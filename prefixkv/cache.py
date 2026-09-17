"""The prefix cache: block-granular lookup, refcounted pinning, LRU eviction.

Modelled on what vLLM V1 actually does, which was read out of
`vllm/v1/core/block_pool.py` and `kv_cache_manager.py`:

  * blocks are hashed in a chain, so position is part of identity;
  * only **complete** blocks are cached -- `cache_full_blocks` returns early at
    `num_cached_blocks >= num_full_blocks`, so the trailing partial block is never
    reusable by anybody;
  * a lookup is capped so at least one token is recomputed for the logit, and the
    cached length must stay block-aligned;
  * blocks in use are refcounted and cannot be evicted; blocks at refcount zero are
    eviction candidates, cached ones in LRU order.

What is deliberately *not* modelled, and why
--------------------------------------------
Scheduling. There is no cost model, no step loop, and no notion of how long a
request takes -- that is `kvsched`'s job, and duplicating it here would mean two
half-calibrated cost models instead of one measured one.

Concurrency enters through exactly one knob: `--concurrency N` pins the blocks of
the N most recent in-flight requests. It is a crude model of "N requests are in
flight" and it is crude on purpose, because the honest alternative is a full
scheduler. What it captures is the part that matters for reuse: **pinned blocks
cannot be evicted, so concurrency competes with the cache for the same memory.**
At `--concurrency 1` requests are served one at a time and eviction is pure LRU;
that understates pressure, and the CLI says so rather than leaving it implied.
"""
from .blocks import cacheable_length, full_blocks


class Stats:
    """Counters, kept apart from the cache so they can be summed and compared."""

    __slots__ = ("lookups", "hit_tokens", "prompt_tokens", "cacheable_tokens",
                 "inserted_blocks", "evicted_blocks", "pin_blocked_evictions",
                 "capacity_misses", "alignment_lost_tokens", "logit_lost_tokens")

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, 0)

    def as_dict(self):
        d = {s: getattr(self, s) for s in self.__slots__}
        d["hit_rate"] = (self.hit_tokens / self.prompt_tokens
                         if self.prompt_tokens else 0.0)
        return d


class PrefixCache:
    """Fixed-capacity block cache with chained hashing and LRU eviction.

    `capacity_blocks` is the whole pool. There is no watermark: vLLM's defaults to
    `0.0` (disabled) and relies on a scheduler rule instead, and this library has no
    scheduler to put a rule in. Saying that plainly is better than inventing a
    reservation and quietly changing the hit rate by it.
    """

    def __init__(self, capacity_blocks, block_size, concurrency=1):
        if capacity_blocks < 1:
            raise ValueError("capacity_blocks must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.capacity_blocks = int(capacity_blocks)
        self.block_size = int(block_size)
        self.concurrency = int(concurrency)
        self.stats = Stats()
        self._blocks = {}      # hash -> True (contents are irrelevant to reuse)
        self._order = []       # hashes, least-recently-used first
        self._refs = {}        # hash -> pin count
        self._inflight = []    # list of (request_id, [hashes]) -- the pin window

    # -- state -----------------------------------------------------------------
    @property
    def used_blocks(self):
        return len(self._blocks)

    @property
    def free_blocks(self):
        return self.capacity_blocks - self.used_blocks

    @property
    def pinned_blocks(self):
        return sum(1 for h, n in self._refs.items() if n > 0 and h in self._blocks)

    def occupancy(self):
        return self.used_blocks / self.capacity_blocks

    # -- lookup ----------------------------------------------------------------
    def lookup(self, token_ids):
        """Longest cached block-aligned prefix, in tokens.

        Walks the block chain from the start and stops at the first miss. It cannot
        skip a missing block and resume: the chain means block k's identity depends
        on blocks 0..k-1, so a gap makes everything after it a different block.
        """
        limit = cacheable_length(len(token_ids), self.block_size)
        hit = 0
        for _, h, _ in full_blocks(token_ids[:limit], self.block_size):
            if h not in self._blocks:
                break
            hit += self.block_size
            self._touch(h)
        return hit

    def _touch(self, h):
        try:
            self._order.remove(h)
        except ValueError:
            pass
        self._order.append(h)

    # -- insertion and eviction -------------------------------------------------
    def insert(self, token_ids):
        """Cache **every** complete block of the prompt. Returns hashes stored.

        Note the insertion range is not the `cacheable_length` used for lookup. The
        first version used it, reasoning that a block the cache "will never be
        allowed to serve" only inflates occupancy. That conflated two things: the
        logit rule limits what *this* request can look up, while the block can still
        be served to a **later, longer** request whose prompt extends past it.

        Real vLLM caches every full block -- `cache_full_blocks` is driven by
        `num_full_blocks`, not by any `n - 1` cap -- measured directly at 18 of 18
        configurations. Capping insertion here made `aligned_ceiling` report less
        reuse than a real engine achieves, i.e. not a ceiling. See
        `oracle.aligned_ceiling` for the full story and the measured cost.
        """
        limit = (len(token_ids) // self.block_size) * self.block_size
        stored = []
        for _, h, _ in full_blocks(token_ids[:limit], self.block_size):
            if h in self._blocks:
                self._touch(h)
                stored.append(h)
                continue
            if self.free_blocks <= 0 and not self._evict_one():
                self.stats.capacity_misses += 1
                break
            self._blocks[h] = True
            self._order.append(h)
            self.stats.inserted_blocks += 1
            stored.append(h)
        return stored

    def _evict_one(self):
        """Drop the least-recently-used unpinned block. False if everything is pinned."""
        for h in list(self._order):
            if self._refs.get(h, 0) > 0:
                self.stats.pin_blocked_evictions += 1
                continue
            self._order.remove(h)
            del self._blocks[h]
            self.stats.evicted_blocks += 1
            return True
        return False

    # -- the pin window ----------------------------------------------------------
    def begin(self, request_id, hashes):
        for h in hashes:
            self._refs[h] = self._refs.get(h, 0) + 1
        self._inflight.append((request_id, list(hashes)))
        while len(self._inflight) > self.concurrency:
            self._release(*self._inflight.pop(0))

    def _release(self, request_id, hashes):
        for h in hashes:
            n = self._refs.get(h, 0) - 1
            if n <= 0:
                self._refs.pop(h, None)
            else:
                self._refs[h] = n

    def drain(self):
        while self._inflight:
            self._release(*self._inflight.pop(0))

    # -- one request -------------------------------------------------------------
    def serve(self, request):
        """Look up, account, insert, pin. Returns the per-request record.

        The three token counts are kept separate all the way out, because collapsing
        them is how a prefix-cache report stops being checkable:

            prompt      what the request brought
            cacheable   what block granularity and the logit rule allow at best
            hit         what was actually in the cache
        """
        tokens = request.token_ids()
        prompt = len(tokens)
        cap = cacheable_length(prompt, self.block_size)
        hit = self.lookup(tokens)
        stored = self.insert(tokens)
        self.begin(request.id, stored)

        s = self.stats
        s.lookups += 1
        s.prompt_tokens += prompt
        s.cacheable_tokens += cap
        s.hit_tokens += hit
        # The two structural losses, separated. Alignment loss is what block
        # granularity costs; logit loss is the one token that can never be a hit and
        # the block-alignment rounding it drags with it.
        s.alignment_lost_tokens += (prompt - 1) % self.block_size if prompt > 1 else 0
        s.logit_lost_tokens += 1 if prompt > 1 else 0
        return {"request": request.id, "prompt": prompt, "cacheable": cap,
                "hit": hit, "computed": prompt - hit,
                "blocks_stored": len(stored)}

    def run(self, trace):
        records = [self.serve(r) for r in trace]
        self.drain()
        return records

    def __repr__(self):
        return (f"PrefixCache({self.used_blocks}/{self.capacity_blocks} blocks, "
                f"block_size={self.block_size}, concurrency={self.concurrency})")


def capacity_blocks_for(pool_bytes, block_size, bytes_per_token):
    """Blocks that fit. Every term is passed in; nothing is guessed here."""
    per_block = block_size * bytes_per_token
    if per_block <= 0:
        raise ValueError("block_size and bytes_per_token must be positive")
    return max(0, int(pool_bytes // per_block))


__all__ = ["PrefixCache", "Stats", "capacity_blocks_for"]
