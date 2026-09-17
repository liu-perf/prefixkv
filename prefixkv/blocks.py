"""Block hashing: the mechanism that decides what a prefix cache can reuse.

A paged KV cache stores keys and values in fixed-size blocks, so reuse happens at
block granularity, and a block is reusable only if **every token in it matches and
every block before it matches too**. Both halves matter:

  * every token in it -- one differing token anywhere in a block kills the whole
    block, so a 15/16 match is a miss, not a 94% hit;
  * every block before it -- position is part of identity. The same 16 tokens
    appearing at a different offset are a different block, because the keys and
    values were computed against a different context.

The second is why the hash is **chained**: a block's hash is a hash of (its
parent's hash, its own tokens). vLLM does the same thing for the same reason. A
content-only hash would happily match a block that means something else.

Two consequences fall straight out of the arithmetic and are the source of most of
this library's findings:

1. **Only complete blocks are cacheable.** A shared prefix of L tokens yields
   `L // block_size` reusable blocks, and the remaining `L % block_size` tokens are
   recomputed every time. With a 100-token system prompt and 16-token blocks, four
   tokens are permanently wasted -- small. With a hundred distinct 20-token shared
   preambles, it is a quarter of them.

2. **The last token cannot be a hit.** The model has to run on at least one token
   to produce a logit, so a lookup is capped at `len(prompt) - 1`. When a prompt is
   exactly a previously cached sequence -- which is what every turn of a
   conversation looks like -- that cap can cost a whole block, because the cached
   length has to stay block-aligned. vLLM's source calls this out in a comment on
   `max_cache_hit_length`.
"""

#: Chained-hash mixing constants. FNV-1a's 64-bit offset basis and prime, used
#: because the hash needs to be deterministic across machines and Python versions
#: -- `hash()` on a tuple is neither, and a cache whose hit rate depends on
#: PYTHONHASHSEED is not measuring anything.
_OFFSET = 0xCBF29CE484222325
_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1


def _fnv(state, value):
    state ^= value & _MASK
    return (state * _PRIME) & _MASK


def block_hash(parent_hash, token_ids):
    """Hash of one block, chained to its parent.

    `parent_hash` is None for the first block. The None case is mixed in as a
    distinct constant rather than skipped, so a first block and an identical block
    appearing later cannot collide.
    """
    state = _OFFSET if parent_hash is None else _fnv(_OFFSET, parent_hash)
    state = _fnv(state, 0x9E3779B97F4A7C15 if parent_hash is None else 1)
    for tid in token_ids:
        state = _fnv(state, tid)
    return state


def full_blocks(token_ids, block_size):
    """The complete blocks of a token sequence, as (index, hash, tokens).

    Deliberately drops the trailing partial block instead of padding it. Padding
    would produce a hash that matches nothing real, and a cache that stored it
    would report hits it cannot honour.
    """
    out = []
    parent = None
    n = len(token_ids) // block_size
    for i in range(n):
        chunk = token_ids[i * block_size:(i + 1) * block_size]
        h = block_hash(parent, chunk)
        out.append((i, h, chunk))
        parent = h
    return out


def cacheable_length(n_tokens, block_size):
    """How many tokens of an `n_tokens` prompt can ever be served from cache.

    Two limits, applied in this order:

      1. at least one token must be recomputed to produce a logit  -> n - 1
      2. the cached length must be block-aligned                   -> floor to block

    The order is not interchangeable, and getting it wrong is the difference between
    a plausible number and a right one. For a 64-token prompt with 16-token blocks:
    limit 1 gives 63, limit 2 floors it to 48. **A prompt that is exactly four blocks
    long can reuse only three of them.** Reversing the two would say 64 -> 64 -> 63,
    claiming a reuse that no block-aligned cache can deliver.

    The whole-block loss lands on prompts whose length is an exact multiple of the
    block size -- about one request in `block_size` -- and on those it costs a full
    block. Everywhere else the loss is the remainder alone.
    """
    if n_tokens <= 1 or block_size <= 0:
        return 0
    return ((n_tokens - 1) // block_size) * block_size


def alignment_waste(shared_tokens, block_size):
    """Tokens of a shared prefix that block granularity throws away."""
    if shared_tokens <= 0:
        return 0
    return shared_tokens % block_size


def common_prefix_len(a, b):
    """Length of the longest common prefix of two token lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


__all__ = ["alignment_waste", "block_hash", "cacheable_length", "common_prefix_len",
           "full_blocks"]
