# Blocks, hashing, and the two limits

## Reuse happens at block granularity

A paged KV cache stores keys and values in fixed-size blocks, so a block is reusable
only if **every token in it matches and every block before it matches too**.

Both halves matter. One differing token anywhere in a block kills the whole block — a
15-of-16 match is a miss, not a 94% hit. And position is part of identity: the same 16
tokens at a different offset are a different block, because the keys and values were
computed against a different context.

That second point is why the hash is **chained**: a block's hash is a hash of its
parent's hash and its own tokens. vLLM does the same thing for the same reason. A
content-only hash would cheerfully match a block that means something else.

The hash is FNV-1a rather than Python's `hash()`, because `hash()` on a tuple is not
stable across processes and a hit rate that depends on `PYTHONHASHSEED` is not
measuring anything.

## Limit one: only complete blocks are cacheable

A shared prefix of L tokens yields `L // block_size` reusable blocks. The remaining
`L % block_size` tokens are recomputed every single time, forever.

A 180-token system prompt with 16-token blocks wastes 4 tokens per request. Across
`EXAMPLE_system_prompt.csv` that is 640 tokens. At block size 64 it is 52 per request.
`PK002` reports it.

The trailing partial block is dropped rather than padded. Padding would produce a hash
that matches nothing real, and a cache storing it would report hits it cannot honour.
vLLM handles this the same way — `cache_full_blocks` returns early at
`num_cached_blocks >= num_full_blocks` — with a consequence worth naming: **the partial
tail is not merely wasted, it is unreusable by anybody.**

## Limit two: the last token can never be a hit

The model must run on at least one token to produce a logit, so a lookup is capped at
`len(prompt) - 1`. Then the cached length must be floored to a block boundary.

**The order is not interchangeable.** For a 64-token prompt with 16-token blocks:

```text
right:  64 -> 63 (logit) -> 48 (floor)
wrong:  64 -> 64 (floor) -> 63 (logit)
```

So a prompt that is exactly four blocks long can reuse three of them. The wrong order
claims a reuse no block-aligned cache can deliver.

The whole-block loss lands on prompts whose length is an exact multiple of the block
size — about one request in `block_size`. Everywhere else the loss is the remainder
alone. `PK005` reports it as `info`, because it is small, and because dressing a small
effect up as a finding is its own kind of dishonesty.

## The donor cap that should not have existed

There are two caps in `oracle.aligned_ceiling`, and for a long time this library used
the same quantity for both. That was wrong, and it is the most instructive mistake in
the project, so it is written out in full.

* The **requester** cap is `cacheable_length(req.prompt_tokens, block_size)` — the
  logit rule plus block alignment, as above. Verified against real vLLM's
  `get_computed_blocks` at **76 configurations, zero mismatches**.
* The **donor** cap is `(donor.prompt_tokens // block_size) * block_size` — *all* of
  the donor's complete blocks.

The shipped version used `cacheable_length` for the donor too, reasoning that an
earlier request cannot donate a block it never stored, since a block it "will never be
allowed to serve" is not worth caching.

**The premise is false.** The logit rule limits what a request can **look up**, not
what it **stores**. A later request whose prompt extends past the donor's end can hit
the donor's last full block legitimately. Measured directly on real vLLM: it caches
every full block, **18 of 18 configurations**, whether the donor is still running or
already freed.

The consequence is not a rounding error in kind. With the cap in place the ceiling was
**below** real reuse:

| trace | capped ceiling | real vLLM `prefix_cache_hits` |
|---|---|---|
| `EXAMPLE_multiturn.csv` | 52 544 | **52 656** |
| `EXAMPLE_agent.csv` (69 requests) | 62 800 | **62 864** |

0.2% and 0.1% in magnitude — and fatal in kind, because a ceiling an engine can beat
is not a ceiling, and `capture = hit / aligned_ceiling` could exceed 100%. With the
donor cap corrected, this formula equals real vLLM's reported hits **exactly** on every
trace in `tests/traces/`.

### Why none of the 94 tests caught it

The library's cross-check is that the ceiling formula must equal what a real
chained-hash cache achieves with unlimited memory, **with no shared code between the
two**. No shared code — but a shared premise: `cache.py` applied the same wrong cap
when inserting. So the oracle and the simulated cache agreed with each other while
both were wrong, at 35 configurations.

Worse, a test existed specifically to hold the error in place.
`test_the_donor_cap_is_load_bearing` asserted that removing the cap *overstates* reuse.
It passed, because the cache had the same cap. **That test was protecting a bug.**

And the error grew a false repair on top of itself: `unaligned_ceiling` was originally
computed with a trie, then replaced because the trie "could not apply the donor cap".
The trie had been right; the cap was the bug.

> **Two routes that share no code but share a premise are not two routes.** Agreement
> between them is only evidence if they can disagree. Breaking this particular tie
> needed an outside authority, and that is what replaying these traces through a real
> `vllm serve` was for. The replacement test,
> `test_a_donor_donates_all_of_its_full_blocks`, pins the number a real engine
> produced rather than one this library computed.

### The fixture bug found along the way

While chasing this, the older `test_the_donor_cap_is_load_bearing` exposed something
unrelated and real: the multi-turn fixture had been generated with each history entry
under a *different* segment label from the question it followed, which breaks the
prefix relation between consecutive turns entirely. Real multi-turn appends the reply
after the same question, so turn k's whole prompt **is** a strict prefix of turn k+1's.
The generator now does that.

## Block size is a trade, and it has no default

`EXAMPLE_multiturn.csv`, unlimited memory:

| block size | hit | alignment loss | structural |
|---|---|---|---|
| 1 | 73.3% | 0.0% | 100.0% |
| 8 | 72.7% | 0.5% | 99.3% |
| 16 | 72.1% | 1.2% | 98.4% |
| 32 | 71.1% | 2.2% | 97.0% |
| 64 | 67.8% | 5.5% | 92.5% |
| 128 | 62.2% | 11.0% | 84.9% |
| 256 | 51.5% | 21.7% | 70.3% |

Smaller blocks always capture more reuse. What they cost is not visible here at all —
more blocks means more hash entries, more per-block bookkeeping, and worse kernel
locality, none of which this library measures. So the table is **one side of a trade**,
and `prefixkv sweep --over block_size` prints it rather than recommending a value.

vLLM's default is 16. This library's default is 16 for the same reason: it is what the
ecosystem uses, not because anything here derived it.

### The other side, since measured

A sibling project, `blocktax`, measured the column this table is missing, on vLLM's
Triton attention kernel with a noise floor built from twin controls. Recorded here
because a stated gap that gets filled should say so:

* on a **scattered** block table — real paged memory, block ids not contiguous —
  block sizes 1/2/4 cost **1%–2.4%** of decode attention time, replicated in four,
  three and three independent measurements;
* block sizes **at or above 8 are indistinguishable from each other** in every
  measurement that had the resolution to tell;
* with a **contiguous** block table the penalty disappears even though the kernel's
  tile-size clamping is identical, so the cost is **indirection**, not tile size;
* the block *table* is not the cost. At `block_size=1` it is 0.003% of the KV it
  indexes. The cost of small blocks is work — hash calls and indirection.

Two cautions on reading that across. The measurement is one consumer card, one
kernel, decode only, so the magnitudes do not transfer; and it does **not** cover
scheduler cost or block-table traffic, so the supported claim is "attention does not
charge for blocks ≥ 8", not "small blocks are free".

The joint conclusion is still not a recommended value. Going from 16 to 8 buys the
0.6 points above and costs something no measurement there could resolve — bounded by
0.23%, **not** established to be zero. Which of those wins depends on how a
deployment divides its step between prefill and decode attention, and that is a
property of the deployment.

## Converting blocks to memory

```python
from prefixkv import capacity_blocks_for
bytes_per_token = 8 * 128 * 2 * 2 * 32       # kv_heads, head_dim, K+V, bf16, layers
capacity_blocks_for(1 << 30, 16, bytes_per_token)     # 512
```

131 072 bytes per token across all layers for a Llama-3.1-8B-shaped GQA decoder, so a
16-token block is 2.0 MiB and 1 GiB of cache holds 512 blocks. Every term is passed in;
nothing about the model or the framework is guessed here.
