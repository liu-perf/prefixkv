# Attribution

## The problem

A prefix cache reporting 90% on traffic where every request is identical has told you
about the traffic. A cache reporting 12% on traffic that shares nothing may be doing
everything available to it. **Comparing two caches by hit rate, or one cache across
two workloads, compares the workloads.**

That is not a subtle effect. One cache, one block size, unlimited memory, seven
traces: hit rate spans **0% to 90%** while `capture` is **100% everywhere**. Every one
of those runs did everything that was available to it.

## The four buckets

Every prompt token lands in exactly one:

| bucket | meaning | what moves it |
|---|---|---|
| `hit` | served from the cache | — |
| `capacity_loss` | the aligned ceiling had it, this cache did not | more memory, or less concurrency competing for it |
| `alignment_loss` | a byte-granular cache would have reused it | a smaller block size |
| `novel` | no cache of any granularity could serve it | changing the traffic |

and

```text
hit + capacity_loss + alignment_loss + novel == prompt_tokens
```

## Why that identity can fail

This is the part worth dwelling on, because the obvious comparison —
`L = λW` on a single timeline — **cannot** fail. Both sides of Little's Law are the
same sum divided two ways, so it holds for a correct engine and a broken one alike.

Here the two sides come from different places:

- `hit` is accumulated by the cache while it runs, from block lookups against a
  chained hash table;
- `aligned_ceiling` and `unaligned_ceiling` are arithmetic over the trace's
  `segments` column — pairwise common leading segments, floored to blocks, capped by
  what each donor could itself have stored. **No hashing, no blocks, no cache.**

An error on either side breaks the sum, and three separate guards fire before it gets
that far: per-request records must agree with the counters, `hit` may not exceed the
ceiling, and the four terms must total the prompt count.
`tests/test_attribution.py` inflates a hit count by one token and checks that the
second guard catches it.

The cross-check that makes the ceiling trustworthy is separate and stronger: the
ceiling formula must equal what the real cache achieves with unlimited memory.
**7 traces × 5 block sizes = 35 configurations, zero mismatches.**

## The two ratios

```text
capture    = hit / aligned_ceiling
structural = aligned_ceiling / unaligned_ceiling
```

`capture` is the only number in this library that is about the cache. `structural` is
what block granularity costs on this traffic — a property of the traffic and the
block size, not of the policy.

Report both, always, because the same hit rate means opposite things:

- **high hit, low capture** — the traffic is generous and the cache is wasting it;
  buy memory.
- **low hit, high capture** — the cache is fine and the traffic is not shareable;
  buying memory does nothing.

## The verdict

Rather than a score, `decompose` returns one sentence naming which of four situations
you are in, because they call for different work and a number would let a reader skip
deciding:

1. **nothing is shared** — no cache of any size or granularity helps.
2. **nothing fills a block** — the block size is the limit and memory cannot move it.
   This case was originally folded into the capacity branch, which produced exactly
   the wrong advice: *"add cache memory"* for a limit that no amount of memory
   changes. It is now handled explicitly.
3. **capture below the floor** — capacity or pinning is the limit.
4. **capture high** — the traffic is the limit, and the sentence says what fraction
   of prompt tokens it offers.

## What `compare` refuses

```python
compare([run_on_trace_a, run_on_trace_b])
# AttributionError: asked to rank configurations across 2 different traces ...
# Hit rate is mostly a property of the traffic, so a ranking that mixes traces
# measures the traces. Compare `capture` per trace instead, or rank within one trace.
```

Ranking configurations *within* one trace is fine and is what `prefixkv sweep` does —
and even there the CLI says so on the summary line, because the ranking does not
travel.

## Where the numbers in this repository come from

Every figure quoted anywhere here was produced by running the shipped traces through
the shipped code. They are measurements of generated traffic whose sharing structure
was chosen deliberately — the mechanisms transfer, the percentages are properties of
these seven files, and `README.md`'s Limits section says so rather than leaving it
for the reader to infer.
