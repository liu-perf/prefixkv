# prefixkv

**How much of a prefix cache's saving is the cache, and how much is the traffic.**

A prefix cache's hit rate is mostly a property of the workload. This library refuses
to report one on its own: every prompt token is attributed to exactly one of four
buckets — served from cache, lost to eviction or pinning, lost to block granularity,
or novel and unservable by any cache — and the four sum to the total.

Prefix caching itself is not this repository's: it is vLLM's automatic prefix caching and SGLang's [RadixAttention](https://arxiv.org/abs/2312.07104). The measurements here are against vLLM's implementation. What is added is the attribution — and the identity that lets it fail.

```console
$ prefixkv run tests/traces/EXAMPLE_agent.csv --capacity-blocks 96 --audit
hit: [info] 63088 tokens (79.5% of prompt)
capacity_loss: [info] 1536 tokens (1.9%) -- available at this block size but evicted or pinned
alignment_loss: [info] 492 tokens (0.6%) -- reusable at byte granularity, thrown away by block granularity
novel: [info] 14250 tokens (18.0%) -- no cache of any granularity could serve these
partition: [ok] the four buckets sum to 79366 prompt tokens exactly; `hit` comes from the cache's counters and the ceilings from arithmetic over the segments column, so this can fail
capture: [info] 97.6% of the reuse available at this block size
structural: [info] block granularity allows 99.2% of this traffic's total reuse
verdict: [info] the cache captured 98% of a ceiling that is 99% of the traffic's total reuse; the limit is the traffic, which offers 82% reusable prompt tokens
```

Zero dependencies, no tokenizer, no model, no GPU. The inputs are CSV files and the
outputs are counts of tokens.

---

## Three findings

### ① The same content, one ordering change, and the hit rate goes to zero

`EXAMPLE_system_prompt.csv` and `EXAMPLE_shared_last.csv` contain **byte-for-byte the
same segments** — a 180-token shared preamble and 160 unique questions, 37 803 prompt
tokens either way. The only difference is that the second puts the shared part
*after* the unique part.

| trace | prompt tokens | hit | novel |
|---|---|---|---|
| `EXAMPLE_system_prompt.csv` | 37 803 | **74.0%** | 24.3% |
| `EXAMPLE_shared_last.csv` | 37 803 | **0.0%** | 100.0% |

A prefix cache reuses a *prefix*. Put anything unique in front of the shared part and
every one of those tokens becomes unreachable — and **no cache configuration recovers
it.** More memory does not help. A smaller block size does not help. It is a one-line
change to prompt assembly, and it is worth 74 points.

The reason this needs a tool is that both runs report the same hit rate a workload
that shares nothing would: **0%.** `PK001` is what distinguishes "there is nothing to
reuse" from "there is plenty and the ordering made it unreachable", and it names the
request and the token count:

```console
r0000: [violation] PK001 180 tokens of reusable segments sit behind the single-use
segment USR:q0000; a prefix cache reuses a prefix, so those tokens can never be
served from cache. Move the shared part first.
```

### ② A hit rate is mostly the traffic, so it is not a claim about a cache

One cache, one block size, unlimited memory, every trace:

| trace | hit | alignment loss | novel | **capture** | structural |
|---|---|---|---|---|---|
| `EXAMPLE_fewshot.csv` | 90.0% | 0.0% | 10.0% | **100%** | 100% |
| `EXAMPLE_agent.csv` | 81.4% | 0.6% | 18.0% | **100%** | 99% |
| `EXAMPLE_system_prompt.csv` | 74.0% | 1.7% | 24.3% | **100%** | 98% |
| `EXAMPLE_multiturn.csv` | 72.1% | 1.2% | 26.7% | **100%** | 98% |
| `EXAMPLE_tiny.csv` | 53.2% | 5.8% | 41.0% | **100%** | 90% |
| `EXAMPLE_no_sharing.csv` | 0.0% | 0.0% | 100.0% | n/a | n/a |
| `EXAMPLE_shared_last.csv` | 0.0% | 0.0% | 100.0% | n/a | n/a |

**Hit rate spans 0% to 90%. Capture is 100% everywhere.** Every one of those runs did
everything available to it; the spread is the workload and nothing else. So:

- `capture = hit / aligned_ceiling` is the only number that is about the cache;
- `structural = aligned_ceiling / unaligned_ceiling` is what block granularity costs
  on this traffic, which is also not about the cache.

Ranking two caches by hit rate on different traffic ranks the traffic. `compare()`
raises rather than doing it silently.

### ③ The capacity cliff is a cliff, not a slope

`EXAMPLE_agent.csv`, 16-token blocks, one request in flight:

| cache blocks | hit | capacity loss | capture |
|---|---|---|---|
| 32 | **0.0%** | 81.4% | 0.0% |
| 48 | 52.9% | 28.5% | 65.0% |
| 64 | 65.5% | 15.9% | 80.4% |
| 96 | 79.5% | 1.9% | 97.6% |
| 128 | 81.3% | 0.1% | 99.9% |
| 256 | 81.4% | 0.0% | 100.0% |

Below the point where **a single prompt fits**, the hit rate is exactly zero: every
block inserted is evicted before the next request can reach it. A capacity plan that
interpolates towards zero misses this entirely.

Concurrency spends the same memory, so it competes with the cache directly. Same
trace, 256 blocks:

| requests in flight | hit | capture | pin-blocked evictions |
|---|---|---|---|
| 1 – 8 | 81.4% | 100.0% | 0 |
| 16 | 77.8% | 95.5% | 14 918 |
| 32 | 68.0% | 83.5% | 21 451 |

To convert blocks into memory, `capacity_blocks_for` does the arithmetic and nothing
is guessed: for a Llama-3.1-8B-shaped GQA decoder across 32 layers the cache holds
131 072 bytes per token, so a 16-token block is **2.0 MiB** and 1 GiB of cache is
**512 blocks**.

---

## The check that can fail

The four buckets are a partition, and unlike `L = λW` on a single timeline that
identity is **not** guaranteed by the arithmetic: `hit` comes from the cache's own
counters, while both ceilings come from arithmetic over the trace's `segments` column
that touches no cache code.

That second route is possible because a request is a list of **named segments**:

```text
# LEN SYS:assistant_v3=180
request_id,arrival_s,segments,output_tokens
r0000,0.418697,SYS:assistant_v3|USR:q0000,163
```

Distinct segment names occupy disjoint token-id ranges, so two requests can never
share *part* of a segment, and the longest common prefix of two requests is exactly
the sum of the lengths of their common leading segments. **The trace declares its own
reuse ceiling, in a column a human can read**, and `prefixkv ceilings` computes it
with no cache involved.

The cross-check: the ceiling formula must equal what the real chained-hash block
cache achieves with unlimited memory. **7 traces × 5 block sizes = 35 configurations,
zero mismatches**, and the two sides share no code.

---

## Two mechanisms worth knowing before you tune anything

**Only complete blocks are cacheable.** A shared prefix of L tokens yields
`L // block_size` reusable blocks; the remainder is recomputed every time. A
180-token system prompt with 16-token blocks wastes 4 tokens per request forever —
640 across this trace. At 64-token blocks it wastes 52.

**The last token can never be a hit**, because the model must run on something to
produce a logit, and the cached length must stay block-aligned afterwards. Applied in
that order: a 64-token prompt with 16-token blocks gives 63, floored to 48. **A prompt
that is exactly four blocks long can reuse only three of them.**

Block size trades the two against each other. `EXAMPLE_multiturn.csv`, unlimited
memory:

| block size | hit | alignment loss | structural |
|---|---|---|---|
| 1 | 73.3% | 0.0% | 100.0% |
| 16 | 72.1% | 1.2% | 98.4% |
| 64 | 67.8% | 5.5% | 92.5% |
| 256 | 51.5% | 21.7% | 70.3% |

---

## Install and run

```bash
pip install -e .

prefixkv trace    tests/traces/EXAMPLE_agent.csv          # what this traffic shares
prefixkv ceilings tests/traces/EXAMPLE_multiturn.csv      # upper bounds, no cache
prefixkv run      tests/traces/EXAMPLE_system_prompt.csv --audit
prefixkv sweep    tests/traces/EXAMPLE_multiturn.csv --over block_size --values 8 16 64
prefixkv --fail-on violation audit tests/traces/EXAMPLE_shared_last.csv
```

Five subcommands. Every human line is `{location}: [{status}] {message}`; `--json`
prints the structure; `--fail-on` is a threshold, not an equality test.

---

## Five rules

| rule | what it finds |
|---|---|
| **PK001** | a reusable segment sitting behind a single-use one, so it can never be reached |
| **PK002** | shared prefixes that are not a multiple of the block size, and what that costs |
| **PK003** | traffic that shares nothing, where prefix caching is not the lever at all |
| **PK004** | capture below the reporting floor, meaning capacity or pinning is the limit |
| **PK005** | the whole-block loss from the logit rule, on the ~1-in-`block_size` prompts it lands on |

`test_every_rule_fires_somewhere_and_stays_quiet_somewhere` asserts each one does
both. A rule that always fires and a rule that never fires are equally useless.

---

## The first version is kept, and still wrong

`prefixkv/naive.py` holds four things you get by writing down the obvious arithmetic,
and `tests/test_naive.py` asserts they are **still** wrong.

1. **`shared / prompt` as a hit rate.** It is the byte-granularity limit with the
   logit rule waived. At block size 1 it is off by exactly **180 tokens** on the
   system-prompt trace — the *first* request's shared prefix, counted as reuse for a
   request that arrived before there was anything to reuse. The error shrinks as the
   trace lengthens, which is why nobody notices it.
2. **Rounding blocks up.** Counts the trailing partial block, which is never hashed.
3. **Capping the donor by the logit rule.** This one *shipped*, and it is the only
   error here that no internal check could catch. The reasoning was that an earlier
   request cannot donate a block it never stored — but the logit rule limits what a
   request can **look up**, not what it **stores**, and a later request with a longer
   prompt can hit the donor's last full block. Real vLLM caches every full block
   (measured, 18 of 18 configurations), so the capped version reported a ceiling
   *below* real reuse: 52 544 on the multi-turn trace where vLLM actually served
   **52 656**. A ceiling an engine can beat is not a ceiling. See
   [docs/blocks.md](docs/blocks.md#the-donor-cap-that-should-not-have-existed).
4. **`hit_rate × prefill_cost × tokens` as an end-to-end saving.** A hit removes
   prefill, not queueing and not decode.

---

## Limits

Stated because they are open, not as modesty.

1. **No scheduler.** Concurrency enters through one knob — `--concurrency N` pins the
   blocks of the N most recent in-flight requests — and nothing else. There is no
   cost model and no step loop; that is a different tool's job, and duplicating it
   here would mean two half-calibrated models instead of one measured one.

   This one has been measured against a real engine rather than left as an
   adjective. Replaying every trace through `vllm serve` and comparing against
   `vllm:prefix_cache_hits`, over 3 cache sizes × 3 concurrencies × 6 traces:

   - **Serially, the model is exact.** At `--concurrency 1`, simulated hits equal
     the real engine's hits *token for token* on all six traces, at cache sizes
     from 160 blocks (tight enough to force repeated eviction) to 4096.
   - **Under concurrency it is pessimistic**, by up to **40.9%** of the ceiling at
     `--concurrency 32` with a tight cache. The pin window holds the last N
     requests' blocks against eviction; a real engine releases them on completion
     while keeping them cached, so it retains reuse this model throws away.
   - That 40.9% is an **upper bound on the model error, not the error**: the two
     "concurrency" knobs are different quantities — this one pins N requests'
     blocks, the experiment's allows N requests in flight, and a real engine holds
     far fewer than N resident at once. Separating the two needs a scheduler's view,
     which is the point of this limit.
2. **No eviction policy but LRU.** Real engines add reference counting subtleties,
   LIFO reuse of uncached blocks for locality, and hash-granularity tricks for
   partial blocks. Only LRU-over-unpinned is modelled.
3. **Segments are atomic.** Two requests share a segment or they do not. Real
   traffic has partial overlaps inside a segment — an edited system prompt, a
   paraphrased question — and this library cannot represent one. It makes the
   ceiling exact and the traffic model coarser.
4. **All traces are synthetic.** Every number here is measured on generated traffic
   whose sharing structure was chosen. The mechanisms transfer; the percentages are
   properties of these seven files.
5. **No cross-request sharing beyond prefixes.** No sliding window, no selective
   eviction, no compression, no disk or CPU tiers.

---

## Licence

MIT.
