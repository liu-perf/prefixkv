# The five rules, and where each came from

Each rule fires on a structural property of the traffic or the run, and each one names
what to do about it. `test_every_rule_fires_somewhere_and_stays_quiet_somewhere`
asserts every rule both fires and stays quiet somewhere in the shipped traces — a rule
that always fires and a rule that never fires are equally useless.

## PK001 — a reusable segment behind a single-use one

**Status: violation.**

A prefix cache reuses a *prefix*. Put the user's question before the shared system
prompt and the shared part is unreachable; the hit rate is zero and no cache
configuration recovers it.

**Source: this repository.** `EXAMPLE_shared_last.csv` holds byte-for-byte the same
segments as `EXAMPLE_system_prompt.csv` with the preamble moved to the back, and the
measured hit rate goes from **74.0% to 0.0%** on identical content.

Why this needs a rule rather than a hit rate: **0% is also what traffic that shares
nothing reports.** PK001 is the only thing in the output that distinguishes "there is
nothing to reuse" from "there is plenty to reuse and the prompt ordering made it
unreachable", and it names the request and the token count.

## PK002 — shared prefixes that are not a multiple of the block size

**Status: info.**

`L % block_size` tokens of every shared prefix fall outside a whole block and are
recomputed every time. On `EXAMPLE_system_prompt.csv` with 16-token blocks that is
**640 tokens across 160 requests**; at block size 64 it is an order more.

Source: arithmetic, cross-checked against the measured `alignment_loss` bucket.

## PK003 — traffic that shares nothing

**Status: warn when the byte-granular ceiling is under 5% of prompt tokens.**

`EXAMPLE_no_sharing.csv` exists for this rule. It is the **control**: without a trace
where a correct cache must report exactly zero, every other number in the suite is
unfalsifiable, because a cache that reported hits on everything would look excellent on
all six other shapes.

The 5% floor is a reporting threshold, not a physical one, and the underlying rate is
printed either way.

## PK004 — capture below the floor

**Status: warn below 75% capture; unknown when there is no reuse to capture.**

`capture = hit / aligned_ceiling`. Below the floor, capacity or pinning is the limit,
and more memory — or less concurrency competing for it — is what moves the number.

The `unknown` case is not cosmetic: **"captured 0% of nothing" and "captured 0% of a
lot" need different words.** Returning `0.0` for both would let a reader conclude their
cache was broken when the traffic simply had no reuse in it.

## PK005 — the logit rule's whole-block loss

**Status: info.**

One token per request can never be a hit, and the cached length must stay block-aligned
afterwards, so a prompt whose length is an exact multiple of the block size loses a full
block rather than a remainder. That is about one request in `block_size`.

**Source: vLLM's `kv_cache_manager.py`**, which caps a lookup at
`max_cache_hit_length = request.num_tokens - 1` and notes in a comment that this can
force recomputing a whole block, because `allocate_slots()` needs a block-aligned
`num_computed_tokens`.

Reported as `info` rather than `warn` because it is small. Two of the five rules are
`info` on purpose: a rule set where everything is urgent is a rule set nobody reads.

## Using them

```bash
prefixkv --fail-on violation audit tests/traces/EXAMPLE_shared_last.csv   # exit 1
prefixkv run   tests/traces/EXAMPLE_agent.csv --capacity-blocks 96 --audit
```

`--fail-on` is a threshold, not an equality test: `--fail-on warn` also fails on
`violation`.
