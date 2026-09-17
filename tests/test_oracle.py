from conftest import ALL_TRACES, UNLIMITED, trace_path, unlimited

from prefixkv import PrefixCache, load_trace, oracle

BLOCK_SIZES = [1, 8, 16, 32, 64]


def test_the_formula_and_the_cache_agree_on_every_trace_and_block_size():
    """The independent cross-check, and the reason this project is checkable.

    Left side: arithmetic over the `segments` column -- pairwise common leading
    segments, floored to blocks, capped by what the donor itself could have stored.
    No hashing, no blocks, no cache.

    Right side: the real chained-hash block cache with capacity beyond any possible
    need.

    The two share no code. 7 traces x 5 block sizes = 35 configurations, and a
    disagreement in any of them would be information rather than noise.
    """
    for name in ALL_TRACES:
        tr = load_trace(trace_path(name))
        for bs in BLOCK_SIZES:
            formula = oracle.aligned_ceiling(tr, bs)
            cache = PrefixCache(UNLIMITED, bs, concurrency=1)
            cache.run(tr)
            assert formula == cache.stats.hit_tokens, (
                f"{name} at block_size={bs}: formula says {formula}, an unlimited "
                f"cache achieved {cache.stats.hit_tokens}")


def test_a_donor_donates_all_of_its_full_blocks(multiturn):
    """The donor cap is the requester's logit rule, not the donor's.

    This test replaces one that asserted the opposite. The earlier version capped a
    donor at its own `cacheable_length`, reasoning that it cannot donate a block it
    never stored -- but the logit rule limits what a request can *look up*, not what
    it *stores*. A donor caches every full block it has, and a later request with a
    longer prompt can hit the donor's last one.

    The numbers below are pinned to a **measurement on real vLLM**, not to this
    library's own arithmetic: replaying `EXAMPLE_multiturn.csv` through
    `vllm serve` with `--block-size 16 --num-gpu-blocks-override 1024` reported
    `vllm:prefix_cache_hits` = 52656, while the capped formula said 52544. The
    engine beat the "ceiling" by 112 tokens, which is how the error was found.

    The reason no internal test could find it: `cache.py` applied the same cap when
    inserting, so the oracle and the simulated cache agreed. Two implementations
    sharing no code but sharing a premise cannot check each other.
    """
    from prefixkv.blocks import cacheable_length
    from prefixkv.oracle import common_leading_segments
    bs = 16

    correct = oracle.aligned_ceiling(multiturn, bs)
    seen, donor_capped = [], 0
    for req in multiturn:
        best = 0
        for earlier in seen:
            shared = common_leading_segments(req, earlier)
            best = max(best, min((shared // bs) * bs,
                                 cacheable_length(earlier.prompt_tokens, bs)))
        donor_capped += min(best, cacheable_length(req.prompt_tokens, bs))
        seen.append(req)

    assert correct == 52656, "this is what real vLLM served on this trace"
    assert donor_capped == 52544, "this is what the shipped bug reported"
    assert correct - donor_capped == 112
    assert donor_capped < correct, (
        "the wrong cap must understate reuse; if it no longer does, this test is not "
        "exercising the thing it exists for"
    )


def test_block_granularity_can_only_lose_reuse():
    for name in ALL_TRACES:
        tr = load_trace(trace_path(name))
        unaligned = oracle.unaligned_ceiling(tr)
        for bs in BLOCK_SIZES:
            assert oracle.aligned_ceiling(tr, bs) <= unaligned


def test_bigger_blocks_never_help(multiturn):
    prev = None
    for bs in (1, 8, 16, 32, 64, 128):
        got = oracle.aligned_ceiling(multiturn, bs)
        if prev is not None:
            assert got <= prev, f"block_size={bs} beat a smaller one"
        prev = got


def test_the_control_has_no_reuse_at_all(no_sharing):
    assert oracle.unaligned_ceiling(no_sharing) == 0
    for bs in BLOCK_SIZES:
        assert oracle.aligned_ceiling(no_sharing, bs) == 0


def test_moving_the_shared_part_last_destroys_all_of_it(system_prompt, shared_last):
    """Same content, one ordering change, and the reuse ceiling goes to zero."""
    assert system_prompt.prompt_tokens == shared_last.prompt_tokens
    assert oracle.unaligned_ceiling(system_prompt) > 0
    assert oracle.unaligned_ceiling(shared_last) == 0


def test_ceilings_reports_a_partition(system_prompt):
    c = oracle.ceilings(system_prompt, 16)
    assert (c["aligned_ceiling"] + c["alignment_loss"] + c["novel_tokens"]
            == c["prompt_tokens"])


def test_a_hand_checkable_case(tiny):
    """EXAMPLE_tiny is small enough to verify with a pencil, so it is verified.

    r0000 is SYS:small(40) + USR:a(9) = 49 tokens, the first request, so no reuse.
    r0001 shares SYS:small = 40 tokens; block-aligned at 16 that is 32; the donor
    r0000 could store cacheable_length(49,16) = 48, so 32 survives.
    """
    from prefixkv.blocks import cacheable_length
    reqs = list(tiny)
    assert reqs[0].prompt_tokens == 49
    assert cacheable_length(49, 16) == 48
    assert oracle.common_leading_segments(reqs[1], reqs[0]) == 40
    assert (40 // 16) * 16 == 32
    cache = unlimited(16)
    records = cache.run(tiny)
    assert records[0]["hit"] == 0
    assert records[1]["hit"] == 32
