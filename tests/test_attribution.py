import pytest
from conftest import ALL_TRACES, UNLIMITED, trace_path

from prefixkv import PrefixCache, decompose, load_trace
from prefixkv.attribution import AttributionError, compare


def report(trace, block_size=16, capacity=UNLIMITED, concurrency=1):
    cache = PrefixCache(capacity, block_size, concurrency)
    records = cache.run(trace)
    return cache, decompose(trace, cache, records)


def test_the_four_buckets_partition_every_prompt_token():
    """The identity that can fail.

    Unlike `L = lambda*W` on one timeline, the four terms come from three
    independent routes -- the cache's counters, a segment-trie walk, and a pairwise
    formula over segment lists -- so an error in any of them breaks the sum.
    """
    for name in ALL_TRACES:
        tr = load_trace(trace_path(name))
        for bs in (8, 16, 64):
            for cap in (64, 512, UNLIMITED):
                _, d = report(tr, bs, cap)
                b = d["buckets"]
                assert (b["hit"] + b["capacity_loss"] + b["alignment_loss"]
                        + b["novel"]) == d["prompt_tokens"], f"{name} bs={bs} cap={cap}"


def test_a_finite_cache_can_never_beat_the_infinite_one():
    for name in ALL_TRACES:
        tr = load_trace(trace_path(name))
        _, big = report(tr, 16, UNLIMITED)
        for cap in (32, 128, 1024):
            _, small = report(tr, 16, cap)
            assert small["buckets"]["hit"] <= big["buckets"]["hit"]


def test_an_inflated_hit_count_is_caught(tiny):
    cache = PrefixCache(UNLIMITED, 16)
    cache.run(tiny)
    cache.stats.hit_tokens += 1          # one token more than the ceiling allows
    with pytest.raises(AttributionError) as exc:
        decompose(tiny, cache)           # no records, so the ceiling check is what fires
    assert "cannot beat an infinite one" in str(exc.value)


def test_records_that_disagree_with_the_counters_are_caught(tiny):
    cache = PrefixCache(UNLIMITED, 16)
    records = cache.run(tiny)
    records[0]["hit"] += 8
    with pytest.raises(AttributionError):
        decompose(tiny, cache, records)


# -- the point of the whole module ------------------------------------------------
def test_hit_rate_is_mostly_the_traffic(system_prompt, no_sharing, agent):
    """One cache, one policy, one block size. Hit rate 0% to 81%.

    Every one of these runs captured 100% of the reuse available to it, so the
    spread is the traffic and nothing else. This is why `capture` exists and why a
    bare hit rate is not a claim about a cache.
    """
    rows = []
    for tr in (no_sharing, system_prompt, agent):
        _, d = report(tr)
        rows.append((tr.source, d["shares"]["hit"], d["capture"]))
    hits = [h for _, h, _ in rows]
    assert min(hits) == 0.0
    assert max(hits) > 0.75
    for _, _, cap in rows:
        assert cap is None or cap > 0.999


def test_the_control_reports_exactly_zero(no_sharing):
    _, d = report(no_sharing)
    assert d["buckets"]["hit"] == 0
    assert d["shares"]["novel"] == 1.0
    assert d["capture"] is None
    assert "shares nothing" in d["verdict"]


def test_moving_the_shared_part_last_costs_every_point(system_prompt, shared_last):
    """Same content, one ordering change. This is the headline finding."""
    _, a = report(system_prompt)
    _, b = report(shared_last)
    assert a["prompt_tokens"] == b["prompt_tokens"]
    assert a["shares"]["hit"] > 0.70
    assert b["shares"]["hit"] == 0.0


def test_capacity_loss_appears_only_when_capacity_binds(agent):
    _, tight = report(agent, 16, 64)
    _, loose = report(agent, 16, UNLIMITED)
    assert tight["shares"]["capacity_loss"] > 0
    assert loose["shares"]["capacity_loss"] == 0
    assert tight["capture"] < loose["capture"]


def test_alignment_loss_grows_with_block_size(multiturn):
    prev = -1.0
    for bs in (1, 8, 16, 32, 64, 128):
        _, d = report(multiturn, bs)
        share = d["shares"]["alignment_loss"]
        assert share >= prev - 1e-12, f"block_size={bs} lost less than a smaller one"
        prev = share
    _, byte = report(multiturn, 1)
    assert byte["shares"]["alignment_loss"] == 0.0


def test_concurrency_competes_with_the_cache(agent):
    _, alone = report(agent, 16, 256, concurrency=1)
    cache, crowded = report(agent, 16, 256, concurrency=16)
    assert crowded["shares"]["hit"] < alone["shares"]["hit"]
    assert cache.stats.pin_blocked_evictions > 0


def test_the_verdict_names_which_of_three_things_is_the_limit(agent, no_sharing):
    # 24 blocks is far below what this trace needs, so capture drops under the
    # reporting floor and the verdict has to name capacity rather than the traffic.
    _, capacity_bound = report(agent, 16, 24)
    assert capacity_bound["capture"] < 0.75
    assert "capacity or pinning" in capacity_bound["verdict"]
    _, block_bound = report(agent, 4096, UNLIMITED)
    assert "block size" in block_bound["verdict"] or "traffic" in block_bound["verdict"]
    _, nothing = report(no_sharing)
    assert "shares nothing" in nothing["verdict"]


def test_comparing_across_traces_is_refused(system_prompt, agent):
    rows = []
    for tr in (system_prompt, agent):
        _, d = report(tr)
        rows.append({"trace": tr.source, **d})
    with pytest.raises(AttributionError) as exc:
        compare(rows)
    assert "mostly a property of the traffic" in str(exc.value)


def test_comparing_within_one_trace_is_allowed(agent):
    rows = []
    for cap in (64, 256, UNLIMITED):
        _, d = report(agent, 16, cap)
        rows.append({"trace": agent.source, **d})
    ranked = compare(rows)
    assert ranked[0]["shares"]["hit"] >= ranked[-1]["shares"]["hit"]
