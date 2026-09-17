import pytest
from conftest import UNLIMITED, unlimited

from prefixkv import PrefixCache, capacity_blocks_for
from prefixkv.blocks import cacheable_length, full_blocks


def test_a_repeated_prompt_hits_the_second_time(tiny):
    cache = unlimited(16)
    records = {r["request"]: r for r in cache.run(tiny)}
    # r0006 has the same segment list as r0000.
    assert records["r0000"]["hit"] == 0
    assert records["r0006"]["hit"] == cacheable_length(records["r0006"]["prompt"], 16)


def test_a_lookup_stops_at_the_first_missing_block():
    """It cannot skip a gap and resume.

    The chain means block k's identity depends on blocks 0..k-1, so everything after
    a miss is a different block than what is stored.
    """
    cache = PrefixCache(UNLIMITED, 4)
    tokens = list(range(40))
    blocks = full_blocks(tokens[:cacheable_length(len(tokens), 4)], 4)
    for i, (_, h, _) in enumerate(blocks):
        if i != 1:                       # store everything except the second block
            cache._blocks[h] = True
            cache._order.append(h)
    assert cache.lookup(tokens) == 4     # one block, then it stops


def test_every_full_block_is_stored_but_the_partial_tail_is_not():
    """Insertion is bounded by full blocks, not by what *this* request can look up.

    This replaces a test that asserted 3 blocks for a 64-token prompt, on the
    reasoning that `cacheable_length(64, 16) == 48` so the fourth block "may never be
    served". It can: not to this request, but to a later one whose prompt extends past
    token 64. Real vLLM stores all four (measured, 18 of 18 configurations), and
    capping insertion here made `aligned_ceiling` report a ceiling a real engine beat.
    """
    cache = PrefixCache(UNLIMITED, 16)
    cache.insert(list(range(64)))
    assert cache.used_blocks == 4         # all four full blocks

    # The trailing partial block is still dropped -- that limit is real, because only
    # complete blocks are hashed.
    cache = PrefixCache(UNLIMITED, 16)
    cache.insert(list(range(70)))
    assert cache.used_blocks == 4         # 70 // 16 == 4, the last 6 tokens are not

    # And a request still cannot serve itself its own last block.
    cache = PrefixCache(UNLIMITED, 16)
    tokens = list(range(64))
    cache.insert(tokens)
    assert cache.lookup(tokens) == 48     # own logit-rule cap, unchanged
    assert cache.lookup(list(range(80))) == 64   # a longer request gets all four


def test_the_control_never_hits(no_sharing):
    cache = unlimited(16)
    cache.run(no_sharing)
    assert cache.stats.hit_tokens == 0


def test_eviction_is_least_recently_used():
    # Four blocks of room; each 9-token sequence stores two blocks
    # (cacheable_length(9, 4) == 8), so two sequences fill the cache exactly.
    cache = PrefixCache(4, 4)
    a, b, c = [list(range(i * 100, i * 100 + 9)) for i in range(3)]
    cache.insert(a)
    cache.insert(b)
    assert cache.used_blocks == 4
    cache.lookup(a)                      # touch a, so b becomes least recent
    cache.insert(c)
    assert cache.lookup(a) > 0
    assert cache.lookup(b) == 0


def test_a_pinned_block_cannot_be_evicted():
    # One block of room; a 5-token sequence stores exactly one block
    # (cacheable_length(5, 4) == 4).
    cache = PrefixCache(1, 4, concurrency=1)
    a = list(range(0, 5))
    cache.begin("a", cache.insert(a))
    cache.insert(list(range(100, 105)))  # must fail: the only block is pinned
    assert cache.stats.pin_blocked_evictions > 0
    assert cache.stats.capacity_misses > 0
    assert cache.lookup(a) > 0


def test_the_pin_window_releases_as_it_slides():
    cache = PrefixCache(4, 4, concurrency=2)
    for i in range(3):
        cache.begin(f"r{i}", cache.insert(list(range(i * 100, i * 100 + 9))))
    assert len(cache._inflight) == 2
    cache.drain()
    assert cache._inflight == []
    assert cache.pinned_blocks == 0


def test_concurrency_reduces_hits_on_a_tight_cache(agent):
    alone = PrefixCache(256, 16, 1)
    alone.run(agent)
    crowded = PrefixCache(256, 16, 32)
    crowded.run(agent)
    assert crowded.stats.hit_tokens < alone.stats.hit_tokens
    assert crowded.stats.pin_blocked_evictions > 0


def test_a_cache_smaller_than_one_prompt_serves_nothing(agent):
    """The cliff.

    Below the point where a single prompt fits, the hit rate is exactly zero: every
    block inserted is evicted before the next request can reach it. It is not a
    gentle slope, and a capacity plan interpolating towards zero would miss it.
    """
    small = PrefixCache(32, 16, 1)
    small.run(agent)
    assert small.stats.hit_tokens == 0
    bigger = PrefixCache(48, 16, 1)
    bigger.run(agent)
    assert bigger.stats.hit_tokens > 0


def test_bad_construction_is_refused():
    with pytest.raises(ValueError):
        PrefixCache(0, 16)
    with pytest.raises(ValueError):
        PrefixCache(16, 0)
    with pytest.raises(ValueError):
        PrefixCache(16, 16, concurrency=0)


def test_capacity_blocks_for_is_plain_arithmetic():
    bytes_per_token = 8 * 128 * 2 * 2 * 32          # Llama-3.1-8B GQA, all 32 layers
    assert bytes_per_token == 131072                # so a 16-token block is 2 MiB
    assert capacity_blocks_for(1 << 30, 16, bytes_per_token) == 512
    with pytest.raises(ValueError):
        capacity_blocks_for(1 << 30, 0, bytes_per_token)
