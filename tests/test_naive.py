"""The first version is still wrong, and these say by how much.

If a later change makes one of them right, the test fails and somebody has to
explain why.
"""
import pytest
from conftest import UNLIMITED

from prefixkv import PrefixCache, decompose, naive
from prefixkv.blocks import cacheable_length


def test_shared_over_prompt_overstates_the_real_hit_rate(system_prompt):
    """The estimate everybody starts from, against the measured one.

    Every request here carries a 180-token shared prefix, so the naive answer is
    180/prompt. The measured answer is lower for two structural reasons the estimate
    has no room for: only 176 of those 180 tokens fall in whole 16-token blocks, and
    one token per request can never be a hit.
    """
    cache = PrefixCache(UNLIMITED, 16)
    report = decompose(system_prompt, cache, cache.run(system_prompt))
    claimed = naive.hit_rate_from_shared_prefix(
        180 * len(system_prompt), system_prompt.prompt_tokens)
    assert claimed > report["shares"]["hit"]
    assert claimed - report["shares"]["hit"] > 0.01


def test_it_is_exactly_right_only_at_byte_granularity(system_prompt):
    """The naive estimate is not stupid, it is a limit case -- of block_size 1 with
    the logit rule waived. Saying which limit makes it clear what it leaves out."""
    cache = PrefixCache(UNLIMITED, 1)
    report = decompose(system_prompt, cache, cache.run(system_prompt))
    claimed_tokens = 180 * len(system_prompt)
    claimed = naive.hit_rate_from_shared_prefix(
        claimed_tokens, system_prompt.prompt_tokens)
    # The remaining gap is exactly 180 tokens: the *first* request's shared prefix,
    # which had no earlier request to inherit it from. The naive estimate counts
    # reuse for a request that arrived before there was anything to reuse -- an
    # error of one prefix, not of one token per request, and it shrinks as the trace
    # gets longer, which is why nobody notices it.
    gap_tokens = round((claimed - report["shares"]["hit"]) * system_prompt.prompt_tokens)
    assert gap_tokens == 180


def test_rounding_blocks_up_counts_one_that_can_never_be_cached():
    """Two errors in one line: it rounds up, counting the trailing partial block that
    is never hashed, and it ignores the logit rule."""
    for n in (17, 33, 64, 100):
        assert naive.blocks_for_prompt_ignoring_alignment(n, 16) > \
            cacheable_length(n, 16) // 16
    assert naive.blocks_for_prompt_ignoring_alignment(64, 16) == 4
    assert cacheable_length(64, 16) // 16 == 3


def test_capping_the_donor_by_the_logit_rule_understates_reuse_by_one_block():
    """The error that shipped, and the only one no internal check could catch.

    Direction matters here more than magnitude: this one made the *ceiling too low*,
    so a real engine could beat it -- and real vLLM did, by 112 tokens on
    `EXAMPLE_multiturn.csv`. A ceiling that can be beaten is not a ceiling.
    """
    bs = 16
    # Donor's whole prompt is a prefix of the new request and is 64 tokens: an exact
    # multiple of the block size, so the wrong cap bites.
    assert naive.donor_capped_reusable_tokens(64, 64, bs) == 48
    assert naive.reusable_tokens(64, 64, bs) == 64
    # The donor stores all four of its full blocks, so 64 is what it can donate.
    assert naive.donor_capped_reusable_tokens(64, 64, bs) < \
        naive.reusable_tokens(64, 64, bs)
    # Anywhere else the shared length is at most donor_len - 1 and both agree.
    assert naive.donor_capped_reusable_tokens(60, 64, bs) == \
        naive.reusable_tokens(60, 64, bs)


def test_savings_from_hit_rate_is_a_prefill_number_not_an_end_to_end_one():
    # Kept separate from the hit rate so the two errors can be shown not to cancel:
    # this one scales with whatever prefill cost you feed it, so it cannot be
    # corrected by fixing the hit rate.
    a = naive.savings_from_hit_rate(0.74, 0.02, 1000)
    b = naive.savings_from_hit_rate(0.74, 0.04, 1000)
    assert b == 2 * a


def test_the_naive_functions_reject_impossible_inputs():
    with pytest.raises(ValueError):
        naive.hit_rate_from_shared_prefix(10, 0)
    with pytest.raises(ValueError):
        naive.blocks_for_prompt_ignoring_alignment(10, 0)
