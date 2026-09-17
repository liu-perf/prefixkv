"""The first version, kept because it is wrong in the usual ways.

`tests/test_naive.py` asserts these are **still** wrong against the measured
attribution. If a later change makes one of them right, the test fails and somebody
has to explain why.

That guard has fired once, and the explanation is worth reading: the entry that used
to live here as "reusable tokens ignoring the donor cap" turned out to be *correct*,
and the version labelled "what it should have said" was the wrong one. The two
swapped places when real vLLM showed that a donor caches every full block. See
`donor_capped_reusable_tokens` -- the only error in this library that no internal
check could catch, because the oracle and the simulated cache shared its premise.
"""
from .blocks import cacheable_length


def hit_rate_from_shared_prefix(shared_tokens, prompt_tokens):
    """shared / prompt, called a hit rate.

    The estimate everybody starts from, and it ignores both structural losses:
    block granularity means only `shared // block_size` blocks are reusable, and one
    token per request can never be a hit. On a 180-token system prompt with 16-token
    blocks it overstates the reusable part by 4 tokens; on a hundred distinct
    20-token preambles it overstates by a quarter.
    """
    if prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    return shared_tokens / prompt_tokens


def savings_from_hit_rate(hit_rate, prefill_cost_per_token_ms, prompt_tokens):
    """hit_rate x cost x tokens, called a time saving.

    Wrong in a way that gets worse as the cache gets better: a cache hit removes the
    *prefill* of those tokens, but the request still pays for its own decode, its
    queueing, and the scheduler step it lands in. Quoting this as an end-to-end
    saving attributes the whole request's latency to prefill.

    Kept as a separate function from the hit rate so the two errors can be shown not
    to cancel.
    """
    return hit_rate * prefill_cost_per_token_ms * prompt_tokens


def blocks_for_prompt_ignoring_alignment(prompt_tokens, block_size):
    """prompt / block_size, rounded up, as the number of reusable blocks.

    Two errors in one line. It rounds *up*, so it counts the trailing partial block
    -- which is never cacheable, because only complete blocks are hashed. And it
    ignores the logit rule. Compare with `blocks.cacheable_length`, which floors
    after subtracting one.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return -(-prompt_tokens // block_size)


def donor_capped_reusable_tokens(shared_tokens, donor_prompt_tokens, block_size):
    """min(floor(shared / block) x block, cacheable_length(donor)) -- capping the
    donor by the logit rule.

    **This one shipped.** It is here because it is the most instructive error in the
    library's history, and because it is the only one that survived every internal
    check.

    The reasoning was: an earlier request only cached up to its own
    `cacheable_length`, so it cannot donate a block it never stored. The premise is
    false. The logit rule limits what a request can **look up**, not what it
    **stores**; a donor stores every full block it has, and a later request with a
    longer prompt can hit the donor's last full block legitimately.

    Measured on real vLLM: it caches every full block, 18 of 18 configurations, donor
    still running or already freed. With this cap in place, `aligned_ceiling`
    reported 52544 on `EXAMPLE_multiturn.csv` while vLLM actually served 52656 --
    a ceiling the engine beat by 112 tokens, which makes it not a ceiling.

    Why every internal check passed: `cache.py` applied the same cap when inserting,
    so the oracle and the simulated cache agreed with each other. Two
    implementations, no shared code, one shared premise. **Agreement between two
    routes is only evidence if they can disagree.** Compare with
    `reusable_tokens` below, which is what the arithmetic should have been.
    """
    return min((shared_tokens // block_size) * block_size,
               cacheable_length(donor_prompt_tokens, block_size))


def reusable_tokens(shared_tokens, donor_prompt_tokens, block_size):
    """What the previous function should have said, for side-by-side comparison.

    The donor contributes all of its complete blocks; the requester's own logit-rule
    cap is applied separately by the caller.
    """
    return min((shared_tokens // block_size) * block_size,
               (donor_prompt_tokens // block_size) * block_size)


__all__ = ["blocks_for_prompt_ignoring_alignment", "donor_capped_reusable_tokens",
           "hit_rate_from_shared_prefix", "reusable_tokens", "savings_from_hit_rate"]
