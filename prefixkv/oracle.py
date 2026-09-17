"""Upper bounds on reuse, computed without touching the cache implementation.

Two ceilings, and the gap between them is what block granularity costs:

`unaligned_ceiling`
    What a cache with infinite memory and **byte granularity** could serve. Pure
    property of the traffic plus the one-token logit rule. No real paged cache can
    reach it; it is the number that says how much reuse the traffic contains at all.

`aligned_ceiling`
    What a cache with infinite memory and a given `block_size` could serve. Pure
    property of the traffic and the block size. A real cache reaches this only when
    it never evicts anything.

Why these are computed from segment structure and not by running a cache
-----------------------------------------------------------------------
Because a bound computed by the thing under test is not a bound. Segments make an
independent computation possible and exact: distinct segment names occupy disjoint
token-id ranges, so two requests can never share *part* of a segment. The longest
common prefix of two requests is therefore **exactly the sum of the lengths of their
common leading segments** -- arithmetic on the `segments` column, with no hashing,
no blocks and no cache involved.

`tests/test_oracle.py` asserts the two routes agree: the aligned ceiling computed
here must equal what `PrefixCache` achieves with capacity set beyond any possible
need. The two share no code, so a disagreement is information -- which is the
property that distinguishes this from a self-consistency check.
"""
from .blocks import cacheable_length


def common_leading_segments(a, b):
    """Tokens in the common leading segments of two requests.

    Exact, because segments are atomic in token space. This is the whole reason the
    ceilings below are computable rather than measurable.
    """
    total = 0
    for sa, sb in zip(a.segments, b.segments):
        if sa.label != sb.label:
            break
        total += sa.length
    return total


def unaligned_ceiling(trace):
    """Byte-granular, infinite-memory reuse, in tokens.

    Defined as `aligned_ceiling(trace, 1)`, and that is the point rather than a
    shortcut: byte granularity is block granularity with a block size of one, so the
    two ceilings must be the same formula evaluated at different granularity, or the
    difference between them stops being "what block granularity costs".

    The first version computed this with a trie over segment labels. It was replaced
    because the trie could not apply the donor cap that `aligned_ceiling` applied,
    which made a run at block_size 1 report a small positive alignment loss where it
    must be exactly zero by definition.

    That diagnosis was right about the symptom and wrong about the cause: the donor
    cap itself was the bug (see `aligned_ceiling`), so the trie's answer had been
    correct and the "fix" was a patch on an error. The definition above is kept
    anyway, because it is the better definition -- it makes the two ceilings the same
    formula by construction, so alignment loss at block_size 1 is zero for a reason
    rather than by arithmetic coincidence.
    """
    return aligned_ceiling(trace, 1)


def aligned_ceiling(trace, block_size):
    """Block-granular, infinite-memory reuse, in tokens.

    Two caps, and they are **not** the same quantity. Getting that wrong is the one
    modelling error in this library that its own tests could not catch; the story is
    worth the space because it is the reason to distrust agreement between two
    routes that share an assumption.

    * The **requester** cap is `cacheable_length(req.prompt_tokens, block_size)`.
      At least one token must be recomputed to produce a logit, and the cached
      length must stay block-aligned, so a lookup can never return more than
      `((n - 1) // block_size) * block_size`. Verified against real vLLM's
      `get_computed_blocks` at 76 configurations, zero mismatches.

    * The **donor** cap is `(earlier.prompt_tokens // block_size) * block_size`:
      *all* of the donor's complete blocks. A donor stores every full block it has,
      including the one containing its own token `n - 1`.

    The first version used `cacheable_length` for both, on the reasoning that a
    block a request "will never be allowed to serve" is not worth caching. That
    reasoning conflates two different things: the logit rule limits what a request
    can **look up**, not what it **stores**, and a later request whose prompt is
    longer may legitimately hit the donor's last full block. Real vLLM caches every
    full block (measured directly: 18 of 18 configurations, donor still running or
    already freed), so the capped version reports a ceiling *below* real reuse --
    which means it is not a ceiling.

    Measured cost of the error, before the fix: 112 tokens of 52544 on
    `EXAMPLE_multiturn.csv` and 64 of 62800 on `EXAMPLE_agent.csv` at block_size 16,
    i.e. 0.2% and 0.1%. Small in magnitude, fatal in kind: `capture` could exceed
    100% against a real engine. After the fix this function's output equals real
    vLLM's `prefix_cache_hits` **exactly** on every trace in `tests/traces/`.

    Why the library's own cross-check missed it: `cache.py` applied the same wrong
    cap when inserting, so the oracle and the cache agreed with each other while
    both were wrong -- two implementations sharing no code but sharing a premise.
    `test_the_donor_cap_is_load_bearing` then pinned the error in place by asserting
    that removing the cap overstates reuse. Only an outside authority could break
    the tie, and that is what real vLLM was for.
    """
    seen = []
    total = 0
    for req in trace:
        cap = cacheable_length(req.prompt_tokens, block_size)
        best = 0
        for earlier in seen:
            shared = common_leading_segments(req, earlier)
            donatable = min((shared // block_size) * block_size,
                            (earlier.prompt_tokens // block_size) * block_size)
            if donatable > best:
                best = donatable
                if best >= cap:
                    break
        total += min(best, cap)
        seen.append(req)
    return total


def novel_tokens(trace):
    """Tokens no cache of any granularity could serve: prompt total minus the
    byte-granular ceiling. Includes each request's unavoidable logit token."""
    return trace.prompt_tokens - unaligned_ceiling(trace)


def ceilings(trace, block_size):
    prompt = trace.prompt_tokens
    unaligned = unaligned_ceiling(trace)
    aligned = aligned_ceiling(trace, block_size)
    if aligned > unaligned:
        raise AssertionError(
            f"the block-aligned ceiling ({aligned}) exceeded the byte-granular one "
            f"({unaligned}). Block granularity can only ever lose reuse, so one of "
            "the two computations is wrong.")
    return {
        "block_size": block_size,
        "prompt_tokens": prompt,
        "unaligned_ceiling": unaligned,
        "aligned_ceiling": aligned,
        "alignment_loss": unaligned - aligned,
        "novel_tokens": prompt - unaligned,
        "unaligned_rate": unaligned / prompt if prompt else 0.0,
        "aligned_rate": aligned / prompt if prompt else 0.0,
    }


__all__ = ["aligned_ceiling", "ceilings", "common_leading_segments", "novel_tokens",
           "unaligned_ceiling"]
