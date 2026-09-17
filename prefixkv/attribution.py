"""Where a prefix cache's saving actually came from.

The problem this module exists for: **a hit rate is mostly a property of the
traffic.** A cache reporting 90% on a trace where every request is identical has
told you about the trace. A cache reporting 12% on traffic that shares nothing may
be doing everything available to it. Comparing two caches by hit rate, or a cache
against itself across two workloads, is comparing the workloads.

So every prompt token is attributed to exactly one of four buckets:

    hit               served from the cache
    capacity_loss     the aligned ceiling had it and this cache did not
                      -- eviction, or blocks pinned by in-flight requests
    alignment_loss    a byte-granular cache would have reused it and block
                      granularity threw it away
    novel             no cache of any granularity could have served it
                      (first appearance, plus each request's logit token)

and the four sum to the prompt total. That identity is the check, and unlike
`L = lambda*W` on one timeline it **can fail**: `hit` comes from the cache's own
counters while the two ceilings come from arithmetic over the `segments` column that
touches no cache code, so an error on either side breaks the sum.

Two routes, not three. An earlier version of this note claimed three by counting a
trie walk that computed the byte-granular ceiling separately; that walk was removed
because it could not apply the donor cap and was therefore slightly wrong, and the
byte-granular ceiling is now the same pairwise formula evaluated at block size 1.

Two ratios come out of it, and they answer different questions:

    capture     = hit / aligned_ceiling      how much of the reuse available at this
                                             block size the cache actually got.
                                             This is the only number that is about
                                             the cache.
    structural  = aligned_ceiling / unaligned_ceiling
                                             how much block granularity costs on
                                             this traffic. A property of the traffic
                                             and the block size, not of the policy.

Report both, always. `capture` near 1.0 with a low hit rate means the cache is fine
and the traffic is not shareable -- which is a completely different engineering
conclusion from a high hit rate with poor capture.
"""
from . import oracle


class AttributionError(AssertionError):
    pass


def decompose(trace, cache, records=None):
    """Attribute every prompt token. `cache` must have already run over `trace`."""
    prompt = trace.prompt_tokens
    hit = cache.stats.hit_tokens
    if records is not None:
        # The records are produced per request; their sum must match the counters.
        # Two accumulations of the same quantity, and if they disagree the counters
        # are lying about something.
        rec_hit = sum(r["hit"] for r in records)
        rec_prompt = sum(r["prompt"] for r in records)
        if (rec_hit, rec_prompt) != (hit, prompt):
            raise AttributionError(
                f"per-request records sum to hit={rec_hit}, prompt={rec_prompt} but "
                f"the cache counters say hit={hit}, prompt={prompt}")

    c = oracle.ceilings(trace, cache.block_size)
    aligned = c["aligned_ceiling"]
    unaligned = c["unaligned_ceiling"]

    if hit > aligned:
        raise AttributionError(
            f"the cache served {hit} tokens from cache but the infinite-memory "
            f"ceiling at block_size={cache.block_size} is {aligned}. A finite cache "
            "cannot beat an infinite one; either the ceiling is computed wrongly or "
            "the cache is counting a hit it did not have.")

    capacity_loss = aligned - hit
    alignment_loss = unaligned - aligned
    novel = prompt - unaligned
    total = hit + capacity_loss + alignment_loss + novel
    if total != prompt:
        raise AttributionError(
            f"attribution does not partition the prompt tokens: "
            f"{hit} + {capacity_loss} + {alignment_loss} + {novel} = {total}, "
            f"expected {prompt}")

    return {
        "block_size": cache.block_size,
        "concurrency": cache.concurrency,
        "capacity_blocks": cache.capacity_blocks,
        "prompt_tokens": prompt,
        "buckets": {
            "hit": hit,
            "capacity_loss": capacity_loss,
            "alignment_loss": alignment_loss,
            "novel": novel,
        },
        "shares": {
            "hit": hit / prompt if prompt else 0.0,
            "capacity_loss": capacity_loss / prompt if prompt else 0.0,
            "alignment_loss": alignment_loss / prompt if prompt else 0.0,
            "novel": novel / prompt if prompt else 0.0,
        },
        "ceilings": c,
        # capture is about the cache; structural is about the traffic.
        "capture": hit / aligned if aligned else None,
        "structural": aligned / unaligned if unaligned else None,
        "verdict": _verdict(hit, aligned, unaligned, prompt),
    }


def _verdict(hit, aligned, unaligned, prompt):
    """One sentence naming which of the three things is the limit.

    Written as a sentence rather than a score because the three cases call for
    different work, and a single number would let a reader skip deciding which case
    they are in.
    """
    if unaligned == 0:
        return ("this traffic shares nothing: no prefix cache of any size or block "
                "granularity can help it, and a cache reporting a hit here would be "
                "reporting a bug")
    if aligned == 0:
        # The traffic shares something, but no prompt is long enough to fill even one
        # block at this block size, so nothing is cacheable. Falling through to the
        # capture branch would blame capacity for a limit that more memory cannot
        # move -- which is the opposite of the advice a reader needs.
        return ("this traffic has reuse, but at this block size no prompt fills a "
                "single block, so nothing is cacheable at all; the block size is the "
                "limit and no amount of cache memory changes it")
    capture = hit / aligned
    structural = aligned / unaligned
    if capture < 0.75:
        return (f"the cache captured {capture:.0%} of the reuse available to it -- "
                "the limit is capacity or pinning, so more cache memory (or less "
                "concurrency competing for it) is what moves this number")
    if structural < 0.75:
        return (f"the cache captured {capture:.0%} of what block granularity allows, "
                f"but block granularity itself only allows {structural:.0%} of the "
                "reuse in this traffic -- the limit is the block size, not the cache")
    return (f"the cache captured {capture:.0%} of a ceiling that is {structural:.0%} "
            f"of the traffic's total reuse; the limit is the traffic, which offers "
            f"{unaligned / prompt:.0%} reusable prompt tokens")


def compare(rows, key="hit"):
    """Rank configurations, and refuse to rank across different traffic.

    Comparing a hit rate measured on one trace against one measured on another is
    the mistake this whole module is about, so the function will not do it silently.
    """
    traces = {r["trace"] for r in rows}
    if len(traces) > 1:
        raise AttributionError(
            f"asked to rank configurations across {len(traces)} different traces "
            f"({', '.join(sorted(traces))}). Hit rate is mostly a property of the "
            "traffic, so a ranking that mixes traces measures the traces. Compare "
            "`capture` per trace instead, or rank within one trace.")
    return sorted(rows, key=lambda r: r["shares"][key], reverse=True)


__all__ = ["AttributionError", "compare", "decompose"]
