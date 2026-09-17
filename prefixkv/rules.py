"""Five things worth telling somebody about their traffic before they tune a cache.

Each rule fires on a structural property of the trace or the run, and each one names
what to do about it. None of them is a score: a rule that only ranked would let a
reader skip deciding which of the five situations they are actually in.

The statuses are the same five the rest of this series uses, and `unknown` sits
above `info` and below `warn` -- a missing measurement is not a pass.
"""
from .blocks import alignment_waste, cacheable_length
from .oracle import ceilings

SEVERITY = {"ok": 0, "info": 1, "unknown": 2, "warn": 3, "violation": 4}

#: Capture below this and the cache -- not the traffic, not the block size -- is the
#: limit. 0.75 is a reporting threshold, not a physical one; it exists so the CLI can
#: say "look here first", and both sides of it are reported either way.
CAPTURE_FLOOR = 0.75
#: Below this share of reusable prompt tokens, prefix caching is not the lever.
SHARE_FLOOR = 0.05


class Finding:
    __slots__ = ("rule", "location", "status", "message")

    def __init__(self, rule, location, status, message):
        self.rule = rule
        self.location = location
        self.status = status
        self.message = message

    def as_dict(self):
        return {"rule": self.rule, "location": self.location,
                "status": self.status, "message": self.message}

    def __str__(self):
        return f"{self.location}: [{self.status}] {self.rule} {self.message}"


def pk001_shared_behind_unique(trace):
    """A reusable segment sitting behind a single-use one can never be reused.

    A prefix cache reuses a *prefix*. Put the user's text before the shared system
    prompt and the shared part becomes unreachable -- the hit rate collapses to zero
    and nothing about the cache configuration will recover it. This is the most
    common real mistake in prompt assembly, it is invisible in a hit-rate number, and
    it is a one-line fix.
    """
    usage = trace.segment_usage()
    out = []
    for req in trace:
        blocked = 0
        seen_unique = None
        for seg in req.segments:
            if usage[seg.label] == 1 and seen_unique is None:
                seen_unique = seg.label
            elif seen_unique is not None and usage[seg.label] > 1:
                blocked += seg.length
        if blocked:
            out.append(Finding(
                "PK001", req.id, "violation",
                f"{blocked} tokens of reusable segments sit behind the single-use "
                f"segment {seen_unique}; a prefix cache reuses a prefix, so those "
                "tokens can never be served from cache. Move the shared part first."))
    return out


def pk002_alignment_waste(trace, block_size):
    """Shared prefixes that are not a multiple of the block size."""
    usage = trace.segment_usage()
    out = []
    shared_run = {}
    for req in trace:
        run = 0
        for seg in req.segments:
            if usage[seg.label] > 1:
                run += seg.length
            else:
                break
        if run:
            shared_run[req.id] = run
    if not shared_run:
        return out
    waste = {rid: alignment_waste(n, block_size) for rid, n in shared_run.items()}
    total = sum(waste.values())
    if total:
        worst = max(waste.items(), key=lambda kv: kv[1])
        out.append(Finding(
            "PK002", "trace", "info",
            f"{total} prompt tokens across {sum(1 for v in waste.values() if v)} "
            f"requests fall outside a whole block at block_size={block_size} and are "
            f"recomputed every time; the worst single request loses {worst[1]} "
            f"tokens. Shared prefixes that are a multiple of the block size lose "
            "none of this."))
    return out


def pk003_nothing_shared(trace, block_size):
    c = ceilings(trace, block_size)
    if c["unaligned_rate"] < SHARE_FLOOR:
        return [Finding(
            "PK003", "trace", "warn",
            f"only {c['unaligned_rate']:.1%} of prompt tokens are reusable even with "
            "byte granularity and infinite memory. Prefix caching is not the lever "
            "for this traffic; any hit rate reported on it is measuring noise.")]
    return [Finding("PK003", "trace", "ok",
                    f"{c['unaligned_rate']:.1%} of prompt tokens are reusable in "
                    "principle, so there is something for a cache to do")]


def pk004_capture(report):
    cap = report.get("capture")
    if cap is None:
        return [Finding("PK004", "cache", "unknown",
                        "no reuse was available, so capture is undefined -- which is "
                        "not the same as a cache that captured nothing")]
    if cap < CAPTURE_FLOOR:
        return [Finding(
            "PK004", "cache", "warn",
            f"captured {cap:.1%} of the reuse available at this block size; "
            f"{report['shares']['capacity_loss']:.1%} of all prompt tokens were lost "
            "to eviction or to blocks pinned by in-flight requests. More cache "
            "memory, or less concurrency competing for it, is what moves this.")]
    return [Finding("PK004", "cache", "ok",
                    f"captured {cap:.1%} of the reuse available at this block size")]


def pk005_logit_rule_cost(trace, block_size):
    """The whole-block loss when a prompt is exactly a previously cached sequence.

    The loss is the remainder on most requests and a whole block on the ones whose
    prompt length is an exact multiple of the block size -- about one in
    `block_size`. It is small in aggregate and invisible unless somebody computes it,
    which is why it is reported as `info` rather than dressed up as a finding.
    """
    out = []
    exact = 0
    lost = 0
    specs = {}
    for req in trace:
        key = req.spec
        prefix_of = specs.get(key)
        if prefix_of is not None:
            exact += 1
        specs[key] = req.id
        cap = cacheable_length(req.prompt_tokens, block_size)
        if req.prompt_tokens > 1:
            lost += req.prompt_tokens - 1 - cap
    if lost:
        out.append(Finding(
            "PK005", "trace", "info",
            f"{lost} tokens across the trace are lost to the rule that the cached "
            f"length must stay block-aligned after reserving one token for the "
            f"logit. At block_size={block_size} a prompt that is exactly a whole "
            "number of blocks long can reuse one block fewer than it contains."))
    if exact:
        out.append(Finding(
            "PK005", "trace", "info",
            f"{exact} requests repeat an earlier request's exact segment list; those "
            "are where the logit rule costs a whole block rather than a few tokens"))
    return out


def audit(trace, block_size, report=None):
    findings = []
    findings += pk003_nothing_shared(trace, block_size)
    findings += pk001_shared_behind_unique(trace)
    findings += pk002_alignment_waste(trace, block_size)
    findings += pk005_logit_rule_cost(trace, block_size)
    if report is not None:
        findings += pk004_capture(report)
    return findings


def worst_status(findings):
    worst = "ok"
    for f in findings:
        if SEVERITY[f.status] > SEVERITY[worst]:
            worst = f.status
    return worst


RULES = ["PK001", "PK002", "PK003", "PK004", "PK005"]

__all__ = ["CAPTURE_FLOOR", "Finding", "RULES", "SEVERITY", "SHARE_FLOOR", "audit",
           "pk001_shared_behind_unique", "pk002_alignment_waste",
           "pk003_nothing_shared", "pk004_capture", "pk005_logit_rule_cost",
           "worst_status"]
