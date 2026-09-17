"""Command line: `trace`, `ceilings`, `run`, `sweep`, `audit`.

Every human-readable line is `{location}: [{status}] {message}`. `--json` prints the
structure instead. `--fail-on` is a threshold, not an equality test.
"""
import argparse
import json
import sys

from . import oracle
from .attribution import AttributionError, decompose
from .cache import PrefixCache
from .rules import SEVERITY, worst_status
from .rules import audit as audit_rules
from .trace import TraceError, load


def _emit(lines, payload, args):
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        for location, status, message in lines:
            print(f"{location}: [{status}] {message}")
    if args.fail_on:
        worst = "ok"
        for _, status, _ in lines:
            if SEVERITY[status] > SEVERITY[worst]:
                worst = status
        if SEVERITY[worst] >= SEVERITY[args.fail_on]:
            return 1
    return 0


def _run_cache(args, trace):
    cache = PrefixCache(args.capacity_blocks, args.block_size, args.concurrency)
    records = cache.run(trace)
    return cache, records, decompose(trace, cache, records)


def cmd_trace(args):
    trace = load(args.trace)
    s = trace.summary()
    lines = [("trace", "info",
              f"{s['requests']} requests, {s['prompt_tokens']} prompt tokens, "
              f"{s['distinct_segments']} distinct segments of which "
              f"{s['reused_segments']} are used more than once")]
    if s["declared"]:
        lines.append(("trace.declared", "info", s["declared"][:200]))
    usage = trace.segment_usage()
    for label, n in list(usage.items())[:args.top]:
        length = trace.registry.get(label).length
        # The label carries a colon and the location field may not: the output
        # format is `{location}: [{status}] {message}`, and a colon inside the
        # location makes every downstream parser -- including this project's own
        # test -- read the line wrongly.
        lines.append((f"segment.{label.replace(':', '.')}", "info" if n > 1 else "ok",
                      f"used by {n} request(s), {length} tokens"
                      + (f", so {length * (n - 1)} token-appearances are repeats"
                         if n > 1 else " -- single use, never reusable")))
    if len(usage) > args.top:
        lines.append(("segment", "info",
                      f"{len(usage) - args.top} further segments not shown "
                      f"(--top {args.top})"))
    return lines, {"summary": s, "usage": usage}


def cmd_ceilings(args):
    trace = load(args.trace)
    lines, payload = [], {}
    for bs in args.block_sizes:
        c = oracle.ceilings(trace, bs)
        payload[str(bs)] = c
        lines.append((f"block_size={bs}", "info",
                      f"byte-granular ceiling {c['unaligned_rate']:.1%} of prompt "
                      f"tokens, block-aligned {c['aligned_rate']:.1%}; alignment "
                      f"costs {c['alignment_loss']} tokens, "
                      f"{c['novel_tokens']} tokens are novel"))
    best = max(payload.items(), key=lambda kv: kv[1]["aligned_ceiling"])
    lines.append(("ceilings", "ok",
                  f"the aligned ceiling is highest at block_size={best[0]}; this is a "
                  "property of the traffic and the block size, with no cache involved"))
    return lines, payload


def cmd_run(args):
    trace = load(args.trace)
    try:
        cache, records, report = _run_cache(args, trace)
    except AttributionError as exc:
        return [("attribution", "violation", str(exc))], {"error": str(exc)}
    b, sh = report["buckets"], report["shares"]
    lines = [
        ("cache", "info",
         f"block_size={report['block_size']}, {report['capacity_blocks']} blocks, "
         f"concurrency={report['concurrency']}"),
        ("hit", "info", f"{b['hit']} tokens ({sh['hit']:.1%} of prompt)"),
        ("capacity_loss", "info" if sh["capacity_loss"] < 0.02 else "warn",
         f"{b['capacity_loss']} tokens ({sh['capacity_loss']:.1%}) -- available at "
         "this block size but evicted or pinned"),
        ("alignment_loss", "info",
         f"{b['alignment_loss']} tokens ({sh['alignment_loss']:.1%}) -- reusable at "
         "byte granularity, thrown away by block granularity"),
        ("novel", "info",
         f"{b['novel']} tokens ({sh['novel']:.1%}) -- no cache of any granularity "
         "could serve these"),
        ("partition", "ok",
         f"the four buckets sum to {report['prompt_tokens']} prompt tokens exactly; "
         "`hit` comes from the cache's counters and the ceilings from arithmetic "
         "over the segments column, so this can fail"),
        ("capture", "info",
         f"{report['capture']:.1%} of the reuse available at this block size"
         if report["capture"] is not None else "undefined: no reuse was available"),
        ("structural", "info",
         f"block granularity allows {report['structural']:.1%} of this traffic's "
         "total reuse" if report["structural"] is not None else "undefined"),
        ("verdict", "info", report["verdict"]),
    ]
    if args.audit:
        for f in audit_rules(trace, args.block_size, report):
            lines.append((f.location, f.status, f"{f.rule} {f.message}"))
    return lines, {"report": report, "records": records if args.records else None}


def cmd_sweep(args):
    trace = load(args.trace)
    lines, rows = [], []
    for value in args.values:
        kw = dict(block_size=args.block_size, capacity_blocks=args.capacity_blocks,
                  concurrency=args.concurrency)
        kw[args.over] = int(value)
        cache = PrefixCache(kw["capacity_blocks"], kw["block_size"], kw["concurrency"])
        records = cache.run(trace)
        try:
            report = decompose(trace, cache, records)
        except AttributionError as exc:
            lines.append((f"{args.over}={value}", "violation", str(exc)))
            continue
        rows.append({args.over: int(value), **report["shares"],
                     "capture": report["capture"], "structural": report["structural"],
                     "trace": trace.source})
        lines.append((f"{args.over}={value}", "info",
                      f"hit {report['shares']['hit']:.1%}, capacity_loss "
                      f"{report['shares']['capacity_loss']:.1%}, alignment_loss "
                      f"{report['shares']['alignment_loss']:.1%}, capture "
                      f"{report['capture']:.1%}"
                      if report["capture"] is not None else "no reuse available"))
    if rows:
        best = max(rows, key=lambda r: r["hit"])
        lines.append(("sweep", "ok",
                      f"highest hit share at {args.over}={best[args.over]}; note that "
                      "this ranking is only valid within one trace, because hit rate "
                      "is mostly a property of the traffic"))
    return lines, {"rows": rows}


def cmd_audit(args):
    trace = load(args.trace)
    report = None
    if args.capacity_blocks:
        _, _, report = _run_cache(args, trace)
    findings = audit_rules(trace, args.block_size, report)
    lines = [(f.location, f.status, f"{f.rule} {f.message}") for f in findings]
    lines.append(("audit", worst_status(findings),
                  f"{len(findings)} finding(s) over {len(trace)} requests"))
    return lines, {"findings": [f.as_dict() for f in findings]}


def build_parser():
    ap = argparse.ArgumentParser(prog="prefixkv", description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-on", choices=sorted(SEVERITY))
    sub = ap.add_subparsers(dest="command", required=True)

    def cache_args(p):
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--capacity-blocks", type=int, default=4096)
        p.add_argument("--concurrency", type=int, default=1)

    p = sub.add_parser("trace", help="what this traffic shares")
    p.add_argument("trace")
    p.add_argument("--top", type=int, default=12)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("ceilings", help="reuse upper bounds, no cache involved")
    p.add_argument("trace")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    p.set_defaults(func=cmd_ceilings)

    p = sub.add_parser("run", help="run a cache and attribute every prompt token")
    p.add_argument("trace")
    cache_args(p)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--records", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sweep", help="vary one knob")
    p.add_argument("trace")
    cache_args(p)
    p.add_argument("--over", default="block_size",
                   choices=("block_size", "capacity_blocks", "concurrency"))
    p.add_argument("--values", nargs="+", default=["8", "16", "32", "64"])
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("audit", help="structural problems in the traffic")
    p.add_argument("trace")
    cache_args(p)
    p.set_defaults(func=cmd_audit)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        lines, payload = args.func(args)
    except FileNotFoundError as exc:
        print(f"input: [violation] cannot read {exc.filename}", file=sys.stderr)
        return 2
    except (TraceError, ValueError, OSError) as exc:
        print(f"input: [violation] {exc}", file=sys.stderr)
        return 2
    return _emit(lines, payload, args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
