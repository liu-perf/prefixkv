"""prefixkv -- how much of a prefix cache's saving is the cache, and how much is the traffic.

A prefix cache's hit rate is mostly a property of the workload. This library refuses
to report one on its own: every prompt token is attributed to exactly one of four
buckets -- served from cache, lost to eviction or pinning, lost to block granularity,
or novel and unservable by any cache -- and the four sum to the total.

    from prefixkv import PrefixCache, load_trace, decompose

    trace = load_trace("tests/traces/EXAMPLE_system_prompt.csv")
    cache = PrefixCache(capacity_blocks=4096, block_size=16)
    records = cache.run(trace)
    print(decompose(trace, cache, records)["verdict"])

Traces declare their own sharing structure in a readable column, which is what makes
the reuse ceiling computable by arithmetic instead of measurable by the thing under
test. Zero dependencies, no tokenizer, no model.
"""
from . import blocks, naive, oracle, rules, segments
from .attribution import AttributionError, compare, decompose
from .blocks import block_hash, cacheable_length, full_blocks
from .cache import PrefixCache, Stats, capacity_blocks_for
from .oracle import aligned_ceiling, ceilings, unaligned_ceiling
from .rules import audit
from .segments import Registry, Segment, SegmentError
from .trace import Request, Trace, TraceError
from .trace import load as load_trace

__version__ = "0.1.0"

__all__ = [
    "AttributionError", "PrefixCache", "Registry", "Request", "Segment",
    "SegmentError", "Stats", "Trace", "TraceError", "aligned_ceiling", "audit",
    "block_hash", "blocks", "cacheable_length", "capacity_blocks_for", "ceilings",
    "compare", "decompose", "full_blocks", "load_trace", "naive", "oracle", "rules",
    "segments", "unaligned_ceiling",
]
