"""Requests built out of named segments, so the sharing structure is visible.

A prefix cache works on token identity, so a trace of prompt *lengths* -- which is
all a scheduling simulator needs -- cannot drive one. It needs the actual tokens.

Writing token ids into a trace file makes it unreadable, and an unreadable fixture
is a fixture nobody checks. So a request here is a list of **named segments**:

    r0007,12.480,SYS:assistant_v3|FEW:math_5shot|USR:q0007,64

Each name expands, deterministically, to a fixed span of token ids. Two requests
that name `SYS:assistant_v3` share those tokens exactly; two that name different
`USR:` segments share none. The consequence is the property this whole library
depends on:

    **the trace declares its own sharing structure, in a column a human can read,
    and therefore its own reuse ceiling can be computed by arithmetic rather than
    measured by the thing under test.**

That is what makes the tool falsifiable. `tests/test_attribution.py` computes the
ceiling from the segment names alone and asserts the cache never exceeds it and
that an infinite cache reaches it.

Token ids are synthetic. There is no tokenizer here and no model -- what a prefix
cache cares about is only whether two token sequences are equal, and equality of
synthetic ids is exactly as sharp as equality of real ones. Using a real tokenizer
would add a dependency, a download, and a licence question, and would not make a
single number in this library more accurate.
"""

#: Segment kinds, and what each one means for sharing. The prefix is part of the
#: name so a reader of a trace can see the structure without a legend.
KINDS = {
    "SYS": "a system prompt; typically identical across a deployment",
    "FEW": "few-shot examples; identical within a task, different across tasks",
    "DOC": "retrieved or pasted context; sometimes shared, often not",
    "ANS": "a previous assistant reply; part of every later turn's prompt",
    "ACT": "a model-emitted tool call; part of every later step's prompt",
    "TOOL": "a tool call and its result, appended to an agent's context",
    "USR": "this request's own text; shared with nothing",
}

#: Distinct segment names must not collide in token space, so each gets its own
#: block of ids. 1e6 apart is arbitrary but enormous compared with any prompt, so
#: two segments can never accidentally look equal.
ID_STRIDE = 1_000_000


class SegmentError(ValueError):
    pass


class Segment:
    """A named, fixed span of token ids.

    Two Segments with the same name are the same tokens. That is the whole
    contract, and it is why the length is part of the name's definition rather
    than per-use: a segment that changed length between uses would share a name
    with something it is not.
    """

    __slots__ = ("kind", "name", "length", "_base")

    def __init__(self, kind, name, length):
        if kind not in KINDS:
            raise SegmentError(f"unknown segment kind {kind!r}; expected one of "
                               f"{', '.join(sorted(KINDS))}")
        if length < 1:
            raise SegmentError(f"{kind}:{name} has length {length}")
        self.kind = kind
        self.name = name
        self.length = int(length)
        self._base = None      # assigned by the registry

    @property
    def label(self):
        return f"{self.kind}:{self.name}"

    def token_ids(self):
        if self._base is None:
            raise SegmentError(f"{self.label} has not been registered")
        return range(self._base, self._base + self.length)

    def __repr__(self):
        return f"Segment({self.label!r}, {self.length})"


class Registry:
    """Assigns each segment name a disjoint span of token ids.

    Deterministic in insertion order, and insertion order comes from sorting the
    labels, so the same trace always produces the same ids on any machine and in
    any Python version. A registry that depended on dict ordering would make every
    hash in the library machine-specific.
    """

    def __init__(self):
        self._segments = {}

    def declare(self, kind, name, length):
        seg = Segment(kind, name, length)
        prior = self._segments.get(seg.label)
        if prior is not None:
            if prior.length != seg.length:
                raise SegmentError(
                    f"{seg.label} was declared with length {prior.length} and again "
                    f"with {seg.length}. A segment name is a promise that the tokens "
                    "are identical; two lengths mean two different things wearing "
                    "one name, and the sharing structure would be a lie.")
            return prior
        self._segments[seg.label] = seg
        self._reindex()
        return seg

    def _reindex(self):
        for i, label in enumerate(sorted(self._segments)):
            self._segments[label]._base = i * ID_STRIDE

    def get(self, label):
        try:
            return self._segments[label]
        except KeyError:
            raise SegmentError(f"{label} was never declared") from None

    def __len__(self):
        return len(self._segments)

    def labels(self):
        return sorted(self._segments)


def expand(segments):
    """Concatenate segments into one token-id list.

    The order matters and is the caller's: a prefix cache can only reuse a *prefix*,
    so putting the shared system prompt after the user's text destroys every hit.
    That failure is real and common -- it is what "the shared part must come first"
    means in practice -- so this function does not sort, reorder, or warn. It
    concatenates, and `prefixkv audit` is what points out the mistake.
    """
    out = []
    for seg in segments:
        out.extend(seg.token_ids())
    return out


def parse_spec(spec, registry, lengths):
    """Parse `SYS:a|FEW:b|USR:c` against a table of declared lengths."""
    segs = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SegmentError(f"{part!r} is not KIND:name")
        kind, name = part.split(":", 1)
        label = f"{kind}:{name}"
        if label not in lengths:
            raise SegmentError(f"{label} has no declared length")
        segs.append(registry.declare(kind, name, lengths[label]))
    if not segs:
        raise SegmentError(f"empty segment spec {spec!r}")
    return segs


__all__ = ["ID_STRIDE", "KINDS", "Registry", "Segment", "SegmentError", "expand",
           "parse_spec"]
