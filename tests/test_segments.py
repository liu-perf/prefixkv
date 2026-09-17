import pytest

from prefixkv.segments import KINDS, Registry, Segment, SegmentError, expand, parse_spec


def test_two_uses_of_a_name_are_the_same_tokens():
    reg = Registry()
    a = reg.declare("SYS", "v1", 40)
    b = reg.declare("SYS", "v1", 40)
    assert a is b
    assert list(a.token_ids()) == list(b.token_ids())


def test_the_same_name_with_two_lengths_is_refused():
    """A segment name is a promise that the tokens are identical.

    Two lengths mean two different things wearing one name, and every sharing
    number computed afterwards would be a lie. This is an error, not a warning.
    """
    reg = Registry()
    reg.declare("SYS", "v1", 40)
    with pytest.raises(SegmentError) as exc:
        reg.declare("SYS", "v1", 41)
    assert "promise" in str(exc.value)


def test_distinct_names_never_overlap_in_token_space():
    reg = Registry()
    segs = [reg.declare("USR", f"q{i}", 900) for i in range(20)]
    seen = set()
    for s in segs:
        ids = set(s.token_ids())
        assert not (ids & seen), f"{s.label} overlaps an earlier segment"
        seen |= ids


def test_ids_do_not_depend_on_declaration_order():
    """Two registries built in opposite orders must agree.

    Otherwise every block hash in the library would depend on the order a generator
    happened to visit segments in, and no fixture would be reproducible.
    """
    a, b = Registry(), Registry()
    for label in ("SYS:x", "FEW:y", "USR:z"):
        kind, name = label.split(":")
        a.declare(kind, name, 10)
    for label in ("USR:z", "FEW:y", "SYS:x"):
        kind, name = label.split(":")
        b.declare(kind, name, 10)
    for label in a.labels():
        assert list(a.get(label).token_ids()) == list(b.get(label).token_ids())


def test_an_unknown_kind_is_refused():
    with pytest.raises(SegmentError):
        Segment("HIST", "x", 10)          # removed on purpose; see make_traces.py
    for kind in KINDS:
        Segment(kind, "x", 10)


def test_a_zero_length_segment_is_refused():
    with pytest.raises(SegmentError):
        Segment("SYS", "x", 0)


def test_an_unregistered_segment_will_not_expand():
    with pytest.raises(SegmentError):
        Segment("SYS", "x", 4).token_ids()


def test_expand_preserves_order_and_does_not_sort():
    """Order is the caller's, and it is load-bearing: a prefix cache reuses a
    prefix, so reordering would change the answer. `expand` must not be helpful."""
    reg = Registry()
    s = reg.declare("SYS", "a", 3)
    u = reg.declare("USR", "b", 2)
    assert expand([s, u]) == list(s.token_ids()) + list(u.token_ids())
    assert expand([u, s]) == list(u.token_ids()) + list(s.token_ids())
    assert expand([s, u]) != expand([u, s])


def test_parse_spec_needs_a_declared_length():
    reg = Registry()
    with pytest.raises(SegmentError):
        parse_spec("SYS:a", reg, {})
    assert len(parse_spec("SYS:a|USR:b", reg, {"SYS:a": 4, "USR:b": 2})) == 2


def test_parse_spec_rejects_malformed_and_empty_specs():
    reg = Registry()
    with pytest.raises(SegmentError):
        parse_spec("justaname", reg, {"justaname": 3})
    with pytest.raises(SegmentError):
        parse_spec("", reg, {})
