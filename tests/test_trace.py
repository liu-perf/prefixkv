import os

import pytest
from conftest import ALL_TRACES, TRACE_DIR, trace_path

from prefixkv.trace import Trace, TraceError, load, write


def test_every_shipped_trace_declares_itself_inside_the_data():
    for name in ALL_TRACES:
        tr = load(trace_path(name))
        assert "SYNTHETIC EXAMPLE" in tr.declared, (
            f"{name} does not say it is synthetic inside the file; a declaration in "
            "the filename does not survive being pasted into a slide")


def test_the_generator_is_deterministic(tmp_path):
    import runpy
    import shutil
    scratch = tmp_path / "traces"
    shutil.copytree(TRACE_DIR, scratch)
    before = {n: (scratch / n).read_bytes() for n in ALL_TRACES}
    mod = runpy.run_path(os.path.join(TRACE_DIR, "make_traces.py"), run_name="not_main")
    mod["main"].__globals__["HERE"] = str(scratch)
    mod["main"]()
    for n in ALL_TRACES:
        assert (scratch / n).read_bytes() == before[n], f"{n} drifted from its generator"


def test_the_sharing_structure_is_readable_in_the_file():
    """The property the whole library rests on: a human can compute the ceiling.

    If the segments column ever stopped naming the structure, the ceilings would
    have to be measured by the thing under test instead of derived.
    """
    text = open(trace_path("EXAMPLE_system_prompt.csv"), encoding="utf-8").read()
    assert "# LEN SYS:assistant_v3=180" in text
    assert "SYS:assistant_v3|USR:q0000" in text


def test_declared_lengths_that_nothing_uses_are_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("# LEN SYS:a=4\n# LEN SYS:unused=9\n"
                 "request_id,arrival_s,segments,output_tokens\nr0,0,SYS:a,4\n",
                 encoding="utf-8")
    with pytest.raises(TraceError) as exc:
        load(str(p))
    assert "no request uses" in str(exc.value)


def test_a_missing_column_names_itself(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("arrival_s,segments\n0,SYS:a\n", encoding="utf-8")
    with pytest.raises(TraceError) as exc:
        load(str(p))
    assert "output_tokens" in str(exc.value)


def test_rows_out_of_order_are_sorted(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("# LEN SYS:a=4\nrequest_id,arrival_s,segments,output_tokens\n"
                 "b,5,SYS:a,1\na,1,SYS:a,1\n", encoding="utf-8")
    assert [r.id for r in load(str(p))] == ["a", "b"]


def test_round_trip(tmp_path):
    original = load(trace_path("EXAMPLE_tiny.csv"))
    p = tmp_path / "rt.csv"
    write(str(p), original, declaration="SYNTHETIC EXAMPLE -- round trip")
    back = load(str(p))
    assert [r.spec for r in back] == [r.spec for r in original]
    assert back.prompt_tokens == original.prompt_tokens


def test_segment_usage_separates_reusable_from_single_use():
    tr = load(trace_path("EXAMPLE_system_prompt.csv"))
    usage = tr.segment_usage()
    assert usage["SYS:assistant_v3"] == len(tr)
    assert sum(1 for v in usage.values() if v == 1) == len(tr)


def test_an_empty_trace_is_an_error():
    with pytest.raises(TraceError):
        Trace([], None)


def test_token_ids_are_cached_not_recomputed():
    tr = load(trace_path("EXAMPLE_tiny.csv"))
    r = list(tr)[0]
    assert r.token_ids() is r.token_ids()


def test_the_control_and_its_mirror_have_identical_token_totals():
    a = load(trace_path("EXAMPLE_system_prompt.csv"))
    b = load(trace_path("EXAMPLE_shared_last.csv"))
    assert a.prompt_tokens == b.prompt_tokens
    assert {r.spec for r in a} != {r.spec for r in b}
