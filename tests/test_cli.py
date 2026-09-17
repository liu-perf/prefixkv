import json
import re

import pytest
from conftest import trace_path

from prefixkv.cli import main

LINE = re.compile(r"^[^:]+: \[(ok|info|unknown|warn|violation)\] .+$")


def run_cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_every_human_line_has_the_house_shape(capsys):
    for cmd in (["trace", trace_path("EXAMPLE_tiny.csv")],
                ["ceilings", trace_path("EXAMPLE_tiny.csv")],
                ["run", trace_path("EXAMPLE_tiny.csv"), "--audit"],
                ["audit", trace_path("EXAMPLE_tiny.csv")]):
        code, out, _ = run_cli(capsys, *cmd)
        assert code == 0
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert lines
        for ln in lines:
            assert LINE.match(ln), f"{cmd[0]}: line does not match the format: {ln!r}"


def test_json_is_parseable(capsys):
    for cmd in ("trace", "ceilings", "run", "audit"):
        code, out, _ = run_cli(capsys, "--json", cmd, trace_path("EXAMPLE_tiny.csv"))
        assert code == 0
        json.loads(out)


def test_run_prints_the_partition_and_both_ratios(capsys):
    code, out, _ = run_cli(capsys, "run", trace_path("EXAMPLE_system_prompt.csv"))
    assert code == 0
    assert "partition: [ok]" in out
    assert "capture: [info]" in out
    assert "structural: [info]" in out
    assert "verdict: [info]" in out


def test_the_control_reports_no_reuse_and_says_so(capsys):
    code, out, _ = run_cli(capsys, "run", trace_path("EXAMPLE_no_sharing.csv"))
    assert code == 0
    assert "novel" in out
    assert "shares nothing" in out
    assert "undefined: no reuse was available" in out


def test_audit_finds_the_shared_part_put_last_and_fails_the_gate(capsys):
    code, out, _ = run_cli(capsys, "--fail-on", "violation", "audit",
                           trace_path("EXAMPLE_shared_last.csv"))
    assert code == 1
    assert "PK001" in out
    assert "Move the shared part first" in out


def test_the_same_content_in_the_right_order_passes_the_gate(capsys):
    code, _, _ = run_cli(capsys, "--fail-on", "violation", "audit",
                         trace_path("EXAMPLE_system_prompt.csv"))
    assert code == 0


def test_fail_on_is_a_threshold_not_an_equality(capsys):
    # The control trace emits a [warn] from PK003 and no violation.
    code, _, _ = run_cli(capsys, "--fail-on", "warn", "audit",
                         trace_path("EXAMPLE_no_sharing.csv"))
    assert code == 1
    code, _, _ = run_cli(capsys, "--fail-on", "violation", "audit",
                         trace_path("EXAMPLE_no_sharing.csv"))
    assert code == 0


def test_sweep_emits_one_row_per_value_and_warns_about_cross_trace_ranking(capsys):
    code, out, _ = run_cli(capsys, "sweep", trace_path("EXAMPLE_multiturn.csv"),
                           "--over", "block_size", "--values", "8", "16", "64")
    assert code == 0
    assert "only valid within one trace" in out
    code, out, _ = run_cli(capsys, "--json", "sweep",
                           trace_path("EXAMPLE_multiturn.csv"),
                           "--over", "block_size", "--values", "8", "16", "64")
    assert len(json.loads(out)["rows"]) == 3


def test_ceilings_needs_no_cache_at_all(capsys):
    code, out, _ = run_cli(capsys, "ceilings", trace_path("EXAMPLE_agent.csv"),
                           "--block-sizes", "1", "16", "256")
    assert code == 0
    assert "no cache involved" in out
    assert out.count("block_size=") >= 3


def test_a_block_size_nothing_can_fill_blames_the_block_size(capsys):
    code, out, _ = run_cli(capsys, "run", trace_path("EXAMPLE_agent.csv"),
                           "--block-size", "4096", "--capacity-blocks", "100000")
    assert code == 0
    assert "the block size is the limit" in out


def test_a_missing_file_exits_two_not_one(capsys):
    code, _, err = run_cli(capsys, "trace", "does-not-exist.csv")
    assert code == 2
    assert "cannot read" in err


def test_a_malformed_trace_exits_two(capsys, tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("arrival_s\n1\n", encoding="utf-8")
    code, _, err = run_cli(capsys, "trace", str(p))
    assert code == 2
    assert "missing column" in err


def test_an_unknown_subcommand_is_a_usage_error(capsys):
    with pytest.raises(SystemExit):
        run_cli(capsys, "explain", trace_path("EXAMPLE_tiny.csv"))
