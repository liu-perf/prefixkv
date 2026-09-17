"""Every CLI line in ci.yml, run for real, exit code compared to the YAML.

The workflow file is the one artefact in this repository that no test could
see, and it had a bug that would have made CI red on every single push:

    prefixkv audit tests/traces/EXAMPLE_shared_last.csv --fail-on violation || test $? -eq 1

`--fail-on` is a **top-level** option. Written after the subcommand, argparse
never reaches the audit logic at all -- it exits 2 with a usage error. Worse,
the `|| test $? -eq 1` guard was there precisely to allow the *expected*
exit 1, and it then compared 2 against 1 and failed. The safeguard turned a
usage error into a failing step instead of absorbing it.

`pytest -q` was green, `ruff` was clean, the bundle's three axes all passed,
and the badge would have been red from the first push. Three documents in
this repository quote the same wrong order and assert "exit 1, which is
correct" -- the intent was right everywhere and the command line was wrong
everywhere. The same mistake was sitting in three of this series'
repositories at once, which is what makes it worth a guard rather than a
one-line correction.

So this file reads the workflow as text, pulls out every `prefixkv`
invocation, runs it in-process, and asserts the exit code agrees with how the
YAML is written.
"""
import os
import re
import shlex
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from prefixkv.cli import main  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI = os.path.join(REPO, ".github", "workflows", "ci.yml")

LINE = re.compile(r"^\s*(?P<bang>!\s*)?prefixkv\s+(?P<rest>.*)$")
PREFIXES = ("tests/", "profiles/", "data/")


def _commands():
    """(may_fail, argv) for every prefixkv invocation in the workflow."""
    with open(CI, encoding="utf-8") as fh:
        joined = re.sub(r"\\\n\s*", " ", fh.read())
    out = []
    for line in joined.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        rest = m.group("rest")
        # a `|| test $? -eq N` tail means the workflow already tolerates a
        # non-zero exit; strip it before parsing the arguments
        guarded = "||" in rest
        argv = [os.path.join(REPO, a) if a.startswith(PREFIXES) else a
                for a in shlex.split(rest.split("||")[0])]
        if any("*" in a for a in argv):      # globs are the shell's job
            continue
        out.append((bool(m.group("bang")) or guarded, argv))
    return out


def test_the_workflow_still_has_cli_lines_to_check():
    assert _commands(), f"no prefixkv invocation found in {CI} -- did the step move?"


def test_no_cli_line_in_ci_dies_on_a_usage_error(capsys):
    """Exit 2 is argparse refusing the command line, not the tool disagreeing.

    This is the one the workflow actually had, and the one the `|| test $? -eq
    1` guard could not absorb, because 2 is not 1.
    """
    for _may_fail, argv in _commands():
        code = main(list(argv))
        capsys.readouterr()
        assert code != 2, (
            "prefixkv " + " ".join(argv) + "\n  exits 2 -- argparse rejected "
            "the command line, so the tool never ran. Check option placement: "
            "top-level options go before the subcommand."
        )


def test_every_cli_line_exits_the_way_the_workflow_assumes(capsys):
    for may_fail, argv in _commands():
        code = main(list(argv))
        capsys.readouterr()
        if not may_fail:
            assert code == 0, (
                "prefixkv " + " ".join(argv) + f"\n  exits {code}, and ci.yml "
                "writes it as an expected success. Under `bash -e` that makes "
                "CI red on every push."
            )
