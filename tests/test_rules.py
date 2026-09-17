from conftest import trace_path

from prefixkv import PrefixCache, decompose, load_trace
from prefixkv.rules import RULES, audit, worst_status


def findings_for(name, block_size=16, capacity=None, concurrency=1):
    tr = load_trace(trace_path(name))
    report = None
    if capacity:
        cache = PrefixCache(capacity, block_size, concurrency)
        report = decompose(tr, cache, cache.run(tr))
    return tr, audit(tr, block_size, report)


def test_pk001_finds_the_shared_part_put_last():
    """The rule exists for a mistake a hit rate cannot show you.

    The hit rate on this trace is 0%, which is also what you would see if the traffic
    simply shared nothing. PK001 is what distinguishes "there is nothing to reuse"
    from "there is plenty to reuse and the prompt order made it unreachable".
    """
    _, f = findings_for("EXAMPLE_shared_last.csv")
    pk001 = [x for x in f if x.rule == "PK001"]
    assert len(pk001) == 160                       # every request
    assert all(x.status == "violation" for x in pk001)
    assert "180 tokens" in pk001[0].message


def test_pk001_is_silent_when_the_order_is_right():
    _, f = findings_for("EXAMPLE_system_prompt.csv")
    assert not [x for x in f if x.rule == "PK001"]


def test_pk002_counts_the_alignment_waste():
    _, f = findings_for("EXAMPLE_system_prompt.csv", block_size=16)
    pk002 = [x for x in f if x.rule == "PK002"]
    assert pk002 and "block_size=16" in pk002[0].message
    # A 180-token shared prefix with 16-token blocks wastes 4 tokens, 160 times.
    assert "640" in pk002[0].message


def test_pk002_waste_grows_with_the_block_size():
    def total(bs):
        _, f = findings_for("EXAMPLE_system_prompt.csv", block_size=bs)
        pk002 = [x for x in f if x.rule == "PK002"]
        return int(pk002[0].message.split()[0]) if pk002 else 0
    assert total(64) > total(16)


def test_pk003_warns_when_nothing_is_shared():
    _, f = findings_for("EXAMPLE_no_sharing.csv")
    pk003 = [x for x in f if x.rule == "PK003"]
    assert pk003[0].status == "warn"
    assert "not the lever" in pk003[0].message


def test_pk003_passes_when_there_is_something_to_do():
    _, f = findings_for("EXAMPLE_agent.csv")
    assert [x for x in f if x.rule == "PK003"][0].status == "ok"


def test_pk004_blames_capacity_only_when_capacity_binds():
    _, tight = findings_for("EXAMPLE_agent.csv", capacity=48)
    _, loose = findings_for("EXAMPLE_agent.csv", capacity=4096)
    assert [x for x in tight if x.rule == "PK004"][0].status == "warn"
    assert [x for x in loose if x.rule == "PK004"][0].status == "ok"


def test_pk004_says_undefined_rather_than_zero_when_there_is_no_reuse():
    """"Captured 0% of nothing" and "captured 0% of a lot" need different words."""
    _, f = findings_for("EXAMPLE_no_sharing.csv", capacity=4096)
    pk004 = [x for x in f if x.rule == "PK004"][0]
    assert pk004.status == "unknown"
    assert "not the same as" in pk004.message


def test_pk005_reports_the_logit_rule_cost():
    _, f = findings_for("EXAMPLE_multiturn.csv")
    assert [x for x in f if x.rule == "PK005"]


def test_every_rule_fires_somewhere_and_stays_quiet_somewhere():
    """A rule that always fires and a rule that never fires are equally useless."""
    fired, quiet = set(), set()
    cases = [("EXAMPLE_shared_last.csv", 16, 4096),
             ("EXAMPLE_no_sharing.csv", 16, 4096),
             ("EXAMPLE_agent.csv", 16, 48),
             ("EXAMPLE_agent.csv", 16, 4096),
             ("EXAMPLE_system_prompt.csv", 16, 4096),
             ("EXAMPLE_multiturn.csv", 16, 4096),
             # Byte granularity: there is no alignment remainder and no whole-block
             # loss, so PK002 and PK005 have nothing to report. Without a case like
             # this, "PK005 fires" would be indistinguishable from "PK005 always
             # fires", which is the same as not having it.
             ("EXAMPLE_multiturn.csv", 1, 4096)]
    for name, bs, cap in cases:
        _, f = findings_for(name, bs, cap)
        noisy = {x.rule for x in f if x.status != "ok"}
        fired |= noisy
        quiet |= set(RULES) - noisy
    assert fired == set(RULES), f"never fired: {sorted(set(RULES) - fired)}"
    assert quiet == set(RULES), f"always fired: {sorted(set(RULES) - quiet)}"


def test_worst_status_is_a_maximum():
    _, f = findings_for("EXAMPLE_shared_last.csv")
    assert worst_status(f) == "violation"
    _, g = findings_for("EXAMPLE_agent.csv", capacity=4096)
    assert worst_status(g) in ("ok", "info")
