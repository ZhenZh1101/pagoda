"""Deadline planner: projections, freezing before the data transition, late-run guard, persistence."""

from pagoda.optim import wsd_multiplier
from pagoda.schedule import DeadlinePlanner, parse_deadline, shard_capacity_steps


def planner(**kw):
    args = dict(deadline_ts=10_000, anneal=0.2, transition_steps=20, max_total=10 ** 6, initial_total=500,
                safety=1.0, freeze_margin=5)
    args.update(kw)
    return DeadlinePlanner(**args)


def test_parse_deadline():
    assert parse_deadline("+2h", now=1000.0) == 1000.0 + 7200
    assert parse_deadline("+90m", now=0.0) == 5400
    assert parse_deadline("1790000000") == 1790000000.0
    assert parse_deadline("2026-10-23T14:00") > 0


def test_replan_tracks_throughput_and_freezes():
    p = planner()
    p.replan(step=100, sec_per_step=1.0, now=100.0)          # 1 step/s -> 10k steps fit
    assert p.total == 100 + 9900 and not p.frozen
    p.replan(step=1000, sec_per_step=2.0, now=1000.0)        # slower: plan shrinks
    assert p.total == 1000 + 4500
    assert wsd_multiplier(1000, p.total, 300, decay_start=p.decay) == 1.0  # LR before the decay unaffected
    t = 1000.0
    for step in range(1001, 6000):                           # tick every step, replan every 10
        t += 2.0
        if p.tick(step):
            break
        if step % 10 == 0:
            p.replan(step=step, sec_per_step=2.0, now=t)
    assert p.frozen and p.transition == step + p.freeze_margin + 1
    total = p.total
    p.replan(step=step + 1, sec_per_step=0.1, now=t + 1)     # frozen: no further change
    assert p.total == total


def test_reserve_is_subtracted():
    p = planner()
    p.replan(step=0, sec_per_step=1.0, reserve_s=1000.0, now=0.0)
    assert p.total == 9000


def test_transient_late_projection_is_not_acted_on():
    # a single bad estimate (e.g. measured across a CORE run) says the run is late ...
    p = planner(deadline_ts=10_000, late_confirm_s=300)
    p.replan(step=1000, sec_per_step=1.0, now=1000.0)
    assert not p.frozen and p.total == 10_000
    msg = p.replan(step=1010, sec_per_step=50.0, now=1010.0)
    assert not p.frozen and "confirming" in msg
    # ... and the next normal estimate clears it
    p.replan(step=1020, sec_per_step=1.0, now=1020.0)
    assert not p.frozen and p.late_since is None


def test_confirmed_late_run_keeps_transition_and_cooldown():
    p = planner(deadline_ts=2000, transition_steps=200, initial_total=50_000, late_confirm_s=300,
                min_transition=50, min_cooldown=300)
    p.replan(step=1000, sec_per_step=1.0, now=1500.0)        # first late signal
    assert not p.frozen
    p.replan(step=1300, sec_per_step=1.0, now=1850.0)        # still late 350 s later -> freeze
    assert p.frozen and p.transition == 1306
    assert p.decay - p.transition >= 50                     # a real 50/50 transition
    assert p.total - p.decay >= 300                         # and a real cooldown, even past the deadline


def test_late_path_is_bounded_in_time():
    # toy scale: a 300-step cooldown floor would dwarf the run; the overrun bound shrinks it
    p = planner(deadline_ts=1000, transition_steps=4, initial_total=340, late_confirm_s=0,
                min_transition=50, min_cooldown=300, max_overrun_s=120)
    p.replan(step=180, sec_per_step=1.8, reserve_s=0.0, now=990.0)  # 10 s left
    assert p.frozen
    assert p.decay - p.transition <= 4                      # never longer than the configured transition
    assert (p.total - 180) * 1.8 <= 10 + 120 + 1.8          # ends within deadline + max_overrun
    assert p.total > p.decay                                # still a (short) cooldown


def test_deadline_passed_freezes_immediately():
    p = planner(deadline_ts=100, transition_steps=200, initial_total=50_000)
    p.replan(step=1000, sec_per_step=1.0, now=200.0)
    assert p.frozen and p.total - p.decay >= 300


def test_capacity_and_state_roundtrip():
    cap = shard_capacity_steps(1.95e9, 0.5e9, 2 ** 17, 0.2, 200)
    assert 18_000 < cap < 19_000
    assert shard_capacity_steps(1.95e9, 0, 2 ** 17, 0.25, 200) == int(1.95e9 // 2 ** 17)  # stage 1 only
    p = DeadlinePlanner(1e9, 0.2, 200, cap, 6000)
    p.sec_per_step = 9.7
    q = DeadlinePlanner(0, 0.2, 200, cap, 1)
    q.load_state_dict(p.state_dict())
    assert q.state_dict() == p.state_dict()
