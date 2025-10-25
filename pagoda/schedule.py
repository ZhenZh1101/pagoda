"""Deadline-driven WSD: pick the total step count from measured throughput.

In a WSD schedule the LR is constant until the decay starts, so the total step
count can be re-estimated at any time during the constant phase without changing
anything already trained. The planner does exactly that: every few steps it
projects how many steps fit before the wall-clock deadline, places the decay and
the stage-2 switch at the last `anneal` fraction of that total, and freezes the
plan a few steps before the 50/50 data transition begins. After freezing nothing
moves, so the decay segment and the data switch stay consistent with what the
prefetcher has already produced.

Throughput comes from the trainer as seconds per step = median recent step time
+ amortised overhead (val, samples, saves); CORE runs are lumpy, so the trainer
passes their remaining cost in `reserve_s` instead. A projection that says the
run is already late must hold for `late_confirm_s` before it is acted on, and the
late path still keeps a real transition and cooldown (it may overrun slightly).

The deadline is an absolute Unix time stored in every checkpoint: a crash or a
night of sleep eats into the same budget instead of extending the run.
"""

import math
import re
import time
from datetime import datetime


def parse_deadline(text, now=None):
    """'+20h' / '+90m' / '+3600s' relative to now, a Unix timestamp, or an ISO time '2026-10-23T14:00'."""
    now = time.time() if now is None else now
    text = str(text).strip()
    m = re.fullmatch(r"\+(\d+(?:\.\d+)?)([hms])", text)
    if m:
        value, unit = float(m.group(1)), m.group(2)
        return now + value * {"h": 3600, "m": 60, "s": 1}[unit]
    if re.fullmatch(r"\d{9,}(\.\d+)?", text):
        return float(text)
    return datetime.fromisoformat(text).timestamp()


class DeadlinePlanner:
    """Holds the plan (total steps, decay start, transition start) and re-estimates it.

    Call `tick(step)` before every step (freezes the plan `freeze_margin` steps
    before the transition begins) and `replan(...)` every few steps.
    """

    def __init__(self, deadline_ts, anneal, transition_steps, max_total, initial_total,
                 safety=0.98, freeze_margin=5, late_confirm_s=300.0, min_transition=50,
                 min_cooldown=300, min_cooldown_frac=0.08, max_overrun_s=40 * 60.0):
        self.deadline = float(deadline_ts)
        self.anneal = anneal
        self.transition_steps = transition_steps
        self.max_total = int(max_total)
        self.safety = safety                # projected steps are discounted by this factor
        self.freeze_margin = freeze_margin  # steps of look-ahead the prefetcher may already have produced
        self.late_confirm_s = late_confirm_s
        self.min_transition = min_transition
        self.min_cooldown = min_cooldown
        self.min_cooldown_frac = min_cooldown_frac
        self.max_overrun_s = max_overrun_s  # the late path may end at most this long after the deadline
        self.frozen = False
        self.late_since = None
        self.sec_per_step = None            # last estimate, kept as a prior across restarts
        self._set_total(max(1, min(int(initial_total), self.max_total)))

    def _set_total(self, total):
        self.total = total
        self.decay = int(round(total * (1 - self.anneal)))
        self.transition = self.decay - self.transition_steps

    def tick(self, step):
        """Freeze once the transition is within the prefetch window. Returns a log line on freezing."""
        if not self.frozen and self.transition <= step + self.freeze_margin + 1:
            self.frozen = True
            return (f"plan frozen at step {step}: {self.total} steps, transition from {self.transition}, "
                    f"decay + stage 2 from {self.decay}")
        return None

    def projected_total(self, step, sec_per_step, reserve_s, now):
        remaining = self.deadline - now - reserve_s
        steps = int(max(0.0, remaining) / sec_per_step * self.safety)
        return max(step + 1, min(step + steps, self.max_total)), remaining

    def replan(self, step, sec_per_step, reserve_s=0.0, now=None):
        """Re-estimate the total from throughput. Returns a log line when the plan changed, else None."""
        if self.frozen or not sec_per_step or sec_per_step <= 0:
            return None
        now = time.time() if now is None else now
        self.sec_per_step = sec_per_step
        projected, remaining = self.projected_total(step, sec_per_step, reserve_s, now)
        earliest = step + self.freeze_margin + 1  # first step whose data has not been produced yet
        derived_transition = int(round(projected * (1 - self.anneal))) - self.transition_steps
        if derived_transition < earliest:
            # the run is late: act only if it persists (or the deadline has passed)
            if self.late_since is None:
                self.late_since = now
            if remaining > 0 and now - self.late_since < self.late_confirm_s:
                return f"late projection at step {step} ({remaining / 3600:.2f} h left); confirming before freezing"
            return self._freeze_late(step, projected, remaining, sec_per_step)
        self.late_since = None
        old = self.total
        self._set_total(projected)
        if abs(projected - old) >= max(1, old // 200):  # report changes > 0.5%
            return (f"replan: {old} -> {self.total} steps ({3600 / sec_per_step:.0f} steps/h, "
                    f"{remaining / 3600:.2f} h left; decay from {self.decay})")
        return None

    def _freeze_late(self, step, projected, remaining, sec_per_step):
        """Start the transition as soon as possible and keep a real cooldown, but never run more than
        `max_overrun_s` past the deadline: the floors shrink to fit that window."""
        self.transition = step + self.freeze_margin + 1
        budget = step + max(1, int((max(0.0, remaining) + self.max_overrun_s) / sec_per_step))
        room = max(0, projected - self.transition)
        trans_len = max(min(self.min_transition, self.transition_steps), min(self.transition_steps, room // 3))
        trans_len = min(trans_len, max(1, (budget - self.transition) // 4))
        self.decay = self.transition + trans_len
        cooldown = max(self.min_cooldown, int(self.min_cooldown_frac * self.decay))
        self.total = max(self.decay + 1, min(max(projected, self.decay + cooldown), budget))
        self.frozen = True
        return (f"late plan frozen at step {step}: transition {self.transition}-{self.decay}, decay to "
                f"{self.total} ({remaining / 3600:.2f} h left; up to {self.max_overrun_s / 60:.0f} min past the deadline "
                f"to keep a cooldown)")

    # ------------------------------------------------------------- persist
    def state_dict(self):
        return {"deadline": self.deadline, "total": self.total, "decay": self.decay,
                "transition": self.transition, "frozen": self.frozen, "sec_per_step": self.sec_per_step}

    def load_state_dict(self, s):
        self.deadline = float(s["deadline"])
        self.total, self.decay, self.transition = int(s["total"]), int(s["decay"]), int(s["transition"])
        self.frozen = bool(s["frozen"])
        self.sec_per_step = s.get("sec_per_step")
        self.late_since = None


def shard_capacity_steps(stage1_tokens, stage2_tokens, tokens_per_step, anneal, transition_steps):
    """Largest total step count for which neither stage would need a second epoch."""
    s1 = stage1_tokens / tokens_per_step
    if not stage2_tokens:  # stage-1-only run: every step reads stage 1
        return max(1, int(math.floor(s1)))
    s2 = stage2_tokens / tokens_per_step
    caps = [(s1 + transition_steps / 2) / (1 - anneal), (s2 - transition_steps / 2) / anneal]
    return max(1, int(math.floor(min(caps))))
