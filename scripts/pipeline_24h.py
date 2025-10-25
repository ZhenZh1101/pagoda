"""Unattended 24-hour pipeline: preflight -> pretrain (deadline mode) -> SFT -> eval ladder -> report.

    bash runs/train_24h.sh                        # normal use: tmux + caffeinate around this script
    python -m scripts.pipeline_24h --run d512_24h # the same, in the foreground
    python -m scripts.pipeline_24h --rehearsal    # ~25 min toy-scale dress rehearsal with injected faults

Time budget from the first launch T0 (absolute Unix times written once to
logs/<run>/pipeline/deadlines.json, so crashes and sleep come out of the same 24 h):

    00:00-00:05  preflight          disk, shards, power, quick tests
    00:05-19:45  pretrain+midtrain  d512 in deadline mode; the planner fits the step count to the window
    19:45-21:30  SFT                2 epochs; the 2nd is dropped automatically if it would not fit
    21:30-22:30  evals              fixed order, each task its own process with a time cap
    22:30-22:35  report             logs/<run>/report_24h.{md,json} + a macOS notification
    22:35-24:00  buffer             one crash/restart, an SFT rerun, or eval overrun

Failure handling: a crashed trainer is relaunched (it resumes from the newest
checkpoint with the same deadline); unrecoverable loss spikes get one relaunch at
0.7x LR with 200 steps of data skipped; a trainer whose log stops for 30 min is
killed and relaunched; at pretrain-deadline + 60 min the trainer is stopped and the
pipeline continues from its newest checkpoint. Re-running this script continues
at the first unfinished stage.
"""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from pagoda.common import CKPT_DIR, CODE_ROOT, DATA_ROOT, LOG_DIR, TOK_DIR, read_json, write_json

H, M = 3600.0, 60.0

# the 24 h recipe (see README "24 小时版"); every relaunch uses exactly these flags
RECIPE = {
    "model": "d512", "seq": 1024, "batch_tokens": 131072, "micro_batch": 16,
    "tokens": 849346560,            # first guess only (6,480 steps); the planner replaces it
    "anneal": 0.25, "transition_steps": 200, "warmup": 300,
    "matrix_lr": 0.02, "embed_lr": 1e-3,  # LR probe (12x256, full WSD): x1.0 beat x0.5 by 0.32 nats, x2.0 ahead of x1.0
    "mix": "configs/stage1.yaml", "mix_anneal": "configs/stage2.yaml", "val_dir": "val",
    "log_every": 50, "eval_every": 500, "eval_tokens": 2 ** 21, "sample_every": 2000,
    "core_frac": 0.25, "core_max_per_task": 500, "core_prior_s": 600,
    "save_every_min": 15, "keep": 3, "milestone_frac": 0.1, "first_save": 100, "auto_recover": 2,
    "replan_every": 10, "late_confirm_s": 300, "deadline_reserve_min": 5, "deadline_max_overrun_min": 45,
}
BUDGET = {"preflight": 5 * M, "pretrain": 19 * H + 45 * M, "sft": 21 * H + 30 * M, "eval": 22 * H + 30 * M,
          "report": 22 * H + 35 * M, "hard_end": 24 * H}
SFT_ARGS = ["--epochs", "2", "--micro-batch", "16"]
TIMING = {"stall_s": 30 * M, "kill_grace_s": 3 * M, "hard_stop_s": 60 * M, "relaunch_wait_s": 30,
          "max_relaunches": 6, "sft_grace_s": 20 * M}
EVAL_LIMIT = 0

# toy scale: same code paths, minutes instead of hours
REHEARSAL_RECIPE = {
    **RECIPE, "model": "tiny", "batch_tokens": 32768, "tokens": 32768 * 400, "transition_steps": 4, "warmup": 5,
    "mix": "configs/smoke/stage1.yaml", "mix_anneal": "configs/smoke/stage2.yaml", "val_dir": "smoke/val",
    "log_every": 5, "eval_every": 40, "eval_tokens": 65536, "sample_every": 200,
    "core_frac": 0.5, "core_max_per_task": 3, "core_prior_s": 10,
    "save_every_min": 1, "first_save": 20, "replan_every": 5, "late_confirm_s": 20, "deadline_reserve_min": 0.5,
    "deadline_max_overrun_min": 2,
}
REHEARSAL_BUDGET = {"preflight": 1 * M, "pretrain": 11 * M, "sft": 16 * M, "eval": 21 * M, "report": 22 * M,
                    "hard_end": 26 * M}
REHEARSAL_SFT_ARGS = ["--epochs", "2", "--micro-batch", "16", "--data", "smoke/sft", "--batch-tokens", "32768",
                      "--warmup", "3", "--eval-every", "10", "--log-every", "5"]
REHEARSAL_TIMING = {"stall_s": 3 * M, "kill_grace_s": 20, "hard_stop_s": 3 * M, "relaunch_wait_s": 2,
                    "max_relaunches": 6, "sft_grace_s": 2 * M}


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.rehearsal = args.rehearsal
        self.run = args.run
        self.recipe = dict(REHEARSAL_RECIPE if self.rehearsal else RECIPE)
        if args.matrix_lr is not None:
            self.recipe["matrix_lr"] = args.matrix_lr
        if args.embed_lr is not None:
            self.recipe["embed_lr"] = args.embed_lr
        self.timing = REHEARSAL_TIMING if self.rehearsal else TIMING
        self.sft_args = REHEARSAL_SFT_ARGS if self.rehearsal else SFT_ARGS
        self.eval_limit = 5 if self.rehearsal else EVAL_LIMIT
        self.dir = LOG_DIR / self.run / "pipeline"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "pipeline.log"
        self.stages_path = self.dir / "stages.json"
        self.stages = read_json(self.stages_path) if self.stages_path.exists() else {}
        self.deadlines = None  # fixed once preflight passes (a failed first launch must not start the clock)
        self.child = None
        self.stop = False
        self.faults = {"kill_before_freeze": False, "sleep": False, "kill_after_freeze": False, "delete_final": False}
        self._lock = open(self.dir / "lock", "w")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another pipeline for run {self.run} is already running (lock {self.dir / 'lock'})")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._on_signal)

    # ------------------------------------------------------------- basics
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except (BrokenPipeError, OSError):  # tee / the terminal may already be gone
            pass

    def _on_signal(self, signum, frame):
        self.stop = True
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)  # the trainer saves a checkpoint on SIGTERM
            except ProcessLookupError:
                pass
        self.log(f"pipeline got signal {signum}: stopping the current stage (re-run to continue)")

    def reap_orphans(self):
        """Stop leftover trainer / SFT / eval processes of this run (e.g. after the pipeline itself died)."""
        pat = rf"^[^ ]*python[0-9.]* -m scripts\.(train|sft|chat_eval|manual_test|identity_eval) .*--run {self.run}( |$)"
        pids = [int(p) for p in subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
                if int(p) != os.getpid()]
        for pid in pids:
            self.log(f"stopping a leftover process of this run (pid {pid}); it saves first if it is the trainer")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        t_end = time.time() + self.timing["kill_grace_s"]
        while pids and time.time() < t_end:
            pids = [p for p in pids if _alive(p)]
            time.sleep(1)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _deadlines(self):
        path = self.dir / "deadlines.json"
        if path.exists() and "pretrain" in self.stages:  # training has started: keep the original T0
            d = read_json(path)
            self.recipe["matrix_lr"] = d.get("matrix_lr", self.recipe["matrix_lr"])
            self.recipe["embed_lr"] = d.get("embed_lr", self.recipe["embed_lr"])
            return d
        t0 = time.time()
        budget = REHEARSAL_BUDGET if self.rehearsal else BUDGET
        d = {"t0": t0, **{k: t0 + v for k, v in budget.items()}, "matrix_lr": self.recipe["matrix_lr"],
             "embed_lr": self.recipe["embed_lr"], "rehearsal": self.rehearsal}
        write_json(path, d)
        self.log("deadlines: " + ", ".join(f"{k} {time.strftime('%a %H:%M', time.localtime(v))}"
                                           for k, v in d.items() if k in budget))
        return d

    def mark(self, stage, status, **extra):
        entry = self.stages.get(stage, {})
        entry.update(status=status, **extra)
        entry.setdefault("start", time.time())
        if status in ("done", "degraded", "failed", "skipped"):
            entry["end"] = time.time()
        self.stages[stage] = entry
        write_json(self.stages_path, self.stages)

    def done(self, stage):
        return self.stages.get(stage, {}).get("status") in ("done", "degraded", "skipped")

    def left(self, key):
        return self.deadlines[key] - time.time()

    def py(self, *module_args):
        return [sys.executable, "-m", *module_args]

    def spawn(self, cmd, log_name):
        logf = open(LOG_DIR / self.run / log_name, "a")
        logf.write(f"\n===== {time.ctime()} $ {' '.join(cmd)}\n")
        logf.flush()
        self.child = subprocess.Popen(cmd, cwd=CODE_ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                      start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        return self.child

    def terminate(self, proc, grace):
        if proc.poll() is not None:
            return proc.returncode
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            return proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            return proc.wait()

    def run_capped(self, cmd, log_name, cap_s):
        """Run to completion or kill after cap_s. Returns (returncode or None on timeout/stop, seconds)."""
        if self.stop:
            return None, 0.0
        t0 = time.time()
        proc = self.spawn(cmd, log_name)
        try:
            rc = proc.wait(timeout=max(1.0, cap_s))
        except subprocess.TimeoutExpired:
            self.log(f"  time cap {cap_s / 60:.0f} min reached: stopping {cmd[2]}")
            self.terminate(proc, 30)
            rc = None
        self.child = None
        return rc, time.time() - t0

    # ------------------------------------------------------------- stages
    def preflight(self):
        problems, warnings = [], []
        free_gb = shutil.disk_usage(DATA_ROOT).free / 1e9
        if free_gb < 10:
            problems.append(f"only {free_gb:.0f} GB free on the data volume")
        r = self.recipe
        stage1 = TOK_DIR / ("smoke/stage1" if self.rehearsal else "stage1")
        stage2 = TOK_DIR / ("smoke/stage2" if self.rehearsal else "stage2")
        sft = TOK_DIR / ("smoke/sft" if self.rehearsal else "sft")
        for p in (stage1 / "manifest.json", stage2 / "manifest.json", TOK_DIR / r["val_dir"] / "manifest.json",
                  sft / "train_manifest.json"):
            if not p.exists():
                problems.append(f"missing {p}")
        others = subprocess.run(["pgrep", "-f", r"^[^ ]*python[0-9.]* -m scripts\.(train|sft|bench)( |$)"],
                                capture_output=True, text=True).stdout.split()
        if others:
            msg = f"other training processes are running (pids {' '.join(others)})"
            (warnings if self.rehearsal else problems).append(msg)
        try:
            batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
            if "AC Power" not in batt:
                warnings.append("not on AC power")
            pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
            if any(line.split()[-1] == "1" for line in pm.splitlines() if "lowpowermode" in line):
                warnings.append("Low Power Mode is on")
        except (OSError, IndexError):
            pass
        tests = subprocess.run(self.py("pytest", "-q", "tests/test_schedule.py", "tests/test_dataloader.py"),
                               cwd=CODE_ROOT, capture_output=True, text=True)
        if tests.returncode != 0:
            problems.append("quick tests failed:\n" + tests.stdout[-2000:])
        for w in warnings:
            self.log(f"preflight warning: {w} (the 24 h budget assumes full speed)")
        if problems:
            for p in problems:
                self.log(f"preflight FAILED: {p}")
            return False
        self.log(f"preflight ok ({free_gb:.0f} GB free)")
        return True

    def train_cmd(self, extra=()):
        r = self.recipe
        flags = ["--model", r["model"], "--seq", r["seq"], "--batch-tokens", r["batch_tokens"],
                 "--micro-batch", r["micro_batch"], "--tokens", r["tokens"], "--anneal", r["anneal"],
                 "--transition-steps", r["transition_steps"], "--warmup", r["warmup"],
                 "--matrix-lr", r["matrix_lr"], "--embed-lr", r["embed_lr"], "--mix", r["mix"],
                 "--mix-anneal", r["mix_anneal"], "--val-dir", r["val_dir"], "--run", self.run,
                 "--log-every", r["log_every"], "--eval-every", r["eval_every"], "--eval-tokens", r["eval_tokens"],
                 "--sample-every", r["sample_every"], "--core-frac", r["core_frac"],
                 "--core-max-per-task", r["core_max_per_task"], "--core-prior-s", r["core_prior_s"],
                 "--save-every-min", r["save_every_min"], "--keep", r["keep"], "--milestone-frac", r["milestone_frac"],
                 "--first-save", r["first_save"], "--auto-recover", r["auto_recover"],
                 "--replan-every", r["replan_every"], "--late-confirm-s", r["late_confirm_s"],
                 "--deadline-reserve-min", r["deadline_reserve_min"],
                 "--deadline-max-overrun-min", r["deadline_max_overrun_min"],
                 "--deadline", repr(self.deadlines["pretrain"]), "--resume", "auto", *extra]
        return self.py("scripts.train", *map(str, flags))

    def pretrain_finished(self):
        p = LOG_DIR / self.run / "pretrain_done.json"
        if not p.exists():
            return False
        d = read_json(p)
        return d.get("step") == d.get("total_steps") and d.get("checkpoint") not in (None, "None")

    def latest_meta(self):
        from pagoda import checkpoint as C
        path = C.latest(CKPT_DIR / self.run)
        return (C.load_meta(path), path) if path else (None, None)

    def usable_or_failed(self, why):
        """'degraded' if the newest checkpoint is far enough along to be worth SFT, else 'failed'."""
        meta, _ = self.latest_meta()
        if meta and meta["step"] >= 0.25 * max(1, meta.get("total_steps") or 1):
            self.log(f"{why}: continuing with the newest checkpoint (step {meta['step']}/{meta.get('total_steps')})")
            return "degraded"
        self.log(f"{why}: the newest checkpoint (step {meta['step'] if meta else None}) is too early to be useful; "
                 f"stopping instead of fine-tuning an untrained model")
        return "failed"

    def pretrain(self):
        relaunches, spike_relaunched, last_crash_step = 0, False, None
        extra, spike_flags, spike_from = [], [], None
        train_log = LOG_DIR / self.run / "train.log"
        while not self.stop:
            if self.pretrain_finished():
                return "done"
            if time.time() > self.deadlines["pretrain"] + self.timing["hard_stop_s"]:
                self.log("pretrain is past its deadline + hard-stop margin: continuing with the newest checkpoint")
                return "degraded"
            if spike_flags and (self.latest_meta()[0] or {}).get("step") != spike_from:
                spike_flags, spike_from = [], None  # the reduced-LR run has saved: its checkpoint carries the state
            flags = spike_flags + extra
            proc = self.spawn(self.train_cmd(flags), "train.stdout.log")
            started = time.time()
            self.log(f"trainer started (pid {proc.pid}){' with ' + ' '.join(flags) if flags else ''}")
            extra = []
            hard_stopped = False
            while proc.poll() is None:
                time.sleep(5)
                now = time.time()
                if self.rehearsal:
                    self.inject_pretrain_faults(proc, now)
                stale = now - max(started, train_log.stat().st_mtime if train_log.exists() else started)
                if stale > self.timing["stall_s"]:
                    self.log(f"watchdog: train.log silent for {stale / 60:.0f} min; restarting the trainer")
                    self.terminate(proc, self.timing["kill_grace_s"])
                elif now > self.deadlines["pretrain"] + self.timing["hard_stop_s"]:
                    self.log(f"hard stop: pretraining ran {self.timing['hard_stop_s'] / 60:.0f} min past its deadline; "
                             f"stopping it (it saves first)")
                    hard_stopped = True
                    self.terminate(proc, self.timing["kill_grace_s"])
            rc = proc.returncode
            self.child = None
            if self.stop:
                return None
            if rc == 0 and self.pretrain_finished():
                return "done"
            if hard_stopped:
                return "degraded"
            meta, _ = self.latest_meta()
            step = meta["step"] if meta else None
            if rc == 2:
                if spike_relaunched:
                    return self.usable_or_failed("trainer stopped again after unrecoverable loss spikes")
                scale = (meta or {}).get("lr_scale", 1.0) * 0.7
                spike_flags, spike_from = ["--lr-scale", f"{scale:.6f}", "--skip-steps", "200"], step
                spike_relaunched = True
                self.log(f"trainer exited after loss spikes: relaunching once at lr_scale {scale:.3f}, skipping 200 steps")
                continue
            relaunches += 1
            self.log(f"trainer exited with code {rc} at checkpoint step {step}; relaunch {relaunches}/{self.timing['max_relaunches']}")
            if relaunches > self.timing["max_relaunches"]:
                return self.usable_or_failed(f"{relaunches - 1} relaunches used up")
            if step is not None and step == last_crash_step and not spike_flags:
                extra = ["--skip-steps", "50"]  # same place twice: step over the data
            last_crash_step = step
            time.sleep(self.timing["relaunch_wait_s"])
        return None

    def sft(self):
        final = CKPT_DIR / self.run / "sft_final.safetensors"
        cmd = self.py("scripts.sft", "--base-run", self.run, "--run", self.run, *self.sft_args,
                      "--deadline", repr(self.deadlines["sft"]))
        cap = self.left("sft") + self.timing["sft_grace_s"]
        rc, secs = self.run_capped(cmd, "sft.stdout.log", cap)
        if rc == 0 and not final.exists() and (CKPT_DIR / self.run / "sft_epoch1.safetensors").exists():
            # finished but the final link is gone (the rehearsal deletes it): relink the best finished epoch
            hist = read_json(LOG_DIR / self.run / "sft_history.json")
            best = min(hist["epochs"], key=lambda e: e["val_loss"])["epoch"]
            final.symlink_to(f"sft_epoch{best}.safetensors")
            self.log(f"sft_final was missing after SFT: relinked to epoch {best}")
        if rc == 0 and final.exists():
            return "done"
        if self.stop:
            return None
        if final.exists():
            self.log(f"SFT exited with {rc}; using the best finished epoch ({os.readlink(final)})")
            return "degraded"
        if self.left("hard_end") > 60 * M or self.rehearsal:
            self.log(f"SFT exited with {rc} and no epoch finished: rerunning with 1 epoch")
            args = [a for a in self.sft_args]
            args[args.index("--epochs") + 1] = "1"
            rc, _ = self.run_capped(self.py("scripts.sft", "--base-run", self.run, "--run", self.run, *args),
                                    "sft.stdout.log", min(self.left("hard_end") - 20 * M, 75 * M) if not self.rehearsal
                                    else 4 * M)
            if self.stop:
                return None
            if final.exists():
                return "degraded"
        return "failed"

    def evals(self):
        lim = self.eval_limit

        def tl(task, n):
            return ["--task-limits", f"{task}={n}"]

        base_limit = ["--limit", str(lim)] if lim else []
        results = {}

        def chat_eval(source, tasks, cap, extra=()):
            cmd = self.py("scripts.chat_eval", "--run", self.run, "--source", source, "--tasks", tasks,
                          *base_limit, *extra)
            rc, secs = self.run_capped(cmd, "eval.stdout.log", cap)
            results[f"{source}:{tasks}"] = {"rc": rc, "seconds": round(secs), "extra": list(extra)}
            self.log(f"  eval {source} {tasks}: {'ok' if rc == 0 else 'rc ' + str(rc)} in {secs / 60:.1f} min")

        def run_tool(name, module_args, cap):
            rc, secs = self.run_capped(self.py(*module_args), "eval.stdout.log", cap)
            results[name] = {"rc": rc, "seconds": round(secs)}
            self.log(f"  {name}: rc {rc} in {secs / 60:.1f} min")

        scale = 0.1 if self.rehearsal else 1.0
        if (CODE_ROOT / "scripts" / "identity_eval.py").exists():
            run_tool("identity_eval", ["scripts.identity_eval", "--run", self.run, "--source", "sft"], 10 * M * scale + 60)
        run_tool("manual_test", ["scripts.manual_test", "--run", self.run, "--source", "sft"], 10 * M * scale + 60)
        chat_eval("sft", "arc_easy,arc_challenge,mmlu", 10 * M * scale + 60)
        gsm_extra = tl("gsm8k_tool", 400) if self.left("eval") < 30 * M and not self.rehearsal else []
        chat_eval("sft", "gsm8k_tool", 40 * M * scale + 60, gsm_extra)
        if self.left("eval") > 10 * M or self.rehearsal:
            chat_eval("sft", "humaneval", 8 * M * scale + 60)
        to_end = self.left("hard_end") - 30 * M
        if to_end > 35 * M or self.rehearsal:
            chat_eval("base", "hellaswag", 35 * M * scale + 60)
        elif to_end > 10 * M:
            chat_eval("base", "hellaswag", 10 * M, tl("hellaswag", 2000))
        else:
            self.log("  skipping HellaSwag (the in-training CORE includes a 500-item HellaSwag)")
        return results

    # ------------------------------------------------------------- faults (rehearsal only)
    def inject_pretrain_faults(self, proc, now):
        f, d = self.faults, self.deadlines
        start, end = d["preflight"], d["pretrain"]
        frac = (now - start) / max(1.0, end - start)
        frozen = "plan frozen" in (LOG_DIR / self.run / "train.log").read_text() if (LOG_DIR / self.run / "train.log").exists() else False
        if not f["kill_before_freeze"] and 0.30 < frac and not frozen and C_latest(self.run):
            f["kill_before_freeze"] = True
            self.log("FAULT: kill -9 the trainer (constant phase)")
            os.killpg(proc.pid, signal.SIGKILL)
        elif f["kill_before_freeze"] and not f["sleep"] and frac > 0.45 and not frozen:
            f["sleep"] = True
            self.log("FAULT: SIGSTOP the trainer for 60 s (simulated sleep)")
            os.killpg(proc.pid, signal.SIGSTOP)
            time.sleep(60)
            os.killpg(proc.pid, signal.SIGCONT)
        elif f["sleep"] and not f["kill_after_freeze"] and frozen and (self.latest_meta()[0] or {}).get("plan", {}).get("frozen"):
            f["kill_after_freeze"] = True
            self.log("FAULT: kill -9 the trainer (after the plan froze)")
            os.killpg(proc.pid, signal.SIGKILL)

    # ------------------------------------------------------------- report
    def report(self):
        run_log = LOG_DIR / self.run
        metrics = []
        if (run_log / "metrics.jsonl").exists():
            metrics = [json.loads(l) for l in (run_log / "metrics.jsonl").read_text().splitlines() if l.strip()]
        meta, ckpt = self.latest_meta()
        done = read_json(run_log / "pretrain_done.json") if (run_log / "pretrain_done.json").exists() else {}
        vals = [m for m in metrics if m["kind"] == "val"]
        cores = [m for m in metrics if m["kind"] == "core"]
        plans = [m for m in metrics if m["kind"] == "plan"]
        n_params = 38_535_168 if self.recipe["model"] == "d512" else None
        tokens = (meta or {}).get("stats", {}).get("tokens_seen", 0)
        sft_hist = read_json(run_log / "sft_history.json") if (run_log / "sft_history.json").exists() else {}
        evals = {}
        for src in ("base", "sft"):
            p = run_log / f"eval_{src}.json"
            if p.exists():
                evals[src] = read_json(p)["tasks"]
        ident = read_json(run_log / "identity_eval_sft.json") if (run_log / "identity_eval_sft.json").exists() else None
        rep = {
            "run": self.run, "rehearsal": self.rehearsal, "deadlines": self.deadlines, "stages": self.stages,
            "recipe": self.recipe, "pretrain": {"finished": bool(done), "steps": (meta or {}).get("step"),
                                                "total_steps": (meta or {}).get("total_steps"), "tokens": tokens,
                                                "tokens_per_param": tokens / n_params if n_params else None,
                                                "recoveries": (meta or {}).get("stats", {}).get("recoveries"),
                                                "lr_scale": (meta or {}).get("lr_scale"),
                                                "plan": (meta or {}).get("plan"), "checkpoint": str(ckpt)},
            "val": [{"step": v["step"], "bpb": v["val_bpb"], "loss": v["val_loss"]} for v in vals],
            "core": [{"step": c["step"], "core": c["core"]} for c in cores],
            "plan_events": [{"step": p["step"], "message": p["message"]} for p in plans],
            "sft": sft_hist, "evals": evals, "identity": ident,
            "manual_sheet": str(run_log / "manual_tests_sft.md"),
        }
        write_json(run_log / "report_24h.json", rep)
        lines = [f"# Pagoda 24 h run `{self.run}`" + (" (rehearsal)" if self.rehearsal else ""), ""]
        t0 = self.deadlines["t0"]
        lines += ["| stage | status | start | end | deadline |", "|---|---|---|---|---|"]
        for name in ("preflight", "pretrain", "sft", "eval", "report"):
            s = self.stages.get(name, {})
            fmt = lambda t: f"T+{(t - t0) / 3600:.2f} h" if t else ""
            lines.append(f"| {name} | {s.get('status', '')} | {fmt(s.get('start'))} | {fmt(s.get('end'))} | "
                         f"{fmt(self.deadlines.get(name))} |")
        pt = rep["pretrain"]
        lines += ["", f"- pretraining: {pt['steps']} / {pt['total_steps']} steps, {tokens / 1e9:.3f}B tokens"
                  + (f" ({pt['tokens_per_param']:.1f} tokens/param)" if pt["tokens_per_param"] else "")
                  + f", recoveries {pt['recoveries']}, lr_scale {pt['lr_scale']}",
                  f"- plan: {json.dumps(pt['plan'])}"]
        if vals:
            lines.append(f"- val bpb: first {vals[0]['val_bpb']:.4f} (step {vals[0]['step']}) -> final "
                         f"{vals[-1]['val_bpb']:.4f} (step {vals[-1]['step']})")
        if cores:
            lines.append("- CORE: " + ", ".join(f"{c['core']:.4f}@{c['step']}" for c in cores))
        for p in plans[-8:]:
            lines.append(f"- plan event @{p['step']}: {p['message']}")
        if sft_hist.get("epochs"):
            lines.append("- SFT: " + ", ".join(f"epoch {e['epoch']} val {e['val_loss']:.4f}" for e in sft_hist["epochs"])
                         + f"; final = epoch {sft_hist.get('final', {}).get('epoch')}")
        lines += ["", "| model | task | result | n |", "|---|---|---|---|"]
        for src, tasks in evals.items():
            for task, r in tasks.items():
                score = {k: v for k, v in r.items() if k in ("accuracy", "acc_norm", "pass@1", "core")}
                lines.append(f"| {src} | {task} | {json.dumps(score)} | {r.get('n', '')} |")
        if ident is not None:
            settings = ident.get("settings") or {}
            ok = bool(settings) and all(v.get("summary", {}).get("pass") for v in settings.values())
            rep["identity_pass"] = ok
            write_json(run_log / "report_24h.json", rep)
            lines.append(f"\n- identity acceptance: {'PASS' if ok else 'FAIL'} (logs/{self.run}/identity_eval_sft.md)")
        lines.append(f"- manual tests to grade: logs/{self.run}/manual_tests_sft.md")
        (run_log / "report_24h.md").write_text("\n".join(lines) + "\n")
        self.log(f"report: {run_log / 'report_24h.md'}")
        subprocess.run(["osascript", "-e", f'display notification "pipeline finished: logs/{self.run}/report_24h.md" '
                                           f'with title "Pagoda 24 h"'], capture_output=True)

    # ------------------------------------------------------------- main
    def main(self):
        self.reap_orphans()
        if not self.done("preflight") or "pretrain" not in self.stages:
            self.stages.pop("preflight", None)
            self.mark("preflight", "running")
            if not self.preflight():
                self.mark("preflight", "failed")
                return 1
            self.mark("preflight", "done")
        self.deadlines = self._deadlines()
        self.log(f"pipeline {self.run} ({'rehearsal' if self.rehearsal else '24 h'}), "
                 f"T0 {time.ctime(self.deadlines['t0'])}, matrix_lr {self.recipe['matrix_lr']}")
        if not self.done("pretrain"):
            self.mark("pretrain", "running")
            status = self.pretrain()
            if status is None:
                return 3
            self.mark("pretrain", status)
            if status == "failed":
                self.log("no usable checkpoint: stopping")
                self.report()
                return 1
        if not self.done("sft"):
            self.mark("sft", "running")
            if self.rehearsal:
                self._sft_fault_thread()
            status = self.sft()
            if status is None:
                return 3
            self.mark("sft", status)
        if not self.done("eval") and (CKPT_DIR / self.run / "sft_final.safetensors").exists():
            self.mark("eval", "running")
            results = self.evals()
            if self.stop:
                return 3
            self.mark("eval", "done", results=results)
        self.mark("report", "done")
        self.report()
        end = time.time()
        self.log(f"pipeline finished at T+{(end - self.deadlines['t0']) / 3600:.2f} h "
                 f"({'within' if end <= self.deadlines['hard_end'] else 'AFTER'} the {('rehearsal' if self.rehearsal else '24 h')} budget)")
        return 0

    def _sft_fault_thread(self):
        """Rehearsal: delete sft_final while epoch 2 runs; the pipeline must still end with a final model."""
        import threading

        def worker():
            final = CKPT_DIR / self.run / "sft_final.safetensors"
            ep1 = CKPT_DIR / self.run / "sft_epoch1.safetensors"
            t_end = time.time() + 10 * M
            while time.time() < t_end and not self.stop:
                if ep1.exists() and (final.exists() or final.is_symlink()):
                    time.sleep(2)
                    final.unlink(missing_ok=True)
                    self.faults["delete_final"] = True
                    self.log("FAULT: deleted sft_final during SFT epoch 2")
                    return
                time.sleep(1)
        threading.Thread(target=worker, daemon=True).start()


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def C_latest(run):
    from pagoda import checkpoint as C
    return C.latest(CKPT_DIR / run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="default d512_24h (rehearsal: rehearsal_24h)")
    ap.add_argument("--rehearsal", action="store_true", help="toy model, smoke shards, minutes-long budget, injected faults")
    ap.add_argument("--matrix-lr", type=float, default=None, help="override the recipe's Muon LR (stored on first launch)")
    ap.add_argument("--embed-lr", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="rehearsal only: delete the previous rehearsal run first")
    args = ap.parse_args()
    args.run = args.run or ("rehearsal_24h" if args.rehearsal else "d512_24h")
    if args.fresh:
        if not args.rehearsal:
            raise SystemExit("--fresh is only for rehearsals")
        shutil.rmtree(CKPT_DIR / args.run, ignore_errors=True)
        shutil.rmtree(LOG_DIR / args.run, ignore_errors=True)
    sys.exit(Pipeline(args).main())


if __name__ == "__main__":
    main()
