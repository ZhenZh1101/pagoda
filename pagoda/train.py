"""Pretraining + mid-training in one WSD run (plan sections 6, 7, 10).

Warmup 300 steps -> constant LR -> at 80% of the steps the LR decays linearly to
0 while the data switches to the stage-2 mix (with a 200-step 50/50 transition
right before the switch). Muon for block matrices, AdamW for the embedding,
global-norm clipping at 1.0, bf16 compute with fp32 master weights.

Monitoring: train loss / grad norm / tok/s / peak memory every `log_every`
steps, val loss + bpb every `eval_every`, fixed-prompt samples every
`sample_every`, CORE at every 25% of progress. Full checkpoints every
`save_every_min` minutes and at step `first_save` (newest 3 kept) and bf16
milestones every 10%.

Deadline mode (`--deadline`, used by the 24 h pipeline): instead of a fixed
token count, pagoda.schedule.DeadlinePlanner re-estimates the total from the
measured step time during the constant-LR phase and freezes it just before the
50/50 transition, so the run ends at the deadline whatever the throughput.
"""

import json
import math
import signal
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from pagoda import checkpoint as C
from pagoda.common import CKPT_DIR, LOG_DIR, TOK_DIR, load_yaml, log
from pagoda.dataloader import Prefetcher, StagedBatches, val_batches
from pagoda.model import GPT, GPTConfig, token_losses
from pagoda.optim import (build_optimizer, clip_by_global_norm, decay_start_step, global_grad_norm, muon_momentum,
                          set_lr, set_muon_momentum, wsd_multiplier)
from pagoda.schedule import DeadlinePlanner, parse_deadline, shard_capacity_steps
from pagoda.tokenizer import get_tokenizer

DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def model_config_from_args(args, vocab_size):
    if args.depth or args.width:
        depth = args.depth or 12
        width = args.width or 640
        hidden = int(round(2.7 * width / 64)) * 64
        cfg = GPTConfig(n_layer=depth, d_model=width, n_head=width // 64, mlp_hidden=hidden)
    else:
        cfg = GPTConfig.from_preset(args.model)
    cfg.vocab_size = vocab_size
    cfg.seq_len = args.seq
    cfg.tie_embeddings = not args.untied
    return cfg


def resolve_shard_dir(mix_path):
    """configs/stageN.yaml -> tok/<shard_dir>."""
    if mix_path is None:
        return None
    cfg = load_yaml(mix_path)
    return TOK_DIR / cfg["shard_dir"]


def build_step_fns(model, opt, clip, token_bytes, compile=True):
    """(micro, apply, evaluate) for gradient accumulation; shared by pretraining and SFT.

    micro(x, y, scale) -> (sum of token losses * scale, n tokens, grads of that sum)
    apply(grads)       -> global grad norm (before clipping); clips and updates in place
    evaluate(x, y)     -> (sum ce, n tokens, sum ce over byte-carrying targets, n bytes)
    """
    def loss_sum(model, x, y, scale):
        ce, valid = token_losses(model, x, y)
        return ce.sum() * scale, valid.sum()

    grad_fn = nn.value_and_grad(model, loss_sum)

    def micro(x, y, scale):
        (loss, n), grads = grad_fn(model, x, y, scale)
        return loss, n, grads

    def apply(grads):
        grads, norm = clip_by_global_norm(grads, clip)
        opt.update(model, grads)
        return norm

    def evaluate(x, y):
        ce, valid = token_losses(model, x, y)
        nbytes = token_bytes[mx.where(valid, y, 0)] * valid
        return ce.sum(), valid.sum(), (ce * (nbytes > 0)).sum(), nbytes.sum()

    if not compile:
        return micro, apply, evaluate
    state = [model.state, opt.state]
    # nn.value_and_grad writes the traced parameters into the model, so `micro` must
    # also list the model state as an output or the model is left holding tracers
    return (mx.compile(micro, inputs=[model.state], outputs=[model.state]),
            mx.compile(apply, inputs=state, outputs=state),
            mx.compile(evaluate, inputs=[model.state]))


STOPPED = "stopped"
DONE = "done"


def _arg(args, name, default):
    """New flags read through getattr so Trainers rebuilt from old checkpoint args still work."""
    return getattr(args, name, default)


class Trainer:
    def __init__(self, args):
        self.args = args
        mx.set_memory_limit(int(args.memory_limit_gb * 1024 ** 3))
        mx.random.seed(args.seed)
        self.tok = get_tokenizer()
        self.cfg = model_config_from_args(args, self.tok.vocab_size)
        self.dtype = DTYPES[args.dtype]
        self.model = GPT(self.cfg, compute_dtype=self.dtype)
        mx.eval(self.model.parameters())

        self.tokens_per_step = args.batch_tokens
        self.micro_tokens = args.micro_batch * args.seq
        if self.tokens_per_step % self.micro_tokens:
            raise SystemExit(f"--batch-tokens {args.batch_tokens} must be a multiple of micro-batch*seq {self.micro_tokens}")
        self.accum = self.tokens_per_step // self.micro_tokens
        self.total_steps = args.max_steps or max(1, math.ceil(args.tokens / self.tokens_per_step))
        self.decay_start = decay_start_step(self.total_steps, args.anneal)
        self.run_dir = CKPT_DIR / args.run
        self.log_dir = LOG_DIR / args.run
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "train.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"

        tc = {"matrix_lr": args.matrix_lr, "muon_momentum": 0.95, "adam_beta1": 0.9, "adam_beta2": 0.95,
              "adam_eps": 1e-8, "embed_lr_tied": args.embed_lr, "embed_lr_untied": 0.2, "head_lr_untied": 0.004}
        self.opt, self.lr_groups = build_optimizer(self.model, tc)

        s1 = resolve_shard_dir(args.mix)
        s2 = resolve_shard_dir(args.mix_anneal)
        self.val_dir = TOK_DIR / args.val_dir
        self.data = StagedBatches(s1, s2, args.micro_batch, args.seq, self.decay_start, args.transition_steps, args.seed)
        self.token_bytes = mx.array(self.tok.token_bytes(), dtype=mx.float32)

        self.step = 0
        self.lr_scale = 1.0 if args.lr_scale is None else args.lr_scale
        self.stats = {"tokens_seen": 0, "train_time": 0.0, "best_val_bpb": None, "recoveries": 0, "skipped_steps": 0,
                      "core_done": 0, "milestones_done": 0, "skip_armed": None, "t_core": None}
        self.loss_scale = 2.0 ** 15 if args.dtype == "fp16" else 1.0
        self._good_steps = 0

        # deadline mode (24 h pipeline): the planner owns total steps, decay start and the data switch
        self.planner = None
        deadline = _arg(args, "deadline", "")
        if deadline:
            self.planner = self._new_planner(parse_deadline(deadline))

        self._dirty = False  # startup changed the state (skip / LR override): save before the first step
        if args.resume:
            self._resume(args.resume)
        self._apply_plan()  # before any skip, so skipped steps are drawn with the real data switch
        if args.skip_steps:
            self._skip_once(args.skip_steps)
        else:
            self.stats["skip_armed"] = None  # a launch without the flag re-arms it

        self._build_compiled()
        self._stop_requested = False
        self._last_save = time.time()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self._on_signal)

    # ------------------------------------------------------------------ setup
    def _on_signal(self, signum, frame):
        if self._stop_requested:
            sys.exit(130)
        self._stop_requested = True  # first: logging can fail if the terminal / tee is already gone
        log(f"signal {signum} received: will save a checkpoint after this step and exit (press again to abort)",
            self.log_file)

    def _build_compiled(self):
        self._micro, self._apply, self._eval = build_step_fns(self.model, self.opt, self.args.clip,
                                                              self.token_bytes, self.args.compile)

    def _new_planner(self, deadline_ts):
        s1 = self.data.s1.shards.total_tokens
        s2 = self.data.s2.shards.total_tokens if self.data.s2 is not None else 0
        cap = shard_capacity_steps(s1, s2, self.tokens_per_step, self.args.anneal, self.args.transition_steps)
        # the prefetcher may run this many steps ahead; the plan must freeze before that window
        lookahead = math.ceil((6 + 1) / self.accum)
        return DeadlinePlanner(deadline_ts, self.args.anneal, self.args.transition_steps, cap, self.total_steps,
                               freeze_margin=max(5, lookahead + 1),
                               late_confirm_s=_arg(self.args, "late_confirm_s", 300.0),
                               max_overrun_s=_arg(self.args, "deadline_max_overrun_min", 40.0) * 60)

    def _apply_plan(self):
        """Push the current plan (planner or fixed) into the LR schedule and the data switch."""
        if self.planner is not None:
            self.total_steps, self.decay_start = self.planner.total, self.planner.decay
            transition = self.planner.transition
        else:
            transition = self.decay_start - self.args.transition_steps
        self.data.set_plan(transition, self.decay_start)

    def _plan_view(self):
        """What the trainer and the data loader currently use."""
        view = (self.total_steps, self.decay_start)
        return view + ((self.data.transition_start, self.data.switch) if self.data.s2 is not None else ())

    def _planner_view(self):
        view = (self.planner.total, self.planner.decay)
        return view + ((self.planner.transition, self.planner.decay) if self.data.s2 is not None else ())

    def _resume(self, which):
        path = C.latest(self.run_dir) if which == "auto" else Path(which)
        if path is None:
            log(f"--resume auto: no checkpoint in {self.run_dir}, starting fresh", self.log_file)
            return
        meta = C.load_meta(path)
        if meta["model_config"] != self.cfg.to_dict():
            raise SystemExit(f"checkpoint config {meta['model_config']} != current {self.cfg.to_dict()}")
        C.load_weights(self.model, path)
        C.load_optimizer(self.opt, path)
        self.data.load_state_dict(meta["data_state"])
        self.step = meta["step"]
        old_stats = meta.get("stats", {})
        self.stats.update(old_stats)
        for key, frac in (("core_done", self.args.core_frac), ("milestones_done", self.args.milestone_frac)):
            if key not in old_stats:  # checkpoints written before crossing counters existed
                n = self._cadence(frac)
                self.stats[key] = int(self.step * n // max(1, meta.get("total_steps") or self.total_steps)) if n else 0
        # --lr-scale is absolute and never compounds. Re-running the command that wrote this checkpoint keeps
        # the checkpoint's value (which may include later automatic 0.7x reductions).
        cli, prev = self.args.lr_scale, (meta.get("args") or {}).get("lr_scale")
        self.lr_scale = meta.get("lr_scale", 1.0) if cli is None or cli == prev else cli
        if self.lr_scale != meta.get("lr_scale", 1.0):
            self._dirty = True

        plan = meta.get("plan")
        if plan is not None:  # a deadline-mode run stays in deadline mode, with its original deadline
            new_deadline = self.planner.deadline if self.planner is not None else None
            self.planner = self.planner or self._new_planner(plan["deadline"])
            self.planner.load_state_dict(plan)
            if new_deadline is not None and _arg(self.args, "reset_deadline", False):
                self.planner.deadline = new_deadline
                log(f"deadline reset to {time.ctime(new_deadline)}", self.log_file)
        elif self.planner is not None:
            # a fixed-token run switched to deadline mode: keep its schedule if it is already past the transition
            old_total = meta.get("total_steps") or self.total_steps
            old_decay = meta.get("decay_start") or decay_start_step(old_total, self.args.anneal)
            old_transition = meta.get("transition_start") or old_decay - self.args.transition_steps
            if self.step >= old_transition - self.planner.freeze_margin - 1:
                self.planner.total, self.planner.decay, self.planner.transition = old_total, old_decay, old_transition
                self.planner.frozen = True
                log(f"deadline mode on a run already in its transition/decay: keeping its schedule ({old_total} steps)",
                    self.log_file)
            else:
                self.planner._set_total(min(old_total, self.planner.max_total))
        else:
            old_total = meta.get("total_steps")
            old_decay = meta.get("decay_start", decay_start_step(old_total, self.args.anneal) if old_total else None)
            if old_total and old_total != self.total_steps:
                old_transition = old_decay - self.args.transition_steps
                if self.step >= old_transition and not _arg(self.args, "force_reschedule", False):
                    raise SystemExit(
                        f"refusing to change the total from {old_total} to {self.total_steps} steps at step {self.step}: "
                        f"the run is already in its transition/decay (from step {old_transition}). Resume with the "
                        f"original --tokens/--max-steps, or pass --force-reschedule if you really mean it.")
                log(f"note: total steps changed {old_total} -> {self.total_steps}", self.log_file)
        log(f"resumed from {path} at step {self.step} (lr_scale {self.lr_scale:.3f})", self.log_file)

    def _skip_data(self, n_steps):
        """Advance the data cursors by n optimizer steps without training (spike recovery)."""
        for s in range(self.step, self.step + n_steps):
            for _ in range(self.accum):
                self.data.batch(s)
        self.stats["skipped_steps"] += n_steps
        log(f"skipped {n_steps} steps of data", self.log_file)

    def _skip_once(self, n_steps):
        """Apply --skip-steps unless the process that wrote this checkpoint already did (the state is
        persisted), so re-running the same command after a crash does not skip again. Relaunch once
        without the flag (or with a different N) to arm it again."""
        if self.stats.get("skip_armed") == n_steps:
            log(f"--skip-steps {n_steps} was already applied by the run that wrote this checkpoint; not skipping again",
                self.log_file)
            return
        self._skip_data(n_steps)
        self.stats["skip_armed"] = n_steps
        self._dirty = True

    # ---------------------------------------------------------------- helpers
    def meta(self, data_state):
        return {"model_config": self.cfg.to_dict(), "data_state": data_state, "stats": self.stats,
                "lr_scale": self.lr_scale, "total_steps": self.total_steps, "decay_start": self.decay_start,
                "transition_start": self.data.transition_start if self.data.s2 is not None else None,
                "plan": self.planner.state_dict() if self.planner is not None else None,
                "args": vars(self.args), "tokens_per_step": self.tokens_per_step}

    def metric(self, **kv):
        kv["step"] = self.step
        kv["time"] = time.time()
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(kv) + "\n")

    def evaluate_val(self):
        tot_ce = tot_n = tot_ce_b = tot_bytes = 0.0
        for rows in val_batches(self.val_dir, self.args.micro_batch, self.args.seq, self.args.eval_tokens):
            r = mx.array(rows)
            ce, n, ce_b, nb = self._eval(r[:, :-1], r[:, 1:])
            mx.eval(ce, n, ce_b, nb)
            tot_ce += ce.item(); tot_n += n.item(); tot_ce_b += ce_b.item(); tot_bytes += nb.item()
        loss = tot_ce / max(tot_n, 1)
        bpb = tot_ce_b / (math.log(2) * max(tot_bytes, 1))
        return loss, bpb

    def sample(self):
        from pagoda.engine import generate
        prompts = json.loads((Path(__file__).resolve().parents[1] / "prompts" / "fixed_prompts.json").read_text())
        out = []
        for p in prompts["pretrain"]:
            ids = self.tok.encode(p, bos=True)
            gen = generate(self.model, self.tok, ids, max_new_tokens=80, temperature=0.0)
            out.append(f"--- {p!r}\n{self.tok.decode(gen)}")
        text = f"===== step {self.step} =====\n" + "\n".join(out) + "\n"
        with open(self.log_dir / "samples.txt", "a") as f:
            f.write(text)
        log(text.rstrip(), None)

    def run_core(self):
        from pagoda.core_eval import evaluate_core
        t0 = time.monotonic()  # does not advance while the Mac sleeps
        res = evaluate_core(self.model, self.tok, max_per_task=self.args.core_max_per_task)
        took = time.monotonic() - t0
        prev = self.stats.get("t_core")
        self.stats["t_core"] = took if prev is None else min(prev, took)  # one stalled run must not inflate the reserve
        self.metric(kind="core", core=res["core"], tasks=res["tasks"], seconds=took)
        log(f"step {self.step} | CORE {res['core']:.4f} ({took:.0f}s)", self.log_file)

    def save(self, data_state, milestone=False):
        t0 = time.time()
        path = C.save_full(self.run_dir, self.step, self.model, self.opt, self.meta(data_state), keep=self.args.keep)
        msg = f"saved {path.name} ({time.time() - t0:.1f}s)"
        if milestone:
            C.save_milestone(self.run_dir, self.step, self.model, self.meta(data_state))
            msg += " + bf16 milestone"
        log(msg, self.log_file)

    def recover(self, reason):
        """Plan section 6: reload the last full checkpoint, LR x0.7, skip 200 steps of data.

        Each recovery compounds on the LR actually in use and is saved right away, so a second spike
        rewinds to the reduced state and skips fresh data instead of replaying the first recovery."""
        path = C.latest(self.run_dir)
        if path is None or self.stats["recoveries"] >= self.args.auto_recover:
            log(f"STOP: {reason}. No automatic recovery left. Resume manually with "
                f"`--resume auto --lr-scale {0.7 * self.lr_scale:.3f} --skip-steps 200`.", self.log_file)
            sys.exit(2)
        meta = C.load_meta(path)
        C.load_weights(self.model, path)
        C.load_optimizer(self.opt, path)
        self.data.load_state_dict(meta["data_state"])
        self.step = meta["step"]
        self.stats = {**self.stats, **meta["stats"], "recoveries": self.stats["recoveries"] + 1}
        self.lr_scale = min(self.lr_scale, meta.get("lr_scale", 1.0)) * 0.7
        if self.planner is not None and meta.get("plan"):
            deadline = self.planner.deadline  # the plan rewinds with the data cursor; the deadline does not
            self.planner.load_state_dict(meta["plan"])
            self.planner.deadline = deadline
        self._apply_plan()
        log(f"RECOVER ({reason}): back to step {self.step}, lr_scale -> {self.lr_scale:.3f}", self.log_file)
        self._skip_data(self.args.recover_skip_steps)
        self.save(self.data.state_dict())  # persist the reduced LR and the advanced cursor
        self._last_save = time.time()

    # ------------------------------------------------------- deadline helpers
    def _cadence(self, frac):
        return max(1, int(round(1 / frac))) if frac and frac > 0 else 0

    def _crossed(self, done, n):
        """Has the run reached the (done+1)-th of n equally spaced points of the current total?"""
        return n and done < n and self.step >= math.ceil((done + 1) * self.total_steps / n)

    def _reserve_s(self, n_core):
        """Wall time the rest of the run needs outside regular steps: remaining CORE runs + final block."""
        t_core = self.stats.get("t_core") or _arg(self.args, "core_prior_s", 600.0)
        cores_left = max(0, n_core - self.stats["core_done"]) if n_core else 0
        return cores_left * t_core + _arg(self.args, "deadline_reserve_min", 5.0) * 60

    # -------------------------------------------------------------- main loop
    def train(self):
        """Returns DONE when the last step finished, STOPPED after a signal-triggered save."""
        a = self.args
        n = self.model.num_params()
        log(f"model {self.cfg.to_dict()}", self.log_file)
        log(f"params total {n['total'] / 1e6:.1f}M, non-embedding {n['non_embedding'] / 1e6:.1f}M; "
            f"{self.total_steps} steps x {self.tokens_per_step} tokens (micro {a.micro_batch}x{a.seq}, accum {self.accum}); "
            f"decay + stage 2 from step {self.decay_start}, transition from {self.data.transition_start}", self.log_file)
        if self.planner is not None:
            log(f"deadline mode: pretraining must end by {time.ctime(self.planner.deadline)}; "
                f"plan {'frozen' if self.planner.frozen else 'adaptive'}, up to {self.planner.max_total} steps", self.log_file)

        n_milestone = self._cadence(a.milestone_frac)
        n_core = self._cadence(a.core_frac)
        first_save = _arg(a, "first_save", 100)
        replan_every = _arg(a, "replan_every", 10)
        if self._dirty:  # a resume-time skip or LR change must survive a crash before the next timed save
            self.save(self.data.state_dict())
            self._dirty = False
        self._last_save = time.time()
        spikes = 0
        window = {"t": time.time(), "tokens": 0, "loss": 0.0, "n": 0}
        dts = deque(maxlen=200)          # pure step times since this process started
        gnorms = deque(maxlen=50)        # recent gradient norms, the baseline for spike detection
        overhead = {"s": 0.0, "steps": 0}  # val / sample / save time, amortised per step (CORE goes in the reserve)
        mx.reset_peak_memory()

        def sec_per_step():
            if len(dts) < 20:
                return self.planner.sec_per_step if self.planner is not None else None
            return statistics.median(dts) + overhead["s"] / max(1, overhead["steps"])

        while self.step < self.total_steps:
            prefetch = Prefetcher(self.data, self.step, self.accum)
            trouble = None
            spikes = 0
            try:
                while self.step < self.total_steps:
                    step = self.step
                    if self.planner is not None:
                        msg = self.planner.tick(step)
                        if msg is None and step % replan_every == 0:
                            msg = self.planner.replan(step, sec_per_step(), self._reserve_s(n_core))
                        if self._plan_view() != self._planner_view():  # sync every change, logged or not
                            self._apply_plan()
                        if msg:
                            log(msg, self.log_file)
                            self.metric(kind="plan", message=msg, **self.planner.state_dict())
                        if self.planner.frozen and self.data.s2 is not None:
                            assert self.data.transition_start == self.planner.transition and \
                                self.data.switch == self.planner.decay, "data switch out of sync with the plan"
                    lrm = wsd_multiplier(step, self.total_steps, a.warmup, a.anneal, self.decay_start) * self.lr_scale
                    set_lr(self.opt, self.lr_groups, lrm)
                    set_muon_momentum(self.opt, muon_momentum(step))

                    t0 = time.time()
                    acc, loss_sum, tok_sum, data_state = None, 0.0, 0.0, None
                    for _ in range(self.accum):
                        s, _, rows, data_state = prefetch.get()
                        assert s == step
                        r = mx.array(rows)
                        loss, ntok, grads = self._micro(r[:, :-1], r[:, 1:], mx.array(self.loss_scale))
                        acc = grads if acc is None else tree_map(mx.add, acc, grads)
                        mx.eval(acc, loss, ntok)
                        loss_sum += loss.item() / self.loss_scale
                        tok_sum += ntok.item()
                    grads = tree_map(lambda g: g / (tok_sum * self.loss_scale), acc)
                    if a.dtype == "fp16":  # dynamic loss scaling: skip the update on overflow
                        gn = global_grad_norm(grads).item()
                        if not math.isfinite(gn):
                            self.loss_scale /= 2
                            log(f"fp16 overflow at step {step}: update skipped, loss scale -> {self.loss_scale:g}", self.log_file)
                            self.step += 1
                            continue
                        self._good_steps += 1
                        if self._good_steps % 1000 == 0:
                            self.loss_scale = min(self.loss_scale * 2, 2.0 ** 24)
                    norm = self._apply(grads)
                    mx.eval(norm, self.model.parameters(), self.opt.state)
                    dt = time.time() - t0

                    train_loss = loss_sum / max(tok_sum, 1)
                    gnorm = norm.item()
                    self.step += 1
                    self.stats["tokens_seen"] += self.tokens_per_step
                    self.stats["train_time"] += dt
                    overhead["steps"] += 1
                    if overhead["steps"] > 3:  # the first steps include compilation
                        dts.append(dt)

                    # --- health checks (plan section 6)
                    if not math.isfinite(train_loss) or not math.isfinite(gnorm):
                        trouble = f"non-finite loss/grad at step {step}"
                        break
                    # a fresh model's gradient norm is naturally large (6-14 at d512) and is clipped anyway:
                    # count spikes only after warmup, and only as a jump ("突增") over the recent median
                    jump = len(gnorms) < 20 or gnorm > 3 * statistics.median(gnorms)
                    gnorms.append(gnorm)
                    if gnorm > a.spike_norm and step >= a.warmup and jump:
                        spikes += 1
                        log(f"warning: grad norm {gnorm:.2f} > {a.spike_norm} at step {step} ({spikes} in a row)", self.log_file)
                        if spikes >= 3:
                            trouble = f"grad norm > {a.spike_norm} for 3 consecutive steps at step {step}"
                            break
                    else:
                        spikes = 0

                    window["tokens"] += self.tokens_per_step
                    window["loss"] += train_loss
                    window["n"] += 1
                    if self.step % a.log_every == 0 or self.step <= 3:
                        el = time.time() - window["t"]
                        tps = window["tokens"] / max(el, 1e-9)
                        peak = mx.get_peak_memory() / 1024 ** 3
                        avg = window["loss"] / window["n"]
                        plan = ""
                        if self.planner is not None:
                            sps = sec_per_step()
                            # same accounting as the planner: remaining steps + remaining CORE runs + final block
                            eta = time.time() + (self.total_steps - self.step) * sps + self._reserve_s(n_core) if sps else None
                            slack = f"{(self.planner.deadline - eta) / 60:+.0f} min" if eta else "n/a"
                            plan = f" | plan {'F' if self.planner.frozen else '~'}{self.total_steps}, slack {slack}"
                        log(f"step {self.step:6d}/{self.total_steps} | loss {avg:.4f} | gnorm {gnorm:.3f} | "
                            f"lr x{lrm:.3f} | {self.data.phase(step)} | {tps:,.0f} tok/s | {dt:.2f}s/step | "
                            f"peak {peak:.1f}GB | {self.stats['tokens_seen'] / 1e9:.3f}B tok{plan}", self.log_file)
                        self.metric(kind="train", loss=avg, grad_norm=gnorm, lr_mult=lrm, tok_per_s=tps,
                                    step_time=dt, peak_mem_gb=peak, phase=self.data.phase(step),
                                    tokens=self.stats["tokens_seen"], epochs=self.data.epochs(),
                                    total_steps=self.total_steps)
                        window = {"t": time.time(), "tokens": 0, "loss": 0.0, "n": 0}

                    last = self.step == self.total_steps
                    t_block = time.time()
                    t_extra = time.monotonic()  # overhead timings ignore system sleep
                    cap = max(120.0, 10 * statistics.median(dts)) if dts else 600.0
                    if a.eval_every and (self.step % a.eval_every == 0 or last):
                        vl, bpb = self.evaluate_val()
                        best = self.stats["best_val_bpb"]
                        self.stats["best_val_bpb"] = bpb if best is None else min(best, bpb)
                        log(f"step {self.step} | val loss {vl:.4f} | val bpb {bpb:.4f}", self.log_file)
                        self.metric(kind="val", val_loss=vl, val_bpb=bpb)
                    if a.sample_every and (self.step % a.sample_every == 0 or last):
                        self.sample()
                    overhead["s"] += min(cap, time.monotonic() - t_extra)
                    if self._crossed(self.stats["core_done"], n_core):
                        self.run_core()  # its cost is counted in the planner's reserve, not per step
                        self.stats["core_done"] += 1
                    t_extra = time.monotonic()
                    milestone = bool(self._crossed(self.stats["milestones_done"], n_milestone))
                    if milestone:
                        self.stats["milestones_done"] += 1
                    timed = (time.time() - self._last_save) / 60 >= a.save_every_min
                    stepped = (a.save_every and self.step % a.save_every == 0) or self.step == first_save
                    if milestone or timed or stepped or last or self._stop_requested:
                        self.save(data_state, milestone=milestone)
                        self._last_save = time.time()
                    overhead["s"] += min(cap, time.monotonic() - t_extra)
                    window["t"] += time.time() - t_block  # tok/s in the log excludes eval / CORE / save time
                    if self._stop_requested:
                        log("stopped by signal after saving", self.log_file)
                        return STOPPED
            finally:
                prefetch.close()  # the loader thread must be stopped before its state is rewound
            if trouble is None:
                break
            self.recover(trouble)
        log(f"training done: {self.stats['tokens_seen'] / 1e9:.3f}B tokens, "
            f"{self.stats['train_time'] / 3600:.2f} h of step time", self.log_file)
        from pagoda.common import write_json
        write_json(self.log_dir / "pretrain_done.json",
                   {"step": self.step, "total_steps": self.total_steps, "tokens": self.stats["tokens_seen"],
                    "best_val_bpb": self.stats["best_val_bpb"], "time": time.time(),
                    "checkpoint": str(C.latest(self.run_dir))})
        return DONE
