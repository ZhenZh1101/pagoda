"""Day-0 smoke test (plan section 10, first row): the whole pipeline at toy scale.

    python -m scripts.smoke_test            # ~10-15 min on an M4 Pro
    python -m scripts.smoke_test --keep     # keep ckpt/smoke_* and tok/smoke afterwards

1. build tiny shards from configs/smoke/*.yaml (separate dirs, real val untouched)
2. train a 4-layer model 40 steps, saving every 10 steps (run A)
3. resume a second run from A's step-20 checkpoint and check that steps 21-40
   reproduce A's losses -> save / load / resume are exact
4. ckpt_check on the newest checkpoint
5. SFT 2 epochs on the smoke SFT shards; chat_eval on every task (5 examples);
   CORE (3 examples per task); a chat reply; the manual-test sheet
"""

import argparse
import json
import shutil
import subprocess
import sys

from pagoda.common import CKPT_DIR, LOG_DIR, TOK_DIR, log

PY = [sys.executable, "-m"]
TRAIN = ["scripts.train", "--model", "tiny", "--seq", "1024", "--batch-tokens", "32768", "--micro-batch", "16",
         "--max-steps", "40", "--warmup", "5", "--transition-steps", "4", "--mix", "configs/smoke/stage1.yaml",
         "--mix-anneal", "configs/smoke/stage2.yaml", "--val-dir", "smoke/val", "--log-every", "1",
         "--eval-every", "20", "--eval-tokens", "65536", "--sample-every", "40", "--save-every", "10", "--keep", "10"]


def run(args, name):
    log(f"$ python -m {' '.join(args)}")
    r = subprocess.run(PY + args, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    (LOG_DIR / "smoke").mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "smoke" / f"{name}.log").write_text(out)
    if r.returncode != 0:
        print(out[-4000:])
        raise SystemExit(f"smoke step '{name}' failed (exit {r.returncode}); log in logs/smoke/{name}.log")
    return out


def losses(run_name):
    out = {}
    for line in (LOG_DIR / run_name / "metrics.jsonl").read_text().splitlines():
        m = json.loads(line)
        if m["kind"] == "train":
            out[m["step"]] = m["loss"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    for d in [CKPT_DIR / "smoke_a", CKPT_DIR / "smoke_b", LOG_DIR / "smoke_a", LOG_DIR / "smoke_b"]:
        shutil.rmtree(d, ignore_errors=True)

    if not args.skip_build:
        run(["scripts.build_shards", "--stage1", "configs/smoke/stage1.yaml", "--stage2", "configs/smoke/stage2.yaml",
             "--sft", "configs/smoke/sft.yaml"], "build")
        run(["scripts.verify_data", "--stage1", "smoke/stage1", "--stage2", "smoke/stage2", "--val", "smoke/val",
             "--sft", "smoke/sft"], "verify")

    run(TRAIN + ["--run", "smoke_a", "--core-frac", "0.5", "--core-max-per-task", "3"], "train_a")
    run(TRAIN + ["--run", "smoke_b", "--resume", str(CKPT_DIR / "smoke_a" / "step_000020"), "--core-frac", "0"],
        "train_b_resume")
    la, lb = losses("smoke_a"), losses("smoke_b")
    diffs = [abs(la[s] - lb[s]) for s in range(21, 41) if s in la and s in lb]
    log(f"resume check: {len(diffs)} steps compared, max |loss_A - loss_B| = {max(diffs):.2e}; "
        f"A: {la[1]:.3f} -> {la[40]:.3f}")
    if len(diffs) < 20 or max(diffs) > 1e-2:
        raise SystemExit("resume does not reproduce the original run")
    if not la[40] < la[1] - 1.0:
        raise SystemExit("loss did not go down in the smoke run")

    run(["scripts.ckpt_check", "--run", "smoke_a", "--steps", "3"], "ckpt_check")
    run(["scripts.sft", "--base-run", "smoke_a", "--run", "smoke_a", "--data", "smoke/sft", "--epochs", "2",
         "--batch-tokens", "32768", "--micro-batch", "16", "--warmup", "3", "--eval-every", "5", "--log-every", "1"], "sft")
    out = run(["scripts.chat_eval", "--run", "smoke_a", "--source", "sft",
               "--tasks", "arc_easy,arc_challenge,mmlu,hellaswag,gsm8k_tool,humaneval", "--limit", "5"], "chat_eval")
    run(["scripts.chat_eval", "--run", "smoke_a", "--source", "base", "--tasks", "core", "--core-max-per-task", "3"],
        "core_eval")
    run(["scripts.chat", "--run", "smoke_a", "--source", "sft", "--prompt", "What is 3 times 12?", "--max-tokens", "64"],
        "chat")
    run(["scripts.manual_test", "--run", "smoke_a", "--source", "sft"], "manual_test")
    log("SMOKE TEST PASSED (logs in logs/smoke/, logs/smoke_a/)")
    if not args.keep:
        for d in [CKPT_DIR / "smoke_a", CKPT_DIR / "smoke_b", LOG_DIR / "smoke_b"]:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
