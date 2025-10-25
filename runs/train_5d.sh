#!/usr/bin/env bash
# 5-day run: 12 x 768 (110M). Token budget from the day-0 benchmark (logs/bench.json):
# 7.76k tok/s x 0.85 x 100 h = 2.37B -> 2.3B (1.84B stage 1 + 0.46B stage 2).
# The plan's pre-benchmark estimate was 1.55B: TOKENS=1.55e9 ./runs/train_5d.sh
# Starts inside a detached tmux session under caffeinate. Re-running resumes from
# the newest checkpoint (exact data position, LR continues, no re-warmup).
#   attach:  tmux attach -t pagoda      detach: Ctrl-b d
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-d768_5d}"
MODEL="${MODEL:-d768}"
TOKENS="${TOKENS:-2.3e9}"
EXTRA="${EXTRA:-}"

CMD="caffeinate -dims uv run python -m scripts.train --model $MODEL --seq 1024 --tokens $TOKENS --anneal 0.2 \
  --mix configs/stage1.yaml --mix-anneal configs/stage2.yaml --run $RUN --resume auto $EXTRA \
  2>&1 | tee -i -a logs/$RUN.stdout.log"
mkdir -p logs
if tmux has-session -t pagoda 2>/dev/null; then
  echo "tmux session 'pagoda' already exists: tmux attach -t pagoda"; exit 1
fi
tmux new-session -d -s pagoda "$CMD; echo; echo 'training exited'; read"
echo "started in tmux session 'pagoda' (run=$RUN). Follow with: tail -f logs/$RUN/train.log"
