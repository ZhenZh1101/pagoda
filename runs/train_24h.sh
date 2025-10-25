#!/usr/bin/env bash
# 24-hour version: 12 x 512 (55M), deadline-driven pretraining -> SFT -> evals -> report, unattended.
# T0 is the first launch; all deadlines are stored in logs/$RUN/pipeline/deadlines.json, so re-running
# this script after a crash, reboot or Ctrl-C continues the same pipeline inside the same 24 h.
#   attach:  tmux attach -t pagoda      detach: Ctrl-b d      progress: tail -f logs/$RUN/pipeline/pipeline.log
# Optional, first launch only: MATRIX_LR=0.03 EMBED_LR=1.5e-3 bash runs/train_24h.sh  (1.5x LR; see README)
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-d512_24h}"
LR_FLAG=""
if [ -n "${MATRIX_LR:-}" ]; then LR_FLAG="$LR_FLAG --matrix-lr $MATRIX_LR"; fi
if [ -n "${EMBED_LR:-}" ]; then LR_FLAG="$LR_FLAG --embed-lr $EMBED_LR"; fi
mkdir -p logs
if tmux has-session -t pagoda 2>/dev/null; then
  echo "tmux session 'pagoda' already exists (another run?): tmux attach -t pagoda"; exit 1
fi
tmux new-session -d -s pagoda \
  "caffeinate -dims uv run python -m scripts.pipeline_24h --run $RUN $LR_FLAG 2>&1 | tee -i -a logs/$RUN.pipeline.log; echo; echo 'pipeline exited'; read"
echo "started in tmux session 'pagoda' (run=$RUN). Progress: tail -f logs/$RUN/pipeline/pipeline.log"
