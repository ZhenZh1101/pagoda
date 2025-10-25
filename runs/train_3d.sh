#!/usr/bin/env bash
# 3-day run: 12 x 640 (80M). Day-0 benchmark: 9.85k tok/s x 0.85 x 55 h = 1.66B -> 1.6B tokens
# (the plan's pre-benchmark estimate was 1.17B). Below 5k tok/s use MODEL=d576 (plan section 3).
set -euo pipefail
cd "$(dirname "$0")"
RUN="${RUN:-d640_3d}" MODEL="${MODEL:-d640}" TOKENS="${TOKENS:-1.6e9}" EXTRA="${EXTRA:-}" ./train_5d.sh
