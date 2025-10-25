#!/usr/bin/env bash
# One-time machine setup from plan section 2. These touch system settings, so run
# them yourself (some need sudo); nothing here is run automatically.
set -euo pipefail
ROOT="${PAGODA_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "1) Exclude the data/checkpoint dirs from Time Machine:"
echo "   tmutil addexclusion \"$ROOT\""
echo "2) Exclude from Spotlight: System Settings > Siri & Spotlight > Spotlight Privacy > add $ROOT"
echo "   (or: sudo mdutil -i off on a dedicated volume; a '.metadata_never_index' file also works on many macOS versions)"
echo "3) Power: keep the 140W charger connected; System Settings > Battery > Energy Mode = High Power (if offered)"
echo "4) Lid open, laptop raised on a hard surface. Disable automatic macOS updates during the run."
echo "5) Training runs inside tmux with caffeinate -dims (runs/train_*.sh do this)."
echo
echo "Directory sizes right now:"
du -sh "$ROOT"/{raw,tok,tokenizer,ckpt,logs} 2>/dev/null || true
df -h "$ROOT" | tail -1
