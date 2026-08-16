#!/usr/bin/env bash
# Resume the kill gate until all 180 requests land.
#
# The measurement host has 25-33% packet loss, so a single pass will not
# finish. Each invocation resumes from the checkpoint and skips completed
# work, so this loop accumulates toward a result instead of restarting.
# The $0.50 cap and the spend ledger are enforced inside killgate.py itself,
# so no amount of looping can exceed the budget.
set -uo pipefail
cd "$(dirname "$0")/.."

PROGRESS="bench/results/.killgate_progress.json"
TOTAL=180
ATTEMPTS="${1:-30}"

count() {
  .venv/bin/python -c "
import json, pathlib
p = pathlib.Path('$PROGRESS')
print(len(json.loads(p.read_text())) if p.exists() else 0)
" 2>/dev/null || echo 0
}

for i in $(seq 1 "$ATTEMPTS"); do
  n=$(count)
  if [ "$n" -ge "$TOTAL" ]; then
    echo ""
    echo "COMPLETE — $n/$TOTAL requests"
    break
  fi
  echo "--- attempt $i/$ATTEMPTS (have $n/$TOTAL) ---"
  .venv/bin/python -m bench.killgate --max-spend 0.50 2>&1 \
    | grep -E "resuming|/180|INCOMPLETE|actual cost|STOP|outstanding" || true
  sleep 5
done

echo ""
echo "=============== FINAL ==============="
.venv/bin/python -m bench.killgate --max-spend 0.50 2>&1 | tail -25
