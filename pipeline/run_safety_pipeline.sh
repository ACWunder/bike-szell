#!/bin/zsh
# Chain the safety-analysis compute:
#   1. wait for the biketrack (A) regen to finish
#   2. re-run 03 + 04 so the comparison CSVs (density_table, metrics_at_biketrack)
#      pick up the new A length
#   3. run 02b safety spatial (A vs C)  — the long one (~25-30 min, Berlin slowest)
# Re-running 07 (report) is done by the orchestrator afterwards.
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOG="$ROOT/pipeline/safety_pipeline.log"
echo "[chain] waiting for regen_biketrack_snapped to finish..." > "$LOG"
while pgrep -f regen_biketrack_snapped >/dev/null; do sleep 20; done
echo "[chain] regen finished. tail:" >> "$LOG"
tail -3 "$ROOT/bikenwgrowth-source/scripts/regen_biketrack_full.log" >> "$LOG" 2>&1 || true

echo "[chain] === 03_analysis_multicity ===" >> "$LOG"
conda run --no-capture-output -n growbikenet python pipeline/03_analysis_multicity.py >> "$LOG" 2>&1
echo "[chain] === 04_analysis_normalized ===" >> "$LOG"
conda run --no-capture-output -n growbikenet python pipeline/04_analysis_normalized.py >> "$LOG" 2>&1
echo "[chain] === 02b_analysis_safety_spatial ===" >> "$LOG"
conda run --no-capture-output -n growbikenet python pipeline/02b_analysis_safety_spatial.py >> "$LOG" 2>&1
echo "[chain] ALL DONE" >> "$LOG"
