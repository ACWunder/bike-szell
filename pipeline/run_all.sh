#!/usr/bin/env bash
# Runs the full analysis pipeline end-to-end and regenerates both PDF reports.
# Designed to be re-runnable: every script reads only from disk, writes only to
# bikenwgrowth-data/analysis_output/, and is idempotent.
#
# Total runtime on a 2024-era MacBook: ~10 min
#   - step 1 (boundaries):  ~1.5 min  (osmnx live, cached after first run)
#   - step 2 (spatial):     <1 min
#   - steps 3–4:            <30 s each
#   - step 5 (bootstrap):   ~3 min
#   - step 6 (gap clusters): <30 s
#   - steps 7–8 (PDFs):     ~15 s each
#
# Usage (from the project root):
#   conda activate growbikenet
#   bash pipeline/run_all.sh

set -e
cd "$(dirname "$0")/.."   # project root

echo "── 1 · effective boundaries ──"
python pipeline/01_compute_effective_boundaries.py

echo
echo "── 2 · spatial (overlap, gap, buffer sweep) ──"
python pipeline/02_analysis_multicity_spatial.py

echo
echo "── 3 · growth curves + coverage bars ──"
python pipeline/03_analysis_multicity.py

echo
echo "── 4 · normalized density + quantile comparison ──"
python pipeline/04_analysis_normalized.py

echo
echo "── 5 · directness bootstrap (10 × 1000 pairs) ──"
python pipeline/05_analysis_directness_bootstrap.py

echo
echo "── 6 · gap clusters (DBSCAN per city) ──"
python pipeline/06_analysis_gap_clusters.py

echo
echo "── 7 · multicity PDF report ──"
python pipeline/07_generate_report_multicity.py

echo
echo "── 8 · Vienna PDF report (v4) ──"
python pipeline/08_generate_report_vienna.py

echo
echo "Mirror final PDFs into reports/ for convenience"
cp bikenwgrowth-data/analysis_output/report_multicity.pdf reports/
cp bikenwgrowth-data/analysis_output/report_vienna_v4.pdf reports/

echo
echo "── DONE ──"
echo "Reports:"
echo "  reports/report_multicity.pdf"
echo "  reports/report_vienna_v4.pdf"
