#!/bin/bash
# Runs one grand-canonical tempering experiment end to end.  Set the variables below, then run (from anywhere):
#   ./config_files/grand_canonical/run_experiment.sh
#
# 1. calibration: runs calibration.ini over ghost_coupling = 0, 1/K, ..., (K-1)/K (CALIBRATION_JOBS jobs each);
# 2. g values:    sample_analysis/compute_ghost_coupling_bias.py writes <OUTPUT>/bias.json (+ bias.png), including
#                 the suggested log_activity;
# 3. production:  runs tempered_bias.ini and direct.ini (PRODUCTION_JOBS jobs each) at that log_activity;
# 4. analysis:    sample_analysis/grand_canonical_analysis.py writes <OUTPUT>/analysis.json.
# The experiment's own settings (K = CALIBRATION_BINS, CALIBRATION_JOBS, PRODUCTION_JOBS, SUB_BINS, TARGET_N, EXACT,
# OUTPUT) are in <EXPERIMENT_DIR>/experiment.env, written by make_experiments.py.
set -euo pipefail

# ---- settings -------------------------------------------------------------------------------------------------------
# experiment folder (relative to src), Python with numpy/scipy/matplotlib, number of parallel jobs
EXPERIMENT_DIR="${EXPERIMENT_DIR:-config_files/grand_canonical/experiments/hard_disks_1d_density_0.5}"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.11}"
MAX_CPUS="${MAX_CPUS:-13}"
# ---------------------------------------------------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # src
export PYTHON MAX_CPUS
source "$EXPERIMENT_DIR/experiment.env"
RUN_LOCAL="config_files/grand_canonical/run_local.sh"

# Grid stops at (K-1)/K where K = number of bins, as the final interval sweeps over [(K-1)/K, 1]
LAST_COUPLING="$("$PYTHON" -c "print(($CALIBRATION_BINS - 1) / $CALIBRATION_BINS)")"
"$RUN_LOCAL" "$EXPERIMENT_DIR/calibration.ini" "$CALIBRATION_JOBS" 0.0 "$LAST_COUPLING" "$((CALIBRATION_BINS - 1))" \
    FixedGhostCouplingKernel ghost_coupling > "$EXPERIMENT_DIR/calibration.log"

"$PYTHON" sample_analysis/compute_ghost_coupling_bias.py "$OUTPUT/calibration" --output "$OUTPUT/bias.json" \
    --sub-bins "$SUB_BINS" --target-number-of-particles "$TARGET_N" --figure "$OUTPUT/bias.png" \
    | tee "$EXPERIMENT_DIR/bias.log"
LOG_ACTIVITY="$("$PYTHON" -c "import json; print(json.load(open('$OUTPUT/bias.json'))['suggested_log_activity'])")"

for scheme in tempered_bias direct; do
    if [ "$scheme" = direct ]; then KERNEL=DirectGrandCanonicalKernel; else KERNEL=TemperedGrandCanonicalKernel; fi
    MAX_CPUS="$(( (MAX_CPUS + 1) / 2 ))" "$RUN_LOCAL" "$EXPERIMENT_DIR/$scheme.ini" "$PRODUCTION_JOBS" \
        "$LOG_ACTIVITY" "$LOG_ACTIVITY" 0 "$KERNEL" log_activity > "$EXPERIMENT_DIR/$scheme.log" &
done
wait

EXACT_FLAG=()
if [ "$EXACT" != "none" ]; then EXACT_FLAG=(--exact "$EXACT"); fi
"$PYTHON" sample_analysis/grand_canonical_analysis.py "$OUTPUT/tempered_bias/log_activity_00" \
    "$OUTPUT/direct/log_activity_00" "${EXACT_FLAG[@]}" \
    --output "$OUTPUT/analysis.json" | tee "$EXPERIMENT_DIR/analysis.log"
