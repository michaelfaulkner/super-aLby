#!/bin/bash
# Runs one grand-canonical experiment end to end, or every point of a sweep:
#   ./config_files/grand_canonical/run_experiment.sh <experiment_dir or sweep_dir>
# The folder is one written by make_experiments.py (absolute, or relative to the current directory). With SULIS=true
# in experiment.env (sulis = true in the spec) the stages are submitted to Slurm by sulis_experiment.sh instead of
# being run here; run it on Sulis then.
# Optional environment variables: STAGES (below), MAX_CPUS (default: number of cores - 1) and PYTHON (default:
# /opt/homebrew/bin/python3.11 if present, else python3; it needs numpy, scipy and matplotlib).
#
# Stages (STAGES, default all four; e.g. STAGES="production analysis" ./run_experiment.sh <dir> to rerun only those):
#   calibration  runs calibration.ini over ghost_coupling = 0, 1/K, ..., (K-1)/K (CALIBRATION_JOBS jobs each);
#   bias         sample_analysis/compute_ghost_coupling_bias.py writes <OUTPUT>/bias.json (+ bias.png), including
#                the suggested log_activity;
#   production   runs <scheme>.ini for every scheme in SCHEMES (PRODUCTION_JOBS jobs each) at one log_activity;
#   analysis     sample_analysis/grand_canonical_analysis.py writes <OUTPUT>/analysis.json.
# calibration and bias are skipped when nothing needs them (a fixed LOG_ACTIVITY and no tabulated bias).
#
# <experiment_dir>/experiment.env (written by make_experiments.py) holds the settings:
#   NAME, OUTPUT, CALIBRATION_BINS (K), CALIBRATION_JOBS, PRODUCTION_JOBS, SUB_BINS,
#   LOG_ACTIVITY  suggested (from the calibration, P(N) peaks near the calibration N) or a number (fixed activity);
#   SCHEMES       production templates to run, e.g. "tempered direct" or "tempered metropolis";
#   SULIS         true or false, with the Slurm settings SULIS_* when true.
# Finished jobs are skipped, so an interrupted experiment can simply be rerun.
set -euo pipefail

ALL_STAGES="calibration bias production analysis"
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_dir or sweep_dir>   (a folder written by make_experiments.py)" >&2
    exit 1
fi
if [ ! -f "$1/experiment.env" ] && [ -f "$1/sweep.json" ]; then   # a sweep: run (or submit) every point in turn
    SWEEP_DIR="$(cd "$1" && pwd)"
    for point in "$SWEEP_DIR"/*/; do
        if [ -f "$point/experiment.env" ]; then "${BASH_SOURCE[0]}" "${point%/}"; fi
    done
    exit 0
fi
if [ ! -f "$1/experiment.env" ]; then
    echo "No experiment.env in $1 (write the experiment with make_experiments.py first)." >&2
    exit 1
fi
EXPERIMENT_DIR="$(cd "$1" && pwd)"   # absolute, so it does not depend on the current directory
STAGES="${STAGES:-$ALL_STAGES}"
for stage in $STAGES; do
    if [[ " $ALL_STAGES " != *" $stage "* ]]; then echo "Unknown stage '$stage' (use: $ALL_STAGES)." >&2; exit 1; fi
done
if [ -z "${PYTHON:-}" ]; then
    if [ -x /opt/homebrew/bin/python3.11 ]; then PYTHON=/opt/homebrew/bin/python3.11; else PYTHON=python3; fi
fi
if [ -z "${MAX_CPUS:-}" ]; then
    CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
    MAX_CPUS="$(( CORES > 1 ? CORES - 1 : 1 ))"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # src: the paths in experiment.env and the configs are relative to it
export PYTHON MAX_CPUS
source "$EXPERIMENT_DIR/experiment.env"
for key in NAME OUTPUT CALIBRATION_BINS CALIBRATION_JOBS PRODUCTION_JOBS SUB_BINS LOG_ACTIVITY \
        SCHEMES SULIS; do
    if [ -z "${!key:-}" ]; then
        echo "$EXPERIMENT_DIR/experiment.env has no $key (regenerate it with make_experiments.py)." >&2
        exit 1
    fi
done
if [ "$SULIS" = true ]; then
    exec config_files/grand_canonical/sulis_experiment.sh "$EXPERIMENT_DIR"
fi
RUN_LOCAL="config_files/grand_canonical/run_local.sh"
read -r -a SCHEME_LIST <<< "$SCHEMES"

get_value() {   # get_value <ini> <key>: the value of the first "key = value" line
    awk -F '=' -v key="$2" '$1 ~ "^" key "[[:space:]]*$" { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' "$1"
}
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

NEEDS_CALIBRATION=false
if [ "$LOG_ACTIVITY" = suggested ]; then NEEDS_CALIBRATION=true; fi
for scheme in "${SCHEME_LIST[@]}"; do
    if grep -q "tabulated_ghost_coupling_bias" "$EXPERIMENT_DIR/$scheme.ini"; then NEEDS_CALIBRATION=true; fi
done

LINE="-----------------------------------------------------------------------------------------"
now() { "$PYTHON" -c "import time; print(time.time())"; }
runtime() { "$PYTHON" -c "import sys; print(float(sys.argv[2]) - float(sys.argv[1]))" "$1" "$(now)"; }
wait_with_progress() {   # wait_with_progress <total runs> <description> <log file> ... : waits for the background
    local total="$1" description="$2"; shift 2  # run_local.sh calls (pids in PIDS), printing every 10 % of the runs
    local step=1 printed=0 count finished running pid status=0
    while true; do
        running=false   # checked before counting, so the final count is printed once the runs have ended
        for pid in "${PIDS[@]}"; do if kill -0 "$pid" 2>/dev/null; then running=true; fi; done
        finished="$(cat "$@" 2>/dev/null | grep -c -E '^(done|skip|FAILED)' || true)"
        while [ "$step" -le 10 ] && [ $((10 * finished)) -ge $((step * total)) ]; do
            count=$(( (step * total + 9) / 10 ))
            if [ "$count" -gt "$printed" ]; then
                echo "$count run$([ "$count" -eq 1 ] || echo s) finished out of a total of $total ($description)."
                printed="$count"
            fi
            step=$((step + 1))
        done
        if ! $running; then break; fi
        sleep 2
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
    if [ "$status" -ne 0 ] || cat "$@" | grep -q '^FAILED'; then
        echo "Some runs failed (see the FAILED lines in the logs in the experiment folder)." >&2
        exit 1
    fi
}
start_stage() {   # start_stage <message>
    echo "$1"
    echo "$LINE"
    STAGE_START="$(now)"
}
end_stage() {   # end_stage <stage name>
    echo "$LINE"
    echo "Total runtime of the $1 = $(runtime "$STAGE_START") seconds."
    echo "$LINE"
}

EXPERIMENT_START="$(now)"
if [ "$LOG_ACTIVITY" = suggested ]; then ACTIVITY_TEXT="the suggested log activity"; else
    ACTIVITY_TEXT="log activity $LOG_ACTIVITY"; fi
echo "Setting up the grand-canonical experiment $NAME (schemes: $(IFS=,; echo "${SCHEME_LIST[*]}" | sed 's/,/, /g');" \
     "$ACTIVITY_TEXT; $MAX_CPUS parallel jobs)."

if has_stage calibration && $NEEDS_CALIBRATION; then
    TOTAL=$((CALIBRATION_BINS * CALIBRATION_JOBS))
    start_stage "Starting the calibration runs ($CALIBRATION_BINS ghost couplings, $CALIBRATION_JOBS jobs each)."
    # Grid stops at (K-1)/K where K = number of bins, as the final interval sweeps over [(K-1)/K, 1]
    LAST_COUPLING="$("$PYTHON" -c "print(($CALIBRATION_BINS - 1) / $CALIBRATION_BINS)")"
    "$RUN_LOCAL" "$EXPERIMENT_DIR/calibration.ini" "$CALIBRATION_JOBS" 0.0 "$LAST_COUPLING" \
        "$((CALIBRATION_BINS - 1))" FixedGhostCouplingKernel ghost_coupling > "$EXPERIMENT_DIR/calibration.log" &
    PIDS=($!)
    wait_with_progress "$TOTAL" "calibration" "$EXPERIMENT_DIR/calibration.log"
    end_stage "calibration"
fi

if has_stage bias && $NEEDS_CALIBRATION; then
    start_stage "Computing the ghost-coupling bias from the calibration runs."
    "$PYTHON" sample_analysis/compute_ghost_coupling_bias.py "$OUTPUT/calibration" --output "$OUTPUT/bias.json" \
        --sub-bins "$SUB_BINS" --figure "$OUTPUT/bias.png" \
        > "$EXPERIMENT_DIR/bias.log"
    "$PYTHON" config_files/grand_canonical/print_experiment_results.py bias "$OUTPUT/bias.json"
    end_stage "bias computation"
fi

if has_stage production; then
    if $NEEDS_CALIBRATION && [ ! -f "$OUTPUT/bias.json" ]; then
        echo "The bias is missing: run the calibration and bias stages first." >&2
        exit 1
    fi
    if [ "$LOG_ACTIVITY" = suggested ]; then
        LOG_ACTIVITY="$("$PYTHON" -c \
            "import json; print(json.load(open('$OUTPUT/bias.json'))['suggested_log_activity'])")"
    fi
    TOTAL=$((PRODUCTION_JOBS * ${#SCHEME_LIST[@]}))
    start_stage "Starting the production runs at log activity $LOG_ACTIVITY ($PRODUCTION_JOBS jobs per scheme)."
    CPUS_PER_SCHEME="$(( MAX_CPUS / ${#SCHEME_LIST[@]} > 0 ? MAX_CPUS / ${#SCHEME_LIST[@]} : 1 ))"
    PIDS=()
    LOGS=()
    for scheme in "${SCHEME_LIST[@]}"; do
        KERNEL="$("$PYTHON" -c "print(''.join(w.capitalize() for w in '$(get_value "$EXPERIMENT_DIR/$scheme.ini" \
            grand_canonical_kernel)'.split('_')))")"
        MAX_CPUS="$CPUS_PER_SCHEME" "$RUN_LOCAL" "$EXPERIMENT_DIR/$scheme.ini" "$PRODUCTION_JOBS" \
            "$LOG_ACTIVITY" "$LOG_ACTIVITY" 0 "$KERNEL" log_activity > "$EXPERIMENT_DIR/$scheme.log" &
        PIDS+=($!)
        LOGS+=("$EXPERIMENT_DIR/$scheme.log")
    done
    wait_with_progress "$TOTAL" "production" "${LOGS[@]}"
    end_stage "production"
fi

if has_stage analysis; then
    start_stage "Analysing the production runs (physical samples only)."
    RUNS=()
    for scheme in "${SCHEME_LIST[@]}"; do
        RUNS+=("$(get_value "$EXPERIMENT_DIR/$scheme.ini" output_directory)/log_activity_00")
    done
    EXACT_FLAG=()   # hard disks have an exact P(N) to compare with
    if [ "$(get_value "$EXPERIMENT_DIR/${SCHEME_LIST[0]}.ini" potential)" = grand_canonical_hard_sphere_potential ]
    then
        EXACT_FLAG=(--exact hard_disks)
    fi
    "$PYTHON" sample_analysis/grand_canonical_analysis.py "${RUNS[@]}" ${EXACT_FLAG[@]+"${EXACT_FLAG[@]}"} \
        --output "$OUTPUT/analysis.json" > "$EXPERIMENT_DIR/analysis.log"
    "$PYTHON" config_files/grand_canonical/print_experiment_results.py analysis "$OUTPUT/analysis.json"
    end_stage "analysis"
fi

echo "Total runtime of the experiment = $(runtime "$EXPERIMENT_START") seconds."
echo "$LINE"
