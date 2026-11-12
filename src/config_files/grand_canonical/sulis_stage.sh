#!/bin/bash
# One stage of a grand-canonical experiment on Slurm (submitted by sulis_experiment.sh; not meant to be run by hand):
#   sulis_stage.sh <experiment_dir> spawn_calibration   writes the calibration job configs and calibration_jobs.txt
#   sulis_stage.sh <experiment_dir> spawn_production    writes the production job configs and production_jobs.txt
#   sulis_stage.sh <experiment_dir> calibration         array task: runs RUNS_PER_TASK lines of calibration_jobs.txt
#   sulis_stage.sh <experiment_dir> bias                computes bias.json, then spawns the production job configs
#   sulis_stage.sh <experiment_dir> production          array task: runs RUNS_PER_TASK lines of production_jobs.txt
#   sulis_stage.sh <experiment_dir> analysis            writes analysis.json and prints the results
# Array task t runs lines t * RUNS_PER_TASK + 1, ..., (t + 1) * RUNS_PER_TASK of the list (default 1) in parallel, one
# per CPU of the task (sulis_experiment.sh sets RUNS_PER_TASK and --cpus-per-task to keep within the Slurm job limit).
# Paths are relative to src.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_dir> <spawn_calibration|spawn_production|calibration|bias|production|analysis>" >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}"   # src (sbatch runs a copy of this script)
EXPERIMENT_DIR="$1"
STAGE="$2"
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "$EXPERIMENT_DIR/experiment.env"
read -r -a SCHEME_LIST <<< "$SCHEMES"

get_value() {   # get_value <ini> <key>: the value of the first "key = value" line
    awk -F '=' -v key="$2" '$1 ~ "^" key "[[:space:]]*$" { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' "$1"
}
section_of() {   # section_of <snake_case_name>: the CamelCase config section of a class
    "$PYTHON" -c "import sys; print(''.join(word.capitalize() for word in sys.argv[1].split('_')))" "$1"
}
spawn() {   # spawn <template.ini> <jobs> <start> <end> <increments> <section> <variable>: like run_local.sh
    local base_dir
    base_dir="$(dirname "$1")/$(basename "$1" .ini)"
    if [ ! -d "$base_dir" ]; then
        "$PYTHON" spawn_configs.py "$@" > /dev/null
    fi
    find "$base_dir" -name "job_*.ini" | sort
}
run_config() {   # run_config <config>: runs one config, unless it has already finished
    local output
    output="$(get_value "$1" output_directory)"
    if [ -f "$output/sim_params.json" ]; then
        echo "The run of $1 has already finished."
        return 0
    fi
    mkdir -p "$output"
    echo "Running $1."
    "$PYTHON" run.py "$1" > "$output/run.log" 2>&1
}
run_task() {   # run_task <list of configs>: runs this array task's block of RUNS_PER_TASK lines in parallel
    local runs_per_task="${RUNS_PER_TASK:-1}" first last config failed=0
    local pids=() configs=()
    first=$(( ${SLURM_ARRAY_TASK_ID:?not an array task} * runs_per_task + 1 ))
    last=$(( first + runs_per_task - 1 ))
    while IFS= read -r config; do
        run_config "$config" &
        pids+=("$!")
        configs+=("$config")
    done < <(sed -n "${first},${last}p" "$1")
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "The run of ${configs[$index]} failed (see its run.log)." >&2
            failed=1
        fi
    done
    return "$failed"
}

case "$STAGE" in
    spawn_calibration)
        LAST_COUPLING="$("$PYTHON" -c "print(1.0 - 2.0 / $CALIBRATION_BINS)")"  # Grid -1, -1 + 2/K, ..., 1 - 2/K
        spawn "$EXPERIMENT_DIR/calibration.ini" "$CALIBRATION_JOBS" -1.0 "$LAST_COUPLING" \
            "$((CALIBRATION_BINS - 1))" FixedGhostCouplingKernel ghost_coupling > "$EXPERIMENT_DIR/calibration_jobs.txt"
        ;;
    spawn_production)
        if [ "$LOG_ACTIVITY" = suggested ]; then
            LOG_ACTIVITY="$("$PYTHON" -c \
                "import json; print(json.load(open('$OUTPUT/bias.json'))['suggested_log_activity'])")"
        fi
        for scheme in "${SCHEME_LIST[@]}"; do
            spawn "$EXPERIMENT_DIR/$scheme.ini" "$PRODUCTION_JOBS" "$LOG_ACTIVITY" "$LOG_ACTIVITY" 0 \
                "$(section_of "$(get_value "$EXPERIMENT_DIR/$scheme.ini" grand_canonical_kernel)")" log_activity
        done > "$EXPERIMENT_DIR/production_jobs.txt"
        ;;
    calibration) run_task "$EXPERIMENT_DIR/calibration_jobs.txt" ;;
    production) run_task "$EXPERIMENT_DIR/production_jobs.txt" ;;
    bias)
        FIGURE=()   # the figure needs matplotlib, which the cluster environment may not have
        if "$PYTHON" -c "import matplotlib" 2>/dev/null; then FIGURE=(--figure "$OUTPUT/bias.png"); fi
        "$PYTHON" sample_analysis/compute_ghost_coupling_bias.py "$OUTPUT/calibration" --output "$OUTPUT/bias.json" \
            --sub-bins "$SUB_BINS" ${FIGURE[@]+"${FIGURE[@]}"} \
            > "$EXPERIMENT_DIR/bias.log"
        "$PYTHON" config_files/grand_canonical/print_experiment_results.py bias "$OUTPUT/bias.json"
        config_files/grand_canonical/sulis_stage.sh "$EXPERIMENT_DIR" spawn_production
        ;;
    analysis)
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
        ;;
    *) echo "Unknown stage $STAGE." >&2; exit 1 ;;
esac
