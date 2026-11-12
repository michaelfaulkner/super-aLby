#!/bin/bash
# Submits one grand-canonical experiment to Slurm (called by run_experiment.sh when experiment.env has SULIS=true):
#   config_files/grand_canonical/sulis_experiment.sh <experiment_dir>
# Stages (STAGES as for run_experiment.sh) are chained with --dependency=afterok:
#   calibration  array job over the calibration runs;
#   bias         one task: bias.json, then the production job configs (at the suggested or fixed log activity);
#   production   array job over the production runs;
#   analysis     one task: analysis.json and the results (in its Slurm log, output/slurm/).
# Slurm settings (account, partition, memory, time per stage) come from experiment.env (the sulis_* spec keys).
# Job limit: Slurm counts every array task as one submitted job, and Sulis allows SULIS_MAX_JOBS (sulis_max_jobs,
# default 500) per user. This experiment gets an equal share of what is left after the jobs already queued, split
# between the jobs of its stages: (SULIS_MAX_JOBS - queued) / SULIS_EXPERIMENTS_LEFT, where run_experiment.sh sets
# SULIS_EXPERIMENTS_LEFT to the sweep points not yet submitted (this one included; 1 for a single experiment). When a
# stage has more runs than its share of tasks, each task runs a block of runs in parallel, one per CPU (RUNS_PER_TASK,
# at most SULIS_CPUS_PER_NODE = 128), so the time limit of a task still fits one run.
set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -f "$1/experiment.env" ]; then
    echo "Usage: $0 <experiment_dir>   (a folder written by make_experiments.py)" >&2
    exit 1
fi
EXPERIMENT_DIR="$(cd "$1" && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # src: Slurm jobs start here (SLURM_SUBMIT_DIR)
EXPERIMENT_DIR="${EXPERIMENT_DIR#"$PWD/"}"
ALL_STAGES="calibration bias production analysis"
STAGES="${STAGES:-$ALL_STAGES}"
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
export PYTHON
source "$EXPERIMENT_DIR/experiment.env"
read -r -a SCHEME_LIST <<< "$SCHEMES"
STAGE_SCRIPT=config_files/grand_canonical/sulis_stage.sh
if ! command -v sbatch > /dev/null; then
    echo "sulis = true, but sbatch was not found: run this on Sulis (or set sulis = false to run locally)." >&2
    exit 1
fi
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
NEEDS_CALIBRATION=false
if [ "$LOG_ACTIVITY" = suggested ]; then NEEDS_CALIBRATION=true; fi
for scheme in "${SCHEME_LIST[@]}"; do
    if grep -q "tabulated_ghost_coupling_bias" "$EXPERIMENT_DIR/$scheme.ini"; then NEEDS_CALIBRATION=true; fi
done
# An existing bias: if tempered.ini reads its bias from somewhere other than <output>/bias.json, copy it there (the
# production log activity and the analysis read it there) and skip the calibration and bias stages.
if [ -f "$EXPERIMENT_DIR/tempered.ini" ] && [[ " ${SCHEME_LIST[*]} " == *" tempered "* ]]; then
    TEMPERED_BIAS="$(awk -F '=' '$1 ~ /^file_name[[:space:]]*$/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit }' \
        "$EXPERIMENT_DIR/tempered.ini")"
    if [ -n "$TEMPERED_BIAS" ] && [ "$TEMPERED_BIAS" != "$OUTPUT/bias.json" ]; then
        if [ ! -f "$TEMPERED_BIAS" ]; then
            echo "tempered.ini reads the bias $TEMPERED_BIAS, which does not exist." >&2
            exit 1
        fi
        mkdir -p "$OUTPUT"
        cp "$TEMPERED_BIAS" "$OUTPUT/bias.json"
        NEEDS_CALIBRATION=false
    fi
fi

mkdir -p output/slurm
SHORT_NAME="$(echo "$NAME" | tr '/' '_' | rev | cut -c1-40 | rev)"   # the end of the name (the sweep point)
COMMON=(--parsable --account="$SULIS_ACCOUNT" --partition="$SULIS_PARTITION" --nodes=1 --ntasks-per-node=1)
MEMORY=(--mem-per-cpu="$SULIS_MEMORY_PER_CPU")
BIAS_MEMORY=(--mem="${SULIS_BIAS_MEMORY:-8G}")   # the bias stage loads every calibration sample at once

ARRAY_STAGES=0   # the stages this script will submit: array stages (one job per task) and single jobs
SINGLE_JOBS=0
if has_stage calibration && $NEEDS_CALIBRATION; then ARRAY_STAGES=$(( ARRAY_STAGES + 1 )); fi
if has_stage bias && $NEEDS_CALIBRATION; then SINGLE_JOBS=$(( SINGLE_JOBS + 1 )); fi
if has_stage production; then ARRAY_STAGES=$(( ARRAY_STAGES + 1 )); fi
if has_stage analysis; then SINGLE_JOBS=$(( SINGLE_JOBS + 1 )); fi
MAX_JOBS="${SULIS_MAX_JOBS:-500}"
CPUS_PER_NODE="${SULIS_CPUS_PER_NODE:-128}"
QUEUED="$(squeue -u "$USER" -r -h 2> /dev/null | wc -l | tr -d ' ')"
BUDGET=$(( (MAX_JOBS - QUEUED) / ${SULIS_EXPERIMENTS_LEFT:-1} ))
TASKS_PER_STAGE=1
if [ "$ARRAY_STAGES" -gt 0 ]; then TASKS_PER_STAGE=$(( (BUDGET - SINGLE_JOBS) / ARRAY_STAGES )); fi
if [ "$TASKS_PER_STAGE" -lt 1 ]; then
    echo "Not enough room under the Slurm job limit ($MAX_JOBS, $QUEUED queued) for $NAME: wait for queued jobs to" \
        "finish, or raise sulis_max_jobs if the limit is higher." >&2
    exit 1
fi

DEPENDENCY=()
submit() {   # submit <stage> <time> [<runs>]: submits one stage after the previous one (an array job over <runs> runs,
             # packed into at most TASKS_PER_STAGE tasks of RUNS_PER_TASK runs each); sets JOB, TASKS and RUNS_PER_TASK
    local array=(--output=output/slurm/slurm-%j.out)
    TASKS=1
    RUNS_PER_TASK=1
    if [ "$#" -eq 3 ]; then
        RUNS_PER_TASK=$(( ($3 + TASKS_PER_STAGE - 1) / TASKS_PER_STAGE ))
        TASKS=$(( ($3 + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
        if [ "$RUNS_PER_TASK" -gt "$CPUS_PER_NODE" ]; then
            echo "The $1 stage of $NAME would need $RUNS_PER_TASK CPUs per task (more than one node): wait for" \
                "queued jobs to finish, or submit fewer experiments at once." >&2
            exit 1
        fi
        array=(--array=0-$(( TASKS - 1 )) --output=output/slurm/slurm-%A_%a.out)
    fi
    local memory=("${MEMORY[@]}")
    if [ "$1" = bias ]; then memory=("${BIAS_MEMORY[@]}"); fi
    JOB="$(RUNS_PER_TASK="$RUNS_PER_TASK" sbatch "${COMMON[@]}" --cpus-per-task="$RUNS_PER_TASK" "${memory[@]}" \
        ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} ${array[@]+"${array[@]}"} --time="$2" \
        --job-name="${SHORT_NAME}_$1" "$STAGE_SCRIPT" "$EXPERIMENT_DIR" "$1")"
    DEPENDENCY=(--dependency=afterok:"$JOB")
}
packing() {   # how the runs of the last array stage were packed
    if [ "$RUNS_PER_TASK" -eq 1 ]; then echo "one task per run"; else
        echo "$TASKS tasks of up to $RUNS_PER_TASK runs in parallel"; fi
}

echo "Submitting the grand-canonical experiment $NAME to Slurm (schemes: $(IFS=,; echo "${SCHEME_LIST[*]}" |
    sed 's/,/, /g'); log activity $LOG_ACTIVITY)."
if has_stage calibration && $NEEDS_CALIBRATION; then
    "$STAGE_SCRIPT" "$EXPERIMENT_DIR" spawn_calibration
    TOTAL="$(wc -l < "$EXPERIMENT_DIR/calibration_jobs.txt" | tr -d ' ')"
    submit calibration "$SULIS_CALIBRATION_TIME" "$TOTAL"
    echo "Calibration: Slurm job $JOB ($TOTAL runs, $(packing))."
fi
if has_stage bias && $NEEDS_CALIBRATION; then
    submit bias "$SULIS_BIAS_TIME"
    echo "Bias: Slurm job $JOB (also writes the production job configs)."
elif has_stage production; then
    if $NEEDS_CALIBRATION && [ ! -f "$OUTPUT/bias.json" ]; then
        echo "The bias is missing: include the calibration and bias stages." >&2
        exit 1
    fi
    "$STAGE_SCRIPT" "$EXPERIMENT_DIR" spawn_production
fi
if has_stage production; then
    TOTAL=$(( PRODUCTION_JOBS * ${#SCHEME_LIST[@]} ))
    submit production "$SULIS_PRODUCTION_TIME" "$TOTAL"
    echo "Production: Slurm job $JOB ($TOTAL runs, $(packing))."
fi
if has_stage analysis; then
    submit analysis "$SULIS_ANALYSIS_TIME"
    echo "Analysis: Slurm job $JOB (the results will be in output/slurm/slurm-$JOB.out)."
fi
