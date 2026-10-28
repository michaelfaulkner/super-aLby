#!/bin/bash
# Submits one grand-canonical experiment to Slurm (called by run_experiment.sh when experiment.env has SULIS=true):
#   config_files/grand_canonical/sulis_experiment.sh <experiment_dir>
# Stages (STAGES as for run_experiment.sh) are chained with --dependency=afterok:
#   calibration  array job, one task per calibration run;
#   bias         one task: bias.json, then the production job configs (at the suggested or fixed log activity);
#   production   array job, one task per production run;
#   analysis     one task: analysis.json and the results (in its Slurm log, output/slurm/).
# Slurm settings (account, partition, memory, time per stage) come from experiment.env (the sulis_* spec keys).
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

mkdir -p output/slurm
SHORT_NAME="$(echo "$NAME" | tr '/' '_' | cut -c1-40)"
COMMON=(--parsable --account="$SULIS_ACCOUNT" --partition="$SULIS_PARTITION" --nodes=1 --ntasks-per-node=1
        --cpus-per-task=1 --mem-per-cpu="$SULIS_MEMORY_PER_CPU")
DEPENDENCY=()
submit() {   # submit <stage> <time> [<array size>]: submits one stage after the previous one, prints its job id
    local array=(--output=output/slurm/slurm-%j.out)
    if [ "$#" -eq 3 ]; then array=(--array=0-$(( $3 - 1 )) --output=output/slurm/slurm-%A_%a.out); fi
    JOB="$(sbatch "${COMMON[@]}" ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} ${array[@]+"${array[@]}"} --time="$2" \
        --job-name="${SHORT_NAME}_$1" "$STAGE_SCRIPT" "$EXPERIMENT_DIR" "$1")"
    DEPENDENCY=(--dependency=afterok:"$JOB")
}

echo "Submitting the grand-canonical experiment $NAME to Slurm (schemes: $(IFS=,; echo "${SCHEME_LIST[*]}" |
    sed 's/,/, /g'); log activity $LOG_ACTIVITY)."
if has_stage calibration && $NEEDS_CALIBRATION; then
    "$STAGE_SCRIPT" "$EXPERIMENT_DIR" spawn_calibration
    TOTAL="$(wc -l < "$EXPERIMENT_DIR/calibration_jobs.txt" | tr -d ' ')"
    submit calibration "$SULIS_CALIBRATION_TIME" "$TOTAL"
    echo "Calibration: Slurm job $JOB ($TOTAL runs)."
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
    echo "Production: Slurm job $JOB ($TOTAL runs)."
fi
if has_stage analysis; then
    submit analysis "$SULIS_ANALYSIS_TIME"
    echo "Analysis: Slurm job $JOB (the results will be in output/slurm/slurm-$JOB.out)."
fi
