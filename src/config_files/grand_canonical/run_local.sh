#!/bin/bash
# Spawns the configs of a grand-canonical template with spawn_configs.py and runs every job locally in parallel.
# Set the variables below, then run (from anywhere):
#   ./config_files/grand_canonical/run_local.sh
# Other scripts (run_experiment.sh) instead pass the seven sweep settings as arguments, which override the variables:
#   ./config_files/grand_canonical/run_local.sh <template.ini> <num_jobs> <start> <end> <num_increments> \
#       <config_header> <config_variable>
#
# The spawned configs are kept in <template_dir>/<template_name>/<config_variable>_XX/job_YY.ini and the outputs go to
# <output_directory of the template>/<config_variable>_XX/job_YY.  Jobs whose output already contains sim_params.json
# are skipped, so the script can be rerun to finish an interrupted sweep.  For a single parameter point use
# NUM_INCREMENTS = 0 and START = END = the template value.
set -euo pipefail

# ---- settings (relative paths are relative to src) ------------------------------------------------------------------
TEMPLATE_INI="config_files/grand_canonical/validation/ideal_gas_1d_tempered.ini"
NUM_JOBS=8
START=0.0
END=0.0
NUM_INCREMENTS=0
CONFIG_HEADER="TemperedGrandCanonicalKernel"
CONFIG_VARIABLE="log_activity"
INCREMENT_TYPE="${INCREMENT_TYPE:-linear}"   # linear or log
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.11}"
MAX_CPUS="${MAX_CPUS:-13}"
# ---------------------------------------------------------------------------------------------------------------------

if [ "$#" -eq 7 ]; then
    TEMPLATE_INI="$1"; NUM_JOBS="$2"; START="$3"; END="$4"; NUM_INCREMENTS="$5"; CONFIG_HEADER="$6"
    CONFIG_VARIABLE="$7"
elif [ "$#" -ne 0 ]; then
    echo "Usage: $0  (settings in the script), or" >&2
    echo "       $0 <template.ini> <num_jobs> <start> <end> <num_increments> <config_header> <config_variable>" >&2
    exit 1
fi

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # src
BASE_DIR="$(dirname "$TEMPLATE_INI")/$(basename "$TEMPLATE_INI" .ini)"

if [ ! -d "$BASE_DIR" ]; then
    "$PYTHON" spawn_configs.py "$TEMPLATE_INI" "$NUM_JOBS" "$START" "$END" "$NUM_INCREMENTS" "$CONFIG_HEADER" \
        "$CONFIG_VARIABLE" "$INCREMENT_TYPE"
fi

run_one() {
    local config_file="$1"
    local output_directory
    output_directory="$(awk -F '=' \
        '$1 ~ /^output_directory[[:space:]]*$/ { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2 }' "$config_file")"
    if [ -f "$output_directory/sim_params.json" ]; then
        echo "skip (done): $config_file"
        return 0
    fi
    mkdir -p "$output_directory"
    if "$PYTHON" run.py "$config_file" > "$output_directory/run.log" 2>&1; then
        echo "done: $config_file"
    else
        echo "FAILED: $config_file (see $output_directory/run.log)"
        return 1
    fi
}
export -f run_one
export PYTHON

find "$BASE_DIR" -name "job_*.ini" | sort | xargs -P "$MAX_CPUS" -I {} bash -c 'run_one "$@"' _ {}
