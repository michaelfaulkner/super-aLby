#!/bin/bash
set -euo pipefail

cd "$PWD" || { echo "Failed to cd to $PWD"; exit 1; }
MAX_CPUS="${MAX_CPUS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc || echo 1)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if [[ -z "${TEMPLATE_INI:-}" || -z "${NUM_JOBS:-}" || -z "${START:-}" || -z "${END:-}" || -z "${NUM_INCREMENTS:-}" || -z "${CONFIG_HEADER:-}" || -z "${CONFIG_VARIABLE:-}" ]]; then
    echo "Error: Required environment variables not set."
    echo "Set TEMPLATE_INI, NUM_JOBS, START, END, NUM_INCREMENTS, CONFIG_HEADER, CONFIG_VARIABLE."
    exit 1
fi

TEMPLATE_BASENAME="$(basename "$TEMPLATE_INI" .ini)"
TEMPLATE_DIRNAME="$(dirname "$TEMPLATE_INI")"
BASE_DIR="${TEMPLATE_DIRNAME}/${TEMPLATE_BASENAME}"

if [ ! -d "$BASE_DIR" ]; then
    if [ -n "${INCREMENT_TYPE:-}" ]; then
        python spawn_configs.py "$TEMPLATE_INI" "$NUM_JOBS" "$START" "$END" "$NUM_INCREMENTS" "$CONFIG_HEADER" "$CONFIG_VARIABLE" "$INCREMENT_TYPE" || { echo "Generation failed"; exit 1; }
    else
        python spawn_configs.py "$TEMPLATE_INI" "$NUM_JOBS" "$START" "$END" "$NUM_INCREMENTS" "$CONFIG_HEADER" "$CONFIG_VARIABLE" || { echo "Generation failed"; exit 1; }
    fi
fi

run_task() {
    local INCREMENT_ID="$1"
    local SUBDIR
    SUBDIR="${BASE_DIR}/${CONFIG_VARIABLE}_$(printf "%02d" "$INCREMENT_ID")"
    local attempts=0

    until [ -d "$SUBDIR" ] && compgen -G "$SUBDIR"/*.ini >/dev/null; do
        attempts=$((attempts+1))
        if [ "$attempts" -gt 600 ]; then
            echo "Timeout waiting for $SUBDIR/*.ini"
            exit 1
        fi
        sleep 1
    done

    local CONFIG_FILE
    for CONFIG_FILE in "$SUBDIR"/*.ini; do
        python run.py "$CONFIG_FILE" || { echo "run.py failed for $CONFIG_FILE"; exit 1; }
    done
}

active_jobs() { jobs -r | wc -l | tr -d ' '; }

for i in $(seq 0 $((NUM_INCREMENTS))); do
    while [ "$(active_jobs)" -ge "$MAX_CPUS" ]; do sleep 1; done
    run_task "$i" &
done

wait
