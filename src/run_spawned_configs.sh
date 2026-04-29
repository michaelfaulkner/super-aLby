#!/bin/bash
set -euo pipefail
exec </dev/null
export PYTHONUNBUFFERED=1

cd "$PWD" || { echo "Failed to cd to $PWD"; exit 1; }
MAX_CPUS="${MAX_CPUS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc || echo 1)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

if [[ -z "${TEMPLATE_INI:-}" || -z "${NUM_JOBS:-}" || -z "${START:-}" || -z "${END:-}" || -z "${NUM_INCREMENTS:-}" || -z "${CONFIG_HEADER:-}" || -z "${CONFIG_VARIABLE:-}" ]]; then
    echo "Error: Required environment variables not set."
    echo "Set TEMPLATE_INI, NUM_JOBS, START, END, NUM_INCREMENTS, CONFIG_HEADER, CONFIG_VARIABLE."
    exit 1
fi

if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || [ "$NUM_JOBS" -lt 1 ]; then
    echo "Error: NUM_JOBS must be an integer greater than or equal to 1 (got '$NUM_JOBS')." >&2
    exit 1
fi

if ! [[ "$MAX_CPUS" =~ ^[0-9]+$ ]] || [ "$MAX_CPUS" -lt 1 ]; then
    echo "Error: MAX_CPUS must be an integer greater than or equal to 1 (got '$MAX_CPUS')." >&2
    exit 1
fi

if ! [[ "$NUM_INCREMENTS" =~ ^[0-9]+$ ]] || [ "$NUM_INCREMENTS" -lt 0 ]; then
    echo "Error: NUM_INCREMENTS must be an integer greater than or equal to 0 (got '$NUM_INCREMENTS')." >&2
    exit 1
fi

if ! awk -v header="$CONFIG_HEADER" -v var="$CONFIG_VARIABLE" '
    $0 ~ "^[[:space:]]*\\["header"\\][[:space:]]*$" { in_section=1; next }
    in_section && /^\[/ { in_section=0 }
    in_section && $1 == var { found=1; exit }
    END { exit(found ? 0 : 1) }
' "$TEMPLATE_INI"; then
    echo "Error: CONFIG_VARIABLE '$CONFIG_VARIABLE' not found under section [$CONFIG_HEADER] in $TEMPLATE_INI" >&2
    exit 1
fi

INI_VALUE="$(awk -F '=' -v key="$CONFIG_VARIABLE" '
    $1 ~ ("^"key"[[:space:]]*$") { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2 }
' "$TEMPLATE_INI")"
if [[ "$INI_VALUE" != "$START" ]]; then
    echo "Error: START ($START) does not match value of $CONFIG_VARIABLE ($INI_VALUE) in $TEMPLATE_INI" >&2
    exit 1
fi

if [ "$NUM_INCREMENTS" -eq 0 ] && [ "$END" != "$START" ]; then
    echo "Error: NUM_INCREMENTS=0 requires END ($END) == START ($START)" >&2
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
        ./super-alby "$CONFIG_FILE" || { echo "Simulation failed for $CONFIG_FILE"; exit 1; }
    done
}

fail=0
declare -a PIDS=()

for i in $(seq 0 $((NUM_INCREMENTS))); do
    run_task "$i" &
    PIDS+=($!)
    if [ "${#PIDS[@]}" -ge "$MAX_CPUS" ]; then
        if ! wait "${PIDS[0]}"; then
            fail=1
        fi
        PIDS=("${PIDS[@]:1}")
    fi
done

for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        fail=1
    fi
done

if [ "$fail" -eq 0 ] && [ -n "${BASE_DIR:-}" ] && [ -d "$BASE_DIR" ]; then
    rm -rf -- "$BASE_DIR"
fi

exit $fail
