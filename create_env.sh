#!/usr/bin/env bash
set -euo pipefail

# Parse optional flags
NOCPP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --nocpp|-n) NOCPP=1; shift ;;
    *) echo "Unknown option: $1" >&2; echo "Usage: $(basename "$0") [--nocpp|-n]" >&2; exit 2 ;;
  esac
done

# Resolve script path (follow symlinks) and set REPO_ROOT robustly
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
if [ "$(basename "$REPO_ROOT")" = "scripts" ]; then
  REPO_ROOT="$(dirname "$REPO_ROOT")"
fi

ENV_FILE="$REPO_ROOT/environment.yml"
ENV_NAME="super-alby"

# Prefer mamba if available
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi

# Create or update the conda environment (conda-forge only)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating conda env: $ENV_NAME"
  "$SOLVER" env update -f "$ENV_FILE" -n "$ENV_NAME" --prune
else
  echo "Creating conda env: $ENV_NAME"
  "$SOLVER" env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

if [ "$NOCPP" -eq 1 ]; then
  echo "Skipping C++ build (nocpp flag set)."
else
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
  make -C "$REPO_ROOT" clean
  if make -C "$REPO_ROOT"; then
    echo "C++ extension built successfully."
  else
    echo "Error building C++ extension. Check the output above for details." >&2
    exit 1
  fi
  conda deactivate
fi

# Create a simple runner wrapper only when creating/updating the env.
WRAPPER="$REPO_ROOT/super-alby"
cat <<'WRAPPER_EOF' > "$WRAPPER"
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="super-alby"
RUN_PY="$REPO_ROOT/src/run.py"

if [ ! -f "$RUN_PY" ]; then
  echo "Error: $RUN_PY not found." >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Install conda/micromamba/mamba." >&2
  exit 3
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' not found. Create it with: bash scripts/create_env.sh" >&2
  exit 4
fi
exec conda run -n "$ENV_NAME" --no-capture-output python "$RUN_PY" "$@"
WRAPPER_EOF

chmod +x "$WRAPPER"

echo "Environment '$ENV_NAME' ready. Wrapper created at '$WRAPPER'." 
echo "Use './super-alby <config-file>' to run."
