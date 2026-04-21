#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="super-alby"

# locate conda base and source conda.sh for non-interactive shells
if [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "conda not found in PATH." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# choose solver
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
else
  SOLVER="conda"
fi

# ensure the conda env exists
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' not found. Create it with scripts/create_env.sh" >&2
  exit 1
fi

# activate and install rpy2 (optional Python<->R bridge)
conda activate "$ENV_NAME"
pip install --upgrade pip
pip install rpy2

# ensure R is available inside the env; if not, try to install r-base from conda-forge
if ! conda run -n "$ENV_NAME" Rscript --version >/dev/null 2>&1; then
  echo "R not present in environment '$ENV_NAME'. Installing r-base from conda-forge..."
  $SOLVER install -n "$ENV_NAME" -c conda-forge r-base -y
fi

# install required CRAN packages (with dependencies) non-interactively using Rscript inside the env
echo "Installing CRAN packages: LaplacesDemon, mcmcse (and dependencies)..."
conda run -n "$ENV_NAME" Rscript -e 'options(repos="https://cloud.r-project.org"); install.packages(c("LaplacesDemon","mcmcse"), dependencies=TRUE)'

echo "rpy2 and required R packages installed into '$ENV_NAME'."
