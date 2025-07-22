#!/bin/bash
#SBATCH --job-name=O1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --output=output/slurm/slurm-%j.out
#SBATCH --error=output/slurm/slurm-%j.out

module purge
module load GCC/11.3.0 OpenMPI/4.1.4
module load matplotlib/3.5.2

cd "${SLURM_SUBMIT_DIR}" || { echo "Failed to cd to $SLURM_SUBMIT_DIR"; exit 1; }

python test.py
