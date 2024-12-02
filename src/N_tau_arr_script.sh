#!/bin/bash

#SBATCH --job-name=super-alby-qo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:0
#SBATCH --mem-per-cpu=3988
#SBATCH --array=1-6
## creating 7 sub-jobs numbered 1-7

## Direct output to the following files.
## (The %j is replaced by the job id.)
#SBATCH -e qo_err_%j.txt
#SBATCH -o qo_out_%j.txt

# Just in case this is not loaded already...

module purge
module load GCC/11.3.0 OpenMPI/4.1.4 SciPy-bundle/2022.05 matplotlib/3.5.2 IPython/8.5.0
# module load languages/intel/2020-u4
# module add languages/anaconda3/2022.11-3.9.13

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}" 
printf "\n\n"


# Specify the path to the k values file
inputs=src/N_tau_values_50.txt

# Extract the t value for the current $SLURM_ARRAY_TASK_ID
tau=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $inputs)
N=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $inputs)



# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID and the k value
echo "This is array task ${SLURM_ARRAY_TASK_ID}, calculating with config file metropolis_${tau}_${N}.ini"


tasknum=${SLURM_ARRAY_TASK_ID}

# Submit
python src/run.py src/config_files/convergence_tests/one_dim_quantum_oscillator_potential/N_tau_50/metropolis_${tau}_${N}.ini
# Output the end time
printf "\n\n"
echo "Ended on: $(date)"