import sys

def main(timestep, N):
    
    N = int(N)
    for i in range(N):
    
        f = open(f"/storage/eng/esrbjv/metropolis_001_scripts/{i}.sh", "w")
        f.write("#!/bin/bash \n" \
        "#SBATCH --job-name=super-alby-qo \n" \
        "#SBATCH --nodes=1 \n" \
        "#SBATCH --ntasks-per-node=1 \n" \
        "#SBATCH --time=48:00:0 \n" \
        "#SBATCH --mem-per-cpu=3988 \n" \
        "\n"
        "#SBATCH -e metropolis_text_output/qo_err_%j.txt \n" \
        "#SBATCH -o metropolis_text_output/qo_out_%j.txt \n" \
        "\n" \
        "module purge \n" \
        "module load GCC/11.3.0 OpenMPI/4.1.4 SciPy-bundle/2022.05 matplotlib/3.5.2 IPython/8.5.0 \n" \
        "\n")
        f.write('cd "${SLURM_SUBMIT_DIR}" \n' \
                '\n' \
                'echo "Running on host $(hostname)" \n' \
                'echo "Started on $(date)" \n' \
                'echo "Directory is $(pwd)" \n' \
                'echo "Slurm job ID is ${SLURM_JOBID}" \n' \
                'echo "This jobs runs on the following machines:" \n' \
                'echo "${SLURM_JOB_NODELIST}" \n' \
                'printf "\n\n" \n' \
                '\n' \
                f'python src/run.py src/config_files/qho/iact_data/metropolis/2/{i}.ini \n' \
                '\n' \
                'printf "\n\n" \n' \
                'echo "Ended on: $(date)") \n')
        
        f.close()

     




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])