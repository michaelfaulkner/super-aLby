import sys
import os
import errno

def main(timestep, N, equilibrium, samples, mass, total_T, initial_position, anharmonicity,
         omega_squared, input_save_str, dir, x_shift):
    timestep_str = timestep.replace(".", "")
    timestep_str = timestep_str.rstrip("0")
    mass_str = mass.replace(".", "")
    mass_str = mass_str.rstrip("0")
    if input_save_str is None:
        input_save_str = timestep_str
    timestep = float(timestep)
    mass = float(mass)
    total_T = int(total_T)
    initial_position = float(initial_position)
    anharmonicity = float(anharmonicity)
    omega_squared = float(omega_squared)
    x_shift = float(x_shift)
    N = int(N)
    for i in range(N):
        output_dir = f"config_files/hpc/anharmonic_metropolis_w2_20/{input_save_str}/{i}.ini"
        print(f"{output_dir}")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(os.path.dirname(output_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(output_dir, "w") as f:
            f.write("[Run] \n" \
            "mediator = metropolis_mediator \n" \
            "number_of_jobs = 1 \n" \
            "max_number_of_cpus = 1 \n" \
            "\n" \
            "[MetropolisMediator] \n" \
            "potential = quantum_harmonic_oscillator_potential \n" \
            "samplers = mean_squared_position_sampler \n" \
            "noise_distribution = unbounded_gaussian_noise_distribution\n" \
            "temperature = 1.0 \n" \
            f"number_of_equilibration_iterations = {equilibrium} \n" \
            f"number_of_observations = {samples} \n" \
            f"proposal_dynamics_adaptor_is_on = True \n" \
            f"output_directory = output/{dir}/{input_save_str}/{i} \n" \
            "\n" \
            "[QuantumHarmonicOscillatorPotential] \n" \
            f"mass = {mass} \n" \
            f"timestep = {timestep} \n" \
            f"anharmonicity = {anharmonicity} \n" \
            f"omega_squared = {omega_squared} \n" \
            f"x_shift = {x_shift} \n" \
            "\n" \
            "[MeanSquaredPositionSampler] \n" \
            f"x_shift = {x_shift} \n" \
            "\n" \
            "[UnboundedGaussianNoiseDistribution] \n" \
            "\n" \
            "[ModelSettings] \n" \
            "number_of_quantum_particles = 1 \n" \
           f"number_of_timeslices = {int(total_T / timestep)} \n" \
            "size_of_particle_space = 1 \n" \
            f"range_of_initial_particle_positions = {initial_position} \n")
        
        #f.close()

     




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8],
         sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12])