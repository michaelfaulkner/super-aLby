import sys
import os
import errno

def main(timestep, N, equilibrium, samples, prefactor, mass, total_T, initial_position, sampling_dist, anharmonicity,
         omega_squared, input_save_str, dir, x_shift, lifting_scheme, refreshment_dist):
    timestep_str = timestep.replace(".", "")
    timestep_str = timestep_str.rstrip("0")
    mass_str = mass.replace(".", "")
    mass_str = mass_str.rstrip("0")
    if input_save_str is None:
        input_save_str = timestep_str
    timestep = float(timestep)
    mass = float(mass)
    total_T = int(total_T)
    #initial_position = float(initial_position)
    sampling_dist = float(sampling_dist)
    anharmonicity = float(anharmonicity)
    omega_squared = float(omega_squared)
    x_shift = float(x_shift)
    N = int(N)
    refreshment_dist = float(refreshment_dist)
    for i in range(N):
        output_dir = f"config_files/hpc/anharmonic_w2_100_x0_5/{input_save_str}/{i}.ini"
        print(f"{output_dir}")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(os.path.dirname(output_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(output_dir, "w") as f:
            f.write("[Run] \n" \
            "mediator = event_chain_mediator \n" \
            "number_of_jobs = 1 \n" \
            "max_number_of_cpus = 1 \n" \
            "\n" \
            "[EventChainMediator] \n" \
            "potential = quantum_harmonic_oscillator_potential \n" \
            "samplers = mean_squared_position_sampler \n" \
            "factor_field = no_factor_field \n" \
            "temperature = 1.0 \n" \
            f"number_of_equilibration_iterations = {equilibrium} \n" \
            f"number_of_observations = {samples} \n" \
            f"normalised_distance_between_measurements = {sampling_dist} \n" \
            f"refreshment_distribution = constant_refreshment_distribution \n" \
            f"output_directory = output/{dir}/{input_save_str}/{i} \n" \
            "\n" \
            "[QuantumHarmonicOscillatorPotential] \n" \
            f"mass = {mass} \n" \
            f"timestep = {timestep} \n" \
            f"anharmonicity = {anharmonicity} \n" \
            f"omega_squared = {omega_squared} \n" \
            f"x_shift = {x_shift} \n" \
            f"fixed_lifting_scheme = {lifting_scheme} \n" \
            "\n" \
            "[NoFactorField] \n" \
            "\n" \
            "[MeanSquaredPositionSampler] \n" \
            f"x_shift = {x_shift} \n" \
            "\n" \
            "[ConstantRefreshmentDistribution] \n" \
            f"normalised_refreshment_distance = {refreshment_dist} \n " \
            "\n" \
            "[ModelSettings] \n" \
            "number_of_quantum_particles = 1 \n" \
           f"number_of_timeslices = {int(total_T / timestep)} \n" \
            "size_of_particle_space = 1 \n" \
            f"range_of_initial_particle_positions = {initial_position} \n")
        
        #f.close()

     




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8],
         sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16])