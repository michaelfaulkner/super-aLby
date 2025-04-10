import sys

def main(timestep, N):
    
    N = int(N)
    for i in range(N):
    
        f = open(f"src/config_files/qho/iact_data/metropolis/{timestep}/{i}.ini", "w")
        f.write("[Run] \n" \
        "mediator = metropolis_mediator \n" \
        "number_of_jobs = 1 \n" \
        "max_number_of_cpus = 1 \n" \
        "\n" \
        "[MetropolisMediator] \n" \
        "potential = quantum_harmonic_oscillator_potential \n" \
        "samplers = mean_squared_position_sampler \n" \
        "noise_distribution = gaussian_noise_distribution \n" \
        "minimum_temperature = 1.0 \n" \
        "maximum_temperature = 1.0 \n" \
        "number_of_temperature_increments = 0 \n" \
        "number_of_equilibration_iterations = 1000 \n" \
        "number_of_observations = 80000 \n" \
        "proposal_dynamics_adaptor_is_on = True \n" \
        "\n" \
        "[QuantumHarmonicOscillatorPotential] \n" \
        "mass = 1.0 \n" \
        "timestep = 0.1 \n" \
        "\n" \
        "[MeanSquaredPositionSampler] \n" \
        f"output_directory = output/iact_data/metropolis/{timestep}/{i} \n" \
        "\n" \
        "[GaussianNoiseDistribution] \n" \
        "\n" \
        "[ModelSettings] \n" \
        "number_of_particles = 1200 \n" \
        "size_of_particle_space = 1 \n" \
        "range_of_initial_particle_positions = 0.0 \n")
        
        f.close()

     




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])