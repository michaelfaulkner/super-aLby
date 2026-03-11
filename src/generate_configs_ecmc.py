import sys
import os
import errno


def main(timestep, N):

    timestep_str = timestep.replace(".", "")
    timestep_str = timestep_str.rstrip("0")
    timestep = float(timestep)

    N = int(N)
    for i in range(N):
        output_dir = f"config_files/qho_iact/ecmc/{timestep_str}/{i}.ini"
        if not os.path.exists(output_dir):
            try:
                os.makedirs(os.path.dirname(output_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(output_dir, "w") as f:
            f.write("[Run] \n" \
            "mediator =  event_chain_mediator \n" \
            "\n" \
            "[EventChainMediator] \n" \
            "potential = quantum_harmonic_oscillator_potential \n" \
            "samplers = mean_squared_position_sampler \n" \
            "temperature = 1.0 \n" \
            "number_of_equilibration_iterations = 1000 \n" \
            "number_of_observations = 80000 \n" \
            "normalised_distance_between_measurements = 1.666 \n" \
            f"output_directory = output/iact_data/ecmc/{timestep_str}/{i} \n" \
            "\n" \
            "[QuantumHarmonicOscillatorPotential] \n" \
            "mass = 1.0 \n" \
            f"timestep = {timestep} \n" \
            "omega_squared = 1.0 \n" \
            "\n" \
            "[MeanSquaredPositionSampler] \n" \
            "\n" \
            "[GaussianNoiseDistribution] \n" \
            "\n" \
            "[ModelSettings] \n" \
            "number_of_quantum_particles = 1 \n" \
            f"number_of_timeslices = {int(120 / timestep)} \n" \
            "size_of_particle_space = 1 \n" \
            "range_of_initial_particle_positions = 0.0 \n")
            
        #f.close()

     




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])