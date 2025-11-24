import sys

def main(timestep, N, equilibrium, samples, prefactor, mass, omega, number_of_timeslices):

    timestep_str = timestep.replace(".", "")
    timestep_str = timestep_str.rstrip("0")
    prefactor_str = prefactor.replace(".","")
    if float(prefactor) < 10.0:
        prefactor_str = prefactor_str.rstrip("0")
    elif prefactor_str[-1] == "0":
        prefactor_str = prefactor_str[:-1]
    else:
        pass 
    mass_str = mass.replace(".", "")
    mass_str = mass_str.rstrip("0")
    timestep = float(timestep)
    mass = float(mass)
    omega = float(omega)
    N = int(N)
    for i in range(N):
    
        f = open(f"config_files/factor_fields_b_vary_30/{prefactor_str}/{i}.ini", "w")
        f.write("[Run] \n" \
        "mediator = event_chain_mediator \n" \
        "number_of_jobs = 1 \n" \
        "max_number_of_cpus = 1 \n" \
        "\n" \
        "[EventChainMediator] \n" \
        "potential = quantum_harmonic_oscillator_potential \n" \
        "samplers = mean_squared_position_sampler \n" \
        "minimum_temperature = 1.0 \n" \
        "maximum_temperature = 1.0 \n" \
        "number_of_temperature_increments = 0 \n" \
        f"number_of_equilibration_iterations = {equilibrium} \n" \
        f"number_of_observations = {samples} \n" \
        "normalised_distance_between_measurements = 1.0 \n" \
        "\n" \
        "[QuantumHarmonicOscillatorPotential] \n" \
        f"mass = {mass} \n" \
        f"omega = {omega} \n" \
        f"timestep = {timestep} \n" \
	    "factor_fields = True \n" \
	    f"factor_fields_prefactor = {prefactor} \n" \
        "\n" \
        "[MeanSquaredPositionSampler] \n" \
        f"output_directory = output/factor_fields_b_vary_30/{prefactor_str}/{i} \n" \
        "\n" \
        "[ModelSettings] \n" \
	"number_of_quantum_particles = 1 \n" \
	f"number_of_timeslices = {int(number_of_timeslices)} \n" \
        "size_of_particle_space = 1 \n" \
        "range_of_initial_particle_positions = 0.0 \n")
        
        f.close()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])