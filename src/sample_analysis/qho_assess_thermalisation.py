import numpy as np
import os
import importlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from markov_chain_diagnostics import get_cumulative_distribution

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")

def analytical_x2(dim_m, N_tau):
    dim_omega = dim_m
    auxilliary = 1 + dim_omega**2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)

    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega**2))) * ((1 + auxilliary**N_tau) / (1 - auxilliary**N_tau))


def main(config_file_string):
    config = parsing.read_config(
        parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperature, number_of_equilibration_iterations,
     number_of_observations, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)

    mass = parsing.get_value(config, strings.to_camel_case(potential), "mass")
    timestep = parsing.get_value(
        config, strings.to_camel_case(potential), "timestep")
    number_of_particles = parsing.get_value(
        config, "ModelSettings", "number_of_particles")

    temperature_index = 0
    thinning_level = None
    number_of_equilibration_iterations = 5000
    mean_sample = sample_getter.get_mean_positions(sample_directory, temperature, number_of_particles,
                                                   number_of_equilibration_iterations, thinning_level=thinning_level)

    mean = np.mean(mean_sample)
    std = np.std(mean_sample)

    analytical_value = analytical_x2(mass * timestep, number_of_particles)
    print(analytical_value)

    print(f"mean: {mean}, std: {std}")

    sub_arr_len = 51000
    num_sub_arrs = 20
    mean_sample_m = np.zeros(sub_arr_len * num_sub_arrs)
    for i in range(num_sub_arrs):
        mean_sample_m[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
        f"output/metropolis_001_checkpoints/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]
    mean_sample_m = mean_sample_m[:]
 
    scaled_sample = np.zeros(sub_arr_len * num_sub_arrs)
    for index in range(1, len(mean_sample_m)):
        scaled_sample[index] = np.sqrt(len(mean_sample_m)) * (mean_sample_m[index] - analytical_value) / std
    


    scaled_cdf = get_cumulative_distribution(scaled_sample)
    plt.plot(scaled_cdf[0], scaled_cdf[1])
    #plt.xlim(-3, 3)
    plt.savefig("test.png")
    plt.clf()

    scaled_cdf = get_cumulative_distribution(mean_sample_m)
    plt.plot(scaled_cdf[0], scaled_cdf[1])
    
    plt.savefig("test0.png")
    plt.clf()

    # normal = np.random.normal(0, 1, 1000)
    # cumulative = get_cumulative_distribution(normal)
    # plt.xlim(-3, 3)
    # plt.plot(cumulative[0], cumulative[1])
 
    # plt.savefig("cdf_test.png")


if __name__ == '__main__':
    main(sys.argv[1])
