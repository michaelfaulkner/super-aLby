from markov_chain_diagnostics import get_sample_mean_and_error
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import sample_getter
import sys

# Add the directory that contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")


def main(config_file_string):
    # use get_specific_heat() from sample_getter.py


    config = parsing.read_config(parsing.parse_options([config_file_string]).config_file)
    (config_file_mediator, potential, _, samplers, sample_directory, temperatures, number_of_equilibration_iterations,
     _, number_of_particles, _) = helper_methods.get_basic_config_data(config_file_string)
    
    if config_file_mediator == "metropolis_mediator":
            thinning_level = 10
    else:
            thinning_level = 1
   


    mean_arr_cv = np.zeros(len(temperatures))
    err_arr_cv = np.zeros(len(temperatures))

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("temperature", fontsize=15, labelpad=10)
    ax.set_ylabel("specific heat per particle", fontsize=15, labelpad=10)

  

    for temperature_index, temperature in enumerate(temperatures):
        print("---------------------------------")
        print(f"Temperature = {temperature:.4f}")
        for sample_index, sampler in enumerate(samplers):
            if sampler == "potential_sampler":
                # expected specific heat is \partial_T E[U] = beta^2 Var[U] (a dimensionless quantity) -- we
                # estimate beta^2 Var[U] / N (the expected specific heat per particle)

                
                specific_heat_mean_and_error = get_sample_mean_and_error(sample_getter.get_specific_heat(
                    sample_directory, temperature, temperature_index, number_of_particles,
                    number_of_equilibration_iterations, thinning_level))

                
                mean_arr_cv[temperature_index] = specific_heat_mean_and_error[0]/number_of_particles
                err_arr_cv[temperature_index] = specific_heat_mean_and_error[1]/number_of_particles
                
                
                print(specific_heat_mean_and_error)

        print("---------------------------------")

    ax.errorbar(temperatures, mean_arr_cv, err_arr_cv, marker=".", markersize=5, color="k")
    fig.savefig("output/cv_temp.png",bbox_inches="tight")

    




if __name__ == '__main__':
    main(sys.argv[1])
