import numpy as np
import os
import importlib
import matplotlib
import matplotlib.pyplot as plt
import sample_getter
import sys
from markov_chain_diagnostics import get_sample_mean_and_error


this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)
helper_methods = importlib.import_module("helper_methods")
parsing = importlib.import_module("base.parsing")
strings = importlib.import_module("base.strings")
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def analytical_x2(dim_m, N_tau):
    dim_omega = dim_m
    auxiliary = 1 + dim_omega ** 2 / 2 - dim_omega * np.sqrt(1 + dim_omega**2 / 4)
    return (1 / (2 * dim_m * dim_omega * np.sqrt(1 + 0.25 * dim_omega ** 2))) * (
            (1 + auxiliary ** N_tau) / (1 - auxiliary ** N_tau))

def str_to_timestep(str):
    split = list(str)
    if split[0] == "0":
        fl = f"0.{split[1]}"
        if len(split) > 2:
            for i in range(1,len(split[1:])):
                fl += f"{split[i+1]}"
        return float(fl)
    else:
        return float(str)


def main(values_filepath):
    r"""
    Produces plots comparing the numerical and analytical values of <x^2> for the 1D quantum harmonic oscillator
        potential.

    Parameters
        ----------
        values_filepath : str
            The path to the N and \delta \tau values file. This file is expected to be a .txt file, formatted like:
            1 001 10000 
            2 005 2000 
            etc.
            This is an artefact of using SLURM array jobs to produce several simulations with differing input values.
        config_folder : str
            The path to the folder containing the corresponding configuration files. These are expected to be named like
            metropolis_001_10000.ini
    """
    tau_values = np.loadtxt(values_filepath, dtype='str')


    analytical_x2_arr = np.zeros(len(tau_values))
    numerical_x2 = np.zeros(len(tau_values))
    timestep_arr = np.zeros(len(tau_values))
    N_arr = np.zeros(len(tau_values))
    m_arr = np.zeros(len(tau_values))


    for index, string in enumerate(tau_values):      
            mass = 1.0
            timestep = str_to_timestep(string)
            number_of_particles = 120 / timestep
            timestep_arr[index] = timestep
            if string != "001":
                sample_directory = f"output/metropolis/{string}"
                mean_sample = np.load(os.path.join(sample_directory,"temperature_00_sample_of_mean_positions.npy"))
                mean_sample_mean = np.mean(mean_sample)
                numerical_x2[index] = mean_sample_mean / timestep**2
            else:
                index_001 = index
   

    numerical_x2_e = np.zeros(len(tau_values))
    for index, string in enumerate(tau_values):

        mass = 1.0
        timestep = str_to_timestep(string)
        number_of_particles = 120 / timestep
        sample_directory = f"output/event_chain_mediator_lambda_50/{string}" 
        mean_sample = np.load(os.path.join(sample_directory,"temperature_00_sample_of_mean_positions.npy"))
        mean_sample = mean_sample[:30000]
        mean_sample_mean = np.mean(mean_sample)
        numerical_x2_e[index] = mean_sample_mean / timestep ** 2
        timestep_arr[index] = timestep
        analytical_x2_arr[index] = analytical_x2(mass * timestep, number_of_particles)
        
    timestep = 0.01
    number_of_equilibration_iterations = 1000
    metropolis_001_31k = np.load(
        "output/metropolis/mean_squared_positions/31000/temperature_00_sample_of_mean_positions_001_0.npy")
    metropolis_001_31k = np.mean(metropolis_001_31k)
    metropolis_001_31k = metropolis_001_31k / timestep **2
    numerical_x2[index_001] = metropolis_001_31k

    sub_arr_len = 51000
    num_sub_arrs = 20
    metropolis_001_10e6 = np.zeros(sub_arr_len * num_sub_arrs)
    for i in range(num_sub_arrs):
        metropolis_001_10e6[i * sub_arr_len : (i+1) * sub_arr_len] = np.load(
        f"output/metropolis_001_checkpoints/temperature_00_run_{i:02d}_sample_of_mean_positions.npy")[1:, 0]

    metropolis_001_10e6 = metropolis_001_10e6[number_of_equilibration_iterations + 1:-18999]
    metropolis_001_10e6 = np.mean(metropolis_001_10e6)
    metropolis_001_10e6 = metropolis_001_10e6 / timestep**2

    fig, ax = plt.subplots(1,1)
    ax.scatter(timestep_arr[:], analytical_x2_arr[:], marker=".", s = 200.0, color="purple", label="analytical")
    ax.scatter(timestep_arr[:], numerical_x2[:], marker="x", s = 200.0, color="#e8740d", label=r"Metropolis - $3\times 10^4$")
    ax.scatter(timestep, metropolis_001_10e6,  marker="", s = 100.0, color="#f5d20f", label=r"Metropolis - $1\times 10^6$ ")
    ax.scatter(timestep_arr[:], numerical_x2_e[:], marker="x", s = 200.0, color="#f974ef", label=r"ECMC - $3\times 10^4$")

    ax.set_xlabel(r"$\delta \tau$",  fontsize=20)
    ax.set_ylabel(r"$\langle x^2 \rangle$",  fontsize=20)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig("test.pdf")

    plt.clf()
    fig, ax = plt.subplots(1,1)
    ax.scatter(analytical_x2_arr[:], numerical_x2[:], marker = "x", color="#e16f04ff", label = r"Metropolis MC - $3\times 10^4$ samples")
    ax.scatter(analytical_x2_arr[index_001], metropolis_001_10e6,  marker="^", color="#0a48b4ff", label=r"Metropolis - $1\times 10^6$ samples ")

    ax.set_xlabel(r"Analytical $\langle x^2 \rangle$", fontsize = 15, labelpad = -4)
    ax.set_ylabel(r"Numerical $\langle x^2 \rangle$", fontsize = 15, labelpad = -4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig("test0.pdf")

    plt.clf()
    fig, ax = plt.subplots(1,1)
    ax.scatter(analytical_x2_arr[:], numerical_x2_e[:], marker = "x",  color="#e20acdff", label = r"ECMC - $3\times 10^4$ samples")

    ax.set_xlabel(r"Analytical $\langle x^2 \rangle$", fontsize = 15, labelpad = -4)
    ax.set_ylabel(r"Numerical $\langle x^2 \rangle$", fontsize = 15, labelpad = -4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig("test1.pdf")
  
    #print(f"analytical 0.01: {analytical_x2_arr[-1]}, ecmc: {numerical_x2_e[-1]}")
   

if __name__ == '__main__':
    main(sys.argv[1])
