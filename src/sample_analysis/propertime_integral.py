import numpy as np
import sys
import matplotlib.pyplot as plt 
import sample_getter
from markov_chain_diagnostics import get_cumulative_distribution, get_sample_mean_and_error

def analytical_x2(mass, omega, N_tau, timestep):

    auxilliary = 1.0 + 0.5 * timestep**2 * omega**2 - timestep * \
        omega * np.sqrt(1 + 0.25 * timestep**2 * omega**2)

    return (1.0 / (2.0 * mass * omega * np.sqrt(1.0 + 0.25 * timestep**2 * omega**2))) * ((1 + auxilliary**N_tau) /
                                                                                          (1 - auxilliary**N_tau))

def main():

    metrop_10 =  sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_10", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_5 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_5", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_2 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_2", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_1 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_1", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_075 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_075", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_05 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_05", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_025 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_025", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_01 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_01", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_005 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_005", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_001 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_001", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()
    metrop_7 = sample_getter.get_mean_squared_positions("output/convergence_tests"
    "/quantum_harmonic_oscillator_potential/metropolis_7", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()

    metrop_proper_time_dt = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_proper_time_integral/checkpoint_00_sample_of_auxillary_particles.npy")
    metrop_proper_time_x2 =  sample_getter.get_mean_squared_positions("output/" \
    "convergence_tests/quantum_harmonic_oscillator_potential/metropolis_proper_time_integral", 1.0, 0,
                                                              30, 10000
                                                              ).flatten()

    metrop_fixed_dt_array = [1.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]


    analytical_arr = np.zeros(np.shape(metrop_proper_time_dt))
    print(len(metrop_proper_time_dt))
    for i in range(len(metrop_proper_time_dt)):
        analytical_arr[i] = analytical_x2(1.0, 1.0, 30, 1/metrop_proper_time_dt[i])


    cutoff=10000

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(metrop_10))[:cutoff], metrop_10[:cutoff], label="dt=10.0")
    ax.plot(np.arange(len(metrop_5))[:cutoff], metrop_5[:cutoff], label="dt=5.0")
    ax.plot(np.arange(len(metrop_2))[:cutoff], metrop_2[:cutoff], label="dt=2.0")
    ax.plot(np.arange(len(metrop_075))[:cutoff], metrop_075[:cutoff], label="dt=0.75")
    ax.plot(np.arange(len(metrop_025))[:cutoff], metrop_025[:cutoff], label="dt=0.25")
    ax.plot(np.arange(len(metrop_005))[:cutoff], metrop_005[:cutoff], label="dt=0.05")
    ax.plot(np.arange(len(metrop_proper_time_x2))[:cutoff], metrop_proper_time_x2[:cutoff], label="propertime integral")

    ax.plot(np.arange(len(metrop_proper_time_dt))[:cutoff], analytical_arr[:cutoff], label="analytical")

    ax.set_xlabel("sample index")
    ax.set_ylabel("x^2")
    ax.legend()
    plt.savefig("x2_dt.png")

    plt.clf()
    fig, ax = plt.subplots(1,1)

    ax.plot(np.arange(len(metrop_proper_time_dt))[:cutoff], metrop_proper_time_dt[:cutoff])
    ax.set_xlabel("sample index")
    ax.set_ylabel("dt'")
    plt.savefig("dt.png")

    dim = np.shape(metrop_01)[0]

    fixed_dt_weighted = np.zeros((dim * 11))

    fixed_dt_weighted[:dim] = metrop_10
    fixed_dt_weighted[dim:dim*2] = metrop_7
    fixed_dt_weighted[dim*2:dim*3] = metrop_5
    fixed_dt_weighted[dim*3:dim*4] = metrop_2
    fixed_dt_weighted[dim*4:dim*5] = metrop_1
    fixed_dt_weighted[dim*5:dim*6] = metrop_075
    fixed_dt_weighted[dim*6:dim*7] = metrop_05
    fixed_dt_weighted[dim*7:dim*8] = metrop_025
    fixed_dt_weighted[dim*8:dim*9] = metrop_01
    fixed_dt_weighted[dim*9:dim*10] = metrop_005
    fixed_dt_weighted[dim*10:dim*11] = metrop_001


    fixed_cdf = get_cumulative_distribution(fixed_dt_weighted)
    propertime_cdf = get_cumulative_distribution(metrop_proper_time_x2)
    plt.clf()
    plt.plot(fixed_cdf[0], fixed_cdf[1], color='r', linewidth=3, linestyle='-',
                     label=f'fixed')
    plt.plot(propertime_cdf[0], propertime_cdf[1], color='k', linewidth=2, linestyle='-',
                label=f'propertime')
    #plt.vlines(bottom_of_well, 0 ,1.07,  linewidth=2, linestyle='-')

    plt.xlabel(r"$x$", fontsize=15, labelpad=10)
    plt.ylabel(r"$ F_n \left( X < x \right)$", fontsize=15, labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=14, pad=10)
    legend = plt.legend(loc='lower right', fontsize=10)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.tight_layout()
    #plt.show()
    plt.savefig("convergence_propertime.png")




if __name__ == "__main__":
    main()