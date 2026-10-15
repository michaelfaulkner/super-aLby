import numpy as np
import sys
import matplotlib.pyplot as plt 


def main():

    metrop_10 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_10/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_5 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_5/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_3 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_3/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_2 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_2/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_15 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_15/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_1 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_1/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_075 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_075/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_05 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_05/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_025 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_025/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_01 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_01/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_005 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_005/checkpoint_00_sample_of_mean_squared_positions.npy")
    metrop_001 = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_001/checkpoint_00_sample_of_mean_squared_positions.npy")

    metrop_proper_time_dt = np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_proper_time_integral/checkpoint_00_sample_of_auxillary_particles.npy")
    metrop_proper_time_x2 =  np.load("/home/raichkel/super-aLby-private/src/output/convergence_tests/quantum_harmonic_oscillator_potential/metropolis_proper_time_integral/checkpoint_00_sample_of_mean_squared_positions.npy")

    metrop_fixed_dt_array = [1.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]

    cutoff=10000

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(len(metrop_10))[:cutoff], metrop_10[:cutoff], label="dt=10.0")
    ax.plot(np.arange(len(metrop_5))[:cutoff], metrop_5[:cutoff], label="dt=5.0")
    ax.plot(np.arange(len(metrop_3))[:cutoff], metrop_3[:cutoff], label="dt=3.0")
    ax.plot(np.arange(len(metrop_2))[:cutoff], metrop_2[:cutoff], label="dt=2.0")
    ax.plot(np.arange(len(metrop_15))[:cutoff], metrop_15[:cutoff], label="dt=1.5")
    #ax.plot(np.arange(len(metrop_075))[:cutoff], metrop_075[:cutoff], label="dt=0.75")
    ax.plot(np.arange(len(metrop_025))[:cutoff], metrop_025[:cutoff], label="dt=0.25")
    #ax.plot(np.arange(len(metrop_005))[:cutoff], metrop_005[:cutoff], label="dt=0.05")
    ax.plot(np.arange(len(metrop_proper_time_x2))[:cutoff], metrop_proper_time_x2[:cutoff], label="propertime integral")

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



if __name__ == "__main__":
    main()