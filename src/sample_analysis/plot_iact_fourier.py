import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main():

    iact_0_mode_ecmc = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/0_mode/iact.npy")
    iact_0_mode_metrop = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/0_mode_metrop/iact.npy")

    iact_0_amplitude_ecmc = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/0_amplitude/iact.npy")
    iact_0_amplitude_metrop = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/0_amplitude_metrop/iact.npy")

    iact_1_amplitude_ecmc = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/1_amplitude/iact.npy")
    iact_1_amplitude_metrop = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/1_amplitude_metrop/iact.npy")

    iact_2_amplitude_ecmc = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/2_amplitude/iact.npy")
    iact_2_amplitude_metrop = np.load("/home/raichkel/super-aLby-private/src/output/fourier_modes/iact/2_amplitude_metrop/iact.npy")

    iact_0_mode_ecmc_mean = np.mean(iact_0_mode_ecmc)
    iact_0_mode_ecmc_err = np.std(iact_0_mode_ecmc)
    iact_0_mode_metrop_mean = np.mean(iact_0_mode_metrop)
    iact_0_mode_metrop_err = np.std(iact_0_mode_metrop)

    iact_0_amplitude_ecmc_mean = np.mean(iact_0_amplitude_ecmc)
    iact_0_amplitude_ecmc_err = np.std(iact_0_amplitude_ecmc)
    iact_0_amplitude_metrop_mean = np.mean(iact_0_amplitude_metrop)
    iact_0_amplitude_metrop_err = np.std(iact_0_amplitude_metrop)

    iact_1_amplitude_ecmc_mean = np.mean(iact_1_amplitude_ecmc)
    iact_1_amplitude_ecmc_err = np.std(iact_1_amplitude_ecmc)
    iact_1_amplitude_metrop_mean = np.mean(iact_1_amplitude_metrop)
    iact_1_amplitude_metrop_err = np.std(iact_1_amplitude_metrop)

    iact_2_amplitude_ecmc_mean = np.mean(iact_2_amplitude_ecmc)
    iact_2_amplitude_ecmc_err = np.std(iact_2_amplitude_ecmc)
    iact_2_amplitude_metrop_mean = np.mean(iact_2_amplitude_metrop)
    iact_2_amplitude_metrop_err = np.std(iact_2_amplitude_metrop)

    print(f"0 coeff: ECMC = {iact_0_mode_ecmc_mean:.2f} +- {iact_0_mode_ecmc_err:.2f}, Metrop = {iact_0_mode_metrop_mean:.2f}"
          f" +- {iact_0_mode_metrop_err:.2f}")
    print(f"0 amp: ECMC = {iact_0_amplitude_ecmc_mean:.2f} +- {iact_0_amplitude_ecmc_err:.2f}, Metrop = "
          f"{iact_0_amplitude_metrop_mean:.2f} +- {iact_0_amplitude_metrop_err:.2f}")
    print(f"1 amp: ECMC = {iact_1_amplitude_ecmc_mean:.2f} +- {iact_1_amplitude_ecmc_err:.2f}, Metrop = "
          f"{iact_1_amplitude_metrop_mean:.2f} +- {iact_1_amplitude_metrop_err:.2f}")
    print(f"2 amp: ECMC = {iact_2_amplitude_ecmc_mean:.2f} +- {iact_2_amplitude_ecmc_err:.2f}, Metrop = "
          f"{iact_2_amplitude_metrop_mean:.2f} +- {iact_2_amplitude_metrop_err:.2f}")


if __name__ == '__main__':
    main()