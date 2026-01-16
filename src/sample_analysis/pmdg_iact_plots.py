import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(iact_data_path, N):

    ecmc_data_path = os.path.join(iact_data_path, "ecmc_iacf") 
    metropolis_data_path = os.path.join(iact_data_path, "metropolis_iacf")

    N = int(N)
    ecmc_timestep_data = np.load(os.path.join(ecmc_data_path, "iact_ecmc_0.npy"))[:, 1]
    storage_arr = np.zeros((len(timestep_data), N))
    sorted_timestep = timestep_data[np.argsort(timestep_data)]

    for index in range(N):
        iact_timestep =  np.load(os.path.join(iact_data_path, f"iact_ecmc_{index}.npy"))
        iact_data = iact_timestep[:, 0] 
        timestep_data = iact_timestep[:, 1] 
        argsorted_data = np.argsort(timestep_data)
        timestep_argsorted = timestep_data[argsorted_data]
        iact_data = iact_data[argsorted_data] 
        storage_arr[:, index] = iact_data



    iact_mean_arr = np.mean(storage_arr, axis = 1)
    err = np.std(storage_arr, axis=1)
