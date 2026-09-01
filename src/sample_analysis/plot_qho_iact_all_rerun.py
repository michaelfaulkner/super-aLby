import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(metropolis_iact_data_path, ecmc_iact_data_path, metropolis_iact_data_path_2, ecmc_iact_data_path_2, ecmc_iact_data_path_3,
          ecmc_iact_data_path_4, ecmc_iact_data_path_5, ecmc_iact_data_path_6, ecmc_iact_data_path_7, ecmc_iact_data_path_8,
          ecmc_iact_data_path_9, ecmc_iact_data_path_10, ecmc_iact_data_path_11, ecmc_iact_data_path_12, ecmc_iact_data_path_13,
          ecmc_iact_data_path_14, ecmc_iact_data_path_15, ecmc_iact_data_path_16, ecmc_iact_data_path_17,
          ecmc_iact_data_path_18, ecmc_iact_data_path_19,  ecmc_iact_data_path_20,  ecmc_iact_data_path_21,
          ecmc_iact_data_path_22, ecmc_iact_data_path_23, ecmc_iact_data_path_24, ecmc_iact_data_path_25,
          ecmc_iact_data_path_26, ecmc_iact_data_path_27, ecmc_iact_data_path_28, N, ff_N, propertime):

    N = int(N)
    ff_N = int(ff_N)
    propertime = float(propertime)
    #print(f"{ecmc_iact_data_path_13}")

    try:

        metropolis_timestep_data = np.load(os.path.join(metropolis_iact_data_path, "iact_metropolis_0.npy"))[:, 1]
        metropolis_storage_arr = np.zeros((len(metropolis_timestep_data), N))
        metropolis_sorted_timestep = metropolis_timestep_data[np.argsort(metropolis_timestep_data)]
        for index in range(N):
            metropolis_iact_timestep =  np.load(os.path.join(metropolis_iact_data_path, f"iact_metropolis_{index}.npy"))
            metropolis_iact_data = metropolis_iact_timestep[:, 0] 
            metropolis_timestep_data = metropolis_iact_timestep[:, 1] 
            argsorted_data = np.argsort(metropolis_timestep_data)
            metropolis_iact_data = metropolis_iact_data[argsorted_data] 
            metropolis_storage_arr[:, index] = metropolis_iact_data

        metropolis_iact_mean_arr = np.mean(metropolis_storage_arr, axis = 1)
        metropolis_iact_mean_arr = metropolis_iact_mean_arr[metropolis_sorted_timestep >= 0.01]
        metropolis_err = np.std(metropolis_storage_arr, axis=1)
        metropolis_err = metropolis_err[metropolis_sorted_timestep >= 0.01]
        metropolis_sorted_timestep = metropolis_sorted_timestep[metropolis_sorted_timestep >= 0.01]
        metropolis_sorted_N = propertime / metropolis_sorted_timestep

        metrop_data = True
    except:
        metrop_data = False


    ecmc_timestep_data = np.load(os.path.join(ecmc_iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr = np.zeros((len(ecmc_timestep_data), N))
    ecmc_sorted_timestep = ecmc_timestep_data[np.argsort(ecmc_timestep_data)]

    for index in range(N):
        ecmc_iact_timestep =  np.load(os.path.join(ecmc_iact_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_iact_data = ecmc_iact_timestep[:, 0] 
        ecmc_timestep_data = ecmc_iact_timestep[:, 1] 
        argsorted_data = np.argsort(ecmc_timestep_data)
        ecmc_timestep_argsorted = ecmc_timestep_data[argsorted_data]
        ecmc_iact_data = ecmc_iact_data[argsorted_data] 
        ecmc_storage_arr[:, index] = ecmc_iact_data

    ecmc_iact_mean_arr = np.mean(ecmc_storage_arr, axis = 1)
    ecmc_iact_mean_arr = ecmc_iact_mean_arr[ecmc_sorted_timestep >= 0.01]
    ecmc_err = np.std(ecmc_storage_arr, axis=1)
    ecmc_err = ecmc_err[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_timestep = ecmc_sorted_timestep[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_N = propertime / ecmc_sorted_timestep

    metropolis_timestep_data_2 = np.load(os.path.join(metropolis_iact_data_path_2, "iact_metropolis_0.npy"))[:, 1]
    metropolis_storage_arr_2 = np.zeros((len(metropolis_timestep_data_2), N))
    metropolis_sorted_timestep_2 = metropolis_timestep_data_2[np.argsort(metropolis_timestep_data_2)]
    for index_2 in range(N):
        metropolis_iact_timestep_2 =  np.load(os.path.join(metropolis_iact_data_path_2, f"iact_metropolis_{index_2}.npy"))
        metropolis_iact_data_2 = metropolis_iact_timestep_2[:, 0] 
        metropolis_timestep_data_2 = metropolis_iact_timestep_2[:, 1] 
        argsorted_data_2 = np.argsort(metropolis_timestep_data_2)
        metropolis_iact_data_2 = metropolis_iact_data_2[argsorted_data_2] 
        metropolis_storage_arr_2[:, index_2] = metropolis_iact_data_2

    metropolis_iact_mean_arr_2 = np.mean(metropolis_storage_arr_2, axis = 1)
    metropolis_iact_mean_arr_2 = metropolis_iact_mean_arr_2[metropolis_sorted_timestep_2 >= 0.01]
    metropolis_err_2 = np.std(metropolis_storage_arr_2, axis=1)
    metropolis_err_2 = metropolis_err_2[metropolis_sorted_timestep_2 >= 0.01]
    metropolis_sorted_timestep_2 = metropolis_sorted_timestep_2[metropolis_sorted_timestep_2 >= 0.01]
    metropolis_sorted_N_2 = propertime / metropolis_sorted_timestep_2


    ecmc_timestep_data_2 = np.load(os.path.join(ecmc_iact_data_path_2, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_2 = np.zeros((len(ecmc_timestep_data_2), N))
    ecmc_sorted_timestep_2 = ecmc_timestep_data_2[np.argsort(ecmc_timestep_data_2)]

    for index_2 in range(N):
        ecmc_iact_timestep_2 =  np.load(os.path.join(ecmc_iact_data_path_2, f"iact_ecmc_{index_2}.npy"))
        ecmc_iact_data_2 = ecmc_iact_timestep_2[:, 0] 
        ecmc_timestep_data_2 = ecmc_iact_timestep_2[:, 1] 
        argsorted_data_2 = np.argsort(ecmc_timestep_data_2)
        ecmc_timestep_argsorted_2 = ecmc_timestep_data_2[argsorted_data_2]
        ecmc_iact_data_2 = ecmc_iact_data_2[argsorted_data_2] 
        ecmc_storage_arr_2[:, index_2] = ecmc_iact_data_2

    ecmc_iact_mean_arr_2 = np.mean(ecmc_storage_arr_2, axis = 1)
    ecmc_iact_mean_arr_2 = ecmc_iact_mean_arr_2[ecmc_sorted_timestep_2 >= 0.01]
    ecmc_err_2 = np.std(ecmc_storage_arr_2, axis=1)
    ecmc_err_2 = ecmc_err_2[ecmc_sorted_timestep_2 >= 0.01]
    ecmc_sorted_timestep_2 = ecmc_sorted_timestep_2[ecmc_sorted_timestep_2 >= 0.01]
    ecmc_sorted_N_2 = propertime / ecmc_sorted_timestep_2



    ecmc_timestep_data_3 = np.load(os.path.join(ecmc_iact_data_path_3, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_3 = np.zeros((len(ecmc_timestep_data_3), N))
    ecmc_sorted_timestep_3 = ecmc_timestep_data_3[np.argsort(ecmc_timestep_data_3)]

    for index_3 in range(N):
        ecmc_iact_timestep_3 =  np.load(os.path.join(ecmc_iact_data_path_3, f"iact_ecmc_{index_3}.npy"))
        ecmc_iact_data_3 = ecmc_iact_timestep_3[:, 0] 
        ecmc_timestep_data_3 = ecmc_iact_timestep_3[:, 1] 
        argsorted_data_3 = np.argsort(ecmc_timestep_data_3)
        ecmc_timestep_argsorted_3 = ecmc_timestep_data_3[argsorted_data_3]
        ecmc_iact_data_3 = ecmc_iact_data_3[argsorted_data_3] 
        ecmc_storage_arr_3[:, index_3] = ecmc_iact_data_3

    ecmc_iact_mean_arr_3 = np.mean(ecmc_storage_arr_3, axis = 1)
    ecmc_iact_mean_arr_3 = ecmc_iact_mean_arr_3[ecmc_sorted_timestep_3 >= 0.01]
    ecmc_err_3 = np.std(ecmc_storage_arr_3, axis=1)
    ecmc_err_3 = ecmc_err_3[ecmc_sorted_timestep_3 >= 0.01]
    ecmc_sorted_timestep_3 = ecmc_sorted_timestep_3[ecmc_sorted_timestep_3 >= 0.01]
    ecmc_sorted_N_3 = propertime / ecmc_sorted_timestep_3




    ecmc_timestep_data_4 = np.load(os.path.join(ecmc_iact_data_path_4, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_4 = np.zeros((len(ecmc_timestep_data_4), N))
    ecmc_sorted_timestep_4 = ecmc_timestep_data_4[np.argsort(ecmc_timestep_data_4)]

    for index_4 in range(N):
        ecmc_iact_timestep_4 =  np.load(os.path.join(ecmc_iact_data_path_4, f"iact_ecmc_{index_4}.npy"))
        ecmc_iact_data_4 = ecmc_iact_timestep_4[:, 0] 
        ecmc_timestep_data_4 = ecmc_iact_timestep_4[:, 1] 
        argsorted_data_4 = np.argsort(ecmc_timestep_data_4)
        ecmc_timestep_argsorted_4 = ecmc_timestep_data_4[argsorted_data_4]
        ecmc_iact_data_4 = ecmc_iact_data_4[argsorted_data_4] 
        ecmc_storage_arr_4[:, index_4] = ecmc_iact_data_4

    ecmc_iact_mean_arr_4 = np.mean(ecmc_storage_arr_4, axis = 1)
    ecmc_iact_mean_arr_4 = ecmc_iact_mean_arr_4[ecmc_sorted_timestep_4 >= 0.01]
    ecmc_err_4 = np.std(ecmc_storage_arr_4, axis=1)
    ecmc_err_4 = ecmc_err_4[ecmc_sorted_timestep_4 >= 0.01]
    ecmc_sorted_timestep_4 = ecmc_sorted_timestep_4[ecmc_sorted_timestep_4 >= 0.01]
    ecmc_sorted_N_4 = propertime / ecmc_sorted_timestep_4




    ecmc_timestep_data_5 = np.load(os.path.join(ecmc_iact_data_path_5, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_5 = np.zeros((len(ecmc_timestep_data_5), N))
    ecmc_sorted_timestep_5 = ecmc_timestep_data_5[np.argsort(ecmc_timestep_data_5)]

    for index_5 in range(N):
        ecmc_iact_timestep_5 =  np.load(os.path.join(ecmc_iact_data_path_5, f"iact_ecmc_{index_5}.npy"))
        ecmc_iact_data_5 = ecmc_iact_timestep_5[:, 0] 
        ecmc_timestep_data_5 = ecmc_iact_timestep_5[:, 1] 
        argsorted_data_5 = np.argsort(ecmc_timestep_data_5)
        ecmc_timestep_argsorted_5 = ecmc_timestep_data_5[argsorted_data_5]
        ecmc_iact_data_5 = ecmc_iact_data_5[argsorted_data_5] 
        ecmc_storage_arr_5[:, index_5] = ecmc_iact_data_5

    ecmc_iact_mean_arr_5 = np.mean(ecmc_storage_arr_5, axis = 1)
    ecmc_iact_mean_arr_5 = ecmc_iact_mean_arr_5[ecmc_sorted_timestep_5 >= 0.01]
    ecmc_err_5 = np.std(ecmc_storage_arr_5, axis=1)
    ecmc_err_5 = ecmc_err_5[ecmc_sorted_timestep_5 >= 0.01]
    ecmc_sorted_timestep_5 = ecmc_sorted_timestep_5[ecmc_sorted_timestep_5 >= 0.01]
    ecmc_sorted_N_5 = propertime / ecmc_sorted_timestep_5



    ecmc_timestep_data_6 = np.load(os.path.join(ecmc_iact_data_path_6, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_6 = np.zeros((len(ecmc_timestep_data_6), N))
    ecmc_sorted_timestep_6 = ecmc_timestep_data_6[np.argsort(ecmc_timestep_data_6)]

    for index_6 in range(N):
        ecmc_iact_timestep_6 =  np.load(os.path.join(ecmc_iact_data_path_6, f"iact_ecmc_{index_6}.npy"))
        ecmc_iact_data_6 = ecmc_iact_timestep_6[:, 0] 
        ecmc_timestep_data_6 = ecmc_iact_timestep_6[:, 1] 
        argsorted_data_6 = np.argsort(ecmc_timestep_data_6)
        ecmc_timestep_argsorted_6 = ecmc_timestep_data_6[argsorted_data_6]
        ecmc_iact_data_6 = ecmc_iact_data_6[argsorted_data_6] 
        ecmc_storage_arr_6[:, index_6] = ecmc_iact_data_6

    ecmc_iact_mean_arr_6 = np.mean(ecmc_storage_arr_6, axis = 1)
    ecmc_iact_mean_arr_6 = ecmc_iact_mean_arr_6[ecmc_sorted_timestep_6 >= 0.01]
    ecmc_err_6 = np.std(ecmc_storage_arr_6, axis=1)
    ecmc_err_6 = ecmc_err_6[ecmc_sorted_timestep_6 >= 0.01]
    ecmc_sorted_timestep_6 = ecmc_sorted_timestep_6[ecmc_sorted_timestep_6 >= 0.01]
    ecmc_sorted_N_6 = propertime / ecmc_sorted_timestep_6


    ecmc_timestep_data_7 = np.load(os.path.join(ecmc_iact_data_path_7, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_7 = np.zeros((len(ecmc_timestep_data_7), N))
    ecmc_sorted_timestep_7 = ecmc_timestep_data_7[np.argsort(ecmc_timestep_data_7)]

    for index_7 in range(N):
        ecmc_iact_timestep_7 =  np.load(os.path.join(ecmc_iact_data_path_7, f"iact_ecmc_{index_7}.npy"))
        ecmc_iact_data_7 = ecmc_iact_timestep_7[:, 0] 
        ecmc_timestep_data_7 = ecmc_iact_timestep_7[:, 1] 
        argsorted_data_7 = np.argsort(ecmc_timestep_data_7)
        ecmc_timestep_argsorted_7 = ecmc_timestep_data_7[argsorted_data_7]
        ecmc_iact_data_7 = ecmc_iact_data_7[argsorted_data_7] 
        ecmc_storage_arr_7[:, index_7] = ecmc_iact_data_7

    ecmc_iact_mean_arr_7 = np.mean(ecmc_storage_arr_7, axis = 1)
    ecmc_iact_mean_arr_7 = ecmc_iact_mean_arr_7[ecmc_sorted_timestep_7 >= 0.01]
    ecmc_err_7 = np.std(ecmc_storage_arr_7, axis=1)
    ecmc_err_7 = ecmc_err_7[ecmc_sorted_timestep_7 >= 0.01]
    ecmc_sorted_timestep_7 = ecmc_sorted_timestep_7[ecmc_sorted_timestep_7 >= 0.01]
    ecmc_sorted_N_7 = propertime / ecmc_sorted_timestep_7


    ecmc_timestep_data_8 = np.load(os.path.join(ecmc_iact_data_path_8, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_8 = np.zeros((len(ecmc_timestep_data_8), N))
    ecmc_sorted_timestep_8 = ecmc_timestep_data_8[np.argsort(ecmc_timestep_data_8)]

    for index_8 in range(N):
        ecmc_iact_timestep_8 =  np.load(os.path.join(ecmc_iact_data_path_8, f"iact_ecmc_{index_8}.npy"))
        ecmc_iact_data_8 = ecmc_iact_timestep_8[:, 0] 
        ecmc_timestep_data_8 = ecmc_iact_timestep_8[:, 1] 
        argsorted_data_8 = np.argsort(ecmc_timestep_data_8)
        ecmc_timestep_argsorted_8 = ecmc_timestep_data_8[argsorted_data_8]
        ecmc_iact_data_8 = ecmc_iact_data_8[argsorted_data_8] 
        ecmc_storage_arr_8[:, index_8] = ecmc_iact_data_8

    ecmc_iact_mean_arr_8 = np.mean(ecmc_storage_arr_8, axis = 1)
    ecmc_iact_mean_arr_8 = ecmc_iact_mean_arr_8[ecmc_sorted_timestep_8 >= 0.01]
    ecmc_err_8 = np.std(ecmc_storage_arr_8, axis=1)
    ecmc_err_8 = ecmc_err_8[ecmc_sorted_timestep_8 >= 0.01]
    ecmc_sorted_timestep_8 = ecmc_sorted_timestep_8[ecmc_sorted_timestep_8 >= 0.01]
    ecmc_sorted_N_8 = propertime / ecmc_sorted_timestep_8


    ecmc_timestep_data_9 = np.load(os.path.join(ecmc_iact_data_path_9, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_9 = np.zeros((len(ecmc_timestep_data_9), N))
    ecmc_sorted_timestep_9 = ecmc_timestep_data_9[np.argsort(ecmc_timestep_data_9)]

    for index_9 in range(N):
        ecmc_iact_timestep_9 =  np.load(os.path.join(ecmc_iact_data_path_9, f"iact_ecmc_{index_9}.npy"))
        ecmc_iact_data_9 = ecmc_iact_timestep_9[:, 0] 
        ecmc_timestep_data_9 = ecmc_iact_timestep_9[:, 1] 
        argsorted_data_9 = np.argsort(ecmc_timestep_data_9)
        ecmc_timestep_argsorted_9 = ecmc_timestep_data_9[argsorted_data_9]
        ecmc_iact_data_9 = ecmc_iact_data_9[argsorted_data_9] 
        ecmc_storage_arr_9[:, index_9] = ecmc_iact_data_9

    ecmc_iact_mean_arr_9 = np.mean(ecmc_storage_arr_9, axis = 1)
    ecmc_iact_mean_arr_9 = ecmc_iact_mean_arr_9[ecmc_sorted_timestep_9 >= 0.01]
    ecmc_err_9 = np.std(ecmc_storage_arr_9, axis=1)
    ecmc_err_9 = ecmc_err_9[ecmc_sorted_timestep_9 >= 0.01]
    ecmc_sorted_timestep_9 = ecmc_sorted_timestep_9[ecmc_sorted_timestep_9 >= 0.01]
    ecmc_sorted_N_9 = propertime / ecmc_sorted_timestep_9


    ecmc_timestep_data_10 = np.load(os.path.join(ecmc_iact_data_path_10, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_10 = np.zeros((len(ecmc_timestep_data_10), N))
    ecmc_sorted_timestep_10 = ecmc_timestep_data_10[np.argsort(ecmc_timestep_data_10)]

    for index_10 in range(N):
        ecmc_iact_timestep_10 =  np.load(os.path.join(ecmc_iact_data_path_10, f"iact_ecmc_{index_10}.npy"))
        ecmc_iact_data_10 = ecmc_iact_timestep_10[:, 0] 
        ecmc_timestep_data_10 = ecmc_iact_timestep_10[:, 1] 
        argsorted_data_10 = np.argsort(ecmc_timestep_data_10)
        ecmc_timestep_argsorted_10 = ecmc_timestep_data_10[argsorted_data_10]
        ecmc_iact_data_10 = ecmc_iact_data_10[argsorted_data_10] 
        ecmc_storage_arr_10[:, index_10] = ecmc_iact_data_10

    ecmc_iact_mean_arr_10 = np.mean(ecmc_storage_arr_10, axis = 1)
    ecmc_iact_mean_arr_10 = ecmc_iact_mean_arr_10[ecmc_sorted_timestep_10 >= 0.01]
    ecmc_err_10 = np.std(ecmc_storage_arr_10, axis=1)
    ecmc_err_10 = ecmc_err_10[ecmc_sorted_timestep_10 >= 0.01]
    ecmc_sorted_timestep_10 = ecmc_sorted_timestep_10[ecmc_sorted_timestep_10 >= 0.01]
    ecmc_sorted_N_10 = propertime / ecmc_sorted_timestep_10

    ecmc_timestep_data_11 = np.load(os.path.join(ecmc_iact_data_path_11, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_11 = np.zeros((len(ecmc_timestep_data_11), N))
    ecmc_sorted_timestep_11 = ecmc_timestep_data_11[np.argsort(ecmc_timestep_data_11)]

    for index_11 in range(N):
        ecmc_iact_timestep_11 =  np.load(os.path.join(ecmc_iact_data_path_11, f"iact_ecmc_{index_11}.npy"))
        ecmc_iact_data_11 = ecmc_iact_timestep_11[:, 0] 
        ecmc_timestep_data_11 = ecmc_iact_timestep_11[:, 1] 
        argsorted_data_11 = np.argsort(ecmc_timestep_data_11)
        ecmc_timestep_argsorted_11 = ecmc_timestep_data_11[argsorted_data_11]
        ecmc_iact_data_11 = ecmc_iact_data_11[argsorted_data_11] 
        ecmc_storage_arr_11[:, index_11] = ecmc_iact_data_11

    ecmc_iact_mean_arr_11 = np.mean(ecmc_storage_arr_11, axis = 1)
    ecmc_iact_mean_arr_11 = ecmc_iact_mean_arr_11[ecmc_sorted_timestep_11 >= 0.01]
    ecmc_err_11 = np.std(ecmc_storage_arr_11, axis=1)
    ecmc_err_11 = ecmc_err_11[ecmc_sorted_timestep_11 >= 0.01]
    ecmc_sorted_timestep_11 = ecmc_sorted_timestep_11[ecmc_sorted_timestep_11 >= 0.01]
    ecmc_sorted_N_11 = propertime / ecmc_sorted_timestep_11

    ecmc_timestep_data_12 = np.load(os.path.join(ecmc_iact_data_path_12, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_12 = np.zeros((len(ecmc_timestep_data_12), N))
    ecmc_sorted_timestep_12 = ecmc_timestep_data_12[np.argsort(ecmc_timestep_data_12)]

    for index_12 in range(N):
        ecmc_iact_timestep_12 =  np.load(os.path.join(ecmc_iact_data_path_12, f"iact_ecmc_{index_12}.npy"))
        ecmc_iact_data_12 = ecmc_iact_timestep_12[:, 0] 
        ecmc_timestep_data_12 = ecmc_iact_timestep_12[:, 1] 
        argsorted_data_12 = np.argsort(ecmc_timestep_data_12)
        ecmc_timestep_argsorted_12 = ecmc_timestep_data_12[argsorted_data_12]
        ecmc_iact_data_12 = ecmc_iact_data_12[argsorted_data_12] 
        ecmc_storage_arr_12[:, index_12] = ecmc_iact_data_12

    ecmc_iact_mean_arr_12 = np.mean(ecmc_storage_arr_12, axis = 1)
    ecmc_iact_mean_arr_12 = ecmc_iact_mean_arr_12[ecmc_sorted_timestep_12 >= 0.01]
    ecmc_err_12 = np.std(ecmc_storage_arr_12, axis=1)
    ecmc_err_12 = ecmc_err_12[ecmc_sorted_timestep_12 >= 0.01]
    ecmc_sorted_timestep_12 = ecmc_sorted_timestep_12[ecmc_sorted_timestep_12 >= 0.01]
    ecmc_sorted_N_12 = propertime / ecmc_sorted_timestep_12

    
    ecmc_timestep_data_13 = np.load(os.path.join(ecmc_iact_data_path_13, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_13 = np.zeros((len(ecmc_timestep_data_13), N))
    ecmc_sorted_timestep_13 = ecmc_timestep_data_13[np.argsort(ecmc_timestep_data_13)]

    for index_13 in range(N):
        ecmc_iact_timestep_13 =  np.load(os.path.join(ecmc_iact_data_path_13, f"iact_ecmc_{index_13}.npy"))
        ecmc_iact_data_13 = ecmc_iact_timestep_13[:, 0] 
        ecmc_timestep_data_13 = ecmc_iact_timestep_13[:, 1] 
        argsorted_data_13 = np.argsort(ecmc_timestep_data_13)
        ecmc_timestep_argsorted_13 = ecmc_timestep_data_13[argsorted_data_13]
        ecmc_iact_data_13 = ecmc_iact_data_13[argsorted_data_13] 
        ecmc_storage_arr_13[:, index_13] = ecmc_iact_data_13

    ecmc_iact_mean_arr_13 = np.mean(ecmc_storage_arr_13, axis = 1)
    ecmc_iact_mean_arr_13 = ecmc_iact_mean_arr_13[ecmc_sorted_timestep_13 >= 0.01]
    ecmc_err_13 = np.std(ecmc_storage_arr_13, axis=1)
    ecmc_err_13 = ecmc_err_13[ecmc_sorted_timestep_13 >= 0.01]
    ecmc_sorted_timestep_13 = ecmc_sorted_timestep_13[ecmc_sorted_timestep_13 >= 0.01]
    ecmc_sorted_N_13 = propertime / ecmc_sorted_timestep_13

    ecmc_timestep_data_14 = np.load(os.path.join(ecmc_iact_data_path_14, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_14 = np.zeros((len(ecmc_timestep_data_14), N))
    ecmc_sorted_timestep_14 = ecmc_timestep_data_14[np.argsort(ecmc_timestep_data_14)]

    for index_14 in range(N):
        ecmc_iact_timestep_14 =  np.load(os.path.join(ecmc_iact_data_path_14, f"iact_ecmc_{index_14}.npy"))
        ecmc_iact_data_14 = ecmc_iact_timestep_14[:, 0] 
        ecmc_timestep_data_14 = ecmc_iact_timestep_14[:, 1] 
        argsorted_data_14 = np.argsort(ecmc_timestep_data_14)
        ecmc_timestep_argsorted_14 = ecmc_timestep_data_14[argsorted_data_14]
        ecmc_iact_data_14 = ecmc_iact_data_14[argsorted_data_14] 
        ecmc_storage_arr_14[:, index_14] = ecmc_iact_data_14

    ecmc_iact_mean_arr_14 = np.mean(ecmc_storage_arr_14, axis = 1)
    ecmc_iact_mean_arr_14 = ecmc_iact_mean_arr_14[ecmc_sorted_timestep_14 >= 0.01]
    ecmc_err_14 = np.std(ecmc_storage_arr_14, axis=1)
    ecmc_err_14 = ecmc_err_14[ecmc_sorted_timestep_14 >= 0.01]
    ecmc_sorted_timestep_14 = ecmc_sorted_timestep_14[ecmc_sorted_timestep_14 >= 0.01]
    ecmc_sorted_N_14 = propertime / ecmc_sorted_timestep_14

    ecmc_timestep_data_15 = np.load(os.path.join(ecmc_iact_data_path_15, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_15 = np.zeros((len(ecmc_timestep_data_15), N))
    ecmc_sorted_timestep_15 = ecmc_timestep_data_15[np.argsort(ecmc_timestep_data_15)]

    for index_15 in range(N):
        if index_15 != 4:
            ecmc_iact_timestep_15 =  np.load(os.path.join(ecmc_iact_data_path_15, f"iact_ecmc_{index_15}.npy"))
            ecmc_iact_data_15 = ecmc_iact_timestep_15[:, 0] 
            ecmc_timestep_data_15 = ecmc_iact_timestep_15[:, 1] 
            argsorted_data_15 = np.argsort(ecmc_timestep_data_15)
            ecmc_timestep_argsorted_15 = ecmc_timestep_data_15[argsorted_data_15]
            ecmc_iact_data_15 = ecmc_iact_data_15[argsorted_data_15] 
            ecmc_storage_arr_15[:, index_15] = ecmc_iact_data_15

    ecmc_iact_mean_arr_15 = np.mean(ecmc_storage_arr_15, axis = 1)
    ecmc_iact_mean_arr_15 = ecmc_iact_mean_arr_15[ecmc_sorted_timestep_15 >= 0.01]
    ecmc_err_15 = np.std(ecmc_storage_arr_15, axis=1)
    ecmc_err_15 = ecmc_err_15[ecmc_sorted_timestep_15 >= 0.01]
    ecmc_sorted_timestep_15 = ecmc_sorted_timestep_15[ecmc_sorted_timestep_15 >= 0.01]
    ecmc_sorted_N_15 = propertime / ecmc_sorted_timestep_15

    ecmc_timestep_data_16 = np.load(os.path.join(ecmc_iact_data_path_16, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_16 = np.zeros((len(ecmc_timestep_data_16), N))
    ecmc_sorted_timestep_16 = ecmc_timestep_data_16[np.argsort(ecmc_timestep_data_16)]

    for index_16 in range(N):
        if index_16 != 4:
            ecmc_iact_timestep_16 =  np.load(os.path.join(ecmc_iact_data_path_16, f"iact_ecmc_{index_16}.npy"))
            ecmc_iact_data_16 = ecmc_iact_timestep_16[:, 0] 
            ecmc_timestep_data_16 = ecmc_iact_timestep_16[:, 1] 
            argsorted_data_16 = np.argsort(ecmc_timestep_data_16)
            ecmc_timestep_argsorted_16 = ecmc_timestep_data_16[argsorted_data_16]
            ecmc_iact_data_16 = ecmc_iact_data_16[argsorted_data_16] 
            ecmc_storage_arr_16[:, index_16] = ecmc_iact_data_16

    ecmc_iact_mean_arr_16 = np.mean(ecmc_storage_arr_16, axis = 1)
    ecmc_iact_mean_arr_16 = ecmc_iact_mean_arr_16[ecmc_sorted_timestep_16 >= 0.01]
    ecmc_err_16 = np.std(ecmc_storage_arr_16, axis=1)
    ecmc_err_16 = ecmc_err_16[ecmc_sorted_timestep_16 >= 0.01]
    ecmc_sorted_timestep_16 = ecmc_sorted_timestep_16[ecmc_sorted_timestep_16 >= 0.01]
    ecmc_sorted_N_16 = propertime / ecmc_sorted_timestep_16

    ecmc_timestep_data_17 = np.load(os.path.join(ecmc_iact_data_path_17, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_17 = np.zeros((len(ecmc_timestep_data_17), N))
    ecmc_sorted_timestep_17 = ecmc_timestep_data_17[np.argsort(ecmc_timestep_data_17)]

    for index_17 in range(N):
        ecmc_iact_timestep_17 =  np.load(os.path.join(ecmc_iact_data_path_17, f"iact_ecmc_{index_17}.npy"))
        ecmc_iact_data_17 = ecmc_iact_timestep_17[:, 0] 
        ecmc_timestep_data_17 = ecmc_iact_timestep_17[:, 1] 
        argsorted_data_17 = np.argsort(ecmc_timestep_data_17)
        ecmc_timestep_argsorted_17 = ecmc_timestep_data_17[argsorted_data_17]
        ecmc_iact_data_17 = ecmc_iact_data_17[argsorted_data_17] 
        ecmc_storage_arr_17[:, index_17] = ecmc_iact_data_17

    ecmc_iact_mean_arr_17 = np.mean(ecmc_storage_arr_17, axis = 1)
    ecmc_iact_mean_arr_17 = ecmc_iact_mean_arr_17[ecmc_sorted_timestep_17 >= 0.01]
    ecmc_err_17 = np.std(ecmc_storage_arr_17, axis=1)
    ecmc_err_17 = ecmc_err_17[ecmc_sorted_timestep_17 >= 0.01]
    ecmc_sorted_timestep_17 = ecmc_sorted_timestep_17[ecmc_sorted_timestep_17 >= 0.01]
    ecmc_sorted_N_17 = propertime / ecmc_sorted_timestep_17


    ecmc_timestep_data_18 = np.load(os.path.join(ecmc_iact_data_path_18, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_18 = np.zeros((len(ecmc_timestep_data_18), N))
    ecmc_sorted_timestep_18 = ecmc_timestep_data_18[np.argsort(ecmc_timestep_data_18)]

    for index_18 in range(N):
        ecmc_iact_timestep_18 =  np.load(os.path.join(ecmc_iact_data_path_18, f"iact_ecmc_{index_18}.npy"))
        ecmc_iact_data_18 = ecmc_iact_timestep_18[:, 0] 
        ecmc_timestep_data_18 = ecmc_iact_timestep_18[:, 1] 
        argsorted_data_18 = np.argsort(ecmc_timestep_data_18)
        ecmc_timestep_argsorted_18 = ecmc_timestep_data_18[argsorted_data_18]
        ecmc_iact_data_18 = ecmc_iact_data_18[argsorted_data_18] 
        ecmc_storage_arr_18[:, index_18] = ecmc_iact_data_18

    ecmc_iact_mean_arr_18 = np.mean(ecmc_storage_arr_18, axis = 1)
    ecmc_iact_mean_arr_18 = ecmc_iact_mean_arr_18[ecmc_sorted_timestep_18 >= 0.01]
    ecmc_err_18 = np.std(ecmc_storage_arr_18, axis=1)
    ecmc_err_18 = ecmc_err_18[ecmc_sorted_timestep_18 >= 0.01]
    ecmc_sorted_timestep_18 = ecmc_sorted_timestep_18[ecmc_sorted_timestep_18 >= 0.01]
    ecmc_sorted_N_18 = propertime / ecmc_sorted_timestep_18

    ecmc_timestep_data_19 = np.load(os.path.join(ecmc_iact_data_path_19, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_19 = np.zeros((len(ecmc_timestep_data_19), N))
    ecmc_sorted_timestep_19 = ecmc_timestep_data_19[np.argsort(ecmc_timestep_data_19)]

    for index_19 in range(N):
        ecmc_iact_timestep_19 =  np.load(os.path.join(ecmc_iact_data_path_19, f"iact_ecmc_{index_19}.npy"))
        ecmc_iact_data_19 = ecmc_iact_timestep_19[:, 0] 
        ecmc_timestep_data_19 = ecmc_iact_timestep_19[:, 1] 
        argsorted_data_19 = np.argsort(ecmc_timestep_data_19)
        ecmc_timestep_argsorted_19 = ecmc_timestep_data_19[argsorted_data_19]
        ecmc_iact_data_19 = ecmc_iact_data_19[argsorted_data_19] 
        ecmc_storage_arr_19[:, index_19] = ecmc_iact_data_19

    ecmc_iact_mean_arr_19 = np.mean(ecmc_storage_arr_19, axis = 1)
    ecmc_iact_mean_arr_19 = ecmc_iact_mean_arr_19[ecmc_sorted_timestep_19 >= 0.01]
    ecmc_err_19 = np.std(ecmc_storage_arr_19, axis=1)
    ecmc_err_19 = ecmc_err_19[ecmc_sorted_timestep_19 >= 0.01]
    ecmc_sorted_timestep_19 = ecmc_sorted_timestep_19[ecmc_sorted_timestep_19 >= 0.01]
    ecmc_sorted_N_19 = propertime / ecmc_sorted_timestep_19

    ecmc_timestep_data_20 = np.load(os.path.join(ecmc_iact_data_path_20, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_20 = np.zeros((len(ecmc_timestep_data_20), N))
    ecmc_sorted_timestep_20 = ecmc_timestep_data_20[np.argsort(ecmc_timestep_data_20)]

    for index_20 in range(N):
        ecmc_iact_timestep_20 =  np.load(os.path.join(ecmc_iact_data_path_20, f"iact_ecmc_{index_20}.npy"))
        ecmc_iact_data_20 = ecmc_iact_timestep_20[:, 0] 
        ecmc_timestep_data_20 = ecmc_iact_timestep_20[:, 1] 
        argsorted_data_20 = np.argsort(ecmc_timestep_data_20)
        ecmc_timestep_argsorted_20 = ecmc_timestep_data_20[argsorted_data_20]
        ecmc_iact_data_20 = ecmc_iact_data_20[argsorted_data_20] 
        ecmc_storage_arr_20[:, index_20] = ecmc_iact_data_20

    ecmc_iact_mean_arr_20 = np.mean(ecmc_storage_arr_20, axis = 1)
    ecmc_iact_mean_arr_20 = ecmc_iact_mean_arr_20[ecmc_sorted_timestep_20 >= 0.01]
    ecmc_err_20 = np.std(ecmc_storage_arr_20, axis=1)
    ecmc_err_20 = ecmc_err_20[ecmc_sorted_timestep_20 >= 0.01]
    ecmc_sorted_timestep_20 = ecmc_sorted_timestep_20[ecmc_sorted_timestep_20 >= 0.01]
    ecmc_sorted_N_20 = propertime / ecmc_sorted_timestep_20

    ecmc_timestep_data_21 = np.load(os.path.join(ecmc_iact_data_path_21, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_21 = np.zeros((len(ecmc_timestep_data_21), N))
    ecmc_sorted_timestep_21 = ecmc_timestep_data_21[np.argsort(ecmc_timestep_data_21)]

    for index_21 in range(N):
        ecmc_iact_timestep_21 =  np.load(os.path.join(ecmc_iact_data_path_20, f"iact_ecmc_{index_20}.npy"))
        ecmc_iact_data_21 = ecmc_iact_timestep_21[:, 0] 
        ecmc_timestep_data_21 = ecmc_iact_timestep_21[:, 1] 
        argsorted_data_21 = np.argsort(ecmc_timestep_data_21)
        ecmc_timestep_argsorted_21 = ecmc_timestep_data_21[argsorted_data_21]
        ecmc_iact_data_21 = ecmc_iact_data_21[argsorted_data_21] 
        ecmc_storage_arr_21[:, index_21] = ecmc_iact_data_21

    ecmc_iact_mean_arr_21 = np.mean(ecmc_storage_arr_21, axis = 1)
    ecmc_iact_mean_arr_21 = ecmc_iact_mean_arr_21[ecmc_sorted_timestep_21 >= 0.01]
    ecmc_err_21 = np.std(ecmc_storage_arr_21, axis=1)
    ecmc_err_21 = ecmc_err_21[ecmc_sorted_timestep_21 >= 0.01]
    ecmc_sorted_timestep_21 = ecmc_sorted_timestep_21[ecmc_sorted_timestep_21 >= 0.01]
    ecmc_sorted_N_21 = propertime / ecmc_sorted_timestep_21

    ecmc_timestep_data_22 = np.load(os.path.join(ecmc_iact_data_path_22, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_22 = np.zeros((len(ecmc_timestep_data_22), N))
    ecmc_sorted_timestep_22 = ecmc_timestep_data_22[np.argsort(ecmc_timestep_data_22)]

    for index_22 in range(N):
        ecmc_iact_timestep_22 =  np.load(os.path.join(ecmc_iact_data_path_22, f"iact_ecmc_{index_22}.npy"))
        ecmc_iact_data_22 = ecmc_iact_timestep_22[:, 0] 
        ecmc_timestep_data_22 = ecmc_iact_timestep_22[:, 1] 
        argsorted_data_22 = np.argsort(ecmc_timestep_data_22)
        ecmc_timestep_argsorted_22 = ecmc_timestep_data_22[argsorted_data_22]
        ecmc_iact_data_22 = ecmc_iact_data_22[argsorted_data_22] 
        ecmc_storage_arr_22[:, index_22] = ecmc_iact_data_22

    ecmc_iact_mean_arr_22 = np.mean(ecmc_storage_arr_22, axis = 1)
    ecmc_iact_mean_arr_22 = ecmc_iact_mean_arr_22[ecmc_sorted_timestep_22 >= 0.01]
    ecmc_err_22 = np.std(ecmc_storage_arr_22, axis=1)
    ecmc_err_22 = ecmc_err_22[ecmc_sorted_timestep_22 >= 0.01]
    ecmc_sorted_timestep_22 = ecmc_sorted_timestep_22[ecmc_sorted_timestep_22 >= 0.01]
    ecmc_sorted_N_22 = propertime / ecmc_sorted_timestep_22

    ecmc_timestep_data_23 = np.load(os.path.join(ecmc_iact_data_path_23, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_23 = np.zeros((len(ecmc_timestep_data_23), N))
    ecmc_sorted_timestep_23 = ecmc_timestep_data_23[np.argsort(ecmc_timestep_data_23)]

    for index_23 in range(N):
        ecmc_iact_timestep_23 =  np.load(os.path.join(ecmc_iact_data_path_23, f"iact_ecmc_{index_23}.npy"))
        ecmc_iact_data_23 = ecmc_iact_timestep_23[:, 0] 
        ecmc_timestep_data_23 = ecmc_iact_timestep_23[:, 1] 
        argsorted_data_23 = np.argsort(ecmc_timestep_data_23)
        ecmc_timestep_argsorted_23 = ecmc_timestep_data_23[argsorted_data_23]
        ecmc_iact_data_23 = ecmc_iact_data_23[argsorted_data_23] 
        ecmc_storage_arr_23[:, index_23] = ecmc_iact_data_23

    ecmc_iact_mean_arr_23 = np.mean(ecmc_storage_arr_23, axis = 1)
    ecmc_iact_mean_arr_23 = ecmc_iact_mean_arr_23[ecmc_sorted_timestep_23 >= 0.01]
    ecmc_err_23 = np.std(ecmc_storage_arr_23, axis=1)
    ecmc_err_23 = ecmc_err_23[ecmc_sorted_timestep_23 >= 0.01]
    ecmc_sorted_timestep_23 = ecmc_sorted_timestep_23[ecmc_sorted_timestep_23 >= 0.01]
    ecmc_sorted_N_23 = propertime / ecmc_sorted_timestep_23

    ecmc_timestep_data_24 = np.load(os.path.join(ecmc_iact_data_path_24, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_24 = np.zeros((len(ecmc_timestep_data_24), N))
    ecmc_sorted_timestep_24 = ecmc_timestep_data_24[np.argsort(ecmc_timestep_data_24)]

    for index_24 in range(N):
        ecmc_iact_timestep_24 =  np.load(os.path.join(ecmc_iact_data_path_24, f"iact_ecmc_{index_24}.npy"))
        ecmc_iact_data_24 = ecmc_iact_timestep_24[:, 0] 
        ecmc_timestep_data_24 = ecmc_iact_timestep_24[:, 1] 
        argsorted_data_24 = np.argsort(ecmc_timestep_data_24)
        ecmc_timestep_argsorted_24 = ecmc_timestep_data_24[argsorted_data_24]
        ecmc_iact_data_24 = ecmc_iact_data_24[argsorted_data_24] 
        ecmc_storage_arr_24[:, index_24] = ecmc_iact_data_24

    ecmc_iact_mean_arr_24 = np.mean(ecmc_storage_arr_24, axis = 1)
    ecmc_iact_mean_arr_24 = ecmc_iact_mean_arr_24[ecmc_sorted_timestep_24 >= 0.01]
    ecmc_err_24 = np.std(ecmc_storage_arr_24, axis=1)
    ecmc_err_24 = ecmc_err_24[ecmc_sorted_timestep_24 >= 0.01]
    ecmc_sorted_timestep_24 = ecmc_sorted_timestep_24[ecmc_sorted_timestep_24 >= 0.01]
    ecmc_sorted_N_24 = propertime / ecmc_sorted_timestep_24

    ecmc_timestep_data_25 = np.load(os.path.join(ecmc_iact_data_path_25, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_25 = np.zeros((len(ecmc_timestep_data_25), N))
    ecmc_sorted_timestep_25 = ecmc_timestep_data_25[np.argsort(ecmc_timestep_data_25)]

    for index_25 in range(N):
        ecmc_iact_timestep_25 =  np.load(os.path.join(ecmc_iact_data_path_25, f"iact_ecmc_{index_25}.npy"))
        ecmc_iact_data_25 = ecmc_iact_timestep_25[:, 0] 
        ecmc_timestep_data_25 = ecmc_iact_timestep_25[:, 1] 
        argsorted_data_25 = np.argsort(ecmc_timestep_data_25)
        ecmc_timestep_argsorted_25 = ecmc_timestep_data_25[argsorted_data_25]
        ecmc_iact_data_25 = ecmc_iact_data_25[argsorted_data_25] 
        ecmc_storage_arr_25[:, index_25] = ecmc_iact_data_25

    ecmc_iact_mean_arr_25 = np.mean(ecmc_storage_arr_25, axis = 1)
    ecmc_iact_mean_arr_25 = ecmc_iact_mean_arr_25[ecmc_sorted_timestep_25 >= 0.01]
    ecmc_err_25 = np.std(ecmc_storage_arr_25, axis=1)
    ecmc_err_25 = ecmc_err_25[ecmc_sorted_timestep_25 >= 0.01]
    ecmc_sorted_timestep_25 = ecmc_sorted_timestep_25[ecmc_sorted_timestep_25 >= 0.01]
    ecmc_sorted_N_25 = propertime / ecmc_sorted_timestep_25

    ecmc_timestep_data_26 = np.load(os.path.join(ecmc_iact_data_path_26, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_26 = np.zeros((len(ecmc_timestep_data_26), N))
    ecmc_sorted_timestep_26 = ecmc_timestep_data_26[np.argsort(ecmc_timestep_data_26)]

    for index_26 in range(N):
        ecmc_iact_timestep_26 =  np.load(os.path.join(ecmc_iact_data_path_26, f"iact_ecmc_{index_26}.npy"))
        ecmc_iact_data_26 = ecmc_iact_timestep_26[:, 0] 
        ecmc_timestep_data_26 = ecmc_iact_timestep_26[:, 1] 
        argsorted_data_26 = np.argsort(ecmc_timestep_data_26)
        ecmc_timestep_argsorted_26 = ecmc_timestep_data_26[argsorted_data_26]
        ecmc_iact_data_26 = ecmc_iact_data_26[argsorted_data_26] 
        ecmc_storage_arr_26[:, index_26] = ecmc_iact_data_26

    ecmc_iact_mean_arr_26 = np.mean(ecmc_storage_arr_26, axis = 1)
    ecmc_iact_mean_arr_26 = ecmc_iact_mean_arr_26[ecmc_sorted_timestep_26 >= 0.01]
    ecmc_err_26 = np.std(ecmc_storage_arr_26, axis=1)
    ecmc_err_26 = ecmc_err_26[ecmc_sorted_timestep_26 >= 0.01]
    ecmc_sorted_timestep_26 = ecmc_sorted_timestep_26[ecmc_sorted_timestep_26 >= 0.01]
    ecmc_sorted_N_26 = propertime / ecmc_sorted_timestep_26

    ecmc_timestep_data_27 = np.load(os.path.join(ecmc_iact_data_path_27, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_27 = np.zeros((len(ecmc_timestep_data_27), N))
    ecmc_sorted_timestep_27 = ecmc_timestep_data_27[np.argsort(ecmc_timestep_data_27)]

    for index_27 in range(N):
        ecmc_iact_timestep_27 =  np.load(os.path.join(ecmc_iact_data_path_27, f"iact_ecmc_{index_27}.npy"))
        ecmc_iact_data_27 = ecmc_iact_timestep_27[:, 0] 
        ecmc_timestep_data_27 = ecmc_iact_timestep_27[:, 1] 
        argsorted_data_27 = np.argsort(ecmc_timestep_data_27)
        ecmc_timestep_argsorted_27 = ecmc_timestep_data_27[argsorted_data_27]
        ecmc_iact_data_27 = ecmc_iact_data_27[argsorted_data_27] 
        ecmc_storage_arr_27[:, index_27] = ecmc_iact_data_27

    ecmc_iact_mean_arr_27 = np.mean(ecmc_storage_arr_27, axis = 1)
    ecmc_iact_mean_arr_27 = ecmc_iact_mean_arr_27[ecmc_sorted_timestep_27 >= 0.01]
    ecmc_err_27 = np.std(ecmc_storage_arr_27, axis=1)
    ecmc_err_27 = ecmc_err_27[ecmc_sorted_timestep_27 >= 0.01]
    ecmc_sorted_timestep_27 = ecmc_sorted_timestep_27[ecmc_sorted_timestep_27 >= 0.01]
    ecmc_sorted_N_27 = propertime / ecmc_sorted_timestep_27

    ecmc_timestep_data_28 = np.load(os.path.join(ecmc_iact_data_path_28, "iact_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr_28 = np.zeros((len(ecmc_timestep_data_28), N))
    ecmc_sorted_timestep_28 = ecmc_timestep_data_28[np.argsort(ecmc_timestep_data_28)]

    for index_28 in range(N):
        ecmc_iact_timestep_28 =  np.load(os.path.join(ecmc_iact_data_path_28, f"iact_ecmc_{index_28}.npy"))
        ecmc_iact_data_28 = ecmc_iact_timestep_28[:, 0] 
        ecmc_timestep_data_28 = ecmc_iact_timestep_28[:, 1] 
        argsorted_data_28 = np.argsort(ecmc_timestep_data_28)
        ecmc_timestep_argsorted_28 = ecmc_timestep_data_28[argsorted_data_28]
        ecmc_iact_data_28 = ecmc_iact_data_28[argsorted_data_28] 
        ecmc_storage_arr_28[:, index_28] = ecmc_iact_data_28

    ecmc_iact_mean_arr_28 = np.mean(ecmc_storage_arr_28, axis = 1)
    ecmc_iact_mean_arr_28 = ecmc_iact_mean_arr_28[ecmc_sorted_timestep_28 >= 0.01]
    ecmc_err_28 = np.std(ecmc_storage_arr_28, axis=1)
    ecmc_err_28 = ecmc_err_28[ecmc_sorted_timestep_28 >= 0.01]
    ecmc_sorted_timestep_28 = ecmc_sorted_timestep_28[ecmc_sorted_timestep_28 >= 0.01]
    ecmc_sorted_N_28 = propertime / ecmc_sorted_timestep_28

    if metrop_data:
        m_fit_trim = -7
        m_coeffs = np.polyfit(np.log(metropolis_sorted_N[:m_fit_trim]), np.log(metropolis_iact_mean_arr[:m_fit_trim]), deg=1)
        fitted_m = m_coeffs[1] + np.multiply(np.log(metropolis_sorted_N[:m_fit_trim]), m_coeffs[0])
        print(f"Metropolis: {m_coeffs[0]}")

    e_fit_trim = -13
    e_coeffs = np.polyfit(np.log(ecmc_sorted_N[:e_fit_trim]), np.log(ecmc_iact_mean_arr[:e_fit_trim]), deg=1)
    fitted_e = e_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:e_fit_trim]), e_coeffs[0])
    print(f"old ECMC: {e_coeffs[0]}")


    e_fit_trim_10 = -7
    e_coeffs_10 = np.polyfit(np.log(ecmc_sorted_N_10[:e_fit_trim_10]), np.log(ecmc_iact_mean_arr_10[:e_fit_trim_10]), deg=1)
    fitted_e_10 = e_coeffs_10[1] + np.multiply(np.log(ecmc_sorted_N_10[:e_fit_trim_10]), e_coeffs_10[0])
    print(f"ECMC sd=500: {e_coeffs_10[0]}")
    
    e_fit_trim_17 = -6
    e_coeffs_17 = np.polyfit(np.log(ecmc_sorted_N_17[:e_fit_trim_17]), np.log(ecmc_iact_mean_arr_17[:e_fit_trim_17]), deg=1)
    fitted_e_17 = e_coeffs_17[1] + np.multiply(np.log(ecmc_sorted_N_17[:e_fit_trim_17]), e_coeffs_17[0])
    print(f"ECMC ff=1.0, sd=500: {e_coeffs_17[0]}")

    e_fit_trim_22 = -9
    e_coeffs_22 = np.polyfit(np.log(ecmc_sorted_N_22[:e_fit_trim_22]), np.log(ecmc_iact_mean_arr_22[:e_fit_trim_22]), deg=1)
    fitted_e_22 = e_coeffs_22[1] + np.multiply(np.log(ecmc_sorted_N_22[:e_fit_trim_22]), e_coeffs_22[0])
    print(f"ECMC alt lifitng scheme, sd=500: {e_coeffs_22[0]}")

    e_fit_trim_26 = -9
    e_coeffs_26 = np.polyfit(np.log(ecmc_sorted_N_26[:e_fit_trim_26]), np.log(ecmc_iact_mean_arr_26[:e_fit_trim_26]), deg=1)
    fitted_e_26 = e_coeffs_26[1] + np.multiply(np.log(ecmc_sorted_N_26[:e_fit_trim_26]), e_coeffs_26[0])
    print(f"ECMC ff=1.0, sd=Nt: {e_coeffs_26[0]}")

    e_fit_trim_27 = -8
    e_coeffs_27 = np.polyfit(np.log(ecmc_sorted_N_27[:e_fit_trim_27]), np.log(ecmc_iact_mean_arr_27[:e_fit_trim_27]), deg=1)
    fitted_e_27 = e_coeffs_27[1] + np.multiply(np.log(ecmc_sorted_N_27[:e_fit_trim_27]), e_coeffs_27[0])
    print(f"ECMC sd=Nt: {e_coeffs_27[0]}")

    fig, ax = plt.subplots(1, 1, figsize = (9, 7))

    if metrop_data:
        ax.plot(metropolis_sorted_N[:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
        ax.errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis MC")
        #ax.annotate(f"M coeff: {m_coeffs[0]:.2f}", xy = (8*10e1, 3*10e1), weight = "bold")

        #ax.errorbar(metropolis_sorted_N_2, metropolis_iact_mean_arr_2, metropolis_err_2, fmt='^', capsize=3, markersize=4, color="#16e104ff", label="Metropolis MC 2 ")
        #ax.annotate(f"M coeff: {m_coeffs[0]:.2f}", xy = (8*10e1, 3*10e1), weight = "bold")
        pass


    ax.plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e), color="#d97dd9ff")
    ax.plot(ecmc_sorted_N_10[:e_fit_trim_10], np.exp(fitted_e_10), color="#8051a9ff")
    ax.plot(ecmc_sorted_N_17[:e_fit_trim_17], np.exp(fitted_e_17), color="#7c82edff")
    ax.plot(ecmc_sorted_N_22[:e_fit_trim_22], np.exp(fitted_e_22), color="#4f4f4fff")
    ax.plot(ecmc_sorted_N_26[:e_fit_trim_26], np.exp(fitted_e_26), color="#6b4670ff")
    ax.plot(ecmc_sorted_N_27[:e_fit_trim_27], np.exp(fitted_e_27), color="#a3bf32ff")

    

    
    ax.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    #ax.errorbar(ecmc_sorted_N_2, ecmc_iact_mean_arr_2, ecmc_err_2, fmt='o', capsize=3, markersize=4, color="#af82e5ff", label="ECMC sd=100, rf = 10e12")
    #ax.errorbar(ecmc_sorted_N_3, ecmc_iact_mean_arr_3, ecmc_err_3, fmt='o', capsize=3, markersize=4, color="#0d5b26ff", label="ECMC sd= 100, rf = 1.0")
    #ax.errorbar(ecmc_sorted_N_4, ecmc_iact_mean_arr_4, ecmc_err_4, fmt='o', capsize=3, markersize=4, color="#e281c8ff", label="ECMC sd=dt")
    #ax.errorbar(ecmc_sorted_N_5, ecmc_iact_mean_arr_5, ecmc_err_5, fmt='^', capsize=3, markersize=7, color="#ff001eff", label="ECMC sd=100, rf = Nt")
    #ax.errorbar(ecmc_sorted_N_6, ecmc_iact_mean_arr_6, ecmc_err_6, fmt='^', capsize=3, markersize=7, color="#2b0330ff", label="ECMC sd=120, rf = Nt")
    #ax.errorbar(ecmc_sorted_N_7, ecmc_iact_mean_arr_7, ecmc_err_7, fmt='o', capsize=3, markersize=4, color="#cdde37ff", label="ECMC sd = 1.0, rf = inf")
    #ax.errorbar(ecmc_sorted_N_8, ecmc_iact_mean_arr_8, ecmc_err_8, fmt='o', capsize=3, markersize=4, color="#3140aeff", label="ECMC sd = 120, rf = 10e12")
    #ax.errorbar(ecmc_sorted_N_9, ecmc_iact_mean_arr_9, ecmc_err_9, fmt='o', capsize=3, markersize=4, color="#64e15dff", label="ECMC sd = 500, rf = 10e12")
    ax.errorbar(ecmc_sorted_N_10, ecmc_iact_mean_arr_10, ecmc_err_10, fmt='*', capsize=3, markersize=9, color="#b8aaffff", label="ECMC sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_11, ecmc_iact_mean_arr_11, ecmc_err_11, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = Nt, rf = inf")
    #ax.errorbar(ecmc_sorted_N_12, ecmc_iact_mean_arr_12, ecmc_err_12, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = 500, rf = Nt")
    #ax.errorbar(ecmc_sorted_N_13, ecmc_iact_mean_arr_13, ecmc_err_13, fmt='o', capsize=3, markersize=4, color="#a0e9adff", label="ECMC sd = 50, rf = inf")
    #ax.errorbar(ecmc_sorted_N_14, ecmc_iact_mean_arr_14, ecmc_err_14, fmt='o', capsize=3, markersize=4, color="#982be6ff", label="ECMC sd = 200, rf = inf")
    #ax.errorbar(ecmc_sorted_N_15, ecmc_iact_mean_arr_15, ecmc_err_15, fmt='*', capsize=3, markersize=9, color="#0d5120ff", label="ECMC sd = 1000, rf = inf")
    #ax.errorbar(ecmc_sorted_N_16, ecmc_iact_mean_arr_16, ecmc_err_16, fmt='*', capsize=3, markersize=9, color="#be1313ff", label="ECMC sd = 250, rf = inf")
    ax.errorbar(ecmc_sorted_N_17, ecmc_iact_mean_arr_17, ecmc_err_17, fmt='o', capsize=3, markersize=4, color="#6018b9ff", label="ECMC FF = 1.0, sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_18, ecmc_iact_mean_arr_18, ecmc_err_18, fmt='o', capsize=3, markersize=4, color="#fa4f06ff", label="ECMC FF = 10.0, sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_19, ecmc_iact_mean_arr_19, ecmc_err_19, fmt='o', capsize=3, markersize=4, color="#22c31aff", label="ECMC FF = 0.1, sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_20, ecmc_iact_mean_arr_20, ecmc_err_20, fmt='o', capsize=3, markersize=4, color="#76efd5ff", label="ECMC FF = 0.5, sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_21, ecmc_iact_mean_arr_21, ecmc_err_21, fmt='o', capsize=3, markersize=4, color="#f11f14ff", label="ECMC FF = 1.0, sd = 50, rf = inf")
    ax.errorbar(ecmc_sorted_N_22, ecmc_iact_mean_arr_22, ecmc_err_22, fmt='o', capsize=3, markersize=4, color="#10052cff", label="ECMC alt-lifting-scheme, sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_23, ecmc_iact_mean_arr_23, ecmc_err_23, fmt='o', capsize=3, markersize=4, color="#e8a1d1ff", label="ECMC alt-lifting-scheme, FF 1.0, sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_24, ecmc_iact_mean_arr_24, ecmc_err_24, fmt='o', capsize=3, markersize=4, color="#e8a1d1ff", label="ECMC FF 10.0, sd = 500, rf = Nt")
    #ax.errorbar(ecmc_sorted_N_25, ecmc_iact_mean_arr_25, ecmc_err_25, fmt='o', capsize=3, markersize=4, color="#24a245ff", label="ECMC alt-lifting-scheme, FF 10.0, sd = 500, rf = inf")
    ax.errorbar(ecmc_sorted_N_26, ecmc_iact_mean_arr_26, ecmc_err_26, fmt='o', capsize=3, markersize=4, color="#b011e1ff", label="ECMC FF 1.0, sd = Nt, rf = inf")
    ax.errorbar(ecmc_sorted_N_27, ecmc_iact_mean_arr_27, ecmc_err_27, fmt='o', capsize=3, markersize=4, color="#e15311ff", label="ECMC sd = Nt, rf = inf")
    #ax.errorbar(ecmc_sorted_N_28, ecmc_iact_mean_arr_28, ecmc_err_28, fmt='o', capsize=3, markersize=4, color="#289a11ff", label="ECMC FF 10.0, sd = Nt, rf = inf")











    

    
    #ax.annotate(f"E coeff: {e_coeffs[0]:.2f}", xy = (4*10e2, 4*10e1), weight = "bold")
    



    
   
    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("IACT", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties, bbox_to_anchor=(1.0, 1.0))
    #ax.set_ylim(0.17e5, 0.8e7)
   


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("iact_compare.png")#, transparent=True)
    plt.clf()


    fig1, ax1 = plt.subplots(1, 1, figsize = (9, 7))

    ax1.errorbar(ecmc_sorted_N_13, ecmc_iact_mean_arr_13, ecmc_err_13, fmt='o', capsize=3, markersize=4, color="#a0e9adff", label="ECMC sd = 50, rf = inf")
    ax1.errorbar(ecmc_sorted_N_21, ecmc_iact_mean_arr_21, ecmc_err_21, fmt='o', capsize=3, markersize=4, color="#f11f14ff", label="ECMC FF = 1.0, sd = 50, rf = inf")



    ax1.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    #ax1.errorbar(ecmc_sorted_N_2, ecmc_iact_mean_arr_2, ecmc_err_2, fmt='o', capsize=3, markersize=4, color="#af82e5ff", label="ECMC sd=100, rf = 10e12")
    #ax1.errorbar(ecmc_sorted_N_3, ecmc_iact_mean_arr_3, ecmc_err_3, fmt='o', capsize=3, markersize=7, color="#03280eff", label="ECMC sd= 100, rf = 1.0")
    #ax1.errorbar(ecmc_sorted_N_4, ecmc_iact_mean_arr_4, ecmc_err_4, fmt='*', capsize=3, markersize=9, color="#ed7ed0ff", label="ECMC sd=dt")
    #ax1.errorbar(ecmc_sorted_N_5, ecmc_iact_mean_arr_5, ecmc_err_5, fmt='^', capsize=3, markersize=7, color="#ff001eff", label="ECMC sd=100, rf = Nt")
    #ax1.errorbar(ecmc_sorted_N_6, ecmc_iact_mean_arr_6, ecmc_err_6, fmt='^', capsize=3, markersize=7, color="#2b0330ff", label="ECMC sd=120, rf = Nt")
    #ax1.errorbar(ecmc_sorted_N_7, ecmc_iact_mean_arr_7, ecmc_err_7, fmt='^', capsize=3, markersize=7, color="#5e6048ff", label="ECMC sd = 1.0, rf = inf")
    #ax1.errorbar(ecmc_sorted_N_8, ecmc_iact_mean_arr_8, ecmc_err_8, fmt='o', capsize=3, markersize=4, color="#3140aeff", label="ECMC sd = 120, rf = 10e12")
    #ax1.errorbar(ecmc_sorted_N_9, ecmc_iact_mean_arr_9, ecmc_err_9, fmt='o', capsize=3, markersize=4, color="#64e15dff", label="ECMC sd = 500, rf = 10e12")
    #ax1.errorbar(ecmc_sorted_N_10, ecmc_iact_mean_arr_10, ecmc_err_10, fmt='*', capsize=3, markersize=9, color="#b8aaffff", label="ECMC sd = 500, rf = inf")
    #ax1.errorbar(ecmc_sorted_N_11, ecmc_iact_mean_arr_11, ecmc_err_11, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = Nt, rf = inf")
    #ax1.errorbar(ecmc_sorted_N_12, ecmc_iact_mean_arr_12, ecmc_err_12, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = 500, rf = 10e12")

    ax1.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax1.set_ylabel("IACT", fontsize=15, weight = "bold")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties, bbox_to_anchor=(1.0, 1.0))
    #ax.set_ylim(0.17e5, 0.8e7)
   


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("iact_compare_1.png")#, transparent=True)
    plt.clf()

    fig2, ax2 = plt.subplots(1, 1, figsize = (9, 7))

    ax2.errorbar(ecmc_sorted_N_22, ecmc_iact_mean_arr_22, ecmc_err_22, fmt='o', capsize=3, markersize=4, color="#10052cff", label="ECMC alt-lifting-scheme, sd = 500, rf = inf")
    ax2.errorbar(ecmc_sorted_N_23, ecmc_iact_mean_arr_23, ecmc_err_23, fmt='o', capsize=3, markersize=4, color="#e8a1d1ff", label="ECMC alt-lifting-scheme, FF 1.0, sd = 500, rf = inf")
    
    ax2.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    #ax2.errorbar(ecmc_sorted_N_2, ecmc_iact_mean_arr_2, ecmc_err_2, fmt='o', capsize=3, markersize=4, color="#af82e5ff", label="ECMC sd=100, rf = 10e12")
    #ax2.errorbar(ecmc_sorted_N_3, ecmc_iact_mean_arr_3, ecmc_err_3, fmt='o', capsize=3, markersize=7, color="#03280eff", label="ECMC sd= 100, rf = 1.0")
    #ax2.errorbar(ecmc_sorted_N_4, ecmc_iact_mean_arr_4, ecmc_err_4, fmt='*', capsize=3, markersize=9, color="#ed7ed0ff", label="ECMC sd=dt, rf=Nt")
    #ax2.errorbar(ecmc_sorted_N_5, ecmc_iact_mean_arr_5, ecmc_err_5, fmt='^', capsize=3, markersize=7, color="#ff001eff", label="ECMC sd=100, rf = Nt")
    #ax2.errorbar(ecmc_sorted_N_6, ecmc_iact_mean_arr_6, ecmc_err_6, fmt='^', capsize=3, markersize=7, color="#2b0330ff", label="ECMC sd=120, rf = Nt")
    #ax2.errorbar(ecmc_sorted_N_7, ecmc_iact_mean_arr_7, ecmc_err_7, fmt='^', capsize=3, markersize=7, color="#5e6048ff", label="ECMC sd = 1.0, rf = inf")
    #ax2.errorbar(ecmc_sorted_N_8, ecmc_iact_mean_arr_8, ecmc_err_8, fmt='o', capsize=3, markersize=4, color="#3140aeff", label="ECMC sd = 120, rf = 10e12")
    #ax2.errorbar(ecmc_sorted_N_9, ecmc_iact_mean_arr_9, ecmc_err_9, fmt='o', capsize=3, markersize=4, color="#64e15dff", label="ECMC sd = 500, rf = 10e12")
    #ax2.errorbar(ecmc_sorted_N_10, ecmc_iact_mean_arr_10, ecmc_err_10, fmt='*', capsize=3, markersize=9, color="#b8aaffff", label="ECMC sd = 500, rf = inf")
    #ax2.errorbar(ecmc_sorted_N_11, ecmc_iact_mean_arr_11, ecmc_err_11, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = Nt, rf = inf")
    #ax2.errorbar(ecmc_sorted_N_12, ecmc_iact_mean_arr_12, ecmc_err_12, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = 500, rf = 10e12")

    ax2.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax2.set_ylabel("IACT", fontsize=15, weight = "bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties, bbox_to_anchor=(1.0, 1.0))
    #ax.set_ylim(0.17e5, 0.8e7)
    


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("iact_compare_2.png")#, transparent=True)
    plt.clf()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],
         sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16], sys.argv[17],
         sys.argv[18], sys.argv[19], sys.argv[20], sys.argv[21], sys.argv[22], sys.argv[23], sys.argv[24], sys.argv[25], 
         sys.argv[26], sys.argv[27], sys.argv[28], sys.argv[29], sys.argv[30], sys.argv[31], sys.argv[32], sys.argv[33])