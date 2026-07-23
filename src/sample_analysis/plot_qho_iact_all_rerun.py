import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(metropolis_iact_data_path, ecmc_iact_data_path, metropolis_iact_data_path_2, ecmc_iact_data_path_2, ecmc_iact_data_path_3,
          ecmc_iact_data_path_4, ecmc_iact_data_path_5, ecmc_iact_data_path_6, ecmc_iact_data_path_7, ecmc_iact_data_path_8,
          ecmc_iact_data_path_9, ecmc_iact_data_path_10, ecmc_iact_data_path_11, ecmc_iact_data_path_12, N, ff_N, propertime):

    N = int(N)
    ff_N = int(ff_N)
    propertime = float(propertime)

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


    if metrop_data:
        m_fit_trim = -1#-7
        m_coeffs = np.polyfit(np.log(metropolis_sorted_N[:m_fit_trim]), np.log(metropolis_iact_mean_arr[:m_fit_trim]), deg=1)
        fitted_m = m_coeffs[1] + np.multiply(np.log(metropolis_sorted_N[:m_fit_trim]), m_coeffs[0])
        print(m_coeffs)

    e_fit_trim = -1
  
    e_coeffs = np.polyfit(np.log(ecmc_sorted_N[:e_fit_trim]), np.log(ecmc_iact_mean_arr[:e_fit_trim]), deg=1)

    fitted_e = e_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:e_fit_trim]), e_coeffs[0])


    
    #print(e_coeffs)

    fig, ax = plt.subplots(1, 1, figsize = (9, 7))

    if metrop_data:
        #ax.plot(metropolis_sorted_N[:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
        #ax.errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis MC")
        #ax.annotate(f"M coeff: {m_coeffs[0]:.2f}", xy = (8*10e1, 3*10e1), weight = "bold")

        #ax.errorbar(metropolis_sorted_N_2, metropolis_iact_mean_arr_2, metropolis_err_2, fmt='^', capsize=3, markersize=4, color="#16e104ff", label="Metropolis MC 2 ")
        #ax.annotate(f"M coeff: {m_coeffs[0]:.2f}", xy = (8*10e1, 3*10e1), weight = "bold")
        pass
    #ax.plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e), color="#d97dd9ff")

    
    ax.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    #ax.errorbar(ecmc_sorted_N_2, ecmc_iact_mean_arr_2, ecmc_err_2, fmt='o', capsize=3, markersize=4, color="#af82e5ff", label="ECMC sd=100, rf = 10e12")
    #ax.errorbar(ecmc_sorted_N_3, ecmc_iact_mean_arr_3, ecmc_err_3, fmt='o', capsize=3, markersize=4, color="#0d5b26ff", label="ECMC sd= 100, rf = 1.0")
    #ax.errorbar(ecmc_sorted_N_4, ecmc_iact_mean_arr_4, ecmc_err_4, fmt='o', capsize=3, markersize=4, color="#e281c8ff", label="ECMC sd=dt")
    #ax.errorbar(ecmc_sorted_N_5, ecmc_iact_mean_arr_5, ecmc_err_5, fmt='^', capsize=3, markersize=7, color="#ff001eff", label="ECMC sd=100, rf = Nt")
    ax.errorbar(ecmc_sorted_N_6, ecmc_iact_mean_arr_6, ecmc_err_6, fmt='^', capsize=3, markersize=7, color="#2b0330ff", label="ECMC sd=120, rf = Nt")
    #ax.errorbar(ecmc_sorted_N_7, ecmc_iact_mean_arr_7, ecmc_err_7, fmt='o', capsize=3, markersize=4, color="#cdde37ff", label="ECMC sd = 1.0, rf = inf")
    ax.errorbar(ecmc_sorted_N_8, ecmc_iact_mean_arr_8, ecmc_err_8, fmt='o', capsize=3, markersize=4, color="#3140aeff", label="ECMC sd = 120, rf = 10e12")
    #ax.errorbar(ecmc_sorted_N_9, ecmc_iact_mean_arr_9, ecmc_err_9, fmt='o', capsize=3, markersize=4, color="#64e15dff", label="ECMC sd = 500, rf = 10e12")
    ax.errorbar(ecmc_sorted_N_10, ecmc_iact_mean_arr_10, ecmc_err_10, fmt='*', capsize=3, markersize=9, color="#b8aaffff", label="ECMC sd = 500, rf = inf")
    #ax.errorbar(ecmc_sorted_N_11, ecmc_iact_mean_arr_11, ecmc_err_11, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = Nt, rf = inf")
    #ax.errorbar(ecmc_sorted_N_12, ecmc_iact_mean_arr_12, ecmc_err_12, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = 500, rf = 10e12")




    

    
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

    ax1.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    ax1.errorbar(ecmc_sorted_N_2, ecmc_iact_mean_arr_2, ecmc_err_2, fmt='o', capsize=3, markersize=4, color="#af82e5ff", label="ECMC sd=100, rf = 10e12")
    ax1.errorbar(ecmc_sorted_N_3, ecmc_iact_mean_arr_3, ecmc_err_3, fmt='o', capsize=3, markersize=7, color="#03280eff", label="ECMC sd= 100, rf = 1.0")
    #ax1.errorbar(ecmc_sorted_N_4, ecmc_iact_mean_arr_4, ecmc_err_4, fmt='*', capsize=3, markersize=9, color="#ed7ed0ff", label="ECMC sd=dt")
    ax1.errorbar(ecmc_sorted_N_5, ecmc_iact_mean_arr_5, ecmc_err_5, fmt='^', capsize=3, markersize=7, color="#ff001eff", label="ECMC sd=100, rf = Nt")
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
    
    ax2.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    #ax2.errorbar(ecmc_sorted_N_2, ecmc_iact_mean_arr_2, ecmc_err_2, fmt='o', capsize=3, markersize=4, color="#af82e5ff", label="ECMC sd=100, rf = 10e12")
    #ax2.errorbar(ecmc_sorted_N_3, ecmc_iact_mean_arr_3, ecmc_err_3, fmt='o', capsize=3, markersize=7, color="#03280eff", label="ECMC sd= 100, rf = 1.0")
    ax2.errorbar(ecmc_sorted_N_4, ecmc_iact_mean_arr_4, ecmc_err_4, fmt='*', capsize=3, markersize=9, color="#ed7ed0ff", label="ECMC sd=dt, rf=Nt")
    #ax2.errorbar(ecmc_sorted_N_5, ecmc_iact_mean_arr_5, ecmc_err_5, fmt='^', capsize=3, markersize=7, color="#ff001eff", label="ECMC sd=100, rf = Nt")
    #ax2.errorbar(ecmc_sorted_N_6, ecmc_iact_mean_arr_6, ecmc_err_6, fmt='^', capsize=3, markersize=7, color="#2b0330ff", label="ECMC sd=120, rf = Nt")
    ax2.errorbar(ecmc_sorted_N_7, ecmc_iact_mean_arr_7, ecmc_err_7, fmt='^', capsize=3, markersize=7, color="#5e6048ff", label="ECMC sd = 1.0, rf = inf")
    #ax2.errorbar(ecmc_sorted_N_8, ecmc_iact_mean_arr_8, ecmc_err_8, fmt='o', capsize=3, markersize=4, color="#3140aeff", label="ECMC sd = 120, rf = 10e12")
    #ax2.errorbar(ecmc_sorted_N_9, ecmc_iact_mean_arr_9, ecmc_err_9, fmt='o', capsize=3, markersize=4, color="#64e15dff", label="ECMC sd = 500, rf = 10e12")
    #ax2.errorbar(ecmc_sorted_N_10, ecmc_iact_mean_arr_10, ecmc_err_10, fmt='*', capsize=3, markersize=9, color="#b8aaffff", label="ECMC sd = 500, rf = inf")
    ax2.errorbar(ecmc_sorted_N_11, ecmc_iact_mean_arr_11, ecmc_err_11, fmt='o', capsize=3, markersize=4, color="#00f7ffff", label="ECMC sd = Nt, rf = inf")
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
         sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15], sys.argv[16], sys.argv[17])