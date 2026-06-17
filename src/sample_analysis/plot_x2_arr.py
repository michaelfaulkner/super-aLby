import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(metropolis_x2_data_path, ecmc_x2_data_path, factor_fields_x2_data_path, N, ff_N, propertime):

    N = int(N)
    ff_N = int(ff_N)
    propertime = float(propertime)

    try:
        metropolis_timestep_data = np.load(os.path.join(metropolis_x2_data_path, "x2_metropolis_0.npy"))[:, 1]
        metropolis_storage_arr = np.zeros((len(metropolis_timestep_data), N))
        metropolis_sorted_timestep = metropolis_timestep_data[np.argsort(metropolis_timestep_data)]
        for index in range(N):
            metropolis_x2_timestep =  np.load(os.path.join(metropolis_x2_data_path, f"x2_metropolis_{index}.npy"))
            metropolis_x2_data = metropolis_x2_timestep[:, 0] 
            metropolis_timestep_data = metropolis_x2_timestep[:, 1] 
            argsorted_data = np.argsort(metropolis_timestep_data)
            metropolis_x2_data = metropolis_x2_data[argsorted_data] 
            metropolis_storage_arr[:, index] = metropolis_x2_data

        metropolis_x2_mean_arr = np.mean(metropolis_storage_arr, axis = 1)
        metropolis_x2_mean_arr = metropolis_x2_mean_arr[metropolis_sorted_timestep >= 0.01]
        metropolis_err = np.std(metropolis_storage_arr, axis=1)
        metropolis_err = metropolis_err[metropolis_sorted_timestep >= 0.01]
        metropolis_sorted_timestep = metropolis_sorted_timestep[metropolis_sorted_timestep >= 0.01]
        metropolis_sorted_N = propertime / metropolis_sorted_timestep

        metrop_data = True
    except:
        metrop_data = False


    ecmc_timestep_data = np.load(os.path.join(ecmc_x2_data_path, "x2_ecmc_0.npy"))[:, 1]
    ecmc_storage_arr = np.zeros((len(ecmc_timestep_data), N))
    ecmc_sorted_timestep = ecmc_timestep_data[np.argsort(ecmc_timestep_data)]

    for index in range(N):
        ecmc_x2_timestep =  np.load(os.path.join(ecmc_x2_data_path, f"x2_ecmc_{index}.npy"))
        ecmc_x2_data = ecmc_x2_timestep[:, 0] 
        ecmc_timestep_data = ecmc_x2_timestep[:, 1] 
        argsorted_data = np.argsort(ecmc_timestep_data)
        ecmc_timestep_argsorted = ecmc_timestep_data[argsorted_data]
        ecmc_x2_data = ecmc_x2_data[argsorted_data] 
        ecmc_storage_arr[:, index] = ecmc_x2_data

    ecmc_x2_mean_arr = np.mean(ecmc_storage_arr, axis = 1)

    ecmc_x2_mean_arr = ecmc_x2_mean_arr[ecmc_sorted_timestep >= 0.01]
    ecmc_err = np.std(ecmc_storage_arr, axis=1)
    ecmc_err = ecmc_err[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_timestep = ecmc_sorted_timestep[ecmc_sorted_timestep >= 0.01]
    ecmc_sorted_N = propertime / ecmc_sorted_timestep
    try:
        ff_timestep_data = np.load(os.path.join(factor_fields_x2_data_path, "x2_ecmc_0.npy"))[:, 1]
        ff_storage_arr = np.zeros((len(ff_timestep_data), ff_N))
        ff_sorted_timestep = ff_timestep_data[np.argsort(ff_timestep_data)]

        for index in range(ff_N):
            ff_x2_timestep =  np.load(os.path.join(factor_fields_x2_data_path, f"x2_ecmc_{index}.npy"))
            ff_x2_data = ff_x2_timestep[:, 0] 
            ff_timestep_data = ff_x2_timestep[:, 1] 
            argsorted_data = np.argsort(ff_timestep_data)
            ff_timestep_argsorted = ff_timestep_data[argsorted_data]
            ff_x2_data = ff_x2_data[argsorted_data] 
            ff_storage_arr[:, index] = ff_x2_data
        
        ff_x2_mean_arr = np.mean(ff_storage_arr, axis = 1)
        ff_x2_mean_arr = ff_x2_mean_arr[ff_sorted_timestep >= 0.01]
        ff_err = np.std(ff_storage_arr, axis=1)
        ff_err = ff_err[ff_sorted_timestep >= 0.01]
        ff_sorted_timestep = ff_sorted_timestep[ff_sorted_timestep >= 0.01]
        ff_sorted_N = propertime / ff_sorted_timestep
        ff_data = True

    except:
        ff_data = False

    #if metrop_data:
        #m_fit_trim = -9#-7
        #m_coeffs = np.polyfit(np.log(metropolis_sorted_N[:m_fit_trim]), np.log(metropolis_x2_mean_arr[:m_fit_trim]), deg=1)
        #fitted_m = m_coeffs[1] + np.multiply(np.log(metropolis_sorted_N[:m_fit_trim]), m_coeffs[0])
        #print(m_coeffs)

    #e_fit_trim = -13 #-13
    #ff_fit_trim = -13
  
    #e_coeffs = np.polyfit(np.log(ecmc_sorted_N[:e_fit_trim]), np.log(ecmc_x2_mean_arr[:e_fit_trim]), deg=1)

    #fitted_e = e_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:e_fit_trim]), e_coeffs[0])


    
    #print(e_coeffs)

    fig, ax = plt.subplots(1, 1)

    if metrop_data:
        #ax.plot(metropolis_sorted_N[:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
        ax.errorbar(metropolis_sorted_N, metropolis_x2_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis MC")
        #a#x.annotate(f"M coeff: {m_coeffs[0]:.2f}", xy = (8*10e1, 3*10e1), weight = "bold")
    
    #ax.plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e), color="#d97dd9ff")


    if ff_data:
   
        #ff_coeffs = np.polyfit(np.log(ff_sorted_N[:ff_fit_trim]), np.log(ff_x2_mean_arr[:ff_fit_trim]), deg=1)
        #fitted_ff = ff_coeffs[1] + np.multiply(np.log(ff_sorted_N[:ff_fit_trim]), ff_coeffs[0])
        #print(ff_coeffs)
       # ax.plot(ff_sorted_N[:ff_fit_trim], np.exp(fitted_ff), color="#a7385eff")
        ax.errorbar(ff_sorted_N, ff_x2_mean_arr, ff_err, fmt='o', capsize=3, markersize=4, color="#950834ff", label="ECMC with Factor Fields")
        #ax.annotate(f"FF coeff: {ff_coeffs[0]:.2f}", xy = (4*10e2, 0.7*10e1), weight = "bold")

    
    ax.errorbar(ecmc_sorted_N, ecmc_x2_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC")
    

    
    #ax.annotate(f"E coeff: {e_coeffs[0]:.2f}", xy = (4*10e2, 4*10e1), weight = "bold")
    



    
   
    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("<x^2>", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_properties = {'weight':'bold'}
    plt.legend(prop=legend_properties)
    #ax.set_ylim(0.17e5, 0.8e7)
   


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("test_qho.png")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])