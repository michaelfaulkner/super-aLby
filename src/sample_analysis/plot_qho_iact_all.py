import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(metropolis_iact_data_path, ecmc_iact_data_path, ecmc_symm_iact_data_path, N, propertime, ecmc_event_rate,
         ecmc_symm_event_rate, metropolis_iact_qah_data_path, ecmc_iact_qah_data_path, ecmc_symm_iact_qah_data_path,
         ecmc_qah_event_rate, ecmc_symm_qah_event_rate ):

    N = int(N)
    propertime = float(propertime)


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
    metropolis_iact_mean_arr = metropolis_sorted_N * metropolis_iact_mean_arr



    metropolis_qah_timestep_data = np.load(os.path.join(metropolis_iact_qah_data_path, "iact_metropolis_0.npy"))[:, 1]
    metropolis_qah_storage_arr = np.zeros((len(metropolis_qah_timestep_data), N))
    metropolis_qah_sorted_timestep = metropolis_qah_timestep_data[np.argsort(metropolis_qah_timestep_data)]
    for index in range(N):
        metropolis_qah_iact_timestep =  np.load(os.path.join(metropolis_iact_qah_data_path, f"iact_metropolis_{index}.npy"))
        metropolis_qah_iact_data = metropolis_qah_iact_timestep[:, 0] 
        metropolis_qah_timestep_data = metropolis_qah_iact_timestep[:, 1] 
        argsorted_data_qah = np.argsort(metropolis_qah_timestep_data)
        metropolis_qah_iact_data = metropolis_qah_iact_data[argsorted_data_qah] 
        metropolis_qah_storage_arr[:, index] = metropolis_qah_iact_data

    metropolis_qah_iact_mean_arr = np.mean(metropolis_qah_storage_arr, axis = 1)
    metropolis_qah_iact_mean_arr = metropolis_qah_iact_mean_arr[metropolis_qah_sorted_timestep > 0.01]
    metropolis_qah_err = np.std(metropolis_qah_storage_arr, axis=1)
    metropolis_qah_err = metropolis_qah_err[metropolis_qah_sorted_timestep > 0.01]
    metropolis_qah_sorted_timestep = metropolis_qah_sorted_timestep[metropolis_qah_sorted_timestep > 0.01]
    metropolis_qah_sorted_N = propertime / metropolis_qah_sorted_timestep
    metropolis_qah_iact_mean_arr = metropolis_qah_sorted_N * metropolis_qah_iact_mean_arr



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

    ecmc_qah_timestep_data = np.load(os.path.join(ecmc_iact_qah_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_qah_storage_arr = np.zeros((len(ecmc_qah_timestep_data), N))
    ecmc_qah_sorted_timestep = ecmc_qah_timestep_data[np.argsort(ecmc_qah_timestep_data)]

    for index in range(N):
        ecmc_qah_iact_timestep =  np.load(os.path.join(ecmc_iact_qah_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_qah_iact_data = ecmc_qah_iact_timestep[:, 0] 
        ecmc_qah_timestep_data = ecmc_qah_iact_timestep[:, 1] 
        argsorted_data_qah = np.argsort(ecmc_qah_timestep_data)
        ecmc_qah_timestep_argsorted = ecmc_qah_timestep_data[argsorted_data_qah]
        ecmc_qah_iact_data = ecmc_qah_iact_data[argsorted_data_qah] 
        ecmc_qah_storage_arr[:, index] = ecmc_qah_iact_data

    ecmc_qah_iact_mean_arr = np.mean(ecmc_qah_storage_arr, axis = 1)
    ecmc_qah_iact_mean_arr = ecmc_qah_iact_mean_arr[ecmc_qah_sorted_timestep >= 0.01]
    ecmc_qah_err = np.std(ecmc_qah_storage_arr, axis=1)
    ecmc_qah_err = ecmc_qah_err[ecmc_qah_sorted_timestep >= 0.01]
    ecmc_qah_sorted_timestep = ecmc_qah_sorted_timestep[ecmc_qah_sorted_timestep >= 0.01]
    ecmc_qah_sorted_N = propertime / ecmc_qah_sorted_timestep


    ecmc_symm_timestep_data = np.load(os.path.join(ecmc_symm_iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_symm_storage_arr = np.zeros((len(ecmc_symm_timestep_data), N))
    ecmc_symm_sorted_timestep = ecmc_symm_timestep_data[np.argsort(ecmc_symm_timestep_data)]

    for index in range(N):
        ecmc_symm_iact_timestep =  np.load(os.path.join(ecmc_symm_iact_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_symm_iact_data = ecmc_symm_iact_timestep[:, 0] 
        ecmc_symm_timestep_data = ecmc_symm_iact_timestep[:, 1] 
        argsorted_data = np.argsort(ecmc_symm_timestep_data)
        ecmc_symm_timestep_argsorted = ecmc_symm_timestep_data[argsorted_data]
        ecmc_symm_iact_data = ecmc_symm_iact_data[argsorted_data] 
        ecmc_symm_storage_arr[:, index] = ecmc_symm_iact_data
    
    ecmc_symm_iact_mean_arr = np.mean(ecmc_symm_storage_arr, axis = 1)
    ecmc_symm_iact_mean_arr = ecmc_symm_iact_mean_arr[ecmc_symm_sorted_timestep >= 0.01]
    ecmc_symm_err = np.std(ecmc_symm_storage_arr, axis=1)
    ecmc_symm_err = ecmc_symm_err[ecmc_symm_sorted_timestep >= 0.01]
    ecmc_symm_sorted_timestep = ecmc_symm_sorted_timestep[ecmc_symm_sorted_timestep >= 0.01]
    ecmc_symm_sorted_N = propertime / ecmc_symm_sorted_timestep
  
    ecmc_qah_symm_timestep_data = np.load(os.path.join(ecmc_symm_iact_qah_data_path, "iact_ecmc_0.npy"))[:, 1]
    ecmc_qah_symm_storage_arr = np.zeros((len(ecmc_qah_symm_timestep_data), N))
    ecmc_qah_symm_sorted_timestep = ecmc_qah_symm_timestep_data[np.argsort(ecmc_qah_symm_timestep_data)]

    for index in range(N):
        ecmc_qah_symm_iact_timestep =  np.load(os.path.join(ecmc_symm_iact_qah_data_path, f"iact_ecmc_{index}.npy"))
        ecmc_qah_symm_iact_data = ecmc_qah_symm_iact_timestep[:, 0] 
        ecmc_qah_symm_timestep_data = ecmc_qah_symm_iact_timestep[:, 1] 
        argsorted_data_qah = np.argsort(ecmc_qah_symm_timestep_data)
        ecmc_qah_symm_timestep_argsorted = ecmc_qah_symm_timestep_data[argsorted_data_qah]
        ecmc_qah_symm_iact_data = ecmc_qah_symm_iact_data[argsorted_data_qah] 
        ecmc_qah_symm_storage_arr[:, index] = ecmc_qah_symm_iact_data
    
    ecmc_qah_symm_iact_mean_arr = np.mean(ecmc_qah_symm_storage_arr, axis = 1)
    ecmc_qah_symm_iact_mean_arr = ecmc_qah_symm_iact_mean_arr[ecmc_qah_symm_sorted_timestep >= 0.01]
    ecmc_qah_symm_err = np.std(ecmc_qah_symm_storage_arr, axis=1)
    ecmc_qah_symm_err = ecmc_qah_symm_err[ecmc_qah_symm_sorted_timestep >= 0.01]
    ecmc_qah_symm_sorted_timestep = ecmc_qah_symm_sorted_timestep[ecmc_qah_symm_sorted_timestep >= 0.01]
    ecmc_qah_symm_sorted_N = propertime / ecmc_qah_symm_sorted_timestep

    m_fit_trim = -7
    e_fit_trim = -8
    ecmc_symm_fit_trim = -8

    m_coeffs = np.polyfit(np.log(metropolis_sorted_N[1:m_fit_trim]), np.log(metropolis_iact_mean_arr[1:m_fit_trim]), deg=1)
    fitted_m = m_coeffs[1] + np.multiply(np.log(metropolis_sorted_N[1:m_fit_trim]), m_coeffs[0])

    e_coeffs = np.polyfit(np.log(ecmc_sorted_N[:e_fit_trim]), np.log(ecmc_iact_mean_arr[:e_fit_trim]), deg=1)
    fitted_e = e_coeffs[1] + np.multiply(np.log(ecmc_sorted_N[:e_fit_trim]), e_coeffs[0])

    ecmc_symm_coeffs = np.polyfit(np.log(ecmc_symm_sorted_N[:ecmc_symm_fit_trim]), np.log(ecmc_symm_iact_mean_arr[:ecmc_symm_fit_trim]), deg=1)
    fitted_ecmc_symm = ecmc_symm_coeffs[1] + np.multiply(np.log(ecmc_symm_sorted_N[:ecmc_symm_fit_trim]), ecmc_symm_coeffs[0])

    m_fit_trim_qah = -7
    e_fit_trim_qah = -8
    ecmc_symm_fit_trim_qah = -8

    m_coeffs_qah = np.polyfit(np.log(metropolis_qah_sorted_N[:m_fit_trim_qah]), np.log(metropolis_qah_iact_mean_arr[:m_fit_trim_qah]), deg=1)
    fitted_m_qah = m_coeffs_qah[1] + np.multiply(np.log(metropolis_qah_sorted_N[:m_fit_trim_qah]), m_coeffs_qah[0])

    e_coeffs_qah = np.polyfit(np.log(ecmc_qah_sorted_N[:e_fit_trim_qah]), np.log(ecmc_qah_iact_mean_arr[:e_fit_trim_qah]), deg=1)
    fitted_e_qah = e_coeffs_qah[1] + np.multiply(np.log(ecmc_qah_sorted_N[:e_fit_trim_qah]), e_coeffs_qah[0])

    ecmc_symm_coeffs_qah = np.polyfit(np.log(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah]), np.log(ecmc_qah_symm_iact_mean_arr[:ecmc_symm_fit_trim_qah]), deg=1)
    fitted_ecmc_symm_qah = ecmc_symm_coeffs_qah[1] + np.multiply(np.log(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah]), ecmc_symm_coeffs_qah[0])


    fig, ax = plt.subplots(1, 1)


    ax.plot(metropolis_sorted_N[1:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
    ax.errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis MC")

    ax.plot(ecmc_symm_sorted_N[:ecmc_symm_fit_trim], np.exp(fitted_ecmc_symm), color="#a7385eff")
    ax.errorbar(ecmc_symm_sorted_N, ecmc_symm_iact_mean_arr, ecmc_symm_err, fmt='o', capsize=3, markersize=4, color="#950834ff", label="ECMC Symmetric")

    ax.errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC Asymmetric")
    ax.plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e), color="#a73896ff")

    
    ax.annotate(f"{e_coeffs[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    ax.annotate(f"{ecmc_symm_coeffs[0]:.2f}", xy = (10e2, 2*10e2), weight = "bold", fontsize=18)
    ax.annotate(f"{m_coeffs[0]:.2f}", xy = (1*10e2, 1*10e4), weight = "bold", fontsize=18)
   
   
    ax.set_xlabel(r"$N_{\tau}$", fontsize=30, weight = "bold")
    ax.set_ylabel(r"$s_{int}$", fontsize=45, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=20)
    legend_properties = {'weight':'bold'}
    legend = ax.legend(loc='lower right', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    plt.tight_layout()
   

    plt.savefig("iact_mean_positions.pdf")#, transparent=True)
    plt.clf()

    fig, ax = plt.subplots(1, 1)


    ax.plot(metropolis_qah_sorted_N[:m_fit_trim_qah], np.exp(fitted_m_qah), color="#3bd27dff")
    ax.errorbar(metropolis_qah_sorted_N, metropolis_qah_iact_mean_arr, metropolis_qah_err, fmt='^', capsize=3, markersize=4, color="#18a553ff", label="Metropolis MC")

    ax.plot(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah], np.exp(fitted_ecmc_symm_qah), color="#47e7bfff")
    ax.errorbar(ecmc_qah_symm_sorted_N, ecmc_qah_symm_iact_mean_arr, ecmc_qah_symm_err, fmt='o', capsize=3, markersize=4, color="#11c7caff", label="ECMC Symmetric")

    ax.errorbar(ecmc_qah_sorted_N, ecmc_qah_iact_mean_arr, ecmc_qah_err, fmt='o', capsize=3, markersize=4, color="#651585ff", label="ECMC Asymmetric")
    ax.plot(ecmc_qah_sorted_N[:e_fit_trim_qah], np.exp(fitted_e_qah), color="#bf58d1ff")

    
    ax.annotate(f"{e_coeffs_qah[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    ax.annotate(f"{ecmc_symm_coeffs_qah[0]:.2f}", xy = (10e2, 2*10e2), weight = "bold", fontsize=18)
    ax.annotate(f"{m_coeffs_qah[0]:.2f}", xy = (1*10e2, 1*10e4), weight = "bold", fontsize=18)
   
   
    ax.set_xlabel(r"$N_{\tau}$", fontsize=30, weight = "bold")
    ax.set_ylabel(r"$s_{int}$", fontsize=45, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=20)
    legend_properties = {'weight':'bold'}
    legend = ax.legend(loc='lower right', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    plt.tight_layout()
   

    plt.savefig("iact_mean_positions_qah.pdf")#, transparent=True)
    plt.clf()

    ecmc_mean_events = pd.read_csv(ecmc_event_rate, header=0, index_col="timestep")
    ecmc_mean_event_rate_mean = ecmc_mean_events.mean(axis=1)
    print(ecmc_mean_event_rate_mean)

    ecmc_symm_mean_events = pd.read_csv(ecmc_symm_event_rate, header=0, index_col="timestep")
    ecmc_symm_mean_event_rate_mean = ecmc_symm_mean_events.mean(axis=1)
    print(ecmc_symm_mean_event_rate_mean)

    ecmc_comp_effort = ecmc_iact_mean_arr * ecmc_mean_event_rate_mean.to_numpy()

    ecmc_symm_comp_effort = ecmc_symm_iact_mean_arr * ecmc_symm_mean_event_rate_mean.to_numpy()

    e_coeffs_ce = np.polyfit(np.log(ecmc_sorted_N[:e_fit_trim]), np.log(ecmc_comp_effort[:e_fit_trim]), deg=1)
    fitted_e_ce = e_coeffs_ce[1] + np.multiply(np.log(ecmc_sorted_N[:e_fit_trim]), e_coeffs_ce[0])

    ecmc_symm_coeffs_ce = np.polyfit(np.log(ecmc_symm_sorted_N[:ecmc_symm_fit_trim]), np.log(ecmc_symm_comp_effort[:ecmc_symm_fit_trim]), deg=1)
    fitted_ecmc_symm_ce = ecmc_symm_coeffs_ce[1] + np.multiply(np.log(ecmc_symm_sorted_N[:ecmc_symm_fit_trim]), ecmc_symm_coeffs_ce[0])


    ecmc_qah_mean_events = pd.read_csv(ecmc_qah_event_rate, header=0, index_col="timestep")
    ecmc_qah_mean_event_rate_mean = ecmc_qah_mean_events.mean(axis=1)

    ecmc_qah_symm_mean_events = pd.read_csv(ecmc_symm_qah_event_rate, header=0, index_col="timestep")
    ecmc_qah_symm_mean_event_rate_mean = ecmc_qah_symm_mean_events.mean(axis=1)
 

    ecmc_qah_comp_effort = ecmc_qah_iact_mean_arr * ecmc_qah_mean_event_rate_mean.to_numpy()

    ecmc_qah_symm_comp_effort = ecmc_qah_symm_iact_mean_arr * ecmc_qah_symm_mean_event_rate_mean.to_numpy()

    e_coeffs_ce_qah = np.polyfit(np.log(ecmc_qah_sorted_N[:e_fit_trim_qah]), np.log(ecmc_qah_comp_effort[:e_fit_trim_qah]), deg=1)
    fitted_e_ce_qah = e_coeffs_ce_qah[1] + np.multiply(np.log(ecmc_qah_sorted_N[:e_fit_trim_qah]), e_coeffs_ce_qah[0])

    ecmc_symm_coeffs_ce_qah = np.polyfit(np.log(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah]), np.log(ecmc_qah_symm_comp_effort[:ecmc_symm_fit_trim_qah]), deg=1)
    fitted_ecmc_symm_ce_qah = ecmc_symm_coeffs_ce_qah[1] + np.multiply(np.log(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah]), ecmc_symm_coeffs_ce_qah[0])



    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(metropolis_sorted_N[1:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
    ax2.errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis")
    ax2.annotate(f"{m_coeffs[0]:.2f}", xy = (2*10e2, 5*10e4), weight = "bold", fontsize=18)

    ax2.errorbar(ecmc_symm_sorted_N, ecmc_symm_comp_effort, ecmc_symm_err, fmt='o',
                 capsize=3, markersize=4, color="#950834ff", label="ECMC Symmetric")

    ax2.plot(ecmc_symm_sorted_N[:ecmc_symm_fit_trim], np.exp(fitted_ecmc_symm_ce), color="#a7385eff")
    ax2.annotate(f"{ecmc_symm_coeffs_ce[0]:.2f}", xy = (0.7*10e2, 7*10e2), weight = "bold", fontsize=18)
    
    ax2.errorbar(ecmc_sorted_N, ecmc_comp_effort, ecmc_err, fmt='o', capsize=3,
                 markersize=4, color="#e20acdff", label="ECMC Asymmetric")
    ax2.plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e_ce), color="#a73896ff")
    ax2.annotate(f"{e_coeffs_ce[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    
    ax2.set_xlabel(r"$N_{\tau}$", fontsize=30, weight = "bold")
    ax2.set_ylabel(r"$\Omega$", fontsize=45, weight = "bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=20)
    legend_properties = {'weight':'bold'}
    legend = ax2.legend(loc='lower right', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
 
    plt.tight_layout()
       
    plt.tight_layout()
    plt.savefig("iact_mean_positions_comp_effort.pdf")
    plt.clf()



    fig2, ax2 = plt.subplots(1, 1)

    ax2.plot(metropolis_qah_sorted_N[:m_fit_trim_qah], np.exp(fitted_m_qah), color="#3bd27dff")
    ax2.errorbar(metropolis_qah_sorted_N, metropolis_qah_iact_mean_arr, metropolis_qah_err, fmt='^', capsize=3, markersize=4, color="#18a553ff", label="Metropolis")
    ax2.annotate(f"{m_coeffs_qah[0]:.2f}", xy = (2*10e2, 5*10e4), weight = "bold", fontsize=18)

    ax2.errorbar(ecmc_qah_symm_sorted_N, ecmc_qah_symm_comp_effort, ecmc_symm_err, fmt='o',
                 capsize=3, markersize=4, color="#11c7caff", label="ECMC Symmetric")

    ax2.plot(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah], np.exp(fitted_ecmc_symm_ce_qah), color="#47e7bfff")
    ax2.annotate(f"{ecmc_symm_coeffs_ce_qah[0]:.2f}", xy = (0.7*10e2, 7*10e2), weight = "bold", fontsize=18)
    
    ax2.errorbar(ecmc_qah_sorted_N, ecmc_qah_comp_effort, ecmc_qah_err, fmt='o', capsize=3,
                 markersize=4, color="#651585ff", label="ECMC Asymmetric")
    ax2.plot(ecmc_qah_sorted_N[:e_fit_trim_qah], np.exp(fitted_e_ce_qah), color="#bf58d1ff")
    ax2.annotate(f"{e_coeffs_ce_qah[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    
    ax2.set_xlabel(r"$N_{\tau}$", fontsize=30, weight = "bold")
    ax2.set_ylabel(r"$\Omega$", fontsize=45, weight = "bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.tick_params(direction="in", left="off",labelleft="off", axis='both', which='both', labelsize=20)
    legend_properties = {'weight':'bold'}
    legend = ax2.legend(loc='lower right', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    for tick in ax2.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_yticklabels():
        tick.set_fontweight('bold')
 
    plt.tight_layout()
       
    plt.tight_layout()
    plt.savefig("iact_mean_positions_comp_effort_qah.pdf")
    plt.clf()

    fig, ax = plt.subplots(2,2, sharex=True, sharey="row", figsize=(10,6))
    ax[0,0].plot(metropolis_sorted_N[1:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
    ax[0,0].errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis")

    ax[0,0].plot(ecmc_symm_sorted_N[:ecmc_symm_fit_trim], np.exp(fitted_ecmc_symm), color="#a7385eff")
    ax[0,0].errorbar(ecmc_symm_sorted_N, ecmc_symm_iact_mean_arr, ecmc_symm_err, fmt='o', capsize=3, markersize=4, color="#950834ff", label="ECMC Symmetric")

    ax[0,0].errorbar(ecmc_sorted_N, ecmc_iact_mean_arr, ecmc_err, fmt='o', capsize=3, markersize=4, color="#e20acdff", label="ECMC Asymmetric")
    ax[0,0].plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e), color="#a73896ff")

    
    ax[0,0].annotate(f"{e_coeffs[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    ax[0,0].annotate(f"{ecmc_symm_coeffs[0]:.2f}", xy = (10e2, 2*10e2), weight = "bold", fontsize=18)
    ax[0,0].annotate(f"{m_coeffs[0]:.2f}", xy = (1*10e2, 1*10e4), weight = "bold", fontsize=18)

    ax[0,1].plot(metropolis_qah_sorted_N[:m_fit_trim_qah], np.exp(fitted_m_qah), color="#3bd27dff")
    ax[0,1].errorbar(metropolis_qah_sorted_N, metropolis_qah_iact_mean_arr, metropolis_qah_err, fmt='^', capsize=3, markersize=4, color="#18a553ff", label="Metropolis")

    ax[0,1].plot(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah], np.exp(fitted_ecmc_symm_qah), color="#47e7bfff")
    ax[0,1].errorbar(ecmc_qah_symm_sorted_N, ecmc_qah_symm_iact_mean_arr, ecmc_qah_symm_err, fmt='o', capsize=3, markersize=4, color="#11c7caff", label="ECMC Symmetric")

    ax[0,1].errorbar(ecmc_qah_sorted_N, ecmc_qah_iact_mean_arr, ecmc_qah_err, fmt='o', capsize=3, markersize=4, color="#651585ff", label="ECMC Asymmetric")
    ax[0,1].plot(ecmc_qah_sorted_N[:e_fit_trim_qah], np.exp(fitted_e_qah), color="#bf58d1ff")

    
    ax[0,1].annotate(f"{e_coeffs_qah[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    ax[0,1].annotate(f"{ecmc_symm_coeffs_qah[0]:.2f}", xy = (10e2, 2*10e2), weight = "bold", fontsize=18)
    ax[0,1].annotate(f"{m_coeffs_qah[0]:.2f}", xy = (1*10e2, 1*10e4), weight = "bold", fontsize=18)


    ax[1,0].plot(metropolis_sorted_N[1:m_fit_trim], np.exp(fitted_m), color="#f9a37bff")
    ax[1,0].errorbar(metropolis_sorted_N, metropolis_iact_mean_arr, metropolis_err, fmt='^', capsize=3, markersize=4, color="#e16f04ff", label="Metropolis")
    ax[1,0].annotate(f"{m_coeffs[0]:.2f}", xy = (2*10e2, 5*10e4), weight = "bold", fontsize=18)

    ax[1,0].errorbar(ecmc_symm_sorted_N, ecmc_symm_comp_effort, ecmc_symm_err, fmt='o',
                 capsize=3, markersize=4, color="#950834ff", label="ECMC Symmetric")

    ax[1,0].plot(ecmc_symm_sorted_N[:ecmc_symm_fit_trim], np.exp(fitted_ecmc_symm_ce), color="#a7385eff")
    ax[1,0].annotate(f"{ecmc_symm_coeffs_ce[0]:.2f}", xy = (0.7*10e2, 7*10e2), weight = "bold", fontsize=18)
    
    ax[1,0].errorbar(ecmc_sorted_N, ecmc_comp_effort, ecmc_err, fmt='o', capsize=3,
                 markersize=4, color="#e20acdff", label="ECMC Asymmetric")
    ax[1,0].plot(ecmc_sorted_N[:e_fit_trim], np.exp(fitted_e_ce), color="#a73896ff")
    ax[1,0].annotate(f"{e_coeffs_ce[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)

    ax[1,1].plot(metropolis_qah_sorted_N[:m_fit_trim_qah], np.exp(fitted_m_qah), color="#3bd27dff")
    ax[1,1].errorbar(metropolis_qah_sorted_N, metropolis_qah_iact_mean_arr, metropolis_qah_err, fmt='^', capsize=3, markersize=4, color="#18a553ff", label="Metropolis")
    ax[1,1].annotate(f"{m_coeffs_qah[0]:.2f}", xy = (2*10e2, 5*10e4), weight = "bold", fontsize=18)

    ax[1,1].errorbar(ecmc_qah_symm_sorted_N, ecmc_qah_symm_comp_effort, ecmc_symm_err, fmt='o',
                 capsize=3, markersize=4, color="#11c7caff", label="ECMC Symmetric")

    ax[1,1].plot(ecmc_qah_symm_sorted_N[:ecmc_symm_fit_trim_qah], np.exp(fitted_ecmc_symm_ce_qah), color="#47e7bfff")
    ax[1,1].annotate(f"{ecmc_symm_coeffs_ce_qah[0]:.2f}", xy = (0.7*10e2, 7*10e2), weight = "bold", fontsize=18)
    
    ax[1,1].errorbar(ecmc_qah_sorted_N, ecmc_qah_comp_effort, ecmc_qah_err, fmt='o', capsize=3,
                 markersize=4, color="#651585ff", label="ECMC Asymmetric")
    ax[1,1].plot(ecmc_qah_sorted_N[:e_fit_trim_qah], np.exp(fitted_e_ce_qah), color="#bf58d1ff")
    ax[1,1].annotate(f"{e_coeffs_ce_qah[0]:.2f}", xy = (4*10e2, 0.5*10e1), weight = "bold", fontsize=18)
    
    ax[1,0].set_xlabel(r"$N_{\tau}$", fontsize=30, weight = "bold")
    ax[1,1].set_xlabel(r"$N_{\tau}$", fontsize=30, weight = "bold")
    ax[0,0].set_ylabel(r"$s_{int}$", fontsize=45, weight = "bold")
    ax[1,0].set_ylabel(r"$\Omega$", fontsize=45, weight = "bold")
    ax[0,0].set_xscale("log")
    ax[0,1].set_xscale("log")
    ax[1,0].set_xscale("log")
    ax[1,1].set_xscale("log")
    ax[0,0].set_yscale("log")
    ax[0,1].set_yscale("log")
    ax[1,0].set_yscale("log")
    ax[1,1].set_yscale("log")
    ax[0,0].tick_params(direction="in", which='both', labelsize=20)
    ax[0,1].tick_params(direction="in", which='both', labelsize=20)
    ax[1,0].tick_params(direction="in", which='both', labelsize=20)
    ax[1,1].tick_params(direction="in", which='both', labelsize=20)
    legend_properties = {}
    legend = ax[0,0].legend(loc='upper left', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    legend_properties = {}
    legend = ax[0,1].legend(loc='upper left', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    legend_properties = {}
    legend = ax[1,0].legend(loc='upper left', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    legend_properties = {}
    legend = ax[1,1].legend(loc='upper left', fontsize=10, prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)
    plt.tight_layout()
    plt.savefig("iact_comp_effort.pdf")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10],
         sys.argv[11], sys.argv[12])