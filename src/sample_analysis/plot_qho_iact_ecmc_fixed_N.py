import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(iact_data_path, N):

    N = int(N)
    m_data = np.load(os.path.join(iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    sorted_m = m_data[np.argsort(m_data)]
    storage_arr = np.zeros((len(m_data), N))

    for index in range(N):
        iact_m =  np.load(os.path.join(iact_data_path, f"iact_ecmc_{index}.npy"))
        iact_data = iact_m[:, 0] 
        m_data = iact_m[:, 1] 
        argsorted_data = np.argsort(m_data)
        m_argsorted = m_data[argsorted_data]
        iact_data = iact_data[argsorted_data] 
        storage_arr[:, index] = iact_data
    
    #print(storage_arr)
    min = 0.01
    fit_index = -2
    start_fit = -7

    sorted_m = np.trim_zeros(sorted_m, trim="f")
    iact_mean_arr = np.mean(storage_arr[np.nonzero(sorted_m >= min)], axis = 1)
    print(sorted_m)
    print(iact_mean_arr)

    err = np.std(storage_arr[np.nonzero(sorted_m >= min)], axis=1)
    sorted_m = sorted_m[np.nonzero(sorted_m >= min)]
  
    e_coeffs = np.polyfit(np.log(sorted_m[start_fit:]), np.log(iact_mean_arr[start_fit:]), deg=1)
    fitted_e = e_coeffs[1] + np.multiply(np.log(sorted_m[start_fit:]), e_coeffs[0])
    print(e_coeffs)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(sorted_m, iact_mean_arr, err, fmt='o', capsize=3, markersize=3.5, color="#ed1171ff")
    ax.plot(sorted_m[start_fit:], np.exp(fitted_e), color="#d97dd9ff")

    ax.set_xlabel(r"$m$", fontsize=20, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15, labelpad=0)
    ax.set_title(r"$b=4.0$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.annotate(f"Fit Coefficient: {e_coeffs[0]:.3f}", xy=(np.median(sorted_m), np.median(iact_mean_arr)-0.1*(np.max(iact_mean_arr)-np.min(iact_mean_arr))))
    #print(ax.get_ylim())
    #ax.set_ylim(0, 1.5e1)
    #ax.set_xlim(38, 13000)
    plt.tight_layout()
    plt.savefig("iact_fixed_N_4.pdf")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])