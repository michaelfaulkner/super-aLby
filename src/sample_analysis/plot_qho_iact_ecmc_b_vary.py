import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def main(iact_data_path, N):

    N = int(N)
    b_data = np.load(os.path.join(iact_data_path, "iact_ecmc_0.npy"))[:, 1]
    sorted_b = b_data[np.argsort(b_data)]
    storage_arr = np.zeros((len(b_data), N))

    for index in range(N):
        iact_b =  np.load(os.path.join(iact_data_path, f"iact_ecmc_{index}.npy"))
        iact_data = iact_b[:, 0] 
        b_data = iact_b[:, 1] 
        argsorted_data = np.argsort(b_data)
        b_argsorted = b_data[argsorted_data]
        iact_data = iact_data[argsorted_data] 
        storage_arr[:, index] = iact_data
    
    #print(storage_arr)
    min = 0.01
    fit_index = -2
    start_fit = -7

    sorted_b = np.trim_zeros(sorted_b, trim="f")
    iact_mean_arr = np.mean(storage_arr[np.nonzero(sorted_b >= min)], axis = 1)
   

    err = np.std(storage_arr[np.nonzero(sorted_b >= min)], axis=1)
    sorted_b = sorted_b[np.nonzero(sorted_b >= min)]
 
    # e_coeffs = np.polyfit(np.log(sorted_b[start_fit:]), np.log(iact_mean_arr[start_fit:]), deg=1)
    # fitted_e = e_coeffs[1] + np.multiply(np.log(sorted_b[start_fit:]), e_coeffs[0])
    # print(e_coeffs)

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(sorted_b, iact_mean_arr, err, fmt='o', capsize=3, markersize=3.5, color="#ed1171ff")
    #ax.plot(sorted_b[start_fit:], np.exp(fitted_e), color="#d97dd9ff")

    ax.set_xlabel(r"$b$", fontsize=20)#, labelpad=-10)
    ax.set_ylabel("IACT", fontsize=15, labelpad=0)
    ax.set_title(r"IACT for $\langle x^2 \rangle$, $\tau=$40")
    ax.set_xlim(-1.0, 22.0)
    #ax.set_xscale("log")
    #ax.set_yscale("log")
    #ax.annotate(f"Fit Coefficient: {e_coeffs[0]:.3f}", xy=(np.median(sorted_b), np.median(iact_mean_arr)-0.1*(np.max(iact_mean_arr)-np.min(iact_mean_arr))))
    #print(ax.get_ylim())
    #ax.set_ylim(0, 1.5e1)
    #ax.set_xlim(38, 13000)
    plt.tight_layout()
    plt.savefig("iact_b_vary_40.pdf")
    plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])