import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')

def main(ecmc_rmsd_data_path, propertime):

    propertime = float(propertime)

    rmsd_data = np.load(ecmc_rmsd_data_path)
    event_index = np.arange(len(rmsd_data))

    cutoff = 5000
    


    #rmsd_fit_trim = -7
    #rmsd_coeffs = np.polyfit(np.log(event_index[:rmsd_fit_trim]), np.log(rmsd_data[:rmsd_fit_trim]), deg=1)
    #fitted_rmsd = rmsd_coeffs[1] + np.multiply(np.log(event_index[:rmsd_fit_trim]), rmsd_coeffs[0])
   # print(f"rmsd scaling: {rmsd_coeffs[0]}")

    

    fig, ax = plt.subplots(1, 1)

    #ax.plot(event_index[:rmsd_fit_trim], np.exp(fitted_rmsd), color="#d97dd9ff")
    ax.scatter(event_index[:cutoff], rmsd_data[:cutoff], marker='o', s=4, color="#e20acdff", label="RMSD")
    


    ax.set_xlabel("MC Event index", fontsize=10, weight = "bold")
    ax.set_ylabel("RMSD", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    legend_properties = {'weight':'bold'}
    #plt.legend(prop=legend_properties)
    #ax.set_ylim(0.17e5, 0.8e7)
   


    #plt.title(f"IACT of x^2 for QHO with (x+2)^2")
    plt.tight_layout()
    plt.savefig("rmsd_vs_events.png")#, transparent=True)
    plt.clf()




if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])