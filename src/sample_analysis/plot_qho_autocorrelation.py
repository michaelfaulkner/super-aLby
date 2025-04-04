import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main(iact_data_path, acf_data_folder, ecmc=True, distance_between_measurements=1):

    iact_timestep = np.load(iact_data_path)
    

    iact_data = iact_timestep[:, 0] 
    timestep_data = iact_timestep[:, 1]
    
        
    fig, ax = plt.subplots(1, 1)
    ax.scatter(timestep_data, iact_data)
    ax.set_xlabel("delta tau")
    ax.set_ylabel("iact")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # if ecmc:
    plt.title(f"Integrated Autocorrelation Time for ECMC, lambda = {distance_between_measurements}")
    plt.savefig(f"iact_ecmc_{distance_between_measurements}.png")
    # else:
    # plt.title(f"Integrated Autocorrelation Time for Metropolis MC")
    # plt.savefig("iact_metropolis_1.png")
    plt.clf()

    for index, timestep in enumerate(timestep_data):
        timestep_str = str(timestep).replace(".", "")
        acf = np.load(os.path.join(acf_data_folder, f"acf_ecmc_{timestep_str}.npy"))
        fig, ax = plt.subplots(1, 1)
        plt.plot(np.arange(0, len(acf)), acf)
        plt.xlabel("sample index")
        plt.ylabel("autocorrelation function")
        

        plt.title(f"Autocorrelation Function for ECMC, delta tau = {timestep}, lambda = {distance_between_measurements}")
        plt.savefig(f"acf_ecmc_{timestep_str}_{distance_between_measurements}.png")
 
        # plt.title(f"Autocorrelation Function for Metropolis MC, delta tau = {timestep}")
        # plt.savefig(f"acf_metropolis_{timestep_str}_1.png")
      

        plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])