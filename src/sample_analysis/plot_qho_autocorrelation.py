import numpy as np
import matplotlib.pyplot as plt
import sys

def main(iact_data_path, acf_data_path):

    iact_timestep = np.load(iact_data_path)
    acf = np.load(acf_data_path)

    iact_data = iact_timestep[:, 0] 
    timestep_data = iact_timestep[:, 1]

        
    fig, ax = plt.subplots(1, 1)
    ax.scatter(timestep_data, iact_data)
    ax.set_xlabel("delta tau")
    ax.set_ylabel("iact")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig("iact_metropolis.png")
    plt.clf()

    for index, timestep in enumerate(timestep_data):
        print(timestep)
        acf_data = acf[index, :]
        acf_data = np.trim_zeros(acf_data, 'b')

        fig, ax = plt.subplots(1, 1)
        plt.plot(np.arange(0, len(acf_data)), acf_data)
        plt.xlabel("sample index")
        plt.ylabel("autocorrelation function")
        plt.title(f"Autocorrelation Function for Metropolis MC, delta tau = {timestep}")
        timestep_str = str(timestep).replace(".", "")
        plt.savefig(f"acf_metropolis_{timestep_str}.png")
        plt.clf()



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])