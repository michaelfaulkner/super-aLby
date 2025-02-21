import numpy as np
import matplotlib.pyplot as plt
import sys

def main(iact_data_path):

    iact_timestep = np.load(iact_data_path)

    iact_data = iact_timestep[:, 0] 
    timestep_data = iact_timestep[:, 1]

        
    fig, ax = plt.subplots(1, 1)
    ax.scatter(timestep_data, iact_data)
    ax.set_xlabel("delta tau")
    ax.set_ylabel("iact")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig("iact.png")

if __name__ == '__main__':
    main(sys.argv[1])