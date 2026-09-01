import numpy as np
import sys
import matplotlib.pyplot as plt 

def main(filepath, proper_time):

    proper_time = float(proper_time)
    data = np.loadtxt(filepath)


    timestep = data[:, 1]
    number_of_particles = proper_time / timestep
    mean_events = data[:, 2]

    

    fig, ax = plt.subplots(1,1)
    ax.scatter(number_of_particles, mean_events, color="#61429e")
    ax.set_xlabel(r"$N_{\tau}$", fontsize=20, labelpad=-10, weight = "bold")
    ax.set_ylabel("No. of Events", fontsize=15, weight = "bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig("events.png")






if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2]) 