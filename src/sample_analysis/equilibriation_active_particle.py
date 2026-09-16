import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('Agg')


def main(active_particle_index_data_rand, active_particle_index_data_pm, active_particle_index_data_p, cutoff):

    cutoff = int(cutoff)
    
    active_particle_index_rand = np.load(active_particle_index_data_rand)
    active_particle_index_pm = np.load(active_particle_index_data_pm)
    active_particle_index_p = np.load(active_particle_index_data_p)


    fig, ax = plt.subplots(1,3, sharey=True, sharex=True)

    ax[0].scatter(np.arange(len(active_particle_index_rand))[:cutoff], active_particle_index_rand[:cutoff], color="#b363d6", label ="Random", s=1.0)
    ax[1].scatter(np.arange(len(active_particle_index_p))[:cutoff], active_particle_index_p[:cutoff], color="#bf1b4f", label =r"$x=100$", s=1.0)
    ax[2].scatter(np.arange(len(active_particle_index_pm))[:cutoff], active_particle_index_pm[:cutoff], color="#9fe379", label =r"$x=\pm 100$", s=1.0)

    ax[0].set_ylabel("Active particle index")
    ax[1].set_xlabel("Event index")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()


    plt.savefig("active_particle_thermalisation.pdf")




if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])