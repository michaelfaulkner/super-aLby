import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')


def main(active_particle_index_data_pm, active_particle_index_data_p, positions_data_pm, positions_data_p, cutoff):

    cutoff = int(cutoff)
    start = 0
    
    active_particle_index_pm = np.load(active_particle_index_data_pm)
    active_particle_index_p = np.load(active_particle_index_data_p)

    positions_pm = np.load(positions_data_pm)
    positions_p = np.load(positions_data_p)

    print(np.shape(positions_p))
    N_t = len(positions_p[0,:])

    mean_squared_positions_pm = np.mean(np.square(positions_pm), axis=1)
    mean_squared_positions_p = np.mean(np.square(positions_p), axis =1)


    fig, ax = plt.subplots(2,2, sharey = "row", sharex=True)
    ax[0,0].plot(np.arange(len(mean_squared_positions_p))[start:cutoff], mean_squared_positions_p[start:cutoff], color="#bf1b4f", label =r"$x=100$")
    ax[0,1].plot(np.arange(len(mean_squared_positions_pm))[start:cutoff], mean_squared_positions_pm[start:cutoff], color="#9fe379", label =r"$x=\pm 100$")
    ax[1,0].scatter(np.arange(len(active_particle_index_p))[:cutoff], active_particle_index_p[:cutoff], color="#bf1b4f", label =r"$x=100$", s=1.0)
    ax[1,1].scatter(np.arange(len(active_particle_index_pm))[:cutoff], active_particle_index_pm[:cutoff], color="#9fe379", label =r"$x=\pm 100$", s=1.0)

    ax[1,0].set_ylabel("Active particle index", fontsize =12)
    ax[0,0].set_ylabel(r"$<x^2>$", fontsize =15)

    ax[1,0].set_xlabel("Event index", fontsize =12)
    ax[1,1].set_xlabel("Event index", fontsize =12)

    ax[0,0].tick_params(axis="x", direction="in", left="off",labelleft="off")
    ax[0,1].tick_params(axis="x",direction="in", left="off",labelleft="off")
    ax[1,0].tick_params(axis="x",direction="in", left="off",labelleft="off")
    ax[1,1].tick_params(axis="x",direction="in", left="off",labelleft="off")

    ax[0,0].tick_params(axis="y", direction="in", left="off",labelleft="off")
    ax[0,1].tick_params(axis="y",direction="in", left="off",labelleft=False)
    ax[1,0].tick_params(axis="y",direction="in", left="off",labelleft="off")
    ax[1,1].tick_params(axis="y",direction="in", left="off",labelleft="off")


    ax[0,0].legend()
    ax[0,1].legend()
    #ax[1,0].legend()
    #ax[1,1].legend()



    plt.savefig("active_particle_thermalisation_symm.pdf")
    plt.clf()

    # print(np.shape(positions_p))
    # fig, ax = plt.subplots(2,2, sharex =True, sharey=True)
    # ax[0,0].scatter(np.arange(len(positions_p[0,:])), positions_p[0,:], color="#bf1b4f", label =r"$x=100$")
    # ax[1,0].scatter(np.arange(len(positions_pm[0,:])), positions_pm[0,:], color="#9fe379", label =r"$x=\pm100$")
    # ax[0,1].scatter(np.arange(len(positions_p[500,:])), positions_p[500,:], color="#bf1b4f", label =r"$x=100$",)
    # ax[1,1].scatter(np.arange(len(positions_pm[500,:])), positions_pm[500,:], color="#9fe379", label =r"$x=\pm100$")
    # plt.savefig("ap_positions.pdf")







if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])