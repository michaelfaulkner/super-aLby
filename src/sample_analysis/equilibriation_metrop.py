import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')


def main(positions_data_pm, positions_data_p, cutoff):

    cutoff = int(cutoff)
    start = 0


    positions_pm = np.load(positions_data_pm)
    positions_p = np.load(positions_data_p)

    print(np.shape(positions_p))
    N_t = len(positions_p[0,:])

    mean_squared_positions_pm = np.mean(np.square(positions_pm), axis=1)
    mean_squared_positions_p = np.mean(np.square(positions_p), axis =1)


    fig, ax = plt.subplots(1,1, sharey = "row", sharex=True)
    ax.plot(8*np.arange(len(mean_squared_positions_p))[np.arange(len(mean_squared_positions_p))<cutoff],
            mean_squared_positions_p[np.arange(len(mean_squared_positions_p))<cutoff], color="#c11f70", label =r"$x=100$")
    #ax[1].plot(np.arange(len(mean_squared_positions_pm))[np.arange(len(mean_squared_positions_pm))<cutoff], mean_squared_positions_pm[np.arange(len(mean_squared_positions_pm))<cutoff], color="#9fe379", label =r"$x=\pm 100$")

    ax.set_ylabel(r"$X^2$", fontsize =20, weight='bold')

   # ax[1].set_xlabel("Sample index", fontsize =15, weight='bold')
    ax.set_xlabel("Event index", fontsize =15, weight='bold')

    ax.tick_params(axis="x", direction="in", left="off",labelleft="off", labelsize=15)
   # ax[1].tick_params(axis="x",direction="in", left="off",labelleft="off")

    ax.tick_params(axis="y", direction="in", left="off",labelleft="off", labelsize=15)
   # ax[1].tick_params(axis="y",direction="in", left="off",labelleft=False)
 
    # legend_properties = {'weight':'bold', 'size': 15}
    # legend = ax[0].legend(loc='lower right', prop=legend_properties)
    # legend.get_frame().set_edgecolor('k')
    # legend.get_frame().set_lw(1.5)

    # legend = ax[1].legend(loc='lower right', prop=legend_properties)
    # legend.get_frame().set_edgecolor('k')
    # legend.get_frame().set_lw(1.5)

    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold') 
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

        ax.set_ylim(0.0, 10500)


    # for tick in ax[1].get_xticklabels():
    #     tick.set_fontweight('bold') 
    # for tick in ax[1].get_yticklabels():
    #     tick.set_fontweight('bold')

    plt.tight_layout()


    plt.savefig("thermalisation_metrop.pdf")
    plt.clf()

    print(np.shape(positions_p))
    fig, ax = plt.subplots(2,2, sharex =True, sharey=True)
    ax[0,0].scatter(np.arange(len(positions_p[0,:])), positions_p[0,:], color="#bf1b4f", label =r"$x=100$")
    ax[1,0].scatter(np.arange(len(positions_pm[0,:])), positions_pm[0,:], color="#9fe379", label =r"$x=\pm100$")
    ax[0,1].scatter(np.arange(len(positions_p[10000,:])), positions_p[10000,:], color="#bf1b4f", label =r"$x=100$",)
    ax[1,1].scatter(np.arange(len(positions_pm[10000,:])), positions_pm[10000,:], color="#9fe379", label =r"$x=\pm100$")
    plt.savefig("metrop_positions.pdf")







if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])