import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')


def main(active_particle_index_data_p_symm, active_particle_index_data_p_asymm, positions_data_p_symm, positions_data_p_asymm, cutoff):

    cutoff = int(cutoff)
    start = 0
    
    active_particle_index_p_symm = np.load(active_particle_index_data_p_symm)
    active_particle_index_p_asymm = np.load(active_particle_index_data_p_asymm)

    positions_p_symm = np.load(positions_data_p_symm)
    positions_p_asymm = np.load(positions_data_p_asymm)

    N_t = len(positions_p_symm[0,:])

    mean_squared_positions_p_symm = np.mean(np.square(positions_p_symm), axis=1)
    mean_squared_positions_p_asymm = np.mean(np.square(positions_p_asymm), axis =1)


    fig, ax = plt.subplots(2,2, sharey = "row", sharex=True, figsize = (12,6))
    ax[0,0].plot(np.arange(len(mean_squared_positions_p_symm))[start:cutoff], mean_squared_positions_p_symm[start:cutoff], color="#c11f70", label ="Symmetric")
    ax[0,1].plot(np.arange(len(mean_squared_positions_p_asymm))[start:cutoff], mean_squared_positions_p_asymm[start:cutoff], color="#f0bf42", label ="Asymmetric")
    ax[1,0].scatter(np.arange(len(active_particle_index_p_symm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#c11f70", label ="Symmetric", s=1.0)
    ax[1,1].scatter(np.arange(len(active_particle_index_p_asymm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#f0bf42", label ="Asymmetric", s=1.0)


    inset_ax_10 = inset_axes(ax[1,0], width="30%", height="50%", loc="lower right", borderpad=2)
    inset_ax_10.plot(np.arange(len(active_particle_index_p_symm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#c11f70")
    inset_ax_10.scatter(np.arange(len(active_particle_index_p_symm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#c11f70", label =r"$x=100$", s=5.0)
    inset_ax_10.set_xlim(0,25)

    inset_ax_11 = inset_axes(ax[1,1], width="30%", height="50%", loc="lower right", borderpad=2)
    inset_ax_11.plot(np.arange(len(active_particle_index_p_asymm))[:cutoff], active_particle_index_p_asymm[:cutoff], color="#f0bf42")
    inset_ax_11.scatter(np.arange(len(active_particle_index_p_asymm))[:cutoff], active_particle_index_p_asymm[:cutoff], color="#f0bf42", label =r"$x=\pm100$", s=5.0)
    inset_ax_11.set_xlim(0,25)
    #inset_ax_11.set_ylim(3,7)


    ax[1,0].set_ylabel("Active particle index", fontsize =15, weight='bold')
    ax[0,0].set_ylabel(r"$X^2$", fontsize =25, weight='bold')

    ax[1,0].set_xlabel("Event index", fontsize =17, weight='bold')
    ax[1,1].set_xlabel("Event index", fontsize =17, weight='bold')

    ax[0,0].tick_params(axis="x", direction="in", left="off",labelleft="off")
    ax[0,1].tick_params(axis="x",direction="in", left="off",labelleft="off")
    ax[1,0].tick_params(axis="x",direction="in", left="off",labelleft="off")
    ax[1,1].tick_params(axis="x",direction="in", left="off",labelleft="off")

    ax[0,0].tick_params(axis="y", direction="in", left="off",labelleft="off")
    ax[0,1].tick_params(axis="y",direction="in", left="off",labelleft=False)
    ax[1,0].tick_params(axis="y",direction="in", left="off",labelleft="off")
    ax[1,1].tick_params(axis="y",direction="in", left="off",labelleft="off")


    inset_ax_10.tick_params(direction="in")
    inset_ax_11.tick_params(direction="in")

    legend_properties = {'weight':'bold', 'size': 15}
    legend = ax[0,0].legend(loc='lower left', prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    legend = ax[0,1].legend(loc='lower left', prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    #ax[1,0].legend()
    #ax[1,1].legend()

    for tick in ax[0,0].get_xticklabels():
        tick.set_fontweight('bold') 
    for tick in ax[0,0].get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax[0,1].get_xticklabels():
        tick.set_fontweight('bold') 
    for tick in ax[0,1].get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax[1,0].get_xticklabels():
        tick.set_fontweight('bold') 
    for tick in ax[1,0].get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax[1,1].get_xticklabels():
        tick.set_fontweight('bold') 
    for tick in ax[1,1].get_yticklabels():
        tick.set_fontweight('bold')
    for tick in inset_ax_10.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in inset_ax_10.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in inset_ax_11.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in inset_ax_11.get_yticklabels():
        tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig("active_particle_thermalisation_both.pdf")
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