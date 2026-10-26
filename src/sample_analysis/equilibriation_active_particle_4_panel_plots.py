import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.use('Agg')


def main(active_particle_index_data_p_symm, active_particle_index_data_p_asymm, positions_data_p_symm, positions_data_p_asymm, metrop_data_p, cutoff):

    cutoff = int(cutoff)
    start = 0
    
    active_particle_index_p_symm = np.load(active_particle_index_data_p_symm)
    active_particle_index_p_asymm = np.load(active_particle_index_data_p_asymm)

    positions_p_symm = np.load(positions_data_p_symm)
    positions_p_asymm = np.load(positions_data_p_asymm)

    metrop_p = np.load(metrop_data_p)

    N_t = len(positions_p_symm[0,:])

    mean_squared_positions_p_symm = np.mean(np.square(positions_p_symm), axis=1)
    mean_squared_positions_p_asymm = np.mean(np.square(positions_p_asymm), axis =1)


    fig, ax = plt.subplots(2,2, sharey = "row", sharex=True, figsize = (12,6))
    ax[0,0].plot(np.arange(len(mean_squared_positions_p_symm))[start:cutoff], mean_squared_positions_p_symm[start:cutoff], color="#c11f70", label ="ECMC - \nSymmetric lifting", linewidth=2.0)
    ax[0,1].plot(np.arange(len(mean_squared_positions_p_asymm))[start:cutoff], mean_squared_positions_p_asymm[start:cutoff], color="#f0bf42", label ="ECMC - \nAsymmetric lifting", linewidth=2.0)
    ax[1,0].scatter(np.arange(len(active_particle_index_p_symm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#c11f70", label ="Symmetric lifting", s=1.0)
    ax[1,1].scatter(np.arange(len(active_particle_index_p_asymm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#f0bf42", label ="Asymmetric lifting", s=1.0)

    inset_ax_00 = inset_axes(ax[0,0], width = "40%", height="50%", loc = "lower right", borderpad = 1.7)
    inset_ax_00.plot(np.arange(len(metrop_p))[start:cutoff], metrop_p[start:cutoff], color = "#2e8316", label = "Metrop MC")
    lims_00 = ax[0,0].get_ylim()
    print(lims_00)
    inset_ax_00.set_ylim(lims_00[0], lims_00[1])

    inset_ax_10 = inset_axes(ax[1,0], width="30%", height="50%", loc="lower right", borderpad=2.5)
    inset_ax_10.plot(np.arange(len(active_particle_index_p_symm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#c11f70")
    inset_ax_10.scatter(np.arange(len(active_particle_index_p_symm))[:cutoff], active_particle_index_p_symm[:cutoff], color="#c11f70", label =r"$x=100$", s=5.0)
    inset_ax_10.set_xlim(0,25)

    inset_ax_11 = inset_axes(ax[1,1], width="30%", height="50%", loc="lower right", borderpad=2.5)
    inset_ax_11.plot(np.arange(len(active_particle_index_p_asymm))[:cutoff], active_particle_index_p_asymm[:cutoff], color="#f0bf42")
    inset_ax_11.scatter(np.arange(len(active_particle_index_p_asymm))[:cutoff], active_particle_index_p_asymm[:cutoff], color="#f0bf42", label =r"$x=\pm100$", s=5.0)
    inset_ax_11.set_xlim(0,25)
    #inset_ax_11.set_ylim(3,7)


    ax[1,0].set_ylabel("Active particle index", fontsize =19)
    ax[0,0].set_ylabel(r"$|\mathbf{x}|^2$", fontsize =30, labelpad=-5)

    ax[1,0].set_xlabel("Event index", fontsize =20)
    ax[1,1].set_xlabel("Event index", fontsize =20)

    ax[0,0].tick_params(axis="x", direction="in", left=False,labelleft=False, labelsize=16)
    ax[0,1].tick_params(axis="x",direction="in", left=True,labelleft=False, labelsize=16)
    ax[1,0].tick_params(axis="x",direction="in", left=False,labelleft=False, labelsize=16)
    ax[1,1].tick_params(axis="x",direction="in", left=False,labelleft=False, labelsize=16)

    ax[0,0].tick_params(axis="y", direction="in", left=True,labelleft=True, labelsize=18)
    ax[0,1].tick_params(axis="y",direction="in", left=True,labelleft=False, labelsize=18)
    ax[1,0].tick_params(axis="y",direction="in", left=True,labelleft=True, labelsize=18)
    ax[1,1].tick_params(axis="y",direction="in", left=True,labelleft=False, labelsize=18)

    inset_ax_00.tick_params(direction="in", labelsize=15)
    inset_ax_10.tick_params(direction="in", labelsize=16)
    inset_ax_11.tick_params(direction="in", labelsize=16)

    legend_properties = {'size': 14}
    handles, labels = ax[0,0].get_legend_handles_labels()
    inset_handles, inset_labels = inset_ax_00.get_legend_handles_labels()
    legend = ax[0,0].legend(handles+inset_handles, labels+inset_labels, loc='lower left', prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    legend = ax[0,1].legend(loc='upper right', prop=legend_properties)
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_lw(1.5)

    plt.setp(ax[0,0].spines.values(), lw=2.0, color='black')
    plt.setp(ax[0,1].spines.values(), lw=2.0, color='black')
    plt.setp(ax[1,0].spines.values(), lw=2.0, color='black')
    plt.setp(ax[1,1].spines.values(), lw=2.0, color='black')
    plt.setp(inset_ax_00.spines.values(), lw=1.5, color='black')
    plt.setp(inset_ax_00.spines.values(), lw=1.5, color='black')
    plt.setp(inset_ax_11.spines.values(), lw=1.5, color='black')
 

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
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])