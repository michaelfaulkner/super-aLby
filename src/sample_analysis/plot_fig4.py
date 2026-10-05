from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, LogLocator, NullLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)

output_file = "output/paper_data/plots/comp_effort_vs_b_soft_disks.pdf"
data_directory = "output/paper_data/fig4"


def load_data_file(file_name):
    return np.load(os.path.join(data_directory, file_name))


def main():
    os.chdir(src_directory)

    _, ax = plt.subplots()
    Ns_er = np.array([8, 16, 32, 48, 64, 96])
    Ns_inset = np.array([16, 32, 48, 64, 96])
    colors = ['teal', 'rebeccapurple', 'darkgreen', 'palevioletred', 'burlywood', 'cornflowerblue', 'tan']

    for i, N in enumerate(Ns_er):
        data = load_data_file(f'soft_disk_paper_N_{N}_L_{N}_comp_effort_sweep.npy')
        b, _, idx = load_data_file(
            f'soft_disk_paper_N_{N}_L_{N}_index_and_state_space_velocity_factor_field_prefactor_sweep.npy')
        errors = load_data_file(f'soft_disk_paper_N_{N}_std_comp_effort_vs_b.npy')[1]
        mask = ~np.isnan(idx)
        b, idx = b[mask], idx[mask]
        m, c = np.polyfit(b, idx, 1)
        b_star = -c / m
        if N == 96:
            data = data[:, :-2]
            errors = errors[:-2]
        y_data = N * data[1]
        errors = N * errors
        y_min = min(y_data)
        y_data_norm = y_data / y_min
        errors_norm = errors / y_min
        ax.errorbar(data[0] / b_star, y_data_norm, errors_norm, capsize=3, zorder=3, marker='o', markersize='7',
                    alpha=0.7, color=colors[i])

    min_efforts_inset = []
    b0_efforts_inset = []
    for N in Ns_inset:
        data_inset = load_data_file(f'soft_disk_paper_N_{N}_L_{N}_comp_effort_sweep.npy')
        min_efforts_inset.append(np.min(data_inset[1]) * N)
        b0_efforts_inset.append(data_inset[1][0] * N)

    custom_handles = [
        Line2D([0], [0], color=colors[0], linestyle='-', marker='o', lw=2.5, label='N = 8'),
        Line2D([0], [0], color=colors[1], linestyle='-', marker='o', lw=2.5, label='N = 16'),
        Line2D([0], [0], color=colors[2], linestyle='-', marker='o', lw=2.5, label='N = 32'),
        Line2D([0], [0], color=colors[3], linestyle='-', marker='o', lw=2.5, label='N = 48'),
        Line2D([0], [0], color=colors[4], linestyle='-', marker='o', lw=2.5, label='N = 64'),
        Line2D([0], [0], color=colors[5], linestyle='-', marker='o', lw=2.5, label='N = 96')
    ]

    leg = ax.legend(handles=custom_handles, fontsize=9, framealpha=1.0, facecolor='white', edgecolor='black',
                    loc='lower right')
    leg.get_frame().set_linewidth(3)
    leg.get_frame().set_edgecolor("k")

    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$b/b^*_P$', fontsize=20, labelpad=3)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\Omega/\Omega(b^*)$', fontsize=20, labelpad=1)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=18)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.tick_params(axis='y', which='major', pad=3)
    ax.tick_params(axis='both', which='minor', direction='in', width=3, length=4)
    ax.set_xlim(-0.1, 2.1)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    axins = ax.inset_axes([0.31, 0.59, 0.38, 0.38])

    axins.plot(Ns_inset, min_efforts_inset, marker='s', color='black', linestyle='-', alpha=0.8, label=r'$b=b^*$')
    axins.plot(Ns_inset, b0_efforts_inset, marker='s', color='dimgray', linestyle='-', alpha=0.8, label=r'$b=0$')

    N_line = np.array([22.6, 45.2])

    prefactor_star = (min_efforts_inset[1] / (32 ** 1.5)) * 0.6
    prefactor_b0 = (b0_efforts_inset[1] / (32 ** 2.5)) * 0.6

    scaling_star = prefactor_star * (N_line ** 1.5)
    scaling_b0 = prefactor_b0 * (N_line ** 2.5)

    axins.plot(N_line, scaling_star, color='black', linestyle=':', linewidth=2)
    axins.plot(N_line, scaling_b0, color='dimgray', linestyle=':', linewidth=2)

    axins.text(N_line[-1] * 1.1, scaling_star[-1], r'$N^{3/2}$', fontsize=12, va='center', color='black')
    axins.text(N_line[-1] * 1.1, scaling_b0[-1], r'$N^{5/2}$', fontsize=12, va='center', color='dimgray')

    axins.set_xscale('log')
    axins.set_yscale('log')

    ticks = [20, 50]
    axins.set_xticks(ticks)
    axins.set_xticklabels([str(t) for t in ticks])
    axins.xaxis.set_minor_formatter(NullFormatter())
    axins.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=10))
    axins.yaxis.set_minor_locator(NullLocator())

    axins.set_xlabel(r'$N$', fontsize=16, labelpad=3)
    axins.set_ylabel(r'$\Omega$', fontsize=16, labelpad=3)
    axins.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=14)
    axins.tick_params(axis='x', which='major', pad=2)
    axins.tick_params(axis='y', which='major', pad=3)
    axins.tick_params(axis='both', which='minor', direction='in', width=3, length=4)
    axins.grid(alpha=0.5)

    leg_inset = axins.legend(loc='upper left', fontsize=9)
    leg_inset.get_frame().set_linewidth(3)
    leg_inset.get_frame().set_edgecolor("k")

    for spine in axins.spines.values():
        spine.set_linewidth(3)

    ax.set_ylim(top=2.28*10**2)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print('Plot saved.')


if __name__ == '__main__':
    main()
