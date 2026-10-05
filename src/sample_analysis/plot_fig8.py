import matplotlib.pyplot as plt
import numpy as np
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)

output_file = "output/paper_data/plots/mixing_time_scaling.pdf"
data_directory = "output/paper_data/fig8"


def load_data_file(file_name):
    return np.load(os.path.join(data_directory, file_name))


def main():
    os.chdir(src_directory)

    _, ax = plt.subplots(figsize=(8, 6))

    data = load_data_file('paper_mixing_times_mixing_time_results.npy')

    N_arr = data[0][2:]
    b0_mt = data[1][2:]
    b_opt_mt = data[2][2:]

    n_mid_start = int(len(N_arr) * 0.20)
    n_mid_end = int(len(N_arr) * 0.80)
    mid_N_arr = N_arr[n_mid_start:n_mid_end]

    ax.plot(N_arr, b0_mt, color='darkblue', marker='8', markersize=8, linestyle='None', alpha=0.8, label=r'$b=0$')

    valid_idx_b0 = np.where(~np.isnan(b0_mt))[0]
    if len(valid_idx_b0) > 0:
        idx = max(valid_idx_b0[0], n_mid_start)
        prefactor_n25 = (b0_mt[idx] / (N_arr[idx] ** 2.5)) * 2.5

        y_line_b0 = prefactor_n25 * mid_N_arr**2.5
        ax.plot(mid_N_arr, y_line_b0, color='darkblue', linestyle=':', linewidth=2.5, alpha=0.8)

        ax.text(mid_N_arr[-1] * 0.58, y_line_b0[-1] - 300, r'$N^{5/2}$',
                color='darkblue', fontsize=18, verticalalignment='center')

    ax.plot(N_arr, b_opt_mt, color='firebrick', marker='8', markersize=8, linestyle='None', alpha=0.8,
            label=r'$b=b^*_P$')

    valid_idx_bopt = np.where(~np.isnan(b_opt_mt))[0]
    if len(valid_idx_bopt) > 0:
        idx = max(valid_idx_bopt[0], n_mid_start)
        prefactor_n2 = (b_opt_mt[idx] / (N_arr[idx] ** 2)) * 0.6

        y_line_bopt = prefactor_n2 * mid_N_arr**2
        ax.plot(mid_N_arr, y_line_bopt, color='firebrick', linestyle=':', linewidth=2.5, alpha=0.8)

        ax.text(mid_N_arr[-1] * 1.08, y_line_bopt[-1], r'$N^2$',
                color='firebrick', fontsize=18, verticalalignment='center')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(alpha=0.5, linestyle='-', which='major')

    ax.set_xlabel(r'$N$', fontsize=20, labelpad=3)
    ax.set_ylabel(r'Mixing Time', fontsize=20, labelpad=3)

    ax.set_xticks([10, 20])
    ax.set_xticklabels(['10', '20'])

    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=18)
    ax.tick_params(axis='x', which='major', pad=6)
    ax.tick_params(axis='y', which='major', pad=3)
    ax.tick_params(axis='both', which='minor', direction='in', width=3, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.legend(fontsize=14, loc='lower right', framealpha=1.0, edgecolor='black')

    plt.subplots_adjust(right=0.85)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print('Plot saved.')


if __name__ == '__main__':
    main()
