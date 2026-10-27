from get_actual_figure_data import get_actual_figure_data
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../../")
sys.path.insert(0, src_directory)

output_file = "output/optimal_sampling_strategies_figs/plots/mixing_time_scaling.pdf"


def main():
    os.chdir(src_directory)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    N_arr, b0_mt = get_actual_figure_data("fig8", "mixing_time_in_events_b_zero_vs_N")
    _, b_opt_mt = get_actual_figure_data("fig8", "mixing_time_in_events_b_star_vs_N")

    n_mid_start = int(len(N_arr) * 0.20)
    n_mid_end = int(len(N_arr) * 0.80)
    mid_N_arr = N_arr[n_mid_start:n_mid_end]

    ax.plot(N_arr, b0_mt, color='darkblue', marker='8', markersize=10, linestyle='None', alpha=0.8, label=r'$b=0$')

    valid_idx_b0 = np.where(~np.isnan(b0_mt))[0]
    if len(valid_idx_b0) > 0:
        idx = max(valid_idx_b0[0], n_mid_start)
        prefactor_n25 = (b0_mt[idx] / (N_arr[idx] ** 2.5)) * 1.5

        y_line_b0 = prefactor_n25 * mid_N_arr**2.5
        ax.plot(mid_N_arr, y_line_b0, color='darkblue', linestyle=':', linewidth=3.75, alpha=0.8)

        ax.text(mid_N_arr[-1] * 0.58, y_line_b0[-1] - 800, r'$N^{5/2}$',
                color='darkblue', fontsize=22, verticalalignment='center')

    ax.plot(N_arr, b_opt_mt, color='firebrick', marker='8', markersize=10, linestyle='None', alpha=0.8,
            label=r'$b=b^*_P$')

    valid_idx_bopt = np.where(~np.isnan(b_opt_mt))[0]
    if len(valid_idx_bopt) > 0:
        idx = max(valid_idx_bopt[0], n_mid_start)
        prefactor_n2 = (b_opt_mt[idx] / (N_arr[idx] ** 2)) * 0.6

        y_line_bopt = prefactor_n2 * mid_N_arr**2
        ax.plot(mid_N_arr, y_line_bopt, color='firebrick', linestyle=':', linewidth=3.75, alpha=0.8)

        ax.text(mid_N_arr[-1] * 0.63, y_line_bopt[-1] - 60, r'$N^2$',
                color='firebrick', fontsize=22, verticalalignment='center')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(alpha=0.5, linestyle='-', which='major')

    ax.set_xlabel(r'$N$', fontsize=20, labelpad=3)
    ax.set_ylabel(r'$\tau_{\rm mix} / (n_{\rm events} / (\sqrt{\beta k} v_0))$', fontsize=20, labelpad=3)

    ax.set_xticks([10, 20])
    ax.set_xticklabels(['10', '20'])

    ax.set_xticks([], minor=True)

    ax.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=18)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.tick_params(axis='y', which='major', pad=3)
    ax.tick_params(axis='both', which='minor', direction='in', width=3, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    leg = ax.legend(fontsize=18, loc='lower right', framealpha=1.0, edgecolor='black')
    leg.get_frame().set_linewidth(3)

    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.95)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.show()

    print('Plot saved.')


if __name__ == '__main__':
    main()
