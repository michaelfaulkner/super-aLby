from matplotlib.lines import Line2D
from get_actual_figure_data import get_actual_figure_data
import matplotlib.pyplot as plt
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../../")
sys.path.insert(0, src_directory)

output_file = "output/optimal_sampling_strategies_figs/plots/index_velocity_pressure.pdf"


def main():
    os.chdir(src_directory)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    for tag, color in (("p2_ghc_beta1_N16_L32", 'firebrick'), ("p2_ghc_beta0p5_N16_L32", 'orange'),
                       ("sd_kappa2_beta1_N8_L8", 'darkblue'), ("p4_ghc_beta1_N8_L16", 'darkgreen')):
        x, y = get_actual_figure_data("fig7", f"idx_space_velocity_over_v0_{tag}_vs_b_over_bstar")
        ax.plot(x, y, color=color, marker='8', markersize=8, linestyle='None', alpha=0.8)
        x, y = get_actual_figure_data("fig7", f"beta_pressure_{tag}_vs_b_over_bstar")
        ax.plot(x, y, color=color, linestyle='-', linewidth=2.5, alpha=0.8)

    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$b/b^*_P$', fontsize=20, labelpad=3)
    ax.set_ylabel(r'$X / (1/r_0)$', fontsize=20, labelpad=3)

    ax.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=18)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.tick_params(axis='y', which='major', pad=3)
    ax.tick_params(axis='both', which='minor', direction='in', width=3, length=4)

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    marker_handles = [
        Line2D([0], [0], color='grey', marker='8', linestyle='None', markersize=8, label=r'$X =  v_{\rm idx} / v_0 $'),
        Line2D([0], [0], color='grey', linestyle='-', linewidth=2.5, label=r'$X = \beta P$')
    ]
    leg_vars = ax.legend(handles=marker_handles, fontsize=11, framealpha=1.0, facecolor='white', edgecolor='black',
                         loc='upper left')
    leg_vars.get_frame().set_linewidth(3)
    ax.add_artist(leg_vars)

    color_handles = [
        Line2D([0], [0], color='firebrick', linestyle='-', linewidth=2.5, label=r'p=2 GHC, $\beta = 1.0$'),
        Line2D([0], [0], color='orange', linestyle='-', linewidth=2.5, label=r'p=2 GHC, $\beta = 0.5$'),
        Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=2.5, label=r'p=4 GHC, $\beta = 1.0$'),
        Line2D([0], [0], color='darkblue', linestyle='-', linewidth=2.5, label=r'$\kappa$=2 SD, $\beta = 1.0$')
    ]
    leg_colors = ax.legend(handles=color_handles, fontsize=11, framealpha=1.0, facecolor='white', edgecolor='black',
                           loc='lower right', bbox_to_anchor=(0.8, 0.0))
    leg_colors.get_frame().set_linewidth(3)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.95)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300)
    plt.show()

    print('Plot saved.')


if __name__ == '__main__':
    main()
