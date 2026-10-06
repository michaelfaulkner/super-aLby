from matplotlib.lines import Line2D
from get_actual_figure_data import get_actual_figure_data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../../")
sys.path.insert(0, src_directory)

output_file = "output/optimal_sampling_strategies_figs/plots/harm_chain_figures_2.pdf"


def apply_main_style(ax, xlabel, ylabel, xpad=0, ypad=0):
    ax.grid(alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=20, labelpad=xpad)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=ypad)
    ax.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=18)
    ax.tick_params(axis='x', which='major', pad=2)
    ax.tick_params(axis='y', which='major', pad=3)
    ax.tick_params(axis='both', which='minor', direction='in', width=3, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(3)


def apply_inset_style(ax_inset, xlabel, ylabel, xpad=0, ypad=0):
    ax_inset.grid(alpha=0.5)
    ax_inset.set_xlabel(xlabel, fontsize=16, labelpad=xpad)
    ax_inset.set_ylabel(ylabel, fontsize=16, labelpad=ypad)
    ax_inset.tick_params(axis='both', which='major', direction='in', width=3, length=5, labelsize=14)
    ax_inset.tick_params(axis='x', which='major', pad=2)
    ax_inset.tick_params(axis='y', which='major', pad=3)
    ax_inset.tick_params(axis='both', which='minor', direction='in', width=3, length=4)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(3)


def style_legend(leg):
    leg.get_frame().set_linewidth(3)
    leg.get_frame().set_edgecolor("k")


def style_inset_legend(leg_inset):
    leg_inset.get_frame().set_linewidth(1.5)
    leg_inset.get_frame().set_edgecolor("k")


def main():
    os.chdir(src_directory)

    fig2, axs2 = plt.subplots(1, 2, figsize=(12.0, 5.0))

    fig2.subplots_adjust(wspace=0.15)

    # ==========================================
    # SUBPLOT 1
    # ==========================================
    ax = axs2[0]
    ps = np.array([2, 4])
    colors_4 = ['mediumblue', 'black']

    cell_horizon = [get_actual_figure_data("fig3", f"a_comp_effort_cell_horizon_p{p}_vs_density") for p in ps]
    four_factor = [get_actual_figure_data("fig3", f"a_comp_effort_four_factor_p{p}_vs_density") for p in ps]
    densities = np.array([x for x, _ in cell_horizon])
    comp_stars = np.array([y for _, y in cell_horizon])
    comp_stars_naive = np.array([y for _, y in four_factor])

    for i, rows in enumerate(zip(densities, comp_stars, ps)):
        dense_row, comp_star_row, p = rows
        ax.plot(dense_row, comp_star_row, color=colors_4[i], marker='o', markersize=8, linewidth=2.0)

    for i, rows in enumerate(zip(densities, comp_stars_naive, ps)):
        dense_row, comp_star_row, p = rows
        ax.plot(dense_row, comp_star_row, color=colors_4[i], marker='s', markersize=8, linewidth=2.0)

    shift_up = 1.5

    x_min = np.percentile(densities[0], 10)
    x_max = np.percentile(densities[0], 40)
    x_range_short = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    text_x_offset = x_range_short[0] * 0.98

    line = (comp_stars[0][0] * shift_up) * (x_range_short / densities[0][0])**(0)
    ax.plot(x_range_short, line, '--', color=colors_4[0], linewidth=2.0)
    ax.text(text_x_offset, line[0], r'const', color=colors_4[0], fontsize=15, va='bottom', ha='right')

    line = (comp_stars[1][0] * shift_up) * (x_range_short / densities[1][0])**(-1)
    ax.plot(x_range_short, line, '--', color=colors_4[1], linewidth=2.0)
    ax.text(text_x_offset, line[0], r'$\rho^{-1}$', color=colors_4[1], fontsize=15, va='bottom', ha='right')

    line = (comp_stars_naive[0][0] * shift_up) * (x_range_short / densities[0][0])**(-2)
    ax.plot(x_range_short, line, ':', color=colors_4[0], linewidth=2.0)
    ax.text(x_range_short[0] * 0.8, line[0] * 1.4, r'$\rho^{-2}$', color=colors_4[0], fontsize=15, va='bottom',
            ha='left')

    line = (comp_stars_naive[1][0] * shift_up) * (x_range_short / densities[1][0])**(-4)
    ax.plot(x_range_short, line, ':', color=colors_4[1], linewidth=2.0)
    idx_4 = 40
    ax.text(x_range_short[idx_4]*1.1, line[idx_4] * 1.4, r'$\rho^{-4}$', color=colors_4[1], fontsize=15, va='bottom',
            ha='center')

    custom_handles_4 = [
        Line2D([0], [0], color='gray', marker='o', linestyle='-', markersize=8, label='Cell Horizon'),
        Line2D([0], [0], color='gray', marker='s', linestyle='-', markersize=8, label='Four Factor'),
        Line2D([0], [0], color='none', label='Blue: p = 2'),
        Line2D([0], [0], color='none', label='Black: p = 4')
    ]

    leg_4 = ax.legend(handles=custom_handles_4, fontsize=12, framealpha=1, edgecolor='black', loc='upper right')
    style_legend(leg_4)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))

    apply_main_style(ax, r'$\rho / \left( \beta k \right)^{1/p}$', r'$\Omega(b^*_P)$', xpad=0, ypad=-2)

    ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}" if x == 0.05 else f"{x:.1f}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.tick_params(axis='x', which='major', pad=2)

    # ==========================================
    # SUBPLOT 2
    # ==========================================
    ax_right = axs2[1]
    Ns_inset_4, comp_efforts_4 = get_actual_figure_data("fig3", "b_comp_effort_times_N_vs_N")
    _, iacts_4 = get_actual_figure_data("fig3", "b_iact_times_N_vs_N")

    ax_right.plot(Ns_inset_4, comp_efforts_4, color='peru', marker='D', linestyle='None', markersize=8)

    ax_right.plot(Ns_inset_4, iacts_4, color='peru', marker='*', linestyle='None', markersize=14)

    log_start_4 = np.log10(Ns_inset_4[0])
    log_end_4 = np.log10(Ns_inset_4[-1])
    log_span_4 = log_end_4 - log_start_4
    Ns_dotted_4 = np.logspace(log_start_4 + 0.25 * log_span_4, log_end_4 - 0.25 * log_span_4, 50)

    shift_above_4 = 0.4
    pref_4 = (comp_efforts_4[-1] / (Ns_inset_4[-1] ** 1.5)) * shift_above_4
    ax_right.plot(Ns_dotted_4, pref_4 * (Ns_dotted_4 ** 1.5), color='peru', linestyle=':', linewidth=2, zorder=1)

    x_lbl_4 = Ns_dotted_4[20]
    ax_right.text(x_lbl_4, pref_4 * (x_lbl_4 ** 1.5) * 0.65, r'$N^{3/2}$', color='peru', fontsize=16, ha='center',
                  va='bottom')

    custom_handles_right = [
        Line2D([0], [0], color='gray', marker='*', linestyle='-', markersize=12,
               label=r'X = $\tau / ((\beta k)^{1 / 4} v_0)^{-1}$'),
        Line2D([0], [0], color='gray', marker='D', linestyle='-', markersize=8, label=r'X = $\Omega$')
    ]

    ax_right.set_xscale('log')
    ax_right.set_yscale('log')

    ax_right.set_yticks([10**3, 10**4])

    ax_right.set_xticks([200, 300, 400])
    ax_right.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_right.get_xaxis().set_minor_locator(ticker.NullLocator())

    apply_main_style(ax_right, r'$N$', 'X', xpad=-1, ypad=-15)

    leg_right = ax_right.legend(handles=custom_handles_right, fontsize=12, loc='lower right', framealpha=1,
                                edgecolor='black')
    style_legend(leg_right)

    label_props = dict(fontsize=22, va='top', ha='right',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    axs2[0].text(0.98, 0.59, '(a)', transform=axs2[0].transAxes, **label_props)
    axs2[1].text(0.12, 0.96, '(b)', transform=axs2[1].transAxes, **label_props)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print('Plot saved.')


if __name__ == '__main__':
    main()
