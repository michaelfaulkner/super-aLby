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

output_file = "output/optimal_sampling_strategies_figs/plots/harm_chain_figures_1.pdf"


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

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    # ==========================================
    # SUBPLOT 1
    # ==========================================
    ax = axs[0, 0]

    Ns_er = np.array([3, 8, 128])
    colors_1 = ['purple', 'orange', 'blue', 'grey', 'firebrick', 'black']

    for i, N in enumerate(Ns_er):
        x, y = get_actual_figure_data("fig2", f"a_mean_event_rate_cell_horizon_N{N}_vs_b_over_bstar")
        ax.plot(x, y, '8', markersize=8, alpha=0.5, color=colors_1[i])
        x, y = get_actual_figure_data("fig2", f"a_mean_event_rate_prediction_cell_horizon_N{N}_vs_b_over_bstar")
        ax.plot(x, y, color=colors_1[i], alpha=0.5, linewidth=2.0)

    x, y = get_actual_figure_data("fig2", "a_mean_event_rate_nonfactorised_N64_vs_b_over_bstar")
    ax.plot(x, y, 'D', markersize=6, alpha=0.5, color='k')
    ax.axhline(0.564, color='darkgreen', linestyle='--', linewidth=2.0)
    ax.set_ylim(top=2.05)

    custom_handles_1 = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.0, label='        Prediction'),
        Line2D([0], [0], color=colors_1[0], lw=2.0, marker='8', linestyle='None',
               label=r'      $\Lambda=\Lambda_{\rm ch}$, N=3'),
        Line2D([0], [0], color=colors_1[1], lw=2.0, marker='8', linestyle='None',
               label=r'      $\Lambda=\Lambda_{\rm ch}$, N=8'),
        Line2D([0], [0], color=colors_1[2], lw=2.0, marker='8', linestyle='None',
               label=r'      $\Lambda=\Lambda_{\rm ch}$, N=128'),
        Line2D([0], [0], color='black', lw=2.0, marker='D', markersize=6, linestyle='None',
               label=r'      $\Lambda=\Lambda_{nf}$, N=64')
    ]

    leg_1 = ax.legend(handles=custom_handles_1, fontsize=8, loc='lower right', framealpha=1, edgecolor='black', ncol=1,
                      handlelength=1.5, handletextpad=-1.5)
    style_legend(leg_1)
    apply_main_style(ax, r'$b/b^*_P$', r'$\Lambda / (\sqrt{\beta k} v_0)$', xpad=0, ypad=-2)

    ax_inset = ax.inset_axes([0.3365, 0.5765, 0.397, 0.397])
    x, y = get_actual_figure_data("fig2", "a_inset_event_rate_ratio_cell_horizon_to_nonfactorised_vs_N")
    ax_inset.plot(x, y, 'k*', markersize=11, label='ECMC Data')
    x, y = get_actual_figure_data("fig2", "a_inset_event_rate_ratio_prediction_vs_N")
    ax_inset.plot(x, y, 'k--', linewidth=2.0, label='Prediction')
    ax_inset.axhline(np.sqrt(2), color='firebrick', linestyle='--', linewidth=2.0,
                     label=r'$N\rightarrow \infty = \sqrt{2}$')

    ax_inset.set_xscale('log')
    ax_inset.set_xticks([10, 100])
    ax_inset.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax_inset.get_xaxis().set_minor_locator(ticker.NullLocator())

    apply_inset_style(ax_inset, r'$N$', r'$\Lambda_{\rm ch} (b^*_P)/\Lambda_{nf}$', xpad=-1, ypad=-4)
    leg_inset_1 = ax_inset.legend(fontsize=8, labelspacing=0.3, borderpad=0.3, handlelength=1.5)
    style_inset_legend(leg_inset_1)

    # ==========================================
    # SUBPLOT 2
    # ==========================================
    ax = axs[0, 1]
    Ns_er_2 = np.array([16, 32, 64, 128])
    colors_2 = ['red', 'black', 'limegreen', 'blue', 'slategrey', 'black']

    for i, N in enumerate(Ns_er_2):
        x, y = get_actual_figure_data("fig2", f"b_iact_normalised_cell_horizon_N{N}_vs_b_over_bstar")
        ax.plot(x, y, marker='p', markersize=8, alpha=0.7, color=colors_2[i])
        x, y = get_actual_figure_data("fig2", f"b_iact_normalised_nonfactorised_N{N}")
        ax.plot(x, y, linestyle='--', color=colors_2[i], alpha=0.5, linewidth=2.0)

    custom_handles_2 = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.0, label='    Non-fact.'),
        Line2D([0], [0], color=colors_2[0], linestyle='-', marker='p', lw=2.0, markersize=8, label='    N = 16'),
        Line2D([0], [0], color=colors_2[1], linestyle='-', marker='p', lw=2.0, markersize=8, label='    N = 32'),
        Line2D([0], [0], color=colors_2[2], linestyle='-', marker='p', lw=2.0, markersize=8, label='    N = 64'),
        Line2D([0], [0], color=colors_2[3], linestyle='-', marker='p', lw=2.0, markersize=8, label='    N = 128')
    ]

    leg_2 = ax.legend(handles=custom_handles_2, fontsize=9, framealpha=1, loc='upper center', edgecolor='black', ncol=1)
    style_legend(leg_2)
    ax.set_yscale('log')
    apply_main_style(ax, r'$b/b^*_P$', r'$\tau/\tau_{\rm ch}(b^*_P)$', xpad=0, ypad=-2)

    # ==========================================
    # SUBPLOT 3
    # ==========================================
    ax = axs[1, 0]
    for i, N in enumerate(Ns_er_2):
        x, y = get_actual_figure_data("fig2", f"c_comp_effort_normalised_cell_horizon_N{N}_vs_b_over_bstar")
        ax.plot(x, y, marker='v', markersize=8, linestyle='-', alpha=0.7, color=colors_2[i])
        x, y = get_actual_figure_data("fig2", f"c_comp_effort_normalised_nonfactorised_N{N}")
        ax.plot(x, y, linestyle='--', alpha=0.5, color=colors_2[i], linewidth=2.0)

    custom_handles_3 = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2.0, label='    Non-fact.'),
        Line2D([0], [0], color=colors_2[0], linestyle='-', lw=2.0, marker='v', markersize=8, label='    N = 16'),
        Line2D([0], [0], color=colors_2[1], linestyle='-', lw=2.0, marker='v', markersize=8, label='    N = 32'),
        Line2D([0], [0], color=colors_2[2], linestyle='-', lw=2.0, marker='v', markersize=8, label='    N = 64'),
        Line2D([0], [0], color=colors_2[3], linestyle='-', lw=2.0, marker='v', markersize=8, label='    N = 128')
    ]

    leg_3 = ax.legend(handles=custom_handles_3, fontsize=9, framealpha=1, loc='upper center', edgecolor='black', ncol=1)
    style_legend(leg_3)
    ax.set_yscale('log')
    apply_main_style(ax, r'$b/b^*_P$', r'$\Omega/\Omega_{\rm ch}(b^*_P)$', xpad=0, ypad=-2)

    # ==========================================
    # SUBPLOT 4
    # ==========================================
    ax = axs[1, 1]

    for kind, color in (('nonfactorised', 'r'), ('b_zero', 'b'), ('b_star', 'k')):
        x, y = get_actual_figure_data("fig2", f"d_iact_times_N_{kind}_vs_N")
        ax.plot(x, y, color=color, marker='*', linestyle='-', markersize=10)

    for kind, color in (('nonfactorised', 'r'), ('b_zero', 'b'), ('b_star', 'k')):
        x, y = get_actual_figure_data("fig2", f"d_comp_effort_times_N_{kind}_vs_N")
        ax.plot(x, y, color=color, marker='D', linestyle='-', markersize=7)

    Ns_inset_2, y_red_omega = get_actual_figure_data("fig2", "d_comp_effort_times_N_nonfactorised_vs_N")
    _, y_blue_omega = get_actual_figure_data("fig2", "d_comp_effort_times_N_b_zero_vs_N")
    _, y_black_omega = get_actual_figure_data("fig2", "d_comp_effort_times_N_b_star_vs_N")

    log_start = np.log10(Ns_inset_2[0])
    log_end = np.log10(Ns_inset_2[-1])
    log_span = log_end - log_start
    Ns_dotted = np.logspace(log_start + 0.375 * log_span, log_start + 0.625 * log_span, 50)

    mid_idx = 2
    N_mid = Ns_inset_2[mid_idx]
    x_lbl = Ns_dotted[25]

    pref_black_omega = (y_black_omega[mid_idx] / (N_mid ** 1.5)) * 0.65
    pref_blue_omega = (y_blue_omega[mid_idx] / (N_mid ** 2.5)) * 0.65
    pref_red_omega = (y_red_omega[mid_idx] / (N_mid ** 3)) * 1.35

    ax.plot(Ns_dotted, pref_black_omega * (Ns_dotted ** 1.5) / 0.5, 'k:', linewidth=2, zorder=1)
    ax.plot(Ns_dotted, pref_blue_omega * (Ns_dotted ** 2.5) / 0.5, 'b:', linewidth=2, zorder=1)
    ax.plot(Ns_dotted, pref_red_omega * (Ns_dotted ** 3) / 0.5, 'r:', linewidth=2, zorder=1)

    ax.text(x_lbl + 15, (pref_black_omega * (x_lbl ** 1.5) * 0.75) / 0.13, r'$N^{3/2}$', color='k', fontsize=14,
            ha='center', va='top')
    ax.text(x_lbl + 17, (pref_blue_omega * (x_lbl ** 2.5) * 0.9) / 0.3, r'$N^{5/2}$', color='b', fontsize=14,
            ha='center', va='bottom')
    ax.text(x_lbl + 9, (pref_red_omega * (x_lbl ** 3) * 1.95) / 0.5, r'$N^3$', color='r', fontsize=14, ha='center',
            va='bottom')

    custom_handles_colors = [
        Line2D([0], [0], color='r', lw=2.0, label='Non-fact.'),
        Line2D([0], [0], color='b', lw=2.0, label=r'$b=0$'),
        Line2D([0], [0], color='k', lw=2.0, label=r'$b=b^*_P$')
    ]

    custom_handles_shapes = [
        Line2D([0], [0], color='gray', marker='*', linestyle='-', markersize=10,
               label=r'X = $\tau / (\sqrt{\beta k} v_0)^{-1}$'),
        Line2D([0], [0], color='gray', marker='D', linestyle='-', markersize=8, label=r'X = $\Omega$')
    ]

    leg_colors = ax.legend(handles=custom_handles_colors, loc='upper left', fontsize=10, framealpha=1,
                           edgecolor='black', ncol=1)
    style_legend(leg_colors)
    ax.add_artist(leg_colors)

    leg_shapes = ax.legend(handles=custom_handles_shapes, loc='lower right', fontsize=10, framealpha=1,
                           edgecolor='black', ncol=1)
    style_legend(leg_shapes)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_yticks([10**3, 10**4, 10**5])
    ax.set_xticks([50, 100])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.get_xaxis().set_minor_locator(ticker.NullLocator())

    apply_main_style(ax, r'$N$', 'X', xpad=0, ypad=-2)

    label_props = dict(fontsize=22, va='top', ha='right',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    axs[0, 0].text(0.85, 0.98, '(a)', transform=axs[0, 0].transAxes, **label_props)
    axs[0, 1].text(0.77, 0.98, '(b)', transform=axs[0, 1].transAxes, **label_props)
    axs[1, 0].text(0.76, 0.98, '(c)', transform=axs[1, 0].transAxes, **label_props)
    axs[1, 1].text(0.4, 0.98, '(d)', transform=axs[1, 1].transAxes, **label_props)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print('Plot saved.')


if __name__ == '__main__':
    main()
