from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)

output_file = "output/paper_data/plots/index_velocity_pressure.pdf"
data_directory = "output/paper_data/fig7"


def load_data_file(file_name):
    return np.load(os.path.join(data_directory, file_name))


def main():
    os.chdir(src_directory)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    b, _, idx = load_data_file(
        'paper_poly_potential_b_sweep_N16_L32_power2_'
        'index_and_state_space_velocity_factor_field_prefactor_sweep.npy')[:, ::4]
    b_star = 2.0

    ax.plot(b / b_star, idx, color='firebrick', marker='8', markersize=8, linestyle='None', alpha=0.8)
    ax.plot(b / b_star, b - 32 * 1 / 16, color='firebrick', linestyle='-', linewidth=2.5, alpha=0.8)

    b, _, idx = load_data_file(
        'paper_poly_potential_b_sweep_N16_L32_power2_temp_2_'
        'index_and_state_space_velocity_factor_field_prefactor_sweep.npy')[:, ::4]
    b_star = 2.0

    ax.plot(b / b_star, idx / (0.5 * 1) ** -0.5, color='orange', marker='8', markersize=8, linestyle='None', alpha=0.8)
    ax.plot(b / b_star, (.5 * b - .5 * 32 * 1 / 16) / (0.5 * 1) ** -0.5, color='orange', linestyle='-', linewidth=2.5,
            alpha=0.8)

    b, _, idx = load_data_file(
        'paper_idx_space_N8_L8_sd_index_and_state_space_velocity_factor_field_prefactor_sweep.npy')
    b_star = 3.06

    data_base = "paper_idx_space_N8_L8_sd"
    final_prefactor_means = []
    for p in range(1):
        job_means = []
        for j in range(1):
            file_name = f"{data_base}_factor_field_prefactor_{p:02d}_job_{j:02d}_checkpoint_00_sample_of_potential.npy"
            data = load_data_file(file_name)
            job_means.append(np.mean(data))
        if job_means:
            final_prefactor_means.append(np.mean(job_means))
        else:
            final_prefactor_means.append(np.nan)
    result_array = np.array(final_prefactor_means)

    ax.plot(b / b_star, idx / 0.75, color='darkblue', marker='8', markersize=8, linestyle='None', alpha=0.8)
    ax.plot(b / b_star, (-b + 8 / 8 * (1 + 2 / 8 * result_array[0]) - 1 / 8) / 0.75, color='darkblue', linestyle='-',
            linewidth=2.5, alpha=0.8)

    b, _, idx = load_data_file(
        'paper_idx_space_N8_L16_power4_index_and_state_space_velocity_factor_field_prefactor_sweep.npy')
    b_star = 16.9 / 2

    data_base = "paper_idx_space_N8_L16_power4"
    final_prefactor_means = []
    for p in range(1):
        job_means = []
        for j in range(1):
            file_name = f"{data_base}_factor_field_prefactor_{p:02d}_job_{j:02d}_checkpoint_00_sample_of_potential.npy"
            data = load_data_file(file_name)
            job_means.append(np.mean(data))
        if job_means:
            final_prefactor_means.append(np.mean(job_means))
        else:
            final_prefactor_means.append(np.nan)
    result_array = np.array(final_prefactor_means)

    ax.plot(b / b_star, idx, color='darkgreen', marker='8', markersize=8, linestyle='None', alpha=0.8)
    ax.plot(b / b_star, b + 8 / 16 * (1 - 4.0 * result_array[0] / 8), color='darkgreen', linestyle='-', linewidth=2.5,
            alpha=0.8)

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
