import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../")
sys.path.insert(0, src_directory)

data_directory = "output/paper_data/fig6"
output_file_name = "output/paper_data/plots/combined_mixing.png"

data_b_opt = "harmonic_chain_event_particle_positions_from_low_prob_strong_coupling_b_opt_temp_0001_N_8_new_no_cell"
data_b_0 = "harmonic_chain_event_particle_positions_from_low_prob_strong_coupling_b_0_paper"


def load_data(data_name, remove_eq=0, number_of_equilibration_iterations=0):

    event_particle_position_path = os.path.join(
        data_directory, f'{data_name}_checkpoint_00_sample_of_event_particle_position.npy')
    pos_sample = np.load(event_particle_position_path)[remove_eq * number_of_equilibration_iterations:]

    event_active_particle_index_path = os.path.join(
        data_directory, f'{data_name}_checkpoint_00_sample_of_event_active_particle_index.npy')
    idx_sample = np.load(event_active_particle_index_path)[remove_eq * number_of_equilibration_iterations:]

    idx_sample = idx_sample + 1

    return pos_sample, idx_sample


def main_combined(data_left, data_right, n, m, file_name, vlines=1, remove_eq=0, inset=True):
    pos_left, idx_left = load_data(data_left, remove_eq)
    pos_right, idx_right = load_data(data_right, remove_eq)

    n_cols_left = np.shape(pos_left)[1]
    n_cols_right = np.shape(pos_right)[1]

    plt.rcParams['axes.linewidth'] = 5.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.major.width'] = 3.9
    plt.rcParams['ytick.major.width'] = 3.9
    plt.rcParams['xtick.minor.width'] = 3.9
    plt.rcParams['ytick.minor.width'] = 3.9

    plt.rcParams['ytick.major.pad'] = 5.5

    fig, ax = plt.subplots(2, 2, sharex='col', figsize=(18, 10))

    x_data = np.arange(n, m + 1) / 1e4

    # ==========================================
    # SUBPLOTS 3 AND 4
    # ==========================================

    for i in range(n_cols_left):
        ax[0, 1].plot(x_data, pos_left[:, i][n:m+1], alpha=0.7, linewidth=3.25, label=f'$i={i+1}$')

    if inset:
        axins = ax[0, 1].inset_axes([0.40, 0.42, 0.42, 0.54])
        for i in range(n_cols_left):
            axins.plot(x_data, pos_left[:, i][n:m+1], alpha=0.7, linewidth=1.95)

        inset_x_max = 1.5
        inset_x_min = x_data[0] - 0.005

        if n_cols_left > 7:
            brown_data = pos_left[n:m+1, 5]
            pink_data = pos_left[n:m+1, 6]
            grey_data = pos_left[n:m+1, 7]

            move_mask_brown = brown_data > (brown_data[0] + 1e-6)
            move_idx_brown = np.argmax(move_mask_brown) if np.any(move_mask_brown) else 0

            if move_idx_brown > 0:
                inset_x_max = x_data[move_idx_brown] + 0.10

        axins.set_xlim(inset_x_min, inset_x_max)

        mask = (x_data >= inset_x_min) & (x_data <= inset_x_max)
        if np.any(mask):
            y_max = np.max(pos_left[n:m+1, :][mask])
            y_min = np.min(pos_left[n:m+1, :][mask])
            axins.set_ylim(y_min - 0.2, y_max + 0.25)

        axins.set_xticks([])
        axins.set_yticks([0, 2, 4, 6])
        axins.set_yticklabels(['0', '1', '2', '3'])
        axins.tick_params(axis='y', direction='in', top=False, right=False, labelsize=26, pad=8)

        indicator_c = ax[0, 1].indicate_inset_zoom(axins, edgecolor="black", alpha=1.0, linewidth=1.95)
        connectors_c = indicator_c[1] if isinstance(indicator_c, tuple) else indicator_c.connectors

        for i, c in enumerate(connectors_c):
            if i == 1:
                c.set_visible(True)
            else:
                c.set_visible(False)

        if n_cols_left > 7:
            move_mask_grey = grey_data > (grey_data[0] + 1e-6)
            move_idx_grey = np.argmax(move_mask_grey) if np.any(move_mask_grey) else 0
            idx1 = min(move_idx_grey + 50, len(x_data) - 1)

            x_pos_ins1 = x_data[idx1]
            y_pink_ins1 = pink_data[idx1]
            y_grey_ins1 = grey_data[idx1]

            axins.annotate('', xy=(x_pos_ins1, y_pink_ins1), xytext=(x_pos_ins1, y_grey_ins1),
                           arrowprops=dict(arrowstyle='<->', color='black', linewidth=3.25))
            axins.text(x_pos_ins1 + 0.005, (y_pink_ins1 + y_grey_ins1) / 2,
                       r'$x_{N,7} \approx L/N$', va='center', ha='left', fontsize=28)

            idx2 = max(0, move_idx_brown - 150)
            x_pos_ins2 = x_data[idx2]
            y_brown_ins2 = brown_data[idx2]
            y_pink_ins2 = pink_data[idx2]

            axins.annotate('', xy=(x_pos_ins2, y_brown_ins2), xytext=(x_pos_ins2, y_pink_ins2),
                           arrowprops=dict(arrowstyle='<->', color='black', linewidth=3.25))
            axins.text(x_pos_ins2 + 0.005, (y_brown_ins2 + y_pink_ins2) / 2,
                       r'$x_{7,6} \approx L/N$', va='center', ha='left', fontsize=28)

    last_y_vals = np.sort(pos_left[m, :])
    x_pos = x_data[-1]

    for i in range(len(last_y_vals) - 1):
        ax[0, 1].annotate('', xy=(x_pos, last_y_vals[i]), xytext=(x_pos, last_y_vals[i+1]),
                          arrowprops=dict(arrowstyle='<->', color='black', linewidth=3.25))

    ax[0, 1].text(x_pos - 0.008 * (x_data[-1] - x_data[0]), ((last_y_vals[0] + last_y_vals[1]) / 2) - 0.25,
                  r'$x_{i+1,i} \approx L/N$', va='center', ha='right', fontsize=28)

    leg1 = ax[0, 1].legend(fontsize=25, loc='upper left', bbox_to_anchor=(0.01, 1.00), ncol=2,
                           columnspacing=0.8, frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
    leg1.get_frame().set_linewidth(3.9)

    ax[0, 1].set_yticks(np.arange(0, 17, 2))
    ax[0, 1].set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', r'$N$'])
    ax[0, 1].tick_params(axis='both', labelsize=28)

    x_range = x_data[-1] - x_data[0]
    ax[0, 1].set_xlim(x_data[0] - 0.01 * x_range, x_data[-1] + 0.01 * x_range)

    active_data_left = idx_left[n:m+1]
    ax[1, 1].plot(x_data, active_data_left, 'o', markersize=1, color='darkred')

    ax[1, 1].set_yticks(np.arange(1, 9))

    max_idx = int(np.max(active_data_left))
    min_idx = int(np.min(active_data_left))

    label_multiplier = 1

    for p in range(max_idx, min_idx, -1):
        mask_curr = active_data_left == p
        mask_next = active_data_left == p - 1

        if np.any(mask_curr) and np.any(mask_next):
            idx_curr = np.argmax(mask_curr)
            idx_next = np.argmax(mask_next)

            if idx_next > idx_curr:
                x_curr = x_data[idx_curr]
                x_next = x_data[idx_next]
                delta_x = x_next - x_curr
                y_pos = p - 1

                if delta_x > 0.01:
                    ax[1, 1].annotate('', xy=(x_curr, y_pos), xytext=(x_next, y_pos),
                                      arrowprops=dict(arrowstyle='<->', color='black', linewidth=3.25))

                    if label_multiplier == 1:
                        lbl_text = r'$\Lambda \cdot L/N$'
                    else:
                        lbl_text = fr'${label_multiplier} \cdot \Lambda \cdot L/N$'

                    ax[1, 1].text(x_curr, y_pos + 0.1,
                                  lbl_text, ha='left', va='bottom', fontsize=28,
                                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

                    label_multiplier += 1

    if inset:
        axins2 = ax[1, 1].inset_axes([0.55, 0.42, 0.42, 0.54])

        hit_7_indices = np.where(active_data_left == 7)[0]
        if len(hit_7_indices) > 0:
            window_start = min(hit_7_indices[0] + 10, len(active_data_left) - 26)
        else:
            window_start = 0

        window_end = min(window_start + 25, len(active_data_left) - 1)

        while active_data_left[window_end] != 7 and window_end < len(active_data_left) - 1:
            window_end += 1

        axins2.plot(x_data[window_start:window_end+1], active_data_left[window_start:window_end+1],
                    '-o', markersize=3, color='darkred', linewidth=1.56)

        axins2.set_xlim(x_data[window_start], x_data[window_end])
        axins2.set_ylim(6.8, 8.2)
        axins2.set_yticks([7, 8])
        axins2.set_xticks([])

        axins2.tick_params(axis='y', direction='in', top=False, right=False, labelsize=26, pad=8)

        indicator = ax[1, 1].indicate_inset_zoom(inset_ax=axins2, edgecolor="black", alpha=1.0, linewidth=1.95)
        connectors = indicator[1] if isinstance(indicator, tuple) else indicator.connectors

        for c in connectors:
            c.set_visible(False)

        con = patches.ConnectionPatch(
            xyA=(x_data[window_start], 6.8), coordsA=axins2.transData,
            xyB=(x_data[window_end], 8.2), coordsB=ax[1, 1].transData,
            color="black", linewidth=1.95
        )
        ax[1, 1].add_artist(con)

    ax[1, 1].tick_params(axis='both', labelsize=28)
    ax[1, 1].set_xlabel('Number of events / $10^4$', fontsize=32, labelpad=12)

    # ==========================================
    # SUBPLOTS 1 AND 2
    # ==========================================

    for i in range(n_cols_right):
        ax[0, 0].plot(x_data, pos_right[:, i][n:m+1], alpha=0.7, linewidth=3.25, label=f'$i={i+1}$')

    last_y_vals_right = np.sort(pos_right[m, :])

    for i in range(len(last_y_vals_right) - 1):
        ax[0, 0].annotate('', xy=(x_pos, last_y_vals_right[i]), xytext=(x_pos, last_y_vals_right[i+1]),
                          arrowprops=dict(arrowstyle='<->', color='black', linewidth=3.25))

    ax[0, 0].text(x_pos - 0.008 * (x_data[-1] - x_data[0]), (last_y_vals_right[0] + last_y_vals_right[1]) / 2,
                  r'$x_{i+1,i} \ll L/N$', va='center', ha='right', fontsize=28)

    leg2 = ax[0, 0].legend(fontsize=25, loc='upper left', bbox_to_anchor=(0.01, 1.00), ncol=2,
                           columnspacing=0.8, frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)
    leg2.get_frame().set_linewidth(3.9)

    y_max_plot = np.max(pos_right[n:m+1, :])
    max_tick = np.ceil(y_max_plot * 5) / 5.0
    yticks = np.arange(0, max_tick + 0.1, 0.2)

    ytick_labels = []
    for val in yticks:
        if np.isclose(val, 0):
            ytick_labels.append('0')
        else:
            val_n_l = val / 2.0
            if val_n_l.is_integer():
                ytick_labels.append(f'{int(val_n_l)}')
            else:
                ytick_labels.append(f'{val_n_l:.1f}')

    ax[0, 0].set_yticks(yticks)
    ax[0, 0].set_yticklabels(ytick_labels)
    ax[0, 0].tick_params(axis='both', labelsize=28)

    ax[0, 0].set_ylabel(r'$x_i \cdot N/L$', fontsize=32)

    ax[0, 0].set_ylim(bottom=-0.05, top=max_tick + 0.04)
    ax[0, 0].set_xlim(x_data[0] - 0.01 * x_range, x_data[-1] + 0.01 * x_range)

    active_data_right = idx_right[n:m+1]
    ax[1, 0].plot(x_data, active_data_right, 'o', markersize=1, color='darkred')

    ax[1, 0].set_yticks(np.arange(1, 9))
    ax[1, 0].set_ylim(0.5, 8.5)

    if inset:
        axins3 = ax[1, 0].inset_axes([0.55, 0.42, 0.42, 0.54])

        window_start_r = 0
        window_end_r = min(100, len(active_data_right) - 1)

        axins3.plot(x_data[window_start_r:window_end_r+1], active_data_right[window_start_r:window_end_r+1],
                    '-o', markersize=3, color='darkred', linewidth=1.56)

        axins3.set_xlim(x_data[window_start_r], x_data[window_end_r])
        axins3.set_ylim(0.5, 8.5)
        axins3.set_yticks(np.arange(1, 9))
        axins3.set_xticks([])

        axins3.tick_params(axis='y', direction='in', top=False, right=False, labelsize=26, pad=8)

        indicator_r = ax[1, 0].indicate_inset_zoom(inset_ax=axins3, edgecolor="black", alpha=1.0, linewidth=1.95)
        connectors_r = indicator_r[1] if isinstance(indicator_r, tuple) else indicator_r.connectors

        for i, c in enumerate(connectors_r):
            if i == 0:
                c.set_visible(True)
            else:
                c.set_visible(False)

    ax[1, 0].set_ylabel(r'$a$', fontsize=32)
    ax[1, 0].tick_params(axis='both', labelsize=28)
    ax[1, 0].set_xlabel('Number of events / $10^4$', fontsize=32, labelpad=12)

    if vlines == 1:
        for x_val in x_data:
            for row in range(2):
                for col in range(2):
                    ax[row, col].axvline(x_val, color="black", linestyle=":", linewidth=0.8, alpha=0.4)

    label_props = dict(fontsize=32, va='top', ha='right',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax[0, 0].text(0.98, 0.98, '(a)', transform=ax[0, 0].transAxes, **label_props)
    ax[1, 0].text(0.98, 0.12, '(b)', transform=ax[1, 0].transAxes, **label_props)
    ax[0, 1].text(0.98, 0.98, '(c)', transform=ax[0, 1].transAxes, **label_props)
    ax[1, 1].text(0.98, 0.12, '(d)', transform=ax[1, 1].transAxes, **label_props)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.05)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    print(f"Figure successfully saved to {file_name}")


def main():
    os.chdir(src_directory)

    n, m = 0, 40000

    main_combined(
        data_left=data_b_opt,
        data_right=data_b_0,
        n=n,
        m=m,
        vlines=0,
        file_name=output_file_name,
        inset=True
    )
    plt.show()


if __name__ == "__main__":
    main()
