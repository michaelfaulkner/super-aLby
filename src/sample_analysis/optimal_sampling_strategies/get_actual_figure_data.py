import get_simulation_data
import glob
import importlib
import json
import numpy as np
import os
import sys
from scipy.special import erf


def figure_data_directory(figure):
    return os.path.join(get_simulation_data.output_directory, "actual_figure_data", figure)


def get_actual_figure_data(figure, name):
    path = os.path.join(figure_data_directory(figure), name + ".npy")
    if not os.path.exists(path):
        figure_makers[figure]()
    return np.load(path)


def _load(path):
    return get_simulation_data.get_simulation_data(os.path.join(get_simulation_data.output_directory, path))


def _save(figure, name, *rows):
    os.makedirs(figure_data_directory(figure), exist_ok=True)
    np.save(os.path.join(figure_data_directory(figure), name + ".npy"),
            np.vstack([np.asarray(row, dtype=float).reshape(-1) for row in rows]))


def _mean_event_rate_prediction(b, temperature, k, n, length):
    delta = b - length / n
    sigma = np.sqrt(temperature / k * (1.0 - 1.0 / n))
    return (k / temperature * (delta * erf(delta / (np.sqrt(2.0) * sigma)) +
                               np.sqrt(2.0 / np.pi) * sigma * np.exp(-delta ** 2 / (2.0 * sigma ** 2))))


def _non_factorised_value(data):
    return data[1][0]


def _std_comp_effort(run):
    path = os.path.join(get_simulation_data.output_directory, run, "std_comp_effort_vs_b.npy")
    if not os.path.exists(path):
        run_directory = os.path.dirname(path)
        config_file = get_simulation_data.get_config_file(run_directory)
        bash_file = os.path.relpath(config_file[:-4], get_simulation_data.src_directory) + ".sh"
        get_simulation_data.check_simulations_have_run(run_directory, "*_00/job_*", bash_file)
        sys.path.insert(0, get_simulation_data.src_directory)
        sys.path.insert(0, os.path.dirname(get_simulation_data.this_directory))
        helper_methods = importlib.import_module("helper_methods")
        get_iact = importlib.import_module("markov_chain_diagnostics").get_iact
        sweep_name = helper_methods.read_variable_from_sh_file(config_file[:-4] + ".sh", "CONFIG_VARIABLE")
        prefactors = _load(f"{run}/structure_factor_iact_vs_prefactor.npy")[0]
        stds = []
        for i in range(len(prefactors)):
            sweep_directory = os.path.join(run_directory, f"{sweep_name}_{i:02d}")
            if not os.path.isdir(sweep_directory):
                stds.append(0.0)
                continue
            comp_efforts = []
            for job_directory in glob.glob(os.path.join(sweep_directory, "job_*")):
                sim_params_path = os.path.join(job_directory, "sim_params.json")
                sample_path = os.path.join(job_directory, "checkpoint_00_sample_of_structure_factor.npy")
                if not os.path.exists(sim_params_path) or not os.path.exists(sample_path):
                    continue
                try:
                    with open(sim_params_path, "r") as f:
                        sim_params = json.load(f)
                    mean_rate = sim_params.get("mean_event_rate")
                    acceptance_rate = sim_params.get("acceptance_rate", 1.0)
                    if acceptance_rate == 0:
                        continue
                    comp_efforts.append(get_iact(np.load(sample_path).flatten()) * mean_rate / acceptance_rate)
                except Exception as e:
                    print(f"Error in {job_directory}: {e}")
            if len(comp_efforts) > 1:
                stds.append(np.std(comp_efforts, ddof=1))
            elif len(comp_efforts) == 1:
                stds.append(0.0)
            else:
                stds.append(np.nan)
        np.save(path, np.vstack([prefactors, stds]))
    return np.load(path)[1]


def make_fig2():
    figure = "fig2"
    for n in (3, 8, 128):
        data = _load(f"fig2/N{n}/L{2 * n}_power2/mean_event_rate_sweep.npy")
        prediction = np.array([_mean_event_rate_prediction(b, 1.0, 1.0, n, 2.0 * n) for b in data[0]])
        _save(figure, f"a_mean_event_rate_cell_horizon_N{n}_vs_b_over_bstar", data[0] / 2.0, data[1])
        _save(figure, f"a_mean_event_rate_prediction_cell_horizon_N{n}_vs_b_over_bstar", data[0] / 2.0, prediction)
    data = _load("fig2/N64/L128_power2_unfactor_sweep/mean_event_rate_sweep.npy")
    _save(figure, "a_mean_event_rate_nonfactorised_N64_vs_b_over_bstar", data[0] / 2.0, data[1])

    inset_ns = np.array([3, 8, 64, 512])
    ratios = np.zeros(len(inset_ns))
    for i, n in enumerate(inset_ns):
        ratios[i] = min(_load(f"fig2/N{n}/L{2 * n}_power2/mean_event_rate_sweep.npy")[1])
    ratios /= 0.564
    x_fit = np.linspace(min(inset_ns), max(inset_ns), 10000)
    _save(figure, "a_inset_event_rate_ratio_cell_horizon_to_nonfactorised_vs_N", inset_ns, ratios)
    _save(figure, "a_inset_event_rate_ratio_prediction_vs_N", x_fit,
          np.sqrt(2 / np.pi * (1 - 1 / x_fit)) * np.sqrt(np.pi))

    for panel, file_name, name in (("b", "structure_factor_iact_vs_prefactor", "iact_normalised"),
                                   ("c", "comp_effort_sweep", "comp_effort_normalised")):
        for n in (16, 32, 64, 128):
            data = _load(f"fig2/N{n}/L{2 * n}_power2/{file_name}.npy")
            y_data = n * data[1]
            minimum = min(y_data)
            data_non_factorised = _load(f"fig2/N{n}/L{2 * n}_power2_unfactor/{file_name}.npy")
            non_factorised = n * _non_factorised_value(data_non_factorised) / minimum
            _save(figure, f"{panel}_{name}_cell_horizon_N{n}_vs_b_over_bstar", data[0] / 2.0, y_data / minimum)
            _save(figure, f"{panel}_{name}_nonfactorised_N{n}", [0.0, 2.0], [non_factorised, non_factorised])

    ns = np.array([32, 48, 64, 96, 128])
    for name, file_name in (("iact", "structure_factor_iact_vs_prefactor"), ("comp_effort", "comp_effort_sweep")):
        at_b_star, at_b_zero, non_factorised = np.zeros(len(ns)), np.zeros(len(ns)), np.zeros(len(ns))
        for i, n in enumerate(ns):
            data = _load(f"fig2/N{n}/L{2 * n}_power2/{file_name}.npy")
            at_b_star[i] = min(data[1])
            at_b_zero[i] = data[1][0]
            non_factorised[i] = _non_factorised_value(_load(f"fig2/N{n}/L{2 * n}_power2_unfactor/{file_name}.npy"))
        _save(figure, f"d_{name}_times_N_nonfactorised_vs_N", ns, ns * non_factorised)
        _save(figure, f"d_{name}_times_N_b_zero_vs_N", ns, ns * at_b_zero)
        _save(figure, f"d_{name}_times_N_b_star_vs_N", ns, ns * at_b_star)


def make_fig3():
    figure = "fig3"
    n_particles = 16
    powers = (2, 4)
    lengths = (32, 64, 96, 160)
    densities = np.zeros((len(powers), len(lengths)))
    cell_horizon = np.zeros((len(powers), len(lengths)))
    four_factor = np.zeros((len(powers), len(lengths)))
    for i, p in enumerate(powers):
        for j, length in enumerate(lengths):
            densities[i, j] = n_particles / length
            cell_horizon[i, j] = min(_load(f"fig3/3a/L{length}_power{p}/comp_effort_sweep.npy")[1])
            try:
                four_factor[i, j] = min(_load(f"fig3/3a/L{length}_power{p}_naive/comp_effort_sweep.npy")[1])
            except Exception:
                four_factor[i, j] = np.nan
    for i, p in enumerate(powers):
        _save(figure, f"a_comp_effort_cell_horizon_p{p}_vs_density", densities[i], cell_horizon[i])
        _save(figure, f"a_comp_effort_four_factor_p{p}_vs_density", densities[i], four_factor[i])

    ns = np.array([144, 160, 192, 256, 384, 512])
    comp_efforts, iacts = np.zeros(len(ns)), np.zeros(len(ns))
    for i, n in enumerate(ns):
        comp_efforts[i] = n * min(_load(f"fig3/3b/N{n}_L{2 * n}_power4/comp_effort_sweep.npy")[1])
        iacts[i] = n * min(_load(f"fig3/3b/N{n}_L{2 * n}_power4/structure_factor_iact_vs_prefactor.npy")[1])
    _save(figure, "b_comp_effort_times_N_vs_N", ns, comp_efforts)
    _save(figure, "b_iact_times_N_vs_N", ns, iacts)


def make_fig4():
    figure = "fig4"
    for n in (8, 16, 32, 48, 64, 96):
        run = f"fig4/N{n}_L{n}"
        data = _load(f"{run}/comp_effort_sweep.npy")
        b, _, idx = _load(f"{run}/index_and_state_space_velocity_factor_field_prefactor_sweep.npy")
        errors = _std_comp_effort(run)
        mask = ~np.isnan(idx)
        b, idx = b[mask], idx[mask]
        slope, intercept = np.polyfit(b, idx, 1)
        b_star = -intercept / slope
        if n == 96:
            data = data[:, :-2]
            errors = errors[:-2]
        y_data = n * data[1]
        errors = n * errors
        minimum = min(y_data)
        _save(figure, f"comp_effort_normalised_vs_b_over_bstar_N{n}_L{n}", data[0] / b_star, y_data / minimum,
              errors / minimum)
    inset_ns = np.array([16, 32, 48, 64, 96])
    at_b_star, at_b_zero = [], []
    for n in inset_ns:
        data = _load(f"fig4/N{n}_L{n}/comp_effort_sweep.npy")
        at_b_star.append(np.min(data[1]) * n)
        at_b_zero.append(data[1][0] * n)
    _save(figure, "inset_comp_effort_times_N_vs_N_b_star", inset_ns, at_b_star)
    _save(figure, "inset_comp_effort_times_N_vs_N_b_zero", inset_ns, at_b_zero)


def make_fig6():
    figure = "fig6"
    first_event, last_event = 0, 40000
    for name, positions_panel, index_panel in (("b0", "a", "b"), ("b_opt", "c", "d")):
        positions = _load(f"fig6/{name}/checkpoint_00_sample_of_event_particle_position.npy")
        active_index = _load(f"fig6/{name}/checkpoint_00_sample_of_event_active_particle_index.npy") + 1
        if len(positions) < last_event + 1:
            raise ValueError(f"fig6/{name} has {len(positions)} events but the figure needs {last_event + 1}; "
                             f"increase number_of_observations in its configuration file.")
        events = np.arange(first_event, last_event + 1) / 1e4
        for i in range(positions.shape[1]):
            _save(figure, f"{positions_panel}_{name}_particle_{i + 1}_position_vs_number_of_events_over_1e4", events,
                  positions[:, i][first_event:last_event + 1])
        _save(figure, f"{index_panel}_{name}_active_particle_index_vs_number_of_events_over_1e4", events,
              active_index[first_event:last_event + 1])


def make_fig7():
    figure = "fig7"
    velocity_file = "index_and_state_space_velocity_factor_field_prefactor_sweep.npy"

    def save_pair(tag, b, b_star, velocity, pressure):
        _save(figure, f"idx_space_velocity_over_v0_{tag}_vs_b_over_bstar", b / b_star, velocity)
        _save(figure, f"beta_pressure_{tag}_vs_b_over_bstar", b / b_star, pressure)

    def mean_potential(run):
        sample = _load(f"fig7/{run}/factor_field_prefactor_00/job_00/checkpoint_00_sample_of_potential.npy")
        return np.mean([np.mean(sample)])

    b, _, idx = _load(f"fig7/N16_L32_power2/{velocity_file}")[:, ::4]
    save_pair("p2_ghc_beta1_N16_L32", b, 2.0, idx, b - 32 * 1 / 16)
    b, _, idx = _load(f"fig7/N16_L32_power2_temp_2/{velocity_file}")[:, ::4]
    save_pair("p2_ghc_beta0p5_N16_L32", b, 2.0, idx / (0.5 * 1) ** -0.5,
              (.5 * b - .5 * 32 * 1 / 16) / (0.5 * 1) ** -0.5)
    b, _, idx = _load(f"fig7/N8_L8_sd/{velocity_file}")
    save_pair("sd_kappa2_beta1_N8_L8", b, 3.06, idx / 0.75,
              (-b + 8 / 8 * (1 + 2 / 8 * mean_potential("N8_L8_sd")) - 1 / 8) / 0.75)
    b, _, idx = _load(f"fig7/N8_L16_power4/{velocity_file}")
    save_pair("p4_ghc_beta1_N8_L16", b, 16.9 / 2, idx, b + 8 / 16 * (1 - 4.0 * mean_potential("N8_L16_power4") / 8))


def make_fig8():
    data = _load("fig8/mixing_time_events_results.npy")
    _save("fig8", "mixing_time_in_events_b_zero_vs_N", data[0][2:], data[1][2:])
    _save("fig8", "mixing_time_in_events_b_star_vs_N", data[0][2:], data[2][2:])


figure_makers = {"fig2": make_fig2, "fig3": make_fig3, "fig4": make_fig4, "fig6": make_fig6, "fig7": make_fig7,
                 "fig8": make_fig8}
