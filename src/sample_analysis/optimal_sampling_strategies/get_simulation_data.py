import glob
import numpy as np
import os
import subprocess
import sys

this_directory = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.abspath(this_directory + "/../../")
output_directory = os.path.join(src_directory, "output", "optimal_sampling_strategies_figs")
config_directory = os.path.join(src_directory, "config_files", "optimal_sampling_strategies_figs")

iact = "structure_factor_iact_vs_prefactor.npy"
mean_event_rate = "mean_event_rate_sweep.npy"
acceptance_rate = "acceptance_rate_sweep.npy"
comp_effort = "comp_effort_sweep.npy"
velocities = "index_and_state_space_velocity_factor_field_prefactor_sweep.npy"
mixing_times_in_events = "mixing_time_events_results.npy"

mixing_time_scripts = {
    mixing_times_in_events: "get_mixing_time_events_figure.py",
}

sample_analysis_scripts = {
    iact: ("../plot_iact_vs_prefactor.py", ["structure_factor"], []),
    mean_event_rate: ("../plot_mean_event_rate_sweep.py", [], []),
    acceptance_rate: ("../plot_acceptance_rate_vs_prefactor.py", [], []),
    comp_effort: ("../plot_comp_effort_vs_prefactor.py", [], [iact, mean_event_rate, acceptance_rate]),
    velocities: ("../plot_index_and_state_space_velocity_sweep.py", [], []),
}


def get_simulation_data(path):
    path = path if os.path.isabs(path) else os.path.join(src_directory, path)
    make_simulation_data(path)
    return np.load(path)


def make_simulation_data(path):
    if os.path.exists(path):
        return
    file_name, run_directory = os.path.basename(path), os.path.dirname(path)
    if file_name in mixing_time_scripts:
        script = mixing_time_scripts[file_name]
        check_simulations_have_run(run_directory, "b0_temp_1_N_*/temperature_00/job_*",
                                   f"each bash file in {_relative_to_src(config_directory)}/fig8")
        _run_script(script, [])
    elif file_name in sample_analysis_scripts:
        script, arguments, prerequisites = sample_analysis_scripts[file_name]
        config_file = get_config_file(run_directory)
        check_simulations_have_run(run_directory, "*_00/job_*", f"{_relative_to_src(config_file[:-4])}.sh")
        for prerequisite in prerequisites:
            make_simulation_data(os.path.join(run_directory, prerequisite))
        _run_script(script, [config_file] + arguments)
    else:
        raise FileNotFoundError(f"{path} does not exist.  Run the bash file or configuration file that makes it (see "
                                f"the README) before running this script.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{script} did not make {path}.")


def _relative_to_src(path):
    return os.path.relpath(path, src_directory)


def get_config_file(run_directory):
    config_file = os.path.join(config_directory, os.path.relpath(run_directory, output_directory)) + ".ini"
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Cannot find the configuration file {config_file} that makes the data in "
                                f"{run_directory}.")
    return config_file


def check_simulations_have_run(run_directory, pattern, bash_file):
    if not glob.glob(os.path.join(run_directory, pattern)):
        raise FileNotFoundError(f"There is no simulation output in {run_directory}.  First run {bash_file} (see the "
                                f"README).")


def _run_script(script, arguments):
    shown_arguments = [_relative_to_src(a) if os.path.isabs(a) else a for a in arguments]
    script_path = os.path.normpath(os.path.join(this_directory, script))
    print(f"Running {_relative_to_src(script_path)} {' '.join(shown_arguments)}", flush=True)
    subprocess.run([sys.executable, script_path] + arguments, cwd=src_directory, check=True,
                   env={**os.environ, "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"}, stdout=subprocess.DEVNULL)
