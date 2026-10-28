"""Prints the results of a grand-canonical experiment stage (used by run_experiment.sh and sulis_stage.sh).

Usage (from src):
    python config_files/grand_canonical/print_experiment_results.py bias <bias.json>
    python config_files/grand_canonical/print_experiment_results.py analysis <analysis.json>
"""
import json
import math
import os
import sys


def print_bias(file_name):
    bias = json.load(open(file_name))
    error = f" +- {bias['total_error']:.4f}" if bias.get("total_error") else ""
    print(f"Excess chemical potential beta mu_ex = eta(1) = {bias['total']:.4f}{error} (free-energy perturbation).")
    if "thermodynamic_integration_total" in bias:
        print(f"Excess chemical potential from thermodynamic integration = "
              f"{bias['thermodynamic_integration_total']:.4f}.")
    if "exact_total_hard_disks" in bias:
        print(f"Exact excess chemical potential (hard disks) = {bias['exact_total_hard_disks']:.4f}.")
    print(f"Smallest survival probability to the end of a calibration bin = "
          f"{min(record['survival_to_bin_end'] for record in bias['bins']):.3f}.")
    print(f"Suggested log activity (maximum of P(N) near N = {bias['number_of_real_particles']}) = "
          f"{bias['suggested_log_activity']:.4f}.")


def print_analysis(file_name):
    runs = json.load(open(file_name))
    runs = runs if isinstance(runs, list) else [runs]

    def name(run):
        return os.path.basename(os.path.dirname(os.path.normpath(run["run_directory"])))

    for run in runs:
        print(f"{name(run)} ({run['number_of_jobs']} jobs): <N> = {run['mean_number'][0]:.4f} +- "
              f"{run['mean_number'][1]:.4f}, Var N = {run['variance_number'][0]:.4f} +- "
              f"{run['variance_number'][1]:.4f}, tau_N = {run['iact_number_in_time'][0]:.4g}, kept fraction = "
              f"{run['physical_fraction'][0]:.3f}.")
        if "effort" in run:
            print(f"{name(run)} computational effort = {run['effort'][0]:.4g} +- {run['effort'][1]:.2g} proposed "
                  f"moves per independent sample ({run['effort_particle_events_only'][0]:.4g} from particle events).")
    exact = runs[0].get("exact")
    if exact:
        print(f"Exact (hard disks): <N> = {exact['mean_number']:.4f}, Var N = {exact['variance_number']:.4f}.")
        for run in runs:
            print(f"Deviation of {name(run)} from exact: <N> {run['mean_number_deviation_in_errors']:+.2f}, "
                  f"Var N {run['variance_number_deviation_in_errors']:+.2f} standard errors.")
    for run in runs[1:]:
        deviations = [f"{label} {(run[key][0] - runs[0][key][0]) / math.hypot(run[key][1], runs[0][key][1]):+.2f}"
                      for key, label in (("mean_number", "<N>"), ("variance_number", "Var N"))]
        print(f"Difference between {name(run)} and {name(runs[0])}: {deviations[0]}, {deviations[1]} standard "
              f"errors.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("bias", "analysis"):
        raise SystemExit(__doc__)
    (print_bias if sys.argv[1] == "bias" else print_analysis)(sys.argv[2])
