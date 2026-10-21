"""Analysis of grand-canonical EventChainMediator runs (any grand_canonical_kernel).

Usage (from src):
    python sample_analysis/grand_canonical_analysis.py <run_directory> [<run_directory> ...]
        [--exact poisson|tonks] [--output <file.json>]

Each run directory contains job_XX sub-directories (as produced by spawn_configs.py), or is itself a single job
directory.  For every job the script keeps the post-equilibration observations, selects the physical ones (ghost
coupling lambda <= 0, or no ghost), and computes P(N), <N>, Var N, <U>, the integrated autocorrelation time of N and
rates of completed insertions.  Results are pooled over jobs (mean and standard error over jobs).

Exact references:
    poisson: ideal gas, P(N) = (zV)^N exp(-zV) / N! (truncated to N >= 1 for the direct scheme).
    tonks:   1D hard disks of diameter sigma on a ring of length L, P(N) proportional to
             z^N L (L - N sigma)^(N - 1) / N! (N >= 1) and P(0) proportional to 1 (truncated to N >= 1 for the direct
             scheme).
"""
import argparse
import glob
import json
import math
import os
import sys
import numpy as np
from scipy.special import gammaln

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_directory)
from markov_chain_diagnostics import get_iact


def get_job_directories(run_directory):
    """Returns the job directories in run_directory (or run_directory itself if it holds a single job)."""
    if os.path.isfile(os.path.join(run_directory, "sim_params.json")):
        return [run_directory]
    return sorted(directory for directory in glob.glob(os.path.join(run_directory, "job_*"))
                  if os.path.isfile(os.path.join(directory, "sim_params.json")))


def load_job(job_directory):
    """Loads the post-equilibration samples and the parameters of a single job."""
    with open(os.path.join(job_directory, "sim_params.json")) as f:
        parameters = json.load(f)
    """EventChainMediator stores the grand-canonical parameters under "grand_canonical"."""
    if "grand_canonical" in parameters:
        parameters = dict(parameters["grand_canonical"], event_chain=parameters)
    parameters.setdefault("statistics", {})
    """Kernel statistics cover the whole run (equilibration included), so rates are per unit of the whole run's
        particle displacement.  Older runs of the standalone mediator recorded their own production-only total_time."""
    parameters["statistics"].setdefault("total_time", (parameters["number_of_equilibration_iterations"] +
                                                       parameters["number_of_observations"]) *
                                        parameters["distance_between_measurements"])
    start = parameters["number_of_equilibration_iterations"] + 1  # + 1 for the initial observation
    samples = {}
    for name in ["particle_number", "ghost_coupling", "potential", "ghost_profile"]:
        file_name = os.path.join(job_directory, f"checkpoint_00_sample_of_{name}.npy")
        if os.path.isfile(file_name):
            samples[name] = np.load(file_name)[start:, 0]
    return parameters, samples


def get_physical_mask(samples):
    coupling = samples.get("ghost_coupling")
    if coupling is None:
        return np.ones(len(samples["particle_number"]), dtype=bool)
    return np.isnan(coupling) | (coupling <= 0.0)


def get_exact_distribution(exact, parameters, maximum_number):
    """Returns the exact P(N) for N = 0..maximum_number (normalised), or None."""
    numbers = np.arange(maximum_number + 1)
    log_activity_volume = parameters["log_activity"] + math.log(parameters["volume"])
    if exact == "poisson":
        log_weights = numbers * log_activity_volume - gammaln(numbers + 1)
    elif exact == "tonks":
        potential = parameters["potential"]
        if potential["type"] != "hard_sphere":
            raise ValueError("The tonks reference needs hard disks.")
        length, sigma = potential["box"][0], potential["diameter"]
        log_weights = np.full(len(numbers), - np.inf)
        log_weights[0] = 0.0
        allowed = (numbers >= 1) & (numbers * sigma < length)
        n = numbers[allowed]
        log_weights[allowed] = (n * parameters["log_activity"] + math.log(length) +
                                (n - 1) * np.log(length - n * sigma) - gammaln(n + 1))
    else:
        return None
    if parameters["scheme"] == "direct":
        log_weights[0] = - np.inf
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / np.sum(weights)


def analyse_job(job_directory):
    parameters, samples = load_job(job_directory)
    numbers = samples["particle_number"]
    physical = get_physical_mask(samples)
    physical_numbers = numbers[physical].astype(int)
    statistics = parameters["statistics"]
    total_time = statistics["total_time"]
    try:
        iact_number = float(get_iact(numbers)) if np.std(numbers) > 0.0 else float("nan")
    except IndexError:
        iact_number = float("nan")
    result = {"job_directory": job_directory, "number_of_physical_observations": int(np.sum(physical)),
              "physical_fraction": float(np.mean(physical)),
              "mean_number": float(np.mean(physical_numbers)), "variance_number": float(np.var(physical_numbers)),
              "iact_number_in_observations": iact_number,
              "iact_number_in_time": iact_number * parameters["distance_between_measurements"],
              "histogram": np.bincount(physical_numbers).tolist()}
    """Wall-clock runtime (from run.log, written by config_files/grand_canonical/run_local.sh) per unit of simulated
        time, to convert tau_N into seconds.  Runs of a sweep share the machine, so this is only a rough cost."""
    log_file = os.path.join(job_directory, "run.log")
    if os.path.isfile(log_file):
        with open(log_file) as f:
            runtimes = [float(line.split("=")[1].split()[0]) for line in f if "Total runtime of the simulation" in line]
        if runtimes:
            simulated_time = ((parameters["number_of_equilibration_iterations"] + parameters["number_of_observations"])
                              * parameters["distance_between_measurements"])
            result["seconds_per_unit_time"] = runtimes[-1] / simulated_time
            result["iact_number_in_seconds"] = result["iact_number_in_time"] * result["seconds_per_unit_time"]
    if "potential" in samples:
        result["mean_energy"] = float(np.mean(samples["potential"][physical]))
        result["mean_energy_per_particle"] = float(np.mean(samples["potential"][physical] /
                                                           np.maximum(physical_numbers, 1)))
    if parameters["scheme"] == "tempered":
        result["insertions_per_unit_time"] = statistics["insertion_acceptances"] / total_time
        result["deletions_per_unit_time"] = statistics["deletion_acceptances"] / total_time
        result["insertion_acceptance"] = (statistics["insertion_acceptances"] /
                                          max(statistics["insertion_attempts"], 1))
        result["deletion_acceptance"] = (statistics["deletion_acceptances"] /
                                         max(statistics["deletion_attempts"] -
                                             statistics["deletion_null_proposals"], 1))
        result["upward_passages_per_unit_time"] = statistics["upward_passages"] / total_time
        """Flatness of the lambda marginal on (0, 1], from the observations of lambda (40 bins)."""
        couplings = samples.get("ghost_coupling")
        if couplings is not None and np.any(couplings > 0.0):
            density, _ = np.histogram(couplings[couplings > 0.0], bins=40, range=(0.0, 1.0))
            result["coupling_density_relative_spread_on_0_1"] = float(np.std(density) / np.mean(density))
    elif parameters["scheme"] == "direct":
        result["insertions_per_unit_time"] = statistics["direct_insertion_acceptances"] / total_time
        result["deletions_per_unit_time"] = statistics["direct_deletion_acceptances"] / total_time
        result["insertion_acceptance"] = (statistics["direct_insertion_acceptances"] /
                                          max(statistics["direct_insertion_attempts"], 1))
        result["deletion_acceptance"] = (statistics["direct_deletion_acceptances"] /
                                         max(statistics["direct_deletion_attempts"] -
                                             statistics["direct_deletion_null_proposals"], 1))
    if "seconds_per_unit_time" in result and "insertions_per_unit_time" in result:
        result["insertions_per_second"] = result["insertions_per_unit_time"] / result["seconds_per_unit_time"]
    return parameters, result


def mean_and_error(values):
    values = np.array([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    if len(values) == 1:
        return [float(values[0]), float("nan")]
    return [float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))]


def analyse_run(run_directory, exact=None):
    """Analyses all jobs of a run and pools them."""
    job_directories = get_job_directories(run_directory)
    if not job_directories:
        raise FileNotFoundError(f"No finished jobs found in {run_directory}")
    analyses = [analyse_job(job_directory) for job_directory in job_directories]
    parameters = analyses[0][0]
    results = [result for _, result in analyses]
    pooled = {"run_directory": run_directory, "number_of_jobs": len(results), "scheme": parameters["scheme"],
              "log_activity": parameters["log_activity"], "volume": parameters["volume"],
              "potential": parameters["potential"], "bias_total": parameters.get("bias", {}).get("total"),
              "bias_class": parameters.get("bias", {}).get("class")}
    for key in results[0]:
        if key in ["job_directory", "histogram"]:
            continue
        pooled[key] = mean_and_error([result.get(key) for result in results])
    maximum_number = max(len(result["histogram"]) for result in results) - 1
    histograms = np.array([np.pad(result["histogram"], (0, maximum_number + 1 - len(result["histogram"])))
                           for result in results], dtype=float)
    distributions = histograms / np.sum(histograms, axis=1, keepdims=True)
    pooled["distribution"] = np.mean(distributions, axis=0).tolist()
    pooled["distribution_error"] = (np.std(distributions, axis=0, ddof=1) / math.sqrt(len(results))).tolist() \
        if len(results) > 1 else None
    if exact is not None:
        exact_distribution = get_exact_distribution(exact, parameters, maximum_number + 10)
        numbers = np.arange(len(exact_distribution))
        exact_mean = float(np.sum(numbers * exact_distribution))
        exact_variance = float(np.sum(numbers ** 2 * exact_distribution) - exact_mean ** 2)
        pooled["exact"] = {"type": exact, "mean_number": exact_mean, "variance_number": exact_variance,
                           "distribution": exact_distribution.tolist()}
        pooled["mean_number_deviation_in_errors"] = ((pooled["mean_number"][0] - exact_mean) /
                                                     pooled["mean_number"][1])
        pooled["variance_number_deviation_in_errors"] = ((pooled["variance_number"][0] - exact_variance) /
                                                         pooled["variance_number"][1])
        """Chi-squared of the pooled histogram against the exact distribution (bins with expected count >= 5)."""
        counts = np.sum(histograms, axis=0)
        total = np.sum(counts)
        expected = exact_distribution[:len(counts)] * total
        mask = expected >= 5.0
        effective = total / max(pooled["iact_number_in_observations"][0], 1.0)
        chi_squared = float(np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask]) * effective / total)
        pooled["chi_squared_iact_corrected"] = chi_squared
        pooled["chi_squared_degrees_of_freedom"] = int(np.sum(mask)) - 1
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directories", nargs="+")
    parser.add_argument("--exact", choices=["poisson", "tonks"], default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summaries = [analyse_run(run_directory, args.exact) for run_directory in args.run_directories]
    for summary in summaries:
        line = (f"{summary['run_directory']}: scheme={summary['scheme']} jobs={summary['number_of_jobs']} "
                f"<N>={summary['mean_number'][0]:.4f}+-{summary['mean_number'][1]:.4f} "
                f"VarN={summary['variance_number'][0]:.4f}+-{summary['variance_number'][1]:.4f}")
        if "exact" in summary:
            exact = summary['exact']
            line += (f" | exact <N>={exact['mean_number']:.4f} VarN={exact['variance_number']:.4f}"
                     f" dev={summary['mean_number_deviation_in_errors']:+.2f}sig/"
                     f"{summary['variance_number_deviation_in_errors']:+.2f}sig")
        if "insertions_per_unit_time" in summary:
            line += f" | insertions/time={summary['insertions_per_unit_time'][0]:.4g}"
        line += f" | tau_N(time)={summary['iact_number_in_time'][0]:.4g}"
        line += f" | kept (lambda <= 0)={100.0 * summary['physical_fraction'][0]:.1f}%"
        print(line)
    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summaries if len(summaries) > 1 else summaries[0], f, indent=1)


if __name__ == "__main__":
    main()
