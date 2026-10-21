"""Computes the ghost-coupling bias (the g values) from fixed-coupling calibration runs.

Usage (from src):
    python sample_analysis/compute_ghost_coupling_bias.py <calibration_directory> --output <bias.json>
        [--sub-bins 4] [--target-number-of-particles N] [--figure <file.png>]

<calibration_directory> holds one sub-directory per calibration coupling (e.g. ghost_coupling_XX, as produced by
spawn_configs.py), each holding job_YY directories of EventChainMediator runs with fixed_ghost_coupling_kernel
and a GhostProfileSampler.  The couplings lambda_0 = 0 < lambda_1 < ... < lambda_{K-1} < 1 define the bins
[lambda_k, lambda_{k+1}] (lambda_K = 1).

This is the analogue of pre-computing <U>_beta for temperature-augmented ECMC.  The bias eta(lambda) should be the
dimensionless free energy beta F(lambda) of growing the ghost, so that the marginal of lambda is flat.  Within bin k, the
run at lambda_k gives the free-energy change to any lambda in the bin by free-energy perturbation (Zwanzig):
    eta(lambda) - eta(lambda_k) = - ln < exp(- beta [U_g(lambda) - U_g(lambda_k)]) >_{lambda_k},
where U_g(lambda) = lambda^n A for soft spheres (A is the recorded profile) and, for hard spheres, the average is the
probability that the ghost clearance c (the recorded profile) is at least lambda.  eta is evaluated on --sub-bins
equally spaced points per bin and its slopes g on the resulting fine grid are written to the output file, which
TabulatedGhostCouplingBias reads.  eta(1) is the estimated beta * mu_ex of the calibration system; the suggested
log_activity ln((N + 1) / V) + eta(1) puts the maximum of P(N) near the calibration N (insertion and deletion
acceptances near 1).  For soft spheres the thermodynamic-integration estimate (trapezoid of beta <dU_g/d lambda>) is
also reported as a check.
"""
import argparse
import glob
import json
import math
import os
import sys
import numpy as np

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_directory)
from grand_canonical_analysis import load_job


def get_ghost_energy_difference(potential, profile, coupling, new_coupling):
    if potential["type"] == "soft_sphere":
        return (new_coupling ** potential["power"] - coupling ** potential["power"]) * profile
    return np.where(profile >= new_coupling, 0.0, np.inf)


def load_calibration(calibration_directory):
    """Returns a list of (coupling, parameters, [profile sample of each job], [number of particles of each job])."""
    increment_directories = sorted(glob.glob(os.path.join(calibration_directory, "*_[0-9][0-9]")))
    if not increment_directories:
        increment_directories = [calibration_directory]
    calibration = []
    for increment_directory in increment_directories:
        job_directories = sorted(glob.glob(os.path.join(increment_directory, "job_*")))
        job_directories = [job for job in job_directories if os.path.isfile(os.path.join(job, "sim_params.json"))]
        if not job_directories:
            continue
        profiles, numbers, parameters = [], [], None
        for job_directory in job_directories:
            parameters, samples = load_job(job_directory)
            if parameters["scheme"] != "calibration":
                raise ValueError(f"{job_directory} is not a calibration run.")
            profiles.append(samples["ghost_profile"])
            numbers.append(int(samples["particle_number"][0]))
        calibration.append((parameters["fixed_ghost_coupling"], parameters, profiles, numbers))
    calibration.sort(key=lambda item: item[0])
    return calibration


def compute_bias(calibration, sub_bins):
    couplings = np.array([item[0] for item in calibration])
    if abs(couplings[0]) > 1.0e-12 or couplings[-1] >= 1.0:
        raise ValueError("The calibration couplings must start at 0 and stay below 1.")
    parameters = calibration[0][1]
    potential = parameters["potential"]
    beta = 1.0 / parameters["temperature"]
    bin_edges = np.append(couplings, 1.0)
    number_of_jobs = min(len(item[2]) for item in calibration)
    nodes, eta = [0.0], [0.0]
    eta_per_job = np.zeros(number_of_jobs)
    eta_per_job_at_nodes = [np.zeros(number_of_jobs)]
    bins = []
    for k, (coupling, _, profiles, numbers) in enumerate(calibration):
        sub_nodes = np.linspace(bin_edges[k], bin_edges[k + 1], sub_bins + 1)[1:]
        pooled = np.concatenate(profiles)
        weights = np.exp(- beta * get_ghost_energy_difference(potential, pooled[:, None], coupling,
                                                              sub_nodes[None, :]))
        survival = np.mean(weights, axis=0)
        if np.any(survival <= 0.0):
            raise ValueError(f"No calibration sample at lambda = {coupling} survives to the end of its bin.  Use a finer "
                             f"calibration grid.")
        job_survival = np.array([np.mean(np.exp(- beta * get_ghost_energy_difference(
            potential, profile[:, None], coupling, sub_nodes[None, :])), axis=0) for profile in profiles[:number_of_jobs]])
        base = eta[-1]
        for j, sub_node in enumerate(sub_nodes):
            nodes.append(float(sub_node))
            eta.append(float(base - math.log(survival[j])))
        with np.errstate(divide='ignore'):
            job_increments = - np.log(job_survival[:, -1])
        eta_per_job = eta_per_job + job_increments
        bin_record = {"coupling": float(coupling), "bin_end": float(bin_edges[k + 1]),
                      "number_of_samples": int(len(pooled)), "survival_to_bin_end": float(survival[-1]),
                      "eta_increment": float(- math.log(survival[-1])), "number_of_real_particles": numbers[0]}
        if potential["type"] == "soft_sphere":
            bin_record["thermodynamic_integration_slope"] = float(
                beta * np.mean(potential["power"] * coupling ** (potential["power"] - 1.0) * pooled)) \
                if coupling > 0.0 else 0.0
        bins.append(bin_record)
    nodes = np.array(nodes)
    eta = np.array(eta)
    slopes = np.diff(eta) / np.diff(nodes)
    finite_jobs = eta_per_job[np.isfinite(eta_per_job)]
    result = {"nodes": nodes.tolist(), "slopes": slopes.tolist(), "eta_at_nodes": eta.tolist(),
              "total": float(eta[-1]),
              "total_error": float(np.std(finite_jobs, ddof=1) / math.sqrt(len(finite_jobs)))
              if len(finite_jobs) > 1 else None,
              "method": "free_energy_perturbation", "sub_bins": sub_bins,
              "calibration_couplings": couplings.tolist(), "bins": bins,
              "number_of_real_particles": calibration[0][3][0], "volume": parameters["volume"],
              "temperature": parameters["temperature"], "potential": potential}
    if potential["type"] == "soft_sphere":
        ti_couplings = np.append(couplings, 1.0)
        ti_slopes = [record["thermodynamic_integration_slope"] for record in bins]
        """The slope at lambda = 1 is not measured: the last trapezoid uses the FEP increment of the last bin instead."""
        result["thermodynamic_integration_total"] = float(np.trapezoid(ti_slopes, couplings) +
                                                          bins[-1]["eta_increment"])
        result["thermodynamic_integration_slopes"] = ti_slopes
    if potential["type"] == "hard_sphere":
        length, sigma, n = potential["box"][0], potential["diameter"], calibration[0][3][0]
        result["exact_total_tonks"] = float(- (n * math.log(length - (n + 1) * sigma) - math.log(length) -
                                               (n - 1) * math.log(length - n * sigma)))
    return result


def plot_bias(result, file_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nodes, eta, slopes = np.array(result["nodes"]), np.array(result["eta_at_nodes"]), np.array(result["slopes"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(nodes, eta, 'k-')
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$\eta(\lambda)$")
    title = rf"$\eta(1) = {result['total']:.3f}$"
    if result.get("total_error"):
        title += rf"$\pm {result['total_error']:.3f}$"
    if "exact_total_tonks" in result:
        title += rf" (exact {result['exact_total_tonks']:.3f})"
    if "thermodynamic_integration_total" in result:
        title += rf" (TI {result['thermodynamic_integration_total']:.3f})"
    axes[0].set_title(title)
    axes[1].stairs(slopes, nodes, color='firebrick')
    if "thermodynamic_integration_slopes" in result:
        axes[1].plot(result["calibration_couplings"], result["thermodynamic_integration_slopes"], 'ko', ms=4,
                     label=r"$\beta\langle \partial_\lambda U_g\rangle$ (TI)")
        axes[1].legend()
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$g(\lambda) = \eta'(\lambda)$")
    survival = [record["survival_to_bin_end"] for record in result["bins"]]
    axes[2].semilogy(result["calibration_couplings"], survival, 'ko-')
    axes[2].set_xlabel(r"$\lambda_k$")
    axes[2].set_ylabel(r"$\langle e^{-\beta \Delta U_g}\rangle_{\lambda_k}$ to bin end")
    figure.tight_layout()
    figure.savefig(file_name, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("calibration_directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sub-bins", type=int, default=4)
    parser.add_argument("--target-number-of-particles", type=int, default=None)
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()
    result = compute_bias(load_calibration(args.calibration_directory), args.sub_bins)
    target = result["number_of_real_particles"] if args.target_number_of_particles is None \
        else args.target_number_of_particles
    result["suggested_log_activity"] = float(math.log((target + 1) / result["volume"]) + result["total"])
    result["suggested_log_activity_target_number_of_particles"] = target
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=1)
    message = f"eta(1) = beta mu_ex = {result['total']:.4f}"
    if result["total_error"] is not None:
        message += f" +- {result['total_error']:.4f}"
    if "exact_total_tonks" in result:
        message += f" (exact Tonks {result['exact_total_tonks']:.4f})"
    if "thermodynamic_integration_total" in result:
        message += f" (TI {result['thermodynamic_integration_total']:.4f})"
    message += (f"; min bin survival = {min(record['survival_to_bin_end'] for record in result['bins']):.3g}; "
                f"suggested log_activity (N = {target}) = {result['suggested_log_activity']:.4f}")
    print(message)
    if args.figure is not None:
        plot_bias(result, args.figure)


if __name__ == "__main__":
    main()
