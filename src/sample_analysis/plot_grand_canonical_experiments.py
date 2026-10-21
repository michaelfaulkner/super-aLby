"""Summary figures and table of the grand-canonical tempering experiments.

Usage (from src):
    python sample_analysis/plot_grand_canonical_experiments.py [--root output/grand_canonical/experiments]
        [--figure-directory <dir>]

Reads <root>/<experiment>/analysis.json and bias.json (written by config_files/grand_canonical/run_experiment.sh) and
writes
    p_of_n.png       P(N) of every scheme (and the exact Tonks distribution for hard disks) for every experiment;
    efficiency.png   completed insertions per unit time and tau_N (in units of time) against density;
    bias.png         the calibrated eta(lambda) (g values = slopes) of every experiment;
    summary.json     one row per experiment and scheme;
    summary.md       the same as a markdown table.
"""
import argparse
import glob
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEME_STYLE = {"tempered_bias": ("firebrick", "-", "o", "tempered, calibrated g"),
                "direct": ("black", ":", "^", "direct insertion")}
FAMILIES = ["hard_disks_1d", "soft_disks_1d"]


def scheme_name(run_directory):
    return os.path.basename(os.path.dirname(os.path.normpath(run_directory)))


def load_experiments(root):
    experiments = []
    for directory in sorted(glob.glob(os.path.join(root, "*"))):
        analysis_file, bias_file = os.path.join(directory, "analysis.json"), os.path.join(directory, "bias.json")
        if not (os.path.isfile(analysis_file) and os.path.isfile(bias_file)):
            continue
        name = os.path.basename(directory)
        family = next((f for f in FAMILIES if name.startswith(f)), None)
        if family is None:
            continue
        density = float(name.split("_")[-1])
        with open(analysis_file) as f:
            runs = json.load(f)
        with open(bias_file) as f:
            bias = json.load(f)
        experiments.append({"name": name, "family": family, "density": density, "bias": bias,
                            "runs": {scheme_name(run["run_directory"]): run for run in runs}})
    return experiments


def plot_distributions(experiments, file_name):
    columns = 4
    rows = math.ceil(len(experiments) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows), squeeze=False)
    for axis, experiment in zip(axes.flat, experiments):
        for scheme, (colour, line, marker, label) in SCHEME_STYLE.items():
            run = experiment["runs"].get(scheme)
            if run is None:
                continue
            distribution = np.array(run["distribution"])
            numbers = np.arange(len(distribution))
            error = np.array(run["distribution_error"]) if run["distribution_error"] is not None else None
            axis.errorbar(numbers, distribution, yerr=error, color=colour, ls=line, marker=marker, ms=3, lw=1.2,
                          label=label)
        exact = next((run["exact"] for run in experiment["runs"].values() if "exact" in run), None)
        if exact is not None:
            distribution = np.array(exact["distribution"])
            axis.plot(np.arange(len(distribution)), distribution, color="grey", lw=4, alpha=0.4, zorder=0,
                      label="exact (Tonks)")
        axis.set_title(experiment["name"].replace("_", " "), fontsize=9)
        axis.set_xlabel("$N$")
        axis.set_ylabel("$P(N)$")
        mean = experiment["runs"]["direct"]["mean_number"][0] if "direct" in experiment["runs"] else None
        if mean is not None:
            spread = 5.0 * math.sqrt(max(experiment["runs"]["direct"]["variance_number"][0], 1.0))
            axis.set_xlim(mean - spread, mean + spread)
    for axis in list(axes.flat)[len(experiments):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(file_name, dpi=150)
    plt.close(figure)


def plot_efficiency(experiments, file_name):
    families = [family for family in FAMILIES if any(e["family"] == family for e in experiments)]
    figure, axes = plt.subplots(2, len(families), figsize=(4.4 * len(families), 7.0), squeeze=False)
    for column, family in enumerate(families):
        members = sorted([e for e in experiments if e["family"] == family], key=lambda e: e["density"])
        densities = [e["density"] for e in members]
        for scheme, (colour, line, marker, label) in SCHEME_STYLE.items():
            for row, key in enumerate(["insertions_per_unit_time", "iact_number_in_time"]):
                values = [e["runs"][scheme][key] if scheme in e["runs"] else [np.nan, np.nan] for e in members]
                axes[row, column].errorbar(densities, [v[0] for v in values], yerr=[v[1] for v in values],
                                           color=colour, ls=line, marker=marker, label=label)
        axes[0, column].set_yscale("log")
        axes[1, column].set_yscale("log")
        axes[0, column].set_title(family.replace("_", " "))
        axes[0, column].set_ylabel("accepted insertions per unit time")
        axes[1, column].set_ylabel(r"$\tau_N$ (time units)")
        axes[1, column].set_xlabel(r"density $\rho\sigma$")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(file_name, dpi=150)
    plt.close(figure)


def plot_biases(experiments, file_name):
    families = [family for family in FAMILIES if any(e["family"] == family for e in experiments)]
    figure, axes = plt.subplots(1, len(families), figsize=(4.4 * len(families), 3.6), squeeze=False)
    for column, family in enumerate(families):
        members = sorted([e for e in experiments if e["family"] == family], key=lambda e: e["density"])
        colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(members)))
        for colour, experiment in zip(colours, members):
            bias = experiment["bias"]
            axes[0, column].plot(bias["nodes"], bias["eta_at_nodes"], color=colour,
                                 label=f"{experiment['density']}: $\\eta(1)$ = {bias['total']:.2f}")
        axes[0, column].set_title(family.replace("_", " "))
        axes[0, column].set_xlabel(r"$\lambda$")
        axes[0, column].set_ylabel(r"$\eta(\lambda)$")
        axes[0, column].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(file_name, dpi=150)
    plt.close(figure)


def write_summary(experiments, directory):
    rows = []
    for experiment in experiments:
        bias = experiment["bias"]
        for scheme in SCHEME_STYLE:
            run = experiment["runs"].get(scheme)
            if run is None:
                continue
            """Independent check (no exact result needed): the canonical calibration at N_cal gives beta mu_ex(N_cal),
                hence the predicted ratio P(N_cal + 1) / P(N_cal) = z V exp(- beta mu_ex) / (N_cal + 1)."""
            calibration_number = bias["number_of_real_particles"]
            distribution = np.array(run["distribution"])
            distribution_error = np.array(run["distribution_error"]) if run["distribution_error"] else None
            measured_ratio, measured_ratio_error = None, None
            if calibration_number + 1 < len(distribution) and distribution[calibration_number] > 0.0:
                measured_ratio = float(distribution[calibration_number + 1] / distribution[calibration_number])
                if distribution_error is not None and distribution[calibration_number + 1] > 0.0:
                    measured_ratio_error = float(measured_ratio * math.sqrt(
                        (distribution_error[calibration_number] / distribution[calibration_number]) ** 2 +
                        (distribution_error[calibration_number + 1] / distribution[calibration_number + 1]) ** 2))
            predicted_ratio = math.exp(run["log_activity"] + math.log(run["volume"]) - bias["total"] -
                                       math.log(calibration_number + 1))
            predicted_ratio_error = predicted_ratio * bias["total_error"] if bias.get("total_error") else None
            rows.append({"experiment": experiment["name"], "scheme": scheme, "jobs": run["number_of_jobs"],
                         "calibration_number": calibration_number,
                         "measured_ratio": [measured_ratio, measured_ratio_error],
                         "predicted_ratio": [predicted_ratio, predicted_ratio_error],
                         "iact_number_seconds": run.get("iact_number_in_seconds"),
                         "insertions_per_second": run.get("insertions_per_second"),
                         "log_activity": run["log_activity"], "eta_1": bias["total"],
                         "eta_1_error": bias.get("total_error"), "eta_1_exact": bias.get("exact_total_tonks"),
                         "eta_1_thermodynamic_integration": bias.get("thermodynamic_integration_total"),
                         "mean_number": run["mean_number"], "variance_number": run["variance_number"],
                         "exact_mean_number": run["exact"]["mean_number"] if "exact" in run else None,
                         "mean_deviation_in_errors": run.get("mean_number_deviation_in_errors"),
                         "insertions_per_unit_time": run.get("insertions_per_unit_time"),
                         "insertion_acceptance": run.get("insertion_acceptance"),
                         "tau_number_time": run["iact_number_in_time"],
                         "physical_fraction": run["physical_fraction"],
                         "coupling_density_spread": run.get("coupling_density_relative_spread_on_0_1")})
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def fmt(pair, digits=3):
        if pair is None or pair[0] is None or not np.isfinite(pair[0]):
            return "-"
        if pair[1] is None or not np.isfinite(pair[1]):
            return f"{pair[0]:.{digits}f}"
        return f"{pair[0]:.{digits}f} ± {pair[1]:.{digits}f}"

    lines = ["| experiment | scheme | η(1) = βμ_ex (calibration) | ⟨N⟩ | exact ⟨N⟩ | Var N | "
             "P(N_cal+1)/P(N_cal) measured | predicted from calibration | insertions / time | insertions / s | "
             "τ_N (time) | τ_N (s) | λ-density spread on (0,1] | kept (λ ≤ 0) |", "|" + "---|" * 14]
    for row in rows:
        eta = f"{row['eta_1']:.3f}" + (f" ± {row['eta_1_error']:.3f}" if row["eta_1_error"] else "")
        if row["eta_1_exact"] is not None:
            eta += f" (exact {row['eta_1_exact']:.3f})"
        if row["eta_1_thermodynamic_integration"] is not None:
            eta += f" (TI {row['eta_1_thermodynamic_integration']:.3f})"
        exact_mean = "-" if row["exact_mean_number"] is None else f"{row['exact_mean_number']:.3f}"
        lines.append(f"| {row['experiment']} | {row['scheme']} | {eta} | {fmt(row['mean_number'])} | {exact_mean} | "
                     f"{fmt(row['variance_number'])} | {fmt(row['measured_ratio'])} | {fmt(row['predicted_ratio'])} | "
                     f"{fmt(row['insertions_per_unit_time'], 4)} | {fmt(row['insertions_per_second'], 1)} | "
                     f"{fmt(row['tau_number_time'], 1)} | {fmt(row['iact_number_seconds'], 3)} | "
                     f"{fmt(row['coupling_density_spread'], 2)} | {100.0 * row['physical_fraction'][0]:.1f}% |")
    with open(os.path.join(directory, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="output/grand_canonical/experiments")
    parser.add_argument("--figure-directory", default=None)
    args = parser.parse_args()
    directory = args.figure_directory or args.root
    os.makedirs(directory, exist_ok=True)
    experiments = load_experiments(args.root)
    plot_distributions(experiments, os.path.join(directory, "p_of_n.png"))
    plot_efficiency(experiments, os.path.join(directory, "efficiency.png"))
    plot_biases(experiments, os.path.join(directory, "bias.png"))
    write_summary(experiments, directory)
    print(f"Wrote figures and summary for {len(experiments)} experiments to {directory}")


if __name__ == "__main__":
    main()
