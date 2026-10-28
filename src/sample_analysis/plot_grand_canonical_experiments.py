"""Summary figures and table of the grand-canonical tempering experiments.

Usage (from src):
    python sample_analysis/plot_grand_canonical_experiments.py [--root output/grand_canonical/experiments]
        [--figure-directory <dir>]

Reads <root>/<experiment>/analysis.json and bias.json (written by config_files/grand_canonical/run_experiment.sh) and
writes
    p_of_n.png       P(N) of every scheme (and the exact distribution for hard disks) for every experiment, plus one
                     p_of_n.png in each experiment's folder;
    efficiency.png   completed insertions per unit time and tau_N (in units of time) against density;
    bias.png         the calibrated eta(lambda) (g values = slopes) of every experiment;
    summary.json     one row per experiment and scheme;
    summary.md       the same as a markdown table.
For every experiment (any folder below root with an analysis.json) it also writes, from the physical samples:
    <scheme>/cdf_<array>.png      the CDF of each sample array (mean over jobs);
    cdf_comparison_<array>.png    the CDFs of all schemes on the same axes, with the Kolmogorov-Smirnov p-value of each
                                  scheme against the first;
    cdf_comparison.json           the Kolmogorov-Smirnov statistics, effective sample sizes and p-values.
"""
import argparse
import glob
import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.ticker
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grand_canonical_analysis import get_job_directories, get_physical_mask, load_job  # noqa: E402
from markov_chain_diagnostics import get_iact  # noqa: E402

SCHEME_STYLE = {"tempered": ("firebrick", "-", "o", "tempered"),
                "direct": ("black", ":", "^", "direct"),
                "metropolis": ("royalblue", "-", "s", "Metropolis")}
"""Colours and markers for any other scheme, in order of appearance (colour-blind safe)."""
EXTRA_COLOURS = ["#009e73", "#e69f00", "#cc79a7", "#56b4e9", "#d55e00", "#7f7f7f"]
EXTRA_MARKERS = ["D", "v", "P", "X", "*", "h"]
FAMILIES = ["hard_disks_1d", "soft_disks_1d"]
"""The ghost's own samplers are not physical observables, so they get no CDF."""
EXCLUDED_SAMPLERS = ("ghost_coupling", "ghost_profile")
CDF_POINTS = 400
AXIS_LABELS = {"particle_number": "$N$", "structure_factor": "$S(q)$", "potential": "$U$"}
"""Figure style of the factor-field paper figures (.dev_nbs/October 26/wc_05_10_26/factor_fields_box)."""
PAPER_STYLE = {"font.size": 18, "axes.linewidth": 3.0, "xtick.major.width": 3.0, "ytick.major.width": 3.0,
               "xtick.major.size": 8, "ytick.major.size": 8, "xtick.minor.width": 2.0, "ytick.minor.width": 2.0}
PANEL_SIZE = (7.6, 6.2)


def style_axes(axes, x_label, y_label, legend_location="best", legend_title=None, title=None):
    """Applies the paper style to one set of axes: ticks, labels, grid, framed legend and semibold text."""
    axes.minorticks_off()
    axes.tick_params(axis="both", which="major", labelsize=20, pad=6)
    axes.set_xlabel(x_label, fontsize=24)
    axes.set_ylabel(y_label, fontsize=22)
    axes.grid(True, which="major", color="0.88", lw=1.0, zorder=0)
    if title is not None:
        axes.set_title(title, fontsize=18)
    if axes.get_legend_handles_labels()[0]:
        legend = axes.legend(loc=legend_location, fontsize=16, framealpha=1.0, edgecolor="black", borderpad=0.5,
                             handlelength=1.4, title=legend_title, title_fontsize=14)
        legend.get_frame().set_linewidth(3.0)
    # Half-strength emboldening: a thin outline stroke on the normal-weight text
    semibold = [path_effects.withStroke(linewidth=0.9, foreground="black")]
    for text_label in axes.get_xticklabels() + axes.get_yticklabels() + [axes.xaxis.label, axes.yaxis.label]:
        text_label.set_path_effects(semibold)


def save_figure(figure, file_name):
    figure.tight_layout()
    figure.savefig(file_name, dpi=200)
    plt.close(figure)


def get_scheme_style(scheme, schemes):
    """(colour, line style, marker, label) of a scheme; schemes not in SCHEME_STYLE get the next extra colour."""
    if scheme in SCHEME_STYLE:
        return SCHEME_STYLE[scheme]
    index = [other for other in schemes if other not in SCHEME_STYLE].index(scheme) % len(EXTRA_COLOURS)
    return EXTRA_COLOURS[index], "-", EXTRA_MARKERS[index], scheme


def get_schemes(runs_by_experiment):
    """The schemes of a list of {scheme: run} dictionaries, the known ones first and the rest in order of appearance."""
    found = [scheme for runs in runs_by_experiment for scheme in runs]
    return [scheme for scheme in SCHEME_STYLE if scheme in found] + list(
        dict.fromkeys(scheme for scheme in found if scheme not in SCHEME_STYLE))


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
        if family is None or "_density_" not in name:  # density sweeps only
            continue
        density = float(name.split("_")[-1])
        with open(analysis_file) as f:
            runs = json.load(f)
        with open(bias_file) as f:
            bias = json.load(f)
        experiments.append({"name": name, "directory": directory, "family": family, "density": density, "bias": bias,
                            "runs": {scheme_name(run["run_directory"]): run for run in runs}})
    return experiments


def plot_distribution(axes, experiment):
    """P(N) of every scheme of one experiment, with the exact distribution for hard disks."""
    schemes = get_schemes([experiment["runs"]])
    for scheme in schemes:
        colour, line, marker, label = get_scheme_style(scheme, schemes)
        run = experiment["runs"].get(scheme)
        if run is None:
            continue
        distribution = np.array(run["distribution"])
        error = np.array(run["distribution_error"]) if run["distribution_error"] is not None else None
        axes.errorbar(np.arange(len(distribution)), distribution, yerr=error, color=colour, ls=line, marker=marker,
                      ms=8, lw=2.5, label=label, zorder=3)
    exact = next((run["exact"] for run in experiment["runs"].values() if "exact" in run), None)
    if exact is not None:
        distribution = np.array(exact["distribution"])
        axes.plot(np.arange(len(distribution)), distribution, color="grey", lw=7, alpha=0.4, zorder=2, label="exact")
    reference = experiment["runs"].get("direct") or next(iter(experiment["runs"].values()))
    spread = 5.0 * math.sqrt(max(reference["variance_number"][0], 1.0))
    axes.set_xlim(reference["mean_number"][0] - spread, reference["mean_number"][0] + spread)
    axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


def plot_distributions(experiments, file_name):
    """Writes the overview of every experiment to file_name and one P(N) figure into each experiment's folder."""
    with plt.rc_context(PAPER_STYLE):
        for experiment in experiments:
            figure, axes = plt.subplots(figsize=PANEL_SIZE)
            plot_distribution(axes, experiment)
            style_axes(axes, "$N$", "$P(N)$", "upper right")
            save_figure(figure, os.path.join(experiment["directory"], "p_of_n.png"))
        columns = min(len(experiments), 4)
        rows = math.ceil(len(experiments) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(PANEL_SIZE[0] * columns, PANEL_SIZE[1] * rows),
                                    squeeze=False)
        for panel, experiment in zip(axes.flat, experiments):
            plot_distribution(panel, experiment)
            style_axes(panel, "$N$", "$P(N)$", "upper right", title=experiment["name"].replace("_", " "))
        for panel in list(axes.flat)[len(experiments):]:
            panel.axis("off")
        save_figure(figure, file_name)

def plot_efficiency(experiments, file_name):
    families = [family for family in FAMILIES if any(e["family"] == family for e in experiments)]
    with plt.rc_context(PAPER_STYLE):
        figure, axes = plt.subplots(2, len(families), figsize=(PANEL_SIZE[0] * len(families), PANEL_SIZE[1] * 2),
                                    squeeze=False)
        for column, family in enumerate(families):
            members = sorted([e for e in experiments if e["family"] == family], key=lambda e: e["density"])
            densities = [e["density"] for e in members]
            schemes = get_schemes([e["runs"] for e in members])
            for scheme in schemes:
                colour, line, marker, label = get_scheme_style(scheme, schemes)
                for row, key in enumerate(["insertions_per_unit_time", "iact_number_in_time"]):
                    values = [e["runs"][scheme][key] if scheme in e["runs"] else [np.nan, np.nan] for e in members]
                    axes[row, column].errorbar(densities, [v[0] for v in values], yerr=[v[1] for v in values],
                                               color=colour, ls=line, marker=marker, ms=9, lw=2.5, label=label)
            for row in range(2):
                axes[row, column].set_yscale("log")
            style_axes(axes[0, column], r"density $\rho\sigma$", "insertions per unit time", "lower left",
                       title=family.replace("_", " "))
            style_axes(axes[1, column], r"density $\rho\sigma$", r"$\tau_N$ (time units)")
            axes[1, column].get_legend().remove()
        save_figure(figure, file_name)

def plot_biases(experiments, file_name):
    families = [family for family in FAMILIES if any(e["family"] == family for e in experiments)]
    with plt.rc_context(PAPER_STYLE):
        figure, axes = plt.subplots(1, len(families), figsize=(PANEL_SIZE[0] * len(families), PANEL_SIZE[1]),
                                    squeeze=False)
        for column, family in enumerate(families):
            members = sorted([e for e in experiments if e["family"] == family], key=lambda e: e["density"])
            colours = plt.cm.viridis(np.linspace(0.0, 0.85, len(members)))
            for colour, experiment in zip(colours, members):
                bias = experiment["bias"]
                axes[0, column].plot(bias["nodes"], bias["eta_at_nodes"], color=colour, lw=2.5,
                                     label=f"{experiment['density']}: $\\eta(1)$ = {bias['total']:.2f}")
            style_axes(axes[0, column], r"$\lambda$", r"$\eta(\lambda)$", "lower right",
                       title=family.replace("_", " "))
        save_figure(figure, file_name)

def write_summary(experiments, directory):
    rows = []
    for experiment in experiments:
        bias = experiment["bias"]
        for scheme in get_schemes([experiment["runs"]]):
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
                         "eta_1_error": bias.get("total_error"), "eta_1_exact": bias.get("exact_total_hard_disks"),
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


def load_physical_observables(job_directory):
    """Returns {sample array name: physical post-equilibration samples} for one job (one entry per column)."""
    parameters, samples = load_job(job_directory)
    start = parameters["number_of_equilibration_iterations"] + 1
    physical = get_physical_mask(samples)
    observables = {}
    for file_name in sorted(glob.glob(os.path.join(job_directory, "checkpoint_00_sample_of_*.npy"))):
        name = os.path.basename(file_name)[len("checkpoint_00_sample_of_"):-len(".npy")]
        if name in EXCLUDED_SAMPLERS:
            continue
        values = np.load(file_name)[start:]
        values = values.reshape(len(values), -1)
        if len(values) != len(physical):
            continue
        values = values[physical]
        for column in range(values.shape[1]):
            observables[name if values.shape[1] == 1 else f"{name}_{column}"] = values[:, column]
    return observables


def find_production_runs(experiment_directory):
    """Returns {scheme: run directory} for the production runs listed in the experiment's analysis.json."""
    with open(os.path.join(experiment_directory, "analysis.json")) as f:
        analysed = json.load(f)
    analysed = analysed if isinstance(analysed, list) else [analysed]
    return {scheme_name(run["run_directory"]): run["run_directory"] for run in analysed
            if os.path.isdir(run["run_directory"])}


def get_cdf_grid(samples):
    """The distinct values of a discrete sample (e.g. N), else CDF_POINTS points from its 0.1% to 99.9% quantile."""
    pooled = np.concatenate(samples)
    distinct = np.unique(pooled)
    if len(distinct) <= CDF_POINTS:
        return distinct
    return np.linspace(*np.quantile(pooled, [0.001, 0.999]), CDF_POINTS)


def get_mean_cdf(job_samples, grid):
    """Empirical CDF of every job on the grid, averaged over jobs."""
    return np.mean([np.searchsorted(np.sort(sample), grid, side="right") / len(sample) for sample in job_samples],
                   axis=0)


def get_effective_sample_size(job_samples):
    """Sum over jobs of (samples / integrated autocorrelation time): the number of independent samples."""
    size = 0.0
    for sample in job_samples:
        try:
            iact = max(float(get_iact(sample)), 1.0) if np.std(sample) > 0.0 else float(len(sample))
        except IndexError:  # the autocorrelation never drops below the cutoff
            iact = float(len(sample))
        size += len(sample) / iact
    return size


def get_ks_p_value(statistic, first_size, second_size):
    """Two-sample Kolmogorov-Smirnov p-value (asymptotic) for effective sample sizes; conservative for discrete data."""
    from scipy.stats import kstwobign
    return float(kstwobign.sf(statistic * math.sqrt(first_size * second_size / (first_size + second_size))))


def plot_cdf(curves, name, file_name, annotation=None):
    """One figure per sample array; curves[scheme] = (grid, mean CDF)."""
    with plt.rc_context(PAPER_STYLE):
        figure, axes = plt.subplots(figsize=PANEL_SIZE)
        schemes = get_schemes([curves])
        for index, scheme in enumerate(schemes):
            grid, mean = curves[scheme]
            colour, line, _, label = get_scheme_style(scheme, schemes)
            # Each later line is thinner, so curves that agree still show every colour
            axes.step(grid, mean, where="post", color=colour, ls=line, lw=max(6.0 - 2.5 * index, 1.5), label=label)
        if name == "particle_number":
            axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        style_axes(axes, AXIS_LABELS.get(name, name.replace("_", " ")), "CDF", "lower right", annotation)
        save_figure(figure, file_name)

def plot_cdfs(experiment_directory):
    """Writes <scheme>/cdf_<array>.png, cdf_comparison_<array>.png and cdf_comparison.json for one experiment."""
    runs = find_production_runs(experiment_directory)
    if not runs:
        return False
    # Physical samples of every job of every scheme
    observables = {scheme: [load_physical_observables(job) for job in get_job_directories(run)]
                   for scheme, run in runs.items()}
    # Sample arrays present in every job of a scheme and not constant (e.g. the potential of hard disks)
    names = {scheme: sorted(name for name in set.intersection(*[set(job) for job in jobs])
                            if len(np.unique(np.concatenate([job[name] for job in jobs]))) > 1)
             for scheme, jobs in observables.items()}
    # One grid per sample array, shared by the schemes so that their CDFs can be compared point by point
    all_names = sorted(set().union(*names.values()))
    grids = {name: get_cdf_grid([job[name] for jobs in observables.values() for job in jobs if name in job])
             for name in all_names}
    curves = {scheme: {name: (grids[name], get_mean_cdf([job[name] for job in observables[scheme]], grids[name]))
                       for name in names[scheme]} for scheme in observables}
    for scheme, run in runs.items():
        for name in names[scheme]:
            plot_cdf({scheme: curves[scheme][name]}, name, os.path.join(os.path.dirname(run), f"cdf_{name}.png"))
    # Compare every scheme with the first one: Kolmogorov-Smirnov test with autocorrelation-corrected sample sizes
    common = sorted(set.intersection(*[set(scheme_names) for scheme_names in names.values()]))
    if len(runs) < 2 or not common:
        return True
    reference = next(iter(curves))
    comparison = {}
    for name in common:
        sizes = {scheme: get_effective_sample_size([job[name] for job in observables[scheme]]) for scheme in curves}
        for scheme in list(curves)[1:]:
            statistic = float(np.max(np.abs(curves[scheme][name][1] - curves[reference][name][1])))
            comparison.setdefault(name, {})[f"{scheme} vs {reference}"] = {
                "ks_statistic": statistic, "effective_sample_sizes": [sizes[scheme], sizes[reference]],
                "p_value": get_ks_p_value(statistic, sizes[scheme], sizes[reference])}
        annotation = "\n".join(f"KS p = {values['p_value']:.2f}" if len(comparison[name]) == 1 else
                               f"{pair}: p = {values['p_value']:.2f}" for pair, values in comparison[name].items())
        plot_cdf({scheme: curves[scheme][name] for scheme in curves}, name,
                 os.path.join(experiment_directory, f"cdf_comparison_{name}.png"), annotation)
    with open(os.path.join(experiment_directory, "cdf_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=1)
    return True


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
    # CDFs of every experiment, written next to its runs
    experiment_directories = sorted(os.path.dirname(file_name) for file_name in
                                    glob.glob(os.path.join(args.root, "**", "analysis.json"), recursive=True))
    plotted = sum(plot_cdfs(experiment_directory) for experiment_directory in experiment_directories)
    print(f"Wrote CDF figures for {plotted} experiments")


if __name__ == "__main__":
    main()
