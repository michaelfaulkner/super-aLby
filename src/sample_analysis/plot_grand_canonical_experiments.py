"""Figures of one grand-canonical experiment: the CDF of every sample array of every scheme, their comparison, and
the calibrated bias.

Usage (from src):
    python sample_analysis/plot_grand_canonical_experiments.py <experiment_dir or sweep_dir>

The folder is the experiment's config folder written by make_experiments.py (config_files/grand_canonical/experiments/
<name>, with experiment.env; for a sweep, every point is plotted). run_experiment.sh runs this at the end of the
analysis stage (PLOT=false turns it off). Reads <OUTPUT>/analysis.json and <OUTPUT>/bias.json and writes, from the
physical samples:
    <OUTPUT>/<scheme>/cdf_<array>.png   the CDF of each sample array (mean over jobs);
    <OUTPUT>/cdf_comparison_<array>.png the CDFs of all schemes on the same axes, with the Kolmogorov-Smirnov p-value
                                        of each scheme against the first;
    <OUTPUT>/cdf_comparison.json        the Kolmogorov-Smirnov statistics, effective sample sizes and p-values;
    <OUTPUT>/bias.png                   eta(lambda) and its slopes g (if the experiment has a calibrated bias).
"""
import argparse
import glob
import json
import math
import os
import re
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


def get_output_directory(experiment_directory):
    """OUTPUT from the experiment's experiment.env."""
    with open(os.path.join(experiment_directory, "experiment.env")) as f:
        match = re.search(r"^OUTPUT=(.*)$", f.read(), re.MULTILINE)
    if match is None:
        raise ValueError(f"No OUTPUT in {experiment_directory}/experiment.env")
    return match.group(1).strip().strip('"')


def plot_experiment(experiment_directory):
    output = get_output_directory(experiment_directory)
    if not os.path.isfile(os.path.join(output, "analysis.json")):
        print(f"No analysis.json in {output} (run the analysis stage first)")
        return
    if plot_cdfs(output):
        print(f"Wrote the CDF figures of {output}")
    bias_file = os.path.join(output, "bias.json")
    if os.path.isfile(bias_file):
        from compute_ghost_coupling_bias import plot_bias
        with open(bias_file) as f:
            plot_bias(json.load(f), os.path.join(output, "bias.png"))
        print(f"Wrote {output}/bias.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_directory")
    directory = parser.parse_args().experiment_directory
    if not os.path.isfile(os.path.join(directory, "experiment.env")) and os.path.isfile(
            os.path.join(directory, "sweep.json")):
        points = sorted(os.path.dirname(file_name) for file_name in
                        glob.glob(os.path.join(directory, "*", "experiment.env")))
    else:
        points = [directory]
    for point in points:
        plot_experiment(point)


if __name__ == "__main__":
    main()
