"""Tests of grand-canonical event-chain Monte Carlo in EventChainMediator (run from src).

    PYTHON=/opt/homebrew/bin/python3.11 python config_files/grand_canonical/tests/run_tests.py
        [--groups unit,basic,metropolis,edge] [--jobs 12] [--quick] [--max-cpus 13]

Groups:
    unit        config_files/grand_canonical/tests/unit_tests.py (event distances against brute force and exact
                survival functions, ghost functions, bias intervals, jump bookkeeping, configuration errors).
    basic       exact results: canonical runs with a grand-canonical potential, ideal gas (Poisson) and hard disks
                (exact Tonks P(N)) with the tempered and direct kernels.
    metropolis  comparison of <N>, Var N and <U> with an independent grand-canonical Metropolis code
                (metropolis_grand_canonical.py): soft disks at T = 1 and T = 2.
    edge        edge cases with exact results: low activity (N = 0 matters), high-density hard disks, temperature
                independence of hard disks, initial N far from equilibrium, arbitrary (wrong) biases with positive and
                negative slopes, the exact lambda marginal of the ideal gas under a bias, extreme kernel parameters and
                constant refreshment distances.
All systems are disks on a ring (the grand-canonical potentials are 1D only).
Each statistical test runs --jobs independent jobs in parallel with run.py (configs in
config_files/grand_canonical/tests/<test>/, outputs in output/grand_canonical/tests/<test>/), pools them and passes if
every compared quantity is within 3.5 standard errors (combined over jobs / Metropolis seeds) of its reference.
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.special import gammaln

ROOT = "config_files/grand_canonical/tests"
OUTPUT = "output/grand_canonical/tests"
PYTHON = os.environ.get("PYTHON", sys.executable)
EQUILIBRATION = 2000
SOFT = ["epsilon = 1.0", "sigma = 1.0", "power = 12.0"]
HARD = ["diameter = 1.0"]
IDEAL = ["epsilon = 0.0"]


def config(name, potential, potential_lines, kernel, kernel_lines, length, number, observations, job,
           temperature=1.0, refreshment="uniform", extra_sections=(), samplers=None):
    """Writes an EventChainMediator config with all distances normalised by number (actual distance 1)."""
    section = "".join(word.capitalize() for word in potential.split("_"))
    samplers = samplers or "particle_number_sampler, ghost_coupling_sampler, potential_sampler"
    lines = ["[Run]", "mediator = event_chain_mediator", "", "[EventChainMediator]", f"potential = {potential}",
             f"samplers = {samplers}", f"refreshment_distribution = {refreshment}_refreshment_distribution"]
    if kernel is not None:
        lines.append(f"grand_canonical_kernel = {kernel}")
    lines += [f"temperature = {temperature}", f"number_of_equilibration_iterations = {EQUILIBRATION}",
              f"number_of_observations = {observations}",
              f"normalised_distance_between_measurements = {1.0 / number!r}",
              f"output_directory = {OUTPUT}/{name}/job_{job:02d}", "", f"[{section}]"] + list(potential_lines)
    if refreshment == "uniform":
        lines += ["", "[UniformRefreshmentDistribution]", "normalised_lower_limit = 0.0",
                  f"normalised_upper_limit = {2.0 / number!r}"]
    else:
        lines += ["", "[ConstantRefreshmentDistribution]", f"normalised_refreshment_distance = {1.0 / number!r}"]
    if kernel is not None:
        lines += ["", f"[{''.join(word.capitalize() for word in kernel.split('_'))}]"] + list(kernel_lines)
    for extra in extra_sections:
        lines += [""] + list(extra)
    lines += ["", "[ModelSettings]", f"number_of_particles = {number}", "dimensionality_of_particle_space = 1",
              f"size_of_particle_space = {length}", "range_of_initial_particle_positions = [-1.0, 1.0]", ""]
    return "\n".join(lines)


def tempered(log_activity, number, parked_length=1.0, coupling_moves=1.0, move_length=1.0, bias_file=None):
    lines = [f"log_activity = {log_activity!r}", f"parked_length = {parked_length}",
             f"normalised_distance_between_coupling_moves = {coupling_moves / number!r}",
             f"mean_coupling_move_length = {move_length}"]
    extras = []
    if bias_file is not None:
        lines = ["ghost_coupling_bias = tabulated_ghost_coupling_bias"] + lines
        extras = [["[TabulatedGhostCouplingBias]", f"file_name = {bias_file}"]]
    return "tempered_grand_canonical_kernel", lines, extras


def direct(log_activity, number):
    return ("direct_grand_canonical_kernel",
            [f"log_activity = {log_activity!r}", f"normalised_distance_between_direct_moves = {1.0 / number!r}"], [])


def tonks_log_activity(length, target):
    """Activity at which the exact Tonks P(N) (disks of diameter 1) peaks near target."""
    return (math.log(target + 1) + (target - 1) * math.log(length - target) -
            target * math.log(length - (target + 1)))


def exact_distribution(kind, log_activity, length, truncate=False):
    numbers = np.arange(0, 400)
    if kind == "poisson":
        log_weights = numbers * (log_activity + math.log(length)) - gammaln(numbers + 1)
    else:
        log_weights = np.full(len(numbers), -np.inf)
        log_weights[0] = 0.0
        allowed = (numbers >= 1) & (numbers < length)
        n = numbers[allowed]
        log_weights[allowed] = n * log_activity + math.log(length) + (n - 1) * np.log(length - n) - gammaln(n + 1)
    if truncate:
        log_weights[0] = -np.inf
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / np.sum(weights)


def write_bias(name, slope):
    """A linear bias eta = slope * lambda (deliberately not the free energy)."""
    path = os.path.join(ROOT, "biases", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"nodes": np.linspace(0.0, 1.0, 11).tolist(), "slopes": [slope] * 10}, f)
    return path


class Test:
    """A statistical test: named runs (each a function job -> config text) and a check."""

    def __init__(self, name, group, runs, check, metropolis=None):
        self.name, self.group, self.runs, self.check, self.metropolis = name, group, runs, check, metropolis


def run(potential, potential_lines, kernel_spec, length, number, observations, **options):
    kernel, kernel_lines, extras = kernel_spec if kernel_spec is not None else (None, [], [])

    def make(name, job):
        return config(name, potential, potential_lines, kernel, kernel_lines, length, number, observations, job,
                      extra_sections=extras, **options)
    return make


def define_tests(observations):
    tests = []
    half = observations // 2
    """basic"""
    tests.append(Test("canonical_soft_spheres", "basic", {
        "no_kernel": run("grand_canonical_soft_sphere_potential", SOFT, None, 20.0, 15, half),
        "parked_ghost": run("grand_canonical_soft_sphere_potential", SOFT,
                            ("fixed_ghost_coupling_kernel", ["ghost_coupling = 0.0"], []), 20.0, 15, half)},
        {"type": "energy_agreement"}))
    for kernel_name, spec, truncate in [("tempered", tempered(0.0, 20), False), ("direct", direct(0.0, 20), True)]:
        tests.append(Test(f"ideal_gas_{kernel_name}", "basic", {
            "run": run("grand_canonical_soft_sphere_potential", IDEAL, spec, 20.0, 20, observations)},
            {"type": "exact", "kind": "poisson", "log_activity": 0.0, "length": 20.0, "truncate": truncate}))
    log_z = tonks_log_activity(30.0, 18)
    for kernel_name, spec, truncate in [("tempered", tempered(log_z, 18), False), ("direct", direct(log_z, 18), True)]:
        tests.append(Test(f"tonks_{kernel_name}", "basic", {
            "run": run("grand_canonical_hard_sphere_potential", HARD, spec, 30.0, 18, observations)},
            {"type": "exact", "kind": "tonks", "log_activity": log_z, "length": 30.0, "truncate": truncate}))

    """metropolis: same system in ECMC (tempered with g = 0, and direct) and in the independent Metropolis code."""
    for name, temperature in [("soft_disks_1d", 1.0), ("soft_disks_1d_temperature_2", 2.0)]:
        tests.append(Test(f"metropolis_{name}", "metropolis", {
            "tempered": run("grand_canonical_soft_sphere_potential", SOFT, tempered(1.0, 15), 20.0, 15, half,
                            temperature=temperature),
            "direct": run("grand_canonical_soft_sphere_potential", SOFT, direct(1.0, 15), 20.0, 15, half,
                          temperature=temperature)},
            {"type": "metropolis", "energy": True},
            metropolis={"kind": "soft", "length": "20.0", "log_activity": 1.0, "temperature": temperature,
                        "initial_number": 15}))

    """edge"""
    log_low = math.log(0.1)
    tests.append(Test("edge_low_activity_tempered", "edge", {
        "run": run("grand_canonical_soft_sphere_potential", IDEAL, tempered(log_low, 1), 5.0, 1, observations)},
        {"type": "exact", "kind": "poisson", "log_activity": log_low, "length": 5.0, "truncate": False,
         "check_zero": True}))
    log_high = tonks_log_activity(30.0, 24)
    tests.append(Test("edge_high_density_tonks", "edge", {
        "tempered": run("grand_canonical_hard_sphere_potential", HARD, tempered(log_high, 24), 30.0, 24, observations),
        "direct": run("grand_canonical_hard_sphere_potential", HARD, direct(log_high, 24), 30.0, 24, observations)},
        {"type": "exact", "kind": "tonks", "log_activity": log_high, "length": 30.0, "truncate": False,
         "truncate_runs": ["direct"]}))
    tests.append(Test("edge_hard_disks_temperature_2", "edge", {
        "run": run("grand_canonical_hard_sphere_potential", HARD, tempered(log_z, 18), 30.0, 18, observations,
                   temperature=2.0)},
        {"type": "exact", "kind": "tonks", "log_activity": log_z, "length": 30.0, "truncate": False}))
    for start in [5, 26]:
        tests.append(Test(f"edge_initial_number_{start}", "edge", {
            "run": run("grand_canonical_hard_sphere_potential", HARD, tempered(log_z, start), 30.0, start,
                       observations)},
            {"type": "exact", "kind": "tonks", "log_activity": log_z, "length": 30.0, "truncate": False}))
    for label, slope in [("positive", 3.0), ("negative", -2.0)]:
        bias = write_bias(f"linear_{label}", slope)
        tests.append(Test(f"edge_wrong_bias_{label}_tonks", "edge", {
            "run": run("grand_canonical_hard_sphere_potential", HARD, tempered(log_z, 18, bias_file=bias), 30.0, 18,
                       observations)},
            {"type": "exact", "kind": "tonks", "log_activity": log_z, "length": 30.0, "truncate": False}))
        tests.append(Test(f"edge_wrong_bias_{label}_soft_disks", "edge", {
            "tempered": run("grand_canonical_soft_sphere_potential", SOFT, tempered(1.0, 15, bias_file=bias), 20.0, 15,
                            observations),
            "direct": run("grand_canonical_soft_sphere_potential", SOFT, direct(1.0, 15), 20.0, 15, observations)},
            {"type": "agreement", "energy": True}))
    bias = write_bias("linear_positive", 3.0)
    tests.append(Test("edge_ideal_gas_lambda_marginal", "edge", {
        "run": run("grand_canonical_soft_sphere_potential", IDEAL, tempered(0.0, 20, bias_file=bias), 20.0, 20,
                   observations)},
        {"type": "lambda_marginal", "slope": 3.0, "parked_length": 1.0, "log_activity": 0.0, "length": 20.0}))
    tests.append(Test("edge_kernel_parameters_tonks", "edge", {
        "short_parked_frequent_moves": run("grand_canonical_hard_sphere_potential", HARD,
                                           tempered(log_z, 18, parked_length=0.2, coupling_moves=0.05,
                                                    move_length=0.1), 30.0, 18, observations),
        "long_parked_rare_moves": run("grand_canonical_hard_sphere_potential", HARD,
                                      tempered(log_z, 18, parked_length=3.0, coupling_moves=10.0, move_length=5.0),
                                      30.0, 18, observations),
        "constant_refreshment": run("grand_canonical_hard_sphere_potential", HARD, tempered(log_z, 18), 30.0, 18,
                                    observations, refreshment="constant")},
        {"type": "exact", "kind": "tonks", "log_activity": log_z, "length": 30.0, "truncate": False}))
    return tests


def load_jobs(directory):
    """Per job: physical N samples, physical energies and all lambda samples (post-equilibration)."""
    jobs = []
    for job in sorted(glob.glob(os.path.join(directory, "job_*"))):
        if not os.path.isfile(os.path.join(job, "sim_params.json")):
            continue
        numbers = np.load(os.path.join(job, "checkpoint_00_sample_of_particle_number.npy"))[EQUILIBRATION + 1:, 0]
        coupling_file = os.path.join(job, "checkpoint_00_sample_of_ghost_coupling.npy")
        couplings = (np.load(coupling_file)[EQUILIBRATION + 1:, 0] if os.path.isfile(coupling_file)
                     else np.full(len(numbers), np.nan))
        physical = np.isnan(couplings) | (couplings <= 0.0)
        energies = np.load(os.path.join(job, "checkpoint_00_sample_of_potential.npy"))[EQUILIBRATION + 1:, 0]
        jobs.append({"numbers": numbers[physical], "energies": energies[physical], "couplings": couplings})
    return jobs


def pooled(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(len(values)))


def moments(jobs):
    return (pooled([job["numbers"].mean() for job in jobs]), pooled([job["numbers"].var() for job in jobs]),
            pooled([job["energies"].mean() for job in jobs]))


def evaluate(test, data, references):
    """Returns (list of (label, deviation in standard errors), text)."""
    check = test.check
    deviations, parts = [], []
    if check["type"] == "exact":
        for run_name, jobs in data.items():
            truncate = check["truncate"] or run_name in check.get("truncate_runs", [])
            distribution = exact_distribution(check["kind"], check["log_activity"], check["length"], truncate)
            numbers = np.arange(len(distribution))
            exact_mean = float(np.sum(numbers * distribution))
            exact_variance = float(np.sum(numbers ** 2 * distribution) - exact_mean ** 2)
            mean, variance, _ = moments(jobs)
            deviations += [(f"{run_name} <N>", (mean[0] - exact_mean) / mean[1]),
                           (f"{run_name} Var N", (variance[0] - exact_variance) / variance[1])]
            parts.append(f"{run_name}: <N> {mean[0]:.4f}±{mean[1]:.4f} (exact {exact_mean:.4f}), Var N "
                         f"{variance[0]:.4f}±{variance[1]:.4f} (exact {exact_variance:.4f})")
            if check.get("check_zero"):
                zero = pooled([np.mean(job["numbers"] == 0) for job in jobs])
                deviations.append((f"{run_name} P(0)", (zero[0] - distribution[0]) / zero[1]))
                parts.append(f"P(0) {zero[0]:.4f}±{zero[1]:.4f} (exact {distribution[0]:.4f})")
    elif check["type"] in ("agreement", "metropolis"):
        summaries = {run_name: moments(jobs) for run_name, jobs in data.items()}
        if check["type"] == "metropolis":
            summaries["metropolis"] = references
        keys = list(summaries)
        quantities = [("<N>", 0), ("Var N", 1)] + ([("<U>", 2)] if check.get("energy") else [])
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                for label, index in quantities:
                    first, second = summaries[keys[i]][index], summaries[keys[j]][index]
                    deviations.append((f"{label} {keys[i]} vs {keys[j]}",
                                       (first[0] - second[0]) / math.hypot(first[1], second[1])))
        for key in keys:
            parts.append(f"{key}: " + ", ".join(f"{label} {summaries[key][index][0]:.4f}±{summaries[key][index][1]:.4f}"
                                                for label, index in quantities))
    elif check["type"] == "energy_agreement":
        no_kernel, parked = data["no_kernel"], data["parked_ghost"]
        fixed = all(np.all(job["numbers"] == job["numbers"][0]) for job in no_kernel + parked)
        first, second = moments(no_kernel)[2], moments(parked)[2]
        deviations.append(("<U> no kernel vs parked ghost", (first[0] - second[0]) / math.hypot(first[1], second[1])))
        if not fixed:
            deviations.append(("N fixed", np.inf))
        parts.append(f"N fixed: {fixed}; <U> no kernel {first[0]:.4f}±{first[1]:.4f}, parked ghost "
                     f"{second[0]:.4f}±{second[1]:.4f}")
    elif check["type"] == "lambda_marginal":
        """Ideal gas: U_g = 0, so lambda has density proportional to exp(eta(lambda)) on [-l, 1] (eta = 0 for
        lambda <= 0), independently of N.  Check P(lambda <= 0) and P(lambda > 1/2 | lambda > 0)."""
        jobs = data["run"]
        slope, parked = check["slope"], check["parked_length"]
        growing_weight = (math.exp(slope) - 1.0) / slope
        exact_parked = parked / (parked + growing_weight)
        exact_upper = (math.exp(slope) - math.exp(slope / 2)) / slope / growing_weight
        parked_fraction = pooled([np.mean(job["couplings"] <= 0.0) for job in jobs])
        upper_fraction = pooled([np.mean(job["couplings"][job["couplings"] > 0.0] > 0.5) for job in jobs])
        distribution = exact_distribution("poisson", check["log_activity"], check["length"])
        exact_mean = float(np.sum(np.arange(len(distribution)) * distribution))
        mean = moments(jobs)[0]
        deviations += [("P(lambda <= 0)", (parked_fraction[0] - exact_parked) / parked_fraction[1]),
                       ("P(lambda > 1/2 | lambda > 0)", (upper_fraction[0] - exact_upper) / upper_fraction[1]),
                       ("<N>", (mean[0] - exact_mean) / mean[1])]
        parts.append(f"P(lambda<=0) {parked_fraction[0]:.4f}±{parked_fraction[1]:.4f} (exact {exact_parked:.4f}), "
                     f"P(lambda>1/2|lambda>0) {upper_fraction[0]:.4f}±{upper_fraction[1]:.4f} (exact "
                     f"{exact_upper:.4f}), <N> {mean[0]:.4f}±{mean[1]:.4f} (exact {exact_mean:.4f})")
    return deviations, "; ".join(parts)


def run_processes(commands, max_cpus):
    """Runs (label, argument list) commands with at most max_cpus at a time.  Returns the failures."""
    pending, running, failures = list(commands), [], []
    while pending or running:
        while pending and len(running) < max_cpus:
            label, arguments = pending.pop(0)
            running.append((label, subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
        for label, process in list(running):
            if process.poll() is not None:
                if process.returncode != 0:
                    failures.append((label, process.stderr.read().decode()[-600:]))
                running.remove((label, process))
        time.sleep(0.5)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", default="unit,basic,metropolis,edge")
    parser.add_argument("--jobs", type=int, default=12, help="independent jobs per run (and Metropolis seeds)")
    parser.add_argument("--quick", action="store_true", help="1e5 instead of 4e5 observations per job")
    parser.add_argument("--max-cpus", type=int, default=13)
    args = parser.parse_args()
    groups = args.groups.split(",")
    all_passed, summary = True, []

    if "unit" in groups:
        result = subprocess.run([PYTHON, os.path.join(ROOT, "unit_tests.py"), os.path.join(ROOT, "unit_settings.ini")],
                                capture_output=True, text=True)
        print("\n".join(line for line in result.stdout.splitlines() if line.startswith(("PASS", "FAIL"))))
        if result.returncode != 0:
            all_passed = False
            print(result.stderr[-1000:])
        summary.append({"test": "unit_tests", "passed": result.returncode == 0})

    tests = [test for test in define_tests(100000 if args.quick else 400000) if test.group in groups]
    commands = []
    for test in tests:
        for run_name, make in test.runs.items():
            directory = os.path.join(ROOT, test.name, run_name)
            shutil.rmtree(directory, ignore_errors=True)
            shutil.rmtree(os.path.join(OUTPUT, test.name, run_name), ignore_errors=True)
            os.makedirs(directory)
            for job in range(args.jobs):
                path = os.path.join(directory, f"job_{job:02d}.ini")
                with open(path, "w") as f:
                    f.write(make(f"{test.name}/{run_name}", job))
                commands.append((path, [PYTHON, "run.py", path]))
        if test.metropolis is not None:
            directory = os.path.join(OUTPUT, test.name, "metropolis")
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory)
            reference = test.metropolis
            sweeps = 40000 if args.quick else 160000
            for seed in range(args.jobs):
                commands.append((f"{test.name} metropolis seed {seed}", [
                    PYTHON, os.path.join(ROOT, "metropolis_grand_canonical.py"), reference["kind"], reference["length"],
                    str(reference["log_activity"]), str(reference["temperature"]), str(sweeps), str(seed),
                    str(reference["initial_number"]), os.path.join(directory, f"seed_{seed:02d}.json")]))
    if commands:
        print(f"Running {len(commands)} jobs on {args.max_cpus} processes ...", flush=True)
        failures = run_processes(commands, args.max_cpus)
        for label, error in failures:
            print(f"FAILED to run {label}:\n{error}")
            all_passed = False

    for test in tests:
        data = {run_name: load_jobs(os.path.join(OUTPUT, test.name, run_name)) for run_name in test.runs}
        references = None
        if test.metropolis is not None:
            seeds = [json.load(open(path)) for path in
                     sorted(glob.glob(os.path.join(OUTPUT, test.name, "metropolis", "seed_*.json")))]
            references = (pooled([s["mean_number"] for s in seeds]), pooled([s["variance_number"] for s in seeds]),
                          pooled([s["mean_energy"] for s in seeds]))
        deviations, text = evaluate(test, data, references)
        passed = all(abs(value) < 3.5 for _, value in deviations)
        all_passed &= passed
        worst = max(deviations, key=lambda item: abs(item[1]))
        print(f"{'PASS' if passed else 'FAIL'}  [{test.group}] {test.name}: {text}  (largest deviation: {worst[0]} "
              f"{worst[1]:+.2f} sigma)", flush=True)
        summary.append({"test": test.name, "group": test.group, "passed": passed, "result": text,
                        "deviations_in_standard_errors": {label: float(value) for label, value in deviations}})
    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
