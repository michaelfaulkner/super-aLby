"""Writes the config templates of the grand-canonical experiments defined in experiments.ini type file.

Usage: python config_files/grand_canonical/make_experiments.py [<name> ...] [--spec <file.ini>]

The spec (default config_files/grand_canonical/experiments.ini, where every key is documented) has one section per
experiment and shared settings in [DEFAULT]. Every setting comes from the spec: there are no defaults here, and a
missing key is an error. With no names, every experiment in the spec is (re)written; otherwise only the named ones.
A section with a sweep (sweep = <key>, sweep_start, sweep_end, sweep_increments, sweep_type = linear or log, as in
run_spawned_configs.sh) is written as one experiment per value, experiments/<name>/<key>_XX/, plus
experiments/<name>/sweep.json listing the values; run_experiment.sh runs (or, with sulis = true, submits) every point.
For every experiment, config_files/grand_canonical/experiments/<name>/ receives
    calibration.ini        fixed_ghost_coupling_kernel, swept over ghost_coupling (the bias runs);
    <scheme>.ini           one production template per scheme in schemes:
                               tempered    tempered_grand_canonical_kernel with the bias computed from calibration,
                               direct      direct_grand_canonical_kernel in ECMC (classic insertion/deletion),
                               metropolis  grand-canonical Metropolis (direct_grand_canonical_kernel);
    experiment.env         settings read by run_experiment.sh.
All production runs of an experiment use the same activity, so they sample the same grand-canonical distribution and
must agree. log_activity = suggested takes the activity from the calibration, which puts the maximum of P(N) near
number_of_particles (the optimal choice for tempering); a number fixes it instead (e.g. to compare with another code).
"""
import argparse
import configparser
import json
import os

THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.join(THIS_DIRECTORY, "experiments")
OUTPUT_ROOT = "output/grand_canonical/experiments"
SCHEME_KERNELS = {"tempered": "tempered_grand_canonical_kernel", "direct": "direct_grand_canonical_kernel",
                  "metropolis": "direct_grand_canonical_kernel"}
INTEGER_KEYS = {"number_of_particles", "calibration_bins", "calibration_jobs", "calibration_observations", "sub_bins",
                "production_jobs", "production_observations", "equilibration", "moves_per_sweep"}
FLOAT_KEYS = {"ring_length", "temperature", "distance_between_measurements", "parked_length",
              "distance_between_coupling_moves", "mean_coupling_move_length", "distance_between_direct_moves",
              "noise_width", "refreshment_lower_limit", "refreshment_upper_limit", "refreshment_distance"}
LIST_KEYS = {"schemes", "samplers"}
STRING_KEYS = {"potential", "log_activity", "refreshment", "range_of_initial_particle_positions"}
BOOLEAN_KEYS = {"coupled_lifting", "redraw_ghost"}
"""Keys every experiment needs; the refreshment keys depend on the refreshment type."""
"""Optional per-scheme run lengths, e.g. tempered_production_observations, overriding the shared key for one scheme."""
OVERRIDE_KEYS = {f"{scheme}_{key}" for scheme in SCHEME_KERNELS for key in ("production_observations", "equilibration")}
REQUIRED_KEYS = (INTEGER_KEYS | FLOAT_KEYS | LIST_KEYS | STRING_KEYS | BOOLEAN_KEYS) - {"refreshment_lower_limit",
                                                                          "refreshment_upper_limit",
                                                                          "refreshment_distance"}
REFRESHMENT_KEYS = {"uniform": ("refreshment_lower_limit", "refreshment_upper_limit"),
                    "constant": ("refreshment_distance",)}
SWEEP_KEYS = ("sweep", "sweep_start", "sweep_end", "sweep_increments", "sweep_type")
"""Slurm settings (needed when sulis = true): written to experiment.env for run_experiment.sh / sulis_experiment.sh."""
SULIS_KEYS = ("sulis_account", "sulis_partition", "sulis_memory_per_cpu", "sulis_calibration_time",
              "sulis_bias_time", "sulis_production_time", "sulis_analysis_time")


def make_experiment(name, s, potential_parameters, sulis):
    """
    Writes one experiment from its settings s (a dict of the spec keys). Distances in the spec are actual distances
    (particle displacement); the configs use the EventChainMediator convention of distances normalised by the initial
    number of particles N, so every distance is divided by N here. Metropolis runs one observation per sweep, with
    moves_per_sweep insertion/deletion attempts after each sweep, and drops ghost_coupling_sampler (it has no ghost).
    """
    schemes, samplers, number_of_particles = s["schemes"], s["samplers"], s["number_of_particles"]
    unknown_schemes = sorted(set(schemes) - set(SCHEME_KERNELS))
    if unknown_schemes:
        raise SystemExit(f"Experiment [{name}]: unknown scheme(s) {unknown_schemes}; choose from "
                         f"{sorted(SCHEME_KERNELS)}.")
    if not samplers:
        raise SystemExit(f"Experiment [{name}]: give the production samplers (samplers = ...).")
    required = {"particle_number_sampler"} | ({"ghost_coupling_sampler"} if "tempered" in schemes else set())
    missing = sorted(required - set(samplers))
    if missing:
        raise SystemExit(f"Experiment [{name}]: samplers must include {', '.join(missing)} (the analysis needs N, and "
                         f"lambda to keep only the physical lambda <= 0 samples of tempered runs).")
    directory = os.path.join(EXPERIMENT_ROOT, name)
    os.makedirs(directory, exist_ok=True)
    output = f"{OUTPUT_ROOT}/{name}"
    potential = s["potential"]
    log_activity = s["log_activity"]
    start_log_activity = 0.0 if log_activity == "suggested" else log_activity

    def normalised(distance):
        return repr(distance / number_of_particles)

    def section(name_snake):
        return "".join(word.capitalize() for word in name_snake.split("_"))

    def potential_lines():
        return ["", f"[{section(potential)}]"] + [f"{key} = {value}" for key, value in potential_parameters.items()]

    def refreshment_lines():
        if s["refreshment"] == "uniform":
            return ["", "[UniformRefreshmentDistribution]",
                    f"normalised_lower_limit = {normalised(s['refreshment_lower_limit'])}",
                    f"normalised_upper_limit = {normalised(s['refreshment_upper_limit'])}"]
        return ["", "[ConstantRefreshmentDistribution]",
                f"normalised_refreshment_distance = {normalised(s['refreshment_distance'])}"]

    def model_lines():
        return ["", "[ModelSettings]", f"number_of_particles = {number_of_particles}",
                "dimensionality_of_particle_space = 1", f"size_of_particle_space = {s['ring_length']}",
                f"range_of_initial_particle_positions = {s['range_of_initial_particle_positions']}", ""]

    def run_length(scheme, key):
        """Returns the scheme's own value of key (e.g. tempered_equilibration) if given, else the shared value."""
        return s.get(f"{scheme}_{key}", s[key])

    def write_ecmc(file_name, ecmc_samplers, equilibration, observations, kernel, kernel_lines):
        lines = ["[Run]", "mediator = event_chain_mediator", "",
                 "[EventChainMediator]", f"potential = {potential}", f"samplers = {ecmc_samplers}",
                 f"refreshment_distribution = {s['refreshment']}_refreshment_distribution",
                 f"grand_canonical_kernel = {kernel}",
                 f"temperature = {s['temperature']}", f"number_of_equilibration_iterations = {equilibration}",
                 f"number_of_observations = {observations}",
                 f"normalised_distance_between_measurements = {normalised(s['distance_between_measurements'])}",
                 f"output_directory = {output}/{os.path.splitext(file_name)[0]}"]
        lines += potential_lines() + refreshment_lines() + ["", f"[{section(kernel)}]"] + kernel_lines
        with open(os.path.join(directory, file_name), "w") as f:
            f.write("\n".join(lines + model_lines()))

    def write_metropolis(file_name):
        metropolis_samplers = ", ".join(sampler for sampler in samplers if sampler != "ghost_coupling_sampler")
        lines = ["[Run]", "mediator = metropolis_mediator", "",
                 "[MetropolisMediator]", f"potential = {potential}", f"samplers = {metropolis_samplers}",
                 "noise_distribution = toroidal_uniform_noise_distribution",
                 "grand_canonical_kernel = direct_grand_canonical_kernel",
                 f"grand_canonical_moves_per_sweep = {s['moves_per_sweep']}", f"temperature = {s['temperature']}",
                 f"number_of_equilibration_iterations = {run_length('metropolis', 'equilibration')}",
                 f"number_of_observations = {run_length('metropolis', 'production_observations')}",
                 "proposal_dynamics_adaptor_is_on = True",
                 f"output_directory = {output}/{os.path.splitext(file_name)[0]}"]
        lines += potential_lines() + ["", "[ToroidalUniformNoiseDistribution]",
                                      f"initial_width_of_noise_distribution = {s['noise_width']}",
                                      "", "[DirectGrandCanonicalKernel]", f"log_activity = {start_log_activity}"]
        with open(os.path.join(directory, file_name), "w") as f:
            f.write("\n".join(lines + model_lines()))

    production_samplers = ", ".join(samplers)
    write_ecmc("calibration.ini", "particle_number_sampler, ghost_profile_sampler", s["equilibration"],
               s["calibration_observations"],
               "fixed_ghost_coupling_kernel", ["ghost_coupling = 0.0"])
    for scheme in schemes:
        if scheme == "tempered":
            write_ecmc("tempered.ini", production_samplers, run_length(scheme, "equilibration"),
                       run_length(scheme, "production_observations"), SCHEME_KERNELS[scheme],
                       ["ghost_coupling_bias = tabulated_ghost_coupling_bias", f"log_activity = {start_log_activity}",
                        f"parked_length = {s['parked_length']}",
                        f"normalised_distance_between_coupling_moves = "
                        f"{normalised(s['distance_between_coupling_moves'])}",
                        f"mean_coupling_move_length = {s['mean_coupling_move_length']}",
                        f"coupled_lifting = {s['coupled_lifting']}",
                        f"redraw_ghost = {s['redraw_ghost']}",
                        "", "[TabulatedGhostCouplingBias]", f"file_name = {output}/bias.json"])
        elif scheme == "direct":
            write_ecmc("direct.ini", production_samplers, run_length(scheme, "equilibration"),
                       run_length(scheme, "production_observations"), SCHEME_KERNELS[scheme],
                       [f"log_activity = {start_log_activity}",
                        f"normalised_distance_between_direct_moves = {normalised(s['distance_between_direct_moves'])}"])
        else:
            write_metropolis("metropolis.ini")
    with open(os.path.join(directory, "experiment.env"), "w") as f:
        f.write(f"NAME={name}\nOUTPUT={output}\nCALIBRATION_BINS={s['calibration_bins']}\n"
                f"CALIBRATION_JOBS={s['calibration_jobs']}\nPRODUCTION_JOBS={s['production_jobs']}\n"
                f"SUB_BINS={s['sub_bins']}\n"
                f"LOG_ACTIVITY={log_activity}\nSCHEMES=\"{' '.join(schemes)}\"\n")
        f.write(f"SULIS={'true' if sulis is not None else 'false'}\n")
        for key, value in (sulis or {}).items():
            f.write(f"{key.upper()}={value}\n")


def read_experiment(section):
    """Converts one spec section into (settings, potential parameters, Slurm settings or None); keys the spec does not
    define are potential parameters."""
    settings, potential_parameters = {}, {}
    if "sulis" not in section:
        raise SystemExit(f"Experiment [{section.name}] needs a value for: sulis (true or false).")
    try:
        on_sulis = section.getboolean("sulis")
    except ValueError:
        raise SystemExit(f"Experiment [{section.name}]: sulis must be true or false.")
    sulis = None
    if on_sulis:
        missing = [key for key in SULIS_KEYS if not section.get(key, "").strip()]
        if missing:
            raise SystemExit(f"Experiment [{section.name}]: sulis = true needs {', '.join(missing)}.")
        sulis = {key: section[key].strip() for key in SULIS_KEYS}
    for key, value in section.items():
        if key == "sulis" or key in SULIS_KEYS:
            continue
        try:
            if key in INTEGER_KEYS or key in OVERRIDE_KEYS:
                settings[key] = int(value)
            elif key in FLOAT_KEYS:
                settings[key] = float(value)
            elif key in LIST_KEYS:
                settings[key] = tuple(item.strip() for item in value.split(",") if item.strip())
            elif key in STRING_KEYS:
                settings[key] = value.strip()
            elif key in BOOLEAN_KEYS:
                settings[key] = section.getboolean(key)
            else:
                potential_parameters[key] = value
        except ValueError:
            raise SystemExit(f"Experiment [{section.name}]: {key} = {value} is not a valid value.")
    missing = sorted(REQUIRED_KEYS - set(settings))
    if missing:
        raise SystemExit(f"Experiment [{section.name}] needs a value for: {', '.join(missing)}.")
    if settings["refreshment"] not in REFRESHMENT_KEYS:
        raise SystemExit(f"Experiment [{section.name}]: refreshment must be one of {sorted(REFRESHMENT_KEYS)}.")
    missing = [key for key in REFRESHMENT_KEYS[settings["refreshment"]] if key not in settings]
    if missing:
        raise SystemExit(f"Experiment [{section.name}]: {settings['refreshment']} refreshment needs "
                         f"{', '.join(missing)}.")
    if settings["log_activity"] != "suggested":
        try:
            settings["log_activity"] = float(settings["log_activity"])
        except ValueError:
            raise SystemExit(f"Experiment [{section.name}]: log_activity must be suggested or a number.")
    return settings, potential_parameters, sulis


def get_sweep_values(section):
    """Returns the sweep values (as written in the configs) of a section with a sweep, like spawn_configs.py."""
    name = section.name
    missing = [key for key in SWEEP_KEYS if not section.get(key, "").strip()]
    if missing:
        raise SystemExit(f"Experiment [{name}]: a sweep needs {', '.join(missing)}.")
    key = section["sweep"].strip()
    if key in SWEEP_KEYS or key == "sulis" or key in SULIS_KEYS or key in LIST_KEYS or key in (
            "potential", "refreshment", "range_of_initial_particle_positions"):
        raise SystemExit(f"Experiment [{name}]: cannot sweep {key} (sweep a numerical key).")
    try:
        start, end = float(section["sweep_start"]), float(section["sweep_end"])
        increments = int(section["sweep_increments"])
    except ValueError:
        raise SystemExit(f"Experiment [{name}]: sweep_start and sweep_end must be numbers and sweep_increments an "
                         f"integer.")
    if increments < 0 or (increments == 0 and start != end):
        raise SystemExit(f"Experiment [{name}]: sweep_increments must be >= 0 (and 0 only if sweep_start = sweep_end).")
    kind = section["sweep_type"].strip()
    if kind == "linear":
        values = [start + i * (end - start) / increments if increments else start for i in range(increments + 1)]
    elif kind == "log":
        if start <= 0.0 or end <= 0.0:
            raise SystemExit(f"Experiment [{name}]: a log sweep needs positive sweep_start and sweep_end.")
        values = [start * (end / start) ** (i / increments) if increments else start for i in range(increments + 1)]
    else:
        raise SystemExit(f"Experiment [{name}]: sweep_type must be linear or log.")
    if key in INTEGER_KEYS or key in OVERRIDE_KEYS:
        if any(abs(value - round(value)) > 1.0e-9 for value in values):
            raise SystemExit(f"Experiment [{name}]: {key} must be an integer, but the sweep gives {values}.")
        return key, [str(int(round(value))) for value in values]
    return key, [format(value, ".12g") for value in values]


def get_point_section(spec, name, key, value):
    """Returns the section of one sweep point: the experiment's settings with key = value (interpolation such as
    %(ring_length)s then uses the swept value)."""
    raw = {option: spec.get(name, option, raw=True) for option in spec.options(name) if option not in SWEEP_KEYS}
    raw[key] = value
    point = configparser.ConfigParser()
    point.read_dict({name: raw})
    return point[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="experiments to write (default: all in the spec)")
    parser.add_argument("--spec", default=os.path.join(THIS_DIRECTORY, "experiments.ini"))
    args = parser.parse_args()
    spec = configparser.ConfigParser()
    if not spec.read(args.spec):
        raise SystemExit(f"Cannot read the spec file {args.spec}.")
    unknown = set(args.names) - set(spec.sections())
    if unknown:
        raise SystemExit(f"Unknown experiment(s) {sorted(unknown)}; choose from {spec.sections()}.")
    if any(key in spec.defaults() for key in SWEEP_KEYS):
        raise SystemExit("Put sweep keys in the section of the experiment to sweep, not in [DEFAULT].")
    for name in spec.sections():
        if args.names and name not in args.names:
            continue
        if not spec.get(name, "sweep", fallback="").strip():
            make_experiment(name, *read_experiment(spec[name]))
            print(f"written experiments/{name}")
            continue
        key, values = get_sweep_values(spec[name])
        points = [f"{key}_{index:02d}" for index in range(len(values))]
        for point, value in zip(points, values):
            make_experiment(f"{name}/{point}", *read_experiment(get_point_section(spec, name, key, value)))
        with open(os.path.join(EXPERIMENT_ROOT, name, "sweep.json"), "w") as f:
            json.dump({"name": name, "sweep": key, "values": values, "points": points}, f, indent=1)
        print(f"written experiments/{name} (sweep over {key}: {', '.join(values)})")


if __name__ == "__main__":
    main()
