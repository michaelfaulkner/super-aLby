"""Writes the config templates of the grand-canonical tempering experiments.

Usage (from src):
    python config_files/grand_canonical/make_experiments.py

For every model (disks on a ring) and density below, config_files/grand_canonical/experiments/<name>/ receives
    calibration.ini        fixed_ghost_coupling_kernel, swept over ghost_coupling (the g-value runs);
    tempered_bias.ini      tempered_grand_canonical_kernel with the tabulated bias computed from the calibration;
    direct.ini             direct_grand_canonical_kernel (classic insertion/deletion, the baseline);
    experiment.env         settings read by run_experiment.sh.
All production runs of an experiment use the same activity (the suggested_log_activity of the calibration), so they
sample the same grand-canonical distribution and must agree.
"""
import os

THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.join(THIS_DIRECTORY, "experiments")
OUTPUT_ROOT = "output/grand_canonical/experiments"


def make_experiment(name, potential, potential_parameters, ring_length, number_of_particles, exact,
                    calibration_bins=20, calibration_jobs=4, calibration_observations=50000,
                    production_jobs=8, production_observations=200000, equilibration=5000,
                    distance_between_measurements=1.0, sub_bins=4):
    """
    All distances follow the EventChainMediator convention: normalised by the initial number of particles.  With N
    particles, a normalised distance of d / N is an actual distance d (time units: particle or coupling displacement).
    """
    directory = os.path.join(EXPERIMENT_ROOT, name)
    os.makedirs(directory, exist_ok=True)
    output = f"{OUTPUT_ROOT}/{name}"
    potential_section = "".join(word.capitalize() for word in potential.split("_"))
    per_particle = 1.0 / number_of_particles

    def write(file_name, samplers, observations, kernel, kernel_lines):
        lines = ["[Run]", "mediator = event_chain_mediator", "",
                 "[EventChainMediator]", f"potential = {potential}", f"samplers = {samplers}",
                 "refreshment_distribution = uniform_refreshment_distribution"]
        if kernel is not None:
            lines.append(f"grand_canonical_kernel = {kernel}")
        lines += ["temperature = 1.0", f"number_of_equilibration_iterations = {equilibration}",
                  f"number_of_observations = {observations}",
                  f"normalised_distance_between_measurements = {distance_between_measurements * per_particle!r}",
                  f"output_directory = {output}/{os.path.splitext(file_name)[0]}"]
        lines += ["", f"[{potential_section}]"] + [f"{key} = {value}" for key, value in potential_parameters.items()]
        lines += ["", "[UniformRefreshmentDistribution]", "normalised_lower_limit = 0.0",
                  f"normalised_upper_limit = {2.0 * per_particle!r}"]
        if kernel is not None:
            kernel_section = "".join(word.capitalize() for word in kernel.split("_"))
            lines += ["", f"[{kernel_section}]"] + kernel_lines
        lines += ["", "[ModelSettings]", f"number_of_particles = {number_of_particles}",
                  "dimensionality_of_particle_space = 1", f"size_of_particle_space = {ring_length}",
                  "range_of_initial_particle_positions = [-1.0, 1.0]", ""]
        with open(os.path.join(directory, file_name), "w") as f:
            f.write("\n".join(lines))

    production_samplers = "particle_number_sampler, ghost_coupling_sampler, potential_sampler"
    tempered_lines = ["log_activity = 0.0", "parked_length = 1.0",
                      f"normalised_distance_between_coupling_moves = {per_particle!r}",
                      "mean_coupling_move_length = 1.0"]
    write("calibration.ini", "particle_number_sampler, ghost_profile_sampler", calibration_observations,
          "fixed_ghost_coupling_kernel", ["ghost_coupling = 0.0"])
    write("tempered_bias.ini", production_samplers, production_observations, "tempered_grand_canonical_kernel",
          ["ghost_coupling_bias = tabulated_ghost_coupling_bias"] + tempered_lines +
          ["", "[TabulatedGhostCouplingBias]", f"file_name = {output}/bias.json"])
    write("direct.ini", production_samplers, production_observations, "direct_grand_canonical_kernel",
          ["log_activity = 0.0", f"normalised_distance_between_direct_moves = {per_particle!r}"])
    with open(os.path.join(directory, "experiment.env"), "w") as f:
        f.write(f"NAME={name}\nOUTPUT={output}\nCALIBRATION_BINS={calibration_bins}\n"
                f"CALIBRATION_JOBS={calibration_jobs}\nPRODUCTION_JOBS={production_jobs}\nSUB_BINS={sub_bins}\n"
                f"TARGET_N={number_of_particles}\nEXACT={exact}\n")


def main():
    hard = {"diameter": 1.0}
    soft = {"epsilon": 1.0, "sigma": 1.0, "power": 12.0}
    """1D hard disks (Tonks gas, exact P(N)) on a ring of length 40."""
    for density, number in [(0.3, 12), (0.5, 20), (0.7, 28), (0.8, 32)]:
        make_experiment(f"hard_disks_1d_density_{density}", "grand_canonical_hard_sphere_potential", hard, 40.0,
                        number, "tonks")
    """1D soft disks, u = (sigma / r)^12 at temperature 1, on a ring of length 40."""
    for density, number in [(0.5, 20), (0.8, 32), (1.0, 40)]:
        make_experiment(f"soft_disks_1d_density_{density}", "grand_canonical_soft_sphere_potential", soft, 40.0,
                        number, "none")


if __name__ == "__main__":
    main()
