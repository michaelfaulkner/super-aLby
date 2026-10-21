"""Module for the GhostProfileSampler class."""
from .observation_sampler import ObservationSampler
import numpy as np


class GhostProfileSampler(ObservationSampler):
    """
    Class for taking observations of the ghost energy profile (for grand-canonical event-chain Monte Carlo): for soft
    spheres, the energy that the ghost would have at lambda = 1; for hard spheres, the clearance
    c = min_j r_gj / sigma.  In a calibration run (fixed_ghost_coupling_kernel) with the ghost held at coupling
    lambda_k, these samples give the free-energy change of growing the ghost to any lambda.
    """

    def __init__(self):
        """The constructor of the GhostProfileSampler class."""
        super().__init__()

    def get_empty_sample_array(self, total_number_of_iterations):
        return np.zeros((total_number_of_iterations, 1))

    def get_observation(self, momenta, positions, potential):
        """NaN if there is no ghost (including potentials without ghost coupling, i.e. all canonical potentials)."""
        if getattr(potential, "ghost_coupling", None) is None:
            return np.nan
        return potential.get_ghost_profile(positions)

    def output_sample(self, sample, checkpoint_index, output_directory):
        self._write_sample_to_file(sample, f"checkpoint_{checkpoint_index:02d}_sample_of_ghost_profile.npy",
                                   output_directory)

    def get_sample(self, checkpoint_index, output_directory):
        return self._read_sample_from_file(f"checkpoint_{checkpoint_index:02d}_sample_of_ghost_profile.npy",
                                           output_directory)
