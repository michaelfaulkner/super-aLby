"""Module for the GhostCouplingSampler class."""
from .observation_sampler import ObservationSampler
import numpy as np


class GhostCouplingSampler(ObservationSampler):
    """
    Class for taking observations of the ghost coupling lambda (for grand-canonical event-chain Monte Carlo).
    Observations with lambda <= 0 are exact grand-canonical samples of the real particles; NaN is recorded if there is
    no ghost. This sample is used to determine which measurements are physical and to diagnose whether the lambda
    space is being sampled in the desired fashion e.g. uniformly.
    """

    def __init__(self):
        """The constructor of the GhostCouplingSampler class."""
        super().__init__()

    def get_empty_sample_array(self, total_number_of_iterations):
        return np.zeros((total_number_of_iterations, 1))

    def get_observation(self, momenta, positions, potential):
        """NaN if there is no ghost (including potentials without ghost coupling, i.e. all canonical potentials)."""
        ghost_coupling = getattr(potential, "ghost_coupling", None)
        return np.nan if ghost_coupling is None else ghost_coupling

    def output_sample(self, sample, checkpoint_index, output_directory):
        self._write_sample_to_file(sample, f"checkpoint_{checkpoint_index:02d}_sample_of_ghost_coupling.npy",
                                   output_directory)

    def get_sample(self, checkpoint_index, output_directory):
        return self._read_sample_from_file(f"checkpoint_{checkpoint_index:02d}_sample_of_ghost_coupling.npy",
                                           output_directory)
