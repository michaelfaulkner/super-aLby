"""Module for the ParticleNumberSampler class."""
from .observation_sampler import ObservationSampler
import numpy as np


class ParticleNumberSampler(ObservationSampler):
    """
    Class for taking observations of the number of real particles N (for grand-canonical event-chain Monte Carlo).  The
    ghost particle is not counted.
    """

    def __init__(self):
        """The constructor of the ParticleNumberSampler class."""
        super().__init__()

    def get_empty_sample_array(self, total_number_of_iterations):
        return np.zeros((total_number_of_iterations, 1))

    def get_observation(self, momenta, positions, potential):
        """Potentials without a ghost (all canonical potentials) have N = len(positions)."""
        if hasattr(potential, "get_number_of_real_particles"):
            return potential.get_number_of_real_particles(positions)
        return len(positions)

    def output_sample(self, sample, checkpoint_index, output_directory):
        self._write_sample_to_file(sample, f"checkpoint_{checkpoint_index:02d}_sample_of_particle_number.npy",
                                   output_directory)

    def get_sample(self, checkpoint_index, output_directory):
        return self._read_sample_from_file(f"checkpoint_{checkpoint_index:02d}_sample_of_particle_number.npy",
                                           output_directory)
