"""Module for the abstract SignFlipNoiseDistribution class."""
from .noise_distribution import NoiseDistribution


class SignFlipNoiseDistribution(NoiseDistribution):
    r"""Class for flipping the sign of each Cartesian component of the active-particle position."""

    def __init__(self, initial_width_of_noise_distribution=None):
        """
        The constructor of the SignFlipNoiseDistribution class.

        Parameters
        ----------
        initial_width_of_noise_distribution : None
            A dummy variable to allow Mediator to tune the width of continuous proposal distributions.
        """
        super().__init__(initial_width_of_noise_distribution)

    def get_candidate_position(self, active_particle_index, positions):
        """
        Returns a candidate position for the active particle in the Metropolis algorithm.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is an int and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is an int and
            represents one Cartesian component of the proposed position of the active particle.
        """
        return - positions[active_particle_index]
