"""Module for the abstract DiscreteNoiseDistribution class."""
from .noise_distribution import NoiseDistribution
from model_settings import dimensionality_of_particle_space
import numpy as np


class DiscreteNoiseDistribution(NoiseDistribution):
    """Class for generating a change in position of plus or minus one."""

    def __init__(self, initial_width_of_noise_distribution: None):
        """
        The constructor of the DiscreteNoiseDistribution class.

        Parameters
        ----------
        initial_width_of_noise_distribution : None
            A dummy variable to allow Mediator to tune the width of continuous proposal distributions.
        """
        super().__init__(initial_width_of_noise_distribution)

    def get_finite_change_in_position(self, number_of_active_particles):
        """
        Returns a proposed finite change in position for each active particle.

        Parameters
        ----------
        number_of_active_particles : int
            An integer representing the number of active particles requiring a proposed change in position.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_active_particles, dimensionality_of_particle_space); each
            element is a float and represents one Cartesian component of the proposed change in position of a single
            active particle.
        """
        return np.random.choice([-1, 1], size=(number_of_active_particles, dimensionality_of_particle_space))
