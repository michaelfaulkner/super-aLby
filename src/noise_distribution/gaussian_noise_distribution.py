"""Module for the abstract GaussianNoiseDistribution class."""
from .continuous_noise_distribution import ContinuousNoiseDistribution
from model_settings import dimensionality_of_particle_space
import numpy as np


class GaussianNoiseDistribution(ContinuousNoiseDistribution):
    """
    This class is used to generate a proposed discrete change in position for each active particle using a Gaussian
    noise distribution.
    """

    def __init__(self, initial_width_of_noise_distribution: float = 0.1):
        """
        The constructor of the GaussianNoiseDistribution class.

        Parameters
        ----------
        initial_width_of_noise_distribution : float
            The initial standard deviation of the Gaussian distribution.
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
        return np.random.normal(0.0, self.width_of_noise_distribution,
                                size=(number_of_active_particles, dimensionality_of_particle_space))
