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

    def get_candidate_position(self, active_particle_index, positions):
        """
        Returns a candidate position for the active particle in the Metropolis algorithm.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the proposed position of the active particle.
        """
        return positions[active_particle_index] + np.random.normal(0.0, self.width_of_noise_distribution,
                                                                   size=dimensionality_of_particle_space)
