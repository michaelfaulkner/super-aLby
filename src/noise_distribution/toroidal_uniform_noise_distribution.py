"""Module for the ToroidalUniformNoiseDistribution class."""
from .continuous_noise_distribution import ContinuousNoiseDistribution
from model_settings import dimensionality_of_particle_space
import numpy as np
from base.vectors import get_shortest_vectors_on_torus


class ToroidalUniformNoiseDistribution(ContinuousNoiseDistribution):
    """
    This class provides functionality for noise distributions that propose discrete changes in the position of the
        active particle (on a toroidal Euclidean particle space) using a uniform noise distribution.
    """

    def __init__(self, initial_width_of_noise_distribution: float = 0.1):
        """
        The constructor of the ToroidalUniformNoiseDistribution class.

        Parameters
        ----------
        initial_width_of_noise_distribution : float
            The initial width of the uniform distribution.
        """
        super().__init__(initial_width_of_noise_distribution)

    def get_candidate_position(self, active_particle_index, positions):
        """
        Returns a candidate position for the active particle in the Metropolis algorithm.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        positions : numpy.ndarray(number_of_particles, dimensionality_of_particle_space)
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
        return get_shortest_vectors_on_torus((positions[active_particle_index] +
                                              np.random.uniform(-0.5 * self.width_of_noise_distribution,
                                                                0.5 * self.width_of_noise_distribution,
                                                                size=dimensionality_of_particle_space)))
