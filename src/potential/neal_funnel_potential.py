"""Module for the NealFunnelPotential class."""
from .one_dimensional_particle_space_potential import OneDimensionalParticleSpacePotential
from base.logging import log_init_arguments
import logging
import math
import numpy as np


class NealFunnelPotential(OneDimensionalParticleSpacePotential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0 .
    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the NealFunnel class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If dimensionality_of_particle_space does not equal 1.
        """
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        float
            The potential.
        """
        positions = np.reshape(positions, tuple([positions.shape[i] for i in range(len(positions.shape) - 1)]))
        return positions[0] ** 2 / 18.0 + 9 * positions[0] / 2.0 + (
                math.exp(-positions[0]) * np.sum(positions[1:len(positions)] ** 2) / 2.0)

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        numpy.ndarray
            The gradient.
        """
        positions = np.reshape(positions, tuple([positions.shape[i] for i in range(len(positions.shape) - 1)]))
        gradient = np.zeros(len(positions))
        gradient[0] = positions[0] / 9.0 + 9 / 2.0 - (
                math.exp(-positions[0]) * np.sum(positions[1:len(positions)] ** 2) / 2.0)
        gradient[1:len(positions)] = 2.0 * positions[1:len(positions)] * math.exp(-positions[0])
        return self._get_higher_dimension_array(gradient)
