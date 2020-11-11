"""Module for the NealFunnelPotential class."""
from .potential import Potential
from base.logging import log_init_arguments
from model_settings import dimensionality_of_particle_space
import logging
import math
import numpy as np


class NealFunnelPotential(Potential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0

    x is an n-dimensional vector of floats.
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
        base.exceptions.ValueError
            If dimensionality_of_particle_space does not equal 1.
        """
        if dimensionality_of_particle_space != 1:
            raise ValueError("Give either None, a float or a list of two float values for "
                             "range_of_initial_particle_positions {0}.".format(self.__class__.__name__))
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, position):
        """
        Returns the potential for the given position.

        Parameters
        ----------
        position : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        float
            The potential.
        """
        return position[0] ** 2 / 18.0 + 9 * position[0] / 2.0 + (
                math.exp(-position[0]) * np.sum(position[1:len(position)] ** 2) / 2.0)

    def get_gradient(self, position):
        """
        Returns the gradient of the potential for the given position.

        Parameters
        ----------
        position : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        numpy.ndarray
            The gradient.
        """
        gradient = np.zeroes(len(position))
        gradient[0] = position[0] / 9.0 + 9 / 2.0 - (
                math.exp(-position[0]) * np.sum(position[1:len(position)] ** 2) / 2.0)
        gradient[1:len(position)] = 2.0 * position[1:len(position)] * math.exp(
            -position[0])
        return gradient
