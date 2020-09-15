"""Module for the InversePowerPotential class."""
import logging
import math
import numpy as np
from base.logging import log_init_arguments
from .potential import Potential


# noinspection PyMethodOverriding
class NealFunnel(Potential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0

    x is an n-dimensional vector of floats.
    """

    def __init__(self, prefactor=1.0):
        """
        The constructor of the NealFunnel class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__(prefactor=prefactor)

    def derivative(self, separation):
        """
        Return the derivative of the potential.

        Parameters
        ----------
        separation
            For physics models, the separation vector r_ij; in this case, the Bayesian parameter value.

        Returns
        -------
        float
            The derivative.
        """
        deriv = np.array([0.0 for _ in range(len(separation))])
        deriv[0] = separation[0] / 9.0 + 9 / 2.0 - (
                    math.exp(-separation[0]) * np.sum(separation[1:len(separation)] ** 2) / 2.0)
        deriv[1:len(separation)] = 2.0 * separation[1:len(separation)] * math.exp(-separation[0])
        return deriv

    def potential(self, separation):
        """
        Return the potential for the given separation.

        Parameters
        ----------
        separation
            For physics models, the separation vector r_ij; in this case, the Bayesian parameter value.

        Returns
        -------
        float
            The potential.
        """
        return separation[0] ** 2 / 18.0 + 9 * separation[0] / 2.0 + math.exp(-separation[0]) * np.sum(
            separation[1:10] ** 2) / 2.0
