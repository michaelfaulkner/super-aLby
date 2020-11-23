"""Module for the ExponentialPowerPotential class."""
from .potential import Potential
from base.logging import log_init_arguments
import logging
import numpy as np


class ExponentialPowerPotential(Potential):
    """
    This class implements the exponential power potential U = sum(x[i] ** power / power)

    x is an n-dimensional vector of floats.
    """

    def __init__(self, power: float = 2, prefactor: float = 1.0):
        """
        The constructor of the ExponentialPowerPotential class.

        Parameters
        ----------
        power : int
            The power to which each component of the positions is raised.
        prefactor : float
            The prefactor k of the potential.
        """
        self._one_over_power = 1.0 / power
        self._power = power
        self._power_minus_two = power - 2
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power, prefactor=prefactor)

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
        return self._one_over_power * np.sum(np.absolute(positions) ** self._power)

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
        return positions * np.absolute(positions) ** self._power_minus_two
