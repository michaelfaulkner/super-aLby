"""Module for the ExponentialPowerPotential class."""
from base.logging import log_init_arguments
from .potential import Potential
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
            The power to which each component of the position is raised.
        prefactor : float
            The prefactor k of the potential.
        """
        self._one_over_power = 1.0 / power
        self._power = power
        self._power_minus_two = power - 2
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power, prefactor=prefactor)

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
        return self._one_over_power * np.sum(np.absolute(position) ** self._power)

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
        numpy array
            The gradient.
        """
        return position * np.absolute(position) ** self._power_minus_two
