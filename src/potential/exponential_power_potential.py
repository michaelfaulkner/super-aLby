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

    def __init__(self, power: int = 2, prefactor: float = 1.0):
        """
        The constructor of the ExponentialPowerPotential class.

        Parameters
        ----------
        power : int
            The power to which each component of the support_variable is raised.
        prefactor : float
            The prefactor k of the potential.
        """
        self._one_over_power = 1.0 / power
        self._power = power
        self._power_minus_two = power - 2
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power, prefactor=prefactor)

    def current_value(self, support_variable, charges=None):
        """
        Returns the potential for the given support_variable.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.
        charges : optional
            All the charges needed to calculate the potential; not used in this potential class.

        Returns
        -------
        float
            The potential.
        """
        return self._one_over_power * np.sum(support_variable ** self._power)

    def gradient(self, support_variable, charges=None):
        """
        Returns the gradient of the potential for the given support_variable.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.
        charges : optional
            All the charges needed to calculate the gradient; not used in this potential class.

        Returns
        -------
        numpy array
            The gradient.
        """
        return support_variable * support_variable ** self._power_minus_two
