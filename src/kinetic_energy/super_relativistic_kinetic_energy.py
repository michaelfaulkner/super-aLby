"""Module for the SuperRelativisticKineticEnergy class."""
from .relativistic_kinetic_energy import RelativisticKineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
import logging
import numpy as np


class SuperRelativisticKineticEnergy(RelativisticKineticEnergy):
    """
    This class implements the super-relativistic kinetic energy
        K = sum((1 + gamma^(-1) p[i] ** 2) ** (power / 2) / power)
    """

    def __init__(self, gamma: float = 1.0, power: float = 1.0):
        """
        The constructor of the SuperRelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momenta values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        power : float
            The power to which each momenta-dependent part of the standard relativistic kinetic energy is raised (in
            order to form the super-relativistic kinetic energy). For potentials with leading order term |x|^a, the
            optimal choice that ensures robust dynamics is given by power = 1 + 1 / (a - 1) for a >= 2 and
            power = 1 + 1 / (a + 1) for a <= -1.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the power equals 0.
        """
        if power == 0.0:
            raise ConfigurationError(
                "Give a value not equal to 0.0 as the power associated with the kinetic energy {0}.".format(
                    self.__class__.__name__))
        self._power_over_two = 0.5 * power
        self._power_over_two_minus_one = self._power_over_two - 1.0
        self._one_over_power = 1.0 / power
        super().__init__(gamma=gamma)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, gamma=gamma, power=power)

    def get_value(self, momenta):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The kinetic energy.
        """
        return self._one_over_power * np.sum((1 + self._one_over_gamma * momenta ** 2) ** self._power_over_two)

    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        numpy.ndarray
            The gradient of the kinetic energy.
        """
        return self._one_over_gamma * momenta * (
                1 + self._one_over_gamma * momenta ** 2) ** self._power_over_two_minus_one
