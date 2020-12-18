"""Module for the SuperRelativisticZigZagKineticEnergy class."""
from .zig_zag_kinetic_energy import ZigZagKineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta
import logging
import numpy as np


class SuperRelativisticZigZagKineticEnergy(ZigZagKineticEnergy):
    """
    This class implements the super-relativistic kinetic energy

        K = sum((1 + gamma^(-1) p[i] ** 2) ** (power / 2) / power),

    using multiple one-dimensional zig-zag algorithms to draw observations from its probability distribution.
    """

    def __init__(self, gamma: float = 1.0, power: float = 1.0, zig_zag_observation_parameter: float = 10.0):
        """
        The constructor of the SuperRelativisticZigZagKineticEnergy class.

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
        zig_zag_observation_parameter : float
            The distance travelled through one-component momentum space (during the zig-zag algorithm) between
            observations of the one-component momentum distribution.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the gamma equals 0.0.
        base.exceptions.ConfigurationError
            If the power equals 0.
        base.exceptions.ConfigurationError
            If zig_zag_observation_rate is less than 0.0.
        """
        if gamma == 0.0:
            raise ConfigurationError("Give a value not equal to 0.0 as the tuning parameter for the relativistic "
                                     "kinetic energy {0}.".format(self.__class__.__name__))
        if power == 0.0:
            raise ConfigurationError(
                "Give a value not equal to 0.0 as the power associated with the kinetic energy {0}.".format(
                    self.__class__.__name__))
        self._one_over_gamma = 1.0 / gamma
        self._root_gamma = gamma ** 0.5
        self._power_over_two = 0.5 * power
        self._power_over_two_minus_one = self._power_over_two - 1.0
        self._minus_power_over_beta = - power / beta
        self._one_over_power = 1.0 / power
        self._two_over_power = 2.0 / power
        super().__init__(zig_zag_observation_parameter=zig_zag_observation_parameter)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, gamma=gamma, power=power,
                           zig_zag_observation_parameter=zig_zag_observation_parameter)

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

    def _get_distance_through_uphill_region(self):
        r"""
        Returns the distance $|\eta|$ travelled (before the next zig-zag event) through the uphill part of
        one-dimensional momentum space. This is calculated by inverting

            $ \rand(0.0, 1.0) =
                \exp \left[- \beta * \int_0^{\eta} \left(\frac{\partial K}{\partial p}\right)^+ dp \right] $

        Returns
        -------
        float
            The distance travelled through the uphill part of one-dimensional momentum space.
        """
        return self._root_gamma * (
                (1.0 + self._minus_power_over_beta * np.log(1.0 - np.random.random())) ** self._two_over_power -
                1.0) ** 0.5
