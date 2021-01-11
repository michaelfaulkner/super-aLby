"""Module for the SuperRelativisticKineticEnergy class."""
from .zig_zag_kinetic_energy import ZigZagKineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta
import logging
import numpy as np


class SuperRelativisticKineticEnergy(ZigZagKineticEnergy):
    """
    This class implements the super-relativistic kinetic energy

        K = sum((1 + gamma^(-1) |p[i]| ** power_2) ** (power_1 / power_2) / power_1),

    using multiple one-dimensional zig-zag algorithms to draw observations from its probability distribution.
    """

    def __init__(self, gamma: float = 1.0, power_1: float = 1.0, power_2: float = 2.0,
                 zig_zag_observation_parameter: float = 5.0):
        """
        The constructor of the SuperRelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momenta values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        power_1 : float
            The power to which each momenta- and power_2-dependent part of the super-relativistic kinetic energy is
            raised. For potentials with leading order term |x|^{a_1}, the optimal choice that ensures robust dynamics
            is given by power_1 = 1 + 1 / (a_1 - 1) for a_1 >= 2 and power_1 = 1 + 1 / (a_1 + 1) for a_1 <= -1.
        power_2 : float
            The power to which each momentum component is raised within the innermost parentheses of the
            super-relativistic kinetic energy. For potentials with lowest order term |x|^{a_2}, the optimal choice that
            ensures robust dynamics is given by power_2 = 1 - 1 / (a_2 - 1) for a_2 >= 2 and power_2 = 1 - 1 / (a_2 +
            1) for a_2 <= -1.
        zig_zag_observation_parameter : float
            The normalised distance travelled through one-component momentum space (during the zig-zag algorithm)
            between observations of the one-component momentum distribution. zig_zag_observation_parameter / beta is the
            (non-normalised) distance travelled between observations.

        Raises
        ------
        base.exceptions.ConfigurationError
            If gamma is not greater than 0.0.
        base.exceptions.ConfigurationError
            If power_1 is less than 1.0.
        base.exceptions.ConfigurationError
            If power_2 is less than 0.0.
        base.exceptions.ConfigurationError
            If zig_zag_observation_rate is less than 0.0.
        """
        if gamma <= 0.0:
            raise ConfigurationError(f"Give a greater than 0.0 as the tuning parameter gamma for "
                                     f"{self.__class__.__name__}.")
        if power_1 < 1.0:
            raise ConfigurationError(f"Give a value not less than 1.0 as the power_1 associated with "
                                     f"{self.__class__.__name__}.")
        if power_2 < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as the power_2 associated with "
                                     f"{self.__class__.__name__}.")
        self._gamma = gamma
        self._one_over_gamma = 1.0 / gamma
        self._power_1_over_beta = power_1 / beta
        self._one_over_power_1 = 1.0 / power_1
        self._power_2 = power_2
        self._one_over_power_2 = 1.0 / power_2
        self._power_2_minus_two = power_2 - 2.0
        self._power_ratio = power_1 / power_2
        self._power_ratio_minus_one = self._power_ratio - 1.0
        self._one_over_power_ratio = 1.0 / self._power_ratio
        super().__init__(zig_zag_observation_parameter=zig_zag_observation_parameter)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, gamma=gamma, power_1=power_1,
                           power_2=power_2, zig_zag_observation_parameter=zig_zag_observation_parameter)

    def get_value(self, momenta):
        """
        Returns the kinetic energy for the given particle momenta.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.

        Returns
        -------
        float
            The kinetic energy.
        """
        return self._one_over_power_1 * np.sum((1 + self._one_over_gamma * np.abs(momenta) ** self._power_2) **
                                               self._power_ratio)

    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy for the given particle momenta.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the kinetic energy of a single
            particle.
        """
        return self._one_over_gamma * momenta * np.abs(momenta) ** self._power_2_minus_two * (
                1 + self._one_over_gamma * np.abs(momenta) ** self._power_2) ** self._power_ratio_minus_one

    def _get_distance_from_origin_to_event(self):
        r"""
        Returns the distance $|\eta|$ travelled (before the next zig-zag event) through the uphill part of
        one-dimensional momentum space, i.e., from the origin to $\eta$. This is calculated by inverting

            $ \rand(0.0, 1.0) =
                \exp \left[- \beta * \int_0^{\eta} \left(\frac{\partial K}{\partial p}\right)^+ dp \right] $

        Returns
        -------
        float
            The distance travelled (before the next zig-zag event) through the uphill part of one-dimensional momentum
            space.
        """
        return (self._gamma * ((1.0 - self._power_1_over_beta * np.log(1.0 - np.random.random())) **
                               self._one_over_power_ratio - 1.0)) ** self._one_over_power_2
