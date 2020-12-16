"""Module for the ExponentialPowerKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta, dimensionality_of_particle_space, number_of_particles
import logging
import numpy as np


class ExponentialPowerKineticEnergy(KineticEnergy):
    """
    This class implements the exponential-power kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, power: float = 2.0, zig_zag_observation_parameter: float = 20.0):
        """
        The constructor of the ExponentialPowerKineticEnergy class.

        Parameters
        ----------
        power : float
            The power to which each momenta component is raised. For potentials with leading order term |x|^a, the
            optimal choice that ensures robust dynamics is given by power = 1 + 1 / (a - 1) for a >= 2 and
            power = 1 + 1 / (a + 1) for a <= -1.
        zig_zag_observation_parameter : float
            The distance travelled through one-component momentum space (during the zig-zag algorithm) between
            observations of the one-component momentum distribution.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the power equals 0.0.
        base.exceptions.ConfigurationError
            If zig_zag_observation_rate is less than 0.0.
        """
        if power == 0.0:
            raise ConfigurationError(
                "Give a value not equal to 0.0 as the power associated with the kinetic energy {0}.".format(
                    self.__class__.__name__))
        if zig_zag_observation_parameter < 0.0:
            raise ConfigurationError(
                "Give a value not less than 0.0 for zig_zag_observation_rate {0}.".format(self.__class__.__name__))
        self._power = power
        self._power_minus_two = power - 2.0
        self._minus_power_over_beta = - power / beta
        self._one_over_power = 1.0 / power
        if dimensionality_of_particle_space == 1:
            self._stored_momenta = np.array([[1.0e-3 * np.random.choice((-1.0, 1.0)), np.random.choice((-1.0, 1.0))]
                                             for _ in range(number_of_particles)])
        else:
            self._stored_momenta = np.array([[[1.0e-3 * np.random.choice((-1.0, 1.0)), np.random.choice((-1.0, 1.0))]
                                              for _ in range(dimensionality_of_particle_space)]
                                             for _ in range(number_of_particles)])
        self._zig_zag_observation_parameter = zig_zag_observation_parameter
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power)

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
        return self._one_over_power * np.sum(np.absolute(momenta) ** self._power)

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
        return momenta * np.absolute(momenta) ** self._power_minus_two

    def get_momentum_observations(self):
        """
        Return an observation of the momenta from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momenta associated with each positions.
        """
        if dimensionality_of_particle_space == 1:
            self._stored_momenta = np.array([self._get_single_momentum_observation(momentum)
                                             for momentum in self._stored_momenta])
            return self._stored_momenta[:, 0]
        self._stored_momenta = np.array([[self._get_single_momentum_observation(component) for component in momentum]
                                         for momentum in self._stored_momenta])
        return self._stored_momenta[:, :, 0]

    def _get_single_momentum_observation(self, momentum):
        distance_to_travel_before_observation = self._zig_zag_observation_parameter
        if np.sign(momentum[0]) == momentum[1]:
            displacement_magnitude = self._get_uphill_displacement_magnitude()
            if distance_to_travel_before_observation < displacement_magnitude:
                return [momentum[0] + distance_to_travel_before_observation * momentum[1], momentum[1]]
            distance_to_travel_before_observation -= displacement_magnitude
            momentum[0] += displacement_magnitude * momentum[1]
        while True:
            direction_of_motion = - np.sign(momentum[0])
            displacement_magnitude = self._get_uphill_displacement_magnitude() + abs(momentum[0])
            if distance_to_travel_before_observation < displacement_magnitude:
                return [momentum[0] + distance_to_travel_before_observation * direction_of_motion, direction_of_motion]
            distance_to_travel_before_observation -= displacement_magnitude
            momentum[0] += displacement_magnitude * direction_of_motion

    def _get_uphill_displacement_magnitude(self):
        return (self._minus_power_over_beta * np.log(1.0 - np.random.random())) ** self._one_over_power
