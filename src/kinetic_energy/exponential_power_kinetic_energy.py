"""Module for the ExponentialPowerKineticEnergy class."""
from .zig_zag_kinetic_energy import ZigZagKineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta
import logging
import numpy as np


class ExponentialPowerKineticEnergy(ZigZagKineticEnergy):
    """
    This class implements the exponential-power kinetic energy

        K = sum(p[i] ** power / power),

    using multiple one-dimensional zig-zag algorithms to draw observations from its probability distribution.
    """

    def __init__(self, power: float = 2.0, zig_zag_observation_parameter: float = 5.0):
        """
        The constructor of the ExponentialPowerKineticEnergy class.

        Parameters
        ----------
        power : float
            The power to which each momenta component is raised. For potentials with leading order term |x|^a, the
            optimal choice that ensures robust dynamics is given by power = 1 + 1 / (a - 1) for a >= 2 and
            power = 1 + 1 / (a + 1) for a <= -1.
        zig_zag_observation_parameter : float
            The normalised distance travelled through one-component momentum space (during the zig-zag algorithm)
            between observations of the one-component momentum distribution. zig_zag_observation_parameter / beta is the
            (non-normalised) distance travelled between observations.

        Raises
        ------
        base.exceptions.ConfigurationError
            If power is less than 1.0.
        base.exceptions.ConfigurationError
            If zig_zag_observation_rate is less than 0.0.
        """
        super().__init__(zig_zag_observation_parameter=zig_zag_observation_parameter)
        if power < 1.0:
            raise ConfigurationError(f"Give a value not less than 1.0 as the power associated with "
                                     f"{self.__class__.__name__}.")
        self._power = power
        self._power_minus_two = power - 2.0
        self._minus_power_over_beta = - power / beta
        self._one_over_power = 1.0 / power
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power,
                           zig_zag_observation_parameter=zig_zag_observation_parameter)

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
        return self._one_over_power * np.sum(np.absolute(momenta) ** self._power)

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
        return momenta * np.absolute(momenta) ** self._power_minus_two

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
        return (self._minus_power_over_beta * np.log(1.0 - np.random.random())) ** self._one_over_power
