"""Module for the ExponentialPowerKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta, dimensionality_of_particle_space, number_of_particles
import logging
import numpy as np
import rpy2.robjects.packages as r_packages
import rpy2.robjects.numpy2ri as n2ri

n2ri.activate()
generalised_power_distribution = r_packages.importr('normalp')


class ExponentialPowerKineticEnergy(KineticEnergy):
    """
    This class implements the exponential-power kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, power: float = 2.0):
        """
        The constructor of the ExponentialPowerKineticEnergy class.

        Parameters
        ----------
        power : float
            The power to which each momenta component is raised. For potentials with leading order term |x|^a, the
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
        self._one_over_power = 1.0 / power
        self._powerth_root_of_one_over_beta = beta ** (- 1.0 / power)
        self._powerth_root_of_power_over_beta = (power / beta) ** (1.0 / power)
        self._power = power
        self._power_minus_two = power - 2.0
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
        """if dimensionality_of_particle_space == 1:
            return stats.gennorm.rvs(self._power, scale=self._power_over_beta_to_powerth_root, size=number_of_particles)
        return np.array([stats.gennorm.rvs(self._power, scale=self._power_over_beta_to_powerth_root,
                                           size=dimensionality_of_particle_space) for _ in range(number_of_particles)])
                                           """
        # todo why doesn't the following commented-out code work?
        """return self._powerth_root_of_power_over_beta * np.random.choice((- 1.0, 1.0),
                                                                        size=dimensionality_of_momenta_array) * (
                   - np.log(1.0 - np.random.random(size=dimensionality_of_momenta_array))) * self._one_over_power"""
        if dimensionality_of_particle_space == 1:
            return np.array(generalised_power_distribution.rnormp(
                number_of_particles, sigmap=self._powerth_root_of_one_over_beta, p=self._power))
        return np.array([generalised_power_distribution.rnormp(dimensionality_of_particle_space,
                                                               sigmap=self._powerth_root_of_one_over_beta,
                                                               p=self._power) for _ in range(number_of_particles)])
