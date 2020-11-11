"""Module for the ExponentialPowerKineticEnergy class."""
from .kinetic_energy import KineticEnergy
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
            The power to which each momentum component is raised. For potentials with leading order term |x|^a, the
            optimal choice that ensures robust dynamics is given by power = 1 + 1 / (a - 1) for a >= 2 and
            power = 1 + 1 / (a + 1) for a <= -1.

        Raises
        ------
        base.exceptions.ValueError
            If the power equals 0.
        """
        if power == 0.0:
            raise ValueError(
                "Give a value not equal to 0.0 as the power associated with the kinetic energy {0}.".format(
                    self.__class__.__name__))
        self._one_over_power = 1.0 / power
        self._one_over_beta_to_powerth_root = beta ** (- 1.0 / power)
        self._power = power
        self._power_minus_two = power - 2.0
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power)

    def get_value(self, momentum):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momentum associated with each position.

        Returns
        -------
        float
            The kinetic energy.
        """
        return self._one_over_power * np.sum(np.absolute(momentum) ** self._power)

    def get_gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momentum associated with each position.

        Returns
        -------
        numpy.ndarray
            The gradient of the kinetic energy.
        """
        return momentum * np.absolute(momentum) ** self._power_minus_two

    def get_momentum_observation(self):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momentum associated with each position.
        """
        if dimensionality_of_particle_space == 1:
            return np.array(generalised_power_distribution.rnormp(
                number_of_particles, sigmap=self._one_over_beta_to_powerth_root, p=self._power))
        return np.array([generalised_power_distribution.rnormp(dimensionality_of_particle_space,
                                                               sigmap=self._one_over_beta_to_powerth_root,
                                                               p=self._power) for _ in range(number_of_particles)])
