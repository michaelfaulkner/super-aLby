"""Module for the RelativisticKineticEnergy class."""
from base.logging import log_init_arguments
from .relativistic_kinetic_energy import RelativisticKineticEnergy
import logging
import numpy as np


class StandardRelativisticKineticEnergy(RelativisticKineticEnergy):
    """
    This class implements the relativistic kinetic energy K = sum((1 + gamma^(-1) p[i] ** 2) ** (1 / 2))
    """

    def __init__(self, gamma: float = 1.0):
        """
        The constructor of the RelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        """
        super().__init__(gamma=gamma)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, gamma=gamma)

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
        return np.sum((1 + self._one_over_gamma * momentum ** 2) ** 0.5)

    def get_gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momentum associated with each position.

        Returns
        -------
        numpy array
            The gradient of the kinetic energy.
        """
        return self._one_over_gamma * momentum * (1 + self._one_over_gamma * momentum ** 2) ** (- 0.5)
