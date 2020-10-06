"""Module for the RelativisticKineticEnergy class."""
from .relativistic_kinetic_energy import RelativisticKineticEnergy
import numpy as np


class StandardRelativisticKineticEnergy(RelativisticKineticEnergy):
    """
    This class implements the relativistic kinetic energy K = sum((1 + gamma^(-1) p[i] ** 2) ** (1 / 2))
    """

    def __init__(self, gamma=1.0):
        """
        The constructor of the RelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        """
        super().__init__(gamma=gamma)

    def current_value(self, momentum):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The kinetic energy.
        """
        return np.sum((1 + self._one_over_gamma * momentum ** 2) ** 0.5)

    def gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        numpy array
            The gradient of the kinetic energy.
        """
        return self._one_over_gamma * momentum * (1 + self._one_over_gamma * momentum ** 2) ** (- 0.5)
