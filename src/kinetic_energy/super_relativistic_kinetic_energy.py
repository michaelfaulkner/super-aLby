"""Module for the GeneralisedPowerKineticEnergy class."""
import numpy as np
from .kinetic_energy import KineticEnergy


# noinspection PyMethodOverriding
class SuperRelativisticKineticEnergy(KineticEnergy):
    """
    This class implements the super-relativistic kinetic energy
        K = \sum_i (1 + gamma^(-1) p[i] ** 2) ** (power / 2) / power
    """

    def __init__(self, gamma=1.0, power=2, prefactor=1.0):
        """
        The constructor of the SuperRelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        power : int
            Either the power to which each momentum variable is raised (the generalised-power case) or twice the power
            to which each momentum-dependent part of the relativistic kinetic energy are raised (the super-relativistic
            case).
        prefactor : float, optional
            A general multiplicative prefactor of the potential (and therefore of the kinetic energy).
        """
        self._one_over_gamma = 1.0 / gamma
        super().__init__(power=power, prefactor=prefactor)

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
        return self._one_over_gamma * momentum * (
                1 + self._one_over_gamma * momentum ** 2) ** (self._power_over_two - 1)

    def kinetic_energy(self, momentum):
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
        return np.sum((1 + self._one_over_gamma * momentum ** 2) ** self._power_over_two) / self._power
