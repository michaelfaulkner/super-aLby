"""Module for the SuperRelativisticKineticEnergy class."""
from .kinetic_energy_with_adaptive_rejection_sampling import KineticEnergyWithAdaptiveRejectionSampling
import numpy as np


class SuperRelativisticKineticEnergy(KineticEnergyWithAdaptiveRejectionSampling):
    """
    This class implements the super-relativistic kinetic energy
        K = sum((1 + gamma^(-1) p[i] ** 2) ** (power / 2) / power)
    """

    def __init__(self, gamma=1.0, power=2):
        """
        The constructor of the SuperRelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        power : int
            Twice the power to which each momentum-dependent part of the relativistic kinetic energy is raised. For
            potentials with leading order term |x|^a, the optimal choice that ensures robust dynamics is given by
            power = 1 + 1 / (a - 1) for a >= 2 and power = 1 + 1 / (a + 1) for a <= -1.
        """
        self._power_over_two = power / 2
        self._power_over_two_minus_one = self._power_over_two - 1
        self._power_minus_one = power - 1
        self._power_minus_two = power - 2
        super().__init__(gamma=gamma, power=power)

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
                1 + self._one_over_gamma * momentum ** 2) ** self._power_over_two_minus_one

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
        return self._one_over_power * np.sum((1 + self._one_over_gamma * momentum ** 2) ** self._power_over_two)
