"""Module for the GeneralisedPowerKineticEnergy class."""
from .kinetic_energy import KineticEnergy
import numpy as np
import rpy2.robjects.packages as r_packages
import rpy2.robjects.numpy2ri as n2ri
n2ri.activate()
generalised_power_distribution = r_packages.importr('normalp')


class GeneralisedPowerKineticEnergy(KineticEnergy):
    """
    This class implements the generalised-power kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, power=2, prefactor=1.0):
        """
        The constructor of the GeneralisedPowerKineticEnergy class.

        Parameters
        ----------
        power : int
            Either the power to which each momentum variable is raised (the generalised-power case) or twice the power
            to which each momentum-dependent part of the relativistic kinetic energy are raised (the super-relativistic
            case).
        prefactor : float, optional
            A general multiplicative prefactor of the potential (and therefore of the kinetic energy).
        """
        self._power = power
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
        return momentum * np.absolute(momentum) ** (self._power - 2)

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
        return self._one_over_power * np.sum(np.absolute(momentum ** self._power))

    def momentum_observation(self, momentum):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Parameters
        ----------
        momentum : numpy_array
            The current momentum associated with each support_variable.

        Returns
        -------
        numpy_array
            A new momentum associated with each support_variable.
        """
        return np.array(generalised_power_distribution.rnormp(len(momentum), p=self._power))