"""Module for the GaussianKineticEnergy class."""
import numpy as np
from .kinetic_energy import KineticEnergy


# noinspection PyMethodOverriding
class GaussianKineticEnergy(KineticEnergy):
    """
    This class implements the Gaussian kinetic energy K = \sum_i p[i] ** 2 / 2
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
        return momentum

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
        return 0.5 * np.sum(np.absolute(momentum ** 2))
