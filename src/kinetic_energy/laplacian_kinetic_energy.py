"""Module for the LaplacianKineticEnergy class."""
import numpy as np
from .kinetic_energy import KineticEnergy


class LaplacianKineticEnergy(KineticEnergy):
    """
    This class implements the Laplacian kinetic energy K = sum(|p[i]|)
    """

    def __init__(self, power=1, prefactor=1.0):
        """
        The constructor of the LaplacianKineticEnergy class.

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
        return np.sign(momentum)

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
        return np.sum(np.absolute(momentum))

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
        return np.random.laplace(size=len(momentum))
