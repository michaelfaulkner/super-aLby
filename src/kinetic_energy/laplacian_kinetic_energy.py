"""Module for the LaplacianKineticEnergy class."""
import numpy as np
from .kinetic_energy import KineticEnergy


class LaplacianKineticEnergy(KineticEnergy):
    """
    This class implements the Laplacian kinetic energy K = sum(|p[i]|)
    """

    def __init__(self):
        """
        The constructor of the LaplacianKineticEnergy class.
        """
        super().__init__()

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
