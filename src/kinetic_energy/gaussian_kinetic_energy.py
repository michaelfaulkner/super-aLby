"""Module for the GaussianKineticEnergy class."""
from base.logging import log_init_arguments
from .kinetic_energy import KineticEnergy
import logging
import numpy as np


class GaussianKineticEnergy(KineticEnergy):
    """
    This class implements the Gaussian kinetic energy K = sum(p[i] ** 2 / 2)
    """

    def __init__(self):
        """
        The constructor of the GaussianKineticEnergy class.
        """
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)

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
        return 0.5 * np.sum(momentum ** 2)

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
        return np.random.normal(size=len(momentum))
