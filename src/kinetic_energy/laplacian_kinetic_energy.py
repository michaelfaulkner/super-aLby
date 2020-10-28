"""Module for the LaplacianKineticEnergy class."""
from base.logging import log_init_arguments
from .kinetic_energy import KineticEnergy
import logging
import numpy as np


class LaplacianKineticEnergy(KineticEnergy):
    """
    This class implements the Laplacian kinetic energy K = sum(|p[i]|)
    """

    def __init__(self):
        """
        The constructor of the LaplacianKineticEnergy class.
        """
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)

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
        return np.sum(np.absolute(momentum))

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
        return np.sign(momentum)

    def get_momentum_observation(self, momentum):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Parameters
        ----------
        momentum : numpy.ndarray
            The current momentum associated with each position.

        Returns
        -------
        numpy.ndarray
            A new momentum associated with each position.
        """
        return np.random.laplace(size=len(momentum))
