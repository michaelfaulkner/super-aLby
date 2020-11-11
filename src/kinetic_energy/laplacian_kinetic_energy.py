"""Module for the LaplacianKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.logging import log_init_arguments
from model_settings import dimensionality_of_momenta_array, one_over_beta
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

    def get_momentum_observation(self):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momentum associated with each position.
        """
        return np.random.laplace(scale=one_over_beta, size=dimensionality_of_momenta_array)
