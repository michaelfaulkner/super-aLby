"""Module for the GaussianKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.logging import log_init_arguments
from model_settings import beta, dimensionality_of_momenta_array, one_over_root_beta
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
        return 0.5 * beta * np.sum(momentum ** 2)

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
        return beta * momentum

    def get_momentum_observation(self):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momentum associated with each position.
        """
        return np.random.normal(scale=one_over_root_beta, size=dimensionality_of_momenta_array)
