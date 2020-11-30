"""Module for the GaussianKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.logging import log_init_arguments
from model_settings import dimensionality_of_momenta_array, one_over_root_beta
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

    def get_value(self, momenta):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The kinetic energy.
        """
        return 0.5 * np.sum(momenta ** 2)

    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        numpy.ndarray
            The gradient of the kinetic energy.
        """
        return momenta

    def get_momentum_observations(self):
        """
        Return an observation of the momenta from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momenta associated with each positions.
        """
        return np.random.normal(scale=one_over_root_beta, size=dimensionality_of_momenta_array)
