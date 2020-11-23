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
        return np.sum(np.absolute(momenta))

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
        return np.sign(momenta)

    def get_momentum_observation(self):
        """
        Return an observation of the momenta from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momenta associated with each positions.
        """
        return np.random.laplace(scale=one_over_beta, size=dimensionality_of_momenta_array)
