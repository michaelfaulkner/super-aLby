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
        Returns the kinetic energy for the given particle momenta.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.

        Returns
        -------
        float
            The kinetic energy.
        """
        return np.sum(np.absolute(momenta))

    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy for the given particle momenta.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the kinetic energy of a single
            particle.
        """
        return np.sign(momenta)

    def get_momentum_observations(self):
        """
        Returns an observation of the momenta from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the newly observed momentum of a single particle.
        """
        return np.random.laplace(scale=one_over_beta, size=dimensionality_of_momenta_array)
