"""Module for the GaussianKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.logging import log_init_arguments
from model_settings import dimensionality_of_momenta_array
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
        return 0.5 * np.sum(momenta ** 2)

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
        return momenta

    def get_momentum_observations(self, temperature):
        """
        Returns an observation of the momenta from the kinetic-energy distribution.

        Parameters
        ----------
        temperature : float
            The sampling temperature.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the newly observed momentum of a single particle.
        """
        return np.random.normal(scale=temperature ** 0.5, size=dimensionality_of_momenta_array)
