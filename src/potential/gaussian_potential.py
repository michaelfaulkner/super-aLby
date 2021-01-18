"""Module for the GaussianPotential class."""
from .potential import Potential
from base. exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import size_of_particle_space
import logging
import numpy as np


class GaussianPotential(Potential):
    """
    This class implements the Gaussian potential U = sum(x[i] ** 2 / 2)
    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the GaussianPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        base.exceptions.ConfigurationError
            If element is not None for element in size_of_particle_space.
        """
        super().__init__(prefactor=prefactor)
        for element in size_of_particle_space:
            if element is not None:
                raise ConfigurationError(f"For each component of size_of_particle_space, give None when using "
                                         f"{self.__class__.__name__}.")
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.

        Returns
        -------
        float
            The potential.
        """
        return 0.5 * np.sum(positions ** 2)

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        return positions
