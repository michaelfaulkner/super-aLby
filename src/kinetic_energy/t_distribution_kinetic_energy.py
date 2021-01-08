"""Module for the TDistributionKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta, dimensionality_of_momenta_array
import logging
import numpy as np


class TDistributionKineticEnergy(KineticEnergy):
    """
    This class implements the t-distribution kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, degrees_of_freedom: int = 1):
        """
        The constructor of the TDistributionKineticEnergy class.

        Parameters
        ----------
        degrees_of_freedom : int
            Number of degrees of freedom of t-distribution.

        Raises
        ------
        base.exceptions.ConfigurationError
            If degrees_of_freedom is less than 1.
        base.exceptions.ConfigurationError
            If beta does not equal 1.0.
        """
        if degrees_of_freedom < 1:
            raise ConfigurationError(f"Give a value not less than 1 as degrees_of_freedom for "
                                     f"{self.__class__.__name__}.")
        if beta != 1.0:
            raise ConfigurationError(f"Set beta equal to 1.0 when using {self.__class__.__name__}.")
        self._degrees_of_freedom = float(degrees_of_freedom)
        self._degrees_of_freedom_minus_one = self._degrees_of_freedom - 1.0
        self._degrees_of_freedom_plus_one = self._degrees_of_freedom + 1.0
        self._degrees_of_freedom_plus_one_over_two = 0.5 * self._degrees_of_freedom_plus_one
        self._one_over_degrees_of_freedom = 1.0 / self._degrees_of_freedom
        super().__init__()
        log_init_arguments(
            logging.getLogger(__name__).debug, self.__class__.__name__, degrees_of_freedom=degrees_of_freedom)

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
        return np.sum(self._degrees_of_freedom_plus_one_over_two * np.log(self._one_over_degrees_of_freedom *
                                                                          (1.0 + momenta ** 2)))

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
        return self._degrees_of_freedom_plus_one * momenta / (self._degrees_of_freedom + momenta ** 2)

    def get_momentum_observations(self):
        """
        Returns an observation of the momenta from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the newly observed momentum of a single particle.
        """
        return np.random.standard_t(df=self._degrees_of_freedom, size=dimensionality_of_momenta_array)
