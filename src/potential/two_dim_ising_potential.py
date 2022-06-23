"""Module for the TwoDimIsingPotential class."""
from .potential import Potential
from base. exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import number_of_particles, size_of_particle_space
import logging
import numpy as np


class TwoDimIsingPotential(Potential):
    r"""
    This class implements the two-dimensional Ising potential
        U = - prefactor * exchange_constant * \sum_{i = 1}^N \sum_{j \in S_i} x[i] * x[j],
        where S_i is the set of the 2D neighbours of particle i and x[i] = \pm 1 is the spin of particle i.
    """

    def __init__(self, lattice_length: int = 8, exchange_constant: float = 1.0, prefactor: float = 1.0):
        """
        The constructor of the TwoDimIsingPotential class.

        Parameters
        ----------
        lattice_length : int
            The number of lattice sites along each dimension of the square lattice.
        exchange_constant : float
            The spin--spin exchange constant.
        prefactor : float
            The prefactor k of the potential.
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        base.exceptions.ConfigurationError
            If exchange_constant is not greater than 0.0.
        """
        super().__init__(prefactor=prefactor)
        if exchange_constant == 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for exchange_constant in "
                                     f"{self.__class__.__name__}.")
        # todo configure integer-valued size_of_particle_space in model_settings
        if not (type(size_of_particle_space) and size_of_particle_space >= 2):
            raise ConfigurationError(f"size_of_particle_space must be an integer greater than 1 when using "
                                     f"{self.__class__.__name__}.")
        self._lattice_length = lattice_length
        self._potential_constant = - prefactor * exchange_constant
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           exchange_constant=exchange_constant, prefactor=prefactor)

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
        return self._potential_constant * np.sum([positions[index] * positions[self._get_east_neighbour(index)] +
                                                  positions[index] * positions[self._get_north_neighbour(index)]
                                                  for index in range(number_of_particles)])

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the potential difference resulting from moving the single active particle to candidate_position.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the proposed position of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        sum_of_neighbouring_spins = (positions[self._get_east_neighbour(active_particle_index)] +
                                     positions[self._get_north_neighbour(active_particle_index)] +
                                     positions[self._get_west_neighbour(active_particle_index)] +
                                     positions[self._get_south_neighbour(active_particle_index)])
        return self._potential_constant * sum_of_neighbouring_spins * (candidate_position -
                                                                       positions[active_particle_index])

    def initialised_position_array(self):
        """
        Returns the initial positions array.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of length number_of_particles; each element is 1 or -1 and represents the
            position (or, equivalently, spin value) of a single particle (or, equivalently, spin).
        """
        return np.random.choice([-1, 1], size=number_of_particles)

    def _get_east_neighbour(self, lattice_site_index):
        return lattice_site_index + (
                lattice_site_index + 1) % self._lattice_length - lattice_site_index - 1 % self._lattice_length

    def _get_west_neighbour(self, lattice_site_index):
        return lattice_site_index + (lattice_site_index - 1 + self._lattice_length) % self._lattice_length - (
                lattice_site_index + self._lattice_length) % self._lattice_length

    def _get_north_neighbour(self, lattice_site_index):
        return lattice_site_index + self._lattice_length * (
                (int(lattice_site_index / self._lattice_length) + 1) % self._lattice_length -
                (int(lattice_site_index / self._lattice_length)) % self._lattice_length)

    def _get_south_neighbour(self, lattice_site_index):
        return lattice_site_index + self._lattice_length * (
                (int(lattice_site_index / self._lattice_length) + self._lattice_length - 1) % self._lattice_length -
                (int(lattice_site_index / self._lattice_length) + self._lattice_length) % self._lattice_length)
