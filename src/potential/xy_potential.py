import numpy as np
from .continuous_potential import ContinuousPotential
from model_settings import dimensionality_of_particle_space, number_of_particles, range_of_initial_particle_positions
from base.logging import log_init_arguments
from base. exceptions import ConfigurationError
import logging


class XyPotential(ContinuousPotential):

    """
    This class implements the 2D XY model potential
    """

    def __init__(self, prefactor: float = 1.0,  lattice_dimensionality: int = 2): # , **kwargs
        """
        The constructor of the XyPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        """

        super().__init__(prefactor=prefactor)
        if lattice_dimensionality != 2:
            raise ConfigurationError(f"Give a value of 2 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        lattice_length = number_of_particles ** (1 / lattice_dimensionality)
        if not lattice_length.is_integer():
            raise ConfigurationError(
                f"For the value of number_of_particles in ModelSettings, give lattice_length ** lattice_dimensionality "
                f"when using {self.__class__.__name__}, where lattice_length is an integer not less than 2.")
        
        self._lattice_dimensionality = lattice_dimensionality
        self._lattice_length = int(lattice_length)
        self.potential_constant = prefactor
        

        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, positions):

        """
        Returns the potential function for the given particle positions. Here, 'positions' is a misnomer, inherited from
        the naming conventions of the parent ContinuousPotential class. A 'position' given here is actually a scalar value
        corresponding to the phase/angle of the spin of the corresponding particle.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        
        Returns
        -------
        float
            The potential.
        """
        # for index in number of particles
        # do the nearest neighbours sum:
        # sum (-cos(\theta_j - \theta_index))
        # \theta_index is positions[index]
        # \theta_j is positions[self.get_jth_neighbour(index)] etc etc
        # with j being nearest neighbours of index particle
        # add to running total

        return self.potential_constant * np.sum([-(np.cos(positions[self._get_north_neighbour(index)]-positions[index]) +
                                                   np.cos(positions[self._get_east_neighbour(index)]-positions[index]) +
                                                   np.cos(positions[self._get_south_neighbour(index)]-positions[index]) +
                                                   np.cos(positions[self._get_west_neighbour(index)]-positions[index]))
                                                    for index in range(number_of_particles)])


    def get_gradient(self, positions):

        """
        Returns the gradient of the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        nothing.
        """

        pass

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the potential difference resulting from moving the single active particle to candidate_position.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents the spin angle of the proposed spin of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """


        # potential difference = potential of proposed move - potential of current state
        # get sum of -cos(\theta_j - \theta_candidate) + cos(\theta_j - \theta_i)
        # where \theta_j are nearest neighbour spins
        # \theta_i is the current spin: positions[active_particle_index]
        # \theta_candidate is the proposed spin: candidate_position
        # write/check for nearest neighbour finding function

        # syntax from ising potential:
        #        sum_of_neighbouring_spins = (positions[self._get_east_neighbour(active_particle_index)] +
        #                              positions[self._get_north_neighbour(active_particle_index)] +
        #                              positions[self._get_west_neighbour(active_particle_index)] +
        #                              positions[self._get_south_neighbour(active_particle_index)])
        # return self.potential_constant * sum_of_neighbouring_spins * (candidate_position -
        #                                                               positions[active_particle_index])
        current_potential = -(np.cos(positions[self._get_north_neighbour(active_particle_index)] - positions[ active_particle_index]) +
                                                   np.cos(positions[self._get_east_neighbour(active_particle_index)] - positions[active_particle_index]) +
                                                   np.cos(positions[self._get_south_neighbour( active_particle_index)] - positions[active_particle_index]) +
                                                   np.cos(positions[self._get_west_neighbour(active_particle_index)] - positions[active_particle_index]))
        
        candidate_potential = -(np.cos(positions[self._get_north_neighbour(active_particle_index)] - candidate_position) +
                                                   np.cos(positions[self._get_east_neighbour(active_particle_index)] - candidate_position) +
                                                   np.cos(positions[self._get_south_neighbour( active_particle_index)] - candidate_position) +
                                                   np.cos(positions[self._get_west_neighbour(active_particle_index)] - candidate_position))
        
        #print(f"for index {active_particle_index}: current potential: {self.potential_constant*current_potential}, candidate potential: {self.potential_constant*candidate_potential}, potential difference: {self.potential_constant*(candidate_potential - current_potential)}")

        return self.potential_constant*(candidate_potential - current_potential)





    def get_neighbours(self, lattice_site_index):
        """Returns a list of the four neighbours (on the 2D lattice) of lattice_site_index"""
        return [self._get_east_neighbour(lattice_site_index), self._get_north_neighbour(lattice_site_index),
                self._get_west_neighbour(lattice_site_index), self._get_south_neighbour(lattice_site_index)]

    def _get_east_neighbour(self, lattice_site_index):
        """Returns the eastwards neighbour (on the 2D lattice) of lattice_site_index"""
        return lattice_site_index + (
                lattice_site_index + 1) % self._lattice_length - lattice_site_index % self._lattice_length

    def _get_north_neighbour(self, lattice_site_index):
        """Returns the northwards neighbour (on the 2D lattice) of lattice_site_index"""
        return lattice_site_index + self._lattice_length * (
                (int(lattice_site_index / self._lattice_length) + 1) % self._lattice_length -
                (int(lattice_site_index / self._lattice_length)) % self._lattice_length)

    def _get_west_neighbour(self, lattice_site_index):
        """Returns the westwards neighbour (on the 2D lattice) of lattice_site_index"""
        return lattice_site_index + (lattice_site_index - 1 + self._lattice_length) % self._lattice_length - (
                lattice_site_index + self._lattice_length) % self._lattice_length

    def _get_south_neighbour(self, lattice_site_index):
        """Returns the southwards neighbour (on the 2D lattice) of lattice_site_index"""
        return lattice_site_index + self._lattice_length * (
                (int(lattice_site_index / self._lattice_length) + self._lattice_length - 1) % self._lattice_length -
                (int(lattice_site_index / self._lattice_length) + self._lattice_length) % self._lattice_length)
