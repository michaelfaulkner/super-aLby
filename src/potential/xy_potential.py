import numpy as np
from .continuous_potential import ContinuousPotential
from model_settings import number_of_particles
from base.logging import log_init_arguments
from base. exceptions import ConfigurationError
import logging
from helper_methods import get_east_neighbour, get_north_neighbour, get_west_neighbour, get_south_neighbour


class XyPotential(ContinuousPotential):

    """
    This class implements the 2D XY model potential
    """

    def __init__(self, prefactor: float = 1.0,  lattice_dimensionality: int = 2): # , **kwargs
        """
        The constructor of the XyPotential class.
        6 
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

        # do east and north neighbours only - periodic BCs
        # neighbour - self for these
        # opposite for south and west


        return self.potential_constant * np.sum([-(np.cos(positions[get_north_neighbour(index, self._lattice_length)]-positions[index]) +
                                                   np.cos(positions[get_east_neighbour(index, self._lattice_length)]-positions[index]))
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
        Returns the potential difference resulting from moving the single active particle's spin to candidate_position.

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

        current_potential = self.sum_nearest_neighbours(active_particle_index, positions[active_particle_index], positions)
        candidate_potential = self.sum_nearest_neighbours(active_particle_index, candidate_position, positions)

        return self.potential_constant*(candidate_potential - current_potential)


    def sum_nearest_neighbours(self, active_particle_index, lattice_site_value, positions):

        """
        Returns the potential at lattice_site_index by performing a sum over nearest neighbours.

        Parameters
        ----------
        lattice_site_index : int
            The index of the lattice site.
        lattice_site_value : float
            The phase of the spin of the particle at lattice_site_index.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        float
            The potential at lattice_site_index.
        """

        return  -(np.cos(positions[get_north_neighbour(active_particle_index, self._lattice_length)] - lattice_site_value) +
                        np.cos(positions[get_east_neighbour(active_particle_index, self._lattice_length)] - lattice_site_value) +
                        np.cos(lattice_site_value -positions[get_south_neighbour(active_particle_index, self._lattice_length)]) +
                        np.cos(lattice_site_value - positions[get_west_neighbour(active_particle_index, self._lattice_length)]))
