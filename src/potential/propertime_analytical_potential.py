"""Module for the PropertimeAnalyticalPotential class"""
import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import number_of_quantum_particles, number_of_timeslices, number_of_particles
from model_settings import dimensionality_of_particle_space 


class PropertimeAnalyticalPotential(EuclideanSubspacePotential):
    """
    Abstract class for calculating analytical values for propertiem integrals.
    """
    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0, omega_squared: float = 1.0,
                 timestep: float = 1.0):
        """
        The constructor of the WorldlinePotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        lattice_dimensionality : int
            The number of Cartesian dimensions of the lattice.
        mass : float
            The mass of the particle
        timestep : float
            The size of the time step
        cpp_implementation : bool
            Whether or not to use the C++ implementation of the get_next_event() function -- provides a large speedup.
        proper_time_integral: bool
            Whether or not to integrate over the proper time 

        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor=prefactor)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        if dimensionality_of_particle_space > 1:
            raise ConfigurationError(f"Functionality for dimensionality_of_particle_space greater than one not yet "
                                     f"available for {self.__class__.__name__}.")
        self._lattice_dimensionality = lattice_dimensionality
        self._mass = mass
        self._timestep = timestep
        self.auxillary_particle_value = np.array([1 / self._timestep])
        self.omega_squared = omega_squared


    def get_value(self, positions):
        """
        Returns the dimensionless action for the given particle positions.  Note that the dimensional action
            S * self._timestep is analogous to the potential of a statistical-physics model (since hbar is considered
            analogous to the inverse temperature (beta) of a stat-physics model; S denotes the raw action).

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.

        Returns
        -------
        float
            The dimensionless action.
        """
        
        return self._mass **2 * number_of_particles / self.auxillary_particle_value
    
    def get_gradient(self, positions):
        # TODO implement get_gradient() function in this class
        """
        Returns the gradient of the dimensionless action for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential (i.e. dimensionless
            action) of a single quantum particle.
        """
        raise SystemError(f"The get_gradient method of {self.__class__.__name__} has not been written.")
    
    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the difference in dimensionless action resulting from moving the single active particle to
            candidate_position.  Note that the dimensional action S * self._timestep is analogous to the potential of a
            statistical-physics model (since hbar is considered analogous to the inverse temperature (beta) of a
            stat-physics model; S denotes the raw action).

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : float or numpy.ndarray
            The proposed position of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.

        Returns
        -------
        float
            The dimensionless-action difference.
        """
  
        return -1.0


    def get_potential_difference_of_auxillary_particle(self, candidate_auxillary_particle_value, 
                                                       auxillary_particle_value, positions):
        
         
        current_value = self._mass **2 * number_of_particles / self.auxillary_particle_value
        
        candidate_value = self._mass **2 * number_of_particles / candidate_auxillary_particle_value
        
        return (candidate_value - current_value)

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):

        pass

    def get_initial_positions(self):
        return get_initial_positions_of_smooth_potential(self.__class__.__name__)

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        pass

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        pass

    @staticmethod
    def get_random_event_chain_velocity():
        pass

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        pass