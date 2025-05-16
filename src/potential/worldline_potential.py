"""Module for the WorldlinePotential class"""
import numpy as np
from .smooth_potential import SmoothPotential
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from model_settings import number_of_quantum_particles, number_of_timeslices, number_of_particles
from helper_methods import get_east_neighbour_worldline, get_west_neighbour_worldline

class WorldlinePotential(SmoothPotential, metaclass=ABCMeta):
    """
    Abstract class for worldline potentials. The extra methods provided are those required to calculate the action
    of the system.
    """
    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0,
                  timestep: float = 1.0, **kwargs):
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
        timestep : float
            The size of the time step
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor=prefactor)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        self._lattice_dimensionality = lattice_dimensionality
        self._mass = mass
        self._timestep = timestep
        self._omega = mass

    def get_value(self, positions):
        """
        Returns the dimensionless action for the given particle positions.  Note that the dimensional action
            S * self._timestep is analogous to the potential of a statistical-physics model (since hbar is considered
            analogous to the inverse temperature (beta) of a stat-physics model; S denotes the raw action).

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        Returns
        -------
        float
            The dimensionless action.
        """
        dimensionless_action = 0.0
        for active_particle_index in range(0, number_of_particles):
            dimensionless_action += self._get_pairwise_dimensionless_action(
                positions[active_particle_index], positions[get_east_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                                                number_of_quantum_particles)])
        return dimensionless_action
    
    def get_gradient(self, positions):
        # TODO implement get_gradient() function in this class
        """
        Returns the gradient of the dimensionless action for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
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
        candidate_position : float
            A float representing the proposed position of the active particle.
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.

        Returns
        -------
        float
            The dimensionless-action difference.
        """
        current_dimensionless_action = (
                self._get_pairwise_dimensionless_action(positions, active_particle_index, 
                    positions[get_west_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                            number_of_quantum_particles)],
                    positions[active_particle_index]) +
                self._get_pairwise_dimensionless_action(positions, active_particle_index,
                    positions[active_particle_index],
                    positions[get_east_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                            number_of_quantum_particles)]))
        
        candidate_dimensionless_action = (
                self._get_pairwise_dimensionless_action(positions, active_particle_index,
                    positions[get_west_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                            number_of_quantum_particles)], candidate_position) +
                self._get_pairwise_dimensionless_action(positions, active_particle_index, candidate_position,
                    positions[get_east_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                            number_of_quantum_particles)]))
        return candidate_dimensionless_action - current_dimensionless_action
    
    
    def _get_pairwise_dimensionless_action(self, positions, active_particle_index, position_at_active_particle_index,
                                            position_at_neighbouring_worldline_index):
        """
        Returns the contribution to the dimensionless action from a given pair of positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The index of the active particle.
        position_at_active_particle_index : float
            The position of the particle at the active particle index.
        position_at_neighbouring_worldline_index : float
            The position of the same index quantum particle at a neighbouring timeslice.
        Returns
        -------
        float
            The pairwise contribution to the dimensionless action.
        """

        return self._get_kinetic_action_term(active_particle_index,  position_at_active_particle_index, 
                                            position_at_neighbouring_worldline_index) + \
                                            self._get_potential_action_term(positions, active_particle_index,
                                                                             position_at_active_particle_index)
    
    def _get_kinetic_action_term(self, active_particle_index,  position_at_active_particle_index, 
                                 position_at_neighbouring_worldline_index):
        """
        Returns the kinetic energy contribution to the pairwise dimensionless action.
        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The index of the active particle.
        position_at_neighbouring_worldline_index : float
            The position of the same index quantum particle at a neighbouring timeslice
        Returns
        -------
        float
            The kinetic energy contribution to the pairwise dimensionless action.
        """

        return 0.5 * self._mass * (position_at_neighbouring_worldline_index -
                position_at_active_particle_index)**2 / self._timestep
    
    @abstractmethod
    def _get_potential_action_term(self, positions, active_particle_index):
        """
        Returns the potential energy contribution to teh pairwise dimensionless action
        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The index of the active particle.
        Returns
        -------
        float
            The potential energy contribution to the pairwise dimensionless action.
        """
        raise NotImplementedError
