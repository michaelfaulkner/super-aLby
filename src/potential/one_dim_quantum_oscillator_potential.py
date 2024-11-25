"""Module for the OneDimQuantumOscillatorPotential class"""
from .potential import Potential
from base.logging import log_init_arguments
import logging
import numpy as np
from model_settings import number_of_particles, dimensionality_of_particle_space, range_of_initial_particle_positions
from base. exceptions import ConfigurationError

class OneDimQuantumOscillatorPotential(Potential):
    r"""
    This class implements a one dimensional quantum harmonic oscillator.
    The 'potential' is taken to be the dimensionless action, 
    S = \sum_{i=1}^{N_{\tau}}[0.5 * m(x_{i+1}-x_i)^2 + 0.5 * m * \omega^2 * x_i^2]
    with the name 'potential' being a misnomer that is an aterfact of the parent Potential class.
    'Dimensionless' here indicates that the parameters m, \omega and x have been rescaled by factors of \delta \tau,
    the timestep, see Westbroek et. al. 2018.
    
    """
    def __init__(self, prefactor: float = 1.0, dimensionless_mass: float = 1.0, lattice_dimensionality: int = 1, timestep: float = 0.1):
        r"""
        The constructor of the  OneDimQuantumOscillatorPotential class

        Parameters
        ----------
        prefactor : float
            The force constant, k, of the potential.
        mass : float
            The mass of the particle.
        lattice_dimensionality : int 
            The number of Cartesian dimensions of the lattice.
        timestep : float
            The size of the time step, \delta \tau.
        """
        super().__init__(prefactor=prefactor)
        if lattice_dimensionality != 1:
            raise ConfigurationError(f"Give a value of 1 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        self._timestep = timestep
        self._dimensionless_m = dimensionless_mass
        self._lattice_dimensionality = lattice_dimensionality
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor, dimensionless_mass=dimensionless_mass, lattice_dimensionality=lattice_dimensionality, timestep=timestep)
        self._dimensionless_omega = self._dimensionless_m
        self._k = self._dimensionless_omega**2 * self._dimensionless_m / (self._timestep**3)

        
    def get_value(self, positions):
        """
        Returns the dimensionless action for the given particle positions.

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
        dimensionless_positions = self.get_dimensionless_position(positions)
        action = 0.0
        for i in range(0, number_of_particles):
            if i < number_of_particles-1:
                action += self.get_action_at_index(dimensionless_positions[i], dimensionless_positions[i+1])
            else: # impose periodic BCs
                action += self.get_action_at_index(dimensionless_positions[i], dimensionless_positions[0])
        return action


    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the difference in dimensionless action resulting from moving the single active particle to candidate_position.

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
            The difference in dimensionless action resulting from moving the single active particle to candidate_position.
        """
        dimensionless_positions = self.get_dimensionless_position(positions)
        if active_particle_index < number_of_particles-1:
            current_action = (self.get_action_at_index(dimensionless_positions[active_particle_index-1], dimensionless_positions[active_particle_index]) +
                                self.get_action_at_index(dimensionless_positions[active_particle_index], dimensionless_positions[active_particle_index+1]))
            candidate_action = (self.get_action_at_index(dimensionless_positions[active_particle_index-1], candidate_position / self._timestep) +
                                self.get_action_at_index(candidate_position / self._timestep, dimensionless_positions[active_particle_index+1]))
        else: # periodic BCs - note index = 0 case is accounted for above as 0-1 = -1 and array[-1] gives last element
            current_action = (self.get_action_at_index(dimensionless_positions[active_particle_index-1], dimensionless_positions[active_particle_index]) +
                                self.get_action_at_index(dimensionless_positions[active_particle_index], dimensionless_positions[0]))
            candidate_action = (self.get_action_at_index(dimensionless_positions[active_particle_index-1], candidate_position / self._timestep) +
                                self.get_action_at_index(candidate_position / self._timestep, dimensionless_positions[0]))

        return candidate_action - current_action

    def initialised_position_array(self):
        """
        Returns the initial positions array. 
        NOTE Currently only has functionality for initialising all positions as 0.0.

        TODO add functionality to specify initial and final positions, calculate a simple
        straight path between them and initilaise with this.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        """
        if dimensionality_of_particle_space != 1:
            raise ConfigurationError(
                f"Give a value of type None, float or int for size_of_particle_space in the ModelSettings"
                f"section when using {self.__class__.__name__}")

        if range_of_initial_particle_positions != 0.0:
            raise ConfigurationError(
                f"Give a value of 0.0 for range_of_initial_particle_positions in the ModelSettings"
                f"section when using {self.__class__.__name__}. Functionality not yet provided for"
                f"non-zero initial path.")

        return np.zeros((number_of_particles,1))
        

    def get_gradient_at_index(self, positions, index):
        """
        Returns the action gradient at a given index

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        index : int
            The time index, i, of the position being considered.
        Returns
        -------
        float
            The dimensionless action gradient."""
        
        return self._dimensionless_m * ((2 + self._dimensionless_omega**2) * positions[index]
                                         - positions[index+1] - positions[index-1])
    

    def get_dimensionless_position(self, positions):
        r"""
        Returns the dimensionless position, x/(\delta \tau)

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        Returns
        -------
        float
            The dimensionless position.

        """

        return positions / self._timestep
    
    def get_action_at_index(self, dimensionless_position_at_index, dimensionless_position_at_next_index):
        """
        Returns the contribution to the dimensionless action from a given index and position.
        Parameters
        ----------
        index : int
            The time index, i, of the position being considered.
        position_at_index : float
            The position of the particle at that index.
        position_at_next_index : float
            The position of the particle at index+1.
        Returns
        -------
        float
            The contribution to the dimensionless action at the given index.
        """

        return (0.5 * self._dimensionless_m * (dimensionless_position_at_next_index - dimensionless_position_at_index)**2 +
                0.5 * self._dimensionless_m * self._dimensionless_omega**2 * dimensionless_position_at_index**2)   