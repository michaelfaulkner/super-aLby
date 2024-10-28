"""Module for the OneDimQuantumOscillatorPotential class"""
from .potential import Potential
from base.logging import log_init_arguments
import logging
import numpy as np
from model_settings import number_of_particles, dimensionality_of_particle_space,  range_of_initial_particle_positions
from base. exceptions import ConfigurationError

class OneDimQuantumOscillatorPotential(Potential):
    r"""
    This class implements a one dimensional quantum harmonic oscillator.
    The 'potential' is taken to be the dimensionless action, 
    S = \sum_{i=1}^{N_{\tau}}[0.5 * m(x_{i+1}-x_i)^2 + 0.5 * m * \omega^2 * x_i^2]
    with the name 'potential' being a misnomer that is an aterfact of the parent Potential class.
    
    """
    def __init__(self, prefactor: float = 1.0, mass: float = 1.0, lattice_dimensionality: int = 1, timestep : float = 0.1, force_constant : float = 1.0):
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
        self._k = force_constant
        self._m = mass
        self._lattice_dimensionality = lattice_dimensionality
        self._timestep = timestep
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor, mass=mass, lattice_dimensionality=lattice_dimensionality, timestep=timestep)
        self._omega = np.sqrt(self._k/self._m)
        self._dimensionless_m = self._m * self._timestep
        self._dimensionless_omega = self._omega * self._timestep

        
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

        current_action = self.get_value(positions)
        positions_with_candidate = positions
        positions_with_candidate[active_particle_index] = candidate_position
        candidate_action = self.get_value(positions_with_candidate)
       
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

        return np.zeros(number_of_particles)
        



    def get_dimensionless_position(self, positions):
        r"""
        Returns the dimensionless position, x/(\delta \tau)

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
            The dimensionless position.

        """

        return positions/self._timestep
    
    def get_action_at_index(self, position_at_index, position_at_next_index):
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

        return (0.5 * self._m * (position_at_next_index - position_at_index)**2 +
                0.5 * self._m * self._dimensionless_omega**2 * position_at_index**2)   