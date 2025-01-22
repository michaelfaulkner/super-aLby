"""Module for the QuantumHarmonicOscillatorPotential class"""
import numpy as np
from .continuous_potential import ContinuousPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
import logging
from model_settings import number_of_particles
from helper_methods import get_east_neighbour, get_west_neighbour


class QuantumHarmonicOscillatorPotential(ContinuousPotential):
    r"""
    This class implements the (currently one-dimensional) potential for the quantum harmonic oscillator resulting
        from the Wick rotation of the Feynman path integral.  The potential corresponds to the dimensionless action,
        \delta\tau \sum_{i=1}^{N_{\tau}}[0.5 * m(x_{i+1} - x_i)^2 / (\delta\tau)^2 + 0.5 * m * \omega^2 * x_i^2],
        where m and \omega are the mass and frequency, respectively.
    """
    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0,
                 timestep: float = 0.1):
        r"""
        The constructor of the QuantumHarmonicOscillatorPotential class

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.
        lattice_dimensionality : int
            The number of Cartesian dimensions of the lattice.
        mass : float
            The mass of the particle.
        timestep : float
            The size of the time step, \delta \tau.
        """
        super().__init__(prefactor=prefactor)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        if lattice_dimensionality != 1:
            raise ConfigurationError(f"Give a value of 1 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        self._lattice_dimensionality = lattice_dimensionality
        self._mass = mass
        self._timestep = timestep
        self._omega = self._mass
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor,
                           lattice_dimensionality=lattice_dimensionality, mass=mass, timestep=timestep)

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
        for particle_index in range(0, number_of_particles):
            dimensionless_action += self._get_pairwise_dimensionless_action(
                positions[particle_index], positions[get_east_neighbour(particle_index, number_of_particles)])
        return dimensionless_action

    def get_gradient(self, positions): #TODO implement get_gradient() function in this class
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
        pass

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
                self._get_pairwise_dimensionless_action(
                    positions[get_west_neighbour(active_particle_index, number_of_particles)],
                    positions[active_particle_index]) +
                self._get_pairwise_dimensionless_action(
                    positions[active_particle_index],
                    positions[get_east_neighbour(active_particle_index, number_of_particles)]))
        candidate_dimensionless_action = (
                self._get_pairwise_dimensionless_action(
                    positions[get_west_neighbour(active_particle_index, number_of_particles)], candidate_position) +
                self._get_pairwise_dimensionless_action(
                    candidate_position, positions[get_east_neighbour(active_particle_index, number_of_particles)]))
        return candidate_dimensionless_action - current_dimensionless_action

    def _get_gradient_at_index(self, positions, particle_index):
        """
        Returns the gradient of the dimensionless action with respect to the particle position at particle_index.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        particle_index : int
            The particle index (i.e., the discretised-time index).
        Returns
        -------
        float
            The dimensionless-action gradient at particle_index.
        """
        return self._mass / self._timestep * ((2.0 + self._timestep ** 2 * self._omega ** 2) * positions[particle_index] -
                             positions[get_west_neighbour(particle_index, number_of_particles)] -
                             positions[get_east_neighbour(particle_index, number_of_particles)]).item()

    def _get_pairwise_dimensionless_action(self, position_at_index, position_at_east_index):
        """
        Returns the contribution to the dimensionless action from a given pair of positions.

        Parameters
        ----------
        position_at_index : float
            The position of the particle at some particle index.
        position_at_east_index : float
            The position of the particle at the site east of the particle index.
        Returns
        -------
        float
            The pairwise contribution to the dimensionless action.
        """

        return 0.5 * self._mass * ((position_at_east_index - position_at_index) ** 2 / self._timestep +
                                   self._timestep * self._omega ** 2 * position_at_index ** 2)

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event for a given active particle index.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The active particle index (i.e., the discretised-time index).
        movement_direction : int
            The direction of movement of the particle, either 1 or -1.
        
        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        """
        shortest_distance_to_next_factor_event = 1.0e10
        neighbouring_indices = np.zeros(3, dtype=np.int32)
        neighbouring_indices[0] = get_west_neighbour(active_particle_index, number_of_particles)
        neighbouring_indices[1] = active_particle_index
        neighbouring_indices[2] = get_east_neighbour(active_particle_index, number_of_particles)
  
        distance_to_next_event = 0.0
        vetoing_index = None
        position_at_index = positions[active_particle_index].item()

        for i in range(3):
            uphill_energy = - np.log(np.random.uniform(0,1))
            
            if i != 1: # considering the neighbour terms
                neighbour_position = positions[neighbouring_indices[i]].item()
                bottom_of_well = neighbour_position
                if (((movement_direction > 0) and (position_at_index < bottom_of_well)) or
                ((movement_direction < 0) and (position_at_index > bottom_of_well))):
                    """advance to the bottom of the well"""
                    distance_to_next_event += np.abs(bottom_of_well - position_at_index)
                    initial_position = bottom_of_well
                else:
                    initial_position = position_at_index
                position_at_index = initial_position
            
                initial_action = 0.5 * self._mass / self._timestep * (initial_position - neighbour_position)**2
                final_action = (uphill_energy + initial_action).item()
                roots = np.roots([0.5 * self._mass / self._timestep, -self._mass / self._timestep * neighbour_position, 0.5 * self._mass / self._timestep * neighbour_position **2 - final_action])
                if (movement_direction > 0) and (roots[0] > 0):
                    final_position = roots[0]
                else:
                    final_position = roots[1]
                if (movement_direction < 0) and (roots[0] < 0):
                    final_position = roots[0]
                else:
                    final_position = roots[1]

                    
            else: # consider x^2 term
                bottom_of_well = 0.0
                if (((movement_direction > 0) and (position_at_index < bottom_of_well)) or
                ((movement_direction < 0) and (position_at_index > bottom_of_well))):
                    """advance to the bottom of the well"""
                    distance_to_next_event += np.abs(bottom_of_well - position_at_index)
                    initial_position = bottom_of_well
                else:
                    initial_position = position_at_index
                position_at_index = initial_position
                initial_action = 0.5 * self._mass * self._timestep * self._omega**2 * initial_position**2
                final_action = uphill_energy + initial_action
                roots = np.roots([0.5 * self._mass * self._timestep * self._omega**2, 0.0, -final_action])
                if (movement_direction > 0) and (roots[0] > 0):
                    final_position = roots[0]
                else:
                    final_position = roots[1]
                if (movement_direction < 0) and (roots[0] < 0):
                    final_position = roots[0]
                else:
                    final_position = roots[1]

            distance_to_next_factor_event = np.abs(final_position - initial_position).item()
            #print(f"neighbour: {i} distance to next factor event: {distance_to_next_factor_event}")
            if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
                shortest_distance_to_next_factor_event = distance_to_next_factor_event
                vetoing_index = neighbouring_indices[i]
                
        #print(f"neighboring indices: {neighbouring_indices}")

        distance_to_next_event += shortest_distance_to_next_factor_event
        return distance_to_next_event, vetoing_index

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction,
                                    veto_index, number_of_bounces, number_of_continuations):
        """Chooses the index and direction for the next active particle in the markov chain"""
        initial_a = active_particle_index
        initial_v = movement_direction
  
        if veto_index == active_particle_index:
            movement_direction = movement_direction * -1
            number_of_bounces += 1
        else:
            active_particle_index = veto_index
            number_of_continuations += 1

        if active_particle_index == initial_a and movement_direction == initial_v:
            raise Exception("Chose the same index and direction twice in a row")
        #print(f"next active particle was chosen: {active_particle_index}")
        return active_particle_index, movement_direction, number_of_bounces, number_of_continuations

    def update_position(self, positions, displacement_distance, active_particle_index, movement_direction):
        """ Updates position of the active particle."""
        positions[active_particle_index] += displacement_distance * movement_direction
