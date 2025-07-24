"""Module for the TemporalFactorField class."""
from .factor_field import FactorField
import numpy as np
from helper_methods import get_east_worldline_neighbour, get_west_worldline_neighbour
from model_settings import number_of_quantum_particles, number_of_timeslices

class TemporalFactorField(FactorField):
    """
    This class implements factor fields along the Euclidean time axis for worldline-type models.
    """

    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1):

        """
        The constructor of the FactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        """
         
        super().__init__(prefactor)
         

         
    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        temperature : float
            The sampling temperature.
        movement_direction : int
            The active-particle direction of motion.

        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        veto_index : int
            The particle index responsible for the event.
        """
        shortest_distance_to_next_factor_event = 1.0e10
        initial_position = positions[active_particle_index].item()
        worldline_neighbours = [get_west_worldline_neighbour(active_particle_index, number_of_quantum_particles,
                                                             number_of_timeslices),
                                get_east_worldline_neighbour(active_particle_index, number_of_quantum_particles,
                                                             number_of_timeslices)]
        for index, worldline_neighbour in enumerate(worldline_neighbours):
            if worldline_neighbour != active_particle_index:
                uphill_energy = - np.log(np.random.uniform(0, 1))
                neighbour_position = positions[worldline_neighbour].item()
            
                if index == 0: # west (i-1) neighbour
                    initial_action = self._prefactor * (initial_position - neighbour_position)
                    final_action = uphill_energy + initial_action
                    final_position = neighbour_position + final_action / self._prefactor
                elif index == 1: # east (i+1) neighbour
                    initial_action = self._prefactor * (neighbour_position - initial_position)
                    final_action = uphill_energy + initial_action
                    final_position = neighbour_position - final_action / self._prefactor

                distance_to_next_factor_event = np.abs(final_position - initial_position)

                if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
                    shortest_distance_to_next_factor_event = distance_to_next_factor_event
                    vetoing_index = active_particle_index

        return shortest_distance_to_next_factor_event, vetoing_index
