"""Module for the HardDiskFactorField class."""
from .factor_field import FactorField
import numpy as np
from model_settings import number_of_particles
from base.vectors import get_shortest_vectors_on_torus


class HardDiskFactorField(FactorField):
    """
    Class for implementing factor fields (in event-chain Monte Carlo) for the hard disk model.
    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the HardDiskFactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.

        """
        super().__init__(prefactor)

    def get_next_event(self, positions, disk_radii, active_particle_index, temperature, movement_direction):
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
        hop_displacement : numpy.ndarray
            Net displacement through state space from active to vetoing particle.
        """
        
        vetoing_index = None
        for index in range(1, number_of_particles):
            candidate = (active_particle_index - index) % number_of_particles
            if (disk_radii[active_particle_index] == disk_radii[candidate]):
                vetoing_index = candidate
                break
        if vetoing_index is not None:
            distance_to_next_factor_event = - np.log(np.random.uniform(0.0, 1.0, 1)) / self._prefactor * temperature
            hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0]
                                                            - positions[active_particle_index, 0])
            return distance_to_next_factor_event, vetoing_index, hop_displacement
        else:
            return float('inf'), None, None
        '''
        neg_neighbour_index, pos_neighbour_index = ((active_particle_index - 1) % number_of_particles,
                                                (active_particle_index + 1) % number_of_particles)
        vetoing_index = neg_neighbour_index if movement_direction > 0.0 else pos_neighbour_index
        # pressure = number_of_particles * temperature / (size_of_particle_space -
                                                        #2.0 * number_of_particles * self._disk_radius)
        distance_to_next_factor_event = - np.log(np.random.uniform(0.0, 1.0, 1)) / self._prefactor * temperature
        hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0]
                                                            - positions[active_particle_index, 0])
        return distance_to_next_factor_event, vetoing_index, hop_displacement
        '''
        
        

