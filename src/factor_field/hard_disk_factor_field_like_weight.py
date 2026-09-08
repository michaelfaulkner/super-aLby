"""Module for the HardDiskFactorField class."""
from .factor_field import FactorField
import numpy as np
from model_settings import number_of_particles
from base.vectors import get_shortest_vectors_on_torus


class HardDiskFactorFieldLikeWeight(FactorField):
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

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction,
                       disk_radii):
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
        disk_radii : numpy.ndarray
            A one-dimensional numpy array of length number_of_particles.  Element i is a float and represents the
            relative radius of the particle i in hard-disk models.

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
            if disk_radii[active_particle_index] == disk_radii[candidate]:
                vetoing_index = candidate
                break
        if vetoing_index is not None:
            if disk_radii[active_particle_index] == 1.0:
                distance_to_next_factor_event = (- np.log(np.random.uniform(0.0, 1.0, 1)) /
                                                 (self._prefactor * (3/4) * temperature))
                hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0] -
                                                                 positions[active_particle_index, 0])
                return distance_to_next_factor_event, vetoing_index, hop_displacement
            if disk_radii[active_particle_index] == 2.0:
                distance_to_next_factor_event = (- np.log(np.random.uniform(0.0, 1.0, 1)) /
                                                 (self._prefactor * (1/4) * temperature))
                hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0] -
                                                                 positions[active_particle_index, 0])
                return distance_to_next_factor_event, vetoing_index, hop_displacement
        else:
            return float('inf'), None, None
