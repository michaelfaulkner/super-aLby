"""Module for the HardDiskFactorField class."""
from .factor_field import FactorField
import numpy as np
from model_settings import size_of_particle_space, number_of_particles
from base.vectors import get_shortest_vectors_on_torus
from base.exceptions import ConfigurationError

class HardDiskFactorField(FactorField):
    """
    Class for implementing factor fields (in event-chain Monte Carlo) for the hard disk model.
    """
    def __init__(self, prefactor: float = 1.0, ff_type: str = 'like_weight', disk_radius_a: float = 1.0,
                 disk_radius_b: float = 2.0, particle_ratio_a: int = 1, particle_ratio_b: int = 1):
        # todo complete docstring to explain each of the input variables
        """
        The constructor of the HardDiskFactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.

        """
        self._ff_type = ff_type
        self._disk_radius_a = disk_radius_a
        self._disk_radius_b = disk_radius_b
        self._particle_ratio_a = particle_ratio_a
        self._particle_ratio_b = particle_ratio_b
        self._prefactor = number_of_particles / (
                size_of_particle_space - (self._particle_ratio_a / (self._particle_ratio_a + self._particle_ratio_b))
                * self._disk_radius_a - (self._particle_ratio_b / (self._particle_ratio_a + self._particle_ratio_b))
                * self._disk_radius_b)

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
        if self._ff_type == 'like_weight':
            return self.like_weight(positions, active_particle_index, temperature, movement_direction,
                            disk_radii)
        elif self._ff_type == 'all':
            return self.all(positions, active_particle_index, temperature, movement_direction)
        elif self._ff_type == 'no':
            return float('inf'), None, None
        else:
            raise ConfigurationError(f"Invalid ff_type '{self._ff_type}' in {self.__class__.__name__}.")

    def all(self, positions, active_particle_index, temperature, movement_direction):
        # todo add docstring - and give a bit of extra detail about what this method does
        neg_neighbour_index, pos_neighbour_index = ((active_particle_index - 1) % number_of_particles,
                                                    (active_particle_index + 1) % number_of_particles)
        vetoing_index = neg_neighbour_index if movement_direction > 0.0 else pos_neighbour_index
        distance_to_next_factor_event = (- np.log(np.random.uniform(0.0, 1.0, 1)) /
                                         self._prefactor * temperature)
        hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0] -
                                                         positions[active_particle_index, 0])
        return distance_to_next_factor_event, vetoing_index, hop_displacement

    def like_weight(self, positions, active_particle_index, temperature, movement_direction,
                                disk_radii):
        # todo add docstring - and give a bit of extra detail about what this method does
        vetoing_index = None
        for index in range(1, number_of_particles):
            candidate = (active_particle_index - index) % number_of_particles
            if disk_radii[active_particle_index] == disk_radii[candidate]:
                vetoing_index = candidate
                break
        if vetoing_index is not None:
            if disk_radii[active_particle_index] == self._disk_radius_a:
                distance_to_next_factor_event = (- np.log(np.random.uniform(0.0, 1.0, 1)) /
                                                 (self._prefactor * (self._particle_ratio_a/(self._particle_ratio_a+self._particle_ratio_b)) * temperature))
                hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0] -
                                                                 positions[active_particle_index, 0])
                return distance_to_next_factor_event, vetoing_index, hop_displacement
            if disk_radii[active_particle_index] == self._disk_radius_b:
                distance_to_next_factor_event = (- np.log(np.random.uniform(0.0, 1.0, 1)) /
                                                 (self._prefactor * (self._particle_ratio_b/(self._particle_ratio_a+self._particle_ratio_b)) * temperature))
                hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_index, 0] -
                                                                 positions[active_particle_index, 0])
                return distance_to_next_factor_event, vetoing_index, hop_displacement
        else:
            return float('inf'), None, None
