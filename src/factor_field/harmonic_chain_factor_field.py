"""Module for the HarmonicChainFactorField class."""
from .factor_field import FactorField
import numpy as np
from model_settings import number_of_particles, size_of_particle_space
from base.vectors import get_shortest_vectors_on_torus


class HarmonicChainFactorField(FactorField):
    """
    Class for factor field for harmonic chain model.

    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the HarmonicChainFactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.


        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor)
        self._prefactor = prefactor

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
        neg_neighbour_index, pos_neighbour_index = self._get_neighbours(active_particle_index)
        vetoing_index = pos_neighbour_index if movement_direction > 0.0 else neg_neighbour_index
        distance_to_next_factor_event = - np.log(np.random.uniform(0.0, 1.0)) / self._prefactor * temperature
        return distance_to_next_factor_event, vetoing_index

    @staticmethod
    def _get_neighbours(active_particle_index):
        """
        Return indices of neighbours to active particle.
        """
        neg_neighbour_index, pos_neighbour_index = ((active_particle_index - 1) % number_of_particles,
                                                    (active_particle_index + 1) % number_of_particles)
        return neg_neighbour_index, pos_neighbour_index
