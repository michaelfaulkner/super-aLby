"""Module for the TemporalFactorField class."""
from .factor_field import FactorField
import numpy as np

class TemporalFactorField(FactorField):
    """
    This class implements factor fields along the Euclidean time axis for worldline-type models.
    """

    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1):

        """
        The constructor of the TemporalFactorField class.

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
            is a float and represents the position of a single quantum particle.
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
        distance_to_next_factor_event = - np.log(np.random.uniform(0, 1)) * temperature / self._prefactor

        return distance_to_next_factor_event, active_particle_index, None
