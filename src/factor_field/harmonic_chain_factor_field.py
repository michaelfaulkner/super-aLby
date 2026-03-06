"""Module for the HarmonicChainFactorField class."""
from .factor_field import FactorField
import numpy as np
from model_settings import number_of_particles, size_of_particle_space


class HarmonicChainFactorField(FactorField):
    """
    Class for implementing factor fields (in event-chain Monte Carlo) for the harmonic-chain model.
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
            If prefactor is less than 0.0.
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
        hop_displacement : numpy.ndarray
            Net displacement through state space from active to vetoing particle.
        """
        if self._prefactor < 10e-10:
            return np.inf, None, None
        neg_neighbour_index, pos_neighbour_index = ((active_particle_index - 1) % number_of_particles,
                                                    (active_particle_index + 1) % number_of_particles)
        vetoing_index = pos_neighbour_index if movement_direction > 0.0 else neg_neighbour_index
        distance_to_next_factor_event = - np.log(np.random.uniform(0.0, 1.0, 1)) / self._prefactor * temperature

        active_particle_position = positions[active_particle_index].copy()
        vetoing_particle_position = positions[vetoing_index].copy()
        if active_particle_index == number_of_particles - 1 and vetoing_index == 0:
            vetoing_particle_position += size_of_particle_space
        elif active_particle_index == 0 and vetoing_index == number_of_particles - 1:
            vetoing_particle_position -= size_of_particle_space

        return distance_to_next_factor_event, vetoing_index, vetoing_particle_position - active_particle_position

