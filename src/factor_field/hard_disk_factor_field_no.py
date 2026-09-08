"""Module for the HardDiskFactorField class."""
from .factor_field import FactorField
import numpy as np
from model_settings import number_of_particles
from base.vectors import get_shortest_vectors_on_torus


class HardDiskFactorFieldNo(FactorField):
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
        return float('inf'), None, None
