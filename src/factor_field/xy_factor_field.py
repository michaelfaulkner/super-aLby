"""Module for the XyFactorField class."""
from .factor_field import FactorField
import numpy as np
from helper_methods import get_neighbours
from model_settings import number_of_particles


class XyFactorField(FactorField):
    """
    Class for implementing factor fields (in event-chain Monte Carlo) for the 2DXY model.  This is currently a work in
        progress as we have not finalised how to implement factor fields for the 2DXY model.
    """

    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 2):
        """
        The constructor of the XyFactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        lattice_dimensionality : int
            The dimensionality of the lattice on which the XY model is defined.  We currently only provide functionality
            for the 2DXY model, i.e. for lattice_dimensionality equal to two.


        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor)
        lattice_length = number_of_particles ** (1 / lattice_dimensionality)
        self._lattice_length = int(lattice_length + 1.0e-12)
        self._prefactor = prefactor / self._lattice_length

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
        active_spin_value = positions[active_particle_index, 0]
        vetoing_index = None

        neighbours = get_neighbours(active_particle_index, self._lattice_length)
        neighbour_pairs = [[i, j] for i, j in zip(neighbours[:len(neighbours)//2], neighbours[len(neighbours)//2:])]
        uphill_index = 0  # Select 2 of the 4 neighbour spins to have uphill factor field slopes.
        for neighbour_index_pair in neighbour_pairs:
            for neighbour_count, neighbouring_spin_index in enumerate(neighbour_index_pair):
                non_active_spin_value = positions[neighbouring_spin_index, 0]
                initial_spin_value_difference = self._get_spin_difference(active_spin_value, non_active_spin_value)
                if neighbour_count == uphill_index:
                    uphill_energy = - temperature / self._prefactor * np.log(1.0 - np.random.rand())
                    distance_to_next_factor_event = uphill_energy / self._prefactor
                else:
                    distance_to_next_factor_event = 1.0e10

                if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
                    shortest_distance_to_next_factor_event = distance_to_next_factor_event
                    vetoing_index = neighbouring_spin_index

        return shortest_distance_to_next_factor_event, vetoing_index

    @staticmethod
    def _get_spin_difference(spin_value_one, spin_value_two):
        """ returns the difference between two spin angles"""
        return (spin_value_one - spin_value_two + np.pi) % (2.0 * np.pi) - np.pi