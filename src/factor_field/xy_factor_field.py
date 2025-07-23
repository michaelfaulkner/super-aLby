"""Module for the XyFactorField class."""
from .factor_field import FactorField
import numpy as np
from helper_methods import get_neighbours
from model_settings import number_of_particles


class XyFactorField(FactorField):
    """
    Class for factor field for XY model.

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
        self._prefactor = prefactor / (2.0 * self._lattice_length)
        self._potential_constant = 1.0
        self._boundary_spin_value = 0.0

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
        vetoing_index = None

        neighbour_indices = get_neighbours(active_particle_index, self._lattice_length)
        neighbouring_spin_index = neighbour_indices[3]  # South neighbour
        uphill_energy = - temperature / self._potential_constant * np.log(1.0 - np.random.rand())
        distance_to_next_factor_event = uphill_energy / self._prefactor

        shortest_distance_to_next_factor_event = distance_to_next_factor_event
        vetoing_index = neighbouring_spin_index

        return shortest_distance_to_next_factor_event, vetoing_index

    def choose_next_active_particle(self, positions, active_particle_index, temperature, movement_direction,
                                    veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

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
        veto_index : int
            The particle index responsible for the event.

        Returns
        -------
        active_particle_index: int
            The index of the next active particle.
        movement_direction : int
            The next active-particle direction of motion.
        """
        if veto_index >= (number_of_particles - self._lattice_length):
            hard_distance, hard_veto_index = self._get_hard_distance(positions)
            soft_distance, soft_veto_index = self.get_next_event(positions, active_particle_index, temperature,
                                                                 movement_direction)
            print(f'Soft: {soft_distance}, Hard: {hard_distance}')
            if soft_distance < hard_distance:
                self.soft_wins += 1
                print('Soft win')
                self._update_global_position(positions, soft_distance, -movement_direction)
                veto_index = soft_veto_index
            else:
                self._update_global_position(positions, hard_distance, -movement_direction)
                veto_index = hard_veto_index
                print('Hard win')
            self.total += 1
        return veto_index, movement_direction

    def _get_hard_distance(self, positions):
        """Returns the distance to the next hard collision event and the index of the particle that triggers
        the event"""
        hard_distances = [self._get_spin_difference(bottom_row_spin, self._boundary_spin_value)[0] for
                          bottom_row_spin in positions[:self._lattice_length]]
        hard_distance, veto_index = min(hard_distances), hard_distances.index(min(hard_distances))
        return hard_distance, veto_index

    @staticmethod
    def _get_spin_difference(spin_value_one, spin_value_two):
        """Returns the difference between two spin angles"""
        return (spin_value_one - spin_value_two) % (2.0 * np.pi)

    @staticmethod
    def _update_global_position(positions, displacement_distance, movement_direction):
        """Updates the position of all particles"""
        positions[:] = (positions + movement_direction * displacement_distance) % (2.0 * np.pi)