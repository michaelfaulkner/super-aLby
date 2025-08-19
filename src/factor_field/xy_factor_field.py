"""Module for the XyFactorField class."""
from .factor_field import FactorField
import numpy as np
from helper_methods import get_north_neighbour, get_east_neighbour, get_south_neighbour, get_west_neighbour
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
        distance_to_next_factor_event = 1.0e10
        active_row = positions[active_particle_index - (active_particle_index % self._lattice_length):
                               active_particle_index - (active_particle_index % self._lattice_length) +
                               self._lattice_length]
        row_winding_number = sum(self._get_spin_difference(active_row[(i+1) % self._lattice_length],
                                                           active_row[i])[0] for i in range(self._lattice_length))
        if abs(row_winding_number) < 1.0e-12:
            return np.inf, None
        prefactor = self._prefactor * abs(row_winding_number)
        uphill_energy = - temperature * np.log(1.0 - np.random.rand())
        distance_to_next_factor_event = uphill_energy / prefactor
        veto_index = get_west_neighbour(active_particle_index, self._lattice_length) if (
                (row_winding_number * movement_direction) < 0) else (get_east_neighbour(active_particle_index,
                                                                                        self._lattice_length))
        return distance_to_next_factor_event, veto_index

    def get_next_event_row_column(self, positions, active_particle_index, temperature, movement_direction):
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
        distance_to_next_factor_event = 1.0e10
        active_row = positions[active_particle_index - (active_particle_index % self._lattice_length):
                               active_particle_index - (active_particle_index % self._lattice_length) +
                               self._lattice_length]
        active_column = positions[active_particle_index % self._lattice_length:
                                  (active_particle_index % self._lattice_length) + self._lattice_length *
                                  self._lattice_length: self._lattice_length]
        row_winding_number = sum(self._get_spin_difference(active_row[(i+1) % self._lattice_length],
                                                           active_row[i])[0] for i in range(self._lattice_length))
        column_winding_number = sum(self._get_spin_difference(active_column[(i + 1) % self._lattice_length],
                                                              active_column[i])[0] for i in range(self._lattice_length))
        if abs(row_winding_number) < 1.0e-12 and abs(column_winding_number) < 1.0e-12:
            return np.inf, None
        if abs(row_winding_number) < 1.0e-12 and not abs(column_winding_number) < 1.0e-12:
            winding_number = column_winding_number
            row_or_column = 1
        elif abs(column_winding_number) < 1.0e-12:
            winding_number = row_winding_number
            row_or_column = 0
        else:
            if np.random.rand() < 0.5:
                winding_number, row_or_column = row_winding_number, 0
            else:
                winding_number, row_or_column = column_winding_number, 1
        prefactor = self._prefactor * abs(winding_number)
        uphill_energy = - temperature * np.log(1.0 - np.random.rand())
        distance_to_next_factor_event = uphill_energy / prefactor
        if row_or_column == 0:
            veto_index = get_west_neighbour(active_particle_index, self._lattice_length) if (
                    (row_winding_number * movement_direction) < 0) else (get_east_neighbour(active_particle_index,
                                                                                            self._lattice_length))
        else:
            veto_index = get_south_neighbour(active_particle_index, self._lattice_length) if (
                    (column_winding_number * movement_direction) < 0) else (get_north_neighbour(active_particle_index,
                                                                                                self._lattice_length))
        return distance_to_next_factor_event, veto_index

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
        return veto_index, movement_direction

    @staticmethod
    def _get_spin_difference(spin_value_one, spin_value_two):
        """Returns the difference between two spin angles"""
        return (spin_value_one - spin_value_two + np.pi + 1.0e-12) % (2.0 * np.pi) - (np.pi + 1.0e-12)

    @staticmethod
    def _update_global_position(positions, displacement_distance, movement_direction):
        """Updates the position of all particles"""
        positions[:] = positions + movement_direction * displacement_distance + 1.0e-12