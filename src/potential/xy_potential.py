import numpy as np
from .continuous_potential import ContinuousPotential
from model_settings import number_of_particles
from base.logging import log_init_arguments
from base. exceptions import ConfigurationError
import logging
from helper_methods import get_east_neighbour, get_north_neighbour, get_west_neighbour, get_south_neighbour


class XyPotential(ContinuousPotential):

    """
    This class implements the 2D XY model potential
    """

    def __init__(self, prefactor: float = 1.0,  lattice_dimensionality: int = 2): 
        """
        The constructor of the XyPotential class.
        6 
        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        """
        super().__init__(prefactor=prefactor)
        if lattice_dimensionality != 2:
            raise ConfigurationError(f"Give a value of 2 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        lattice_length = number_of_particles ** (1 / lattice_dimensionality)
        if not lattice_length.is_integer():
            raise ConfigurationError(
                f"For the value of number_of_particles in ModelSettings, give lattice_length ** lattice_dimensionality "
                f"when using {self.__class__.__name__}, where lattice_length is an integer not less than 2.")
        
        self._lattice_dimensionality = lattice_dimensionality
        self._lattice_length = int(lattice_length)
        self.potential_constant = prefactor
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor,
                           lattice_dimensionality=lattice_dimensionality)

    def get_value(self, positions):

        """
        Returns the potential function for the given particle positions. Here, 'positions' is a misnomer, inherited from
        the naming conventions of the parent ContinuousPotential class. A 'position' given here is actually a scalar
        value corresponding to the phase/angle of the spin of the corresponding particle.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        
        Returns
        -------
        float
            The potential.
        """

        return self.potential_constant * 0.5 * np.sum([self._sum_nearest_neighbours(
            index, positions[index].item(), positions) for index in range(number_of_particles)])

    def get_gradient(self, positions):
        # TODO implement get_gradient() function in this class
        """
        Returns the gradient of the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        nothing.
        """
        raise SystemError(f"The get_gradient method of {self.__class__.__name__} has not been written.")

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the potential difference resulting from moving the single active particle's spin to candidate_position.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : float
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents the spin angle of the proposed spin of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        current_potential = self._sum_nearest_neighbours(active_particle_index, positions[active_particle_index].item(),
                                                         positions)
        candidate_potential = self._sum_nearest_neighbours(active_particle_index, candidate_position, positions)
        return self.potential_constant * (candidate_potential - current_potential)

    def _sum_nearest_neighbours(self, active_particle_index, active_particle_position, positions):

        """
        Returns the potential at lattice_site_index by performing a sum over nearest neighbours.

        Parameters
        ----------
        active_particle_index : int
            The index of the active_particle.
        active_particle_position : float
            The phase of the spin of the particle at active_particle_index.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        float
            The potential at lattice_site_index.
        """
        return -(np.cos(positions[get_north_neighbour(active_particle_index, self._lattice_length)] -
                        active_particle_position) +
                 np.cos(positions[get_east_neighbour(active_particle_index, self._lattice_length)] -
                        active_particle_position) +
                 np.cos(active_particle_position -
                        positions[get_south_neighbour(active_particle_index, self._lattice_length)]) +
                 np.cos(active_particle_position -
                        positions[get_west_neighbour(active_particle_index, self._lattice_length)]))

    @staticmethod
    def get_random_event_chain_velocity():
        """Uniformly samples a direction of motion for the active particle from chosen velocity distribution"""
        return 1.0

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature,
                                                  movement_direction):
        """
        Returns the distance to the next particle event for a given active particle index,
        as well as the particle index responsible for that event.  Used for ECMC.

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
            The direction of movement of the active particle.
        
        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        veto_index : int
            The particle index responsible for the event.
        """
        shortest_distance_to_next_factor_event = 1.0e10
        active_spin_value = positions[active_particle_index]
        neighbouring_spin_indices = np.zeros(4, dtype=np.int8)
        neighbouring_spin_indices[0] = get_north_neighbour(active_particle_index, self._lattice_length)
        neighbouring_spin_indices[1] = get_south_neighbour(active_particle_index, self._lattice_length)
        neighbouring_spin_indices[2] = get_east_neighbour(active_particle_index, self._lattice_length)
        neighbouring_spin_indices[3] = get_west_neighbour(active_particle_index, self._lattice_length)
        vetoing_spin_index = None

        for i in range(4):
            non_active_spin_value = positions[neighbouring_spin_indices[i]]
            initial_spin_value_difference = self._get_spin_difference(active_spin_value, non_active_spin_value)
            uphill_energy = - temperature * np.log(1.0 - np.random.rand())

            if initial_spin_value_difference > 0.0:
                initial_two_spin_potential = 1.0 - np.cos(initial_spin_value_difference)
                no_of_complete_spin_rotations = int(0.5 * (initial_two_spin_potential + uphill_energy))
                final_two_spin_potential = ((no_of_complete_spin_rotations + 1.0) * 2.0 - initial_two_spin_potential -
                                            uphill_energy).item()
                final_spin_value_difference = np.arccos(1.0 - final_two_spin_potential)
                distance_to_next_factor_event = ((no_of_complete_spin_rotations + 0.5) * 2.0 * np.pi -
                                                 initial_spin_value_difference - final_spin_value_difference).item()
           
            else:
                no_of_complete_spin_rotations = int(0.5 * uphill_energy)
                final_two_spin_potential = ((no_of_complete_spin_rotations + 1.0) * 2.0 - uphill_energy).item()
                final_spin_value_difference = np.arccos(1.0 - final_two_spin_potential)
                distance_to_next_factor_event = ((no_of_complete_spin_rotations + 0.5) * 2.0 * np.pi -
                                                 initial_spin_value_difference - final_spin_value_difference).item()

            if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
                shortest_distance_to_next_factor_event = distance_to_next_factor_event
                vetoing_spin_index = neighbouring_spin_indices[i]
                
        return shortest_distance_to_next_factor_event, vetoing_spin_index

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction,
                                    veto_index):
        """
        Chooses the index and direction for the next active particle in the markov chain for ECMC.
        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        movement_direction : int
            The direction of movement of the active particle.
        veto_index : int
            The particle index responsible for the event. 
        """
        return veto_index, movement_direction

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """ Updates position of the active particle."""
        positions[active_particle_index] = (positions[active_particle_index] + displacement_distance) % (2.0 * np.pi)

    @staticmethod
    def _get_spin_difference(spin_value_one, spin_value_two):
        """ returns the difference between two spin angles"""
        return (spin_value_one - spin_value_two + np.pi) % (2.0 * np.pi) - np.pi
