"""Module for the HardDiskPotential class"""
import itertools
import math
import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base.exceptions import ConfigurationError
from base.vectors import get_shortest_vectors_on_torus
from linked_lists.two_dimensional_linked_lists import TwoDimensionalLinkedLists
from model_settings import size_of_particle_space, number_of_particles


class HardDiskPotential(EuclideanSubspacePotential):
    r"""
    This class implements the potential functionality for event-chain simulation of the hard-disk model.  Some abstract
        methods from SoftMatterPotential are not relevant due to the non-smooth nature of the 'potential' function.
        We include these methods as dummy methods.
    """

    def __init__(self, prefactor: float = 1.0, particle_radius: float = 1.0):
        r"""
        The constructor of the HardDiskPotential class

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If each Cartesian component of size_of_particle_space is not equal.
        base.exceptions.ConfigurationError
            If particle_radius is greater than half the length of the particle space along any Cartesian dimension.
        """
        if not math.isclose(size_of_particle_space[0], size_of_particle_space[1]):
            raise ConfigurationError(
                f"Set each Cartesian component of size_of_particle_space to a common float when using "
                f"{self.__class__.__name__}, as this class currently provides only for square compact subspaces.")
        for linear_length in size_of_particle_space:
            if particle_radius > 0.5 * linear_length:
                raise ConfigurationError(
                    f"Give a value of less than half the length of the particle space (along each Cartesian dimension) "
                    f"for particle_radius in {self.__class__.__name__}.  This ensures at least two cells along each "
                    f"Cartesian direction, which avoids the possibility of self collision in event-chain Monte Carlo.")
        super().__init__(prefactor=prefactor)
        self._particle_radius = particle_radius
        number_of_cells_in_each_direction = np.int_(size_of_particle_space / (2.0 * self._particle_radius))
        self._linked_lists = TwoDimensionalLinkedLists(number_of_cells_in_each_direction)

    def get_value(self, positions):
        """
        This is a dummy method as it is not relevant to hard-sphere models.  For a smooth potential function, the
            functionality provides Mediator with the current value of the potential.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential function.
        """
        pass

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        This is a dummy method as it is not relevant to hard-sphere models.  For some candidate configuration and
            smooth potential function, the functionality provides MetropolisMediator with the increase in the value of
            the potential function (relative to the current configuration).

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the proposed position of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        pass

    def get_initial_positions(self):
        # todo test this function in terminal etc
        """
        Returns the initial positions array.  Creates a close-packed configuration.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g., three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
        dx = [1.00001 * 2.0 * self._particle_radius, 0.0]
        dy = [1.00001 * self._particle_radius, 1.00001 * self._particle_radius * np.sqrt(3.0)]
        positions = np.zeros((number_of_particles, 2))
        for index_x in range(int(number_of_particles ** 0.5)):
            """***NOTE THAT the commented-out code is adapted from HistoricDisks***"""
            """for index_y in range(int(number_of_particles ** 0.5) + 2):
                if index_y * int(number_of_particles ** 0.5) + index_x + 1 > number_of_particles:
                    continue"""
            for index_y in range(int(number_of_particles ** 0.5)):
                positions[index_x + index_y * int(number_of_particles ** 0.5)][0] = get_shortest_vectors_on_torus(
                    (index_x * dx[0] + index_y * dy[0]) % size_of_particle_space[0])
                positions[index_x + index_y * int(number_of_particles ** 0.5)][1] = get_shortest_vectors_on_torus(
                    (index_y * dy[1]) % size_of_particle_space[1])
        self._linked_lists.reset_linked_lists(positions)
        return positions

    @staticmethod
    def get_random_event_chain_velocity():
        """
        Uniformly samples a direction of motion for the active particle from chosen velocity distribution.

        Returns
        ----------
        random_event_chain_velocity : int or numpy.ndarray
            The uniformly sampled event-chain velocity of the active particle.  If the state space of each particle is
            a subset of the real line, the method should output an integer; otherwise it should output a one-dimensional
            numpy array (of integers) of length dimensionality_of_particle_space, where the nth component represents the
            velocity of the active particle along the nth Cartesian direction.
        """
        if np.random.uniform() < 0.5:
            return np.array([1, 0])
        return np.array([0, 1])

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature,
                                                  movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of the corresponding particle.
        active_particle_index : int
            The active particle index.
        temperature : float
            The sampling temperature.  NB, we will map this to the packing fraction for hard spheres.
        movement_direction : numpy.ndarray
            A one-dimensional numpy array of size 2; the element first/second element is 0 or 1 and represents the
            direction of motion along the x/y direction.

        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        vetoing_particle_index : int
            The index of the particle that triggers the event.
        """
        # todo adapt EventChainMediator._generate_sample_at_current_temperature() to remove following line (since we
        #  perform its operation in initialised_position_array())
        self._linked_lists.reset_linked_lists(positions)
        active_particle_position = positions[active_particle_index]
        active_cell = np.int_(active_particle_position // self._linked_lists.cell_size)
        # todo can probably simplify these if statements
        if movement_direction[1] == 0:
            # active particle is advancing in x direction
            distance_to_edge_of_active_cell = self._linked_lists.cell_size[0] - (
                    active_particle_position[0] - active_cell[0] * self._linked_lists.cell_size[0])
        else:
            # active particle is advancing in y direction
            distance_to_edge_of_active_cell = self._linked_lists.cell_size[1] - (
                    active_particle_position[1] - active_cell[1] * self._linked_lists.cell_size[1])
        shortest_distance_to_next_event = distance_to_edge_of_active_cell
        vetoing_particle_index = active_particle_index
        # todo can probably reduce range of this iteration by accounting for direction of motion
        for candidate_cell in itertools.product(range(active_cell[0] - 1, active_cell[0] + 1),
                                                range(active_cell[1] - 1, active_cell[1] + 1)):
            # todo extract (as method) following repeated code
            candidate_cell_index = self._linked_lists.get_cell_index([
                int((element + self._linked_lists.number_of_cells_in_each_direction[index] / 2) %
                    self._linked_lists.number_of_cells_in_each_direction[index] -
                    self._linked_lists.number_of_cells_in_each_direction[index] / 2)
                for index, element in enumerate(candidate_cell)])
            candidate_particle_index = self._linked_lists.leading_particle_of_cell[candidate_cell_index]
            while candidate_particle_index is not None:
                candidate_particle_position = positions[candidate_particle_index]
                displacement_to_candidate_particle = candidate_particle_position - active_particle_position
                distance_to_possible_collision = 1.0e10
                # todo can probably simplify these if statements
                if movement_direction[1] == 0:
                    # active particle is advancing in x direction
                    if (displacement_to_candidate_particle[0] > 0.0 and
                            np.abs(displacement_to_candidate_particle[1]) < 2.0 * self._particle_radius):
                        # collision possible
                        distance_to_possible_collision = displacement_to_candidate_particle[0] - (
                                4.0 * self._particle_radius ** 2 - displacement_to_candidate_particle[1] ** 2) ** 0.5
                else:
                    # active particle is advancing in y direction
                    if (displacement_to_candidate_particle[1] > 0.0 and
                            np.abs(displacement_to_candidate_particle[0]) < 2.0 * self._particle_radius):
                        # collision possible
                        distance_to_possible_collision = displacement_to_candidate_particle[1] - (
                                4.0 * self._particle_radius ** 2 - displacement_to_candidate_particle[0] ** 2) ** 0.5
                if distance_to_possible_collision < shortest_distance_to_next_event:
                    shortest_distance_to_next_event = distance_to_possible_collision
                    vetoing_particle_index = candidate_particle_index
                candidate_particle_index = self._linked_lists.next_particle_in_same_cell[candidate_particle_index]
        return shortest_distance_to_next_event, vetoing_particle_index

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """Chooses the index and direction for the next active particle in the markov chain"""
        return veto_index, movement_direction

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        # todo check this - copied from copied from QuantumHarmonic Oscillator but might not translate!!!
        """ Updates position of the active particle."""
        positions[active_particle_index] += displacement_distance * movement_direction
