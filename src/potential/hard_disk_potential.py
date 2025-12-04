
"""Module for the HardDiskPotential class"""
import itertools
import math
import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base.exceptions import ConfigurationError, MediatorError
from base.vectors import get_shortest_vectors_on_torus
from linked_lists.two_dimensional_linked_lists import TwoDimensionalLinkedLists
from model_settings import size_of_particle_space, number_of_particles, dimensionality_of_particle_space


class HardDiskPotential(EuclideanSubspacePotential):
    r"""
    This class implements the potential functionality for event-chain simulations of the hard-disk model.  Functionality
        is currently provided only for hard disks in a 2D box with a (1:1) aspect ratio.

    N.B. the abstract get_gradient() method (defined in EuclideanSubspacePotential) is not relevant due to the
        non-smooth nature of the 'potential' function.

    For 72 hard disks in a 2D box with a (1:1) aspect ratio, the simulations defined in config_files/2d_hard_disk_tests
        tested the event-chain code against data provided at the following URL:

        https://github.com/jellyfysh/HistoricDisks/blob/master/DigitizedData/ThisWork.csv

        1) config_files/2d_hard_disk_tests/packing_fraction_point_688 predicted
            \beta P (2 \sigma)^2 = 8.377193860693 +- 0.012134909492, compared with 8.39654 +- 0.00040 at the URL.

        2) config_files/2d_hard_disk_tests/packing_fraction_point_698 predicted
            \beta P (2 \sigma)^2 = 8.521643548125 +- 0.013374142768, compared with 8.5118 +- 0.0010 at the URL.

        3) config_files/2d_hard_disk_tests/packing_fraction_point_698 predicted
            \beta P (2 \sigma)^2 = 8.548840320398 +- 0.012093065068, compared with 8.55170 +- 0.00059 at the URL.

        The final two simulations agreed (with the published data) within the simulation error.  The first resulted in
            a minor discrepancy.  Given that these tests were run before any code optimisation, we were therefore happy
            to conclude that the code is working correctly.  Note that our simulations produced 10^6 samples after
            discarding 10^5 equilibration samples.
    """

    def __init__(self, prefactor: float = 1.0, disk_radius_a: float = 1.0, disk_radius_b: float = 1.0, packing_fraction: float = 0.5):
        r"""
        The constructor of the HardDiskPotential class

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.
        disk_radius : float, optional
            The radius of each disk.
        packing_fraction : float, optional
            The packing fraction of the disks.  This corresponds to the mean disk density.

        Raises
        ------
        base.exceptions.ConfigurationError
            If each Cartesian component of size_of_particle_space is not equal.
        base.exceptions.ConfigurationError
            If disk_radius is greater than half the length of the particle space along any Cartesian dimension.
        base.exceptions.ConfigurationError
            If packing_fraction is less than 0.1 or greater than 0.8 (the theoretical minimum and maximum are zero and
            approximately 0.9, respectively).
        """
        super().__init__(prefactor=prefactor)
        for linear_length in size_of_particle_space:
            if (disk_radius_a or disk_radius_b) > 0.5 * linear_length:
                raise ConfigurationError(
                    f"Give a value of less than half the length of the particle space (along each Cartesian dimension) "
                    f"for disk_radius in {self.__class__.__name__}.  This ensures at least two cells along each "
                    f"Cartesian direction, which avoids the possibility of self collision in event-chain Monte Carlo.")
        if dimensionality_of_particle_space != 1 and math.isclose(disk_radius_a, disk_radius_b):
            raise ConfigurationError("Binary-mixture functionality is only available for 1D hard-sphere models.  For "
                                     "dimensionality_of_particle_space > 1, set disk_radius_a equal to disk_radius_b.")
        if not (0.1 <= packing_fraction <= 0.8):
            raise ConfigurationError(f"Give a value not less than 0.1 and not greater than 0.8 for packing_fraction in "
                                     f"{self.__class__.__name__}.")
        self._disk_radius_a = disk_radius_a
        self._disk_radius_b = disk_radius_b
        self._disk_radius = disk_radius_a if dimensionality_of_particle_space > 1 else None
        self._packing_fraction = packing_fraction
        self._disk_radii = np.array([self._disk_radius_a if (i % 2) == 0 else self._disk_radius_b for i in range(number_of_particles)], dtype=float)
        number_of_cells_in_each_direction = np.int_(size_of_particle_space / (2.0 * max(self._disk_radius_a, self._disk_radius_b)))
        if dimensionality_of_particle_space > 1:
            if not math.isclose(size_of_particle_space[0], size_of_particle_space[1]):
                raise ConfigurationError(
                    f"Set each Cartesian component of size_of_particle_space to a common float when using "
                    f"{self.__class__.__name__}, as this class currently provides only for square compact subspaces.")
            self._linked_lists = TwoDimensionalLinkedLists(number_of_cells_in_each_direction)
        self._active_cell_index = 0
        print(f"System length along each Cartesian dimension is {size_of_particle_space}.")
        print(f"Number of cells along each Cartesian dimension is {number_of_cells_in_each_direction}.")

    def _radius_for_index(self, idx: int) -> float:
        """Return disk radius for a given particle index."""
        return float(self._disk_radii[idx])

    def get_value(self, positions):
        """
        For hard-sphere models, this method throws a ValueError if there are disk overlaps and returns 0.0 otherwise.

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
        return float('inf') if self._check_for_disk_overlaps(positions)[0] else 0.0

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        This method supports 1D systems only.  For some candidate configuration and smooth potential function,
            the functionality provides MetropolisMediator with the increase in the value of the potential function
            (relative to the current configuration).

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
        active_radius = self._radius_for_index(active_particle_index)
        for neighbour_index in range(number_of_particles):
            neighbour_radius = self._radius_for_index(neighbour_index)
            minimum_allowed_separation = active_radius + neighbour_radius
            if (neighbour_index != active_particle_index and np.linalg.norm(get_shortest_vectors_on_torus(
                    positions[neighbour_index] - candidate_position)) < minimum_allowed_separation):
                return float('inf')
        return 0.0

    def get_gradient(self, positions):
        """
        Throws an error if used in this case.  The method is only valid for smooth potential functions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        raise MediatorError(f"get_gradient() is not a valid method for {self.__class__.__name__} as this is not a "
                            f"smooth potential function.")

    def get_initial_positions(self):
        """
        Returns the initial positions array.  This is a close-packed configuration as described in
            self._get_candidate_initial_positions().

        N.B. as it is challenging to generate close-packed configurations of hard disks, we provide two different
            attempts at creating a close-packed configuration with no overlaps, via
            self._get_candidate_initial_positions().  We recommend choosing number_of_particles equal to either a
            square number or the product of two adjacent integers.  This avoids non-complete rows or columns of disks
            (in the closed-packed configuration).

        N.B. for a (2:3^0.5) aspect ratio, the high-packing limit is 0.906899682117 (12 significant figures).

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g. three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
        if dimensionality_of_particle_space == 1:
            positions = self._get_candidate_initial_positions(number_of_particles)
            return positions
        try:
            index_range = [int(number_of_particles ** 0.5), int(number_of_particles ** 0.5 + 2)]
            positions = self._get_candidate_initial_positions(index_range)
            overlap_exists, particle_index_1, particle_index_2, minimal_separation_distance = (
                self._check_for_disk_overlaps(positions))
            if overlap_exists:
                raise ValueError(f"Disks {particle_index_1} and {particle_index_2} are overlapping. "
                                 f"Their minimal separation distance is {minimal_separation_distance}.")
            print("Using the primary method for generating initial hard-disk configurations (see "
                  "HardDiskPotential.get_initial_positions()).")
        except ValueError:
            try:
                index_range = [int(number_of_particles ** 0.5 + 1), int(number_of_particles ** 0.5 + 1)]
                positions = self._get_candidate_initial_positions(index_range)
                overlap_exists, particle_index_1, particle_index_2, minimal_separation_distance = (
                    self._check_for_disk_overlaps(positions))
                if overlap_exists:
                    raise ValueError(f"Disks {particle_index_1} and {particle_index_2} are overlapping. "
                                     f"Their minimal separation distance is {minimal_separation_distance}.")
                print("Using the alternative method for generating initial hard-disk configurations (due to overlaps "
                      "induced by the primary method - see HardDiskPotential.get_initial_positions()).")
            except ValueError:
                raise ConfigurationError(
                    f"Both methods for generating initial hard-disk configurations have failed (see "
                    f"HardDiskPotential.get_initial_positions()).  As it is challenging to generate close-packed "
                    f"configurations of hard disks, we recommend choosing number_of_particles equal to either a square "
                    f"number or the product of two adjacent integers.  This avoids non-complete rows or columns of "
                    f"disks (in the closed-packed configuration).  Alternatively, consider increasing "
                    f"number_of_particles or decreasing packing_fraction, e.g. for a (1:1) aspect ratio with the "
                    f"primary initial-configuration method, we believe that number_of_particles should be greater than "
                    f"32 to guarantee a valid initial configuration for packing_fraction = 0.688 (though a thorough "
                    f"analysis is required).")
        self._linked_lists.reset_linked_lists(positions)
        return positions

    def _get_candidate_initial_positions(self, index_range):
        """
        Returns a candidate for the initial positions array.  The function creates a close-packed configuration on a
            [hexagonal lattice](https://en.wikipedia.org/wiki/Hexagonal_lattice) with index_range[0] the maximum number
            of disks along any row and index_range[1] the total number of rows.  This reflects the fully packed
            configuration presented in figure 4 of Statist. Sci. 39, 137 (2024).

        N.B. As it is challenging to generate close-packed configurations of hard disks, for 2D configurations,
            we recommend choosing number_of_particles equal to either a square number or the product of two adjacent
            integers. This avoids non-complete rows of disks (in the closed-packed configuration).

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g. three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
        positions = np.zeros((number_of_particles, dimensionality_of_particle_space))
        if dimensionality_of_particle_space == 1:
            positions[0, 0] = 0.0
            for index in range(1, number_of_particles):
                previous_radius = self._radius_for_index(index - 1)
                current_radius = self._radius_for_index(index)
                step = 1.00001 * (previous_radius + current_radius) / self._packing_fraction
                positions[index, 0] = positions[index - 1, 0] + step
            positions = get_shortest_vectors_on_torus(positions)
            return positions
        delta_x = 1.00001 * 2.0 * self._disk_radius
        delta_y = [1.00001 * self._disk_radius, 1.00001 * self._disk_radius * 3.0 ** 0.5]
        for index_x in range(index_range[0]):
            for index_y in range(index_range[1]):
                if index_x + index_y * index_range[0] + 1 > number_of_particles:
                    """pass to correct for using index_range[1] = int(number_of_particles ** 0.5 + 2) - which we use as 
                        int(number_of_particles ** 0.5) is too small for a non-square number_of_particles"""
                    continue
                positions[index_x + index_y * index_range[0], 0] = (index_x * delta_x + index_y * delta_y[0]
                                                                    ) % size_of_particle_space[0]
                positions[index_x + index_y * index_range[0], 1] = (index_y * delta_y[1]) % size_of_particle_space[1]
        positions = get_shortest_vectors_on_torus(positions)
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
        if dimensionality_of_particle_space == 1:
            return 1
        if np.random.uniform() < 0.5:
            return np.array([1, 0])
        return np.array([0, 1])

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
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
        hop_displacement : numpy.ndarray
            Net displacement through state space from active to vetoing particle.
        """
        if dimensionality_of_particle_space == 1:
            vetoing_particle_index = (active_particle_index + 1) % number_of_particles if movement_direction > 0 else (
                    (active_particle_index - 1) % number_of_particles)
            active_radius = self._radius_for_index(active_particle_index)
            veto_radius = self._radius_for_index(vetoing_particle_index)
            separation = get_shortest_vectors_on_torus(
                positions[vetoing_particle_index, 0] - positions[active_particle_index, 0])
            distance_to_next_event = separation - (active_radius + veto_radius)
            hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_particle_index]
                                                             - positions[active_particle_index])
            return distance_to_next_event, vetoing_particle_index, hop_displacement
        # todo fix bug in 2D hard-disk code that appeared after correcting EventChainMediator structure
        self.pointer_hop_distance = 0.0
        active_particle_position = positions[active_particle_index]
        if self.cell_boundary_event:
            self._linked_lists.move_particle_to_new_cell(active_particle_position, active_particle_index,
                                                         self._active_cell_index)
        self.cell_boundary_event = True
        active_cell = self._linked_lists.get_cell(active_particle_position)
        self._active_cell_index = self._linked_lists.get_cell_index(active_cell)
        """NB, following would have to be adapted for a negative direction of motion"""
        distance_to_edge_of_active_cell = ((1.0 + np.dot(movement_direction, active_cell)) *
                                           np.dot(movement_direction, self._linked_lists.cell_size) -
                                           np.dot(movement_direction, active_particle_position +
                                                  0.5 * size_of_particle_space))
        shortest_distance_to_next_event, pointer_hop_distance = distance_to_edge_of_active_cell, 0.0
        vetoing_particle_index = active_particle_index
        motion_index, other_index = self._get_motion_index_and_other_index(movement_direction)
        for candidate_cell in itertools.product(range(active_cell[0] - motion_index, active_cell[0] + 2),
                                                range(active_cell[1] - other_index, active_cell[1] + 2)):
            candidate_cell = candidate_cell % self._linked_lists.number_of_cells_in_each_direction
            candidate_cell_index = self._linked_lists.get_cell_index(candidate_cell)
            candidate_particle_index = self._linked_lists.leading_particle_of_cell[candidate_cell_index]
            while candidate_particle_index is not None:
                if candidate_particle_index != active_particle_index:
                    candidate_particle_position = positions[candidate_particle_index]
                    displacement_to_candidate_particle = get_shortest_vectors_on_torus(candidate_particle_position -
                                                                                       active_particle_position)
                    distance_to_possible_collision, candidate_pointer_hop_distance = 1.0e10, 0.0
                    if np.abs(displacement_to_candidate_particle[other_index]) < 2.0 * self._disk_radius:
                        # collision possible
                        if displacement_to_candidate_particle[motion_index] < 0.0:
                            displacement_to_candidate_particle[motion_index] += size_of_particle_space[motion_index]
                        candidate_pointer_hop_distance = (4.0 * self._disk_radius ** 2 -
                                                          displacement_to_candidate_particle[other_index] ** 2) ** 0.5
                        distance_to_possible_collision = (displacement_to_candidate_particle[motion_index] -
                                                          candidate_pointer_hop_distance)
                    if distance_to_possible_collision < shortest_distance_to_next_event:
                        self.cell_boundary_event = False
                        shortest_distance_to_next_event = distance_to_possible_collision
                        vetoing_particle_index = candidate_particle_index
                        self.pointer_hop_distance = candidate_pointer_hop_distance
                candidate_particle_index = self._linked_lists.next_particle_in_same_cell[candidate_particle_index]
        hop_displacement = get_shortest_vectors_on_torus(positions[vetoing_particle_index]
                                                         - positions[active_particle_index])
        return shortest_distance_to_next_event, vetoing_particle_index, hop_displacement

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
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
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """ Updates the position of the active particle."""
        positions[active_particle_index] += displacement_distance * movement_direction
        positions[active_particle_index] = get_shortest_vectors_on_torus(positions[active_particle_index])

    def _check_for_disk_overlaps(self, positions):
        for particle_index_1 in range(number_of_particles):
            radius_particle_index_1 = self._radius_for_index(particle_index_1)
            for particle_index_2 in range(particle_index_1 + 1, number_of_particles):
                radius_particle_index_2 = self._radius_for_index(particle_index_2)
                minimal_separation_distance = np.linalg.norm(get_shortest_vectors_on_torus(positions[particle_index_1] -
                                                                                           positions[particle_index_2]))
                if (minimal_separation_distance < (radius_particle_index_1 + radius_particle_index_2) and not
                        abs(minimal_separation_distance - (radius_particle_index_1 + radius_particle_index_2)) < 1.0e-12):
                    return True, particle_index_1, particle_index_2, minimal_separation_distance
        return False, None, None, None

    @staticmethod
    def _get_motion_index_and_other_index(movement_direction):
        motion_index = 0  # assume that the active particle is advancing in x direction
        if movement_direction[0] == 0:
            motion_index = 1  # the active particle is actually advancing in y direction
        other_index = 1 - motion_index
        return motion_index, other_index

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        #print(self._disk_radii)
        if dimensionality_of_particle_space != 1:
            raise MediatorError("portals only implemented for 1D hard-sphere systems.")
        if veto_index is None or veto_index == active_particle_index:
            return None             
        if np.random.uniform() >= 0.5:
            return None
        active_radius = self._radius_for_index(active_particle_index)
        veto_radius = self._radius_for_index(veto_index)
        seperation = float(np.linalg.norm(get_shortest_vectors_on_torus(positions[active_particle_index] - positions[veto_index])))
        if (seperation > (active_radius + veto_radius) + 1e-12):
            return None
        next_index = (veto_index + 1) % number_of_particles
        next_radius = self._radius_for_index(next_index)
        veto_position = float(positions[veto_index, 0])
        next_position = float(positions[next_index, 0])
        #print(active_particle_index, veto_index, next_index)
        gap = (next_position - veto_position) % size_of_particle_space      
        required_gap = veto_radius + 2.0 * active_radius + next_radius
        if gap + 1e-12 < required_gap:
            return None
        candidate_position = veto_position + (veto_radius + active_radius) + 1e-8
        wrapped_position = ((candidate_position + size_of_particle_space/2) % size_of_particle_space) - size_of_particle_space/2
        return float(wrapped_position)
