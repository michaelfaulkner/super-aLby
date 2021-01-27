"""Module for the LennardJonesPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
from typing import Sequence
import itertools
import logging
import numpy as np


class LennardJonesPotential(SoftMatterPotential):
    r"""
    This class implements the two-particle Lennard-Jones potential

        $ U = k * \sum_{i > j} U_{{\rm LJ}, ij} $ ,

    where

        $ U_{{\rm LJ}, ij} = \begin{cases}
                                U_{{\rm LJ}, ij}^{\rm bare}(r_{ij}) - U_{{\rm LJ}, ij}^{\rm bare}(r_{\rm c}) \,
                                    {\rm if} \, r_{ij} \le r_{\rm c} \\
                                0 \, {\rm if} \, r_{ij} > r_{\rm c}
                             \end{cases} $

    is the two-particle Lennard-Jones potential, and

        $ U_{{\rm LJ}, ij}^{\rm bare}(r_{ij}) = 4 \epsilon \left[\left(\frac{\sigma}{r_{ij}}\right)^{12} -
            \left(\frac{\sigma}{r_{ij}}\right)^6\right]$

    is the bare two-particle Lennard-Jones potential. In the above, $\epsilon$ is bare well depth, $\sigma$ is the
    characteristic length scale of the Lennard-Jones potential, and $r_c$ is the cutoff distance at which the bare
    potential is truncated. If using a finite cutoff distance $r_c$, we recommend $r_c \ge 2.5 \sigma$.
    """

    def __init__(self, characteristic_length: float = 1.0, well_depth: float = 1.0, cutoff_length: float = 2.5,
                 use_linked_lists: bool = False, prefactor: float = 1.0) -> None:
        """
        The constructor of the LennardJonesPotential class.

        The default values are optimized so that the result has machine precision.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
        cutoff_length : float, optional
            The cutoff distance at which the bare potential is truncated. If no cutoff is required, give inf in the
            configuration file.
        use_linked_lists : bool, optional
            If True, use the linked-lists cell algorithm.
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        base.exceptions.ConfigurationError
            If element is less than 2.0 * characteristic_length for element in size_of_particle_space.
        base.exceptions.ConfigurationError
            If cutoff_length is less than 2.5 * characteristic_length.
        base.exceptions.ConfigurationError
            If characteristic_length is less than 0.5.
        base.exceptions.ConfigurationError
            If use_linked_lists is True and dimensionality_of_particle_space does not equal 3.
        base.exceptions.ConfigurationError
            If use_linked_lists is True and cutoff_length is inf.
        """
        super().__init__(prefactor)
        for element in size_of_particle_space:
            if element < 2.0 * characteristic_length:
                raise ConfigurationError(f"Ensure that the value of each component of size_of_particle_space is not "
                                         f"less than twice the value of characteristic_length in "
                                         f"{self.__class__.__name__}.")
        if cutoff_length < 2.5 * characteristic_length:
            raise ConfigurationError(f"Ensure that the value of cutoff_length is not less than 2.5 times the value of "
                                     f"characteristic_length in {self.__class__.__name__}.")
        if characteristic_length < 0.5:
            raise ConfigurationError(f"Give a value not less than 0.5 for characteristic_length in "
                                     f"{self.__class__.__name__}.")
        if use_linked_lists is True and dimensionality_of_particle_space != 3:
            raise ConfigurationError(f"For size_of_particle_space, give a one-dimensional list of length 3 (and "
                                     f"composed of floats) when use_linked_lists in {self.__class__.__name__} is True. "
                                     f"This is because the dimensionality of particle space must be 3 when using the "
                                     f"linked-lists algorithm in {self.__class__.__name__}.")
        if use_linked_lists is True and cutoff_length == float('inf'):
            raise ConfigurationError(f"When the value of cutoff_length in {self.__class__.__name__} is inf, give a "
                                     f"value of False for use_linked_lists in {self.__class__.__name__}.")
        self._potential_12_constant = 4.0 * prefactor * well_depth * characteristic_length ** 12
        self._potential_6_constant = 4.0 * prefactor * well_depth * characteristic_length ** 6
        self._gradient_12_constant = 12.0 * self._potential_12_constant
        self._gradient_6_constant = 6.0 * self._potential_6_constant
        self._characteristic_length = characteristic_length
        if cutoff_length == float('inf'):
            self._cutoff_length = None
            self._bare_potential_at_cut_off = 0.0
        else:
            self._cutoff_length = cutoff_length
            self._bare_potential_at_cut_off = (self._potential_12_constant * cutoff_length ** (- 12.0) -
                                               self._potential_6_constant * cutoff_length ** (- 6.0))
        self._use_linked_lists = use_linked_lists
        if self._use_linked_lists:
            self._number_of_cells_in_each_direction = np.int_(size_of_particle_space / self._cutoff_length)
            self._total_number_of_cells = int(np.prod(self._number_of_cells_in_each_direction))
            self._cell_size = size_of_particle_space / self._number_of_cells_in_each_direction
            self._leading_particle_of_cell = [None for _ in range(self._total_number_of_cells)]
            self._particle_links = [None for _ in range(number_of_particles)]
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           characteristic_length=characteristic_length, well_depth=well_depth,
                           cutoff_length=cutoff_length, use_linked_lists=use_linked_lists, prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential.
        """
        if self._use_linked_lists:
            potential = 0.0
            self._reset_linked_lists(positions)
            for cell_one in itertools.product(range(self._number_of_cells_in_each_direction[0]),
                                              range(self._number_of_cells_in_each_direction[1]),
                                              range(self._number_of_cells_in_each_direction[2])):
                cell_one_index = self._get_cell_index(cell_one)
                for cell_two in itertools.product(range(cell_one[0] - 1, cell_one[0] + 1),
                                                  range(cell_one[1] - 1, cell_one[1] + 1),
                                                  range(cell_one[2] - 1, cell_one[2] + 1)):
                    cell_two_index = self._get_cell_index([int((element + self._number_of_cells_in_each_direction[index]
                                                                / 2) % self._number_of_cells_in_each_direction[index] -
                                                               self._number_of_cells_in_each_direction[index] / 2) for
                                                           index, element in enumerate(cell_two)])
                    particle_one_index = self._leading_particle_of_cell[cell_one_index]
                    while particle_one_index is not None:
                        particle_two_index = self._leading_particle_of_cell[cell_two_index]
                        while particle_two_index is not None:
                            if particle_one_index > particle_two_index:
                                separation_distance = np.linalg.norm(get_shortest_vectors_on_torus(
                                    positions[particle_one_index] - positions[particle_two_index]))
                                if separation_distance <= self._cutoff_length:
                                    potential += self._get_non_zero_two_particle_potential(separation_distance)
                            particle_two_index = self._particle_links[particle_two_index]
                        particle_one_index = self._particle_links[particle_one_index]
            return potential
        elif self._cutoff_length is not None:
            return sum([self._get_two_particle_potential(positions[i], positions[j]) for i in range(number_of_particles)
                        for j in range(i + 1, number_of_particles)])
        return sum([self._get_non_zero_two_particle_potential(
            np.linalg.norm(get_shortest_vectors_on_torus(positions[i] - positions[j]))) for i in
            range(number_of_particles) for j in range(i + 1, number_of_particles)])

    def _get_two_particle_potential(self, position_one, position_two):
        """
        Returns the Lennard-Jones potential for two particles.

        Parameters
        ----------
        position_one : numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle one.
        position_two
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle two.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential.
        """
        separation_distance = np.linalg.norm(get_shortest_vectors_on_torus(position_one - position_two))
        if separation_distance <= self._cutoff_length:
            return self._get_non_zero_two_particle_potential(separation_distance)
        return 0.0

    def _get_non_zero_two_particle_potential(self, separation_distance):
        """
        Returns the Lennard-Jones potential for two particles whose shortest separation distance is not greater than
        self._cutoff_length.

        Parameters
        ----------
        separation_distance : float
            The shortest separation distance between the two particles.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential (for all cases for which it is non-zero).
        """
        return (self._potential_12_constant * separation_distance ** (- 12.0) -
                self._potential_6_constant * separation_distance ** (- 6.0) - self._bare_potential_at_cut_off)

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        gradient = np.zeros((number_of_particles, dimensionality_of_particle_space))
        if self._use_linked_lists:
            self._reset_linked_lists(positions)
            for cell_one in itertools.product(range(self._number_of_cells_in_each_direction[0]),
                                              range(self._number_of_cells_in_each_direction[1]),
                                              range(self._number_of_cells_in_each_direction[2])):
                cell_one_index = self._get_cell_index(cell_one)
                for cell_two in itertools.product(range(cell_one[0] - 1, cell_one[0] + 1),
                                                  range(cell_one[1] - 1, cell_one[1] + 1),
                                                  range(cell_one[2] - 1, cell_one[2] + 1)):
                    cell_two_index = self._get_cell_index([int((element + self._number_of_cells_in_each_direction[index]
                                                                / 2) % self._number_of_cells_in_each_direction[index] -
                                                               self._number_of_cells_in_each_direction[index] / 2) for
                                                           index, element in enumerate(cell_two)])
                    particle_one_index = self._leading_particle_of_cell[cell_one_index]
                    while particle_one_index is not None:
                        particle_two_index = self._leading_particle_of_cell[cell_two_index]
                        while particle_two_index is not None:
                            if particle_one_index > particle_two_index:
                                separation_vector = get_shortest_vectors_on_torus(positions[particle_one_index] -
                                                                                  positions[particle_two_index])
                                separation_distance = np.linalg.norm(separation_vector)
                                if separation_distance <= self._cutoff_length:
                                    two_particle_gradient = self._get_non_zero_two_particle_gradient(
                                        separation_vector, separation_distance)
                                    gradient[particle_one_index] += two_particle_gradient
                                    gradient[particle_two_index] -= two_particle_gradient
                            particle_two_index = self._particle_links[particle_two_index]
                        particle_one_index = self._particle_links[particle_one_index]
            return gradient
        elif self._cutoff_length is not None:
            for i in range(number_of_particles):
                for j in range(i + 1, number_of_particles):
                    two_particle_gradient = self._get_two_particle_gradient(positions[i], positions[j])
                    gradient[i] += two_particle_gradient
                    gradient[j] -= two_particle_gradient
            return gradient
        for i in range(number_of_particles):
            for j in range(i + 1, number_of_particles):
                separation_vector = get_shortest_vectors_on_torus(positions[i] - positions[j])
                two_particle_gradient = self._get_non_zero_two_particle_gradient(separation_vector,
                                                                                 np.linalg.norm(separation_vector))
                gradient[i] += two_particle_gradient
                gradient[j] -= two_particle_gradient
        return gradient

    def _get_two_particle_gradient(self, position_one, position_two):
        """
        Returns the gradient of the Lennard-Jones potential for two particles.

        Parameters
        ----------
        position_one : numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle one.
        position_two
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle two.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the gradient of the two-particle Lennard-Jones potential.
        """
        separation_vector = get_shortest_vectors_on_torus(position_one - position_two)
        separation_distance = np.linalg.norm(separation_vector)
        if separation_distance <= self._cutoff_length:
            return self._get_non_zero_two_particle_gradient(separation_vector, separation_distance)
        return 0.0

    def _get_non_zero_two_particle_gradient(self, separation_vector, separation_distance):
        """
        Returns the Lennard-Jones potential for two particles whose shortest separation distance is not greater than
        self._cutoff_length.

        Parameters
        ----------
        separation_distance : float
            The shortest separation distance between the two particles.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential (for all cases for which it is non-zero).
        """
        return - separation_vector * (self._gradient_12_constant * separation_distance ** (- 14.0) -
                                      self._gradient_6_constant * separation_distance ** (- 8.0))

    def _reset_linked_lists(self, positions):
        """
        Resets the linked lists (self._leading_particle_of_cell and self._particle_links) that allow us to avoid
        computing the bare two-particle potential and two-particle gradient for two particles whose minimum separation
        distance is greater than self._cutoff_length.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.
        """
        self._leading_particle_of_cell = [None for _ in range(self._total_number_of_cells)]
        for index, position in enumerate(positions):
            cell = np.int_(position // self._cell_size)
            cell_index = self._get_cell_index(cell)
            self._particle_links[index] = self._leading_particle_of_cell[cell_index]
            self._leading_particle_of_cell[cell_index] = index

    def _get_cell_index(self, cell):
        """
        Gets the cell index for some given cell coordinates.

        Parameters
        ----------
        cell : Sequence[int]
            A one-dimensional Python list of size 3; each element is an int and represents one Cartesian component of
            the cell coordinates.
        """
        return (cell[0] + self._number_of_cells_in_each_direction[0] * cell[1] +
                self._number_of_cells_in_each_direction[0] * self._number_of_cells_in_each_direction[1] * cell[2])
