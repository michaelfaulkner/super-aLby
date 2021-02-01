"""Module for the LennardJonesPotentialWithLinkedLists class."""
from .lennard_jones_potentials_with_cutoff import LennardJonesPotentialsWithCutoff
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
from typing import Sequence
import itertools
import logging
import numpy as np


class LennardJonesPotentialWithLinkedLists(LennardJonesPotentialsWithCutoff):
    r"""
    With linked-lists, this class implements the Lennard-Jones potential

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

    is the bare two-particle Lennard-Jones potential. In the above, $\epsilon$ is the bare well depth, $\sigma$ is the
    characteristic length scale of the Lennard-Jones potential, and $r_c$ is the cutoff distance at which the bare
    two-particle potential is truncated. We recommend $r_c \ge 2.5 \sigma$.
    """

    def __init__(self, characteristic_length: float = 1.0, well_depth: float = 1.0, cutoff_length: float = 2.5,
                 prefactor: float = 1.0) -> None:
        """
        The constructor of the LennardJonesPotentialWithLinkedLists class.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
        cutoff_length : float, optional
            The cutoff distance at which the bare potential is truncated.
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
        """
        super().__init__(characteristic_length, well_depth, cutoff_length, prefactor)
        if dimensionality_of_particle_space != 3:
            raise ConfigurationError(f"For size_of_particle_space, give a one-dimensional list of length 3 (and "
                                     f"composed of floats) in {self.__class__.__name__}. This is because the "
                                     f"dimensionality of particle space must be 3 when using the linked-lists algorithm "
                                     f"in {self.__class__.__name__}.")
        self._number_of_cells_in_each_direction = np.int_(size_of_particle_space / self._cutoff_length)
        self._total_number_of_cells = int(np.prod(self._number_of_cells_in_each_direction))
        self._cell_size = size_of_particle_space / self._number_of_cells_in_each_direction
        self._leading_particle_of_cell = [None for _ in range(self._total_number_of_cells)]
        self._particle_links = [None for _ in range(number_of_particles)]
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           characteristic_length=characteristic_length, well_depth=well_depth,
                           cutoff_length=cutoff_length, prefactor=prefactor)

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
        potential = 0.0
        self._reset_linked_lists(positions)
        for cell_one in itertools.product(range(self._number_of_cells_in_each_direction[0]),
                                          range(self._number_of_cells_in_each_direction[1]),
                                          range(self._number_of_cells_in_each_direction[2])):
            cell_one_index = self._get_cell_index(cell_one)
            for cell_two in itertools.product(range(cell_one[0] - 1, cell_one[0] + 1),
                                              range(cell_one[1] - 1, cell_one[1] + 1),
                                              range(cell_one[2] - 1, cell_one[2] + 1)):
                cell_two_index = self._get_cell_index([int((element + self._number_of_cells_in_each_direction[index] /
                                                            2) % self._number_of_cells_in_each_direction[index] -
                                                           self._number_of_cells_in_each_direction[index] / 2)
                                                       for index, element in enumerate(cell_two)])
                particle_one_index = self._leading_particle_of_cell[cell_one_index]
                while particle_one_index is not None:
                    particle_two_index = self._leading_particle_of_cell[cell_two_index]
                    while particle_two_index is not None:
                        if particle_one_index > particle_two_index:
                            potential += self._get_two_particle_potential(positions[particle_one_index],
                                                                          positions[particle_two_index])
                        particle_two_index = self._particle_links[particle_two_index]
                    particle_one_index = self._particle_links[particle_one_index]
        return potential

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
        self._reset_linked_lists(positions)
        for cell_one in itertools.product(range(self._number_of_cells_in_each_direction[0]),
                                          range(self._number_of_cells_in_each_direction[1]),
                                          range(self._number_of_cells_in_each_direction[2])):
            cell_one_index = self._get_cell_index(cell_one)
            for cell_two in itertools.product(range(cell_one[0] - 1, cell_one[0] + 1),
                                              range(cell_one[1] - 1, cell_one[1] + 1),
                                              range(cell_one[2] - 1, cell_one[2] + 1)):
                cell_two_index = self._get_cell_index([int((element + self._number_of_cells_in_each_direction[index] /
                                                            2) % self._number_of_cells_in_each_direction[index] -
                                                           self._number_of_cells_in_each_direction[index] / 2)
                                                       for index, element in enumerate(cell_two)])
                particle_one_index = self._leading_particle_of_cell[cell_one_index]
                while particle_one_index is not None:
                    particle_two_index = self._leading_particle_of_cell[cell_two_index]
                    while particle_two_index is not None:
                        if particle_one_index > particle_two_index:
                            two_particle_gradient = self._get_two_particle_gradient(positions[particle_one_index],
                                                                                    positions[particle_two_index])
                            gradient[particle_one_index] += two_particle_gradient
                            gradient[particle_two_index] -= two_particle_gradient
                        particle_two_index = self._particle_links[particle_two_index]
                    particle_one_index = self._particle_links[particle_one_index]
        return gradient

    def _reset_linked_lists(self, positions):
        """
        Resets the linked lists (self._leading_particle_of_cell and self._particle_links) that allow us to avoid
        computing the bare two-particle potential and two-particle gradient for two particles whose cells are separated
        by at least one other cell in any Cartesian direction (since the length of each dimension of each cell is not
        less than self._cutoff_length).

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
