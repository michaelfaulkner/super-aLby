"""Module for the LinkedLists class."""
from model_settings import number_of_particles, size_of_particle_space
from abc import ABCMeta, abstractmethod
from typing import Sequence
import numpy as np


class LinkedLists(metaclass=ABCMeta):
    r"""
    The abstract LinkedLists class.  For models of particles existing on a shared compact manifold, this class
        implements the functionality required for linked (particle) lists between neighbouring cells, where these cells
        are hypercubic and tessellate the manifold.  Different child classes correspond to different dimensionality of
        the compact manifold.
    """

    def __init__(self, number_of_cells_in_each_direction: Sequence[int]) -> None:
        """
        The constructor of the LinkedLists class.

        Parameters
        ----------
        number_of_cells_in_each_direction : int, optional
            Sequence of integers, where each represents the number of cells along each Cartesian direction.

        Raises
        ------
        base.exceptions.ConfigurationError
            If number_of_cells_in_each_direction is not a list and any element number_of_cells_in_each_direction is not
            greater than 0.
        """
        for element in number_of_cells_in_each_direction:
            if element < 1:
                raise ValueError(f"number_of_cells_in_each_direction in {self.__class__.__name__} must be a Python "
                                 f"list composed of integers, each greater than 0.")
        self.number_of_cells_in_each_direction = number_of_cells_in_each_direction
        self._total_number_of_cells = int(np.prod(self.number_of_cells_in_each_direction))
        self.cell_size = size_of_particle_space / self.number_of_cells_in_each_direction
        self.leading_particle_of_cell = [None for _ in range(self._total_number_of_cells)]
        self.next_particle_in_same_cell = [None for _ in range(number_of_particles)]

    def reset_linked_lists(self, positions):
        """
        Resets the linked lists (self._leading_particle_of_cell and self._next_particle_in_same_cell).

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.
        """
        self.leading_particle_of_cell = [None for _ in range(self._total_number_of_cells)]
        for particle_index, position in enumerate(positions):
            cell = self.get_cell(position)
            cell_index = self.get_cell_index(cell)
            self.next_particle_in_same_cell[particle_index] = self.leading_particle_of_cell[cell_index]
            self.leading_particle_of_cell[cell_index] = particle_index

    def get_cell(self, position):
        """
        Gets the cell coordinates for some given particle position.

        Parameters
        ----------
        position : numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the particle position.

        Returns
        ----------
        cell : numpy.ndarray
            The cell coordinates.  This is a one-dimensional numpy array of length dimensionality_of_particle_space.
            Each component is an integer and represents one Cartesian cell coordinate.
        """
        return np.int_((position + 0.5 * size_of_particle_space) // self.cell_size)

    @abstractmethod
    def get_cell_index(self, cell):
        """
        Gets the cell index for some given cell coordinates.

        Parameters
        ----------
        cell : Sequence[int]
            A one-dimensional Python list of size dimensionality_of_particle_space; each element is an int and
            represents one Cartesian component of the cell coordinates.
        """
        raise NotImplementedError
