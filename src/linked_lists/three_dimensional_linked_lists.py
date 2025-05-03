"""Module for the ThreeDimensionalLinkedLists class."""
from typing import Sequence
from linked_lists.linked_lists import LinkedLists


class ThreeDimensionalLinkedLists(LinkedLists):
    r"""
    The ThreeDimensionalLinkedLists class.  For models of particles existing on a shared compact 3D manifold, this
    class implements the functionality required for linked (particle) lists between neighbouring cells, where these
    cells are cubic and tessellate the manifold.
    """

    def __init__(self, number_of_cells_in_each_direction: Sequence[int]) -> None:
        """
        The constructor of the ThreeDimensionalLinkedLists class.

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
        super().__init__(number_of_cells_in_each_direction)

    def get_cell_index(self, cell):
        """
        Gets the cell index for some given cell coordinates.

        Parameters
        ----------
        cell : Sequence[int]
            A one-dimensional Python list of size 3; each element is an int and represents one Cartesian component of
            the cell coordinates.
        """
        return (cell[0] + self.number_of_cells_in_each_direction[0] * cell[1] +
                self.number_of_cells_in_each_direction[0] * self.number_of_cells_in_each_direction[1] * cell[2])
