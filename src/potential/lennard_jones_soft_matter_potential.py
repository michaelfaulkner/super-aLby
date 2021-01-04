"""Module for the CoulombSoftMatterPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
import logging
import numpy as np


class LennardJonesSoftMatterPotential(SoftMatterPotential):
    r"""
    This class implements the two-particle Lennard-Jones potential+.
    """

    def __init__(self, characteristic_length: float = 0.1, prefactor: float = 1.0) -> None:
        """
        The constructor of the CoulombSoftMatterPotential class.

        The default values are optimized so that the result has machine precision.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        prefactor : float, optional
            The prefactor k of the potential.
        """
        self._characteristic_length = characteristic_length
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            The particle position vectors {r_i}.

        Returns
        -------
        float
            The potential.
        """
        separation_distance = np.linalg.norm(
            get_shortest_vectors_on_torus(positions[0] - positions[1])) / self._characteristic_length
        return separation_distance ** (- 12.0) - separation_distance ** (- 6.0)

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            The particle position vectors {r_i}.

        Returns
        -------
        numpy.ndarray
            The gradient.
        """
        separation_vector = get_shortest_vectors_on_torus(positions[0] - positions[1])
        separation_distance = np.linalg.norm(separation_vector)
        zero_particle_gradient = separation_vector * (
                - 12.0 * self._characteristic_length ** 12.0 * separation_distance ** (- 14.0)
                + 6.0 * self._characteristic_length ** 6.0 * separation_distance ** (- 8.0))
        return np.array([zero_particle_gradient, - zero_particle_gradient])
