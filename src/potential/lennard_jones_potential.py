"""Module for the LennardJonesPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import number_of_particles
import logging
import numpy as np


class LennardJonesPotential(SoftMatterPotential):
    # todo debug then add functionality for many particles
    r"""
    This class implements the two-particle Lennard-Jones potential+.
    """

    def __init__(self, characteristic_length: float = 1.0, prefactor: float = 1.0) -> None:
        """
        The constructor of the LennardJonesPotential class.

        The default values are optimized so that the result has machine precision.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        base.exceptions.ConfigurationError
            If number_of_particles does not equal two.
        """
        if number_of_particles != 2:
            raise ConfigurationError(f"Give a value of 2 as the number_of_particles in [ModelSettings] when using "
                                     f"{self.__class__.__name__} as it is currently a two-particle potential.")
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
        return separation_distance ** (- 12) - separation_distance ** (- 6)

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
                - separation_distance ** (- 14) * 12.0 * self._characteristic_length ** 12
                + separation_distance ** (- 8) * 6.0 * self._characteristic_length ** 6)
        return np.array([zero_particle_gradient, - zero_particle_gradient])
