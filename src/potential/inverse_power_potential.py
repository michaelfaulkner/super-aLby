"""Module for the InversePowerPotential class."""
from .potential import Potential
from base.logging import log_init_arguments
from base.vectors import get_shortest_vector_on_ring, get_shortest_vector_on_torus
from model_settings import dimensionality_of_momenta_array, dimensionality_of_particle_space
import logging
import numpy as np


class InversePowerPotential(Potential):
    """This class implements the inverse power potential U = sum(|| positions[i] || ** (- power) / power)"""

    def __init__(self, power: float = 1.0, prefactor: float = 1.0):
        """
        The constructor of the InversePowerPotential class.

        Parameters
        ----------
        power : int
            Minus 1 multiplied by the power to which the norm of each component of the positions is raised (i.e., the
            norm of each particle position vector).
        prefactor : float
            The prefactor k of the potential.
        """
        self._one_over_power = 1.0 / power
        self._negative_power = - power
        self._negative_power_minus_two = - power - 2.0
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power, prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        float
            The potential.
        """
        if dimensionality_of_particle_space == 1:
            return self._one_over_power * sum(
                [abs(get_shortest_vector_on_ring(position, 0)) ** self._negative_power for position in positions])
        return self._one_over_power * sum(
            [np.linalg.norm(get_shortest_vector_on_torus(position)) ** self._negative_power for position in positions])

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        numpy.ndarray
            The gradient.
        """
        if dimensionality_of_particle_space == 1:
            return np.array([self._one_particle_gradient(get_shortest_vector_on_ring(position, 0))
                             for position in positions])
        return np.array([self._one_particle_gradient(get_shortest_vector_on_torus(position)) for position in positions])

    def _one_particle_gradient(self, shortest_position_vector):
        return - shortest_position_vector * np.linalg.norm(shortest_position_vector) ** self._negative_power_minus_two
