"""Module for the InversePowerPotential class."""
from .potential import Potential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
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

        Raises
        ------
        base.exceptions.ConfigurationError
            If power is less than 1.0.
        """
        if power < 1.0:
            raise ConfigurationError(f"Give a value not less than 1.0 as power in {self.__class__.__name__}.")
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
        return self._one_over_power * np.sum(
            np.linalg.norm(get_shortest_vectors_on_torus(positions), axis=1) ** self._negative_power)

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
        toroidal_positions = get_shortest_vectors_on_torus(positions)
        return - toroidal_positions * np.linalg.norm(toroidal_positions, axis=1) ** self._negative_power_minus_two
