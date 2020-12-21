"""Module for the InversePowerPotential class."""
from .potential import Potential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vector_on_ring, get_shortest_vector_on_torus
from model_settings import dimensionality_of_particle_space, size_of_particle_space
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
            If type(size_of_particle_space) is None.
        base.exceptions.ConfigurationError
            If power is less than 1.0.
        base.exceptions.ConfigurationError
            If power is less than float(dimensionality_of_particle_space + 2).
        """
        if type(size_of_particle_space) is None:
            raise ConfigurationError(f"When using {self.__class__.__name__}, give a value either of type float or of "
                                     f"type list (where the type of each of its elements is a float) as "
                                     f"size_of_particle_space in the INI section [ModelSettings].")
        if power < 1.0:
            raise ConfigurationError(f"Give a value not less than 1.0 as power in {self.__class__.__name__}.")
        if power < float(dimensionality_of_particle_space + 2):
            raise ConfigurationError(f"Give a value not less than the dimensionality of particle space plus 2 as power "
                                     f"in {self.__class__.__name__}: if type(size_of_particle_space) is list, give a "
                                     f"value not less than len(size_of_particle_space) + 2 as power; if "
                                     f"type(size_of_particle_space) if float, give a value not less than 3 as power. "
                                     f"This ensures that potential at the boundaries is negligible relative to the "
                                     f"rest of the space; otherwise, we would have to account for image potentials (as "
                                     f"we do with the Coulomb potential.")
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
            for index, position in enumerate(positions):
                positions[index] = self._one_particle_gradient(get_shortest_vector_on_ring(position, 0))
            return positions
        for index, position in enumerate(positions):
            positions[index] = self._one_particle_gradient(get_shortest_vector_on_torus(position))
        return positions

    def _one_particle_gradient(self, shortest_position_vector):
        return - shortest_position_vector * np.linalg.norm(shortest_position_vector) ** self._negative_power_minus_two
