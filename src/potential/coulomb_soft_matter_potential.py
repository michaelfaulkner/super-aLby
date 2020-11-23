"""Module for the CoulombSoftMatterPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_corrected_separation, permutation_3d
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
import logging
import numpy as np


class CoulombSoftMatterPotential(SoftMatterPotential):
    r"""
    This class implements the machine-precise Coulomb potential
    U_ij = k * c_i * c_j * \sum_{\vec{n}\in\np.b{Z}^3} 1/ (|\vec{r_ij}+\vec{n}\vec{L}|).

    k is a prefactor, c_i and c_j are the charges of the involved units and r_ij = r_j - r_i is the separation between
    the units. \vec{L} are the sides of the three-dimensional simulation box with periodic boundary conditions.
    This class is (currently) only implemented for a three-dimensional cube, and for like charges.
    The conditionally convergent sum can be consistently defined in terms of tin-foil boundary conditions. Then,
    the sum is absolutely convergent. Ewald summation splits the sum up partly in position space and partly in Fourier
    space. The summation has three parameters: the cutoff in Fourier space, the cutoff in position space, and a
    convergence factor alpha, which balances the converging speeds of the two sums.
    """

    def __init__(self, alpha: float = 3.45, fourier_cutoff: int = 6, position_cutoff: int = 2,
                 prefactor: float = 1.0) -> None:
        """
        The constructor of the CoulombSoftMatterPotential class.

        The default values are optimized so that the result has machine precision.

        Parameters
        ----------
        alpha : float, optional
            The convergence factor alpha of the Ewald summation.
        fourier_cutoff : int, optional
            The cutoff in Fourier space of the Ewald summation.
        position_cutoff : int, optional
            The cutoff in position space of the Ewald summation.
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the dimension does not equal three.
        base.exceptions.ConfigurationError
            If the cutoff in Fourier space is negative.
        base.exceptions.ConfigurationError
            If the cutoff in position space is negative.
        """
        if not dimensionality_of_particle_space == 3:
            raise ConfigurationError("The potential {0} can only be used in three-dimensional particle space."
                                     .format(self.__class__.__name__))
        if alpha <= 0.0:
            raise ConfigurationError("The convergence factor alpha must be > 0.0 in the class {0}."
                                     .format(self.__class__.__name__))
        if fourier_cutoff < 0:
            raise ConfigurationError("The argument fourier_cutoff must be >= 0 in the class {0}."
                                     .format(self.__class__.__name__))
        if position_cutoff < 0:
            raise ConfigurationError("The argument position_cutoff must be >= 0 in the class {0}."
                                     .format(self.__class__.__name__))

        self._system_length = size_of_particle_space[0]  # todo convert to non-cubic particle spaces
        pi_sq = np.pi * np.pi
        self._fourier_cutoff = fourier_cutoff
        self._position_cutoff = position_cutoff

        self._alpha = alpha / self._system_length
        self._alpha_sq = self._alpha * self._alpha
        self._fourier_cutoff_sq = self._fourier_cutoff * self._fourier_cutoff
        self._position_cutoff_sq = self._position_cutoff * self._position_cutoff
        self._two_alpha_over_root_pi = 2 * self._alpha / np.sqrt(np.pi)
        self.two_pi_over_length = 2 * np.pi / self._system_length
        length_sq = self._system_length * self._system_length

        fourier_list = [[[0.0 for _ in range(self._fourier_cutoff + 1)] for _ in
                         range(self._fourier_cutoff + 1)] for _ in range(self._fourier_cutoff + 1)]
        fourier_list_for_potential = [[[0.0 for _ in range(self._fourier_cutoff + 1)] for _ in
                                       range(self._fourier_cutoff + 1)] for _ in range(self._fourier_cutoff + 1)]
        for k in range(0, self._fourier_cutoff + 1):
            for j in range(0, self._fourier_cutoff + 1):
                for i in range(1, self._fourier_cutoff + 1):
                    if j == 0 and k == 0:
                        coefficient = 1.0
                    elif k == 0:
                        coefficient = 2.0
                    elif j == 0:
                        coefficient = 2.0
                    else:
                        coefficient = 4.0

                    norm_sq = i * i + j * j + k * k
                    base_fourier_list_component = 2 * coefficient * (
                            np.exp(- pi_sq * norm_sq / self._alpha_sq / length_sq) / norm_sq / self._system_length)
                    fourier_list_for_potential[i][j][k] = base_fourier_list_component / np.pi
                    # todo change the following to 2 * base_fourier_list_component / self._system_length
                    fourier_list[i][j][k] = 4 * i * coefficient * (np.exp(
                        - pi_sq * norm_sq / self._alpha_sq / length_sq) / norm_sq / length_sq)

        self._fourier_array = tuple(
            [tuple([tuple(fourier_list[i][j]) for j in range(self._fourier_cutoff + 1)]) for i in
             range(self._fourier_cutoff + 1)])

        self._fourier_array_for_potential = tuple(
            [tuple([tuple(fourier_list_for_potential[i][j]) for j in range(self._fourier_cutoff + 1)])
             for i in range(self._fourier_cutoff + 1)])

        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           alpha=alpha, fourier_cutoff=fourier_cutoff, position_cutoff=position_cutoff,
                           prefactor=prefactor)
        
    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            The particle-particle separation vectors {r_ij}.

        Returns
        -------
        float
            The potential.
        """
        # todo add functionality for non-like charges
        potential = 0.0
        for i in range(number_of_particles):
            for j in range(i, number_of_particles):
                separation = get_corrected_separation(positions[i] - positions[j])
                for direction in range(3):
                    permuted_separation = permutation_3d(separation, direction)
                    potential += (self._get_two_particle_position_space_potential(*permuted_separation) +
                                  self._get_two_particle_fourier_space_potential(*permuted_separation))
        return potential

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            The particle-particle separation vectors {r_ij}.

        Returns
        -------
        numpy.ndarray
            The gradient.
        """
        gradient = np.array([np.zeros(3) for _ in range(number_of_particles)])
        for i in range(number_of_particles):
            for j in range(i, number_of_particles):
                separation = get_corrected_separation(positions[i] - positions[j])
                for direction in range(3):
                    permuted_separation = permutation_3d(separation, direction)
                    two_particle_gradient = (self._get_two_particle_position_space_gradient(*permuted_separation) +
                                             self._get_two_particle_fourier_space_gradient(*permuted_separation))
                    gradient[i][direction] += two_particle_gradient
                    gradient[j][direction] -= two_particle_gradient
        return gradient

    def _get_two_particle_position_space_potential(self, separation_x, separation_y, separation_z):
        """Returns the position-space part of the Ewald sum of the two-particle potential."""
        two_particle_position_space_potential = 0.0

        for k in range(- self._position_cutoff, self._position_cutoff + 1):
            vector_z_sq = (separation_z + k * self._system_length) * (separation_z + k * self._system_length)
            cutoff_y = int((self._position_cutoff_sq - k * k) ** 0.5)
            for j in range(- cutoff_y, cutoff_y + 1):
                vector_y_sq = (separation_y + j * self._system_length) * (separation_y + j * self._system_length)
                cutoff_x = int((self._position_cutoff_sq - j * j - k * k) ** 0.5)
                for i in range(- cutoff_x, cutoff_x + 1):
                    vector_x = separation_x + i * self._system_length
                    vector_sq = vector_x * vector_x + vector_y_sq + vector_z_sq
                    vector_norm = vector_sq ** 0.5
                    two_particle_position_space_potential += np.erfc(self._alpha * vector_norm) / vector_norm

        return two_particle_position_space_potential

    def _get_two_particle_fourier_space_potential(self, separation_x, separation_y, separation_z):
        """Returns the Fourier-space part of the Ewald sum of the two-particle potential."""
        two_particle_fourier_space_potential = 0.0

        delta_cos_x = np.cos(self.two_pi_over_length * separation_x)
        delta_sin_x = np.sin(self.two_pi_over_length * separation_x)
        delta_cos_y = np.cos(self.two_pi_over_length * separation_y)
        delta_sin_y = np.sin(self.two_pi_over_length * separation_y)
        delta_cos_z = np.cos(self.two_pi_over_length * separation_z)
        delta_sin_z = np.sin(self.two_pi_over_length * separation_z)

        cos_x = delta_cos_x
        sin_x = delta_sin_x
        cos_y = 1.0
        sin_y = 0.0
        cos_z = 1.0
        sin_z = 0.0

        for i in range(1, self._fourier_cutoff + 1):
            cutoff_y = int((self._fourier_cutoff_sq - i * i) ** 0.5)
            for j in range(0, cutoff_y + 1):
                cutoff_z = int((self._fourier_cutoff_sq - i * i - j * j) ** 0.5)
                for k in range(0, cutoff_z + 1):
                    two_particle_fourier_space_potential += (self._fourier_array_for_potential[i][j][k] *
                                                             cos_x * cos_y * cos_z)

                    if k != cutoff_z:
                        store_cos_z = cos_z
                        cos_z = store_cos_z * delta_cos_z - sin_z * delta_sin_z
                        sin_z = sin_z * delta_cos_z + store_cos_z * delta_sin_z
                    elif j != cutoff_y:
                        store_cos_y = cos_y
                        cos_y = store_cos_y * delta_cos_y - sin_y * delta_sin_y
                        sin_y = sin_y * delta_cos_y + store_cos_y * delta_sin_y
                        cos_z = 1.0
                        sin_z = 0.0
                    elif i != self._fourier_cutoff:
                        store_cos_x = cos_x
                        cos_x = store_cos_x * delta_cos_x - sin_x * delta_sin_x
                        sin_x = sin_x * delta_cos_x + store_cos_x * delta_sin_x
                        cos_y = 1.0
                        sin_y = 0.0
                        cos_z = 1.0
                        sin_z = 0.0

        return two_particle_fourier_space_potential

    def _get_two_particle_position_space_gradient(self, separation_x, separation_y, separation_z):
        """Returns the position-space part of the Ewald sum of the two-particle gradient."""
        two_particle_position_space_gradient = 0.0

        for k in range(- self._position_cutoff, self._position_cutoff + 1):
            vector_z_sq = (separation_z + k * self._system_length) * (separation_z + k * self._system_length)
            cutoff_y = int((self._position_cutoff_sq - k * k) ** 0.5)
            for j in range(- cutoff_y, cutoff_y + 1):
                vector_y_sq = (separation_y + j * self._system_length) * (separation_y + j * self._system_length)
                cutoff_x = int((self._position_cutoff_sq - j * j - k * k) ** 0.5)
                for i in range(- cutoff_x, cutoff_x + 1):
                    vector_x = separation_x + i * self._system_length
                    vector_sq = vector_x * vector_x + vector_y_sq + vector_z_sq
                    vector_norm = vector_sq ** 0.5
                    two_particle_position_space_gradient += (vector_x * (
                            self._two_alpha_over_root_pi * np.exp(- self._alpha_sq * vector_sq) + np.erfc(
                                self._alpha * vector_norm) / vector_norm) / vector_sq)

        return two_particle_position_space_gradient

    def _get_two_particle_fourier_space_gradient(self, separation_x, separation_y, separation_z):
        """Returns the Fourier-space part of the Ewald sum of the two-particle gradient."""
        two_particle_fourier_space_gradient = 0.0

        delta_cos_x = np.cos(self.two_pi_over_length * separation_x)
        delta_sin_x = np.sin(self.two_pi_over_length * separation_x)
        delta_cos_y = np.cos(self.two_pi_over_length * separation_y)
        delta_sin_y = np.sin(self.two_pi_over_length * separation_y)
        delta_cos_z = np.cos(self.two_pi_over_length * separation_z)
        delta_sin_z = np.sin(self.two_pi_over_length * separation_z)

        cos_x = delta_cos_x
        sin_x = delta_sin_x
        cos_y = 1.0
        sin_y = 0.0
        cos_z = 1.0
        sin_z = 0.0

        for i in range(1, self._fourier_cutoff + 1):
            cutoff_y = int((self._fourier_cutoff_sq - i * i) ** 0.5)
            for j in range(0, cutoff_y + 1):
                cutoff_z = int((self._fourier_cutoff_sq - i * i - j * j) ** 0.5)
                for k in range(0, cutoff_z + 1):
                    two_particle_fourier_space_gradient += self._fourier_array[i][j][k] * sin_x * cos_y * cos_z

                    if k != cutoff_z:
                        store_cos_z = cos_z
                        cos_z = store_cos_z * delta_cos_z - sin_z * delta_sin_z
                        sin_z = sin_z * delta_cos_z + store_cos_z * delta_sin_z
                    elif j != cutoff_y:
                        store_cos_y = cos_y
                        cos_y = store_cos_y * delta_cos_y - sin_y * delta_sin_y
                        sin_y = sin_y * delta_cos_y + store_cos_y * delta_sin_y
                        cos_z = 1.0
                        sin_z = 0.0
                    elif i != self._fourier_cutoff:
                        store_cos_x = cos_x
                        cos_x = store_cos_x * delta_cos_x - sin_x * delta_sin_x
                        sin_x = sin_x * delta_cos_x + store_cos_x * delta_sin_x
                        cos_y = 1.0
                        sin_y = 0.0
                        cos_z = 1.0
                        sin_z = 0.0

        return two_particle_fourier_space_gradient
