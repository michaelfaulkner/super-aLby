"""Module for the CoulombPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus, get_permuted_3d_vector
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
import logging
import math
import numpy as np


class CoulombPotential(SoftMatterPotential):
    # todo add functionality for non-like charges
    r"""
    This class implements the machine-precise Coulomb potential

        $ U = k * \sum_{i > j} [
            \sum_{\vec{n}\in\np.b{Z}^3} \erfc(\alpha |\vec{r}_{ij} + \vec{n}L|) / |\vec{r_ij} + \vec{n}\vec{L}| +
            \frac{1}{\pi} \sum_{\vec{m} \ne 0} \frac{\exp( -\pi^2 |\vec{m}|^2 / \alpha^2 L^2)}{|\vec{m}^2|}
                \cos(\frac{2\pi}{L}\vec{m} \cdot \vec{r}){ij})] $

    k is a prefactor and r_ij = r_i - r_j is the separation vector between particles i and j (which we correct for
    periodic boundaries conditions). \vec{L} are the sides of the three-dimensional simulation box with periodic
    boundary conditions. As our code is only based on potential differences, we have not subtracted the self-energy term

        $ U_{\rm self}(\alpha, \{ c_i \}) = \alpha \sum_i c_i^2 / \pi^{1 / 2} +  \pi (\sum_i c_i)^2 / 2 L^3 \alpha^2 $

    from the potential, where $ \{ c_i \} \equiv \{ c_i : i = 1, \dots , N \} $ is the set of charge values of the
    particles, all of which are currently set to unity. This class is (currently) only implemented for a
    three-dimensional cube, and for charges of value unity, both of which will be generalised in the future.
    We have chosen tin-foil boundary conditions, which is a bit of a misnomer as this refers to medium in which the
    infinite number of copies of the system are embedded. Ewald summation splits the sum into position-space and
    Fourier-space parts. The summation has three parameters: the cutoff in Fourier space, the cutoff in position space,
    and a convergence factor alpha, which balances the converging speeds of the two sums.
    """

    def __init__(self, alpha: float = 3.45, fourier_cutoff: int = 6, position_cutoff: int = 2,
                 prefactor: float = 1.0) -> None:
        """
        The constructor of the CoulombPotential class.

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
            If model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        base.exceptions.ConfigurationError
            If the values of each component of model_settings.size_of_particle_space are not equal (since the Ewald sum
            is currently only configured for the three-dimensional cube).
        base.exceptions.ConfigurationError
            If model_settings.dimensionality_of_particle_space does not equal three.
        base.exceptions.ConfigurationError
            If the convergence parameter alpha is not greater than 0.0.
        base.exceptions.ConfigurationError
            If the cutoff in Fourier space is less than 0.
        base.exceptions.ConfigurationError
            If the cutoff in position space is less than 0.
        base.exceptions.ConfigurationError
            If model_settings.number_of_particles does not equal two.
        """
        super().__init__(prefactor)
        if dimensionality_of_particle_space != 3:
            raise ConfigurationError(f"For size_of_particle_space, give a list of length 3, where each component is a "
                                     f"list of two float values, when using {self.__class__.__name__}.")
        if (size_of_particle_space[0] != size_of_particle_space[1] or size_of_particle_space[1] !=
                size_of_particle_space[2] or size_of_particle_space[2] != size_of_particle_space[0]):
            raise ConfigurationError(f"When using {self.__class__.__name__}, give equal values for each component of "
                                     f"size_of_particle_space.")
        if alpha <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for the convergence factor alpha of "
                                     f"{self.__class__.__name__}.")
        if fourier_cutoff < 0:
            raise ConfigurationError(f"Give a value not less than 0 for fourier_cutoff in {self.__class__.__name__}.")
        if position_cutoff < 0:
            raise ConfigurationError(f"Give a value not less than 0 for position_cutoff in {self.__class__.__name__}.")
        if number_of_particles != 2:
            raise ConfigurationError(f"Give a value of 2 as the number_of_particles in [ModelSettings] when using "
                                     f"{self.__class__.__name__} as it is currently a two-particle potential.")
        self._system_length = size_of_particle_space[0]  # todo convert to non-cubic particle space
        self._fourier_cutoff = fourier_cutoff
        self._position_cutoff = position_cutoff
        self._alpha = alpha / self._system_length
        self._alpha_sq = self._alpha * self._alpha
        self._fourier_cutoff_sq = self._fourier_cutoff * self._fourier_cutoff
        self._position_cutoff_sq = self._position_cutoff * self._position_cutoff
        self._two_alpha_over_root_pi = 2 * self._alpha / np.sqrt(np.pi)
        self._two_pi_over_length = 2 * np.pi / self._system_length
        self._length_sq = self._system_length ** 2
        self._fourier_array = np.zeros((self._fourier_cutoff + 1, self._fourier_cutoff + 1, self._fourier_cutoff + 1))
        for k in range(0, self._fourier_cutoff + 1):
            for j in range(0, self._fourier_cutoff + 1):
                for i in range(0, self._fourier_cutoff + 1):
                    if (i == 0 and j == 0) or (j == 0 and k == 0) or (k == 0 and i == 0):
                        coefficient = 1.0
                    elif i == 0 or j == 0 or k == 0:
                        coefficient = 2.0
                    else:
                        coefficient = 4.0
                    norm_sq = i ** 2 + j ** 2 + k ** 2
                    if not (i == 0 and j == 0 and k == 0):
                        self._fourier_array[i, j, k] = 2.0 * prefactor * coefficient * np.exp(- np.pi ** 2 * norm_sq /
                                                                                              self._alpha_sq /
                                                                                              self._length_sq) / norm_sq
        self._fourier_array.flags.writeable = False
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           alpha=alpha, fourier_cutoff=fourier_cutoff, position_cutoff=position_cutoff,
                           prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, 3); each element is a float and represents one
            Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential.
        """
        return sum([self._get_two_particle_potential(positions[i], positions[j]) for i in range(number_of_particles)
                    for j in range(i + 1, number_of_particles)])

    def _get_two_particle_potential(self, position_one, position_two):
        """
        Returns the two-particle potential between particles one and two.

        Parameters
        ----------
        position_one : numpy.ndarray
            A one-dimensional numpy array of size 3; each element is a float and represents one Cartesian component of
            the position of particle one.
        position_two : numpy.ndarray
            A one-dimensional numpy array of size 3; each element is a float and represents one Cartesian component of
            the position of particle two.

        Returns
        -------
        float
            The two-particle potential.
        """
        separation = get_shortest_vectors_on_torus(position_one - position_two)
        return (self._get_two_particle_position_space_potential(separation) +
                self._get_two_particle_fourier_space_potential(separation))

    def _get_two_particle_position_space_potential(self, separation):
        """Returns the position-space part of the Ewald sum of the two-particle potential."""
        two_particle_position_space_potential = 0.0
        for k in range(- self._position_cutoff, self._position_cutoff + 1):
            vector_z_sq = (separation[2] + k * self._system_length) * (separation[2] + k * self._system_length)
            cutoff_y = int((self._position_cutoff_sq - k * k) ** 0.5)
            for j in range(- cutoff_y, cutoff_y + 1):
                vector_y_sq = (separation[1] + j * self._system_length) * (separation[1] + j * self._system_length)
                cutoff_x = int((self._position_cutoff_sq - j * j - k * k) ** 0.5)
                for i in range(- cutoff_x, cutoff_x + 1):
                    vector_x = separation[0] + i * self._system_length
                    vector_sq = vector_x * vector_x + vector_y_sq + vector_z_sq
                    vector_norm = vector_sq ** 0.5
                    two_particle_position_space_potential += math.erfc(self._alpha * vector_norm) / vector_norm
        return self._prefactor * two_particle_position_space_potential

    def _get_two_particle_fourier_space_potential(self, separation):
        """Returns the Fourier-space part of the Ewald sum of the two-particle potential."""
        cos, sin = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        delta_cos, delta_sin = self._set_delta_cos_and_delta_sin(separation)
        two_particle_fourier_space_potential = 0.0
        for i in range(0, self._fourier_cutoff + 1):
            cutoff_y = int((self._fourier_cutoff_sq - i * i) ** 0.5)
            for j in range(0, cutoff_y + 1):
                cutoff_z = int((self._fourier_cutoff_sq - i * i - j * j) ** 0.5)
                for k in range(0, cutoff_z + 1):
                    two_particle_fourier_space_potential += (self._fourier_array[i, j, k] * cos[0] * cos[1] * cos[2] /
                                                             self._system_length / np.pi)
                    cos, sin = self._update_trig_functions(cos, sin, delta_cos, delta_sin, cutoff_y, cutoff_z, i, j, k)
        return two_particle_fourier_space_potential

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
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        gradient = np.zeros((number_of_particles, 3))
        for i in range(number_of_particles):
            for j in range(i + 1, number_of_particles):
                separation = get_shortest_vectors_on_torus(positions[i] - positions[j])
                for direction in range(3):
                    permuted_separation = get_permuted_3d_vector(separation, direction)
                    two_particle_gradient = (self._get_two_particle_position_space_gradient(permuted_separation) +
                                             self._get_two_particle_fourier_space_gradient(permuted_separation))
                    gradient[i][direction] += two_particle_gradient
                    gradient[j][direction] -= two_particle_gradient
        return gradient

    def _get_two_particle_position_space_gradient(self, separation):
        """Returns the position-space part of the Ewald sum of the two-particle gradient."""
        two_particle_position_space_gradient = 0.0
        for k in range(- self._position_cutoff, self._position_cutoff + 1):
            vector_z_sq = (separation[2] + k * self._system_length) * (separation[2] + k * self._system_length)
            cutoff_y = int((self._position_cutoff_sq - k * k) ** 0.5)
            for j in range(- cutoff_y, cutoff_y + 1):
                vector_y_sq = (separation[1] + j * self._system_length) * (separation[1] + j * self._system_length)
                cutoff_x = int((self._position_cutoff_sq - j * j - k * k) ** 0.5)
                for i in range(- cutoff_x, cutoff_x + 1):
                    vector_x = separation[0] + i * self._system_length
                    vector_sq = vector_x * vector_x + vector_y_sq + vector_z_sq
                    vector_norm = vector_sq ** 0.5
                    two_particle_position_space_gradient -= (vector_x * (self._two_alpha_over_root_pi *
                                                                         np.exp(- self._alpha_sq * vector_sq) +
                                                                         math.erfc(self._alpha * vector_norm) /
                                                                         vector_norm) / vector_sq)
        return self._prefactor * two_particle_position_space_gradient

    def _get_two_particle_fourier_space_gradient(self, separation):
        """Returns the Fourier-space part of the Ewald sum of the two-particle gradient."""
        delta_cos, delta_sin = self._set_delta_cos_and_delta_sin(separation)
        cos, sin = [delta_cos[0], 1.0, 1.0], [delta_sin[0], 0.0, 0.0]
        two_particle_fourier_space_gradient = 0.0
        for i in range(1, self._fourier_cutoff + 1):
            cutoff_y = int((self._fourier_cutoff_sq - i * i) ** 0.5)
            for j in range(0, cutoff_y + 1):
                cutoff_z = int((self._fourier_cutoff_sq - i * i - j * j) ** 0.5)
                for k in range(0, cutoff_z + 1):
                    two_particle_fourier_space_gradient -= (2.0 * i * self._fourier_array[i, j, k] *
                                                            sin[0] * cos[1] * cos[2] / self._length_sq)
                    cos, sin = self._update_trig_functions(cos, sin, delta_cos, delta_sin, cutoff_y, cutoff_z, i, j, k)
        return two_particle_fourier_space_gradient

    def _set_delta_cos_and_delta_sin(self, separation):
        """Returns the delta_cos and delta_sin functions for the two-particle Fourier-space calculations."""
        return ([np.cos(self._two_pi_over_length * separation[0]), np.cos(self._two_pi_over_length * separation[1]),
                 np.cos(self._two_pi_over_length * separation[2])],
                [np.sin(self._two_pi_over_length * separation[0]), np.sin(self._two_pi_over_length * separation[1]),
                 np.sin(self._two_pi_over_length * separation[2])])

    def _update_trig_functions(self, cos, sin, delta_cos, delta_sin, cutoff_y, cutoff_z, i, j, k):
        """Recalculates the cos, sin, delta_cos and delta_sin functions (which we implement after each iteration
            through Fourier space) for the two-particle Fourier-space calculations."""
        if k != cutoff_z:
            store_cos_z = cos[2]
            cos[2] = store_cos_z * delta_cos[2] - sin[2] * delta_sin[2]
            sin[2] = sin[2] * delta_cos[2] + store_cos_z * delta_sin[2]
        elif j != cutoff_y:
            store_cos_y = cos[1]
            cos[1] = store_cos_y * delta_cos[1] - sin[1] * delta_sin[1]
            sin[1] = sin[1] * delta_cos[1] + store_cos_y * delta_sin[1]
            cos[2] = 1.0
            sin[2] = 0.0
        elif i != self._fourier_cutoff:
            store_cos_x = cos[0]
            cos[0] = store_cos_x * delta_cos[0] - sin[0] * delta_sin[0]
            sin[0] = sin[0] * delta_cos[0] + store_cos_x * delta_sin[0]
            cos[1] = 1.0
            sin[1] = 0.0
            cos[2] = 1.0
            sin[2] = 0.0
        return cos, sin
