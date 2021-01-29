"""Module for the LennardJonesPotentialsWithCutoff class."""
from .lennard_jones_potentials import LennardJonesPotentials
from base.exceptions import ConfigurationError
from base.vectors import get_shortest_vectors_on_torus
from abc import ABCMeta, abstractmethod
import numpy as np


class LennardJonesPotentialsWithCutoff(LennardJonesPotentials, metaclass=ABCMeta):
    r"""
    Abstract class for Lennard-Jones potentials (with a cutoff distance on the length scale of the interaction
    potential)

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
        The constructor of the LennardJonesPotential class.

        The default values are optimized so that the result has machine precision.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
        cutoff_length : float, optional
            The cutoff distance at which the bare potential is truncated. If no cutoff is required, give inf in the
            configuration file.
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
        super().__init__(characteristic_length, well_depth, prefactor)
        if cutoff_length < 2.5 * characteristic_length:
            raise ConfigurationError(f"Ensure that the value of cutoff_length is not less than 2.5 times the value of "
                                     f"characteristic_length in {self.__class__.__name__}.")
        self._cutoff_length = cutoff_length
        self._bare_potential_at_cut_off = (self._potential_12_constant * cutoff_length ** (- 12.0) -
                                           self._potential_6_constant * cutoff_length ** (- 6.0))

    @abstractmethod
    def get_value(self, positions):
        """
        Returns the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, positions):
        """
        Returns the gradient of the potential function for the given particle positions.

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
        raise NotImplementedError

    def _get_two_particle_potential(self, position_one, position_two):
        """
        Returns the Lennard-Jones potential for two particles.

        Parameters
        ----------
        position_one : numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle one.
        position_two
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle two.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential.
        """
        separation_distance = np.linalg.norm(get_shortest_vectors_on_torus(position_one - position_two))
        if separation_distance <= self._cutoff_length:
            return self._get_non_zero_two_particle_potential(separation_distance)
        return 0.0

    def _get_two_particle_gradient(self, position_one, position_two):
        """
        Returns the gradient of the Lennard-Jones potential for two particles.

        Parameters
        ----------
        position_one : numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle one.
        position_two
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the position of particle two.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the gradient of the two-particle Lennard-Jones potential.
        """
        separation_vector = get_shortest_vectors_on_torus(position_one - position_two)
        separation_distance = np.linalg.norm(separation_vector)
        if separation_distance <= self._cutoff_length:
            return self._get_non_zero_two_particle_gradient(separation_vector, separation_distance)
        return 0.0
