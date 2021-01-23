"""Module for the LennardJonesPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import number_of_particles
import logging
import numpy as np


class LennardJonesPotential(SoftMatterPotential):
    r"""
    This class implements the two-particle Lennard-Jones potential

        $ U = k * \sum_{i > j} U_{{\rm LJ}, ij} $ ,

    where

        $ U_{{\rm LJ}, ij} = \begin{cases}
                                U_{{\rm LJ}, ij}^{\rm bare}(r_{ij}) - U_{{\rm LJ}, ij}^{\rm bare}(r_{\rm c}) \,
                                    {\rm if} \, r_{ij} \le r_{\rm c} \\
                                0 \, {\rm if} \, r_{ij} > r_{\rm c}
                             \end{cases} $

    is the two-particle Lennard-Jones potential, and

        $ U_{{\rm LJ}, ij}^{\rm bare}(r_{ij}) = 4 \epsilon \left[\left(\frac{\sigma}{r_{ij}}\right)^12 -
            \left(\frac{\sigma}{r_{ij}}\right)^6\right]$

    is the bare two-particle Lennard-Jones potential. In the above, $\epsilon$ is bare well depth, $\sigma$ is the
    characteristic length scale of the Lennard-Jones potential, and $r_c$ is the cutoff distance at which the bare
    potential is truncated. If using a finite cutoff distance $r_c$, we recommend $r_c \ge 2.5 \sigma$.
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
        """
        super().__init__(prefactor)
        self._potential_prefactor_12 = 4.0 * well_depth * characteristic_length ** 12
        self._potential_prefactor_6 = 4.0 * well_depth * characteristic_length ** 6
        self._gradient_prefactor_12 = 12.0 * self._potential_prefactor_12
        self._gradient_prefactor_6 = 6.0 * self._potential_prefactor_6
        self._characteristic_length = characteristic_length
        if cutoff_length == float('inf'):
            self._cutoff_length = None
            self._bare_potential_at_cut_off = 0.0
        else:
            self._cutoff_length = cutoff_length
            self._bare_potential_at_cut_off = (self._potential_prefactor_12 * cutoff_length ** (- 12.0) -
                                               self._potential_prefactor_6 * cutoff_length ** (- 6.0))

        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential.
        """
        return sum([self._get_bare_two_particle_potential(positions[i], positions[j])
                    for i in range(number_of_particles) for j in range(i + 1, number_of_particles)])

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

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
        gradient = np.zeros((number_of_particles, 3))
        for i in range(number_of_particles):
            for j in range(i + 1, number_of_particles):
                two_particle_gradient = self._get_two_particle_gradient(positions[i], positions[j])
                gradient[i] += two_particle_gradient
                gradient[j] -= two_particle_gradient
        return gradient

    def _get_bare_two_particle_potential(self, position_one, position_two):
        """
        Returns the bare Lennard-Jones potential for two particles.

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
            The bare two_particle Lennard-Jones potential.
        """
        separation_distance = np.linalg.norm(get_shortest_vectors_on_torus(position_one - position_two))
        if self._cutoff_length is None or separation_distance <= self._cutoff_length:
            return (self._potential_prefactor_12 * separation_distance ** (- 12.0) -
                    self._potential_prefactor_6 * separation_distance ** (- 6.0) - self._bare_potential_at_cut_off)
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
        if self._cutoff_length is None or separation_distance <= self._cutoff_length:
            return - separation_vector * (self._gradient_prefactor_12 * separation_distance ** (- 14.0) -
                                          self._gradient_prefactor_6 * separation_distance ** (- 8.0))
        return 0.0
