"""Module for the LennardJonesPotentialWithoutLinkedLists class."""
from .lennard_jones_potentials_with_cutoff import LennardJonesPotentialsWithCutoff
from base.logging import log_init_arguments
from model_settings import dimensionality_of_particle_space, number_of_particles
import logging
import numpy as np


class LennardJonesPotentialWithoutLinkedLists(LennardJonesPotentialsWithCutoff):
    r"""
    Without linked-lists, this class implements the Lennard-Jones potential

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
        The constructor of the LennardJonesPotentialWithoutLinkedLists class.

        NOTE THAT:
            i) The Metropolis algorithm does not seem to converge for two Lennard-Jones particles for which the
            value of each component of size_of_particle_space is greater than twice the value of characteristic_length
            -- perhaps due to too much time spent with particles independently drifting around.
            ii) Newtonian- and relativistic-dynamics-based algorithms do not seem to converge for two Lennard-Jones
            particles for which the value of each component of size_of_particle_space is less than twice the value of
            characteristic_length -- perhaps due to discontinuities in the potential gradients.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
        cutoff_length : float, optional
            The cutoff distance at which the bare potential is truncated.
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
        base.exceptions.ConfigurationError
            If use_linked_lists is True and dimensionality_of_particle_space does not equal 3.
        base.exceptions.ConfigurationError
            If use_linked_lists is True and cutoff_length is inf.
        """
        super().__init__(characteristic_length, well_depth, cutoff_length, prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           characteristic_length=characteristic_length, well_depth=well_depth,
                           cutoff_length=cutoff_length, prefactor=prefactor)

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
        return sum([self._get_two_particle_potential(positions[i], positions[j]) for i in range(number_of_particles)
                    for j in range(i + 1, number_of_particles)])

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
        gradient = np.zeros((number_of_particles, dimensionality_of_particle_space))
        for i in range(number_of_particles):
            for j in range(i + 1, number_of_particles):
                two_particle_gradient = self._get_two_particle_gradient(positions[i], positions[j])
                gradient[i] += two_particle_gradient
                gradient[j] -= two_particle_gradient
        return gradient

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        # TODO write the code for this method!
        """
        Returns the potential difference resulting from moving the single active particle to candidate_position.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the proposed position of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        raise SystemError(f"The get_potential_difference method of {self.__class__.__name__} has not been written.")
