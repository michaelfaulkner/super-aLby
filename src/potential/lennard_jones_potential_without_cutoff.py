"""Module for the LennardJonesPotentialWithoutCutoff class."""
from .lennard_jones_potentials import LennardJonesPotentials
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import dimensionality_of_particle_space, number_of_particles
import logging
import numpy as np


class LennardJonesPotentialWithoutCutoff(LennardJonesPotentials):
    r"""
    This class implements the Lennard-Jones potential (without a cutoff distance on the length scale of the interaction
    potential)

        $ U = k * \sum_{i > j} U_{{\rm LJ}, ij} $ ,

    where

        $ U_{{\rm LJ}, ij} = 4 \epsilon \left[\left(\frac{\sigma}{r_{ij}}\right)^{12} -
            \left(\frac{\sigma}{r_{ij}}\right)^6\right]$

    is the two-particle Lennard-Jones potential. In the above, $\epsilon$ is the well depth, $\sigma$ is the
    characteristic length scale of the Lennard-Jones potential.
    """

    def __init__(self, characteristic_length: float = 1.0, well_depth: float = 1.0, prefactor: float = 1.0) -> None:
        """
        The constructor of the LennardJonesPotentialWithoutCutoff class.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
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
            If characteristic_length is less than 0.5.
        """
        super().__init__(characteristic_length, well_depth, prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           characteristic_length=characteristic_length, well_depth=well_depth, prefactor=prefactor)

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
        return sum([self._get_non_zero_two_particle_potential(np.linalg.norm(get_shortest_vectors_on_torus(
            positions[i] - positions[j]))) for i in range(number_of_particles) for j in
            range(i + 1, number_of_particles)])

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
                separation_vector = get_shortest_vectors_on_torus(positions[i] - positions[j])
                two_particle_gradient = self._get_non_zero_two_particle_gradient(separation_vector,
                                                                                 np.linalg.norm(separation_vector))
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
