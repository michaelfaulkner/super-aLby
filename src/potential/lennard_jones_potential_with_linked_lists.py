"""Module for the LennardJonesPotentialWithLinkedLists class."""
from .lennard_jones_potentials_with_cutoff import LennardJonesPotentialsWithCutoff
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from linked_lists.three_dimensional_linked_lists import ThreeDimensionalLinkedLists
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
import itertools
import logging
import numpy as np


class LennardJonesPotentialWithLinkedLists(LennardJonesPotentialsWithCutoff):
    r"""
    With linked-lists, this class implements the Lennard-Jones potential

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
        The constructor of the LennardJonesPotentialWithLinkedLists class.

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
        """
        super().__init__(characteristic_length, well_depth, cutoff_length, prefactor)
        if dimensionality_of_particle_space != 3:
            raise ConfigurationError(f"For size_of_particle_space, give a one-dimensional list of length 3 (and "
                                     f"composed of floats) in {self.__class__.__name__}. This is because the "
                                     f"dimensionality of particle space must be 3 when using the linked-lists "
                                     f"algorithm in {self.__class__.__name__}.")
        number_of_cells_in_each_direction = np.int_(size_of_particle_space / self._cutoff_length)
        self._linked_lists = ThreeDimensionalLinkedLists(number_of_cells_in_each_direction)
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
        potential = 0.0
        self._linked_lists.reset_linked_lists(positions)
        for cell_one in itertools.product(range(self._linked_lists.number_of_cells_in_each_direction[0]),
                                          range(self._linked_lists.number_of_cells_in_each_direction[1]),
                                          range(self._linked_lists.number_of_cells_in_each_direction[2])):
            cell_one_index = self._linked_lists.get_cell_index(cell_one)
            for cell_two in itertools.product(range(cell_one[0] - 1, cell_one[0] + 1),
                                              range(cell_one[1] - 1, cell_one[1] + 1),
                                              range(cell_one[2] - 1, cell_one[2] + 1)):
                cell_two_index = self._linked_lists.get_cell_index([
                    int((element + self._linked_lists.number_of_cells_in_each_direction[index] / 2) %
                        self._linked_lists.number_of_cells_in_each_direction[index] -
                        self._linked_lists.number_of_cells_in_each_direction[index] / 2)
                    for index, element in enumerate(cell_two)])
                particle_one_index = self._linked_lists.leading_particle_of_cell[cell_one_index]
                while particle_one_index is not None:
                    particle_two_index = self._linked_lists.leading_particle_of_cell[cell_two_index]
                    while particle_two_index is not None:
                        if particle_one_index > particle_two_index:
                            potential += self._get_two_particle_potential(positions[particle_one_index],
                                                                          positions[particle_two_index])
                        particle_two_index = self._linked_lists.next_particle_in_same_cell[particle_two_index]
                    particle_one_index = self._linked_lists.next_particle_in_same_cell[particle_one_index]
        return potential

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
        self._linked_lists.reset_linked_lists(positions)
        for cell_one in itertools.product(range(self._linked_lists.number_of_cells_in_each_direction[0]),
                                          range(self._linked_lists.number_of_cells_in_each_direction[1]),
                                          range(self._linked_lists.number_of_cells_in_each_direction[2])):
            cell_one_index = self._linked_lists.get_cell_index(cell_one)
            for cell_two in itertools.product(range(cell_one[0] - 1, cell_one[0] + 1),
                                              range(cell_one[1] - 1, cell_one[1] + 1),
                                              range(cell_one[2] - 1, cell_one[2] + 1)):
                cell_two_index = self._linked_lists.get_cell_index([
                    int((element + self._linked_lists.number_of_cells_in_each_direction[index] / 2) %
                        self._linked_lists.number_of_cells_in_each_direction[index] -
                        self._linked_lists.number_of_cells_in_each_direction[index] / 2)
                    for index, element in enumerate(cell_two)])
                particle_one_index = self._linked_lists.leading_particle_of_cell[cell_one_index]
                while particle_one_index is not None:
                    particle_two_index = self._linked_lists.leading_particle_of_cell[cell_two_index]
                    while particle_two_index is not None:
                        if particle_one_index > particle_two_index:
                            two_particle_gradient = self._get_two_particle_gradient(positions[particle_one_index],
                                                                                    positions[particle_two_index])
                            gradient[particle_one_index] += two_particle_gradient
                            gradient[particle_two_index] -= two_particle_gradient
                        particle_two_index = self._linked_lists.next_particle_in_same_cell[particle_two_index]
                    particle_one_index = self._linked_lists.next_particle_in_same_cell[particle_one_index]
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

    @staticmethod
    def get_random_event_chain_velocity():
        """Uniformly samples a direction of motion for the active particle from chosen velocity distribution"""
        raise SystemError(f"The get_random_event_chain_velocity method has not been written.")

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature,
                                                  movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        temperature : float
            The sampling temperature.
        movement_direction : int
            The direction of movement of the active particle.
        
        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        vetoing_particle_index : int
            The index of the particle that triggers the event.
        """
        raise SystemError(f"The get_distance_to_next_event_and_veto_index method of {self.__class__.__name__} has not "
                          f"been written.  Functionality of ECMC for {self.__class__.__name__} is not yet provided.")
    
    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction for the next active particle in the markov chain for ECMC.
        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        movement_direction : int
            The direction of movement of the active particle.
        veto_index : int
            The particle index responsible for the event. 
        """
        raise SystemError(f"The choose_next_active_particle method of {self.__class__.__name__} has not been written.")

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """ Updates position of the active particle following an event."""
        raise SystemError(f"The update_position method has not been written.")
