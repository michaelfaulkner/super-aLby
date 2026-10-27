"""Module for the OneDimLennardJonesPotential class."""
import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base.vectors import get_shortest_vectors_on_torus
from base.exceptions import ConfigurationError
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import (dimensionality_of_particle_space, size_of_particle_space, number_of_particles)


class OneDimLennardJonesPotential(EuclideanSubspacePotential):
    r"""
    Class for one-dimensional Lennard-Jones potential

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
    two-particle potential is truncated. If using a cutoff distance $r_c$, we recommend $r_c \ge 2.5 \sigma$.
    """

    def __init__(self, characteristic_length: float = 1.0, well_depth: float = 1.0, prefactor: float = 1.0,
                 factor_field_prefactor: float = 0.0, use_cell_horizon: bool = False) -> None:
        """
        The constructor of the OneDimLennardJonesPotentials class.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
        prefactor : float, optional
            The prefactor k of the potential.
        factor_field_prefactor : float
            Prefactor of the factor field contribution to the potential.
        use_cell_horizon : bool
            Determines whether to use cell horizon method.

        Raises
        ------
        base.exceptions.ConfigurationError
            If model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        base.exceptions.ConfigurationError
            If characteristic_length is less than 0.5.
        """
        super().__init__(prefactor)
        # check_model_settings_of_soft_matter_potential(size_of_particle_space, dimensionality_of_particle_space,
        #                                              range_of_initial_particle_positions, self.__class__.__name__)
        if characteristic_length < 0.5:
            raise ConfigurationError(f"Give a value not less than 0.5 for characteristic_length in "
                                     f"{self.__class__.__name__}.")
        self._well_depth = well_depth
        self._characteristic_length = characteristic_length
        self._potential_12_constant = 4.0 * prefactor * well_depth * characteristic_length ** 12
        self._potential_6_constant = 4.0 * prefactor * well_depth * characteristic_length ** 6
        self._gradient_12_constant = 12.0 * self._potential_12_constant
        self._gradient_6_constant = 6.0 * self._potential_6_constant
        self._equilibrium_separation = (2.0 ** (1.0 / 6.0)) * characteristic_length
        self._grad_max_separation = (26.0 / 7.0) ** (1.0 / 6.0) * characteristic_length
        self._max_grad = 504 * self._well_depth / (169 * self._characteristic_length * (26.0 / 7.0) ** (1.0 / 6.0))
        self._energy_midpoint = ((self._potential_12_constant / ((size_of_particle_space[0] / 2.0) ** 12)) -
                                 self._potential_6_constant / ((size_of_particle_space[0] / 2.0) ** 6))
        self._use_cell_horizon = use_cell_horizon
        self._factor_field_prefactor = factor_field_prefactor

    def get_initial_positions(self):
        """
        Returns the initial positions array.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g. two particles
            (confined to one-dimensional space) at positions 0.0 and 1.0 is represented by [[0.0] [1.0]]; three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
        return np.sort(get_initial_positions_of_smooth_potential(self.__class__.__name__), axis=0)

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
        for i in range(number_of_particles):
            for j in range(i + 1, number_of_particles):
                separation_vector = get_shortest_vectors_on_torus(positions[i] - positions[j])
                separation_distance = float(np.linalg.norm(separation_vector))
                potential += self._get_non_zero_two_particle_potential(separation_distance)
        return potential

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
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
        candidate_potential = self._get_single_particle_potential(active_particle_index, candidate_position, positions)
        current_potential = self._get_single_particle_potential(active_particle_index, positions[active_particle_index],
                                                                positions)
        return candidate_potential - current_potential

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

    @staticmethod
    def get_random_event_chain_velocity():
        """
        Uniformly samples a direction of motion for the active particle from chosen velocity distribution.

        Returns
        ----------
        random_event_chain_velocity : int or numpy.ndarray
            The uniformly sampled event-chain velocity of the active particle.  If the state space of each particle is
            a subset of the real line, the method should output an integer; otherwise it should output a one-dimensional
            numpy array (of integers) of length dimensionality_of_particle_space, where the nth component represents the
            velocity of the active particle along the nth Cartesian direction.
        """
        return 1

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
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
            The active-particle direction of motion.

        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        veto_index : int
            The particle index responsible for the event.
        hop_displacement : numpy.ndarray
            Net displacement through state space from active to vetoing particle.
        """
        active_particle_position = positions[active_particle_index][0]
        final_distance_to_next_factor_event = np.inf
        vetoing_index = active_particle_index
        hop_displacement = 0.0
        if self._use_cell_horizon:
            final_max_rate = None

            pos_neighbour_index, neg_neighbour_index = ((active_particle_index + 1) % number_of_particles,
                                                        (active_particle_index - 1) % number_of_particles)
            front_neighbour_index = pos_neighbour_index if movement_direction > 0 else neg_neighbour_index
            back_neighbour_index = neg_neighbour_index if movement_direction > 0 else pos_neighbour_index
            front_neighbour_separation = ((positions[front_neighbour_index][0] - active_particle_position)
                                          * movement_direction)
            front_neighbour_separation = front_neighbour_separation % size_of_particle_space[0]
            energy_threshold = -temperature * np.log(np.random.uniform(0.0, 1.0))
            initial_separation = min(front_neighbour_separation, self._equilibrium_separation)
            energy_start = self._get_non_zero_two_particle_potential(initial_separation)
            energy_target = energy_start + energy_threshold
            z = 0.5 * (1.0 + np.sqrt(1.0 + energy_target / self._well_depth))
            target_separation = self._characteristic_length / (z ** (1.0 / 6.0))
            front_distance_to_next_factor_event = front_neighbour_separation - target_separation
            final_distance_to_next_factor_event = front_distance_to_next_factor_event
            vetoing_index = front_neighbour_index

            for particle_index in range(number_of_particles):
                if particle_index == active_particle_index or particle_index == front_neighbour_index:
                    continue
                particle_separation = positions[particle_index][0] - active_particle_position
                particle_separation -= size_of_particle_space[0] * np.round(particle_separation /
                                                                            size_of_particle_space[0])
                abs_separation = float(abs(particle_separation))
                energy_threshold = -temperature * np.log(np.random.uniform(0.0, 1.0))
                if particle_separation > 0.0:
                    max_rate = None
                    initial_separation = min(abs_separation, self._equilibrium_separation)
                    energy_start = self._get_non_zero_two_particle_potential(initial_separation)
                    energy_target = energy_start + energy_threshold
                    z = 0.5 * (1.0 + np.sqrt(1.0 + energy_target / self._well_depth))
                    target_separation = self._characteristic_length / (z ** (1.0 / 6.0))
                    candidate_distance_to_next_factor_event = abs_separation - target_separation
                else:
                    if abs_separation < self._grad_max_separation:
                        max_grad = self._max_grad
                    else:
                        if abs_separation + front_neighbour_separation < size_of_particle_space[0] - abs_separation:
                            max_grad = -self._get_non_zero_two_particle_gradient(particle_separation)
                        elif (abs_separation + front_neighbour_separation < size_of_particle_space[0] -
                              self._grad_max_separation):
                            max_grad = -self._get_non_zero_two_particle_gradient(size_of_particle_space[0] +
                                                                                 particle_separation)
                        else:
                            max_grad = self._max_grad
                    if particle_index == back_neighbour_index:
                        max_rate = max(0.0, max_grad + self._factor_field_prefactor)
                    else:
                        max_rate = max(0.0, max_grad)
                    candidate_distance_to_next_factor_event = np.inf if max_rate < 1.0e-10 else (
                            -np.log(np.random.uniform(0.0, 1.0)) * temperature / max_rate)

                if candidate_distance_to_next_factor_event < final_distance_to_next_factor_event:
                    final_distance_to_next_factor_event = candidate_distance_to_next_factor_event
                    final_max_rate = max_rate
                    vetoing_index = particle_index

            shortest_distance_to_next_factor_event = final_distance_to_next_factor_event

            if final_max_rate is None:
                active_particle_position += shortest_distance_to_next_factor_event * movement_direction
                particle_separation = positions[vetoing_index][0] - active_particle_position
                active_particle_position -= shortest_distance_to_next_factor_event * movement_direction
                particle_separation -= size_of_particle_space[0] * np.round(
                    particle_separation / size_of_particle_space[0])
                max_grad = -self._get_non_zero_two_particle_gradient(particle_separation)
                max_rate = max(0.0, max_grad)
                if vetoing_index == front_neighbour_index:
                    actual_rate = max(0.0, max_grad - self._factor_field_prefactor)
                else:
                    actual_rate = max(0.0, max_grad)
                if actual_rate > 0.0 and np.random.uniform(0.0, 1.0) < actual_rate / max_rate:
                    hop_displacement = (positions[vetoing_index][0] - active_particle_position) * movement_direction
                else:
                    vetoing_index, hop_displacement = (active_particle_index,
                                                       shortest_distance_to_next_factor_event * movement_direction)
            else:
                active_particle_position += shortest_distance_to_next_factor_event * movement_direction
                particle_separation = positions[vetoing_index][0] - active_particle_position
                active_particle_position -= shortest_distance_to_next_factor_event * movement_direction
                particle_separation -= size_of_particle_space[0] * np.round(
                    particle_separation / size_of_particle_space[0])
                if vetoing_index == back_neighbour_index:
                    actual_rate = max(0.0, -self._get_non_zero_two_particle_gradient(particle_separation) +
                                      self._factor_field_prefactor)
                else:
                    actual_rate = max(0.0, -self._get_non_zero_two_particle_gradient(particle_separation))

                if actual_rate > 0.0 and np.random.uniform(0.0, 1.0) < actual_rate / final_max_rate:
                    hop_displacement = (positions[vetoing_index][0] - active_particle_position) * movement_direction
                else:
                    vetoing_index, hop_displacement = (active_particle_index,
                                                       shortest_distance_to_next_factor_event * movement_direction)

        else:
            for particle_index in range(number_of_particles):
                if particle_index == active_particle_index:
                    continue
                particle_position = positions[particle_index][0]
                particle_separation = particle_position - active_particle_position
                particle_separation -= size_of_particle_space[0] * np.round(particle_separation /
                                                                            size_of_particle_space[0])
                abs_separation = float(abs(particle_separation))
                energy_threshold = -temperature * np.log(np.random.uniform(0.0, 1.0))
                if particle_separation > 0.0:
                    initial_separation = min(abs_separation, self._equilibrium_separation)
                    energy_start = (self._potential_12_constant / (initial_separation ** 12) -
                                    self._potential_6_constant / (initial_separation ** 6))
                    energy_target = energy_start + energy_threshold
                    z = 0.5 * (1.0 + np.sqrt(1.0 + energy_target / self._well_depth))
                    target_separation = self._characteristic_length / (z ** (1.0 / 6.0))
                    candidate_distance_to_next_factor_event = abs_separation - target_separation
                else:
                    initial_separation = max(abs_separation, self._equilibrium_separation)
                    energy_start = (self._potential_12_constant / (initial_separation ** 12) -
                                    self._potential_6_constant / (initial_separation ** 6))
                    energy_target = energy_start + energy_threshold
                    if energy_target > self._energy_midpoint:
                        distance_to_midpoint = size_of_particle_space[0] / 2.0 - abs_separation
                        remaining_energy_threshold = energy_threshold - (self._energy_midpoint - energy_start)
                        z = 0.5 * (1.0 + np.sqrt(remaining_energy_threshold / self._well_depth))
                        extra_separation = self._characteristic_length / (z ** (1.0 / 6.0))
                        candidate_distance_to_next_factor_event = (distance_to_midpoint +
                                                                   (size_of_particle_space[0] / 2.0 - extra_separation))
                    else:
                        z = 0.5 * (1.0 - np.sqrt(1.0 + energy_target / self._well_depth))
                        target_separation = self._characteristic_length / (z ** (1.0 / 6.0))
                        candidate_distance_to_next_factor_event = target_separation - abs_separation

                if candidate_distance_to_next_factor_event < final_distance_to_next_factor_event:
                    final_distance_to_next_factor_event = candidate_distance_to_next_factor_event
                    vetoing_index = particle_index
                    hop_displacement = positions[vetoing_index][0] - active_particle_position

        return final_distance_to_next_factor_event, vetoing_index, hop_displacement

    def _get_single_particle_potential(self, active_particle_index, particle_position, positions):
        """
        Calculates the interaction potential between a single particle and all other particles.
        """
        potential = 0.0
        for particle_index in range(number_of_particles):
            if particle_index == active_particle_index:
                continue
            separation_vector = get_shortest_vectors_on_torus(positions[particle_index] - particle_position)
            separation_distance = float(np.linalg.norm(separation_vector))
            potential += self._get_non_zero_two_particle_potential(separation_distance)
        return potential

    def _get_non_zero_two_particle_potential(self, separation_distance):
        """
        Returns the Lennard-Jones potential for two particles whose shortest separation distance is not greater than
        self._cutoff_length.

        Parameters
        ----------
        separation_distance : float
            The shortest separation distance between the two particles.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential (for all cases for which it is non-zero).
        """
        return (self._potential_12_constant * separation_distance ** (- 12.0) -
                self._potential_6_constant * separation_distance ** (- 6.0))

    def _get_non_zero_two_particle_gradient(self, separation_vector):
        """
        Returns the Lennard-Jones potential for two particles whose shortest separation distance is not greater than
        self._cutoff_length.

        Parameters
        ----------
        separation_vector : numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the gradient of the shortest separation vector between the two
            particles.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential (for all cases for which it is non-zero).
        """
        separation_distance = abs(separation_vector)
        return - separation_vector * (self._gradient_12_constant * separation_distance ** (- 14.0) -
                                      self._gradient_6_constant * separation_distance ** (- 8.0))

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of each corresponding particle.
        active_particle_index : int
            The active particle index
        movement_direction : int
            The active-particle direction of motion.
        veto_index : int
            The particle index responsible for the event.

        Returns
        -------
        active_particle_index: int
            The index of the next active particle.
        movement_direction : int
            The next active-particle direction of motion.
        """
        return veto_index, movement_direction

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """Updates the position of the active particle following an event."""
        positions[active_particle_index] = positions[active_particle_index] + movement_direction * displacement_distance

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """Propose candidate via teleportation portal kernel."""
        raise SystemError(f"The get_portal_candidate method of {self.__class__.__name__} has not been written.")

