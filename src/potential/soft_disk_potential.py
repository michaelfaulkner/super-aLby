"""Module for the SoftDiskPotential class."""
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base.exceptions import ConfigurationError
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import size_of_particle_space, number_of_particles
import numpy as np


class SoftDiskPotential(EuclideanSubspacePotential):
    """
    This class implements the soft disk potential U = epsilon * (sigma / r) ** power.
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, power: float = 12.0, cutoff: float = None,
                 factor_field_prefactor: float = 0.0, use_cell_horizon: bool = False):
        """
        The constructor of the SoftDiskPotential class.

        Parameters
        ----------
        epsilon : float
            The energy scale of the potential.
        sigma : float
            The distance scale of the potential.
        power : float
            The steepness of the repulsive potential.
        cutoff : float
            The interaction cutoff distance in units of sigma.
        factor_field_prefactor : float
            Prefactor of the factor field contribution to the potential.
        use_cell_horizon : bool
            Determines whether to use cell horizon method.

        Raises
        ------
        base.exceptions.ConfigurationError
            If dimensionality of size_of_particle_space is greater than 1.
        """
        super().__init__(prefactor=epsilon)
        self._epsilon = epsilon
        self._sigma = sigma
        self._power = power
        self._factor_field_prefactor = factor_field_prefactor
        self._use_cell_horizon = use_cell_horizon
        self._cutoff = size_of_particle_space if cutoff is None else cutoff
        if len(size_of_particle_space) > 1:
            raise ConfigurationError(f'{self.__class__.__name__} only supports 1D space. Provided: '
                                     f'{len(size_of_particle_space)}')

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
        positions = np.sort(get_initial_positions_of_smooth_potential(self.__class__.__name__), axis=0)
        return positions

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
        potential : float
            The potential.
        """
        potential = 0.0
        for i in range(number_of_particles):
            for j in range(i + 1, number_of_particles):
                particle_separation = positions[j] - positions[i]
                particle_separation -= size_of_particle_space * np.round(particle_separation / size_of_particle_space)
                particle_separation = float(np.abs(particle_separation))
                if 0.0 < particle_separation < self._cutoff:
                    potential += self._epsilon * (self._sigma / particle_separation) ** self._power
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
        gradient = np.zeros_like(positions)
        for particle_index in range(number_of_particles):
            gradient[particle_index] = self.get_single_particle_gradient(positions, particle_index)
        return gradient

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
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        candidate_potential = self._get_single_particle_potential(active_particle_index, candidate_position, positions)
        current_potential = self._get_single_particle_potential(active_particle_index, positions[active_particle_index],
                                                                positions)
        return candidate_potential - current_potential

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
            is a float and represents the position of each particle.
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
        vetoing_index : int
            The index of the particle that triggers the event.
        hop_displacement : numpy.ndarray
            Net displacement through state space from active to vetoing particle.
        """
        active_particle_position = positions[active_particle_index][0]
        final_candidate_distance_to_next_factor_event = np.inf
        vetoing_index = active_particle_index

        if self._use_cell_horizon:
            final_max_rate = None

            pos_neighbour_index, neg_neighbour_index = ((active_particle_index + 1) % number_of_particles,
                                                        (active_particle_index - 1) % number_of_particles)
            front_neighbour_index = pos_neighbour_index if movement_direction > 0 else neg_neighbour_index
            back_neighbour_index = neg_neighbour_index if movement_direction > 0 else pos_neighbour_index
            front_neighbour_separation = ((positions[front_neighbour_index][0] - active_particle_position)
                                          * movement_direction)
            front_neighbour_separation = front_neighbour_separation % size_of_particle_space[0]

            alpha = (-np.log(np.random.uniform(0.0, 1.0)) * temperature /
                     (self._epsilon * self._sigma ** self._power) + 1 / front_neighbour_separation ** self._power)
            front_distance_to_next_factor_event = front_neighbour_separation - alpha ** (-1 / self._power)
            final_candidate_distance_to_next_factor_event = front_distance_to_next_factor_event
            vetoing_index = front_neighbour_index

            for particle_index in range(number_of_particles):
                if particle_index == active_particle_index or particle_index == front_neighbour_index:
                    continue
                particle_separation = positions[particle_index][0] - active_particle_position
                particle_separation -= size_of_particle_space[0] * np.round(particle_separation /
                                                                            size_of_particle_space[0])
                abs_separation = np.abs(particle_separation)
                if particle_separation * movement_direction > 0.0:
                    max_rate = None
                    alpha = (-np.log(np.random.uniform(0.0, 1.0)) * temperature /
                             (self._epsilon * self._sigma ** self._power) + 1 / abs_separation ** self._power)
                    candidate_distance_to_next_factor_event = abs_separation - alpha ** (-1 / self._power)
                else:
                    max_separation = abs_separation + front_distance_to_next_factor_event
                    if particle_index == back_neighbour_index:
                        max_rate = max(0.0, self._get_potential_gradient(max_separation) + self._factor_field_prefactor)
                        candidate_distance_to_next_factor_event = np.inf if max_rate < 1.0e-10 else (
                                -np.log(np.random.uniform(0.0, 1.0)) * temperature / max_rate)
                    elif max_separation < size_of_particle_space[0] / 2.0:
                        continue
                    else:
                        max_rate = None
                        free_distance = size_of_particle_space[0] / 2 - abs_separation
                        abs_separation = size_of_particle_space[0] / 2
                        alpha = (-np.log(np.random.uniform(0.0, 1.0)) * temperature /
                                 (self._epsilon * self._sigma ** self._power) + 1 / abs_separation ** self._power)
                        candidate_distance_to_next_factor_event = abs_separation - alpha ** (-1 / self._power)
                        candidate_distance_to_next_factor_event += free_distance

                if candidate_distance_to_next_factor_event < final_candidate_distance_to_next_factor_event:
                    final_candidate_distance_to_next_factor_event = candidate_distance_to_next_factor_event
                    final_max_rate = max_rate
                    vetoing_index = particle_index

            shortest_distance_to_next_factor_event = final_candidate_distance_to_next_factor_event

            if np.isinf(shortest_distance_to_next_factor_event):
                neg_neighbour_position, pos_neighbour_position = (positions[neg_neighbour_index][0],
                                                                  positions[pos_neighbour_index][0])
                if active_particle_index == 0:
                    neg_neighbour_position -= size_of_particle_space[0]
                elif active_particle_index == number_of_particles - 1:
                    pos_neighbour_position += size_of_particle_space[0]
                midpoint = (neg_neighbour_position + pos_neighbour_position) / 2.0
                distance_to_midpoint = (midpoint - active_particle_position) * movement_direction
                if distance_to_midpoint <= 0.0:
                    distance_to_midpoint = size_of_particle_space[0] / 4.0
                shortest_distance_to_next_factor_event, vetoing_index, hop_displacement = (
                    distance_to_midpoint, active_particle_index, distance_to_midpoint * movement_direction)

            elif final_max_rate is None:
                active_particle_position += shortest_distance_to_next_factor_event * movement_direction
                particle_separation = positions[vetoing_index][0] - active_particle_position
                active_particle_position -= shortest_distance_to_next_factor_event * movement_direction
                particle_separation -= size_of_particle_space[0] * np.round(
                    particle_separation / size_of_particle_space[0])
                abs_separation = np.abs(particle_separation)
                max_grad = -self._get_potential_gradient(abs_separation)
                if vetoing_index == front_neighbour_index:
                    actual_grad = max_grad - self._factor_field_prefactor
                else:
                    actual_grad = max_grad
                max_rate = max(0.0, max_grad)
                actual_rate = max(0.0, actual_grad)
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
                abs_separation = np.abs(particle_separation)

                if vetoing_index == back_neighbour_index:
                    actual_rate = max(0.0, self._get_potential_gradient(abs_separation) + self._factor_field_prefactor)
                else:
                    actual_rate = max(0.0, self._get_potential_gradient(abs_separation))

                if actual_rate > 0.0 and np.random.uniform(0.0, 1.0) < actual_rate / final_max_rate:
                    hop_displacement = (positions[vetoing_index][0] - active_particle_position) * movement_direction
                else:
                    vetoing_index, hop_displacement = (active_particle_index,
                                                       shortest_distance_to_next_factor_event * movement_direction)

        else:
            for particle_index in range(number_of_particles):
                if particle_index == active_particle_index:
                    continue
                particle_separation = positions[particle_index][0] - active_particle_position
                particle_separation -= size_of_particle_space[0] * np.round(particle_separation / size_of_particle_space[0])
                abs_separation = np.abs(particle_separation)
                if particle_separation * movement_direction > 0.0:
                    alpha = (-np.log(np.random.uniform(0.0, 1.0)) * temperature /
                             (self._epsilon * self._sigma ** self._power) + 1 / abs_separation ** self._power)
                    candidate_distance_to_next_factor_event = abs_separation - alpha ** (-1 / self._power)
                else:
                    free_distance = size_of_particle_space[0] / 2 - abs_separation
                    abs_separation = size_of_particle_space[0] / 2
                    alpha = (-np.log(np.random.uniform(0.0, 1.0)) * temperature /
                             (self._epsilon * self._sigma ** self._power) + 1 / abs_separation ** self._power)
                    candidate_distance_to_next_factor_event = abs_separation - alpha ** (-1 / self._power)
                    candidate_distance_to_next_factor_event += free_distance
                if candidate_distance_to_next_factor_event < final_candidate_distance_to_next_factor_event:
                    final_candidate_distance_to_next_factor_event = candidate_distance_to_next_factor_event
                    vetoing_index = particle_index

            shortest_distance_to_next_factor_event, hop_displacement = (
                final_candidate_distance_to_next_factor_event, (positions[vetoing_index][0] - active_particle_position)
                * movement_direction)

        return shortest_distance_to_next_factor_event, vetoing_index, hop_displacement

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

    def _get_potential_gradient(self, abs_separation):
        """
        Calculates the potential gradient for a given separation.
        """
        if 0.0 < abs_separation < self._cutoff:
            return (-self._power * self._epsilon * (self._sigma ** self._power) /
                    (abs_separation ** (self._power + 1)))
        return 0.0

    def _get_single_particle_potential(self, active_particle_index, particle_position, positions):
        """
        Calculates the interaction potential between a single particle and all other particles
        within the cutoff distance.
        """
        potential = 0.0
        for particle_index in range(number_of_particles):
            if particle_index == active_particle_index:
                continue
            particle_separation = positions[particle_index][0] - particle_position
            particle_separation -= size_of_particle_space * np.round(particle_separation / size_of_particle_space[0])
            abs_separation = abs(particle_separation)
            if 0.0 < abs_separation < self._cutoff:
                potential += self._epsilon * (self._sigma / abs_separation) ** self._power
        return potential

    def get_single_particle_gradient(self, positions, single_particle_index):
        """
        Returns the gradient of the potential for a single particle position.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.
        single_particle_index : int
            Index of particle in positions array to evaluate the gradient of the potential with respect to.

        Returns
        -------
        float
            Value of the gradient of the potential for the single particle.
        """
        gradient = 0.0
        for particle_index in range(number_of_particles):
            if particle_index == single_particle_index:
                continue
            particle_separation = positions[particle_index][0] - positions[single_particle_index][0]
            particle_separation -= size_of_particle_space[0] * np.round(particle_separation / size_of_particle_space[0])
            abs_separation = float(np.abs(particle_separation))
            if 0.0 < abs_separation < self._cutoff:
                force = -self._get_potential_gradient(abs_separation)
                if particle_separation > 0.0:
                    gradient += force + self._factor_field_prefactor
                else:
                    gradient -= force + self._factor_field_prefactor
        return gradient

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """Updates the position of the active particle following an event."""
        positions[active_particle_index] = positions[active_particle_index] + movement_direction * displacement_distance
        if positions[active_particle_index] > 1e10:
            positions -= 1e10
        elif positions[active_particle_index] < -1e10:
            positions += 1e10

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """Propose candidate via teleportation portal kernel."""
        raise SystemError(f"The get_portal_candidate method is not valid for {self.__class__.__name__}.")
