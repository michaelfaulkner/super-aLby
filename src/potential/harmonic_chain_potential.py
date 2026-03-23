"""Module for the HarmonicChainPotential class."""
import numpy as np
from .harmonic_chain_potentials import HarmonicChainPotentials
from helper_methods import get_initial_positions_of_smooth_potential


class HarmonicChainPotential(HarmonicChainPotentials):
    """
    This class implements the harmonic-chain potential U = prefactor * sum((x[i] - x[i-1]) ** 2) / 2 with periodic
        boundary conditions such that x[N] = x[0] + L (with each x[i] defined on the entire real line).

    This is equivalent to a model of real-valued springs x_tilde[i] with potential
        U = prefactor * sum(x_tilde[i] ** 2) / 2 and subject to the constraint sum(x_tilde[i]) = L.
    """

    def __init__(self, prefactor: float = 1.0, equilibrium_length: float = 0.0, use_cell_horizon: bool = False,
                 cell_horizon: float = 1.0):
        """
        The constructor of the HarmonicChainPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        equilibrium_length : float
            The separation of particles associated with the minimum of the potential.
        use_cell_horizon : bool
            Determines whether to use cell horizon method.
        cell_horizon : float
            Horizon over which to measure maximum potential gradient.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the dimensionality of size_of_particle_space is greater than 1.
        """
        super().__init__(prefactor=prefactor, equilibrium_length=equilibrium_length)
        self._use_cell_horizon = use_cell_horizon
        if use_cell_horizon:
            self._cell_horizon = cell_horizon

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
        active_particle_position = positions[active_particle_index]
        neg_neighbour_index, pos_neighbour_index = self._get_neighbours(active_particle_index)
        neg_neighbour_position, pos_neighbour_position = self._get_neighbour_positions(positions, active_particle_index,
                                                                                       neg_neighbour_index,
                                                                                       pos_neighbour_index)

        if self._use_cell_horizon:
            active_particle_position += self._cell_horizon * movement_direction
            max_pos_grad, max_neg_grad = self._get_single_particle_gradient(positions, active_particle_index)
            active_particle_position -= self._cell_horizon * movement_direction
            if movement_direction > 0.0:
                max_pos_grad += self._equilibrium_length
                max_neg_grad -= self._equilibrium_length
            else:
                max_pos_grad -= self._equilibrium_length
                max_neg_grad += self._equilibrium_length

            max_rate_pos = np.maximum(0.0, movement_direction * max_pos_grad)
            max_rate_neg = np.maximum(0.0, movement_direction * max_neg_grad)

            candidate_distance_to_next_factor_event_pos = np.inf if max_rate_pos < 1e-12 else (
                    -np.log(np.random.uniform(0.0, 1.0)) / max_rate_pos)
            candidate_distance_to_next_factor_event_neg = np.inf if max_rate_neg < 1e-12 else (
                    -np.log(np.random.uniform(0.0, 1.0)) / max_rate_neg)

            if max_rate_pos < 1e-12 and max_rate_neg < 1e-12:
                return self._cell_horizon, active_particle_index, self._cell_horizon

            candidate_distance_to_next_factor_event, vetoing_index, max_rate = (
                min((candidate_distance_to_next_factor_event_pos, pos_neighbour_index, max_rate_pos),
                    (candidate_distance_to_next_factor_event_neg, neg_neighbour_index, max_rate_neg)))

            if candidate_distance_to_next_factor_event > self._cell_horizon:
                shortest_distance_to_next_factor_event, vetoing_index, hop_displacement = (
                    self._cell_horizon, active_particle_index, self._cell_horizon)

            else:
                active_particle_position += candidate_distance_to_next_factor_event * movement_direction
                actual_pos_grad, actual_neg_grad = self._get_single_particle_gradient(positions, active_particle_index)
                active_particle_position -= candidate_distance_to_next_factor_event * movement_direction
                if movement_direction > 0.0:
                    actual_pos_grad += self._equilibrium_length
                    actual_neg_grad -= self._equilibrium_length
                else:
                    actual_pos_grad -= self._equilibrium_length
                    actual_neg_grad += self._equilibrium_length

                if vetoing_index == pos_neighbour_index:
                    actual_rate = np.maximum(0.0, movement_direction * actual_pos_grad)
                else:
                    actual_rate = np.maximum(0.0, movement_direction * actual_neg_grad)

                if np.random.uniform(0.0, 1.0) < actual_rate / max_rate:
                    shortest_distance_to_next_factor_event = candidate_distance_to_next_factor_event
                    hop_displacement = positions[vetoing_index][0] - active_particle_position[0]
                else:
                    shortest_distance_to_next_factor_event, vetoing_index, hop_displacement = (
                        candidate_distance_to_next_factor_event, active_particle_index,
                        candidate_distance_to_next_factor_event)

        else:
            neg_dist_to_eq, pos_dist_to_eq = (neg_neighbour_position + self._equilibrium_length -
                                              active_particle_position[0],
                                              pos_neighbour_position - self._equilibrium_length -
                                              active_particle_position[0])
            neg_dist_to_eq *= movement_direction
            pos_dist_to_eq *= movement_direction
            rand_neg, rand_pos = (- temperature * np.log(np.random.uniform(0.0, 1.0)) / self._potential_constant,
                                  - temperature * np.log(np.random.uniform(0.0, 1.0)) / self._potential_constant)
            distance_to_next_neg_factor_event = (neg_dist_to_eq + rand_neg ** 0.5 if neg_dist_to_eq > 0
                                                 else neg_dist_to_eq + (rand_neg + (-neg_dist_to_eq) ** 2) ** 0.5)
            distance_to_next_pos_factor_event = (pos_dist_to_eq + rand_pos ** 0.5 if pos_dist_to_eq > 0
                                                 else pos_dist_to_eq + (rand_pos + (-pos_dist_to_eq) ** 2) ** 0.5)

            shortest_distance_to_next_factor_event, vetoing_index, hop_displacement = (
                min((distance_to_next_neg_factor_event, neg_neighbour_index, neg_neighbour_position
                     - active_particle_position[0]),
                    (distance_to_next_pos_factor_event, pos_neighbour_index, pos_neighbour_position
                     - active_particle_position[0])))

        return shortest_distance_to_next_factor_event, vetoing_index, hop_displacement

    def _get_single_particle_gradient(self, positions, single_particle_index):
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
        tuple
            Value of the gradient of the potential for the single particle.
        """
        single_particle_position = positions[single_particle_index][0]
        neg_neighbour_index, pos_neighbour_index = self._get_neighbours(single_particle_index)
        neg_neighbour_position, pos_neighbour_position = self._get_neighbour_positions(positions, single_particle_index,
                                                                                       neg_neighbour_index,
                                                                                       pos_neighbour_index)

        pos_gradient_value = (single_particle_position - pos_neighbour_position)
        neg_gradient_value = (single_particle_position - neg_neighbour_position)

        return 2.0 * self._potential_constant * pos_gradient_value, 2.0 * self._potential_constant * neg_gradient_value
