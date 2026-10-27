"""Module for the UnfactorisedHarmonicChainPotential class."""
import numpy as np
from .harmonic_chain_potentials import HarmonicChainPotentials
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import number_of_particles


class UnfactorisedHarmonicChainPotential(HarmonicChainPotentials):
    """
    This class implements the unfactorised harmonic-chain potential U = prefactor * sum((x[i] - x[i-1]) ** 2) / 2 with
        periodic boundary conditions such that x[N] = x[0] + L (with each x[i] defined on the entire real line).

    This is equivalent to a model of real-valued springs x_tilde[i] with potential
        U = prefactor * sum(x_tilde[i] ** 2) / 2 and subject to the constraint sum(x_tilde[i]) = L.
    """

    def __init__(self, prefactor: float = 1.0, equilibrium_length: float = 0.0):
        """
        The constructor of the UnfactorisedHarmonicChainPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        equilibrium_length : float
            The separation of particles associated with the minimum of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the dimensionality of size_of_particle_space is greater than 1.
        """
        super().__init__(prefactor=prefactor, equilibrium_length=equilibrium_length)
        self._gradients = np.zeros(number_of_particles)

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
        for particle_index in range(number_of_particles):
            self._gradients[particle_index] = self._get_single_particle_gradient(positions, particle_index)
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
        midpoint = (neg_neighbour_position + pos_neighbour_position) / 2.0
        net_dist_to_eq = (midpoint - active_particle_position[0]) * movement_direction
        rand_net = -temperature * np.log(np.random.uniform(0.0, 1.0)) / (2.0 * self._potential_constant)
        if net_dist_to_eq > 0:
            distance_to_next_event = net_dist_to_eq + rand_net ** 0.5
        else:
            distance_to_next_event = net_dist_to_eq + (rand_net + (-net_dist_to_eq) ** 2) ** 0.5

        positions[active_particle_index] += distance_to_next_event * movement_direction
        self._gradients[active_particle_index] = self._get_single_particle_gradient(positions, active_particle_index)
        self._gradients[neg_neighbour_index] = self._get_single_particle_gradient(positions, neg_neighbour_index)
        self._gradients[pos_neighbour_index] = self._get_single_particle_gradient(positions, pos_neighbour_index)
        positions[active_particle_index] -= distance_to_next_event * movement_direction

        vetoing_index = self._get_vetoing_index()

        hop_displacement = positions[vetoing_index][0] - active_particle_position[0]

        return distance_to_next_event, vetoing_index, hop_displacement

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
        float
            Value of the gradient of the potential for the single particle.
        """
        single_particle_position = positions[single_particle_index][0]
        neg_neighbour_index, pos_neighbour_index = self._get_neighbours(single_particle_index)
        neg_neighbour_position, pos_neighbour_position = self._get_neighbour_positions(positions, single_particle_index,
                                                                                       neg_neighbour_index,
                                                                                       pos_neighbour_index)
        gradient_value = 2.0 * single_particle_position - neg_neighbour_position - pos_neighbour_position
        return 2.0 * self._potential_constant * gradient_value

    def _get_vetoing_index(self):
        rates = np.maximum(0, -self._gradients)
        total_rate = rates.sum()
        if total_rate < 1.0e-10:
            return np.random.choice(np.arange(len(self._gradients)))
        probs = rates / total_rate
        return np.random.choice(np.arange(len(self._gradients)), p=probs)
