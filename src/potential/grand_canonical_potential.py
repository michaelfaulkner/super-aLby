"""Module for the abstract GrandCanonicalPotential class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from .euclidean_subspace_potential import EuclideanSubspacePotential
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
import numpy as np


class GrandCanonicalPotential(EuclideanSubspacePotential, metaclass=ABCMeta):
    r"""
    Abstract class for pairwise potentials of particles on a ring whose number of particles may change, for use with
    EventChainMediator or MetropolisMediator together with a grand-canonical kernel.

    Unlike the canonical potentials, the number of particles is not fixed. It is always read from the positions array,
    which has shape (number_of_rows, 1).

    If self.ghost_coupling is not None, the last row of positions is the ghost particle and self.ghost_coupling is
    its coupling lambda in [-parked_length, 1].  Every ghost-real pair interacts as a real-real pair with its length
    scale multiplied by lambda (lambda <= 0 means no interaction, lambda = 1 means a full particle).
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the GrandCanonicalPotential class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If dimensionality_of_particle_space is not 1.
        """
        super().__init__(prefactor=prefactor, **kwargs)
        if dimensionality_of_particle_space != 1:
            raise ConfigurationError(f"{self.__class__.__name__} only supports dimensionality_of_particle_space = 1.")
        self._initial_number_of_real_particles = number_of_particles
        self.ghost_coupling = None
        self._ghost_profile = None

    def get_volume(self):
        return float(size_of_particle_space[0])

    def has_ghost(self):
        return self.ghost_coupling is not None

    def get_number_of_real_particles(self, positions):
        """Returns the number of real (non-ghost) particles."""
        return len(positions) - 1 if self.has_ghost() else len(positions)

    def wrap(self, position):
        """Returns position mapped onto the ring."""
        return np.mod(position, size_of_particle_space[0])

    def get_random_position(self):
        """Returns a position (array of shape (1,)) drawn uniformly from the ring."""
        return np.random.uniform(0.0, size_of_particle_space[0], 1)

    def _get_minimum_image(self, separations):
        return separations - size_of_particle_space[0] * np.round(separations / size_of_particle_space[0])

    def _get_separations(self, positions, index):
        """Returns the minimum-image separations positions[j] - positions[index] for all j."""
        return self._get_minimum_image(positions[:, 0] - positions[index, 0])

    def _get_scales(self, number_of_rows, index):
        """
        Returns the factor that multiplies each pair interaction in the potential with the particle at index. It is 1
        for two real particles, lambda for a pair with the ghost when lambda > 0, and 0 when the ghost is parked.
        """
        scales = np.ones(number_of_rows)
        if self.has_ghost():
            ghost_index = number_of_rows - 1
            ghost_scale = max(self.ghost_coupling, 0.0)
            if index == ghost_index:
                scales[:] = ghost_scale
            else:
                scales[ghost_index] = ghost_scale
        scales[index] = 0.0
        return scales

    def get_initial_positions(self):
        """
        Returns the initial positions of the particles (number_of_particles real, 1 ghost)

        Returns
        -------
        numpy.ndarray
            Array of shape (number_of_rows, 1).
        """
        positions = self.wrap(get_initial_positions_of_smooth_potential(self.__class__.__name__))
        if self.has_ghost():
            positions = np.concatenate([positions, self.get_random_position()[None, :]])
        self._check_initial_positions(positions)
        return positions

    def _check_initial_positions(self, positions):
        """Raises a ConfigurationError if the initial positions have infinite energy (e.g. overlapping hard disks)."""
        if not np.isfinite(self.get_total_value(positions)):
            raise ConfigurationError(f"The initial positions of {self.__class__.__name__} have infinite energy "
                                     f"(overlapping or coinciding particles).  Change "
                                     f"range_of_initial_particle_positions or the density.")

    @abstractmethod
    def _get_pair_energies(self, distances, scales):
        """
        Returns the pair energies for the given distances and length-scale factors.
        """
        raise NotImplementedError

    def get_value(self, positions):
        """Returns the potential energy of the real particles."""
        real_positions = positions[:self.get_number_of_real_particles(positions), 0]
        first_indices, second_indices = np.triu_indices(len(real_positions), k=1)
        distances = np.abs(self._get_minimum_image(real_positions[second_indices] - real_positions[first_indices]))
        return float(np.sum(self._get_pair_energies(distances, 1.0)))

    def get_total_value(self, positions):
        """Returns the potential energy of all particles, including the ghost at its current coupling."""
        energy = self.get_value(positions)
        if self.has_ghost():
            energy += self.get_single_particle_energy(positions, len(positions) - 1)
        return energy

    def get_single_particle_energy(self, positions, index):
        """Returns the interaction energy of particle index with all other rows (ghost pairs scaled by lambda)."""
        scales = self._get_scales(len(positions), index)
        mask = scales > 0.0
        return float(np.sum(self._get_pair_energies(np.abs(self._get_separations(positions, index)[mask]),
                                                    scales[mask])))

    def get_insertion_energy(self, positions, candidate_position):
        """
        Returns the energy of a full particle at candidate_position interacting with all real particles.  Only used
        without a ghost (direct insertion).
        """
        number_of_real_particles = self.get_number_of_real_particles(positions)
        distances = np.abs(self._get_minimum_image(positions[:number_of_real_particles, 0] - candidate_position[0]))
        return float(np.sum(self._get_pair_energies(distances, np.ones(number_of_real_particles))))

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """Returns the energy change on moving particle active_particle_index to candidate_position."""
        old_energy = self.get_single_particle_energy(positions, active_particle_index)
        trial_positions = positions.copy()
        trial_positions[active_particle_index] = self.wrap(candidate_position)
        return self.get_single_particle_energy(trial_positions, active_particle_index) - old_energy

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next event of the active particle, the index of the particle that triggers it and
        the hop displacement.

        Parameters
        ----------
        positions : numpy.ndarray
            Array of shape (number_of_rows, 1).
        active_particle_index : int
            The active particle index.
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
        scales = self._get_scales(len(positions), active_particle_index)
        interacting_indices = np.nonzero(scales > 0.0)[0]
        # If ghost is active all interactions are zero if lambda <= 0. Null event occurs and ghost moves around ring
        if len(interacting_indices) == 0:
            return size_of_particle_space[0], active_particle_index, 0.0
        separations = self._get_separations(positions, active_particle_index)[interacting_indices]
        distance, vetoing_index = self._get_next_event_from_separations(separations, scales[interacting_indices],
                                                                        temperature)
        if vetoing_index is None:
            return distance, active_particle_index, 0.0
        hop_displacement = self._get_minimum_image(separations[vetoing_index] - distance)
        return distance, int(interacting_indices[vetoing_index]), float(hop_displacement)

    @abstractmethod
    def _get_next_event_from_separations(self, separations, scales, temperature):
        """
        Returns the distance to the next event of the active particle given the minimum-image separations of the
        particles it interacts with (in (-L/2, L/2]) and their length-scale factors, and the index of the vetoing
        particle, or None for a null event.
        """
        raise NotImplementedError

    @staticmethod
    def get_random_event_chain_velocity():
        """Direction of event chain velocity."""
        return 1

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """Lifts the event-chain label to the vetoing particle."""
        return veto_index, movement_direction

    def update_position(self, positions, displacement_distance, active_particle_index, movement_direction):
        """Moves the active particle forward by displacement_distance and maps it back onto the ring."""
        positions[active_particle_index] = np.mod(positions[active_particle_index] + displacement_distance,
                                                  size_of_particle_space[0])

    def get_gradient(self, positions):
        raise NotImplementedError(f"{self.__class__.__name__} is only used with event-chain Monte Carlo.")

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        raise NotImplementedError(f"Teleportation portals are not implemented for {self.__class__.__name__}.")

    def _get_ghost_distances(self, positions):
        """Returns the distances from the ghost (last row) to every real particle."""
        return np.abs(self._get_minimum_image(positions[:-1, 0] - positions[-1, 0]))

    @abstractmethod
    def get_ghost_profile(self, positions):
        """
        Returns a scalar that determines the ghost energy as a function of lambda at the current positions, and caches
        it in self._ghost_profile for moves of lambda at fixed positions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ghost_energy(self, coupling, profile):
        """Returns the ghost energy U_g(coupling) given the profile."""
        raise NotImplementedError

    @abstractmethod
    def get_ghost_force(self, coupling):
        """Returns dU_g / d lambda at the cached profile."""
        raise NotImplementedError

    @abstractmethod
    def get_next_coupling_event(self, coupling, temperature, direction):
        """
        Returns the distance in lambda to the next event of the rate [direction * beta * dU_g / d lambda]^+ while lambda
        moves from coupling in the given direction (+1 growing, -1 shrinking), for the cached profile (numpy.inf if it
        never occurs).  This is the coupling analogue of get_next_event: the event occurs once the ghost energy has
        increased by an exponential random variable times the temperature.  For repulsive ghost interactions
        (dU_g / d lambda >= 0) only growing produces events; attractive ones must also handle direction = -1.
        """
        raise NotImplementedError

    @abstractmethod
    def get_ghost_wall(self):
        """Returns the coupling at which the cached ghost becomes blocked by a hard contact (hard spheres only)."""
        raise NotImplementedError

    @abstractmethod
    def choose_coupling_event_particle(self, positions):
        """
        Returns the index of the real particle whose ghost pair interaction caused a coupling event of the growing ghost
        (chosen in proportion to its share of dU_g / d lambda) and its separation from the ghost.
        """
        raise NotImplementedError

    @abstractmethod
    def get_lift_ratio(self, distance, coupling):
        """Returns |d u / d r| / (d u / d lambda) for a ghost pair at the given distance and coupling."""
        raise NotImplementedError

    @abstractmethod
    def get_description(self):
        """Returns a JSON-serialisable dictionary that describes the potential."""
        raise NotImplementedError
