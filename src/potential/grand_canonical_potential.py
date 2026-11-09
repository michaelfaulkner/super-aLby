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
    its coupling lambda in [-1 - parked_length, 1], the ghost diameter in units of sigma.  Every ghost-real pair
    interacts as a real-real pair with its length scale multiplied by (1 + lambda) / 2, i.e. the additive contact
    (1 + lambda) sigma / 2 (lambda <= -1 means no interaction, lambda = 1 means a full particle).
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
        """Coupling below which only the ghost moves in the compression flow (-1: the whole flow is one region)."""
        self.compression_split = -1.0

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

    @staticmethod
    def get_ghost_scale(coupling):
        """Returns the length-scale factor max((1 + lambda) / 2, 0) of the ghost-real pairs at coupling lambda."""
        return np.maximum(0.5 * (1.0 + np.asarray(coupling, dtype=float)), 0.0)

    def _get_minimum_image(self, separations):
        return separations - size_of_particle_space[0] * np.round(separations / size_of_particle_space[0])

    def _get_separations(self, positions, index):
        """Returns the minimum-image separations positions[j] - positions[index] for all j."""
        return self._get_minimum_image(positions[:, 0] - positions[index, 0])

    def _get_scales(self, number_of_rows, index):
        """
        Returns the factor that multiplies each pair interaction in the potential with the particle at index. It is 1
        for two real particles, (1 + lambda) / 2 for a pair with the ghost when lambda > -1, and 0 when the ghost is
        parked.
        """
        scales = np.ones(number_of_rows)
        if self.has_ghost():
            ghost_index = number_of_rows - 1
            ghost_scale = float(self.get_ghost_scale(self.ghost_coupling))
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
        """Returns the interaction energy of particle index with all other rows (ghost pairs at the ghost scale)."""
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
        # If ghost is active all interactions are zero if lambda <= -1. Null event occurs and ghost moves around ring
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

    def get_compression_contact(self):
        """Returns the reference contact distance c of real pairs in the compression flow."""
        raise NotImplementedError(f"{self.__class__.__name__} has no compression flow.")

    def get_compression_chain(self, positions, window):
        """
        Returns (left_wall, movers, right_wall) for the compression flow of the ghost (last row): the rows that move
        with lambda, in ring order from the left wall, i.e. [L_w, ..., L_1, ghost, R_1, ..., R_w] with
        w = min(window, (N - 1) // 2), and the two frozen rows just outside them (the same row if N is odd and the
        window covers the ring).  window = 0 moves only the ghost between its two neighbours.
        """
        number_of_real_particles = len(positions) - 1
        ghost_index = number_of_real_particles
        window = min(window, (number_of_real_particles - 1) // 2)
        forward = np.mod(positions[:-1, 0] - positions[-1, 0], size_of_particle_space[0])
        if window == 0:  # Only the two neighbours of the ghost are needed
            return int(np.argmax(forward)), np.array([ghost_index]), int(np.argmin(forward))
        """Real particles in ring order going forwards from the ghost: R_1 first, L_1 last."""
        order = np.argsort(forward)
        movers = np.concatenate([order[number_of_real_particles - window:], [ghost_index], order[:window]])
        return int(order[number_of_real_particles - 1 - window]), movers.astype(int), int(order[window])

    def _get_compression_gaps(self, positions, chain, coupling):
        """Returns the excess gaps between consecutive rows of the chain (from the left wall to the right wall), which
        gaps belong to the ghost, and the reference contact distances at the given coupling."""
        left_wall, movers, right_wall = chain
        length = size_of_particle_space[0]
        contact = self.get_compression_contact()
        offsets = np.mod(positions[movers, 0] - positions[left_wall, 0], length)
        span = np.mod(positions[right_wall, 0] - positions[left_wall, 0], length)
        if right_wall == left_wall:
            span = length
        ghost_pairs = np.zeros(len(movers) + 1, dtype=bool)
        ghost_position = int(np.nonzero(movers == len(positions) - 1)[0][0])
        ghost_pairs[[ghost_position, ghost_position + 1]] = True
        contacts = np.where(ghost_pairs, self.get_ghost_scale(coupling) * contact, contact)
        gaps = np.diff(np.concatenate([[0.0], offsets, [span]])) - contacts
        return gaps, ghost_pairs, contacts

    def get_compression_flow(self, positions, chain, coupling):
        """
        Returns (rate_constant, free_length, free_length_slope, velocities) of the compression flow: the free length of
        the chain shrinks by free_length_slope = c per unit growth of lambda, the flow changes the configuration
        volume at the relative rate rate_constant / free_length (rate_constant = c * number of moving rows), and
        velocities holds d x / d lambda of every row (zero for the rows that do not move).
        """
        gaps, ghost_pairs, _ = self._get_compression_gaps(positions, chain, coupling)
        contact = self.get_compression_contact()
        free_length = float(np.sum(gaps))
        gap_velocities = -contact * gaps / free_length + np.where(ghost_pairs, 0.5 * contact, 0.0)
        velocities = np.zeros(len(positions))
        velocities[chain[1]] = np.cumsum(gap_velocities)[:-1]
        return contact * len(chain[1]), free_length, contact, velocities

    def apply_compression(self, positions, chain, old_coupling, new_coupling):
        """
        Moves the rows of the chain along the compression flow from old_coupling to new_coupling: every excess gap of
        the chain is scaled by the same factor.
        """
        left_wall, movers, _ = chain
        gaps, ghost_pairs, _ = self._get_compression_gaps(positions, chain, old_coupling)
        contact = self.get_compression_contact()
        free_length = np.sum(gaps)
        scale = (free_length - (new_coupling - old_coupling) * contact) / free_length
        new_contacts = np.where(ghost_pairs, self.get_ghost_scale(new_coupling) * contact, contact)
        offsets = np.cumsum(gaps * scale + new_contacts)[:-1]
        positions[movers, 0] = np.mod(positions[left_wall, 0] + offsets, size_of_particle_space[0])

    """The energy along the compression flow.  The defaults are for potentials whose energy does not change along the
        flow (hard spheres); GrandCanonicalSoftSpherePotential overrides them.  Between synchronisations the flow line
        is cached by prepare_compression, and the reversal rate of lambda is split into two factors: the near factor
        [v_lambda (beta dU_near/dlambda + J - g)]^+ (pairs within a few ring neighbours, evaluated exactly) and one
        factor per remaining (tail) pair, sampled together by thinning with a single random pair per proposal."""

    def prepare_compression(self, positions, velocities, coupling, lowest, highest, temperature):
        """Caches the flow line through positions (moving at velocities = d x / d lambda from coupling) and returns the
        interval of lambda (inside [lowest, highest]) in which the cache and its bounds are valid."""
        return lowest, highest

    def get_compression_energy_rate(self, coupling, direction):
        """Returns the near-factor energy rate beta d U_near / d t at coupling, with lambda moving in direction."""
        return 0.0

    def get_compression_energy_rate_bound(self, coupling, step, direction):
        """Returns an upper bound of get_compression_energy_rate while lambda moves a further distance step."""
        return 0.0

    def get_compression_step_cap(self, coupling, direction):
        """Returns the longest lambda step for which get_compression_energy_rate_bound is valid."""
        return np.inf

    def get_compression_tail_rate_bound(self):
        """Returns the bound of the summed tail-pair reversal rates, valid in the interval of prepare_compression."""
        return 0.0

    def accept_compression_tail_event(self, coupling, direction):
        """Thinning of a tail proposal at coupling: returns True if one random tail pair reverses lambda."""
        return False

    @abstractmethod
    def get_description(self):
        """Returns a JSON-serialisable dictionary that describes the potential."""
        raise NotImplementedError
