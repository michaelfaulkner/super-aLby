"""Module for the abstract GrandCanonicalPotential class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from .euclidean_subspace_potential import EuclideanSubspacePotential
from model_settings import dimensionality_of_particle_space, number_of_particles, size_of_particle_space
import numpy as np


class GrandCanonicalPotential(EuclideanSubspacePotential, metaclass=ABCMeta):
    r"""
    Abstract class for pairwise potentials of disks on a ring whose number of particles may change, for use with
    EventChainMediator together with a grand-canonical kernel.

    Unlike the canonical potentials, the number of particles is not fixed: it is always read from the positions array,
    which has shape (number_of_rows, 1).  Particles live on the periodic interval [0, L) (L = size_of_particle_space,
    dimensionality_of_particle_space must be 1) and interact through all pairs with the minimum-image convention.  The
    interaction does not depend on particle order, so particles can be appended and removed freely.  Event chains always
    move in the positive direction (movement_direction = 1).

    If self.ghost_coupling is not None, the last row of positions is the ghost particle and self.ghost_coupling is
    its coupling lambda in [- parked_length, 1].  Every ghost-real pair interacts as a real-real pair with its length
    scale multiplied by lambda (lambda <= 0 means no interaction, lambda = 1 means a full particle).  Children define
    the pair energy as a function of the distance and of this length-scale factor.
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

    @property
    def volume(self):
        return float(size_of_particle_space[0])

    @property
    def has_ghost(self):
        return self.ghost_coupling is not None

    def get_number_of_real_particles(self, positions):
        """Returns the number of real (non-ghost) particles."""
        return len(positions) - 1 if self.has_ghost else len(positions)

    def wrap(self, position):
        """Returns position mapped onto the ring."""
        return np.mod(position, size_of_particle_space[0])

    def get_random_position(self):
        """Returns a position (array of shape (1,)) drawn uniformly from the ring."""
        return np.random.uniform(0.0, size_of_particle_space[0], 1)

    def _get_minimum_image(self, separations):
        return separations - size_of_particle_space[0] * np.round(separations / size_of_particle_space[0])

    def _get_separations(self, positions, index):
        """Returns the minimum-image separations positions[j] - positions[index] for all j (shape (number_of_rows,))."""
        return self._get_minimum_image(positions[:, 0] - positions[index, 0])

    def _get_scales(self, number_of_rows, index):
        """
        Returns the length-scale factor of each pair (index, j): 1 for real-real pairs, the effective ghost coupling
        max(lambda, 0) for ghost-real pairs, and 0 for j = index.
        """
        scales = np.ones(number_of_rows)
        if self.has_ghost:
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
        Returns the initial positions: number_of_particles (+ 1 with a ghost) evenly spaced sites on the ring, randomly
        assigned to the rows (so the ghost, the last row, sits on a random site).

        Returns
        -------
        numpy.ndarray
            Array of shape (number_of_rows, 1).
        """
        number_of_rows = self._initial_number_of_real_particles + (1 if self.has_ghost else 0)
        spacing = size_of_particle_space[0] / number_of_rows
        sites = (np.random.permutation(number_of_rows) + 0.5) * spacing
        self._check_initial_positions(sites[:, None], spacing)
        return sites[:, None]

    def _check_initial_positions(self, positions, spacing):
        """
        Raises a ConfigurationError if the initial positions have infinite energy (only relevant to hard cores).
        Children may also check the spacing (see GrandCanonicalHardSpherePotential).
        """
        if not np.isfinite(self.get_total_value(positions)):
            raise ConfigurationError(f"The initial lattice of {self.__class__.__name__} has overlapping particles.  "
                                     f"Use a smaller density.")

    @abstractmethod
    def _get_pair_energies(self, distances, scales):
        """
        Returns the pair energies for the given (non-negative) distances and length-scale factors (vectorised).  A scale
        of 0 means no interaction.
        """
        raise NotImplementedError

    def get_value(self, positions):
        """Returns the potential energy of the real particles (the ghost is excluded)."""
        real_positions = positions[:self.get_number_of_real_particles(positions), 0]
        upper = np.triu_indices(len(real_positions), k=1)
        distances = np.abs(self._get_minimum_image(real_positions[:, None] - real_positions[None, :]))[upper]
        return float(np.sum(self._get_pair_energies(distances, np.ones(len(distances)))))

    def get_total_value(self, positions):
        """Returns the potential energy of all particles, including the ghost at its current coupling."""
        energy = self.get_value(positions)
        if self.has_ghost:
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
        the hop displacement (the displacement from the active to the vetoing particle at the event).  If the vetoing
        index equals active_particle_index, the event is a null event (no lift; the mediator simply re-evaluates).

        Parameters
        ----------
        positions : numpy.ndarray
            Array of shape (number_of_rows, 1).
        active_particle_index : int
            The active particle index (may be the ghost).
        temperature : float
            The sampling temperature.
        movement_direction : int
            Always 1 (motion in the positive direction).

        Returns
        -------
        float
            Distance to the next event (always finite).
        int
            Index of the vetoing particle.
        float
            Hop displacement.
        """
        scales = self._get_scales(len(positions), active_particle_index)
        partner_indices = np.nonzero(scales > 0.0)[0]
        if len(partner_indices) == 0:
            return size_of_particle_space[0], active_particle_index, size_of_particle_space[0]
        separations = self._get_separations(positions, active_particle_index)[partner_indices]
        distance, best = self._get_next_event_of_partners(separations, scales[partner_indices], temperature)
        if best is None:
            return distance, active_particle_index, distance
        hop = self._get_minimum_image(separations[best] - distance)
        return distance, int(partner_indices[best]), float(hop)

    @abstractmethod
    def _get_next_event_of_partners(self, separations, scales, temperature):
        """
        Returns the distance to the next event of the active particle given the minimum-image separations of its
        interacting partners (in (- L / 2, L / 2]) and their length-scale factors, and the index (into separations) of
        the vetoing partner, or None for a null event (always a finite distance; see
        EventChainMediator, which takes the minimum over candidate event tuples).
        """
        raise NotImplementedError

    @staticmethod
    def get_random_event_chain_velocity():
        """Returns 1: event chains always move in the positive direction."""
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
        Returns a scalar that determines the ghost energy as a function of lambda at the current positions (see
        get_ghost_energy), and caches it in self._ghost_profile for moves of lambda at fixed positions (read by
        get_ghost_force, get_next_coupling_event and get_ghost_wall).
        """
        raise NotImplementedError

    @abstractmethod
    def get_ghost_energy(self, coupling, profile):
        """Returns the ghost energy U_g(coupling) given the profile (vectorised over coupling or profile)."""
        raise NotImplementedError

    @abstractmethod
    def get_ghost_force(self, coupling):
        """Returns dU_g / d lambda at the cached profile (zero for hard cores, where the wall acts instead)."""
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
        """Returns the coupling at which the cached ghost becomes blocked by a hard contact (numpy.inf if never)."""
        raise NotImplementedError

    @abstractmethod
    def get_description(self):
        """Returns a JSON-serialisable dictionary that describes the potential."""
        raise NotImplementedError
