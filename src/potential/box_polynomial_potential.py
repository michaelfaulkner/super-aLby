"""Module for the BoxPolynomialPotential class."""
import numpy as np
from typing import Sequence
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base.exceptions import ConfigurationError
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import number_of_particles, size_of_particle_space


class BoxPolynomialPotential(EuclideanSubspacePotential):
    """
    This class implements a non-toroidal polynomial chain on [0, L]: N particles joined by N - 1 springs, each with
        potential U(r) = sum_p prefactor_p * |r| ** power_p. The boundary is set by the options below.

    Tethers.  If tethered is True, particle 0 and particle N - 1 are joined by identical springs to fixed anchors at
        x = 0 and x = L, so there are N + 1 springs whose lengths sum to L.

    Walls.  walls = 'hard' (default) confines every particle to [0, L]; walls = 'soft' gives harmonic walls
        wall_stiffness * (min(0, x) ** 2 + max(0, x - L) ** 2) / 2; walls = 'none' requires tethered = True.

    Factor field.  factor_field_prefactor = b adds -b * r to every spring factor.

    Events.  Spring events lift to the neighbouring particle; wall, tether and linear-piece events reverse the direction
        of motion of the active particle, and are signalled to the mediator by a veto index of -1 or N.  Spring events
        are exact (closed form) for harmonic springs with any field, or a single power without a field.  Otherwise, set
        use_cell_horizon = True.
    """

    def __init__(self, powers: Sequence[float] = None, prefactors: Sequence[float] = None,
                 factor_field_prefactor: float = 0.0, tethered: bool = False, walls: str = 'hard',
                 wall_stiffness: float = 1.0, use_cell_horizon: bool = False, cell_horizon: float = 1.0):
        """
        The constructor of the BoxPolynomialPotential class.

        Parameters
        ----------
        powers : Sequence[float]
            The powers of the spring potential (default 2).
        prefactors : Sequence[float]
            The prefactors of the spring potential (default 0.5).
        factor_field_prefactor : float
            The factor field b, the same on every spring.
        tethered : bool
            If True, tether the end particles by springs to fixed anchors at x = 0 and x = L.
        walls : str
            'hard', 'soft' or 'none'.
        wall_stiffness : float
            The stiffness of soft walls.
        use_cell_horizon : bool
            If True, sample spring events by thinning over a cell horizon.
        cell_horizon : float
            The length of the cell horizon.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the options are invalid or inconsistent (see the class docstring).
        """
        super().__init__(prefactor=1.0)
        self._powers = np.atleast_1d(np.asarray([2.0] if powers is None else powers, dtype=float))
        self._prefactors = np.atleast_1d(np.asarray([0.5] if prefactors is None else prefactors, dtype=float))
        if len(size_of_particle_space) > 1:
            raise ConfigurationError(f'{self.__class__.__name__} only supports 1D space.')
        if (self._powers.size != self._prefactors.size or np.any(self._prefactors <= 0.0) or
                np.any(self._powers < 1.0)):
            raise ConfigurationError(f'{self.__class__.__name__} requires equal numbers of powers and prefactors, '
                                     f'with powers >= 1 and prefactors > 0 (convex springs).')
        if walls not in ('hard', 'soft', 'none'):
            raise ConfigurationError(f"Give 'hard', 'soft' or 'none' for walls in {self.__class__.__name__}.")
        if walls == 'none' and not tethered:
            raise ConfigurationError(f"walls = 'none' in {self.__class__.__name__} requires tethered = True.")
        if walls == 'soft' and wall_stiffness <= 0.0:
            raise ConfigurationError(f'Give a positive value for wall_stiffness in {self.__class__.__name__}.')
        self._box_length = size_of_particle_space[0]
        self._tethered = tethered
        self._walls = walls
        self._wall_stiffness = wall_stiffness
        self._use_cell_horizon = use_cell_horizon
        self._cell_horizon = cell_horizon
        self._is_harmonic = self._powers.tolist() == [2.0]
        self._spring_terms = list(zip(self._prefactors.tolist(), self._powers.tolist()))
        self._factor_field_prefactor = factor_field_prefactor
        if not (use_cell_horizon or self._is_harmonic or (self._powers.size == 1 and factor_field_prefactor == 0.0)):
            raise ConfigurationError(f'Exact events of {self.__class__.__name__} are only available for harmonic '
                                     f'springs or a single power without a factor field; set use_cell_horizon = True.')
        self._linear_piece_coefficients = np.zeros(number_of_particles)
        if not tethered:
            self._linear_piece_coefficients[0] -= factor_field_prefactor
            self._linear_piece_coefficients[-1] += factor_field_prefactor

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
        potential : float
            The potential.
        """
        particle_positions = positions[:, 0]
        if not np.all(self._is_inside_hard_walls(particle_positions)):
            return np.inf  # Reject move if particle moves outside hard walls
        potential = np.sum(self._get_spring_energy(self._get_spring_lengths(particle_positions)))
        potential += self._get_soft_wall_energy(particle_positions)
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
        particle_positions = positions[:, 0]
        spring_forces = self._get_spring_derivative(self._get_spring_lengths(particle_positions))
        if not self._tethered:
            spring_forces = np.concatenate(([0.0], spring_forces, [0.0]))
        gradient = spring_forces[:-1] - spring_forces[1:]
        if self._walls == 'soft':
            gradient += self._wall_stiffness * (np.minimum(0.0, particle_positions) +
                                                np.maximum(0.0, particle_positions - self._box_length))
        return gradient.reshape(-1, 1)

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
        current_position = positions[active_particle_index][0]
        candidate = candidate_position[0]
        if not self._is_inside_hard_walls(candidate):
            return np.inf  # Reject move if particle moves outside hard walls
        (neg_neighbour_index, neg_neighbour_position), (pos_neighbour_index, pos_neighbour_position) = (
            self._get_neighbours(positions, active_particle_index))
        potential_difference = 0.0
        if neg_neighbour_index is not None:  # Neighbour indices are none for particle 1 and N if untethered
            potential_difference += (self._get_spring_energy(candidate - neg_neighbour_position) -
                                     self._get_spring_energy(current_position - neg_neighbour_position))
        if pos_neighbour_index is not None:
            potential_difference += (self._get_spring_energy(pos_neighbour_position - candidate) -
                                     self._get_spring_energy(pos_neighbour_position - current_position))
        potential_difference += (self._get_soft_wall_energy(np.array([candidate])) -
                                 self._get_soft_wall_energy(np.array([current_position])))
        return potential_difference

    @staticmethod
    def get_random_event_chain_velocity():
        """
        Returns the initial direction of motion of the active particle.

        Returns
        -------
        int
            The initial direction of motion, +1.
        """
        return 1

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event (in ECMC), the index of the particle that triggers the event and
        the hop displacement.  Boundary (one-particle) events return a veto index of -1 or N.

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
        hop_displacement : float
            Net displacement through state space from active to vetoing particle.
        """
        active_particle_position = positions[active_particle_index][0]
        (neg_neighbour_index, neg_neighbour_position), (pos_neighbour_index, pos_neighbour_position) = (
            self._get_neighbours(positions, active_particle_index))
        spring_pieces = []  # each spring piece is (initial_spring_length, sign, veto index, hop)
        if neg_neighbour_index is not None:
            spring_pieces.append((active_particle_position - neg_neighbour_position, movement_direction,
                                  neg_neighbour_index, 0.0 if neg_neighbour_index < 0 else
                                  neg_neighbour_position - active_particle_position))
        if pos_neighbour_index is not None:
            spring_pieces.append((pos_neighbour_position - active_particle_position, -movement_direction,
                                  pos_neighbour_index, 0.0 if pos_neighbour_index > number_of_particles - 1 else
                                  pos_neighbour_position - active_particle_position))
        if self._use_cell_horizon:
            candidate_events = [self._get_cell_horizon_event(spring_pieces, active_particle_index, temperature)]
        else:
            candidate_events = [(self._get_spring_event_distance(initial_spring_length, sign, temperature), veto_index,
                                 hop) for initial_spring_length, sign, veto_index, hop in spring_pieces]
        # In untethered case, have to simulate extra events from the two end particles to satisfy sum to L constraint.
        linear_piece_rate = (np.maximum(0.0, movement_direction *
                                        self._linear_piece_coefficients[active_particle_index]) / temperature)
        if linear_piece_rate > 1e-12:
            candidate_events.append((-np.log(np.random.uniform(0.0, 1.0)) / linear_piece_rate,
                                     self._get_boundary_index(movement_direction), 0.0))
        candidate_events.append((self._get_wall_distance(active_particle_position, movement_direction, temperature),
                                 self._get_boundary_index(movement_direction), 0.0))
        return min(candidate_events)

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.  Spring events lift to the
        vetoing particle; boundary events (veto index outside [0, N - 1]) reverse the direction of motion of the active
        particle.

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
        if veto_index < 0 or veto_index >= number_of_particles:
            return active_particle_index, -movement_direction
        return veto_index, movement_direction

    def update_position(self, positions, displacement_distance, active_particle_index, movement_direction):
        """
        Updates the position of the active particle following an event.  Hard walls clip round-off beyond the wall.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of each particle.
        displacement_distance : float
            The distance moved by the active particle.
        active_particle_index : int
            The active particle index
        movement_direction : int
            The active-particle direction of motion.
        """
        new_position = positions[active_particle_index] + movement_direction * displacement_distance
        if self._walls == 'hard':  # Cheap guard for floating-point error. Hard wall limits handled in get_next_event.
            new_position = np.clip(new_position, 0.0, self._box_length)
        positions[active_particle_index] = new_position

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """
        Propose candidate via teleportation portal kernel.
        """
        raise SystemError(f"The get_portal_candidate method of {self.__class__.__name__} has not been written.")

    def _get_spring_energy(self, spring_lengths):
        """
        Returns the spring potential U(r) = sum_p prefactor_p * |r| ** power_p.

        Parameters
        ----------
        spring_lengths : float or numpy.ndarray
            The spring lengths.

        Returns
        -------
        float or numpy.ndarray
            The spring potential of each spring length.
        """
        spring_energy = 0.0
        for power, prefactor in zip(self._powers, self._prefactors):
            spring_energy += prefactor * np.abs(spring_lengths) ** power
        return spring_energy

    def _get_spring_derivative(self, spring_lengths):
        """
        Returns the derivative U'(r) of the spring potential.

        Parameters
        ----------
        spring_lengths : float or numpy.ndarray
            The spring length(s) r.

        Returns
        -------
        float or numpy.ndarray
            The derivative of the spring potential at each spring length.
        """
        spring_derivative = 0.0
        for power, prefactor in zip(self._powers, self._prefactors):
            spring_derivative += prefactor * power * np.abs(spring_lengths) ** (power - 1.0) * np.sign(spring_lengths)
        return spring_derivative

    def _get_scalar_spring_derivative(self, spring_length):
        """
        Returns the derivative U'(r) of the spring potential for a single spring length.

        Parameters
        ----------
        spring_length : float
            The spring length r.

        Returns
        -------
        float
            The derivative of the spring potential.
        """
        if spring_length == 0.0:
            return 0.0
        sign = 1.0 if spring_length > 0.0 else -1.0
        spring_derivative = 0.0
        for prefactor, power in self._spring_terms:
            spring_derivative += prefactor * power * abs(spring_length) ** (power - 1.0) * sign
        return spring_derivative

    def _get_spring_event_distance(self, initial_spring_length, sign, temperature):
        """
        Returns the exact (closed-form) distance to the next event of a single spring.

        Parameters
        ----------
        initial_spring_length : float
            The current spring length.
        sign : int
            +1 if the spring lengthens as the active particle moves, -1 if it shortens.
        temperature : float
            The sampling temperature.

        Returns
        -------
        float
            The distance to the next event of the spring.
        """
        prefactor, power = self._spring_terms[0]
        # Factor field can only be applied analytically for p = 2
        spring_minimum = self._factor_field_prefactor / (2.0 * prefactor) if self._is_harmonic else 0.0
        spring_extension = initial_spring_length - spring_minimum
        energy_increase = -temperature * np.log(np.random.uniform(0.0, 1.0)) / prefactor
        if sign * spring_extension >= 0.0:
            return (abs(spring_extension) ** power + energy_increase) ** (1.0 / power) - abs(spring_extension)
        return abs(spring_extension) + energy_increase ** (1.0 / power)

    def _get_cell_horizon_event(self, spring_pieces, active_particle_index, temperature):
        """
        Returns the next spring event by thinning over the cell horizon.  Each spring is bounded by its event rate at
        the end of the horizon, and the earliest candidate is accepted with probability rate / bound.
        Rejected and horizon events return the active particle as the vetoing index.

        Parameters
        ----------
        spring_pieces : list
            One tuple (initial_spring_length, sign, veto_index, hop_displacement) per spring of the active particle,
            where sign is +1 if the spring lengthens as the active particle moves and -1 if it shortens.
        active_particle_index : int
            The active particle index
        temperature : float
            The sampling temperature.

        Returns
        -------
        distance_to_next_event : float
            The distance to the next spring event.
        vetoing_index : int
            The index of the particle that triggers the event.
        hop_displacement : float
            Net displacement through state space from active to vetoing particle.
        """
        candidates = []
        for initial_spring_length, sign, veto_index, hop in spring_pieces:
            max_rate = max(0.0, sign * (self._get_scalar_spring_derivative(
                initial_spring_length + sign * self._cell_horizon) - self._factor_field_prefactor))
            if max_rate > 1e-12:
                candidates.append((-np.log(np.random.uniform(0.0, 1.0)) * temperature / max_rate,
                                   initial_spring_length, sign, veto_index, hop, max_rate))
        if not candidates or min(candidates)[0] > self._cell_horizon:
            return self._cell_horizon, active_particle_index, 0.0
        distance_to_next_event, initial_neighbour_separation, sign, veto_index, hop, max_rate = min(candidates)
        actual_rate = max(0.0, sign * (self._get_scalar_spring_derivative(
            initial_neighbour_separation + sign * distance_to_next_event) - self._factor_field_prefactor))
        if np.random.uniform(0.0, 1.0) < actual_rate / max_rate:
            return distance_to_next_event, veto_index, hop
        return distance_to_next_event, active_particle_index, 0.0

    def _get_neighbours(self, positions, particle_index):
        """
        Returns the index and position of the negative and positive spring neighbours of a particle, or (None, None)
        where there is no spring.  A tether anchor has index -1 (at x = 0) or N (at x = L).

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of each particle.
        particle_index : int
            The index of the particle.

        Returns
        -------
        tuple
            (index, position) of the negative and of the positive neighbour, each (None, None) if there is no spring.
        """
        neg_neighbour = ((particle_index - 1, positions[particle_index - 1][0]) if particle_index > 0 else
                         (-1, 0.0) if self._tethered else (None, None))
        pos_neighbour = ((particle_index + 1, positions[particle_index + 1][0]) if particle_index < number_of_particles - 1
                         else (number_of_particles, self._box_length) if self._tethered else (None, None))
        return neg_neighbour, pos_neighbour

    def _get_spring_lengths(self, particle_positions):
        """
        Returns the spring lengths (including the tethers if tethered).

        Parameters
        ----------
        particle_positions : numpy.ndarray
            A one-dimensional numpy array of the particle positions.

        Returns
        -------
        numpy.ndarray
            The spring lengths.
        """
        if self._tethered:
            return np.diff(np.concatenate(([0.0], particle_positions, [self._box_length])))
        return np.diff(particle_positions)

    def _is_inside_hard_walls(self, particle_positions):
        """
        Returns whether each position satisfies the hard-wall constraints (always True without hard walls).

        Parameters
        ----------
        particle_positions : float or numpy.ndarray
            The particle positions.

        Returns
        -------
        numpy.ndarray
            Whether each position lies inside [0, L].
        """
        if self._walls != 'hard':
            return np.full(np.shape(particle_positions), True)
        return (particle_positions >= 0.0) & (particle_positions <= self._box_length)

    @staticmethod
    def _get_boundary_index(direction):
        """
        Returns the veto index signalling a one-particle (direction-reversing) event: N on the right, -1 on the left.

        Parameters
        ----------
        direction : int
            The active-particle direction of motion.

        Returns
        -------
        int
            The veto index.
        """
        return number_of_particles if direction > 0 else - 1

    def _get_soft_wall_energy(self, particle_positions):
        """
        Returns the soft-wall energy (zero for hard or no walls).

        Parameters
        ----------
        particle_positions : numpy.ndarray
            A one-dimensional numpy array of the particle positions.

        Returns
        -------
        float
            The soft-wall energy.
        """
        if self._walls != 'soft':
            return 0.0
        return 0.5 * self._wall_stiffness * np.sum(np.minimum(0.0, particle_positions) ** 2 +
                                                   np.maximum(0.0, particle_positions - self._box_length) ** 2)

    def _get_wall_distance(self, active_particle_position, direction, temperature):
        """
        Returns the distance to the wall event in the direction of motion: deterministic for hard walls, infinite without
        walls, and for soft walls the point where the wall energy (wall_stiffness / 2) * overshoot ** 2 has risen by
        -temperature * log(u), where overshoot is the distance past the wall ahead (negative inside the box).

        Parameters
        ----------
        active_particle_position : float
            The position of the active particle.
        direction : int
            The active-particle direction of motion.
        temperature : float
            The sampling temperature.

        Returns
        -------
        float
            The distance to the wall event.
        """
        if self._walls == 'none':
            return np.inf
        if self._walls == 'hard':
            return (max(0.0, self._box_length - active_particle_position) if direction > 0
                    else max(0.0, active_particle_position))
        overshoot = (active_particle_position - self._box_length) if direction > 0 else - active_particle_position
        energy_increase = - 2.0 * temperature * np.log(np.random.uniform(0.0, 1.0)) / self._wall_stiffness
        if overshoot >= 0.0:  # already past the wall
            return np.sqrt(overshoot ** 2 + energy_increase) - overshoot
        return -overshoot + np.sqrt(energy_increase)  # first reach the wall
