"""Module for the TemperedGrandCanonicalKernel class."""
from base.exceptions import ConfigurationError
from ghost_coupling_bias.ghost_coupling_bias import GhostCouplingBias
from ghost_coupling_bias.no_ghost_coupling_bias import NoGhostCouplingBias
from model_settings import number_of_particles
from .grand_canonical_kernel import GrandCanonicalKernel
import numpy as np


class TemperedGrandCanonicalKernel(GrandCanonicalKernel):
    r"""
    Class for grand-canonical ECMC with nonreversible tempering of a single ghost particle.

    The ghost is the last row of positions. Its coupling lambda (the ghost diameter in units of sigma) lies in
    [-1 - parked_length, 1] and moves at speed v_lambda.
    The augmented target is z^N / N! * exp(- beta * U(x, y; lambda) + eta(lambda)) / V, uniform over the
    N + 1 event-chain labels and the directions. eta is given by ghost_coupling_bias.

    Kernel events (coupling moves) start after exponentially distributed distances through the state space with mean
    normalised_distance_between_coupling_moves * number_of_particles. During coupling moves the particles are frozen
    and lambda moves at unit speed for an exponentially distributed distance with mean mean_coupling_move_length:
      - flips of v_lambda at rate [v_lambda * (beta * dU_g / d lambda - g(lambda))]^+, sampled by thinning
        the sum of the neighbour rate [v_lambda * beta * dU_g / d lambda]^+ and the bias rate [- v_lambda * g(lambda)]^+
      - a hard wall at the ghost clearance (only for hard spheres)
      - insertion at lambda = 1: the ghost becomes real and a new parked ghost appears at a uniform position at
        lambda = -1 - parked_length, accepted with min(1, z V / (N + 2) * exp(eta(-1 - parked_length) - eta(1)));
      - deletion at lambda = -1 - parked_length: null if the event-chain label is on the parked ghost, else the parked
        ghost is removed and a uniformly chosen real particle becomes the ghost at lambda = 1, accepted with
        min(1, (N + 1) / (z V) * exp(eta(1) - eta(-1 - parked_length))). A rejected jump flips v_lambda.
    """

    def __init__(self, ghost_coupling_bias: GhostCouplingBias = NoGhostCouplingBias(), log_activity: float = 0.0,
                 parked_length: float = 2.0, normalised_distance_between_coupling_moves: float = 0.05,
                 mean_coupling_move_length: float = 2.0, compression_flow: bool = False,
                 compression_window: int = None):
        """
        The constructor of the TemperedGrandCanonicalKernel class.

        Parameters
        ----------
        ghost_coupling_bias : ghost_coupling_bias.ghost_coupling_bias.GhostCouplingBias, optional
            The bias eta(lambda); its slopes are the g values.
        log_activity : float, optional
            log z = beta * mu.
        parked_length : float, optional
            Length of the parked interval [-1 - parked_length, -1] of lambda.
        normalised_distance_between_coupling_moves : float, optional
            Mean distance through the state space between coupling moves, per particle.
        mean_coupling_move_length : float, optional
            Mean distance travelled by lambda during one set of coupling moves.
        compression_flow : bool, optional
            If True, the ghost's neighbourhood moves with lambda during coupling moves. The excess gaps of the ghost
            and its compression_window neighbours on each side are scaled together (for hard spheres, below
            lambda = 0 only the ghost moves, keeping its relative place between its two neighbours).
        compression_window : int, optional
            Number of neighbours on each side of the ghost that move during compression_flow moves (None: the whole
            ring).

        Raises
        ------
        base.exceptions.ConfigurationError
            If ghost_coupling_bias is not a GhostCouplingBias, or if any distance is not greater than 0.0.
        base.exceptions.ConfigurationError
            If compression_flow and coupled_lifting are both True, or if compression_window is negative.
        """
        super().__init__()
        if not isinstance(ghost_coupling_bias, GhostCouplingBias):
            raise ConfigurationError(f"Give a GhostCouplingBias child class as ghost_coupling_bias in "
                                     f"{self.__class__.__name__}.")
        if min(parked_length, normalised_distance_between_coupling_moves, mean_coupling_move_length) <= 0.0:
            raise ConfigurationError(f"Give values greater than 0.0 for parked_length, "
                                     f"normalised_distance_between_coupling_moves and mean_coupling_move_length in "
                                     f"{self.__class__.__name__}.")
        if compression_window is not None and compression_window < 0:
            raise ConfigurationError(f"Give a non-negative compression_window in {self.__class__.__name__}.")
        self._bias = ghost_coupling_bias
        self._log_activity = log_activity
        self._parked_length = parked_length
        self._lowest_coupling = -1.0 - parked_length  # The bottom of the parked interval
        self._mean_distance_between_coupling_moves = normalised_distance_between_coupling_moves * number_of_particles
        self._mean_coupling_move_length = mean_coupling_move_length
        self._compression_flow = compression_flow
        self._compression_window = compression_window if compression_window is not None else number_of_particles
        """Compression flow, set up by _set_up_compression_flow: the region of lambda (None = parked, inner or outer),
            the coupling at which the particles were last moved, the chain of moving rows, J = rate constant / free
            length, the free length and its decrease per unit lambda, the coupling interval in which the flow cache
            is valid, and the last thinning step length."""
        self._flow_region = None
        self._synchronised_coupling = None
        self._flow_chain = None
        self._flow_rate_constant = None
        self._free_length = None
        self._free_length_slope = None
        self._flow_lowest_coupling = None
        self._flow_highest_coupling = None
        self._thinning_step = None
        self._activity_times_volume = None
        self._coupling_direction = 1  # Lambda space velocity
        self._last_extreme = None
        self._statistics = {key: 0 for key in [
            "coupling_moves", "insertion_attempts", "insertion_acceptances", "deletion_attempts",
            "deletion_null_proposals", "deletion_acceptances", "coupling_bias_flips", "coupling_thinning_rejections",
            "coupling_wall_flips", "upward_passages", "downward_passages", "compression_flips",
            "compression_thinning_rejections", "compression_tail_flips"]}

    def initialize(self, potential, temperature):
        super().initialize(potential, temperature)
        if self._compression_flow and not hasattr(potential, "get_compression_chain"):
            raise ConfigurationError(f"compression_flow in {self.__class__.__name__} needs a GrandCanonicalPotential "
                                     f"with a compression flow.")
        potential.ghost_coupling = self._lowest_coupling + 0.5 * self._parked_length
        self._activity_times_volume = float(np.exp(self._log_activity)) * potential.get_volume()  # z * V

    def get_distance_to_next_event(self):
        return -np.log(np.random.uniform(0.0, 1.0)) * self._mean_distance_between_coupling_moves

    def handle_event(self, positions, active_particle_index, movement_direction):
        """
        Runs one set of coupling moves: the particles are frozen while lambda travels an exponentially distributed
        distance with mean mean_coupling_move_length.
        """
        self._count("coupling_moves")
        # How far we can move through lambda space before returning to particle state space
        remaining_length = -np.log(np.random.uniform(0.0, 1.0)) * self._mean_coupling_move_length
        if self._compression_flow:
            positions, active_particle_index = self._run_compression_moves(positions, active_particle_index,
                                                                           remaining_length)
            return positions, active_particle_index, movement_direction
        # Cache ghost profile (fixed while particles are frozen)
        self._potential.get_ghost_profile(positions)
        beta = 1.0 / self._temperature
        while True:
            coupling = self._potential.ghost_coupling
            if self._coupling_direction > 0:
                if coupling < -1.0:  # Moving through lambda < -1 is free as ghost potential is zero
                    next_stop, kind = -1.0, "reach_zero"  # The ghost starts interacting at lambda = -1
                else:
                    next_stop, kind = 1.0, "top"
                    wall = self._potential.get_ghost_wall()  # Value of lambda that ghost hits hard stop (hard spheres)
                    if wall <= next_stop:  # Do we hit a hard wall before reaching lambda = 1?
                        next_stop, kind = wall, "wall"
            else:
                if coupling > -1.0:
                    next_stop, kind = -1.0, "park"  # Stop at -1: below it the ghost no longer interacts
                else:
                    next_stop, kind = self._lowest_coupling, "bottom"
            if kind in ("reach_zero", "bottom"):  # Ghost particle not interacting
                proposal_distance = np.inf
            else:
                # Next flip proposal = the earlier of the ghost-pair event and the bias event
                proposal_distance = min(self._potential.get_next_coupling_event(coupling, self._temperature,
                                                                                self._coupling_direction),
                                        self._bias.get_next_event(coupling, self._coupling_direction))
            distance_to_next_stop = max(self._coupling_direction * (next_stop - coupling), 0.0)
            candidates = [distance_to_next_stop, remaining_length, proposal_distance]
            choice = int(np.argmin(candidates))
            distance_to_next_event = candidates[choice]
            new_coupling = next_stop if choice == 0 else coupling + self._coupling_direction * distance_to_next_event
            self._potential.ghost_coupling = new_coupling
            remaining_length -= distance_to_next_event
            if choice == 1:  # Return to particle space moves
                return positions, active_particle_index, movement_direction
            if choice == 0:
                if kind == "wall":
                    self._count("coupling_wall_flips")
                    self._coupling_direction = -1
                elif kind == "top":
                    positions = self._attempt_insertion(positions)
                elif kind == "bottom":  # Deletion attempt (the label stays with its particle)
                    positions, active_particle_index = self._attempt_deletion(positions, active_particle_index)
            else:
                force_rate = beta * self._potential.get_ghost_force(new_coupling)
                bias_slope = self._bias.get_slope(new_coupling, self._coupling_direction)
                rate = max(0.0, self._coupling_direction * (force_rate - bias_slope))
                rate_bound = (max(self._coupling_direction * force_rate, 0.0) +
                              max(-self._coupling_direction * bias_slope, 0.0))
                if np.random.uniform(0.0, 1.0) * rate_bound < rate:
                    self._count("coupling_bias_flips")
                    self._coupling_direction = -self._coupling_direction
                else:
                    self._count("coupling_thinning_rejections")

    def _get_compression_region(self):
        coupling, direction = self._potential.ghost_coupling, self._coupling_direction
        split = self._potential.compression_split
        if coupling < -1.0 or (coupling == -1.0 and direction < 0):
            return None
        if coupling < split or (coupling == split and direction < 0):
            return "inner"
        return "outer"

    def _run_compression_moves(self, positions, active_particle_index, remaining_length):
        """
        Coupling moves with the compression flow.  While the ghost interacts, lambda moves
        together with its neighbourhood growing lambda scales the free gaps of the moving rows down, so no hard wall is
        ever hit, and shrinking lambda scales them up.  The flow changes the configuration volume at the relative rate
        J = rate_constant / free_length per unit lambda, so lambda reverses at rate [v_lambda (J - g(lambda))]^+, only
        where the neighbourhood is denser (growing) or more dilute (shrinking) than the bias expects.  The particles are
        moved lazily, at the end of the coupling moves, at lambda = compression_split and lambda = -1, and before
        jumps.  Returns the positions and the (stored) active particle index.
        """
        potential = self._potential
        split = potential.compression_split
        self._thinning_step = 0.5
        self._set_up_compression_flow(positions)
        while True:
            coupling, direction = potential.ghost_coupling, self._coupling_direction
            if self._flow_region is None:  # Parked ghost: lambda moves freely to -1 or to -1 - parked_length
                stop = -1.0 if direction > 0 else self._lowest_coupling
                if remaining_length < abs(stop - coupling):
                    potential.ghost_coupling = coupling + direction * remaining_length
                    return positions, active_particle_index
                remaining_length -= abs(stop - coupling)
                potential.ghost_coupling = stop
                if direction < 0:
                    positions, active_particle_index = self._attempt_deletion(positions, active_particle_index)
                self._set_up_compression_flow(positions)
                continue
            free = self._free_length - self._free_length_slope * (coupling - self._synchronised_coupling)
            if direction > 0 and free <= 1.0e-12:  # Jammed neighbourhood (reached only through round-off)
                self._count("compression_flips")
                self._coupling_direction = -1
                continue
            """A step goes at most to the edge of the interval where the flow cache is valid (and, growing, at most
                until the free length halves).  The near-factor rate is bounded with the extreme bias slope of that
                step (also a bound for shorter ones).  If more than 3 proposals are expected, the step restarts from
                twice the previous step length and is halved until the bound is tight."""
            target = self._flow_highest_coupling if direction > 0 else self._flow_lowest_coupling
            step_length = min(abs(target - coupling), potential.get_compression_step_cap(coupling, direction))
            if direction > 0:
                step_length = min(step_length, 0.25 * free / self._free_length_slope)
            end = coupling + direction * step_length
            smallest_slope, largest_slope = self._bias.get_slope_range(min(coupling, end), max(coupling, end))
            slope_bound = smallest_slope if direction > 0 else -largest_slope
            restarted = False
            while True:
                end_free = free - direction * self._free_length_slope * step_length
                bound = max(potential.get_compression_energy_rate_bound(coupling, step_length, direction) +
                            direction * self._flow_rate_constant / end_free - slope_bound, 0.0)
                if bound * step_length <= 3.0 or step_length < 2.0e-12:
                    break
                if not restarted and 2.0 * self._thinning_step < step_length:
                    step_length, restarted = 2.0 * self._thinning_step, True
                else:
                    step_length *= 0.5
            self._thinning_step = max(step_length, 2.0e-6)
            end = coupling + direction * step_length
            if step_length >= abs(target - coupling):
                end = target  # Exactly, so that the edge of the interval is recognised below
            proposal = -np.log(np.random.uniform(0.0, 1.0)) / bound if bound > 0.0 else np.inf
            tail_bound = potential.get_compression_tail_rate_bound()
            candidates = [remaining_length, step_length, proposal,
                          -np.log(np.random.uniform(0.0, 1.0)) / tail_bound if tail_bound > 0.0 else np.inf]
            choice = int(np.argmin(candidates))
            step = candidates[choice]
            potential.ghost_coupling = coupling + direction * step
            remaining_length -= step
            if choice == 0:
                self._move_particles_along_compression_flow(positions)
                return positions, active_particle_index
            if choice == 2:  # Near factor: J - g and the energy of the near pairs
                rate = (direction * (self._flow_rate_constant / (free - direction * self._free_length_slope * step) -
                                     self._bias.get_slope(potential.ghost_coupling, direction)) +
                        potential.get_compression_energy_rate(potential.ghost_coupling, direction))
                if np.random.uniform(0.0, 1.0) * bound < rate:
                    self._count("compression_flips")
                    self._coupling_direction = -direction
                else:
                    self._count("compression_thinning_rejections")
                continue
            if choice == 3:  # Tail factor: one random far pair
                if potential.accept_compression_tail_event(potential.ghost_coupling, direction):
                    self._count("compression_tail_flips")
                    self._coupling_direction = -direction
                continue
            potential.ghost_coupling = end
            if direction > 0 and end >= 1.0:
                self._move_particles_along_compression_flow(positions)
                number_of_rows = len(positions)
                positions = self._attempt_insertion(positions)
                if len(positions) != number_of_rows:
                    self._set_up_compression_flow(positions)
            elif end in (-1.0, split, self._flow_lowest_coupling, self._flow_highest_coupling):
                self._move_particles_along_compression_flow(positions)
                self._set_up_compression_flow(positions)

    def _set_up_compression_flow(self, positions):
        """Sets up the flow of the region lambda is in (or entering) through the current positions."""
        potential = self._potential
        split = potential.compression_split
        self._flow_region = self._get_compression_region()
        self._synchronised_coupling = potential.ghost_coupling
        if self._flow_region is None:  # Not in compression flow region
            return
        lowest, highest = (-1.0, split) if self._flow_region == "inner" else (split, 1.0)
        if len(positions) == 1:  # No real particles: the ghost grows freely
            self._flow_chain, self._flow_rate_constant = None, 0.0
            self._free_length, self._free_length_slope = np.inf, 1.0
            velocities = np.zeros(1)
        else:
            window = self._compression_window if self._flow_region == "outer" else 0
            self._flow_chain = potential.get_compression_chain(positions, window)
            (self._flow_rate_constant, self._free_length, self._free_length_slope,
             velocities) = potential.get_compression_flow(positions, self._flow_chain, potential.ghost_coupling)
            if self._free_length <= 0.0:
                raise RuntimeError(f"The compression flow has no free length ({self._free_length}): reduce the "
                                   f"compression_contact of {potential.__class__.__name__}.")
        self._flow_lowest_coupling, self._flow_highest_coupling = potential.prepare_compression(
            positions, velocities, potential.ghost_coupling, lowest, highest, self._temperature)

    def _move_particles_along_compression_flow(self, positions):
        """Moves the particles along the flow to the current lambda (the flow line, and its cache, stay valid)."""
        potential = self._potential
        if (self._flow_region is not None and self._flow_chain is not None and
                potential.ghost_coupling != self._synchronised_coupling):
            potential.apply_compression(positions, self._flow_chain, self._synchronised_coupling,
                                        potential.ghost_coupling)
            self._free_length -= self._free_length_slope * (potential.ghost_coupling - self._synchronised_coupling)
        self._synchronised_coupling = potential.ghost_coupling

    def _attempt_insertion(self, positions):
        """Insertion attempt when lambda reaches 1 while growing."""
        self._count("insertion_attempts")
        if self._last_extreme == "bottom":
            self._count("upward_passages")
        self._last_extreme = "top"
        number_of_real_particles = len(positions) - 1
        ratio = (self._activity_times_volume / (number_of_real_particles + 2) *
                 np.exp(self._bias.get_value(self._lowest_coupling) - self._bias.get_value(1.0)))
        if np.random.uniform(0.0, 1.0) < ratio:
            self._count("insertion_acceptances")
            self._last_extreme = "bottom"
            self._potential.ghost_coupling = self._lowest_coupling  # New parked ghost at the bottom, still growing
            positions = np.vstack((positions, self._potential.get_random_position()))
            self._potential.get_ghost_profile(positions)  # Profile of the new ghost
            return positions
        self._coupling_direction = -1  # Flip velocity if insertion rejected
        return positions

    def _attempt_deletion(self, positions, active_particle_index):
        """Deletion attempt when lambda reaches -1 - parked_length while shrinking."""
        self._count("deletion_attempts")
        if self._last_extreme == "top":
            self._count("downward_passages")
        self._last_extreme = "bottom"
        number_of_real_particles = len(positions) - 1
        if active_particle_index == number_of_real_particles:  # Label on the parked ghost (always the case if N = 0)
            self._count("deletion_null_proposals")
            self._coupling_direction = 1  # Can't delete active particle so flip velocity and grow lambda
            return positions, active_particle_index
        ratio = ((number_of_real_particles + 1) / self._activity_times_volume *
                 np.exp(self._bias.get_value(1.0) - self._bias.get_value(self._lowest_coupling)))
        if np.random.uniform(0.0, 1.0) >= ratio:
            self._coupling_direction = 1
            return positions, active_particle_index
        self._count("deletion_acceptances")
        self._last_extreme = "top"
        chosen_index = np.random.randint(0, number_of_real_particles)
        last_real_index = number_of_real_particles - 1
        positions = positions[:-1]
        positions[[chosen_index, last_real_index]] = positions[[last_real_index, chosen_index]]
        if active_particle_index == chosen_index:
            active_particle_index = last_real_index
        elif active_particle_index == last_real_index:
            active_particle_index = chosen_index
        self._potential.ghost_coupling = 1.0
        self._potential.get_ghost_profile(positions)
        self._coupling_direction = -1
        return positions, active_particle_index

    def get_summary(self):
        return {"scheme": "tempered", "coupling_convention": "diameter", "log_activity": self._log_activity,
                "volume": self._potential.get_volume(),
                "parked_length": self._parked_length,
                "mean_distance_between_coupling_moves": self._mean_distance_between_coupling_moves,
                "mean_coupling_move_length": self._mean_coupling_move_length,
                "compression_flow": self._compression_flow, "compression_window": self._compression_window,
                "potential": self._potential.get_description(),
                "bias": self._bias.get_description(),
                "statistics": self._statistics}
