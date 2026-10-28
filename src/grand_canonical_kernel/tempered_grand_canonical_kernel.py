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

    The ghost is the last row of positions. Its coupling lambda lies in [-parked_length, 1] and moves at speed v_lambda.
    The augmented target is z^N / N! * exp(- beta * U(x, y; lambda) + eta(lambda)) / V, uniform over the
    N + 1 event-chain labels and the directions. eta is given by ghost_coupling_bias.

    Kernel events (coupling moves) start after exponentially distributed distances through the state space with mean
    normalised_distance_between_coupling_moves * number_of_particles. During coupling moves the particles are frozen
    and lambda moves at unit speed for an exponentially distributed distance with mean mean_coupling_move_length:
      - flips of v_lambda at rate [v_lambda * (beta * dU_g / d lambda - g(lambda))]^+, sampled by thinning
        the sum of the neighbour rate [v_lambda * beta * dU_g / d lambda]^+ and the bias rate [- v_lambda * g(lambda)]^+
      - a hard wall at the ghost clearance (only for hard spheres)
      - uniform resampling of the ghost position when lambda crosses 0 upwards
      - insertion at lambda = 1: the ghost becomes real and a new parked ghost appears at a uniform position at
        lambda = -parked_length, accepted with min(1, z V / (N + 2) * exp(eta(- parked_length) - eta(1)));
      - deletion at lambda = -parked_length: null if the event-chain label is on the parked ghost, else the parked
        ghost is removed and a uniformly chosen real particle becomes the ghost at lambda = 1, accepted with
        min(1, (N + 1) / (z V) * exp(eta(1) - eta(- parked_length))). A rejected jump flips v_lambda.
    """

    def __init__(self, ghost_coupling_bias: GhostCouplingBias = NoGhostCouplingBias(), log_activity: float = 0.0,
                 parked_length: float = 1.0, normalised_distance_between_coupling_moves: float = 0.05,
                 mean_coupling_move_length: float = 1.0, coupled_lifting: bool = True,
                 redraw_ghost: bool = False):
        """
        The constructor of the TemperedGrandCanonicalKernel class.

        Parameters
        ----------
        ghost_coupling_bias : ghost_coupling_bias.ghost_coupling_bias.GhostCouplingBias, optional
            The bias eta(lambda); its slopes are the g values.
        log_activity : float, optional
            log z = beta * mu.
        parked_length : float, optional
            Length of the parked interval [- parked_length, 0] of lambda.
        normalised_distance_between_coupling_moves : float, optional
            Mean distance through the state space between coupling moves, per particle.
        mean_coupling_move_length : float, optional
            Mean distance travelled by lambda during one set of coupling moves.
        coupled_lifting : bool, optional
            If True, ghost-pair events can lift between lambda and the particles.
        redraw_ghost : bool, optional
            If True, the ghost position is redrawn uniformly whenever lambda crosses 0 upwards.

        Raises
        ------
        base.exceptions.ConfigurationError
            If ghost_coupling_bias is not a GhostCouplingBias, or if any distance is not greater than 0.0.
        """
        super().__init__()
        if not isinstance(ghost_coupling_bias, GhostCouplingBias):
            raise ConfigurationError(f"Give a GhostCouplingBias child class as ghost_coupling_bias in "
                                     f"{self.__class__.__name__}.")
        if min(parked_length, normalised_distance_between_coupling_moves, mean_coupling_move_length) <= 0.0:
            raise ConfigurationError(f"Give values greater than 0.0 for parked_length, "
                                     f"normalised_distance_between_coupling_moves and mean_coupling_move_length in "
                                     f"{self.__class__.__name__}.")
        self._bias = ghost_coupling_bias
        self._log_activity = log_activity
        self._parked_length = parked_length
        self._mean_distance_between_coupling_moves = normalised_distance_between_coupling_moves * number_of_particles
        self._mean_coupling_move_length = mean_coupling_move_length
        self._coupled_lifting = coupled_lifting
        self._redraw_ghost = redraw_ghost
        self._lifted_from_particles = False
        self._activity_times_volume = None
        self._coupling_direction = 1  # Lambda space velocity
        self._last_extreme = None
        self._statistics = {key: 0 for key in [
            "coupling_moves", "insertion_attempts", "insertion_acceptances", "deletion_attempts",
            "deletion_null_proposals", "deletion_acceptances", "coupling_bias_flips", "coupling_pair_flips",
            "coupling_thinning_rejections",
            "coupling_wall_flips", "ghost_redraws", "upward_passages", "downward_passages", "lifts_to_particle",
            "lifts_to_coupling"]}

    def initialize(self, potential, temperature):
        super().initialize(potential, temperature)
        potential.ghost_coupling = - 0.5 * self._parked_length
        self._activity_times_volume = float(np.exp(self._log_activity)) * potential.get_volume()  # z * V

    def get_distance_to_next_event(self):
        return -np.log(np.random.uniform(0.0, 1.0)) * self._mean_distance_between_coupling_moves

    def handle_event(self, positions, active_particle_index, movement_direction):
        """
        Runs one set of coupling moves: the particles are frozen while lambda travels an exponentially distributed
        distance with mean mean_coupling_move_length.
        """
        self._count("coupling_moves")
        if self._lifted_from_particles:
            # A particle hit the ghost and lifted into lambda so lambda shrinks and the active particle resampled
            self._lifted_from_particles = False
            self._coupling_direction = -1
            active_particle_index = np.random.randint(0, len(positions))
        # How far we can move through lambda space before returning to particle state space
        remaining_length = -np.log(np.random.uniform(0.0, 1.0)) * self._mean_coupling_move_length
        # Cache ghost profile (fixed while particles are frozen)
        self._potential.get_ghost_profile(positions)
        beta = 1.0 / self._temperature
        while True:
            coupling = self._potential.ghost_coupling
            if self._coupling_direction > 0:
                if coupling < 0.0:  # Moving through lambda < 0 is free as ghost potential is zero
                    next_stop, kind = 0.0, "reach_zero"  # Ghost position redrawn at lambda = 0 (redraw_ghost)
                else:
                    next_stop, kind = 1.0, "top"
                    wall = self._potential.get_ghost_wall()  # Value of lambda that ghost hits hard stop (hard spheres)
                    if wall <= next_stop:  # Do we hit a hard wall before reaching lambda = 1?
                        next_stop, kind = wall, "wall"
            else:
                if coupling > 0.0:
                    next_stop, kind = 0.0, "park"  # Stop at 0: below it the ghost no longer interacts
                else:
                    next_stop, kind = -self._parked_length, "bottom"
            if kind in ("reach_zero", "bottom"):  # Ghost particle not interacting
                proposal_distance = pair_distance = np.inf
            else:
                # Next flip proposal = the earlier of the ghost-pair event and the bias event
                pair_distance = self._potential.get_next_coupling_event(coupling, self._temperature,
                                                                        self._coupling_direction)
                proposal_distance = min(pair_distance, self._bias.get_next_event(coupling, self._coupling_direction))
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
                if kind == "reach_zero" and self._redraw_ghost:
                    self._count("ghost_redraws")
                    positions[-1] = self._potential.get_random_position()
                    self._potential.get_ghost_profile(positions)  # Profile of the redrawn ghost
                elif kind == "wall":
                    lifted_index = (self._attempt_lift_to_particle(positions, new_coupling)
                                    if self._coupled_lifting else None)
                    if lifted_index is not None:
                        return positions, lifted_index, movement_direction
                    self._count("coupling_wall_flips")
                    self._coupling_direction = -1
                elif kind == "top":
                    positions = self._attempt_insertion(positions)
                elif kind == "bottom":  # Deletion attempt (the label stays with its particle)
                    positions, active_particle_index = self._attempt_deletion(positions, active_particle_index)
            elif self._coupled_lifting:
                # Ghost pairs and the bias are separate factors (no thinning)
                if proposal_distance == pair_distance:
                    lifted_index = self._attempt_lift_to_particle(positions, new_coupling)
                    if lifted_index is not None:
                        return positions, lifted_index, movement_direction
                    self._count("coupling_pair_flips")
                else:
                    self._count("coupling_bias_flips")
                self._coupling_direction = -self._coupling_direction
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

    def _get_lift_constant(self, number_of_rows):
        """
        K = (weight of one particle direction) / (weight of one lambda direction) = 2 * (mean particle distance between
        coupling moves) / (mean coupling move length * number_of_rows). The particles and lambda share the time in the
        ratio of their switching distances, the particle share is uniform over the rows and the lambda share over
        the two directions.
        """
        return 2.0 * self._mean_distance_between_coupling_moves / (self._mean_coupling_move_length * number_of_rows)

    def _attempt_lift_to_particle(self, positions, coupling):
        """
        Lifts from growing lambda to the particle moving away from the ghost pair that caused the event, with
        probability min(1, K * |du/dr| / (du/dlambda)).  Returns its index, or None (lambda then flips).  The lambda
        direction stored for the next coupling moves is redrawn.
        """
        index, separation = self._potential.choose_coupling_event_particle(positions)
        lifted_index = index if separation > 0.0 else len(positions) - 1
        ratio = self._get_lift_constant(len(positions)) * self._potential.get_lift_ratio(abs(separation), coupling)
        if np.random.uniform(0.0, 1.0) >= ratio:
            return None
        self._count("lifts_to_particle")
        self._coupling_direction = 1 if np.random.uniform(0.0, 1.0) < 0.5 else -1
        return lifted_index

    def lifts_into_coupling(self, positions, active_particle_index, vetoing_index, hop_displacement):
        """
        At a ghost-pair event of a particle moving towards the interacting ghost, lifts into shrinking lambda with
        probability min(1, 1 / (K * |du/dr| / (du/dlambda))). Otherwise, the usual lift to the vetoing particle.
        """
        ghost_index = len(positions) - 1
        if (not self._coupled_lifting or active_particle_index == vetoing_index or
                ghost_index not in (active_particle_index, vetoing_index) or self._potential.ghost_coupling <= 0.0):
            return False
        ratio = self._get_lift_constant(len(positions)) * self._potential.get_lift_ratio(abs(hop_displacement),
                                                                                         self._potential.ghost_coupling)
        if np.random.uniform(0.0, 1.0) >= 1.0 / ratio:
            return False
        self._count("lifts_to_coupling")
        self._lifted_from_particles = True
        return True

    def _attempt_insertion(self, positions):
        """Insertion attempt when lambda reaches 1 while growing."""
        self._count("insertion_attempts")
        if self._last_extreme == "bottom":
            self._count("upward_passages")
        self._last_extreme = "top"
        number_of_real_particles = len(positions) - 1
        ratio = (self._activity_times_volume / (number_of_real_particles + 2) *
                 np.exp(self._bias.get_value(-self._parked_length) - self._bias.get_value(1.0)))
        if np.random.uniform(0.0, 1.0) < ratio:
            self._count("insertion_acceptances")
            self._last_extreme = "bottom"
            self._potential.ghost_coupling = -self._parked_length  # New ghost parked at -parked_length, still growing
            positions = np.vstack((positions, self._potential.get_random_position()))
            self._potential.get_ghost_profile(positions)  # Profile of the new ghost (needed when redraw_ghost = False)
            return positions
        self._coupling_direction = -1  # Flip velocity if insertion rejected
        return positions

    def _attempt_deletion(self, positions, active_particle_index):
        """Deletion attempt when lambda reaches -parked_length while shrinking."""
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
                 np.exp(self._bias.get_value(1.0) - self._bias.get_value(-self._parked_length)))
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
        return {"scheme": "tempered", "log_activity": self._log_activity, "volume": self._potential.get_volume(),
                "parked_length": self._parked_length,
                "mean_distance_between_coupling_moves": self._mean_distance_between_coupling_moves,
                "mean_coupling_move_length": self._mean_coupling_move_length, "coupled_lifting": self._coupled_lifting,
                "redraw_ghost": self._redraw_ghost,
                "potential": self._potential.get_description(),
                "bias": self._bias.get_description(),
                "statistics": self._statistics}
