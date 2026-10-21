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

    The ghost is the last row of positions; its coupling lambda lies in [- parked_length, 1] and moves with direction
    sigma_lambda.  The augmented target is z^N / N! * exp(- beta * U(x, y; lambda) + eta(lambda)) / V, uniform over the
    N + 1 event-chain labels and the directions; eta is given by ghost_coupling_bias (eta = 0 for lambda <= 0, where
    the ghost is parked and non-interacting, so observations with lambda <= 0 are exact grand-canonical samples).

    Kernel events (coupling moves) start after exponentially distributed distances through the state space with mean
    normalised_distance_between_coupling_moves * number_of_particles. During coupling moves the particles are frozen
    and lambda moves at unit speed for an exponentially distributed distance with mean mean_coupling_move_length:
      * flips of sigma_lambda at rate [sigma_lambda * (beta * dU_g / d lambda - g(lambda))]^+, sampled by thinning
        the sum of the neighbour rate [sigma_lambda * beta * dU_g / d lambda]^+ and the bias rate
        [- sigma_lambda * g(lambda)]^+ (valid for repulsive and attractive ghost interactions);
      * a hard wall at the ghost clearance (hard spheres);
      * a uniform redraw of the ghost position when lambda crosses 0 upwards;
      * insertion at lambda = 1: the ghost becomes real and a new parked ghost appears at a uniform position at
        lambda = - parked_length, accepted with min(1, z V / (N + 2) * exp(eta(- parked_length) - eta(1)));
      * deletion at lambda = - parked_length: null if the event-chain label is on the parked ghost, else the parked
        ghost is removed and a uniformly chosen real particle becomes the ghost at lambda = 1, accepted with
        min(1, (N + 1) / (z V) * exp(eta(1) - eta(- parked_length))) (N real particles before the deletion);
      a rejected jump flips sigma_lambda.
    """

    def __init__(self, ghost_coupling_bias: GhostCouplingBias = NoGhostCouplingBias(), log_activity: float = 0.0,
                 parked_length: float = 1.0, normalised_distance_between_coupling_moves: float = 0.05,
                 mean_coupling_move_length: float = 1.0):
        """
        The constructor of the TemperedGrandCanonicalKernel class.

        Parameters
        ----------
        ghost_coupling_bias : ghost_coupling_bias.ghost_coupling_bias.GhostCouplingBias, optional
            The bias eta(lambda); its slopes are the g values.
        log_activity : float, optional
            log z = beta * mu (thermal wavelength 1).
        parked_length : float, optional
            Length of the parked interval [- parked_length, 0] of lambda.
        normalised_distance_between_coupling_moves : float, optional
            Mean distance through the state space between coupling moves, per particle.
        mean_coupling_move_length : float, optional
            Mean distance travelled by lambda during one set of coupling moves.

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
        self._activity_times_volume = None
        self._coupling_direction = 1  # Lambda space velocity
        self._last_extreme = None
        self._statistics = {key: 0 for key in [
            "coupling_moves", "insertion_attempts", "insertion_acceptances", "deletion_attempts",
            "deletion_null_proposals", "deletion_acceptances", "coupling_bias_flips", "coupling_thinning_rejections",
            "coupling_wall_flips", "ghost_redraws", "upward_passages", "downward_passages"]}

    def initialize(self, potential, temperature):
        super().initialize(potential, temperature)
        potential.ghost_coupling = - 0.5 * self._parked_length
        self._activity_times_volume = float(np.exp(self._log_activity)) * potential.volume

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
        # Establish lambda landscape
        self._potential.get_ghost_profile(positions)
        beta = 1.0 / self._temperature
        while True:
            coupling = self._potential.ghost_coupling
            if self._coupling_direction > 0:
                if coupling < 0.0:
                    segment_end, slope, kind = 0.0, 0.0, "reach_zero"  # Ghost position resampled at lambda=0
                else:
                    # Identify biasing associated with current value of lambda
                    segment_end, slope = self._bias.get_segment(coupling, 1)
                    kind = "top" if segment_end == 1.0 else "node"
                    wall = self._potential.get_ghost_wall()  # Value of lambda that ghost hits hard stop.
                    if wall <= segment_end:  # Do we hit a hard wall before we leave the segment?
                        segment_end, kind = wall, "wall"
            else:
                if coupling > 0.0:
                    segment_end, slope = self._bias.get_segment(coupling, -1)
                    kind = "node"  # No reach_zero event as no need to resample position when lambda becomes < 0
                else:
                    segment_end, slope, kind = -self._parked_length, 0.0, "bottom"
            if kind in ("reach_zero", "bottom"):  # Ghost particle not interacting
                proposal_distance = np.inf
            else:
                # Propose distance to lambda velocity flip. Factorise the coupling potential into the neighbour factor
                # with rate [v * beta * dU_g / d lambda]^+ and the bias factor with rate [-v * g]^+ and take the earlier
                # event. Their sum bounds the flip rate [v * (beta * dU_g / d lambda - g)]^+, and Poisson thinning below
                # reduces it to that rate.
                bias_rate = max(-self._coupling_direction * slope, 0.0)
                proposal_distance = min(
                    self._potential.get_next_coupling_event(coupling, self._temperature, self._coupling_direction),
                    -np.log(np.random.uniform(0.0, 1.0)) / bias_rate if bias_rate > 0.0 else np.inf)
            candidates = [abs(segment_end - coupling), remaining_length, proposal_distance]
            choice = int(np.argmin(candidates))
            step = candidates[choice]
            new_coupling = segment_end if choice == 0 else coupling + self._coupling_direction * step
            self._potential.ghost_coupling = new_coupling
            remaining_length -= step
            if choice == 1:  # Return to particle space moves
                return positions, active_particle_index, movement_direction
            if choice == 0:
                if kind == "reach_zero":
                    self._count("ghost_redraws")
                    # Resample position as lambda hit zero from below
                    positions[-1] = self._potential.get_random_position()
                    self._potential.get_ghost_profile(positions)
                elif kind == "wall":
                    self._count("coupling_wall_flips")
                    self._coupling_direction = -1
                elif kind == "top":
                    positions = self._attempt_insertion(positions)
                elif kind == "bottom":  # If deleted active particle must select a new one
                    positions, active_particle_index = self._attempt_deletion(positions, active_particle_index)
            else:
                force_rate = beta * self._potential.get_ghost_force(new_coupling)
                rate = max(0.0, self._coupling_direction * (force_rate - slope))
                rate_bound = (max(self._coupling_direction * force_rate, 0.0) +
                              max(- self._coupling_direction * slope, 0.0))
                if np.random.uniform(0.0, 1.0) * rate_bound < rate:
                    self._count("coupling_bias_flips")
                    self._coupling_direction = -self._coupling_direction
                else:
                    self._count("coupling_thinning_rejections")

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
            self._potential.ghost_coupling = - self._parked_length
            return np.vstack((positions, self._potential.get_random_position()))
        self._coupling_direction = -1
        return positions

    def _attempt_deletion(self, positions, active_particle_index):
        """Deletion attempt when lambda reaches - parked_length while shrinking."""
        self._count("deletion_attempts")
        if self._last_extreme == "top":
            self._count("downward_passages")
        self._last_extreme = "bottom"
        number_of_real_particles = len(positions) - 1
        if active_particle_index == number_of_real_particles:  # Label on the parked ghost (always the case if N = 0)
            self._count("deletion_null_proposals")
            self._coupling_direction = 1
            return positions, active_particle_index
        ratio = ((number_of_real_particles + 1) / self._activity_times_volume *
                 np.exp(self._bias.get_value(1.0) - self._bias.get_value(- self._parked_length)))
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
        return {"scheme": "tempered", "log_activity": self._log_activity, "volume": self._potential.volume,
                "parked_length": self._parked_length,
                "mean_distance_between_coupling_moves": self._mean_distance_between_coupling_moves,
                "mean_coupling_move_length": self._mean_coupling_move_length,
                "potential": self._potential.get_description(),
                "bias": {"class": self._bias.__class__.__name__, "nodes": self._bias.nodes.tolist(),
                         "slopes": self._bias.slopes.tolist(), "total": self._bias.total},
                "statistics": self._statistics}
