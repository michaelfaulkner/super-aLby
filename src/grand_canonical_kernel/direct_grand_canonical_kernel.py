"""Module for the DirectGrandCanonicalKernel class."""
from base.exceptions import ConfigurationError
from model_settings import number_of_particles
from .grand_canonical_kernel import GrandCanonicalKernel
import numpy as np


class DirectGrandCanonicalKernel(GrandCanonicalKernel):
    r"""
    Class for direct (untempered) insertions and deletions with no ghost particle.

    Kernel events are separated by exponentially distributed distances through the state space with mean
    normalised_distance_between_direct_moves * number_of_particles (the initial number of particles).  At each event,
    with probability 1/2, a particle is proposed at a uniform position and accepted with
    min(1, z V exp(-beta dU) / (N + 1)) otherwise a uniformly chosen non-active particle is proposed for deletion and
    accepted with min(1, N exp(beta dU) / (z V)).  With MetropolisMediator the deleted particle is chosen among all N,
    with the same acceptance probabilities.
    """

    def __init__(self, log_activity: float = 0.0, normalised_distance_between_direct_moves: float = 0.05):
        """
        The constructor of the DirectGrandCanonicalKernel class.

        Parameters
        ----------
        log_activity : float, optional
            log z = beta * mu.
        normalised_distance_between_direct_moves : float, optional
            Mean distance through the state space between direct moves, per particle.

        Raises
        ------
        base.exceptions.ConfigurationError
            If normalised_distance_between_direct_moves is not greater than 0.0.
        """
        super().__init__()
        if normalised_distance_between_direct_moves <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for normalised_distance_between_direct_moves in "
                                     f"{self.__class__.__name__}.")
        self._log_activity = log_activity
        self._mean_distance_between_direct_moves = normalised_distance_between_direct_moves * number_of_particles
        self._activity_times_volume = None
        self._statistics = {key: 0 for key in ["direct_insertion_attempts", "direct_insertion_acceptances",
                                               "direct_deletion_attempts", "direct_deletion_null_proposals",
                                               "direct_deletion_acceptances"]}

    def initialize(self, potential, temperature):
        super().initialize(potential, temperature)
        potential.ghost_coupling = None
        self._activity_times_volume = float(np.exp(self._log_activity)) * potential.get_volume()

    def get_distance_to_next_event(self):
        return -np.log(np.random.uniform(0.0, 1.0)) * self._mean_distance_between_direct_moves

    def handle_event(self, positions, active_particle_index, movement_direction):
        beta = 1.0 / self._temperature
        number_of_real_particles = len(positions)
        # Attempt insertion/deletion with p=0.5 each
        if np.random.uniform(0.0, 1.0) < 0.5:
            self._count("direct_insertion_attempts")
            candidate_position = self._potential.get_random_position()
            energy = self._potential.get_insertion_energy(positions, candidate_position)
            # Grand canonical insertion acceptance probability
            if np.random.uniform(0.0, 1.0) < (
                    self._activity_times_volume * np.exp(-beta * energy) / (number_of_real_particles + 1)):
                self._count("direct_insertion_acceptances")
                positions = np.vstack((positions, candidate_position))
            return positions, active_particle_index, movement_direction
        self._count("direct_deletion_attempts")
        if active_particle_index is None:  # Metropolis mediator used
            if number_of_real_particles == 0:
                self._count("direct_deletion_null_proposals")
                return positions, active_particle_index, movement_direction
            chosen_index = np.random.randint(0, number_of_real_particles)
        else:
            if number_of_real_particles <= 1:  # Can't delete only (and active) particle
                self._count("direct_deletion_null_proposals")
                return positions, active_particle_index, movement_direction
            chosen_index = np.random.randint(0, number_of_real_particles - 1)
            if chosen_index >= active_particle_index:  # Avoid picking the active particle
                chosen_index += 1
        energy = self._potential.get_single_particle_energy(positions, chosen_index)
        # Grand canonical deletion acceptance probability
        if np.random.uniform(0.0, 1.0) < number_of_real_particles * np.exp(beta * energy) / self._activity_times_volume:
            self._count("direct_deletion_acceptances")
            last_index = number_of_real_particles - 1
            # Move particle to be deleted to end then pop
            positions[[chosen_index, last_index]] = positions[[last_index, chosen_index]]
            if active_particle_index == last_index:
                active_particle_index = chosen_index
            positions = positions[:-1]
        return positions, active_particle_index, movement_direction

    def get_summary(self):
        return {"scheme": "direct", "log_activity": self._log_activity, "volume": self._potential.get_volume(),
                "mean_distance_between_direct_moves": self._mean_distance_between_direct_moves,
                "potential": self._potential.get_description(), "statistics": self._statistics}
