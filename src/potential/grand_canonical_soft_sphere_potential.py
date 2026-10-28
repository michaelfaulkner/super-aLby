"""Module for the GrandCanonicalSoftSpherePotential class."""
from base.exceptions import ConfigurationError
from .grand_canonical_potential import GrandCanonicalPotential
from model_settings import size_of_particle_space
import numpy as np


class GrandCanonicalSoftSpherePotential(GrandCanonicalPotential):
    r"""
    Class for soft disks on a ring, U(r) = epsilon * (sigma / r) ** power, for use with EventChainMediator or
    MetropolisMediator and a grand-canonical kernel.

    A ghost-real pair at coupling lambda > 0 has sigma -> lambda * sigma, i.e. U_g(r, lambda) = lambda ** power * U(r).
    Hence, at fixed positions, the ghost energy is U_g(lambda) = lambda ** power * A with A = sum_j u(r_gj) (the energy
    of a full particle at the ghost position), and dU_g / d lambda = power * lambda ** (power - 1) * A.
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, power: float = 12.0):
        """
        The constructor of the GrandCanonicalSoftSpherePotential class.

        Parameters
        ----------
        epsilon : float, optional
            The energy scale of the potential.
        sigma : float, optional
            The length scale of the potential.
        power : float, optional
            The steepness of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If epsilon < 0, sigma <= 0 or power <= 1.
        """
        super().__init__(prefactor=1.0)
        if epsilon < 0.0 or sigma <= 0.0 or power <= 1.0:
            raise ConfigurationError(f"Give epsilon >= 0, sigma > 0 and power > 1 in {self.__class__.__name__}.")
        self._epsilon = epsilon
        self._sigma = sigma
        self._power = power

    def _get_pair_energies(self, distances, scales):
        return self._epsilon * (scales * self._sigma / distances) ** self._power

    def _get_next_event_from_separations(self, separations, scales, temperature):
        """
        Returns the distance to the next factor event and the vetoing particle.
        """
        if self._epsilon == 0.0:  # Ideal gas reference
            return size_of_particle_space[0], None
        half_length = 0.5 * size_of_particle_space[0]
        ahead = separations > 0.0
        abs_separations = np.asarray(np.where(ahead, separations, half_length))
        free_distances = np.asarray(np.where(ahead, 0.0, half_length + separations))
        alpha = (-np.log(np.random.uniform(0.0, 1.0, len(separations))) * temperature /
                 (self._epsilon * (scales * self._sigma) ** self._power) + 1 / abs_separations ** self._power)
        distances = free_distances + abs_separations - alpha ** (-1 / self._power)
        vetoing_index = int(np.argmin(distances))
        return max(float(distances[vetoing_index]), 0.0), vetoing_index

    def get_ghost_profile(self, positions):
        """Returns sum over real particles the energy the ghost would have at lambda=1."""
        distances = self._get_ghost_distances(positions)
        # Set all scales to 1 including ghost particle as lambda=1.
        self._ghost_profile = float(np.sum(self._get_pair_energies(distances, np.ones(len(distances)))))
        return self._ghost_profile

    def get_ghost_energy(self, coupling, profile):
        coupling = np.asarray(coupling, dtype=float)
        return np.where(coupling > 0.0, coupling ** self._power * profile, 0.0)

    def get_ghost_force(self, coupling):
        return self._power * coupling ** (self._power - 1.0) * self._ghost_profile

    def get_next_coupling_event(self, coupling, temperature, direction):
        if direction < 0 or self._ghost_profile <= 0.0:  
            return np.inf
        target = coupling ** self._power - np.log(np.random.uniform(0.0, 1.0)) * temperature / self._ghost_profile
        return target ** (1.0 / self._power) - coupling

    def get_ghost_wall(self):
        return np.inf  # No hard walls for soft sphere potential

    def choose_coupling_event_particle(self, positions):
        """d u_j / d lambda is proportional to u_j at full coupling, i.e. to r_j ** (- power)."""
        separations = self._get_minimum_image(positions[:-1, 0] - positions[-1, 0])
        weights = (np.min(np.abs(separations)) / np.abs(separations)) ** self._power
        index = int(np.random.choice(len(weights), p=weights / np.sum(weights)))
        return index, float(separations[index])

    def get_lift_ratio(self, distance, coupling):
        """u = epsilon * (lambda * sigma / r) ** power, so |du/dr| / (du/dlambda) = lambda / r."""
        return coupling / distance

    def get_description(self):
        return {"type": "soft_sphere", "epsilon": self._epsilon, "sigma": self._sigma, "power": self._power,
                "box": [float(size_of_particle_space[0])]}
