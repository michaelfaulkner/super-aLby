"""Module for the GrandCanonicalSoftSpherePotential class."""
from base.exceptions import ConfigurationError
from .grand_canonical_potential import GrandCanonicalPotential
from model_settings import size_of_particle_space
import numpy as np


class GrandCanonicalSoftSpherePotential(GrandCanonicalPotential):
    r"""
    Class for soft disks on a ring, u(r) = epsilon * (sigma / r) ** power, for every pair with the minimum-image
    convention (no cutoff), for use with EventChainMediator and a grand-canonical kernel.

    A ghost-real pair at coupling lambda > 0 has sigma -> lambda * sigma, i.e., u_g(r; lambda) = lambda ** power * u(r).
    Hence, at fixed positions, the ghost energy is U_g(lambda) = lambda ** power * A with A = sum_j u(r_gj) (the energy
    of a full particle at the ghost position), and dU_g / d lambda = power * lambda ** (power - 1) * A.
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 1.0, power: float = 12.0):
        """
        The constructor of the GrandCanonicalSoftSpherePotential class.

        Parameters
        ----------
        epsilon : float, optional
            The energy scale of the potential (0.0 gives the ideal gas).
        sigma : float, optional
            The length scale of the potential.
        power : float, optional
            The steepness of the potential (must be greater than 1 so that the ghost energy is smooth at lambda = 0).

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

    def _get_next_event_of_partners(self, separations, scales, temperature):
        """
        Returns the distance to the next factor event and the vetoing partner.  Each pair factor has its own exact
        (inverted) event distance: the event occurs when the pair energy has increased by an exponential random variable
        times the temperature.  A partner ahead at distance r is approached at once.  A partner behind first recedes
        (no events) until its minimum image flips to L / 2 ahead, after a distance L / 2 - |r|, and is then approached.
        Without any interaction (epsilon = 0) a null event one ring length ahead is returned.
        """
        if self._epsilon == 0.0:
            return size_of_particle_space[0], None
        prefactors = self._epsilon * (scales * self._sigma) ** self._power
        half_length = 0.5 * size_of_particle_space[0]
        ahead = separations > 0.0
        start = np.where(ahead, separations, half_length)
        offset = np.where(ahead, 0.0, half_length + separations)
        target_energy = (prefactors / start ** self._power -
                         np.log(np.random.uniform(0.0, 1.0, len(separations))) * temperature)
        distances = offset + start - (prefactors / target_energy) ** (1.0 / self._power)
        best = int(np.argmin(distances))
        return max(float(distances[best]), 0.0), best

    def get_ghost_profile(self, positions):
        """Returns sum over real particles the energy the ghost would have at lambda=1."""
        distances = self._get_ghost_distances(positions)
        # Set all scales to 1 including ghost particle as lambda=1
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
        return np.inf

    def get_description(self):
        return {"type": "soft_sphere", "epsilon": self._epsilon, "sigma": self._sigma, "power": self._power,
                "box": [float(size_of_particle_space[0])]}
