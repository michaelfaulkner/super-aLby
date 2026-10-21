"""Module for the GrandCanonicalHardSpherePotential class."""
from base.exceptions import ConfigurationError
from .grand_canonical_potential import GrandCanonicalPotential
from model_settings import size_of_particle_space
import numpy as np


class GrandCanonicalHardSpherePotential(GrandCanonicalPotential):
    r"""
    Class for hard disks of diameter sigma on a ring (the Tonks gas), for use with EventChainMediator and a
    grand-canonical kernel.

    The ghost is non-additive: a ghost-real pair at coupling lambda > 0 has contact distance lambda * sigma.  At
    lambda <= 0 the ghost is a non-interacting point, so it can be placed anywhere; at lambda = 1 it is a full disk.
    At fixed positions, the ghost can grow up to its clearance c = min_j r_gj / sigma, beyond which its energy is
    infinite, i.e., c acts as a hard wall on lambda.
    """

    def __init__(self, diameter: float = 1.0):
        """
        The constructor of the GrandCanonicalHardSpherePotential class.

        Parameters
        ----------
        diameter : float, optional
            The disk diameter sigma (the real-real contact distance).

        Raises
        ------
        base.exceptions.ConfigurationError
            If diameter is not positive or not less than half the ring length.
        """
        super().__init__(prefactor=1.0)
        if diameter <= 0.0 or diameter >= 0.5 * size_of_particle_space[0]:
            raise ConfigurationError(f"Give 0 < diameter < (ring length) / 2 in {self.__class__.__name__}.")
        self._diameter = diameter

    def _check_initial_positions(self, positions, spacing):
        """
        Also raises a ConfigurationError if the initial lattice is jammed, i.e., its spacing equals the diameter: a full
        ring of touching disks cannot move, and event chains would lift around it forever without moving.
        """
        super()._check_initial_positions(positions, spacing)
        if spacing <= self._diameter * (1.0 + 1.0e-12):
            raise ConfigurationError(f"The initial lattice of {self.__class__.__name__} has spacing {spacing}, so the "
                                     f"ring of touching disks is jammed.  Use a longer ring or fewer particles.")

    def _get_pair_energies(self, distances, scales):
        return np.where(distances < scales * self._diameter, np.inf, 0.0)

    def _get_next_event_of_partners(self, separations, scales, temperature):
        """
        Returns the distance to the next collision and the partner the active disk collides with.  Partners behind the
        active disk are reached after going around the ring.
        """
        contacts = scales * self._diameter
        distances = np.where(separations > 0.0, separations, size_of_particle_space[0] + separations) - contacts
        best = int(np.argmin(distances))
        return max(float(distances[best]), 0.0), best

    def get_ghost_profile(self, positions):
        """Returns (and caches) the distance min_j r_gj / sigma in lambda space that the ghost particle can expand."""
        if len(positions) == 1:
            self._ghost_profile = np.inf
        else:
            self._ghost_profile = float(np.min(self._get_ghost_distances(positions)) / self._diameter)
        return self._ghost_profile

    def get_ghost_energy(self, coupling, profile):
        coupling = np.asarray(coupling, dtype=float)
        return np.where(coupling > profile, np.inf, 0.0)

    def get_ghost_force(self, coupling):
        return 0.0

    def get_next_coupling_event(self, coupling, temperature, direction):
        return np.inf  # Only interactions for hard spheres are hard collisions handled in self.get_ghost_wall()

    def get_ghost_wall(self):
        return self._ghost_profile

    def get_description(self):
        return {"type": "hard_sphere", "diameter": self._diameter, "box": [float(size_of_particle_space[0])]}
