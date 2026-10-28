"""Module for the GrandCanonicalHardSpherePotential class."""
from base.exceptions import ConfigurationError
from .grand_canonical_potential import GrandCanonicalPotential
from model_settings import size_of_particle_space
import numpy as np


class GrandCanonicalHardSpherePotential(GrandCanonicalPotential):
    r"""
    Class for hard disks on a ring, for use with EventChainMediator or MetropolisMediator and a grand-canonical
    kernel.
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
            If diameter is not positive or not less than half the size_of_particle_space.
        """
        super().__init__(prefactor=1.0)
        if diameter <= 0.0 or diameter >= 0.5 * size_of_particle_space[0]:
            raise ConfigurationError(f"Give 0 < diameter < size_of_particle_space / 2 in {self.__class__.__name__}.")
        self._diameter = diameter

    def get_initial_positions(self):
        """
        Returns the initial positions.

        Returns
        -------
        numpy.ndarray
            Array of shape (number_of_rows, 1).
        """
        number_of_rows = self._initial_number_of_real_particles + (1 if self.has_ghost() else 0)
        spacing = size_of_particle_space[0] / number_of_rows
        if spacing <= self._diameter:
            raise ConfigurationError(f"The initial lattice of {self.__class__.__name__} has spacing {spacing}, so the "
                                     f"ring of touching disks is jammed.  Use a larger size_of_particle_space or fewer "
                                     f"particles.")
        positions = ((np.arange(number_of_rows) + 0.5) * spacing)[:, None]
        self._check_initial_positions(positions)
        return positions

    def _get_pair_energies(self, distances, scales):
        return np.where(distances < scales * self._diameter, np.inf, 0.0)

    def _get_next_event_from_separations(self, separations, scales, temperature):
        """
        Returns the distance to the next collision and the particle the active particle collides with.
        """
        contacts = scales * self._diameter
        distances = np.where(separations > 0.0, separations, size_of_particle_space[0] + separations) - contacts
        vetoing_index = int(np.argmin(distances))
        return max(float(distances[vetoing_index]), 0.0), vetoing_index

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
        return np.inf

    def get_ghost_wall(self):
        return self._ghost_profile

    def choose_coupling_event_particle(self, positions):
        """Used during lifting moves from lambda space to particle space to select which particle subsequently
        takes over."""
        separations = self._get_minimum_image(positions[:-1, 0] - positions[-1, 0])
        index = int(np.argmin(np.abs(separations)))
        return index, float(separations[index])

    def get_lift_ratio(self, distance, coupling):
        """The contact gap r - lambda * sigma closes at speed 1 in r and sigma in lambda."""
        return 1.0 / self._diameter

    def get_description(self):
        return {"type": "hard_sphere", "diameter": self._diameter, "box": [float(size_of_particle_space[0])]}
