"""Module for the TetheredHarmonicChainPotential class."""
import numpy as np
from .harmonic_chain_potential import HarmonicChainPotential
from model_settings import number_of_particles, size_of_particle_space


class TetheredHarmonicChainPotential(HarmonicChainPotential):
    """
    This class implements the harmonic-chain potential
        U = prefactor * sum((x[i] - x[i-1] - equilibrium_length) ** 2) / 2 for a non-toroidal chain in the box [0, L]:
        particle 0 is tethered by an identical spring to a fixed anchor at x = 0 and particle N - 1 to a fixed anchor
        at x = L.  There are therefore N + 1 springs whose lengths sum to L, which is the box analogue of the constraint
        sum(x_tilde[i]) = L of the toroidal HarmonicChainPotential.

    In event-chain Monte Carlo, an event triggered by a tether spring has no particle to lift to, so the active
        particle remains active and its direction of motion is reversed.  Such events are signalled to the mediator by
        a veto index lying outside the range [0, N - 1] (-1 for the anchor at 0; N for the anchor at L).
    """

    def __init__(self, prefactor: float = 1.0, equilibrium_length: float = 0.0, use_cell_horizon: bool = False,
                 cell_horizon: float = 1.0):
        """
        The constructor of the TetheredHarmonicChainPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        equilibrium_length : float
            The separation of particles associated with the minimum of the potential (the factor field is
            prefactor * equilibrium_length).
        use_cell_horizon : bool
            Determines whether to use cell horizon method.
        cell_horizon : float
            Horizon over which to measure maximum potential gradient.
        """
        super().__init__(prefactor=prefactor, equilibrium_length=equilibrium_length,
                         use_cell_horizon=use_cell_horizon, cell_horizon=cell_horizon)

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
        extended_positions = np.concatenate(([0.0], positions[:, 0], [size_of_particle_space[0]]))
        return self._potential_constant * np.sum((np.diff(extended_positions) - self._equilibrium_length) ** 2)

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
        extended_positions = np.concatenate(([0.0], positions[:, 0], [size_of_particle_space[0]]))
        spring_extensions = np.diff(extended_positions) - self._equilibrium_length
        return (2.0 * self._potential_constant * (spring_extensions[:-1] - spring_extensions[1:])).reshape(-1, 1)

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.  If the event was triggered by a
        tether spring (veto_index outside [0, N - 1]), the active particle remains active and reverses direction.

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

    @staticmethod
    def _get_neighbours(active_particle_index):
        """
        Return indices of neighbours to active particle.  The neighbour 'beyond' either end of the chain is a fixed
        anchor, labelled -1 (at x = 0) or number_of_particles (at x = L).
        """
        return active_particle_index - 1, active_particle_index + 1

    @staticmethod
    def _get_neighbour_positions(positions, active_particle_index, neg_neighbour_index, pos_neighbour_index):
        neg_neighbour_position = positions[neg_neighbour_index][0] if neg_neighbour_index >= 0 else 0.0
        pos_neighbour_position = (positions[pos_neighbour_index][0] if pos_neighbour_index < number_of_particles
                                  else size_of_particle_space[0])
        return neg_neighbour_position, pos_neighbour_position

    def _sum_nearest_neighbours(self, active_particle_index, candidate_position, positions):
        """
        Returns the potential at active_particle_index by performing a sum over the (possibly anchor) neighbours.
        """
        neg_neighbour_index, pos_neighbour_index = self._get_neighbours(active_particle_index)
        neg_neighbour_position, pos_neighbour_position = self._get_neighbour_positions(
            positions, active_particle_index, neg_neighbour_index, pos_neighbour_index)
        neg_displacement, pos_displacement = (candidate_position - neg_neighbour_position,
                                              pos_neighbour_position - candidate_position)
        return (neg_displacement - self._equilibrium_length) ** 2 + (pos_displacement - self._equilibrium_length) ** 2
