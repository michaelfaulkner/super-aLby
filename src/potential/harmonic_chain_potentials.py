"""Module for the abstract HarmonicChainPotentials class."""
from abc import ABCMeta
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base. exceptions import ConfigurationError
from model_settings import size_of_particle_space, number_of_particles


class HarmonicChainPotentials(EuclideanSubspacePotential, metaclass=ABCMeta):
    """
    This class provides harmonic-chain functionality that is common to both HarmonicChainPotential and
        UnfoactorisedHarmonicChainPotential.
    """

    def __init__(self, prefactor: float = 1.0, equilibrium_length: float = 0.0):
        """
        The constructor of the HarmonicChainPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        equilibrium_length : float
            The separation of particles associated with the minimum of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the dimensionality of size_of_particle_space is greater than 1.
        """
        super().__init__(prefactor=prefactor)
        self._potential_constant = 0.5 * prefactor
        self._equilibrium_length = equilibrium_length
        if len(size_of_particle_space) > 1:
            raise ConfigurationError(f'{self.__class__.__name__} only supports 1D space. Provided: '
                                     f'{len(size_of_particle_space)}')

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
        potential = 0.0
        for particle_index in range(number_of_particles):
            neg_neighbour_index = (particle_index - 1) % number_of_particles
            neg_neighbour_position = positions[neg_neighbour_index][0]
            if particle_index == 0:
                neg_neighbour_position -= size_of_particle_space
            potential += (positions[particle_index] - neg_neighbour_position - self._equilibrium_length) ** 2
        return self._potential_constant * potential

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
        gradient_value = 0.0
        for particle_index in range(number_of_particles):
            neg_neighbour_index = (particle_index - 1) % number_of_particles
            neg_neighbour_position = positions[neg_neighbour_index][0]
            if particle_index == 0:
                neg_neighbour_position -= size_of_particle_space
            gradient_value += 2.0 * (positions[particle_index] - neg_neighbour_position - self._equilibrium_length)
        return self._potential_constant * gradient_value

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
        return (self._potential_constant *
                (self._sum_nearest_neighbours(active_particle_index, candidate_position, positions) -
                 self._sum_nearest_neighbours(active_particle_index, positions[active_particle_index], positions)))

    def _sum_nearest_neighbours(self, active_particle_index, candidate_position, positions):

        """
        Returns the potential at active_particle_index by performing a sum over nearest neighbours.

        Parameters
        ----------
        active_particle_index : int
            The index of the active_particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length 1 whose sole element is a float and represents the proposed phase of
            the spin of the active particle at active_particle_index.  This is a numpy array his is because the ith
            component of the positions array is a one-dimensional numpy array of length 1.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.
        Returns
        -------
        float
            The potential at lattice_site_index.
        """
        neg_neighbour_index, pos_neighbour_index = self._get_neighbours(active_particle_index)
        neg_neighbour_position, pos_neighbour_position = (positions[neg_neighbour_index][0],
                                                          positions[pos_neighbour_index][0])
        if active_particle_index == number_of_particles - 1:
            pos_neighbour_position += size_of_particle_space
        if active_particle_index == 0:
            neg_neighbour_position -= size_of_particle_space
        neg_displacement, pos_displacement = (candidate_position - neg_neighbour_position,
                                              pos_neighbour_position - candidate_position)
        return (neg_displacement - self._equilibrium_length) ** 2 + (pos_displacement - self._equilibrium_length) ** 2

    @staticmethod
    def get_random_event_chain_velocity():
        """
        Uniformly samples a direction of motion for the active particle from chosen velocity distribution.

        Returns
        ----------
        random_event_chain_velocity : int or numpy.ndarray
            The uniformly sampled event-chain velocity of the active particle.  If the state space of each particle is
            a subset of the real line, the method should output an integer; otherwise it should output a one-dimensional
            numpy array (of integers) of length dimensionality_of_particle_space, where the nth component represents the
            velocity of the active particle along the nth Cartesian direction.
        """
        return 1

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

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
        return veto_index, movement_direction

    @staticmethod
    def _get_neighbours(active_particle_index):
        """
        Return indices of neighbours to active particle.
        """
        neg_neighbour_index, pos_neighbour_index = ((active_particle_index - 1) % number_of_particles,
                                                    (active_particle_index + 1) % number_of_particles)
        return neg_neighbour_index, pos_neighbour_index

    @staticmethod
    def _get_neighbour_positions(positions, active_particle_index, neg_neighbour_index, pos_neighbour_index):
        neg_neighbour_position, pos_neighbour_position = (positions[neg_neighbour_index][0],
                                                          positions[pos_neighbour_index][0])
        if active_particle_index == number_of_particles - 1:
            pos_neighbour_position += size_of_particle_space[0]
        elif active_particle_index == 0:
            neg_neighbour_position -= size_of_particle_space[0]
        return neg_neighbour_position, pos_neighbour_position

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """Updates the position of the active particle following an event."""
        positions[active_particle_index] = positions[active_particle_index] + movement_direction * displacement_distance
        if positions[active_particle_index] > 1e10:
            positions -= 1e10
        elif positions[active_particle_index] < -1e10:
            positions += 1e10

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """Propose candidate via teleportation portal kernel."""
        raise SystemError(f"The get_portal_candidate method of {self.__class__.__name__} has not been written.")
