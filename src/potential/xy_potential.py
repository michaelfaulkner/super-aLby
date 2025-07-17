import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from model_settings import number_of_particles
from base.exceptions import ConfigurationError
from helper_methods import get_neighbours, get_initial_positions_of_smooth_potential


class XyPotential(EuclideanSubspacePotential):
    """
    This class implements the 2DXY model potential.
    """

    def __init__(self, prefactor: float = 1.0,  lattice_dimensionality: int = 2):
        """
        The constructor of the XyPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        lattice_dimensionality : int
            The dimensionality of the lattice on which the XY model is defined.  We currently only provide functionality
            for the 2DXY model, i.e. for lattice_dimensionality equal to two.
        """
        super().__init__(prefactor=prefactor)
        if lattice_dimensionality != 2:
            raise ConfigurationError(f"Give a value of 2 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        lattice_length = number_of_particles ** (1 / lattice_dimensionality)
        if not lattice_length.is_integer():
            raise ConfigurationError(
                f"For the value of number_of_particles in ModelSettings, give lattice_length ** lattice_dimensionality "
                f"when using {self.__class__.__name__}, where lattice_length is an integer not less than 2.")
        self._lattice_dimensionality = lattice_dimensionality
        self._lattice_length = int(lattice_length + 1.0e-12)
        self.potential_constant = prefactor

    def get_initial_positions(self):
        """
        Returns the initial positions array.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g. two particles
            (confined to one-dimensional space) at positions 0.0 and 1.0 is represented by [[0.0] [1.0]]; three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
        return get_initial_positions_of_smooth_potential(self.__class__.__name__)

    def get_value(self, positions):

        """
        Returns the potential function for the given particle positions. Here, 'positions' is a misnomer, inherited from
        the naming conventions of the parent ContinuousPotential class. A 'position' given here is actually a scalar
        value corresponding to the phase/angle of the spin of the corresponding particle.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.

        Returns
        -------
        float
            The potential.
        """

        return self.potential_constant * 0.5 * np.sum([self._sum_nearest_neighbours(index, positions[index], positions)
                                                       for index in range(number_of_particles)])

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        nothing.
        """
        raise SystemError(f"The get_gradient method of {self.__class__.__name__} has not been written.")

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the potential difference resulting from moving the single active particle's spin to candidate_position.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length 1 whose sole element is a float and represents the proposed phase of
            the spin of the active particle at active_particle_index.  This is a numpy array his is because the ith
            component of the positions array is a one-dimensional numpy array of length 1.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        return self.potential_constant * (
                self._sum_nearest_neighbours(active_particle_index, candidate_position, positions) -
                self._sum_nearest_neighbours(active_particle_index, positions[active_particle_index], positions))

    def _sum_nearest_neighbours(self, active_particle_index, active_particle_position, positions):

        """
        Returns the potential at lattice_site_index by performing a sum over nearest neighbours.

        Parameters
        ----------
        active_particle_index : int
            The index of the active_particle.
        active_particle_position : numpy.ndarray
            A one-dimensional numpy array of length 1 whose sole element is a float and represents the phase of the spin
            of the active particle.  This is because the ith component of the positions array is a one-dimensional numpy
            array of length 1.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, 1); each element is a float and represents the
            phase of the spin of its corresponding particle.
        Returns
        -------
        float
            The potential at lattice_site_index.
        """
        return -np.sum([np.cos(positions[neighbouring_spin_index, 0] - active_particle_position[0])
                        for neighbouring_spin_index in get_neighbours(active_particle_index, self._lattice_length)])

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

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        temperature : float
            The sampling temperature.
        movement_direction : int
            The active-particle direction of motion.

        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        vetoing_particle_index : int
            The index of the particle that triggers the event.
        """
        shortest_distance_to_next_factor_event = 1.0e10
        active_spin_value = positions[active_particle_index, 0]
        vetoing_spin_index = None

        for neighbouring_spin_index in get_neighbours(active_particle_index, self._lattice_length):
            non_active_spin_value = positions[neighbouring_spin_index, 0]
            initial_spin_value_difference = self._get_spin_difference(active_spin_value, non_active_spin_value)
            uphill_energy = - temperature / self.potential_constant * np.log(1.0 - np.random.rand())

            if initial_spin_value_difference > 0.0:
                initial_two_spin_potential = 1.0 - np.cos(initial_spin_value_difference)
                no_of_complete_spin_rotations = int(0.5 * (initial_two_spin_potential + uphill_energy))
                final_two_spin_potential = ((no_of_complete_spin_rotations + 1.0) * 2.0 - initial_two_spin_potential -
                                            uphill_energy)
                final_spin_value_difference = np.arccos(1.0 - final_two_spin_potential)
                distance_to_next_factor_event = ((no_of_complete_spin_rotations + 0.5) * 2.0 * np.pi -
                                                 initial_spin_value_difference - final_spin_value_difference)

            else:
                no_of_complete_spin_rotations = int(0.5 * uphill_energy)
                final_two_spin_potential = ((no_of_complete_spin_rotations + 1.0) * 2.0 - uphill_energy)
                final_spin_value_difference = np.arccos(1.0 - final_two_spin_potential)
                distance_to_next_factor_event = ((no_of_complete_spin_rotations + 0.5) * 2.0 * np.pi -
                                                 initial_spin_value_difference - final_spin_value_difference)

            if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
                shortest_distance_to_next_factor_event = distance_to_next_factor_event
                vetoing_spin_index = neighbouring_spin_index

        return shortest_distance_to_next_factor_event, vetoing_spin_index

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
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
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """Updates the position of the active particle."""
        positions[active_particle_index] = (positions[active_particle_index] + displacement_distance) % (2.0 * np.pi)

    @staticmethod
    def _get_spin_difference(spin_value_one, spin_value_two):
        """Returns the difference between two spin angles"""
        return (spin_value_one - spin_value_two + np.pi + 1.0e-12) % (2.0 * np.pi) - (np.pi + 1.0e-12)

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """Propose candidate via teleportation portal kernel."""
        return (2.0 * positions[veto_index] - positions[active_particle_index]) % (2.0 * np.pi)
