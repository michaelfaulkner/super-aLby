"""Module for the NealFunnelPotential class."""
from .one_dim_euclidean_particle_space_potential import OneDimEuclideanParticleSpacePotential
import math
import numpy as np


class NealFunnelPotential(OneDimEuclideanParticleSpacePotential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0 .
    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the NealFunnel class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If dimensionality_of_particle_space does not equal 1.
        """
        super().__init__(prefactor=prefactor)

    def get_value(self, positions):
        """
        Returns the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.

        Returns
        -------
        float
            The potential.
        """
        positions = np.reshape(positions, tuple([positions.shape[i] for i in range(len(positions.shape) - 1)]))
        return self._prefactor * (positions[0] ** 2 / 18.0 + 9 * positions[0] / 2.0 + math.exp(-positions[0]) *
                                  np.sum(positions[1:len(positions)] ** 2) / 2.0)

    def get_gradient(self, positions):
        """
        Returns the gradient of the potential for the given positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        positions = np.reshape(positions, tuple([positions.shape[i] for i in range(len(positions.shape) - 1)]))
        gradient = np.zeros(len(positions))
        gradient[0] = positions[0] / 9.0 + 9 / 2.0 - (
                math.exp(-positions[0]) * np.sum(positions[1:len(positions)] ** 2) / 2.0)
        gradient[1:len(positions)] = 2.0 * positions[1:len(positions)] * math.exp(-positions[0])
        return self._prefactor * self._get_higher_dimension_array(gradient)

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        # TODO write the code for this method!
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
            is a float and represents one Cartesian component of the position of a single particle. In this case, the
            entire positions array corresponds to the Bayesian parameter.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        raise SystemError(f"The get_potential_difference method of {self.__class__.__name__} has not been written.")

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
        raise SystemError(f"The get_random_event_chain_velocity method has not been written.")

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
        hop_displacement : numpy.ndarray
            Net displacement through state space from active to vetoing particle.
        """
        raise SystemError(f"The get_distance_to_next_event_and_veto_index method of {self.__class__.__name__} has not "
                          f"been written.  Functionality of ECMC for {self.__class__.__name__} is not yet provided.")

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
        raise SystemError(f"The choose_next_active_particle method of {self.__class__.__name__} has not been written.")

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """Updates the position of the active particle following an event."""
        raise SystemError(f"The update_position method has not been written.")

    def get_swap_candidate(self, positions, active_particle_index):
        """Propose candidate configuration via a Metropolis swap kernel."""
        raise SystemError(f"The functionality (within MetropolisMediator) for Metropolis swaps has only been written "
                          f"for hard-sphere models.")
