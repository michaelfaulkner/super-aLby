"""Module for the InversePowerPotential class."""
from .continuous_potential import ContinuousPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import size_of_particle_space
import logging
import numpy as np


class InversePowerPotential(ContinuousPotential):
    # TODO this class does not work with DeterministicMediator -- acceptance rates are 0 or 1, depending on step size
    """This class implements the inverse power potential U = prefactor * sum(| positions[i] | ** (- power)) / power"""

    def __init__(self, power: float = 1.0, prefactor: float = 1.0):
        """
        The constructor of the InversePowerPotential class.

        Parameters
        ----------
        power : int
            Minus 1 multiplied by the power to which the norm of each component of the positions is raised (i.e., the
            norm of each particle position vector).
        prefactor : float
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If type(element) is not np.float64 for element in size_of_particle_space.
        base.exceptions.ConfigurationError
            If power is less than 1.0.
        """
        super().__init__(prefactor=prefactor)
        for element in size_of_particle_space:
            if type(element) != np.float64:
                raise ConfigurationError(f"For each component of size_of_particle_space, give a float value when using "
                                         f"{self.__class__.__name__}.")
        if power < 1.0:
            raise ConfigurationError(f"Give a value not less than 1.0 as power in {self.__class__.__name__}.")
        self._potential_constant = prefactor / power
        self._negative_power = - power
        self._negative_power_minus_two = - power - 2.0
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, power=power, prefactor=prefactor)
        # TODO remove the following warning once the class works with DeterministicMediator
        Warning(f"{self.__class__.__name__} does not currently work with DeterministicMediator.  Acceptance rates are "
                f"either 0 or 1, depending on step size of the numerical integrator.")

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
        return self._potential_constant * np.sum(np.linalg.norm(get_shortest_vectors_on_torus(positions), axis=1) **
                                                 self._negative_power)

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
        toroidal_positions = get_shortest_vectors_on_torus(positions)
        return - self._prefactor * toroidal_positions * np.linalg.norm(toroidal_positions,
                                                                       axis=1) ** self._negative_power_minus_two

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
    
    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event for a given active particle index,
        as well as the particle index responsible for that event. Used for ECMC.

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
            The direction of movement of the particle, either 1 or -1.
        
        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        veto_index : int
            The particle index responsible for the event.
        """
        raise SystemError(f"The get_distance_to_next_event_and_veto_index method of {self.__class__.__name__} has not been written."
                          "Functionality of ECMC for {self.__class__.__name__} is not currently provided.")

    
    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, n_indices_chosen,
                                    veto_index):
        """
        Chooses the index and direction for the next active particle in the markov chain for ECMC.
        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        movement_direction : int
            The direction of movement of the particle, either 1 or -1.
        veto_index : int
            The particle index responsible for the event. 
        """
        raise SystemError(f"The choose_next_active_particle method of {self.__class__.__name__} has not been written.")
        