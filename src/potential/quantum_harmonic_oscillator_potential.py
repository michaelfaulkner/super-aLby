"""Module for the QuantumHarmonicOscillatorPotential class"""
import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from base.exceptions import ConfigurationError
from model_settings import number_of_particles
from helper_methods import get_east_neighbour, get_west_neighbour
from helper_methods import get_initial_positions_of_smooth_potential


class QuantumHarmonicOscillatorPotential(EuclideanSubspacePotential):
    r"""
    This class implements the (currently one-dimensional) potential for the quantum harmonic oscillator resulting
        from the Wick rotation of the Feynman path integral.  The potential corresponds to the dimensionless action,
        \delta\tau \sum_{i=1}^{N_{\tau}}[0.5 * m(x_{i+1} - x_i)^2 / (\delta\tau)^2 + 0.5 * m * \omega^2 * x_i^2],
        where m and \omega are the mass and frequency, respectively.
    """
    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0,
                 timestep: float = 0.1):
        r"""
        The constructor of the QuantumHarmonicOscillatorPotential class

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.
        lattice_dimensionality : int
            The number of Cartesian dimensions of the lattice.
        mass : float
            The mass of the particle.
        timestep : float
            The size of the time step, \delta \tau.
        """
        super().__init__(prefactor=prefactor)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        if lattice_dimensionality != 1:
            raise ConfigurationError(f"Give a value of 1 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        self._lattice_dimensionality = lattice_dimensionality
        self._mass = mass
        self._timestep = timestep
        self._omega = self._mass

    def get_initial_positions(self):
        """
        Returns the initial positions array.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g., two particles
            (confined to one-dimensional space) at positions 0.0 and 1.0 is represented by [[0.0] [1.0]]; three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
        return get_initial_positions_of_smooth_potential(self.__class__.__name__)

    def get_value(self, positions):
        """
        Returns the dimensionless action for the given particle positions.  Note that the dimensional action
            S * self._timestep is analogous to the potential of a statistical-physics model (since hbar is considered
            analogous to the inverse temperature (beta) of a stat-physics model; S denotes the raw action).

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        Returns
        -------
        float
            The dimensionless action.
        """
        dimensionless_action = 0.0
        for particle_index in range(0, number_of_particles):
            dimensionless_action += self._get_pairwise_dimensionless_action(
                positions[particle_index], positions[get_east_neighbour(particle_index, number_of_particles)])
        return dimensionless_action

    def get_gradient(self, positions):
        # TODO implement get_gradient() function in this class
        """
        Returns the gradient of the dimensionless action for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        raise SystemError(f"The get_gradient method of {self.__class__.__name__} has not been written.")

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the difference in dimensionless action resulting from moving the single active particle to
            candidate_position.  Note that the dimensional action S * self._timestep is analogous to the potential of a
            statistical-physics model (since hbar is considered analogous to the inverse temperature (beta) of a
            stat-physics model; S denotes the raw action).

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : float
            A float representing the proposed position of the active particle.
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.

        Returns
        -------
        float
            The dimensionless-action difference.
        """
        current_dimensionless_action = (
                self._get_pairwise_dimensionless_action(
                    positions[get_west_neighbour(active_particle_index, number_of_particles)],
                    positions[active_particle_index]) +
                self._get_pairwise_dimensionless_action(
                    positions[active_particle_index],
                    positions[get_east_neighbour(active_particle_index, number_of_particles)]))
        
        candidate_dimensionless_action = (
                self._get_pairwise_dimensionless_action(
                    positions[get_west_neighbour(active_particle_index, number_of_particles)], candidate_position) +
                self._get_pairwise_dimensionless_action(
                    candidate_position, positions[get_east_neighbour(active_particle_index, number_of_particles)]))
        return candidate_dimensionless_action - current_dimensionless_action

    def _get_gradient_at_index(self, positions, particle_index):
        """
        Returns the gradient of the dimensionless action with respect to the particle position at particle_index.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        particle_index : int
            The particle index (i.e., the discretised-time index).
        Returns
        -------
        float
            The dimensionless-action gradient at particle_index.
        """
        return self._mass / self._timestep * (
                (2.0 + self._timestep ** 2 * self._omega ** 2) * positions[particle_index] -
                positions[get_west_neighbour(particle_index, number_of_particles)] -
                positions[get_east_neighbour(particle_index, number_of_particles)]).item()

    def _get_pairwise_dimensionless_action(self, position_at_index, position_at_east_index):
        """
        Returns the contribution to the dimensionless action from a given pair of positions.

        Parameters
        ----------
        position_at_index : float or numpy.ndarray
            The position of the particle at some particle index.
        position_at_east_index : float or numpy.ndarray
            The position of the particle at the site east of the particle index.
        Returns
        -------
        float
            The pairwise contribution to the dimensionless action.
        """

        return 0.5 * self._mass * ((position_at_east_index - position_at_index) ** 2 / self._timestep +
                                   self._timestep * self._omega ** 2 * position_at_index ** 2)

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
        return np.random.choice((-1, 1))

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature,
                                                  movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The active particle index (i.e., the discretised-time index).
        temperature : float
            The sampling temperature.  NB, we set temperature = 1.0 (for QHO) as this quantity is for stat-phys models.
        movement_direction : int
            The direction of movement of the active particle.
        
        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        vetoing_particle_index : int
            The index of the particle that triggers the event.
        """
        shortest_distance_to_next_factor_event = 1.0e10
        neighbouring_indices = np.zeros(3, dtype=np.int32)
        neighbouring_indices[0] = get_west_neighbour(active_particle_index, number_of_particles)
        neighbouring_indices[1] = active_particle_index
        neighbouring_indices[2] = get_east_neighbour(active_particle_index, number_of_particles)

        vetoing_index = None
        initial_position = positions[active_particle_index].item()

        for i in range(3):
            uphill_energy = - np.log(np.random.uniform(0, 1))
            if i != 1:  # considering the neighbour terms
                neighbour_position = positions[neighbouring_indices[i]].item()
                bottom_of_well = neighbour_position
                if ((movement_direction > 0 and initial_position < bottom_of_well) or
                        (movement_direction < 0 and initial_position > bottom_of_well)):
                    """advance to the bottom of the well"""
                    intermediate_position = bottom_of_well
                else:
                    intermediate_position = initial_position

                initial_action = 0.5 * (self._mass / self._timestep) * (intermediate_position - neighbour_position) ** 2
                final_action = uphill_energy + initial_action
                roots = np.roots([0.5 * self._mass / self._timestep, -(self._mass / self._timestep)
                                  * neighbour_position, (0.5 * self._mass / self._timestep) * neighbour_position ** 2
                                  - final_action])
                final_position_wrt_factor_event = self.get_final_position_wrt_factor_event(movement_direction, roots)
                    
            else:  # consider x^2 term
                bottom_of_well = 0.0
                if (((movement_direction > 0) and (initial_position < bottom_of_well)) or
                        ((movement_direction < 0) and (initial_position > bottom_of_well))):
                    """advance to the bottom of the well"""
                    intermediate_position = bottom_of_well
                else:
                    intermediate_position = initial_position
                
                initial_action = 0.5 * self._mass * self._timestep * self._omega ** 2 * intermediate_position ** 2
                final_action = uphill_energy + initial_action
                roots = np.roots([0.5 * self._mass * self._timestep * self._omega ** 2, 0.0, -final_action])
                final_position_wrt_factor_event = self.get_final_position_wrt_factor_event(movement_direction, roots)

            distance_to_next_factor_event = np.abs(final_position_wrt_factor_event - initial_position)

            if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
                shortest_distance_to_next_factor_event = distance_to_next_factor_event
                vetoing_index = neighbouring_indices[i]
        
        return shortest_distance_to_next_factor_event, vetoing_index

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """Chooses the index and direction for the next active particle in the markov chain"""
        initial_a = active_particle_index
        initial_v = movement_direction
  
        if veto_index == active_particle_index:
            movement_direction = movement_direction * -1
        else:
            active_particle_index = veto_index
        if active_particle_index == initial_a and movement_direction == initial_v:
            raise Exception("Chose the same index and direction twice in a row")
        return active_particle_index, movement_direction

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """ Updates position of the active particle."""
        positions[active_particle_index] += displacement_distance * movement_direction

    @staticmethod
    def get_final_position_wrt_factor_event(movement_direction, roots):
        if (movement_direction > 0) and (roots[0] > roots[1]):
            return roots[0]
        elif (movement_direction > 0) and (roots[0] < roots[1]):
            return roots[1]
        elif (movement_direction < 0) and (roots[0] < roots[1]):
            return roots[0]
        else:
            return roots[1]
