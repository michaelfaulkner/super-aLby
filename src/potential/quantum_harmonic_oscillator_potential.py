"""Module for the QuantumHarmonicOscillatorPotential class"""
import numpy as np
from .worldline_potential import WorldlinePotential
from base.exceptions import ConfigurationError
from model_settings import number_of_quantum_particles, number_of_timeslices, number_of_particles
from helper_methods import get_east_neighbour_worldline, get_west_neighbour_worldline
from helper_methods import get_initial_positions_of_smooth_potential


class QuantumHarmonicOscillatorPotential(WorldlinePotential):
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
        super().__init__(prefactor=prefactor, lattice_dimensionality=lattice_dimensionality, mass=mass,
                          timestep=timestep)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        if lattice_dimensionality != 1:
            raise ConfigurationError(f"Give a value of 1 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        
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
                positions[particle_index], positions[get_east_neighbour_worldline(particle_index, number_of_timeslices,
                                                                                  number_of_quantum_particles)])
        return dimensionless_action

    def get_gradient(self, positions):
        # TODO implement get_gradient() function in this class
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

    def _get_gradient_at_index(self, positions, active_particle_index):
        """
        Returns the gradient of the dimensionless action with respect to the particle position at particle_index.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The particle index (i.e., the discretised-time index).
        Returns
        -------
        float
            The dimensionless-action gradient at particle_index.
        """
        return self._mass / self._timestep * (
                (2.0 + self._timestep ** 2 * self._omega ** 2) * positions[active_particle_index] -
                positions[get_west_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                       number_of_quantum_particles)] -
                positions[get_east_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                       number_of_quantum_particles)]).item()

    def _get_potential_action_term(self, positions, active_particle_index, position_at_active_particle_index):
        """
        Returns the potential energy contribution to teh pairwise dimensionless action

        Parameters
        ----------
        position_at_active_particle_index : float
            The position of the particle at some particle index.
        active_particle_index : int
            The index of the active particle.
        position_at_active_particle_index : float
            The position of the particle at the active particle index.

        Returns
        -------
        float
            The potential energy contribution to the pairwise dimensionless action.
        """
        return 0.5 * self._mass * self._timestep * self._omega ** 2 * position_at_active_particle_index ** 2

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
        
        worldline_neighbours = [get_west_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                             number_of_quantum_particles),
                                get_east_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                             number_of_quantum_particles)]

        shortest_distance_to_next_factor_event, vetoing_index = \
            self._get_distance_to_next_kinetic_event_and_veto_index(positions, active_particle_index,
                                                                    movement_direction, worldline_neighbours)
         # consider x^2 term
        initial_position = initial_position = positions[active_particle_index].item()
        uphill_energy = - np.log(np.random.uniform(0, 1))
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
        final_position_wrt_factor_event = self._get_final_position_wrt_quadratic_event(movement_direction, roots)

        distance_to_next_factor_event = np.abs(final_position_wrt_factor_event - initial_position)

        if distance_to_next_factor_event < shortest_distance_to_next_factor_event:
            shortest_distance_to_next_factor_event = distance_to_next_factor_event
            vetoing_index = active_particle_index

        return shortest_distance_to_next_factor_event, vetoing_index

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction for the next active particle in the markov chain.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The active particle index (i.e., the discretised-time index).
        movement_direction : int
            The direction of movement of the active particle.
        veto_index : int
            The particle index responsible for the event. 

        Returns
        -------
        active_particle_index : int
            The next active particle index (i.e., the discretised-time index) in the event chain.
        movement_direction : int
            The direction of movement of the next active particle.
        """
        initial_a = active_particle_index
        initial_v = movement_direction
  
        if veto_index == active_particle_index:
            movement_direction = movement_direction * -1
        else:
            active_particle_index = veto_index
        if active_particle_index == initial_a and movement_direction == initial_v:
            raise Exception("The same combination of active particle index and direction of motion has been chosen "
                            "twice in a row.")
        return active_particle_index, movement_direction

    def update_position(self, positions, displacement_distance, active_particle_index, movement_direction):
        """
        Updates position of the active particle following an event.

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        displacement_distance : float
            The displacement that the current position of the active particle will be updated using.
        active_particle_index : int
            The active particle index (i.e., the discretised-time index).
        movement_direction : int
            The direction of movement of the active particle.

        Returns
        -------
        new_position : float
            The updated position of the active particle
        """
        positions[active_particle_index] += displacement_distance * movement_direction
