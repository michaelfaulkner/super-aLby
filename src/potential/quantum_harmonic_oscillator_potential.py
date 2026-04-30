"""Module for the QuantumHarmonicOscillatorPotential class"""
import numpy as np
import cmath
from .worldline_potential import WorldlinePotential
from base.exceptions import ConfigurationError
from helper_methods import get_initial_positions_of_smooth_potential
from model_settings import number_of_quantum_particles, number_of_timeslices, number_of_particles
from potential.cpp_quantum_harmonic_oscillator import cpp_qho 
class QuantumHarmonicOscillatorPotential(WorldlinePotential):
    r"""
    This class implements the (currently one-dimensional) potential for the quantum harmonic oscillator resulting
        from the Wick rotation of the Feynman path integral.  The potential corresponds to the dimensionless action,
        \delta\tau \sum_{i=1}^{N_{\tau}}[0.5 * m(x_{i+1} - x_i)^2 / (\delta\tau)^2 + 0.5 * m * \omega^2 * x_i^2],
        where m and \omega are the mass and frequency, respectively.
    """
    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0, omega_squared: float = 1.0,
                 timestep: float = 0.1, cpp_implementation: bool = True, anharmonicity: float = 0.0):
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
        omega_squared : float
            The squared frequency of the oscillations
        timestep : float
            The size of the time step, \delta \tau.
        cpp_implementation : bool
            Whether or not to use the C++ implementation of the get_next_event() function -- provides a large speedup.
        anharmonicity: float
            The anharmonicity parameter in the potential.
        
        """
        super().__init__(prefactor=prefactor, lattice_dimensionality=lattice_dimensionality, mass=mass,
                         timestep=timestep, cpp_implementation=cpp_implementation)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        if lattice_dimensionality != 1:
            raise ConfigurationError(f"Give a value of 1 for lattice_dimensionality in {self.__class__.__name__} - "
                                     f"functionality for other dimensions not yet provided.")
        if anharmonicity < 0.0:
            raise ConfigurationError(f"Give a value greater than or equal to 0.0 for anharmonicity in "
                                     f"{self.__class__.__name__} to ensure a well-defined probability distribution.")
        if omega_squared <= 0.0 and anharmonicity == 0.0:
            raise ConfigurationError(f"For anharmonicity equal to 0.0, give a value greater than 0.0 for omega_squared "
                                     f"in {self.__class__.__name__} to ensure a well-defined probability distribution.")
        if anharmonicity <= 0.0 and omega_squared == 0.0:
            raise ConfigurationError(f"For omega_squared equal to 0.0, give a value greater than 0.0 for anharmonicity "
                                     f"in {self.__class__.__name__} to ensure a well-defined probability distribution.")
        self._omega_squared = omega_squared
        self._anharmonicity = anharmonicity
        if self._anharmonicity != 0.0:
            self._magnitude_of_double_well_position = np.sqrt(-self._mass * self._omega_squared /
                                                               (4 * self._anharmonicity))
        else:
            self._magnitude_of_double_well_position = 0.0
        

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
        candidate_position : float or numpy.ndarray
            The proposed position of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.

        Returns
        -------
        float
            The dimensionless-action difference.
        """
        if self._cpp_implementation:
    
            active_particle_position = positions[active_particle_index]
            west_neighbour_position = positions[self._get_west_worldline_neighbour(active_particle_index)]
            east_neighbour_position = positions[self._get_east_worldline_neighbour(active_particle_index)]

            diff = cpp_qho.get_potential_difference(active_particle_position.item(),
                    west_neighbour_position.item(), east_neighbour_position.item(), self._mass, self._timestep, self._omega_squared,
                    self._anharmonicity, candidate_position.item())
            
            return diff
        
        else:
            return super().get_potential_difference(active_particle_index, candidate_position, positions)


    def _get_gradient_at_index(self, positions, active_particle_index):
        """
        Returns the gradient of the dimensionless action with respect to the particle position at particle_index.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.

        Returns
        -------
        float
            The dimensionless-action gradient at active_particle_index.
        """
        return (2.0 * self._mass / self._timestep + self._mass * self._timestep * self._omega_squared) \
                * positions[active_particle_index] + 4.0 * self._anharmonicity * self._timestep \
                * positions[active_particle_index]**3 - self._mass / self._timestep \
                * (positions[self._get_west_worldline_neighbour(active_particle_index)] \
                   + positions[self._get_east_worldline_neighbour(active_particle_index)])

    def _get_potential_action_term(self, positions, active_particle_index, position_at_active_particle_index):
        """
        Returns the potential energy contribution to the pairwise dimensionless action

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.
        position_at_active_particle_index : float or numpy.ndarray
            The position of the active particle.

        Returns
        -------
        float
            The potential energy contribution to the pairwise dimensionless action.
        """
        return 0.5 * self._mass * self._timestep * self._omega_squared * position_at_active_particle_index ** 2 \
                    + self._anharmonicity * self._timestep * position_at_active_particle_index**4

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

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.
        temperature : float
            The sampling temperature.  N.B. we set temperature = 1.0 (for QHO) as this quantity is for stat-phys models.
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
        
        worldline_neighbours = [self._get_west_worldline_neighbour(active_particle_index),
                                self._get_east_worldline_neighbour(active_particle_index)]
        
        self._kinetic_U_west = - np.log(np.random.uniform(0.0, 1.0))
        self._kinetic_U_east = - np.log(np.random.uniform(0.0, 1.0))
        self._potential_U = - np.log(np.random.uniform(0.0, 1.0))

        if self._cpp_implementation:
            event = cpp_qho.get_next_event(active_particle_index, worldline_neighbours[1],
                    worldline_neighbours[0], number_of_quantum_particles, number_of_timeslices,
                    positions[active_particle_index].item(), positions[worldline_neighbours[1]].item(),
                    positions[worldline_neighbours[0]].item(), self._mass, self._timestep, movement_direction,
                    self._anharmonicity, self._omega_squared, self._magnitude_of_double_well_position,
                    self._kinetic_U_west, self._kinetic_U_east, self._potential_U)
            
            return event.shortest_distance_to_next_event, event.vetoing_index, None
        
        else:
            (shortest_distance_to_next_event, vetoing_index) = \
                self._get_next_kinetic_event(positions, active_particle_index, movement_direction, worldline_neighbours)
            """now consider the potential part of the action"""
            initial_position = positions[active_particle_index].item()
            uphill_energy = self._potential_U  #- np.log(np.random.uniform(0.0, 1.0))

            if self._anharmonicity == 0.0 or self._omega_squared == 0.0:
                bottom_of_well = 0.0
            else:
                if initial_position != 0.0:
                    bottom_of_well = self._magnitude_of_double_well_position * np.sign(initial_position)
                else:
                    if movement_direction > 0:
                        bottom_of_well = self._magnitude_of_double_well_position
                    else:
                        bottom_of_well = -self._magnitude_of_double_well_position

            if ((movement_direction > 0 and initial_position < bottom_of_well) or 
                    (movement_direction < 0 and initial_position > bottom_of_well)):
                """advance to the bottom of the potential well"""
                intermediate_position = bottom_of_well
            else:
                intermediate_position = initial_position
                    
            initial_action = 0.5 * self._mass * self._timestep * self._omega_squared * intermediate_position ** 2 + \
                            self._timestep * self._anharmonicity * intermediate_position ** 4
            final_action = uphill_energy  + initial_action

            if self._anharmonicity == 0.0:
                roots = self._get_harmonic_potential_roots(final_action)
            else:
                roots = self._get_anharmonic_potential_roots(final_action)

            if self._anharmonicity > 0.0 > self._omega_squared:
                if np.isreal(roots).all():
                    final_position = self._get_final_position_of_non_tunnel_event(intermediate_position, movement_direction,
                                                                                roots)
                else:
                    if ((movement_direction > 0 and intermediate_position > 0.0) or
                            (movement_direction < 0 and intermediate_position < 0.0)):
                        final_position = self._get_final_position_of_single_well_event(movement_direction,
                                                                                        roots[np.isreal(roots)])
                    else:
                        remaining_barrier_height = self._get_barrier_height(intermediate_position)
                        bottom_of_well *= -1
                        intermediate_position = bottom_of_well
                        final_action -= remaining_barrier_height
                        roots = self._get_anharmonic_potential_roots(final_action)
                        if np.isreal(roots).all():
                            final_position = self._get_final_position_of_non_tunnel_event(intermediate_position,
                                                                                        movement_direction, roots)
                        else:
                            final_position = self._get_final_position_of_single_well_event(movement_direction,
                                                                                        roots[np.isreal(roots)])
            else:
                final_position = self._get_final_position_of_single_well_event(movement_direction, roots)

            distance_to_next_factor_event = np.abs(final_position - initial_position)
            #print(f"py potential proposed: {distance_to_next_factor_event}, {active_particle_index}")

            if distance_to_next_factor_event < shortest_distance_to_next_event:
                shortest_distance_to_next_event = distance_to_next_factor_event
                vetoing_index = active_particle_index
        
            return shortest_distance_to_next_event, vetoing_index, None

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.
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
        Updates the position of the active particle following an event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the position of a single quantum particle.
        displacement_distance : float
            The displacement distance of the active particle to the next event.
        active_particle_index : int
            The index of the active particle.
        movement_direction : int
            The active-particle direction of motion.
        """
        positions[active_particle_index] += displacement_distance * movement_direction

    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """Propose candidate via teleportation portal kernel."""
        raise SystemError(f"The get_portal_candidate method of {self.__class__.__name__} has not been written.")

    @staticmethod
    def _get_final_position_of_non_tunnel_event(position, movement_direction, roots):
        """
        Returns the correct root of the quartic equation for an event that does not tunnel through the barrier of a
            double-well potential.

        Parameters
        ----------
        position : float
            Position of particle after movement down the potential landscape has been made.
        movement_direction : int
            The active-particle direction of motion.
        roots : numpy.ndarray
            Array of roots of the quadratic equation given by the kinetic term of the action. 
        Returns
        -------
            The correct root of the equation according to the direction of motion.
        """
        sorted_roots = np.sort(roots)
        if position < 0.0: # A < x < B, i.e. in the left-hand well
            if movement_direction > 0:
                return sorted_roots[1]
            else:
                return sorted_roots[0]
        else: # C < x < D, i.e. in the right-hand well
            if movement_direction > 0:
                return sorted_roots[3]
            else:
                return sorted_roots[2]
        
    def _get_barrier_height(self, position):
        """
        Finds the barrier height between the two wells of a double-well quartic potential.

        Parameters
        ----------

        Returns
        -------
            The barrier height.
        """
        barrier_height = np.abs(-(self._timestep * self._anharmonicity * position **4 + 
                                0.5 * self._mass * self._timestep * self._omega_squared * position **2))

        return barrier_height
    
    def _quartic_action_term(self, position):
        """
        Returns the value of the quartic action term at x = position, disregarding the constant term.

        Parameters
        ----------
        position : float
            The position x.
        Returns
        ---------
            The value of the quartic action term.
        """
        return self._timestep * self._anharmonicity * position**4 + \
            0.5 * self._mass * self._timestep * self._omega_squared * position**2 

        
    def _get_harmonic_potential_roots(self, final_action):
        """
        Returns the final position due to a potential event with a harmonic potential

        Parameters
        ----------
        final_action : float
            The value of the final action for the kinetic term at this event time
        Returns
        -------
            ndarray containing both roots of the equation
        """
        a = 0.5 * self._mass * self._timestep * self._omega_squared
        b = 0.0
        c = -final_action
        return self._get_quadratic_roots(a, b, c)
    
    def _get_anharmonic_potential_roots(self, final_action):

        """
        Returns the final position due to a potential event with an anharmonic potential. Treats the potential as a 
        quadratic where u = x**2

        Parameters
        ----------
        final_action : float
            The value of the final action for the kinetic term at this event time
        Returns
        -------
            ndarray containing all 4 roots of the equation
        """
        roots = np.zeros(4, dtype=complex)

        a = self._timestep * self._anharmonicity
        b = 0.5 * self._mass * self._timestep * self._omega_squared
        c = -final_action

        u0, u1 = self._get_quadratic_roots(a, b, c)

        roots[0] = cmath.sqrt(u0)
        roots[1] = -cmath.sqrt(u0)
        roots[2] = cmath.sqrt(u1)
        roots[3] = -cmath.sqrt(u1)

        return roots
