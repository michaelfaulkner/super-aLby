"""Module for the WorldlinePotential class"""
import numpy as np
from .euclidean_subspace_potential import EuclideanSubspacePotential
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from model_settings import number_of_quantum_particles, number_of_timeslices, number_of_particles
from model_settings import dimensionality_of_particle_space 


class WorldlinePotential(EuclideanSubspacePotential, metaclass=ABCMeta):
    """
    Abstract class for worldline potentials.  The extra methods provided are those required to calculate the action
        of the system.
    """
    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0, omega_squared: float = 1.0,
                 timestep: float = 1.0, cpp_implementation: bool = False, **kwargs):
        """
        The constructor of the WorldlinePotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        lattice_dimensionality : int
            The number of Cartesian dimensions of the lattice.
        mass : float
            The mass of the particle
        timestep : float
            The size of the time step
        cpp_implementation : bool
            Whether or not to use the C++ implementation of the get_next_event() function -- provides a large speedup.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor=prefactor)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        if dimensionality_of_particle_space > 1:
            raise ConfigurationError(f"Functionality for dimensionality_of_particle_space greater than one not yet "
                                     f"available for {self.__class__.__name__}.")
        self._lattice_dimensionality = lattice_dimensionality
        self._mass = mass
        self._timestep = timestep
        self._cpp_implementation = cpp_implementation
        if self._cpp_implementation: #TODO check if using ECMC
            print(f"Using C++ implementation of get_next_event() for ECMC")

    def get_value(self, positions):
        """
        Returns the dimensionless action for the given particle positions.  Note that the dimensional action
            S * self._timestep is analogous to the potential of a statistical-physics model (since hbar is considered
            analogous to the inverse temperature (beta) of a stat-physics model; S denotes the raw action).

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.

        Returns
        -------
        float
            The dimensionless action.
        """
        dimensionless_action = 0.0
        for particle_index in range(0, number_of_particles):
            """NB, this is one of the cases that prevents dimensionality_of_particle_space > 1 (see __init__())."""
            dimensionless_action += self._get_pairwise_dimensionless_action(
                positions, particle_index, positions[particle_index, 0],
                positions[self._get_east_worldline_neighbour(particle_index), 0])
        return dimensionless_action
    
    def get_gradient(self, positions):
        # TODO implement get_gradient() function in this class
        """
        Returns the gradient of the dimensionless action for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential (i.e. dimensionless
            action) of a single quantum particle.
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
  
        current_dimensionless_action = (
                self._get_pairwise_dimensionless_action(
                    positions, self._get_west_worldline_neighbour(active_particle_index),
                    positions[self._get_west_worldline_neighbour(active_particle_index)],
                    positions[active_particle_index]) +
                self._get_pairwise_dimensionless_action(
                    positions, active_particle_index, positions[active_particle_index],
                    positions[self._get_east_worldline_neighbour(active_particle_index)]))
        
        candidate_dimensionless_action = (
                self._get_pairwise_dimensionless_action(
                    positions, self._get_west_worldline_neighbour(active_particle_index),
                    positions[self._get_west_worldline_neighbour(active_particle_index)], candidate_position) +
                self._get_pairwise_dimensionless_action(
                    positions, active_particle_index, candidate_position,
                    positions[self._get_east_worldline_neighbour(active_particle_index)]))
        
        return (candidate_dimensionless_action - current_dimensionless_action)

    def _get_pairwise_dimensionless_action(self, positions, active_particle_index, position_at_active_particle_index,
                                           position_at_neighbouring_worldline_index):
        """
        Returns the contribution to the dimensionless action from a given pair of positions.  This is pairwise in the
            sense of worldline timeslices, not quantum particles.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.
        position_at_active_particle_index : float or numpy.ndarray
            The position of the particle at the active particle index.
        position_at_neighbouring_worldline_index : float or numpy.ndarray
            The position of the same index quantum particle at a neighbouring timeslice.

        Returns
        -------
        float
            The pairwise contribution to the dimensionless action.
        """
        return (self._get_kinetic_action_term(
            position_at_active_particle_index, position_at_neighbouring_worldline_index) +
                self._get_potential_action_term(positions, active_particle_index, position_at_active_particle_index))
    
    def _get_kinetic_action_term(self, position_at_active_particle_index, position_at_neighbouring_worldline_index):
        """
        Returns the kinetic energy contribution to the pairwise dimensionless action.

        Parameters
        ----------
        position_at_active_particle_index : float or numpy.ndarray
            The position of the particle at the active particle index.
        position_at_neighbouring_worldline_index : float or numpy.ndarray
            The position of the same index quantum particle at a neighbouring timeslice

        Returns
        -------
        float
            The kinetic energy contribution to the pairwise dimensionless action.
        """
        return 0.5 * self._mass * \
        (position_at_neighbouring_worldline_index - position_at_active_particle_index) ** 2 / self._timestep
    
    @abstractmethod
    def _get_potential_action_term(self, positions, active_particle_index, position_at_active_particle_index):
        """
        Returns the potential energy contribution to the pairwise dimensionless action

        Parameters
        -------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.
        position_at_active_particle_index : float
            The position of the particle at the active particle index.

        Returns
        -------
        float
            The potential energy contribution to the pairwise dimensionless action.
        """
        raise NotImplementedError

    def _get_next_kinetic_event(self, positions, active_particle_index, movement_direction, worldline_neighbours,
                                kinetic_U_west, kinetic_U_east):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.
        active_particle_index : int
            The index of the active particle.
        movement_direction : int
            The active-particle direction of motion.
        worldline_neighbours : List[int]
            A one-dimensional list containing the particles indices of the worldline neighbours of the active particle.
        kinetic_U_west : float
            Randomly drawn uphill energy for the west kinetic-energy term.
        kinetic_U_east : float
            Randomly drawn uphill energy for the east kinetic-energy term.
        
        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event according to the kinetic term of the action.
        vetoing_particle_index : int
            The index of the particle that triggers the event according to the kinetic term of the action.
        """
        shortest_distance_to_next_kinetic_event = 1.0e10
        vetoing_index = None
        initial_position = positions[active_particle_index].item()

        for i, worldline_neighbour in enumerate(worldline_neighbours):
            if worldline_neighbour != active_particle_index:
                if i == 0:
                    uphill_energy = kinetic_U_west #- np.log(np.random.uniform(0, 1))
                else:
                    uphill_energy = kinetic_U_east
                neighbour_position = positions[worldline_neighbour].item()
                bottom_of_well = neighbour_position
                if ((movement_direction > 0 and initial_position < bottom_of_well) or
                        (movement_direction < 0 and initial_position > bottom_of_well)):
                    """advance to the bottom of the well"""
                    intermediate_position = bottom_of_well
                else:
                    intermediate_position = initial_position
                initial_action = 0.5 * (self._mass / self._timestep) * (intermediate_position - neighbour_position) ** 2
                final_action = uphill_energy + initial_action
                roots = self._analytic_kinetic_term_roots(neighbour_position, final_action)
                final_position = self._get_final_position_of_single_well_event(movement_direction, roots)
                distance_to_candidate_kinetic_event = np.abs(final_position - initial_position)
                #print(f"py kinetic proposed: {distance_to_candidate_kinetic_event}, {worldline_neighbour}")
                if distance_to_candidate_kinetic_event < shortest_distance_to_next_kinetic_event:
                    shortest_distance_to_next_kinetic_event = distance_to_candidate_kinetic_event
                    vetoing_index = worldline_neighbour
                
        return shortest_distance_to_next_kinetic_event, vetoing_index
    
    @abstractmethod
    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.
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
        raise NotImplementedError
    
    @abstractmethod
    def update_position(self, positions, displacement_distance, active_particle_index, movement_direction):
        """
        Updates the position of the active particle following an event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single quantum particle.
        displacement_distance : float
            The displacement that the current position of the active particle will be updated using.
        active_particle_index : int
            The index of the active particle.
        movement_direction : int
            The active-particle direction of motion.
        """
        raise NotImplementedError
    
    def _analytic_kinetic_term_roots(self, neighbour_position, final_action):
        """
        Returns the final position due to a kinetic event with the neighbouring particle at neighbour_position

        Parameters
        ----------
        neighbour_position : float 
            The position of the neighbouring particle
        final_action : float
            The value of the final action for the kinetic term at this event time
        Returns
        -------
            ndarray containing both roots of the equation

        """
        a = 0.5 * self._mass / self._timestep
        b = - self._mass / self._timestep * neighbour_position
        c = a * neighbour_position**2 - final_action
        return self._get_quadratic_roots(a, b, c)

    @staticmethod
    def _get_quadratic_roots(a, b, c):
        """
        Returns the two roots of the quadratic equation a*x^2 + b*x + c = 0.

        Parameters
        ----------
        a : float
            Coefficient of the quadratic term.
        b : float
            Coefficient of the linear term.
        c : float
            Coefficient of the constant term.
        Returns
        -------
            ndarray containing both roots of the quadratic equation

        """
        roots = np.zeros(2)
        roots[0] = (-b + np.emath.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        roots[1] = (-b - np.emath.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return roots

    @staticmethod
    def _get_final_position_of_single_well_event(movement_direction, roots):
        """
        Returns the correct root of the quadratic equation for an event generated by a quadratic potential term.

        Parameters
        ----------
        movement_direction : int
            The active-particle direction of motion.
        roots : numpy.ndarray
            Array of roots of the quadratic equation given by the kinetic term of the action.
        Returns
        -------
            The correct root of the equation according to the direction of motion.
        """
        assert(len(roots) == 2)
        
        if (movement_direction > 0) and (roots[0] > roots[1]):
            return roots[0]
        elif (movement_direction > 0) and (roots[0] < roots[1]):
            return roots[1]
        elif (movement_direction < 0) and (roots[0] < roots[1]):
            return roots[0]
        else:
            return roots[1]
        
    @staticmethod
    def _get_east_worldline_neighbour(lattice_site_index):
        """Returns the eastwards timeslice neighbour of lattice_site_index."""

        return int((lattice_site_index + number_of_quantum_particles) %
                   (number_of_timeslices * number_of_quantum_particles))

    @staticmethod
    def _get_west_worldline_neighbour(lattice_site_index):
        """Returns the westwards timeslice neighbour of lattice_site_index."""
        return int((lattice_site_index - number_of_quantum_particles) %
                   (number_of_timeslices * number_of_quantum_particles))
