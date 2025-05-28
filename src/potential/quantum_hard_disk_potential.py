"""Module for the QuantumHardDiskPotential class"""
import numpy as np
from .worldline_potential import WorldlinePotential
from base.exceptions import ConfigurationError
from base.vectors import get_shortest_vectors_on_torus
from model_settings import size_of_particle_space, number_of_quantum_particles, number_of_timeslices
from model_settings import number_of_particles, size_of_particle_space_over_two
from helper_methods import get_east_neighbour_worldline, get_west_neighbour_worldline


class QuantumHardDiskPotential(WorldlinePotential):
    r"""
    This class implements a 2-body quantum hard disk model in the worldline formalism.
    The potential corresponds to the dimensionless action,
        \delta\tau \sum_{i=1}^{N_{\tau}}[0.5 * m(x_{i+1} - x_i)^2 / (\delta\tau)^2 + V(r)],
        where m and \omega are the mass and frequency, respectively, and V(r) is the hard disk potential.
    """

    def __init__(self, prefactor: float = 1.0, lattice_dimensionality: int = 1, mass: float = 1.0,
                 timestep: float = 0.1, disk_radius: float = 1.0):
        r"""
        The constructor of the QuantumHarmonicHardDiskPotential class

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
        disk_radius : float, optional
            The radius of each disk.
        """
        super().__init__(prefactor=prefactor, lattice_dimensionality=lattice_dimensionality, mass=mass,
                         timestep=timestep)
        if prefactor != 1.0:
            raise ConfigurationError(f"Give a value of 1.0 for prefactor in {self.__class__.__name__} - functionality "
                                     f"for other values is not yet provided.")
        self._disk_radius = disk_radius

    def get_initial_positions(self):
        """
        Returns the initial positions array.  Creates an equidistant configuration.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g., three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """

        # distance_between_particles = size_of_particle_space / number_of_quantum_particles
        # positions = np.zeros((number_of_particles, 1))
        # for particle_index in range(number_of_particles):
        #     quantum_particle_index = particle_index % number_of_quantum_particles
        #     positions[particle_index] = get_shortest_vectors_on_torus(
        #         quantum_particle_index * distance_between_particles)
        positions = np.zeros((number_of_particles, 1))
        for particle_index in range(number_of_particles):
            quantum_particle_index = particle_index % number_of_quantum_particles
            positions[particle_index] = quantum_particle_index * 2.0 * self._disk_radius + quantum_particle_index * 0.5 * self._disk_radius
        return positions

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

        return 1.0

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
        pass

    def _get_potential_action_term(self, positions, active_particle_index, position_at_active_particle_index):
        r"""
        Returns the hard disk potential which goes as:
        V(r) = \inf ir r < \sigma
                0 if r \geq \sigma
        where r = |x_i - y_j| and \sigma is the disk radius

        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        active_particle_index : int
            The index of the active particle.
        position_at_active_particle_index : float
            The position of the particle at the active particle index.
        Returns
        -------
        float
            The potential energy contribution to the pairwise dimensionless action.
        """

        quantum_particles_at_timeslice = \
            self.get_quantum_particles_at_timeslice(
                active_particle_index, number_of_quantum_particles)
        dist = 10e6
        dist_new = None
        for quantum_particle in quantum_particles_at_timeslice:
            if quantum_particle != active_particle_index:
                dist_new = \
                    np.abs(get_shortest_vectors_on_torus(
                        position_at_active_particle_index - positions[quantum_particle]))

                if dist_new < dist:
                    dist = dist_new

        if dist < 2 * self._disk_radius:
            hard_disk_potential = 10e10
        else:
            hard_disk_potential = 0.0
            
        return hard_disk_potential

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
        vetoing_index : int
            The index of the particle that triggers the event.
            """
        worldline_neighbours = [get_west_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                             number_of_quantum_particles),
                                get_east_neighbour_worldline(active_particle_index, number_of_timeslices,
                                                             number_of_quantum_particles)]
        quantum_particle_neighbours = \
            self.get_quantum_particles_at_timeslice(
                active_particle_index, number_of_quantum_particles)

        shortest_distance_to_next_factor_event, vetoing_index = \
            self.get_kinetic_term_distance_to_next_event_and_veto_index(positions, active_particle_index,
                                                                        temperature, movement_direction,
                                                                        worldline_neighbours)
        #shortest_distance_to_next_factor_event = 1.0e10
        distance_to_next_potential_event = 1.0e10
        potential_veto_index = None
        initial_position = positions[active_particle_index].item()
       
        for quantum_particle_neighbour in quantum_particle_neighbours:
            if quantum_particle_neighbour != active_particle_index:
                neighbour_quantum_particle_position = positions[quantum_particle_neighbour].item()
                
                distance_to_next_particle = get_shortest_vectors_on_torus(neighbour_quantum_particle_position -
                                                                           initial_position)
                #print(f"initial sep: {distance_to_next_particle}")
                                
                if distance_to_next_particle < 0.0: # we need to move in direction +1
                    distance_to_next_particle += size_of_particle_space
                    #print(f"boundary event, dist: {distance_to_next_particle}")                                       
              
                distance_to_next_particle = np.abs(distance_to_next_particle) - 2.0 * self._disk_radius

                #print(f"initial: {initial_position}, neighbour: {neighbour_quantum_particle_position}")
    
                if distance_to_next_particle < distance_to_next_potential_event:
                    distance_to_next_potential_event = distance_to_next_particle
                    potential_veto_index = quantum_particle_neighbour
        if shortest_distance_to_next_factor_event < distance_to_next_potential_event:
            #print(f"shortest dist (due to kinetic): {shortest_distance_to_next_factor_event}, potential dist was {distance_to_next_potential_event}")
            pass
        else:
            vetoing_index = potential_veto_index
            shortest_distance_to_next_factor_event = distance_to_next_potential_event
            #print(f"shortest dist (due to potential): {shortest_distance_to_next_factor_event}")
        
        #print(f"shortest dist: {shortest_distance_to_next_factor_event}, veto: {vetoing_index}, active: {active_particle_index}")
     
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
        new_active_particle_index : int
            The next active particle index (i.e., the discretised-time index) in the event chain.
        new_movement_direction : int
            The direction of movement of the next active particle.
        """
        return veto_index, movement_direction

    def update_position(self, positions, displacement_distance, active_particle_index, movement_direction):
        """
        Updates position of the active particle following an event.
        Parameters
        ----------
        positions : numpy.ndarray
            A one-dimensional numpy array of size (number_of_particles), indexed by time step; each element
            is a float and represents the position of the worldline at that time step.
        displacement_distrance : float
            The displacement that the current position of the cative particle will be updated using.
        active_particle_index : int
            The active particle index (i.e., the discretised-time index).
        movement_direction : int
            The direction of movement of the active particle.
        Returns
        -------
        new_position : float
            The updated position of the active particle
        """
        
        old_position = np.copy(positions[active_particle_index])
        
        positions[active_particle_index] += displacement_distance * \
            movement_direction
        positions[active_particle_index] = get_shortest_vectors_on_torus(
            positions[active_particle_index])

        #print(f"movement direction: {movement_direction}, displacement: {displacement_distance} moved particle {active_particle_index} at {old_position} to {positions[active_particle_index]} \n"
            #" ----------------------------------------------------------------------------------------------")

    @staticmethod
    def get_quantum_particles_at_timeslice(active_particle_index, number_of_quantum_particles):
        """ Returns the indices of all quantum particles at current timeslice."""
        quantum_particle_index = active_particle_index % number_of_quantum_particles
        timeslice_index = active_particle_index // number_of_quantum_particles
        quantum_particles_at_timeslice = np.zeros(number_of_quantum_particles)
        for index in range(number_of_quantum_particles):
            quantum_particle_index = number_of_quantum_particles * timeslice_index + index
            quantum_particles_at_timeslice[index] = quantum_particle_index
        return quantum_particles_at_timeslice.astype(int)

    def _remove_overlap(self, active_particle_position, neighbour_particle_position, displacement):
        """Checks if moving a given displacement in the +1 direction would create an overlap, and is so, returns the
        maximum displacement that can be moved without creating an overlap"""
        init_displacement = displacement.copy()
        separation = np.abs(get_shortest_vectors_on_torus(active_particle_position + \
                   displacement) - neighbour_particle_position)
        if separation < 2.0 * self._disk_radius:
            displacement -= (2.0 * self._disk_radius - separation)
            if displacement < 0.0:
                displacement = 0.0
            #print(f"fixed overlap (separation {separation}), changed {init_displacement} to {displacement}")

        return displacement
        