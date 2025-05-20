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

        distance_between_particles = size_of_particle_space / number_of_quantum_particles
        positions = np.zeros((number_of_particles, 1))
        for particle_index in range(number_of_particles):
            quantum_particle_index = particle_index % number_of_quantum_particles
            positions[particle_index] = get_shortest_vectors_on_torus(size_of_particle_space_over_two +
                                                                    quantum_particle_index * distance_between_particles)
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
        """
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
        #find timeslice
        timeslice_index = active_particle_index // number_of_quantum_particles
        #find quantum particle index
        quantum_particle_index = active_particle_index % number_of_quantum_particles

        dist = 10e6
        for quantum_particle in range(number_of_quantum_particles):
            if quantum_particle != quantum_particle_index:
                quantum_particle_location = number_of_quantum_particles * timeslice_index + quantum_particle
                dist_new = np.abs(position_at_active_particle_index - positions[quantum_particle_location])
                if dist_new < dist:
                    dist = dist_new
        
        if dist < self._disk_radius:
            hard_disk_potential = 10e10
        else:
            hard_disk_potential = 0.0
        return hard_disk_potential

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature,
                                            movement_direction):
        raise  SystemError(f"The get_distance_to_next_event_and_veto_index method of {self.__class__.__name__} "
                            "has not been written.")
    
    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
            raise  SystemError(f"The choose_next_active_particle method of {self.__class__.__name__} "
                            "has not been written.")
    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """ Updates position of the active particle following an event."""
        pass