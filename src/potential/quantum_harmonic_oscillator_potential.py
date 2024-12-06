"""Module for the QuantumHarmonicOscillatorPotential class"""
import numpy as np
from .continuous_potential import ContinuousPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
import logging
from model_settings import number_of_particles
from helper_methods import get_east_neighbour, get_west_neighbour


class QuantumHarmonicOscillatorPotential(ContinuousPotential):
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
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor,
                           lattice_dimensionality=lattice_dimensionality, mass=mass, timestep=timestep)

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
        pass

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

    def get_gradient_at_index(self, positions, particle_index):
        """
        Returns the gradient of the dimensional action with respect to the particle position at particle_index.

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
        return self._mass * ((2.0 + self._timestep ** 2 * self._omega ** 2) * positions[particle_index] -
                             positions[get_west_neighbour(particle_index, number_of_particles)] -
                             positions[get_east_neighbour(particle_index, number_of_particles)]).item() / self._timestep

    def _get_pairwise_dimensionless_action(self, position_at_index, position_at_east_index):
        """
        Returns the contribution to the dimensionless action from a given pair of positions.

        Parameters
        ----------
        position_at_index : float
            The position of the particle at some particle index.
        position_at_east_index : float
            The position of the particle at the site east of the particle index.
        Returns
        -------
        float
            The pairwise contribution to the dimensionless action.
        """

        return 0.5 * self._mass * ((position_at_east_index - position_at_index) ** 2 / self._timestep +
                                   self._timestep * self._omega ** 2 * position_at_index ** 2)

    def get_distance_to_next_event(self, dimensionless_position_at_index, dimensionless_position_at_east_index,
                                   dimensionless_position_at_west_index, movement_direction, move_num):
        proposed_move_dimensionless = 0
        distance_travelled_in_move = 0
        possible_move_dimensionless = ((dimensionless_position_at_east_index + dimensionless_position_at_west_index)
                            / (2 + self._dimensionless_omega**2))
        if dimensionless_position_at_index < possible_move_dimensionless:
            dimensionless_position_at_index += possible_move_dimensionless * movement_direction
            proposed_move_dimensionless += possible_move_dimensionless
            distance_travelled_in_move += np.abs(possible_move_dimensionless)

        random_value = np.random.uniform(0.0, 1.0)
        a = 0.5 * self._dimensionless_m * (2.0 + self._dimensionless_omega**2)
        b = 0.5 * self._dimensionless_m * (4.0 * dimensionless_position_at_index 
                                              + 2.0 * self._dimensionless_omega**2 * dimensionless_position_at_index
                                              - 2.0 * dimensionless_position_at_east_index 
                                              - 2.0 * dimensionless_position_at_west_index).item()
        c = np.log(random_value)

        eta = np.roots([c,b,a])
        if eta[0] > 0:
            eta = eta[0]
        else: 
            eta = eta[1]
        #TODO pick one of the roots
        

        proposed_move_dimensionless += eta * movement_direction
        distance_travelled_in_move += np.abs(eta)

        move_num += 1

        return proposed_move_dimensionless, move_num, eta, distance_travelled_in_move