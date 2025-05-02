"""Module for the HardDiskPotential class"""
import numpy as np
from .soft_matter_potential import SoftMatterPotential
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
import logging
from model_settings import number_of_particles


class HardDiskPotential(SoftMatterPotential):
    r"""
    This class implements the potential for the hard-disk model...
    """

    def __init__(self, prefactor: float = 1.0):
        r"""
        The constructor of the HardDiskPotential class

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.
        """
        super().__init__(prefactor=prefactor)

    def get_value(self, positions):
        # todo work out how to account for this not being a relevant method
        """
        Returns the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        float
            The potential function.
        """
        pass

    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        # todo work out how to account for this not being a relevant method (you'd probably create a special
        #  HardDiskMetropolisMediator class)
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
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        pass

    def get_gradient(self, positions):
        # todo work out how to account for this not being a relevant method
        """
        Returns the gradient of the potential function for the given particle positions.

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

    @staticmethod
    def get_random_event_chain_velocity():
        """Uniformly samples a direction of motion for the active particle from chosen velocity distribution"""
        pass

    def get_distance_to_next_event_and_veto_index(self, positions, active_particle_index, temperature,
                                                  movement_direction):
        """
        Returns the distance to the next particle event for a given active particle index.

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
        """
        pass

    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """Chooses the index and direction for the next active particle in the markov chain"""
        pass

    @staticmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        # todo check this - copied from copied from QuantumHarmonic Oscillator but don't think it traslates!!!
        """ Updates position of the active particle."""
        positions[active_particle_index] += displacement_distance * movement_direction
