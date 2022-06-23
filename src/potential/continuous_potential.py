"""Module for the abstract ContinuousPotential class."""
from .potential import Potential
from abc import ABCMeta, abstractmethod
from model_settings import dimensionality_of_particle_space, number_of_particles, range_of_initial_particle_positions
import numpy as np


class ContinuousPotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for continuous potentials used in the algorithm code.

    A general continuous-potential class provides the function itself, its gradient and the potential difference
        resulting from the displacement of a single particle.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the ContinuousPotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor, **kwargs)

    @abstractmethod
    def get_value(self, positions):
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
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, positions):
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
        raise NotImplementedError

    @abstractmethod
    def get_potential_difference(self, active_particle_index, candidate_position, positions):
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
        raise NotImplementedError

    def initialised_position_array(self):
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
        if dimensionality_of_particle_space == 1:
            if type(range_of_initial_particle_positions) == float:
                return np.array(
                    [np.atleast_1d(range_of_initial_particle_positions) for _ in range(number_of_particles)])
            else:
                return np.array([np.atleast_1d(np.random.uniform(*range_of_initial_particle_positions))
                                 for _ in range(number_of_particles)])
        else:
            if type(range_of_initial_particle_positions[0]) == float:
                return np.array([range_of_initial_particle_positions for _ in range(number_of_particles)])
            else:
                return np.array([[np.random.uniform(*axis_range) for axis_range in range_of_initial_particle_positions]
                                 for _ in range(number_of_particles)])
