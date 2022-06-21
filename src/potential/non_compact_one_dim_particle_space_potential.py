"""Module for the abstract NonCompactOneDimParticleSpacePotential class."""
from .potential import Potential
from base.exceptions import ConfigurationError
from model_settings import dimensionality_of_particle_space, size_of_particle_space
from abc import ABCMeta, abstractmethod
import numpy as np


class NonCompactOneDimParticleSpacePotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for potentials restricted to non-compact one-dimensional particle space.

    A general potential class provides the function itself and its gradient.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the NonCompactOneDimParticleSpacePotential class.

        This abstract class verifies that i) element is None for each element of size_of_particle_space, and ii) the
        dimensionality of particle space is one. The static method _get_higher_dimension_array() is also provided,
        which is currently used as a workaround such that its child classes work with the new form of the positions
        array: positions = [[a] [b] [c]] (as opposed to [a b c] used in the Biometrika project). No other additional
        functionality is provided.

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
            If element is not None for element in size_of_particle_space.
        base.exceptions.ConfigurationError
            If dimensionality_of_particle_space does not equal 1.
        """
        super().__init__(prefactor, **kwargs)
        for element in size_of_particle_space:
            if element is not None:
                raise ConfigurationError(f"For each component of size_of_particle_space, give None when using "
                                         f"{self.__class__.__name__}.")
        if dimensionality_of_particle_space != 1:
            raise ConfigurationError(f"Give either None or a list of two float values for size_of_particle_space when "
                                     f"using {self.__class__.__name__} as {self.__class__.__name__} is restricted to "
                                     f"one-dimensional particle space.")

    @abstractmethod
    def get_value(self, positions):
        """
        Returns the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

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
            is a float and represents one Cartesian component of the position of a single particle.

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

    @staticmethod
    def _get_higher_dimension_array(array):
        new_dimensionality_of_array = [component for component in array.shape]
        new_dimensionality_of_array.append(-1)
        return np.reshape(array, tuple(new_dimensionality_of_array))
