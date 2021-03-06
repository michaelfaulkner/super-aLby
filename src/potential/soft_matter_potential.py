"""Module for the abstract SoftMatterPotential class."""
from .potential import Potential
from base.exceptions import ConfigurationError
from model_settings import range_of_initial_particle_positions, size_of_particle_space
from abc import ABCMeta, abstractmethod
import numpy as np


class SoftMatterPotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for soft-matter potentials, which are potentials that are functions of particle-separation vectors.

    A general potential class provides the function itself and its gradient.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the SoftMatterPotential class.

        This abstract class verifies that i) type(element) is np.float64 for each element of size_of_particle_space,
        and ii) the initial particle positions do not all coincide, as this would generate divergences. No other
        additional functionality is provided.

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
            If type(element) is not np.float64 for element in size_of_particle_space.
        base.exceptions.ConfigurationError
            If model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        """
        super().__init__(prefactor, **kwargs)
        for element in size_of_particle_space:
            if type(element) != np.float64:
                raise ConfigurationError(f"For each component of size_of_particle_space, give a float value when using "
                                         f"{self.__class__.__name__}.")
        conditions = [type(component) == list and len(component) == 2 and type(bound) == float
                      for component in range_of_initial_particle_positions for bound in component]
        for condition in np.atleast_1d(conditions):
            if not condition:
                raise ConfigurationError(f"For each component of range_of_initial_particle_positions, give a list of "
                                         f"two float values to avoid numerical divergences (due to initial "
                                         f"configuration) in {self.__class__.__name__}.")

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
