"""Module for the abstract OneDimensionalParticleSpacePotential class."""
from .potential import Potential
from base.exceptions import ConfigurationError
from model_settings import dimensionality_of_particle_space
from abc import ABCMeta, abstractmethod


class OneDimensionalParticleSpacePotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for potentials restricted to one-dimensional particle space.

    A general potential class provides the function itself and its gradient.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the OneDimensionalParticleSpacePotential class.

        This abstract class verifies that the initial particle positions do not all coincide, as this would generate
        divergences. No other additional functionality is provided.

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
            If dimensionality_of_particle_space does not equal 1.
        """
        if dimensionality_of_particle_space != 1:
            raise ConfigurationError(f"Give either None or a list of two float values for size_of_particle_space when "
                                     f"using {self.__class__.__name__} as {self.__class__.__name__} is restricted to "
                                     f"one-dimensional particle space.")
        super().__init__(prefactor, **kwargs)

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
