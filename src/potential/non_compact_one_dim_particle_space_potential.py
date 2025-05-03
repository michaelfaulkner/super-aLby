"""Module for the abstract NonCompactOneDimParticleSpacePotential class."""
from .continuous_potential import ContinuousPotential
from base.exceptions import ConfigurationError
from model_settings import dimensionality_of_particle_space, size_of_particle_space
from abc import ABCMeta
import numpy as np


class NonCompactOneDimParticleSpacePotential(ContinuousPotential, metaclass=ABCMeta):
    """
    Abstract class for potentials restricted to non-compact, continuous one-dimensional particle space.
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

    @staticmethod
    def _get_higher_dimension_array(array):
        new_dimensionality_of_array = [component for component in array.shape]
        new_dimensionality_of_array.append(-1)
        return np.reshape(array, tuple(new_dimensionality_of_array))
