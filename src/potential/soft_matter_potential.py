"""Module for the abstract SoftMatterPotential class."""
from .potential import Potential
from base.exceptions import ConfigurationError
from model_settings import range_of_initial_particle_positions
from abc import ABCMeta, abstractmethod
import numpy as np


class SoftMatterPotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for soft-matter potentials used in the algorithm code.

    A general potential class provides the function itself and its gradient.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the SoftMatterPotential class.

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
            If the model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        """
        conditions = [type(component) == list and len(component) == 2 and type(bound) == float
                      for component in range_of_initial_particle_positions for bound in component]
        for condition in np.atleast_1d(conditions):
            if not condition:
                raise ConfigurationError(f"For each component of range_of_initial_particle_positions, give a list of "
                                         f"two float values to avoid numerical divergences (due to initial "
                                         f"configuration) in {self.__class__.__name__}.")
        super().__init__(prefactor, **kwargs)

    @abstractmethod
    def get_value(self, positions):
        """
        Return the potential function for certain separations and charges.

        Parameters
        ----------
        positions : numpy.ndarray
            One or many particle-particle separation vectors {r_ij}.

        Returns
        -------
        float
            The potential function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, positions):
        """
        Return the gradient of the potential for certain separation and charges.

        Parameters
        ----------
        positions : numpy.ndarray
            One or many particle-particle separation vectors {r_ij}.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError
