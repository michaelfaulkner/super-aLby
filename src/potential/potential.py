"""Module for the abstract Potential class."""
from base.exceptions import ConfigurationError
from abc import ABCMeta, abstractmethod


class Potential(metaclass=ABCMeta):
    """
    Abstract class for potentials used in the algorithm code.

    A general potential class provides the function itself and its gradient.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the Potential class.

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
        if prefactor == 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as prefactor for {self.__class__.__name__}.")
        self._prefactor = prefactor
        super().__init__(**kwargs)

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
