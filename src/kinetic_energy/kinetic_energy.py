"""Module for the abstract KineticEnergy class."""
from abc import ABCMeta, abstractmethod


class KineticEnergy(metaclass=ABCMeta):
    """
    Abstract class for kinetic energies used in the algorithm code.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
        observation of the momenta.
    """

    def __init__(self, **kwargs):
        """
        The constructor of the KineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def get_value(self, momenta):
        """
        Return the kinetic-energy function.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The kinetic-energy function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, momenta):
        """
        Return the gradient of the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError

    @abstractmethod
    def get_momentum_observations(self):
        """
        Return an observation of the momenta from the kinetic-energy distribution.

        Returns
        -------
        numpy.ndarray
            A new momenta associated with each positions.
        """
        raise NotImplementedError
