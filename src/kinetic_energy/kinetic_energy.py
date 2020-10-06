"""Module for the abstract KineticEnergy class."""
from abc import ABCMeta, abstractmethod


class KineticEnergy(metaclass=ABCMeta):
    """
    Abstract class for kinetic energies used in the algorithm code.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
        observation of the momentum.
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
    def gradient(self, momentum):
        """
        Return the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError

    @abstractmethod
    def current_value(self, momentum):
        """
        Return the kinetic-energy function.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The kinetic-energy function.
        """
        raise NotImplementedError

    @abstractmethod
    def momentum_observation(self, momentum):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Parameters
        ----------
        momentum : numpy_array
            The current momentum associated with each support_variable.

        Returns
        -------
        numpy_array
            A new momentum associated with each support_variable.
        """
        raise NotImplementedError
