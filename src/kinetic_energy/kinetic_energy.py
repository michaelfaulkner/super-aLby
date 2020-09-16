"""Module for the abstract KineticEnergy class."""
from abc import ABCMeta, abstractmethod


class KineticEnergy(metaclass=ABCMeta):
    """
    Abstract class for kinetic energies used in the algorithm code.

    A general kinetic-energy class provides the function itself and its gradient.
    """

    def __init__(self, power=2, prefactor=1.0, **kwargs):
        """
        The constructor of the KineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        power : int
            Either the power to which each momentum variable is raised (the generalised-power case) or twice the power
            to which each momentum-dependent part of the relativistic kinetic energy are raised (the super-relativistic
            case).
        prefactor : float, optional
            A general multiplicative prefactor of the potential (and therefore of the kinetic energy).
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ValueError
            If the power equals 0.
        base.exceptions.ValueError
            If the prefactor equals 0.0.
        """
        if power == 0:
            raise ValueError("Give a value not equal to 0 as the power associated with the kinetic energy {0}.".format(
                self.__class__.__name__))
        if prefactor == 0.0:
            raise ValueError("Give a value not equal to 0.0 as the prefactor for the potential {0}.".format(
                self.__class__.__name__))
        self._power = power
        self._prefactor = prefactor
        super().__init__(**kwargs)

    @abstractmethod
    def gradient(self, momentum):
        """
        Return the gradient of the potential for certain separation and charges.

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
    def potential(self, momentum):
        """
        Return the potential function for certain separations and charges.

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
