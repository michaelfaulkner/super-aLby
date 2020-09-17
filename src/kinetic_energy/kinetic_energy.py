"""Module for the abstract KineticEnergy class."""
from abc import ABCMeta, abstractmethod


class KineticEnergy(metaclass=ABCMeta):
    """
    Abstract class for kinetic energies used in the algorithm code.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
        observation of the momentum.
    """

    def __init__(self, power=2, **kwargs):
        """
        The constructor of the KineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        power : int
            Either the power to which each momentum component is raised (the generalised-power case) or twice the power
            to which each momentum-dependent part of the relativistic kinetic energy is raised (the super-relativistic
            case). For potentials with leading order term |x|^a, the optimal choice that ensures robust dynamics is
            given by power = 1 + 1 / (a - 1) for a >= 2 and power = 1 + 1 / (a + 1) for a <= -1.
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
        self._one_over_power = 1 / power
        self._power = power
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
    def kinetic_energy(self, momentum):
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
