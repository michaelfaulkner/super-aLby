"""Module for the abstract Potential class."""
from abc import ABCMeta, abstractmethod


class Potential(metaclass=ABCMeta):
    """
    Abstract class for potentials used in the algorithm code.

    A general potential class provides the function itself and its gradient.
    """

    def __init__(self, prefactor=1.0, **kwargs):
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
            If the prefactor equals 0.
        """
        if prefactor == 0.0:
            raise ValueError("Give a prefactor not equal to 0.0 as the prefactor for the potential {0}.".format(
                self.__class__.__name__))
        self._prefactor = prefactor
        super().__init__(**kwargs)

    @abstractmethod
    def gradient(self, support_variable, charges=None):
        """
        Return the gradient of the potential for certain separation and charges.

        Parameters
        ----------
        support_variable
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        charges : optional
            All the charges needed to calculate the derivative.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError

    @abstractmethod
    def potential(self, support_variable, charges=None):
        """
        Return the potential function for certain separations and charges.

        Parameters
        ----------
        support_variable
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        charges : optional
            All the charges needed to calculate the potential function.

        Returns
        -------
        float
            The potential function.
        """
        raise NotImplementedError
