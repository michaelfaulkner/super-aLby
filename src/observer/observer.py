"""Module for the abstract Observer class."""
from abc import ABCMeta, abstractmethod


class Observer(metaclass=ABCMeta):
    """
    Abstract class for taking observations of the state of the system.
    """

    def __init__(self, **kwargs):
        """
        The constructor of the Observer class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def get_observation(self, momentum, position, charges=None):
        """
        Return the observation after each iteration of the Markov chain.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each position.
        position : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        charges : optional
            All the charges needed to calculate the potential and its gradient.

        Returns
        -------
        numpy_array
            The observation.
        """
        raise NotImplementedError
