"""Module for the MomentumObserver class."""
from .observer import Observer


class MomentumObserver(Observer):
    """
    Class for taking observations of the momentum of the system.
    """

    def __init__(self, filename: str):
        """
        The constructor of the MomentumObserver class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        filename : str
            The filename onto which the sample is written at the end of the run.
        """
        super().__init__(filename)

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
            The observation of the momentum.
        """
        return momentum
