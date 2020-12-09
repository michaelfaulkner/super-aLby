"""Module for the StandardPositionSampler class."""
from .position_sampler import PositionSampler
from base.logging import log_init_arguments
import logging


class StandardPositionSampler(PositionSampler):
    """
    Class for taking observations of particle positions without correcting for periodic boundaries.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the StandardPositionSampler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        output_directory : str
            The filename onto which the sample is written at the end of the run.
        """
        super().__init__(output_directory)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           output_directory=output_directory)

    def get_observation(self, momenta, positions):
        """
        Return the observation after each iteration of the Markov chain.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.
        positions : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            The observation of the positions.
        """
        return positions
