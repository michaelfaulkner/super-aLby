"""Module for the ToroidalPositionSampler class."""
from .position_sampler import PositionSampler
from base.logging import log_init_arguments
from base.vectors import get_shortest_vector_on_ring, get_shortest_vector_on_torus
from model_settings import dimensionality_of_particle_space
import logging
import numpy as np


class ToroidalPositionSampler(PositionSampler):
    """
    Class for taking observations of particle positions on the torus, i.e., corrected for periodic boundaries.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the ToroidalPositionSampler class.

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
        if dimensionality_of_particle_space == 1:
            return np.array([get_shortest_vector_on_ring(position, 0) for position in positions])
        return np.array([get_shortest_vector_on_torus(position) for position in positions])
