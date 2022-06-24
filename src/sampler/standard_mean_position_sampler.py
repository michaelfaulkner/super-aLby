"""Module for the StandardMeanPositionSampler class."""
from .mean_position_sampler import MeanPositionSampler
from base.logging import log_init_arguments
import logging
import numpy as np


class StandardMeanPositionSampler(MeanPositionSampler):
    """
    Class for taking observations of mean particle positions without correcting for periodic boundaries.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the StandardMeanPositionSampler class.

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
        Returns an observation of the system for the given particle momenta and positions.

        Parameters
        ----------
        momenta : None or numpy.ndarray
            None or a two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each
            element is a float and represents one Cartesian component of the momentum of a single particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            The observation of the positions.
        """
        return np.mean(positions, axis=0)
