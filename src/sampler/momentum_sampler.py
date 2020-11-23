"""Module for the MomentumSampler class."""
from base.logging import log_init_arguments
from model_settings import number_of_particles, dimensionality_of_particle_space
from .sampler import Sampler
import logging
import numpy as np


class MomentumSampler(Sampler):
    """
    Class for taking observations of the momenta of the system.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the MomentumSampler class.

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

    def initialise_sample_array(self, total_number_of_iterations):
        """
        Generate array that stores the sample.

        Parameters
        ----------
        total_number_of_iterations : int
            The total number of iterations of the Markov chain.

        Returns
        -------
        numpy.ndarray
            Numpy array of zeros of the required structure.
        """
        if dimensionality_of_particle_space == 1:
            return np.zeros((total_number_of_iterations + 1, number_of_particles))
        else:
            return np.zeros((total_number_of_iterations + 1, number_of_particles, dimensionality_of_particle_space))

    def get_observation(self, momentum, position):
        """
        Return the observation after each iteration of the Markov chain.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momenta associated with each positions.
        position : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            The observation of the momenta.
        """
        return momentum

    def output_sample(self, sample):
        """
        Following completion of the Markov chain, print the sample to the output file.

        Parameters
        ----------
        sample : numpy.ndarray
            The sample generated by the Markov chain.
        """
        self._write_sample_to_file(sample, "sample_of_momenta.csv")

    def get_sample(self):
        """
        In order to analyse the Markov chain in an external program, read the sample from the output file.

        Returns
        ----------
        numpy.ndarray
            The sample generated by the Markov chain.
        """
        return self._read_sample_from_file("sample_of_momenta.csv")
