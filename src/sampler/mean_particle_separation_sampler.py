"""Module for the ParticleSeparationSampler class."""
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vector_on_torus
from model_settings import dimensionality_of_particle_space, number_of_particle_pairs, number_of_particles, \
    size_of_particle_space
from .sampler import Sampler
import logging
import numpy as np


class MeanParticleSeparationSampler(Sampler):
    """
    Class for taking observations of the positions of the system.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the ParticleSeparationSampler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        output_directory : str
            The filename onto which the sample is written at the end of the run.
        """
        if dimensionality_of_particle_space == 1:
            raise ConfigurationError("Cannot use {0} when size_of_particle_space is one dimensional (i.e., when it is "
                                     "equal to None or a float).".format(self.__class__.__name__))
        for component in np.atleast_1d(size_of_particle_space):
            if component is None:
                raise ConfigurationError(
                    "All components of size_of_particle_space must be floats when using {0}.".format(
                        self.__class__.__name__))
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
        return np.zeros((total_number_of_iterations + 1, number_of_particle_pairs))

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
        float
            The observation of the mean of all shortest (on the torus) particle-separation vectors.
        """
        return sum(
            [np.linalg.norm(get_shortest_vector_on_torus(positions[i] - positions[j]))
             for i in range(number_of_particles) for j in range(i + 1, number_of_particles)]) / number_of_particle_pairs

    def output_sample(self, sample):
        """
        Following completion of the Markov chain, print the sample to the output file.

        Parameters
        ----------
        sample : numpy.ndarray
            The sample generated by the Markov chain.
        """
        self._write_sample_to_file(sample, "sample_of_particle_separations.csv")

    def get_sample(self):
        """
        In order to analyse the Markov chain in an external program, read the sample from the output file.

        Returns
        ----------
        numpy.ndarray
            The sample generated by the Markov chain.
        """
        return self._read_sample_from_file("sample_of_particle_separations.csv")
