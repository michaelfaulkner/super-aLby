"""Module for the MeanParticleSeparationSampler class."""
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import number_of_particle_pairs, number_of_particles, size_of_particle_space
from .sampler import Sampler
import logging
import numpy as np


class MeanParticleSeparationSampler(Sampler):
    """
    Class for taking observations of the mean particle-particle separation distances.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the MeanParticleSeparationSampler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        output_directory : str
            The filename onto which the sample is written at the end of the run.

        Raises
        ------
        base.exceptions.ConfigurationError
            If dimensionality_of_particle_space does not equal 1.
        """
        super().__init__(output_directory)
        for component in np.atleast_1d(size_of_particle_space):
            if component is None:
                raise ConfigurationError(f"Give a float for each component of size_of_particle_space in [ModelSettings]"
                                         f" when using {self.__class__.__name__} as {self.__class__.__name__} is "
                                         f"designed for toroidal systems.")
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
        return np.zeros((total_number_of_iterations + 1, 1))

    def get_observation(self, momenta, positions):
        """
        Returns an observation of the system for the given particle momenta and positions.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        float
            The observation of the mean of all shortest (on the torus) particle-separation vectors.
        """
        return sum([np.linalg.norm(get_shortest_vectors_on_torus(positions[i] - positions[j])) for i in
                    range(number_of_particles) for j in range(i + 1, number_of_particles)]) / number_of_particle_pairs

    def output_sample(self, sample):
        """
        Following completion of the Markov chain, print the sample to the output file.

        Parameters
        ----------
        sample : numpy.ndarray
            The sample generated by the Markov chain.
        """
        self._write_sample_to_file(sample, "sample_of_mean_particle_separations.csv")

    def get_sample(self):
        """
        In order to analyse the Markov chain in an external program, read the sample from the output file.

        Returns
        ----------
        numpy.ndarray
            The sample generated by the Markov chain.
        """
        return self._read_sample_from_file("sample_of_mean_particle_separations.csv")
