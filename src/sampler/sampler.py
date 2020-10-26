"""Module for the abstract Sampler class."""
from abc import ABCMeta, abstractmethod
import numpy as np
import os


class Sampler(metaclass=ABCMeta):
    """
    Abstract class for taking observations of the state of the system.
    """

    def __init__(self, output_directory: str, **kwargs):
        """
        The constructor of the Sampler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        output_directory : str
            The name of the directory into which the sample file is written at the end of the run.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        self._output_directory = output_directory
        os.makedirs(self._output_directory, exist_ok=True)
        super().__init__(**kwargs)

    @abstractmethod
    def get_observation(self, momentum, position):
        """
        Return the observation after each iteration of the Markov chain.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each position.
        position : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.

        Returns
        -------
        numpy_array
            The observation.
        """
        raise NotImplementedError

    @abstractmethod
    def output_sample(self, sample):
        """
        Following completion of the Markov chain, print the sample to the output file.

        Parameters
        ----------
        sample : numpy_array
            The sample generated by the Markov chain.
        """
        raise NotImplementedError

    @abstractmethod
    def initialise_sample_array(self, total_number_of_iterations):
        """
        Generate array that stores the sample.

        Parameters
        ----------
        total_number_of_iterations : int
            The total number of iterations of the Markov chain.

        Returns
        -------
        ndarray
            Numpy array of zeros of the required structure.
        """
        raise NotImplementedError

    def _write_sample_to_file(self, sample, sample_file_string):
        with open(os.path.join(os.getcwd(), self._output_directory, sample_file_string), 'w') as file:
            np.savetxt(file, np.reshape(sample, (sample.shape[0], -1)), delimiter=',')
