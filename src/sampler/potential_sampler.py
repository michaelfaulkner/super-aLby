"""Module for the PotentialSampler class."""
from .sampler import Sampler
from base.logging import log_init_arguments
import logging
import numpy as np


class PotentialSampler(Sampler):
    """
    Class for taking observations of the potential.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the PotentialSampler class.

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
        return np.zeros((total_number_of_iterations + 1, 1))

    def get_observation(self, momenta, positions, potential):
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
        potential : float or potential.potential.Potential
            If a float, the current value of the potential; otherwise, an instance of the chosen child class of
            potential.potential.Potential.

        Returns
        -------
        float
            The observation of the potential.
        """
        if type(potential) == np.float64:
            return potential
        return potential.get_value(positions)

    def output_sample(self, sample, temperature_index):
        """
        Following completion of the Markov chain, print the sample to the output file.

        Parameters
        ----------
        sample : numpy.ndarray
            The sample generated by the Markov chain.
        temperature_index : int
            The index of the iteration through the list sampling temperatures.
        """
        self._write_sample_to_file(sample, f"temperature_{temperature_index:02d}_sample_of_potential.npy")

    def get_sample(self, temperature_index):
        """
        In order to analyse the Markov chain in an external program, read the sample from the output file.

        Parameters
        ----------
        temperature_index : int
            The index of the iteration through the list sampling temperatures.

        Returns
        ----------
        numpy.ndarray
            The sample generated by the Markov chain.
        """
        return self._read_sample_from_file(f"temperature_{temperature_index:02d}_sample_of_potential.npy")
