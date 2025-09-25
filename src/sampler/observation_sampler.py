"""Module for the ObservationSampler class."""
from .sampler import Sampler
from abc import ABCMeta, abstractmethod


class ObservationSampler(Sampler, metaclass=ABCMeta):
    """
    Abstract class for taking observations of the state of the system at observation events.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the ObservationSampler class.

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

    @abstractmethod
    def get_empty_sample_array(self, total_number_of_iterations):
        """
        Generate numpy array to store the sample.

        Parameters
        ----------
        total_number_of_iterations : int
            The total number of iterations of the Markov chain.

        Returns
        -------
        numpy.ndarray
            Numpy array of zeros of the required structure.
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        raise NotImplementedError
