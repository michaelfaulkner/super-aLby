"""Module for the abstract ContinuousNoiseDistribution class."""
from .noise_distribution import NoiseDistribution
from base.exceptions import ConfigurationError
from abc import ABCMeta, abstractmethod


class ContinuousNoiseDistribution(NoiseDistribution, metaclass=ABCMeta):
    """Abstract class for continuous noise distributions."""

    def __init__(self, initial_width_of_noise_distribution: float = 0.1, **kwargs):
        """
        The constructor of the ContinuousNoiseDistribution class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        initial_width_of_noise_distribution : float
            The initial width of the noise distribution (the standard deviation if a non-compact distribution).
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(initial_width_of_noise_distribution, **kwargs)
        if initial_width_of_noise_distribution <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as initial_width_of_noise_distribution in "
                                     f"{self.__class__.__name__}.")

    @abstractmethod
    def get_candidate_position(self, active_particle_index, positions):
        """
        Returns a candidate position for the active particle in the Metropolis algorithm.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the proposed position of the active particle.
        """
        raise NotImplementedError
