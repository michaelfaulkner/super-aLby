"""Module for the abstract NoiseDistribution class."""
from abc import ABCMeta, abstractmethod
from typing import Union


class NoiseDistribution(metaclass=ABCMeta):
    """Abstract class for noise distributions."""

    def __init__(self, initial_width_of_noise_distribution=Union[None, float], **kwargs):
        """
        The constructor of the NoiseDistribution class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        initial_width_of_noise_distribution : None or float
            If a continuous noise distribution, this is the initial width of the noise distribution (the standard
            deviation if a non-compact distribution); otherwise, we require None.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self.initial_width_of_noise_distribution = initial_width_of_noise_distribution
        self.width_of_noise_distribution = initial_width_of_noise_distribution

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
            is a float or int and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float or int
            and represents one Cartesian component of the proposed position of the active particle.
        """
        raise NotImplementedError
