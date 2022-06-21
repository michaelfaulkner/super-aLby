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
    def get_finite_change_in_position(self, number_of_active_particles):
        """
        Returns a proposed finite change in position for each active particle.

        Parameters
        ----------
        number_of_active_particles : int
            An integer representing the number of active particles requiring a proposed change in position.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_active_particles, dimensionality_of_particle_space); each
            element is a float and represents one Cartesian component of the proposed change in position of a single
            active particle.
        """
        raise NotImplementedError
