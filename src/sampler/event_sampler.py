"""Module for the EventSampler class."""
from .sampler import Sampler
from abc import ABCMeta, abstractmethod


class EventSampler(Sampler, metaclass=ABCMeta):
    """
    Abstract class for taking observations of the system at event times.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the EventSampler class.

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

    def get_empty_sample_array(self):
        """
        Generate a Python list to store the event sample.

        Returns
        -------
        Sequence[float or int]
            Variable length list to account for random number of events.
        """
        return []

    @abstractmethod
    def get_observation(self, positions, potential, active_particle_index, vetoing_index, distance_to_next_event):
        """
        Returns an observation of the system at an event for the given system state.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.
        potential : float or potential.potential.Potential
            If a float, the current value of the potential; otherwise, an instance of the chosen child class of
            potential.potential.Potential.
        active_particle_index : int
            The active particle index
        vetoing_index : int
            The index of the particle that triggers the event.
        distance_to_next_event : float
            Distance to next ECMC event.
        """
        raise NotImplementedError
