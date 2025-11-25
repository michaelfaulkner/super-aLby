"""Module for the abstract RefreshmentDistribution class."""
from abc import ABCMeta, abstractmethod
from model_settings import number_of_particles


class RefreshmentDistribution(metaclass=ABCMeta):
    """Abstract class for velocity-refreshment distributions within the event-chain Monte Carlo algorithm."""

    def __init__(self, normalised_refreshment_lengthscale=1.0, **kwargs):
        """
        The constructor of the RefreshmentDistribution class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self.normalised_refreshment_lengthscale = normalised_refreshment_lengthscale

    @abstractmethod
    def get_refreshment_distance(self):
        """
        Returns a velocity refreshment distance, the cumulative distance through the state space active particles
        in a single chain move before the velocity is resampled within the Event Chain algorithm.

        Parameters
        ----------

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array with single element equal to the refreshment distance.
        """
        raise NotImplementedError
