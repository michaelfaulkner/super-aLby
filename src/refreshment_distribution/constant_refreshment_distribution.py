"""Module for the ConstantRefreshmentDistribution class."""
import numpy as np
from model_settings import number_of_particles
from base.exceptions import ConfigurationError
from refreshment_distribution.refreshment_distribution import RefreshmentDistribution


class ConstantRefreshmentDistribution(RefreshmentDistribution):
    """
    Class for returning a constant velocity refreshment distance within the Event Chain algorithm.
    """

    def __init__(self, normalised_refreshment_distance: float = 1.0):
        """
        The constructor of the ConstantRefreshmentDistribution class.

        Parameters
        ----------
        normalised_refreshment_distance : float or int
            Constant velocity refreshment distance per particle.
        """
        super().__init__()
        if normalised_refreshment_distance <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for normalised_refreshment_distance in "
                                     f"{self.__class__.__name__}.")
        self._refreshment_distance = number_of_particles * normalised_refreshment_distance

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
        return np.array([self._refreshment_distance])
