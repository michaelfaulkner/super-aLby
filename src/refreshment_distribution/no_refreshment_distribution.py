"""Module for the NoRefreshmentDistribution class."""
import numpy as np
from model_settings import number_of_particles
from base.exceptions import ConfigurationError
from refreshment_distribution.refreshment_distribution import RefreshmentDistribution

class NoRefreshmentDistribution(RefreshmentDistribution):
    """
    Class for returning no velocity refreshments, i.e. having an infinite refreshment distance, within the event-chain
    Monte Carlo algorithm.
    """

    def __init__(self):
        """
        The constructor of the NoRefreshmentDistribution class.

        Parameters
        ----------
        None
        """
        super().__init__()
        self._refreshment_distance = np.inf

    def get_refreshment_distance(self):
        """
        Returns an infinite velocity refreshment distance, the cumulative distance through the state space active particles
        in a single chain move before the velocity is resampled within the Event Chain algorithm, effectively ensuring
        that no refreshments occur.

        Parameters
        ----------

        Returns
        -------
        numpy.ndarray
            A one-dimensional numpy array with single element equal to the refreshment distance.
        """
        return np.array([self._refreshment_distance])
