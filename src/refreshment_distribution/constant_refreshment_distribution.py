"""Module for the ConstantRefreshmentDistribution class."""
import numpy as np
from model_settings import number_of_particles
from base.exceptions import ConfigurationError
from refreshment_distribution.refreshment_distribution import RefreshmentDistribution


class ConstantRefreshmentDistribution(RefreshmentDistribution):
    """
    Class for returning a constant velocity-refreshment distance within the event-chain Monte Carlo algorithm.
    """

    def __init__(self, normalised_refreshment_distance: float = 1.0):
        """
        The constructor of the ConstantRefreshmentDistribution class.

        N.B. We recommend setting the value of normalised_refreshment_distance to min(size_of_particle_space) or simply
            to one if this value is O(1).  For the 2D hard-disk model, the value of min(size_of_particle_space) is
            disk_radius * (pi * number_of_particles / packing_fraction) ** 0.5 for a square aspect ratio.

        N.B. This velocity-refreshment distribution should not be used for the 1D hard-disk model.  This is because
            the Markov process then fails to converge at certain packing fractions, as it becomes too deterministic.

        Parameters
        ----------
        normalised_refreshment_distance : float or int
            Constant velocity refreshment distance per particle.
        """
        super().__init__(normalised_refreshment_distance)
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
