"""Module for the UniformRefreshmentDistribution class."""
import numpy as np
from base.exceptions import ConfigurationError
from model_settings import size_of_particle_space, number_of_particles
from refreshment_distribution.refreshment_distribution import RefreshmentDistribution


class UniformRefreshmentDistribution(RefreshmentDistribution):
    """
    Class for sampling the velocity refreshment distance within the Event Chain algorithm from a uniform distribution.
    """

    def __init__(self, normalised_lower_limit: float = 0.0,
                 normalised_upper_limit: float = size_of_particle_space / number_of_particles):
        """
        The constructor of the UniformRefreshmentDistribution class.

        Parameters
        ----------
        normalised_lower_limit : float or int
            Lower limit of uniform distribution per particle.
        normalised_upper_limit : float or int
            Upper limit of uniform distribution per particle.
        """
        super().__init__()
        if (normalised_lower_limit < 0.0 or normalised_upper_limit < 0.0 or normalised_upper_limit <
                normalised_lower_limit):
            raise ConfigurationError(f"Give values for normalised_lower_limit and normalised_upper_limit >= 0.0 with "
                                     f"normalised_upper_limit >= normalised_lower_limit in {self.__class__.__name__}. "
                                     f"Provided: normalised_lower_limit={normalised_lower_limit}, "
                                     f"normalised_upper_limit={normalised_upper_limit}.")
        self._normalised_lower_limit, self._normalised_upper_limit = (
            number_of_particles * normalised_lower_limit, normalised_upper_limit)

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
        return number_of_particles * np.random.uniform(self._normalised_lower_limit, self._normalised_upper_limit, 1)
