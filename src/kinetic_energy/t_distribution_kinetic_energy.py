"""Module for the TDistributionKineticEnergy class."""
from base.logging import log_init_arguments
from .kinetic_energy import KineticEnergy
import logging
import numpy as np


class TDistributionKineticEnergy(KineticEnergy):
    """
    This class implements the t-distribution kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, degrees_of_freedom: int = 1):
        """
        The constructor of the TDistributionKineticEnergy class.

        Parameters
        ----------
        degrees_of_freedom : int
            Number of degrees of freedom of t-distribution.

        Raises
        ------
        base.exceptions.ValueError
            If degrees_of_freedom is less than 1.
        """
        if degrees_of_freedom < 1:
            raise ValueError("Give a value not less than 1 as the number of degrees of freedom of the t-distribution "
                             "kinetic energy {0}.".format(self.__class__.__name__))
        self._degrees_of_freedom = float(degrees_of_freedom)
        self._degrees_of_freedom_minus_one = self._degrees_of_freedom - 1.0
        self._degrees_of_freedom_plus_one = self._degrees_of_freedom + 1.0
        self._degrees_of_freedom_plus_one_over_two = 0.5 * self._degrees_of_freedom_plus_one
        self._one_over_degrees_of_freedom = 1.0 / self._degrees_of_freedom
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           degrees_of_freedom=degrees_of_freedom)

    def get_value(self, momentum):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momentum associated with each position.

        Returns
        -------
        float
            The kinetic energy.
        """
        return np.sum(self._degrees_of_freedom_plus_one_over_two * np.log(self._one_over_degrees_of_freedom *
                                                                          (1.0 + momentum ** 2)))

    def get_gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momentum associated with each position.

        Returns
        -------
        numpy.ndarray
            The gradient of the kinetic energy.
        """
        return self._degrees_of_freedom_plus_one * momentum / (self._degrees_of_freedom + momentum ** 2)

    def get_momentum_observation(self, momentum):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Parameters
        ----------
        momentum : numpy.ndarray
            The current momentum associated with each position.

        Returns
        -------
        numpy.ndarray
            A new momentum associated with each position.
        """
        return np.random.standard_t(df=self._degrees_of_freedom, size=len(momentum))
