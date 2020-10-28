"""Module for the TDistributionKineticEnergy class."""
from base.logging import log_init_arguments
from .kinetic_energy import KineticEnergy
import logging
import numpy as np


class TDistributionKineticEnergy(KineticEnergy):
    """
    This class implements the t-distribution kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, nu: int = 1):
        """
        The constructor of the ExponentialPowerKineticEnergy class.

        Parameters
        ----------
        nu : int
            Number of degrees of freedom.

        Raises
        ------
        base.exceptions.ValueError
            If n is less than 1.
        """
        if nu < 1:
            raise ValueError("Give a value not less than 1 as the number of degrees of freedom of the t-distribution "
                             "kinetic energy {0}.".format(self.__class__.__name__))
        self._nu = float(nu)
        self._nu_minus_one = self._nu - 1.0
        self._nu_plus_one = self._nu + 1.0
        self._nu_plus_one_over_two = 0.5 * self._nu_plus_one
        self._one_over_nu = 1.0 / self._nu
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, nu=nu)

    def get_value(self, momentum):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each position.

        Returns
        -------
        float
            The kinetic energy.
        """
        return np.sum(self._nu_plus_one_over_two * np.log(self._one_over_nu * (1.0 + momentum ** 2)))

    def get_gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each position.

        Returns
        -------
        numpy array
            The gradient of the kinetic energy.
        """
        return self._nu_plus_one * momentum / (self._nu + momentum ** 2)

    def get_momentum_observation(self, momentum):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Parameters
        ----------
        momentum : numpy_array
            The current momentum associated with each position.

        Returns
        -------
        numpy_array
            A new momentum associated with each position.
        """
        return np.random.standard_t(df=self._nu, size=len(momentum))
