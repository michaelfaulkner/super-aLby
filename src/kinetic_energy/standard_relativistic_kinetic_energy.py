"""Module for the RelativisticKineticEnergy class."""
from .relativistic_kinetic_energy import RelativisticKineticEnergy
import numpy as np

import rpy2.robjects.packages as r_packages
import rpy2.robjects.numpy2ri as numpy2ri
adaptive_rejection_sampling = r_packages.importr('ars')
numpy2ri.activate()


class StandardRelativisticKineticEnergy(RelativisticKineticEnergy):
    """
    This class implements the relativistic kinetic energy K = sum((1 + gamma^(-1) p[i] ** 2) ** (1 / 2))
    """

    def __init__(self, gamma=1.0):
        """
        The constructor of the RelativisticKineticEnergy class.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the kinetic energy transforms from
            Gaussian to generalised-power behaviour.
        """
        super().__init__(gamma=gamma)

    def gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        numpy array
            The gradient of the kinetic energy.
        """
        return self._one_over_gamma * momentum * (1 + self._one_over_gamma * momentum ** 2) ** (- 0.5)

    def kinetic_energy(self, momentum):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The kinetic energy.
        """
        return np.sum((1 + self._one_over_gamma * momentum ** 2) ** 0.5)

    def momentum_observation(self, momentum):
        """
        Returns an observation of the momentum from the kinetic-energy distribution using adaptive rejection sampling.

        Parameters
        ----------
        momentum : numpy_array
            The current momentum associated with each support_variable.

        Returns
        -------
        numpy_array
            A new momentum associated with each support_variable.
        """
        return np.array(adaptive_rejection_sampling.ars(len(momentum), - self.kinetic_energy(momentum), - self.gradient(momentum)))

