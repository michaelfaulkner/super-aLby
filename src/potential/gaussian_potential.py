"""Module for the GaussianPotential class."""
from .potential import Potential
from base.logging import log_init_arguments
import logging
import numpy as np


class GaussianPotential(Potential):
    """
    This class implements the Gaussian potential U = sum(x[i] ** 2 / 2)
    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the GaussianPotential class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        """
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, prefactor=prefactor)

    def get_value(self, position):
        """
        Returns the potential for the given position.

        Parameters
        ----------
        position : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        float
            The potential.
        """
        return 0.5 * np.sum(position ** 2)

    def get_gradient(self, position):
        """
        Returns the gradient of the potential for the given position.

        Parameters
        ----------
        position : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        numpy.ndarray
            The gradient.
        """
        return position
