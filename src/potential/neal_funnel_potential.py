"""Module for the NealFunnelPotential class."""
import math
import numpy as np
from .potential import Potential


# noinspection PyMethodOverriding
class NealFunnelPotential(Potential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0

    x is an n-dimensional vector of floats.
    """

    def __init__(self, prefactor=1.0):
        """
        The constructor of the NealFunnel class.

        Parameters
        ----------
        prefactor : float
            The prefactor k of the potential.
        """
        super().__init__(prefactor=prefactor)

    def gradient(self, support_variable):
        """
        Return the gradient of the potential.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        numpy array
            The gradient.
        """
        gradient = np.zeroes(len(support_variable))
        gradient[0] = support_variable[0] / 9.0 + 9 / 2.0 - (
                math.exp(-support_variable[0]) * np.sum(support_variable[1:len(support_variable)] ** 2) / 2.0)
        gradient[1:len(support_variable)] = 2.0 * support_variable[1:len(support_variable)] * math.exp(
            -support_variable[0])
        return gradient

    def potential(self, support_variable):
        """
        Return the potential for the given separation.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        float
            The potential.
        """
        return support_variable[0] ** 2 / 18.0 + 9 * support_variable[0] / 2.0 + (
                math.exp(-support_variable[0]) * np.sum(support_variable[1:len(support_variable)] ** 2) / 2.0)
