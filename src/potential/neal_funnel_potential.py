"""Module for the NealFunnelPotential class."""
from .potential import Potential
import math
import numpy as np


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

    def current_value(self, support_variable, charges=None):
        """
        Returns the potential for the given support_variable.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.
        charges : optional
            All the charges needed to calculate the potential; not used in this potential class.

        Returns
        -------
        float
            The potential.
        """
        return support_variable[0] ** 2 / 18.0 + 9 * support_variable[0] / 2.0 + (
                math.exp(-support_variable[0]) * np.sum(support_variable[1:len(support_variable)] ** 2) / 2.0)

    def gradient(self, support_variable, charges=None):
        """
        Returns the gradient of the potential for the given support_variable.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.
        charges : optional
            All the charges needed to calculate the gradient; not used in this potential class.

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
