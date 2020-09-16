"""Module for the SmoothPinballLossPotential class."""
import logging
import math
import numpy as np
from base.logging import log_init_arguments
from .potential import Potential


# noinspection PyMethodOverriding
class SmoothPinballLossPotential(Potential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0

    x is an n-dimensional vector of floats.
    """

    def __init__(self, tau, sigma, lambda_hyperparameter, x, y, xi=0.1, q=2.0, prefactor=1.0):
        """
        The constructor of the NealFunnel class.

        Parameters
        ----------
        tau : float
            Quantile number.
        sigma : float
            Learning rate / observation noise.
        lambda_hyperparameter : float
            Regularising factor in prior.
        x : float
            Design matrix (measured data).
        y : float
            Response (measured data).
        xi : float
            Pinball smoother (xi --> 0 recovers nonsmooth pinball loss).
        q : float
            Power used in prior.
        prefactor : float
            The prefactor k of the potential.
        """
        super().__init__(prefactor=prefactor)
        self._tau = tau
        self._sigma = sigma
        self._xi = xi
        self._xi_dot_sigma = xi * sigma
        self._lambda_hyperparameter = lambda_hyperparameter
        self._q = q
        self._x = x
        self._y = y
        self._beta_function_value = self.__beta_function(xi * (1 - tau), xi * tau)
        self._x_sum = np.sum(self._x, axis=0)

    def gradient(self, support_variable):
        """
        Return the gradient of the potential.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        numpy array
            The gradient.
        """
        prior_gradient = np.array(
            [self._q * self._lambda_hyperparameter * np.sign(component) * np.absolute(component) ** (self._q - 1) for
             component in support_variable])
        logistic_term = self.__logistic_function((self._y - np.inner(self._x, support_variable)) / self._xi_dot_sigma)
        mid_term = np.array([np.inner(logistic_term, self._x[:, i]) for i in range(len(support_variable))])
        return (1 - self._tau) / self._sigma * self._x_sum + 1 / self._sigma * mid_term + prior_gradient

    def potential(self, support_variable):
        """
        Return the potential for the given separation.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the Bayesian
            parameter value.

        Returns
        -------
        float
            The potential.
        """
        x_dot_beta = np.inner(self._x, support_variable)
        pinball_loss = (self._tau - 1) * (self._y - x_dot_beta) / self._sigma + self._xi * np.logaddexp(
            0.0, (self._y - x_dot_beta) / self._xi_dot_sigma) + np.log(self._xi * self._sigma * self._beta_function_value)
        prior_vec = np.absolute(support_variable) ** self._q
        return np.sum(pinball_loss) + self._lambda_hyperparameter * np.sum(prior_vec)

    @staticmethod
    def __beta_function(self, a, b):
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    @staticmethod
    def __logistic_function(self, a):
        return np.exp(-np.logaddexp(0, -a))
