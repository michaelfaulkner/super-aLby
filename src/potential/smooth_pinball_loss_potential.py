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
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__(prefactor=prefactor)
        self.tau = tau
        self.sigma = sigma
        self.xi = xi
        self.xi_dot_sigma = xi * sigma
        self.lambda_hyperparameter = lambda_hyperparameter
        self.q = q
        self.x = x
        self.y = y
        self.beta_function_value = self.__beta_function(xi * (1 - tau), xi * tau)
        self.x_sum = np.sum(self.x, axis=0)

    def gradient(self, support_variable):
        """
        Return the gradient of the potential.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, the separation vector r_ij; in this case, the Bayesian parameter value.

        Returns
        -------
        numpy array
            The gradient.
        """
        prior_gradient = np.array(
            [self.q * self.lambda_hyperparameter * np.sign(component) * np.absolute(component) ** (self.q - 1) for
             component in support_variable])
        logistic_term = self.__logistic_function((self.y - np.inner(self.x, support_variable)) / self.xi_dot_sigma)
        mid_term = np.array([np.inner(logistic_term, self.x[:, i]) for i in range(len(support_variable))])
        return (1 - self.tau) / self.sigma * self.x_sum + 1 / self.sigma * mid_term + prior_gradient

    def potential(self, support_variable):
        """
        Return the potential for the given separation.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, the separation vector r_ij; in this case, the Bayesian parameter value.

        Returns
        -------
        float
            The potential.
        """
        x_dot_beta = np.inner(self.x, support_variable)
        pinball_loss = (self.tau - 1) * (self.y - x_dot_beta) / self.sigma + self.xi * np.logaddexp(
            0.0, (self.y - x_dot_beta) / self.xi_dot_sigma) + np.log(self.xi * self.sigma * self.beta_function_value)
        prior_vec = np.absolute(support_variable) ** self.q
        return np.sum(pinball_loss) + self.lambda_hyperparameter * np.sum(prior_vec)

    @staticmethod
    def __beta_function(self, a, b):
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    @staticmethod
    def __logistic_function(self, a):
        return np.exp(-np.logaddexp(0, -a))
