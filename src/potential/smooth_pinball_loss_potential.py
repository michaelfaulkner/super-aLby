"""Module for the SmoothPinballLossPotential class."""
from base.logging import log_init_arguments
from .potential import Potential
import logging
import math
import numpy as np


class SmoothPinballLossPotential(Potential):
    """
    This class implements the Neal's funnel potential
        U = x[0] ** 2 / 18.0 + 9 * x[0] / 2.0 + exp(-x[0]) * np.sum(x[1:len(x)] ** 2) / 2.0

    x is an n-dimensional vector of floats.
    """

    def __init__(self, tau: float, sigma: float, lambda_hyperparameter: float, x: str, y: str, xi: float = 0.1,
                 power: float = 2.0, prefactor: float = 1.0):
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
        x : str
            Input data file representing the design matrix (measured data).
        y : str
            Input data file representing the response (measured data).
        xi : float
            Pinball smoother (xi --> 0 recovers nonsmooth pinball loss).
        power : float
            Power used in prior.
        prefactor : float
            The prefactor k of the potential.
        """
        self._tau = tau
        self._sigma = sigma
        self._xi = xi
        self._xi_dot_sigma = xi * sigma
        self._lambda_hyperparameter = lambda_hyperparameter
        self._power = power
        self._power_minus_one = power - 1.0
        self._x = np.loadtxt(x, dtype=float, delimiter=',')
        self._y = np.loadtxt(y, dtype=float, delimiter=',')
        self._beta_function_value = self._beta_function(xi * (1.0 - tau), xi * tau)
        self._x_sum = np.sum(self._x, axis=0)
        super().__init__(prefactor=prefactor)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, tau=tau, sigma=sigma,
                           lambda_hyperparameter=lambda_hyperparameter, x=x, y=y, xi=xi, power=power,
                           prefactor=prefactor)

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
        x_dot_beta = np.inner(self._x, position)
        pinball_loss = ((self._tau - 1) * (self._y - x_dot_beta) / self._sigma + self._xi *
                        np.logaddexp(0.0, (self._y - x_dot_beta) / self._xi_dot_sigma) +
                        np.log(self._xi * self._sigma * self._beta_function_value))
        prior_vec = np.absolute(position) ** self._power
        return np.sum(pinball_loss) + self._lambda_hyperparameter * np.sum(prior_vec)

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
        prior_gradient = np.array([self._power * self._lambda_hyperparameter * np.sign(component) * np.absolute(
            component) ** self._power_minus_one for component in position])
        logistic_term = self._logistic_function((self._y - np.inner(self._x, position)) / self._xi_dot_sigma)
        mid_term = np.array([np.inner(logistic_term, self._x[:, i]) for i in range(len(position))])
        return (1 - self._tau) / self._sigma * self._x_sum + 1 / self._sigma * mid_term + prior_gradient

    @staticmethod
    def _beta_function(a, b):
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    @staticmethod
    def _logistic_function(a):
        return np.exp(-np.logaddexp(0, -a))
