"""Module for the abstract KineticEnergyWithAdaptiveRejectionSampling class."""
from .kinetic_energy import KineticEnergy
from adaptive_rejection_sampling import AdaptiveRejectionSampling
from base.logging import log_init_arguments
from model_settings import beta, dimensionality_of_particle_space, number_of_particles
from abc import ABCMeta, abstractmethod
import logging
import numpy as np


class RelativisticKineticEnergy(KineticEnergy, metaclass=ABCMeta):
    """
    Abstract class for relativistic kinetic energies (both StandardRelativisticKineticEnergy and
        SuperRelativisticKineticEnergy), which we require because both inherited classes use the adaptive rejection
        sampling defined in RelativisticKineticEnergy.momentum_observation() to draw the momenta observations.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
        observation of the momenta.
    """

    def __init__(self, gamma: float = 1.0, **kwargs):
        """
        The constructor of the RelativisticKineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momenta values near which the (super- and standard) relativistic
            kinetic energies transform from Gaussian to generalised-power behaviour.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ValueError
            If the gamma equals 0.0.
        """
        if gamma == 0.0:
            raise ValueError(
                "Give a value not equal to 0.0 as the tuning parameter for the relativistic kinetic energy {0}.".format(
                    self.__class__.__name__))
        self._one_over_gamma = 1.0 / gamma
        self._adaptive_rejection_sampling_instance = AdaptiveRejectionSampling(self._negative_beta_dot_current_value,
                                                                               self._negative_beta_dot_gradient)
        super().__init__(**kwargs)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, gamma=gamma)

    @abstractmethod
    def get_value(self, momenta):
        """
        Returns the kinetic-energy function.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The kinetic-energy function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError

    def get_momentum_observation(self):
        """
        Returns an observation of the momenta from the kinetic-energy distribution using adaptive rejection sampling.

        Returns
        -------
        numpy.ndarray
            A new momenta associated with each positions.
        """
        if dimensionality_of_particle_space == 1:
            return np.array(self._adaptive_rejection_sampling_instance.draw(number_of_particles))
        else:
            return np.array([self._adaptive_rejection_sampling_instance.draw(dimensionality_of_particle_space)
                             for _ in range(number_of_particles)])

    def _negative_beta_dot_current_value(self, momentum):
        """
        Returns the product of minus 1 and the kinetic-energy function.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The product of minus 1 and the kinetic-energy function.
        """
        return - beta * self.get_value(momentum)

    def _negative_beta_dot_gradient(self, momentum):
        """
        Returns the product of minus 1 and the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The product of minus 1 and the derivative.
        """
        return - beta * self.get_gradient(momentum)
