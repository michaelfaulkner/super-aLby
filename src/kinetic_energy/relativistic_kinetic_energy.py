"""Module for the abstract KineticEnergyWithAdaptiveRejectionSampling class."""
from .kinetic_energy import KineticEnergy
from adaptive_rejection_sampling import AdaptiveRejectionSampling
from abc import ABCMeta, abstractmethod
import numpy as np


class RelativisticKineticEnergy(KineticEnergy, metaclass=ABCMeta):
    """
    Abstract class for relativistic kinetic energies (both StandardRelativisticKineticEnergy and
        SuperRelativisticKineticEnergy), which we require because both inherited classes use the adaptive rejection
        sampling defined in RelativisticKineticEnergy.momentum_observation() to draw the momentum observations.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
        observation of the momentum.
    """

    def __init__(self, gamma=1.0, **kwargs):
        """
        The constructor of the RelativisticKineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the (super- and standard) relativistic
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
        self._adaptive_rejection_sampling_instance = AdaptiveRejectionSampling(self._negative_current_value, self._negative_gradient)
        super().__init__(**kwargs)

    @abstractmethod
    def current_value(self, momentum):
        """
        Returns the kinetic-energy function.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The kinetic-energy function.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError

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
        return np.array(self._adaptive_rejection_sampling_instance.draw(len(momentum)))

    def _negative_current_value(self, momentum):
        """
        Returns the product of minus 1 and the kinetic-energy function.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The product of minus 1 and the kinetic-energy function.
        """
        return - self.current_value(momentum)

    def _negative_gradient(self, momentum):
        """
        Returns the product of minus 1 and the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The product of minus 1 and the derivative.
        """
        return - self.gradient(momentum)
