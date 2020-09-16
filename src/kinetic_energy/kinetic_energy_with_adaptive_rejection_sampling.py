"""Module for the abstract KineticEnergyWithAdaptiveRejectionSampling class."""
from .kinetic_energy import KineticEnergy
from adaptive_rejection_sampling import AdaptiveRejectionSampling
from abc import ABCMeta, abstractmethod
import numpy as np


class KineticEnergyWithAdaptiveRejectionSampling(KineticEnergy, metaclass=ABCMeta):
    """
    Abstract class for kinetic energies that use adaptive rejection sampling to draw the momentum observations.

    A general kinetic-energy class provides the function itself and its gradient.
    """

    def __init__(self, power=2, prefactor=1.0, **kwargs):
        """
        The constructor of the KineticEnergyWithAdaptiveRejectionSampling class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        power : int
            Either the power to which each momentum variable is raised (the generalised-power case) or twice the power
            to which each momentum-dependent part of the relativistic kinetic energy are raised (the super-relativistic
            case).
        prefactor : float, optional
            A general multiplicative prefactor of the potential (and therefore of the kinetic energy).
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ValueError
            If the power equals 0.
        base.exceptions.ValueError
            If the prefactor equals 0.0.
        """
        super().__init__(power=power, prefactor=prefactor, **kwargs)

    @abstractmethod
    def gradient(self, momentum):
        """
        Return the gradient of the kinetic energy.

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

    @abstractmethod
    def kinetic_energy(self, momentum):
        """
        Return the kinetic-energy function.

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
        return np.array(
            AdaptiveRejectionSampling(self.kinetic_energy(momentum), self.gradient(momentum)).draw(len(momentum)))
