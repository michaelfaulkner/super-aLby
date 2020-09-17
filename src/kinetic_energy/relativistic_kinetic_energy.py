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

    def __init__(self, gamma=1.0, power=2, **kwargs):
        """
        The constructor of the KineticEnergyWithAdaptiveRejectionSampling class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        gamma : float
            The tuning parameter that controls the momentum values near which the (super- and standard) relativistic
            kinetic energies transform from Gaussian to generalised-power behaviour.
        power : int
            Either the power to which each momentum component is raised (the generalised-power case) or twice the power
            to which each momentum-dependent part of the relativistic kinetic energy is raised (the super-relativistic
            case). For potentials with leading order term |x|^a, the optimal choice that ensures robust dynamics is
            given by power = 1 + 1 / (a - 1) for a >= 2 and power = 1 + 1 / (a + 1) for a <= -1.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ValueError
            If the power equals 0.
        base.exceptions.ValueError
            If the prefactor equals 0.0.
        """
        self._one_over_gamma = 1.0 / gamma
        super().__init__(power=power, **kwargs)

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
            AdaptiveRejectionSampling(- self.kinetic_energy(momentum), - self.gradient(momentum)).draw(len(momentum)))
