"""Module for the GeneralisedPowerKineticEnergy class."""
from .kinetic_energy import KineticEnergy
import numpy as np
import rpy2.robjects.packages as r_packages
import rpy2.robjects.numpy2ri as n2ri
n2ri.activate()
generalised_power_distribution = r_packages.importr('normalp')


class GeneralisedPowerKineticEnergy(KineticEnergy):
    """
    This class implements the generalised-power kinetic energy K = sum(p[i] ** power / power)
    """

    def __init__(self, power: int = 2):
        """
        The constructor of the GeneralisedPowerKineticEnergy class.

        Parameters
        ----------
        power : int
            The power to which each momentum component is raised. For potentials with leading order term |x|^a, the
            optimal choice that ensures robust dynamics is given by power = 1 + 1 / (a - 1) for a >= 2 and
            power = 1 + 1 / (a + 1) for a <= -1.

        Raises
        ------
        base.exceptions.ValueError
            If the power equals 0.
        """
        if power == 0:
            raise ValueError("Give a value not equal to 0 as the power associated with the kinetic energy {0}.".format(
                self.__class__.__name__))
        super().__init__()
        self._one_over_power = 1 / power
        self._power = power
        self._power_minus_two = power - 2

    def current_value(self, momentum):
        """
        Returns the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        float
            The kinetic energy.
        """
        return self._one_over_power * np.sum(np.absolute(momentum) ** self._power)

    def gradient(self, momentum):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.

        Returns
        -------
        numpy array
            The gradient of the kinetic energy.
        """
        return momentum * np.absolute(momentum) ** self._power_minus_two

    def momentum_observation(self, momentum):
        """
        Return an observation of the momentum from the kinetic-energy distribution.

        Parameters
        ----------
        momentum : numpy_array
            The current momentum associated with each support_variable.

        Returns
        -------
        numpy_array
            A new momentum associated with each support_variable.
        """
        return np.array(generalised_power_distribution.rnormp(len(momentum), p=self._power))
