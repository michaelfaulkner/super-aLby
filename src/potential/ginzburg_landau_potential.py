"""Module for the GinzburgLandauPotential class."""
from .potential import Potential
import numpy as np


class GinzburgLandauPotential(Potential):
    """
    This class implements the Ginzburg-Landau potential with one-dimensional order parameter on a three-dimensional
        periodic cubic lattice.
    """

    def __init__(self, alpha, beta, lambda_hyperparameter, lattice_length, prefactor=1.0):
        """
        The constructor of the NealFunnel class.

        Parameters
        ----------
        alpha : float
            ajouter
        beta : float
            ajouter
        lambda_hyperparameter : float
            ajouter
        lattice_length : int
            Number of lattice sites in each Cartesian direction (cubic lattice)
        prefactor : float
            The prefactor k of the potential.
        """
        super().__init__(prefactor=prefactor)
        self._alpha = alpha
        self._beta = beta
        self._lambda_hyperparameter = lambda_hyperparameter
        self._lattice_length = lattice_length
        self._lattice_volume = lattice_length ** 3
        self._one_minus_beta = (1 - beta)
        self._beta_dot_alpha = beta * alpha
        self._beta_dot_lambda = beta * lambda_hyperparameter

    def gradient(self, support_variable, charges=None):
        """
        Return the gradient of the potential.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the entire
            array of superconducting phase.
        charges : optional
            All the charges needed to calculate the gradient; not used in this potential class.

        Returns
        -------
        numpy array
            The gradient.
        """
        return (self._one_minus_beta * support_variable - self._beta_dot_alpha * (
                self.__pos_x_translation(support_variable) + self.__neg_x_translation(support_variable) +
                self.__pos_y_translation(support_variable) + self.__neg_y_translation(support_variable) +
                self.__pos_z_translation(support_variable) + self.__neg_z_translation(support_variable) -
                6 * support_variable) + self._beta_dot_lambda * support_variable ** 3)

    def potential(self, support_variable, charges=None):

        """
        Return the potential for the given separation.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; in this case, the entire
            array of superconducting phase.
        charges : optional
            All the charges needed to calculate the potential; not used in this potential class.

        Returns
        -------
        float
            The potential.
        """
        return np.sum(
            0.5 * self._one_minus_beta * support_variable ** 2 + 0.5 * self._beta_dot_alpha * (
                    (self.__pos_x_translation(support_variable) - support_variable) ** 2 +
                    (self.__pos_y_translation(support_variable) - support_variable) ** 2 +
                    (self.__pos_z_translation(support_variable) - support_variable) ** 2) +
            0.25 * self._beta_dot_lambda * support_variable ** 4)

    def __pos_x_translation(self, psi):
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))  # reshapes to a Len*Len*Len matrix
        b = np.pad(a, (0, 1), mode='wrap')  # copies the 0th entry at each matrix level to (Len+1)th entry
        c = np.delete(b, self._lattice_length, 0)  # deletes the (Len)th entry at the highest matrix level
        d = np.delete(c, self._lattice_length, 1)  # deletes the (Len)th entry at the second-highest matrix level
        e = np.delete(d, 0, 2)  # deletes the 0th entry at the lowest matrix level
        return np.reshape(e, self._lattice_volume)  # reshapes to an Len**3-dim vector

    def __pos_y_translation(self, psi):
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))
        b = np.pad(a, (0, 1), mode='wrap')
        c = np.delete(b, self._lattice_length, 0)
        d = np.delete(c, 0, 1)
        e = np.delete(d, self._lattice_length, 2)
        return np.reshape(e, self._lattice_volume)

    def __pos_z_translation(self, psi):
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))
        b = np.pad(a, (0, 1), mode='wrap')
        c = np.delete(b, 0, 0)
        d = np.delete(c, self._lattice_length, 1)
        e = np.delete(d, self._lattice_length, 2)
        return np.reshape(e, self._lattice_volume)

    def __neg_x_translation(self, psi):
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))
        b = np.pad(a, (1, 0), mode='wrap')
        c = np.delete(b, 0, 0)
        d = np.delete(c, 0, 1)
        e = np.delete(d, self._lattice_length, 2)
        return np.reshape(e, self._lattice_volume)

    def __neg_y_translation(self, psi):
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))
        b = np.pad(a, (1, 0), mode='wrap')
        c = np.delete(b, 0, 0)
        d = np.delete(c, self._lattice_length, 1)
        e = np.delete(d, 0, 2)
        return np.reshape(e, self._lattice_volume)

    def __neg_z_translation(self, psi):
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))
        b = np.pad(a, (1, 0), mode='wrap')
        c = np.delete(b, self._lattice_length, 0)
        d = np.delete(c, 0, 1)
        e = np.delete(d, 0, 2)
        return np.reshape(e, self._lattice_volume)

    # todo integrer __reshape_and_pad() dans les fonctions ci-dessous
    def __reshape_and_pad(self, psi, positive_translation):
        # reshape to a self.lattice_length * self.lattice_length * self.lattice_length matrix
        a = np.reshape(psi, (self._lattice_length, self._lattice_length, self._lattice_length))
        # copies the 0th entry at each matrix level to (self.lattice_length + 1)th entry
        if positive_translation:
            return np.pad(a, (0, 1), mode='wrap')
        return np.pad(a, (1, 0), mode='wrap')
