import logging
import math
import numpy as np
from base.logging import log_init_arguments
from .potential import Potential


# noinspection PyMethodOverriding
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
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__(prefactor=prefactor)
        self.alpha = alpha
        self.beta = beta
        self.lambda_hyperparameter = lambda_hyperparameter
        self.lattice_length = lattice_length
        self.lattice_volume = lattice_length ** 3
        self.one_minus_beta = (1 - beta)
        self.beta_dot_alpha = beta * alpha
        self.beta_dot_lambda = beta * lambda_hyperparameter

    def gradient(self, support_variable):
        """
        Return the gradient of the potential.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, the separation vector r_ij; in this case, the superconducting phase.

        Returns
        -------
        numpy array
            The gradient.
        """
        lamb = self.lamb
        beta = self.beta
        alpha = self.alpha
        Len = self.Len
        z1 = (1 - beta)
        z2 = beta * alpha
        z3 = beta * lamb
        psi_posx = self.__pos_x(psi)
        psi_posy = self.__pos_y(psi)
        psi_posz = self.__pos_z(psi)
        psi_negx = self.__neg_x(psi)
        psi_negy = self.negy(psi)
        psi_negz = self.__neg_z(psi)
        vec = np.empty(Len ** 3)
        vec = z1 * psi - z2 * (
                    psi_posx + psi_negx + psi_posy + psi_negy + psi_posz + psi_negz - 6 * psi) + z3 * psi ** 3
        return vec

    def potential(self, support_variable):
        """
        Return the potential for the given separation.

        Parameters
        ----------
        support_variable : numpy array
            For soft-matter models, the separation vector r_ij; in this case, the superconducting phase.

        Returns
        -------
        float
            The potential.
        """
        return np.sum(
            0.5 * self.one_minus_beta * support_variable ** 2 + 0.5 * self.beta_dot_alpha * (
                    (self.__pos_x(support_variable) - support_variable) ** 2 +
                    (self.__pos_y(support_variable) - support_variable) ** 2 +
                    (self.__pos_z(support_variable) - support_variable) ** 2) +
            0.25 * self.beta_dot_lambda * support_variable ** 4)

    def __pos_x(self, psi):
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))  # reshapes to a Len*Len*Len matrix
        b = np.pad(a, (0, 1), mode='wrap')  # copies the 0th entry at each matrix level to (Len+1)th entry
        c = np.delete(b, self.lattice_length, 0)  # deletes the (Len)th entry at the highest matrix level
        d = np.delete(c, self.lattice_length, 1)  # deletes the (Len)th entry at the second-highest matrix level
        e = np.delete(d, 0, 2)  # deletes the 0th entry at the lowest matrix level
        return np.reshape(e, self.lattice_volume)  # reshapes to an Len**3-dim vector

    def __pos_y(self, psi):
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        b = np.pad(a, (0, 1), mode='wrap')
        c = np.delete(b, self.lattice_length, 0)
        d = np.delete(c, 0, 1)
        e = np.delete(d, self.lattice_length, 2)
        return np.reshape(e, self.lattice_volume)

    def __pos_z(self, psi):
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        b = np.pad(a, (0, 1), mode='wrap')
        c = np.delete(b, 0, 0)
        d = np.delete(c, self.lattice_length, 1)
        e = np.delete(d, self.lattice_length, 2)
        return np.reshape(e, self.lattice_volume)

    def __neg_x(self, psi):
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        b = np.pad(a, (1, 0), mode='wrap')
        c = np.delete(b, 0, 0)
        d = np.delete(c, 0, 1)
        e = np.delete(d, self.lattice_length, 2)
        return np.reshape(e, self.lattice_volume)

    def negy(self, psi):
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        b = np.pad(a, (1, 0), mode='wrap')
        c = np.delete(b, 0, 0)
        d = np.delete(c, self.lattice_length, 1)
        e = np.delete(d, 0, 2)
        return np.reshape(e, self.lattice_volume)

    def __neg_z(self, psi):
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        b = np.pad(a, (1, 0), mode='wrap')
        c = np.delete(b, self.lattice_length, 0)
        d = np.delete(c, 0, 1)
        e = np.delete(d, 0, 2)
        return np.reshape(e, self.lattice_volume)

    def __reshape_and_pad(self, psi):
        # reshape to a self.lattice_length * self.lattice_length * self.lattice_length matrix
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        # copies the 0th entry at each matrix level to (self.lattice_length + 1)th entry
        return np.pad(a, (0, 1), mode='wrap')

    def __reshape_and_pad(self, psi, positive_translation):
        # reshape to a self.lattice_length * self.lattice_length * self.lattice_length matrix
        a = np.reshape(psi, (self.lattice_length, self.lattice_length, self.lattice_length))
        # copies the 0th entry at each matrix level to (self.lattice_length + 1)th entry
        if positive_translation:
            return np.pad(a, (0, 1), mode='wrap')
        return np.pad(a, (1, 0), mode='wrap')
