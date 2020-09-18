"""Module for the LeapfrogIntegrator class."""
from .integrator import Integrator
import numpy as np


class LeapfrogIntegrator(Integrator):
    """
    This class implements the leapfrog numerical integrator.
    """

    def __init__(self, kinetic_energy_instance, potential_instance):
        """
        The constructor of the LeapfrogIntegrator class.

        Parameters
        ----------
        kinetic_energy_instance : instance of Python class
            instance of KineticEnergy class.
        potential_instance : instance of Python class
            instance of Potential class.
        """
        super().__init__(kinetic_energy_instance=kinetic_energy_instance, potential_instance=potential_instance)

    def get_flow(self, momentum, support_variable, number_of_integration_steps, step_size, charges=None):
        """
        Return the Hamiltonian / (super-)relativistic flow between times
            t_0 and t_0 + step_size * number_of_integration_steps.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each support_variable.
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        number_of_integration_steps : int, optional
            number of  numerical integration steps between initial and candidate configurations.
        step_size : int, optional
            step size of numerical integration.
        charges : optional
            All the charges needed to calculate the potential and its gradient.

        Returns
        -------
        numpy_array
            The flow.
        """
        support_variable_flow = np.empty((len(support_variable), number_of_integration_steps))
        momentum_flow = np.empty((len(momentum), number_of_integration_steps))
        support_variable_flow[:, 0] = support_variable
        momentum_flow[:, 0] = momentum
        for i in range(number_of_integration_steps):
            intermediate_momentum = (momentum_flow[:, i] - 0.5 * step_size *
                                     self._potential_instance.gradient(support_variable_flow[:, i], charges=charges))
            support_variable_flow[:, i + 1] = (support_variable_flow[:, i] + step_size *
                                               self._kinetic_energy_instance.gradient(intermediate_momentum))
            momentum_flow[:, i + 1] = (intermediate_momentum - 0.5 * step_size *
                                       self._potential_instance.gradient(support_variable_flow[:, i + 1],
                                                                         charges=charges))
        return np.vstack((momentum_flow, support_variable_flow))
