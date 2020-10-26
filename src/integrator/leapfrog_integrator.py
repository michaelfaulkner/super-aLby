"""Module for the LeapfrogIntegrator class."""
from base.logging import log_init_arguments
from .integrator import Integrator
import logging


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
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           kinetic_energy_instance=kinetic_energy_instance, potential_instance=potential_instance)

    def get_candidate_configuration(self, momentum, position, number_of_integration_steps, step_size):
        """
        Return the Hamiltonian / (super-)relativistic flow between times
            t_0 and t_0 + step_size * number_of_integration_steps.

        Parameters
        ----------
        momentum : numpy_array
            The momentum associated with each position.
        position : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        number_of_integration_steps : int, optional
            number of  numerical integration steps between initial and candidate configurations.
        step_size : int, optional
            step size of numerical integration.

        Returns
        -------
        numpy_array
            The flow.
        """
        half_step_size = 0.5 * step_size
        momentum = momentum - half_step_size * self._potential_instance.get_gradient(position)
        for _ in range(number_of_integration_steps - 1):
            position = position + step_size * self._kinetic_energy_instance.get_gradient(momentum)
            momentum = momentum - step_size * self._potential_instance.get_gradient(position)
        position = position + step_size * self._kinetic_energy_instance.get_gradient(momentum)
        momentum = momentum - half_step_size * self._potential_instance.get_gradient(position)
        return momentum, position
