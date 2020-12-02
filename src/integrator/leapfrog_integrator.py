"""Module for the LeapfrogIntegrator class."""
from base.logging import log_init_arguments
from .integrator import Integrator
import logging


class LeapfrogIntegrator(Integrator):
    """
    This class implements the leapfrog numerical integrator.
    """

    def __init__(self):
        """
        The constructor of the LeapfrogIntegrator class.
        """
        super().__init__()
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)

    def get_candidate_configuration(self, momenta, positions, kinetic_energy_instance, potential_instance,
                                    number_of_integration_steps, step_size):
        """
        Return the Hamiltonian / (super-)relativistic flow between times
            t_0 and t_0 + step_size * number_of_integration_steps.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.
        positions : numpy.ndarray
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        kinetic_energy_instance : instantiated Python class
            Instance of an inherited KineticEnergy class; contains all methods associated with the kinetic energy.
        potential_instance : instantiated Python class
            Instance of an inherited Potential class; contains all methods associated with the potential.
        number_of_integration_steps : int, optional
            number of  numerical integration steps between initial and candidate configurations.
        step_size : int, optional
            step size of numerical integration.

        Returns
        -------
        numpy.ndarray
            The flow.
        """
        # todo change to += and -= everywhere below
        half_step_size = 0.5 * step_size
        momenta = momenta - half_step_size * potential_instance.get_gradient(positions)
        for _ in range(number_of_integration_steps - 1):
            positions = positions + step_size * kinetic_energy_instance.get_gradient(momenta)
            momenta = momenta - step_size * potential_instance.get_gradient(positions)
        positions = positions + step_size * kinetic_energy_instance.get_gradient(momenta)
        momenta = momenta - half_step_size * potential_instance.get_gradient(positions)
        return momenta, positions
