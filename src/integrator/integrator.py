"""Module for the abstract Integrator class."""
from abc import ABCMeta, abstractmethod


class Integrator(metaclass=ABCMeta):
    """
    Abstract class for numerical integrators used to generate the Hamiltonian / (super-)relativistic flow between times
        t_0 and t_0 + step_size * number_of_integration_steps.

    A general integrator class provides the get_flow() function.
    """

    def __init__(self, **kwargs):
        """
        The constructor of the Integrator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def get_candidate_configuration(self, momentum, position, kinetic_energy_instance, potential_instance,
                                    number_of_integration_steps, step_size):
        """
        Return the Hamiltonian / (super-)relativistic flow between times
            t_0 and t_0 + step_size * number_of_integration_steps.

        Parameters
        ----------
        momentum : numpy.ndarray
            The momenta associated with each positions.
        position : numpy.ndarray
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
        raise NotImplementedError
