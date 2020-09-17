"""Module for the abstract Integrator class."""
from abc import ABCMeta, abstractmethod


class Integrator(metaclass=ABCMeta):
    """
    Abstract class for numerical integrators used to generate the Hamiltonian / (super-)relativistic flow between times
        t_0 and t_0 + step_size * number_of_integration_steps.

    A general integrator class provides the get_flow() function.
    """

    def __init__(self, kinetic_energy_instance, potential_instance, step_size=1, number_of_integration_steps=10,
                 **kwargs):
        """
        The constructor of the Integrator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kinetic_energy_instance : instance of Python class
            instance of KineticEnergy class.
        potential_instance : instance of Python class
            instance of Potential class.
        step_size : int, optional
            step size of numerical integration.
        number_of_integration_steps : int, optional
            number of  numerical integration steps between initial and candidate configurations.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ValueError
            If the step_size equals 0.
            If the number_of_integration_steps equals 0.
        """
        if step_size == 0:
            raise ValueError("Give a value not equal to 0.0 as the step size of the numerical integrator {0}.".format(
                self.__class__.__name__))
        if number_of_integration_steps == 0:
            raise ValueError("Give a value not equal to 0.0 as the number of numerical integration steps {0}.".format(
                self.__class__.__name__))
        self._kinetic_energy_instance = kinetic_energy_instance
        self._potential_instance = potential_instance
        self._step_size = step_size
        self._number_of_integration_steps = number_of_integration_steps
        super().__init__(**kwargs)

    @abstractmethod
    def flow(self, support_variable, momentum, charges=None):
        """
        Return the Hamiltonian / (super-)relativistic flow between times
            t_0 and t_0 + step_size * number_of_integration_steps.

        Parameters
        ----------
        support_variable : numpy_array
            For soft-matter models, one or many particle-particle separation vectors {r_ij}; for Bayesian models, the
            parameter value; for the Ginzburg-Landau potential on a lattice, the entire array of superconducting phase.
        momentum : numpy_array
            The momentum associated with each support_variable.
        charges : optional
            All the charges needed to calculate the potential and its gradient.

        Returns
        -------
        numpy_array
            The flow.
        """
        raise NotImplementedError
