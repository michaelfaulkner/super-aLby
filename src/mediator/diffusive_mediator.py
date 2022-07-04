"""Module for the DiffusiveMediator class."""
from .mediator import Mediator
from abc import ABCMeta, abstractmethod
from potential.potential import Potential
from sampler.sampler import Sampler


class DiffusiveMediator(Mediator, metaclass=ABCMeta):
    """Abstract DiffusiveMediator class.  This is the parent class for all mediators that use diffusive dynamics."""

    def __init__(self, potential: Potential, sampler: Sampler, minimum_temperature: float = 1.0,
                 maximum_temperature: float = 1.0, number_of_temperature_values: int = 1,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 proposal_dynamics_adaptor_is_on: bool = True, **kwargs):
        r"""
        The constructor of the DiffusiveMediator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        sampler : sampler.sampler.Sampler
            Instance of the chosen child class of sampler.sampler.Sampler.
        minimum_temperature : float, optional
            The minimum value of the model temperature, n.b., the temperature is the reciprocal of the inverse
            temperature, beta (up to a proportionality constant).
        maximum_temperature : float, optional
            The maximum value of the model temperature, n.b., the temperature is the reciprocal of the inverse
            temperature, beta (up to a proportionality constant).
        number_of_temperature_values : int, optional
            The number of temperature values to iterate over.
        number_of_equilibration_iterations : int, optional
            Number of equilibration iterations of the Markov process.
        number_of_observations : int, optional
            Number of sample observations, i.e., the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
        proposal_dynamics_adaptor_is_on : bool, optional
            When True, the step size of the integrator is tuned during the equilibration process.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If potential is not an instance of some child class of potential.potential.Potential.
        base.exceptions.ConfigurationError
            If sampler is not an instance of some child class of sampler.sampler.Sampler.
        base.exceptions.ConfigurationError
            If number_of_equilibration_iterations is less than 0.
        base.exceptions.ConfigurationError
            If number_of_observations is not greater than 0.
        base.exceptions.ConfigurationError
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        """
        super().__init__(potential, sampler, minimum_temperature, maximum_temperature, number_of_temperature_values,
                         number_of_equilibration_iterations, number_of_observations, proposal_dynamics_adaptor_is_on,
                         **kwargs)

    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        super()._reset_arrays_and_counters(temperature)
        self._sample[0, :] = self._sampler.get_observation(None, self._positions, self._potential)

    @abstractmethod
    def _generate_single_observation(self, markov_chain_step_index, temperature):
        """Advances the Markov chain by one step and adds a single observation to the sample."""
        raise NotImplementedError

    @abstractmethod
    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step (DeterministicMediator) or the width of the proposal
            distribution (MetropolisMediator)."""
        raise NotImplementedError

    @abstractmethod
    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        raise NotImplementedError
