"""Module for the Mediator class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from helper_methods import get_temperatures
from potential.potential import Potential
from run import get_ordinal
from sampler.sampler import Sampler
from typing import Sequence


class Mediator(metaclass=ABCMeta):
    """Abstract Mediator class."""

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], minimum_temperature: float = 1.0,
                 maximum_temperature: float = 1.0, number_of_temperature_increments: int = 0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 proposal_dynamics_adaptor_is_on: bool = True, **kwargs):
        r"""
        The constructor of the Mediator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
        minimum_temperature : float, optional
            The minimum value of the model temperature, n.b., the temperature is the reciprocal of the inverse
            temperature, beta (up to a proportionality constant).
        maximum_temperature : float, optional
            The maximum value of the model temperature, n.b., the temperature is the reciprocal of the inverse
            temperature, beta (up to a proportionality constant).
        number_of_temperature_increments : int, optional
            number_of_temperature_increments + 1 is the number of temperature values to iterate over.
        number_of_equilibration_iterations : int, optional
            Number of equilibration iterations of the Markov process.
        number_of_observations : int, optional
            Number of sample observations, i.e., the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
        proposal_dynamics_adaptor_is_on : bool, optional
            When True, the size of either the numerical integration step (DeterministicMediator) or the width of the
            proposal distribution (MetropolisMediator) is tuned during the equilibration process.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If potential is not an instance of some child class of potential.potential.Potential.
        base.exceptions.ConfigurationError
            If samplers is not a sequence of instances of some child classes of sampler.sampler.Sampler.
        base.exceptions.ConfigurationError
            If minimum_temperature is less than 0.0.
        base.exceptions.ConfigurationError
            If maximum_temperature is less than 0.0.
        base.exceptions.ConfigurationError
            If maximum_temperature is less than minimum_temperature.
        base.exceptions.ConfigurationError
            If number_of_temperature_increments is less than 0.
        base.exceptions.ConfigurationError
            If number_of_temperature_increments is 0 and minimum_temperature does not equal maximum_temperature.
        base.exceptions.ConfigurationError
            If number_of_equilibration_iterations is less than 0.
        base.exceptions.ConfigurationError
            If number_of_observations is not greater than 0.
        base.exceptions.ConfigurationError
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        """
        super().__init__(**kwargs)
        if not isinstance(potential, Potential):
            raise ConfigurationError(f"Give a potential class as the value for potential in {self.__class__.__name__}.")
        for sampler in samplers:
            if not isinstance(sampler, Sampler):
                raise ConfigurationError(f"Give a list of sampler classes as the value for samplers in "
                                         f"{self.__class__.__name__}.")
        if minimum_temperature < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as minimum_temperature in "
                                     f"{self.__class__.__name__}.")
        if maximum_temperature < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as maximum_temperature in "
                                     f"{self.__class__.__name__}.")
        if maximum_temperature < minimum_temperature:
            raise ConfigurationError(f"Give values of minimum_temperature and maximum_temperature in "
                                     f"{self.__class__.__name__} such that the value of maximum_temperature is not "
                                     f"less than the value of minimum_temperature.")
        if number_of_temperature_increments < 0:
            raise ConfigurationError(f"Give a value not less than 0 as number_of_temperature_increments in "
                                     f"{self.__class__.__name__}.")
        if number_of_temperature_increments == 0 and minimum_temperature != maximum_temperature:
            raise ConfigurationError(f"As the value of number_of_temperature_increments is equal to 0, give equal "
                                     f"values of minimum_temperature and maximum_temperature in "
                                     f"{self.__class__.__name__}.")
        if number_of_equilibration_iterations < 0:
            raise ConfigurationError(f"Give a value not less than 0 as number_of_equilibration_iterations in "
                                     f"{self.__class__.__name__}.")
        if number_of_observations <= 0:
            raise ConfigurationError(f"Give a value greater than 0 as number_of_observations in "
                                     f"{self.__class__.__name__}.")
        if type(proposal_dynamics_adaptor_is_on) is not bool:
            raise ConfigurationError(f"Give a value of type bool as proposal_dynamics_adaptor_is_on in "
                                     f"{self.__class__.__name__}.")
        self._potential = potential
        self._samplers = samplers
        self._temperatures = get_temperatures(minimum_temperature, maximum_temperature,
                                              number_of_temperature_increments)
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._number_of_observations_between_screen_prints_for_clock = int(number_of_observations / 10)
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._proposal_dynamics_adaptor_is_on = proposal_dynamics_adaptor_is_on
        """The following objects are set in self._reset_arrays_and_counters()"""
        self._positions = None
        self._samples = None
        self._number_of_accepted_trajectories = None

    def generate_sample(self):
        """Runs the Markov chain in order to generate the sample."""
        for temperature_index, temperature in enumerate(self._temperatures):
            self._print_temperature_message(temperature, temperature_index)
            self._reset_arrays_and_counters(temperature)
            for markov_chain_index in range(self._total_number_of_iterations):
                if markov_chain_index == self._number_of_equilibration_iterations:
                    self._number_of_accepted_trajectories = 0
                self._generate_single_observation(markov_chain_index, temperature)
                if (self._proposal_dynamics_adaptor_is_on and
                        markov_chain_index < self._number_of_equilibration_iterations and
                        (markov_chain_index + 1) % 100 == 0):
                    self._proposal_dynamics_adaptor()
                    self._number_of_accepted_trajectories = 0
                if (markov_chain_index + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                    current_sample_size = markov_chain_index + 1
                    print(f"{current_sample_size} observations drawn out of a total of "
                          f"{self._total_number_of_iterations} (including {self._number_of_equilibration_iterations} "
                          f"equilibration observations).")
            [sampler.output_sample(self._samples[sampler_index], temperature_index) for sampler_index, sampler in
             enumerate(self._samplers)]
            self._print_markov_chain_summary()

    def _print_temperature_message(self, temperature, temperature_index):
        """Prints details of the current sampling temperature before each temperature iteration."""
        if len(self._temperatures) == 1:
            print("---------------------------------------------")
            print(f"Temperature = {temperature:.4f} (only temperature value)")
            print("---------------------------------------------")
        else:
            print("--------------------------------------------------")
            print(f"Temperature = {temperature:.4f} ({get_ordinal(temperature_index + 1)} of {len(self._temperatures)} "
                  f"temperature values)")
            print("--------------------------------------------------")

    @abstractmethod
    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        self._positions = self._potential.initialised_position_array()
        self._samples = [sampler.initialise_sample_array(self._total_number_of_iterations) for sampler in
                         self._samplers]
        self._number_of_accepted_trajectories = 0

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
