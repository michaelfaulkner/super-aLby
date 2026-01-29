"""Module for the ReversibleMediator class."""
from .mediator import Mediator
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from potential.potential import Potential
from sampler.sampler import Sampler
from typing import Sequence


class ReversibleMediator(Mediator, metaclass=ABCMeta):
    """Abstract ReversibleMediator class."""

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], temperature: float = 1.0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 output_directory: str = None, proposal_dynamics_adaptor_is_on: bool = True, **kwargs):
        r"""
        The constructor of the ReversibleMediator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
        temperature : float, optional
            The model temperature, n.b., the temperature is the reciprocal of the inverse temperature, beta (up to a
            proportionality constant).
        number_of_equilibration_iterations : int, optional
            Number of equilibration iterations of the Markov process.
        number_of_observations : int, optional
            Number of sample observations, i.e. the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
        output_directory : str
            The name of the directory into which the sample file is written at the end of the run.
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
            If temperature is less than 0.0.
        base.exceptions.ConfigurationError
            If number_of_equilibration_iterations is less than 0.
        base.exceptions.ConfigurationError
            If number_of_observations is not greater than 0.
        base.exceptions.ConfigurationError
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        """
        super().__init__(potential, samplers, temperature, number_of_equilibration_iterations, number_of_observations,
                         output_directory, **kwargs)
        if type(proposal_dynamics_adaptor_is_on) is not bool:
            raise ConfigurationError(f"Give a value of type bool as proposal_dynamics_adaptor_is_on in "
                                     f"{self.__class__.__name__}.")
        self._proposal_dynamics_adaptor_is_on = proposal_dynamics_adaptor_is_on
        """The following object is set in self._set_arrays_and_counters()"""
        self._number_of_accepted_trajectories = None

    @abstractmethod
    def _set_arrays_and_counters(self):
        """Sets the arrays (e.g. the sample array) and counters before the Markov process."""
        super()._set_arrays_and_counters()
        self._number_of_accepted_trajectories = 0

    def _run_markov_process(self):
        """Runs the Markov chain with model temperature equal to self._temperature."""
        self.tunnelling_events = 0
        self.events = 0
        for markov_chain_index in range(self._total_number_of_iterations):
            if markov_chain_index == self._number_of_equilibration_iterations:
                self._number_of_accepted_trajectories = 0
            self._generate_single_observation(markov_chain_index)
            if (self._proposal_dynamics_adaptor_is_on and markov_chain_index < self._number_of_equilibration_iterations
                    and markov_chain_index % 100 == 0):
                self._proposal_dynamics_adaptor()
                self._number_of_accepted_trajectories = 0
            super()._print_sample_progress(markov_chain_index)

    @abstractmethod
    def _generate_single_observation(self, markov_chain_step_index):
        """Advances the Markov chain by one step and adds a single observation to the sample."""
        raise NotImplementedError

    @abstractmethod
    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step (DeterministicMediator) or the width of the proposal
            distribution (MetropolisMediator)."""
        raise NotImplementedError
