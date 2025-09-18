"""Module for the Mediator class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from potential.potential import Potential
from sampler.sampler import Sampler
from typing import Sequence
import numpy as np
import os
from model_settings import number_of_particles, system_volume


class Mediator(metaclass=ABCMeta):
    """Abstract Mediator class."""

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], temperature: float = 1.0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000, **kwargs):
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
        temperature : float, optional
            The model temperature, n.b., the temperature is the reciprocal of the inverse temperature, beta (up to a
            proportionality constant).
        number_of_equilibration_iterations : int, optional
            Number of equilibration iterations of the Markov process.
        number_of_observations : int, optional
            Number of sample observations, i.e. the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
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
        """
        super().__init__(**kwargs)
        if not isinstance(potential, Potential):
            raise ConfigurationError(f"Give a potential class as the value for potential in {self.__class__.__name__}.")
        for sampler in samplers:
            if not isinstance(sampler, Sampler):
                raise ConfigurationError(f"Give a list of sampler classes as the value for samplers in "
                                         f"{self.__class__.__name__}.")
        if temperature < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as temperature in {self.__class__.__name__}.")
        if number_of_equilibration_iterations < 0:
            raise ConfigurationError(f"Give a value not less than 0 as number_of_equilibration_iterations in "
                                     f"{self.__class__.__name__}.")
        if number_of_observations <= 0:
            raise ConfigurationError(f"Give a value greater than 0 as number_of_observations in "
                                     f"{self.__class__.__name__}.")
        self._potential = potential
        self._samplers = [sampler for sampler in samplers if 'event' not in str(sampler)]
        self._event_samplers = [sampler for sampler in samplers if 'event' in str(sampler)]
        self._temperature = temperature
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._number_of_observations_between_screen_prints_for_clock = int(self._total_number_of_iterations / 10)
        """The following objects are set in self._set_arrays_and_counters()"""
        self._momenta = None
        self._positions = None
        self._samples = None
        self._event_samples = None
        self._initial_samples = None
        self._initial_event_samples = None
        self._checkpoint_index = None

    def generate_sample(self, restart_flag):
        """Generates a sample with the model temperature equal to self._temperature."""
        self._set_arrays_and_counters()
        if restart_flag:
            self._reload_configuration_from_file_and_reset()
            self._checkpoint_index = self.get_checkpoint_index()
        else:
            self._get_initial_sample()
        self._run_markov_process()
        if not restart_flag:
            self._samples = [np.concatenate((self._initial_samples[sampler_index], self._samples[sampler_index]))
                             for sampler_index, sampler in enumerate(self._samplers)]
            self._event_samples = [np.concatenate((self._initial_event_samples[event_sampler_index],
                                                   self._event_samples[event_sampler_index])) for
                                   event_sampler_index, sampler in enumerate(self._event_samplers)]
        [sampler.output_sample(self._samples[sampler_index], self._checkpoint_index)
         for sampler_index, sampler in enumerate(self._samplers)]
        [event_sampler.output_sample(self._event_samples[event_sampler_index], self._checkpoint_index) 
         for event_sampler_index, event_sampler in enumerate(self._event_samplers)]
        self._write_checkpoint_index_and_configuration()
        self._print_markov_process_summary()

    @abstractmethod
    def _set_arrays_and_counters(self):
        """Sets the arrays (e.g. the sample array) and counters before the Markov process."""
        self._positions = self._potential.get_initial_positions()
        self._samples = [sampler.get_empty_sample_array(self._total_number_of_iterations) for sampler in self._samplers]
        self._event_samples = [event_sampler.get_empty_sample_array() for event_sampler in self._event_samplers]
        self._checkpoint_index = 0

    def _reload_configuration_from_file_and_reset(self):
        """Reloads position data from a previous sub-run in the case of checkpointing."""
        self._positions = np.load(os.path.join(os.getcwd(), self._samplers[0].output_directory,
                                               "configuration_at_checkpoint.npy"))

    def get_checkpoint_index(self):
        """Finds run index if checkpointing is being used."""
        return int(np.loadtxt(os.path.join(os.getcwd(), self._samplers[0].output_directory, "checkpoint_index.txt"),
                              dtype='int')) + 1

    def _get_initial_sample(self):
        self._initial_samples = [sampler.get_empty_sample_array(1) for sampler in self._samplers]
        self._initial_event_samples = [event_sampler.get_empty_sample_array() for event_sampler in self._event_samplers]
        for sampler_index, sampler in enumerate(self._samplers):
            if "PressureSampler" in str(sampler):
                self._initial_samples[sampler_index][0, :] = number_of_particles / system_volume  # ideal-gas pressure
            else:
                self._initial_samples[sampler_index][0, :] = sampler.get_observation(self._momenta, self._positions,
                                                                                     self._potential)
        [self._initial_event_samples[event_sampler_index].append(event_sampler.get_observation(
            self._positions, self._potential)) for event_sampler_index, event_sampler in enumerate(self._event_samplers)]

    @abstractmethod
    def _run_markov_process(self):
        """Runs the Markov process with model temperature equal to self._temperature."""
        raise NotImplementedError

    def _print_sample_progress(self, markov_chain_index):
        """Prints (to screen) details of the current sampling process."""
        if (markov_chain_index % self._number_of_observations_between_screen_prints_for_clock == 0 and
                markov_chain_index != 0):
            print(f"{markov_chain_index} observations drawn out of a total of {self._total_number_of_iterations} "
                  f"(including {self._number_of_equilibration_iterations} equilibration observations).")

    def _write_checkpoint_index_and_configuration(self):
        """Saves current run index and final position state of the system."""
        np.savetxt(os.path.join(os.getcwd(),  self._samplers[0].output_directory, "checkpoint_index.txt"),
                   [self._checkpoint_index], fmt="%02d")
        np.save(os.path.join(os.getcwd(),  self._samplers[0].output_directory, "configuration_at_checkpoint.npy"),
                self._positions)

    @abstractmethod
    def _print_markov_process_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        raise NotImplementedError
