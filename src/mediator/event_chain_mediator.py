"""Module for EventChainMediator class"""
import importlib
import numpy as np
from base.exceptions import ConfigurationError
from .mediator import Mediator
from potential.euclidean_subspace_potential import EuclideanSubspacePotential
from sampler.sampler import Sampler
from typing import Sequence
from model_settings import number_of_particles, size_of_particle_space, system_volume
parsing = importlib.import_module("base.parsing")


class EventChainMediator(Mediator):
    """The EventChainMediator class provides functionality for the event-chain Monte Carlo algorithm."""

    def __init__(self, potential: EuclideanSubspacePotential, samplers: Sequence[Sampler],
                 minimum_temperature: float = 1.0, maximum_temperature: float = 1.0,
                 number_of_temperature_increments: int = 0, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, normalised_distance_between_measurements: float = 1.0):
        r"""
        Constructor of the EventChainMediator class.

        Parameters
        ----------
        potential : potential.euclidean_subspace_potential.EuclideanSubspacePotential
            Instance of the chosen child class of potential.euclidean_subspace_potential.EuclideanSubspacePotential.
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
        normalised_distance_between_measurements : float, optional
            Total distance through state space between samples (normalised as indicated by operations below).

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
            If normalised_distance_between_measurements is not greater than 0.0.
        """
        super().__init__(potential, samplers, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations)
        """Re-instantiate self._potential as EuclideanSubspacePotential contains additional abstract methods."""
        self._potential = potential
        if normalised_distance_between_measurements <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as normalised_distance_between_measurements in "
                                     f"{self.__class__.__name__}.")
        if "HardDiskPotential" in str(potential) and "QuantumHardDiskPotential" not in str(potential):
            self._distance_between_measurements = (normalised_distance_between_measurements * number_of_particles *
                                                   np.min(size_of_particle_space))
        else:
            self._distance_between_measurements = normalised_distance_between_measurements * number_of_particles
        print(f"distance between measurements: {self._distance_between_measurements}")
        for sampler_index, sampler in enumerate(self._samplers):
            if "PressureSampler" in str(sampler):
                sampler.distance_between_measurements = self._distance_between_measurements
        self._total_number_of_events = 0

    def _generate_sample_at_current_temperature(self, temperature_index, temperature, restart_flag):
        """Runs the Markov process at temperature in order to generate the sample at temperature."""
        self._total_number_of_events = 0
        super()._generate_sample_at_current_temperature(temperature_index, temperature, restart_flag)
        for markov_chain_index in range(self.number_of_markov_iterations):
            active_particle_index = np.random.randint(0, number_of_particles)
            movement_direction = self._potential.get_random_event_chain_velocity()
            distance_to_next_measurement = self._distance_between_measurements
            while True:
                distance_to_next_event, vetoing_index = self._potential.get_distance_to_next_event_and_veto_index(
                    self._positions, active_particle_index, temperature, movement_direction)
                if distance_to_next_measurement < distance_to_next_event:
                    self._potential.update_position(self._positions, distance_to_next_measurement,
                                                    active_particle_index, movement_direction)
                    self._potential.cell_boundary_event = False
                    for sampler_index, sampler in enumerate(self._samplers):
                        self._samples[sampler_index][markov_chain_index, :] = sampler.get_observation(
                            None, self._positions, self._potential)
                    break
                else:
                    distance_to_next_measurement -= distance_to_next_event
                    self._potential.update_position(self._positions, distance_to_next_event, active_particle_index,
                                                    movement_direction)
                    active_particle_index, movement_direction = self._potential.choose_next_active_particle(
                        self._positions, active_particle_index, movement_direction, vetoing_index)
                    self._total_number_of_events += 1
              
            super()._print_sample_progress(markov_chain_index)


    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Mean event rate per particle = "
              f"{self._total_number_of_events / self._number_of_observations / number_of_particles}")

    def _reset_arrays_and_counters(self, temperature, restart_flag):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        super()._reset_arrays_and_counters(temperature, restart_flag)
        self._momenta = None
        if not restart_flag:
            self._get_initial_sample()
        
