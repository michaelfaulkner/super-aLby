"""Module for EventChainMediator class"""
import importlib
import numpy as np
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from .mediator import Mediator
from potential.potential import Potential
from sampler.sampler import Sampler
from typing import Sequence
import logging
# NOTE this might not work?
from model_settings import number_of_particles
from helper_methods import get_east_neighbour, get_west_neighbour

parsing = importlib.import_module("base.parsing")


class EventChainMediator(Mediator):
    """The EventChainMediator class provides functionality for the event-chain Monte Carlo algorithm."""

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], minimum_temperature: float = 1.0,
                 maximum_temperature: float = 1.0, number_of_temperature_increments: int = 0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 distance_between_measurements: float = 1.0):
        r"""
        Constructor of the EventChainMediator class.

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
        distance_between_measurements : float, optional
            Total distance through state space between samples.

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
            If distance_between_measurements is not greater than 0.0.
        """
        super().__init__(potential, samplers, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations)
        if distance_between_measurements <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as distance_between_measurements in "
                                     f"{self.__class__.__name__}.")
        self._distance_between_measurements = distance_between_measurements
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, samplers=samplers, minimum_temperature=minimum_temperature,
                           maximum_temperature=maximum_temperature,
                           number_of_temperature_increments=number_of_temperature_increments,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           distance_between_measurements=distance_between_measurements)

        """The following objects are for testing"""
        self._move_num = None
        self._indices = np.zeros((number_of_observations * 100, 7))
        self._n_indices_chosen = 0

    def _generate_sample_at_current_temperature(self, temperature_index, temperature):
        """Runs the Markov process at temperature in order to generate the sample at temperature."""
        self._total_number_of_events = 0
        for markov_chain_index in range(self._total_number_of_iterations):
            if markov_chain_index == 0:
                active_particle_index = np.random.randint(0, number_of_particles)
                movement_direction = np.random.choice((-1.0, 1.0))

            active_particle_index, movement_direction = self._generate_single_observation(markov_chain_index, temperature, movement_direction,
                                              active_particle_index)
            super()._print_sample_progress(markov_chain_index)

        # get most chosen index - testing for debugging
        mode_index = np.bincount(self._indices[:self._n_indices_chosen, 0].astype(int)).argmax()
        print(f"most visited index was {mode_index}")
        # TODO read output folder from config file
        np.save("output/event_chain_mediator/temperature_00_sample_of_indices.npy",
                self._indices[:self._n_indices_chosen])

    def _generate_single_observation(self, markov_chain_index, temperature, movement_direction,
                                     active_particle_index=None):
        """Advances the Markov chain to the next sampling instance and adds a single observation to the sample."""
        distance_travelled = 0.0
        while distance_travelled < self._distance_between_measurements:
            east_particle_index = get_east_neighbour(active_particle_index, number_of_particles)
            west_particle_index = get_west_neighbour(active_particle_index, number_of_particles)
            active_particle_position = self._positions[active_particle_index]
            east_particle_position = self._positions[east_particle_index]
            west_particle_position = self._positions[west_particle_index]
            #############################################################
            # sample some more data for testing
            self._indices[self._n_indices_chosen, 3] = self._positions[active_particle_index]
            self._indices[self._n_indices_chosen, 4] = west_particle_position
            self._indices[self._n_indices_chosen, 5] = east_particle_position
            self._indices[self._n_indices_chosen, 6] = self._potential._get_pairwise_dimensionless_action(
                active_particle_position, east_particle_position)
            ##############################################################

            # TODO to generalise, we should pass self._positions and active_particle_index to
            #  get_distance_to_next_event() 
            # sort out east/west in potential class
            #############################################################
            distance_to_next_event, self._move_num, eta = self._potential.get_distance_to_next_event(
                active_particle_position, east_particle_position, west_particle_position, movement_direction,
                self._move_num)

            if distance_travelled + distance_to_next_event >= self._distance_between_measurements:
                distance_to_measurement = self._distance_between_measurements - distance_travelled
                distance_travelled += np.abs(distance_to_measurement)
                self._update_position(distance_to_measurement, active_particle_index, movement_direction)
                for sampler_index, sampler in enumerate(self._samplers):
                    self._samples[sampler_index][markov_chain_index + 1, :] = sampler.get_observation(
                        None, self._positions, self._potential)
            else:
                distance_travelled += distance_to_next_event
                self._update_position(distance_to_next_event, active_particle_index,  movement_direction)
                ###########################
                self._indices[self._n_indices_chosen, 2] = eta
                self._indices[self._n_indices_chosen, 1] = distance_to_next_event
                self._indices[self._n_indices_chosen, 0] = active_particle_index
                ############################
                active_particle_index, movement_direction, self._n_indices_chosen = self._potential.choose_next_active_particle(
                    active_particle_index, east_particle_index, west_particle_index, self._positions,
                    movement_direction, self._n_indices_chosen)
                
                self._total_number_of_events += 1
        return active_particle_index, movement_direction

    def _update_position(self, displacement_distance, active_particle_index, movement_direction):
        """ Updates position of the active particle."""
        self._positions[active_particle_index] += displacement_distance * movement_direction

    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Mean event rate per particle = "
              f"{self._total_number_of_events / self._number_of_observations / number_of_particles}")

    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        super()._reset_arrays_and_counters(temperature)
        self._move_num = 0
        for sampler_index, sampler in enumerate(self._samplers):
            self._samples[sampler_index][0, :] = sampler.get_observation(None, self._positions, self._potential)
