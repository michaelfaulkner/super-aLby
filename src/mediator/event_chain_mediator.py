"""Module for EventChainMediator class"""
import importlib
import numpy as np
from base.exceptions import ConfigurationError
from .mediator import Mediator
from factor_field.factor_field import FactorField
from factor_field.no_factor_field import NoFactorField
from potential.euclidean_subspace_potential import EuclideanSubspacePotential
from sampler.sampler import Sampler
from typing import Sequence
from model_settings import number_of_particles, size_of_particle_space, system_volume
parsing = importlib.import_module("base.parsing")


class EventChainMediator(Mediator):
    """The EventChainMediator class provides functionality for the event-chain Monte Carlo algorithm."""

    def __init__(self, potential: EuclideanSubspacePotential, samplers: Sequence[Sampler],
                 factor_field: FactorField = NoFactorField(), minimum_temperature: float = 1.0,
                 maximum_temperature: float = 1.0, number_of_temperature_increments: int = 0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 normalised_distance_between_measurements: float = 1.0,
                 normalised_distance_between_velocity_refreshments: float = 1.0, teleportation_portal: bool = False):
        r"""
        Constructor of the EventChainMediator class.

        Parameters
        ----------
        potential : potential.euclidean_subspace_potential.EuclideanSubspacePotential
            Instance of the chosen child class of potential.euclidean_subspace_potential.EuclideanSubspacePotential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
        factor_field : factor_field.factor_field.FactorField
            Instance of the chosen child class of factor_field.factor_field.FactorField.  Choose no_factor_field in the
            configuration file if you do not want to use a factor field.
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
            Number of sample observations, i.e. the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
        normalised_distance_between_measurements : float, optional
            Total distance through state space between samples (normalised as indicated by the operations below).
        normalised_distance_between_velocity_refreshments : float, optional
            Total distance through state space between velocity refreshments (normalised as indicated by the operations
            below).
        teleportation_portal : bool, optional
            When True, a teleportation portal is attempted at each event induced by the potential.

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
        base.exceptions.ConfigurationError
            If normalised_distance_between_velocity_refreshments is not greater than 0.0.
        """
        super().__init__(potential, samplers, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations)
        """Re-instantiate self._potential as EuclideanSubspacePotential contains additional abstract methods."""
        self._potential = potential
        if normalised_distance_between_measurements <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as normalised_distance_between_measurements in "
                                     f"{self.__class__.__name__}.")
        if normalised_distance_between_velocity_refreshments <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as "
                                     f"normalised_distance_between_velocity_refreshments in {self.__class__.__name__}.")
        self._distance_between_measurements = normalised_distance_between_measurements * number_of_particles
        self._distance_between_velocity_refreshments = (normalised_distance_between_velocity_refreshments *
                                                        number_of_particles)
        if "HardDiskPotential" in str(potential):
            self._distance_between_measurements *= np.min(size_of_particle_space)
            self._distance_between_velocity_refreshments *= np.min(size_of_particle_space)
        print(f"Distance between event-chain measurements is {self._distance_between_measurements}")
        print(f"Distance between event-chain velocity refreshments is {self._distance_between_measurements}")
        for sampler_index, sampler in enumerate(self._samplers):
            if "PressureSampler" in str(sampler):
                sampler.distance_between_measurements = self._distance_between_measurements
        """The following object is set in self._reset_arrays_and_counters()"""
        self._total_number_of_events = None
        self._factor_field = factor_field
        self._teleportation_portal = teleportation_portal

    def _generate_sample_at_current_temperature(self, temperature_index, temperature):
        """Runs the Markov process at temperature in order to generate the sample at temperature."""
        self.active_particle_index = np.random.randint(0, number_of_particles)
        movement_direction = self._potential.get_random_event_chain_velocity()
        distance_to_next_velocity_refreshment = self._distance_between_velocity_refreshments
        for markov_chain_index in range(self._total_number_of_iterations):
            distance_to_next_measurement = self._distance_between_measurements
            while True:
                candidate_events = [self._potential.get_next_event(
                                        self._positions, self.active_particle_index, temperature, movement_direction),
                                    self._factor_field.get_next_event(
                                        self._positions, self.active_particle_index, temperature, movement_direction)]
                distance_to_next_event, vetoing_index = min(candidate_events)

                if (distance_to_next_measurement < distance_to_next_event and
                        distance_to_next_measurement < distance_to_next_velocity_refreshment):
                    self._potential.update_position(self._positions, distance_to_next_measurement,
                                                    self.active_particle_index, movement_direction)
                    self._potential.cell_boundary_event = False
                    distance_to_next_velocity_refreshment -= distance_to_next_measurement
                    for sampler_index, sampler in enumerate(self._samplers):
                        self._samples[sampler_index][markov_chain_index, :] = sampler.get_observation(
                            None, self._positions, self._potential)
                    break

                elif distance_to_next_velocity_refreshment < distance_to_next_event:
                    self._potential.update_position(self._positions, distance_to_next_velocity_refreshment,
                                                    self.active_particle_index, movement_direction)
                    self._potential.cell_boundary_event = False
                    distance_to_next_measurement -= distance_to_next_velocity_refreshment
                    self.active_particle_index = np.random.randint(0, number_of_particles)
                    movement_direction = self._potential.get_random_event_chain_velocity()
                    distance_to_next_velocity_refreshment = self._distance_between_velocity_refreshments

                else:
                    self._potential.update_position(self._positions, distance_to_next_event,
                                                    self.active_particle_index, movement_direction)
                    if self._teleportation_portal:
                        portal_candidate = self._potential.get_portal_candidate(self._positions, self.active_particle_index,
                                                                                vetoing_index, movement_direction)
                        potential_difference = self._potential.get_potential_difference(self.active_particle_index,
                                                                                        portal_candidate,
                                                                                        self._positions)
                        if (potential_difference < 0.0 or np.random.uniform(0.0, 1.0)
                                < np.exp(- potential_difference / temperature)):
                            self._positions[self.active_particle_index] = portal_candidate
                        else:
                            self.active_particle_index, movement_direction = self._potential.choose_next_active_particle(
                                self._positions, self.active_particle_index, movement_direction, vetoing_index)
                    else:
                        self.active_particle_index, movement_direction = self._potential.choose_next_active_particle(
                            self._positions, self.active_particle_index, movement_direction, vetoing_index)
                    self._total_number_of_events += 1
                    distance_to_next_measurement -= distance_to_next_event
                    distance_to_next_velocity_refreshment -= distance_to_next_event

            super()._print_sample_progress(markov_chain_index)

    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Mean event rate per particle = "
              f"{self._total_number_of_events / self._number_of_observations / number_of_particles}")

    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g. the sample array) and counters before each temperature iteration."""
        super()._reset_arrays_and_counters(temperature)
        self._total_number_of_events = 0
