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
        """
        super().__init__(potential, samplers, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations,
                         )
        if distance_between_measurements < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as distance_between_measurements in "
                                     f"{self.__class__.__name__}.")

        self._timestep = self._potential._timestep
        self._dimensionless_omega = self._potential._dimensionless_omega  # NOTE need to rename to be public
        self._dimensionless_mass = self._potential._dimensionless_m
        self._distance_between_measurements = distance_between_measurements
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, samplers=samplers, minimum_temperature=minimum_temperature,
                           maximum_temperature=maximum_temperature,
                           number_of_temperature_increments=number_of_temperature_increments,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations)

        """The following objects are for testing"""
        self._move_num = None
        self._initial_index = None
        self._indices = np.zeros((number_of_observations * 100, 7))
        self._n_indices_chosen = 0

    def _generate_sample_at_current_temperature(self, temperature_index, temperature):
        """Runs the Markov process at temperature in order to generate the sample at temperature."""
        self._total_number_of_events = 0
        active_particle_index = None
        for markov_chain_index in range(self._total_number_of_iterations):
            active_particle_index = np.random.randint(0, number_of_particles)
            movement_direction = np.random.choice((-1.0, 1.0))
            if markov_chain_index == 0:
                self._initial_index = active_particle_index
                print(f"started at index {active_particle_index}, direction {movement_direction}")
                # store active particle
                self._indices[self._n_indices_chosen, 0] = active_particle_index
                self._n_indices_chosen += 1
            active_particle_index, movement_direction = self._generate_single_observation(markov_chain_index,
                                                                                          temperature,
                                                                                          movement_direction,
                                                                                          active_particle_index)
            if (markov_chain_index + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                current_sample_size = markov_chain_index + 1
                print(f"{current_sample_size} observations drawn out of a total of "
                      f"{self._total_number_of_iterations} (including {self._number_of_equilibration_iterations} "
                      f"equilibration observations).")
        # get most chosen index
        mode_index = np.bincount(self._indices[:self._n_indices_chosen, 0].astype(int)).argmax()
        print(f"started at index {self._initial_index}, most visited index was {mode_index}")
        # TODO read output folder from config file
        np.save("output/event_chain_mediator/temperature_00_sample_of_indices.npy",
                self._indices[:self._n_indices_chosen])

    def _generate_single_observation(self, markov_chain_index, temperature, movement_direction,
                                     active_particle_index=None):
        """Advances the Markov chain to the next sampling instance and adds a single observation to the sample."""
        distance_travelled = 0.0

        while distance_travelled < self._distance_between_measurements:  # i.e. we will always start before we reach lambda
            # NOTE may have to think more about edge cases where this might not effectively catch the sampling moment.
            active_particle_index, movement_direction, distance_travelled = self._generate_next_event(
                markov_chain_index, distance_travelled,
                active_particle_index, movement_direction)
            # NOTE may need to consider if this always catches cases where we propose a move than exceeds lambda
            # does the simulation continue correctly after this case?

        return active_particle_index, movement_direction

    def _generate_next_event(self, markov_chain_index, distance_travelled, active_particle_index, movement_direction):
        """Runs the Markov chain until the next event"""
        # TODO implement variable speed_of_chain (how would that work?)
        a_plus_one_index = get_east_neighbour(active_particle_index, number_of_particles)
        a_minus_one_index = get_west_neighbour(active_particle_index, number_of_particles)
        dimensionless_positions = self._potential.get_dimensionless_position(self._positions)
        dimensionless_position_a = dimensionless_positions[active_particle_index]
        dimensionless_position_a_plus_1 = dimensionless_positions[a_plus_one_index]
        dimensionless_position_a_minus_1 = dimensionless_positions[a_minus_one_index]
        #############################################################
        # sample some more data for testing
        self._indices[self._n_indices_chosen, 3] = dimensionless_position_a
        self._indices[self._n_indices_chosen, 4] = dimensionless_position_a_minus_1
        self._indices[self._n_indices_chosen, 5] = dimensionless_position_a_plus_1
        self._indices[self._n_indices_chosen, 6] = self._potential.get_action_at_index(dimensionless_position_a,
                                                                                       dimensionless_position_a_plus_1)
        ##############################################################

        proposed_move_dimensionless, self._move_num, self._indices[
            self._n_indices_chosen, 2], distance_travelled_in_move = self._potential.get_distance_to_next_event(
            dimensionless_position_a,
            dimensionless_position_a_plus_1,
            dimensionless_position_a_minus_1,
            movement_direction, self._move_num)
        #NOTE this is the only point where self._timestep is used here - posible to move to potential?
        proposed_move = proposed_move_dimensionless * self._timestep 

        if distance_travelled + distance_travelled_in_move > self._distance_between_measurements or distance_travelled + distance_travelled_in_move == self._distance_between_measurements:
            allowed_move = self._distance_between_measurements - distance_travelled
            distance_travelled += np.abs(allowed_move)
            self.update_position(allowed_move, active_particle_index, distance_travelled, movement_direction)
            for sampler_index, sampler in enumerate(self._samplers):
                self._samples[sampler_index][markov_chain_index + 1, :] = sampler.get_observation(
                    None, self._positions, self._potential)

        else:
            distance_travelled += distance_travelled_in_move
            self.update_position(proposed_move, active_particle_index, distance_travelled, movement_direction)
            ###########################
            self._indices[self._n_indices_chosen, 1] = proposed_move
            ############################
            active_particle_index, movement_direction = self.choose_next_active_particle(active_particle_index,
                                                                                         a_plus_one_index,
                                                                                         a_minus_one_index,
                                                                                         movement_direction)

        return active_particle_index, movement_direction, distance_travelled

    def update_position(self, move, active_particle_index, distance_travelled, movement_direction):
        """ Updates position and distance travelled for the active particle"""
        self._positions[active_particle_index] += move
        # NOTE changed to account for distance_travelled in potential.get_distance
        # due to possibility of moving back and forth? not sure if physically possible but
        # should account for all edge cases
        # distance_travelled += np.abs(move)        
        # return distance_travelled

    def choose_next_active_particle(self, active_particle_index, active_particle_plus_1_index,
                                    active_particle_minus_1_index, movement_direction):
        """Chooses the index and direction for the next active particle in the markov chain"""
        initial_a = active_particle_index
        initial_v = movement_direction

        site_a_gradient = self._potential.get_gradient_at_index(self._positions, active_particle_index)
        site_a_minus_1_gradient = self._potential.get_gradient_at_index(self._positions,
                                                                        active_particle_minus_1_index)
        site_a_plus_1_gradient = self._potential.get_gradient_at_index(self._positions,
                                                                       active_particle_plus_1_index)
        total_action_gradients = np.abs(site_a_minus_1_gradient) + np.abs(site_a_gradient) + np.abs(
            site_a_plus_1_gradient)
        probabilities = np.zeros(3)

        probabilities[0] = site_a_minus_1_gradient / total_action_gradients
        probabilities[1] = probabilities[0] + site_a_gradient / total_action_gradients
        probabilities[2] = probabilities[1] + site_a_plus_1_gradient / total_action_gradients

        rand = np.random.uniform(0.0, 1.0)

        # NOTE this probably accounts for all cases/number of options, but this should be checked
        if rand < probabilities[0]:
            active_particle_index = active_particle_minus_1_index
        elif rand < probabilities[1]:
            movement_direction = movement_direction * -1
        else:
            active_particle_index = active_particle_plus_1_index

        if active_particle_index == initial_a and movement_direction == initial_v:
            raise Exception("Chose the same index and direction twice in a row")

        self._indices[self._n_indices_chosen, 0] = active_particle_index
        self._n_indices_chosen += 1
        return active_particle_index, movement_direction

    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Mean event rate per particle = "
              f"{self._total_number_of_events / self._number_of_observations / number_of_particles}")

    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        super()._reset_arrays_and_counters(temperature)
        self._dimensionless_positions = self._potential.get_dimensionless_position(self._positions)
        self._move_num = 0
        for sampler_index, sampler in enumerate(self._samplers):
            self._samples[sampler_index][0, :] = sampler.get_observation(None, self._positions, self._potential)
