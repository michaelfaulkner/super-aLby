"""Module for EventChainMediator class"""
import os
import json
import importlib
import numpy as np
from base.exceptions import ConfigurationError
from .mediator import Mediator
from factor_field.factor_field import FactorField
from factor_field.no_factor_field import NoFactorField
from potential.euclidean_subspace_potential import EuclideanSubspacePotential
from sampler.sampler import Sampler
from typing import Sequence
from model_settings import number_of_particles, size_of_particle_space
parsing = importlib.import_module("base.parsing")


class EventChainMediator(Mediator):
    """The EventChainMediator class provides functionality for the event-chain Monte Carlo algorithm."""

    def __init__(self, potential: EuclideanSubspacePotential, samplers: Sequence[Sampler],
                 factor_field: FactorField = NoFactorField(), temperature: float = 1.0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 output_directory: str = None, normalised_distance_between_measurements: float = 1.0,
                 normalised_distance_between_velocity_refreshments: float = 1.0, teleportation_portal: bool = False):
        r"""
        Constructor of the EventChainMediator class.  Note that this class works only with potential classes that
            inherit from EuclideanSubspacePotential (essentially continuous spaces).

        Parameters
        ----------
        potential : potential.euclidean_subspace_potential.EuclideanSubspacePotential
            Instance of the chosen child class of potential.euclidean_subspace_potential.EuclideanSubspacePotential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
        factor_field : factor_field.factor_field.FactorField
            Instance of the chosen child class of factor_field.factor_field.FactorField.  Choose no_factor_field in the
            configuration file if you do not want to use a factor field.
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
            If temperature is less than 0.0.
        base.exceptions.ConfigurationError
            If number_of_equilibration_iterations is less than 0.
        base.exceptions.ConfigurationError
            If number_of_observations is not greater than 0.
        base.exceptions.ConfigurationError
            If normalised_distance_between_measurements is not greater than 0.0.
        base.exceptions.ConfigurationError
            If normalised_distance_between_velocity_refreshments is not greater than 0.0.
        """
        super().__init__(potential, samplers, temperature, number_of_equilibration_iterations, number_of_observations,
                         output_directory)
        """Re-instantiate self._potential as EuclideanSubspacePotential contains additional abstract methods."""
        self._potential = potential
        if normalised_distance_between_measurements <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for normalised_distance_between_measurements in "
                                     f"{self.__class__.__name__}.")
        if normalised_distance_between_velocity_refreshments <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for "
                                     f"normalised_distance_between_velocity_refreshments in {self.__class__.__name__}.")
        self._distance_between_measurements = normalised_distance_between_measurements * number_of_particles
        self._distance_between_velocity_refreshments = (normalised_distance_between_velocity_refreshments *
                                                        number_of_particles)
        if "HardDiskPotential" in str(potential) and len(size_of_particle_space) > 1:
            self._distance_between_measurements *= np.min(size_of_particle_space)
            self._distance_between_velocity_refreshments *= np.min(size_of_particle_space)
        print(f"Distance between event-chain measurements is {self._distance_between_measurements}")
        print(f"Distance between event-chain velocity refreshments is {self._distance_between_velocity_refreshments}")
        for sampler_index, sampler in enumerate(self._samplers):
            if "PressureSampler" in str(sampler):
                sampler.distance_between_measurements = self._distance_between_measurements
                if (abs(normalised_distance_between_measurements -
                        normalised_distance_between_velocity_refreshments) > 1.0e-12 and
                        normalised_distance_between_measurements > normalised_distance_between_velocity_refreshments):
                    raise ConfigurationError(f"Give a value not less than normalised_distance_between_measurements for "
                                             f"normalised_distance_between_velocity_refreshments in "
                                             f"{self.__class__.__name__}.  This is to avoid errors due to the subtle "
                                             f"calculation of pressure estimates made via the pointer-hop distance "
                                             f"(though this is not fully understood).")
        """The following object is set in self._set_arrays_and_counters()"""
        self._total_number_of_events = None
        self._factor_field = factor_field
        self._teleportation_portal = teleportation_portal

    def _run_markov_process(self):
        """Runs the Markov process with model temperature equal to self._temperature."""
        active_particle_index = np.random.randint(0, number_of_particles)
        movement_direction = self._potential.get_random_event_chain_velocity()
        # distance_to_next_velocity_refreshment = self._distance_between_velocity_refreshments
        distance_to_next_velocity_refreshment = np.random.uniform(0, size_of_particle_space - 2.0 * number_of_particles)
        for markov_chain_index in range(self._total_number_of_iterations):
            distance_to_next_measurement = self._distance_between_measurements
            taken_measurement = False
            while True:
                candidate_events = [self._potential.get_next_event(
                                        self._positions, active_particle_index, self._temperature, movement_direction),
                                    self._factor_field.get_next_event(
                                        self._positions, active_particle_index, self._temperature, movement_direction)]
                distance_to_next_event, vetoing_index = min(candidate_events)

                if (distance_to_next_measurement < distance_to_next_event and
                        distance_to_next_measurement < distance_to_next_velocity_refreshment):
                    self._potential.update_position(self._positions, distance_to_next_measurement,
                                                    active_particle_index, movement_direction)
                    self._potential.cell_boundary_event = False
                    distance_to_next_velocity_refreshment -= distance_to_next_measurement
                    distance_to_next_event -= distance_to_next_measurement
                    for sampler_index, sampler in enumerate(self._samplers):
                        self._samples[sampler_index][markov_chain_index, :] = sampler.get_observation(
                            None, self._positions, self._potential)
                    taken_measurement = True

                if distance_to_next_velocity_refreshment < distance_to_next_event:
                    self._potential.update_position(self._positions, distance_to_next_velocity_refreshment,
                                                    active_particle_index, movement_direction)
                    self._potential.cell_boundary_event = False
                    distance_to_next_measurement -= distance_to_next_velocity_refreshment
                    active_particle_index = np.random.randint(0, number_of_particles)
                    movement_direction = self._potential.get_random_event_chain_velocity()
                    # distance_to_next_velocity_refreshment = self._distance_between_velocity_refreshments
                    distance_to_next_velocity_refreshment = np.random.uniform(0, size_of_particle_space -
                                                                              2.0 * number_of_particles)
                    if taken_measurement:
                        break

                else:
                    self._potential.update_position(self._positions, distance_to_next_event,
                                                    active_particle_index, movement_direction)
                    [self._event_samples[event_sampler_index].append(event_sampler.get_observation(
                        self._positions, self._potential, active_particle_index, vetoing_index, distance_to_next_event))
                        for event_sampler_index, event_sampler in enumerate(self._event_samplers)]
                    self._potential.aggregate_pointer_hop_distance += self._potential.pointer_hop_distance

                    if self._teleportation_portal:
                        portal_candidate = self._potential.get_portal_candidate(self._positions, active_particle_index,
                                                                                vetoing_index, movement_direction)
                        potential_difference = self._potential.get_potential_difference(active_particle_index,
                                                                                        portal_candidate,
                                                                                        self._positions)
                        if (potential_difference < 0.0 or np.random.uniform(0.0, 1.0)
                                < np.exp(- potential_difference / self._temperature)):
                            self._positions[active_particle_index] = portal_candidate
                        else:
                            active_particle_index, movement_direction = self._potential.choose_next_active_particle(
                                self._positions, active_particle_index, movement_direction, vetoing_index)
                    else:
                        active_particle_index, movement_direction = self._potential.choose_next_active_particle(
                            self._positions, active_particle_index, movement_direction, vetoing_index)
                    self._total_number_of_events += 1
                    distance_to_next_velocity_refreshment -= distance_to_next_event
                    if taken_measurement:
                        break
                    distance_to_next_measurement -= distance_to_next_event

            super()._print_sample_progress(markov_chain_index)
        self._write_state_and_index_space_velocities()

    def _print_markov_process_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Mean event rate per particle = "
              f"{self._total_number_of_events / self._number_of_observations / number_of_particles}")

    def _set_arrays_and_counters(self):
        """Sets the arrays (e.g. the sample array) and counters before the Markov process."""
        super()._set_arrays_and_counters()
        self._total_number_of_events = 0

    def _write_state_and_index_space_velocities(self):
        """Saves average state space and index space velocities"""
        state_space_velocity = self._potential.state_space_displacement / self._potential.total_event_distance
        index_space_velocity = self._potential.index_space_displacement / self._potential.number_of_index_space_moves
        with open(os.path.join(self._output_directory, "state_and_index_space_velocities.json"), "w") as f:
            json.dump({"state_space_velocity": state_space_velocity, "index_space_velocity": index_space_velocity}, f)
