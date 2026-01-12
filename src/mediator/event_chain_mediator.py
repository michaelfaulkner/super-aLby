"""Module for EventChainMediator class"""
import os
import json
import importlib
import numpy as np
from base.exceptions import ConfigurationError
from .mediator import Mediator
from factor_field.factor_field import FactorField
from factor_field.no_factor_field import NoFactorField
from refreshment_distribution.refreshment_distribution import RefreshmentDistribution
from refreshment_distribution.constant_refreshment_distribution import ConstantRefreshmentDistribution
from potential.euclidean_subspace_potential import EuclideanSubspacePotential
from sampler.sampler import Sampler
from typing import Sequence
from model_settings import number_of_particles, size_of_particle_space, dimensionality_of_particle_space

parsing = importlib.import_module("base.parsing")


class EventChainMediator(Mediator):
    """The EventChainMediator class provides functionality for the event-chain Monte Carlo algorithm."""

    def __init__(self, potential: EuclideanSubspacePotential, samplers: Sequence[Sampler],
                 factor_field: FactorField = NoFactorField(),
                 refreshment_distribution: RefreshmentDistribution = ConstantRefreshmentDistribution(),
                 temperature: float = 1.0, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, output_directory: str = None,
                 normalised_distance_between_measurements: float = 1.0, teleportation_portal: bool = False):
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
        refreshment_distribution : refreshment_distribution.refreshment_distribution.RefreshmentDistribution
            Instance of the chosen child class of
            refreshment_distribution.refreshment_distribution.RefreshmentDistribution.
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
        teleportation_portal : bool, optional
            When True, a teleportation portal is attempted at each event induced by the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If potential is not an instance of some child class of potential.potential.Potential.
        base.exceptions.ConfigurationError
            If factor_field is not an instance of some child class of factor_field.factor_field.FactorField.
        base.exceptions.ConfigurationError
            If refreshment_distribution is not an instance of some child class of
                refreshment_distribution.refreshment_distribution.RefreshmentDistribution.
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
        """
        super().__init__(potential, samplers, temperature, number_of_equilibration_iterations, number_of_observations,
                         output_directory)
        if not isinstance(factor_field, FactorField):
            raise ConfigurationError(f"Give a factor-field class as the value for factor_field in "
                                     f"{self.__class__.__name__}.")
        if not isinstance(refreshment_distribution, RefreshmentDistribution):
            raise ConfigurationError(f"Give a refreshment-distribution class as the value for refreshment_distribution "
                                     f"in {self.__class__.__name__}.")
        """Re-instantiate self._potential as EuclideanSubspacePotential contains additional abstract methods."""
        self._potential = potential
        self._factor_field = factor_field
        self._refreshment_distribution = refreshment_distribution
        if normalised_distance_between_measurements <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 for normalised_distance_between_measurements in "
                                     f"{self.__class__.__name__}.")
        self._distance_between_measurements = normalised_distance_between_measurements * number_of_particles
        if "HardDiskPotential" in str(potential) and len(size_of_particle_space) > 1:
            self._distance_between_measurements *= np.min(size_of_particle_space)
        self._free_space = np.atleast_1d(size_of_particle_space)[0]
        if "HardDiskPotential" in str(potential):
            self._free_space -= 2.0 * number_of_particles
        print(f"Distance between event-chain measurements is {self._distance_between_measurements}")
        for sampler_index, sampler in enumerate(self._samplers):
            if "PressureSampler" in str(sampler):
                sampler.distance_between_measurements = self._distance_between_measurements
                if (abs(normalised_distance_between_measurements -
                        self._refreshment_distribution.normalised_refreshment_lengthscale) > 1.0e-12 and
                        normalised_distance_between_measurements >
                        self._refreshment_distribution.normalised_refreshment_lengthscale):
                    raise ConfigurationError(f"Give a value not less than normalised_distance_between_measurements for "
                                             f"normalised_distance_between_velocity_refreshments in "
                                             f"{self.__class__.__name__}.  This is to avoid errors due to the subtle "
                                             f"calculation of pressure estimates made via the pointer-hop distance "
                                             f"(though this is not fully understood).")
        """The following object is set in self._set_arrays_and_counters()"""
        (self._total_number_of_events, self._state_space_displacement, self._total_event_distance,
         self._index_space_displacement, self._number_of_index_space_moves) = None, None, None, None, None
        self._teleportation_portal = teleportation_portal
        self.active_particle_index = None # this is set in self._generate_sample_at_current_temperature()
        self._index_of_current_active_particle_sample = 0

    def _run_markov_process(self):
        """Runs the Markov process with model temperature equal to self._temperature."""
        ff_events = 0
        active_particle_index = np.random.randint(0, number_of_particles)
        distance_to_next_measurement = 0.0
        movement_direction = self._potential.get_random_event_chain_velocity()
        distance_to_next_velocity_refreshment = self._refreshment_distribution.get_refreshment_distance()
        for markov_chain_index in range(self._total_number_of_iterations):
            distance_to_next_measurement += self._distance_between_measurements
            taken_measurement = False
            while True:
                candidate_events = [self._potential.get_next_event(
                                        self._positions, active_particle_index, self._temperature, movement_direction),
                                    self._factor_field.get_next_event(
                                        self._positions, active_particle_index, self._temperature, movement_direction)]

                distance_to_next_event, vetoing_index, hop_displacement = min(candidate_events)
                if np.argmin([candidate_events[0][0], candidate_events[1][0]]) == 1:
                    FF_event = True
                    ff_events += 1
                else:
                    FF_event = False
                #print(f"distance to next event: {distance_to_next_event}, FF: {FF_event}")
                self._update_state_and_index_space_displacements(distance_to_next_event, active_particle_index,
                                                                 vetoing_index, hop_displacement)

                if (distance_to_next_measurement < distance_to_next_event and
                        distance_to_next_measurement < distance_to_next_velocity_refreshment):
                    self._potential.update_position(self._positions, distance_to_next_measurement,
                                                    active_particle_index, movement_direction)
                    distance_to_next_velocity_refreshment -= distance_to_next_measurement
                    distance_to_next_event -= distance_to_next_measurement
                    distance_to_next_measurement = 0.0
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
                    distance_to_next_velocity_refreshment = self._refreshment_distribution.get_refreshment_distance()

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
                    distance_to_next_measurement -= distance_to_next_event
                    if taken_measurement:
                        break

            super()._print_sample_progress(markov_chain_index)
        self._write_state_and_index_space_velocities()
        print(f"total events: {self._total_number_of_events}, factor field events: {ff_events}" 
              f"\n {ff_events/ self._total_number_of_events} of events were ff")

    def _print_markov_process_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Mean event rate per particle = "
              f"{self._total_number_of_events / self._number_of_observations / number_of_particles}")

    def _set_arrays_and_counters(self):
        """Sets the arrays (e.g. the sample array) and counters before the Markov process."""
        super()._set_arrays_and_counters()
        self._total_number_of_events, self._number_of_index_space_moves = 0, 0
        self._state_space_displacement, self._total_event_distance, self._index_space_displacement = 0.0, 0.0, 0.0

    def _write_state_and_index_space_velocities(self):
        """Saves average state space and index space velocities"""
        state_space_velocity = None if self._total_event_distance == 0.0 else (
                self._state_space_displacement / self._total_event_distance)
        index_space_velocity = None if self._number_of_index_space_moves == 0.0 else (
                self._index_space_displacement / self._number_of_index_space_moves)
        with open(os.path.join(self._output_directory, "state_and_index_space_velocities.json"), "w") as f:
            json.dump({"state_space_velocity": state_space_velocity, "index_space_velocity": index_space_velocity}, f)

    def _update_state_and_index_space_displacements(self, displacement_distance, active_particle_index, vetoing_index,
                                                    hop_displacement):
        """Updates the state- and index-space displacements following each particle-event sampling.  This is to measure
            their mean values over the entire simulation.  N.B. we apply this method before checking whether the next
            event is the particle event, a velocity-refreshment event or a sampling/measurement event.  For the latter,
            this is because we continue motion without re-sampling the next particle event; for the
            velocity-refreshment events, it is because we can choose between including none or all of the current
            piecewise trajectory (this may change at non-constant speed but we would have to check)."""
        # todo add functionality for greater than 1D particle space
        if dimensionality_of_particle_space == 1:
            if hop_displacement:
                self._state_space_displacement += hop_displacement[0]
                self._total_event_distance += displacement_distance[0]
            if vetoing_index == (active_particle_index + 1) % number_of_particles:
                self._index_space_displacement += 1
            elif vetoing_index == (active_particle_index - 1) % number_of_particles:
                self._index_space_displacement -= 1
            self._number_of_index_space_moves += 1
