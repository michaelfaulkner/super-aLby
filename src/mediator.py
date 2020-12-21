"""Module for the Mediator class."""
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta, dimensionality_of_particle_space, number_of_particles, \
    range_of_initial_particle_positions
import logging
import numpy as np


class Mediator:
    """
    Mediator class.

    The class provides the Markov-chain function (as get_sample()).
    """

    def __init__(self, integrator_instance, kinetic_energy_instance, potential_instance, sampler_instance,
                 number_of_equilibration_iterations=10000, number_of_observations=100000, initial_step_size=0.1,
                 max_number_of_integration_steps=10, randomise_number_of_integration_steps=False,
                 step_size_adaptor_is_on=True, use_metropolis_accept_reject=True):
        """
        The constructor of the Mediator class.

        Parameters
        ----------
        integrator_instance : Python class instance

        kinetic_energy_instance : Python class instance

        potential_instance : Python class instance

        sampler_instance : Python class instance

        number_of_equilibration_iterations : int, optional

        number_of_observations : int, optional

        initial_step_size : float, optional

        max_number_of_integration_steps : int, optional

        randomise_number_of_integration_steps : Boolean, optional

        step_size_adaptor_is_on : Boolean, optional

        use_metropolis_accept_reject : Boolean, optional

        Raises
        ------
        base.exceptions.ConfigurationError
            If number_of_equilibration_iterations is less than 0.
        base.exceptions.ConfigurationError
            If number_of_observations is not greater than 0.
        base.exceptions.ConfigurationError
            If initial_step_size is not greater than 0.0.
        base.exceptions.ConfigurationError
            If max_number_of_integration_steps is not greater than 0.
        base.exceptions.ConfigurationError
            If type(randomise_number_of_integration_steps) is not bool.
        base.exceptions.ConfigurationError
            If type(step_size_adaptor_is_on) is not bool.
        base.exceptions.ConfigurationError
            If type(use_metropolis_accept_reject) is not bool.

        """
        if number_of_equilibration_iterations < 0:
            raise ConfigurationError(f"Give a value not less than 0 as number_of_equilibration_iterations in "
                                     f"{self.__class__.__name__}.")
        if number_of_observations <= 0:
            raise ConfigurationError(f"Give a value greater than 0 as number_of_observations in "
                                     f"{self.__class__.__name__}.")
        if initial_step_size <= 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as initial_step_size in "
                                     f"{self.__class__.__name__}.")
        if max_number_of_integration_steps <= 0:
            raise ConfigurationError(f"Give a value greater than 0 as max_number_of_integration_steps in "
                                     f"{self.__class__.__name__}.")
        if type(randomise_number_of_integration_steps) is not bool:
            raise ConfigurationError(f"Give a value of type bool as randomise_number_of_integration_steps in "
                                     f"{self.__class__.__name__}.")
        if type(step_size_adaptor_is_on) is not bool:
            raise ConfigurationError(f"Give a value of type bool as randomise_number_of_integration_steps in "
                                     f"{self.__class__.__name__}.")
        if type(use_metropolis_accept_reject) is not bool:
            raise ConfigurationError(f"Give a value of type bool as randomise_number_of_integration_steps in "
                                     f"{self.__class__.__name__}.")

        self._integrator = integrator_instance
        self._kinetic_energy = kinetic_energy_instance
        self._potential = potential_instance
        self._sampler = sampler_instance
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._number_of_observations_between_screen_prints_for_clock = int(number_of_observations / 10)
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._initial_step_size = initial_step_size
        self._step_size = beta * initial_step_size
        self._max_number_of_integration_steps = max_number_of_integration_steps
        self._randomise_number_of_integration_steps = randomise_number_of_integration_steps
        self._step_size_adaptor_is_on = step_size_adaptor_is_on
        self._use_metropolis_accept_reject = use_metropolis_accept_reject
        self._momenta = self._kinetic_energy.get_momentum_observations()
        self._positions = self._initialise_position_array()
        self._current_potential = self._potential.get_value(self._positions)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           integrator_instance=integrator_instance, kinetic_energy_instance=kinetic_energy_instance,
                           potential_instance=potential_instance, sampler_instance=sampler_instance,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           initial_step_size=initial_step_size,
                           max_number_of_integration_steps=max_number_of_integration_steps,
                           randomise_number_of_integration_steps=randomise_number_of_integration_steps,
                           step_size_adaptor_is_on=step_size_adaptor_is_on,
                           use_metropolis_accept_reject=use_metropolis_accept_reject)

    def get_sample(self):
        """
        Runs the Markov chain and returns the generated sample.

        Returns
        -------
        numpy.ndarray
            The sample generated by the Markov chain.
        """
        number_of_accepted_trajectories = 0
        number_of_integration_steps = self._max_number_of_integration_steps
        number_of_numerical_divergences = 0
        sample = self._sampler.initialise_sample_array(self._total_number_of_iterations)
        sample[0, :] = self._sampler.get_observation(self._momenta, self._positions)

        for i in range(self._total_number_of_iterations):
            if i == self._number_of_equilibration_iterations:
                number_of_accepted_trajectories = 0
            if self._randomise_number_of_integration_steps:
                number_of_integration_steps = 1 + np.random.randint(self._max_number_of_integration_steps)

            candidate_momenta, candidate_positions = (
                self._integrator.get_candidate_configuration(self._momenta, self._positions, self._kinetic_energy,
                                                             self._potential, number_of_integration_steps,
                                                             self._step_size))
            candidate_potential = self._potential.get_value(candidate_positions)
            current_energy = self._kinetic_energy.get_value(self._momenta) + self._current_potential
            energy_change = self._kinetic_energy.get_value(candidate_momenta) + candidate_potential - current_energy

            if energy_change / current_energy > 100.0:
                number_of_numerical_divergences += 1
            if self._use_metropolis_accept_reject:
                if energy_change < 0.0 or np.random.uniform(0, 1) < np.exp(- beta * energy_change):
                    self._update_system_state(candidate_momenta, candidate_positions, candidate_potential)
                    number_of_accepted_trajectories += 1
            else:
                self._update_system_state(candidate_momenta, candidate_positions, candidate_potential)

            sample[i + 1, :] = self._sampler.get_observation(self._momenta, self._positions)
            self._momenta = self._kinetic_energy.get_momentum_observations()

            if (i + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                current_sample_size = i + 1
                print(f"{current_sample_size} observations drawn out of a total of {self._total_number_of_iterations} "
                      f"(including equilibration observations).")

            if self._step_size_adaptor_is_on and i < self._number_of_equilibration_iterations and (i + 1) % 100 == 0:
                acceptance_rate = number_of_accepted_trajectories / 100.0
                if acceptance_rate > 0.9:
                    self._step_size *= 1.1
                elif acceptance_rate < 0.8:
                    self._step_size *= 0.9
                number_of_accepted_trajectories = 0

        self._print_markov_chain_summary(number_of_accepted_trajectories / self._number_of_observations,
                                         number_of_numerical_divergences)

        return sample

    def _update_system_state(self, new_momenta, new_positions, new_potential):
        self._momenta = new_momenta
        self._positions = new_positions
        self._current_potential = new_potential

    @staticmethod
    def _initialise_position_array():
        if dimensionality_of_particle_space == 1:
            if type(range_of_initial_particle_positions) == float:
                return range_of_initial_particle_positions * np.ones(number_of_particles)
            else:
                return np.random.uniform(*range_of_initial_particle_positions, size=number_of_particles)
        else:
            if type(range_of_initial_particle_positions[0]) == float:
                return np.array([range_of_initial_particle_positions for _ in range(number_of_particles)])
            else:
                return np.array([[np.random.uniform(*axis_range) for axis_range in range_of_initial_particle_positions]
                                 for _ in range(number_of_particles)])

    def _print_markov_chain_summary(self, acceptance_rate, number_of_numerical_divergences):
        print(f"Metropolis-Hastings acceptance rate = {acceptance_rate}")
        print(f"Number of numerical instabilities (relative energy increases by two orders of magnitude) = "
              f"{number_of_numerical_divergences}")
        self._step_size /= beta
        if self._step_size_adaptor_is_on:
            print(f"Initial numerical step size = {self._initial_step_size}")
            print(f"Final numerical step size = {self._step_size}")
        else:
            print(f"Numerical step size = {self._step_size}")
        if self._randomise_number_of_integration_steps:
            print(f"Max number of integration steps = {self._max_number_of_integration_steps}; number of integration "
                  f"steps was randomised.")
        else:
            print(f"Number of integration steps = {self._max_number_of_integration_steps}")
