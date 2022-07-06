"""Module for the MetropolisMediator class."""
from .diffusive_mediator import DiffusiveMediator
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import number_of_particles
from noise_distribution.noise_distribution import NoiseDistribution
from potential.potential import Potential
from sampler.sampler import Sampler
from typing import Sequence
import logging
import numpy as np
import random


class MetropolisMediator(DiffusiveMediator):
    """The MetropolisMediator class provides functionality for the Metropolis algorithm, which is equivalent to the
        Metropolis-within-Gibbs algorithm blocked to the level of single-particle dynamics."""

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], noise_distribution: NoiseDistribution,
                 minimum_temperature: float = 1.0, maximum_temperature: float = 1.0,
                 number_of_temperature_increments: int = 0, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, proposal_dynamics_adaptor_is_on: bool = True):
        r"""
        The constructor of the MetropolisMediator class.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
        noise_distribution : noise_distribution.noise_distribution.NoiseDistribution
            Instance of the chosen child class of noise_distribution.noise_distribution.NoiseDistribution.
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
            When True, the step size of the integrator is tuned during the equilibration process.

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
        base.exceptions.ConfigurationError
            If noise_distribution is not an instance of some child class of
            noise_distribution.noise_distribution.NoiseDistribution.
        """
        super().__init__(potential, samplers, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations,
                         proposal_dynamics_adaptor_is_on)
        if not isinstance(noise_distribution, NoiseDistribution):
            raise ConfigurationError(f"Give a noise_distribution class as the value for noise_distribution in "
                                     f"{self.__class__.__name__}.")
        self._target_acceptance_rate = 0.44  # TODO add functionality so the user can set self._target_acceptance_rate
        self._noise_distribution = noise_distribution
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, samplers=samplers, noise_distribution=noise_distribution,
                           minimum_temperature=minimum_temperature, maximum_temperature=maximum_temperature,
                           number_of_temperature_increments=number_of_temperature_increments,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           proposal_dynamics_adaptor_is_on=proposal_dynamics_adaptor_is_on)

    def _advance_markov_chain(self, markov_chain_step_index, temperature):
        """Advances the Markov chain by one step."""
        particles_to_update = [index for index in range(number_of_particles)]
        random.shuffle(particles_to_update)  # randomises order of elements in particles_to_update
        for active_particle_index in particles_to_update:
            candidate_position = self._noise_distribution.get_candidate_position(active_particle_index, self._positions)
            potential_difference = self._potential.get_potential_difference(active_particle_index, candidate_position,
                                                                            self._positions)
            if potential_difference < 0.0 or np.random.uniform(0.0, 1.0) < np.exp(- potential_difference / temperature):
                self._positions[active_particle_index] = candidate_position
                self._number_of_accepted_trajectories += 1

    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step or the width of the proposal distribution."""
        acceptance_rate = self._number_of_accepted_trajectories / 100.0 / number_of_particles
        if type(self._noise_distribution.width_of_noise_distribution) == float:
            if acceptance_rate > 1.1 * self._target_acceptance_rate:
                self._noise_distribution.width_of_noise_distribution *= 1.1
            elif acceptance_rate < 0.9 * self._target_acceptance_rate:
                self._noise_distribution.width_of_noise_distribution *= 0.9

    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Acceptance rate = "
              f"{self._number_of_accepted_trajectories / self._number_of_observations / number_of_particles}")
        if type(self._noise_distribution.width_of_noise_distribution) == float:
            if self._proposal_dynamics_adaptor_is_on:
                print(f"Initial width of noise distribution = "
                      f"{self._noise_distribution.initial_width_of_noise_distribution}")
                print(f"Final width of noise distribution = {self._noise_distribution.width_of_noise_distribution}")
            else:
                print(f"Width of noise distribution = {self._noise_distribution.width_of_noise_distribution}")
