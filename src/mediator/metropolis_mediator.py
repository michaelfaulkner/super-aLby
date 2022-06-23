"""Module for the MetropolisMediator class."""
from .mediator import Mediator
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import beta, number_of_particles, size_of_particle_space
from noise_distribution.noise_distribution import NoiseDistribution
from potential.potential import Potential
from sampler.sampler import Sampler
import logging
import numpy as np
import random


class MetropolisMediator(Mediator):
    """The MetropolisMediator class provides functionality for the Metropolis algorithm, which is equivalent to the
        Metropolis-within-Gibbs algorithm blocked to the level of single-particle dynamics."""

    def __init__(self, potential: Potential, sampler: Sampler, noise_distribution: NoiseDistribution,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 proposal_dynamics_adaptor_is_on: bool = True):
        r"""
        The constructor of the MetropolisMediator class.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        sampler : sampler.sampler.Sampler
            Instance of the chosen child class of sampler.sampler.Sampler.
        noise_distribution : noise_distribution.noise_distribution.NoiseDistribution
            Instance of the chosen child class of noise_distribution.noise_distribution.NoiseDistribution.
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
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        base.exceptions.ConfigurationError
            If type(use_metropolis_accept_reject) is not bool
        base.exceptions.ConfigurationError
            If potential is an instance of LennardJonesPotentialWithoutCutoff and element is greater than 2.0 *
            LennardJonesPotentialWithoutCutoff.characteristic_length for element in size_of_particle_space.
        """
        super().__init__(potential, sampler, number_of_equilibration_iterations, number_of_observations,
                         proposal_dynamics_adaptor_is_on)
        self._target_acceptance_rate = 0.44  # TODO add functionality so the user can set self._target_acceptance_rate
        self._noise_distribution = noise_distribution
        self._sample[0, :] = self._sampler.get_observation(None, self._positions)
        if "LennardJonesPotentialWithoutCutoff" in str(self._potential):
            for element in size_of_particle_space:
                if 2.0 * self._potential.characteristic_length < element:
                    raise ConfigurationError(f"When using {self.__class__.__name__}, ensure that the value of each "
                                             f"component of size_of_particle_space is not greater than twice the value "
                                             f"of characteristic_length in LennardJonesPotentialWithoutCutoff.")
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, sampler=sampler, noise_distribution=noise_distribution,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           proposal_dynamics_adaptor_is_on=proposal_dynamics_adaptor_is_on)

    def _generate_single_observation(self, markov_chain_step_index):
        """Advances the Markov chain by one step and adds a single observation to the sample."""
        particles_to_update = [index for index in range(number_of_particles)]
        random.shuffle(particles_to_update)  # randomises order of elements in particles_to_update
        for active_particle_index in particles_to_update:
            candidate_position = (self._positions[active_particle_index] +
                                  self._noise_distribution.get_finite_change_in_position(1)[0])
            potential_difference = self._potential.get_potential_difference(active_particle_index, candidate_position,
                                                                            self._positions)
            if potential_difference < 0.0 or np.random.uniform(0.0, 1.0) < np.exp(- beta * potential_difference):
                self._positions[active_particle_index] = candidate_position
                self._number_of_accepted_trajectories += 1
        self._sample[markov_chain_step_index + 1, :] = self._sampler.get_observation(None, self._positions)

    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step or the width of the proposal distribution."""
        acceptance_rate = self._number_of_accepted_trajectories / 100.0 / number_of_particles
        if type(self._noise_distribution.width_of_noise_distribution) is not None:
            if acceptance_rate > 1.1 * self._target_acceptance_rate:
                self._noise_distribution.width_of_noise_distribution *= 1.1
            elif acceptance_rate < 0.9 * self._target_acceptance_rate:
                self._noise_distribution.width_of_noise_distribution *= 0.9

    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        print(f"Acceptance rate = "
              f"{self._number_of_accepted_trajectories / self._number_of_observations / number_of_particles}")
        if type(self._noise_distribution.width_of_noise_distribution) is not None:
            if self._proposal_dynamics_adaptor_is_on:
                print(f"Initial width of noise distribution = "
                      f"{self._noise_distribution.initial_width_of_noise_distribution}")
                print(f"Final width of noise distribution = {self._noise_distribution.width_of_noise_distribution}")
            else:
                print(f"Width of noise distribution = {self._noise_distribution.width_of_noise_distribution}")