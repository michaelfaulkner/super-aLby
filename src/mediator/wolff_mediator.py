"""Module for the WolffMediator class."""
from .diffusive_mediator import DiffusiveMediator
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from model_settings import number_of_particles
from potential.ising_potential import IsingPotential
from sampler.sampler import Sampler
import logging
import numpy as np


class WolffMediator(DiffusiveMediator):
    """The WolffMediator class provides functionality for the Wolff algorithm for the square-lattice Ising model."""

    def __init__(self, potential: IsingPotential, sampler: Sampler, minimum_temperature: float = 1.0,
                 maximum_temperature: float = 1.0, number_of_temperature_values: int = 1,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 proposal_dynamics_adaptor_is_on: bool = False):
        r"""
        The constructor of the MetropolisMediator class.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of potential.ising_potential.IsingPotential (the only permitted potential for WolffMediator).
        sampler : sampler.sampler.Sampler
            Instance of the chosen child class of sampler.sampler.Sampler.
        minimum_temperature : float, optional
            The minimum value of the model temperature, n.b., the temperature is the reciprocal of the inverse
            temperature, beta (up to a proportionality constant).
        maximum_temperature : float, optional
            The maximum value of the model temperature, n.b., the temperature is the reciprocal of the inverse
            temperature, beta (up to a proportionality constant).
        number_of_temperature_values : int, optional
            The number of temperature values to iterate over.
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
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        base.exceptions.ConfigurationError
            If potential is not an instance of potential.ising_potential.IsingPotential.
        base.exceptions.ConfigurationError
            If proposal_dynamics_adaptor_is_on is not False.
        """
        super().__init__(potential, sampler, minimum_temperature, maximum_temperature, number_of_temperature_values,
                         number_of_equilibration_iterations, number_of_observations, proposal_dynamics_adaptor_is_on)
        if isinstance(potential, IsingPotential):
            self._potential_constant = self._potential.potential_constant
        else:
            raise ConfigurationError(f"Give a value of ising_potential for potential in {self.__class__.__name__}.")
        if proposal_dynamics_adaptor_is_on:
            raise ConfigurationError(f"Give a value of False for proposal_dynamics_adaptor_is_on in "
                                     f"{self.__class__.__name__}.")
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, sampler=sampler, minimum_temperature=minimum_temperature,
                           maximum_temperature=maximum_temperature,
                           number_of_temperature_values=number_of_temperature_values,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           proposal_dynamics_adaptor_is_on=proposal_dynamics_adaptor_is_on)

    def _generate_single_observation(self, markov_chain_step_index, temperature):
        """Advances the Markov chain by one step and adds a single observation to the sample."""
        # todo check the following line after testing code with celui d'après
        # prob_of_adding_spin_to_cluster = 1.0 - np.exp(2.0 * self._potential_constant / temperature)
        prob_of_adding_spin_to_cluster = 1.0 - np.exp(- 2.0 / temperature)
        base_lattice_site = np.random.choice(number_of_particles)
        stack = [base_lattice_site]
        self._positions[base_lattice_site] *= -1
        while stack:
            current_lattice_site = stack.pop()
            for neighbouring_lattice_site in self._potential.get_neighbours(current_lattice_site):
                if (self._positions[neighbouring_lattice_site] == -self._positions[base_lattice_site] and
                        np.random.rand() < prob_of_adding_spin_to_cluster):
                    self._positions[neighbouring_lattice_site] *= -1
                    stack.append(neighbouring_lattice_site)
        self._sample[markov_chain_step_index + 1, :] = self._sampler.get_observation(None, self._positions,
                                                                                     self._potential)

    def _proposal_dynamics_adaptor(self):
        """Proposal dynamics cannot be adapted in the Wolff algorithm."""
        pass

    def _print_markov_chain_summary(self):
        """Markov-process summary not printed to screen as acceptance rates and proposal dynamics are not relevant."""
        pass
