"""Module for the SwendsenWangMediator class."""
from .ising_cluster_mediator import IsingClusterMediator
from base.logging import log_init_arguments
from model_settings import number_of_particles
from potential.ising_potential import IsingPotential
from sampler.sampler import Sampler
from typing import Sequence
import logging
import numpy as np
import random


class SwendsenWangMediator(IsingClusterMediator):
    """The SwendsenWangMediator class provides functionality for the Swendsen-Wang algorithm for the square-lattice
        Ising model."""

    def __init__(self, potential: IsingPotential, samplers: Sequence[Sampler], minimum_temperature: float = 1.0,
                 maximum_temperature: float = 1.0, number_of_temperature_increments: int = 0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 proposal_dynamics_adaptor_is_on: bool = False):
        r"""
        The constructor of the SwendsenWangMediator class.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of potential.ising_potential.IsingPotential (the only permitted potential for WolffMediator).
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
            If potential is not an instance of potential.ising_potential.IsingPotential.
        base.exceptions.ConfigurationError
            If proposal_dynamics_adaptor_is_on is not False.
        """
        super().__init__(potential, samplers, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations,
                         proposal_dynamics_adaptor_is_on)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, samplers=samplers, minimum_temperature=minimum_temperature,
                           maximum_temperature=maximum_temperature,
                           number_of_temperature_increments=number_of_temperature_increments,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           proposal_dynamics_adaptor_is_on=proposal_dynamics_adaptor_is_on)

    def _advance_markov_chain(self, markov_chain_step_index, temperature):
        """Advances the Markov chain by one step."""
        prob_of_adding_neighbour_to_cluster = self._get_prob_of_adding_neighbour_to_cluster(temperature)
        # create list of particles in random order
        remaining_particles = list(range(number_of_particles))
        random.shuffle(remaining_particles)
        while remaining_particles:
            base_lattice_site = remaining_particles.pop()
            cluster = [base_lattice_site]
            extremity_sites_of_cluster = [base_lattice_site]
            while extremity_sites_of_cluster:
                current_lattice_site = extremity_sites_of_cluster.pop()
                for neighbouring_lattice_site in self._potential.get_neighbours(current_lattice_site):
                    if (neighbouring_lattice_site in remaining_particles and
                            self._positions[neighbouring_lattice_site] == self._positions[base_lattice_site] and
                            np.random.rand() < prob_of_adding_neighbour_to_cluster):
                        cluster.append(neighbouring_lattice_site)
                        extremity_sites_of_cluster.append(neighbouring_lattice_site)
                        remaining_particles.remove(neighbouring_lattice_site)
            # now flip all spin values within cluster with probability one half
            if np.random.rand() < 0.5:
                for lattice_site in cluster:
                    self._positions[lattice_site] *= -1
