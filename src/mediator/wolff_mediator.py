"""Module for the WolffMediator class."""
from .ising_cluster_mediator import IsingClusterMediator
from model_settings import number_of_particles
from potential.ising_potential import IsingPotential
from helper_methods import get_neighbours
from sampler.sampler import Sampler
from typing import Sequence
import numpy as np


class WolffMediator(IsingClusterMediator):
    """The WolffMediator class provides functionality for the Wolff algorithm for the square-lattice Ising model."""

    def __init__(self, potential: IsingPotential, samplers: Sequence[Sampler], temperature: float = 1.0,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 output_directory: str = None, proposal_dynamics_adaptor_is_on: bool = False):
        r"""
        The constructor of the WolffMediator class.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of potential.ising_potential.IsingPotential (the only permitted potential for WolffMediator).
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
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
        proposal_dynamics_adaptor_is_on : bool, optional
            When True, the step size of the integrator is tuned during the equilibration process.

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
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        base.exceptions.ConfigurationError
            If potential is not an instance of potential.ising_potential.IsingPotential.
        base.exceptions.ConfigurationError
            If proposal_dynamics_adaptor_is_on is not False.
        """
        super().__init__(potential, samplers, temperature, number_of_equilibration_iterations, number_of_observations,
                         output_directory, proposal_dynamics_adaptor_is_on)
        """Re-instantiate self._potential as IsingPotential contains lattice_length."""
        self._potential = potential

    def _advance_markov_chain(self, markov_chain_step_index):
        """Advances the Markov chain by one step."""
        prob_of_adding_neighbour_to_cluster = self._get_prob_of_adding_neighbour_to_cluster(self._temperature)
        base_lattice_site = np.random.choice(number_of_particles)
        extremity_sites_of_cluster = [base_lattice_site]
        self._positions[base_lattice_site] *= -1
        while extremity_sites_of_cluster:
            current_lattice_site = extremity_sites_of_cluster.pop()
            for neighbouring_lattice_site in get_neighbours(current_lattice_site, self._potential.lattice_length):
                if (self._positions[neighbouring_lattice_site] == -self._positions[base_lattice_site] and
                        np.random.rand() < prob_of_adding_neighbour_to_cluster):
                    self._positions[neighbouring_lattice_site] *= -1
                    extremity_sites_of_cluster.append(neighbouring_lattice_site)
