"""Module for the ToroidalLeapfrogIntegrator class."""
from .deterministic_mediator import DeterministicMediator
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import size_of_particle_space
from kinetic_energy.kinetic_energy import KineticEnergy
from potential.continuous_potential import ContinuousPotential
from sampler.sampler import Sampler
from typing import Sequence
import logging
import numpy as np


class ToroidalLeapfrogMediator(DeterministicMediator):
    """
    This class implements the mediator using the leapfrog numerical integrator with corrections of the particle
    positions to account for the toroidal geometry (using base.vectors.get_shortest_vectors_on_torus()).
    """

    def __init__(self, potential: ContinuousPotential, samplers: Sequence[Sampler], kinetic_energy: KineticEnergy,
                 minimum_temperature: float = 1.0, maximum_temperature: float = 1.0,
                 number_of_temperature_increments: int = 0, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, proposal_dynamics_adaptor_is_on: bool = True,
                 initial_step_size: float = 0.1, max_number_of_integration_steps: int = 10,
                 randomise_number_of_integration_steps: bool = False, use_metropolis_accept_reject: bool = True):
        r"""
        The constructor of the ToroidalLeapfrogMediator class.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.continuous_potential.ContinuousPotential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
        kinetic_energy : kinetic_energy.kinetic_energy.KineticEnergy
            Instance of the chosen child class of kinetic_energy.kinetic_energy.KineticEnergy.
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
        initial_step_size : float, optional
            The initial step size of the integrator.
        max_number_of_integration_steps : int, optional
            The maximum number of numerical integration steps at each iteration of the Markov process.
        randomise_number_of_integration_steps : bool, optional
            When True, Mediator sets the number of numerical integration steps (at each iteration of the Markov
            process) by drawing uniformly from the set $\{ 1, 2, \dots , max_number_of_integration_steps \}$; when
            False, the number of numerical integration steps is always max_number_of_integration_steps.
        use_metropolis_accept_reject : bool, optional
            When True, the Metropolis step is used following the generation of each candidate configuration; when
            False, all candidate configurations are accepted.

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
            If kinetic_energy is not an instance of some child class of kinetic_energy.kinetic_energy.KineticEnergy.
        base.exceptions.ConfigurationError
            If initial_step_size is not greater than 0.0.
        base.exceptions.ConfigurationError
            If max_number_of_integration_steps is not greater than 0.
        base.exceptions.ConfigurationError
            If type(randomise_number_of_integration_steps) is not bool.
        base.exceptions.ConfigurationError
            If type(use_metropolis_accept_reject) is not bool
        base.exceptions.ConfigurationError
            If type(element) is not np.float64 for element in size_of_particle_space.
        """
        super().__init__(potential, samplers, kinetic_energy, minimum_temperature, maximum_temperature,
                         number_of_temperature_increments, number_of_equilibration_iterations, number_of_observations,
                         proposal_dynamics_adaptor_is_on, initial_step_size, max_number_of_integration_steps,
                         randomise_number_of_integration_steps, use_metropolis_accept_reject)
        for element in size_of_particle_space:
            if type(element) != np.float64:
                raise ConfigurationError(f"For each component of size_of_particle_space, give a float value when using "
                                         f"{self.__class__.__name__}.")
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, samplers=samplers, kinetic_energy=kinetic_energy,
                           minimum_temperature=minimum_temperature, maximum_temperature=maximum_temperature,
                           number_of_temperature_increments=number_of_temperature_increments,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           proposal_dynamics_adaptor_is_on=proposal_dynamics_adaptor_is_on,
                           initial_step_size=initial_step_size,
                           max_number_of_integration_steps=max_number_of_integration_steps,
                           randomise_number_of_integration_steps=randomise_number_of_integration_steps,
                           use_metropolis_accept_reject=use_metropolis_accept_reject)

    def _get_candidate_configuration(self, temperature):
        """
        Returns the candidate momenta, positions and potential after self._number_of_integration_steps integration
        steps.

        Parameters
        ----------
        temperature : float
            The sampling temperature.

        Returns
        -------
        numpy.ndarray
            The candidate momenta. A two-dimensional numpy array of size (number_of_particles,
            dimensionality_of_particle_space); each element is a float and represents one Cartesian component of the
            candidate momentum of a single particle.
        numpy.ndarray
            The candidate positions. A two-dimensional numpy array of size (number_of_particles,
            dimensionality_of_particle_space); each element is a float and represents one Cartesian component of the
            candidate position of a single particle.
        float
            The potential of the candidate configuration.
        """
        candidate_momenta = (self._momenta -
                             0.5 * self._step_size * self._potential.get_gradient(self._positions) / temperature)
        candidate_positions = get_shortest_vectors_on_torus(
            self._positions + self._step_size * self._kinetic_energy.get_gradient(candidate_momenta) / temperature)
        for _ in range(self._number_of_integration_steps - 1):
            candidate_momenta -= self._step_size * self._potential.get_gradient(candidate_positions) / temperature
            candidate_positions = get_shortest_vectors_on_torus(
                candidate_positions + self._step_size * self._kinetic_energy.get_gradient(candidate_momenta) /
                temperature)
        return (candidate_momenta - 0.5 * self._step_size *
                self._potential.get_gradient(candidate_positions) / temperature, candidate_positions,
                self._potential.get_value(candidate_positions))
