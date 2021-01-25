"""Module for the ToroidalLeapfrogIntegrator class."""
from .mediator import Mediator
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
from model_settings import size_of_particle_space
from kinetic_energy.kinetic_energy import KineticEnergy
from potential.potential import Potential
from sampler.sampler import Sampler
import logging
import numpy as np


class ToroidalLeapfrogMediator(Mediator):
    """
    This class implements the mediator using the leapfrog numerical integrator with corrections of the particle
    positions to account for the toroidal geometry (using base.vectors.get_shortest_vectors_on_torus()).
    """

    def __init__(self, kinetic_energy: KineticEnergy, potential: Potential, sampler: Sampler,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 initial_step_size: float = 0.1, max_number_of_integration_steps: int = 10,
                 randomise_number_of_integration_steps: bool = False, step_size_adaptor_is_on: bool = True,
                 use_metropolis_accept_reject: bool = True):
        r"""
        The constructor of the ToroidalLeapfrogMediator class.

        Parameters
        ----------
        kinetic_energy : kinetic_energy.kinetic_energy.KineticEnergy
            Instance of the chosen child class of kinetic_energy.kinetic_energy.KineticEnergy.
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        sampler : sampler.sampler.Sampler
            Instance of the chosen child class of sampler.sampler.Sampler.
        number_of_equilibration_iterations : int, optional
            Number of equilibration iterations of the Markov process.
        number_of_observations : int, optional
            Number of sample observations, i.e., the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
        initial_step_size : float, optional
            The initial step size of the integrator.
        max_number_of_integration_steps : int, optional
            The maximum number of numerical integration steps at each iteration of the Markov process.
        randomise_number_of_integration_steps : bool, optional
            When True, Mediator sets the number of numerical integration steps (at each iteration of the Markov
            process) by drawing uniformly from the set $\{ 1, 2, \dots , max_number_of_integration_steps \}$; when
            False, the number of numerical integration steps is always max_number_of_integration_steps.
        step_size_adaptor_is_on : bool, optional
            When True, the step size of the integrator is tuned during the equilibration process.
        use_metropolis_accept_reject : bool, optional
            When True, the Metropolis step is used following the generation of each candidate configuration; when
            False, all candidate configurations are accepted.

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
        base.exceptions.ConfigurationError
            If type(element) is not np.float64 for element in size_of_particle_space.
        """
        super().__init__(kinetic_energy, potential, sampler, number_of_equilibration_iterations, number_of_observations,
                         initial_step_size, max_number_of_integration_steps, randomise_number_of_integration_steps,
                         step_size_adaptor_is_on, use_metropolis_accept_reject)
        for element in size_of_particle_space:
            if type(element) != np.float64:
                raise ConfigurationError(f"For each component of size_of_particle_space, give a float value when using "
                                         f"{self.__class__.__name__}.")
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           kinetic_energy=kinetic_energy, potential=potential, sampler=sampler,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           initial_step_size=initial_step_size,
                           max_number_of_integration_steps=max_number_of_integration_steps,
                           randomise_number_of_integration_steps=randomise_number_of_integration_steps,
                           step_size_adaptor_is_on=step_size_adaptor_is_on,
                           use_metropolis_accept_reject=use_metropolis_accept_reject)

    def _get_candidate_configuration(self):
        """
        Returns the candidate momenta, positions and potential after self._number_of_integration_steps integration
        steps.

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
        candidate_momenta = self._momenta - 0.5 * self._step_size * self._potential.get_gradient(self._positions)
        candidate_positions = get_shortest_vectors_on_torus(self._positions + self._step_size *
                                                            self._kinetic_energy.get_gradient(candidate_momenta))
        for _ in range(self._number_of_integration_steps - 1):
            candidate_momenta -= self._step_size * self._potential.get_gradient(candidate_positions)
            candidate_positions = get_shortest_vectors_on_torus(candidate_positions + self._step_size *
                                                                self._kinetic_energy.get_gradient(candidate_momenta))
        return (candidate_momenta - 0.5 * self._step_size * self._potential.get_gradient(candidate_positions),
                candidate_positions, self._potential.get_value(candidate_positions))
