"""Module for the EuclideanLeapfrogIntegrator class."""
from .euclidean_and_lazy_toroidal_leapfrog_mediators import EuclideanAndLazyToroidalLeapfrogMediators
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from kinetic_energy.kinetic_energy import KineticEnergy
from model_settings import size_of_particle_space
from potential.continuous_potential import ContinuousPotential
from sampler.sampler import Sampler
import logging


class EuclideanLeapfrogMediator(EuclideanAndLazyToroidalLeapfrogMediators):
    """
    This class implements the mediator using the leapfrog numerical integrator on Euclidean space.
    """

    def __init__(self, potential: ContinuousPotential, sampler: Sampler, kinetic_energy: KineticEnergy,
                 minimum_temperature: float = 1.0, maximum_temperature: float = 1.0,
                 number_of_temperature_values: int = 1, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, proposal_dynamics_adaptor_is_on: bool = True,
                 initial_step_size: float = 0.1, max_number_of_integration_steps: int = 10,
                 randomise_number_of_integration_steps: bool = False, use_metropolis_accept_reject: bool = True):
        r"""
        The constructor of the EuclideanLeapfrogMediator class.

        Parameters
        ----------
        potential : potential.continuous_potential.ContinuousPotential
            Instance of the chosen child class of potential.continuous_potential.ContinuousPotential.
        sampler : sampler.sampler.Sampler
            Instance of the chosen child class of sampler.sampler.Sampler.
        kinetic_energy : kinetic_energy.kinetic_energy.KineticEnergy
            Instance of the chosen child class of kinetic_energy.kinetic_energy.KineticEnergy.
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
            If type(use_metropolis_accept_reject) is not bool.
        base.exceptions.ConfigurationError
            If element is not None for element in size_of_particle_space.
        """
        super().__init__(potential, sampler, kinetic_energy, minimum_temperature, maximum_temperature,
                         number_of_temperature_values, number_of_equilibration_iterations, number_of_observations,
                         proposal_dynamics_adaptor_is_on, initial_step_size, max_number_of_integration_steps,
                         randomise_number_of_integration_steps, use_metropolis_accept_reject)
        for element in size_of_particle_space:
            if element is not None:
                raise ConfigurationError(f"For each component of size_of_particle_space, give None when using "
                                         f"{self.__class__.__name__}.")
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, sampler=sampler, kinetic_energy=kinetic_energy,
                           minimum_temperature=minimum_temperature, maximum_temperature=maximum_temperature,
                           number_of_temperature_values=number_of_temperature_values,
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
        return self._get_candidate_configuration_without_toroidal_corrections(temperature)
