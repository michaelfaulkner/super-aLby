"""Module for the LeapfrogIntegrator class."""
from .mediator import Mediator
from base.logging import log_init_arguments
from kinetic_energy.kinetic_energy import KineticEnergy
from potential.potential import Potential
from sampler.sampler import Sampler
import logging


class LeapfrogMediator(Mediator):
    """
    This class implements the mediator using the leapfrog numerical integrator.
    """

    def __init__(self, kinetic_energy: KineticEnergy, potential: Potential, sampler: Sampler,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 initial_step_size: float = 0.1, max_number_of_integration_steps: int = 10,
                 randomise_number_of_integration_steps: bool = False, step_size_adaptor_is_on: bool = True,
                 use_metropolis_accept_reject: bool = True):
        """
        The constructor of the LeapfrogMediator class.

        Parameters
        ----------
        kinetic_energy : kinetic_energy.kinetic_energy.KineticEnergy

        potential : potential.potential.Potential

        sampler : from sampler.sampler.Sampler

        number_of_equilibration_iterations : int, optional

        number_of_observations : int, optional

        initial_step_size : float, optional

        max_number_of_integration_steps : int, optional

        randomise_number_of_integration_steps : bool, optional

        step_size_adaptor_is_on : bool, optional

        use_metropolis_accept_reject : bool, optional

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
        super().__init__(kinetic_energy, potential, sampler,
                         number_of_equilibration_iterations, number_of_observations, initial_step_size,
                         max_number_of_integration_steps, randomise_number_of_integration_steps,
                         step_size_adaptor_is_on, use_metropolis_accept_reject)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           kinetic_energy_instance=kinetic_energy, potential_instance=potential,
                           sampler_instance=sampler,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           initial_step_size=initial_step_size,
                           max_number_of_integration_steps=max_number_of_integration_steps,
                           randomise_number_of_integration_steps=randomise_number_of_integration_steps,
                           step_size_adaptor_is_on=step_size_adaptor_is_on,
                           use_metropolis_accept_reject=use_metropolis_accept_reject)

    def _get_candidate_configuration(self):
        """
        Returns the candidate momenta and positions after self._number_of_integration_steps integration steps.

        Returns
        -------
        numpy.ndarray
            The candidate momenta and positions.
        """
        candidate_momenta, candidate_positions = self._momenta, self._positions
        half_step_size = 0.5 * self._step_size
        candidate_momenta -= half_step_size * self._potential.get_gradient(self._positions)
        for _ in range(self._number_of_integration_steps - 1):
            candidate_positions += self._step_size * self._kinetic_energy.get_gradient(candidate_momenta)
            candidate_momenta -= self._step_size * self._potential.get_gradient(candidate_positions)
        candidate_positions += self._step_size * self._kinetic_energy.get_gradient(candidate_momenta)
        candidate_momenta -= half_step_size * self._potential.get_gradient(candidate_positions)
        return candidate_momenta, candidate_positions
