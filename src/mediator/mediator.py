"""Module for the Mediator class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from kinetic_energy.kinetic_energy import KineticEnergy
from model_settings import beta, dimensionality_of_particle_space, number_of_particles, \
    range_of_initial_particle_positions
from potential.potential import Potential
from sampler.sampler import Sampler
import numpy as np


class Mediator(metaclass=ABCMeta):
    """Abstract Mediator class."""

    def __init__(self, kinetic_energy: KineticEnergy, potential: Potential, sampler: Sampler,
                 number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                 initial_step_size: float = 0.1, max_number_of_integration_steps: int = 10,
                 randomise_number_of_integration_steps: bool = False, step_size_adaptor_is_on: bool = True,
                 use_metropolis_accept_reject: bool = True, **kwargs):
        """
        The constructor of the Mediator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

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

        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

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

        self._kinetic_energy = kinetic_energy
        self._potential = potential
        self._sampler = sampler
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._number_of_observations_between_screen_prints_for_clock = int(number_of_observations / 10)
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._initial_step_size = initial_step_size
        self._step_size = beta * initial_step_size
        self._max_number_of_integration_steps = max_number_of_integration_steps
        self._number_of_integration_steps = self._max_number_of_integration_steps
        self._randomise_number_of_integration_steps = randomise_number_of_integration_steps
        self._step_size_adaptor_is_on = step_size_adaptor_is_on
        self._use_metropolis_accept_reject = use_metropolis_accept_reject
        self._momenta = self._kinetic_energy.get_momentum_observations()
        self._positions = self._initialise_position_array()
        self._sample = self._sampler.initialise_sample_array(self._total_number_of_iterations)
        self._sample[0, :] = self._sampler.get_observation(self._momenta, self._positions)
        self._current_potential = self._potential.get_value(self._positions)
        self._number_of_accepted_trajectories = 0
        self._number_of_numerical_divergences = 0
        super().__init__(**kwargs)

    def generate_sample(self):
        """Runs the Markov chain in order to generate the sample."""
        for i in range(self._total_number_of_iterations):
            if i == self._number_of_equilibration_iterations:
                self._number_of_accepted_trajectories = 0
            if self._randomise_number_of_integration_steps:
                self._number_of_integration_steps = 1 + np.random.randint(self._max_number_of_integration_steps)

            candidate_momenta, candidate_positions = self._get_candidate_configuration()
            candidate_potential = self._potential.get_value(candidate_positions)
            current_energy = self._kinetic_energy.get_value(self._momenta) + self._current_potential
            energy_change = self._kinetic_energy.get_value(candidate_momenta) + candidate_potential - current_energy

            if energy_change / current_energy > 100.0:
                self._number_of_numerical_divergences += 1
            if self._use_metropolis_accept_reject:
                if energy_change < 0.0 or np.random.uniform(0, 1) < np.exp(- beta * energy_change):
                    self._update_system_state(candidate_momenta, candidate_positions, candidate_potential)
                    self._number_of_accepted_trajectories += 1
            else:
                self._update_system_state(candidate_momenta, candidate_positions, candidate_potential)

            self._sample[i + 1, :] = self._sampler.get_observation(self._momenta, self._positions)
            self._momenta = self._kinetic_energy.get_momentum_observations()

            if (i + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                current_sample_size = i + 1
                print(f"{current_sample_size} observations drawn out of a total of {self._total_number_of_iterations} "
                      f"(including equilibration observations).")

            if self._step_size_adaptor_is_on and i < self._number_of_equilibration_iterations and (i + 1) % 100 == 0:
                acceptance_rate = self._number_of_accepted_trajectories / 100.0
                if acceptance_rate > 0.9:
                    self._step_size *= 1.1
                elif acceptance_rate < 0.8:
                    self._step_size *= 0.9
                self._number_of_accepted_trajectories = 0

    @abstractmethod
    def _get_candidate_configuration(self):
        """
        Returns the candidate momenta and positions after self._number_of_integration_steps integration steps.

        Returns
        -------
        numpy.ndarray
            The candidate momenta and positions.
        """
        raise NotImplementedError

    def _update_system_state(self, new_momenta, new_positions, new_potential):
        self._momenta = new_momenta
        self._positions = new_positions
        self._current_potential = new_potential

    def post_run_methods(self):
        self._print_markov_chain_summary()
        self._sampler.output_sample(self._sample)

    @staticmethod
    def _initialise_position_array():
        if dimensionality_of_particle_space == 1:
            if type(range_of_initial_particle_positions) == float:
                return np.array(
                    [np.atleast_1d(range_of_initial_particle_positions) for _ in range(number_of_particles)])
            else:
                return np.array([np.atleast_1d(np.random.uniform(*range_of_initial_particle_positions))
                                 for _ in range(number_of_particles)])
        else:
            if type(range_of_initial_particle_positions[0]) == float:
                return np.array([range_of_initial_particle_positions for _ in range(number_of_particles)])
            else:
                return np.array([[np.random.uniform(*axis_range) for axis_range in range_of_initial_particle_positions]
                                 for _ in range(number_of_particles)])

    def _print_markov_chain_summary(self):
        acceptance_rate = self._number_of_accepted_trajectories / self._number_of_observations
        print(f"Metropolis-Hastings acceptance rate = {acceptance_rate}")
        print(f"Number of numerical instabilities (relative energy increases by two orders of magnitude) = "
              f"{self._number_of_numerical_divergences}")
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
