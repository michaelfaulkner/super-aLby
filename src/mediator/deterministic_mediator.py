"""Module for the DeterministicMediator class."""
from .mediator import Mediator
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from kinetic_energy.kinetic_energy import KineticEnergy
from potential.continuous_potential import ContinuousPotential
from potential.ising_potential import IsingPotential
from sampler.sampler import Sampler
import numpy as np


class DeterministicMediator(Mediator, metaclass=ABCMeta):
    """Abstract DeterministicMediator class.  This is the parent class for all mediators that use Newtonian,
        relativistic or super-relativistic dynamics."""

    def __init__(self, potential: ContinuousPotential, sampler: Sampler, kinetic_energy: KineticEnergy,
                 minimum_temperature: float = 1.0, maximum_temperature: float = 1.0,
                 number_of_temperature_values: int = 1, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, proposal_dynamics_adaptor_is_on: bool = True,
                 initial_step_size: float = 0.1, max_number_of_integration_steps: int = 10,
                 randomise_number_of_integration_steps: bool = False, use_metropolis_accept_reject: bool = True,
                 **kwargs):
        r"""
        The constructor of the DeterministicMediator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

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
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If potential is not an instance of some child class of potential.potential.Potential.
        base.exceptions.ConfigurationError
            If sampler is not an instance of some child class of sampler.sampler.Sampler.
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
        """
        super().__init__(potential, sampler, minimum_temperature, maximum_temperature, number_of_temperature_values,
                         number_of_equilibration_iterations, number_of_observations, proposal_dynamics_adaptor_is_on,
                         **kwargs)
        if not isinstance(kinetic_energy, KineticEnergy):
            raise ConfigurationError(f"Give a kinetic_energy class as the value for kinetic_energy in "
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
        if type(use_metropolis_accept_reject) is not bool:
            raise ConfigurationError(f"Give a value of type bool as randomise_number_of_integration_steps in "
                                     f"{self.__class__.__name__}.")
        if isinstance(potential, IsingPotential):
            raise ConfigurationError(f"Do not give IsingPotential as the value for potential in "
                                     f"{self.__class__.__name__} as child classes of DeterministicMediator cannot "
                                     f"resolve potentials defined on discrete configuration space.")
        """In the following line, we re-declare self._potential (it has already been declared in Mediator.__init__()) 
            as the default potential of DeterministicMediator is ContinuousPotential, which contains get_gradient()."""
        self._potential = potential
        self._kinetic_energy = kinetic_energy
        self._initial_step_size = initial_step_size
        self._step_size = initial_step_size
        self._max_number_of_integration_steps = max_number_of_integration_steps
        self._number_of_integration_steps = self._max_number_of_integration_steps
        self._randomise_number_of_integration_steps = randomise_number_of_integration_steps
        self._use_metropolis_accept_reject = use_metropolis_accept_reject
        self._target_acceptance_rate = 0.85  # TODO add functionality so the user can set self._target_acceptance_rate
        """The following objects are set in self._reset_arrays_and_counters()"""
        self._momenta = None
        self._current_potential = None
        self._number_of_unstable_trajectories = None

    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        super()._reset_arrays_and_counters(temperature)
        self._momenta = self._kinetic_energy.get_momentum_observations(temperature)
        self._current_potential = self._potential.get_value(self._positions)
        self._sample[0, :] = self._sampler.get_observation(self._momenta, self._positions, self._current_potential)
        self._number_of_unstable_trajectories = 0

    def _generate_single_observation(self, markov_chain_step_index, temperature):
        """Advances the Markov chain by one step and adds a single observation to the sample."""
        if self._randomise_number_of_integration_steps:
            self._number_of_integration_steps = 1 + np.random.randint(self._max_number_of_integration_steps)
        candidate_momenta, candidate_positions, candidate_potential = self._get_candidate_configuration(temperature)
        current_energy = self._kinetic_energy.get_value(self._momenta) + self._current_potential
        energy_change = self._kinetic_energy.get_value(candidate_momenta) + candidate_potential - current_energy
        if energy_change / current_energy > 1000.0:
            self._number_of_unstable_trajectories += 1
        if self._use_metropolis_accept_reject:
            if energy_change < 0.0 or np.random.uniform(0.0, 1.0) < np.exp(- energy_change / temperature):
                self._update_system_state(candidate_momenta, candidate_positions, candidate_potential)
                self._number_of_accepted_trajectories += 1
        else:
            self._update_system_state(candidate_momenta, candidate_positions, candidate_potential)
        self._momenta = self._kinetic_energy.get_momentum_observations(temperature)
        self._sample[markov_chain_step_index + 1, :] = self._sampler.get_observation(self._momenta, self._positions,
                                                                                     self._current_potential)

    @abstractmethod
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
        raise NotImplementedError

    def _update_system_state(self, new_momenta, new_positions, new_potential):
        """
        Resets the stored system state following an accepted proposal.

        Parameters
        ----------
        new_momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.
        new_positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.
        new_potential : float
            The potential of the accepted configuration.
        """
        self._momenta, self._positions, self._current_potential = new_momenta, new_positions, new_potential

    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step or the width of the proposal distribution."""
        acceptance_rate = self._number_of_accepted_trajectories / 100.0
        if acceptance_rate > 1.05 * self._target_acceptance_rate:
            self._step_size *= 1.1
        elif acceptance_rate < 0.95 * self._target_acceptance_rate:
            self._step_size *= 0.9

    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        if self._use_metropolis_accept_reject:
            acceptance_rate = self._number_of_accepted_trajectories / self._number_of_observations
            print(f"Metropolis-Hastings acceptance rate = {acceptance_rate}")
        if self._proposal_dynamics_adaptor_is_on:
            print(f"Initial numerical step size = {self._initial_step_size}")
            print(f"Final numerical step size = {self._step_size}")
        else:
            print(f"Numerical step size = {self._step_size}")
        print(f"Number of unstable numerical trajectories (defined as a relative energy increases by three orders of "
              f"magnitude) = {self._number_of_unstable_trajectories}")
        if self._randomise_number_of_integration_steps:
            print(f"Maximum number of integration steps = {self._max_number_of_integration_steps} (the number of "
                  f"integration steps was randomised).")
        else:
            print(f"Number of integration steps = {self._max_number_of_integration_steps}")
