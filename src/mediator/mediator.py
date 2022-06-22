"""Module for the Mediator class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
from model_settings import dimensionality_of_particle_space, number_of_particles, range_of_initial_particle_positions
from potential.potential import Potential
from sampler.sampler import Sampler
import numpy as np


class Mediator(metaclass=ABCMeta):
    """Abstract Mediator class."""

    def __init__(self, potential: Potential, sampler: Sampler, number_of_equilibration_iterations: int = 10000,
                 number_of_observations: int = 100000, proposal_dynamics_adaptor_is_on: bool = True, **kwargs):
        r"""
        The constructor of the Mediator class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        sampler : sampler.sampler.Sampler
            Instance of the chosen child class of sampler.sampler.Sampler.
        number_of_equilibration_iterations : int, optional
            Number of equilibration iterations of the Markov process.
        number_of_observations : int, optional
            Number of sample observations, i.e., the sample size. This is equal to the number of post-equilibration
            iterations of the Markov process.
        proposal_dynamics_adaptor_is_on : bool, optional
            When True, the size of either the numerical integration step (DeterministicMediator) or the width of the
            proposal distribution (MetropolisMediator) is tuned during the equilibration process.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If number_of_equilibration_iterations is less than 0.
        base.exceptions.ConfigurationError
            If number_of_observations is not greater than 0.
        base.exceptions.ConfigurationError
            If type(proposal_dynamics_adaptor_is_on) is not bool.
        """
        super().__init__(**kwargs)
        if number_of_equilibration_iterations < 0:
            raise ConfigurationError(f"Give a value not less than 0 as number_of_equilibration_iterations in "
                                     f"{self.__class__.__name__}.")
        if number_of_observations <= 0:
            raise ConfigurationError(f"Give a value greater than 0 as number_of_observations in "
                                     f"{self.__class__.__name__}.")
        if type(proposal_dynamics_adaptor_is_on) is not bool:
            raise ConfigurationError(f"Give a value of type bool as proposal_dynamics_adaptor_is_on in "
                                     f"{self.__class__.__name__}.")
        self._potential = potential
        self._sampler = sampler
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._number_of_observations_between_screen_prints_for_clock = int(number_of_observations / 10)
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._proposal_dynamics_adaptor_is_on = proposal_dynamics_adaptor_is_on
        self._positions = self._initialise_position_array()
        self._sample = self._sampler.initialise_sample_array(self._total_number_of_iterations)
        self._number_of_accepted_trajectories = 0
        self._use_metropolis_accept_reject = True

    def generate_sample(self):
        """Runs the Markov chain in order to generate the sample."""
        for markov_chain_index in range(self._total_number_of_iterations):
            if markov_chain_index == self._number_of_equilibration_iterations:
                self._number_of_accepted_trajectories = 0
            self._generate_single_observation(markov_chain_index)
            if (self._proposal_dynamics_adaptor_is_on and markov_chain_index < self._number_of_equilibration_iterations
                    and (markov_chain_index + 1) % 100 == 0):
                self._proposal_dynamics_adaptor()
                self._number_of_accepted_trajectories = 0
            if (markov_chain_index + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                current_sample_size = markov_chain_index + 1
                print(f"{current_sample_size} observations drawn out of a total of {self._total_number_of_iterations} "
                      f"(including equilibration observations).")

    def post_simulation_methods(self):
        """Runs all methods that follow the Markov process."""
        self._sampler.output_sample(self._sample)
        self._print_markov_chain_summary()

    @staticmethod
    def _initialise_position_array():
        # TODO move to potential class
        """
        Returns the initial positions array.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle, e.g., two particles
            (confined to one-dimensional space) at positions 0.0 and 1.0 is represented by [[0.0] [1.0]]; three
            particles (confined to two-dimensional space) at positions (0.0, 1.0), (2.0, 3.0) and (- 1.0, - 2.0) is
            represented by [[0.0 1.0] [2.0 3.0] [-1.0 -2.0]].
        """
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

    @abstractmethod
    def _generate_single_observation(self, markov_chain_step_index):
        """Advances the Markov chain by one step and adds a single observation to the sample."""
        raise NotImplementedError

    @abstractmethod
    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step (DeterministicMediator) or the width of the proposal
            distribution (MetropolisMediator)."""
        raise NotImplementedError

    @abstractmethod
    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        raise NotImplementedError
