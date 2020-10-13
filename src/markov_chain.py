"""Module for the MarkovChain class."""
from base.logging import log_init_arguments
import logging
import numpy as np


class MarkovChain:
    """
    MarkovChain class.

    The class provides the Markov-chain function (as run()).
    """

    def __init__(self, dimension_of_target_distribution, integrator_instance, kinetic_energy_instance,
                 potential_instance, observer_instance, number_of_equilibration_iterations=1000,
                 number_of_observations=1000, initial_step_size=1.0, max_number_of_integration_steps=10,
                 randomise_initial_momenta=False, randomise_initial_position=False,
                 randomise_number_of_integration_steps=False, step_size_adaptor_is_on=True,
                 use_metropolis_accept_reject=True):
        """
        The constructor of the MarkovChain class.

        Parameters
        ----------
        dimension_of_target_distribution : int

        integrator_instance : Python class instance

        kinetic_energy_instance : Python class instance

        potential_instance : Python class instance

        observer_instance : Python class instance

        number_of_equilibration_iterations : int, optional

        number_of_observations : int, optional

        initial_step_size : float, optional

        max_number_of_integration_steps : int, optional

        randomise_number_of_integration_steps : Boolean, optional

        randomise_initial_momenta : Boolean, optional

        randomise_initial_position : Boolean, optional

        step_size_adaptor_is_on : Boolean, optional

        use_metropolis_accept_reject : Boolean, optional

        Raises
        ------
        base.exceptions.ValueError
            If the prefactor equals 0.
        """
        if dimension_of_target_distribution == 0:
            raise ValueError("Give a value not equal to 0 as the dimension of the target distribution {0}.".format(
                self.__class__.__name__))
        if initial_step_size == 0.0:
            raise ValueError(
                "Give a value not equal to 0 as the initial step size of the numerical integrator {0}.".format(
                    self.__class__.__name__))
        if max_number_of_integration_steps == 0:
            raise ValueError(
                "Give a value not equal to 0 as the maximum number of numerical integration steps {0}.".format(
                    self.__class__.__name__))
        if number_of_observations == 0:
            raise ValueError(
                "Give a value not equal to 0 as the number of observations of target distribution {0}.".format(
                    self.__class__.__name__))
        self._dimension_of_target_distribution = dimension_of_target_distribution
        self._integrator_instance = integrator_instance
        self._kinetic_energy_instance = kinetic_energy_instance
        self._potential_instance = potential_instance
        self._observer_instance = observer_instance
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._step_size = initial_step_size
        self._max_number_of_integration_steps = max_number_of_integration_steps
        self._randomise_number_of_integration_steps = randomise_number_of_integration_steps
        self._randomise_initial_momenta = randomise_initial_momenta
        self._randomise_initial_position = randomise_initial_position
        self._step_size_adaptor_is_on = step_size_adaptor_is_on
        self._use_metropolis_accept_reject = use_metropolis_accept_reject
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           dimension_of_target_distribution=dimension_of_target_distribution,
                           integrator_instance=integrator_instance, kinetic_energy_instance=kinetic_energy_instance,
                           potential_instance=potential_instance, observer_instance=observer_instance,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations, initial_step_size=initial_step_size,
                           max_number_of_integration_steps=max_number_of_integration_steps,
                           randomise_initial_momenta=randomise_initial_momenta,
                           randomise_initial_position=randomise_initial_position,
                           randomise_number_of_integration_steps=randomise_number_of_integration_steps,
                           step_size_adaptor_is_on=step_size_adaptor_is_on,
                           use_metropolis_accept_reject=use_metropolis_accept_reject)

    def get_sample(self, charges=None):
        """
        Runs the Markov chain and returns the generated observations of the target and momentum distributions.

        Parameters
        ----------
        charges : optional
            All the charges needed to run the Markov chain.

        Returns
        -------
        float
            The observations (of the target and momentum distributions) generated by the Markov chain.
     """
        initial_step_size = self._step_size
        number_of_accepted_trajectories = 0
        number_of_numerical_divergences_during_equilibration = 0
        number_of_numerical_divergences_during_equilibrated_process = 0
        number_of_integration_steps = self._max_number_of_integration_steps
        momentum = self._initialise_momentum_or_position(initialise_momentum=True)
        position = self._initialise_momentum_or_position(initialise_momentum=False)
        sample = np.zeros((self._dimension_of_target_distribution,
                           self._number_of_equilibration_iterations + self._number_of_observations + 1))

        for i in range(self._number_of_equilibration_iterations + self._number_of_observations):
            momentum = self._kinetic_energy_instance.momentum_observation(momentum)
            if self._randomise_number_of_integration_steps:
                number_of_integration_steps = 1 + np.random.randint(self._max_number_of_integration_steps)
            momentum_candidate, position_candidate = self._integrator_instance.get_candidate_configuration(
                momentum, position, number_of_integration_steps, self._step_size, charges=None)

            if self._use_metropolis_accept_reject:
                delta_hamiltonian = (self._kinetic_energy_instance.current_value(momentum_candidate) -
                                     self._kinetic_energy_instance.current_value(momentum) +
                                     self._potential_instance.current_value(position_candidate,
                                                                            charges=charges) -
                                     self._potential_instance.current_value(position, charges=charges))
                if abs(delta_hamiltonian) > 1000.0:
                    if i < self._number_of_equilibration_iterations:
                        number_of_numerical_divergences_during_equilibration += 1
                    else:
                        number_of_numerical_divergences_during_equilibrated_process += 1
                if np.random.uniform(0, 1) < np.exp(- delta_hamiltonian):
                    position = position_candidate
                    momentum = momentum_candidate
                    number_of_accepted_trajectories += 1
            else:
                position = position_candidate
                momentum = momentum_candidate
            sample[:, i] = self._observer_instance.get_observation(momentum, position)

            if self._step_size_adaptor_is_on and i < self._number_of_equilibration_iterations and (i + 1) % 100 == 0:
                acceptance_rate = number_of_accepted_trajectories / 100.0
                if acceptance_rate > 0.8:
                    self._step_size *= 1.1
                elif acceptance_rate < 0.6:
                    self._step_size *= 0.9
                number_of_accepted_trajectories = 0

        acceptance_rate = number_of_accepted_trajectories / float(self._number_of_observations)
        print(
            "Max number of integration steps = %d; initial numerical step size = %.3f; final numerical step size = %.3f"
            % (self._max_number_of_integration_steps, initial_step_size, self._step_size))
        print("Metropolis-Hastings acceptance rate = %f" % acceptance_rate)
        print("Number of numerical divergences during equilibration = %d" %
              number_of_numerical_divergences_during_equilibration)
        print("Number of numerical divergences during equilibrated process = %d" %
              number_of_numerical_divergences_during_equilibrated_process)
        return sample

    def _initialise_momentum_or_position(self, initialise_momentum):
        if initialise_momentum:
            randomise_initial_values = self._randomise_initial_momenta
        else:
            randomise_initial_values = self._randomise_initial_position
        if randomise_initial_values:
            return np.random.uniform(low=-10.0, high=10.0, size=self._dimension_of_target_distribution)
        else:
            return np.zeros(self._dimension_of_target_distribution)
