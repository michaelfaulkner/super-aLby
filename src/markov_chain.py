"""Module for the MarkovChain class."""
import numpy as np


class MarkovChain:
    """
    MarkovChain class.

    The class provides the Markov-chain function (as run()).
    """

    def __init__(self, dimension_of_target_distribution, integrator_instance, kinetic_energy_instance,
                 potential_instance, initial_step_size=1.0, max_number_of_integration_steps=10,
                 number_of_equilibration_iterations=10000, number_of_observations=1100, step_size_adaptor_is_on=True,
                 use_metropolis_accept_reject=True, randomise_number_of_integration_steps=False):
        """
        The constructor of the MarkovChain class.

        Parameters
        ----------
        dimension_of_target_distribution : int

        integrator_instance : Python class instance

        kinetic_energy_instance : Python class instance

        potential_instance : Python class instance

        initial_step_size : float, optional

        max_number_of_integration_steps : int, optional

        number_of_equilibration_iterations : int, optional

        number_of_observations : int, optional

        use_metropolis_accept_reject : Boolean, optional

        randomise_number_of_integration_steps : Boolean, optional

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
        self._step_size = initial_step_size
        self._max_number_of_integration_steps = max_number_of_integration_steps
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._step_size_adaptor_is_on = step_size_adaptor_is_on
        self._use_metropolis_accept_reject = use_metropolis_accept_reject
        self._randomise_number_of_integration_steps = randomise_number_of_integration_steps

    def run(self, charges=None):
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
        support_variable = np.zeros(self._dimension_of_target_distribution)
        support_variable_sample = np.zeros(
            (self._dimension_of_target_distribution,
             self._number_of_equilibration_iterations + self._number_of_observations + 1))
        support_variable_sample[:, 0] = support_variable
        momentum = np.zeros(self._dimension_of_target_distribution)
        momentum_sample = np.zeros(
            (self._dimension_of_target_distribution,
             self._number_of_equilibration_iterations + self._number_of_observations + 1))
        momentum_sample[:, 0] = momentum

        for i in range(self._number_of_equilibration_iterations + self._number_of_observations):
            momentum = self._kinetic_energy_instance.momentum_observation(momentum)
            if self._randomise_number_of_integration_steps:
                number_of_integration_steps = 1 + np.random.randint(self._max_number_of_integration_steps)
            momentum_candidate, support_variable_candidate = self._integrator_instance.get_candidate_configuration(
                momentum, support_variable, number_of_integration_steps, self._step_size, charges=None)

            if self._use_metropolis_accept_reject:
                delta_hamiltonian = (self._kinetic_energy_instance.current_value(momentum_candidate) -
                                     self._kinetic_energy_instance.current_value(momentum) +
                                     self._potential_instance.current_value(support_variable_candidate,
                                                                            charges=charges) -
                                     self._potential_instance.current_value(support_variable, charges=charges))
                if abs(delta_hamiltonian) > 1000.0:
                    if i < self._number_of_equilibration_iterations:
                        number_of_numerical_divergences_during_equilibration += 1
                    else:
                        number_of_numerical_divergences_during_equilibrated_process += 1
                if np.random.uniform(0, 1) < np.exp(- delta_hamiltonian):
                    support_variable = support_variable_candidate
                    momentum = momentum_candidate
                    number_of_accepted_trajectories += 1
            else:
                support_variable = support_variable_candidate
                momentum = momentum_candidate
            support_variable_sample[:, i] = support_variable
            momentum_sample[:, i] = momentum

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
        return (momentum_sample, support_variable_sample, self._step_size, acceptance_rate,
                number_of_numerical_divergences_during_equilibration,
                number_of_numerical_divergences_during_equilibrated_process)
