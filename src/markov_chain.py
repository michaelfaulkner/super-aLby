"""Module for the MarkovChain class."""
from base.logging import log_init_arguments
from model_settings import number_of_particles, dimensionality_of_particle_space
import logging
import numpy as np


class MarkovChain:
    """
    MarkovChain class.

    The class provides the Markov-chain function (as run()).
    """

    def __init__(self, integrator_instance, kinetic_energy_instance, potential_instance, observer_instance,
                 range_of_initial_particle_positions, number_of_equilibration_iterations=5000,
                 number_of_observations=1000, initial_step_size=0.1, max_number_of_integration_steps=10,
                 randomise_number_of_integration_steps=False, step_size_adaptor_is_on=True,
                 use_metropolis_accept_reject=True):
        """
        The constructor of the MarkovChain class.

        Parameters
        ----------
        integrator_instance : Python class instance

        kinetic_energy_instance : Python class instance

        potential_instance : Python class instance

        observer_instance : Python class instance

        range_of_initial_particle_positions : float or list of floats

        number_of_equilibration_iterations : int, optional

        number_of_observations : int, optional

        initial_step_size : float, optional

        max_number_of_integration_steps : int, optional

        randomise_number_of_integration_steps : Boolean, optional

        step_size_adaptor_is_on : Boolean, optional

        use_metropolis_accept_reject : Boolean, optional

        Raises
        ------
        base.exceptions.ValueError
            If the prefactor equals 0.
        """
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
        condition = ((dimensionality_of_particle_space == 1 and
                      ((type(range_of_initial_particle_positions) == list and
                        type(range_of_initial_particle_positions[0]) != float) or
                       (type(range_of_initial_particle_positions) == list and
                        len(range_of_initial_particle_positions) > 2))) or
                     (dimensionality_of_particle_space > 1 and
                      (type(range_of_initial_particle_positions) == float or
                       len(range_of_initial_particle_positions) > dimensionality_of_particle_space or
                       (type(range_of_initial_particle_positions[0]) == list and
                        type(range_of_initial_particle_positions[0][0]) != float))))
        if condition:
            raise ValueError("Give initial particle positions that agree with the size of particle space {0}.".format(
                self.__class__.__name__))
        self._integrator = integrator_instance
        self._kinetic_energy = kinetic_energy_instance
        self._potential = potential_instance
        self._observer = observer_instance
        self._range_of_initial_particle_positions = range_of_initial_particle_positions
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._step_size = initial_step_size
        self._max_number_of_integration_steps = max_number_of_integration_steps
        self._randomise_number_of_integration_steps = randomise_number_of_integration_steps
        self._step_size_adaptor_is_on = step_size_adaptor_is_on
        self._use_metropolis_accept_reject = use_metropolis_accept_reject
        if dimensionality_of_particle_space == 1:
            self._dimensionality_of_position_array = number_of_particles
            self._dimensionality_of_sample_array = (self._total_number_of_iterations + 1, number_of_particles)
        else:
            self._dimensionality_of_position_array = (number_of_particles, dimensionality_of_particle_space)
            self._dimensionality_of_sample_array = (
                self._total_number_of_iterations + 1, number_of_particles, dimensionality_of_particle_space)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           integrator_instance=integrator_instance, kinetic_energy_instance=kinetic_energy_instance,
                           potential_instance=potential_instance, observer_instance=observer_instance,
                           randomise_number_of_integration_steps=randomise_number_of_integration_steps,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations, initial_step_size=initial_step_size,
                           max_number_of_integration_steps=max_number_of_integration_steps,
                           range_of_initial_particle_positions=range_of_initial_particle_positions,
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
        number_of_integration_steps = self._max_number_of_integration_steps
        momenta = np.zeros(self._dimensionality_of_position_array)
        positions = self._initialise_position_array()
        sample = np.zeros(self._dimensionality_of_sample_array)
        sample[0, :] = self._observer.get_observation(momenta, positions)

        for i in range(self._total_number_of_iterations):
            momenta = self._kinetic_energy.get_momentum_observation(momenta)
            if self._randomise_number_of_integration_steps:
                number_of_integration_steps = 1 + np.random.randint(self._max_number_of_integration_steps)
            candidate_momenta, candidate_positions = self._integrator.get_candidate_configuration(
                momenta, positions, number_of_integration_steps, self._step_size, charges=None)

            if self._use_metropolis_accept_reject:
                delta_hamiltonian = (self._kinetic_energy.get_value(candidate_momenta) -
                                     self._kinetic_energy.get_value(momenta) +
                                     self._potential.get_value(candidate_positions, charges=charges) -
                                     self._potential.get_value(positions, charges=charges))
                if np.random.uniform(0, 1) < np.exp(- delta_hamiltonian):
                    positions = candidate_positions
                    momenta = candidate_momenta
                    number_of_accepted_trajectories += 1
            else:
                positions = candidate_positions
                momenta = candidate_momenta
            sample[i + 1, :] = self._observer.get_observation(momenta, positions)

            if self._step_size_adaptor_is_on and i < self._number_of_equilibration_iterations and (i + 1) % 100 == 0:
                acceptance_rate = number_of_accepted_trajectories / 100.0
                if acceptance_rate > 0.8:
                    self._step_size *= 1.1
                elif acceptance_rate < 0.6:
                    self._step_size *= 0.9
                number_of_accepted_trajectories = 0

        acceptance_rate = number_of_accepted_trajectories / float(self._number_of_observations)
        print("Max number of integration steps = %d" % self._max_number_of_integration_steps)
        print("Initial numerical step size = %.3f" % initial_step_size)
        print("Final numerical step size = %.3f" % self._step_size)
        print("Metropolis-Hastings acceptance rate = %f" % acceptance_rate)
        return sample

    def _initialise_position_array(self):
        if dimensionality_of_particle_space == 1:
            if type(self._range_of_initial_particle_positions) == float:
                return self._range_of_initial_particle_positions * np.ones(number_of_particles)
            else:
                return np.random.uniform(*self._range_of_initial_particle_positions, size=number_of_particles)
        else:
            # todo ValueError for soft-matter models w/all particles at same initial position
            if type(self._range_of_initial_particle_positions[0]) == float:
                return np.array([self._range_of_initial_particle_positions for _ in range(number_of_particles)])
            else:
                return np.array([[np.random.uniform(*axis_range) for axis_range in
                                  self._range_of_initial_particle_positions] for _ in range(number_of_particles)])
