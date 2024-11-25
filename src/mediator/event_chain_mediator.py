"""Module for EventChainMediator class"""
import logging
import numpy as np
import random
from base.exceptions import ConfigurationError
from helper_methods import get_temperatures
from potential.potential import Potential
from run import get_ordinal
from sampler.sampler import Sampler
from typing import Sequence
from noise_distribution.noise_distribution import NoiseDistribution
from base.logging import log_init_arguments
                        #NOTE this might not work?
from model_settings import number_of_particles, distance_between_measurements, speed_of_chain

class EventChainMediator():
    """
    Class for event chain Monte Carlo algorithms

    #NOTE Large amount of duplicated code between this and Mediator/DiffusiveMediator etc. classes
    #TODO re-structure code to avoid this 
    """

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], noise_distribution: NoiseDistribution,
                minimum_temperature: float = 1.0, maximum_temperature: float = 1.0, number_of_temperature_increments: int = 0,
                number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                proposal_dynamics_adaptor_is_on: bool = True, **kwargs):
        r"""
        Constructor of the EventChainMediator class

        #NOTE not currently designed for cooperative inheritance as in diffusive cases. 
        #TODO generate structure similar to parent-child-child classes of mediator-diffusive-metropolis etc.


        Parameters
        ----------
        potential : potential.potential.Potential
            Instance of the chosen child class of potential.potential.Potential.
        samplers : Sequence[sampler.sampler.Sampler]
            Sequence of instances of the chosen child classes of sampler.sampler.Sampler.
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
            When True, the size of either the numerical integration step (DeterministicMediator) or the width of the
            proposal distribution (MetropolisMediator) is tuned during the equilibration process.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

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
        """

        super().__init__(**kwargs)
        if not isinstance(potential, Potential):
            raise ConfigurationError(f"Give a potential class as the value for potential in {self.__class__.__name__}.")
        for sampler in samplers:
            if not isinstance(sampler, Sampler):
                raise ConfigurationError(f"Give a list of sampler classes as the value for samplers in "
                                         f"{self.__class__.__name__}.")
        if minimum_temperature < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as minimum_temperature in "
                                     f"{self.__class__.__name__}.")
        if maximum_temperature < 0.0:
            raise ConfigurationError(f"Give a value not less than 0.0 as maximum_temperature in "
                                     f"{self.__class__.__name__}.")
        if maximum_temperature < minimum_temperature:
            raise ConfigurationError(f"Give values of minimum_temperature and maximum_temperature in "
                                     f"{self.__class__.__name__} such that the value of maximum_temperature is not "
                                     f"less than the value of minimum_temperature.")
        if number_of_temperature_increments < 0:
            raise ConfigurationError(f"Give a value not less than 0 as number_of_temperature_increments in "
                                     f"{self.__class__.__name__}.")
        if number_of_temperature_increments == 0 and minimum_temperature != maximum_temperature:
            raise ConfigurationError(f"As the value of number_of_temperature_increments is equal to 0, give equal "
                                     f"values of minimum_temperature and maximum_temperature in "
                                     f"{self.__class__.__name__}.")
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
        self._samplers = samplers
        self._temperatures = get_temperatures(minimum_temperature, maximum_temperature,
                                              number_of_temperature_increments)
        self._number_of_equilibration_iterations = number_of_equilibration_iterations
        self._number_of_observations = number_of_observations
        self._number_of_observations_between_screen_prints_for_clock = int(number_of_observations / 10)
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._proposal_dynamics_adaptor_is_on = proposal_dynamics_adaptor_is_on
        """The following objects are set in self._reset_arrays_and_counters()"""
        self._positions = None
        self._samples = None
        self._number_of_accepted_trajectories = None
        self._noise_distribution = noise_distribution
        self._dimensionless_positions = self._potential.get_dimensionless_position(self._positions)
        self._dimensionless_omega = self._potential._dimensionless_omega #NOTE need to rename to be public
        self._dimensionless_mass = self._potential._dimensionless_m

        if not isinstance(noise_distribution, NoiseDistribution):
            raise ConfigurationError(f"Give a noise_distribution class as the value for noise_distribution in "
                                     f"{self.__class__.__name__}.")
        
        self._noise_distribution = noise_distribution
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential, samplers=samplers, noise_distribution=noise_distribution,
                           minimum_temperature=minimum_temperature, maximum_temperature=maximum_temperature,
                           number_of_temperature_increments=number_of_temperature_increments,
                           number_of_equilibration_iterations=number_of_equilibration_iterations,
                           number_of_observations=number_of_observations,
                           proposal_dynamics_adaptor_is_on=proposal_dynamics_adaptor_is_on)



    def generate_sample(self):
        """Runs the Markov chain in order to generate the sample."""
        for temperature_index, temperature in enumerate(self._temperatures):
            self._print_temperature_message(temperature, temperature_index)
            self._reset_arrays_and_counters(temperature)
            
            # pick randomly positive or negative
            if np.random.uniform(0.0, 1.0) < 0.5:
                movement_direction = -1
            else:
                movement_direction = 1
            active_particle_index = None
            for markov_chain_index in range(self._total_number_of_iterations):
                active_particle_index, movement_direction = self._generate_single_observation(markov_chain_index, temperature, movement_direction, active_particle_index)

                if (markov_chain_index + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                    current_sample_size = markov_chain_index + 1
                    print(f"{current_sample_size} observations drawn out of a total of "
                          f"{self._total_number_of_iterations} (including {self._number_of_equilibration_iterations} "
                          f"equilibration observations).")
            [sampler.output_sample(self._samples[sampler_index], temperature_index) for sampler_index, sampler in
             enumerate(self._samplers)]
            self._print_markov_chain_summary()






    def _print_temperature_message(self, temperature, temperature_index):
        """Prints details of the current sampling temperature before each temperature iteration."""

    
    def _generate_single_observation(self, markov_chain_index, temperature, movement_direction, active_particle_index = None):
        """Advances the Markov chain to the next sampling instance and adds a single observation to the sample."""

        # read in lambda from config file OR DO THIS IN INIT
        distance_travelled = 0
        if markov_chain_index == 0:
            # randomly select the active particle index
            active_particle_index = np.random.randint(0, number_of_particles)


        while distance_travelled < distance_between_measurements: # i.e. we will always start before we reach lambda
            #NOTE may have to think more about edge cases where this might not effectively catch the sampling moment.
            active_particle_index, movement_direction, distance_travelled = self._generate_next_event(markov_chain_index, distance_travelled,
                                                                                                       active_particle_index, movement_direction)
        
        return active_particle_index, movement_direction
        



    def _generate_next_event(self, markov_chain_step_index, distance_travelled, active_particle_index, movement_direction):
        """Runs the Markov chain until the next event"""
        
        # start by checking max(x_a, (x_a+1 + x_a-1) / (2 + \omega ^2))
        dimensionless_position_a = self._dimensionless_positions[active_particle_index]
        dimensionless_position_a_plus_1 = self._dimensionless_positions[active_particle_index+1]
        dimensionless_position_a_minus_1 = self._dimensionless_positions[active_particle_index-1]
        

        possible_move = ((dimensionless_position_a_plus_1 + dimensionless_position_a_minus_1)
                            / (2 + self._dimensionless_omega**2))

        if dimensionless_position_a < possible_move:
        #   translate site a to (x_a+1 + x_a-1) / (2 + \omega ^2)
            if distance_travelled + possible_move > distance_between_measurements: #TODO might need to make this >= lambda

                allowed_move = distance_between_measurements - distance_travelled
                dimensionless_position_a, distance_travelled = self.update_position(allowed_move, active_particle_index, distance_travelled)


                for sampler_index, sampler in enumerate(self._samplers):
                    self._samples[sampler_index][markov_chain_step_index + 1, :] = sampler.get_observation(
                        None, self._positions, self._potential)
                
            else:
                dimensionless_position_a, distance_travelled = self.update_position(possible_move, active_particle_index, distance_travelled)

        # calculate eta (i.e. the distance that this markov chain would take us)
        initial_action = (self._potential.get_action_at_index(dimensionless_position_a_minus_1, dimensionless_position_a)
                          + self._potential.get_action_at_index(dimensionless_position_a, dimensionless_position_a_plus_1))
                        #NOTE this might give errors due to pass by copy/reference?? check
        random_value = np.random.uniform(0.0, 1.0)
        a = self._dimensionless_mass * (1 + 0.5 * self._dimensionless_omega**2)
        b = -1 * self._dimensionless_mass * (dimensionless_position_a_plus_1 + dimensionless_position_a_minus_1)
        c = (0.5 * self._dimensionless_mass * (dimensionless_position_a_plus_1**2 
                + dimensionless_position_a_minus_1**2 + self._dimensionless_omega**2 * 
                dimensionless_position_a_minus_1**2) - initial_action * np.log(random_value))
        # use quadratic eqn, take +ve root?
        eta = np.roots([c,b,a]) - dimensionless_position_a
        print(eta) #TODO pick one of the roots
        eta = eta[0]

        # our proposed move may take us past lambda, at which point we should measure
        if distance_travelled + eta > distance_between_measurements: #TODO might need to make this >= lambda

            allowed_move = distance_between_measurements - distance_travelled
            dimensionless_position_a, distance_travelled = self.update_position(allowed_move, active_particle_index, distance_travelled)

            for sampler_index, sampler in enumerate(self._samplers):
                self._samples[sampler_index][markov_chain_step_index + 1, :] = sampler.get_observation(
                    None, self._positions, self._potential)
        
        else:
            dimensionless_position_a, distance_travelled = self.update_position(eta, active_particle_index, distance_travelled)

            # choose our new active particle
            # and what direction it will move in

        return active_particle_index, movement_direction, distance_travelled



    def update_position(self, move, active_particle_index, distance_travelled):
        """ Updates position and distance travelled for the active particle"""
        self._dimensionless_positions[active_particle_index] += move
        dimensionless_position_a = self._dimensionless_positions[active_particle_index]
        distance_travelled += move

        return dimensionless_position_a, distance_travelled
    
    def choose_next_active_particle(self, active_particle_index, movement_direction):
        """Chooses the index and direction for the next active particle in the markov chain"""
        rand = np.random.uniform(0.0, 1.0)
        site_a_gradient = self._potential.get_gradient_at_index(self._dimensionless_positions, active_particle_index)
        site_a_minus_1_gradient = self._potential.get_gradient_at_index(self._dimensionless_positions, active_particle_index-1)
        site_a_plus_1_gradient = self._potential.get_gradient_at_index(self._dimensionless_positions, active_particle_index)
        total_action_gradients = site_a_minus_1_gradient + site_a_gradient + site_a_plus_1_gradient

        prob_arr = np.zeros((6,2))
        count = 0
        for v in [movement_direction, movement_direction * -1]:
            for grad in [site_a_minus_1_gradient, site_a_gradient, site_a_plus_1_gradient]:
                probability = np.max([0,-grad * v /total_action_gradients])

                prob_arr[count,0] = probability
                prob_arr[count, 1] = v
                count += 1

        nonzero = np.nonzero(prob_arr[:,0])
        for index, probability in enumerate(prob_arr[nonzero,0]):
            if rand > prob_arr[index-1,0] and rand < probability: #TODO make this account for edge cases
                active_particle_index = # this might not work tbh 
                # think of another way of doing this?


        # check current movement direction
        # ignore the choice that corresponds to conitnuing in the direction and site we were just at
        # 
        if rand < -site_a_minus_1_gradient / total_action_gradients:
            active_particle_index -= 1
        elif (rand == -site_a_minus_1_gradient / total_action_gradients or rand > -site_a_minus_1_gradient / total_action_gradients) and rand < -site_a_gradient / total_action_gradients:
            pass
        elif (rand == -site_a_gradient / total_action_gradients or rand > -site_a_gradient / total_action_gradients):
            active_particle_index += 1





    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step (DeterministicMediator) or the width of the proposal
            distribution (MetropolisMediator)."""
        
    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""









    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        self._positions = self._potential.initialised_position_array()
        self._samples = [sampler.initialise_sample_array(self._total_number_of_iterations) for sampler in
                         self._samplers]
        self._number_of_accepted_trajectories = 0
        for sampler_index, sampler in enumerate(self._samplers):
            self._samples[sampler_index][0, :] = sampler.get_observation(None, self._positions, self._potential)
