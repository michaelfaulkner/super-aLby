"""Module for EventChainMediator class"""
import logging
import numpy as np
from base.exceptions import ConfigurationError
import importlib
from helper_methods import get_temperatures
from potential.potential import Potential
from run import get_ordinal
from sampler.sampler import Sampler
from typing import Sequence
from noise_distribution.noise_distribution import NoiseDistribution
from base.logging import log_init_arguments
                        #NOTE this might not work?
from model_settings import number_of_particles, distance_between_measurements, speed_of_chain
from helper_methods import get_east_neighbour, get_west_neighbour
parsing = importlib.import_module("base.parsing")

class EventChainMediator():
    """
    Class for event chain Monte Carlo algorithms

    #NOTE Large amount of duplicated code between this and Mediator/DiffusiveMediator etc. classes
    #TODO re-structure code to avoid this 
    """

    def __init__(self, potential: Potential, samplers: Sequence[Sampler], noise_distribution: NoiseDistribution,
                minimum_temperature: float = 1.0, maximum_temperature: float = 1.0, number_of_temperature_increments: int = 0,
                number_of_equilibration_iterations: int = 10000, number_of_observations: int = 100000,
                proposal_dynamics_adaptor_is_on: bool = True, distance_between_measurements : float = 1.0, speed_of_chain : float  = 1.0):
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
        self._number_of_observations_between_screen_prints_for_clock = 1 #int(number_of_observations / 10)
        self._total_number_of_iterations = number_of_equilibration_iterations + number_of_observations
        self._proposal_dynamics_adaptor_is_on = proposal_dynamics_adaptor_is_on
        self._timestep = self._potential._timestep 
        self._dimensionless_omega = self._potential._dimensionless_omega #NOTE need to rename to be public
        self._dimensionless_mass = self._potential._dimensionless_m
        self._output_directory = parsing.get_value()
        """The following objects are set in self._reset_arrays_and_counters()"""
        self._positions = None
        self._samples = None
        self._number_of_accepted_trajectories = None
        self._noise_distribution = noise_distribution
        ######################## for testing 
        self._move_num = None
        self._initial_index = None
        self._indices = np.zeros((number_of_observations*100, 7))
        self._n_indices_chosen = 0
        ##########################

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
            if np.random.uniform(0.0, 1.0) < 0.5:
                movement_direction = -1
            else:
                movement_direction = 1
            active_particle_index = None
            for markov_chain_index in range(self._total_number_of_iterations):
                if markov_chain_index == 0:
                    active_particle_index = np.random.randint(0, number_of_particles)
                    self._initial_index = active_particle_index
                    print(f"started at index {active_particle_index}, direction {movement_direction}")
                    # store active particle
                    self._indices[self._n_indices_chosen,0] = active_particle_index
                    self._n_indices_chosen += 1
                active_particle_index, movement_direction = self._generate_single_observation(markov_chain_index, temperature, movement_direction, active_particle_index)
                if (markov_chain_index + 1) % self._number_of_observations_between_screen_prints_for_clock == 0:
                    current_sample_size = markov_chain_index + 1
                    print(f"{current_sample_size} observations drawn out of a total of "
                          f"{self._total_number_of_iterations} (including {self._number_of_equilibration_iterations} "
                          f"equilibration observations).")
            [sampler.output_sample(self._samples[sampler_index], temperature_index) for sampler_index, sampler in
             enumerate(self._samplers)]
            # get most chosen index
            mode_index = np.bincount(self._indices[:self._n_indices_chosen,0].astype(int)).argmax()
            print(f"started at index {self._initial_index}, most visited index was {mode_index}")
            # TODO read output folder from config file
            np.save("output/event_chain_mediator/temperature_00_sample_of_indices.npy", self._indices[:self._n_indices_chosen])
            self._print_markov_chain_summary()
            


    def _print_temperature_message(self, temperature, temperature_index):
        """Prints details of the current sampling temperature before each temperature iteration."""
        if len(self._temperatures) == 1:
                    print("---------------------------------------------")
                    print(f"Temperature = {temperature:.4f} (only temperature value)")
                    print("---------------------------------------------")
        else:
            print("--------------------------------------------------")
            print(f"Temperature = {temperature:.4f} ({get_ordinal(temperature_index + 1)} of {len(self._temperatures)} "
                  f"temperature values)")
            print("--------------------------------------------------")
    

    def _generate_single_observation(self, markov_chain_index, temperature, movement_direction, active_particle_index = None):
        """Advances the Markov chain to the next sampling instance and adds a single observation to the sample."""
        distance_travelled = 0

        while distance_travelled < distance_between_measurements: # i.e. we will always start before we reach lambda
            #NOTE may have to think more about edge cases where this might not effectively catch the sampling moment.
            active_particle_index, movement_direction, distance_travelled = self._generate_next_event(markov_chain_index, distance_travelled,
                                                                                                       active_particle_index, movement_direction)
            #NOTE may need to consider if this always catches cases where we propose a move than exceeds lambda
            # does the simulation continue correctly after this case?
        
        return active_particle_index, movement_direction
        

    def _generate_next_event(self, markov_chain_index, distance_travelled, active_particle_index, movement_direction):
        """Runs the Markov chain until the next event"""
        #TODO implement variable speed_of_chain (how would that work?)
        a_plus_one_index = get_east_neighbour(active_particle_index, number_of_particles)
        a_minus_one_index = get_west_neighbour(active_particle_index, number_of_particles)
        position_a = self._positions[active_particle_index]
        dimensionless_position_a = position_a / self._timestep
        position_a_plus_1 = self._positions[a_plus_one_index]
        dimensionless_position_a_plus_1 = position_a_plus_1  / self._timestep
        position_a_minus_1 = self._positions[a_minus_one_index]
        dimensionless_position_a_minus_1 = position_a_minus_1 / self._timestep
        #############################################################
        # sample some more data for testing
        self._indices[self._n_indices_chosen,3] = dimensionless_position_a
        self._indices[self._n_indices_chosen,4] = dimensionless_position_a_minus_1
        self._indices[self._n_indices_chosen,5] = dimensionless_position_a_plus_1
        self._indices[self._n_indices_chosen,6] = self._potential.get_action_at_index(dimensionless_position_a, dimensionless_position_a_plus_1)
        ##############################################################
        
        proposed_move_dimensionless, self._move_num, self._indices[self._n_indices_chosen,2], distance_travelled_in_move = self._potential.get_distance_to_next_event(dimensionless_position_a,
                                                                    dimensionless_position_a_plus_1,
                                                                    dimensionless_position_a_minus_1,
                                                                    movement_direction, self._move_num)
        proposed_move = proposed_move_dimensionless * self._timestep

        if distance_travelled + distance_travelled_in_move > distance_between_measurements or distance_travelled + distance_travelled_in_move == distance_between_measurements:
            allowed_move = distance_between_measurements - distance_travelled
            distance_travelled = self.update_position(allowed_move, active_particle_index, distance_travelled, movement_direction)
            for sampler_index, sampler in enumerate(self._samplers):
                self._samples[sampler_index][markov_chain_index + 1, :] = sampler.get_observation(
                    None, self._positions, self._potential)
       
        else:
            distance_travelled += distance_travelled_in_move
            self.update_position(proposed_move, active_particle_index, distance_travelled, movement_direction)
            ###########################
            self._indices[self._n_indices_chosen,1] = proposed_move
            ############################
            active_particle_index, movement_direction = self.choose_next_active_particle(active_particle_index,
                                                                                        a_plus_one_index,
                                                                                        a_minus_one_index,
                                                                                        movement_direction)
        
            
        return active_particle_index, movement_direction, distance_travelled


    def update_position(self, move, active_particle_index, distance_travelled, movement_direction):
        """ Updates position and distance travelled for the active particle"""
        self._positions[active_particle_index] += move 
        # NOTE changed to account for distance_travelled in potential.get_distance
        # due to possibility of moving back and forth? not sure if physically possible but
        # should account for all edge cases
        # distance_travelled += np.abs(move)        
        # return distance_travelled
    
    def choose_next_active_particle(self, active_particle_index, active_particle_plus_1_index,
                                    active_particle_minus_1_index, movement_direction):
        """Chooses the index and direction for the next active particle in the markov chain"""
        initial_a = active_particle_index
        initial_v = movement_direction
     
        site_a_gradient = self._potential.get_gradient_at_index(self._positions, active_particle_index)
        site_a_minus_1_gradient = self._potential.get_gradient_at_index(self._positions,
                                                                        active_particle_minus_1_index)
        site_a_plus_1_gradient = self._potential.get_gradient_at_index(self._positions,
                                                                        active_particle_plus_1_index)
        total_action_gradients = np.abs(site_a_minus_1_gradient) + np.abs(site_a_gradient) + np.abs(site_a_plus_1_gradient)
        probabilities = np.zeros(3)

        probabilities[0] = site_a_minus_1_gradient /  total_action_gradients
        probabilities[1] = probabilities[0] + site_a_gradient / total_action_gradients
        probabilities[2] = probabilities[1] + site_a_plus_1_gradient / total_action_gradients

        
        rand = np.random.uniform(0.0, 1.0)

        #NOTE this probably accounts for all cases/number of options, but this should be checked
        if rand < probabilities[0]:
            active_particle_index = active_particle_minus_1_index 
        elif rand < probabilities[1]:
            movement_direction = movement_direction * -1   
        else:
            active_particle_index = active_particle_plus_1_index
        
                
        if active_particle_index == initial_a and movement_direction == initial_v:
            raise Exception("Chose the same index and direction twice in a row")
        
        self._indices[self._n_indices_chosen,0] = active_particle_index
        self._n_indices_chosen += 1
        return active_particle_index, movement_direction




    def _proposal_dynamics_adaptor(self):
        """Tunes the size of either the numerical integration step (DeterministicMediator) or the width of the proposal
            distribution (MetropolisMediator). In EventChainMediator this is a holdover only."""
        pass
        
    def _print_markov_chain_summary(self):
        """Prints a summary of the completed Markov process to the screen."""
        #TODO what should this print? acceptance rate and width of noise distribution not relevant
        pass

    def _reset_arrays_and_counters(self, temperature):
        """Sets or resets the arrays (e.g., the sample array) and counters before each temperature iteration."""
        self._positions = self._potential.initialised_position_array()
        self._dimensionless_positions = self._potential.get_dimensionless_position(self._positions)
        self._samples = [sampler.initialise_sample_array(self._total_number_of_iterations) for sampler in
                         self._samplers]
        self._number_of_accepted_trajectories = 0
        self._move_num = 0
        for sampler_index, sampler in enumerate(self._samplers):
            self._samples[sampler_index][0, :] = sampler.get_observation(None, self._positions, self._potential)
