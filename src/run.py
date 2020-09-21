import markov_chain
import integrator.leapfrog_integrator
import kinetic_energy.gaussian_kinetic_energy
import potential.exponential_power_potential
import numpy as np

potential_instance = potential.exponential_power_potential.ExponentialPowerPotential()
kinetic_energy_instance = kinetic_energy.gaussian_kinetic_energy.GaussianKineticEnergy()
integrator_instance = integrator.leapfrog_integrator.LeapfrogIntegrator(kinetic_energy_instance, potential_instance)
markov_chain_instance = markov_chain.MarkovChain(integrator_instance, kinetic_energy_instance, potential_instance,
                                                 initial_step_size=1.0, max_number_of_integration_steps=10,
                                                 number_of_equilibration_iterations=100, number_of_observations=1100,
                                                 step_size_adaptor_is_on=True, use_metropolis_accept_reject=True,
                                                 randomise_number_of_integration_steps=False)
support_variable = np.zeros(10)
momentum_sample, support_variable_sample, adapted_step_size, acceptance_rate, number_of_numerical_divergences = (
    markov_chain_instance.run(support_variable, charges=None))
