import markov_chain
import integrator.leapfrog_integrator
import kinetic_energy.gaussian_kinetic_energy
import potential.exponential_power_potential

integrator_instance = integrator.leapfrog_integrator.LeapfrogIntegrator()
potential_instance = potential.exponential_power_potential.ExponentialPowerPotential()
kinetic_energy_instance = kinetic_energy.gaussian_kinetic_energy.GaussianKineticEnergy
markov_chain_instance = markov_chain.MarkovChain(integrator_instance, potential_instance, kinetic_energy_instance,
                                                 initial_step_size=1, max_number_of_integration_steps=10,
                                                 number_of_equilibration_iterations=100, number_of_observations=1100,
                                                 use_metropolis_accept_reject=True,
                                                 randomise_number_of_integration_steps=False)
