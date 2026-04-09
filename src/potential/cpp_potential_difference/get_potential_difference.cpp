#include<iostream>
#include <cmath>
#include "get_potential_difference.hpp"
#include <pybind11/pybind11.h>

double get_potential_difference(double active_particle_position, double west_neighbour_position,
    double east_neighbour_position, double mass, double timestep,
    double omega_squared, double anharmonicity, double candidate_position){

    double current_action = get_pairwise_action(west_neighbour_position, active_particle_position, mass, timestep,
        omega_squared, anharmonicity) +
        get_pairwise_action(active_particle_position, east_neighbour_position, mass, timestep,
        omega_squared, anharmonicity);

    double candidate_action = get_pairwise_action(west_neighbour_position, candidate_position, mass, timestep,
        omega_squared, anharmonicity) +
        get_pairwise_action(candidate_position, east_neighbour_position, mass, timestep,
        omega_squared, anharmonicity);
    
    return candidate_action - current_action;
}


double get_pairwise_action(double active_particle_position, double neighbour_position, double mass,
        double timestep, double omega_squared, double anharmonicity){
    
    return get_kinetic_action_term(active_particle_position, neighbour_position, mass, timestep) +
        get_potential_action_term(active_particle_position, mass, timestep, omega_squared, anharmonicity);
}

double get_kinetic_action_term(double active_particle_position, double neighbour_position, double mass,
                        double timestep){
    double diff = neighbour_position - active_particle_position;
    return 0.5 * mass / timestep * pow(diff, 2);
}

double get_potential_action_term(double active_particle_position, double mass, double timestep, double omega_squared,
    double anharmonicity){
    
        return 0.5 * mass * timestep * omega_squared * pow(active_particle_position, 2) +
            anharmonicity * timestep * pow(active_particle_position, 4);
}

int main(){
    return 0;
}