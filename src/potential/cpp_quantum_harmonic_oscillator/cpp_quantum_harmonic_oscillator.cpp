#include <iostream>
#include <random>
#include <cmath>
#include <complex>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <vector>
#include "cpp_quantum_harmonic_oscillator.hpp"
#include <cassert>
#include <pybind11/pybind11.h>
struct next_event{
  double shortest_distance_to_next_event;
  int vetoing_index;
};

struct next_event get_next_kinetic_event(int active_particle_index, int east_neighbour_index, int west_neighbour_index,
    int number_of_quantum_particles, int number_of_timeslices, double active_particle_position,
    double east_neighbour_position, double west_neighbour_position, double mass, double timestep,
    int movement_direction, double uphill_energy_west, double uphill_energy_east){

    struct next_event proposed_kinetic_event;

    proposed_kinetic_event.shortest_distance_to_next_event = 1.0e10;
    proposed_kinetic_event.vetoing_index = number_of_quantum_particles * number_of_timeslices; // should error if returned

    double initial_position = active_particle_position;

    for(int neighbour = 0; neighbour < 2; neighbour++){
        double uphill_energy;
        double neighbour_position;
        int possible_veto;
        if(neighbour == 0){
            neighbour_position = west_neighbour_position;
            possible_veto = west_neighbour_index;
            uphill_energy = uphill_energy_west;
        }
        else{
            neighbour_position = east_neighbour_position;
            possible_veto = east_neighbour_index;
            uphill_energy = uphill_energy_east;
        }

        double bottom_of_well = neighbour_position;
        double intermediate_position;
        if((movement_direction > 0 && initial_position < bottom_of_well)||
        (movement_direction < 0 && initial_position > bottom_of_well)){
            intermediate_position = bottom_of_well;
        }
        else{
            intermediate_position = initial_position;
        }
        double diff = intermediate_position - neighbour_position;
        double initial_action = 0.5 * (mass /  timestep) * pow(diff, 2);
        double final_action = uphill_energy + initial_action;

        double a = 0.5 * mass / timestep;
        double b = -mass / timestep * neighbour_position;
        double c = a * pow(neighbour_position, 2) - final_action;

        std::vector<double> roots = get_real_quadratic_roots(a, b, c);
        double final_position = get_final_position_of_single_well_event(movement_direction, roots);
        double distance_to_possible_event = std::abs(final_position - initial_position);

        if(distance_to_possible_event < proposed_kinetic_event.shortest_distance_to_next_event){
            proposed_kinetic_event.shortest_distance_to_next_event = distance_to_possible_event;
            proposed_kinetic_event.vetoing_index = possible_veto;
        }
    }
    return proposed_kinetic_event;
}
struct next_event get_next_event(int active_particle_index, int east_neighbour_index, int west_neighbour_index,
    int number_of_quantum_particles, int number_of_timeslices, double active_particle_position,
    double east_neighbour_position, double west_neighbour_position, double mass, double timestep,
    int movement_direction, double anharmonicity, double omega_squared, double magnitude_of_double_well_position,
    double uphill_energy_west, double uphill_energy_east, double uphill_energy_potential, double x_shift){
    


    struct next_event shortest_event = get_next_kinetic_event(active_particle_index, east_neighbour_index,
        west_neighbour_index, number_of_quantum_particles, number_of_timeslices, active_particle_position,
        east_neighbour_position, west_neighbour_position, mass, timestep, movement_direction, uphill_energy_west,
        uphill_energy_east);  

    double initial_position = active_particle_position;
    double uphill_energy = uphill_energy_potential;
    double epsilon = 1e-10;
    double bottom_of_well;

    if(std::abs(anharmonicity) <= epsilon || std::abs(omega_squared) <= epsilon){
        bottom_of_well = x_shift;
    }
    else{
        if(std::abs(initial_position) >= epsilon){
            if(initial_position < 0.0){
                bottom_of_well = -magnitude_of_double_well_position;
            }
            else{
                bottom_of_well = magnitude_of_double_well_position;
            }
        }
        else{
            if(movement_direction > 0){
                bottom_of_well = magnitude_of_double_well_position;
            }
            else{
                bottom_of_well = -magnitude_of_double_well_position;
            }
        }
    }

    double intermediate_position;
    if ((movement_direction > 0  && initial_position < bottom_of_well)
        || (movement_direction < 0 && initial_position > bottom_of_well)){
        intermediate_position = bottom_of_well;
    }
    else{
        intermediate_position = initial_position;
    }

    double initial_action = 0.5 * mass * timestep * omega_squared * pow((intermediate_position - x_shift), 2)
                            + timestep * anharmonicity * pow(intermediate_position, 4);
    double final_action = uphill_energy + initial_action;

    std::vector<double> real_roots_arr;
    std::vector<std::complex<double>> roots_arr; 
    if(std::abs(anharmonicity) <= epsilon){ 
        real_roots_arr = get_harmonic_potential_roots(mass, timestep, omega_squared, final_action, x_shift);
    }
    else{
        roots_arr = get_anharmonic_potential_roots(mass, timestep, anharmonicity, omega_squared, final_action);
    }
    double final_position;
    if(anharmonicity > 0.0 && 0.0 > omega_squared){
        bool real_roots = check_all_roots_real(roots_arr);
        if(real_roots){
            real_roots_arr = get_real_elements(roots_arr);
            final_position = get_final_position_of_non_tunnel_event(intermediate_position, movement_direction,
                            real_roots_arr);
        }
        else{
            if((movement_direction > 0 && intermediate_position > 0.0) ||
                (movement_direction < 0 && intermediate_position < 0.0)){
                real_roots_arr = get_real_elements(roots_arr);
                final_position = get_final_position_of_single_well_event(movement_direction, real_roots_arr);
                }
            else{
                double remaining_barrier_height = get_barrier_height(intermediate_position, timestep, mass,
                                                    anharmonicity, omega_squared);
                bottom_of_well *= -1;
                intermediate_position = bottom_of_well;
                final_action -= remaining_barrier_height;
                roots_arr = get_anharmonic_potential_roots(mass, timestep, anharmonicity, omega_squared, final_action);
                bool real_roots = check_all_roots_real(roots_arr);
            
                if(real_roots){
                    std::vector<double> real_roots_arr = get_real_elements(roots_arr);
                    final_position = get_final_position_of_non_tunnel_event(intermediate_position, movement_direction,
                                        real_roots_arr);
                }
                else{
                    std::vector<double> real_roots_arr = get_real_elements(roots_arr);
                    final_position = get_final_position_of_single_well_event(movement_direction, real_roots_arr);
                }
            }
        }
    }
    else{
        final_position = get_final_position_of_single_well_event(movement_direction, real_roots_arr);
    }
    double distance_to_next_factor_event = std::abs(final_position - initial_position);

    if(distance_to_next_factor_event < shortest_event.shortest_distance_to_next_event){
        shortest_event.shortest_distance_to_next_event = distance_to_next_factor_event;
        shortest_event.vetoing_index = active_particle_index;
    }

    return shortest_event;
}

double get_barrier_height(double position, double timestep, double mass, double anharmonicity, double omega_squared){
  return std::abs(-(timestep * anharmonicity * pow(position, 4) +
                    0.5 * mass * timestep * omega_squared * pow(position, 2)));
}

std::vector<double> get_real_elements(std::vector<std::complex<double>> complex_arr){

    double epsilon = 1e-10;
    std::vector<double> real_arr;
    for(long unsigned int i = 0; i < complex_arr.size(); i++){
        std::complex<double> ith_root = complex_arr.at(i);
        if(std::abs(ith_root.imag()) <= epsilon){
            real_arr.push_back(ith_root.real());
        }
    }

    return real_arr;
}

double get_final_position_of_non_tunnel_event(double position, int movement_direction, std::vector<double> roots){

    std::sort(roots.begin(), roots.end());

    if(position < 0.0){
        if(movement_direction > 0){
            return roots.at(1);
        }
        else{
            return roots.at(0);
        }
    }
    else{
        if(movement_direction > 0){
            return roots.at(3);
        }
        else{
            return roots.at(2);
        }
    }
}


bool check_all_roots_real(std::vector<std::complex<double>> roots){
    bool real_roots = true;
    for(int i = 0; i < 4; i++){
        std::complex<double> ith_root = roots.at(i);
        if(ith_root.imag() != 0.0){
            real_roots = false;
        }
    }
    return real_roots;
}

std::vector<double> get_harmonic_potential_roots(double mass, double timestep, double omega_squared, double final_action, double x_shift){
  
    std::vector<double> roots; 
    double a = 0.5 * mass * timestep * omega_squared;
    double b = - mass * omega_squared * timestep * x_shift;
    double c = 0.5 * mass * omega_squared * timestep * pow(x_shift, 2) - final_action;

    roots = get_real_quadratic_roots(a, b, c);
    return roots;
}

std::vector<std::complex<double>> get_anharmonic_potential_roots(double mass, double timestep, double anharmonicity,
                      double omega_squared, double final_action){
    std::vector<std::complex<double>> roots_U; 
    std::vector<std::complex<double>> roots; 
    double a = timestep * anharmonicity;
    double b = 0.5 * mass * timestep * omega_squared;
    double c = -final_action;

    roots_U = get_quadratic_roots(a, b, c);

    roots.push_back(sqrt(roots_U.at(0))); 
    roots.push_back(-sqrt(roots_U.at(0)));
    roots.push_back(sqrt(roots_U.at(1))); 
    roots.push_back(-sqrt(roots_U.at(1)));

    return roots;
}

std::vector<std::complex<double>> get_quadratic_roots(double a, double b, double c){
    // complex type because we could sqrt a -ve number here
    std::vector<std::complex<double>> roots; 
    std::complex<double> discriminant = pow(b, 2) - 4.0 * a * c;
    roots.push_back((-b + std::sqrt(discriminant)) / (2.0 * a));
    roots.push_back((-b - std::sqrt(discriminant)) / (2.0 * a));
    return roots;
}


std::vector<double> get_real_quadratic_roots(double a, double b, double c){
    std::vector<double> roots;

    double discriminant = pow(b, 2) - 4.0 * a * c;
    roots.push_back((-b + std::sqrt(discriminant)) / (2.0 * a));
    roots.push_back((-b - std::sqrt(discriminant)) / (2.0 * a));
    return roots;
}


double get_final_position_of_single_well_event(int movement_direction, std::vector<double> roots){
    assert(roots.size() == 2);

    std::sort(roots.begin(), roots.end());
    double return_root;
    if(movement_direction > 0){
       return_root = roots.at(1);
    }
    else{
        return_root = roots.at(0);
    }
    return return_root;
}

double get_potential_difference(double active_particle_position, double west_neighbour_position,
    double east_neighbour_position, double mass, double timestep,
    double omega_squared, double anharmonicity, double candidate_position, double x_shift){

    double current_action = get_pairwise_action(west_neighbour_position, active_particle_position, mass, timestep,
        omega_squared, anharmonicity, x_shift) +
        get_pairwise_action(active_particle_position, east_neighbour_position, mass, timestep,
        omega_squared, anharmonicity, x_shift);

    double candidate_action = get_pairwise_action(west_neighbour_position, candidate_position, mass, timestep,
        omega_squared, anharmonicity, x_shift) +
        get_pairwise_action(candidate_position, east_neighbour_position, mass, timestep,
        omega_squared, anharmonicity, x_shift);
    
    return candidate_action - current_action;
}


double get_pairwise_action(double active_particle_position, double neighbour_position, double mass,
        double timestep, double omega_squared, double anharmonicity, double x_shift){
    
    return get_kinetic_action_term(active_particle_position, neighbour_position, mass, timestep) +
        get_potential_action_term(active_particle_position, mass, timestep, omega_squared, anharmonicity, x_shift);
}

double get_kinetic_action_term(double active_particle_position, double neighbour_position, double mass,
                        double timestep){
    double diff = neighbour_position - active_particle_position;
    return 0.5 * mass / timestep * pow(diff, 2);
}

double get_potential_action_term(double active_particle_position, double mass, double timestep, double omega_squared,
    double anharmonicity, double x_shift){
    
        return 0.5 * mass * timestep * omega_squared * pow((active_particle_position - x_shift), 2) +
            anharmonicity * timestep * pow(active_particle_position, 4);
}


int main() {
    
    return 0;
} 