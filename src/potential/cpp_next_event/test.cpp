#include<iostream>
#include <fstream> 
#include<stdio.h>
#include<vector>
#include <algorithm>
#include <complex>
#include <math.h>
#include <random>


struct next_event{
  double shortest_distance_to_next_event;
  int vetoing_index;
};

struct next_event get_next_kinetic_event(int active_particle_index, int east_neighbour_index, int west_neighbour_index, int number_of_quantum_particles, int number_of_timeslices, 
    double active_particle_position, double east_neighbour_position, double west_neighbour_position, double mass,
    double timestep, int movement_direction){

    struct next_event proposed_kinetic_event;

    proposed_kinetic_event.shortest_distance_to_next_event = 1.0e10;
    proposed_kinetic_event.vetoing_index = number_of_quantum_particles * number_of_timeslices; // should error if returned

    double initial_position = active_particle_position;

    for(int neighbour = 0; neighbour < 2; neighbour++){
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 generator(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dist(1.0, 2.0);
        double uphill_energy = -log(dist(generator));
        double neighbour_position;
        int possible_veto;
        if(neighbour == 0){
            neighbour_position = west_neighbour_position;
            possible_veto = west_neighbour_index;
        }
        else{
            neighbour_position = east_neighbour_position;
            possible_veto = east_neighbour_index;
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
        double distance_to_possible_event = abs(final_position - initial_position);

        if(distance_to_possible_event < proposed_kinetic_event.shortest_distance_to_next_event){
            proposed_kinetic_event.shortest_distance_to_next_event = distance_to_possible_event;
            proposed_kinetic_event.vetoing_index = possible_veto;
        }
    }
    return proposed_kinetic_event;
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

























int main(){
//     std::vector<double> vect;
//     std::default_random_engine generator;
//     std::uniform_real_distribution<double> dist(0.0, 1.0);
//     for(int i; i < 20; i++){
//         vect.push_back(dist(generator));
//     }

//    std::ofstream outfile("dist.txt");
//    for(int i; i < 20; i++){
//     outfile << vect.at(i) << ",";
//    }

//    outfile.close();

    double x = 2.0;

    std::cout << x << -x << std::endl;


    return 0;
}