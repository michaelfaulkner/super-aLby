#include<iostream>
#include<stdio.h>
#include<vector>
#include <algorithm>

int get_east_worldline_neighbour(int lattice_site_index, int number_of_quantum_particles, int number_of_timeslices){
    return (lattice_site_index + number_of_quantum_particles) %
              (number_of_timeslices * number_of_quantum_particles);
}

int get_west_worldline_neighbour(int lattice_site_index, int number_of_quantum_particles, int number_of_timeslices){
    int mod = (lattice_site_index - number_of_quantum_particles) %
            (number_of_timeslices * number_of_quantum_particles);
    if(mod < 0){
        return number_of_timeslices * number_of_quantum_particles + mod;
    }
    else{
        return mod;
    }
}

int main(){
    int n_e = get_east_worldline_neighbour(1, 1, 50);
    int n_w = get_west_worldline_neighbour(1, 1, 50);

    int n_e1 = get_east_worldline_neighbour(49, 1, 50);
    int n_w1 = get_west_worldline_neighbour(0, 1, 50);
    //n_w1 = get_west_worldline_neighbour(-1, 1, 50);
    int a = 0 - 1;
    int b = 50 * 1;
    std::cout << ((a % b) + b) % b << std::endl;
     a = 5 - 1;
    std::cout << ((a % b) + b) % b << std::endl;

    // std::cout << -1%50 << std::endl;
    // std::cout << n_e << std::endl;
    // std::cout << n_w << std::endl;

    // std::cout << n_e1 << std::endl;
    // std::cout << n_w1 << std::endl;

    return 0;
}