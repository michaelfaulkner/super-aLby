#include <vector>
#include <complex>


struct next_event get_next_kinetic_event(int active_particle_index, int east_neighbour_index, int west_neighbour_index,
    int number_of_quantum_particles, int number_of_timeslices, double active_particle_position,
    double east_neighbour_position, double west_neighbour_position, double mass, double timestep,
    int movement_direction, double uphill_energy_west, double uphill_energy_east);

struct next_event get_next_event(int active_particle_index,  int east_neighbour_index, int west_neighbour_index,
    int number_of_quantum_particles, int number_of_timeslices, double active_particle_position,
    double east_neighbour_position, double west_neighbour_position, double mass, double timestep,
    int movement_direction, double anharmonicity, double omega_squared, double magnitude_of_double_well_position, 
    double uphill_energy_west, double uphill_energy_east, double uphill_energy_potential);

double get_barrier_height(double position, double timestep, double mass, double anharmonicity, double omega_squared);

double get_barrier_height(double position, double timestep, double mass, double anharmonicity, double omega_squared);

std::vector<double> get_real_elements(std::vector<std::complex<double>> complex_arr);

double get_final_position_of_non_tunnel_event(double position, int movement_direction, std::vector<double> roots);

bool check_all_roots_real(std::vector<std::complex<double>> roots);

std::vector<double> get_harmonic_potential_roots(double mass, double timestep, double omega_squared,
    double final_action);

std::vector<std::complex<double>> get_anharmonic_potential_roots(double mass, double timestep, double anharmonicity,
                      double omega_squared, double final_action);

std::vector<std::complex<double>> get_quadratic_roots(double a, double b, double c);

std::vector<double> get_real_quadratic_roots(double a, double b, double c);

double get_final_position_of_single_well_event(int movement_direction, std::vector<double> roots);

double get_potential_difference(double active_particle_position, double west_neighbour_position,
    double east_neighbour_position, double mass, double timestep,
    double omega_squared, double anharmonicity, double candidate_position);

double get_pairwise_action(double active_particle_position, double neighbour_position, double mass,
    double timestep, double omega_squared, double anharmonicity);

double get_kinetic_action_term(double active_particle_position, double neighbour_position, double mass,
                        double timestep);

double get_potential_action_term(double active_particle_position, double mass, double timestep, double omega_squared,
    double anharmonicity);