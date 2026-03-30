#include <vector>
#include <complex>
#include <array>

// function declarations
double get_barrier_height(double position, double timestep, double mass, double anharmonicity, double omega_squared);

std::vector<double> get_real_elements(std::vector<std::complex<double>> complex_arr);

double get_final_position_of_non_tunnel_event(double position, int movement_direction, std::vector<double> roots);

bool check_all_roots_real(std::vector<std::complex<double>> roots);

std::vector<double>  get_harmonic_potential_roots(double mass, double timestep, double omega_squared,
                                                                    double final_action);

std::vector<std::complex<double>> get_anharmonic_potential_roots(double mass, double timestep, double anharmonicity,
                                                                        double omega_squared, double final_action);

std::array<std::complex<double>, 2> get_quadratic_roots(double a, double b, double c);

std::array<double, 2> get_real_quadratic_roots(double a, double b, double c);

double get_final_position_of_single_well_event(int movement_direction, std::array<double, 2> roots);

double get_final_position_of_single_well_event(int movement_direction, std::vector<double> roots);

int get_east_worldline_neighbour(int lattice_site_index, int number_of_quantum_particles, int number_of_timeslices);

int get_west_worldline_neighbour(int lattice_site_index, int number_of_quantum_particles, int number_of_timeslices);
// end function declarations

