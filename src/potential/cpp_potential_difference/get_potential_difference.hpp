double get_potential_difference(double active_particle_position, double west_neighbour_position,
    double east_neighbour_position, double mass, double timestep,
    double omega_squared, double anharmonicity, double candidate_position);

double get_pairwise_action(double active_particle_position, double neighbour_position, double mass,
    double timestep, double omega_squared, double anharmonicity);

double get_kinetic_action_term(double active_particle_position, double neighbour_position, double mass,
                        double timestep);

double get_potential_action_term(double active_particle_position, double mass, double timestep, double omega_squared,
    double anharmonicity);