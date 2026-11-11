import numpy as np
import sys


def main(mass: float = 1.0, omega_squared: float = 1.0, dt_prime: float = 0.1, number_of_particles: int = 100):

    mass = float(mass)
    omega_squared = float(omega_squared)
    dt_prime = float(dt_prime)
    number_of_particles = int(number_of_particles)

    x2 = 1.0 / (2.0 * mass * np.sqrt(omega_squared) * (1.0 / dt_prime**2) 
                            * np.sqrt(1 + omega_squared / (4.0 * dt_prime**2)))

    R = 1 + np.sqrt(omega_squared) / dt_prime * \
        (np.sqrt(omega_squared) / (2 * dt_prime) - np.sqrt(1 +
                        omega_squared / (4 * dt_prime**2)))



    print(x2 * ((1 + R**number_of_particles) / (1 - R**number_of_particles)))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])