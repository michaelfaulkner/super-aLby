"""Module for the NoFactorField class."""
from .factor_field import FactorField


class NoFactorField(FactorField):
    """
    Class for empty factor field.

    """

    def __init__(self, prefactor: float = 1.0):
        """
        The constructor of the NoFactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor)

    def get_next_event(self, positions, active_particle_index, temperature, movement_direction):
        """
        Returns the distance to the next particle event (in ECMC) and the index of the particle that triggers the event.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        temperature : float
            The sampling temperature.
        movement_direction : int
            The active-particle direction of motion.

        Returns
        ----------
        distance_to_next_event : float
            The distance to the next particle event
        veto_index : int
            The particle index responsible for the event.
        """
        return float('inf'), None

    def choose_next_active_particle(self, positions, active_particle_index, temperature, movement_direction,
                                    veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
        temperature : float
            The sampling temperature.
        movement_direction : int
            The active-particle direction of motion.
        veto_index : int
            The particle index responsible for the event.

        Returns
        -------
        active_particle_index: int
            The index of the next active particle.
        movement_direction : int
            The next active-particle direction of motion.
        """
        return
