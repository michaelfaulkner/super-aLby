"""Module for the abstract FactorField class."""
from base.exceptions import ConfigurationError
from abc import ABCMeta, abstractmethod


class FactorField(metaclass=ABCMeta):
    """
    Abstract class for factor fields.

    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the FactorField class.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If prefactor is not greater than 0.0.
        """
        if prefactor == 0.0:
            raise ConfigurationError(f"Give a value greater than 0.0 as prefactor for {self.__class__.__name__}.")
        self._prefactor = prefactor

    @abstractmethod
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
        raise NotImplementedError
