"""Module for the abstract KineticEnergy class."""
from abc import ABCMeta, abstractmethod


class KineticEnergy(metaclass=ABCMeta):
    """
    Abstract class for kinetic energies used in the algorithm code.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
        observation of the momenta.
    """

    def __init__(self, **kwargs):
        """
        The constructor of the KineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def get_value(self, momenta):
        """
        Returns the kinetic energy for the given particle momenta.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.

        Returns
        -------
        float
            The kinetic energy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy for the given particle momenta.

        Parameters
        ----------
        momenta : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the momentum of a single particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the kinetic energy of a single
            particle.
        """
        raise NotImplementedError

    @abstractmethod
    def get_momentum_observations(self, temperature):
        """
        Returns an observation of the momenta from the kinetic-energy distribution.

        Parameters
        ----------
        temperature : float
            The sampling temperature.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the newly observed momentum of a single particle.
        """
        raise NotImplementedError
