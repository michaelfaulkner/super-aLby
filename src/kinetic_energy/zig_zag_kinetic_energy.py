"""Module for the abstract ZigZagKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.exceptions import ConfigurationError
from model_settings import dimensionality_of_momenta_array, dimensionality_of_particle_space
from abc import ABCMeta, abstractmethod
import numpy as np


class ZigZagKineticEnergy(KineticEnergy, metaclass=ABCMeta):
    """
    Abstract class for kinetic energies that use multiple one-dimensional zig-zag algorithms to draw observations from
    its probability distribution.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
    observation of the momenta.
    """

    def __init__(self, zig_zag_observation_parameter: float = 10.0, **kwargs):
        """
        The constructor of the ZigZagKineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        zig_zag_observation_parameter : float
            The distance travelled through one-component momentum space (during the zig-zag algorithm) between
            observations of the one-component momentum distribution.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If zig_zag_observation_rate is less than 0.0.
        """
        if zig_zag_observation_parameter < 0.0:
            raise ConfigurationError(
                "Give a value not less than 0.0 for zig_zag_observation_parameter {0}.".format(self.__class__.__name__))
        self._stored_momenta = 1.0e-3 * np.random.choice((-1.0, 1.0), dimensionality_of_momenta_array)
        self._zig_zag_observation_parameter = zig_zag_observation_parameter
        super().__init__(**kwargs)

    @abstractmethod
    def get_value(self, momenta):
        """
        Returns the kinetic-energy function.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The kinetic-energy function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, momenta):
        """
        Returns the gradient of the kinetic energy.

        Parameters
        ----------
        momenta : numpy.ndarray
            The momenta associated with each positions.

        Returns
        -------
        float
            The derivative.
        """
        raise NotImplementedError

    def get_momentum_observations(self):
        """
        Returns an observation of the momenta from the kinetic-energy distribution using multiple one-dimensional
        zig-zag algorithms.

        Returns
        -------
        numpy.ndarray
            A new momenta associated with each positions.
        """
        if dimensionality_of_particle_space == 1:
            self._stored_momenta = np.array(
                [self._get_single_momentum_observation(momentum) for momentum in self._stored_momenta])
            return self._stored_momenta
        self._stored_momenta = np.array([[self._get_single_momentum_observation(component) for component in momentum]
                                         for momentum in self._stored_momenta])
        return self._stored_momenta

    def _get_single_momentum_observation(self, stored_momentum):
        """
        Returns an observation of a single momentum component from the kinetic-energy distribution using a
        one-dimensional zig-zag algorithm.

        This one-dimensional zig-zag algorithm obtains an observation of a single Cartesian component of the momentum
        of a single particle. Motion is always initialised towards the centre of the space as we found this to converge
        more quickly than either continuing in the direction of motion at the time of the previous observation (which
        was stored in self._stored_momenta) or initialising the motion away from the centre of the space with
        probability 1/2.

        Parameters
        ----------
        stored_momentum : float
            A single component of the stored momentum.

        Returns
        -------
        float
            The observation of a single momentum component.
        """
        distance_left_before_observation = self._zig_zag_observation_parameter
        while True:
            distance_to_next_event = self._get_distance_through_uphill_region() + abs(stored_momentum)
            if distance_left_before_observation < distance_to_next_event:
                return stored_momentum - distance_left_before_observation * np.sign(stored_momentum)
            distance_left_before_observation -= distance_to_next_event
            stored_momentum -= distance_to_next_event * np.sign(stored_momentum)

    @abstractmethod
    def _get_distance_through_uphill_region(self):
        r"""
        Returns the distance $|\eta|$ travelled (before the next zig-zag event) through the uphill part of
        one-dimensional momentum space. This is calculated by inverting

            $ \rand(0.0, 1.0) =
                \exp \left[- \beta * \int_0^{\eta} \left(\frac{\partial K}{\partial p}\right)^+ dp \right] $

        Returns
        -------
        float
            The distance travelled through the uphill part of one-dimensional momentum space.
        """
        raise NotImplementedError
