"""Module for the abstract ZigZagKineticEnergy class."""
from .kinetic_energy import KineticEnergy
from base.exceptions import ConfigurationError
from model_settings import beta, dimensionality_of_momenta_array, dimensionality_of_particle_space
from abc import ABCMeta, abstractmethod
import numpy as np


class ZigZagKineticEnergy(KineticEnergy, metaclass=ABCMeta):
    """
    Abstract class for kinetic energies that use multiple one-dimensional zig-zag algorithms to draw observations from
    its probability distribution.

    A general kinetic-energy class provides the function itself, its gradient, and the method for drawing a new
    observation of the momenta.
    """

    def __init__(self, zig_zag_observation_parameter: float = 5.0, **kwargs):
        """
        The constructor of the ZigZagKineticEnergy class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        zig_zag_observation_parameter : float
            The normalised distance travelled through one-component momentum space (during the zig-zag algorithm)
            between observations of the one-component momentum distribution. zig_zag_observation_parameter / beta is the
            (non-normalised) distance travelled between observations.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If zig_zag_observation_rate is less than 5.0.
        """
        if zig_zag_observation_parameter < 5.0:
            raise ConfigurationError(f"Give a not less than 5.0 as zig_zag_observation_parameter for "
                                     f"{self.__class__.__name__}.")
        self._distance_between_zig_zag_observations = zig_zag_observation_parameter / beta
        self._stored_momenta = 1.0e-3 * np.random.choice((-1.0, 1.0), dimensionality_of_momenta_array)
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
        [self._get_single_momentum_observation(index) for index, _ in np.ndenumerate(self._stored_momenta)]
        return self._stored_momenta

    def _get_single_momentum_observation(self, index):
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
        index : float
            Index of a single component of the stored momentum.
        """
        distance_left_before_observation = self._distance_between_zig_zag_observations
        momentum_sign = np.sign(self._stored_momenta[index])
        if distance_left_before_observation < abs(self._stored_momenta[index]):
            self._stored_momenta[index] -= momentum_sign * distance_left_before_observation
            return
        momentum_sign *= - 1.0
        while True:
            distance_to_next_event = self._get_distance_through_uphill_region()
            if distance_left_before_observation < distance_to_next_event:
                self._stored_momenta[index] = momentum_sign * distance_left_before_observation
                break
            elif distance_left_before_observation < 2.0 * distance_to_next_event:
                self._stored_momenta[index] = momentum_sign * (distance_to_next_event -
                                                               distance_left_before_observation)
                break
            distance_left_before_observation -= 2.0 * distance_to_next_event
            momentum_sign *= - 1.0

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
