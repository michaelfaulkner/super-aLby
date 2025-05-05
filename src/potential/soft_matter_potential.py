"""Module for the abstract SoftMatterPotential class."""
from .smooth_potential import SmoothPotential
from base.exceptions import ConfigurationError
from model_settings import dimensionality_of_particle_space, range_of_initial_particle_positions, size_of_particle_space
from abc import ABCMeta
import numpy as np


class SoftMatterPotential(SmoothPotential, metaclass=ABCMeta):
    """
    Abstract class for soft-matter potentials, which are potentials that are smooth functions of particle-separation
        vectors.

    The only additional structure provided by this class (relative to SmoothPotential) are raising exceptions to ensure
        a valid combined choice of size_of_particle_space, range_of_initial_particle_positions and
        dimensionality_of_particle_space.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the SoftMatterPotential class.

        This abstract class verifies that i) type(element) is np.float64 for each element of size_of_particle_space,
        and ii) the initial particle positions do not all coincide, as this would generate divergences. No other
        additional functionality is provided.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If type(element) is not np.float64 for element in size_of_particle_space.
        base.exceptions.ConfigurationError
            If model_settings.range_of_initial_particle_positions does not give a real-valued interval for each
            component of the initial positions of each particle.
        """
        super().__init__(prefactor, **kwargs)
        if dimensionality_of_particle_space == 1:
            if not type(size_of_particle_space) is np.float64:
                raise ConfigurationError(
                    f"Give a float (representing the volume of the one-dimensional particle space) for the value of "
                    f"size_of_particle_space in the ModelSettings section when using {self.__class__.__name__} (or any "
                    f"child class of SoftMatterPotential) with a one-dimensional particle space.")
        else:
            if not (type(size_of_particle_space) is np.ndarray and
                    dimensionality_of_particle_space == len(size_of_particle_space) and
                    [type(component) is np.float64 for component in size_of_particle_space]):
                raise ConfigurationError(
                    f"Give a list of dimensionality_of_particle_space floats (each representing the length of the "
                    f"corresponding Cartesian dimension of the dimensionality_of_particle_space-dimensional particle "
                    f"space) for the value of size_of_particle_space in the ModelSettings section when using "
                    f"{self.__class__.__name__} (or any child class of SoftMatterPotential) with a particle space of "
                    f"dimension dimensionality_of_particle_space.")
        if dimensionality_of_particle_space == 1:
            if not (type(range_of_initial_particle_positions) is list and
                    len(range_of_initial_particle_positions) == 2 and
                    [type(bound) is float for bound in range_of_initial_particle_positions]):
                raise ConfigurationError(
                    f"Give a list of two floats (representing the bounds of the interval from which each particle "
                    f"position is chosen) for the value of range_of_initial_particle_positions in the ModelSettings "
                    f"section when using {self.__class__.__name__} (or any child class of SoftMatterPotential) with a "
                    f"one-dimensional particle space.")
        else:
            if not (type(range_of_initial_particle_positions) is list and
                    (len(range_of_initial_particle_positions) == dimensionality_of_particle_space and
                     [type(component) is list and len(component) == 2 and type(bound) is float
                      for component in range_of_initial_particle_positions for bound in component])):
                raise ConfigurationError(
                    f"Give a list of dimensionality_of_particle_space lists of two floats for the value of "
                    f"range_of_initial_particle_positions in the ModelSettings section when using "
                    f"{self.__class__.__name__} (or any child class of SoftMatterPotential) with a particle space of "
                    f"dimension dimensionality_of_particle_space.  Each element of the list corresponds to a Cartesian "
                    f"component of each particle position and each sub-list represents the bounds of the interval from "
                    f"which the corresponding initial Cartesian component is randomly chosen.")
