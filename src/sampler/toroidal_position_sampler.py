"""Module for the ToroidalPositionSampler class."""
from .position_sampler import PositionSampler
from base.logging import log_init_arguments
from base.vectors import get_shortest_vectors_on_torus
import logging


class ToroidalPositionSampler(PositionSampler):
    """
    Class for taking observations of particle positions on the torus, i.e., corrected for periodic boundaries.
    """

    def __init__(self, output_directory: str):
        """
        The constructor of the ToroidalPositionSampler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        output_directory : str
            The filename onto which the sample is written at the end of the run.
        """
        super().__init__(output_directory)
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           output_directory=output_directory)

    def get_observation(self, momenta, positions):
        """
        Returns an observation of the system for the given particle momenta and positions.

        Parameters
        ----------
        momenta : None or numpy.ndarray
            None or a two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each
            element is a float and represents one Cartesian component of the momentum of a single particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            The observation of the positions.
        """
        return get_shortest_vectors_on_torus(positions)
