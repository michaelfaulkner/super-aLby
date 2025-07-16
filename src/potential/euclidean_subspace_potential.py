"""Module for the abstract EuclideanSubspacePotential class."""
from .potential import Potential
from abc import ABCMeta, abstractmethod


class EuclideanSubspacePotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for potentials defined in Euclidean subspaces.

    Apart from get_gradient(), the additional structure provided by this class (relative to Potential) are those methods
        required for event-chain Monte Carlo.

    NB, self.aggregate_pointer_hop_distance (defined below) is the aggregate distance travelled through the state space
        by the active-particle pointer due to hops when the active particle changes.  self.cell_boundary_event (defined
        below) is used to track whether the next event is a cell-boundary event.  Each is currently only used for ECMC
        in HardDiskPotential
    """

    def __init__(self, prefactor: float = 1.0, **kwargs):
        """
        The constructor of the EuclideanSubspacePotential class.

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
            If prefactor is not greater than 0.0.
        """
        super().__init__(prefactor, **kwargs)
        """the following are currently only used for ECMC in HardDiskPotential - see main docstring above for details"""
        self.aggregate_pointer_hop_distance = 0.0
        self.cell_boundary_event = False

    @abstractmethod
    def get_gradient(self, positions):
        """
        Returns the gradient of the potential function for the given particle positions.  This should throw an error if
            used with a non-smooth potential, eg, a hard-sphere potential.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_random_event_chain_velocity():
        """
        Uniformly samples a direction of motion for the active particle from chosen velocity distribution.

        Returns
        ----------
        random_event_chain_velocity : int or numpy.ndarray
            The uniformly sampled event-chain velocity of the active particle.  If the state space of each particle is
            a subset of the real line, the method should output an integer; otherwise it should output a one-dimensional
            numpy array (of integers) of length dimensionality_of_particle_space, where the nth component represents the
            velocity of the active particle along the nth Cartesian direction.
        """
        raise NotImplementedError

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

    @abstractmethod
    def choose_next_active_particle(self, positions, active_particle_index, movement_direction, veto_index):
        """
        Chooses the index and direction of motion of the next active particle in ECMC.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents the spin angle of its corresponding particle.
        active_particle_index : int
            The active particle index
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def update_position(positions, displacement_distance, active_particle_index, movement_direction):
        """Updates the position of the active particle following an event."""
        raise NotImplementedError

    @abstractmethod
    def get_portal_candidate(self, positions, active_particle_index, veto_index, movement_direction):
        """Propose candidate via teleportation portal kernel."""
        raise NotImplementedError
