"""Module for the abstract GrandCanonicalKernel class."""
from abc import ABCMeta, abstractmethod
from base.exceptions import ConfigurationError
import numpy as np


class GrandCanonicalKernel(metaclass=ABCMeta):
    r"""
    Abstract class for the particle-number moves of event-chain Monte Carlo in the grand-canonical ensemble.

    EventChainMediator races the next event of the kernel against the potential, refreshment and measurement events.
    When the kernel event wins, the mediator calls handle_event, which performs the whole kernel move (e.g. an
    insertion or deletion attempt, or a set of coupling moves, during which the particles are frozen and the kernel
    moves the ghost coupling).
    """

    def __init__(self, **kwargs):
        """
        The constructor of the GrandCanonicalKernel class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.
        """
        super().__init__(**kwargs)
        self._potential = None
        self._temperature = None
        self._statistics = {}

    def initialize(self, potential, temperature):
        """
        Stores the potential and temperature and sets the ghost coupling of the potential.  Called by the constructor of
        EventChainMediator, before the initial positions are generated.
        """
        from potential.grand_canonical_potential import GrandCanonicalPotential
        if not isinstance(potential, GrandCanonicalPotential):
            raise ConfigurationError(f"{self.__class__.__name__} needs a GrandCanonicalPotential child class as the "
                                     f"potential of EventChainMediator.")
        self._potential = potential
        self._temperature = temperature

    def _count(self, key):
        """
        Counts a kernel statistic e.g. insertion/deletion attempts and acceptances, flips of lambda.
        """
        self._statistics[key] = self._statistics.get(key, 0) + 1

    def get_distance_to_next_event(self):
        """Returns the distance through the state space (in position mode) to the next kernel event."""
        return np.inf

    def handle_event(self, positions, active_particle_index, movement_direction):
        """
        Performs a kernel move.  Returns (positions, active_particle_index, movement_direction). Positions may be a new
        array if the number of particles changed.
        """
        return positions, active_particle_index, movement_direction

    @abstractmethod
    def get_summary(self):
        """Returns a JSON-serialisable dictionary describing the kernel and its production-phase statistics."""
        raise NotImplementedError
