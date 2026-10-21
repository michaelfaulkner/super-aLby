"""Module for the NoGhostCouplingBias class."""
from .ghost_coupling_bias import GhostCouplingBias


class NoGhostCouplingBias(GhostCouplingBias):
    """Class for the unbiased ghost coupling, eta(lambda) = 0."""

    def __init__(self):
        """The constructor of the NoGhostCouplingBias class."""
        super().__init__([0.0, 1.0], [0.0])
