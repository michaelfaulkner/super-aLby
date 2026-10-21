"""Module for the abstract GhostCouplingBias class."""
from abc import ABCMeta
from base.exceptions import ConfigurationError
import numpy as np


class GhostCouplingBias(metaclass=ABCMeta):
    r"""
    Abstract class for the bias eta(lambda) on the ghost-coupling coordinate lambda of grand-canonical ECMC.

    The augmented target of TemperedGrandCanonicalKernel is proportional to
    z^N / N! * exp(- beta * U(x, y; lambda) + eta(lambda)), with lambda in [- parked_length, 1].  eta is zero for
    lambda <= 0, where the ghost is parked (non-interacting), and its slope g(lambda) = d eta / d lambda is piecewise
    constant on (0, 1] between successive nodes.  The ideal bias is the dimensionless free energy of growing the ghost,
    for which the marginal distribution of lambda on (0, 1] is flat.  Any bias leaves the physical (lambda <= 0)
    samples exact: it only changes how efficiently lambda travels between the parked interval and lambda = 1.
    """

    def __init__(self, nodes, slopes):
        """
        The constructor of the GhostCouplingBias class.

        Parameters
        ----------
        nodes : Sequence[float]
            Strictly increasing coupling values with nodes[0] = 0 and nodes[-1] = 1.
        slopes : Sequence[float]
            The slope g of eta on each interval [nodes[k], nodes[k + 1]]; len(slopes) = len(nodes) - 1.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the nodes do not start at 0, end at 1 and increase strictly, or if the number of slopes is wrong.
        """
        self._nodes = np.asarray(nodes, dtype=float)  # Values of lambda to evaluate biasing at
        self._slopes = np.asarray(slopes, dtype=float)  # g values
        if len(self._nodes) < 2 or abs(self._nodes[0]) > 1.0e-12 or abs(self._nodes[-1] - 1.0) > 1.0e-12:
            raise ConfigurationError(f"The nodes of {self.__class__.__name__} must start at 0.0 and end at 1.0.")
        if np.any(np.diff(self._nodes) <= 0.0):
            raise ConfigurationError(f"The nodes of {self.__class__.__name__} must increase strictly.")
        if len(self._slopes) != len(self._nodes) - 1:
            raise ConfigurationError(f"{self.__class__.__name__} needs exactly one slope per interval between nodes.")
        self._nodes[0], self._nodes[-1] = 0.0, 1.0
        self._eta_at_nodes = np.concatenate(([0.0], np.cumsum(self._slopes * np.diff(self._nodes))))  # Bias values

    @property
    def nodes(self):
        return self._nodes

    @property
    def slopes(self):
        return self._slopes

    @property
    def total(self):
        """eta(1) - eta(0), i.e., the (estimated) dimensionless excess chemical potential beta * mu_ex."""
        return float(self._eta_at_nodes[-1])

    def get_value(self, coupling):
        """Returns biasing eta(coupling), which is zero for coupling <= 0 and piecewise linear on [0, 1]."""
        return float(np.interp(coupling, self._nodes, self._eta_at_nodes))

    def get_segment(self, coupling, direction):
        """
        Determines the interval the ghost coupling lambda is moving through and returns the node at its far end and the
        interval's slope.

        Parameters
        ----------
        coupling : float
            The current ghost coupling lambda.
        direction : int
            +1 if lambda is growing, -1 if lambda is shrinking.

        Returns
        -------
        float
            The coupling at which the current interval ends.
        float
            The slope g of eta on the current interval.
        """
        if direction > 0:
            index = int(np.searchsorted(self._nodes, coupling, side='right')) - 1
            index = min(max(index, 0), len(self._slopes) - 1)
            return self._nodes[index + 1], self._slopes[index]
        index = int(np.searchsorted(self._nodes, coupling, side='left')) - 1
        index = min(max(index, 0), len(self._slopes) - 1)
        return self._nodes[index], self._slopes[index]
