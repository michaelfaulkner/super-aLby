"""Module for the abstract GhostCouplingBias class."""
from abc import ABCMeta
from base.exceptions import ConfigurationError
import numpy as np


class GhostCouplingBias(metaclass=ABCMeta):
    r"""
    Abstract class for the bias eta(lambda) on the ghost-coupling lambda for grand-canonical ECMC.

    The augmented target of TemperedGrandCanonicalKernel is proportional to
    z^N / N! * exp(-beta * U(x, y; lambda) + eta(lambda)), with lambda in [-parked_length, 1].  eta is zero for
    lambda <= 0, where the ghost is parked (non-interacting), and its slope g(lambda) = d eta / d lambda is piecewise
    constant on (0, 1] between successive nodes.  The ideal bias is the dimensionless free energy of growing the ghost,
    for which the marginal distribution of lambda on (0, 1] is flat.
    """

    def __init__(self, nodes, slopes):
        """
        The constructor of the GhostCouplingBias class.

        Parameters
        ----------
        nodes : Sequence[float]
            Strictly increasing coupling values with nodes[0] = 0 and nodes[-1] = 1.
        slopes : Sequence[float]
            The slope g of eta on each interval [nodes[k], nodes[k + 1]].

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
        """Running integrals over lambda of the bias event rates [g]^+ (lambda shrinking) and [- g]^+ (growing)."""
        self._shrinking_rate_integrals = np.concatenate(
            ([0.0], np.cumsum(np.maximum(self._slopes, 0.0) * np.diff(self._nodes))))
        self._growing_rate_integrals = np.concatenate(
            ([0.0], np.cumsum(np.maximum(-self._slopes, 0.0) * np.diff(self._nodes))))

    def get_description(self):
        """Returns the nodes, the slopes and eta(1) = beta * mu_ex (the total) as a JSON-serialisable dictionary."""
        return {"class": self.__class__.__name__, "nodes": self._nodes.tolist(), "slopes": self._slopes.tolist(),
                "total": float(self._eta_at_nodes[-1])}

    def get_value(self, coupling):
        """Returns biasing eta(coupling), which is zero for coupling <= 0 and piecewise linear on [0, 1]."""
        return float(np.interp(coupling, self._nodes, self._eta_at_nodes))

    def get_next_event(self, coupling, direction):
        """
        Returns the distance in lambda to the next bias event of lambda moving from coupling (in [0, 1]) in the given
        direction (+1 growing, -1 shrinking), i.e. where the integrated rate [- direction * g]^+ reaches an
        exponential random variable, or numpy.inf if that happens outside [0, 1].  The integrated rate is piecewise
        linear between the nodes, so it is inverted exactly.
        """
        target = -np.log(np.random.uniform(0.0, 1.0))
        if direction > 0:  # Will usually return np.inf as bias event rate is usually zero for growing moves
            integrals = self._growing_rate_integrals
            value = float(np.interp(coupling, self._nodes, integrals)) + target  # Integrated rate event must reach
            if value >= integrals[-1]:  # No bias event before lambda=1
                return np.inf
            index = int(np.searchsorted(integrals, value, side='right')) - 1
        else:
            integrals = self._shrinking_rate_integrals
            value = float(np.interp(coupling, self._nodes, integrals)) - target
            if value <= 0.0:  # No bias event before lambda=0
                return np.inf
            index = int(np.searchsorted(integrals, value, side='left')) - 1
        rate = -direction * self._slopes[index]
        event_coupling = self._nodes[index] + (value - integrals[index]) / rate  # Final lambda value
        return direction * (event_coupling - coupling)

    def get_slope(self, coupling, direction):
        """
        Returns the slope g of eta on the interval the ghost coupling lambda is moving through (at a node, the interval
        it is entering).

        Parameters
        ----------
        coupling : float
            The current ghost coupling lambda.
        direction : int
            +1 if lambda is growing, -1 if lambda is shrinking.

        Returns
        -------
        float
            The slope g of eta on the current interval.
        """
        index = int(np.searchsorted(self._nodes, coupling, side='right' if direction > 0 else 'left')) - 1
        return self._slopes[min(max(index, 0), len(self._slopes) - 1)]
