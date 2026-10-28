"""Module for the FixedGhostCouplingKernel class."""
from base.exceptions import ConfigurationError
from .grand_canonical_kernel import GrandCanonicalKernel


class FixedGhostCouplingKernel(GrandCanonicalKernel):
    """
    Class for canonical event chains with a ghost particle held at a fixed coupling. Use for calibration runs to
    measure the g values of TemperedGrandCanonicalKernel. The ghost is the last row of positions.
    """

    def __init__(self, ghost_coupling: float = 0.0):
        """
        The constructor of the FixedGhostCouplingKernel class.

        Parameters
        ----------
        ghost_coupling : float, optional
            The fixed ghost coupling lambda in [0, 1].

        Raises
        ------
        base.exceptions.ConfigurationError
            If ghost_coupling is not in [0, 1].
        """
        super().__init__()
        if not 0.0 <= ghost_coupling <= 1.0:
            raise ConfigurationError(f"Give 0 <= ghost_coupling <= 1 in {self.__class__.__name__}.")
        self._ghost_coupling = ghost_coupling

    def initialize(self, potential, temperature):
        super().initialize(potential, temperature)
        potential.ghost_coupling = self._ghost_coupling

    def get_summary(self):
        return {"scheme": "calibration", "fixed_ghost_coupling": self._ghost_coupling,
                "volume": self._potential.get_volume(), "potential": self._potential.get_description()}
