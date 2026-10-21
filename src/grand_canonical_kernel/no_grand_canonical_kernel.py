"""Module for the NoGrandCanonicalKernel class."""
from .grand_canonical_kernel import GrandCanonicalKernel


class NoGrandCanonicalKernel(GrandCanonicalKernel):
    """Class for canonical event-chain Monte Carlo (fixed number of particles, no ghost).  This is the default."""

    def __init__(self):
        """The constructor of the NoGrandCanonicalKernel class."""
        super().__init__()

    def initialize(self, potential, temperature):
        """Works with any potential: a GrandCanonicalPotential is used without a ghost."""
        self._potential = potential
        self._temperature = temperature
        if hasattr(potential, "ghost_coupling"):
            potential.ghost_coupling = None

    def get_summary(self):
        return {"scheme": "canonical"}
