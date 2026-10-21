"""Module for the TabulatedGhostCouplingBias class."""
from base.exceptions import ConfigurationError
from .ghost_coupling_bias import GhostCouplingBias
import json
import os


class TabulatedGhostCouplingBias(GhostCouplingBias):
    """
    Class for a ghost-coupling bias read from a JSON file with keys "nodes" and "slopes" (see GhostCouplingBias).  Such
    files are written by sample_analysis/compute_ghost_coupling_bias.py from fixed-coupling calibration runs.
    """

    def __init__(self, file_name: str = None, slope_scale: float = 1.0):
        """
        The constructor of the TabulatedGhostCouplingBias class.

        Parameters
        ----------
        file_name : str
            Path (relative to the working directory) of the JSON file containing the nodes and slopes.
        slope_scale : float, optional
            Multiplies every slope (1.0 uses the tabulated bias; values other than 1.0 are for testing sensitivity).

        Raises
        ------
        base.exceptions.ConfigurationError
            If file_name does not exist.
        """
        if file_name is None or not os.path.isfile(file_name):
            raise ConfigurationError(f"Give the path of an existing JSON file as file_name in "
                                     f"{self.__class__.__name__}.  Given: {file_name}")
        with open(file_name) as f:
            table = json.load(f)
        super().__init__(table["nodes"], [slope_scale * slope for slope in table["slopes"]])
        self.file_name = file_name
