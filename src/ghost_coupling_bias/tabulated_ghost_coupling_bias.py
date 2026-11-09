"""Module for the TabulatedGhostCouplingBias class."""
from base.exceptions import ConfigurationError
from .ghost_coupling_bias import GhostCouplingBias
import json
import numpy as np
import os


class TabulatedGhostCouplingBias(GhostCouplingBias):
    """
    Class for ghost-coupling bias read from an existing JSON file. Such files are written by
    sample_analysis/compute_ghost_coupling_bias.py from fixed-coupling calibration runs.  Files without
    "coupling_convention": "diameter" use the old coupling lambda_old = (1 + lambda) / 2 (nodes from 0 to 1) and are
    converted.
    """

    def __init__(self, file_name: str = None):
        """
        The constructor of the TabulatedGhostCouplingBias class.

        Parameters
        ----------
        file_name : str
            Path of the JSON file.

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
        nodes, slopes = np.asarray(table["nodes"], dtype=float), np.asarray(table["slopes"], dtype=float)
        if table.get("coupling_convention") != "diameter":
            nodes, slopes = 2.0 * nodes - 1.0, 0.5 * slopes
        super().__init__(nodes, slopes)
        self.file_name = file_name
