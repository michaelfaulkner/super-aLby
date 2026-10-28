"""Module for the TabulatedGhostCouplingBias class."""
from base.exceptions import ConfigurationError
from .ghost_coupling_bias import GhostCouplingBias
import json
import os


class TabulatedGhostCouplingBias(GhostCouplingBias):
    """
    Class for ghost-coupling bias read from an existing JSON file. Such files are written by
    sample_analysis/compute_ghost_coupling_bias.py from fixed-coupling calibration runs.
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
        super().__init__(table["nodes"], table["slopes"])
        self.file_name = file_name
