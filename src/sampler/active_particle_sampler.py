from .sampler import Sampler
import numpy as np

class ActiveParticleSampler(Sampler):
    """
    Class for taking observations of the active particle within EventChainMediator.
    """
    def  __init__(self, output_directory: str):
        """
        The constructor of the PressureSampler class.

        Parameters
        ----------
        output_directory : str
            The filename onto which the sample is written at the end of the run.
        """
        super().__init__(output_directory)