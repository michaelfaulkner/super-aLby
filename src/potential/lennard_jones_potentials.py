"""Module for the LennardJonesPotential class."""
from .soft_matter_potential import SoftMatterPotential
from base.exceptions import ConfigurationError
from abc import ABCMeta, abstractmethod


class LennardJonesPotentials(SoftMatterPotential, metaclass=ABCMeta):
    r"""
    Abstract class for Lennard-Jones potentials

        $ U = k * \sum_{i > j} U_{{\rm LJ}, ij} $ ,

    where

        $ U_{{\rm LJ}, ij} = \begin{cases}
                                U_{{\rm LJ}, ij}^{\rm bare}(r_{ij}) - U_{{\rm LJ}, ij}^{\rm bare}(r_{\rm c}) \,
                                    {\rm if} \, r_{ij} \le r_{\rm c} \\
                                0 \, {\rm if} \, r_{ij} > r_{\rm c}
                             \end{cases} $

    is the two-particle Lennard-Jones potential, and

        $ U_{{\rm LJ}, ij}^{\rm bare}(r_{ij}) = 4 \epsilon \left[\left(\frac{\sigma}{r_{ij}}\right)^{12} -
            \left(\frac{\sigma}{r_{ij}}\right)^6\right]$

    is the bare two-particle Lennard-Jones potential. In the above, $\epsilon$ is the bare well depth, $\sigma$ is the
    characteristic length scale of the Lennard-Jones potential, and $r_c$ is the cutoff distance at which the bare
    two-particle potential is truncated. If using a cutoff distance $r_c$, we recommend $r_c \ge 2.5 \sigma$.
    """

    def __init__(self, characteristic_length: float = 1.0, well_depth: float = 1.0, prefactor: float = 1.0) -> None:
        """
        The constructor of the LennardJonesPotentials class.

        NOTE THAT:
            i) The Metropolis algorithm does not seem to converge for two Lennard-Jones particles for which the
            value of each component of size_of_particle_space is greater than twice the value of characteristic_length
            -- perhaps due to too much time spent with particles independently drifting around.
            ii) Newtonian- and relativistic-dynamics-based algorithms do not seem to converge for two Lennard-Jones
            particles for which the value of each component of size_of_particle_space is less than twice the value of
            characteristic_length -- perhaps due to discontinuities in the potential gradients.

        Parameters
        ----------
        characteristic_length : float, optional
            The characteristic length scale of the two-particle Lennard-Jones potential.
        well_depth : float, optional
            The well depth of the bare two-particle Lennard-Jones potential.
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If model_settings.range_of_initial_particle_positions does not give an real-valued interval for each
            component of the initial positions of each particle.
        base.exceptions.ConfigurationError
            If characteristic_length is less than 0.5.
        """
        super().__init__(prefactor)
        if characteristic_length < 0.5:
            raise ConfigurationError(f"Give a value not less than 0.5 for characteristic_length in "
                                     f"{self.__class__.__name__}.")
        self._potential_12_constant = 4.0 * prefactor * well_depth * characteristic_length ** 12
        self._potential_6_constant = 4.0 * prefactor * well_depth * characteristic_length ** 6
        self._gradient_12_constant = 12.0 * self._potential_12_constant
        self._gradient_6_constant = 6.0 * self._potential_6_constant
        self._bare_potential_at_cut_off = 0.0

    @abstractmethod
    def get_value(self, positions):
        """
        Returns the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        float
            The potential function.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, positions):
        """
        Returns the gradient of the potential function for the given particle positions.

        Parameters
        ----------
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle.

        Returns
        -------
        numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the gradient of the potential of a single particle.
        """
        raise NotImplementedError

    @abstractmethod
    def get_potential_difference(self, active_particle_index, candidate_position, positions):
        """
        Returns the potential difference resulting from moving the single active particle to candidate_position.

        Parameters
        ----------
        active_particle_index : int
            The index of the active particle.
        candidate_position : numpy.ndarray
            A one-dimensional numpy array of length dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the proposed position of the active particle.
        positions : numpy.ndarray
            A two-dimensional numpy array of size (number_of_particles, dimensionality_of_particle_space); each element
            is a float and represents one Cartesian component of the position of a single particle. For Bayesian
            models, the entire positions array corresponds to the parameter; for the Ginzburg-Landau potential on a
            lattice, the entire positions array corresponds to the entire array of superconducting phase.

        Returns
        -------
        float
            The potential difference resulting from moving the single active particle to candidate_position.
        """
        raise NotImplementedError

    def _get_non_zero_two_particle_potential(self, separation_distance):
        """
        Returns the Lennard-Jones potential for two particles whose shortest separation distance is not greater than
        self._cutoff_length.

        Parameters
        ----------
        separation_distance : float
            The shortest separation distance between the two particles.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential (for all cases for which it is non-zero).
        """
        return (self._potential_12_constant * separation_distance ** (- 12.0) -
                self._potential_6_constant * separation_distance ** (- 6.0) - self._bare_potential_at_cut_off)

    def _get_non_zero_two_particle_gradient(self, separation_vector, separation_distance):
        """
        Returns the Lennard-Jones potential for two particles whose shortest separation distance is not greater than
        self._cutoff_length.

        Parameters
        ----------
        separation_vector : numpy.ndarray
            A one-dimensional numpy array of size dimensionality_of_particle_space; each element is a float and
            represents one Cartesian component of the gradient of the shortest separation vector between the two
            particles.
        separation_distance : float
            The shortest separation distance between the two particles.

        Returns
        -------
        float
            The two-particle Lennard-Jones potential (for all cases for which it is non-zero).
        """
        return - separation_vector * (self._gradient_12_constant * separation_distance ** (- 14.0) -
                                      self._gradient_6_constant * separation_distance ** (- 8.0))
