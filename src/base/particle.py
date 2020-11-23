# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019 The JeLLyFysh organization
# (see the AUTHORS file for the full list of authors)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either > version 3 of the License, or (at your option) any
# later version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2019] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, Werner Krauth
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# arXiv e-prints: 1907.12502 (2019), https://arxiv.org/abs/1907.12502
#
"""Module for the Particle class."""
from typing import Mapping, Sequence


class Particle(object):
    """
    Class to store physical state information, meaning the positions and charges of a particle.

    A particle can be a point mass or a composite point object.

    Attributes
    ----------
    position : Sequence[float]
        The positions of the particle.
    charge : Mapping[str, float] or None
        A map from the name onto the value of the charge of the particle.
    """

    def __init__(self, position: Sequence[float], charge: Mapping[str, float] = None) -> None:
        """
        The constructor of the Particle class.

        Parameters
        ----------
        position : Sequence[float]
            The initial positions of the particle.
        charge : Mapping[str, float] or None, optional
            A map from the name onto the value of the charge of the particle.
        """
        self.position = position
        self.charge = charge
