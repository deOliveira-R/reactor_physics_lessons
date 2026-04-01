"""Data model for microscopic cross section data of a single isotope."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

NG = 421  # number of energy groups


@dataclass
class Isotope:
    """Microscopic cross section data for one isotope at one temperature.

    All cross sections are in barns; energy boundaries in eV.
    Dimensions:
        sig0    : (n_sig0,)           — background cross section base points
        sigC    : (n_sig0, NG)        — radiative capture
        sigL    : (n_sig0, NG)        — (n,alpha)
        sigF    : (n_sig0, NG)        — fission
        sigT    : (n_sig0, NG)        — total
        sigS    : [3][n_sig0] of (NG, NG) sparse  — scattering (3 Legendre)
        sig2    : (NG, NG) sparse     — (n,2n) matrix
        nubar   : (NG,)              — average neutrons per fission
        chi     : (NG,)              — fission spectrum
        eg      : (NG+1,)            — energy group boundaries
    """

    name: str
    aw: float  # atomic weight (amu)
    temp: float  # temperature (K)
    eg: np.ndarray  # energy group boundaries
    sig0: np.ndarray  # sigma-zero base points

    sigC: np.ndarray
    sigL: np.ndarray
    sigF: np.ndarray
    sigT: np.ndarray

    nubar: np.ndarray  # (NG,)
    chi: np.ndarray  # (NG,)

    sigS: list[list[csr_matrix]] = field(default_factory=list)  # [legendre][sig0_idx]
    sig2: csr_matrix = field(default_factory=lambda: csr_matrix((NG, NG)))

    @property
    def n_sig0(self) -> int:
        return len(self.sig0)

    @property
    def ng(self) -> int:
        return NG

    @property
    def is_fissile(self) -> bool:
        return np.any(self.sigF > 0)
