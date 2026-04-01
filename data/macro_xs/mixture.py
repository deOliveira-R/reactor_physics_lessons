"""Macroscopic cross section computation for isotope mixtures.

A Mixture holds the isotopes, their number densities, and the resulting
macroscopic cross sections used by reactor solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from data.micro_xs.isotope import NG, Isotope
from .interpolation import interp_sig_s, interp_xs_field
from .sigma_zeros import solve_sigma_zeros


@dataclass
class Mixture:
    """Macroscopic cross sections for a homogeneous mixture.

    Attributes
    ----------
    SigC, SigL, SigF, SigP, SigT : (NG,) arrays
        Macroscopic capture, (n,alpha), fission, production, total XS in 1/cm.
    SigS : list of (NG, NG) sparse matrices, one per Legendre order.
    Sig2 : (NG, NG) sparse — macroscopic (n,2n) matrix.
    chi  : (NG,) — fission spectrum of the mixture.
    eg   : (NG+1,) — energy group boundaries in eV.
    """

    SigC: np.ndarray
    SigL: np.ndarray
    SigF: np.ndarray
    SigP: np.ndarray
    SigT: np.ndarray
    SigS: list[csr_matrix]
    Sig2: csr_matrix
    chi: np.ndarray
    eg: np.ndarray


def compute_macro_xs(
    isotopes: list[Isotope],
    number_densities: np.ndarray,
    escape_xs: float = 0.0,
    n_legendre: int = 3,
    fissile_indices: Optional[list[int]] = None,
) -> Mixture:
    """Compute macroscopic cross sections for a mixture of isotopes.

    Parameters
    ----------
    isotopes : list[Isotope]
        Microscopic cross section data for each isotope.
    number_densities : (n_iso,) array
        Number densities in 1/(barn*cm).
    escape_xs : float
        Escape cross section in 1/cm (0 for infinite medium).
    n_legendre : int
        Number of Legendre scattering components (default 3).
    fissile_indices : list[int] or None
        Indices into `isotopes` for fissile nuclides.  If None, auto-detected.

    Returns
    -------
    Mixture with all macroscopic cross sections.
    """
    n_iso = len(isotopes)
    aDen = np.asarray(number_densities)
    eg = isotopes[0].eg

    # --- Sigma-zero iterations ---
    print("  Sigma-zero iterations...", end=" ", flush=True)
    sig0 = solve_sigma_zeros(isotopes, aDen, escape_xs)
    print("done.")

    # --- Interpolate microscopic XS at converged sigma-zeros ---
    print("  Interpolating cross sections...", end=" ", flush=True)

    sigC = np.array([interp_xs_field(iso.sigC, iso, sig0[i]) for i, iso in enumerate(isotopes)])
    sigL = np.array([interp_xs_field(iso.sigL, iso, sig0[i]) for i, iso in enumerate(isotopes)])
    sigF = np.array([interp_xs_field(iso.sigF, iso, sig0[i]) for i, iso in enumerate(isotopes)])

    sigS_list: list[list[csr_matrix]] = []
    for j in range(n_legendre):
        sigS_j = [interp_sig_s(iso, j, sig0[i]) for i, iso in enumerate(isotopes)]
        sigS_list.append(sigS_j)

    print("done.")

    # --- Sum to macroscopic XS ---
    # Vector XS: (NG,) = sum_i micro_i(NG) * N_i
    SigC = sigC.T @ aDen
    SigL = sigL.T @ aDen
    SigF = sigF.T @ aDen

    # Production XS: only from fissile isotopes
    if fissile_indices is None:
        fissile_indices = [i for i, iso in enumerate(isotopes) if iso.is_fissile]
    SigP = sum(
        isotopes[i].nubar * sigF[i] * aDen[i] for i in fissile_indices
    ) if fissile_indices else np.zeros(NG)

    # Scattering matrices
    SigS = []
    for j in range(n_legendre):
        mat = sum(sigS_list[j][i] * aDen[i] for i in range(n_iso))
        SigS.append(mat)

    # (n,2n) matrix
    Sig2 = sum(iso.sig2 * aDen[i] for i, iso in enumerate(isotopes))

    # Total XS
    SigT = SigC + SigL + SigF + np.array(SigS[0].sum(axis=1)).ravel() + np.array(Sig2.sum(axis=1)).ravel()

    # Fission spectrum — use first fissile isotope's chi (simplification)
    chi = np.zeros(NG)
    if fissile_indices:
        chi = isotopes[fissile_indices[0]].chi.copy()

    return Mixture(
        SigC=SigC, SigL=SigL, SigF=SigF, SigP=SigP, SigT=SigT,
        SigS=SigS, Sig2=Sig2, chi=chi, eg=eg,
    )
