"""Homogeneous infinite reactor eigenvalue solver.

Solves for the neutron spectrum and k-infinity in an infinite homogeneous
medium by iterating on the slowing-down / fission source equation:

    diag(SigT) * phi = chi * P/k + SigS0^T * phi + 2 * Sig2^T * phi

where P = (SigP + 2*colsum(Sig2)) . phi is the total production rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags

from data.macro_xs.mixture import Mixture
from data.micro_xs.isotope import NG


@dataclass
class HomogeneousResult:
    """Result of a homogeneous infinite reactor calculation."""

    k_inf: float
    flux: np.ndarray  # (NG,) — group fluxes normalised to 100 n/cm3/s production
    eg_mid: np.ndarray  # (NG,) — mid-group energies (eV)
    de: np.ndarray  # (NG,) — energy bin widths (eV)
    du: np.ndarray  # (NG,) — lethargy bin widths
    sig_prod: float  # one-group production XS (1/cm)
    sig_abs: float  # one-group absorption XS (1/cm)
    mixture: Mixture

    @property
    def flux_per_energy(self) -> np.ndarray:
        return self.flux / self.de

    @property
    def flux_per_lethargy(self) -> np.ndarray:
        return self.flux / self.du


def solve_homogeneous_infinite(
    mix: Mixture,
    n_iter: int = 5,
) -> HomogeneousResult:
    """Solve the eigenvalue problem for an infinite homogeneous reactor.

    Parameters
    ----------
    mix : Mixture
        Macroscopic cross sections for the homogeneous medium.
    n_iter : int
        Number of power iterations (5 is typically sufficient).

    Returns
    -------
    HomogeneousResult
    """
    sig2_colsum = np.array(mix.Sig2.sum(axis=1)).ravel()
    SigS0_T = mix.SigS[0].T.tocsr()
    Sig2_T = mix.Sig2.T.tocsr()

    # LHS matrix: diag(SigT) - SigS0^T - 2*Sig2^T
    A = diags(mix.SigT) - SigS0_T - 2.0 * Sig2_T

    phi = np.ones(NG)
    k_inf = 1.0

    for _ in range(n_iter):
        prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
        abs_rate = (mix.SigC + mix.SigF + mix.SigL + sig2_colsum) @ phi
        k_inf = prod_rate / abs_rate
        print(f"    k_inf = {k_inf:.5f}")

        # Solve: A * phi = chi * prod_rate / k_inf
        from scipy.sparse.linalg import spsolve
        rhs = mix.chi * prod_rate / k_inf
        phi = spsolve(A.tocsc(), rhs)

        # Normalise so total production = 100 n/cm3/s
        prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
        phi *= 100.0 / prod_rate

    prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
    abs_rate = (mix.SigC + mix.SigF + mix.SigL + sig2_colsum) @ phi
    total_flux = phi.sum()

    eg = mix.eg
    eg_mid = 0.5 * (eg[:NG] + eg[1 : NG + 1])
    de = eg[1 : NG + 1] - eg[:NG]
    du = np.log(eg[1 : NG + 1] / eg[:NG])

    return HomogeneousResult(
        k_inf=k_inf,
        flux=phi,
        eg_mid=eg_mid,
        de=de,
        du=du,
        sig_prod=prod_rate / total_flux,
        sig_abs=abs_rate / total_flux,
        mixture=mix,
    )
