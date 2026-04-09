"""Homogeneous infinite reactor eigenvalue solver.

Solves for the neutron spectrum and k-infinity in an infinite homogeneous
medium.  The transport equation reduces to:

    (diag(Σ_t) − Σ_s0^T − 2·Σ₂^T) · φ = χ · P / k

where P = (Σ_p + 2·colsum(Σ₂)) · φ is the total production rate.

The solver satisfies the ``EigenvalueSolver`` protocol from
``numerics.eigenvalue`` and can be used with the generic
``power_iteration`` function.

.. seealso:: :ref:`theory-homogeneous` — Key Facts, eigenvalue equations, scattering convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.numerics.eigenvalue import power_iteration


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HomogeneousResult:
    """Result of a homogeneous infinite reactor calculation."""

    k_inf: float
    flux: np.ndarray  # (NG,) — group fluxes normalised to 100 n/cm³/s production
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


# ---------------------------------------------------------------------------
# Solver class (satisfies EigenvalueSolver protocol)
# ---------------------------------------------------------------------------

class HomogeneousSolver:
    """Eigenvalue solver for an infinite homogeneous medium.

    The removal matrix A = diag(Σ_t) − Σ_s0^T − 2·Σ₂^T absorbs both
    scattering and (n,2n) into the LHS, so ``solve_fixed_source`` is a
    single sparse direct solve per iteration.
    """

    def __init__(self, mix: Mixture) -> None:
        self.mix = mix
        self.ng = mix.ng
        self.sig2_colsum = np.array(mix.Sig2.sum(axis=1)).ravel()

        # Pre-build the removal matrix (constant across iterations)
        SigS0_T = mix.SigS[0].T.tocsr()
        Sig2_T = mix.Sig2.T.tocsr()
        self._A = diags(mix.SigT) - SigS0_T - 2.0 * Sig2_T

    def initial_flux_distribution(self) -> np.ndarray:
        return np.ones(self.ng)

    def compute_fission_source(
        self, flux_distribution: np.ndarray, keff: float,
    ) -> np.ndarray:
        prod_rate = (self.mix.SigP + 2.0 * self.sig2_colsum) @ flux_distribution
        return self.mix.chi * prod_rate / keff

    def solve_fixed_source(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        return spsolve(self._A.tocsc(), fission_source)

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        prod = (self.mix.SigP + 2.0 * self.sig2_colsum) @ flux_distribution
        abso = self.mix.absorption_xs @ flux_distribution
        return float(prod / abso)

    def converged(
        self, keff: float, keff_old: float,
        flux_distribution: np.ndarray, flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        if iteration < 3:
            return False
        return abs(keff - keff_old) < 1e-10


# ---------------------------------------------------------------------------
# Convenience wrapper (preserves existing call signature)
# ---------------------------------------------------------------------------

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
    solver = HomogeneousSolver(mix)
    k_inf, keff_history, phi = power_iteration(solver, max_iter=n_iter)

    for k in keff_history:
        print(f"    k_inf = {k:.5f}")

    # Normalise flux so total production = 100 n/cm³/s
    sig2_colsum = solver.sig2_colsum
    prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
    phi *= 100.0 / prod_rate

    # Post-processing
    prod_rate = (mix.SigP + 2.0 * sig2_colsum) @ phi
    abs_rate = mix.absorption_xs @ phi
    total_flux = phi.sum()

    ng = mix.ng
    eg = mix.eg
    eg_mid = 0.5 * (eg[:ng] + eg[1:ng + 1])
    de = eg[1:ng + 1] - eg[:ng]
    du = np.log(eg[1:ng + 1] / eg[:ng])

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
