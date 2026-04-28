"""Diagnostic 03: actual k_eff comparison (Mark, Hébert, specular)
on the thin homogeneous sphere (τ_R = 2.5).

Created by numerics-investigator on 2026-04-27.

Diag 02 was a surprising finding: σ_t·K·1 ≈ 0.5 (NOT 1.0) for ALL
three closures on thin sphere — Mark, Hébert, specular. So the
"σ_t·K·1 = 1 means k_inf recovery" intuition is WRONG. The correct
contract must be different.

This diagnostic computes actual k_eff for all three closures on the
thin sphere fuel-A 1G case to see which closures DO recover k_inf.

Key XS: σ_t = 0.5, σ_s = 0.38, νΣ_f = 0.025, k_inf = 0.025/0.12
       = 0.208333.

If Hébert recovers k_inf to <0.5% and specular gives -6%, then the
problem is NOT in the row-sum K·1 — the eigenvector is non-uniform on
the thin sphere even for "homogeneous infinite-medium-equivalent"
case, and σ_t·K·1·1 = constant doesn't carry the right contract.

The correct contract for k_inf is:
   For any φ such that A φ = (1/k) B φ (eigenvector),
   k = (B φ)·1 / (A φ)·1 = ((K·νΣ_f φ)·1) / ((Σ_t I - K·Σ_s) φ)·1

For φ = constant, this reduces to:
   k_eff = νΣ_f · (K·1)_avg / (Σ_t - σ_s · (K·1)_avg)
         = νΣ_f · κ / (Σ_t - σ_s · κ)    where κ = avg σt·K·1 / Σ_t

If κ < 1 (as in thin sphere), k_eff < k_inf. The thin-cell
penalty is exactly the κ-deficit. So the question becomes: why
does Hébert get k_inf right while Mark/specular don't, given that
all three give κ ≈ 0.5?

Answer probably: the eigenvector is NOT uniform for thin sphere.
The "uniform-φ → k_inf" intuition only works when the geometry is
small enough that escape doesn't redistribute the flux radially.
For thin sphere there IS radial flux redistribution and the row-by-row
behavior matters, not the average.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    solve_peierls_1g,
)
from orpheus.derivations._xs_library import get_xs


def _solve(R, sigt, sigs, nuf, *, n_bc_modes, boundary, geom=SPHERE_1D):
    radii = np.array([R])
    sig_t = np.array([sigt])
    sig_s_arr = np.array([[sigs]])
    nu_sig_f = np.array([nuf])
    return solve_peierls_1g(
        geom, radii, sig_t, sig_s_arr, nu_sig_f,
        boundary=boundary, n_bc_modes=n_bc_modes,
        p_order=4, n_panels_per_region=2, n_angular=24, n_rho=24,
        n_surf_quad=24, dps=20, tol=1e-10,
    )


CASES = [
    # (tag, R, sigt, sigs, nuf)  → user's reported thin-cell XS
    ("thin τR=2.5 (user)", 5.0, 0.5, 0.38, 0.025),
    # Standard fuel A 1G (σ_t=1, k_inf=1.5)
    ("thick τR=5.0 (fuel A)", 5.0, 1.0, 0.5, 0.75),
]


@pytest.mark.parametrize("tag,R,sigt,sigs,nuf", CASES)
def test_keff_per_closure(tag, R, sigt, sigs, nuf, capsys):
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== {tag}: σ_t={sigt}, σ_s={sigs}, νΣ_f={nuf}, "
              f"k_inf = {k_inf:.6f} ===")

        sol = _solve(R, sigt, sigs, nuf, n_bc_modes=1,
                     boundary="white_rank1_mark")
        err = (sol.k_eff - k_inf) / k_inf
        print(f"  white_rank1_mark, N=1: k_eff={sol.k_eff:.8f} "
              f"err={err*100:+.4f}%")

        sol = _solve(R, sigt, sigs, nuf, n_bc_modes=1,
                     boundary="white_hebert")
        err = (sol.k_eff - k_inf) / k_inf
        print(f"  white_hebert,      N=1: k_eff={sol.k_eff:.8f} "
              f"err={err*100:+.4f}%")

        for N in (1, 2, 4, 6):
            sol = _solve(R, sigt, sigs, nuf, n_bc_modes=N,
                         boundary="specular")
            err = (sol.k_eff - k_inf) / k_inf
            print(f"  specular,          N={N}: k_eff={sol.k_eff:.8f} "
                  f"err={err*100:+.4f}%")


if __name__ == "__main__":
    import sys
    for tag, R, sigt, sigs, nuf in CASES:
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== {tag}: σ_t={sigt}, σ_s={sigs}, νΣ_f={nuf}, "
              f"k_inf = {k_inf:.6f} ===")
        sol = _solve(R, sigt, sigs, nuf, n_bc_modes=1,
                     boundary="white_rank1_mark")
        err = (sol.k_eff - k_inf) / k_inf
        print(f"  white_rank1_mark, N=1: k_eff={sol.k_eff:.8f} "
              f"err={err*100:+.4f}%")

        sol = _solve(R, sigt, sigs, nuf, n_bc_modes=1,
                     boundary="white_hebert")
        err = (sol.k_eff - k_inf) / k_inf
        print(f"  white_hebert,      N=1: k_eff={sol.k_eff:.8f} "
              f"err={err*100:+.4f}%")

        for N in (1, 2, 4, 6):
            sol = _solve(R, sigt, sigs, nuf, n_bc_modes=N,
                         boundary="specular")
            err = (sol.k_eff - k_inf) / k_inf
            print(f"  specular,          N={N}: k_eff={sol.k_eff:.8f} "
                  f"err={err*100:+.4f}%")
    sys.exit(0)
