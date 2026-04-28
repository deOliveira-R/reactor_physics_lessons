"""Diagnostic 11: synthesis — rank-N specular plateau on thin cells
is a STRUCTURAL gap in the boundary closure, NOT a physical limitation
of specular BC.

Created by numerics-investigator on 2026-04-27.

Root-cause findings (see diags 01-14 in this folder):

1. The plateau is NOT a quadrature artifact. At RICH precision the
   thin-sphere ladder is:
     N=4: -6.30 %, N=6: -6.28 %, N=8: -6.25 %, N=10: -6.25 %.

2. The bug is NOT in the P primitive Jacobian. Adding (ρ_max/R)² makes
   it -48 %; adding µ-weight (Marshak) makes it -26 %. The "no_jacobian"
   choice is empirically the best of these three (diag 14).

3. The K_bc = G·R·P specular construction is a SINGLE-BOUNCE operator.
   At rank-1, R = [[1]] and the multi-bounce is recovered by Hébert's
   (1 - P_ss)⁻¹ scalar correction:
     spec rank-1 + (1-P_ss)⁻¹ = -0.27 % on thin sphere (diag 04 N=1)
     specular rank-1 alone     = -8.42 %.

4. The natural rank-N matrix generalisation is
     K_bc^corr = G · R · (I - T·R)⁻¹ · P
   where T_mn = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-σ_t · 2R · µ} dµ is the
   surface-to-surface partial-current transfer matrix for sphere
   specular (chord 2Rµ at surface cosine µ; specular preserves the
   cosine through the chord by sphere symmetry). At rank-1, T[0,0] =
   P_ss to 1e-16 (diag 06 sanity).

5. The matrix correction WORKS at low N (diag 06):
     thin τR=2.5: corrected N=1: -0.27 %, N=2: -0.25 %, N=3: -0.12 %.
     very thin τR=1.0: baseline -42 %, corrected N=1..3: -0.13 %
     to -0.04 %.
   But OVERSHOOTS at high N: corrected N=6: +2.31 %, N=8: +5.62 %.

6. The high-N failure is SPECTRAL: ρ(T·R) → 1 as N grows because
   R_spec entries grow O(N²) while T captures grazing-mode bounces
   that survive reflection forever (chord → 0 as µ → 0). At N=10
   thin sphere ρ(T·R) = 0.87; at very thin ρ(T·R) = 0.94 — close
   to instability of the geometric series.

CONCLUSION: rank-N specular needs a multi-bounce closure that:
  (a) reduces to scalar Hébert at rank-1 (✓ for our naive matrix
      correction at N=1);
  (b) does NOT amplify grazing modes via R_spec at high N
      (FAILED for naive matrix; needs a different formulation).

This synthesis test PINS the plateau and protects against regressions
that would either silently improve it (good — but should be verified
to actually fix the issue) or worsen it.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, solve_peierls_1g,
)


_RICH_KW = {
    "p_order": 6, "n_panels_per_region": 4, "n_angular": 48,
    "n_rho": 48, "n_surf_quad": 48, "dps": 30,
}

# Thin-cell case the user reported (σ_t·R = 2.5)
_R = 5.0
_SIGT = 0.5
_SIGS = 0.38
_NUF = 0.025
_KINF = _NUF / (_SIGT - _SIGS)


def _solve_specular_rich(n_bc_modes):
    return solve_peierls_1g(
        SPHERE_1D, np.array([_R]), np.array([_SIGT]),
        np.array([[_SIGS]]), np.array([_NUF]),
        boundary="specular", n_bc_modes=n_bc_modes, tol=1e-10,
        **_RICH_KW,
    )


def _solve_hebert_rich():
    return solve_peierls_1g(
        SPHERE_1D, np.array([_R]), np.array([_SIGT]),
        np.array([[_SIGS]]), np.array([_NUF]),
        boundary="white_hebert", n_bc_modes=1, tol=1e-10,
        **_RICH_KW,
    )


@pytest.mark.foundation
def test_specular_thin_sphere_plateau_pinned(capsys):
    """Pin the specular thin-cell structural plateau at RICH quadrature.

    If a future fix reduces this plateau substantially, this test will
    fail — at which point inspect, verify the fix is principled (not a
    quadrature artifact), and update the threshold.
    """
    with capsys.disabled():
        print(f"\n=== thin τR=2.5, k_inf={_KINF:.6f} ===")
        sol_h = _solve_hebert_rich()
        err_h = (sol_h.k_eff - _KINF) / _KINF
        print(
            f"  Hebert:  k_eff={sol_h.k_eff:.8f} ({err_h*100:+.4f}%)"
        )
        assert abs(err_h) < 5e-3, (
            f"Hebert should recover k_inf to <0.5% on thin cell — "
            f"got {err_h*100:.4f}%"
        )

        for n_modes, expected_err_pct in [
            (4, -6.30), (6, -6.28), (8, -6.25),
        ]:
            sol = _solve_specular_rich(n_modes)
            err = (sol.k_eff - _KINF) / _KINF * 100
            print(
                f"  spec N={n_modes}: k_eff={sol.k_eff:.8f} "
                f"({err:+.4f}%); expected ~{expected_err_pct:+.2f}%"
            )
            assert abs(err - expected_err_pct) < 0.05, (
                f"Specular N={n_modes} thin-cell err = {err:.4f}% — "
                f"far from expected {expected_err_pct:.2f}%. Either "
                f"the plateau changed (good — investigate) or a "
                f"regression."
            )


if __name__ == "__main__":
    print(f"\n=== thin τR=2.5, k_inf={_KINF:.6f} ===")
    sol_h = _solve_hebert_rich()
    err_h = (sol_h.k_eff - _KINF) / _KINF * 100
    print(f"  Hebert:  k_eff={sol_h.k_eff:.8f} ({err_h:+.4f}%)")
    for n_modes in (4, 6, 8):
        sol = _solve_specular_rich(n_modes)
        err = (sol.k_eff - _KINF) / _KINF * 100
        print(f"  spec N={n_modes}: k_eff={sol.k_eff:.8f} ({err:+.4f}%)")
