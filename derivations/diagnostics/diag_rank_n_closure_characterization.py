"""Rank-N hollow-sphere closure characterization — comprehensive scans.

Issue #119 analysis (2026-04-21):

After extensive investigation (Sanchez-McCormick §III.F.1 implementation,
60+ recipe variants, MC cross-check of W), we've characterized WHY the
rank-N per-face white-BC closure does not close to ≤ 0.1% residual on
hollow sphere. Findings:

1. **Quadrature is not the bottleneck.** Radial-mesh refinement from
   8 nodes → 48 nodes reduces Sanchez N=2 residual from 1.42% to 1.31%
   only. Not converging to zero.

2. **Mode truncation is not the bottleneck.** Going from N=2 to N=4 in
   Sanchez µ-ortho basis gives 1.42% → 1.42% → 1.42%. Literal plateau.

3. **F.4 scalar N=1 is SPECIAL.** Across the tested σ_t·R range:

   | σ_t·R | F.4 N=1 err | Sanchez N=2 err | Sanchez N=3 err |
   |-------|-------------|-----------------|-----------------|
   | 1.0   | 3.27%       | 42.02%          | 49.15%          |
   | 2.5   | 0.55%       | 5.89%           | 5.94%           |
   | 5.0   | 0.077%      | 1.42%           | 1.42%           |
   | 10.0  | 0.26%       | 0.53%           | 0.53%           |
   | 20.0  | 0.45%       | 0.44%           | 0.45%           |

   F.4 has a minimum around σ_t·R=5 (0.077%) — its "sweet spot." It
   degrades modestly at other parameters but STILL beats Sanchez N=2
   at every σ_t·R except extremely thick optical cells (≥20 MFP) where
   both closures converge to ~0.45% residual.

4. **Sanchez N=2 is consistently WORSE than F.4 N=1 at practical cells.**
   At σ_t·R ∈ [1, 10] (typical fuel-pin regime), F.4 is 1-13× better.

5. **The Wigner-Seitz identity IS mathematically exact** for white BC
   on both surfaces of a homogeneous hollow sphere (no net leakage,
   k_eff = k_inf). So the residual at rank-∞ should converge to 0.
   The plateau at ~1.42% for Sanchez indicates something fundamentally
   structural is missing in the canonical rank-N per-face closure —
   NOT a simple prefactor or basis-choice bug.

6. **Production recommendation**: F.4's scalar rank-2 is the best
   closure we've found. It's < 1% residual at practical cells and
   bit-exact for slab. Rank-N per-face for hollow sphere is an OPEN
   RESEARCH PROBLEM and the NotImplementedError guard in
   `build_closure_operator` must stay.

This script combines the refinement / N-scan / parameter-scan tests
that produced the above data. Promote to tests/derivations/ if
quantitative gates are wanted.
"""
from __future__ import annotations
import sys
import numpy as np
import pytest

sys.path.insert(0, "/workspaces/ORPHEUS")
sys.path.insert(0, "/workspaces/ORPHEUS/derivations/diagnostics")

from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry, composite_gl_r, build_volume_kernel,
    build_closure_operator,
)
from diag_sanchez_N_convergence import build_K_bc, solve_k_eff


def solve_k_eff_from_K(K, sig_t_v, sig_s_v, nu_sig_f_v):
    import scipy.linalg as sla
    N_r = K.shape[0]
    A = np.diag(sig_t_v * np.ones(N_r)) - K * sig_s_v
    B = K * nu_sig_f_v
    eigvals = sla.eigvals(np.linalg.solve(A, B))
    return float(np.max(np.real(eigvals)))


def f4_k_eff_at(R, r_0, sig_t_v, sig_s_v, nu_sig_f_v):
    """F.4 scalar rank-2 k_eff for hollow sphere."""
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 4, dps=15, inner_radius=r_0,
    )
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t,
        n_angular=24, n_rho=24, dps=15,
    )
    op = build_closure_operator(
        geom, r_nodes, r_wts, radii, sig_t,
        reflection="white", n_angular=24, n_surf_quad=24, dps=15,
    )
    return solve_k_eff_from_K(K_vol + op.as_matrix(), sig_t_v, sig_s_v, nu_sig_f_v)


def sanchez_k_eff_at(R, r_0, sig_t_v, sig_s_v, nu_sig_f_v, N=2):
    """Sanchez µ-ortho recipe k_eff at rank N."""
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 4, dps=15, inner_radius=r_0,
    )
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t,
        n_angular=24, n_rho=24, dps=15,
    )
    K_bc = build_K_bc(
        geom, r_nodes, r_wts, radii, sig_t, N,
        n_angular=24, n_surf_quad=24, dps=15,
    )
    return solve_k_eff(K_bc, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)


@pytest.mark.slow
def test_sanchez_plateau_persists_under_radial_refinement():
    """Radial-mesh refinement does not close the Sanchez 1.42% plateau.

    Confirms the plateau is a closure-structure issue, not quadrature.
    """
    sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
    k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
    R, r_0 = 5.0, 1.5
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])
    errs = {}
    for n_p, p_ord in [(2, 4), (4, 6), (8, 6)]:
        r_nodes, r_wts, panels = composite_gl_r(
            radii, n_p, p_ord, dps=15, inner_radius=r_0,
        )
        K_vol = build_volume_kernel(
            geom, r_nodes, panels, radii, sig_t,
            n_angular=24, n_rho=24, dps=15,
        )
        K_bc = build_K_bc(
            geom, r_nodes, r_wts, radii, sig_t, 2,
            n_angular=24, n_surf_quad=24, dps=15,
        )
        k = solve_k_eff(K_bc, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        errs[(n_p, p_ord)] = abs(k - k_inf) / k_inf * 100
    # Plateau floor ~1.31%, finest mesh still well above 0.1%
    assert all(e > 1.0 for e in errs.values()), (
        f"Residuals {errs} — plateau may have closed, retire test."
    )


@pytest.mark.slow
def test_f4_beats_sanchez_at_practical_cells():
    """F.4 scalar N=1 is consistently better than Sanchez N=2 at
    practical optical thicknesses (σ_t·R ∈ [1, 10]).

    At σ_t·R > 10 the two converge.
    """
    sig_s_rel = 0.5
    nu_sig_f_rel = 0.75
    k_inf = nu_sig_f_rel / (1.0 - sig_s_rel)
    R, r_0 = 5.0, 1.5
    for sig_t_v in (0.5, 1.0, 2.0):  # σ_t·R ∈ [2.5, 5, 10]
        sig_s_v = sig_s_rel * sig_t_v
        nu_sig_f_v = nu_sig_f_rel * sig_t_v
        k_f4 = f4_k_eff_at(R, r_0, sig_t_v, sig_s_v, nu_sig_f_v)
        k_s2 = sanchez_k_eff_at(R, r_0, sig_t_v, sig_s_v, nu_sig_f_v, N=2)
        err_f4 = abs(k_f4 - k_inf) / k_inf * 100
        err_s2 = abs(k_s2 - k_inf) / k_inf * 100
        assert err_f4 < err_s2 + 0.1, (
            f"σ_t·R = {sig_t_v*R}: F.4 N=1 err={err_f4:.3f}% should beat "
            f"Sanchez N=2 err={err_s2:.3f}% (within 0.1% margin)."
        )


def main():
    print("Rank-N hollow-sphere closure characterization")
    print(f"{'σ_t·R':>8} {'F.4 N=1':>10} {'Sanchez N=2':>13} {'Sanchez N=3':>13}")
    print("-" * 50)
    sig_s_rel = 0.5
    nu_sig_f_rel = 0.75
    k_inf = nu_sig_f_rel / (1.0 - sig_s_rel)
    R, r_0 = 5.0, 1.5
    for sig_t_v in (0.2, 0.5, 1.0, 2.0, 4.0):
        sig_s_v = sig_s_rel * sig_t_v
        nu_sig_f_v = nu_sig_f_rel * sig_t_v
        k_f4 = f4_k_eff_at(R, r_0, sig_t_v, sig_s_v, nu_sig_f_v)
        err_f4 = abs(k_f4 - k_inf) / k_inf * 100
        errs_s = {}
        for N in (2, 3):
            k = sanchez_k_eff_at(R, r_0, sig_t_v, sig_s_v, nu_sig_f_v, N=N)
            errs_s[N] = abs(k - k_inf) / k_inf * 100
        print(f"{sig_t_v*R:>8.1f} {err_f4:>9.3f}% "
              f"{errs_s[2]:>12.3f}% {errs_s[3]:>12.3f}%")


if __name__ == "__main__":
    main()
