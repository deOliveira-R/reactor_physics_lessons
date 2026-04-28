"""Diagnostic: thin-sphere multi-bounce specular k_eff ladder vs N.

Phase 1B of the visibility-cone-substitution rollout
(``.claude/plans/visibility-cone-substitution-rollout.md``).

The Phase 4 multi-bounce specular sphere closure overshoots k_inf for
N ≥ 4 on thin sphere — the matrix-Galerkin (I − T·R)^{-1} has an
unbounded operator norm in the continuous limit (verified in Phase 5
to be a structural pathology, not numerical noise). Pre-Phase 1B
baseline (commit 4dc03cf) measured:

  N=1: ≈ rank-1 hebert (well below k_inf)
  N=2: closer to k_inf
  N=3: very close
  N=4: +0.43 % overshoot
  N=6: +1.80 %
  N=8: +5.62 %

This diagnostic re-runs the same problem with the new vis-cone-
subdivided µ-quadrature in compute_T_specular_sphere and reports
the post-rollout overshoot at each N. The HOMOGENEOUS sphere has
N_reg=1 (no kinks), so vis-cone reduces to plain GL — overshoot
should be unchanged. To exercise the vis-cone path the diagnostic
also runs the same thin-cell parameters with a 2-region split
(r_1=R/2, same σ_t, same σ_s, same νΣ_f, same scattering ratio).
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations.geometry_topologies import SPHERE_POLAR_3D
from orpheus.derivations.peierls_geometry import solve_peierls_1g


def _kinf(sig_t: np.ndarray, sig_s: np.ndarray, nuf: np.ndarray) -> float:
    return float(nuf[0]) / (float(sig_t[0]) - float(sig_s[0, 0]))


def _run_ladder(label: str, R: float, radii: np.ndarray, sig_t: np.ndarray,
                sig_s: np.ndarray, nuf: np.ndarray, ranks=(1, 2, 3, 4, 6, 8)):
    print(f"\n=== {label} ===")
    print(f"  R = {R}, radii = {radii.tolist()}, σ_t = {sig_t.tolist()}")
    k_inf = _kinf(sig_t, sig_s, nuf)
    print(f"  k_inf = {k_inf:.8f}")
    for N in ranks:
        sol = solve_peierls_1g(
            SPHERE_POLAR_3D, radii, sig_t, sig_s, nuf,
            boundary="specular_multibounce", n_bc_modes=N,
            p_order=4, n_panels_per_region=2,
            n_angular=24, n_rho=24, n_surf_quad=24, dps=20,
            tol=1e-10,
        )
        rel = (sol.k_eff - k_inf) / k_inf
        sign = "+" if rel >= 0 else "−"
        print(f"  N={N}: k_eff = {sol.k_eff:.8f}  rel = {sign}{abs(rel)*100:.4f} %")


def main() -> None:
    # Fuel-A-like fixture (matches `homogeneous_fuel_A_1G` in the test suite).
    sig_t = np.array([0.5])
    sig_s = np.array([[0.4]])
    nuf = np.array([0.025])
    R = 5.0

    _run_ladder(
        "Homogeneous thin sphere (N_reg=1, vis-cone trivializes)",
        R, np.array([R]), sig_t, sig_s, nuf,
    )

    _run_ladder(
        "Two-region thin sphere (N_reg=2, exercises vis-cone subdivision)",
        R, np.array([R / 2.0, R]),
        np.array([float(sig_t[0]), float(sig_t[0])]),
        sig_s, nuf,
    )


if __name__ == "__main__":
    main()
