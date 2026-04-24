"""Diagnostic — Issue #129 Stage 1c: scan L at large R to expose the plateau.

Stage 1b showed that holding L=1e-3 fixed and growing R → 1000 makes
the rel_diff plateau at ~3 %. The gap does NOT close to zero. This
strongly suggests the gap is structural — the cylinder Ki₁ kernel and
the slab E₁ kernel see different ray-distribution physics in any
limit, and the gap floor depends on L itself, not on R.

Plan: at large fixed R = 1000 (curvature negligible), scan L over
[1e-4, 1e-3, 1e-2, 1e-1, 1, 10]. If the plateau gap depends on L
through e.g. (Σ_t·L)^{1/2}, we have a structural curvature-correction
even in the planar limit.

Also: at fixed L (small, e.g. L=0.01), check what happens at R far
beyond R=1000 (e.g. R=1e4) — does the plateau truly converge to a
non-zero floor, or does it eventually decay slowly?

Created by numerics-investigator on 2026-04-23 for Issue #129.
INFORMATIONAL.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    SLAB_POLAR_1D,
    solve_peierls_1g,
)


SIG_T = np.array([1.0])
SIG_S = np.array([0.4])
NU_SIG_F = np.array([0.6])


def _solve_slab(L: float) -> float:
    sol = solve_peierls_1g(
        SLAB_POLAR_1D,
        radii=np.array([L]),
        sig_t=SIG_T, sig_s=SIG_S, nu_sig_f=NU_SIG_F,
        boundary="vacuum",
        n_panels_per_region=1, p_order=3,
        n_angular=24, n_rho=24, dps=20,
    )
    return sol.k_eff


def _solve_hollow_cyl(r_0: float, R: float) -> float:
    geom = CurvilinearGeometry(kind="cylinder-1d", inner_radius=r_0)
    sol = solve_peierls_1g(
        geom,
        radii=np.array([R]),
        sig_t=SIG_T, sig_s=SIG_S, nu_sig_f=NU_SIG_F,
        boundary="vacuum",
        n_panels_per_region=1, p_order=3,
        n_angular=24, n_rho=24, dps=20,
    )
    return sol.k_eff


@pytest.mark.slow
def test_issue129_stage1c_L_scan_at_large_R():
    """Scan L at fixed large R = 1000."""
    print("\n=== Issue #129 Stage 1c — L scan at fixed large R = 1000 ===")
    print(f"  Material: Σ_t={SIG_T[0]}, Σ_s={SIG_S[0]}, νΣ_f={NU_SIG_F[0]}")
    R_LARGE = 1000.0
    print(f"  R fixed (large) = {R_LARGE}")
    print()
    print(f"{'L':>14} {'L/R':>14} {'k_slab':>16} {'k_cyl':>16} "
          f"{'rel_diff':>14} {'sqrt(Σt·L)':>14}")
    print("-" * 105)

    for L in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        r_0 = R_LARGE - L
        t0 = time.time()
        k_slab = _solve_slab(L)
        try:
            k_cyl = _solve_hollow_cyl(r_0, R_LARGE)
        except Exception as exc:
            print(f"  L={L} FAILED: {exc}")
            continue
        wall = time.time() - t0
        rel_diff = abs(k_slab - k_cyl) / max(abs(k_slab), 1e-30)
        sqrtL = float(np.sqrt(SIG_T[0] * L))
        print(f"{L:>14.4e} {L/R_LARGE:>14.4e} {k_slab:>16.10e} "
              f"{k_cyl:>16.10e} {rel_diff:>14.4e} {sqrtL:>14.4e}  "
              f"(wall {wall:.1f}s)")

    # If rel_diff scales as sqrt(Σ_t·L), the gap is the in-plane
    # tangential-chord term — fundamental, no clean limit.
    # If rel_diff scales linearly in L/R, hypothesis 1 still alive.


if __name__ == "__main__":
    test_issue129_stage1c_L_scan_at_large_R()
