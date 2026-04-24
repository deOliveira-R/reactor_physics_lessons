"""Diagnostic — Issue #129 Stage 1: R-scaling of the planar-limit gap.

Probes whether the slab vs hollow-cylinder k_eff disagreement at fixed
``L/R = 0.001`` shrinks as ``R`` grows. The original Phase G.4 plan
claimed agreement at 1e-8 for ``r_0/R → 1``; the empirical baseline
shows 22 % at ``R=1``.

Hypothesis 1: a meaningful planar limit needs ``R → ∞`` at fixed ``L``,
not just ``r_0/R → 1`` at fixed ``R``. If the gap shrinks roughly like
``L/R`` as ``R`` grows, hypothesis 1 is correct and the original plan
just had the wrong limit specification.

Created by numerics-investigator on 2026-04-23 for Issue #129.

This diagnostic is INFORMATIONAL — it does not assert a passing
condition. The "expected" outcome is a cleanly tabulated R-scan that
Stage 2/3 can reason from. We mark it ``slow`` because each cylinder
solve at high R is non-trivial (chord lengths are long).

Promotion guidance: this is a one-shot characterisation, not a
regression test. Do NOT promote to the test suite. Stage 3 will
specify the actual regression test (if any).
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    CurvilinearGeometry,
    SLAB_POLAR_1D,
    solve_peierls_1g,
)


# Material — same as Issue #129 baseline.
SIG_T = np.array([1.0])
SIG_S = np.array([0.4])
NU_SIG_F = np.array([0.6])
LR_RATIO = 1.0e-3   # L/R fixed, sweep R itself.


def _solve_slab(L: float) -> tuple[float, float]:
    """Slab L thickness, vacuum BC, unified path. Return (k_eff, wall)."""
    t0 = time.time()
    sol = solve_peierls_1g(
        SLAB_POLAR_1D,
        radii=np.array([L]),
        sig_t=SIG_T,
        sig_s=SIG_S,
        nu_sig_f=NU_SIG_F,
        boundary="vacuum",
        n_panels_per_region=1,
        p_order=3,
        n_angular=24,
        n_rho=24,
        dps=20,
    )
    return sol.k_eff, time.time() - t0


def _solve_hollow_cyl(r_0: float, R: float) -> tuple[float, float]:
    """Hollow cylinder annulus [r_0, R], vacuum BC. Return (k_eff, wall)."""
    geom = CurvilinearGeometry(kind="cylinder-1d", inner_radius=r_0)
    t0 = time.time()
    sol = solve_peierls_1g(
        geom,
        radii=np.array([R]),
        sig_t=SIG_T,
        sig_s=SIG_S,
        nu_sig_f=NU_SIG_F,
        boundary="vacuum",
        n_panels_per_region=1,
        p_order=3,
        n_angular=24,
        n_rho=24,
        dps=20,
    )
    return sol.k_eff, time.time() - t0


@pytest.mark.slow
def test_issue129_stage1_rscan():
    """R-scan at fixed L/R=1e-3.

    Expected R values: 1, 10, 100. At each:
      - slab L=L_R = LR_RATIO * R
      - hollow cyl r_0 = R - L_R, outer R
    Compare k_eff. Hypothesis 1 says rel_diff → 0 as R → ∞.
    """
    print("\n=== Issue #129 Stage 1 — R-scan at fixed L/R = 1e-3 ===")
    print(f"  Material: Σ_t={SIG_T[0]}, Σ_s={SIG_S[0]}, νΣ_f={NU_SIG_F[0]}")
    print(f"  L/R fixed = {LR_RATIO}")
    print()
    print(f"{'R':>10} {'L':>14} {'r_0/R':>10} {'k_slab':>16} "
          f"{'k_cyl':>16} {'rel_diff':>12} {'wall(s)':>10}")
    print("-" * 100)

    rows = []
    for R in (1.0, 10.0, 100.0):
        L = LR_RATIO * R
        r_0 = R - L
        k_slab, wall_s = _solve_slab(L)
        k_cyl, wall_c = _solve_hollow_cyl(r_0, R)
        rel_diff = abs(k_slab - k_cyl) / max(abs(k_slab), 1e-30)
        wall = wall_s + wall_c
        rows.append((R, L, r_0 / R, k_slab, k_cyl, rel_diff, wall))
        print(f"{R:>10.3g} {L:>14.6e} {r_0/R:>10.6f} "
              f"{k_slab:>16.10e} {k_cyl:>16.10e} "
              f"{rel_diff:>12.4e} {wall:>10.2f}")

    # Informational: is the gap shrinking?
    print()
    if len(rows) >= 2:
        for i in range(1, len(rows)):
            r_prev, _, _, _, _, e_prev, _ = rows[i - 1]
            r_now, _, _, _, _, e_now, _ = rows[i]
            ratio = e_prev / max(e_now, 1e-30)
            R_ratio = r_now / r_prev
            print(f"  R: {r_prev:g} → {r_now:g} (×{R_ratio:g})  "
                  f"rel_diff: {e_prev:.3e} → {e_now:.3e} "
                  f"(shrunk ×{ratio:.2g})  "
                  f"vs. naive 1/R: {R_ratio:.2g}, vs. 1/R²: {R_ratio**2:.2g}")

    # Stage 1 success criterion (informational only):
    # If gap shrinks proportionally to 1/R or faster, hypothesis 1
    # confirmed. If gap stays > 10 % even at R=100, hypothesis 1 is
    # incomplete and a structural reason remains.
    biggest_gap_at_R100 = rows[-1][5]
    assert rows is not None, "diagnostic completed"
    print()
    print(f"  At R=100: rel_diff = {biggest_gap_at_R100:.3e}")
    if biggest_gap_at_R100 < 1e-4:
        print("  → Hypothesis 1 strongly supported: gap shrinks to 1e-4 at R=100.")
    elif biggest_gap_at_R100 < 1e-2:
        print("  → Hypothesis 1 partially supported.")
    else:
        print("  → Hypothesis 1 not supported on its own.")


if __name__ == "__main__":
    test_issue129_stage1_rscan()
