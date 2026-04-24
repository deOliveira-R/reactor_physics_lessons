"""Diagnostic — Issue #129 Stage 1b: hold L fixed, vary R independently.

The Stage 1a R-scan held L/R fixed (so L grows with R). This is the
"keep the ratio constant" reading of the original plan. But it
conflates two effects: as R grows the curvature shrinks (good), but as
L also grows the slab itself becomes optically thicker (changes k_eff
absolute scale). To isolate curvature, hold L fixed at the small probe
value (L = 0.001) and let R range over [1, 10, 100, 1000].

Then: r_0 = R - L, so r_0/R = 1 - L/R = 1, 0.9999, 0.999999, …. The
inner cavity becomes geometrically more planar, AND R → ∞ tells us
whether curvature in the cylinder kernel itself washes out.

Created by numerics-investigator on 2026-04-23 for Issue #129.

INFORMATIONAL — does not assert pass/fail. Promotion guidance: not a
permanent test; it's a one-shot characterisation.
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
L_FIXED = 1.0e-3   # Slab thickness — held constant. r_0 = R - L_FIXED.


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
def test_issue129_stage1b_fixed_L():
    """Hold L = 1e-3 fixed, vary R ∈ [1, 10, 100, 1000].

    Slab k_eff is computed once and reused (independent of R).
    Cylinder k_eff varies because curvature changes with R.
    """
    print("\n=== Issue #129 Stage 1b — fixed L, varying R ===")
    print(f"  Material: Σ_t={SIG_T[0]}, Σ_s={SIG_S[0]}, νΣ_f={NU_SIG_F[0]}")
    print(f"  L fixed = {L_FIXED}")
    print()

    t0 = time.time()
    k_slab = _solve_slab(L_FIXED)
    t_slab = time.time() - t0
    print(f"  Slab L={L_FIXED}: k_eff = {k_slab:.10e}  (wall {t_slab:.1f}s)")
    print()

    print(f"{'R':>10} {'r_0/R':>14} {'k_cyl':>16} "
          f"{'rel_diff':>14} {'wall(s)':>10}")
    print("-" * 76)

    rows = []
    for R in (1.0, 10.0, 100.0, 1000.0):
        r_0 = R - L_FIXED
        t0 = time.time()
        try:
            k_cyl = _solve_hollow_cyl(r_0, R)
        except Exception as exc:
            print(f"  FAILED at R={R}: {exc}")
            continue
        wall = time.time() - t0
        rel_diff = abs(k_slab - k_cyl) / max(abs(k_slab), 1e-30)
        rows.append((R, r_0 / R, k_cyl, rel_diff, wall))
        print(f"{R:>10.4g} {r_0/R:>14.10f} {k_cyl:>16.10e} "
              f"{rel_diff:>14.4e} {wall:>10.2f}")

    # Diagnostic: how does rel_diff scale with R for fixed L?
    print()
    if len(rows) >= 2:
        for i in range(1, len(rows)):
            r_prev, _, _, e_prev, _ = rows[i - 1]
            r_now, _, _, e_now, _ = rows[i]
            ratio = e_prev / max(e_now, 1e-30)
            R_ratio = r_now / r_prev
            print(f"  R: {r_prev:g} → {r_now:g} (×{R_ratio:g})  "
                  f"rel_diff: {e_prev:.3e} → {e_now:.3e} "
                  f"(shrunk ×{ratio:.2g})")

    # If gap → 0 with R → ∞ (fixed L), curvature is THE source of
    # the gap, and the R-fixed-L planar limit IS the right limit.
    last = rows[-1] if rows else None
    if last is not None:
        print(f"\n  At R={last[0]}: rel_diff = {last[3]:.3e}, "
              f"k_cyl→k_slab = {last[2]:.10e} → {k_slab:.10e}")


if __name__ == "__main__":
    test_issue129_stage1b_fixed_L()
