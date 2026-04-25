"""Probe C: rank-N on Class B sphere with HOMOGENEOUS Σ_t masquerading as 2R.

Created by numerics-investigator on 2026-04-24.

If this test catches a real bug, promote to ``tests/derivations/test_peierls_rank_n_bc.py``.

Hypothesis under test (per probe-cascade SKILL §"drop one factor at a time"):
    The 1G/2R sphere rank-2 +57 % gap is *MR-coupled*, but is it
    coupled to the **inter-region Σ_t difference** (i.e., σ_t,A=1 vs σ_t,B=2)
    or only to the **multi-region routing path** (i.e., len(radii)>1 hitting
    a different code branch)?

Test design — three configurations, all `n_bc_modes ∈ {1, 2, 3}`, sphere:
  X) radii=[1.0],         sig_t=[1.0]               — true 1R  (control)
  Y) radii=[0.5, 1.0],    sig_t=[1.0, 1.0]          — homog masquerading as 2R
  Z) radii=[0.5, 1.0],    sig_t=[1.0, 2.0]          — true 2R (the failure case)

Reference: bare 1G white-BC sphere with **uniform Σ_t = 1**, k_inf is
analytical: k = ν Σ_f / Σ_a (regardless of flux shape) when there's no leakage.
For sphere R=1.0 MFP at uniform Σ_t=1 and the standard 1G XS (sig_t=1, sig_c=0.2,
sig_f=0.3, sig_s=0.5, ν=2.5) → k_inf = 1.5. Cases X and Y must give IDENTICAL
k_eff (geometry/XS are functionally identical). If Y differs from X — the bug
is in the MR routing path **independently of any Σ_t breakpoint**.

This is the cleanest possible H_B test.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    solve_peierls_mg,
)


_QUAD = dict(
    n_panels_per_region=2, p_order=3,
    n_angular=24, n_rho=24, n_surf_quad=24, dps=15,
)


def _solve_sphere(radii, sig_t_per_region, n_bc_modes, ng_key="1g"):
    """Solve sphere with given (radii, sig_t array per region).

    `sig_t_per_region` is shape (n_regions, ng); we build sig_s, nu_sig_f
    from get_xs("A", ng_key) for ALL regions so material A defines the
    scatter / fission / chi.
    """
    n_regions = len(radii)
    ng = len(sig_t_per_region[0])
    xs_A = get_xs("A", ng_key)
    sig_s = np.stack([xs_A["sig_s"]] * n_regions, axis=0)
    nu_sig_f = np.vstack([xs_A["nu"] * xs_A["sig_f"]] * n_regions)
    chi = np.vstack([xs_A["chi"]] * n_regions)
    sig_t = np.array(sig_t_per_region, dtype=float).reshape(n_regions, ng)

    sol = solve_peierls_mg(
        SPHERE_1D, radii=np.array(radii), sig_t=sig_t,
        sig_s=sig_s, nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_rank1_mark", n_bc_modes=n_bc_modes,
        **_QUAD,
    )
    return sol.k_eff


@pytest.mark.parametrize("n_bc_modes", [1, 2, 3])
def test_probe_c_homogeneous_2r_matches_1r(n_bc_modes):
    """Sphere with radii=[0.5, 1.0] and uniform σ_t=1 must match radii=[1.0]."""
    k_X = _solve_sphere([1.0], [[1.0]], n_bc_modes=n_bc_modes)
    k_Y = _solve_sphere([0.5, 1.0], [[1.0], [1.0]], n_bc_modes=n_bc_modes)
    rel_diff = abs(k_X - k_Y) / max(abs(k_X), 1e-30)
    print(
        f"\n  rank-{n_bc_modes}: k_X(1R, σ=1) = {k_X:.10f}, "
        f"k_Y(2R-homog, σ=1) = {k_Y:.10f}, rel_diff = {rel_diff:.3e}"
    )
    # Quadrature-floor tolerance — these should be near-identical
    # because the geometry is functionally identical.
    assert rel_diff < 1e-3, (
        f"rank-{n_bc_modes} MR-routing bug: k_eff(homog 2R) differs from "
        f"k_eff(1R) by {rel_diff:.3e} relative — this is a routing-path "
        f"bug independent of any Σ_t breakpoint."
    )


def test_probe_c_full_table():
    """Print the full X, Y, Z table for visual inspection."""
    print()
    print("=" * 76)
    print("Probe C: sphere rank-N homogeneous-2R vs true-1R vs true-2R")
    print("  (XS = A 1G; Σ_t,A=1, Σ_t,B=2; BASE quadrature)")
    print("=" * 76)
    cases = [
        ("X: radii=[1.0],       σ=[1.0]",     [1.0],         [[1.0]]),
        ("Y: radii=[0.5,1.0],   σ=[1.0,1.0]", [0.5, 1.0],    [[1.0], [1.0]]),
        ("Z: radii=[0.5,1.0],   σ=[1.0,2.0]", [0.5, 1.0],    [[1.0], [2.0]]),
    ]
    print(f"\n  {'case':<36}  {'rank-1':>14} {'rank-2':>14} {'rank-3':>14}")
    for label, radii, sig in cases:
        ks = []
        for n_bc in (1, 2, 3):
            ks.append(_solve_sphere(radii, sig, n_bc_modes=n_bc))
        print(f"  {label:<36}  {ks[0]:>14.10f} {ks[1]:>14.10f} {ks[2]:>14.10f}")


if __name__ == "__main__":
    test_probe_c_full_table()
    for n in (1, 2, 3):
        try:
            test_probe_c_homogeneous_2r_matches_1r(n)
            print(f"  rank-{n}: PASS")
        except AssertionError as e:
            print(f"  rank-{n}: FAIL — {e}")
