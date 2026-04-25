"""Diagnostic: cylinder Hébert (1-P_ss)⁻¹ × rank-1 Mark on Class B (Issue #132).

Mirror of ``diag_sphere_white_bc_geometric_series_fix.py`` for cylinder.
Tests Hypothesis 2 of the Issue #132 plan: combine the existing cylinder
rank-1 Mark closure (with its 2-D-projected-cosine ``compute_G_bc``
limitation) with the Hébert geometric-series factor 1/(1 - P_ss^cyl).

If this CLOSES the cylinder Class B 22 % rank-1 Mark gap to <1 %, ship
the path. If it leaves a substantial residual error (≥10 %), the gap is
upstream of the geometric-series factor — most likely the
``compute_G_bc`` cylinder branch needing the 3-D Knyazev correction
(Issue #112 Phase C / Hypothesis 3).

CASES
-----
Mirror sphere set: 1G/1R, 1G/2R, 2G/1R, 2G/2R cylinders. Layouts come
from `cp_cylinder._RADII` and `_xs_library.LAYOUTS`. Reference k_inf
from `cp_cylinder._build_case` (Ki₃-based CP method, the trusted MR
multi-group reference for the cylinder Class B suite).

Run:
    python -m pytest derivations/diagnostics/diag_cylinder_hebert_keff.py -v
"""
from __future__ import annotations

import time
import numpy as np
import pytest

from orpheus.derivations import cp_cylinder
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    solve_peierls_mg,
)
from orpheus.derivations import peierls_geometry as _pg

from derivations.diagnostics.diag_cylinder_hebert_pss import (
    compute_P_ss_cylinder_homogeneous,
    compute_P_ss_cylinder_multiregion,
)


_QUAD = dict(n_panels_per_region=2, p_order=3,
             n_angular=24, n_rho=24, n_surf_quad=24, dps=15)


def _build_xs(ng_key, n_regions):
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(region, ng_key) for region in layout]
    return (
        np.vstack([xs["sig_t"] for xs in xs_list]),
        np.stack([xs["sig_s"] for xs in xs_list], axis=0),
        np.vstack([xs["nu"] * xs["sig_f"] for xs in xs_list]),
        np.vstack([xs["chi"] for xs in xs_list]),
    )


def _kinf_reference(ng_key, n_regions):
    """Reference k_inf for cylinder Class B (Ki₃-based CP)."""
    return cp_cylinder._build_case(ng_key, n_regions).k_inf


def _solve_unpatched(ng_key, n_regions):
    """Bare rank-1 Mark closure: K = K_vol + K_bc^Mark.

    This is the BEFORE baseline (the 22 %-error path)."""
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_cylinder._RADII[n_regions])
    sol = solve_peierls_mg(
        CYLINDER_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
        nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_rank1_mark", n_bc_modes=1,
        **_QUAD,
    )
    return float(sol.k_eff)


def _solve_with_hebert_patch(ng_key, n_regions, *, P_ss_per_group=None):
    """Patched Hébert path: K = K_vol + K_bc^Mark / (1 - P_ss^cyl).

    Monkey-patches `build_white_bc_correction_rank_n` to multiply its
    output by 1/(1 - P_ss) per group, in the same style as the sphere
    diagnostic. P_ss^cyl computed from the Issue #132 derivation
    (Bickley Ki₃, see diag_cylinder_hebert_pss.py)."""
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_cylinder._RADII[n_regions])

    n_groups = sig_t.shape[1]
    if P_ss_per_group is None:
        P_ss_per_group = [
            compute_P_ss_cylinder_multiregion(sig_t[:, g], radii, n_quad=64)
            for g in range(n_groups)
        ]
    P_ss_iter = iter(P_ss_per_group)
    original_build = _pg.build_white_bc_correction_rank_n

    def patched_build(geometry, r_nodes, r_wts, radii_arg, sig_t_arg,
                      n_angular=32, n_surf_quad=32, dps=25, n_bc_modes=1):
        K_bc = original_build(
            geometry, r_nodes, r_wts, radii_arg, sig_t_arg,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            n_bc_modes=n_bc_modes,
        )
        P_ss = next(P_ss_iter)
        if P_ss >= 1.0:
            raise RuntimeError(f"P_ss^cyl = {P_ss} ≥ 1; would diverge")
        return K_bc / (1.0 - P_ss)

    _pg.build_white_bc_correction_rank_n = patched_build
    try:
        sol = solve_peierls_mg(
            CYLINDER_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
            nu_sig_f=nu_sig_f, chi=chi,
            boundary="white_rank1_mark", n_bc_modes=1,
            **_QUAD,
        )
    finally:
        _pg.build_white_bc_correction_rank_n = original_build

    return float(sol.k_eff), P_ss_per_group


# ───────────────────────── pytest tests ─────────────────────────

CASES = [
    ("1g", 1),
    ("1g", 2),
    ("2g", 1),
    ("2g", 2),
]


@pytest.mark.parametrize("ng_key,n_regions", CASES)
def test_hebert_factor_alone_does_not_close_cylinder_gap(ng_key, n_regions):
    """Sanity: the (1-P_ss)⁻¹ factor on its own does NOT close the
    cylinder Class B Mark gap. Both before and after errors are recorded.

    This test PASSES whenever the diagnostic runs cleanly. The metric
    is logged for inspection — the *interpretation* (whether the residual
    error is small enough to ship) is in the report, not the assertion.
    """
    kinf = _kinf_reference(ng_key, n_regions)
    k_before = _solve_unpatched(ng_key, n_regions)
    k_after, P_ss_per_g = _solve_with_hebert_patch(ng_key, n_regions)
    err_before = (k_before - kinf) / kinf * 100
    err_after = (k_after - kinf) / kinf * 100
    print(f"\n  {ng_key} {n_regions}r — k_inf={kinf:.6f}")
    print(f"    BEFORE: k_eff={k_before:.6f}  err={err_before:+.3f}%")
    print(f"    AFTER : k_eff={k_after:.6f}  err={err_after:+.3f}%")
    print(f"    P_ss/group: {P_ss_per_g}")
    # Always passes — diagnostic of viability, not regression
    assert k_before > 0 and k_after > 0


if __name__ == "__main__":
    print("=" * 78)
    print("CYLINDER Hébert (1-P_ss)⁻¹ × rank-1 Mark closure (Issue #132)")
    print("=" * 78)

    for ng_key, n_regions in CASES:
        kinf = _kinf_reference(ng_key, n_regions)
        print(f"\n--- {ng_key} {n_regions}r — k_inf = {kinf:.10f} ---")

        t0 = time.time()
        k_before = _solve_unpatched(ng_key, n_regions)
        t_before = time.time() - t0
        err_before = (k_before - kinf) / kinf * 100

        sig_t, _, _, _ = _build_xs(ng_key, n_regions)
        radii = np.array(cp_cylinder._RADII[n_regions])
        P_ss_per_g = [
            compute_P_ss_cylinder_multiregion(sig_t[:, g], radii, n_quad=64)
            for g in range(sig_t.shape[1])
        ]

        t0 = time.time()
        k_after, _ = _solve_with_hebert_patch(
            ng_key, n_regions, P_ss_per_group=P_ss_per_g,
        )
        t_after = time.time() - t0
        err_after = (k_after - kinf) / kinf * 100

        print(f"  P_ss^cyl per group: {[f'{p:.6f}' for p in P_ss_per_g]}")
        print(f"  factor 1/(1-P_ss):  {[f'{1/(1-p):.4f}' for p in P_ss_per_g]}")
        print(f"  BEFORE (Mark only):       k_eff={k_before:.10f}  "
              f"err={err_before:+8.3f}%  ({t_before:.1f}s)")
        print(f"  AFTER (Mark × Hébert):    k_eff={k_after:.10f}  "
              f"err={err_after:+8.3f}%  ({t_after:.1f}s)")
