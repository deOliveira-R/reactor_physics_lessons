"""Probe: rank-N Marshak white-BC on Class B (solid cyl/sph), MR×MG.

Per plan §5 (`.claude/plans/issue-100-103-rank-n-class-b-multi-region.md`),
this script runs the load-bearing 2G/2R rank-N sweep to discriminate
between three outcomes:

- A (clean extension): rank-N gap in MR×MG mirrors the 1G/1R floor.
- B (hidden bug, Issue #131-style): MR×MG gap diverges from 1G/1R
  pattern — gap grows with Σ_t,A − Σ_t,B, or sign flip between
  adjacent ranks, etc.
- C (rank-N actually beats rank-1 Mark): would invalidate the
  2026-04-22 single-region falsification.

Quadrature: BASE preset (n_panels_per_region=2, p_order=3, n_angular=24,
n_rho=24, n_surf_quad=24, dps=15). RICH only if BASE shows interesting
signal worth refining.

Run from repo root: ``python derivations/diagnostics/diag_class_b_rank_n_probe.py``
"""

from __future__ import annotations

import time

import numpy as np

from orpheus.derivations import cp_cylinder, cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SPHERE_1D,
    solve_peierls_mg,
)

_QUAD_BASE = dict(
    n_panels_per_region=2,
    p_order=3,
    n_angular=24,
    n_rho=24,
    n_surf_quad=24,
    dps=15,
)


def _build_xs(ng_key: str, n_regions: int):
    """Mirror cp_cylinder._build_case body — produce the 4 arrays."""
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t = np.vstack([xs["sig_t"] for xs in xs_list])
    sig_s = np.stack([xs["sig_s"] for xs in xs_list], axis=0)
    nu_sig_f = np.vstack([xs["nu"] * xs["sig_f"] for xs in xs_list])
    chi = np.vstack([xs["chi"] for xs in xs_list])
    return sig_t, sig_s, nu_sig_f, chi


def _solve(geometry, ng_key, n_regions, n_bc_modes, *, radii_key):
    cp_module = cp_cylinder if radii_key == "cyl" else cp_sphere
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_module._RADII[n_regions])
    t0 = time.time()
    sol = solve_peierls_mg(
        geometry,
        radii=radii,
        sig_t=sig_t,
        sig_s=sig_s,
        nu_sig_f=nu_sig_f,
        chi=chi,
        boundary="white_rank1_mark",
        n_bc_modes=n_bc_modes,
        **_QUAD_BASE,
    )
    return sol.k_eff, time.time() - t0


def _ref_kinf(geometry_label, ng_key, n_regions):
    cp_module = cp_cylinder if geometry_label == "cyl" else cp_sphere
    return cp_module._build_case(ng_key, n_regions).k_inf


def main():
    print("=" * 72)
    print("Class B rank-N MR×MG probe — BASE quadrature")
    print(f"  preset = {_QUAD_BASE}")
    print("=" * 72)

    for geom_label, geometry in [("cyl", CYLINDER_1D), ("sph", SPHERE_1D)]:
        print(f"\n--- {geom_label.upper()} ---")
        for ng_key, n_regions in [
            ("1g", 1),  # sanity baseline (Issue #112 table)
            ("2g", 1),  # 2G/1R sanity
            ("1g", 2),  # MR sanity
            ("2g", 2),  # the load-bearing case
        ]:
            kinf = _ref_kinf(geom_label, ng_key, n_regions)
            print(
                f"\n{ng_key} {n_regions}r — k_inf={kinf:.10f}  "
                f"(geometry={geom_label})"
            )
            print(f"  {'N':>3} {'k_eff':>16} {'err [%]':>10} {'wall [s]':>10}")
            # cylinder N≥3 known-divergent at thin cell per Issue #112
            ranks = (1, 2, 3, 5) if geom_label == "cyl" else (1, 2, 3, 5, 8)
            for n in ranks:
                k_eff, dt = _solve(
                    geometry, ng_key, n_regions, n, radii_key=geom_label,
                )
                err_pct = 100.0 * (k_eff - kinf) / kinf
                print(f"  {n:>3} {k_eff:>16.10f} {err_pct:>10.3f} {dt:>10.2f}")


if __name__ == "__main__":
    main()
