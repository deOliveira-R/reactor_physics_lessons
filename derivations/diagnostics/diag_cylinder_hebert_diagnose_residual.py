"""Diagnostic: localise the residual cylinder Hébert error to G_bc kernel
(Issue #132 Hypothesis 3 evidence).

Phase-2 diag_cylinder_hebert_keff.py shows that applying the
(1 - P_ss^cyl)⁻¹ factor to the cylinder rank-1 Mark closure REDUCES
the 22 % error to ~11 %, but does NOT close it to <1 % (sphere achieves
<1.5 %). The residual ≈10 % gap must come from the cylinder
``compute_G_bc`` itself — specifically the 2-D-projected-cosine Ki_1/d
form which lacks the 3-D polar-angle integration that produces
higher-order Bickley Ki_{2+k} (Knyazev 1993, Atomic Energy 74,
DOI 10.1007/BF00844623).

This script provides EVIDENCE for that diagnosis WITHOUT building the
Knyazev kernel (that's Issue #112 Phase C work). It does so by:

1. Computing the row-sum balance of K_bc^cyl (rank-1 Mark, then
   ×1/(1-P_ss)). The Hébert closure should give row-sum K·1 = 1 -
   P_esc on a flat-flux probe (white-BC partition-of-unity). For
   sphere this holds bit-exact after the geometric series. For cylinder,
   we expect a constant-but-wrong offset that quantifies the kernel bias.

2. Compute the same on a fixed-source 1G problem: σ_t = 1, ν Σ_f = 0,
   Σ_s = 1 (purely scattering), with white BC. The exact solution is
   φ = constant (white BC + uniform-Q feedback), and the rank-1 Mark
   should converge to the right magnitude; the residual after the
   geometric-series fix is the kernel bias.

If the kernel bias is ~10 % on 1G/1R, then the Knyazev 3-D polar-angle
correction is required to make the Hébert path useful for cylinder.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations import cp_cylinder
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D, SPHERE_1D,
    composite_gl_r,
    build_volume_kernel,
    build_white_bc_correction_rank_n,
    compute_P_esc, compute_G_bc, compute_P_ss_sphere,
)

import sys
sys.path.insert(0, "/workspaces/ORPHEUS")
from derivations.diagnostics.diag_cylinder_hebert_pss import (
    compute_P_ss_cylinder_homogeneous,
    compute_P_ss_cylinder_multiregion,
)


_QUAD = dict(n_panels_per_region=2, p_order=3,
             n_angular=24, n_rho=24, n_surf_quad=24, dps=15)


def assemble_K(geometry, sig_t_g, radii, *, with_hebert: bool, P_ss_func):
    """Assemble K = K_vol + K_bc[/ (1-P_ss)] for one group."""
    r_nodes, r_wts, panels = composite_gl_r(
        radii, _QUAD["n_panels_per_region"], _QUAD["p_order"],
    )
    K_vol = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t_g,
        n_angular=_QUAD["n_angular"], n_rho=_QUAD["n_rho"],
        dps=_QUAD["dps"],
    )
    K_bc = build_white_bc_correction_rank_n(
        geometry, r_nodes, r_wts, radii, sig_t_g,
        n_angular=_QUAD["n_angular"],
        n_surf_quad=_QUAD["n_surf_quad"],
        dps=_QUAD["dps"],
        n_bc_modes=1,
    )
    if with_hebert:
        P_ss = P_ss_func(sig_t_g, radii)
        K_bc = K_bc / (1.0 - P_ss)
    return K_vol, K_bc, r_nodes


def _Pss_sphere_func(sig_t_g, radii):
    return compute_P_ss_sphere(radii, sig_t_g, n_quad=64)


def _Pss_cyl_func(sig_t_g, radii):
    return compute_P_ss_cylinder_multiregion(sig_t_g, radii, n_quad=64)


def row_sum_partition_check(geometry, sig_t_g, radii, P_ss_func, label):
    """For Hébert closure, K·1 should equal Σ_t (so K φ = Σ_t φ on
    constant flux ⇒ flux balance via partition of unity:
    P_abs + P_leak * (1/(1-P_ss)) * <implicit return> = ... ).

    Concretely, for a TRUE Hébert white-BC closure, when φ = 1 the
    integrated source K · φ at each node must reproduce the steady-state
    constant flux: K_vol · 1 + K_bc^Hebert · 1 = Σ_t · 1 (in the
    appropriate normalisation). The fractional deviation quantifies the
    kernel bias."""
    K_vol, K_bc_heb, r_nodes = assemble_K(
        geometry, sig_t_g, radii, with_hebert=True, P_ss_func=P_ss_func,
    )
    K_total = K_vol + K_bc_heb
    one = np.ones(len(r_nodes))
    rhs = K_total @ one
    # Normalise by σ_t (1-region uniform: σ_t scalar, MR: pick avg)
    sig_t_avg = float(np.mean(sig_t_g))
    rel_dev = (rhs - sig_t_avg) / sig_t_avg
    print(f"\n  {label} K · 1 / σ_t (Hébert closure):")
    print(f"    nodes (radii): {r_nodes[:5]}... (showing 5)")
    print(f"    K·1 / σ_t    : {(rhs/sig_t_avg)[:5]}... (showing 5)")
    print(f"    rel dev      : min={rel_dev.min():+.4f}  "
          f"max={rel_dev.max():+.4f}  "
          f"mean={rel_dev.mean():+.4f}")
    return rel_dev


def fixed_source_pure_scatter(geometry, sig_t, radii, P_ss_func, label):
    """Fixed-source, pure-scatter, white BC: exact solution φ=const.

    Σ_s = σ_t, no fission, white BC. With uniform external source Q,
    the steady-state flux is φ = Q / Σ_a = ∞ (Σ_a=0). Use Σ_s = 0.99 σ_t,
    Σ_a = 0.01 σ_t: φ = Q / Σ_a = 100 Q exactly. Compare numerical
    solve."""
    sig_a = 0.01 * sig_t
    sig_s = (sig_t - sig_a)
    Q = 1.0  # uniform source

    sig_t_arr = np.array([sig_t])
    K_vol, K_bc_heb, r_nodes = assemble_K(
        geometry, sig_t_arr, radii, with_hebert=True, P_ss_func=P_ss_func,
    )
    K = K_vol + K_bc_heb

    # Solve (Σ_t I - Σ_s K) φ = K · Q (for fixed source Q applied uniformly)
    # Actually the K matrix already contains the σ_t weighting. Standard
    # transport-fixed-source: σ_t φ = Σ_s K φ + K · Q (where Q is the
    # external source contribution as a per-node source).
    # In ORPHEUS Peierls:  σ_t φ = K (σ_s φ + Q)   (collision form)
    #                  ⇒  (σ_t I - σ_s K) φ = K · Q
    #
    # For uniform Q on solid cell:
    n = len(r_nodes)
    A = sig_t * np.eye(n) - sig_s * K
    b = K @ (Q * np.ones(n))
    phi_num = np.linalg.solve(A, b)

    phi_exact = Q / sig_a  # 1/Σ_a since K conserves on white BC
    rel_dev = (phi_num - phi_exact) / phi_exact
    print(f"\n  {label} fixed-source pure-scatter (σ_t={sig_t}, c=0.99):")
    print(f"    exact φ      : {phi_exact:.4f}")
    print(f"    numeric φ    : min={phi_num.min():.4f}  "
          f"max={phi_num.max():.4f}  mean={phi_num.mean():.4f}")
    print(f"    rel dev      : min={rel_dev.min():+.4f}  "
          f"max={rel_dev.max():+.4f}  "
          f"mean={rel_dev.mean():+.4f}")
    return rel_dev


def main():
    print("=" * 78)
    print("CYLINDER HÉBERT RESIDUAL DIAGNOSIS (Issue #132 Hypothesis 3)")
    print("=" * 78)
    print()
    print("If sphere row-sum dev ≈ 0 % AND cylinder row-sum dev ≈ 10 %,")
    print("the residual is in compute_G_bc (cylinder branch) — the ")
    print("Knyazev Ki_{2+k} 3-D correction is required.")
    print("=" * 78)

    radii_1r = np.array([1.0])
    sig_t_1g = np.array([1.0])

    # Sphere reference: should give near-zero deviation
    print("\n────────────── SPHERE (reference: should be ~0) ──────────────")
    row_sum_partition_check(
        SPHERE_1D, sig_t_1g, radii_1r, _Pss_sphere_func, "sphere 1G/1R",
    )
    fixed_source_pure_scatter(
        SPHERE_1D, 1.0, radii_1r, _Pss_sphere_func, "sphere 1G/1R",
    )

    # Cylinder: probe the Mark kernel bias
    print("\n────────────── CYLINDER (residual ≈ kernel bias) ──────────────")
    row_sum_partition_check(
        CYLINDER_1D, sig_t_1g, radii_1r, _Pss_cyl_func, "cylinder 1G/1R",
    )
    fixed_source_pure_scatter(
        CYLINDER_1D, 1.0, radii_1r, _Pss_cyl_func, "cylinder 1G/1R",
    )

    # 2-region cylinder
    radii_2r = np.array([0.5, 1.0])
    sig_t_2r = np.array([1.0, 1.0])  # uniform σ_t
    print("\n────────────── CYLINDER 2R (uniform σ_t) ──────────────")
    row_sum_partition_check(
        CYLINDER_1D, sig_t_2r, radii_2r, _Pss_cyl_func, "cylinder 1G/2R",
    )


if __name__ == "__main__":
    main()
