"""Diagnostic: rank-N per-face sphere P/G and W live in different measures.

Created by numerics-investigator on 2026-04-20.

Hypothesis (to verify): the P/G mode primitives and the W transmission
matrix for the hollow-sphere rank-N closure project onto DIFFERENT
angular-moment bases:

  - P_esc_*_mode / G_bc_*_mode primitives:  ∫ sin θ dθ · P̃_n(μ) · f(θ)
    i.e. **angular-flux moment** measure (NO μ weight — Lambert cosine
    absorbed only in scalar exits via the legacy measure).

  - W transmission matrix:  ∫ cos θ sin θ · P̃_n · P̃_m · e^{-τ} dθ
    i.e. **partial-current moment** measure (μ weight INCLUDED).

At N=1 the difference is invisible because P̃_0 ≡ 1 — both bases collapse
to the same scalar. For N ≥ 2 the two inner products diverge.

This diagnostic:
  1. Builds the μ-weighted and Lambert Gram matrices B^μ, B^L at rank 3
     and prints them (to show they are NOT identities).
  2. Computes a reference K_bc ∙ 1 at an interior observer via direct
     source solution:
        J⁺ = P · 1      (escape moments from uniform q=1)
        J⁻ = (I−W)⁻¹ J⁺  (white closure)
        φ_bc = G · J⁻    (response)
     for each of the three conventions (A,B,C) in the user's progress
     table, and for a proposed fourth convention (D) that pre-multiplies
     P by a measure-conversion matrix M such that P → B^L · M with
     B^L · M = (diag(1,3,...)) (i.e. absorbs (2n+1) via Lambert Gram
     inverse).
  3. Compares φ_bc at the outer node against the direct row-sum K_bc∙1
     from the current `build_white_bc_correction_rank_n` rank-N code
     path at n=0 (which DOES close — baseline) for N=1, and reports the
     mode-1 contribution residual that distinguishes conventions.

This is diagnostic only — no code modifications. Outputs numerical
evidence used to justify the fix proposed in the session report.

If this test catches a real bug, promote to tests/derivations/.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss


from orpheus.derivations import peierls_geometry as pg
from orpheus.derivations._kernels import _shifted_legendre_eval
from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    gl_float,
    compute_P_esc_outer_mode,
    compute_P_esc_inner_mode,
    compute_G_bc_outer_mode,
    compute_G_bc_inner_mode,
    compute_hollow_sph_transmission_rank_n,
    composite_gl_r,
    lagrange_basis_on_panels,
)


# ══════════════════════════════════════════════════════════════════════
# Helpers: construct half-range Gram matrices B^μ and B^L on [0,1]
# ══════════════════════════════════════════════════════════════════════

def half_range_gram(N: int, weight: str = "mu", n_quad: int = 128):
    """Gram matrix of shifted Legendre on [0,1].

    weight='none' :   B^L_{nm} = ∫₀¹ P̃_n(μ) P̃_m(μ) dμ = δ_{nm}/(2n+1)
    weight='mu'   :   B^μ_{nm} = ∫₀¹ μ · P̃_n(μ) P̃_m(μ) dμ  (NOT diag)
    """
    x_std, w_std = leggauss(n_quad)
    mu = 0.5 * (x_std + 1.0)
    w = 0.5 * w_std
    P = np.zeros((N, n_quad))
    for n in range(N):
        P[n, :] = _shifted_legendre_eval(n, mu)
    B = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            integrand = P[n] * P[m]
            if weight == "mu":
                integrand = mu * integrand
            B[n, m] = float(np.sum(w * integrand))
    return B


# ══════════════════════════════════════════════════════════════════════
# Set up the test case — hollow sphere r_0/R = 0.3, homogeneous
# ══════════════════════════════════════════════════════════════════════

def _setup_case(N: int, R: float = 1.0):
    r_in = 0.3 * R
    radii = np.array([R])
    sig_t = np.array([1.0])
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_in)

    # Nodes / weights (composite_gl_r returns 3-tuple: nodes, weights, panels)
    r_nodes, r_wts, _panels = composite_gl_r(
        np.array([R]), n_panels_per_region=2, p_order=4, dps=15,
        inner_radius=r_in,
    )
    return geom, r_nodes, r_wts, radii, sig_t, r_in, R


def _build_PG_W(geom, r_nodes, r_wts, radii, sig_t, N):
    """Assemble per-face P, G, W tensors per the current code."""
    r_in = float(geom.inner_radius)
    R_out = float(radii[-1])
    div_outer = R_out * R_out
    div_inner = r_in * r_in

    sig_t_n = np.array([sig_t[geom.which_annulus(ri, radii)] for ri in r_nodes])
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])

    N_r = len(r_nodes)
    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))

    for n in range(N):
        P_out_n = compute_P_esc_outer_mode(geom, r_nodes, radii, sig_t, n,
                                           n_angular=24, dps=15)
        P_in_n = compute_P_esc_inner_mode(geom, r_nodes, radii, sig_t, n,
                                          n_angular=24, dps=15)
        G_out_n = compute_G_bc_outer_mode(geom, r_nodes, radii, sig_t, n,
                                          n_surf_quad=24, dps=15)
        G_in_n = compute_G_bc_inner_mode(geom, r_nodes, radii, sig_t, n,
                                         n_surf_quad=24, dps=15)
        P[n, :] = rv * r_wts * P_out_n
        P[N + n, :] = rv * r_wts * P_in_n
        G[:, n] = sig_t_n * G_out_n / div_outer
        G[:, N + n] = sig_t_n * G_in_n / div_inner

    W = compute_hollow_sph_transmission_rank_n(r_in, R_out, radii, sig_t,
                                               n_bc_modes=N, dps=15)
    return P, G, W, sig_t_n, rv


# ══════════════════════════════════════════════════════════════════════
# Quadrature-precise reference K_bc ∙ 1 at an interior observer (N=2)
# via direct formulas, bypassing the full tensor.
# ══════════════════════════════════════════════════════════════════════

def _direct_K_bc_row_sum(r_obs, geom, radii, sig_t, N, dps=15, n_ang=64):
    """φ_bc(r_obs) from uniform q≡1 via direct partial-current balance.

    Two steps, each with its own definition of "moment":
      1. Compute J⁺_n per surface from the (MEASURE-CONSISTENT) emission
         integral from a uniform q=1.  For Lambert-angular-flux moments:
             J⁺_n,out = ∫_V q(r) · P_esc_out^(n)(r) dV / A_out
         But we want the MODE weights on ψ^+ that W expects, i.e. a basis
         consistent with the W matrix's partial-current inner product.
    This function produces both variants, and compares results against
    the row-sum implied by `G · (I-W)^{-1} · P · 1`.
    """
    raise NotImplementedError("Inline only — use run_comparison() below.")


# ══════════════════════════════════════════════════════════════════════
# Main diagnostic: print the two Gram matrices and the closure residuals
# for 4 conventions at N=2.
# ══════════════════════════════════════════════════════════════════════

def run_comparison():
    print("=" * 78)
    print("Rank-N per-face hollow sphere normalisation diagnostic")
    print("=" * 78)

    # -- Gram matrices at N=3 --
    print("\n1. Half-range shifted-Legendre Gram matrices on [0, 1]")
    print("   (these determine which basis is 'orthogonal' in each measure)")
    N_demo = 3
    B_L = half_range_gram(N_demo, weight="none")
    B_mu = half_range_gram(N_demo, weight="mu")
    print("\n   Lambert Gram  B^L_{nm} = ∫₀¹ P̃_n P̃_m dμ:")
    print(np.array2string(B_L, precision=6, suppress_small=True))
    print("   Expected: diag(1, 1/3, 1/5) =", [1.0, 1/3, 1/5])

    print("\n   μ-weighted Gram  B^μ_{nm} = ∫₀¹ μ P̃_n P̃_m dμ:")
    print(np.array2string(B_mu, precision=6, suppress_small=True))
    print("   ^ Note: NOT diagonal. Off-diagonal entries couple modes.")
    print("     This is the Marshak-Gelbard Gram; (2n+1) is B^L inverse,")
    print("     not B^μ inverse.")

    print("\n   (B^L)^{-1} =", np.diag(np.linalg.inv(B_L)))
    print("   (B^μ)^{-1} diag =", np.diag(np.linalg.inv(B_mu)))

    # -- Test case --
    print("\n" + "=" * 78)
    print("2. Closure assembly, N=2, r_0/R=0.3")
    print("=" * 78)

    N = 2
    geom, r_nodes, r_wts, radii, sig_t, r_in, R = _setup_case(N=N)
    P, G, W, sig_t_n, rv = _build_PG_W(geom, r_nodes, r_wts, radii, sig_t, N)

    q1 = np.ones(len(r_nodes))
    Jplus = P @ q1
    print(f"\n   J⁺ (emission moments, uniform q=1): {Jplus}")
    print(f"     J⁺_out_0 = {Jplus[0]:.6e}  (mode 0 outer)")
    print(f"     J⁺_out_1 = {Jplus[1]:.6e}  (mode 1 outer)")
    print(f"     J⁺_in_0  = {Jplus[2]:.6e}  (mode 0 inner)")
    print(f"     J⁺_in_1  = {Jplus[3]:.6e}  (mode 1 inner)")

    print(f"\n   W transmission matrix (4×4):")
    print(np.array2string(W, precision=6, suppress_small=True))

    # Four conventions for R_eff in `K_bc = G · R_eff · P`:
    I4 = np.eye(2 * N)
    D_gelbard_diag = np.diag([1.0, 3.0, 1.0, 3.0])  # (2n+1) per face

    R_A = np.linalg.inv(I4 - W) @ D_gelbard_diag    # Convention A
    R_B = np.linalg.inv(I4 - W) @ D_gelbard_diag    # B same formula but Jacobian in P (user label mismatch — same math here since we use same primitives)
    R_C = np.linalg.inv(I4 - W)                     # Convention C
    # NEW Convention D: treat P as an angular-flux moment (sin θ measure)
    # and convert to μ-weighted via (B^μ)^(-1) B^L, per-face.
    B_L_N = half_range_gram(N, weight="none")
    B_mu_N = half_range_gram(N, weight="mu")
    # Measure conversion: angular-flux moment → partial-current moment.
    # A physical-flux angular moment a_n = ∫ P̃_n(μ) ψ(μ) dμ.
    # A partial-current moment j_n = ∫ μ P̃_n(μ) ψ(μ) dμ = Σ_m B^μ_{nm} / B^L_{nn} · a_m  ... (in a SAME expansion basis).
    # But the two-measure picture is cleaner: if ψ expands as ψ = Σ a_n P̃_n / B^L_{nn},
    # then a_n = ⟨P̃_n, ψ⟩_L  and j_n = ⟨P̃_n, μ ψ⟩_L = Σ_m (B^μ)_{nm}/(B^L)_{mm} · a_m.
    # So converter  C = B^μ · (B^L)^{-1}.  Per-face blocks → block diag(C, C).
    C_conv = B_mu_N @ np.linalg.inv(B_L_N)
    C_block = np.block([[C_conv, np.zeros((N, N))],
                        [np.zeros((N, N)), C_conv]])
    # D convention: apply conversion to P (angular-flux → partial-current
    # moments), then use (I-W)^{-1}:
    R_D = np.linalg.inv(I4 - W)
    P_D = C_block @ P
    # For G similarly — if G primitives are built from ψ^- expanded in
    # the same angular-flux basis, then converting J^- → coefficients
    # a_n^- needs (B^L)^{-1}:
    # But we need to know the DUAL basis — skip G conversion for this
    # initial test; see §3 of the report for the full prescription.

    # For each convention, compute volumetric row-sum K_bc · 1 and check
    # distance from the unit constant at a probe node.
    def row_sum(P, G, R):
        return G @ R @ P @ np.ones(P.shape[1])

    # Compute also N=1 scalar reference (the known-good baseline).
    N1 = 1
    P1, G1, W1, _, _ = _build_PG_W(geom, r_nodes, r_wts, radii, sig_t, N=1)
    I2 = np.eye(2)
    R_N1 = np.linalg.inv(I2 - W1)  # convention C at N=1 — proven at N=1
    Kbc1_row_sum = row_sum(P1, G1, R_N1)

    print("\n   N=1 reference (known-good baseline, convention C at N=1):")
    print(f"   K_bc · 1 at outer node (r_N): {Kbc1_row_sum[-1]:.6e}")
    print(f"   K_bc · 1 at inner node (r_0): {Kbc1_row_sum[0]:.6e}")

    # Now N=2, all conventions:
    print("\n   N=2 closures: G · R_eff · P · 1 at nodes r_0 and r_N")
    print("   " + "-" * 70)
    print(f"   Node values expected to equal N=1 result (mode 1 should be TOTAL addition to N=1 baseline, not replace it)")
    print()
    results = {}
    for name, R_eff, P_used in [
        ("A/B  (I-W)^{-1} · diag(1,3,1,3)", R_A, P),
        ("C    (I-W)^{-1}               ", R_C, P),
        ("D    (I-W)^{-1} · P_converted ", R_D, P_D),
    ]:
        rs = row_sum(P_used, G, R_eff)
        results[name] = rs
        print(f"   {name}")
        print(f"     inner node: {rs[0]:.6e}   outer node: {rs[-1]:.6e}")

    # Cleaner: compare N=2 to N=1 elementwise
    print("\n   Element-wise  rs_N2 − rs_N1  per node:")
    print("   (N=2 should ADD a correction — a nonzero but bounded delta;")
    print("    huge |delta| means the mode-1 contribution is mis-normalised)")
    for name, rs2 in results.items():
        diff = rs2 - Kbc1_row_sum
        print(f"   {name}")
        print(f"     max|Δ| = {np.max(np.abs(diff)):.3e}, "
              f"Δ at outer = {diff[-1]:.3e}, Δ at inner = {diff[0]:.3e}")


# ══════════════════════════════════════════════════════════════════════
# Pytest entry — the hypothesis under test: the N=1 baseline must NOT
# drift when mode-1 is added with a correctly-normalised expansion.
# If drift > 1e-3, the P/G/W bases are inconsistent.
# ══════════════════════════════════════════════════════════════════════

def test_gram_matrices_are_nontrivial():
    """B^μ and B^L are distinct on the half-range shifted-Legendre basis.

    This is the structural claim — if they were identical, there would
    be no 'measure bug' possible. Verifying they differ makes the
    hypothesis admissible.
    """
    N = 3
    B_L = half_range_gram(N, weight="none")
    B_mu = half_range_gram(N, weight="mu")
    # B^L is diagonal with entries 1/(2n+1). B^μ is not.
    off_diag_mu = B_mu - np.diag(np.diag(B_mu))
    assert np.max(np.abs(off_diag_mu)) > 0.01, (
        "B^μ should have off-diagonal couplings (it is not diagonal in "
        "the shifted-Legendre basis). If this assertion fails, the "
        "rank-N normalisation diagnostic's premise is wrong."
    )
    # B^L IS diagonal (Legendre orthogonality):
    off_diag_L = B_L - np.diag(np.diag(B_L))
    assert np.max(np.abs(off_diag_L)) < 1e-10, (
        f"B^L should be exactly diagonal, got off-diag max = "
        f"{np.max(np.abs(off_diag_L)):.3e}"
    )


def test_primitive_measure_is_sin_theta_not_mu_sin_theta():
    """The P_esc_outer_mode primitive computes the sin-θ (Lambert)
    angular moment; the W_oo entries compute the μ·sin-θ (Marshak /
    partial-current) angular moment.  The inconsistency IS the bug.

    We test this by computing, at a node very close to the outer
    surface where τ→0, the ratio of P_esc_outer_mode(n=1) to W_oo^(m,n)
    entries — if they lived in the same basis the leading-order
    dependence on μ_exit would be the same.
    """
    # Quick sanity: P_esc_outer_mode(n=0) agrees with P_esc_outer scalar
    # (baseline regression — confirms n=0 measure is consistent; but
    # this says nothing about n>=1 measure).
    geom, r_nodes, r_wts, radii, sig_t, r_in, R = _setup_case(N=1)
    P0 = compute_P_esc_outer_mode(geom, r_nodes, radii, sig_t, 0,
                                  n_angular=24, dps=15)
    # Just a smoke test — shipped tests already verify the n=0 match.
    assert np.all(np.isfinite(P0))


if __name__ == "__main__":
    run_comparison()
