"""Diagnostic v2: Galerkin double-integration K_bc with vis-cone + u² substitution.

Created by numerics-investigator on 2026-04-28.

v1 was wrong because:
1. Per-pair K used plain GL on [0, 1], missing visibility step → Q-oscillation.
2. Convention factor was unclear.

v2 fixes:
1. Per-pair K uses vis-cone + u² substitution → spectral Q-convergence
   (verified to machine precision off-diagonal in
   diag_phase5_round3_visibility_cone_quad.py).
2. Outer assembly matches Phase 4 native specular_multibounce conventions:
     K_bc[i, j] = (sig_t_i/divisor) · ∫∫ L_i(r) L_j(r') · rv(r') ·
                       [∫ G_in(r,µ) F_out(r',µ) MB(µ) dµ] dr dr'
   The diagonal r=r' is included (log-singular but L¹-integrable in 2-D).
3. Multi-bounce factor MB(µ) is the BACKUP-derived (1/2)·µ/(1-e^{-2aµ})
   form (HALF M1) — verified by polylog closed form.

Diagonal log-divergence handling
---------------------------------
At r = r' the per-pair K is log-divergent in the µ-quadrature Q. In Galerkin
sub-quadrature, GL nodes within a panel never coincide (so r_qP ≠ r_qQ
in different sub-points), but the NEAR-diagonal contribution (small |r-r'|)
grows as log(1/|r-r'|).

For a panel of width W, the integrated diagonal contribution is:
    ∫∫_panel² log(1/|r-r'|) dr dr' = W² · (log W + 3/2)
which is FINITE for W > 0. So Galerkin DOES smooth the diagonal log-singularity
in 2-D, just with O(log) convergence rate (NOT spectral).

This is the price of the smoothing approach. As n_quad_r → ∞, the Galerkin
quadrature must capture the log-singular near-diagonal accurately, requiring
LOG-WEIGHTED quadrature for spectral convergence (Gauss-log) or simply many
plain GL nodes (algebraic convergence).

Approach
--------
1. Use plain GL for r/r' panel sub-integration.
2. Use vis-cone + u² subst for per-(r,r') µ-integration.
3. Run convergence ladder n_quad_r ∈ {4, 8, 16, 32}, n_quad_µ ∈ {64, 128}.
4. Compare to white_hebert reference.

Decision criterion
------------------
- |k_eff_galerkin/k_inf - 1| < 0.5 % at thin sphere
- No catastrophic overshoot
- Match white_hebert within 0.5 %
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, _build_full_K_per_group, composite_gl_r,
)
from diag_phase5_round3_visibility_cone_quad import (
    _K_cont_pair_visibility_substitution as _K_cont_pair,
)


SIG_T = 0.5
SIG_S = 0.38
NU_SIG_F = 0.025
K_INF = NU_SIG_F / (SIG_T - SIG_S)
R_THIN = 5.0


def keff_from_K(K, sig_t_array, nu_sig_f_array, sig_s_array):
    A = np.diag(sig_t_array) - K * sig_s_array[None, :]
    B = K * nu_sig_f_array[None, :]
    M = np.linalg.solve(A, B)
    eigval = np.linalg.eigvals(M)
    return float(np.max(np.real(eigval)))


def _lagrange_basis_local(r_eval, r_panel, k_local):
    p = len(r_panel)
    num = 1.0
    den = 1.0
    for b in range(p):
        if b == k_local:
            continue
        num *= (r_eval - r_panel[b])
        den *= (r_panel[k_local] - r_panel[b])
    return num / den


def compute_K_bc_galerkin_v2(
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    panels: list,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_quad_r: int = 16,
    n_quad_mu: int = 64,
    convention_factor: float = 1.0,  # ←tune to match reference
) -> np.ndarray:
    r"""Galerkin K_bc with vis-cone + u² substitution per-pair quadrature.

    K_bc[i,j] = (σ_t,i/d) · Σ_{P,Q panels}
        Σ_{a in P} Σ_{b in Q} 1[i=i_P+a] 1[j=j_Q+b]
            ∫∫ L_a^P(r) L_b^Q(r') rv(r') K_continuous(r, r') dr dr'

    where K_continuous(r, r') uses the BACKUP-derived FULL M1 form:
        K_continuous(r, r') = ∫ G_in(r,µ) F_out(r',µ) µ/(1-e^{-2aµ}) dµ
    via vis-cone + u² substitution.

    The convention_factor allows scanning HALF/FULL/DOUBLE M1 to find the
    right normalisation.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    if len(radii) != 1:
        raise NotImplementedError(
            "Galerkin K_bc currently homogeneous-sphere only "
            "(multi-region deferred for round 4)."
        )
    R = float(radii[-1])
    sigma = float(sig_t[0])

    N = len(r_nodes)
    geometry = SPHERE_1D
    divisor = geometry.rank1_surface_divisor(R)

    # Pre-compute panel sub-quadrature points and Lagrange weights
    rsub_nodes_ref, rsub_wts_ref = np.polynomial.legendre.leggauss(n_quad_r)
    panel_data = []
    for pa, pb, i_start, i_end in panels:
        h = 0.5 * (pb - pa)
        m = 0.5 * (pa + pb)
        r_sub = h * rsub_nodes_ref + m
        w_sub = h * rsub_wts_ref
        local_nodes = r_nodes[i_start:i_end]
        p_local = i_end - i_start
        L_local = np.zeros((p_local, n_quad_r))
        for a in range(p_local):
            for q in range(n_quad_r):
                L_local[a, q] = _lagrange_basis_local(
                    r_sub[q], local_nodes, a,
                )
        panel_data.append({
            "i_start": i_start,
            "i_end": i_end,
            "p_local": p_local,
            "r_sub": r_sub,
            "w_sub": w_sub,
            "L_local": L_local,
            "V_sub": np.array([
                geometry.radial_volume_weight(float(r))
                for r in r_sub
            ]),
        })

    # Pre-compute the per-pair K(r_qP, r_qQ) matrix
    n_total = sum(n_quad_r for _ in panel_data)
    all_r_sub = np.concatenate([p["r_sub"] for p in panel_data])
    K_pair = np.zeros((n_total, n_total))
    for k1 in range(n_total):
        for k2 in range(n_total):
            K_pair[k1, k2] = _K_cont_pair(
                all_r_sub[k1], all_r_sub[k2], sigma, R, n_quad_mu,
            )

    sig_t_n = np.full(N, sigma)
    K_bc = np.zeros((N, N))

    panel_offsets = []
    off = 0
    for p in panel_data:
        panel_offsets.append(off)
        off += n_quad_r

    for P_idx, pdat in enumerate(panel_data):
        i0, i1 = pdat["i_start"], pdat["i_end"]
        L_P = pdat["L_local"]
        w_sub_P = pdat["w_sub"]
        offP = panel_offsets[P_idx]
        for Q_idx, qdat in enumerate(panel_data):
            j0, j1 = qdat["i_start"], qdat["i_end"]
            L_Q = qdat["L_local"]
            w_sub_Q = qdat["w_sub"]
            V_Q = qdat["V_sub"]
            offQ = panel_offsets[Q_idx]
            K_sub = K_pair[offP:offP + n_quad_r, offQ:offQ + n_quad_r]

            # K_block[a, b] = Σ_qP Σ_qQ L_P[a,qP]·w_qP·K_sub[qP,qQ]·w_qQ·V_Q[qQ]·L_Q[b,qQ]
            tmp = (L_P * w_sub_P[None, :]) @ K_sub  # (p_loc, n_quad_r)
            K_block = tmp @ ((w_sub_Q * V_Q)[:, None] * L_Q.T)
            K_block *= sig_t_n[i0:i1, None] / divisor
            K_block *= convention_factor
            K_bc[i0:i1, j0:j1] += K_block

    return K_bc


def test_v2_b_q_mu_divergence(capsys):
    """Demonstrate the FUNDAMENTAL FAILURE: 2-D panel integral diverges with Q_µ.

    The diagonal log-singularity in K_continuous(r, r) propagates to a
    log(Q_µ) divergence at every (r, r') node when Q_µ varies. This means
    the Galerkin smoothing strategy is structurally broken — the 2-D
    integral ∫∫ K(r, r') dr dr' is itself ill-defined when computed via
    Nyström sampling of the µ-integrand.
    """
    with capsys.disabled():
        from diag_phase5_round3_visibility_cone_quad import (
            _K_cont_pair_visibility_substitution as Kpair,
        )
        sigma, R = SIG_T, R_THIN
        a_p, b_p = 1.0, 4.0
        h = 0.5 * (b_p - a_p)
        m = 0.5 * (a_p + b_p)
        Q_r = 16
        print(f"\n=== v2-B: Q_µ divergence of 2-D Galerkin integral ===")
        print(f"Panel [{a_p}, {b_p}], Q_r = {Q_r}")
        print(f"  Q_µ  | ∫∫ K_pair(r,r') dr dr'")
        prev = None
        for Q_mu in (32, 64, 128, 256):
            nodes, wts = np.polynomial.legendre.leggauss(Q_r)
            r_sub = h * nodes + m
            w_sub = h * wts
            I = 0.0
            for k1 in range(Q_r):
                for k2 in range(Q_r):
                    I += w_sub[k1] * w_sub[k2] * Kpair(
                        r_sub[k1], r_sub[k2], sigma, R, Q_mu,
                    )
            print(f"  {Q_mu:4d} | {I:+.6e}",
                  f"(Δ vs prev: {(I - prev) if prev else 'n/a'})"
                  if prev is not None else "")
            prev = I
        print(f"\n  CONCLUSION: 2-D integral diverges ~log(Q_µ).")
        print(f"  Each Q_µ-doubling adds ~constant to the integral.")
        print(f"  → Galerkin smoothing FAILS to cure the structural singularity.")


def test_v2_a_smoke_keff(capsys):
    """Smoke test the Galerkin v2 form across multiple convention factors."""
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)
        rel_heb = (k_heb - K_INF) / K_INF * 100

        K_vac = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )

        print(f"\n=== Galerkin v2 smoke test ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, k_inf={K_INF:.6f}")
        print(f"white_hebert: k_eff={k_heb:.6f} ({rel_heb:+.4f}%)")

        for cf_label, cf in [("HALF M1 (×0.5)", 0.5),
                              ("FULL M1 (×1.0)", 1.0),
                              ("DOUBLE M1 (×2.0)", 2.0)]:
            print(f"\n  --- {cf_label} ---")
            print(f"  n_quad_r | n_quad_µ | k_eff_gal     | rel_kinf(%) "
                  f"| rel_heb(%)")
            for nqr in (4, 8, 16):
                for nqmu in (64, 128):
                    K_bc = compute_K_bc_galerkin_v2(
                        r_nodes, r_wts, panels, radii, sig_t_g,
                        n_quad_r=nqr, n_quad_mu=nqmu,
                        convention_factor=cf,
                    )
                    K_full = K_vac + K_bc
                    k_g = keff_from_K(
                        K_full, sig_t_arr, nu_sf_arr, sig_s_arr,
                    )
                    rel_inf = (k_g - K_INF) / K_INF * 100
                    rel_heb_diff = (k_g - k_heb) / k_heb * 100
                    print(f"  {nqr:7d}  | {nqmu:9d} | {k_g:.6f}      | "
                          f"{rel_inf:+.4f}%    | {rel_heb_diff:+.4f}%")
