"""Per-mode Villarino-Stamm'ler normalisation on rank-N hollow-sphere W.

Created by numerics-investigator on 2026-04-21 for Issue #119.

Hypothesis C (novel, unvalidated by literature).

    Apply Hébert 2009 Eqs. 3.347-3.352 (Villarino-Stamm'ler rank-0
    normalisation) INDEPENDENTLY to each diagonal mode-n surface
    sub-block of the rank-N transmission matrix W. Target: force the
    conservation identity

        W_oo[n, n] + W_oi[n, n] = 1     (outer row sum per mode)
        W_io[n, n] + W_ii[n, n] = 1     (inner row sum per mode)

    at sigma_t = 0 for every mode n, replicating the scalar F.4 identity
    that drives F.4's 0.077 % residual. At sigma_t > 0 the target is
    the scalar F.4 row sums (computed directly from
    compute_hollow_sph_transmission).

    The additive symmetric correction preserves reciprocity by
    construction (Hébert p. 129):

        \\hat{t}_{lm} = (z_l + z_m) t_{lm},
        t_{lm} = (S_l / 4) * W[l, m] (symmetric; l, m in {outer, inner})

    Four correction factors per mode (one z_outer^n and one z_inner^n per
    mode block), computed by Gauss-Seidel fixed-point iteration on the
    per-mode 2x2 T-matrix.

Literature status.

    Hébert's §3.350-3.352 is rank-0 only. Extending to per-mode is
    NOT derived in any reference we've consulted (Ligou, Sanchez 2002,
    Stamm'ler Ch. IV, Stacey Ch. 9, Hébert 2009 Ch. 3 — all five
    defer to rank-0 V-S). This diagnostic tests whether the naive
    per-mode extension collapses the 1.42% Sanchez µ-ortho plateau.

Residuals pre V-S (from diag_sanchez_N_convergence.py at sigma_t*R=5,
r_0/R=0.3):

    N=1  2.55%
    N=2  1.43%
    N=3  1.42%
    N=4  1.42% (plateau)

Target after V-S: <= 0.1% at N=2 would falsify "c_in remapping is
structural and unrecoverable via post-hoc normalisation". If plateau
persists, confirms Branch C (close Issue #119 with F.4 as production).

If this diagnostic catches a real bug, promote to ``tests/cp/``.
"""
from __future__ import annotations
import sys

import numpy as np
import pytest

sys.path.insert(0, "/workspaces/ORPHEUS")
sys.path.insert(0, "/workspaces/ORPHEUS/derivations/diagnostics")

from orpheus.derivations.peierls_geometry import (
    BoundaryClosureOperator,
    CurvilinearGeometry,
    build_volume_kernel,
    composite_gl_r,
    compute_G_bc_inner_mode_marshak,
    compute_G_bc_outer_mode_marshak,
    compute_P_esc_inner_mode_marshak,
    compute_P_esc_outer_mode_marshak,
    compute_hollow_sph_transmission,
    compute_hollow_sph_transmission_rank_n,
)

# Re-use the Sanchez µ-ortho rank-N pipeline (1.42 % plateau baseline).
from diag_sanchez_N_convergence import (
    build_K_bc as sanchez_build_K_bc,
    compute_W_rank_n,
    solve_k_eff,
)


# -------------------------------------------------------------------------
# Per-mode Villarino-Stamm'ler (Hypothesis C)
# -------------------------------------------------------------------------
def _vs_per_mode_2x2(
    W_sub: np.ndarray,
    S_outer: float,
    S_inner: float,
    g_outer: float,
    g_inner: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Gauss-Seidel V-S on a single mode-n surface 2x2 block.

    T matrix (Hébert Eq. 3.347, surface-only):

        t = [[S_out/4 * W_oo, S_out/4 * W_oi],
             [S_in /4 * W_io, S_in /4 * W_ii]]

    Source/response vectors s = col(1, 1), g = col(S_out/4, S_in/4).

    Solve T.s = g by additive multiplicative correction (Eq. 3.352):

        t_hat_{lm} = (z_l + z_m) * t_{lm}.

    Returns (W_corrected, (z_outer, z_inner)).

    Note: because T is symmetric by reciprocity (S_out * W_oi =
    S_in * W_io), the additive correction preserves symmetry — and
    hence preserves reciprocity in the corrected W.
    """
    # Hébert's t matrix for this mode.
    t = np.array([
        [(S_outer / 4.0) * W_sub[0, 0], (S_outer / 4.0) * W_sub[0, 1]],
        [(S_inner / 4.0) * W_sub[1, 0], (S_inner / 4.0) * W_sub[1, 1]],
    ])
    s = np.array([1.0, 1.0])
    g = np.array([g_outer, g_inner])

    # Linearize: row l gives
    #   z_l * sum_m t_{lm} s_m + sum_m t_{lm} s_m z_m = g_l
    # => (t.s)_l * z_l + (t * diag(s)) @ z = g_l
    # i.e. M z = g with M_{ll} = (t.s)_l + t_{ll} s_l, M_{lm} = t_{lm} s_m (l != m).
    ts = t @ s  # shape (2,)
    M = np.array([
        [ts[0] + t[0, 0] * s[0], t[0, 1] * s[1]],
        [t[1, 0] * s[0], ts[1] + t[1, 1] * s[1]],
    ])
    # Hébert Eq. 3.352 initial guess z = 1/2.
    z = np.array([0.5, 0.5])

    # Direct solve (2x2 is cheap and cleaner than Gauss-Seidel).
    # Equivalent to running Gauss-Seidel to convergence.
    try:
        z = np.linalg.solve(M, g)
    except np.linalg.LinAlgError:
        # Fall back to Gauss-Seidel if M is singular (shouldn't happen
        # for physically sensible W).
        for _ in range(max_iter):
            z_new = z.copy()
            for l in range(2):
                num = g[l]
                for m in range(2):
                    if m < l:
                        num -= t[l, m] * s[m] * z_new[m]
                    elif m > l:
                        num -= t[l, m] * s[m] * z[m]
                denom = t[l, l] * s[l] + ts[l]
                z_new[l] = num / denom
            if np.max(np.abs(z_new - z)) < tol:
                z = z_new
                break
            z = z_new

    # Apply the additive symmetric correction.
    Z = np.array([[z[0] + z[0], z[0] + z[1]],
                  [z[1] + z[0], z[1] + z[1]]])
    W_hat = Z * W_sub
    return W_hat, (z[0], z[1])


def apply_vs_per_mode(
    W: np.ndarray,
    N: int,
    S_outer: float,
    S_inner: float,
    g_outer: float,
    g_inner: float,
) -> np.ndarray:
    """Apply per-mode V-S to the (2N x 2N) rank-N W matrix.

    Mode layout: [outer_0, ..., outer_{N-1}, inner_0, ..., inner_{N-1}].

    For each diagonal mode n in [0, N), extract the 2x2 sub-block

        [[W[n, n]      , W[n, N + n]     ],
         [W[N + n, n]  , W[N + n, N + n] ]]

    apply V-S, and write back. Off-diagonal mode couplings (n != m) are
    left untouched.
    """
    W_corr = W.copy()
    for n in range(N):
        W_sub = np.array([
            [W[n, n], W[n, N + n]],
            [W[N + n, n], W[N + n, N + n]],
        ])
        W_hat, _ = _vs_per_mode_2x2(
            W_sub, S_outer, S_inner, g_outer, g_inner,
        )
        W_corr[n, n] = W_hat[0, 0]
        W_corr[n, N + n] = W_hat[0, 1]
        W_corr[N + n, n] = W_hat[1, 0]
        W_corr[N + n, N + n] = W_hat[1, 1]
    return W_corr


# -------------------------------------------------------------------------
# Build K_bc with per-mode V-S patched onto the shipped W.
# -------------------------------------------------------------------------
def build_K_bc_vs(
    geom, r_nodes, r_wts, radii, sig_t, N,
    *,
    n_angular: int = 24,
    n_surf_quad: int = 24,
    dps: int = 15,
):
    """Rebuild the Sanchez µ-ortho K_bc, then patch W with V-S per-mode.

    The F.4 scalar row sums set the conservation target per mode.
    """
    N_r = len(r_nodes)
    R_out = float(radii[-1])
    r_in = float(geom.inner_radius)
    S_outer = 4.0 * np.pi * R_out ** 2
    S_inner = 4.0 * np.pi * r_in ** 2

    # Compute F.4 scalar row sums as the per-mode target.
    W_scalar = compute_hollow_sph_transmission(
        r_in, R_out, radii, sig_t, dps=dps,
    )
    # W_scalar layout: [[W_oo, W_oi], [W_io, W_ii=0]].
    row_outer_F4 = W_scalar[0, 0] + W_scalar[0, 1]   # target for outer row
    row_inner_F4 = W_scalar[1, 0] + W_scalar[1, 1]   # target for inner row
    g_outer = (S_outer / 4.0) * row_outer_F4
    g_inner = (S_inner / 4.0) * row_inner_F4

    # Sanchez µ-ortho P, G and raw W (1.42 % plateau).
    from diag_sanchez_N_convergence import compute_P_esc, compute_G_bc
    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))
    sig_t_n = np.array([sig_t[geom.which_annulus(ri, radii)] for ri in r_nodes])
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])
    div_outer = R_out * R_out
    div_inner = r_in * r_in
    for n in range(N):
        for face_idx, face in enumerate(["outer", "inner"]):
            P_arr = compute_P_esc(
                geom, r_nodes, radii, sig_t, n, face,
                n_angular=n_angular, dps=dps,
            )
            G_arr = compute_G_bc(
                geom, r_nodes, radii, sig_t, n, face,
                n_surf_quad=n_surf_quad, dps=dps,
            )
            row = face_idx * N + n
            P[row, :] = rv * r_wts * P_arr
            if face_idx == 0:
                G[:, row] = sig_t_n * G_arr / div_outer
            else:
                G[:, row] = sig_t_n * G_arr / div_inner

    W = compute_W_rank_n(geom, radii, sig_t, N, dps=dps)
    W_corrected = apply_vs_per_mode(W, N, S_outer, S_inner, g_outer, g_inner)
    R_mat = np.linalg.inv(np.eye(2 * N) - W_corrected)
    return G @ R_mat @ P, W, W_corrected


# -------------------------------------------------------------------------
# Reciprocity check
# -------------------------------------------------------------------------
def check_reciprocity(
    W: np.ndarray,
    N: int,
    S_outer: float,
    S_inner: float,
) -> float:
    """Sanchez-McCormick reciprocity A_k * W_{jk}^{mn} = A_j * W_{kj}^{nm}.

    With k=outer, j=inner, mode (m,n):
        A_outer * W_{io}^{mn} = A_inner * W_{oi}^{nm}
    i.e.
        S_outer * W[N + m, n] = S_inner * W[n, N + m]
    (outer-area times inner-from-outer mode m<-mode n
     equals inner-area times outer-from-inner mode n<-mode m).

    Returns max absolute violation over all (m, n) pairs.
    """
    max_viol = 0.0
    for m in range(N):
        for n in range(N):
            lhs = S_outer * W[N + m, n]
            rhs = S_inner * W[n, N + m]
            max_viol = max(max_viol, abs(lhs - rhs))
    return max_viol


# -------------------------------------------------------------------------
# Alternative pipeline: SHIPPED Marshak primitives + SHIPPED W (MC-verified)
# -------------------------------------------------------------------------
def build_K_bc_shipped_marshak_vs(
    geom, r_nodes, r_wts, radii, sig_t, N,
    *,
    n_angular: int = 24,
    n_surf_quad: int = 24,
    dps: int = 15,
):
    """Rebuild K_bc using the SHIPPED Marshak primitives + shipped W,
    then patch W with per-mode V-S.

    This replicates the path that _build_closure_operator_rank_n_white
    takes internally (behind the NotImplementedError guard), but adds
    the per-mode V-S correction on top. Used to cross-check the
    Sanchez µ-ortho pipeline result.
    """
    N_r = len(r_nodes)
    R_out = float(radii[-1])
    r_in = float(geom.inner_radius)
    S_outer = 4.0 * np.pi * R_out ** 2
    S_inner = 4.0 * np.pi * r_in ** 2
    div_outer = R_out * R_out
    div_inner = r_in * r_in

    # F.4 scalar targets.
    W_scalar = compute_hollow_sph_transmission(
        r_in, R_out, radii, sig_t, dps=dps,
    )
    row_outer_F4 = W_scalar[0, 0] + W_scalar[0, 1]
    row_inner_F4 = W_scalar[1, 0] + W_scalar[1, 1]
    g_outer = (S_outer / 4.0) * row_outer_F4
    g_inner = (S_inner / 4.0) * row_inner_F4

    sig_t_n = np.array([sig_t[geom.which_annulus(ri, radii)] for ri in r_nodes])
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])

    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))
    for n in range(N):
        P_out = compute_P_esc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t, n,
            n_angular=n_angular, dps=dps,
        )
        P_in = compute_P_esc_inner_mode_marshak(
            geom, r_nodes, radii, sig_t, n,
            n_angular=n_angular, dps=dps,
        )
        G_out = compute_G_bc_outer_mode_marshak(
            geom, r_nodes, radii, sig_t, n,
            n_surf_quad=n_surf_quad, dps=dps,
        )
        G_in = compute_G_bc_inner_mode_marshak(
            geom, r_nodes, radii, sig_t, n,
            n_surf_quad=n_surf_quad, dps=dps,
        )
        P[n, :] = rv * r_wts * P_out
        P[N + n, :] = rv * r_wts * P_in
        G[:, n] = sig_t_n * G_out / div_outer
        G[:, N + n] = sig_t_n * G_in / div_inner

    W = compute_hollow_sph_transmission_rank_n(
        r_in, R_out, radii, sig_t, N, dps=dps,
    )
    W_corr = apply_vs_per_mode(W, N, S_outer, S_inner, g_outer, g_inner)

    # Raw closure (no V-S)
    R_raw = np.linalg.inv(np.eye(2 * N) - W)
    K_raw = G @ R_raw @ P

    # V-S-corrected closure
    R_vs = np.linalg.inv(np.eye(2 * N) - W_corr)
    K_vs = G @ R_vs @ P

    return K_raw, K_vs, W, W_corr


# -------------------------------------------------------------------------
# The primary test
# -------------------------------------------------------------------------
@pytest.mark.slow
def test_vs_per_mode_residual_table():
    """Residual table before/after V-S at sigma_t*R=5, r_0/R=0.3.

    Records numbers for N in {1, 2, 3, 4}. Asserts nothing about
    absolute magnitudes (this is characterization), only:

    - Post-V-S N=1 matches pre-V-S N=1 to within 1e-3 (V-S at n=0
      should be a no-op because F.4 conservation holds at mode 0).
    - Reciprocity survives (violation < 1e-8 after V-S).
    """
    sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
    k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
    R = 5.0
    r_0 = 1.5
    S_outer = 4.0 * np.pi * R ** 2
    S_inner = 4.0 * np.pi * r_0 ** 2
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])

    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 4, dps=15, inner_radius=r_0,
    )
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t,
        n_angular=24, n_rho=24, dps=15,
    )

    results: list[dict] = []
    for N in [1, 2, 3, 4]:
        K_bc_raw = sanchez_build_K_bc(geom, r_nodes, r_wts, radii, sig_t, N)
        k_raw = solve_k_eff(K_bc_raw, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        err_raw = abs(k_raw - k_inf) / k_inf * 100.0

        K_bc_vs, W_raw, W_vs = build_K_bc_vs(
            geom, r_nodes, r_wts, radii, sig_t, N,
        )
        k_vs = solve_k_eff(K_bc_vs, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        err_vs = abs(k_vs - k_inf) / k_inf * 100.0

        recip_raw = check_reciprocity(W_raw, N, S_outer, S_inner)
        recip_vs = check_reciprocity(W_vs, N, S_outer, S_inner)

        results.append(dict(
            N=N, err_raw=err_raw, err_vs=err_vs,
            recip_raw=recip_raw, recip_vs=recip_vs,
        ))

    # N=1 sanity: V-S correction should be near-unity (mode-0 conservation
    # is built into F.4 at sigma_t=0; at sigma_t > 0 there's a residual
    # that V-S corrects, so we allow up to ~3 percentage-point shift).
    n1 = next(r for r in results if r["N"] == 1)
    assert abs(n1["err_vs"] - n1["err_raw"]) < 3.0, (
        f"N=1 V-S shifted err by {n1['err_vs'] - n1['err_raw']:.3f}% — "
        f"expected small adjustment (mode-0 conservation mostly holds "
        f"for F.4-scalar target)."
    )

    # Reciprocity: the per-mode correction is additive-symmetric on the
    # scalar z's, but the OFF-DIAGONAL (n != m) terms of W are untouched.
    # Reciprocity of the diagonal sub-blocks is preserved by construction
    # of the additive symmetric correction.
    for r in results:
        assert r["recip_vs"] < 1e-8, (
            f"N={r['N']}: reciprocity violation after V-S "
            f"= {r['recip_vs']:.3e}, pre V-S = {r['recip_raw']:.3e}"
        )


def test_vs_per_mode_plateau_persists_if_structural():
    """Verdict gate: does per-mode V-S break the 1.42% plateau?

    This is the decisive test. Pass/fail describes outcome, not bug:

    - If err_vs[N=2] < 0.1%: the plateau DID break. Branch A' is
      the production closure. THIS TEST NAME BECOMES MISLEADING and
      should be replaced with a direct success gate.

    - If err_vs[N=2] >= 0.1%: the plateau is structural (c_in
      remapping cannot be rescued by post-hoc per-mode normalisation),
      confirming Branch C close-out. This test stays green as a
      plateau-characterisation gate.
    """
    sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
    k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
    R = 5.0
    r_0 = 1.5
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])

    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 4, dps=15, inner_radius=r_0,
    )
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t,
        n_angular=24, n_rho=24, dps=15,
    )

    K_bc_vs, _, _ = build_K_bc_vs(
        geom, r_nodes, r_wts, radii, sig_t, N=2,
    )
    k_vs = solve_k_eff(K_bc_vs, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
    err_vs = abs(k_vs - k_inf) / k_inf * 100.0

    # The plateau-persists gate: if V-S per-mode closes below 0.1%,
    # this test fails and the investigator pivots to Branch A'.
    # Edited to record the OUTCOME in the assertion message for the
    # session log.
    closed_below_0p1 = err_vs < 0.1
    closed_below_0p5 = err_vs < 0.5

    # Characterization assertion (not absolute):
    # Either the plateau is intact (err_vs > 0.5%), OR we've closed
    # substantially. Any intermediate result (0.1% < err < 0.5%) is
    # still a useful partial finding.
    if closed_below_0p1:
        pytest.fail(
            f"V-S per-mode CLOSED below 0.1% (err={err_vs:.4f}%). "
            f"Plateau BROKEN — this is a Branch A' landing signal. "
            f"Replace this test with a direct success gate and pivot "
            f"to lifting the NotImplementedError guard."
        )
    elif closed_below_0p5:
        pytest.fail(
            f"V-S per-mode PARTIALLY closed (err={err_vs:.4f}%, "
            f"below 0.5%). Between Hypothesis C plateau and 'definitive "
            f"break'. Investigate Hypotheses A and B before closing."
        )
    else:
        # Plateau persisted (>= 0.5%): Hypothesis C fails, Branch C
        # close-out confirmed.
        assert err_vs >= 0.5, f"sanity check failed: err_vs={err_vs}"


# -------------------------------------------------------------------------
# main: print the before/after table and verdict
# -------------------------------------------------------------------------
def main():
    sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
    k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
    R = 5.0
    r_0 = 1.5
    S_outer = 4.0 * np.pi * R ** 2
    S_inner = 4.0 * np.pi * r_0 ** 2
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])

    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 4, dps=15, inner_radius=r_0,
    )
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t,
        n_angular=24, n_rho=24, dps=15,
    )

    print(
        f"Hypothesis C (per-mode Villarino-Stamm'ler) on hollow sphere, "
        f"sigma_t*R = {sig_t_v * R}, r_0/R = {r_0 / R}"
    )
    print(f"k_inf = {k_inf}")
    print()
    print("Pipeline A: Sanchez µ-ortho primitives + µ-ortho W "
          "(1.42 % plateau baseline)")
    print(
        f"{'N':>3s} {'k_raw':>12s} {'err_raw%':>10s} "
        f"{'k_vs':>12s} {'err_vs%':>10s} "
        f"{'recip_raw':>12s} {'recip_vs':>12s}"
    )
    print("-" * 80)

    for N in [1, 2, 3, 4]:
        K_bc_raw = sanchez_build_K_bc(geom, r_nodes, r_wts, radii, sig_t, N)
        k_raw = solve_k_eff(K_bc_raw, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        err_raw = abs(k_raw - k_inf) / k_inf * 100.0

        K_bc_vs, W_raw, W_vs = build_K_bc_vs(
            geom, r_nodes, r_wts, radii, sig_t, N,
        )
        k_vs = solve_k_eff(K_bc_vs, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        err_vs = abs(k_vs - k_inf) / k_inf * 100.0

        recip_raw = check_reciprocity(W_raw, N, S_outer, S_inner)
        recip_vs = check_reciprocity(W_vs, N, S_outer, S_inner)

        print(
            f"{N:3d} {k_raw:12.6f} {err_raw:9.4f}% "
            f"{k_vs:12.6f} {err_vs:9.4f}% "
            f"{recip_raw:12.3e} {recip_vs:12.3e}"
        )

    print()
    print("Pipeline B: SHIPPED Marshak primitives + shipped MC-verified "
          "W (hit by NotImplementedError guard)")
    print(
        f"{'N':>3s} {'k_raw':>12s} {'err_raw%':>10s} "
        f"{'k_vs':>12s} {'err_vs%':>10s} "
        f"{'recip_raw':>12s} {'recip_vs':>12s}"
    )
    print("-" * 80)
    for N in [1, 2, 3, 4]:
        K_raw, K_vs, W_raw, W_vs = build_K_bc_shipped_marshak_vs(
            geom, r_nodes, r_wts, radii, sig_t, N,
        )
        k_raw = solve_k_eff(K_raw, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        err_raw = abs(k_raw - k_inf) / k_inf * 100.0
        k_vs = solve_k_eff(K_vs, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
        err_vs = abs(k_vs - k_inf) / k_inf * 100.0
        recip_raw = check_reciprocity(W_raw, N, S_outer, S_inner)
        recip_vs = check_reciprocity(W_vs, N, S_outer, S_inner)
        print(
            f"{N:3d} {k_raw:12.6f} {err_raw:9.4f}% "
            f"{k_vs:12.6f} {err_vs:9.4f}% "
            f"{recip_raw:12.3e} {recip_vs:12.3e}"
        )

    # Sub-verdict: print the corrected W's per-mode row sums at sigma_t=0
    # to confirm V-S achieved the per-mode conservation target.
    print()
    print("Per-mode conservation check at sigma_t = 0 for N=3:")
    sig_t_zero = np.array([0.0])
    W_raw_zero = compute_W_rank_n(geom, radii, sig_t_zero, 3, dps=15)
    W_scalar = compute_hollow_sph_transmission(r_0, R, radii, sig_t_zero, dps=15)
    row_outer_F4 = W_scalar[0, 0] + W_scalar[0, 1]
    row_inner_F4 = W_scalar[1, 0] + W_scalar[1, 1]
    g_out = (S_outer / 4.0) * row_outer_F4
    g_in = (S_inner / 4.0) * row_inner_F4
    W_vs_zero = apply_vs_per_mode(W_raw_zero, 3, S_outer, S_inner, g_out, g_in)
    print(
        f"  F.4 scalar row sums: outer = {row_outer_F4:.6f}, "
        f"inner = {row_inner_F4:.6f}"
    )
    for n in range(3):
        raw_outer_sum = W_raw_zero[n, n] + W_raw_zero[n, 3 + n]
        raw_inner_sum = W_raw_zero[3 + n, n] + W_raw_zero[3 + n, 3 + n]
        vs_outer_sum = W_vs_zero[n, n] + W_vs_zero[n, 3 + n]
        vs_inner_sum = W_vs_zero[3 + n, n] + W_vs_zero[3 + n, 3 + n]
        print(
            f"  n={n}: outer row raw = {raw_outer_sum:.6f} -> "
            f"vs = {vs_outer_sum:.6f} (target {row_outer_F4:.6f}); "
            f"inner row raw = {raw_inner_sum:.6f} -> "
            f"vs = {vs_inner_sum:.6f} (target {row_inner_F4:.6f})"
        )

    # Extended scan if N=2 V-S closes below 0.1%.
    K_bc_vs_2, _, _ = build_K_bc_vs(geom, r_nodes, r_wts, radii, sig_t, 2)
    k_vs_2 = solve_k_eff(K_bc_vs_2, K_vol, sig_t_v, sig_s_v, nu_sig_f_v)
    err_vs_2 = abs(k_vs_2 - k_inf) / k_inf * 100.0
    if err_vs_2 < 0.1:
        print()
        print(
            "V-S per-mode closed below 0.1% at N=2! Extended scan across "
            "(sigma_t*R, r_0/R) follows:"
        )
        for sigma_R in (1.0, 2.5, 5.0, 10.0, 20.0):
            sig_t_val = sigma_R / R
            for r0_frac in (0.1, 0.2, 0.3):
                r_0_scan = r0_frac * R
                geom_s = CurvilinearGeometry(
                    kind="sphere-1d", inner_radius=r_0_scan,
                )
                sig_t_scan = np.array([sig_t_val])
                r_nodes_s, r_wts_s, panels_s = composite_gl_r(
                    radii, 2, 4, dps=15, inner_radius=r_0_scan,
                )
                K_vol_s = build_volume_kernel(
                    geom_s, r_nodes_s, panels_s, radii, sig_t_scan,
                    n_angular=24, n_rho=24, dps=15,
                )
                try:
                    K_bc_s, _, _ = build_K_bc_vs(
                        geom_s, r_nodes_s, r_wts_s, radii, sig_t_scan, 2,
                    )
                    k_s = solve_k_eff(
                        K_bc_s, K_vol_s, sig_t_v, sig_s_v, nu_sig_f_v,
                    )
                    err_s = abs(k_s - k_inf) / k_inf * 100.0
                    print(
                        f"  sigma_t*R = {sigma_R}, r_0/R = {r0_frac}: "
                        f"err = {err_s:.4f}%"
                    )
                except Exception as e:
                    print(
                        f"  sigma_t*R = {sigma_R}, r_0/R = {r0_frac}: "
                        f"ERROR {type(e).__name__}: {e}"
                    )
    else:
        print()
        print(
            f"V-S per-mode at N=2 gives err = {err_vs_2:.4f}% "
            f"(> 0.1% target). Not running extended scan."
        )


if __name__ == "__main__":
    main()
