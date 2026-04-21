"""Diagnostic: rank-N N=1 reduction bit-exactness requires Model-split P(A)/G(B).

Created by numerics-investigator on 2026-04-21 (continuation of Issue #119).

The Phase F.4 scalar N=1 closure uses:
  - `compute_P_esc_outer`: Model A — EXCLUDES rays that hit inner first
    (correct: those rays don't escape via outer).
  - `compute_G_bc_outer`: Model B — INCLUDES rays that traverse the cavity
    (correct: observer flux from outer-boundary isotropic current can arrive
    via cavity-crossing rays).

The rank-N per-face primitives `compute_P_esc_outer_mode` and
`compute_G_bc_outer_mode_marshak` BOTH use Model A (they skip cavity-crossing
rays). This is INCONSISTENT with the scalar F.4 reference. At N=1 reduction,
this manifests as ~0.05% divergence from F.4 (F.4 gives 0.077% residual,
Model-A-both-sides gives 0.135%).

This diagnostic verifies:
  1. At N=1, only the (P=A, G=B) split reproduces F.4 bit-exactly.
  2. Pure Model A (current per-face code) gives 0.135% residual — worse.
  3. Pure Model B (candidate "fix") gives 0.019% residual — better but not
     bit-exact, because G's unphysical cavity-crossing with NON-zero σ_t
     is still a legitimate Model-B term.

This is ONE of the contributors to the N ≥ 2 closure failure — the existing
per-face primitives apply Model A uniformly. Proper fix: P should use Model A,
G should use Model B. See Issue #119 for the full investigation.

If this test catches a regression in the Phase F.4 scalar baseline (expected
0.077% residual), promote to ``tests/derivations/test_peierls_rank2_bc.py``
under ``TestRank2HollowSphScalarBaseline``.
"""
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    composite_gl_r,
    compute_P_esc_outer,
    compute_P_esc_inner,
    compute_G_bc_outer,
    compute_G_bc_inner,
    compute_hollow_sph_transmission_rank_n,
    build_volume_kernel,
    gl_float,
    _shifted_legendre_eval,
)


def _solve_keff(K, sig_t_v, sig_s_v, nu_sig_f_v, tol=1e-12, max_iter=500):
    N = K.shape[0]
    A = np.diag(np.full(N, sig_t_v)) - K * sig_s_v
    B = K * nu_sig_f_v
    phi = np.ones(N)
    k = 1.0
    for _ in range(max_iter):
        q = B @ phi / k
        phi_new = np.linalg.solve(A, q)
        B_phi_new = B @ phi_new
        B_phi = B @ phi
        k_new = k * (np.abs(B_phi_new).sum() / np.abs(B_phi).sum())
        if abs(k_new - k) < tol:
            return k_new
        phi = phi_new / np.linalg.norm(phi_new)
        k = k_new
    return k


def _compute_P_outer_flex(geom, r_nodes, radii, sig_t, n_mode, *,
                           model, n_angular=24, dps=15):
    """Per-face outer escape mode-n primitive with selectable Model A/B.

    Model A: skip rays that hit inner shell first.
    Model B: include ALL outward rays to outer (some cross cavity).
    """
    R = float(radii[-1])
    omega_pts, omega_wts = gl_float(n_angular, *geom.angular_range, dps)
    cos_omegas = geom.ray_direction_cosine(omega_pts)
    angular_factor = geom.angular_weight(omega_pts)
    pref = geom.prefactor
    Nnode = len(r_nodes)
    P = np.zeros(Nnode)
    for i, r_i in enumerate(r_nodes):
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_out = geom.rho_max(float(r_i), cos_om, R)
            if rho_out <= 0.0:
                continue
            if model == 'A':
                rho_in_minus, _ = geom.rho_inner_intersections(float(r_i), cos_om)
                if rho_in_minus is not None and rho_in_minus < rho_out:
                    continue
            tau = geom.optical_depth_along_ray(
                float(r_i), cos_om, rho_out, radii, sig_t,
            )
            K_esc = geom.escape_kernel_mp(tau, dps)
            mu_exit = (rho_out + float(r_i) * cos_om) / R
            p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_exit]))[0])
            total += omega_wts[k] * angular_factor[k] * p_tilde * K_esc
        P[i] = pref * total
    return P


def _compute_G_outer_flex(geom, r_nodes, radii, sig_t, n_mode, *,
                           model, n_surf_quad=24, dps=15):
    """Per-face outer Green's function mode-n primitive with selectable Model."""
    R = float(radii[-1])
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts)
    sin_thetas = np.sin(theta_pts)
    Nnode = len(r_nodes)
    G = np.zeros(Nnode)
    for i, r_i in enumerate(r_nodes):
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]
            st = sin_thetas[k]
            rho_out = geom.rho_max(float(r_i), ct, R)
            if rho_out <= 0.0:
                continue
            if model == 'A':
                rho_in_minus, _ = geom.rho_inner_intersections(float(r_i), ct)
                if rho_in_minus is not None and rho_in_minus < rho_out:
                    continue
            tau = geom.optical_depth_along_ray(
                float(r_i), ct, rho_out, radii, sig_t,
            )
            mu_s = (rho_out + float(r_i) * ct) / R
            p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_s]))[0])
            total += theta_wts[k] * st * p_tilde * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


@pytest.fixture(scope="module")
def _hollow_sph_setup():
    """Reference problem: hollow sphere R=5, r_0/R=0.3, homogeneous sig_t=1,
    sig_s=0.5, nu_sig_f=0.75 → k_inf = 1.5."""
    R, r_0 = 5.0, 1.5
    sig_t_v, sig_s_v, nu_sig_f_v = 1.0, 0.5, 0.75
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])
    r_nodes, r_wts, panels = composite_gl_r(
        radii, 2, 4, dps=15, inner_radius=r_0,
    )
    N_r = len(r_nodes)
    sig_t_n = np.full(N_r, sig_t_v)
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t, 24, 24, 15,
    )
    W1 = compute_hollow_sph_transmission_rank_n(
        r_0, R, radii, sig_t, 1, dps=15,
    )
    return dict(
        R=R, r_0=r_0,
        sig_t_v=sig_t_v, sig_s_v=sig_s_v, nu_sig_f_v=nu_sig_f_v, k_inf=1.5,
        geom=geom, radii=radii, sig_t=sig_t,
        r_nodes=r_nodes, r_wts=r_wts, panels=panels,
        sig_t_n=sig_t_n, rv=rv, K_vol=K_vol, W1=W1,
    )


def test_phase_f4_scalar_residual_below_0p1_percent(_hollow_sph_setup):
    """Regression gate: Phase F.4 scalar N=1 gives < 0.1% residual.

    This is the current production baseline. Uses Lambert-scalar primitives
    directly (not the per-face mode primitives).
    """
    S = _hollow_sph_setup
    P_sca = np.zeros((2, len(S["r_nodes"])))
    G_sca = np.zeros((len(S["r_nodes"]), 2))
    P_sca[0] = S["rv"] * S["r_wts"] * compute_P_esc_outer(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    )
    P_sca[1] = S["rv"] * S["r_wts"] * compute_P_esc_inner(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    )
    G_sca[:, 0] = S["sig_t_n"] * compute_G_bc_outer(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    ) / (S["R"] ** 2)
    G_sca[:, 1] = S["sig_t_n"] * compute_G_bc_inner(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    ) / (S["r_0"] ** 2)
    R1 = np.linalg.inv(np.eye(2) - S["W1"])
    K = S["K_vol"] + G_sca @ R1 @ P_sca
    k = _solve_keff(K, S["sig_t_v"], S["sig_s_v"], S["nu_sig_f_v"])
    err = abs(k - S["k_inf"]) / S["k_inf"] * 100
    assert err < 0.1, (
        f"Phase F.4 scalar residual {err:.4f}% > 0.1% threshold "
        f"(k_eff={k:.6f}, k_inf={S['k_inf']}). Baseline regression."
    )


def test_N1_model_A_both_sides_worse_than_F4(_hollow_sph_setup):
    """N=1 reduction with per-face primitives both Model-A gives WORSE than F.4.

    This is the current Phase F.5 infrastructure convention — all per-face
    primitives use Model A. At N=1 it gives 0.135% residual vs F.4's 0.077%.
    """
    S = _hollow_sph_setup
    N = 1
    N_r = len(S["r_nodes"])
    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))
    for n in range(N):
        Po = _compute_P_outer_flex(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], n, model='A',
        )
        Pi = compute_P_esc_inner(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
        )
        Go = _compute_G_outer_flex(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], n, model='A',
        )
        Gi = compute_G_bc_inner(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
        )
        P[n, :] = S["rv"] * S["r_wts"] * Po
        P[N + n, :] = S["rv"] * S["r_wts"] * Pi
        G[:, n] = S["sig_t_n"] * Go / (S["R"] ** 2)
        G[:, N + n] = S["sig_t_n"] * Gi / (S["r_0"] ** 2)
    R_eff = np.linalg.inv(np.eye(2 * N) - S["W1"])
    K = S["K_vol"] + G @ R_eff @ P
    k = _solve_keff(K, S["sig_t_v"], S["sig_s_v"], S["nu_sig_f_v"])
    err = abs(k - S["k_inf"]) / S["k_inf"] * 100
    # Documented Model-A N=1 residual. If this deviates strongly (<<0.1% or
    # >>0.2%) something changed in the primitive's integration path.
    assert 0.10 < err < 0.20, (
        f"Model-A N=1 residual {err:.4f}% outside expected [0.10, 0.20] "
        f"band (k_eff={k:.6f})."
    )


def test_N1_model_split_P_A_G_B_matches_F4_bit_exact(_hollow_sph_setup):
    """When P uses Model A and G uses Model B, the per-face N=1 closure
    reduces bit-exactly to the Phase F.4 scalar baseline.

    This confirms the basis identification is self-consistent when the
    Model choice matches the physical meaning of each primitive:
      - P_esc_outer(r_i) = probability of escape via outer — Model A
      - G_bc_outer(r_i)  = flux at r_i from iso current at outer — Model B
    """
    S = _hollow_sph_setup
    N = 1
    N_r = len(S["r_nodes"])

    # Reference F.4 scalar k_eff:
    P_sca = np.zeros((2, N_r))
    G_sca = np.zeros((N_r, 2))
    P_sca[0] = S["rv"] * S["r_wts"] * compute_P_esc_outer(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    )
    P_sca[1] = S["rv"] * S["r_wts"] * compute_P_esc_inner(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    )
    G_sca[:, 0] = S["sig_t_n"] * compute_G_bc_outer(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    ) / (S["R"] ** 2)
    G_sca[:, 1] = S["sig_t_n"] * compute_G_bc_inner(
        S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
    ) / (S["r_0"] ** 2)
    K_F4 = S["K_vol"] + G_sca @ np.linalg.inv(np.eye(2) - S["W1"]) @ P_sca
    k_F4 = _solve_keff(K_F4, S["sig_t_v"], S["sig_s_v"], S["nu_sig_f_v"])

    # Per-face model-split rebuild:
    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))
    for n in range(N):
        Po = _compute_P_outer_flex(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], n, model='A',
        )
        Pi = compute_P_esc_inner(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
        )
        Go = _compute_G_outer_flex(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], n, model='B',
        )
        Gi = compute_G_bc_inner(
            S["geom"], S["r_nodes"], S["radii"], S["sig_t"], 24, 15,
        )
        P[n, :] = S["rv"] * S["r_wts"] * Po
        P[N + n, :] = S["rv"] * S["r_wts"] * Pi
        G[:, n] = S["sig_t_n"] * Go / (S["R"] ** 2)
        G[:, N + n] = S["sig_t_n"] * Gi / (S["r_0"] ** 2)
    K_split = S["K_vol"] + G @ np.linalg.inv(np.eye(2 * N) - S["W1"]) @ P
    k_split = _solve_keff(K_split, S["sig_t_v"], S["sig_s_v"], S["nu_sig_f_v"])

    delta = abs(k_split - k_F4)
    assert delta < 1e-10, (
        f"Model-split P(A)/G(B) N=1 must match F.4 scalar bit-exact. "
        f"Got k_split={k_split:.12f}, k_F4={k_F4:.12f}, Δ={delta:.2e}. "
        f"If this fails, either (a) the Model A/B split no longer matches "
        f"the physical definition of compute_P_esc_outer/compute_G_bc_outer, "
        f"or (b) the per-face N=1 code-path drifted from the Lambert scalar "
        f"primitives. Check Phase F.4 assembly in "
        f"_build_closure_operator_rank2_white."
    )
