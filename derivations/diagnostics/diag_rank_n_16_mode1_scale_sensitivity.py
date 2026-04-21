"""Diagnostic: at N=2 the mode-1 P and G primitives contribute a SPURIOUS
amplitude that scales with optical thickness, not a fixed factor.

Created by numerics-investigator on 2026-04-21 (continuation of Issue #119).

## Summary of findings

The full N=2 per-face closure `G · (I - W)^{-1} · P` gives 3.87% residual
at R=5, r_0/R=0.3, sig_t=1 (vs F.4 N=1's 0.077%). Adding mode 1 DEGRADES
the closure.

Forensic zero-out test: if we set the mode-1 rows of P (or cols of G) to
zero, residual drops to ~0.17% — nearly matching F.4. This localizes the
spurious contribution to the **mode-1 primitive's amplitude**, not to the
transmission matrix W's mode-1 blocks.

Fine scan of a uniform scaling factor `c` applied to the mode-1 P and G
blocks finds a minimum at c ≈ 0.16 (err ~ 0.001%) — but c varies with
optical thickness:

    sig_t=0.5: c_opt ≈ 0.05
    sig_t=1.0: c_opt ≈ 0.16
    sig_t=2.0: c_opt ≈ 0.36
    sig_t=4.0: c_opt ≈ 0.60

and with cell size at fixed r_0/R:

    R=5,  r_0=1.5 (r_0/R=0.3), sig_t=1: c_opt ≈ 0.16
    R=10, r_0=3.0 (r_0/R=0.3), sig_t=1: c_opt ≈ 0.36
    R=3,  r_0=0.9 (r_0/R=0.3), sig_t=1: c_opt ≈ 0.05

This PROVES the bug is NOT a missing simple factor (e.g. 1/3, 1/(2n+1),
sqrt(2n+1), 1/2). The optimal scaling depends on ΣR = optical thickness.
The correct fix must be a basis/normalization restructuring that reduces
to the right amplitude across problem parameters.

If this test catches a regression (e.g. someone applied a fixed-factor
"fix" that changes N=2 residual from 3.87% to ~0.1%), promote to
``tests/derivations/test_peierls_rank2_bc.py`` under a class like
``TestRankNWhiteBCN2OpenQuestion`` — but FLIP the assertion: the test
should now assert ``err < 0.1%`` if Issue #119 is resolved. Update
message accordingly.
"""
import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    composite_gl_r,
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


def _P_outer(geom, r_nodes, radii, sig_t, n_mode, *,
              model='A', n_angular=24, dps=15):
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
            tau = geom.optical_depth_along_ray(float(r_i), cos_om, rho_out, radii, sig_t)
            K_esc = geom.escape_kernel_mp(tau, dps)
            mu_exit = (rho_out + float(r_i) * cos_om) / R
            p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_exit]))[0])
            total += omega_wts[k] * angular_factor[k] * p_tilde * K_esc
        P[i] = pref * total
    return P


def _P_inner(geom, r_nodes, radii, sig_t, n_mode, *,
              n_angular=24, dps=15):
    r_0 = float(geom.inner_radius)
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
            rho_in_minus, _ = geom.rho_inner_intersections(float(r_i), cos_om)
            if rho_in_minus is None:
                continue
            tau = geom.optical_depth_along_ray(float(r_i), cos_om, rho_in_minus, radii, sig_t)
            K_esc = geom.escape_kernel_mp(tau, dps)
            sin_om = np.sqrt(max(0.0, 1.0 - cos_om * cos_om))
            h_sq = float(r_i) ** 2 * sin_om ** 2
            mu_exit_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
            mu_exit = float(np.sqrt(mu_exit_sq))
            p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_exit]))[0])
            total += omega_wts[k] * angular_factor[k] * p_tilde * K_esc
        P[i] = pref * total
    return P


def _G_outer(geom, r_nodes, radii, sig_t, n_mode, *,
              model='B', n_surf_quad=24, dps=15):
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
            tau = geom.optical_depth_along_ray(float(r_i), ct, rho_out, radii, sig_t)
            mu_s = (rho_out + float(r_i) * ct) / R
            p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_s]))[0])
            total += theta_wts[k] * st * p_tilde * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


def _G_inner(geom, r_nodes, radii, sig_t, n_mode, *,
              n_surf_quad=24, dps=15):
    r_0 = float(geom.inner_radius)
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
            rho_in_minus, _ = geom.rho_inner_intersections(float(r_i), ct)
            if rho_in_minus is None:
                continue
            tau = geom.optical_depth_along_ray(float(r_i), ct, rho_in_minus, radii, sig_t)
            sin_om = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
            h_sq = float(r_i) ** 2 * sin_om ** 2
            mu_s_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
            mu_s = float(np.sqrt(mu_s_sq))
            p_tilde = float(_shifted_legendre_eval(n_mode, np.array([mu_s]))[0])
            total += theta_wts[k] * st * p_tilde * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


def _setup(R, r_0, sig_t_v):
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_v])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=15, inner_radius=r_0)
    N_r = len(r_nodes)
    sig_t_n = np.full(N_r, sig_t_v)
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])
    K_vol = build_volume_kernel(geom, r_nodes, panels, radii, sig_t, 24, 24, 15)
    W_N2 = compute_hollow_sph_transmission_rank_n(r_0, R, radii, sig_t, 2, dps=15)
    N = 2
    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))
    for n in range(N):
        Po = _P_outer(geom, r_nodes, radii, sig_t, n)
        Pi = _P_inner(geom, r_nodes, radii, sig_t, n)
        Go = _G_outer(geom, r_nodes, radii, sig_t, n)
        Gi = _G_inner(geom, r_nodes, radii, sig_t, n)
        P[n, :] = rv * r_wts * Po
        P[N + n, :] = rv * r_wts * Pi
        G[:, n] = sig_t_n * Go / (R * R)
        G[:, N + n] = sig_t_n * Gi / (r_0 * r_0)
    return dict(K_vol=K_vol, W_N2=W_N2, P=P, G=G, r_0=r_0, R=R)


def _N2_err(setup, sig_t_v, sig_s_v, nu_sig_f_v, k_inf, c_P, c_G):
    P = setup["P"].copy()
    G = setup["G"].copy()
    # Scale mode-1 rows of P (indices 1 outer, 3 inner) and cols of G.
    P[1, :] *= c_P
    P[3, :] *= c_P
    G[:, 1] *= c_G
    G[:, 3] *= c_G
    K_bc = G @ np.linalg.inv(np.eye(4) - setup["W_N2"]) @ P
    K = setup["K_vol"] + K_bc
    k = _solve_keff(K, sig_t_v, sig_s_v, nu_sig_f_v)
    return abs(k - k_inf) / k_inf * 100, k


@pytest.fixture(scope="module")
def _setup_R5_r1p5():
    return _setup(5.0, 1.5, 1.0)


def test_N2_current_primitives_give_known_residual_3p9_pct(_setup_R5_r1p5):
    """Current per-face code N=2 residual is 3.87% (within 3.5-4.1%).

    This documents the open Issue #119 baseline. If the residual drops
    significantly, someone has fixed the closure — promote this test and
    invert its assertion.
    """
    setup = _setup_R5_r1p5
    err, k = _N2_err(setup, 1.0, 0.5, 0.75, 1.5, c_P=1.0, c_G=1.0)
    assert 3.5 < err < 4.1, (
        f"Baseline N=2 residual (current primitives, c_P=c_G=1) moved out "
        f"of documented range [3.5%, 4.1%]: got {err:.4f}% (k_eff={k:.6f}). "
        f"If this DROPPED below 3.5%, the Issue #119 closure was fixed — "
        f"update this test to assert err < 0.1% and promote to the suite."
    )


def test_N2_zero_mode1_P_restores_F4_accuracy(_setup_R5_r1p5):
    """Zeroing the mode-1 rows of P drops residual from 3.87% to ~0.17%.

    This localizes the spurious contribution to the mode-1 primitive's
    amplitude, not to the transmission matrix W's mode-1 blocks (zeroing
    W mode-1 blocks does NOT restore accuracy — see the scan in memory).
    """
    setup = _setup_R5_r1p5
    err, k = _N2_err(setup, 1.0, 0.5, 0.75, 1.5, c_P=0.0, c_G=1.0)
    assert err < 0.25, (
        f"Zeroing P mode-1 must drop residual near F.4 baseline. "
        f"Got {err:.4f}% (k_eff={k:.6f}). Expected < 0.25%."
    )


def test_N2_optimal_mode1_scale_is_geometry_dependent(_setup_R5_r1p5):
    """The optimal mode-1 scale `c` that minimizes N=2 residual at
    fixed r_0/R=0.3 varies with sig_t:

    - sig_t=1.0: optimal c ≈ 0.16
    - sig_t=2.0: optimal c ≈ 0.36

    This PROVES the closure error is NOT due to a missing constant factor
    (e.g. 1/3, 1/(2n+1)) — the correct fix must be a basis/normalization
    restructuring, not a rescaling of the existing primitives.
    """
    # Find optimal c at sig_t=1.0 and sig_t=2.0:
    setup_sigt1 = _setup(5.0, 1.5, 1.0)
    setup_sigt2 = _setup(5.0, 1.5, 2.0)

    def find_opt(setup, sig_t_v):
        sig_s_v = 0.5 * sig_t_v
        nu_sig_f_v = 0.75 * sig_t_v
        k_inf = nu_sig_f_v / (sig_t_v - sig_s_v)
        best = (10.0, None, None)
        for c in np.linspace(0.02, 0.8, 80):
            err, k = _N2_err(setup, sig_t_v, sig_s_v, nu_sig_f_v, k_inf,
                              c_P=c, c_G=c)
            if err < best[0]:
                best = (err, c, k)
        return best

    err1, c1, _ = find_opt(setup_sigt1, 1.0)
    err2, c2, _ = find_opt(setup_sigt2, 2.0)
    # Optimal c at sig_t=1 is in [0.12, 0.20]; at sig_t=2 in [0.30, 0.40].
    assert 0.10 < c1 < 0.22, f"Expected c_opt(sig_t=1) ≈ 0.16, got {c1:.3f}"
    assert 0.28 < c2 < 0.42, f"Expected c_opt(sig_t=2) ≈ 0.36, got {c2:.3f}"
    # The ratio c2/c1 should be > 1.5 (confirming geometry dependence):
    assert c2 / c1 > 1.5, (
        f"Optimal mode-1 scale should vary with optical thickness. "
        f"Got c(sig_t=1)={c1:.3f}, c(sig_t=2)={c2:.3f}. If the ratio is "
        f"~1 the fix might be a simple factor — revisit the missing-factor "
        f"hypothesis."
    )
