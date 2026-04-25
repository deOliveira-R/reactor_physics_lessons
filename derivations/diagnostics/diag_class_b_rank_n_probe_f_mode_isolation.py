"""Probe F: per-mode contribution to K_bc — which mode breaks the closure?

Created by numerics-investigator on 2026-04-24.

Per probe-cascade SKILL §"drop one factor at a time": the rank-2 closure
adds the n=0 mode (legacy isotropic) PLUS the n=1 mode (Marshak DP_1
partial current with (ρ_max/R)² Jacobian). This probe isolates the
contribution of EACH mode to k_eff:

  variant 1: K_bc = mode-0 only (=rank-1 Mark)
  variant 2: K_bc = mode-0 + mode-1
  variant 3: K_bc = mode-1 only (no isotropic component)

The diff between variant 1 and variant 2 should equal the rank-1 →
rank-2 step. Comparing variant 2 to variant 3 reveals whether the
mode-1 contribution is by itself reasonable or pathological in MR.

This isolates whether the bug is:
  (a) mode-1 primitive misbehaving in MR with σ_t step,
  (b) a normalization mismatch between mode-0 (legacy) and mode-1
      (Marshak), or
  (c) the (2n+1) Marshak weight in the reflection operator.

For each variant we compute k_eff using the same scattering / fission
operator from material A.
"""

from __future__ import annotations

import numpy as np

from orpheus.derivations._xs_library import get_xs
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    BoundaryClosureOperator,
    build_closure_operator,
    build_volume_kernel,
    composite_gl_r,
    compute_P_esc,
    compute_G_bc,
    compute_P_esc_mode,
    compute_G_bc_mode,
    reflection_marshak,
)


def _custom_K_bc(geometry, r_nodes, r_wts, radii, sig_t, *, modes,
                 quad):
    """Build K_bc using only the listed modes. modes is a subset of {0, 1, 2, ...}."""
    R_cell = float(radii[-1])
    N_r = len(r_nodes)
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(N_r)
    ])
    rv = np.array([geometry.radial_volume_weight(float(rj)) for rj in r_nodes])
    divisor = geometry.rank1_surface_divisor(R_cell)

    N_modes = max(modes) + 1
    P = np.zeros((N_modes, N_r))
    G = np.zeros((N_r, N_modes))
    R_mat = np.zeros((N_modes, N_modes))
    R_full = reflection_marshak(N_modes)

    for n in modes:
        if n == 0:
            P_esc = compute_P_esc(geometry, r_nodes, radii, sig_t,
                                  n_angular=quad["n_angular"], dps=quad["dps"])
            G_bc = compute_G_bc(geometry, r_nodes, radii, sig_t,
                                n_surf_quad=quad["n_surf_quad"], dps=quad["dps"])
        else:
            P_esc = compute_P_esc_mode(geometry, r_nodes, radii, sig_t, n,
                                       n_angular=quad["n_angular"], dps=quad["dps"])
            G_bc = compute_G_bc_mode(geometry, r_nodes, radii, sig_t, n,
                                     n_surf_quad=quad["n_surf_quad"], dps=quad["dps"])
        P[n, :] = rv * r_wts * P_esc
        G[:, n] = sig_t_n * G_bc / divisor
        R_mat[n, n] = R_full[n, n]

    op = BoundaryClosureOperator(P=P, G=G, R=R_mat)
    return op.as_matrix()


def _solve_keff_with_K(K_total, sig_t_n, sig_s, nu_sig_f, chi):
    """Direct power iteration for k_eff given K = K_vol + K_bc, 1G."""
    A = np.diag(sig_t_n) - K_total * sig_s
    # Pure 1G with no upscatter, scattering matrix is scalar
    # Σ_t φ - K·Σ_s·φ = (1/k) K·νΣ_f·φ  →  M = inv(A) · K · diag(νΣ_f)
    Minv_K = np.linalg.solve(A, K_total)
    Op = Minv_K * nu_sig_f
    eigvals = np.linalg.eigvals(Op)
    eigvals = eigvals[np.isreal(eigvals)].real
    return float(np.max(eigvals))


def main():
    quad = dict(n_panels_per_region=2, p_order=3,
                n_angular=24, n_rho=24, n_surf_quad=24, dps=15)

    print("=" * 76)
    print("Probe F: per-mode contribution to K_bc (sphere 1G/2R suspect Z)")
    print("=" * 76)

    radii = np.array([0.5, 1.0])
    sig_t = np.array([1.0, 2.0])
    xs_A = get_xs("A", "1g")
    sig_s_scalar = float(xs_A["sig_s"][0, 0])    # 0.5
    nu_sig_f_scalar = float(xs_A["nu"][0] * xs_A["sig_f"][0])  # 2.5 * 0.3 = 0.75

    r_nodes, r_wts, panels = composite_gl_r(
        radii, quad["n_panels_per_region"], quad["p_order"],
        dps=quad["dps"], inner_radius=0.0,
    )
    sig_t_n = np.array([
        sig_t[SPHERE_1D.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])

    K_vol = build_volume_kernel(
        SPHERE_1D, r_nodes, panels, radii, sig_t,
        n_angular=quad["n_angular"], n_rho=quad["n_rho"], dps=quad["dps"],
    )

    print("\n  variant            k_eff      Δ vs rank-1 [%]")
    K_bc_m0 = _custom_K_bc(SPHERE_1D, r_nodes, r_wts, radii, sig_t,
                           modes=[0], quad=quad)
    k_m0 = _solve_keff_with_K(K_vol + K_bc_m0, sig_t_n,
                              sig_s_scalar, nu_sig_f_scalar, chi=1.0)
    print(f"  mode-0 only        {k_m0:.10f}  baseline")

    K_bc_m01 = _custom_K_bc(SPHERE_1D, r_nodes, r_wts, radii, sig_t,
                            modes=[0, 1], quad=quad)
    k_m01 = _solve_keff_with_K(K_vol + K_bc_m01, sig_t_n,
                               sig_s_scalar, nu_sig_f_scalar, chi=1.0)
    print(f"  mode-0 + mode-1    {k_m01:.10f}  {(k_m01-k_m0)/k_m0*100:+.3f}%")

    K_bc_m1 = _custom_K_bc(SPHERE_1D, r_nodes, r_wts, radii, sig_t,
                           modes=[1], quad=quad)
    k_m1 = _solve_keff_with_K(K_vol + K_bc_m1, sig_t_n,
                              sig_s_scalar, nu_sig_f_scalar, chi=1.0)
    print(f"  mode-1 only        {k_m1:.10f}  {(k_m1-k_m0)/k_m0*100:+.3f}%")

    # Compare same on the 1R control where rank-2 is essentially right
    print("\n--- 1R control: σ_t=[1.0], radii=[1.0] ---")
    radii_c = np.array([1.0])
    sig_t_c = np.array([1.0])
    r_nodes_c, r_wts_c, panels_c = composite_gl_r(
        radii_c, quad["n_panels_per_region"], quad["p_order"],
        dps=quad["dps"], inner_radius=0.0,
    )
    sig_t_n_c = np.array([
        sig_t_c[SPHERE_1D.which_annulus(float(r_nodes_c[i]), radii_c)]
        for i in range(len(r_nodes_c))
    ])
    K_vol_c = build_volume_kernel(
        SPHERE_1D, r_nodes_c, panels_c, radii_c, sig_t_c,
        n_angular=quad["n_angular"], n_rho=quad["n_rho"], dps=quad["dps"],
    )
    K_bc_m0_c = _custom_K_bc(SPHERE_1D, r_nodes_c, r_wts_c, radii_c, sig_t_c,
                             modes=[0], quad=quad)
    K_bc_m01_c = _custom_K_bc(SPHERE_1D, r_nodes_c, r_wts_c, radii_c, sig_t_c,
                              modes=[0, 1], quad=quad)
    K_bc_m1_c = _custom_K_bc(SPHERE_1D, r_nodes_c, r_wts_c, radii_c, sig_t_c,
                             modes=[1], quad=quad)
    k_m0_c = _solve_keff_with_K(K_vol_c + K_bc_m0_c, sig_t_n_c,
                                sig_s_scalar, nu_sig_f_scalar, chi=1.0)
    k_m01_c = _solve_keff_with_K(K_vol_c + K_bc_m01_c, sig_t_n_c,
                                 sig_s_scalar, nu_sig_f_scalar, chi=1.0)
    k_m1_c = _solve_keff_with_K(K_vol_c + K_bc_m1_c, sig_t_n_c,
                                sig_s_scalar, nu_sig_f_scalar, chi=1.0)
    print(f"  mode-0 only        {k_m0_c:.10f}  baseline")
    print(f"  mode-0 + mode-1    {k_m01_c:.10f}  {(k_m01_c-k_m0_c)/k_m0_c*100:+.3f}%")
    print(f"  mode-1 only        {k_m1_c:.10f}  {(k_m1_c-k_m0_c)/k_m0_c*100:+.3f}%")


if __name__ == "__main__":
    main()
