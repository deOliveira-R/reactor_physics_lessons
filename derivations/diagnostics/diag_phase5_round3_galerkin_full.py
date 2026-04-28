"""Round 3 — Full Galerkin double integration as last-ditch attempt.

Replace pointwise K_bc[i,j] = K_bc(r_i, r_j) with the full Galerkin
double integral

.. math::
    K^{\\rm Gal}_{ij} = \\int_{V_i} \\int_{V_j} L_i(r) L_j(r')
        K_{\\rm bc}(r, r') \\, dr\\,dr'

over the radial cells V_i, V_j defined by the GL panels. ``L_i(r)``
is a unit indicator on cell i (constant trial functions). The
double integral over r, r' on cells smooths BOTH the diagonal (where
r and r' may coincide) AND the off-diagonal singularity at r = r'.

This is the "natural cure" recommended by R3-B in the M2 memo. If
it converges, it's the production form.
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_round3_adaptive_quadrature import (
    K_INF, NU_SIG_F, R_THIN, SIG_S, SIG_T,
    _F_out_at_mu, _G_in_at_mu, _multi_bounce_factor_half_M1,
    keff_from_K,
)


def _K_bc_pointwise(r_i, r_j, R, sigma, n_quad=64):
    """Per-pair µ integral on visibility cone, half-M1 factor."""
    mu_min_i = float(np.sqrt(max(0.0, 1.0 - (r_i / R) ** 2)))
    mu_min_j = float(np.sqrt(max(0.0, 1.0 - (r_j / R) ** 2)))
    mu_lo = max(mu_min_i, mu_min_j)
    if mu_lo >= 1.0:
        return 0.0
    nodes_unit, wts_unit = np.polynomial.legendre.leggauss(n_quad)
    mu_pts = 0.5 * (nodes_unit + 1.0) * (1.0 - mu_lo) + mu_lo
    mu_wts = 0.5 * (1.0 - mu_lo) * wts_unit
    f_mb = _multi_bounce_factor_half_M1(mu_pts, sigma, R)
    integrand = np.zeros(n_quad)
    for q in range(n_quad):
        integrand[q] = (
            _G_in_at_mu(r_i, R, mu_pts[q], sigma)
            * _F_out_at_mu(r_j, R, mu_pts[q], sigma)
            * f_mb[q]
        )
    return 2.0 * float(np.sum(mu_wts * integrand))


def compute_K_bc_full_galerkin(
    geometry, r_nodes, r_wts, panels, radii, sig_t, *,
    n_quad=64, n_inner=4,
):
    r"""Full Galerkin double integration of the continuous-µ K_bc kernel.

    For each pair (i, j) of GL panels, integrate

    .. math::
        K_{ij} = \\int_{r_i^L}^{r_i^R} \\int_{r_j^L}^{r_j^R}
            L_i(r) L_j(r') K_{\\rm bc}(r, r') \\, dr\\,dr' / w_i

    with constant trial functions (so L_i = 1 on cell i and 0 elsewhere).
    Inner quadrature: GL with ``n_inner`` nodes per cell.

    The /w_i normalisation makes K_ij have the same units as a Nyström
    matrix (rate × cell width on the receive side, source unit on the
    send side).

    NOTE: This requires the ``panels`` structure from ``composite_gl_r``
    to know cell boundaries. We use the GL nodes and weights as
    representative of the panel.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])

    # We need panel boundaries. composite_gl_r returns ``panels`` as
    # ndarray of (panel_lo, panel_hi) edges interleaved with nodes.
    # For simplicity, use the half-distance to neighbours as cell width.
    N = len(r_nodes)
    cell_lo = np.zeros(N)
    cell_hi = np.zeros(N)
    for i in range(N):
        if i == 0:
            cell_lo[i] = 0.0
        else:
            cell_lo[i] = 0.5 * (r_nodes[i - 1] + r_nodes[i])
        if i == N - 1:
            cell_hi[i] = R
        else:
            cell_hi[i] = 0.5 * (r_nodes[i] + r_nodes[i + 1])

    rv = np.array([
        geometry.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(N)
    ])
    divisor = geometry.rank1_surface_divisor(R)

    # Inner GL on [-1,1]
    inner_nodes, inner_wts = np.polynomial.legendre.leggauss(n_inner)

    K_total = np.zeros((N, N))

    for i in range(N):
        # Map inner GL to [cell_lo[i], cell_hi[i]]
        wi = cell_hi[i] - cell_lo[i]
        ri_pts = 0.5 * (inner_nodes + 1.0) * wi + cell_lo[i]
        ri_wts = 0.5 * wi * inner_wts
        for j in range(N):
            wj = cell_hi[j] - cell_lo[j]
            rj_pts = 0.5 * (inner_nodes + 1.0) * wj + cell_lo[j]
            rj_wts = 0.5 * wj * inner_wts

            # Double integral
            K_ij_galerkin = 0.0
            for ai in range(n_inner):
                for aj in range(n_inner):
                    K_pt = _K_bc_pointwise(
                        ri_pts[ai], rj_pts[aj], R, sigma, n_quad=n_quad,
                    )
                    # Galerkin: integrate L_i(r) L_j(r') K(r,r') dr dr'
                    K_ij_galerkin += ri_wts[ai] * rj_wts[aj] * K_pt
            # Apply Nyström-conversion: divide by w_i to recover
            # ``K_ij·q_j`` form (q_j is source per unit volume,
            # K_ij has source-cell-volume baked in already).
            # Use ORPHEUS volume weights similar to per-pair:
            #   K_total[i,j] = (sig_t_n[i]/divisor) · rv[j] · K_galerkin / wi
            # where K_galerkin already has the r' integration (cell j)
            # baked in via rj_wts.
            # Actually: in Nyström convention K_ij · w_j · q_j is the
            # contribution. So K_total[i,j] should be the weighted
            # version.
            K_total[i, j] = (
                (sig_t_n[i] / divisor)
                * rv[j]  # already in K_pt? NO — that was at single r_j
                * (K_ij_galerkin / wi)  # average over receive cell i
                # / wj ?  K_pt has no source-cell-volume baked in either
            )
            # Simpler: K_total[i,j] should match the Nyström K[i,j] at
            # the limit n_inner → 1 (delta functions at r_i, r_j).
            # That is the per-pair point evaluation × r_wts[j].
            # Going beyond n_inner=1 is the cell-averaged correction.

    return K_total


def test_galerkin_full(capsys):
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        # Coarse grid first
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== Full Galerkin double-integration ===")
        print(f"  N_r = {N}, k_inf = {K_INF:.6f}, k_heb = {k_heb:.6f}")
        print(f"  Q   | n_in | k_eff       | rel_heb")
        for n_in in (2, 4, 8):
            for Q in (32, 64, 128):
                K_bc = compute_K_bc_full_galerkin(
                    SPHERE_1D, r_nodes, r_wts, panels,
                    radii, sig_t_g,
                    n_quad=Q, n_inner=n_in,
                )
                k = keff_from_K(
                    K_vol + K_bc, sig_t_arr, nu_sf_arr, sig_s_arr,
                )
                rel_h = (k - k_heb) / k_heb * 100
                print(f"  {Q:3d} | {n_in:3d}  | {k:.6f}   | {rel_h:+.4f}%")
