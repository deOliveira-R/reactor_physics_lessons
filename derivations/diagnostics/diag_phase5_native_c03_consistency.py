"""Phase 5 Front C — consistency probe: do F_out, G_in reproduce
the Phase 4 rank-N moments under polynomial projection?

If yes → the µ-resolved primitives are CORRECT; the Q-divergence in
Probe D comes from the multi-bounce T(µ) singularity at µ→0.

If no → the F_out / G_in derivation has a Jacobian bug that needs
fixing before the K_bc construction works.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _shifted_legendre_eval,
    composite_gl_r,
    compute_G_bc,
    compute_G_bc_mode,
    compute_P_esc,
    compute_P_esc_mode,
)

from derivations.diagnostics.diag_phase5_native_c01_orpheus_form import (
    F_out_mu_sphere, G_in_mu_sphere, R_THIN, SIG_T,
)


def test_F_out_reproduces_P_esc_n0(capsys):
    """Project F_out_mu against P̃_0 = 1 over µ ∈ [0,1] to recover the
    Phase 4 P_esc[0,j].
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        # Phase 4 P_esc (rank-1, n=0 — the isotropic-source escape)
        # The Phase 4 build uses compute_P_esc for n=0, NOT
        # compute_P_esc_mode. Let me get both.
        P_esc_orig = compute_P_esc(
            SPHERE_1D, r_nodes, radii, sig_t_g,
            n_angular=24, dps=15,
        )
        P_esc_mode_0 = compute_P_esc_mode(
            SPHERE_1D, r_nodes, radii, sig_t_g, n_mode=0,
            n_angular=24, dps=15,
        )
        P_esc_mode_1 = compute_P_esc_mode(
            SPHERE_1D, r_nodes, radii, sig_t_g, n_mode=1,
            n_angular=24, dps=15,
        )

        # Native: integrate F_out against P̃_n over µ ∈ [0,1]
        Q = 128
        nodes, wts = np.polynomial.legendre.leggauss(Q)
        mu_pts = 0.5 * (nodes + 1.0)
        mu_wts = 0.5 * wts
        F_out = F_out_mu_sphere(r_nodes, radii, sig_t_g, mu_pts)
        # Project against P̃_0 = 1
        P_esc_native_0 = np.sum(mu_wts[:, None] * F_out, axis=0)
        # Project against P̃_1(µ) = 2µ-1
        Pt_1 = _shifted_legendre_eval(1, mu_pts)
        P_esc_native_1 = np.sum(
            (mu_wts * Pt_1)[:, None] * F_out, axis=0,
        )

        print(f"\n=== F_out projection consistency ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, Q={Q}")
        print(f"  j |  r_j   | P_esc(n=0,Phase4) | P_esc_mode(n=0) | "
              f"P_esc_native(n=0) | rel_err_orig | rel_err_mode")
        for j in range(len(r_nodes)):
            err_orig = (P_esc_native_0[j] - P_esc_orig[j]) / P_esc_orig[j]
            err_mode = (P_esc_native_0[j] - P_esc_mode_0[j]) / P_esc_mode_0[j]
            print(
                f"  {j} | {r_nodes[j]:.3f} | {P_esc_orig[j]:.6e}      | "
                f"{P_esc_mode_0[j]:.6e}    | {P_esc_native_0[j]:.6e}      | "
                f"{err_orig:+.3e} | {err_mode:+.3e}"
            )

        print(f"\n  j |  P_esc_mode(n=1, Phase4) | P_esc_native(n=1) | "
              f"rel_err")
        for j in range(len(r_nodes)):
            err = (P_esc_native_1[j] - P_esc_mode_1[j]) / abs(P_esc_mode_1[j])
            print(
                f"  {j} | {P_esc_mode_1[j]:.6e}            | "
                f"{P_esc_native_1[j]:.6e}      | {err:+.3e}"
            )


def test_G_in_reproduces_G_bc_n0(capsys):
    """Project G_in_mu against P̃_0 = 1 over µ ∈ [0,1] to recover the
    Phase 4 G_bc[i,0].
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        # Phase 4 G_bc rank-1 (uses compute_G_bc directly)
        G_bc_orig = compute_G_bc(
            SPHERE_1D, r_nodes, radii, sig_t_g,
            n_surf_quad=24, dps=15,
        )
        G_bc_mode_0 = compute_G_bc_mode(
            SPHERE_1D, r_nodes, radii, sig_t_g, n_mode=0,
            n_surf_quad=24, dps=15,
        )
        G_bc_mode_1 = compute_G_bc_mode(
            SPHERE_1D, r_nodes, radii, sig_t_g, n_mode=1,
            n_surf_quad=24, dps=15,
        )

        Q = 128
        nodes, wts = np.polynomial.legendre.leggauss(Q)
        mu_pts = 0.5 * (nodes + 1.0)
        mu_wts = 0.5 * wts
        G_in = G_in_mu_sphere(r_nodes, radii, sig_t_g, mu_pts)
        # Project against P̃_0 = 1
        G_bc_native_0 = np.sum(G_in * mu_wts[None, :], axis=1)
        Pt_1 = _shifted_legendre_eval(1, mu_pts)
        G_bc_native_1 = np.sum(
            G_in * (mu_wts * Pt_1)[None, :], axis=1,
        )

        print(f"\n=== G_in projection consistency ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, Q={Q}")
        print(f"  i |  r_i   | G_bc(Phase4)     | G_bc_mode(0) | "
              f"G_bc_native(0)  | rel_err_orig")
        for i in range(len(r_nodes)):
            err = (G_bc_native_0[i] - G_bc_orig[i]) / G_bc_orig[i]
            print(
                f"  {i} | {r_nodes[i]:.3f} | {G_bc_orig[i]:.6e}      | "
                f"{G_bc_mode_0[i]:.6e}      | {G_bc_native_0[i]:.6e}     | "
                f"{err:+.3e}"
            )

        print(f"\n  i |  G_bc_mode(n=1)         | G_bc_native(n=1) | "
              f"rel_err")
        for i in range(len(r_nodes)):
            err = (G_bc_native_1[i] - G_bc_mode_1[i]) / abs(G_bc_mode_1[i])
            print(
                f"  {i} | {G_bc_mode_1[i]:.6e}           | "
                f"{G_bc_native_1[i]:.6e}     | {err:+.3e}"
            )
