"""L1 CP-sphere flux agreement with the Peierls integral-equation reference.

Phase A of the Phase-4.3 sphere campaign. Mirrors the Phase-4.2
cylinder test ``test_peierls_cylinder_flux.py``: CP spherical flux
shapes are compared against the polar-form Peierls reference built
on the unified :class:`~peierls_geometry.CurvilinearGeometry`
(``kind="sphere-1d"``).

Scope limits imposed by the rank-1 white-BC closure
(:func:`~peierls_geometry.build_white_bc_correction` and Issue #103):

- At :math:`R \\lesssim 2` MFP the rank-1 Peierls white BC over/
  undershoots CP's flat-source white BC, making the per-point flux
  comparison unreliable there.
- At :math:`R \\ge 5` MFP both methods converge to the infinite-medium
  spectrum within ~1 %, and the flux-shape agreement is tight.

This file tests the **thick-R regime**. A follow-up commit (once the
higher-rank white BC lands per Issue #103 / N1) will extend the test
to thin spheres."""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.cp.solver import CPParams, solve_cp
from orpheus.derivations._xs_library import get_mixture
from orpheus.derivations.peierls_geometry import (
    build_volume_kernel,
    build_white_bc_correction,
    composite_gl_r,
)
from orpheus.derivations.peierls_sphere import (
    GEOMETRY,
    PeierlsSphereSolution,
)
from orpheus.geometry import CoordSystem, Mesh1D


_SIG_T = 1.0
_SIG_S = 0.5
_NU_SIG_F = 0.75
_K_INF = _NU_SIG_F / (_SIG_T - _SIG_S)  # = 1.5


def _build_peierls_sphere_reference(
    R: float, n_panels: int = 3, p_order: int = 5,
    n_theta: int = 20, n_rho: int = 20, n_phi: int = 20, dps: int = 20,
) -> PeierlsSphereSolution:
    """Solve the 1G 1-region Peierls sphere with white BC and return the
    :class:`PeierlsSphereSolution` carrying both k_eff and interpolable φ."""
    radii = np.array([R])
    sig_t_arr = np.array([_SIG_T])
    sig_s_arr = np.array([_SIG_S])
    nu_sig_f_arr = np.array([_NU_SIG_F])

    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels, p_order, dps=dps,
    )
    K_vol = build_volume_kernel(
        GEOMETRY, r_nodes, panels, radii, sig_t_arr,
        n_angular=n_theta, n_rho=n_rho, dps=dps,
    )
    K_bc = build_white_bc_correction(
        GEOMETRY, r_nodes, r_wts, radii, sig_t_arr,
        n_angular=n_theta, n_surf_quad=n_phi, dps=dps,
    )
    K = K_vol + K_bc

    N = len(r_nodes)
    sig_t_n = np.full(N, _SIG_T)
    sig_s_n = np.full(N, _SIG_S)
    nu_sig_f_n = np.full(N, _NU_SIG_F)
    A = np.diag(sig_t_n) - K * sig_s_n[np.newaxis, :]
    B = K * nu_sig_f_n[np.newaxis, :]

    phi = np.ones(N)
    k_val = 1.0
    B_phi = B @ phi
    prod_old = np.abs(B_phi).sum()
    for _ in range(300):
        q = B_phi / k_val
        phi_new = np.linalg.solve(A, q)
        B_phi_new = B @ phi_new
        prod_new = np.abs(B_phi_new).sum()
        k_new = k_val * prod_new / prod_old if prod_old > 0 else k_val
        nrm = np.abs(phi_new).sum()
        if nrm > 0:
            phi_new = phi_new / nrm
        B_phi_norm = B @ phi_new
        prod_norm = np.abs(B_phi_norm).sum()
        converged = abs(k_new - k_val) < 1e-10
        phi, k_val = phi_new, k_new
        B_phi, prod_old = B_phi_norm, prod_norm
        if converged:
            break

    return PeierlsSphereSolution(
        r_nodes=r_nodes,
        phi_values=phi[:, np.newaxis],
        k_eff=float(k_val),
        cell_radius=R,
        n_groups=1,
        n_quad_r=N,
        n_quad_theta=n_theta * n_rho,
        precision_digits=dps,
        panel_bounds=panels,
    )


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified", "one-group-kinf")
class TestPeierlsSphereSelfConvergence:
    """Peierls sphere self-convergence under quadrature refinement.

    If the operator is right, both the eigenvalue and the flux shape
    must be Cauchy-convergent as the quadrature orders go up."""

    def test_k_eff_cauchy_convergence_at_thick_R(self):
        R = 10.0
        keffs = []
        for n_q in (12, 18, 24):
            sol = _build_peierls_sphere_reference(
                R=R, n_panels=3, p_order=5,
                n_theta=n_q, n_rho=n_q, n_phi=n_q, dps=20,
            )
            keffs.append(sol.k_eff)
        d1 = abs(keffs[1] - keffs[0])
        d2 = abs(keffs[2] - keffs[1])
        assert d2 < d1 or d2 < 5e-5, (
            f"Sphere Peierls k_eff not converging: {keffs}"
        )

    def test_flux_cauchy_convergence_at_thick_R(self):
        R = 10.0
        probe = np.linspace(0.5, 0.9 * R, 10)
        phis = []
        for n_q in (12, 18, 24):
            sol = _build_peierls_sphere_reference(
                R=R, n_panels=3, p_order=5,
                n_theta=n_q, n_rho=n_q, n_phi=n_q, dps=20,
            )
            p = sol.phi(probe, g=0)
            p_norm = p / np.trapezoid(p, probe)
            phis.append(p_norm)
        d1 = np.max(np.abs(phis[1] - phis[0]))
        d2 = np.max(np.abs(phis[2] - phis[1]))
        assert d2 < d1 + 1e-6, (
            f"Sphere Peierls flux profile not converging: "
            f"d1={d1:.3e}, d2={d2:.3e}"
        )


@pytest.mark.l1
@pytest.mark.verifies("peierls-unified", "one-group-kinf")
class TestCPvsPeierlsSphereAtThickR:
    """CP-sphere vs Peierls-sphere flux shape agreement at thick R."""

    def test_k_eff_agrees_at_thick_R(self):
        """CP and Peierls white-BC k_eff agree to < 2 % at R = 10 MFP."""
        R = 10.0
        ref = _build_peierls_sphere_reference(
            R=R, n_panels=3, p_order=5,
            n_theta=20, n_rho=20, n_phi=20, dps=20,
        )

        n_cells = 20
        edges = np.linspace(0.0, R, n_cells + 1)
        materials = {2: get_mixture("A", "1g")}
        mat_ids = np.full(n_cells, 2)
        mesh = Mesh1D(
            edges=edges, mat_ids=mat_ids,
            coord=CoordSystem.SPHERICAL,
        )
        result = solve_cp(materials, mesh, CPParams(keff_tol=1e-8))

        rel_err = abs(result.keff - ref.k_eff) / ref.k_eff
        assert rel_err < 2e-2, (
            f"CP keff = {result.keff:.6f} vs Peierls k_eff = "
            f"{ref.k_eff:.6f} at R = {R}: rel err = {rel_err:.3e}"
        )

    def test_flux_shape_agrees_at_thick_R(self):
        """CP flux and Peierls flux (both white BC, R = 10) agree on
        normalised flux profile to L2 < 5 %."""
        R = 10.0
        ref = _build_peierls_sphere_reference(
            R=R, n_panels=3, p_order=5,
            n_theta=20, n_rho=20, n_phi=20, dps=20,
        )

        n_cells = 20
        edges = np.linspace(0.0, R, n_cells + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        materials = {2: get_mixture("A", "1g")}
        mat_ids = np.full(n_cells, 2)
        mesh = Mesh1D(
            edges=edges, mat_ids=mat_ids,
            coord=CoordSystem.SPHERICAL,
        )
        result = solve_cp(materials, mesh, CPParams(keff_tol=1e-8))

        phi_cp = result.flux[:, 0]
        phi_ref = ref.phi(centers, g=0)

        # Normalise both by sphere-volume-weighted integral (∫ 4π r² φ dr)
        weights = 4.0 * np.pi * centers * centers * (edges[1:] - edges[:-1])
        cp_norm = phi_cp / np.dot(phi_cp, weights)
        ref_norm = phi_ref / np.dot(phi_ref, weights)

        rel_l2 = np.sqrt(
            np.sum((cp_norm - ref_norm) ** 2 * weights)
            / np.sum(weights)
        )
        assert rel_l2 < 0.05, (
            f"CP vs Peierls sphere normalised flux L2 error = "
            f"{rel_l2:.3e} > 5 %"
        )
