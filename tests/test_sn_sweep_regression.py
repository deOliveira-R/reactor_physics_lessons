"""Regression tests for the SN sweep recurrence and SNMesh stencil.

These tests cover bugs and edge cases found during the geometry
migration (2026-04-04).  Each test targets a specific failure mode.

Gotcha #5: _solve_recurrence formula rewrite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An algebraically-equivalent rewrite of the DD recurrence
(``0.5*(psi_in + psi_out)`` → ``2*(psi_in*(1-a)/(1-a+eps) + s) - psi_in``)
produced different numerical results because the cell-average formula
must be ``0.5*(psi_in + psi_out)`` where ``psi_out = a*psi_in + s``.
The rewritten form lost the ``psi0[None, :]`` broadcasting in the
cumulative product and was numerically unstable near ``a → 1``.

Impact: scattering source iteration diverged for multi-group problems
(flux grew by ~1e34 per outer iteration), while 1-group was unaffected.
"""

import numpy as np
import pytest

from orpheus.geometry import Mesh1D, Mesh2D, homogeneous_1d
from orpheus.sn.geometry import SNMesh
from orpheus.sn.quadrature import GaussLegendre1D, LebedevSphere
from orpheus.sn.sweep import _solve_recurrence, _outgoing, transport_sweep


# ═══════════════════════════════════════════════════════════════════════
# _solve_recurrence unit tests
# ═══════════════════════════════════════════════════════════════════════

class TestSolveRecurrence:
    """Unit tests for the DD recurrence solver."""

    def test_single_cell_zero_bc(self):
        """Single cell: psi_avg = 0.5*(0 + a*0 + s) = 0.5*s."""
        a = np.array([[0.8, 0.6]])
        s = np.array([[0.1, 0.2]])
        psi0 = np.zeros(2)
        result = _solve_recurrence(a, s, psi0)
        # psi_in = 0, psi_out = a*0 + s = s
        expected = 0.5 * (0 + s)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_single_cell_nonzero_bc(self):
        """Single cell with nonzero boundary: psi_out = a*psi0 + s."""
        a = np.array([[0.5, 0.7]])
        s = np.array([[0.2, 0.3]])
        psi0 = np.array([1.0, 2.0])
        result = _solve_recurrence(a, s, psi0)
        psi_out = a[0] * psi0 + s[0]
        expected = 0.5 * (psi0[None, :] + psi_out[None, :])
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_multi_cell_dd_relation(self):
        """Cell-average = 0.5*(psi_in + psi_out) for every cell."""
        a = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
        s = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]])
        psi0 = np.array([0.5, 1.0])
        psi_avg = _solve_recurrence(a, s, psi0)

        # Reconstruct psi_in and psi_out cell by cell
        psi_in = psi0.copy()
        for i in range(3):
            psi_out = a[i] * psi_in + s[i]
            expected_avg = 0.5 * (psi_in + psi_out)
            np.testing.assert_allclose(
                psi_avg[i], expected_avg, rtol=1e-13,
                err_msg=f"Cell {i} DD relation violated",
            )
            psi_in = psi_out

    def test_outgoing_matches_last_cell(self):
        """_outgoing must return the outgoing flux of the last cell."""
        a = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
        s = np.array([[0.1, 0.2], [0.15, 0.25], [0.2, 0.3]])
        psi0 = np.array([0.5, 1.0])

        # Walk the recurrence manually
        psi_in = psi0.copy()
        for i in range(3):
            psi_out = a[i] * psi_in + s[i]
            psi_in = psi_out
        expected_outgoing = psi_out

        actual = _outgoing(a, s, psi0)
        np.testing.assert_allclose(actual, expected_outgoing, rtol=1e-13)

    def test_regression_multigroup_scattering_convergence(self):
        """Scattering source iteration must converge for multi-group.

        Gotcha #5: an algebraically-equivalent rewrite of _solve_recurrence
        caused the inner loop to diverge for 2+ groups while 1-group worked.
        """
        from orpheus.derivations._xs_library import get_mixture
        from orpheus.sn.solver import SNSolver

        mix = get_mixture("A", "2g")
        mesh = homogeneous_1d(20, 2.0, mat_id=0)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver({0: mix}, sn_mesh, max_inner=500, inner_tol=1e-10)

        # One outer iteration: flux must remain bounded
        phi = solver.initial_flux_distribution()
        fission = solver.compute_fission_source(phi, 1.0)
        phi_new = solver.solve_fixed_source(fission, phi)

        assert np.all(np.isfinite(phi_new)), "Non-finite flux after one outer iteration"
        assert phi_new.max() < 100, (
            f"Flux blew up to {phi_new.max():.2e} — "
            f"_solve_recurrence may have changed"
        )


# ═══════════════════════════════════════════════════════════════════════
# SNMesh stencil and shape tests
# ═══════════════════════════════════════════════════════════════════════

class TestSNMesh:
    """Tests for the SNMesh augmented geometry."""

    def test_stencil_values_cartesian(self):
        """streaming_x[n,i] must equal 2|μ_x[n]| / dx[i]."""
        mesh = Mesh1D(edges=np.array([0.0, 0.1, 0.3, 0.6]),
                      mat_ids=np.array([0, 1, 2]))
        quad = GaussLegendre1D.create(4)
        sn_mesh = SNMesh(mesh, quad)

        for n in range(quad.N):
            for i in range(sn_mesh.nx):
                expected = 2.0 * abs(quad.mu_x[n]) / mesh.widths[i]
                np.testing.assert_allclose(
                    sn_mesh.streaming_x[n, i], expected, rtol=1e-14,
                )

    def test_stencil_dd_denom_equivalence(self):
        """Precomputed stencil must reproduce the original DD denominator.

        Original: denom = Σ_t + 2|μ_x|/dx + 2|μ_y|/dy
        Stencil:  denom = Σ_t + streaming_x[n,i] + streaming_y[n,j]
        """
        mesh = Mesh2D(
            edges_x=np.linspace(0, 1, 4),  # 3 cells, dx=1/3
            edges_y=np.linspace(0, 0.5, 3),  # 2 cells, dy=0.25
            mat_map=np.zeros((3, 2), dtype=int),
        )
        quad = LebedevSphere.create(order=17)
        sn_mesh = SNMesh(mesh, quad)

        sig_t = 0.5  # scalar for simplicity
        for n in range(quad.N):
            for i in range(sn_mesh.nx):
                for j in range(sn_mesh.ny):
                    old = sig_t + 2*abs(quad.mu_x[n])/mesh.dx[i] + 2*abs(quad.mu_y[n])/mesh.dy[j]
                    new = sig_t + sn_mesh.streaming_x[n, i] + sn_mesh.streaming_y[n, j]
                    np.testing.assert_allclose(new, old, rtol=1e-14)

    def test_mesh1d_shapes(self):
        """SNMesh from Mesh1D must have (N,1) shaped mat_map and volumes."""
        mesh = Mesh1D(edges=np.linspace(0, 1, 6), mat_ids=np.array([0,1,2,1,0]))
        quad = GaussLegendre1D.create(4)
        sn_mesh = SNMesh(mesh, quad)

        assert sn_mesh.nx == 5
        assert sn_mesh.ny == 1
        assert sn_mesh.mat_map.shape == (5, 1)
        assert sn_mesh.volumes.shape == (5, 1)
        assert sn_mesh.is_1d is True

    def test_mesh2d_shapes(self):
        """SNMesh from Mesh2D preserves shapes."""
        mesh = Mesh2D(
            edges_x=np.linspace(0, 1, 4),
            edges_y=np.linspace(0, 1, 3),
            mat_map=np.zeros((3, 2), dtype=int),
        )
        quad = LebedevSphere.create(order=17)
        sn_mesh = SNMesh(mesh, quad)

        assert sn_mesh.nx == 3
        assert sn_mesh.ny == 2
        assert sn_mesh.mat_map.shape == (3, 2)
        assert sn_mesh.volumes.shape == (3, 2)
        assert sn_mesh.is_1d is False

    def test_cylindrical_requires_level_quadrature(self):
        """Cylindrical coords require a quadrature with level structure."""
        from orpheus.geometry import CoordSystem

        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.CYLINDRICAL)
        quad = GaussLegendre1D.create(4)
        with pytest.raises(ValueError, match="level structure"):
            SNMesh(mesh, quad)

    def test_spherical_setup(self):
        """Spherical SNMesh must precompute face areas and α coefficients."""
        from orpheus.geometry import CoordSystem

        mesh = Mesh1D(edges=np.array([0.0, 0.5, 1.0]), mat_ids=np.array([0, 1]),
                      coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(4)
        sn_mesh = SNMesh(mesh, quad)

        assert sn_mesh.curvature == "spherical"
        assert sn_mesh.face_areas is not None
        assert sn_mesh.alpha_half is not None
        assert len(sn_mesh.alpha_half) == quad.N + 1
        # α_{1/2} = 0 and α_{N+1/2} ≈ 0
        np.testing.assert_allclose(sn_mesh.alpha_half[0], 0.0)
        np.testing.assert_allclose(sn_mesh.alpha_half[-1], 0.0, atol=1e-14)

    def test_sweep_1d_2d_consistency(self):
        """1D and 2D sweeps on equivalent meshes must produce similar keff.

        A Mesh1D slab and a Mesh2D with ny=1 must give comparable results
        when using the same quadrature (Lebedev can handle ny=1).
        """
        from orpheus.derivations._xs_library import get_mixture
        from orpheus.sn.solver import SNSolver

        mix = get_mixture("A", "1g")

        # 1D with GL
        mesh_1d = homogeneous_1d(10, 1.0, mat_id=0)
        quad_gl = GaussLegendre1D.create(8)
        solver_1d = SNSolver({0: mix}, SNMesh(mesh_1d, quad_gl),
                             max_inner=500, inner_tol=1e-10)
        phi = solver_1d.initial_flux_distribution()
        keff_1d = 1.0
        for _ in range(50):
            fs = solver_1d.compute_fission_source(phi, keff_1d)
            phi = solver_1d.solve_fixed_source(fs, phi)
            keff_1d = solver_1d.compute_keff(phi)

        # 2D with Lebedev (ny=1)
        mesh_2d = Mesh2D(
            edges_x=np.linspace(0, 1, 11),
            edges_y=np.array([0.0, 1.0]),
            mat_map=np.zeros((10, 1), dtype=int),
        )
        quad_leb = LebedevSphere.create(order=17)
        solver_2d = SNSolver({0: mix}, SNMesh(mesh_2d, quad_leb),
                             max_inner=500, inner_tol=1e-10)
        phi = solver_2d.initial_flux_distribution()
        keff_2d = 1.0
        for _ in range(50):
            fs = solver_2d.compute_fission_source(phi, keff_2d)
            phi = solver_2d.solve_fixed_source(fs, phi)
            keff_2d = solver_2d.compute_keff(phi)

        # Both must match the analytical k_inf (homogeneous, 1G)
        from orpheus.derivations import get
        k_ref = get("sn_slab_1eg_1rg").k_inf
        assert abs(keff_1d - k_ref) < 1e-8
        assert abs(keff_2d - k_ref) < 1e-8
