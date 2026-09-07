"""Cylindrical SN sweep + azimuthal-redistribution regression gates.

Split from the legacy ``tests/sn/test_cylindrical.py`` (SN taxonomy
reorg): the per-sweep regression guards (T2b curvilinear closure) and
the azimuthal α-coefficient / telescoping / equilibrium tests that
exercise the sweep's redistribution path. The end-to-end k-eigenvalue
claims moved to ``tests/sn/eigenvalue/test_keff_curvilinear.py``; the CP
cross-checks moved to
``tests/sn/verification/analytical/test_cp_standoff_curvilinear.py``.
"""

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import SNSolver, solve_sn
from tests.sn._test_helpers import (
    curvilinear_homogeneous_mesh as _homogeneous_mesh,
    curvilinear_two_region_mesh as _two_region_mesh,
    placeholder_materials,
)
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux

# Equation-coverage list preserved verbatim from the legacy
# test_cylindrical module so no verifies(...) edge is lost in the split.
pytestmark = pytest.mark.verifies(
    "transport-cylindrical",
    "alpha-cylindrical",
    "alpha-recursion",
    "wdd-closure",
    "wdd-face",
    "mm-weights",
    "multigroup",
    "reflective-bc",
    "one-group-kinf",
    "matrix-eigenvalue",
    "mg-balance",
    "balance-general",
)


@pytest.mark.l0
class TestCylindricalSweepRegression:
    """Tests targeting issues found in spherical implementation."""

    def test_single_sweep_all_finite(self):
        """A single sweep must produce finite fluxes."""
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once
        from orpheus.transport.source_sinks import AngularSourceSink

        mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        sig_t = np.full((1, *sn_mesh.spatial_shape), 0.5)  # (ng, *spatial)
        Q_iso = np.ones((1, *sn_mesh.spatial_shape))       # (ng, *spatial)
        source = AngularSourceSink.from_isotropic(Q_iso, sn_mesh)

        boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        ang, phi = sweep_once(source, sig_t, sn_mesh, boundary_flux)

        assert np.all(np.isfinite(ang)), "Non-finite angular flux"
        assert np.all(np.isfinite(phi)), "Non-finite scalar flux"

    def test_inner_loop_bounded_multigroup(self):
        """Inner loop must stay bounded for multi-group."""
        mix = get_mixture("A", "2g")
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        fission = solver.compute_fission_source(phi, 1.0)
        phi_new = solver.solve_fixed_source(fission, phi)

        assert np.all(np.isfinite(phi_new)), "Non-finite flux after solve_fixed_source"
        assert phi_new.max() < 1e6, (
            f"Flux blew up to {phi_new.max():.2e}"
        )

    # ``test_both_quadratures_agree`` (product(4,8) vs level_symmetric(4)
    # keff at rtol=1e-6) RETIRED at Q5.6.3: (a) its subject — cross-FAMILY
    # agreement on a cylinder — is unspellable once SNMesh(CYLINDRICAL)
    # admits only the carrying folded_product family; (b) `[M]` 2026-08-08
    # its fixture was Mode-12-degenerate all along: the homogeneous
    # REFLECTIVE cylinder's keff is the flat k_inf = 1.5 exactly
    # (quadrature-blind — product(4,8), LS4, folded(4,8), folded(8,4) and
    # folded(4,16) all measured 1.500000000000), so the 1e-6 agreement
    # never constrained either family's angular wiring.  The live
    # cross-checks of the folded cylinder's angular fidelity are the
    # trajectory_resolvent L1 gate (test_unified_matvec_cylinder) and the
    # MMS σ_y-parity + azimuthal-floor gates (tests/sn/verification/mms).

    def test_requires_level_quadrature(self):
        """Cylindrical SNMesh with GL quadrature must raise ValueError."""
        mesh = _homogeneous_mesh(5, 1.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.gauss_legendre(4)
        with pytest.raises(ValueError, match="level structure"):
            SNMesh(mesh, quad, placeholder_materials())


@pytest.mark.l2
class TestAzimuthalRedistribution:
    """Azimuthal α-coefficient / telescoping / equilibrium sweep gates.

    Split from the legacy ``TestMultiGroupMultiRegion`` — these tests
    exercise the cylindrical sweep's angular-redistribution path
    (α boundary conditions, the αψ telescoping conservation, and the
    fixed-source φ = Q/Σ_t equilibrium), NOT the eigenvalue iteration.
    """

    def test_azimuthal_alpha_boundary_conditions(self):
        """Per-level α coefficients must satisfy α[0] = 0, α[-1] ≈ 0.

        This is the azimuthal analogue of the spherical α boundary check.
        Failure means the ξ-weighted sum doesn't vanish, which would
        cause non-physical angular flux generation.
        """
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        reduced = sn_mesh.reduced
        assert reduced is not None  # 1-D mesh => minted by the ctor (narrowing)
        for p, alpha in enumerate(reduced.angular.alpha_per_level):
            np.testing.assert_allclose(alpha[0], 0.0,
                                       err_msg=f"Level {p}: α[0] ≠ 0")
            np.testing.assert_allclose(alpha[-1], 0.0, atol=1e-13,
                                       err_msg=f"Level {p}: α[-1] ≠ 0")

    def test_angular_flux_at_center_all_positive(self):
        """All ordinate angular fluxes at r≈0 must be positive.

        Tests that azimuthal redistribution correctly couples all
        directions at the centre where A=0.
        """
        mix = get_mixture("A", "1g")
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

        # D-H.1d: Solution.angular_flux is TimedFullField; .interior.values.
        psi_center = result.angular_flux.interior.values[:, 0, 0]
        assert np.all(psi_center > 0), (
            f"Zero/negative angular flux at centre: min={psi_center.min():.4e}"
        )

    def test_redistribution_telescoping_conservation(self):
        """αψ product telescopes to zero on each level per cell.

        The redistribution sum Σ_m (α_{m+1/2}ψ_{m+1/2} − α_{m-1/2}ψ_{m-1/2})
        must vanish for each cell because α[0] = α[M] = 0.
        """
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once
        from orpheus.transport.source_sinks import AngularSourceSink

        mix = get_mixture("A", "1g")
        mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        sig_t = np.full((1, *sn_mesh.spatial_shape), mix.SigT[0])  # (ng, *spatial)
        Q_iso = np.ones((1, *sn_mesh.spatial_shape))               # (ng, *spatial)
        source = AngularSourceSink.from_isotropic(Q_iso, sn_mesh)
        boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        ang, _ = sweep_once(source, sig_t, sn_mesh, boundary_flux)

        reduced = sn_mesh.reduced
        assert reduced is not None  # 1-D mesh => minted by the ctor (narrowing)
        for p, level_idx in enumerate(quad.level_indices):
            alpha = reduced.angular.alpha_per_level[p]
            M = len(level_idx)
            psi_angle = np.zeros(10)
            for m_local in range(M):
                n = level_idx[m_local]
                psi_cell = ang[n, 0, :]
                psi_angle_new = 2.0 * psi_cell - psi_angle
                psi_angle = psi_angle_new
            residual = alpha[M] * psi_angle
            np.testing.assert_allclose(residual, 0.0, atol=1e-12,
                                       err_msg=f"Level {p}: telescoping residual ≠ 0")

    def test_single_cell_uniform_source_equilibrium(self):
        """Two-cell 1G pure absorber with uniform source → φ = Q/Σ_t."""
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once
        from orpheus.transport.source_sinks import AngularSourceSink

        mesh = _homogeneous_mesh(2, 1.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        sig_t = np.ones((1, *sn_mesh.spatial_shape))  # (ng, *spatial)
        Q_iso = np.ones((1, *sn_mesh.spatial_shape))  # (ng, *spatial)
        source = AngularSourceSink.from_isotropic(Q_iso, sn_mesh)
        boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        phi = None
        for _ in range(100):
            # Wave O (#208) O.4a.2 — bare sweep: drive the −B reflective
            # coupling explicitly before each sweep (mirrors the production
            # _solve_fixed_source_si direct loop).
            reflect_outflow_into_inflow(boundary_flux, sn_mesh)
            _, phi = sweep_once(source, sig_t, sn_mesh, boundary_flux)

        phi_avg = np.average(phi[0, :], weights=mesh.volumes)
        np.testing.assert_allclose(phi_avg, 1.0, rtol=0.01,
                                   err_msg="Volume-avg φ ≠ Q/Σ_t for uniform source")
