"""Spherical SN sweep + α-redistribution + BiCGSTAB regression gates.

Split from the legacy ``tests/sn/test_spherical.py`` (SN taxonomy
reorg): the α-coefficient algebraic identities, the per-sweep
regression guards (T2b curvilinear closure), and the BiCGSTAB
inner-solver unit checks. The end-to-end k-eigenvalue claims moved to
``tests/sn/eigenvalue/test_keff_curvilinear.py``; the CP cross-check to
``tests/sn/verification/analytical/test_cp_standoff_curvilinear.py``.
"""

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import SNSolver, solve_sn
from tests.sn._test_helpers import (
    curvilinear_homogeneous_mesh as _homogeneous_mesh,
    placeholder_materials,
)
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux

# Equation-coverage list preserved verbatim from the legacy
# test_spherical module so no verifies(...) edge is lost in the split.
pytestmark = pytest.mark.verifies(
    "transport-spherical",
    "alpha-recursion",
    "wdd-closure",
    "wdd-face",
    "multigroup",
    "reflective-bc",
    "one-group-kinf",
    "matrix-eigenvalue",
    "mg-balance",
    "balance-general",
)


# ── Angular redistribution coefficient identities ────────────────────

@pytest.mark.l0
class TestAlphaCoefficients:
    """Properties of the angular redistribution coefficients."""

    @pytest.mark.parametrize("N", [4, 8, 16, 32])
    def test_alpha_boundary_conditions(self, N):
        """α_{1/2} = 0 and α_{N+1/2} = 0 by GL antisymmetry."""
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(N)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        reduced = sn_mesh.reduced
        assert reduced is not None  # 1-D mesh => minted by the ctor (narrowing)
        np.testing.assert_allclose(reduced.angular.alpha_per_level[0][0], 0.0)
        np.testing.assert_allclose(reduced.angular.alpha_per_level[0][-1], 0.0, atol=1e-14)

    def test_alpha_recursion(self):
        """α_{n+1/2} = α_{n-1/2} − w_n μ_n (Lathrop & Carlson 1966).

        ⛔ Cited "Bailey et al. 2009 Eq. 50" until 2026-08-27 — the
        wrong Bailey paper (a piecewise-linear FE *diffusion* paper,
        unrelated to curvilinear SN), retracted at #168 Phase B.
        """
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        reduced = sn_mesh.reduced
        assert reduced is not None  # 1-D mesh => minted by the ctor (narrowing)
        alpha = reduced.angular.alpha_per_level[0]
        for n in range(quad.N):
            expected = alpha[n] - quad.weights[n] * quad.mu_x[n]
            np.testing.assert_allclose(alpha[n + 1], expected, rtol=1e-14)

    def test_alpha_symmetric(self):
        """α coefficients are symmetric about the midpoint: α[k] = α[N-k].

        Follows from GL symmetry (w_n = w_{N-1-n}, μ_n = -μ_{N-1-n}).
        """
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        reduced = sn_mesh.reduced
        assert reduced is not None  # 1-D mesh => minted by the ctor (narrowing)
        alpha = reduced.angular.alpha_per_level[0]
        N = quad.N
        for k in range(N + 1):
            np.testing.assert_allclose(
                alpha[k], alpha[N - k], atol=1e-14,
                err_msg=f"α not symmetric at k={k}",
            )


# ── Sweep-level regression tests ─────────────────────────────────────

@pytest.mark.l0
class TestSphericalSweepRegression:
    """Tests targeting specific issues found during implementation."""

    def test_uniform_source_converges_to_Q_over_sigt(self):
        """Repeated sweeps with uniform Q and Σ_t must converge to φ = Q/Σ_t.

        This caught the missing weight_norm (1/sum_w) normalization in
        the spherical sweep source term.
        """
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once
        from orpheus.transport.source_sinks import AngularSourceSink

        mesh = _homogeneous_mesh(10, 1.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        sig_t = np.ones((1, *sn_mesh.spatial_shape))  # (ng, *spatial)
        Q_iso = np.ones((1, *sn_mesh.spatial_shape))  # (ng, *spatial)
        source = AngularSourceSink.from_isotropic(Q_iso, sn_mesh)

        boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        phi = None
        for _ in range(200):
            # Wave O (#208) O.4a.2 — bare sweep: drive the −B reflective
            # coupling explicitly before each sweep (the sweep no longer
            # re-applies the BC at entry).  Mirrors _solve_fixed_source_si.
            reflect_outflow_into_inflow(boundary_flux, sn_mesh)
            _, phi = sweep_once(source, sig_t, sn_mesh, boundary_flux)

        V = sn_mesh.volumes
        phi_avg = np.sum(phi[0, :] * V) / V.sum()
        np.testing.assert_allclose(phi_avg, 1.0, rtol=0.10,
                                   err_msg="Volume-avg φ ≠ Q/Σ_t for uniform source")

    def test_single_sweep_all_finite(self):
        """A single sweep must produce finite (non-NaN, non-Inf) fluxes.

        Catches the negative-denominator bug from using signed α
        instead of |α| at the innermost cell where A=0.
        """
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once
        from orpheus.transport.source_sinks import AngularSourceSink

        mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        sig_t = np.full((1, *sn_mesh.spatial_shape), 0.5)  # (ng, *spatial)
        Q_iso = np.ones((1, *sn_mesh.spatial_shape))       # (ng, *spatial)
        source = AngularSourceSink.from_isotropic(Q_iso, sn_mesh)

        boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        ang, phi = sweep_once(source, sig_t, sn_mesh, boundary_flux)

        assert np.all(np.isfinite(ang)), "Non-finite angular flux in first sweep"
        assert np.all(np.isfinite(phi)), "Non-finite scalar flux in first sweep"

    def test_inner_loop_bounded_multigroup(self):
        """Inner scattering iteration must stay bounded for multi-group."""
        mix = get_mixture("A", "2g")
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        fission = solver.compute_fission_source(phi, 1.0)
        phi_new = solver.solve_fixed_source(fission, phi)

        assert np.all(np.isfinite(phi_new)), "Non-finite flux after solve_fixed_source"
        assert phi_new.max() < 1e6, (
            f"Flux blew up to {phi_new.max():.2e} — inner loop may diverge"
        )

    def test_angular_flux_at_center_all_positive(self):
        """All ordinate angular fluxes at the centre must be positive.

        Tests that the angular redistribution correctly couples inward
        and outward ordinates at r≈0, where A=0 and spatial streaming
        vanishes.
        """
        mix = get_mixture("A", "1g")
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

        # D-H.1d: Solution.angular_flux is TimedFullField; bulk values.
        psi_center = result.angular_flux.interior.values[:, 0, 0]  # (N_ord,) for group 0

        assert np.all(psi_center > 0), (
            f"Zero or negative angular flux at centre: {psi_center}"
        )


# ── BiCGSTAB inner solver ────────────────────────────────────────────

@pytest.mark.l0
class TestSphericalBicgstab:
    """Tests for the BiCGSTAB inner solver on spherical geometry."""

    def test_bicgstab_1g_homogeneous_exact(self):
        """BiCGSTAB on 1G homogeneous sphere must match analytical k_inf."""
        case = get("sn_slab_1eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn({0: mix}, mesh, quad,
                          inner_solver="krylov",
                          max_inner=2000, inner_tol=1e-6)

        assert abs(result.keff - case.k_inf) < 1e-4, (
            f"BiCGSTAB keff={result.keff:.8f} vs analytical={case.k_inf:.8f}"
        )

    def test_bicgstab_matches_source_iteration(self):
        """BiCGSTAB and source iteration must agree on spherical geometry."""
        case = get("sn_slab_1eg_1rg")
        mix = next(iter(case.materials.values()))

        keffs = {}
        for label, solver_type in [("SI", "source_iteration"),
                                    ("BC", "krylov")]:
            mesh = _homogeneous_mesh(10, 2.0, mat_id=0,
                                  coord=CoordSystem.SPHERICAL)
            quad = Quadrature.gauss_legendre(8)
            result = solve_sn(
                {0: mix}, mesh, quad,
                inner_solver=solver_type,
                max_inner=500 if solver_type == "SI" else 2000,
                inner_tol=1e-10 if solver_type == "SI" else 1e-6,
            )
            keffs[label] = result.keff

        assert abs(keffs["SI"] - keffs["BC"]) < 1e-4, (
            f"SI keff={keffs['SI']:.8f} vs BC keff={keffs['BC']:.8f}"
        )

    @pytest.mark.catches("ERR-007")
    def test_bicgstab_finite_result(self):
        """BiCGSTAB on 1G spherical must produce finite flux and keff.

        Note: multi-group BiCGSTAB on spherical geometry is unstable
        (the explicit FD operator for angular redistribution is less
        accurate than the DD sweep's implicit treatment). Only 1G is
        expected to converge reliably.
        """
        case = get("sn_slab_1eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn({0: mix}, mesh, quad,
                          inner_solver="krylov",
                          max_inner=2000, inner_tol=1e-6)

        assert np.isfinite(result.keff), f"keff is not finite: {result.keff}"
        assert np.all(np.isfinite(result.scalar_flux.values)), "Non-finite scalar flux"
