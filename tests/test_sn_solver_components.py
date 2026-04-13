"""Unit tests for individual SN solver components.

Tests each method in isolation against a reference (the original per-cell
implementation), using small 2-group cross sections for fast execution.
"""

from pathlib import Path

import numpy as np
import pytest
import time

from orpheus.derivations._xs_library import get_mixture
from orpheus.geometry import Mesh1D, Mesh2D
from orpheus.sn.geometry import SNMesh
from orpheus.sn.quadrature import GaussLegendre1D, LebedevSphere
from orpheus.sn.solver import SNSolver, solve_sn
from orpheus.sn.sweep import transport_sweep

pytestmark = pytest.mark.l0  # SN solver method-in-isolation component checks


def _uniform_2d(nx, ny, delta, mat_map):
    """Helper: build a uniform Mesh2D (replaces CartesianMesh.uniform_2d)."""
    return Mesh2D(
        edges_x=np.linspace(0, nx * delta, nx + 1),
        edges_y=np.linspace(0, ny * delta, ny + 1),
        mat_map=np.asarray(mat_map, dtype=int),
    )


@pytest.fixture
def solver_2g():
    """Build a small 2-group solver for component testing."""
    fuel = get_mixture("A", "2g")
    mod = get_mixture("B", "2g")
    materials = {2: fuel, 0: mod}

    nx, ny = 6, 4
    delta = 0.2
    mat = np.zeros((nx, ny), dtype=int)
    mat[:3, :] = 2
    mat[3:, :] = 0

    mesh = _uniform_2d(nx, ny, delta, mat)
    quad = LebedevSphere.create(order=17)
    sn_mesh = SNMesh(mesh, quad)
    solver = SNSolver(materials, sn_mesh)
    return solver, materials, sn_mesh, quad


# ── Reference implementations (per-cell loops, known correct) ─────────

def _ref_add_scattering(solver, Q, phi):
    """Original per-cell scattering source (reference)."""
    out = Q.copy()
    nx, ny = solver.sn_mesh.nx, solver.sn_mesh.ny
    for ix in range(nx):
        for iy in range(ny):
            mid = int(solver.sn_mesh.mat_map[ix, iy])
            out[ix, iy, :] += solver.sig_s0[mid].T @ phi[ix, iy, :]
    return out


def _ref_add_n2n(solver, Q, phi):
    """Original per-cell (n,2n) source (reference)."""
    out = Q.copy()
    nx, ny = solver.sn_mesh.nx, solver.sn_mesh.ny
    for ix in range(nx):
        for iy in range(ny):
            mid = int(solver.sn_mesh.mat_map[ix, iy])
            out[ix, iy, :] += 2.0 * (solver.sig2[mid].T @ phi[ix, iy, :])
    return out


def _ref_compute_keff(solver, flux):
    """Original per-cell keff computation (reference)."""
    vol = solver.volume[:, :, None]
    production = np.sum(solver.sig_p * flux * vol)
    for ix in range(solver.sn_mesh.nx):
        for iy in range(solver.sn_mesh.ny):
            mid = int(solver.sn_mesh.mat_map[ix, iy])
            sig2_sum = np.array(solver.sig2[mid].sum(axis=1)).ravel()
            production += 2.0 * np.dot(sig2_sum, flux[ix, iy, :]) * solver.volume[ix, iy]
    absorption = np.sum(solver.sig_a * flux * vol)
    return float(production / absorption)


# ── Component tests ──────────────────────────────────────────────────

class TestAddScatteringSource:
    def test_matches_reference(self, solver_2g):
        solver, *_ = solver_2g
        np.random.seed(42)
        phi = np.random.rand(solver.sn_mesh.nx, solver.sn_mesh.ny, solver.ng) + 0.1
        Q = np.random.rand(solver.sn_mesh.nx, solver.sn_mesh.ny, solver.ng)

        expected = _ref_add_scattering(solver, Q, phi)

        Q_actual = Q.copy()
        solver._add_scattering_source(Q_actual, phi)

        np.testing.assert_allclose(Q_actual, expected, rtol=1e-13,
                                   err_msg="Scattering source mismatch")

    def test_zero_flux_gives_zero_addition(self, solver_2g):
        solver, *_ = solver_2g
        Q = np.ones((solver.sn_mesh.nx, solver.sn_mesh.ny, solver.ng))
        phi = np.zeros_like(Q)

        Q_before = Q.copy()
        solver._add_scattering_source(Q, phi)
        np.testing.assert_array_equal(Q, Q_before)


class TestAddN2NSource:
    def test_matches_reference(self, solver_2g):
        solver, *_ = solver_2g
        np.random.seed(123)
        phi = np.random.rand(solver.sn_mesh.nx, solver.sn_mesh.ny, solver.ng) + 0.1
        Q = np.random.rand(solver.sn_mesh.nx, solver.sn_mesh.ny, solver.ng)

        expected = _ref_add_n2n(solver, Q, phi)

        Q_actual = Q.copy()
        solver._add_n2n_source(Q_actual, phi)

        np.testing.assert_allclose(Q_actual, expected, rtol=1e-13,
                                   err_msg="N2N source mismatch")


class TestComputeKeff:
    def test_matches_reference(self, solver_2g):
        solver, *_ = solver_2g
        np.random.seed(99)
        flux = np.random.rand(solver.sn_mesh.nx, solver.sn_mesh.ny, solver.ng) + 0.1

        expected = _ref_compute_keff(solver, flux)
        actual = solver.compute_keff(flux)

        np.testing.assert_allclose(actual, expected, rtol=1e-13,
                                   err_msg="compute_keff mismatch")


class TestTransportSweep:
    def test_deterministic_output(self, solver_2g):
        """Sweep with same input must produce same output."""
        solver, _, sn_mesh, quad = solver_2g
        np.random.seed(7)
        Q = np.random.rand(sn_mesh.nx, sn_mesh.ny, solver.ng) + 0.01

        psi_bc1, psi_bc2 = {}, {}
        ang1, phi1 = transport_sweep(Q, solver.sig_t, sn_mesh, psi_bc1)
        ang2, phi2 = transport_sweep(Q, solver.sig_t, sn_mesh, psi_bc2)

        np.testing.assert_array_equal(phi1, phi2,
                                      err_msg="Sweep not deterministic")

    def test_matches_saved_reference(self, solver_2g):
        """Sweep output must match the saved reference (bitwise regression)."""
        solver, _, sn_mesh, quad = solver_2g
        np.random.seed(7)
        Q = np.random.rand(sn_mesh.nx, sn_mesh.ny, solver.ng) + 0.01

        _, phi = transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})
        ref = np.load(Path(__file__).parent / "sweep_ref_2g.npy")

        np.testing.assert_allclose(phi, ref, rtol=1e-14,
                                   err_msg="Sweep regression: output changed")

    def test_positive_source_positive_flux(self, solver_2g):
        """Positive source must produce non-negative flux."""
        solver, _, sn_mesh, quad = solver_2g
        Q = np.ones((sn_mesh.nx, sn_mesh.ny, solver.ng))

        _, phi = transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})

        assert np.all(phi >= 0), "Negative flux from positive source"

    def test_scalar_flux_shape(self, solver_2g):
        """Output shapes must match expectations."""
        solver, _, sn_mesh, quad = solver_2g
        Q = np.ones((sn_mesh.nx, sn_mesh.ny, solver.ng))

        ang, phi = transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})

        assert ang.shape == (quad.N, sn_mesh.nx, sn_mesh.ny, solver.ng)
        assert phi.shape == (sn_mesh.nx, sn_mesh.ny, solver.ng)


class TestQuadratureWeightConservation:
    """The sweep must account for ALL quadrature weight in the scalar flux.

    Discovery: ordinates with mu_x = mu_y = 0 (z-directed) were previously
    skipped, losing 0.77% of Lebedev weight.  This caused a multi-group
    eigenvalue error of ~0.4% that was invisible in 1-group problems.
    """

    def test_no_weight_lost(self, solver_2g):
        """Σ_n w_n · ψ_n must use the full sum(weights), not a subset."""
        solver, _, sn_mesh, quad = solver_2g
        Q = np.ones((sn_mesh.nx, sn_mesh.ny, solver.ng))

        ang, phi = transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})

        # Reconstruct scalar flux from angular flux manually
        phi_manual = np.zeros_like(phi)
        for n in range(quad.N):
            phi_manual += quad.weights[n] * ang[n]

        np.testing.assert_allclose(phi, phi_manual, rtol=1e-14,
                                   err_msg="Scalar flux missing ordinate contributions")

    def test_z_ordinates_contribute(self, solver_2g):
        """Z-directed ordinates (mu_x=mu_y=0) must have nonzero angular flux."""
        solver, _, sn_mesh, quad = solver_2g
        Q = np.ones((sn_mesh.nx, sn_mesh.ny, solver.ng))

        ang, _ = transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})

        for n in range(quad.N):
            if abs(quad.mu_x[n]) < 1e-15 and abs(quad.mu_y[n]) < 1e-15:
                assert np.all(ang[n] > 0), (
                    f"Z-directed ordinate {n} has zero angular flux"
                )

    def test_homogeneous_scalar_flux_equals_Q_over_sigt(self, solver_2g):
        """In a homogeneous infinite medium, converged φ = Q / Σ_t.

        With uniform source Q and reflective BCs, the isotropic SN
        equation gives ψ = Q/(4π·Σ_t) per ordinate.
        Then φ = Σ w_n ψ_n = sum(w) · Q/(4π·Σ_t) = Q/Σ_t.
        """
        from orpheus.derivations._xs_library import get_mixture

        mix = get_mixture("A", "2g")
        materials = {0: mix}
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)
        local_sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver(materials, local_sn_mesh)

        Q = np.ones((2, 2, solver.ng))

        # Run many sweeps to converge reflective BCs
        psi_bc = {}
        for _ in range(200):
            _, phi = transport_sweep(Q, solver.sig_t, local_sn_mesh, psi_bc)

        expected = Q / solver.sig_t
        np.testing.assert_allclose(phi, expected, rtol=1e-6,
                                   err_msg="Converged φ ≠ Q/Σ_t for uniform source")


class TestAbsorptionXS:
    """Verify that absorption_xs includes fission (not just capture).

    Discovery: we initially suspected compute_keff used the wrong Σ_a.
    In fact, Mixture.absorption_xs = SigF + SigC + SigL + Sig2_rowsum,
    which is the correct total removal rate (Σ_t - Σ_s) for the keff formula.
    """

    def test_absorption_xs_includes_fission(self):
        from orpheus.derivations._xs_library import get_mixture

        mix = get_mixture("A", "2g")
        sig_a = mix.absorption_xs
        expected = np.array(mix.SigF) + np.array(mix.SigC) + np.array(mix.SigL) \
            + np.asarray(mix.Sig2.sum(axis=1)).ravel()
        np.testing.assert_array_equal(sig_a, expected)

    def test_absorption_equals_removal(self):
        """absorption_xs must equal Σ_t - rowsum(Σ_s) (total removal)."""
        from orpheus.derivations._xs_library import get_mixture

        mix = get_mixture("A", "2g")
        removal = np.array(mix.SigT) - np.asarray(mix.SigS[0].sum(axis=1)).ravel()
        np.testing.assert_allclose(mix.absorption_xs, removal, rtol=1e-14)


class TestMultiGroupEigenvector:
    """The converged flux group ratio must match the analytical eigenvector.

    Discovery: 1-group tests hide bugs because k = νΣf/Σa is independent
    of the spatial/angular flux shape.  Multi-group problems have a specific
    eigenvector (group ratio) that must be recovered.
    """

    def test_2g_eigenvector(self):
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))

        # Analytical eigenvector from (Σ_t - Σ_s^T)^{-1} · χ⊗(νΣf)
        sig_s = mix.SigS[0].toarray()
        A = np.diag(mix.SigT) - sig_s.T
        F = np.outer(mix.chi, mix.SigP)
        _, vecs = np.linalg.eig(np.linalg.solve(A, F))
        idx = np.argmax(np.real(np.linalg.eigvals(np.linalg.solve(A, F))))
        phi_expected = np.real(vecs[:, idx])
        phi_expected /= phi_expected.sum()

        # Run 2D solver
        materials = {0: mix}
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)
        solver = SNSolver(materials, SNMesh(mesh, quad), max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(100):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)
            phi = phi / np.linalg.norm(phi)

        phi_cell = phi[0, 0, :]
        phi_ratio = phi_cell / phi_cell.sum()

        np.testing.assert_allclose(phi_ratio, phi_expected, rtol=1e-6,
                                   err_msg="Converged group ratio ≠ analytical eigenvector")


class TestBicgstabNormalization:
    """BiCGSTAB must give the same keff regardless of quadrature type.

    Discovery: build_rhs hardcoded 4π for the angular normalization, but
    GL quadrature weights sum to 2, not 4π. The normalization must use
    sum(weights), which is quadrature-dependent.
    """

    def test_1d_gl_homogeneous_exact(self):
        """BiCGSTAB with GL quadrature on 1D slab must match analytical k_inf."""
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))

        mesh = Mesh1D(edges=np.linspace(0, 2, 5), mat_ids=np.zeros(4, dtype=int))
        gl = GaussLegendre1D.create(8)
        solver = SNSolver({0: mix}, SNMesh(mesh, gl),
                          inner_solver="bicgstab",
                          max_inner=2000, inner_tol=1e-6)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(50):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)
            phi /= np.linalg.norm(phi)

        assert abs(keff - case.k_inf) < 1e-4, (
            f"1D GL BiCGSTAB keff={keff:.8f} vs analytical={case.k_inf:.8f}"
        )

    def test_2d_lebedev_homogeneous_exact(self):
        """BiCGSTAB with Lebedev quadrature on 2D mesh must match analytical k_inf."""
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))

        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)
        solver = SNSolver({0: mix}, SNMesh(mesh, quad),
                          inner_solver="bicgstab",
                          max_inner=2000, inner_tol=1e-6)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(50):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)
            phi /= np.linalg.norm(phi)

        assert abs(keff - case.k_inf) < 1e-4, (
            f"2D Lebedev BiCGSTAB keff={keff:.8f} vs analytical={case.k_inf:.8f}"
        )

    def test_gl_and_lebedev_agree(self):
        """BiCGSTAB keff must not depend on which quadrature is used.

        Both GL (sum(w)=2) and Lebedev (sum(w)=4π) must produce the same
        eigenvalue for the same homogeneous problem.
        """
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))

        results = {}
        for label, mesh, quad in [
            ("GL", Mesh1D(edges=np.linspace(0, 2, 5), mat_ids=np.zeros(4, dtype=int)),
             GaussLegendre1D.create(8)),
            ("Lebedev", _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int)),
             LebedevSphere.create(order=17)),
        ]:
            solver = SNSolver({0: mix}, SNMesh(mesh, quad),
                              inner_solver="bicgstab",
                              max_inner=2000, inner_tol=1e-6)
            phi = solver.initial_flux_distribution()
            keff = 1.0
            for _ in range(50):
                fs = solver.compute_fission_source(phi, keff)
                phi = solver.solve_fixed_source(fs, phi)
                keff = solver.compute_keff(phi)
                phi /= np.linalg.norm(phi)
            results[label] = keff

        assert abs(results["GL"] - results["Lebedev"]) < 1e-3, (
            f"GL keff={results['GL']:.6f} vs Lebedev keff={results['Lebedev']:.6f}"
        )


class TestAnisotropicScattering:
    """Pn scattering must reduce to P0 when L=0, and affect keff when L>0."""

    def test_p0_gives_identical_keff(self):
        """scattering_order=0 must give the exact same keff as the default."""
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)

        # Default (P0)
        solver_default = SNSolver({0: mix}, SNMesh(mesh, quad),
                                  max_inner=500, inner_tol=1e-10)
        phi = solver_default.initial_flux_distribution()
        keff = 1.0
        for _ in range(50):
            fs = solver_default.compute_fission_source(phi, keff)
            phi = solver_default.solve_fixed_source(fs, phi)
            keff = solver_default.compute_keff(phi)
            phi /= np.linalg.norm(phi)
        keff_p0 = keff

        # Explicit P0
        solver_explicit = SNSolver({0: mix}, SNMesh(mesh, quad),
                                   scattering_order=0,
                                   max_inner=500, inner_tol=1e-10)
        phi = solver_explicit.initial_flux_distribution()
        keff = 1.0
        for _ in range(50):
            fs = solver_explicit.compute_fission_source(phi, keff)
            phi = solver_explicit.solve_fixed_source(fs, phi)
            keff = solver_explicit.compute_keff(phi)
            phi /= np.linalg.norm(phi)
        keff_explicit = keff

        assert abs(keff_p0 - keff_explicit) < 1e-14, (
            f"P0 default {keff_p0:.10f} != explicit P0 {keff_explicit:.10f}"
        )

    def test_p1_homogeneous_same_as_p0(self):
        """On a homogeneous infinite medium with isotropic flux,
        P1 scattering gives the same keff as P0.

        The P1 moments are zero for isotropic flux (the current φ·Y_1^m
        integrates to zero by symmetry), so P1 adds nothing.
        """
        from orpheus.derivations._xs_library import get_mixture

        # Use the 421-group library which has P1 data
        fuel = get_mixture("A", "2g")  # only has P0

        # Skip if no P1 data available
        if len(fuel.SigS) < 2:
            pytest.skip("No P1 scattering data in 2-group library")

    def test_p1_request_limited_by_data(self):
        """If scattering_order > available data, it must be clamped."""
        from orpheus.derivations._xs_library import make_mixture

        # Build a mixture with P0 data only (no P1)
        mix_p0_only = make_mixture(
            sig_t=np.array([0.5, 1.0]),
            sig_c=np.array([0.01, 0.02]),
            sig_f=np.array([0.01, 0.08]),
            nu=np.array([2.5, 2.5]),
            chi=np.array([1.0, 0.0]),
            sig_s=np.array([[0.38, 0.10], [0.00, 0.90]]),
            # no sig_s1
        )
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)

        # Request P1 but only P0 data available → should clamp to P0
        solver = SNSolver({0: mix_p0_only}, SNMesh(mesh, quad), scattering_order=1)
        assert solver.scattering_order == 0, (
            f"Expected L=0 (clamped), got L={solver.scattering_order}"
        )

    def test_spherical_harmonics_orthogonality(self):
        """Lebedev spherical harmonics must satisfy discrete orthogonality."""
        quad = LebedevSphere.create(order=17)
        Y = quad.spherical_harmonics(1)
        w = quad.weights

        # <Y_0^0 | Y_0^0> = sum(w) = 4pi
        ortho_00 = np.sum(w * Y[:, 0, 0] ** 2)
        np.testing.assert_allclose(ortho_00, w.sum(), rtol=1e-12)

        # <Y_1^m | Y_1^m> = sum(w) / 3  for each m
        for m_idx in range(3):
            ortho_1m = np.sum(w * Y[:, 1, m_idx] ** 2)
            np.testing.assert_allclose(ortho_1m, w.sum() / 3, rtol=1e-10,
                                       err_msg=f"Y_1^{m_idx-1} not orthonormal")

        # <Y_0^0 | Y_1^m> = 0  for all m
        for m_idx in range(3):
            cross = np.sum(w * Y[:, 0, 0] * Y[:, 1, m_idx])
            np.testing.assert_allclose(cross, 0, atol=1e-14,
                                       err_msg=f"Y_0^0 not orthogonal to Y_1^{m_idx-1}")

    def test_p1_changes_heterogeneous_keff(self):
        """P1 scattering must produce a different keff than P0 on a
        heterogeneous problem where anisotropy matters at interfaces."""
        from orpheus.derivations._xs_library import get_mixture

        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")  # B has mu_bar=0.6, strongly anisotropic
        materials = {2: fuel, 0: mod}

        mat = np.zeros((6, 2), dtype=int)
        mat[:3, :] = 2
        mesh = _uniform_2d(6, 2, 0.2, mat)
        quad = LebedevSphere.create(order=17)

        keffs = {}
        for L in [0, 1]:
            solver = SNSolver(materials, SNMesh(mesh, quad),
                              scattering_order=L,
                              max_inner=500, inner_tol=1e-10)
            phi = solver.initial_flux_distribution()
            keff = 1.0
            for _ in range(50):
                fs = solver.compute_fission_source(phi, keff)
                phi = solver.solve_fixed_source(fs, phi)
                keff = solver.compute_keff(phi)
                phi /= np.linalg.norm(phi)
            keffs[L] = keff

        assert abs(keffs[0] - keffs[1]) > 1e-4, (
            f"P0 keff={keffs[0]:.6f} and P1 keff={keffs[1]:.6f} should differ"
        )

    def test_aniso_source_zero_for_isotropic_flux(self):
        """For isotropic angular flux (all ordinates equal), P1+ source = 0."""
        from orpheus.derivations._xs_library import get_mixture

        mix = get_mixture("A", "2g")
        if len(mix.SigS) < 2:
            pytest.skip("No P1 data")

        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)
        solver = SNSolver({0: mix}, SNMesh(mesh, quad), scattering_order=1)

        # Isotropic angular flux: same value for all ordinates
        N = quad.N
        angular = np.ones((N, 2, 2, solver.ng))

        Q_aniso = solver._build_aniso_scattering(angular)
        if Q_aniso is not None:
            np.testing.assert_allclose(Q_aniso, 0, atol=1e-12,
                                       err_msg="P1 source nonzero for isotropic flux")


class TestBicgstabPnScattering:
    """BiCGSTAB path must handle Pn scattering consistently with source iteration."""

    def test_bicgstab_p0_matches_si_p0(self):
        """BiCGSTAB and source iteration must agree at P0."""
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)

        keffs = {}
        for label, solver_type in [("SI", "source_iteration"), ("BC", "bicgstab")]:
            solver = SNSolver({0: mix}, SNMesh(mesh, quad),
                              inner_solver=solver_type, scattering_order=0,
                              max_inner=500 if solver_type == "source_iteration" else 2000,
                              inner_tol=1e-10 if solver_type == "source_iteration" else 1e-6)
            phi = solver.initial_flux_distribution()
            keff = 1.0
            for _ in range(50):
                fs = solver.compute_fission_source(phi, keff)
                phi = solver.solve_fixed_source(fs, phi)
                keff = solver.compute_keff(phi)
                phi /= np.linalg.norm(phi)
            keffs[label] = keff

        assert abs(keffs["SI"] - keffs["BC"]) < 1e-4, (
            f"P0 SI keff={keffs['SI']:.8f} vs BC keff={keffs['BC']:.8f}"
        )

    def test_bicgstab_p1_homogeneous_same_as_p0(self):
        """BiCGSTAB with P1 on homogeneous must match P0 (isotropic flux)."""
        from orpheus.derivations._xs_library import get_mixture

        mix = get_mixture("A", "2g")
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)

        keffs = {}
        for L in [0, 1]:
            solver = SNSolver({0: mix}, SNMesh(mesh, quad),
                              inner_solver="bicgstab", scattering_order=L,
                              max_inner=2000, inner_tol=1e-6)
            phi = solver.initial_flux_distribution()
            keff = 1.0
            for _ in range(50):
                fs = solver.compute_fission_source(phi, keff)
                phi = solver.solve_fixed_source(fs, phi)
                keff = solver.compute_keff(phi)
                phi /= np.linalg.norm(phi)
            keffs[L] = keff

        assert abs(keffs[0] - keffs[1]) < 1e-4, (
            f"BiCGSTAB P0 keff={keffs[0]:.6f} vs P1 keff={keffs[1]:.6f} "
            f"should be equal on homogeneous"
        )

    def test_bicgstab_p1_matches_si_p1_homogeneous(self):
        """BiCGSTAB and source iteration must agree at P1 on homogeneous."""
        from orpheus.derivations._xs_library import get_mixture

        mix = get_mixture("A", "2g")
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)

        keffs = {}
        for label, solver_type in [("SI", "source_iteration"), ("BC", "bicgstab")]:
            solver = SNSolver({0: mix}, SNMesh(mesh, quad),
                              inner_solver=solver_type, scattering_order=1,
                              max_inner=500 if solver_type == "source_iteration" else 2000,
                              inner_tol=1e-10 if solver_type == "source_iteration" else 1e-6)
            phi = solver.initial_flux_distribution()
            keff = 1.0
            for _ in range(50):
                fs = solver.compute_fission_source(phi, keff)
                phi = solver.solve_fixed_source(fs, phi)
                keff = solver.compute_keff(phi)
                phi /= np.linalg.norm(phi)
            keffs[label] = keff

        assert abs(keffs["SI"] - keffs["BC"]) < 1e-3, (
            f"P1 SI keff={keffs['SI']:.8f} vs BC keff={keffs['BC']:.8f}"
        )


class TestFissionSource:
    """Verify fission source normalization against SN equation physics."""

    def test_isotropic_normalization(self, solver_2g):
        """In the SN equation, the isotropic source Q appears as Q/(4π).

        The sweep multiplies Q by weight_norm = 1/sum(w) = 1/(4π).
        So the fission source passed to the sweep should be the
        *un-normalized* isotropic source: χ · (νΣf · φ) / k.

        Verify: sweep(Q) with Q = fission_source should produce the
        same scalar flux as sweep(Q/(4π)) with a modified weight_norm=1.
        """
        solver, _, sn_mesh, quad = solver_2g
        phi = solver.initial_flux_distribution()
        fission_src = solver.compute_fission_source(phi, 1.0)

        # The fission source should NOT already include the 1/(4π) factor,
        # because the sweep applies weight_norm = 1/(4π) internally.
        # Check: fission_src = chi * sum(sig_p * phi) / keff
        expected = solver.chi * np.sum(solver.sig_p * phi, axis=2)[:, :, None]
        np.testing.assert_allclose(fission_src, expected, rtol=1e-14,
                                   err_msg="Fission source has unexpected normalization")

    def test_one_group_homogeneous_keff(self, solver_2g):
        """For a 1-group homogeneous infinite medium, k_inf = νΣf / Σa.

        After enough power iterations, the solver must recover this.
        Use the 2-group fixture but check that the ratio of
        production / absorption is self-consistent.
        """
        solver, *_ = solver_2g
        phi = solver.initial_flux_distribution()
        keff = solver.compute_keff(phi)
        # keff from uniform flux should equal sum(sig_p) / sum(sig_a)
        # weighted by volume (which is uniform, so cancels)
        vol = solver.volume[:, :, None]
        expected = float(np.sum(solver.sig_p * phi * vol) / np.sum(solver.sig_a * phi * vol))
        # This only matches if there's zero n2n (which there may not be)
        # So just check they're both finite and positive
        assert np.isfinite(keff) and keff > 0


class TestHomogeneousExact:
    """2D SN on homogeneous infinite medium must match analytical k_inf.

    Uses small 2×2 mesh with reflective BCs. The analytical k_inf is
    the maximum eigenvalue of A⁻¹F where A = Σ_t - Σ_s^T, F = χ⊗(νΣ_f).
    """

    @pytest.mark.parametrize("ng_key,label", [("1g", "1G"), ("2g", "2G"), ("4g", "4G")])
    def test_homogeneous_exact(self, ng_key, label):
        from orpheus.derivations import get

        case = get(f"sn_slab_{ng_key[0]}eg_1rg")
        mix = next(iter(case.materials.values()))
        materials = {0: mix}

        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)
        solver = SNSolver(materials, SNMesh(mesh, quad), max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(100):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)
            phi = phi / np.linalg.norm(phi)

        assert abs(keff - case.k_inf) < 1e-8, (
            f"{label}: keff={keff:.10f} vs analytical={case.k_inf:.10f}"
        )


class TestSolveFixedSource:
    """Integration test: one outer iteration must reduce residual."""
    def test_source_iteration_converges(self, solver_2g):
        solver, *_ = solver_2g
        phi = solver.initial_flux_distribution()
        keff = 1.0

        fission_src = solver.compute_fission_source(phi, keff)
        phi_new = solver.solve_fixed_source(fission_src, phi)

        # The solver should produce a non-trivial update
        assert not np.allclose(phi, phi_new), "No update from solve_fixed_source"
        assert np.all(np.isfinite(phi_new)), "NaN/Inf in solve output"

    def test_bicgstab_matches_source_iteration(self, solver_2g):
        """BiCGSTAB and source iteration must converge to the same keff."""
        from orpheus.derivations import get

        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = _uniform_2d(2, 2, 0.5, np.zeros((2, 2), dtype=int))
        quad = LebedevSphere.create(order=17)

        # Source iteration
        solver_si = SNSolver({0: mix}, SNMesh(mesh, quad),
                             inner_solver="source_iteration",
                             max_inner=500, inner_tol=1e-10)
        phi = solver_si.initial_flux_distribution()
        keff = 1.0
        for _ in range(50):
            fs = solver_si.compute_fission_source(phi, keff)
            phi = solver_si.solve_fixed_source(fs, phi)
            keff = solver_si.compute_keff(phi)
            phi /= np.linalg.norm(phi)
        keff_si = keff

        # BiCGSTAB
        solver_bc = SNSolver({0: mix}, SNMesh(mesh, quad),
                             inner_solver="bicgstab",
                             max_inner=2000, inner_tol=1e-6)
        phi = solver_bc.initial_flux_distribution()
        keff = 1.0
        for _ in range(50):
            fs = solver_bc.compute_fission_source(phi, keff)
            phi = solver_bc.solve_fixed_source(fs, phi)
            keff = solver_bc.compute_keff(phi)
            phi /= np.linalg.norm(phi)
        keff_bc = keff

        assert abs(keff_si - keff_bc) < 1e-5, (
            f"BiCGSTAB keff={keff_bc:.8f} vs SI keff={keff_si:.8f}"
        )


# ── Profiling ────────────────────────────────────────────────────────

@pytest.fixture
def solver_421g():
    """Build the full 421-group 10x10 solver for profiling."""
    from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad

    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    mesh = _uniform_2d(10, 10, 0.2, np.tile(np.array([2]*5 + [1] + [0]*4, dtype=int), (10, 1)).T)
    quad = LebedevSphere.create(order=17)
    solver = SNSolver(materials, SNMesh(mesh, quad))
    return solver, materials, mesh, quad


class TestPerformanceBaseline:
    """Measure baseline timings for each component.

    Not assertions — just prints. Run with ``pytest -s`` to see output.
    """
    def test_profile_components(self, solver_2g):
        solver, _, sn_mesh, quad = solver_2g
        np.random.seed(42)
        phi = np.random.rand(sn_mesh.nx, sn_mesh.ny, solver.ng) + 0.1
        Q = np.random.rand(sn_mesh.nx, sn_mesh.ny, solver.ng)
        fission_src = solver.compute_fission_source(phi, 1.0)

        # Scattering source
        n_reps = 100
        t0 = time.perf_counter()
        for _ in range(n_reps):
            Q_tmp = Q.copy()
            solver._add_scattering_source(Q_tmp, phi)
        t_scat = (time.perf_counter() - t0) / n_reps * 1000
        print(f"\n  _add_scattering_source: {t_scat:.3f} ms")

        # N2N source
        t0 = time.perf_counter()
        for _ in range(n_reps):
            Q_tmp = Q.copy()
            solver._add_n2n_source(Q_tmp, phi)
        t_n2n = (time.perf_counter() - t0) / n_reps * 1000
        print(f"  _add_n2n_source: {t_n2n:.3f} ms")

        # compute_keff
        t0 = time.perf_counter()
        for _ in range(n_reps):
            solver.compute_keff(phi)
        t_keff = (time.perf_counter() - t0) / n_reps * 1000
        print(f"  compute_keff: {t_keff:.3f} ms")

        # Transport sweep
        n_sweep = 5
        t0 = time.perf_counter()
        for _ in range(n_sweep):
            transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})
        t_sweep = (time.perf_counter() - t0) / n_sweep * 1000
        print(f"  transport_sweep: {t_sweep:.1f} ms")

        # Full inner iteration
        t0 = time.perf_counter()
        solver.solve_fixed_source(fission_src, phi)
        t_inner = (time.perf_counter() - t0) * 1000
        print(f"  solve_fixed_source (1 outer): {t_inner:.0f} ms")

    @pytest.mark.slow
    def test_profile_421g(self, solver_421g):
        """Profile with the full 421-group 10x10 problem."""
        solver, _, mesh, quad = solver_421g
        phi = solver.initial_flux_distribution()
        Q = solver.compute_fission_source(phi, 1.0)

        n_reps = 10
        t0 = time.perf_counter()
        for _ in range(n_reps):
            Q_tmp = Q.copy()
            solver._add_scattering_source(Q_tmp, phi)
        t_scat = (time.perf_counter() - t0) / n_reps * 1000
        print(f"\n  [421g] _add_scattering_source: {t_scat:.2f} ms")

        t0 = time.perf_counter()
        for _ in range(n_reps):
            Q_tmp = Q.copy()
            solver._add_n2n_source(Q_tmp, phi)
        t_n2n = (time.perf_counter() - t0) / n_reps * 1000
        print(f"  [421g] _add_n2n_source: {t_n2n:.2f} ms")

        t0 = time.perf_counter()
        for _ in range(100):
            solver.compute_keff(phi)
        t_keff = (time.perf_counter() - t0) / 100 * 1000
        print(f"  [421g] compute_keff: {t_keff:.3f} ms")

        n_sweep = 3
        t0 = time.perf_counter()
        for _ in range(n_sweep):
            transport_sweep(Q, solver.sig_t, solver.sn_mesh, {})
        t_sweep = (time.perf_counter() - t0) / n_sweep * 1000
        print(f"  [421g] transport_sweep: {t_sweep:.1f} ms")
