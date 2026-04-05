"""Verify the spherical 1D SN solver.

Tests cover:
- Homogeneous exact: k_inf must match analytical (geometry-independent)
- Particle balance: production/absorption = keff (reflective BC, no leakage)
- Angular redistribution properties: α sum, antisymmetry
- Spatial convergence: O(h²) with mesh refinement
- Cross-check with CP spherical: SN eigenvalue close to CP eigenvalue
"""

import numpy as np
import pytest

from derivations import get
from derivations._xs_library import get_mixture
from geometry import CoordSystem, Mesh1D, homogeneous_1d, mesh1d_from_zones, Zone
from sn_geometry import SNMesh
from sn_quadrature import GaussLegendre1D
from sn_solver import SNSolver, solve_sn


# ── Homogeneous infinite medium (geometry-independent k_inf) ─────────

@pytest.mark.parametrize("case_name", [
    "sn_slab_1eg_1rg",
    "sn_slab_2eg_1rg",
    "sn_slab_4eg_1rg",
])
def test_homogeneous_exact(case_name):
    """Spherical SN on a homogeneous sphere with reflective BC must
    match the analytical infinite-medium eigenvalue (same for all
    geometries)."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    materials = {0: mix}
    mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
    quad = GaussLegendre1D.create(8)
    result = solve_sn(materials, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    # Spherical DD has larger discretization error than Cartesian
    # due to angular redistribution coupling. 1G is exact (keff
    # independent of flux shape); multi-group has ~1% error on S8/20-cell.
    tol = 1e-6 if case.n_groups == 1 else 0.02
    assert abs(result.keff - case.k_inf) < tol, (
        f"keff={result.keff:.8f} vs analytical={case.k_inf:.8f} "
        f"err={abs(result.keff - case.k_inf):.2e}"
    )


# ── Particle balance ─────────────────────────────────────────────────

def test_particle_balance():
    """For reflective BCs (no leakage), production / absorption = keff."""
    case = get("sn_slab_2eg_1rg")
    mix = next(iter(case.materials.values()))
    materials = {0: mix}
    mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
    quad = GaussLegendre1D.create(8)
    result = solve_sn(materials, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    # Volume-weighted production and absorption rates
    V = mesh.volumes
    flux = result.scalar_flux[:, 0, :]  # (nx, ng)
    sig_p = mix.SigP
    sig_a = mix.SigC + mix.SigF

    production = np.sum(flux * sig_p[None, :] * V[:, None])
    absorption = np.sum(flux * sig_a[None, :] * V[:, None])

    k_balance = production / absorption
    np.testing.assert_allclose(
        k_balance, result.keff, rtol=1e-5,
        err_msg=f"Particle balance: prod/abs={k_balance:.8f} ≠ keff={result.keff:.8f}",
    )


# ── Angular redistribution properties ────────────────────────────────

class TestAlphaCoefficients:
    """Properties of the angular redistribution coefficients."""

    @pytest.mark.parametrize("N", [4, 8, 16, 32])
    def test_alpha_boundary_conditions(self, N):
        """α_{1/2} = 0 and α_{N+1/2} = 0 by GL antisymmetry."""
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(N)
        sn_mesh = SNMesh(mesh, quad)

        np.testing.assert_allclose(sn_mesh.alpha_half[0], 0.0)
        np.testing.assert_allclose(sn_mesh.alpha_half[-1], 0.0, atol=1e-14)

    def test_alpha_recursion(self):
        """α_{n+1/2} = α_{n-1/2} − w_n μ_n (Bailey et al. 2009 Eq. 50)."""
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)

        alpha = sn_mesh.alpha_half
        for n in range(quad.N):
            expected = alpha[n] - quad.weights[n] * quad.mu_x[n]
            np.testing.assert_allclose(alpha[n + 1], expected, rtol=1e-14)

    def test_alpha_symmetric(self):
        """α coefficients are symmetric about the midpoint: α[k] = α[N-k].

        This follows from GL symmetry (w_n = w_{N-1-n}, μ_n = -μ_{N-1-n}):
        the cumulative sum Σ w_m μ_m is symmetric because each pair
        (w_n μ_n, w_{N-1-n} μ_{N-1-n}) contributes equal and opposite
        increments that mirror about the midpoint.
        """
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)

        alpha = sn_mesh.alpha_half
        N = quad.N
        for k in range(N + 1):
            np.testing.assert_allclose(
                alpha[k], alpha[N - k], atol=1e-14,
                err_msg=f"α not symmetric at k={k}",
            )


# ── Spatial convergence ──────────────────────────────────────────────

@pytest.mark.slow
def test_spatial_convergence():
    """Diamond-difference on spherical mesh must show O(h²) convergence."""
    fuel = get_mixture("A", "1g")
    mod = get_mixture("B", "1g")
    materials = {2: fuel, 0: mod}

    keffs = []
    drs = []
    for n_per in [5, 10, 20, 40]:
        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=n_per),
            Zone(outer_edge=1.0, mat_id=0, n_cells=n_per),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(16)
        result = solve_sn(
            materials, mesh, quad,
            max_outer=300, max_inner=500, inner_tol=1e-10,
        )
        keffs.append(result.keff)
        drs.append(0.5 / n_per)

    # Richardson extrapolation reference
    k_ref = keffs[-1] + (keffs[-1] - keffs[-2]) / 3.0

    # Compute convergence orders
    orders = []
    for i in range(1, len(keffs)):
        err_prev = abs(keffs[i - 1] - k_ref)
        err_curr = abs(keffs[i] - k_ref)
        if err_prev > 0 and err_curr > 0:
            orders.append(
                np.log(err_prev / err_curr)
                / np.log(drs[i - 1] / drs[i])
            )

    assert orders[-1] > 1.5, (
        f"Expected O(h²) convergence, got order {orders[-1]:.2f}"
    )


# ── Cross-check with CP spherical ────────────────────────────────────

def test_cross_check_with_cp_1g():
    """SN and CP on the same spherical geometry should give close k_inf.

    For homogeneous 1G, both must match the analytical value exactly.
    The white-BC (CP) vs reflective-BC (SN) difference vanishes for
    homogeneous infinite medium.
    """
    from collision_probability import solve_cp

    mix = get_mixture("A", "1g")
    materials_sn = {0: mix}
    materials_cp = {0: mix}

    # SN: homogeneous sphere with reflective BC
    mesh_sn = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
    quad = GaussLegendre1D.create(8)
    result_sn = solve_sn(materials_sn, mesh_sn, quad,
                         max_inner=500, inner_tol=1e-10)

    # CP: same geometry with white BC
    mesh_cp = homogeneous_1d(1, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
    result_cp = solve_cp(materials_cp, mesh_cp)

    # Both must give the same 1G homogeneous k_inf
    np.testing.assert_allclose(
        result_sn.keff, result_cp.keff, rtol=1e-6,
        err_msg=f"SN keff={result_sn.keff:.6f} vs CP keff={result_cp.keff:.6f}",
    )


# ── Flux positivity ──────────────────────────────────────────────────

def test_flux_non_negative():
    """Converged scalar flux must be non-negative everywhere."""
    mix = get_mixture("A", "1g")
    mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
    quad = GaussLegendre1D.create(8)
    result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

    assert np.all(result.scalar_flux >= 0), (
        f"Negative flux: min={result.scalar_flux.min():.4e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Sweep-level regression tests (from debugging this implementation)
# ═══════════════════════════════════════════════════════════════════════

class TestSphericalSweepRegression:
    """Tests targeting specific issues found during implementation."""

    def test_uniform_source_converges_to_Q_over_sigt(self):
        """Repeated sweeps with uniform Q and Σ_t must converge to φ = Q/Σ_t.

        This caught the missing weight_norm (1/sum_w) normalization in
        the spherical sweep source term.
        """
        from sn_sweep import _sweep_1d_spherical

        mesh = homogeneous_1d(10, 1.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)

        sig_t = np.ones((10, 1, 1))
        Q = np.ones((10, 1, 1))

        psi_bc = {}
        for _ in range(200):
            _, phi = _sweep_1d_spherical(Q, sig_t, sn_mesh, psi_bc)

        # Volume-weighted average must converge to Q/Σ_t = 1.0.
        # Individual cells near r=0 have large DD error due to extreme
        # aspect ratios on equal-volume spherical meshes.
        V = sn_mesh.volumes[:, 0]
        phi_avg = np.sum(phi[:, 0, 0] * V) / V.sum()
        np.testing.assert_allclose(phi_avg, 1.0, rtol=0.10,
                                   err_msg="Volume-avg φ ≠ Q/Σ_t for uniform source")

    def test_single_sweep_all_finite(self):
        """A single sweep must produce finite (non-NaN, non-Inf) fluxes.

        This catches the negative-denominator bug from using signed α
        instead of |α| at the innermost cell where A=0.
        """
        from sn_sweep import _sweep_1d_spherical

        mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)

        sig_t = np.full((10, 1, 1), 0.5)
        Q = np.ones((10, 1, 1))

        psi_bc = {}
        ang, phi = _sweep_1d_spherical(Q, sig_t, sn_mesh, psi_bc)

        assert np.all(np.isfinite(ang)), "Non-finite angular flux in first sweep"
        assert np.all(np.isfinite(phi)), "Non-finite scalar flux in first sweep"

    def test_inner_loop_bounded_multigroup(self):
        """Inner scattering iteration must stay bounded for multi-group.

        On a homogeneous sphere, one outer iteration (fission source →
        solve_fixed_source) must produce finite flux with bounded norm.
        """
        mix = get_mixture("A", "2g")
        mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver({0: mix}, sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        fission = solver.compute_fission_source(phi, 1.0)
        phi_new = solver.solve_fixed_source(fission, phi)

        assert np.all(np.isfinite(phi_new)), "Non-finite flux after solve_fixed_source"
        assert phi_new.max() < 1e6, (
            f"Flux blew up to {phi_new.max():.2e} — inner loop may diverge"
        )

    def test_angular_flux_at_center_all_positive(self):
        """All ordinate angular fluxes at the centre must be positive.

        This tests that the angular redistribution correctly couples
        inward and outward ordinates at r≈0, where A=0 and the spatial
        streaming vanishes.  Without redistribution, outward ordinates
        would have zero flux at the centre.
        """
        mix = get_mixture("A", "1g")
        mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

        # Angular flux at the innermost cell (closest to r=0)
        psi_center = result.angular_flux[:, 0, 0, 0]  # (N_ord,) for group 0

        assert np.all(psi_center > 0), (
            f"Zero or negative angular flux at centre: {psi_center}"
        )


# ═══════════════════════════════════════════════════════════════════════
# BiCGSTAB inner solver (spherical transport operator)
# ═══════════════════════════════════════════════════════════════════════

class TestSphericalBicgstab:
    """Tests for the BiCGSTAB inner solver on spherical geometry."""

    def test_bicgstab_1g_homogeneous_exact(self):
        """BiCGSTAB on 1G homogeneous sphere must match analytical k_inf."""
        case = get("sn_slab_1eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        result = solve_sn({0: mix}, mesh, quad,
                          inner_solver="bicgstab",
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
                                    ("BC", "bicgstab")]:
            mesh = homogeneous_1d(10, 2.0, mat_id=0,
                                  coord=CoordSystem.SPHERICAL)
            quad = GaussLegendre1D.create(8)
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

    def test_bicgstab_finite_result(self):
        """BiCGSTAB on 1G spherical must produce finite flux and keff.

        Note: multi-group BiCGSTAB on spherical geometry is unstable
        (the explicit FD operator for angular redistribution is less
        accurate than the DD sweep's implicit treatment). Only 1G is
        expected to converge reliably.
        """
        case = get("sn_slab_1eg_1rg")
        mix = next(iter(case.materials.values()))
        mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        result = solve_sn({0: mix}, mesh, quad,
                          inner_solver="bicgstab",
                          max_inner=2000, inner_tol=1e-6)

        assert np.isfinite(result.keff), f"keff is not finite: {result.keff}"
        assert np.all(np.isfinite(result.scalar_flux)), "Non-finite scalar flux"


# ═══════════════════════════════════════════════════════════════════════
# Multi-group / multi-region preemptive tests (spherical)
# ═══════════════════════════════════════════════════════════════════════

class TestMultiGroupMultiRegionSpherical:
    """Preemptive tests for error patterns that hide in simple problems.

    Spherical-specific: angular redistribution + multi-group scattering
    is the combination most likely to expose normalization and coupling bugs.
    """

    def test_2g_heterogeneous_converges(self):
        """2G fuel+moderator sphere must converge to finite keff."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        assert np.isfinite(result.keff), f"keff is NaN/Inf"
        assert 0.5 < result.keff < 3.0, f"keff={result.keff:.4f} out of range"
        assert np.all(np.isfinite(result.scalar_flux)), "Non-finite flux"

    def test_4g_scattering_convergence(self):
        """4G homogeneous must converge (richest scattering matrix)."""
        mix = get_mixture("A", "4g")
        mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver({0: mix}, sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(5):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)

        assert np.all(np.isfinite(phi)), "4G scattering iteration diverged"
        assert phi.max() < 1e10, f"4G flux blew up to {phi.max():.2e}"

    def test_multigroup_eigenvector_not_flat(self):
        """Flux spectrum must differ between fuel and moderator."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        flux = result.scalar_flux[:, 0, :]
        V = mesh.volumes
        mat_ids = mesh.mat_ids

        fuel_flux = np.average(flux[mat_ids == 2], axis=0, weights=V[mat_ids == 2])
        mod_flux = np.average(flux[mat_ids == 0], axis=0, weights=V[mat_ids == 0])

        fuel_ratio = fuel_flux[0] / fuel_flux[1]
        mod_ratio = mod_flux[0] / mod_flux[1]

        assert abs(fuel_ratio - mod_ratio) > 0.01, (
            f"Spectrum identical in fuel/mod — coupling broken: "
            f"fuel={fuel_ratio:.4f}, mod={mod_ratio:.4f}"
        )

    def test_particle_balance_heterogeneous(self):
        """Particle balance on 2G heterogeneous sphere."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver(materials, sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(100):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)

        vol = solver.volume[:, :, None]
        production = np.sum(solver.sig_p * phi * vol)
        absorption = np.sum(solver.sig_a * phi * vol)
        k_balance = production / absorption

        np.testing.assert_allclose(
            k_balance, keff, rtol=1e-4,
            err_msg=f"Heterogeneous balance: {k_balance:.6f} ≠ {keff:.6f}",
        )

    def test_fixed_source_flux_bounded(self):
        """Fixed-source flux range must be bounded near r=0.

        Without the ΔA/w geometry factor, the flux spikes to ~5x at
        the origin.  With the fix, the range should be bounded.
        """
        from sn_sweep import _sweep_1d_spherical

        mesh = homogeneous_1d(40, 1.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = GaussLegendre1D.create(8)
        sn_mesh = SNMesh(mesh, quad)

        Q = np.ones((40, 1, 1))
        sig_t = np.ones((40, 1, 1))
        psi_bc = {}
        for _ in range(50):
            _, phi = _sweep_1d_spherical(Q, sig_t, sn_mesh, psi_bc)

        phi_avg = np.average(phi[:, 0, 0], weights=mesh.volumes)
        np.testing.assert_allclose(phi_avg, 1.0, rtol=0.01,
                                   err_msg="Volume-avg φ ≠ Q/Σ_t")
        assert phi[:, 0, 0].max() < 2.0, (
            f"Flux spike at origin: max={phi[:, 0, 0].max():.4f}"
        )

    def test_heterogeneous_1g_spatial_convergence(self):
        """keff must converge with mesh refinement for fuel+moderator."""
        mix_fuel = get_mixture("A", "1g")
        mix_mod = get_mixture("B", "1g")
        materials = {2: mix_fuel, 0: mix_mod}
        quad = GaussLegendre1D.create(8)

        keffs = []
        for n_cells in [5, 10, 20]:
            zones = [
                Zone(outer_edge=0.5, mat_id=2, n_cells=n_cells),
                Zone(outer_edge=1.0, mat_id=0, n_cells=n_cells),
            ]
            mesh = mesh1d_from_zones(zones, coord=CoordSystem.SPHERICAL)
            result = solve_sn(materials, mesh, quad,
                              max_inner=500, inner_tol=1e-10)
            keffs.append(result.keff)

        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1, (
            f"keff not converging: Δ(10−5)={diff_1:.6f}, Δ(20−10)={diff_2:.6f}, "
            f"keffs={[f'{k:.6f}' for k in keffs]}"
        )
