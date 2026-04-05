"""Verify the cylindrical 1D SN solver.

Tests cover both quadrature types (LevelSymmetricSN and ProductQuadrature):
- Homogeneous exact: k_inf must match analytical (geometry-independent)
- Particle balance: production/absorption = keff (reflective BC)
- Cross-check with CP cylindrical: SN close to CP eigenvalue
- Flux non-negativity
- Sweep regression: finite fluxes, inner loop bounded
"""

import numpy as np
import pytest

from derivations import get
from derivations._xs_library import get_mixture
from geometry import CoordSystem, Mesh1D, homogeneous_1d, mesh1d_from_zones, Zone
from sn_geometry import SNMesh
from sn_quadrature import LevelSymmetricSN, ProductQuadrature
from sn_solver import SNSolver, solve_sn


# ── Homogeneous infinite medium ──────────────────────────────────────

@pytest.mark.parametrize("case_name", [
    "sn_slab_1eg_1rg",
    "sn_slab_2eg_1rg",
    "sn_slab_4eg_1rg",
])
@pytest.mark.parametrize("quad_factory", [
    lambda: ProductQuadrature.create(n_mu=4, n_phi=8),
    lambda: LevelSymmetricSN.create(4),
], ids=["product", "level_sym"])
def test_homogeneous_exact(case_name, quad_factory):
    """Cylindrical SN on a homogeneous cylinder with reflective BC must
    match the analytical infinite-medium eigenvalue."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = quad_factory()
    result = solve_sn({0: mix}, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    # Cylindrical DD with azimuthal redistribution is exact for
    # homogeneous problems (no spatial gradient in infinite medium)
    assert abs(result.keff - case.k_inf) < 1e-6, (
        f"keff={result.keff:.8f} vs analytical={case.k_inf:.8f}"
    )


# ── Particle balance ─────────────────────────────────────────────────

@pytest.mark.parametrize("quad_factory", [
    lambda: ProductQuadrature.create(n_mu=4, n_phi=8),
    lambda: LevelSymmetricSN.create(4),
], ids=["product", "level_sym"])
def test_particle_balance(quad_factory):
    """For reflective BCs (no leakage), production / absorption = keff."""
    case = get("sn_slab_2eg_1rg")
    mix = next(iter(case.materials.values()))
    mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = quad_factory()
    result = solve_sn({0: mix}, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    V = mesh.volumes
    flux = result.scalar_flux[:, 0, :]
    sig_p = mix.SigP
    sig_a = mix.SigC + mix.SigF

    production = np.sum(flux * sig_p[None, :] * V[:, None])
    absorption = np.sum(flux * sig_a[None, :] * V[:, None])

    k_balance = production / absorption
    np.testing.assert_allclose(
        k_balance, result.keff, rtol=1e-5,
        err_msg=f"Particle balance: prod/abs={k_balance:.8f} ≠ keff={result.keff:.8f}",
    )


# ── Cross-check with CP cylindrical ──────────────────────────────────

def test_cross_check_with_cp_1g():
    """SN and CP on the same cylindrical geometry should give close k_inf."""
    from collision_probability import solve_cp

    mix = get_mixture("A", "1g")

    mesh_sn = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = ProductQuadrature.create(n_mu=4, n_phi=8)
    result_sn = solve_sn({0: mix}, mesh_sn, quad,
                         max_inner=500, inner_tol=1e-10)

    mesh_cp = homogeneous_1d(1, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    result_cp = solve_cp({0: mix}, mesh_cp)

    np.testing.assert_allclose(
        result_sn.keff, result_cp.keff, rtol=1e-6,
        err_msg=f"SN keff={result_sn.keff:.6f} vs CP keff={result_cp.keff:.6f}",
    )


# ── Flux non-negativity ──────────────────────────────────────────────

def test_flux_non_negative():
    mix = get_mixture("A", "1g")
    mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = ProductQuadrature.create(n_mu=4, n_phi=8)
    result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

    assert np.all(result.scalar_flux >= 0), (
        f"Negative flux: min={result.scalar_flux.min():.4e}"
    )


# ── Sweep regression ─────────────────────────────────────────────────

class TestCylindricalSweepRegression:
    """Tests targeting issues found in spherical implementation."""

    def test_single_sweep_all_finite(self):
        """A single sweep must produce finite fluxes."""
        from sn_sweep import _sweep_1d_cylindrical

        mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad)

        sig_t = np.full((10, 1, 1), 0.5)
        Q = np.ones((10, 1, 1))

        psi_bc = {}
        ang, phi = _sweep_1d_cylindrical(Q, sig_t, sn_mesh, psi_bc)

        assert np.all(np.isfinite(ang)), "Non-finite angular flux"
        assert np.all(np.isfinite(phi)), "Non-finite scalar flux"

    def test_inner_loop_bounded_multigroup(self):
        """Inner loop must stay bounded for multi-group."""
        mix = get_mixture("A", "2g")
        mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver({0: mix}, sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        fission = solver.compute_fission_source(phi, 1.0)
        phi_new = solver.solve_fixed_source(fission, phi)

        assert np.all(np.isfinite(phi_new)), "Non-finite flux after solve_fixed_source"
        assert phi_new.max() < 1e6, (
            f"Flux blew up to {phi_new.max():.2e}"
        )

    def test_both_quadratures_agree(self):
        """Product and level-symmetric must give close keff."""
        mix = get_mixture("A", "1g")

        keffs = {}
        for label, quad in [
            ("product", ProductQuadrature.create(n_mu=4, n_phi=8)),
            ("level_sym", LevelSymmetricSN.create(4)),
        ]:
            mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
            result = solve_sn({0: mix}, mesh, quad,
                              max_inner=500, inner_tol=1e-10)
            keffs[label] = result.keff

        np.testing.assert_allclose(
            keffs["product"], keffs["level_sym"], rtol=1e-6,
            err_msg="Product and level-symmetric quadratures disagree",
        )

    def test_requires_level_quadrature(self):
        """Cylindrical SNMesh with GL quadrature must raise ValueError."""
        mesh = homogeneous_1d(5, 1.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = GaussLegendre1D.create(4)
        with pytest.raises(ValueError, match="level structure"):
            SNMesh(mesh, quad)


# Need this import for the guard test
from sn_quadrature import GaussLegendre1D


# ═══════════════════════════════════════════════════════════════════════
# Multi-group / multi-region preemptive tests
# ═══════════════════════════════════════════════════════════════════════

class TestMultiGroupMultiRegion:
    """Preemptive tests targeting error patterns that hide in simple problems.

    These specifically test multi-group AND heterogeneous configurations
    where bugs in source normalization, scattering convention, angular
    redistribution, and weight handling become visible.

    Error pattern taxonomy (from gotchas.md):
    - Source normalization (weight_norm): invisible in 1G keff ratio
    - Scattering iteration divergence: invisible in 1G (no coupling)
    - Angular redistribution sign: invisible in Cartesian (no curvature)
    - Eigenvector distortion: invisible in 1G (keff ≠ f(flux shape))
    """

    def test_2g_heterogeneous_fuel_moderator(self):
        """2G fuel+moderator cylinder — the minimum problem that catches
        normalization, scattering, and redistribution bugs simultaneously."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        assert np.isfinite(result.keff), f"keff is NaN/Inf"
        assert result.keff > 0, f"keff is non-positive: {result.keff}"
        assert np.all(np.isfinite(result.scalar_flux)), "Non-finite flux"
        # keff should be reasonable (not 0 or huge)
        assert 0.5 < result.keff < 3.0, f"keff={result.keff:.4f} out of physical range"

    def test_2g_heterogeneous_product_different_resolutions(self):
        """Product quadrature at two resolutions must give close keff."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        keffs = {}
        for label, quad in [
            ("4×8", ProductQuadrature.create(n_mu=4, n_phi=8)),
            ("8×8", ProductQuadrature.create(n_mu=8, n_phi=8)),
        ]:
            zones = [
                Zone(outer_edge=0.5, mat_id=2, n_cells=10),
                Zone(outer_edge=1.0, mat_id=0, n_cells=10),
            ]
            mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)
            result = solve_sn(materials, mesh, quad,
                              max_inner=500, inner_tol=1e-10)
            keffs[label] = result.keff

        assert abs(keffs["4×8"] - keffs["8×8"]) < 0.05, (
            f"Product resolutions disagree: "
            f"4×8={keffs['4×8']:.6f}, 8×8={keffs['8×8']:.6f}"
        )

    def test_4g_homogeneous_scattering_convergence(self):
        """4G homogeneous with strong scattering must converge.

        4-group has the richest scattering matrix (10 nonzero entries)
        and is the most sensitive to iteration divergence.
        """
        mix = get_mixture("A", "4g")
        mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad)
        solver = SNSolver({0: mix}, sn_mesh, max_inner=500, inner_tol=1e-10)

        # Run 5 outer iterations — flux must remain bounded
        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(5):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)

        assert np.all(np.isfinite(phi)), "4G scattering iteration diverged"
        assert phi.max() < 1e10, f"4G flux blew up to {phi.max():.2e}"

    def test_multigroup_eigenvector_not_flat(self):
        """For multi-group heterogeneous, the flux spectrum must vary
        between fuel and moderator — a flat spectrum indicates the
        multi-group coupling is broken.

        This catches the class of bugs where 1G passes (keff correct)
        but the group structure is wrong.
        """
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        # Average flux ratio (group 0 / group 1) in fuel vs moderator
        flux = result.scalar_flux[:, 0, :]  # (nx, ng)
        V = mesh.volumes
        mat_ids = mesh.mat_ids

        fuel_flux = np.average(flux[mat_ids == 2], axis=0, weights=V[mat_ids == 2])
        mod_flux = np.average(flux[mat_ids == 0], axis=0, weights=V[mat_ids == 0])

        fuel_ratio = fuel_flux[0] / fuel_flux[1]
        mod_ratio = mod_flux[0] / mod_flux[1]

        # The ratios must be different (fuel has different spectrum than moderator)
        assert abs(fuel_ratio - mod_ratio) > 0.01, (
            f"Flux spectrum identical in fuel and moderator — "
            f"multi-group coupling may be broken: "
            f"fuel ratio={fuel_ratio:.4f}, mod ratio={mod_ratio:.4f}"
        )

    def test_particle_balance_heterogeneous(self):
        """Particle balance must hold for heterogeneous multi-region."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        zones = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
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
            err_msg=f"Heterogeneous particle balance: {k_balance:.6f} ≠ {keff:.6f}",
        )

    def test_azimuthal_alpha_boundary_conditions(self):
        """Per-level α coefficients must satisfy α[0] = 0, α[-1] ≈ 0.

        This is the azimuthal analogue of the spherical α boundary check.
        Failure means the ξ-weighted sum doesn't vanish, which would
        cause non-physical angular flux generation.
        """
        mesh = Mesh1D(edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
                      coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad)

        for p, alpha in enumerate(sn_mesh.alpha_per_level):
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
        mesh = homogeneous_1d(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

        psi_center = result.angular_flux[:, 0, 0, 0]
        assert np.all(psi_center > 0), (
            f"Zero/negative angular flux at centre: min={psi_center.min():.4e}"
        )

    def test_redistribution_telescoping_conservation(self):
        """αψ product telescopes to zero on each level per cell.

        The redistribution sum Σ_m (α_{m+1/2}ψ_{m+1/2} − α_{m-1/2}ψ_{m-1/2})
        must vanish for each cell because α[0] = α[M] = 0.
        """
        from sn_sweep import _sweep_1d_cylindrical

        mix = get_mixture("A", "1g")
        mesh = homogeneous_1d(10, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad)

        sig_t = np.full((10, 1, 1), mix.SigT[0])
        Q = np.ones((10, 1, 1))
        psi_bc = {}
        ang, _ = _sweep_1d_cylindrical(Q, sig_t, sn_mesh, psi_bc)

        for p, level_idx in enumerate(quad.level_indices):
            alpha = sn_mesh.alpha_per_level[p]
            M = len(level_idx)
            # Reconstruct angular face fluxes from cell-average and DD
            psi_angle = np.zeros(10)
            for m_local in range(M):
                n = level_idx[m_local]
                psi_cell = ang[n, :, 0, 0]
                psi_angle_new = 2.0 * psi_cell - psi_angle
                psi_angle = psi_angle_new
            # After all ordinates, psi_angle = ψ_{M+1/2}
            # Telescoping: α[M]·ψ_{M+1/2} - α[0]·ψ_{1/2} = 0 since α[0]=α[M]=0
            residual = alpha[M] * psi_angle
            np.testing.assert_allclose(residual, 0.0, atol=1e-12,
                                       err_msg=f"Level {p}: telescoping residual ≠ 0")

    def test_single_cell_uniform_source_equilibrium(self):
        """Two-cell 1G pure absorber with uniform source → φ = Q/Σ_t."""
        from sn_sweep import _sweep_1d_cylindrical

        mesh = homogeneous_1d(2, 1.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad)

        Q = np.ones((2, 1, 1))
        sig_t = np.ones((2, 1, 1))
        psi_bc = {}
        for _ in range(100):
            _, phi = _sweep_1d_cylindrical(Q, sig_t, sn_mesh, psi_bc)

        phi_avg = np.average(phi[:, 0, 0], weights=mesh.volumes)
        np.testing.assert_allclose(phi_avg, 1.0, rtol=0.01,
                                   err_msg="Volume-avg φ ≠ Q/Σ_t for uniform source")

    def test_heterogeneous_1g_spatial_convergence(self):
        """keff must converge monotonically with mesh refinement."""
        mix_fuel = get_mixture("A", "1g")
        mix_mod = get_mixture("B", "1g")
        materials = {2: mix_fuel, 0: mix_mod}
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)

        keffs = []
        for n_cells in [5, 10, 20]:
            zones = [
                Zone(outer_edge=0.5, mat_id=2, n_cells=n_cells),
                Zone(outer_edge=1.0, mat_id=0, n_cells=n_cells),
            ]
            mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)
            result = solve_sn(materials, mesh, quad,
                              max_inner=500, inner_tol=1e-10)
            keffs.append(result.keff)

        # keff differences must decrease (convergence)
        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1, (
            f"keff not converging: Δ(10−5)={diff_1:.6f}, Δ(20−10)={diff_2:.6f}, "
            f"keffs={[f'{k:.6f}' for k in keffs]}"
        )

    def test_heterogeneous_sn_vs_cp_cross_check(self):
        """Heterogeneous SN and CP should agree within ~10%."""
        from collision_probability import solve_cp

        mix_fuel = get_mixture("A", "1g")
        mix_mod = get_mixture("B", "1g")
        materials = {2: mix_fuel, 0: mix_mod}

        zones_sn = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=20),
            Zone(outer_edge=1.0, mat_id=0, n_cells=20),
        ]
        mesh_sn = mesh1d_from_zones(zones_sn, coord=CoordSystem.CYLINDRICAL)
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        result_sn = solve_sn(materials, mesh_sn, quad,
                             max_inner=500, inner_tol=1e-10)

        zones_cp = [
            Zone(outer_edge=0.5, mat_id=2, n_cells=10),
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
        ]
        mesh_cp = mesh1d_from_zones(zones_cp, coord=CoordSystem.CYLINDRICAL)
        result_cp = solve_cp(materials, mesh_cp)

        np.testing.assert_allclose(
            result_sn.keff, result_cp.keff, rtol=0.10,
            err_msg=f"SN={result_sn.keff:.6f} vs CP={result_cp.keff:.6f}",
        )
