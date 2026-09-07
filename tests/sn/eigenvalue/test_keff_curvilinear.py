"""Curvilinear (cylinder + sphere) SN k-eigenvalue verification.

Split from the legacy ``tests/sn/test_cylindrical.py`` /
``test_spherical.py`` (SN taxonomy reorg). This is the standalone home
the spec mandates for curvilinear k-eff — it previously existed ONLY
inside the mixed cyl/sph files and would have been lost in a naive
split. It carries the only ``l2`` markers in the 1-D suite.

The cylindrical eigenvalue tests are below; the spherical ones are
appended by the test_spherical split (phase 4). The per-sweep
regression guards moved to ``tests/sn/sweep/curvilinear/``; the CP
cross-checks to ``tests/sn/verification/analytical/``.
"""

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import BC, CoordSystem
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import SNSolver, solve_sn
from tests.sn._test_helpers import (
    curvilinear_homogeneous_mesh as _homogeneous_mesh,
    curvilinear_two_region_mesh as _two_region_mesh,
)
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux

# Cylinder and sphere carry DIFFERENT equation-coverage lists (the
# legacy modules did too), so the verifies(...) sets are applied
# per-section, NOT as a single module-level pytestmark — a module mark
# would falsely stamp the cylinder labels onto the sphere tests (and
# vice versa). Each constant is the verbatim list from its legacy
# module so no verifies(...) edge is lost in the split.
_CYL_VERIFIES = pytest.mark.verifies(
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
_SPH_VERIFIES = pytest.mark.verifies(
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


# ═══════════════════════════════════════════════════════════════════════
# Cylindrical eigenvalue
# ═══════════════════════════════════════════════════════════════════════

@_CYL_VERIFIES
@pytest.mark.l1
@pytest.mark.parametrize("case_name", [
    "sn_slab_1eg_1rg",
    "sn_slab_2eg_1rg",
    "sn_slab_4eg_1rg",
])
# Two SPLITS of the admitted folded family (Q5.6.3 replaces the pre-flip
# product-vs-LS pair): 4 levels × 4 angles vs 8 levels × 2 angles drive
# different level bookkeeping through the full eigenvalue path.
@pytest.mark.parametrize("quad_factory", [
    lambda: Quadrature.folded_product(n_mu=4, n_phi=8),
    lambda: Quadrature.folded_product(n_mu=8, n_phi=4),
], ids=["folded_4x8", "folded_8x4"])
def test_homogeneous_exact(case_name, quad_factory):
    """Cylindrical SN on a homogeneous cylinder with reflective BC must
    match the analytical infinite-medium eigenvalue."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = quad_factory()
    result = solve_sn({0: mix}, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    assert abs(result.keff - case.k_inf) < 1e-6, (
        f"keff={result.keff:.8f} vs analytical={case.k_inf:.8f}"
    )


@_CYL_VERIFIES
@pytest.mark.l1
@pytest.mark.parametrize("quad_factory", [
    lambda: Quadrature.folded_product(n_mu=4, n_phi=8),
    lambda: Quadrature.folded_product(n_mu=8, n_phi=4),
], ids=["folded_4x8", "folded_8x4"])
def test_particle_balance(quad_factory):
    """For reflective BCs (no leakage), production / absorption = keff."""
    case = get("sn_slab_2eg_1rg")
    mix = next(iter(case.materials.values()))
    mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = quad_factory()
    result = solve_sn({0: mix}, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    V = mesh.volumes
    flux = result.scalar_flux.values.T  # PR-INDEX-5
    sig_p = mix.SigP
    sig_a = mix.SigC + mix.SigF

    production = np.sum(flux * sig_p[None, :] * V[:, None])
    absorption = np.sum(flux * sig_a[None, :] * V[:, None])

    k_balance = production / absorption
    np.testing.assert_allclose(
        k_balance, result.keff, rtol=1e-5,
        err_msg=f"Particle balance: prod/abs={k_balance:.8f} ≠ keff={result.keff:.8f}",
    )


@_CYL_VERIFIES
@pytest.mark.l1
def test_flux_non_negative():
    mix = get_mixture("A", "1g")
    mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
    quad = Quadrature.folded_product(n_mu=4, n_phi=8)
    result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

    assert np.all(result.scalar_flux.values >= 0), (
        f"Negative flux: min={result.scalar_flux.values.min():.4e}"
    )


@_CYL_VERIFIES
@pytest.mark.l2
class TestCylinderMultiGroupMultiRegion:
    """Multi-group heterogeneous cylindrical eigenvalue integration.

    The minimum problems that catch normalization, scattering, and
    eigenvector-distortion bugs simultaneously — invisible in 1G keff.
    """

    def test_2g_heterogeneous_fuel_moderator(self):
        """2G fuel+moderator cylinder — minimum problem catching
        normalization, scattering, and redistribution bugs at once."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
            coord=CoordSystem.CYLINDRICAL,
        )
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        assert np.isfinite(result.keff), "keff is NaN/Inf"
        assert result.keff > 0, f"keff is non-positive: {result.keff}"
        assert np.all(np.isfinite(result.scalar_flux.values)), "Non-finite flux"
        assert 0.5 < result.keff < 3.0, f"keff={result.keff:.4f} out of physical range"

    def test_2g_heterogeneous_product_different_resolutions(self):
        """Folded quadrature at two resolutions must give close keff."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        keffs = {}
        for label, quad in [
            ("4×8", Quadrature.folded_product(n_mu=4, n_phi=8)),
            ("8×8", Quadrature.folded_product(n_mu=8, n_phi=8)),
        ]:
            mesh = _two_region_mesh(
                outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
                coord=CoordSystem.CYLINDRICAL,
            )
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
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL)
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, max_inner=500, inner_tol=1e-10)

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
        """
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
            coord=CoordSystem.CYLINDRICAL,
        )
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        flux = result.scalar_flux.values.T  # PR-INDEX-5  # (nx, ng)
        V = mesh.volumes
        mat_ids = mesh.mat_ids

        fuel_flux = np.average(flux[mat_ids == 2], axis=0, weights=V[mat_ids == 2])
        mod_flux = np.average(flux[mat_ids == 0], axis=0, weights=V[mat_ids == 0])

        fuel_ratio = fuel_flux[0] / fuel_flux[1]
        mod_ratio = mod_flux[0] / mod_flux[1]

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

        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
            coord=CoordSystem.CYLINDRICAL,
        )
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        sn_mesh = SNMesh(mesh, quad, materials)
        solver = SNSolver(sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(100):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)

        vol = solver.volume[None, :]
        production = np.sum(solver.mat_xs.fission_production * phi * vol)
        absorption = np.sum(solver.mat_xs.absorption_cross_section * phi * vol)
        k_balance = production / absorption

        np.testing.assert_allclose(
            k_balance, keff, rtol=1e-4,
            err_msg=f"Heterogeneous particle balance: {k_balance:.6f} ≠ {keff:.6f}",
        )

    def test_heterogeneous_1g_spatial_convergence(self):
        """keff must converge monotonically with mesh refinement."""
        mix_fuel = get_mixture("A", "1g")
        mix_mod = get_mixture("B", "1g")
        materials = {2: mix_fuel, 0: mix_mod}
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)

        keffs = []
        for n_cells in [5, 10, 20]:
            mesh = _two_region_mesh(
                outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(n_cells, n_cells),
                coord=CoordSystem.CYLINDRICAL,
            )
            result = solve_sn(materials, mesh, quad,
                              max_inner=500, inner_tol=1e-10)
            keffs.append(result.keff)

        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1, (
            f"keff not converging: Δ(10−5)={diff_1:.6f}, Δ(20−10)={diff_2:.6f}, "
            f"keffs={[f'{k:.6f}' for k in keffs]}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Spherical eigenvalue (split from the legacy test_spherical.py)
# ═══════════════════════════════════════════════════════════════════════
#
# Grouped in classes (collision-free with the cylinder module functions)
# and decorated with _SPH_VERIFIES — the sphere's own equation-coverage
# list, distinct from the cylinder's.

@_SPH_VERIFIES
@pytest.mark.l1
class TestSphereEigenvalue:
    """Spherical homogeneous-exact / balance / convergence eigenvalue gates."""

    @pytest.mark.parametrize("case_name", [
        "sn_slab_1eg_1rg",
        "sn_slab_2eg_1rg",
        "sn_slab_4eg_1rg",
    ])
    def test_homogeneous_exact(self, case_name):
        """Spherical SN on a homogeneous sphere with reflective BC must
        match the analytical infinite-medium eigenvalue."""
        case = get(case_name)
        mix = next(iter(case.materials.values()))
        materials = {0: mix}
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
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

    def test_particle_balance(self):
        """For reflective BCs (no leakage), production / absorption = keff."""
        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))
        materials = {0: mix}
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        V = mesh.volumes
        flux = result.scalar_flux.values.T  # PR-INDEX-5  # (nx, ng)
        sig_p = mix.SigP
        sig_a = mix.SigC + mix.SigF

        production = np.sum(flux * sig_p[None, :] * V[:, None])
        absorption = np.sum(flux * sig_a[None, :] * V[:, None])

        k_balance = production / absorption
        np.testing.assert_allclose(
            k_balance, result.keff, rtol=1e-5,
            err_msg=f"Particle balance: prod/abs={k_balance:.8f} ≠ keff={result.keff:.8f}",
        )

    @pytest.mark.slow
    def test_spatial_convergence(self):
        """Diamond-difference on spherical mesh must show O(h²) convergence."""
        fuel = get_mixture("A", "1g")
        mod = get_mixture("B", "1g")
        materials = {2: fuel, 0: mod}

        keffs = []
        drs = []
        for n_per in [5, 10, 20, 40]:
            mesh = _two_region_mesh(
                outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(n_per, n_per),
                coord=CoordSystem.SPHERICAL,
            )
            quad = Quadrature.gauss_legendre(16)
            result = solve_sn(
                materials, mesh, quad,
                max_outer=300, max_inner=500, inner_tol=1e-10,
            )
            keffs.append(result.keff)
            drs.append(0.5 / n_per)

        k_ref = keffs[-1] + (keffs[-1] - keffs[-2]) / 3.0

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

    def test_flux_non_negative(self):
        """Converged scalar flux must be non-negative everywhere."""
        mix = get_mixture("A", "1g")
        mesh = _homogeneous_mesh(10, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn({0: mix}, mesh, quad, max_inner=500, inner_tol=1e-10)

        assert np.all(result.scalar_flux.values >= 0), (
            f"Negative flux: min={result.scalar_flux.values.min():.4e}"
        )


@_SPH_VERIFIES
@pytest.mark.l2
class TestMultiGroupMultiRegionSpherical:
    """Multi-group heterogeneous spherical eigenvalue integration.

    Spherical-specific: angular redistribution + multi-group scattering
    is the combination most likely to expose normalization / coupling bugs.
    """

    def test_2g_heterogeneous_converges(self):
        """2G fuel+moderator sphere must converge to finite keff."""
        fuel = get_mixture("A", "2g")
        mod = get_mixture("B", "2g")
        materials = {2: fuel, 0: mod}

        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
            coord=CoordSystem.SPHERICAL,
        )
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        assert np.isfinite(result.keff), "keff is NaN/Inf"
        assert 0.1 < result.keff < 3.0, f"keff={result.keff:.4f} out of range"
        assert np.all(np.isfinite(result.scalar_flux.values)), "Non-finite flux"

    def test_4g_scattering_convergence(self):
        """4G homogeneous must converge (richest scattering matrix)."""
        mix = get_mixture("A", "4g")
        mesh = _homogeneous_mesh(20, 2.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, max_inner=500, inner_tol=1e-10)

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

        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
            coord=CoordSystem.SPHERICAL,
        )
        quad = Quadrature.gauss_legendre(8)
        result = solve_sn(materials, mesh, quad,
                          max_inner=500, inner_tol=1e-10)

        flux = result.scalar_flux.values.T  # PR-INDEX-5
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

        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
            coord=CoordSystem.SPHERICAL,
        )
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, materials)
        solver = SNSolver(sn_mesh, max_inner=500, inner_tol=1e-10)

        phi = solver.initial_flux_distribution()
        keff = 1.0
        for _ in range(100):
            fs = solver.compute_fission_source(phi, keff)
            phi = solver.solve_fixed_source(fs, phi)
            keff = solver.compute_keff(phi)

        vol = solver.volume[None, :]
        production = np.sum(solver.mat_xs.fission_production * phi * vol)
        absorption = np.sum(solver.mat_xs.absorption_cross_section * phi * vol)
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
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once
        from orpheus.transport.source_sinks import AngularSourceSink
        from tests.sn._test_helpers import placeholder_materials

        mesh = _homogeneous_mesh(40, 1.0, mat_id=0, coord=CoordSystem.SPHERICAL)
        quad = Quadrature.gauss_legendre(8)
        sn_mesh = SNMesh(mesh, quad, placeholder_materials())

        sig_t = np.ones((1, *sn_mesh.spatial_shape))    # (ng, *spatial)
        Q_iso = np.ones((1, *sn_mesh.spatial_shape))    # (ng, *spatial)
        source = AngularSourceSink.from_isotropic(Q_iso, sn_mesh)
        boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        phi = None
        for _ in range(50):
            # Wave O (#208) O.4a.2 — the bare ``transport_sweep`` no longer
            # re-applies the reflective BC at entry; drive the −B coupling
            # explicitly (reflect the persisted outflow into the inflow slots)
            # before each sweep — the sweep-tier gates' inter-sweep −B (the
            # drivers deliver it as the ``B`` gain; #448).
            reflect_outflow_into_inflow(boundary_flux, sn_mesh)
            _, phi = sweep_once(source, sig_t, sn_mesh, boundary_flux)

        phi_avg = np.average(phi[0, :], weights=mesh.volumes)
        np.testing.assert_allclose(phi_avg, 1.0, rtol=0.01,
                                   err_msg="Volume-avg φ ≠ Q/Σ_t")
        assert phi[0, :].max() < 2.0, (
            f"Flux spike at origin: max={phi[0, :].max():.4f}"
        )

    def _sphere_1g_keff(self, n_cells: int, n_gl: int) -> float:
        materials = {2: get_mixture("A", "1g"), 0: get_mixture("B", "1g")}
        mesh = _two_region_mesh(
            outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(n_cells, n_cells),
            coord=CoordSystem.SPHERICAL,
        )
        return solve_sn(
            materials, mesh, Quadrature.gauss_legendre(n_gl),
            max_inner=500, inner_tol=1e-10,
        ).keff

    def test_heterogeneous_1g_spatial_convergence(self):
        """keff converges under h-refinement (spatial), from n=10.

        #282 route (a): the spatial ladder is measured from n=10 (n∈[10,20,40])
        — the coarser n=5→10 increment is a REAL near-coincidence of this
        two-region config (Δ≈8e-7, not a solver artifact — it survives
        keff_tol=1e-12), so the legacy n∈[5,10,20] `diff_2 < diff_1` check
        tripped on it.  From n=10 the increments shrink monotonically.
        (numerics-investigator 2026-07-04; the pole cell is O(h^~1.4), so the
        global rate is sub-quadratic but genuinely convergent.)"""
        keffs = [self._sphere_1g_keff(n, 8) for n in (10, 20, 40)]
        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1, (
            f"keff not converging from n=10: Δ(20−10)={diff_1:.6e}, "
            f"Δ(40−20)={diff_2:.6e}, keffs={[f'{k:.8f}' for k in keffs]}"
        )

    def test_heterogeneous_1g_angular_order_consistency(self):
        """The route-(a) ψ½ seed is a CONSISTENT angular closure (#282).

        The seed IS an angular closure, so it changes the O(N) truncation
        (the direct Carlson march and the retired edge-extrapolation differ
        by ~1.7e-3 at GL8 but converge to the SAME transport eigenvalue as
        N→∞ — agreeing to ~1e-6 by GL32).  This gate pins that the NEW seed
        converges in angular order N at a fixed mesh; a seed that did NOT
        converge in N would be an INCONSISTENT closure (a genuine
        regression).  This — NOT the MMS (blind: every curvilinear ansatz is
        ≤ linear-in-μ, the seed's EXACT regime, vv Mode 7) — is what
        certifies the seed re-pose is principled.  (numerics-investigator
        2026-07-04; the dd_regression sphere_2g_3reg N=8 snapshot move is the
        matching §16.D re-baseline.)"""
        keffs = [self._sphere_1g_keff(20, n_gl) for n_gl in (8, 16, 32)]
        diff_1 = abs(keffs[1] - keffs[0])
        diff_2 = abs(keffs[2] - keffs[1])
        assert diff_2 < diff_1 and diff_2 < 5e-4, (
            f"keff not converging in angular order N: Δ(16−8)={diff_1:.3e}, "
            f"Δ(32−16)={diff_2:.3e}, keffs={[f'{k:.8f}' for k in keffs]}"
        )


# ═══════════════════════════════════════════════════════════════════════
# SI ≡ Krylov inner-solver equivalence on the curvilinear EIGENVALUE path
# (ERR-026 manifestation #7 — Issue #196)
# ═══════════════════════════════════════════════════════════════════════
#
# Before the ERR-058 closure-seed fix (Issue #195, 2026-06-12) the
# curvilinear sweep converged to a DIFFERENT fixed point than the
# apply-matvec, so the source-iteration inner (which drives the sweep) and
# the Krylov inner (which drives the matvec) produced eigenvalues differing
# at O(h): 0.286 % on sphere_2g_3reg at n=40, ~30 % per-cell on the
# eigenvector shape, the gap halving under refinement.  Logged as ERR-026
# manifestation #7 (Issue #196).
#
# ERR-058 closed the wrong-fixed-point family: BOTH inner solvers now operate
# on the SAME correct discrete operator (the coupled-pole spatial seed
# ψ(0,+μ)=ψ(0,−μ) + the ``AngularEdgeExtrapolation`` half-angle seed), so they
# MUST converge to the same eigenpair — up to the iteration floor, NOT
# bit-identically (they are different iteration schemes, not the same
# arithmetic).  This gate pins that agreement; any re-introduced
# sweep-vs-matvec closure asymmetry (the ERR-026 class) re-opens the O(h) gap
# and trips here with 4+ orders of margin.
#
# Structural-independence discipline (vv-principles L11): SI≡Krylov alone is
# twin-path agreement — NECESSARY, NOT SUFFICIENT (both share the production
# operator and could share a defect).  It is anchored by the homogeneous
# k_inf legs above (``test_homogeneous_exact`` — closed-form infinite-medium
# eigenvalue) and the Variant-α Green's-function cross-check
# (``verification/analytical/test_phase_c_crosscheck.py``), which supply the
# structurally-independent ground.  Here the flux is genuinely NON-FLAT
# (fuel|moderator, 2G) so the angular-redistribution terms are exercised —
# not a homogeneous/1G degenerate (vv-principles anti-patterns #3 / #4).

_SI_KRYLOV_KEFF_TOL = 1e-7    # bug-era |Δk| ~3.9e-3; observed floor ~1.9e-11
_SI_KRYLOV_SHAPE_TOL = 1e-6   # bug-era ~30 %; observed floor ~2.4e-10


def _assert_si_krylov_eigenvalue_equivalence(materials, mesh, quad) -> float:
    """Solve the eigenvalue problem under both inner solvers; assert the
    converged eigenpair agrees to the iteration floor (ERR-026 manifestation
    #7 catcher).

    Both inners solve the identical ``(L+C−S−F)ψ = (1/k)Fψ`` operator on the
    SAME quadrature, so the equivalence holds for any quadrature.  Returns the
    group-0 radial-profile max/min so the caller can assert the flux is
    genuinely non-flat (else the equivalence is vacuous).
    """
    sol_si = solve_sn(
        materials, mesh, quad, inner_solver="source_iteration",
        keff_tol=1e-12, flux_tol=1e-10, max_inner=500, inner_tol=1e-10,
    )
    sol_kry = solve_sn(
        materials, mesh, quad, inner_solver="krylov",
        keff_tol=1e-12, flux_tol=1e-10, max_inner=4000, inner_tol=1e-10,
    )

    k_si, k_kry = sol_si.keff, sol_kry.keff
    assert k_si is not None and k_kry is not None  # eigenvalue solve sets keff
    dk = abs(k_si - k_kry)
    assert dk < _SI_KRYLOV_KEFF_TOL, (
        f"SI keff={k_si:.10f} vs Krylov keff={k_kry:.10f} "
        f"(|Δk|={dk:.2e} ≥ {_SI_KRYLOV_KEFF_TOL:.0e}) — curvilinear "
        f"SI-vs-Krylov eigenvalue asymmetry (ERR-026 manifestation #7)"
    )

    phi_si = np.asarray(sol_si.scalar_flux.values, dtype=np.float64)   # (ng, nx)
    phi_kry = np.asarray(sol_kry.scalar_flux.values, dtype=np.float64)
    # Eigenvectors are scale-free → L∞-normalise each group before comparison.
    for g in range(phi_si.shape[0]):
        a = phi_si[g] / np.max(np.abs(phi_si[g]))
        b = phi_kry[g] / np.max(np.abs(phi_kry[g]))
        shape_diff = float(np.max(np.abs(a - b)))
        assert shape_diff < _SI_KRYLOV_SHAPE_TOL, (
            f"group {g} flux SHAPE max|Δφ_L∞|={shape_diff:.2e} ≥ "
            f"{_SI_KRYLOV_SHAPE_TOL:.0e} — SI vs Krylov eigenvector diverged "
            f"(ERR-026 manifestation #7)"
        )
    prof = phi_si[0]
    return float(prof.max() / prof.min())


@_SPH_VERIFIES
@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.catches("ERR-026")
def test_si_krylov_eigenvalue_equivalence_sphere():
    """Sphere: SI ≡ Krylov converged eigenpair on a heterogeneous 2G problem.

    Measured 2026-06-12 (post-ERR-058): |Δk|=1.9e-11, L∞ flux-shape
    diff=2.4e-10, radial max/min=3.34 (non-flat guard fires).  Bug-era
    (pre-ERR-058, n=40 sphere_2g_3reg): |Δk|~3.9e-3, shape ~30 %.

    ALSO the value catcher for a dropped SI ``initial_guess`` seed
    (M-SEED-DROP, #226 spec §17 F4 — a MUTATION name, deliberately NOT a
    ``catches`` tag: that marker names a catalogued ``ERR-NNN`` defect,
    and the catalogue is now ``.. error-entry::`` nodes in the corpus, so
    a free-form tag there resolves to nothing and warns a ``-W`` build.
    It lived in the marker until 2026-08-17): a simulated seed-drop moves SI's
    eigenvalue by |Δk| ≈ 3.46e-2 while Krylov's is untouched, reddening
    the equivalence 5 orders above ``_SI_KRYLOV_KEFF_TOL``.  Because this
    gate is ``@slow`` (deselected by the canonical ``-m "not slow"`` run),
    the FAST net for the same contract is the Mode-11 path-spy
    ``tests/sn/solve/test_seed_threading_spy.py`` — value catcher here,
    path catcher there.
    """
    materials = {2: get_mixture("A", "2g"), 0: get_mixture("B", "2g")}
    mesh = _two_region_mesh(
        outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
        coord=CoordSystem.SPHERICAL,
    )
    quad = Quadrature.gauss_legendre(8)
    maxmin = _assert_si_krylov_eigenvalue_equivalence(materials, mesh, quad)
    assert maxmin > 1.2, (
        f"sphere flux too flat (group-0 max/min={maxmin:.3f}); redistribution "
        f"not exercised — SI≡Krylov equivalence would be vacuous"
    )


@_CYL_VERIFIES
@pytest.mark.l1
@pytest.mark.slow
@pytest.mark.catches("ERR-026")
def test_si_krylov_eigenvalue_equivalence_cylinder():
    """Cylinder: SI ≡ Krylov converged eigenpair on a heterogeneous 2G problem.

    Measured 2026-06-12 (post-ERR-058): |Δk|=1.1e-11, L∞ flux-shape
    diff=2.6e-11, radial max/min=1.67 (non-flat guard fires).
    """
    materials = {2: get_mixture("A", "2g"), 0: get_mixture("B", "2g")}
    mesh = _two_region_mesh(
        outers=(0.5, 1.0), mat_ids=(2, 0), n_cells=(10, 10),
        coord=CoordSystem.CYLINDRICAL,
    )
    quad = Quadrature.folded_product(n_mu=4, n_phi=8)
    maxmin = _assert_si_krylov_eigenvalue_equivalence(materials, mesh, quad)
    assert maxmin > 1.2, (
        f"cylinder flux too flat (group-0 max/min={maxmin:.3f}); redistribution "
        f"not exercised — SI≡Krylov equivalence would be vacuous"
    )


# ═══════════════════════════════════════════════════════════════════════
# #9 — P1 directional EIGENVALUE in curvilinear (path-(II) Legendre scatter)
# ═══════════════════════════════════════════════════════════════════════
#
# The L0 operator-source gate
# (``tests/sn/verification/mms/test_curvilinear_aniso_scattering_p1.py``)
# proves the curvilinear ℓ=1 scattering source equals its hand-reference
# per ordinate.  These L1 rows close the loop at the EIGENVALUE level:
# they verify the PHYSICAL DIRECTION of the effect — a forward-peaked P1
# (positive mean cosine, ``SigS[1] >= 0``) preserves the forward
# direction of scattered neutrons, which in a finite VACUUM-bounded
# sphere ENHANCES leakage and therefore LOWERS k_eff.
#
# Mirrors ``test_keff_2d.py::test_p1_changes_heterogeneous_keff`` but
# asserts the SIGN + a leakage-monotone mechanism pin, not merely that
# P0 != P1.  Reached through the public ``solve_sn(..., scattering_order)``
# path.  Materials: fuel = ``get_mixture("A","2g")`` (the only fissile 2g
# mixture; asymmetric downscatter-only P0 avoids ERR-002 + the 1G
# eigenvalue degeneracy; SigS[1] forward-peaked), moderator =
# ``get_mixture("C","2g")``.  Vacuum outer BC is LOAD-BEARING because it
# makes P1's k_eff effect a LEAKAGE effect specifically (the mechanism the
# monotone test isolates).  NB the "no leakage ⟹ P1 cannot change k"
# argument holds ONLY for a HOMOGENEOUS reflective sphere (k = k_inf,
# flux-shape independent — that is the monotone test's genuine k_inf
# control, where reflective Δk = 1.5e-12 ≈ 0).  A HETEROGENEOUS reflective
# sphere has a non-flat flux, so P1 still changes k there via spectral /
# spatial coupling (measured reflective Δ = 2.4e-2) — independent of
# leakage; the vacuum BC is what pins the effect to leakage for the SIGN +
# leakage-monotone claims below.


def _sphere_keff(materials, mesh, scattering_order: int) -> float:
    """k_eff of a spherical SN eigenvalue solve at the given Pℓ order."""
    quad = Quadrature.gauss_legendre(8)
    return solve_sn(
        materials, mesh, quad,
        scattering_order=scattering_order,
        max_outer=300, max_inner=500, inner_tol=1e-10, keff_tol=1e-8,
    ).keff


@pytest.mark.verifies("pn-scatter")
@pytest.mark.l1
class TestSphereP1DirectionalEigenvalue:
    """#9 — forward-peaked P1 LOWERS k_eff in a vacuum-bounded sphere."""

    def test_p1_lowers_heterogeneous_keff(self):
        """[L1] HET vacuum sphere: forward-peaked P1 reduces k_eff vs P0.

        Fuel core (r<5) + moderator shell (R=10), vacuum outer.  The 2G
        fissile mixture makes this a genuine eigenvalue claim (NOT 1G
        degenerate, NOT flux-shape independent).  A forward-peaked P1
        enhances leakage at the outer boundary, so ``keff_P1 < keff_P0``;
        the gap is bounded into a physical band (a runaway gap would
        signal a sign-flipped / absorption-mimicking P1).

        Measured (GL8, post-#291 leakage-inclusive reported k —
        principled re-baseline 2026-07-03; the pre-fix functional read
        0.8648/0.8508 by omitting the ~25% leakage fraction):
        keff_P0=0.7060, keff_P1=0.6788, Δ=2.73e-2 — the Δ roughly
        doubles under the corrected functional because the P1
        leakage-enhancement now enters k directly, and stays inside the
        same physical band.
        """
        materials = {0: get_mixture("A", "2g"), 1: get_mixture("C", "2g")}
        mesh = _two_region_mesh(
            outers=(5.0, 10.0), mat_ids=(0, 1), n_cells=(20, 20),
            coord=CoordSystem.SPHERICAL, bc=BC.vacuum,
        )
        keff_p0 = _sphere_keff(materials, mesh, 0)
        keff_p1 = _sphere_keff(materials, mesh, 1)
        delta = keff_p0 - keff_p1

        assert keff_p1 < keff_p0, (
            f"forward-peaked P1 must LOWER k_eff via enhanced leakage, "
            f"got keff_P0={keff_p0:.6f} keff_P1={keff_p1:.6f} "
            f"(Δ={delta:.3e}); a non-negative Δ means the curvilinear "
            f"P1 scattering source has the wrong directional sign."
        )
        assert 1e-3 < delta < 5e-2, (
            f"keff_P0 - keff_P1 = {delta:.3e} is outside the physical band "
            f"(1e-3, 5e-2): below the floor the effect is numerically "
            f"vacuous; above the ceiling the P1 block is mimicking "
            f"absorption rather than redistributing direction."
        )

    @pytest.mark.slow
    def test_p1_leakage_monotone_with_sphere_size(self):
        """[L1] Leakage-monotone mechanism pin: |Δk| grows as the sphere
        shrinks.

        The directional-eigenvalue NEGATIVE CONTROL.  P1's effect on
        k_eff is a LEAKAGE effect, so it must intensify when the
        surface-to-volume ratio rises (smaller sphere).  A homogeneous
        fissile sphere at R=4 must show a LARGER positive
        ``keff_P0 - keff_P1`` than at R=25, and both must stay positive.
        A P1 that mimicked absorption (volumetric, not surface) would
        instead grow with VOLUME — violating this monotonicity.

        Measured (GL8, post-#291 leakage-inclusive reported k —
        principled re-baseline 2026-07-03):
        Δ(R=4)=9.74e-3 > Δ(R=25)=7.93e-3 > 0.  The ordering margin
        NARROWED vs the pre-fix functional (which read 3.75e-3 vs
        2.89e-4): the corrected k carries the leakage term directly at
        BOTH radii, where the old absorption-only k saw just its
        spectral shadow at R=25.  If refinement ever flips this
        ordering, re-examine the physics claim — do NOT widen a
        tolerance (the assertion is an ordering, and the vv rule
        forbids relaxing contracts to fit).
        """
        materials = {0: get_mixture("A", "2g")}
        deltas = {}
        for radius in (4.0, 25.0):
            mesh = _homogeneous_mesh(
                20, radius, mat_id=0,
                coord=CoordSystem.SPHERICAL, bc=BC.vacuum,
            )
            deltas[radius] = (
                _sphere_keff(materials, mesh, 0)
                - _sphere_keff(materials, mesh, 1)
            )

        assert deltas[4.0] > deltas[25.0] > 0, (
            f"P1's k_eff effect must be leakage-monotone in sphere size: "
            f"Δ(R=4)={deltas[4.0]:.3e} > Δ(R=25)={deltas[25.0]:.3e} > 0 "
            f"required.  A non-monotone or sign-flipped ordering means the "
            f"curvilinear P1 source is not behaving as a surface-leakage "
            f"redistribution (it may be mimicking volumetric absorption)."
        )


# ═══════════════════════════════════════════════════════════════════════
# #10 — the FOLDED (quotient) rule binds the σ-EVEN sub-basis
# ═══════════════════════════════════════════════════════════════════════
#
# The cylinder P1 sibling of #9.  #9 reaches the Pℓ path on a SPHERE
# through ``Quadrature.gauss_legendre(8)`` — an UNFOLDED rule, which
# binds the plain ``SphericalHarmonicBasis`` and is therefore blind to
# everything below.  This section is the folded-cylinder half, and it
# exists because ``Quadrature._harmonic_basis`` makes a CHOICE that
# nothing at the solve tier was asserting (tracker 2.1-W of
# ``.claude/plans/angular_spaces_derived_from_symmetry.md``, #429).
#
# THE CHOICE.  ``folded_product`` is the σ_y-QUOTIENT of the staggered
# product rule (Q5.6): every node carries ξ = μ_y > 0, and the fold's
# defining law (:eq:`discrete-measure-quotient`) — ∫f d(μ/G) = ∫f dμ —
# holds only for G-INVARIANT f.  The σ_y-ODD harmonics are not
# G-invariant, so their discrete moments on the quotient are GARBAGE,
# not zero: `[M]` on ``folded_product(4, 8)`` a FLAT flux analyses to
# +6.486547 in the ξ-carrying l = 1 slot [1, 2], where the unfolded rule
# cancels to 1e-16, and the scattering kernel's raw YᵀW analysis has no
# Gram division anywhere to absorb it.  ``_harmonic_basis`` therefore
# binds ``MirrorEvenSphericalHarmonicBasis`` — the σ-even sub-basis,
# which structurally zeroes the odd columns.
#
# ⛔ WHY NO P0 GATE CAN SEE THIS.  `[M]` at L = 0 the σ-even table is
# BIT-IDENTICAL to the parent's (max|ΔY| = 0.000000e+00); it first
# diverges at L = 1 (max|ΔY| = 8.688461e-01 on ``folded_product(4, 8)``).
# Every folded eigenvalue row in this module above runs at the default
# ``scattering_order=0`` and is structurally blind to the binding.  So
# is every reciprocity/Riesz row in
# ``tests/sn/architecture/test_monomorphic_leaves.py`` that DOES reach
# a folded rule at L = 1: ⟨Ax,y⟩ = ⟨x,A*y⟩ holds for any CONSISTENT
# (M, R) pair because both sides read the SAME table — the
# ``vv-principles`` two-sides-from-one-source tautology.  Those rows
# EXERCISE the fold basis; they cannot ASSERT it.
#
# ⛔ WHAT THE TREE ALREADY HAD, measured rather than assumed.  2.1-W was
# written against `[M]` *"deleting MirrorEvenSphericalHarmonicBasis reds
# 0 of 1913"*, i.e. "no witnesses anywhere".  That is FALSE, twice over,
# and the correction is recorded here so nobody re-derives it.
#
# (a) The DELETION is not an in-class mutation and cannot produce a red:
#     ``directional.py:83`` imports the class at MODULE SCOPE and
#     ``tests/sn/primitives/conftest.py:7`` reaches it transitively
#     (conftest → tests.sn._test_helpers → orpheus.transport →
#     orpheus.numerics → …quadrature → .directional).  `[M]` scoped to
#     that directory, pytest exits rc=4 having collected NOTHING — zero
#     ``^FAILED`` lines and zero ``^ERROR`` lines, so a scanner counting
#     either reads "0 caught".  It measured the import graph.
# (b) Under the REBIND (in-class, ``vv-principles`` #18) `[M]` 7 of 1827
#     tests red over the 80-file population that can reach a folded rule
#     (``-m "not slow"``; control 1827 passed / 0 failed / 0 error).
#     FOUR of those seven are PRE-EXISTING:
#       tests/sn/primitives/test_quadrature_fold.py::TestFoldedHarmonics
#         ::test_flat_moments_are_the_isotropic_moment_alone
#         ::test_the_folded_frame_analysis_is_isotropic_on_a_flat_flux
#       tests/numerics/test_frame.py
#         ::test_parseval_frame_square_closes[folded8x8-L2]
#         ::test_parseval_dressing_installed_on_diagonal_frames[folded8x8-L2]
#
# ⟹ the honest statement of the gap this section closes is NOT "the fold
# basis has no witness".  It is that all four pre-existing catchers are
# OBJECT-tier gates on the basis and its frame — a flat-flux moment
# table, a Parseval collapse — and NONE of them is reached through
# ``solve_sn``.  The eigenvalue tier, where a user meets this choice, had
# no witness at all, and that is what the two rows below supply.
#
# THE REFERENCE IS STRUCTURALLY INDEPENDENT.  In an infinite homogeneous
# medium the transport solution is spatially flat and angularly
# isotropic, so φ_ℓ ≡ 0 for every ℓ ≥ 1 and the anisotropic scattering
# source contributes NOTHING: k = k_inf, whatever the Pℓ truncation.
# ``get("sn_slab_2eg_1rg").k_inf`` is that closed form — a 2-group
# transfer-matrix eigenvalue with no SN solver, no quadrature and no
# harmonic basis anywhere in its chain (the closed-form pillar, which
# is the only pillar that may carry an EIGENVALUE claim; MMS may not).
# ≥2 groups, so the 1-group degeneracy does not apply.
#
# ⭐ MEASURED MUTATION TABLE (2026-08-31, in-process monkeypatch of
# ``Quadrature._harmonic_basis``, ``python -O``).  Columns are the two
# rows below; "keystone" is |k(P1) − k_inf| on the homogeneous
# reflective cylinder, "companion" is k(P0) − k(P1) on the
# heterogeneous vacuum cylinder.
#
#   arm                                        keystone 4x8 / 8x4      companion Δ
#   M0  none (control)                          1.4699e-11 / 1.5106e-11   2.4873e-02   green / green
#   M1  rebind → SphericalHarmonicBasis(L)      4.3231e-02 / 4.9992e-02   5.8161e-01   RED   / RED
#   M3  mirror_axis 1 → 0                       4.3231e-02 / 4.9992e-02   6.0566e-01   RED   / RED
#   M4  mirror_axis 1 → 2                       4.3231e-02 / 4.9992e-02   5.8161e-01   RED   / RED
#   M5  over-mask the RADIAL even slot [1, 1]   1.4788e-11 / 1.4055e-11   0.0000e+00   green / RED
#   M6  over-mask the AXIAL even slot [1, 0]    1.4699e-11 / 1.5106e-11   2.4873e-02   green / green
#
# M1 is the defining mutation (the silent rebind 2.1-W exists to make
# falsifiable) and it is deliberately IN-CLASS: the parent is a
# perfectly legal ``SphericalHarmonicBasis``, so nothing structural
# breaks and only the property under test moves (``vv-principles`` #18).
#
# ⭐ M5 is why the companion row exists and is not decoration: dropping
# a genuine EVEN l = 1 basis function (the ERR-072 declared-not-computed
# family — a hand-listed mask with one slot wrong) is invisible to the
# keystone, because a FLAT flux has no radial current for it to lose.
# The companion is its only catcher in this module.
#
# ⛔ DECLARED BLINDNESS (M6).  Slot [1, 0] carries μ_z (`[M]`
# corr(Y[:,1,0], μ_z) = +1.000), and a 1-D cylinder is symmetric under
# μ_z → −μ_z, so the axial current is identically zero and dropping
# that basis function moves NOTHING — on either row, at any refinement.
# This is a theorem about the 1-D chart, not a gap in these rows: the
# μ_z-carrying l = 1 slot has no witness reachable on any 1-D geometry,
# and a 2-D/3-D fixture is the only place one could live.
#
# ⚠ HARNESS NOTE for anyone re-running the battery: ``Quadrature``
# memoises its frames in ``_angular_frames[L]``, so a ``Quadrature``
# instance that was already solved UNMUTATED returns the cached honest
# frame and the mutation reads bit-identical (`[M]` 0.9726641733732218
# both ways — a false "no teeth").  Build the rule AFTER installing the
# mutation, or install it at ``pytest_configure`` as the battery does.
#
# ERR-080 does not apply here: ``folded_product`` is a genuine 2-D rule
# whose nodes carry real μ_y/μ_z, not a 1-D rule faking azimuth 0.


def _folded_cyl_keff(materials, mesh, quad, scattering_order: int) -> float:
    """k_eff of a cylindrical SN solve on a FOLDED (quotient) rule.

    The ``None`` arm is an explicit ``raise``, not an ``assert``: this
    helper is module-level support code and the canonical runner is
    ``python -O`` (``vv-principles`` Mode 8).
    """
    result = solve_sn(
        materials, mesh, quad,
        scattering_order=scattering_order,
        max_outer=300, max_inner=500, inner_tol=1e-10, keff_tol=1e-8,
    )
    if result.keff is None:
        raise AssertionError(
            f"the folded cylindrical solve at scattering_order="
            f"{scattering_order} returned no eigenvalue at all "
            f"(Solution.keff is None) — there is nothing for the "
            f"quotient-basis claim below to be asserted against"
        )
    return result.keff


@pytest.mark.verifies("pn-scatter", "discrete-measure-quotient")
@pytest.mark.l1
class TestFoldedCylinderP1BindsTheQuotientBasis:
    """#10 — a folded rule's Pℓ path must use the σ-EVEN sub-basis."""

    @pytest.mark.parametrize("quad_factory", [
        lambda: Quadrature.folded_product(n_mu=4, n_phi=8),
        lambda: Quadrature.folded_product(n_mu=8, n_phi=4),
    ], ids=["folded_4x8", "folded_8x4"])
    def test_kinf_is_pl_order_invariant_on_the_quotient(self, quad_factory):
        """[L1] KEYSTONE. Homogeneous reflective cylinder, 2G, folded
        rule: k(P1) = k(P0) = k_inf, the closed form.

        WHAT IT PINS.  On the quotient measure the σ_y-odd harmonics are
        outside the function space; the fold binds the σ-even sub-basis
        so their moments come out EXACT 0.0.  An infinite medium has
        φ_ℓ ≡ 0 for ℓ ≥ 1 anyway, so a correct Pℓ path adds nothing and
        k stays at the analytical k_inf at every truncation order.  Bind
        the PARENT basis instead and the flat flux analyses to +6.49 in
        the ξ-carrying slot, which reconstructs straight into the P1
        source and drags k off the closed form.

        WHAT REDDENS IT — `[M]` 2026-08-31, in-process monkeypatch of
        ``Quadrature._harmonic_basis``, ``python -O``:

        * honest        |k(P1) − k_inf| = 1.4699e-11 (4x8) / 1.5106e-11 (8x4)
        * ``return SphericalHarmonicBasis(L=L)``  → 4.3231e-02 / 4.9992e-02
        * ``mirror_axis=0``                       → 4.3231e-02 / 4.9992e-02
        * ``mirror_axis=2``                       → 4.3231e-02 / 4.9992e-02

        i.e. nine orders of separation against a 1e-6 gate.

        WHAT IT CANNOT SEE, and why the companion row below exists:
        over-masking a genuine EVEN l = 1 slot leaves this row at
        1.4788e-11 — a flat flux has no ℓ = 1 content to lose.  The P0
        leg is likewise a provable non-catcher for the rebind (`[M]` the
        two tables are bit-identical at L = 0); it is kept because it
        pins the OTHER direction — that the σ-even restriction does not
        over-mask the isotropic sector.

        The reference is ``derivations``' 2-group transfer-matrix
        eigenvalue: no solver, no quadrature, no basis in its chain.
        """
        case = get("sn_slab_2eg_1rg")
        mix = next(iter(case.materials.values()))

        # ACTIVATION LEG — the ℓ=1 channel must be live, or the whole
        # row is vacuous: a zero SigS[1] multiplies the garbage moment
        # by zero and no basis error could reach k.
        sig_s1 = mix.SigS[1]
        sig_s1 = np.asarray(
            sig_s1.todense() if hasattr(sig_s1, "todense") else sig_s1
        )
        assert np.abs(sig_s1).max() > 1e-3, (
            f"vacuous fixture: SigS[1] max |.| = {np.abs(sig_s1).max():.3e}; "
            f"with no anisotropic scattering the Pl channel is switched "
            f"off and this row cannot see the harmonic basis at all"
        )

        mesh = _homogeneous_mesh(
            20, 2.0, mat_id=0, coord=CoordSystem.CYLINDRICAL
        )
        keff_p0 = _folded_cyl_keff({0: mix}, mesh, quad_factory(), 0)
        keff_p1 = _folded_cyl_keff({0: mix}, mesh, quad_factory(), 1)

        assert abs(keff_p1 - case.k_inf) < 1e-6, (
            f"P1 on a folded (quotient) rule left the analytical "
            f"infinite-medium eigenvalue: keff_P1={keff_p1:.12f} vs "
            f"k_inf={case.k_inf:.12f} (|Δ|={abs(keff_p1 - case.k_inf):.3e}). "
            f"An infinite medium has phi_l = 0 for every l >= 1, so the "
            f"Pl source MUST be inert here — a non-zero l=1 moment on "
            f"this rule means the harmonic machinery is analysing the "
            f"quotient in the FULL spherical-harmonic basis instead of "
            f"its sigma-even sub-basis (the sigma-odd moments are "
            f"garbage on a folded measure, not zero)."
        )
        assert abs(keff_p0 - case.k_inf) < 1e-6, (
            f"P0 baseline broke: keff_P0={keff_p0:.12f} vs "
            f"k_inf={case.k_inf:.12f}; the sigma-even restriction must "
            f"leave the isotropic sector untouched (it masks only "
            f"sigma-ODD slots), so this leg failing means the mask is "
            f"over-masking, not that the Pl path is wrong."
        )
        assert abs(keff_p1 - keff_p0) < 1e-9, (
            f"the l>=1 channel is not inert in an infinite medium: "
            f"keff_P0={keff_p0:.12f} keff_P1={keff_p1:.12f} "
            f"(|Δ|={abs(keff_p1 - keff_p0):.3e}).  This is the sharper "
            f"form of the claim above — the Pl truncation order must not "
            f"move k when the flux is flat and isotropic."
        )

    def test_p1_leakage_shift_survives_on_a_heterogeneous_quotient_solve(self):
        """[L1] COMPANION — the non-flat-flux row, and the only catcher
        of an over-masked EVEN slot.

        Fuel core (r < 5) + moderator shell (R = 10), VACUUM outer,
        ``folded_product(4, 8)``, 2G.  The flux is non-flat, so the
        radial current φ_1 is genuinely non-zero and the ℓ = 1 channel
        carries real signal — which the keystone's infinite medium, by
        construction, cannot.

        A forward-peaked P1 (``SigS[1] >= 0``) enhances leakage through
        the vacuum boundary, so ``keff_P1 < keff_P0`` and the gap sits in
        a physical band.  `[M]` 2026-08-31, ``python -O``:

        * honest: keff_P0 = 0.9975374278381011,
          keff_P1 = 0.9726641733732218, Δ = 2.4873e-02
        * rebind to the parent SH basis  → Δ = 5.8161e-01  (RED, band)
        * ``mirror_axis`` 1 → 0          → Δ = 6.0566e-01  (RED, band)
        * over-mask the RADIAL even slot [1, 1] → Δ = 0.0000e+00
          (P1 collapses exactly onto P0 — the radial current has
          nowhere to live).  ⭐ The keystone reads 1.4788e-11 under that
          same mutation, i.e. GREEN: this row is its sole catcher here.

        ⭐ WHICH LEG CATCHES WHAT — measured per arm, not argued:

        * the SIGN leg is the ONLY catcher of the over-masked radial
          slot (`[M]` it fires on ``0.9975374278381011 <
          0.9975374278381011``, and the band leg never runs);
        * the SIGN leg is a provable NON-CATCHER for the rebind — the
          mutated keff_P1 = 0.4159 is still *below* keff_P0, so ``<``
          passes and only the BAND fires (`[M]` ``0.5816 < 0.1`` False).

        So neither leg is redundant and neither covers the other.  Do
        not read the sign assertion as coverage of the basis binding;
        on its own it covers the DIRECTION of the P1 effect (the #9
        claim, transplanted to the cylinder).
        """
        materials = {0: get_mixture("A", "2g"), 1: get_mixture("C", "2g")}
        mesh = _two_region_mesh(
            outers=(5.0, 10.0), mat_ids=(0, 1), n_cells=(20, 20),
            coord=CoordSystem.CYLINDRICAL, bc=BC.vacuum,
        )
        quad = Quadrature.folded_product(n_mu=4, n_phi=8)
        keff_p0 = _folded_cyl_keff(materials, mesh, quad, 0)
        keff_p1 = _folded_cyl_keff(materials, mesh, quad, 1)
        delta = keff_p0 - keff_p1

        assert keff_p1 < keff_p0, (
            f"forward-peaked P1 must LOWER k_eff via enhanced leakage on "
            f"a vacuum-bounded cylinder, got keff_P0={keff_p0:.8f} "
            f"keff_P1={keff_p1:.8f} (Δ={delta:.3e}).  Δ == 0 exactly "
            f"means the l=1 radial slot was masked away."
        )
        assert 1e-3 < delta < 1e-1, (
            f"keff_P0 - keff_P1 = {delta:.3e} is outside the physical "
            f"band (1e-3, 1e-1): below the floor the l=1 channel is "
            f"switched off (an over-masked EVEN slot); above the ceiling "
            f"the folded rule's Pl source is being fed sigma-ODD moments "
            f"that are garbage on the quotient, not the sigma-even "
            f"sub-basis the fold binds."
        )
