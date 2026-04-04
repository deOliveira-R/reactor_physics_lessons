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
