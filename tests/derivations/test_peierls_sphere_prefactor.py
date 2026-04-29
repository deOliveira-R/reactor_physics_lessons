"""L0 prefactor and row-sum identity tests for the spherical Peierls kernel.

Mirror of ``test_peierls_cylinder_prefactor.py``. The row-sum identity
:math:`\\sum_j K_{ij}\\,\\Sigma_t(r_j) = \\Sigma_t(r_i)` isolates:

- the :math:`1/2` prefactor (:math:`1/(4\\pi)` Green's function
  normalisation times the :math:`2\\pi` azimuthal fold),
- the :math:`\\sin\\theta` angular measure (the sphere's polar-angle
  weight, distinct from the cylinder's constant-:math:`d\\beta`),
- the :math:`e^{-\\tau}` kernel choice (no dimensional reduction since
  the 3-D point kernel is already 3-D — unlike the cylinder's
  :math:`\\mathrm{Ki}_1` that comes from z-integration),
- the polar :math:`(θ, ρ)` Nyström assembly and the directed ray
  walker.

Failure of any one of these amplifies the row-sum deficit above the
escape-probability floor, surfacing the bug long before an eigenvalue
test could catch it."""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    build_volume_kernel,
    build_white_bc_correction,
    composite_gl_r,
)
from orpheus.derivations.peierls_sphere import GEOMETRY


# ═══════════════════════════════════════════════════════════════════════
# Row-sum identity — vacuum (infinite-medium limit)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereRowSumIdentity:
    """K·Σ_t = Σ_t in the infinite-medium limit; finite-R deficit is
    controlled by the escape probability."""

    @staticmethod
    def _build_thick_sphere(n_theta: int, n_rho: int, dps: int = 20):
        """Reference configuration: R = 10, Σ_t = 1, ~12 radial nodes."""
        R = 10.0
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=5, dps=dps,
        )
        K = build_volume_kernel(
            GEOMETRY, r_nodes, panels, radii, sig_t,
            n_angular=n_theta, n_rho=n_rho, dps=dps,
        )
        return r_nodes, r_wts, K, R

    def test_interior_row_sum_equals_sigma_t(self):
        """For :math:`r_i \\le R/2` (≥ 5 MFP from the boundary), the
        infinite-medium row sum holds to better than :math:`10^{-3}`."""
        r_nodes, _, K, R = self._build_thick_sphere(n_theta=24, n_rho=24)
        sig_t = 1.0
        sig_t_vec = np.full_like(r_nodes, sig_t)
        row_action = K @ sig_t_vec
        interior = r_nodes <= 0.5 * R
        max_deficit = np.abs(row_action[interior] - sig_t).max()
        assert max_deficit < 1e-3, (
            f"Sphere row-sum identity violated by {max_deficit:.3e} in the "
            f"interior; expected < 1e-3 for a 10-MFP sphere. "
            f"row_action[interior] = {row_action[interior]}"
        )

    def test_deficit_grows_toward_boundary(self):
        """The row-sum deficit :math:`\\Sigma_t - (K\\Sigma_t)_i` is
        a monotone-increasing function of :math:`r_i` (escape
        probability grows toward the surface)."""
        r_nodes, _, K, R = self._build_thick_sphere(n_theta=24, n_rho=24)
        sig_t_vec = np.ones_like(r_nodes)
        deficit = 1.0 - (K @ sig_t_vec)
        # Exclude the 1 MFP near the wall where quadrature noise dominates.
        mask = r_nodes <= R - 1.0
        order = np.argsort(r_nodes[mask])
        diffs = np.diff(deficit[mask][order])
        assert np.all(diffs > -1e-5), (
            f"Sphere escape deficit not monotone in r_i: diffs = {diffs}"
        )

    def test_convergence_under_quadrature_refinement(self):
        """Row-sum deficit at an interior probe shrinks as (n_θ, n_ρ)
        increase."""
        R = 10.0
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, _, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=5, dps=20,
        )
        probe = int(np.argmin(np.abs(r_nodes - 2.0)))
        sig_t_vec = np.full_like(r_nodes, sig_t[0])
        deficits = []
        for n_q in (12, 20, 28):
            K = build_volume_kernel(
                GEOMETRY, r_nodes, panels, radii, sig_t,
                n_angular=n_q, n_rho=n_q, dps=20,
            )
            deficits.append(abs(sig_t[0] - (K @ sig_t_vec)[probe]))
        assert deficits[-1] < deficits[0], (
            f"Quadrature refinement did not reduce the sphere deficit: "
            f"{deficits}"
        )
        assert deficits[-1] < 1e-3, (
            f"Finest-quadrature deficit too large on the sphere: "
            f"{deficits[-1]:.3e} at r ≈ {r_nodes[probe]:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Row-sum identity under white-BC rank-1 correction
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereWhiteBCRowSum:
    """With the rank-1 white-BC correction, (K_vol + K_bc)·Σ_t ≈ Σ_t
    with a residual that shrinks toward the thick limit — the same
    phenomenon documented for the cylinder (rank-1 Mark closure is
    correct only in the flat-source / thick-cell limit)."""

    @staticmethod
    def _build_homogeneous_sphere(R: float, n_theta: int, n_rho: int, n_phi: int):
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=5, dps=25,
        )
        K_vol = build_volume_kernel(
            GEOMETRY, r_nodes, panels, radii, sig_t,
            n_angular=n_theta, n_rho=n_rho, dps=25,
        )
        K_bc = build_white_bc_correction(
            GEOMETRY, r_nodes, r_wts, radii, sig_t,
            n_angular=n_theta, n_surf_quad=n_phi, dps=25,
        )
        return r_nodes, sig_t[0], K_vol + K_bc

    def test_thick_sphere_residual_below_two_percent(self):
        """R = 10 MFP: max residual below 2 % (the rank-1 closure is
        near-exact in the thick limit)."""
        r_nodes, sig_t, K_tot = self._build_homogeneous_sphere(
            R=10.0, n_theta=20, n_rho=20, n_phi=32,
        )
        sig_t_vec = np.full_like(r_nodes, sig_t)
        residual = np.abs(K_tot @ sig_t_vec - sig_t_vec).max()
        assert residual < 2e-2, (
            f"R=10 MFP rank-1 white-BC residual too large: {residual:.3e}"
        )

    def test_medium_sphere_residual_below_five_percent(self):
        """R = 5 MFP: residual still reasonable (~1 %), mirroring the
        cylinder's rank-1 behaviour at the same optical size."""
        r_nodes, sig_t, K_tot = self._build_homogeneous_sphere(
            R=5.0, n_theta=20, n_rho=20, n_phi=32,
        )
        sig_t_vec = np.full_like(r_nodes, sig_t)
        residual = np.abs(K_tot @ sig_t_vec - sig_t_vec).max()
        assert residual < 5e-2, (
            f"R=5 MFP rank-1 white-BC residual too large: {residual:.3e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# G_bc vacuum-limit sanity check
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereGBCVacuumLimit:
    """In the Σ_t → 0 limit, the sphere's surface-to-volume Green's
    function tends to the geometric value 4 everywhere: a uniform
    isotropic inward partial current J⁻ fills the interior with
    scalar flux φ = 4 J⁻ (4π sr times ψ_in = J⁻/π equals 4)."""

    def test_vacuum_G_bc_is_four(self):
        from orpheus.derivations.peierls_geometry import compute_G_bc

        radii = np.array([1.0])
        # Σ_t·R = 1e-8: effectively vacuum at double precision.
        sig_t = np.array([1e-8])
        r_nodes = np.array([0.0, 0.25, 0.5, 0.75, 0.99])
        G_bc = compute_G_bc(GEOMETRY, r_nodes, radii, sig_t, n_surf_quad=32, dps=20)
        np.testing.assert_allclose(G_bc, 4.0, rtol=1e-5, atol=1e-5)
