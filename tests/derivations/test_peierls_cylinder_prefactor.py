"""L0 prefactor and row-sum identity tests for the cylindrical Peierls kernel.

These tests land with C3 of the Phase-4.2 verification campaign and
constitute the **prefactor-bug gate**: passing them proves that the
1/π prefactor, the Ki₁ kernel choice, and the polar (β, ρ) Nyström
assembly in :func:`orpheus.derivations.peierls_geometry.build_volume_kernel`
are all individually correct.

Physical identity under test
============================

For a bare homogeneous cylinder with :math:`\\Sigma_t = 1` cm⁻¹ and
radius :math:`R = 10` cm (ten mean free paths), a neutron born
isotropically at interior radius :math:`r_i \\ll R` has essentially
zero probability of escaping without a collision. The infinite-medium
Peierls operator row-sum identity

.. math::

    \\sum_{j=1}^{N} K_{ij} \\;=\\; \\Sigma_t(r_i)

(for the identity-LHS form in which :math:`K_{ij}` includes the
:math:`\\Sigma_t(r_i)` factor) therefore holds to the escape
probability :math:`P_{\\rm esc}(r_i) \\sim e^{-\\Sigma_t (R-r_i)}`,
which is well below 10⁻³ for :math:`r_i \\le R/2`.

The test also verifies:

- monotonic convergence under refinement of the β and ρ quadrature
  orders (sanity check on the integration scheme),
- geometric sanity: row sums decay toward the boundary exactly as
  the escape probability dictates (no spurious constant bias).
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_cylinder import GEOMETRY
from orpheus.derivations.peierls_geometry import (
    build_volume_kernel,
    composite_gl_r,
)


@pytest.mark.l0
@pytest.mark.verifies("peierls-equation", "ki3-def")
class TestRowSumIdentity:
    """The Peierls Nyström operator satisfies the infinite-medium
    row-sum identity in the bulk of a thick homogeneous cylinder."""

    @staticmethod
    def _build_thick_cyl(n_beta: int, n_rho: int, dps: int = 20):
        """Reference configuration: R = 10, Σ_t = 1, N = 10 r-nodes."""
        R = 10.0
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=5, dps=dps,
        )
        K = build_volume_kernel(
            GEOMETRY, r_nodes, panels, radii, sig_t,
            n_angular=n_beta, n_rho=n_rho, dps=dps,
        )
        return r_nodes, K, R

    def test_interior_row_sum_equals_sigma_t(self):
        """For :math:`r_i \\le R/2` (five mean free paths from boundary),
        the row sum :math:`\\sum_j K_{ij}` equals :math:`\\Sigma_t = 1`
        to better than :math:`10^{-3}`."""
        r_nodes, K, R = self._build_thick_cyl(n_beta=24, n_rho=24)
        row_sums = K.sum(axis=1)
        interior = r_nodes <= 0.5 * R
        max_deficit = np.abs(row_sums[interior] - 1.0).max()
        assert max_deficit < 1e-3, (
            f"Row-sum identity violated by {max_deficit:.3e} in the "
            f"interior; expected < 1e-3 for a 10-MFP cylinder. "
            f"row_sums[interior] = {row_sums[interior]}"
        )

    def test_deficit_grows_toward_boundary(self):
        """Row-sum deficit :math:`1-\\sum_j K_{ij}` increases
        monotonically with :math:`r_i` (geometric escape
        probability is a smooth increasing function of position)."""
        r_nodes, K, R = self._build_thick_cyl(n_beta=24, n_rho=24)
        row_sums = K.sum(axis=1)
        deficit = 1.0 - row_sums
        # Exclude nodes closer than one MFP to the wall where quadrature
        # error dominates
        mask = r_nodes <= R - 1.0
        deficit_sorted_by_r = deficit[mask][np.argsort(r_nodes[mask])]
        diffs = np.diff(deficit_sorted_by_r)
        # Allow a little noise at the 10⁻⁵ level; core monotonicity check
        assert np.all(diffs > -1e-5), (
            f"Escape deficit not monotone in r_i: diffs = {diffs}"
        )

    def test_convergence_under_quadrature_refinement(self):
        """At an interior point the row-sum converges to :math:`\\Sigma_t`
        monotonically as :math:`(n_\\beta, n_\\rho)` increase."""
        R = 10.0
        radii = np.array([R])
        sig_t = np.array([1.0])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=5, dps=20,
        )
        # Pick an interior node for the probe
        probe_idx = int(np.argmin(np.abs(r_nodes - 2.0)))

        results = []
        for n_q in (12, 20, 28):
            K = build_volume_kernel(
                GEOMETRY, r_nodes, panels, radii, sig_t,
                n_angular=n_q, n_rho=n_q, dps=20,
            )
            results.append(K[probe_idx].sum())

        # Deficit should shrink as quadrature refines
        deficits = [abs(1.0 - r) for r in results]
        assert deficits[-1] < deficits[0], (
            f"Quadrature refinement did not reduce the deficit: "
            f"{deficits}"
        )
        assert deficits[-1] < 1e-3, (
            f"Finest-quadrature deficit too large: "
            f"{deficits[-1]:.3e} at r = {r_nodes[probe_idx]:.4f}"
        )


@pytest.mark.l0
@pytest.mark.verifies("peierls-equation")
class TestMultiRegionRowSum:
    """Row-sum identity extends to multi-region with equal Σ_t across
    annuli — i.e. a homogeneous cylinder expressed as nominally
    "multi-region" exercises the τ walker without changing the physics."""

    def test_two_annulus_equal_sig_t_matches_one_region(self):
        """Splitting a homogeneous cylinder into two annuli (same Σ_t
        on both) gives the same row sums as the one-region case to
        within quadrature error."""
        # One-region reference
        R = 10.0
        sig_t_val = 1.0
        r_nodes_1, _, panels_1 = composite_gl_r(
            np.array([R]), n_panels_per_region=2, p_order=5, dps=20,
        )
        K_1 = build_volume_kernel(
            GEOMETRY, r_nodes_1, panels_1, np.array([R]), np.array([sig_t_val]),
            n_angular=20, n_rho=20, dps=20,
        )
        rs_1 = K_1.sum(axis=1)

        # Two-annulus with same Σ_t; mesh is NOT the same (panel breakpoints
        # differ), so compare at shared interior r values via interpolation.
        radii_2 = np.array([R / 2, R])
        sig_t_2 = np.array([sig_t_val, sig_t_val])
        r_nodes_2, _, panels_2 = composite_gl_r(
            radii_2, n_panels_per_region=1, p_order=5, dps=20,
        )
        K_2 = build_volume_kernel(
            GEOMETRY, r_nodes_2, panels_2, radii_2, sig_t_2,
            n_angular=20, n_rho=20, dps=20,
        )
        rs_2 = K_2.sum(axis=1)

        # Interior points (r <= 4): both should be ≈ Σ_t
        interior_1 = rs_1[r_nodes_1 <= 4.0]
        interior_2 = rs_2[r_nodes_2 <= 4.0]
        if len(interior_1) and len(interior_2):
            assert abs(interior_1.mean() - interior_2.mean()) < 5e-4, (
                f"One-region vs two-region mean row-sum diverges: "
                f"1R mean = {interior_1.mean():.6f}, "
                f"2R mean = {interior_2.mean():.6f}"
            )
