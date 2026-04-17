"""L0 geometry tests for the cylindrical Peierls walker and y-quadrature.

These tests land with the C2 scaffolding commit of the Phase-4.2
verification campaign. They exercise the two pieces of pure geometry
in :mod:`orpheus.derivations.peierls_cylinder` that have no dependence
on the (not-yet-built) Nyström kernel builder:

1. :func:`composite_gl_y` — composite Gauss–Legendre quadrature on
   :math:`[0, R]` with breakpoints at each annular radius. Tested
   against trivial closed-form integrals.

2. :func:`optical_depths_pm` — the :math:`\\tau^{+}(r,r',y)` /
   :math:`\\tau^{-}(r,r',y)` optical-path walker. Tested against
   homogeneous-medium closed forms, two-annulus diameter transits,
   and the degenerate :math:`r = r'` case.

These are the sanity checks that must pass before the Nyström kernel
builder (C3) is trusted to assemble meaningful matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_cylinder import (
    composite_gl_y,
    optical_depths_pm,
)


# ═══════════════════════════════════════════════════════════════════════
# Composite GL y-quadrature
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("chord-length")
class TestCompositeGLY:
    """Sanity checks for the y-quadrature with breakpoints at r_k."""

    def test_integrates_constant_to_R(self):
        """∫₀ᴿ 1 dy = R (machine precision)."""
        radii = np.array([0.4, 1.0])
        y_pts, y_wts, _ = composite_gl_y(radii, n_panels_per_region=4, p_order=6)
        assert np.isclose(y_wts.sum(), radii[-1], rtol=1e-14)

    def test_integrates_linear_to_half_Rsq(self):
        """∫₀ᴿ y dy = R²/2 (machine precision, GL order ≥ 2 exact)."""
        radii = np.array([0.5, 1.0, 2.0])
        y_pts, y_wts, _ = composite_gl_y(radii, n_panels_per_region=3, p_order=4)
        R = radii[-1]
        integral = np.dot(y_wts, y_pts)
        assert np.isclose(integral, 0.5 * R ** 2, rtol=1e-14)

    def test_panel_bounds_partition_y_axis(self):
        """Concatenation of panel bounds covers [0, R] without gaps."""
        radii = np.array([0.4, 0.9, 1.5])
        _, _, panel_bounds = composite_gl_y(
            radii, n_panels_per_region=2, p_order=4,
        )
        # First panel starts at 0; last ends at R; consecutive panels abut.
        assert panel_bounds[0][0] == pytest.approx(0.0, abs=1e-14)
        assert panel_bounds[-1][1] == pytest.approx(radii[-1], rel=1e-14)
        for p0, p1 in zip(panel_bounds[:-1], panel_bounds[1:]):
            assert p0[1] == pytest.approx(p1[0], rel=1e-14)

    def test_breakpoints_hit_each_annular_radius(self):
        """Each annular radius r_k appears as a panel endpoint."""
        radii = np.array([0.3, 0.7, 1.2])
        _, _, panel_bounds = composite_gl_y(
            radii, n_panels_per_region=2, p_order=4,
        )
        endpoints = sorted({pa for pa, _, _, _ in panel_bounds}
                           | {pb for _, pb, _, _ in panel_bounds})
        for rk in radii:
            assert any(abs(ep - rk) < 1e-13 for ep in endpoints), (
                f"Radius {rk} is not a panel endpoint; endpoints = {endpoints}"
            )


# ═══════════════════════════════════════════════════════════════════════
# τ⁺ / τ⁻ optical-path walker
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("chord-length")
class TestOpticalDepthsPM:
    """Closed-form verification of the τ± walker."""

    def test_homogeneous_1region_closed_form(self):
        r"""For a homogeneous 1-region cylinder with Σ_t = σ and radii R,

        .. math::

           \tau^{+}(r, r', y) &= \sigma \cdot \bigl|\sqrt{r^{2}-y^{2}}
                                              - \sqrt{r'^{2}-y^{2}}\bigr| \\
           \tau^{-}(r, r', y) &= \sigma \cdot \bigl(\sqrt{r^{2}-y^{2}}
                                              + \sqrt{r'^{2}-y^{2}}\bigr)
        """
        radii = np.array([1.0])
        sig_t = np.array([2.3])
        r, r_prime = 0.7, 0.4
        y_pts = np.linspace(0.0, min(r, r_prime) - 1e-6, 11)

        tau_p, tau_m = optical_depths_pm(r, r_prime, y_pts, radii, sig_t)

        s_r = np.sqrt(r ** 2 - y_pts ** 2)
        s_rp = np.sqrt(r_prime ** 2 - y_pts ** 2)
        expected_p = sig_t[0] * np.abs(s_r - s_rp)
        expected_m = sig_t[0] * (s_r + s_rp)

        np.testing.assert_allclose(tau_p, expected_p, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(tau_m, expected_m, rtol=1e-14, atol=1e-14)

    def test_degenerate_r_equals_rprime_tau_plus_zero(self):
        r"""For :math:`r = r'` and :math:`y < r`, the same-side path has
        zero length: :math:`s_{r} - s_{r'} = 0` exactly."""
        radii = np.array([1.0])
        sig_t = np.array([1.5])
        r = 0.6
        y_pts = np.linspace(0.0, r - 1e-6, 8)

        tau_p, tau_m = optical_depths_pm(r, r, y_pts, radii, sig_t)

        assert np.allclose(tau_p, 0.0, atol=1e-14)
        # τ⁻ should be 2·σ·√(r²-y²)
        s_r = np.sqrt(r ** 2 - y_pts ** 2)
        np.testing.assert_allclose(tau_m, 2 * sig_t[0] * s_r, rtol=1e-14, atol=1e-14)

    def test_two_annulus_diameter_transit(self):
        r"""For :math:`r = r' = R` and :math:`y < r_1`, the through-centre
        path crosses the full diameter of both annuli.

        Expected

        .. math::

           \tau^{-}(R, R, y) = 2\,\Sigma_{t,1}\sqrt{r_{1}^{2}-y^{2}}
                             + 2\,\Sigma_{t,2}\bigl(\sqrt{R^{2}-y^{2}}
                                                   - \sqrt{r_{1}^{2}-y^{2}}\bigr).
        """
        r1, R = 0.4, 1.0
        radii = np.array([r1, R])
        sig_t = np.array([0.9, 2.1])

        y_pts = np.linspace(0.0, r1 - 1e-6, 6)
        _, tau_m = optical_depths_pm(R, R, y_pts, radii, sig_t)

        s_in = np.sqrt(r1 ** 2 - y_pts ** 2)
        s_out = np.sqrt(R ** 2 - y_pts ** 2)
        expected = 2 * sig_t[0] * s_in + 2 * sig_t[1] * (s_out - s_in)

        np.testing.assert_allclose(tau_m, expected, rtol=1e-14, atol=1e-14)

    def test_two_annulus_y_between_inner_and_outer(self):
        r"""For :math:`r_1 \le y < R`, the chord does *not* intersect the
        inner annulus. Only :math:`\Sigma_{t,2}` contributes.

        Expected for :math:`r = r' = R`:

        .. math::

           \tau^{-}(R, R, y) = 2\,\Sigma_{t,2}\sqrt{R^{2}-y^{2}},
           \qquad \tau^{+}(R, R, y) = 0.
        """
        r1, R = 0.4, 1.0
        radii = np.array([r1, R])
        sig_t = np.array([0.9, 2.1])

        y_pts = np.linspace(r1, R - 1e-6, 6)
        tau_p, tau_m = optical_depths_pm(R, R, y_pts, radii, sig_t)

        s_out = np.sqrt(R ** 2 - y_pts ** 2)
        np.testing.assert_allclose(tau_m, 2 * sig_t[1] * s_out, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(tau_p, 0.0, atol=1e-14)

    def test_reciprocity_in_r_rprime_swap(self):
        """τ⁺(r, r', y) = τ⁺(r', r, y) and τ⁻(r, r', y) = τ⁻(r', r, y)."""
        radii = np.array([0.4, 1.0])
        sig_t = np.array([0.9, 2.1])
        r, r_prime = 0.3, 0.7
        y_pts = np.linspace(0.0, 0.29, 7)

        tau_p1, tau_m1 = optical_depths_pm(r, r_prime, y_pts, radii, sig_t)
        tau_p2, tau_m2 = optical_depths_pm(r_prime, r, y_pts, radii, sig_t)

        np.testing.assert_allclose(tau_p1, tau_p2, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(tau_m1, tau_m2, rtol=1e-14, atol=1e-14)

    def test_one_region_reduces_to_single_sigma(self):
        """Walker never picks up a wrong region's Σ_t in the 1-region case."""
        radii = np.array([2.5])
        # A clearly off-by-one Σ_t would amplify τ
        sig_t = np.array([3.0])
        r, r_prime = 1.8, 1.2
        y_pts = np.linspace(0.0, 1.1, 9)

        tau_p, tau_m = optical_depths_pm(r, r_prime, y_pts, radii, sig_t)

        # Both taus proportional to sig_t[0] — doubling it doubles them.
        tau_p2, tau_m2 = optical_depths_pm(
            r, r_prime, y_pts, radii, 2 * sig_t,
        )
        np.testing.assert_allclose(tau_p2, 2 * tau_p, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(tau_m2, 2 * tau_m, rtol=1e-14, atol=1e-14)
