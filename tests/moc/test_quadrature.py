"""Tests for the MOC angular quadrature (azimuthal x Tabuchi-Yamamoto polar)."""

import numpy as np
import pytest

from orpheus.moc.quadrature import MOCQuadrature

# MOC TY polar quadrature weights and azimuthal spacing.
# test_azimuthal_angles_{in_range,uniform_spacing} directly verify
# Eq. :label:`azimuthal-angles` (issue #87 Phase B.3).
pytestmark = [pytest.mark.l0, pytest.mark.verifies("azimuthal-angles")]


# ── TY polar weight sums ────────────────────────────────────────────

@pytest.mark.parametrize("n_polar", [1, 2, 3])
def test_polar_weights_sum_to_half(n_polar):
    """TY polar weights must sum to 0.5 (one hemisphere)."""
    q = MOCQuadrature.create(n_azi=4, n_polar=n_polar)
    assert q.omega_polar.sum() == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize("n_polar", [1, 2, 3])
def test_sin_polar_in_unit_interval(n_polar):
    """sin(theta_p) must lie in (0, 1]."""
    q = MOCQuadrature.create(n_azi=4, n_polar=n_polar)
    assert np.all(q.sin_polar > 0)
    assert np.all(q.sin_polar <= 1.0)


# ── Azimuthal weights ───────────────────────────────────────────────

@pytest.mark.parametrize("n_azi", [4, 8, 16, 32])
def test_azimuthal_weights_sum_to_one(n_azi):
    """Azimuthal weights must sum to 1."""
    q = MOCQuadrature.create(n_azi=n_azi, n_polar=1)
    assert q.omega_azi.sum() == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("n_azi", [4, 8, 16])
def test_azimuthal_angles_in_range(n_azi):
    """Azimuthal angles must lie in [0, pi)."""
    q = MOCQuadrature.create(n_azi=n_azi, n_polar=1)
    assert np.all(q.phi >= 0)
    assert np.all(q.phi < np.pi)


@pytest.mark.parametrize("n_azi", [4, 8, 16])
def test_azimuthal_angles_uniform_spacing(n_azi):
    """Azimuthal angles must be uniformly spaced."""
    q = MOCQuadrature.create(n_azi=n_azi, n_polar=1)
    diffs = np.diff(q.phi)
    assert np.allclose(diffs, np.pi / n_azi, atol=1e-14)


# ── Combined weight normalisation ───────────────────────────────────

@pytest.mark.parametrize("n_azi,n_polar", [(4, 1), (8, 2), (16, 3), (32, 3)])
def test_combined_weight_normalisation(n_azi, n_polar):
    """2 * sum(omega_azi) * sum(omega_polar) = 1 (full sphere)."""
    q = MOCQuadrature.create(n_azi=n_azi, n_polar=n_polar)
    total = 2.0 * q.omega_azi.sum() * q.omega_polar.sum()
    assert total == pytest.approx(1.0, abs=1e-12)


# ── Shapes ───────────────────────────────────────────────────────────

def test_shapes():
    """Array dimensions must match n_azi and n_polar."""
    q = MOCQuadrature.create(n_azi=16, n_polar=3)
    assert q.phi.shape == (16,)
    assert q.omega_azi.shape == (16,)
    assert q.sin_polar.shape == (3,)
    assert q.omega_polar.shape == (3,)


# ── TY-3 published values ───────────────────────────────────────────

def test_ty3_values_match_published():
    """TY-3 sin(theta) and weights must match Yamamoto et al. (2007)."""
    q = MOCQuadrature.create(n_azi=4, n_polar=3)
    np.testing.assert_allclose(q.sin_polar, [0.166648, 0.537707, 0.932954], atol=1e-6)
    np.testing.assert_allclose(
        q.omega_polar,
        [0.046233 / 2, 0.283619 / 2, 0.670148 / 2],
        atol=1e-6,
    )


# ── Validation errors ───────────────────────────────────────────────

def test_invalid_n_azi():
    with pytest.raises(ValueError, match="n_azi"):
        MOCQuadrature.create(n_azi=1, n_polar=1)


def test_invalid_n_polar():
    with pytest.raises(ValueError, match="n_polar"):
        MOCQuadrature.create(n_azi=4, n_polar=5)
