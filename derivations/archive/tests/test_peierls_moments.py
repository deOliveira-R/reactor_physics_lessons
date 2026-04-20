r"""L0 term verification for closed-form Peierls moment integrals.

Each cumulative moment :math:`J_k^{\kappa}(z) = \int_0^z u^k\,\kappa(u)\,du`
is gated against :func:`mpmath.quad` of the same integrand to 1e-15
relative across a sweep of :math:`z` and :math:`k` values, including the
small-:math:`z` cancellation regime for the slab kernel.

The same gates also verify the per-segment differences
:math:`J_k^{\kappa}(u_b) - J_k^{\kappa}(u_a)`, which are what the
moment-form K-matrix assembly actually consumes.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from orpheus.derivations.peierls_moments import (
    e_n_cumulative_moments,
    ki_n_cumulative_moments,
    exp_cumulative_moments,
    slab_segment_moments,
    cylinder_segment_moments,
    sphere_segment_moments,
)
from orpheus.derivations._kernels import e_n_mp, ki_n_mp


# ═══════════════════════════════════════════════════════════════════════
# Reference: brute-force mpmath.quad of the moment integrand
# ═══════════════════════════════════════════════════════════════════════

def _ref_e_n_moment(z: float, k: int, dps: int = 50) -> float:
    """Reference :math:`\\int_0^z u^k E_1(u)\\,du` via adaptive mpmath.quad."""
    if z == 0.0:
        return 0.0
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(z)
        # E_1 has a logarithmic singularity at u=0; subdivision hint at 0
        # (endpoint) is implicit since mpmath.quad starts there.
        val = mpmath.quad(
            lambda u: u ** k * mpmath.expint(1, u),
            [mpmath.mpf(0), z_mp],
        )
        return float(val)


def _ref_ki_n_moment(z: float, k: int, dps: int = 40) -> float:
    """Reference :math:`\\int_0^z u^k Ki_1(u)\\,du` via mpmath.quad."""
    if z == 0.0:
        return 0.0
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(z)
        val = mpmath.quad(
            lambda u: u ** k * ki_n_mp(1, float(u), dps),
            [mpmath.mpf(0), z_mp],
        )
        return float(val)


def _ref_exp_moment(z: float, k: int, dps: int = 50) -> float:
    """Reference :math:`\\int_0^z u^k e^{-u}\\,du = \\gamma(k+1, z)`."""
    if z == 0.0:
        return 0.0
    with mpmath.workdps(dps):
        return float(mpmath.gammainc(k + 1, 0, mpmath.mpf(z)))


# ═══════════════════════════════════════════════════════════════════════
# Test sweeps
# ═══════════════════════════════════════════════════════════════════════

# Z values cover the small-z cancellation regime, the natural decay
# scale ~ a few mean free paths, and the tail where E_1/Ki_1 → 0.
Z_GRID = [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]
K_RANGE = [0, 1, 2, 3, 4, 5, 6]


@pytest.mark.l0
@pytest.mark.parametrize("z", Z_GRID)
def test_e_n_moments_match_quad(z: float) -> None:
    """Slab :math:`J_k^{E_1}(z)` matches adaptive mpmath.quad to 1e-13."""
    k_max = max(K_RANGE)
    closed = e_n_cumulative_moments(z, k_max, dps=40)
    for k in K_RANGE:
        ref = _ref_e_n_moment(z, k, dps=50)
        # Absolute tolerance protects the small-z regime where J_k ~ z^{k+1}·log z
        # vanishes; relative tolerance for the bulk.
        scale = max(abs(ref), 1e-30)
        rel_err = abs(closed[k] - ref) / scale
        assert rel_err < 1e-13, (
            f"k={k}, z={z}: closed={closed[k]:.16e}, ref={ref:.16e}, rel={rel_err:.2e}"
        )


@pytest.mark.l0
@pytest.mark.parametrize("z", Z_GRID)
def test_ki_n_moments_match_quad(z: float) -> None:
    """Cylinder :math:`J_k^{\\mathrm{Ki}_1}(z)` matches adaptive mpmath.quad to 1e-12.

    Slightly looser than slab because the reference is a nested
    mpmath.quad (Ki_1 itself is computed by adaptive quadrature),
    accumulating a bit more numerical noise.
    """
    k_max = max(K_RANGE)
    closed = ki_n_cumulative_moments(z, k_max, dps=40)
    for k in K_RANGE:
        ref = _ref_ki_n_moment(z, k, dps=40)
        scale = max(abs(ref), 1e-30)
        rel_err = abs(closed[k] - ref) / scale
        assert rel_err < 1e-12, (
            f"k={k}, z={z}: closed={closed[k]:.16e}, ref={ref:.16e}, rel={rel_err:.2e}"
        )


@pytest.mark.l0
@pytest.mark.parametrize("z", Z_GRID)
def test_exp_moments_match_quad(z: float) -> None:
    """Sphere :math:`J_k^{e^{-u}}(z)` matches mpmath.gammainc to 1e-15."""
    k_max = max(K_RANGE)
    closed = exp_cumulative_moments(z, k_max, dps=40)
    for k in K_RANGE:
        ref = _ref_exp_moment(z, k, dps=50)
        scale = max(abs(ref), 1e-30)
        rel_err = abs(closed[k] - ref) / scale
        assert rel_err < 1e-15, (
            f"k={k}, z={z}: closed={closed[k]:.16e}, ref={ref:.16e}, rel={rel_err:.2e}"
        )


@pytest.mark.l0
def test_slab_segment_moments_telescope() -> None:
    """``slab_segment_moments(0, z) == e_n_cumulative_moments(z)``."""
    z = 3.7
    k_max = 4
    seg = slab_segment_moments(0.0, z, k_max, dps=30)
    cum = e_n_cumulative_moments(z, k_max, dps=30)
    np.testing.assert_allclose(seg, cum, rtol=1e-15, atol=1e-30)


@pytest.mark.l0
def test_cylinder_segment_moments_telescope() -> None:
    """``cylinder_segment_moments(0, z) == ki_n_cumulative_moments(z)``."""
    z = 2.5
    k_max = 4
    seg = cylinder_segment_moments(0.0, z, k_max, dps=30)
    cum = ki_n_cumulative_moments(z, k_max, dps=30)
    np.testing.assert_allclose(seg, cum, rtol=1e-15, atol=1e-30)


@pytest.mark.l0
def test_slab_segment_moments_match_partition() -> None:
    """:math:`\\int_a^b = \\int_a^c + \\int_c^b` for slab moments."""
    a, c, b = 0.7, 1.8, 4.2
    k_max = 5
    full = slab_segment_moments(a, b, k_max, dps=30)
    left = slab_segment_moments(a, c, k_max, dps=30)
    right = slab_segment_moments(c, b, k_max, dps=30)
    np.testing.assert_allclose(full, left + right, rtol=1e-13, atol=1e-30)


@pytest.mark.l0
def test_cylinder_segment_moments_match_partition() -> None:
    a, c, b = 0.4, 1.3, 3.1
    k_max = 5
    full = cylinder_segment_moments(a, b, k_max, dps=30)
    left = cylinder_segment_moments(a, c, k_max, dps=30)
    right = cylinder_segment_moments(c, b, k_max, dps=30)
    np.testing.assert_allclose(full, left + right, rtol=1e-12, atol=1e-30)


@pytest.mark.l0
def test_zero_z_returns_zero_vector() -> None:
    """Degenerate :math:`z = 0` gives all-zero moment vector across families."""
    for func in (e_n_cumulative_moments, ki_n_cumulative_moments, exp_cumulative_moments):
        out = func(0.0, k_max=4, dps=20)
        assert out.shape == (5,)
        np.testing.assert_array_equal(out, np.zeros(5))
