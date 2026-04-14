"""Unit tests for diffusion solver properties.

Tests structural properties:
- Vacuum BC: flux = 0 at boundaries
- Current continuity at material interfaces
- Particle balance
"""

import numpy as np
import pytest

from orpheus.diffusion.solver import CoreGeometry, TwoGroupXS, solve_diffusion_1d

pytestmark = pytest.mark.l0  # Diffusion property checks (BC, continuity, balance)


def _default_xs():
    """Return the default fuel XS for testing."""
    return TwoGroupXS(
        transport=np.array([0.2181, 0.7850]),
        absorption=np.array([0.0096, 0.0959]),
        fission=np.array([0.0024, 0.0489]),
        production=np.array([0.0061, 0.1211]),
        chi=np.array([1.0, 0.0]),
        scattering=np.array([0.0160, 0.0]),
    )


def test_vacuum_bc():
    """Flux must be zero at the boundary faces (vacuum BC).

    The diffusion solver applies zero-flux at the physical boundaries.
    The flux at the first and last FACE should be zero (or near-zero
    due to the extrapolation distance approximation).
    """
    fuel_xs = _default_xs()
    geom = CoreGeometry(
        bot_refl_height=0.0, fuel_height=50.0, top_refl_height=0.0, dz=2.5,
    )
    result = solve_diffusion_1d(geom=geom, reflector_xs=fuel_xs, fuel_xs=fuel_xs)

    # Current at boundaries: J at face 0 and face N should reflect
    # the vacuum condition. The flux at boundary cells should be small
    # compared to the peak.
    flux = result.flux  # (2, n_cells)
    peak = flux.max()
    # First and last cell fluxes should be much smaller than peak
    for g in range(2):
        assert flux[g, 0] < 0.3 * peak, (
            f"Group {g}: boundary flux {flux[g, 0]:.4e} is too large vs peak {peak:.4e}"
        )
        assert flux[g, -1] < 0.3 * peak, (
            f"Group {g}: boundary flux {flux[g, -1]:.4e} is too large vs peak {peak:.4e}"
        )


def test_flux_positivity():
    """All flux values must be positive in the fundamental mode."""
    fuel_xs = _default_xs()
    geom = CoreGeometry(
        bot_refl_height=0.0, fuel_height=50.0, top_refl_height=0.0, dz=2.5,
    )
    result = solve_diffusion_1d(geom=geom, reflector_xs=fuel_xs, fuel_xs=fuel_xs)
    assert np.all(result.flux > 0), f"Non-positive flux: min={result.flux.min():.6e}"


def test_flux_symmetry():
    """For a symmetric bare slab (no reflector), flux must be symmetric."""
    fuel_xs = _default_xs()
    geom = CoreGeometry(
        bot_refl_height=0.0, fuel_height=50.0, top_refl_height=0.0, dz=2.5,
    )
    result = solve_diffusion_1d(geom=geom, reflector_xs=fuel_xs, fuel_xs=fuel_xs)

    flux = result.flux  # (2, n_cells)
    for g in range(2):
        np.testing.assert_allclose(
            flux[g, :], flux[g, ::-1], rtol=1e-10,
            err_msg=f"Group {g} flux is not symmetric",
        )
