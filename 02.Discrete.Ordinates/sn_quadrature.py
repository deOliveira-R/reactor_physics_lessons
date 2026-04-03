"""Angular quadrature for SN transport.

Provides a protocol and two implementations:
- GaussLegendre1D: for 1D slab problems (mu on [-1,1], weights sum to 2)
- LebedevSphere: for 2D/3D problems (directions on unit sphere, weights sum to 4π)

The solver uses ``1/sum(weights)`` as the isotropic normalization factor,
making it quadrature-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AngularQuadrature(Protocol):
    """Contract for any angular quadrature usable by the SN solver."""

    mu_x: np.ndarray       # (N,) x-direction cosines
    mu_y: np.ndarray       # (N,) y-direction cosines (0 for 1D)
    weights: np.ndarray    # (N,) quadrature weights
    N: int                 # number of ordinates

    def reflection_index(self, axis: str) -> np.ndarray:
        """Index array: ref[n] = partner of ordinate n reflected in ``axis``."""
        ...


@dataclass
class GaussLegendre1D:
    """Gauss-Legendre quadrature on [-1, 1] for 1D slab transport.

    mu_x = GL points, mu_y = 0. Weights sum to 2.
    """

    mu_x: np.ndarray
    mu_y: np.ndarray
    weights: np.ndarray
    N: int

    @classmethod
    def create(cls, n_ordinates: int = 16) -> GaussLegendre1D:
        """Build N-point GL quadrature (must be even for SN)."""
        mu, w = np.polynomial.legendre.leggauss(n_ordinates)
        return cls(
            mu_x=mu,
            mu_y=np.zeros(n_ordinates),
            weights=w,
            N=n_ordinates,
        )

    def reflection_index(self, axis: str) -> np.ndarray:
        """GL is symmetric: partner of i is N-1-i for x-reflection."""
        if axis == "x":
            return np.arange(self.N)[::-1].copy()
        else:
            # y-reflection: mu_y=0, so every ordinate is its own partner
            return np.arange(self.N)


@dataclass
class LebedevSphere:
    """Lebedev quadrature on the unit sphere for 2D/3D transport.

    Weights sum to 4π. Directions cover the full sphere.
    """

    mu_x: np.ndarray
    mu_y: np.ndarray
    mu_z: np.ndarray
    weights: np.ndarray
    N: int
    _ref_x: np.ndarray
    _ref_y: np.ndarray
    _ref_z: np.ndarray

    @classmethod
    def create(cls, order: int = 17) -> LebedevSphere:
        """Build Lebedev quadrature from scipy."""
        from scipy.integrate import lebedev_rule
        pts, w = lebedev_rule(order)
        mu_x, mu_y, mu_z = pts[0], pts[1], pts[2]
        n_pts = len(w)

        ref_x = _find_reflections(-mu_x, mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_y = _find_reflections(mu_x, -mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_z = _find_reflections(mu_x, mu_y, -mu_z, mu_x, mu_y, mu_z)

        return cls(
            mu_x=mu_x, mu_y=mu_y, mu_z=mu_z,
            weights=w, N=n_pts,
            _ref_x=ref_x, _ref_y=ref_y, _ref_z=ref_z,
        )

    def reflection_index(self, axis: str) -> np.ndarray:
        if axis == "x":
            return self._ref_x
        elif axis == "y":
            return self._ref_y
        elif axis == "z":
            return self._ref_z
        raise ValueError(f"Unknown axis: {axis}")


def _find_reflections(
    tx: np.ndarray, ty: np.ndarray, tz: np.ndarray,
    rx: np.ndarray, ry: np.ndarray, rz: np.ndarray,
) -> np.ndarray:
    """Find index of closest match in (rx,ry,rz) for each (tx,ty,tz)."""
    n = len(tx)
    ref = np.empty(n, dtype=int)
    for i in range(n):
        dist = (rx - tx[i])**2 + (ry - ty[i])**2 + (rz - tz[i])**2
        ref[i] = np.argmin(dist)
    return ref
