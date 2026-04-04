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


def _build_spherical_harmonics(
    L: int, mu_x: np.ndarray, mu_y: np.ndarray, mu_z: np.ndarray,
) -> np.ndarray:
    """Compute real spherical harmonics Y_l^m for all ordinates.

    Convention (matching MATLAB ``discreteOrdinatesPWR.m``):
        Y_0^0  = 1
        Y_1^-1 = μ_z
        Y_1^0  = μ_x
        Y_1^+1 = μ_y

    Returns shape (N, L+1, 2L+1) with Y[n, l, l+m].
    """
    N = len(mu_x)
    Y = np.zeros((N, L + 1, 2 * L + 1))
    for n in range(N):
        for l in range(L + 1):
            for m in range(-l, l + 1):
                if l == 0 and m == 0:
                    Y[n, l, l + m] = 1.0
                elif l == 1 and m == -1:
                    Y[n, l, l + m] = mu_z[n]
                elif l == 1 and m == 0:
                    Y[n, l, l + m] = mu_x[n]
                elif l == 1 and m == 1:
                    Y[n, l, l + m] = mu_y[n]
    return Y


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

    def spherical_harmonics(self, L: int) -> np.ndarray:
        """(N, L+1, 2L+1) real spherical harmonics Y[n, l, l+m]."""
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

    @property
    def mu(self) -> np.ndarray:
        """Alias for mu_x (the 1D direction cosines)."""
        return self.mu_x

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

    def spherical_harmonics(self, L: int) -> np.ndarray:
        """1D harmonics: Y_0^0=1, Y_1^0=μ_x (only x-component in 1D)."""
        return _build_spherical_harmonics(
            L, self.mu_x, self.mu_y, np.zeros(self.N),
        )


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

    def spherical_harmonics(self, L: int) -> np.ndarray:
        """(N, L+1, 2L+1) real spherical harmonics for Lebedev ordinates."""
        return _build_spherical_harmonics(L, self.mu_x, self.mu_y, self.mu_z)


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
