"""Special-function kernels for collision probability derivations.

Provides E₃ (exponential integral) for slab geometry and
Ki₃/Ki₄ (Bickley-Naylor) for cylindrical geometry.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.integrate import quad
from scipy.special import expn


# ═══════════════════════════════════════════════════════════════════════
# E₃ kernel (slab geometry)
# ═══════════════════════════════════════════════════════════════════════

def e3(x: float) -> float:
    """Third-order exponential integral E₃(x)."""
    return float(expn(3, max(x, 0.0)))


def e3_vec(x: np.ndarray) -> np.ndarray:
    """Vectorised E₃."""
    return expn(3, np.maximum(x, 0.0))


# ═══════════════════════════════════════════════════════════════════════
# Ki₃ / Ki₄ kernels (cylindrical geometry)
# ═══════════════════════════════════════════════════════════════════════

class BickleyTables:
    """Tabulated Ki₃ and Ki₄ Bickley-Naylor functions.

    Ki₃(x) = int_0^{pi/2} exp(-x / sin t) sin(t) dt
    Ki₄(x) = int_x^inf Ki₃(t) dt

    Tables are built once and cached.
    """

    def __init__(self, n_points: int = 20_000, x_max: float = 50.0):
        self.n_points = n_points
        self.x_max = x_max
        self._x = np.linspace(0, x_max, n_points)
        self._dx = self._x[1] - self._x[0]

        # Build Ki₃ table via numerical integration
        ki3 = np.empty(n_points)
        ki3[0] = 1.0
        for i in range(1, n_points):
            ki3[i], _ = quad(
                lambda t, xx=self._x[i]: np.exp(-xx / np.sin(t)) * np.sin(t),
                0, np.pi / 2,
            )
        self._ki3 = ki3

        # Ki₄ = cumulative integral of Ki₃ from x to infinity
        self._ki4 = np.cumsum(ki3[::-1])[::-1] * self._dx
        self._ki4[-1] = 0.0

    def ki3(self, x: float) -> float:
        """Evaluate Ki₃(x) by interpolation."""
        return float(np.interp(x, self._x, self._ki3, right=0.0))

    def ki4(self, x: float) -> float:
        """Evaluate Ki₄(x) by interpolation."""
        return float(np.interp(x, self._x, self._ki4, right=0.0))

    def ki3_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Ki₃."""
        return np.interp(np.asarray(x, dtype=float), self._x, self._ki3, right=0.0)

    def ki4_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Ki₄."""
        return np.interp(np.asarray(x, dtype=float), self._x, self._ki4, right=0.0)


@functools.lru_cache(maxsize=1)
def bickley_tables(
    n_points: int = 20_000, x_max: float = 50.0,
) -> BickleyTables:
    """Get (or build) the cached Bickley-Naylor lookup tables."""
    return BickleyTables(n_points, x_max)
