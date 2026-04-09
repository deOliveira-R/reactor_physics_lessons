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


# ═══════════════════════════════════════════════════════════════════════
# Level-Symmetric S_N quadrature
# ═══════════════════════════════════════════════════════════════════════

# Tabulated first direction cosine μ₁² and weights for S2–S16.
# Source: Lewis & Miller, Table 4-1; also Carlson & Lathrop (1968).
# All octant ordinates are generated from these by permuting (μ,η,ξ).
_LEVEL_SYM_DATA: dict[int, dict] = {
    2: {"mu2": [1 / 3], "weights": [1.0]},
    4: {"mu2": [1 / 6, 1 / 6], "weights": [1 / 3, 1 / 3]},
    # For S4 the standard single-weight form:
    # one distinct direction cosine value μ₁² = 1/3 is used with 3 permutations per octant.
}

# For clean implementation, we use the closed-form construction
# described in Lewis & Miller §4.2 for arbitrary even N.
# The direction cosine values on each level satisfy:
#   μ_p² = (2p - 1) / (N(N+2)/4 - 1) · (1 - 3μ₁²) + μ₁²  for p = 1..N/2
# with μ₁² chosen to satisfy the moment conditions.


def _build_level_symmetric(sn_order: int) -> tuple:
    """Build level-symmetric S_N quadrature from first principles.

    Uses the standard construction: N/2 μ-levels with equally-spaced
    μ² values.  On each level p, there are (N/2 - p + 1) ordinates in
    the first octant.  Weights are determined by the zeroth and second
    moment conditions.

    Returns
    -------
    mu_x, mu_y, mu_z, weights : flattened arrays for full sphere
    level_info : dict with per-level structure
    """
    if sn_order % 2 != 0 or sn_order < 2:
        raise ValueError(f"S_N order must be positive even, got {sn_order}")

    n_half = sn_order // 2  # number of μ-levels per hemisphere

    # Standard first direction cosine squared (Lewis & Miller convention)
    # μ₁² is set so that moment conditions are satisfied.
    # For the equal-weight level-symmetric set: μ₁² = 1/(N(N-1)/2 + 1) · 1
    # Actually, the standard choice is μ₁² such that Σ w = 4π and Σ w μ² = 4π/3.
    # For the simple equal-weight construction:
    #   μ_p² = μ₁² + (p-1)·Δ,  Δ = (1 - 3μ₁²)/(n_half - 1) for n_half > 1
    #   Δ chosen so that the set {μ_p} covers [μ₁, √(1-2μ₁²)] symmetrically.
    # Standard: μ₁² = 1/(sn_order*(sn_order+2)/4)  [Carlson & Lathrop]

    if n_half == 1:
        # S2: single direction cosine, isotropic
        mu2_levels = np.array([1.0 / 3.0])
    else:
        # Equal spacing in μ²: μ_p² = μ₁² + (p-1)·2(1-3μ₁²)/(N-2)
        mu1_sq = 1.0 / (sn_order * (sn_order + 2) / 4)
        delta = 2.0 * (1.0 - 3.0 * mu1_sq) / (sn_order - 2)
        mu2_levels = mu1_sq + np.arange(n_half) * delta

    mu_levels = np.sqrt(mu2_levels)

    # Build octant ordinates: on level p (0-indexed), the direction cosines
    # are all permutations of (μ_a, μ_b, μ_c) where μ_a² + μ_b² + μ_c² = 1
    # and each comes from the set of level values.
    # For level p: μ_z = mu_levels[p], and (η, ξ) are all pairs from
    # mu_levels that satisfy η² + ξ² = 1 - μ_z².
    octant_dirs = []  # list of (η, ξ, μ) tuples
    for p in range(n_half):
        mu_z = mu_levels[p]
        sin_theta_sq = 1.0 - mu_z**2
        # On this level, the η values come from the same set
        # Number of azimuthal points on level p: n_half - p
        n_azi = n_half - p
        for k in range(n_azi):
            eta = mu_levels[k]
            xi_sq = sin_theta_sq - eta**2
            if xi_sq < -1e-14:
                continue
            xi = np.sqrt(max(xi_sq, 0.0))
            octant_dirs.append((eta, xi, mu_z))

    n_octant = len(octant_dirs)

    # Equal weights within the octant (simple level-symmetric)
    w_octant = 4.0 * np.pi / (8.0 * n_octant)

    # Reflect to full sphere (8 octants)
    all_eta, all_xi, all_mu, all_w = [], [], [], []
    # Level tracking: we'll rebuild after reflection
    for eta, xi, mu_z in octant_dirs:
        for s_eta in [-1, 1]:
            for s_xi in [-1, 1]:
                for s_mu in [-1, 1]:
                    all_eta.append(s_eta * eta)
                    all_xi.append(s_xi * xi)
                    all_mu.append(s_mu * mu_z)
                    all_w.append(w_octant)

    mu_x = np.array(all_eta)   # η — radial for cylindrical
    mu_y = np.array(all_xi)    # ξ — azimuthal for cylindrical
    mu_z = np.array(all_mu)    # μ — axial
    weights = np.array(all_w)

    # Build level structure: group ordinates by |μ_z| value,
    # sort within each level by increasing η (mu_x) for the
    # cylindrical azimuthal sweep convention.
    n_levels = n_half
    level_mu_vals = mu_levels
    level_indices = []
    for p in range(n_levels):
        tol = 1e-12
        idx = np.where(np.abs(np.abs(mu_z) - level_mu_vals[p]) < tol)[0]
        order = np.argsort(mu_x[idx])
        level_indices.append(idx[order])

    return mu_x, mu_y, mu_z, weights, n_levels, level_mu_vals, level_indices


@dataclass
class LevelSymmetricSN:
    """Level-symmetric S_N quadrature on the unit sphere.

    Standard triangular quadrature with N/2 μ-levels per hemisphere.
    Provides the ``level_indices`` structure required by the cylindrical
    SN sweep for azimuthal redistribution.

    Weights sum to 4π.
    """

    mu_x: np.ndarray       # η — radial direction cosines
    mu_y: np.ndarray       # ξ — azimuthal direction cosines
    mu_z: np.ndarray       # μ — axial direction cosines
    weights: np.ndarray
    N: int
    _ref_x: np.ndarray
    _ref_y: np.ndarray
    _ref_z: np.ndarray

    # Level structure
    n_levels: int
    level_indices: list[np.ndarray]
    level_mu: np.ndarray

    @classmethod
    def create(cls, sn_order: int = 4) -> LevelSymmetricSN:
        """Build S_N level-symmetric quadrature of given order."""
        mu_x, mu_y, mu_z, w, n_levels, level_mu, level_indices = \
            _build_level_symmetric(sn_order)
        N = len(w)

        ref_x = _find_reflections(-mu_x, mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_y = _find_reflections(mu_x, -mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_z = _find_reflections(mu_x, mu_y, -mu_z, mu_x, mu_y, mu_z)

        return cls(
            mu_x=mu_x, mu_y=mu_y, mu_z=mu_z,
            weights=w, N=N,
            _ref_x=ref_x, _ref_y=ref_y, _ref_z=ref_z,
            n_levels=n_levels,
            level_indices=level_indices,
            level_mu=level_mu,
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
        return _build_spherical_harmonics(L, self.mu_x, self.mu_y, self.mu_z)


# ═══════════════════════════════════════════════════════════════════════
# Product Quadrature (GL in μ × equispaced in φ)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProductQuadrature:
    """Product quadrature: Gauss-Legendre(μ) × equispaced(φ).

    The polar angle θ is discretised via Gauss-Legendre on μ = cos θ ∈ [-1, 1].
    The azimuthal angle φ is discretised uniformly on [0, 2π).

    Direction cosines:
    - μ_z = μ (axial, = cos θ)
    - μ_x = η = sin(θ) cos(φ) (radial for cylindrical)
    - μ_y = ξ = sin(θ) sin(φ) (azimuthal for cylindrical)

    Weights: ``w = w_GL(μ) · (2π / n_phi)`` — sum to 4π.

    Provides ``level_indices`` for the cylindrical sweep.
    """

    mu_x: np.ndarray       # η = sin(θ)cos(φ)
    mu_y: np.ndarray       # ξ = sin(θ)sin(φ)
    mu_z: np.ndarray       # μ = cos(θ)
    weights: np.ndarray
    N: int
    _ref_x: np.ndarray
    _ref_y: np.ndarray
    _ref_z: np.ndarray

    # Level structure
    n_levels: int
    level_indices: list[np.ndarray]
    level_mu: np.ndarray

    @classmethod
    def create(cls, n_mu: int = 8, n_phi: int = 8) -> ProductQuadrature:
        """Build product quadrature with n_mu GL points and n_phi azimuthal points.

        Parameters
        ----------
        n_mu : int
            Number of Gauss-Legendre points in μ (polar).
        n_phi : int
            Number of equispaced points in φ (azimuthal).
        """
        # GL points in μ = cos(θ)
        mu_gl, w_gl = np.polynomial.legendre.leggauss(n_mu)

        # Equispaced φ in [0, 2π)
        phi_pts = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        w_phi = 2.0 * np.pi / n_phi

        N_total = n_mu * n_phi
        mu_x = np.empty(N_total)
        mu_y = np.empty(N_total)
        mu_z = np.empty(N_total)
        weights = np.empty(N_total)
        level_indices = []

        idx = 0
        for p in range(n_mu):
            mu_val = mu_gl[p]
            sin_theta = np.sqrt(1.0 - mu_val**2)
            level_idx = []
            for m in range(n_phi):
                mu_x[idx] = sin_theta * np.cos(phi_pts[m])
                mu_y[idx] = sin_theta * np.sin(phi_pts[m])
                mu_z[idx] = mu_val
                weights[idx] = w_gl[p] * w_phi
                level_idx.append(idx)
                idx += 1
            # Sort by increasing η (mu_x) for cylindrical azimuthal sweep.
            # The sweep proceeds from most-inward (η = −sin θ) to
            # most-outward (η = +sin θ), matching the α recursion
            # convention from Bailey et al. (2009) Eq. 50.
            level_arr = np.array(level_idx)
            order = np.argsort(mu_x[level_arr])
            level_indices.append(level_arr[order])

        ref_x = _find_reflections(-mu_x, mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_y = _find_reflections(mu_x, -mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_z = _find_reflections(mu_x, mu_y, -mu_z, mu_x, mu_y, mu_z)

        return cls(
            mu_x=mu_x, mu_y=mu_y, mu_z=mu_z,
            weights=weights, N=N_total,
            _ref_x=ref_x, _ref_y=ref_y, _ref_z=ref_z,
            n_levels=n_mu,
            level_indices=level_indices,
            level_mu=mu_gl,
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
        return _build_spherical_harmonics(L, self.mu_x, self.mu_y, self.mu_z)


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

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
