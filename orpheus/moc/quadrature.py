"""Angular quadrature for the Method of Characteristics.

Product quadrature: uniform azimuthal angles x Tabuchi-Yamamoto (TY)
polar angles.  The TY quadrature is optimised for the Bickley-function
integrals that arise when the 2-D MOC flat-source solution is integrated
over polar angle (Yamamoto et al., J. Nucl. Sci. Technol. 44(2), 2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Tabuchi-Yamamoto tables ─────────────────────────────────────────
# Each entry: (sin_theta, weight) for one polar angle per half-space.
# Weights sum to 0.5 (one hemisphere); full-sphere sum is 1.0.
# Source: Yamamoto et al. (2007), Table 2; also Knott & Yamamoto (2010).

_TY_TABLES: dict[int, tuple[np.ndarray, np.ndarray]] = {
    1: (
        np.array([0.798184]),
        np.array([0.500000]),
    ),
    2: (
        np.array([0.363900, 0.899900]),
        np.array([0.212854 / 2, 0.787146 / 2]),
    ),
    3: (
        np.array([0.166648, 0.537707, 0.932954]),
        np.array([0.046233 / 2, 0.283619 / 2, 0.670148 / 2]),
    ),
}


@dataclass(frozen=True)
class MOCQuadrature:
    """Product quadrature for 2-D Method of Characteristics.

    Azimuthal: ``n_azi`` uniform angles in [0, pi).  Each azimuthal angle
    defines a family of parallel tracks; the supplementary angle (phi + pi)
    is the same physical track traversed in the opposite direction.

    Polar: Tabuchi-Yamamoto quadrature with ``n_polar`` angles per
    half-space.  The polar angle enters the MOC only through the
    effective 3-D optical thickness  tau / sin(theta_p).

    Attributes
    ----------
    n_azi : int
        Number of azimuthal angles in [0, pi).
    n_polar : int
        Number of polar angles per half-space (1, 2, or 3).
    phi : ndarray, shape (n_azi,)
        Azimuthal angles in [0, pi) (radians).
    omega_azi : ndarray, shape (n_azi,)
        Azimuthal weights, each = 1 / n_azi, summing to 1.
    sin_polar : ndarray, shape (n_polar,)
        sin(theta_p) for each TY polar angle.
    omega_polar : ndarray, shape (n_polar,)
        TY polar weights (sum = 0.5 for one hemisphere).
    """

    n_azi: int
    n_polar: int
    phi: np.ndarray
    omega_azi: np.ndarray
    sin_polar: np.ndarray
    omega_polar: np.ndarray

    @classmethod
    def create(cls, n_azi: int = 16, n_polar: int = 3) -> MOCQuadrature:
        """Build an MOC product quadrature.

        Parameters
        ----------
        n_azi : int
            Number of azimuthal angles in [0, pi).  Must be >= 2.
        n_polar : int
            Number of TY polar angles per half-space (1, 2, or 3).
        """
        if n_azi < 2:
            raise ValueError(f"n_azi must be >= 2, got {n_azi}")
        if n_polar not in _TY_TABLES:
            raise ValueError(
                f"n_polar must be one of {sorted(_TY_TABLES)}, got {n_polar}"
            )

        phi = np.linspace(0, np.pi, n_azi, endpoint=False) + np.pi / (2 * n_azi)
        omega_azi = np.full(n_azi, 1.0 / n_azi)
        sin_polar, omega_polar = _TY_TABLES[n_polar]

        return cls(
            n_azi=n_azi,
            n_polar=n_polar,
            phi=phi,
            omega_azi=omega_azi,
            sin_polar=sin_polar.copy(),
            omega_polar=omega_polar.copy(),
        )
