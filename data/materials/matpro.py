"""MATPRO material property correlations for UO2, gap gases, and Zircaloy.

Port of MATLAB ``matpro.m``.  All correlations are vectorised (work with
numpy arrays) and temperatures are in Kelvin unless noted otherwise.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# UO2 fuel
# ---------------------------------------------------------------------------

UO2_RHO = 10_980.0  # theoretical density (kg/m3)


def uo2_cp(T: np.ndarray) -> np.ndarray:
    """UO2 specific heat (J/kg-K).  MATPRO(2003)."""
    T = np.asarray(T, dtype=float)
    return 162.3 + 0.3038 * T - 2.391e-4 * T**2 + 6.404e-8 * T**3


def uo2_k(T: np.ndarray, Bu: float = 0.0, por: float = 0.05) -> np.ndarray:
    """UO2 thermal conductivity (W/m-K).  MATPRO(2003)."""
    T = np.asarray(T, dtype=float)
    base = (1.0 / (0.0452 + 0.000246 * T + 0.00187 * Bu
                    + 0.038 * (1 - 0.9 * np.exp(-0.04 * Bu))
                    * Bu**0.28 / (1 + 396 * np.exp(-6380.0 / T)))
            + 3.5e9 * np.exp(-16360.0 / T) / T**2)
    return base * 1.0789 * (1 - por) / (1 + por / 2)


def uo2_thexp(T: np.ndarray) -> np.ndarray:
    """UO2 thermal expansion (m/m).  MATPRO(2003)."""
    T = np.asarray(T, dtype=float)
    return (T / 1000 - 0.3 + 4 * np.exp(-5000.0 / T)) / 100


def uo2_E(T: np.ndarray, por: float = 0.05) -> np.ndarray:
    """UO2 Young's modulus (MPa).  MATPRO(2003) p.2-58."""
    T = np.asarray(T, dtype=float)
    return 2.334e5 * (1 - 2.752 * por) * (1 - 1.0915e-4 * T)


UO2_NU = 0.316  # Poisson ratio


def uo2_swelling_rate(dFdt: float, F: np.ndarray, T: np.ndarray) -> np.ndarray:
    """UO2 volumetric swelling rate (1/s).  MATPRO(2003)."""
    T = np.asarray(T, dtype=float)
    F = np.asarray(F, dtype=float)
    solid = 2.5e-29 * dFdt
    dT = np.maximum(2800 - T, 0.0)  # clamp to avoid negative base with fractional exp
    gaseous = np.where(
        T < 2800,
        8.8e-56 * dFdt * dT**11.73 * np.exp(-0.0162 * dT) * np.exp(-8e-27 * F),
        0.0,
    )
    return solid + gaseous


def uo2_creep_rate(sig: np.ndarray, T: np.ndarray) -> np.ndarray:
    """UO2 thermal creep rate (1/s) — simplified."""
    return 5e5 * sig * np.exp(-4e5 / (8.314 * T))


# ---------------------------------------------------------------------------
# Gap gases
# ---------------------------------------------------------------------------

def k_He(T: np.ndarray) -> np.ndarray:
    """He thermal conductivity (W/m-K)."""
    return 2.639e-3 * np.asarray(T, dtype=float)**0.7085


def k_Xe(T: np.ndarray) -> np.ndarray:
    """Xe thermal conductivity (W/m-K)."""
    return 4.351e-5 * np.asarray(T, dtype=float)**0.8616


def k_Kr(T: np.ndarray) -> np.ndarray:
    """Kr thermal conductivity (W/m-K)."""
    return 8.247e-5 * np.asarray(T, dtype=float)**0.8363


# Molecular weights (g/mol)
_MW = np.array([2.0, 54.0, 36.0])  # He, Kr, Xe


def _psi(k1, k2, M1, M2):
    """Prandtl mixing factor for gas conductivities."""
    return (1 + np.sqrt(np.sqrt(M1 / M2) * k1 / k2))**2 / np.sqrt(8 * (1 + M1 / M2))


def gas_mixture_k(T: float, mole_fractions: np.ndarray) -> float:
    """Gas mixture thermal conductivity (W/m-K).  MATPRO.

    Parameters
    ----------
    T : temperature (K)
    mole_fractions : (3,) array — [He, Kr, Xe] mole fractions.
    """
    k = np.array([k_He(T), k_Kr(T), k_Xe(T)])
    x = np.asarray(mole_fractions)
    M = _MW

    result = 0.0
    for i in range(3):
        denom = sum(_psi(k[i], k[j], M[i], M[j]) * x[j] for j in range(3))
        result += k[i] * x[i] / denom
    return float(result)


# ---------------------------------------------------------------------------
# Zircaloy cladding
# ---------------------------------------------------------------------------

ZRY_RHO = 6_600.0  # density (kg/m3)


def zry_cp(T: np.ndarray) -> np.ndarray:
    """Zircaloy specific heat (J/kg-K)."""
    return 252.54 + 0.11474 * np.asarray(T, dtype=float)


def zry_k(T: np.ndarray) -> np.ndarray:
    """Zircaloy thermal conductivity (W/m-K)."""
    T = np.asarray(T, dtype=float)
    return 7.51 + 2.09e-2 * T - 1.45e-5 * T**2 + 7.67e-9 * T**3


def zry_thexp(T: np.ndarray) -> np.ndarray:
    """Zircaloy thermal expansion (m/m).  PNNL(2010) p.3-16."""
    return -2.373e-5 + (np.asarray(T, dtype=float) - 273.15) * 6.721e-6


def zry_E(T: np.ndarray) -> np.ndarray:
    """Zircaloy Young's modulus (MPa).  PNNL(2010) p.3-20."""
    return 1.088e5 - 54.75 * np.asarray(T, dtype=float)


ZRY_NU = 0.3  # Poisson ratio


def zry_creep_rate(sig: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Zircaloy thermal creep rate (1/s) — simplified."""
    return 1e5 * sig * np.exp(-2e5 / (8.314 * T))


def zry_K(T: np.ndarray) -> np.ndarray:
    """Zircaloy strength coefficient (MPa).  MATPRO."""
    T = np.asarray(T, dtype=float)
    return np.where(
        T < 743,
        2.257e9 + T * (-5.644e6 + T * (7.525e3 - T * 4.33167)),
        np.where(
            T < 1090,
            2.522488e6 * np.exp(2.8500027e6 / T**2),
            np.where(
                T < 1255,
                184.1376039e6 - 1.4345448e5 * T,
                4.330e7 + T * (-6.685e4 + T * (37.579 - T * 7.33e-3)),
            ),
        ),
    )


def zry_m(T: np.ndarray) -> np.ndarray:
    """Zircaloy strain rate sensitivity exponent.  MATPRO."""
    T = np.asarray(T, dtype=float)
    return np.where(
        T <= 730,
        0.02,
        np.where(
            T <= 900,
            20.63172161 - 0.07704552983 * T + 9.504843067e-05 * T**2
            - 3.860960716e-08 * T**3,
            -6.47e-02 + T * 2.203e-04,
        ),
    )


def zry_n(T: np.ndarray) -> np.ndarray:
    """Zircaloy strain hardening exponent.  MATPRO."""
    T = np.asarray(T, dtype=float)
    return np.where(
        T < 1099.0772,
        -9.490e-2 + T * (1.165e-3 + T * (-1.992e-6 + T * 9.588e-10)),
        np.where(T < 1600, -0.22655119 + 2.5e-4 * T, 0.17344880),
    )


def zry_burst_stress(T: np.ndarray) -> np.ndarray:
    """Zircaloy burst stress (MPa).  MATPRO(2003) p.4-187."""
    T = np.asarray(T, dtype=float)
    return 10.0 ** (8.42 + T * (2.78e-3 + T * (-4.87e-6 + T * 1.49e-9))) / 1e6
