"""Water / steam equilibrium properties wrapper using pyXSteam.

Port of MATLAB ``H2Oeq.m``.  Given pressure (MPa) and enthalpy (J/kg),
returns mixture, saturated-liquid and saturated-vapor property bundles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyXSteam.XSteam import XSteam

_st = XSteam(XSteam.UNIT_SYSTEM_MKS)  # bar, °C, kg/m3


# ---------------------------------------------------------------------------
# IAPWS 2008 viscosity (without pyXSteam's 900 °C cutoff)
# ---------------------------------------------------------------------------
# Coefficients for the finite-density correction (H_ij matrix).
_H_VISC = np.array([
    [ 0.5132047,  0.3205656,  0.0,         0.0,        -0.7782567,  0.1885447],
    [ 0.2151778,  0.7317883,  1.241044,    1.476783,    0.0,         0.0],
    [-0.2818107, -1.070786,  -1.263184,    0.0,         0.0,         0.0],
    [ 0.1778064,  0.460504,   0.2340379,  -0.4924179,   0.0,         0.0],
    [-0.0417661,  0.0,        0.0,         0.1600435,   0.0,         0.0],
    [ 0.0,       -0.01578386, 0.0,         0.0,         0.0,         0.0],
    [ 0.0,        0.0,        0.0,        -0.003629481, 0.0,         0.0],
])


def _iapws_viscosity(T_K: float, rho: float) -> float:
    """IAPWS 2008 dynamic viscosity (Pa-s).

    Same correlation as pyXSteam but without the 900 °C upper-temperature
    cutoff.  Used as a fallback when pyXSteam returns NaN.

    # TODO DA-20260405-004: Consider using this for ALL viscosity calls (replacing _st.my_ph
    # entirely). Bit-identical below 900 °C, no region dispatch overhead,
    # and T_K/rho are already available at call sites.
    """
    Ts = T_K / 647.226
    rhos = rho / 317.763

    # Dilute-gas (kinetic theory) contribution
    mu0 = Ts**0.5 / (1 + 0.978197 / Ts + 0.579829 / Ts**2 - 0.202354 / Ts**3)

    # Finite-density correction
    inv_Ts_m1 = 1.0 / Ts - 1.0
    rhos_m1 = rhos - 1.0
    S = 0.0
    rhos_pow = 1.0
    for i in range(7):
        inv_pow = 1.0
        for j in range(6):
            S += _H_VISC[i, j] * inv_pow * rhos_pow
            inv_pow *= inv_Ts_m1
        rhos_pow *= rhos_m1
    mu1 = np.exp(rhos * S)

    return float(mu0 * mu1 * 55.071e-6)


@dataclass
class FluidProps:
    """Thermodynamic properties of a fluid state."""
    T: float       # temperature (K)
    p: float       # pressure (MPa)
    h: float       # specific enthalpy (J/kg)
    rho: float     # density (kg/m3)
    k: float       # thermal conductivity (W/m-K)
    mu: float      # dynamic viscosity (Pa-s)
    nu: float      # kinematic viscosity (m2/s)
    c_p: float     # specific heat (J/kg-K)
    void: float = 0.0   # void fraction (-)
    x: float = 0.0      # quality (-)
    Tsat: float = 0.0   # saturation temperature (K)


@dataclass
class SatProps:
    """Saturation properties for one phase."""
    T: float       # saturation temperature (K)
    h: float       # specific enthalpy (J/kg)
    rho: float     # density (kg/m3)
    k: float       # thermal conductivity (W/m-K)
    mu: float      # dynamic viscosity (Pa-s)
    nu: float      # kinematic viscosity (m2/s)
    c_p: float     # specific heat (J/kg-K)
    sig: float = 0.0  # surface tension (N/m) — liquid only


def h2o_properties(p_MPa: float, h_Jkg: float) -> tuple[FluidProps, SatProps, SatProps]:
    """Compute water/steam properties at given pressure and enthalpy.

    Parameters
    ----------
    p_MPa : pressure in MPa
    h_Jkg : specific enthalpy in J/kg

    Returns
    -------
    (pro, Lsat, Vsat) — mixture, saturated liquid, saturated vapor properties.
    """
    p_bar = p_MPa * 10
    h_kJ = h_Jkg / 1e3

    # Saturation temperature
    Tsat_C = _st.tsat_p(p_bar)
    Tsat_K = Tsat_C + 273.15

    # Saturated liquid properties
    hL = _st.hL_p(p_bar)           # kJ/kg
    cpL = _st.CpL_p(p_bar)        # kJ/kg-K
    rhoL = _st.rhoL_p(p_bar)      # kg/m3
    sigL = _st.st_p(p_bar)        # N/m surface tension
    kL = _st.tcL_p(p_bar)         # W/m-K
    muL = _st.my_pt(p_bar, Tsat_C - 1)  # Pa-s (slightly subcooled)
    nuL = muL / rhoL if rhoL > 0 else 0.0

    Lsat = SatProps(
        T=Tsat_K, h=hL * 1e3, rho=rhoL, k=kL, mu=muL, nu=nuL,
        c_p=cpL * 1e3, sig=sigL,
    )

    # Saturated vapor properties
    hV = _st.hV_p(p_bar)          # kJ/kg
    cpV = _st.CpV_p(p_bar)       # kJ/kg-K
    rhoV = _st.rhoV_p(p_bar)     # kg/m3
    kV = _st.tcV_p(p_bar)        # W/m-K
    muV = _st.my_pt(p_bar, Tsat_C + 1)  # Pa-s (slightly superheated)
    nuV = muV / rhoV if rhoV > 0 else 0.0

    Vsat = SatProps(
        T=Tsat_K, h=hV * 1e3, rho=rhoV, k=kV, mu=muV, nu=nuV,
        c_p=cpV * 1e3,
    )

    # Mixture quality
    x = (h_kJ - hL) / max(hV - hL, 1e-10)

    if x < 0:
        # Single-phase liquid
        T_C = _st.t_ph(p_bar, h_kJ)
        T_K = T_C + 273.15
        rho = _st.rho_ph(p_bar, h_kJ)
        k = _st.tc_ph(p_bar, h_kJ)
        mu = _st.my_ph(p_bar, h_kJ)
        if mu != mu:  # NaN fallback
            mu = _iapws_viscosity(T_K, rho)
        nu = mu / rho if rho > 0 else 0.0
        cp = _st.Cp_ph(p_bar, h_kJ) * 1e3  # J/kg-K
        void = 0.0

    elif x >= 1:
        # Single-phase steam
        T_C = _st.t_ph(p_bar, h_kJ)
        T_K = T_C + 273.15
        rho = _st.rho_ph(p_bar, h_kJ)
        k = _st.tc_ph(p_bar, h_kJ)
        mu = _st.my_ph(p_bar, h_kJ)
        if mu != mu:  # NaN fallback
            mu = _iapws_viscosity(T_K, rho)
        nu = mu / rho if rho > 0 else 0.0
        cp = _st.Cp_ph(p_bar, h_kJ) * 1e3
        void = 1.0

    else:
        # Two-phase mixture
        T_K = Tsat_K
        void = _st.vx_ph(p_bar, h_kJ)
        rho = rhoL * (1 - void) + rhoV * void
        k = kL * (1 - x) + kV * x
        mu = muL * (1 - x) + muV * x
        nu = mu / rho if rho > 0 else 0.0
        cp = (cpL * (1 - x) + cpV * x) * 1e3

    pro = FluidProps(
        T=T_K, p=p_MPa, h=h_Jkg, rho=rho, k=k, mu=mu, nu=nu,
        c_p=cp, void=void, x=x, Tsat=Tsat_K,
    )

    return pro, Lsat, Vsat


def h2o_enthalpy(p_MPa: float, T_K: float) -> float:
    """Compute water enthalpy (J/kg) at given pressure and temperature."""
    return _st.h_pt(p_MPa * 10, T_K - 273.15) * 1e3


def h2o_density(p_MPa: float, T_K: float) -> float:
    """Compute water density (kg/m3) at given pressure and temperature."""
    return _st.rho_pt(p_MPa * 10, T_K - 273.15)
