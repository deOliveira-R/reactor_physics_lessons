"""Predefined mixture recipes matching the MATLAB educational cases.

Each recipe loads the required isotopes and computes number densities
following the original physical setup, then returns a Mixture.
"""

from __future__ import annotations

import numpy as np
from pyXSteam.XSteam import XSteam

from orpheus.data.micro_xs import load_isotope
from .mixture import Mixture, compute_macro_xs

_steam = XSteam(XSteam.UNIT_SYSTEM_MKS)  # bar, °C, kg/m3

# Physical constants
_AMU_TO_G = 1.660538e-24  # g per atomic mass unit


def _number_density(density_g_cm3: float, molecular_weight_amu: float) -> float:
    """Convert mass density to molecular number density in 1/(barn*cm)."""
    rho = density_g_cm3 * 1e-24  # g/(barn*cm)
    return rho / _AMU_TO_G / molecular_weight_amu


def _water_density(pressure_MPa: float, temp_K: float) -> float:
    """Water density in g/cm3 via pyXSteam (expects bar, °C)."""
    return _steam.rho_pt(pressure_MPa * 10, temp_K - 273) * 1e-3


def aqueous_uranium(
    temp_K: int = 294,
    pressure_MPa: float = 0.1,
    u_conc_ppm: float = 1000.0,
) -> Mixture:
    """Homogeneous aqueous reactor: water + dissolved U-235.

    Matches MATLAB ``createH2OU``.
    """
    print(f"Building aqueous U mixture (T={temp_K}K, p={pressure_MPa}MPa, U={u_conc_ppm}ppm)")

    H01 = load_isotope("H_001", temp_K)
    O16 = load_isotope("O_016", temp_K)
    U235 = load_isotope("U_235", temp_K)

    u_conc = u_conc_ppm * 1e-6
    mol_weight = 2 * H01.aw + O16.aw + u_conc * U235.aw
    rho = _number_density(_water_density(pressure_MPa, temp_K), mol_weight)

    isotopes = [H01, O16, U235]
    densities = np.array([2 * rho, rho, rho * u_conc])

    return compute_macro_xs(isotopes, densities, escape_xs=0.0)


def pwr_like_mix() -> Mixture:
    """Homogeneous PWR-like unit cell mixture: UO2 + Zry + borated water.

    Matches MATLAB ``createPWR_like_mix``.
    """
    print("Building PWR-like homogeneous mixture")

    # --- Geometry (from input_and_initialize_PWR_like) ---
    fuel_r_out = 4.12e-3     # m
    clad_r_in = 4.22e-3      # m
    clad_r_out = 4.75e-3     # m
    pitch = 13.3e-3           # m
    r_cell = np.sqrt(pitch**2 / np.pi)

    v_frac_fuel = fuel_r_out**2 / r_cell**2
    v_frac_clad = (clad_r_out**2 - clad_r_in**2) / r_cell**2
    v_frac_cool = (r_cell**2 - clad_r_out**2) / r_cell**2

    # --- UO2 fuel (3% enrichment, 5% porosity, 600K) ---
    U235 = load_isotope("U_235", 600)
    U238 = load_isotope("U_238", 600)
    O16_fuel = load_isotope("O_016", 600)

    enrich = 0.03
    porosity = 0.05
    fuel_rho_theoretical = 10980e-3  # g/cm3 (from matpro)
    UO2_aw = U235.aw * enrich + U238.aw * (1 - enrich) + O16_fuel.aw * 2
    rho_UO2 = _number_density(fuel_rho_theoretical * (1 - porosity), UO2_aw)

    fuel_densities = np.array([
        enrich * rho_UO2 * v_frac_fuel,
        (1 - enrich) * rho_UO2 * v_frac_fuel,
        2 * rho_UO2 * v_frac_fuel,
    ])

    # --- Zircaloy cladding (600K) ---
    Zr_names = ["ZR090", "ZR091", "ZR092", "ZR094", "ZR096"]
    Zr_isos = [load_isotope(n, 600) for n in Zr_names]
    mol_fr_Zr = np.array([0.5145, 0.1122, 0.1715, 0.1738, 0.0280])
    Zry_aw = sum(iso.aw * f for iso, f in zip(Zr_isos, mol_fr_Zr))
    clad_rho_kg_m3 = 6600  # from matpro
    rho_Zry = _number_density(clad_rho_kg_m3 * 1e-3, Zry_aw)
    clad_densities = mol_fr_Zr * rho_Zry * v_frac_clad

    # --- Borated water coolant (600K, 16 MPa, 4000 ppm B) ---
    H01 = load_isotope("H_001", 600)
    O16_cool = load_isotope("O_016", 600)
    B10 = load_isotope("B_010", 600)
    B11 = load_isotope("B_011", 600)

    b_conc = 4000e-6
    mol_fr_B = np.array([0.199, 0.801])
    H2OB_aw = 2 * H01.aw + O16_cool.aw + b_conc * (mol_fr_B[0] * B10.aw + mol_fr_B[1] * B11.aw)

    # pyXSteam: rho_pt expects (bar, °C)
    rho_H2OB = _number_density(_water_density(16.0, 600), H2OB_aw)

    cool_densities = np.array([
        2 * rho_H2OB * v_frac_cool,
        rho_H2OB * v_frac_cool,
        rho_H2OB * b_conc * mol_fr_B[0] * v_frac_cool,
        rho_H2OB * b_conc * mol_fr_B[1] * v_frac_cool,
    ])

    # --- Assemble all isotopes ---
    isotopes = [U235, U238, O16_fuel] + Zr_isos + [H01, O16_cool, B10, B11]
    densities = np.concatenate([fuel_densities, clad_densities, cool_densities])

    return compute_macro_xs(isotopes, densities, escape_xs=0.0, fissile_indices=[0, 1])


# ---------------------------------------------------------------------------
# Per-material recipes (used by transport solvers: DO, MoC, MC, etc.)
# ---------------------------------------------------------------------------

# PWR geometry constants (from input_and_initialize_PWR_like.m)
_FUEL_R_OUT_CM = 0.412   # outer fuel radius in cm
_CLAD_R_OUT_CM = 0.475   # outer cladding radius in cm


def uo2_fuel(temp_K: int = 900, enrichment: float = 0.03, porosity: float = 0.05) -> Mixture:
    """Macroscopic XS for UO2 fuel with self-shielding escape XS.

    Matches MATLAB ``createUO2_03``.
    """
    print(f"Building UO2 fuel (T={temp_K}K, enrich={enrichment:.0%}, porosity={porosity:.0%})")

    U235 = load_isotope("U_235", temp_K)
    U238 = load_isotope("U_238", temp_K)
    O16 = load_isotope("O_016", temp_K)

    UO2_aw = U235.aw * enrichment + U238.aw * (1 - enrichment) + O16.aw * 2
    fuel_rho_theoretical = 10980e-3  # g/cm3 (from matpro)
    rho = _number_density(fuel_rho_theoretical * (1 - porosity), UO2_aw)

    isotopes = [U235, U238, O16]
    densities = np.array([enrichment * rho, (1 - enrichment) * rho, 2 * rho])

    escape_xs = 1.0 / (2 * _FUEL_R_OUT_CM)
    return compute_macro_xs(isotopes, densities, escape_xs=escape_xs,
                            n_legendre=2, fissile_indices=[0, 1])


def zircaloy_clad(temp_K: int = 600) -> Mixture:
    """Macroscopic XS for Zircaloy cladding with self-shielding escape XS.

    Matches MATLAB ``createZry``.
    """
    print(f"Building Zircaloy cladding (T={temp_K}K)")

    Zr_names = ["ZR090", "ZR091", "ZR092", "ZR094", "ZR096"]
    Zr_isos = [load_isotope(n, temp_K) for n in Zr_names]
    mol_fr = np.array([0.5145, 0.1122, 0.1715, 0.1738, 0.0280])

    Zry_aw = sum(iso.aw * f for iso, f in zip(Zr_isos, mol_fr))
    clad_rho = 6600 * 1e-3  # g/cm3 (from matpro)
    rho = _number_density(clad_rho, Zry_aw)

    densities = mol_fr * rho

    escape_xs = 1.0 / (2 * _CLAD_R_OUT_CM)
    return compute_macro_xs(Zr_isos, densities, escape_xs=escape_xs,
                            n_legendre=3, fissile_indices=[])


def borated_water(
    temp_K: int = 600,
    pressure_MPa: float = 16.0,
    boron_ppm: float = 4000.0,
) -> Mixture:
    """Macroscopic XS for borated water coolant (no self-shielding).

    Matches MATLAB ``createH2OB``.
    """
    print(f"Building borated water (T={temp_K}K, p={pressure_MPa}MPa, B={boron_ppm:.0f}ppm)")

    H01 = load_isotope("H_001", temp_K)
    O16 = load_isotope("O_016", temp_K)
    B10 = load_isotope("B_010", temp_K)
    B11 = load_isotope("B_011", temp_K)

    b_conc = boron_ppm * 1e-6
    mol_fr_B = np.array([0.199, 0.801])
    H2OB_aw = 2 * H01.aw + O16.aw + b_conc * (mol_fr_B[0] * B10.aw + mol_fr_B[1] * B11.aw)

    rho = _number_density(_water_density(pressure_MPa, temp_K), H2OB_aw)

    isotopes = [H01, O16, B10, B11]
    densities = np.array([
        2 * rho,
        rho,
        rho * b_conc * mol_fr_B[0],
        rho * b_conc * mol_fr_B[1],
    ])

    return compute_macro_xs(isotopes, densities, escape_xs=0.0,
                            n_legendre=3, fissile_indices=[])
