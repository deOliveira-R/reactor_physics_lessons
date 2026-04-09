#!/usr/bin/env python3
"""Coupled thermal-hydraulics and fuel behaviour under LOCA conditions -- self-contained student script.

Simulates a Loss-of-Coolant Accident in a PWR for 600 s, coupling:
  - 1-D axial thermal-hydraulics (2 nodes, two-phase flow with regime selection)
  - 1-D radial fuel/clad heat conduction
  - Cladding thermo-mechanical deformation (elastic + plastic)
  - Cladding failure detection (burst stress criterion)

Physics
-------
A PWR fuel rod is modelled with axial coolant nodes and radial fuel/clad
discretization.  The LOCA transient drives depressurization and flow
coastdown via time-dependent boundary conditions (inlet temperature,
outlet pressure, inlet velocity, and power).

Coolant thermal-hydraulics uses mixture-averaged two-phase properties with
three flow regimes: single-phase forced convection (Dittus-Boelter),
nucleate boiling (Thom), and film boiling.  The Churchill friction factor
handles friction pressure drop.

Cladding mechanics tracks elastic + plastic deformation.  A burst stress
criterion detects cladding failure, after which the inner gas pressure
equalizes with coolant pressure.

Port of MATLAB Module 09 (``thermalHydraulics.m``, ``initializeFuelRod.m``,
``initializeCoolant.m``, ``funRHS.m``, ``cladFailureEvent.m``).

Usage
-----
    python 07_thermal_hydraulics.py

Plots are saved to ``results/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from orpheus.data.materials import matpro
from orpheus.data.materials.h2o_properties import h2o_density, h2o_enthalpy, h2o_properties


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class THParams:
    """Geometry, boundary conditions and time parameters for the LOCA simulation."""

    # --- axial mesh ---
    nz: int = 2
    dz0: float = 1.5  # initial node height (m)

    # --- fuel ---
    fuel_r_in: float = 0.0       # inner fuel radius (m)
    fuel_r_out: float = 4.12e-3  # outer fuel radius (m)
    fuel_nr: int = 20            # radial nodes in fuel
    porosity: float = 0.05       # fuel porosity (-)
    burnup: float = 0.0          # fuel burnup (MWd/kgHM)
    fgr: float = 0.06            # fission gas release fraction (-)

    # --- cladding ---
    clad_r_in: float = 4.22e-3   # inner clad radius (m)
    clad_r_out: float = 4.75e-3  # outer clad radius (m)
    clad_nr: int = 5             # radial nodes in clad

    # --- gap and inner gas ---
    roughness: float = 6e-6      # effective roughness (m)
    p0_ingas: float = 1.2        # fabrication He pressure (MPa)
    v_gas_plenum: float = 10e-6  # gas plenum volume (m3)

    # --- coolant ---
    a_flow0: float = 8.914e-5    # channel flow area (m2)

    # --- time ---
    time_step: float = 1.0       # output time step (s)
    time_end: float = 600.0      # end time (s)

    # --- boundary condition tables (time, value) ---
    # Inlet temperature (K) vs. time (s)
    T0_table: np.ndarray = field(default_factory=lambda: np.array([
        [0.0,   200.0, 210.0, 5000.0],
        [553.0, 553.0, 380.0, 380.0],
    ]))
    # Outlet pressure (MPa) vs. time (s)
    p0_table: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 200.0, 200.3, 200.5, 201.4, 201.9, 202.2, 203.0, 203.8,
         206.3, 207.9, 210.4, 213.1, 216.6, 218.8, 221.0, 224.8, 229.5,
         235.8, 243.9, 250.2, 256.8, 264.1, 273.7, 291.7, 313.5, 334.0,
         5000.0],
        [15.5, 15.5, 14.65727, 13.69425, 12.27957, 11.13598, 10.41352,
         9.63108, 8.54763, 7.52432, 7.52432, 6.56115, 5.35727, 3.79223,
         2.70878, 1.80575, 1.20389, 0.72230, 0.54173, 0.39122, 0.36115,
         0.33108, 0.39122, 0.39122, 0.33108, 0.33108, 0.33108, 0.33108],
    ]))
    # Inlet velocity (m/s) vs. time (s)
    vel0_table: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 200.0, 201.0, 202.0, 203.0, 440.0, 450.0, 5000.0],
        [4.8, 4.8,   2.0,   1.0,   0.002, 0.002, 0.1,   0.1],
    ]))
    # Power (W) vs. time (s)
    power_table: np.ndarray = field(default_factory=lambda: np.array([
        [0.0, 200.0, 201.1, 211.1, 221.1, 231.1, 241.1, 251.1, 261.1,
         271.1, 281.1, 291.1, 301.1, 311.1, 321.1, 331.1, 341.1, 351.1,
         361.1, 371.1, 381.1, 391.1, 401.1, 411.1, 421.1, 431.1, 441.1,
         451.1, 461.1, 471.1, 481.1, 491.1, 5000.0],
        [69140.625, 69140.625, 6659.609, 2551.766, 2203.547, 2020.457,
         1899.090, 1809.574, 1739.305, 1681.844, 1633.465, 1591.848,
         1555.434, 1523.145, 1494.195, 1468.004, 1444.125, 1422.207,
         1401.977, 1383.207, 1365.719, 1349.359, 1334.000, 1319.535,
         1305.875, 1292.938, 1280.660, 1268.980, 1257.848, 1247.219,
         1237.051, 1227.309, 1217.961],
    ]))
    # Axial power peaking factors (per node)
    kz: np.ndarray = field(default_factory=lambda: np.ones(2))


@dataclass
class THResult:
    """Complete time-history of the thermal-hydraulics LOCA calculation."""

    time: np.ndarray  # (nt,)

    # Fuel temperature (K) -- (nz, nf, nt)
    fuel_T: np.ndarray
    # Clad temperature (K) -- (nz, nc, nt)
    clad_T: np.ndarray
    # Coolant temperature (K) -- (nz, nt)
    cool_T: np.ndarray
    # Saturation temperature (K) -- (nz, nt)
    cool_Tsat: np.ndarray
    # CHF temperature (K) -- (nz, nt)
    cool_TCHF: np.ndarray
    # Coolant pressure (MPa) -- (nz, nt)
    cool_p: np.ndarray
    # Coolant enthalpy (J/kg) -- (nz, nt)
    cool_h: np.ndarray
    # Coolant velocity (m/s) -- (nz, nt)
    cool_vel: np.ndarray
    # Void fraction (-) -- (nz, nt)
    cool_void: np.ndarray
    # Equilibrium quality (-) -- (nz, nt)
    cool_x: np.ndarray
    # Flow regime (1/2/3) -- (nz, nt)
    cool_regime: np.ndarray

    # Inner gas pressure (MPa) -- (nt,)
    ingas_p: np.ndarray
    # Fuel outer radius (m) -- (nz, nt)
    fuel_r: np.ndarray
    # Clad inner/outer radius (m) -- (nz, nt)
    clad_r_in: np.ndarray
    clad_r_out: np.ndarray

    # Cladding stresses (MPa) -- (nz, nc, nt)
    clad_sig_r: np.ndarray
    clad_sig_h: np.ndarray
    clad_sig_z: np.ndarray
    # Engineering & burst stress (MPa) -- (nz, nt)
    clad_sig_I: np.ndarray
    clad_sig_B: np.ndarray

    # Effective plastic strain (fraction) -- (nz, nc, nt)
    clad_epsPeff: np.ndarray

    # Clad failure time (s), NaN if no failure
    clad_fail_time: float

    params: THParams = field(repr=False)


# ============================================================================
# Helpers
# ============================================================================


def _interp_between(y: np.ndarray) -> np.ndarray:
    """Midpoint interpolation between adjacent nodes (last axis)."""
    return 0.5 * (y[..., :-1] + y[..., 1:])


def _von_mises(sig_r: np.ndarray, sig_h: np.ndarray, sig_z: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5 * (sig_r - sig_h) ** 2
        + 0.5 * (sig_r - sig_z) ** 2
        + 0.5 * (sig_h - sig_z) ** 2
    )


def _interp_bc(table: np.ndarray, t: float) -> float:
    """Interpolate a boundary condition table at time *t*."""
    return float(np.interp(t, table[0], table[1]))


# ============================================================================
# Initialization
# ============================================================================


def _initialize_th(
    params: THParams,
) -> tuple[dict, np.ndarray]:
    """Build internal parameter dict and initial ODE state vector.

    Returns
    -------
    p : dict  -- precomputed constants carried through the integration.
    y0 : 1-D array -- initial ODE state vector.
    """
    nz = params.nz
    nf = params.fuel_nr
    nc = params.clad_nr
    dz0 = params.dz0

    # --- fuel mesh ---
    fuel_dr0 = (params.fuel_r_out - params.fuel_r_in) / (nf - 1)
    fuel_r0 = np.linspace(params.fuel_r_in, params.fuel_r_out, nf)
    fuel_r0_bnd = np.concatenate(
        ([params.fuel_r_in], _interp_between(fuel_r0), [params.fuel_r_out])
    )
    # volume per node, per axial level (nz, nf)
    fuel_v0_1d = np.pi * np.diff(fuel_r0_bnd ** 2) * dz0  # (nf,)
    fuel_v0 = np.tile(fuel_v0_1d, (nz, 1))                # (nz, nf)
    fuel_mass = matpro.UO2_RHO * (1.0 - params.porosity) * fuel_v0

    # initial power
    pow0 = params.power_table[1, 0]
    lhgr0 = pow0 / (dz0 * nz)
    fuel_qV = np.outer(
        lhgr0 * dz0 * params.kz / fuel_v0.sum(axis=1),
        np.ones(nf),
    )  # (nz, nf)

    # radial heat transfer areas between fuel nodes (nf-1,)
    fuel_a_bnd = 2 * np.pi * _interp_between(fuel_r0) * dz0

    # --- clad mesh ---
    clad_dr0 = (params.clad_r_out - params.clad_r_in) / (nc - 1)
    clad_r0 = np.linspace(params.clad_r_in, params.clad_r_out, nc)
    clad_r0_bnd = np.concatenate(
        ([params.clad_r_in], _interp_between(clad_r0), [params.clad_r_out])
    )
    clad_v0_1d = np.pi * np.diff(clad_r0_bnd ** 2) * dz0  # (nc,)
    clad_v0 = np.tile(clad_v0_1d, (nz, 1))                # (nz, nc)

    # radial heat transfer areas between clad nodes (nc-1,)
    clad_a_bnd = 2 * np.pi * _interp_between(clad_r0) * dz0

    # --- gap ---
    gap_dr0 = params.clad_r_in - params.fuel_r_out
    gap_r0_mid = 0.5 * (params.clad_r_in + params.fuel_r_out)
    gap_a_bnd = 2 * np.pi * gap_r0_mid * dz0  # scalar

    # --- coolant geometry ---
    a_flow0 = params.a_flow0
    a_flow = np.full(nz, a_flow0)
    vol_flow = a_flow * dz0
    area_HX = 2 * np.pi * params.clad_r_out * dz0
    d_hyd = 4 * vol_flow / area_HX

    # --- inner gas initial moles ---
    v_gap0 = dz0 * nz * np.pi * (params.clad_r_in ** 2 - params.fuel_r_out ** 2)
    v_void0 = dz0 * nz * np.pi * params.fuel_r_in ** 2
    mu_He0 = (params.p0_ingas * 1e6) * (params.v_gas_plenum + v_gap0 + v_void0) / (8.31 * 293.0)

    # --- initial coolant state ---
    p0_cool = params.p0_table[1, 0]
    T0_cool = params.T0_table[1, 0]
    h0_cool = h2o_enthalpy(p0_cool, T0_cool)  # J/kg

    # --- pack internal params ---
    p = dict(
        nz=nz, nf=nf, nc=nc, dz0=dz0,
        fuel_r0=fuel_r0, fuel_dr0=fuel_dr0, fuel_r0_bnd=fuel_r0_bnd,
        fuel_v0=fuel_v0, fuel_mass=fuel_mass, fuel_a_bnd=fuel_a_bnd,
        clad_r0=clad_r0, clad_dr0=clad_dr0, clad_r0_bnd=clad_r0_bnd,
        clad_v0=clad_v0, clad_a_bnd=clad_a_bnd,
        gap_dr0=gap_dr0, gap_r0_mid=gap_r0_mid, gap_a_bnd=gap_a_bnd,
        roughness=params.roughness,
        a_flow0=a_flow0, a_flow=a_flow, vol_flow=vol_flow,
        area_HX=area_HX, d_hyd=d_hyd,
        mu_He0=mu_He0, v_gas_plenum=params.v_gas_plenum,
        porosity=params.porosity, burnup=params.burnup,
        fuel_r_in=params.fuel_r_in, fuel_r_out=params.fuel_r_out,
        clad_r_in_fab=params.clad_r_in, clad_r_out_fab=params.clad_r_out,
        T0_table=params.T0_table, p0_table=params.p0_table,
        vel0_table=params.vel0_table, power_table=params.power_table,
        kz=params.kz,
        clad_fail=False,
    )

    # --- initial state vector ---
    # Layout: fuel_T (nz*nf), clad_T (nz*nc), cool_h (nz),
    #         clad_epsPeff (nz*nc), clad_epsP (nz*nc*3)
    n_ode = nz * nf + nz * nc + nz + nz * nc + nz * nc * 3
    y0 = np.zeros(n_ode)
    y0[: nz * nf] = T0_cool            # fuel temperatures
    y0[nz * nf : nz * nf + nz * nc] = T0_cool  # clad temperatures
    y0[nz * nf + nz * nc : nz * nf + nz * nc + nz] = h0_cool  # coolant enthalpy
    # plastic strains start at zero

    return p, y0


# ============================================================================
# State vector packing / unpacking
# ============================================================================

def _unpack(y: np.ndarray, p: dict) -> dict:
    """Unpack flat ODE state vector into named 2-D arrays."""
    nz, nf, nc = p["nz"], p["nf"], p["nc"]
    idx = 0

    fuel_T = y[idx : idx + nz * nf].reshape(nz, nf)
    idx += nz * nf

    clad_T = y[idx : idx + nz * nc].reshape(nz, nc)
    idx += nz * nc

    cool_h = y[idx : idx + nz].copy()
    idx += nz

    clad_epsPeff = y[idx : idx + nz * nc].reshape(nz, nc)
    idx += nz * nc

    clad_epsP = np.empty((3, nz, nc))
    for i in range(3):
        clad_epsP[i] = y[idx : idx + nz * nc].reshape(nz, nc)
        idx += nz * nc

    return dict(
        fuel_T=fuel_T, clad_T=clad_T, cool_h=cool_h,
        clad_epsPeff=clad_epsPeff, clad_epsP=clad_epsP,
    )


def _pack_rates(
    rate_fuel_T: np.ndarray,    # (nz, nf)
    rate_clad_T: np.ndarray,    # (nz, nc)
    rate_cool_h: np.ndarray,    # (nz,)
    rate_epsPeff: np.ndarray,   # (nz, nc)
    rate_epsP: list,            # [3 x (nz, nc)]
) -> np.ndarray:
    """Pack rates into a flat vector matching the state layout."""
    parts = [
        rate_fuel_T.ravel(),
        rate_clad_T.ravel(),
        rate_cool_h,
        rate_epsPeff.ravel(),
    ]
    for i in range(3):
        parts.append(rate_epsP[i].ravel())
    return np.concatenate(parts)


# ============================================================================
# Algebraic pressure computation
# ============================================================================


def _compute_pressure(cool_h, cool_props, vel0, T0, p0, p):
    """Compute coolant pressure at each axial node algebraically.

    Uses the outlet pressure boundary condition and cumulative pressure drops
    from gravity, wall friction, and acceleration.

    Returns (pressure, velocity, mdot, Re) arrays.
    """
    nz = p["nz"]
    dz0 = p["dz0"]
    a_flow = p["a_flow"]
    a_flow0 = p["a_flow0"]
    d_hyd = p["d_hyd"]

    rho_inlet = h2o_density(p0, T0)

    # mass flow rate at inlet
    mdot0 = vel0 * a_flow0 * rho_inlet

    # mass flow rate at junctions (nz+1,) -- uniform
    mdot_jnc = np.full(nz + 1, mdot0)

    # mass flow rate in nodes (nz,) -- midpoint interpolation
    mdot = 0.5 * (mdot_jnc[:-1] + mdot_jnc[1:])

    # extract per-node density and kinematic viscosity
    rho = np.array([pro.rho for pro in cool_props])
    nu = np.array([pro.nu for pro in cool_props])

    # node velocities
    vel = mdot / rho / a_flow

    # Reynolds number
    Re = d_hyd * np.abs(vel) / nu

    # Churchill friction factor
    theta1 = -2.457 * np.log((7.0 / Re) ** 0.9)
    theta2 = 37530.0 / Re
    f_wall = 8 * ((8.0 / Re) ** 12 + 1.0 / (theta1 ** 16 + theta2 ** 16) ** 1.5) ** (1.0 / 12.0)

    # wall friction pressure drop (MPa)
    dp_fric = f_wall * dz0 / (2 * d_hyd) * rho * vel ** 2 / 2 * 1e-6

    # gravity pressure drop (MPa)
    dp_grav = rho * 9.81e-6 * dz0

    # acceleration pressure drop at junctions (nz-1,)
    dp_accel_jnc = -np.diff(rho * vel ** 2) * 1e-6

    # junction pressures from outlet boundary upward:
    # p_jnc[nz] = p0, p_jnc[i] = p_jnc[i+1] + dp_grav[i] + dp_fric[i]
    p_jnc = np.empty(nz + 1)
    p_jnc[nz] = p0
    for i in range(nz - 1, -1, -1):
        p_jnc[i] = p_jnc[i + 1] + dp_grav[i] + dp_fric[i]

    # node pressures (midpoint interpolation + acceleration correction)
    p_node = 0.5 * (p_jnc[:-1] + p_jnc[1:])
    # acceleration correction: cumsum from top of dp_accel
    if nz > 1:
        accel_corr = np.zeros(nz)
        accel_corr[:-1] = np.cumsum(dp_accel_jnc[::-1])[::-1]
        p_node += accel_corr

    return p_node, vel, mdot, Re


# ============================================================================
# Algebraic cladding stress solver
# ============================================================================


def _solve_clad_stress(clad_T, clad_epsP, p_cool, p_ingas, p):
    """Solve for cladding stress components at each axial level.

    For each axial node independently, solve a linear system of
    3*(nc-1) + 3 = 3*nc equations for sig_r, sig_h, sig_z.

    Returns dict with sig_r, sig_h, sig_z arrays (nz, nc).
    """
    nz, nc = p["nz"], p["nc"]
    clad_r0 = p["clad_r0"]
    clad_dr0 = p["clad_dr0"]
    clad_r_mid = _interp_between(clad_r0)

    sig_r = np.zeros((nz, nc))
    sig_h = np.zeros((nz, nc))
    sig_z = np.zeros((nz, nc))

    for iz in range(nz):
        T = clad_T[iz]          # (nc,)
        epsP = clad_epsP[:, iz, :]  # (3, nc)

        # material properties
        E = matpro.zry_E(T)     # (nc,) MPa
        nu_c = matpro.ZRY_NU
        epsT = matpro.zry_thexp(T)  # (nc,)

        # non-elastic strain
        epsNE = np.empty((3, nc))
        for i in range(3):
            epsNE[i] = epsT + epsP[i]

        # compliance
        Cs = 1.0 / E
        Cc = -nu_c / E

        n_unk = 3 * nc
        A = np.zeros((n_unk, n_unk))
        b = np.zeros(n_unk)

        # column offsets
        ir = 0
        ih = nc
        izz = 2 * nc

        row = 0

        # --- stress equilibrium: nc-1 equations ---
        for j in range(nc - 1):
            c_diff = 1.0 / clad_dr0
            c_avg = 0.5 / clad_r_mid[j]
            A[row, ir + j] = -c_diff + c_avg
            A[row, ir + j + 1] = c_diff + c_avg
            A[row, ih + j] = -c_avg
            A[row, ih + j + 1] = -c_avg
            row += 1

        # --- strain compatibility: nc-1 equations ---
        for j in range(nc - 1):
            jp = j + 1
            Cs_j, Cs_jp = Cs[j], Cs[jp]
            Cc_j, Cc_jp = Cc[j], Cc[jp]
            r_ = clad_r_mid[j]
            dr = clad_dr0

            # diff(eps_h)/dr
            A[row, ih + j] += -Cs_j / dr
            A[row, ih + jp] += Cs_jp / dr
            A[row, ir + j] += -Cc_j / dr
            A[row, ir + jp] += Cc_jp / dr
            A[row, izz + j] += -Cc_j / dr
            A[row, izz + jp] += Cc_jp / dr

            # -(eps_r_ - eps_h_)/r_
            dC_j = Cs_j - Cc_j
            dC_jp = Cs_jp - Cc_jp
            A[row, ir + j] += -0.5 * dC_j / r_
            A[row, ir + jp] += -0.5 * dC_jp / r_
            A[row, ih + j] += 0.5 * dC_j / r_
            A[row, ih + jp] += 0.5 * dC_jp / r_

            dNE_h = epsNE[1, jp] - epsNE[1, j]
            rhs_ne = epsNE[0, j] - epsNE[1, j] + epsNE[0, jp] - epsNE[1, jp]
            b[row] = -dNE_h / dr + 0.5 * rhs_ne / r_
            row += 1

        # --- axial strain constant: nc-1 equations ---
        for j in range(nc - 1):
            jp = j + 1
            A[row, izz + j] += Cs[j]
            A[row, ir + j] += Cc[j]
            A[row, ih + j] += Cc[j]
            A[row, izz + jp] += -Cs[jp]
            A[row, ir + jp] += -Cc[jp]
            A[row, ih + jp] += -Cc[jp]
            b[row] = -(epsNE[2, j] - epsNE[2, jp])
            row += 1

        # --- BC1: sig_r(outer) = -p_coolant ---
        A[row, ir + nc - 1] = 1.0
        b[row] = -p_cool[iz]
        row += 1

        # --- BC2: sig_r(inner) = -p_ingas ---
        A[row, ir + 0] = 1.0
        b[row] = -p_ingas
        row += 1

        # --- BC3: integral of sig_z = p_inner*r_inner^2 - p_outer*r_outer^2 ---
        clad_dr2 = np.diff(clad_r0 ** 2)
        for j in range(nc - 1):
            A[row, izz + j] += 0.5 * clad_dr2[j]
            A[row, izz + j + 1] += 0.5 * clad_dr2[j]
        b[row] = p_ingas * clad_r0[0] ** 2 - p_cool[iz] * clad_r0[-1] ** 2
        row += 1

        assert row == n_unk

        sigma = np.linalg.solve(A, b)
        sig_r[iz] = sigma[ir : ir + nc]
        sig_h[iz] = sigma[ih : ih + nc]
        sig_z[iz] = sigma[izz : izz + nc]

    return dict(sig_r=sig_r, sig_h=sig_h, sig_z=sig_z)


# ============================================================================
# Wall heat transfer with regime selection
# ============================================================================


def _wall_heat_transfer(Twall, cool_props, Lsat_arr, Vsat_arr, cool_p, cool_h, Re, d_hyd):
    """Compute wall heat flux (W/m2) with flow regime selection.

    Returns (qw, regime, TCHF) arrays of shape (nz,).
    """
    nz = len(Twall)

    T_mix = np.array([pro.T for pro in cool_props])
    Tsat = np.array([pro.Tsat for pro in cool_props])
    void = np.array([pro.void for pro in cool_props])
    k_mix = np.array([pro.k for pro in cool_props])
    cp_mix = np.array([pro.c_p for pro in cool_props])
    mu_mix = np.array([pro.mu for pro in cool_props])

    rhoL = np.array([L.rho for L in Lsat_arr])
    rhoV = np.array([V.rho for V in Vsat_arr])
    hL = np.array([L.h for L in Lsat_arr])
    hV = np.array([V.h for V in Vsat_arr])
    kV = np.array([V.k for V in Vsat_arr])
    cpV = np.array([V.c_p for V in Vsat_arr])
    nuV = np.array([V.nu for V in Vsat_arr])
    sigL = np.array([L.sig for L in Lsat_arr])

    dTw = Twall - T_mix
    dTwSat = Twall - Tsat
    drhoSat = rhoL - rhoV
    dhLat = hV - hL
    dhSub = np.maximum(hL - cool_h, 0.0)

    # Prandtl number
    Pr = cp_mix * mu_mix / k_mix

    # subcooling and void correction multipliers
    kSub = 1.0 + 0.1 * (rhoL / rhoV) ** 0.75 * dhSub / dhLat
    kVoid = 1.0 - void

    # Single-phase forced convection (Dittus-Boelter)
    qw_1phase = np.maximum(4.36, 0.023 * Re ** 0.8 * Pr ** 0.4) * (k_mix / d_hyd) * dTw

    # Nucleate boiling (Thom)
    qw_NB = 2000 * np.abs(dTwSat) * dTwSat * np.exp(cool_p / 4.34)

    # Critical heat flux
    qw_CHF = (
        0.14 * dhLat
        * (9.81 * sigL * rhoV ** 2 * drhoSat) ** 0.25
        * kSub * kVoid
    )

    # Wall temperature at CHF
    TCHF = Tsat + np.sqrt(np.maximum(qw_CHF * np.exp(-cool_p / 4.34) / 2000, 0.0))

    # Film boiling
    qw_FB = (
        0.25
        * (9.81 * kV ** 2 * cpV * drhoSat / nuV) ** 0.333
        * kSub * dTwSat
    )

    # Regime selection
    qw = np.zeros(nz)
    regime = np.zeros(nz, dtype=int)

    Tsat_L = np.array([L.T for L in Lsat_arr])

    for iz in range(nz):
        if Twall[iz] < Tsat_L[iz]:
            # Single-phase liquid forced convection
            regime[iz] = 1
            qw[iz] = qw_1phase[iz]
        elif Twall[iz] < TCHF[iz]:
            # Pre-CHF boiling
            regime[iz] = 2
            qw[iz] = max(qw_1phase[iz], qw_NB[iz])
        else:
            # Post-CHF film boiling or single-phase steam
            regime[iz] = 3
            qw[iz] = max(qw_1phase[iz] * (1 - kVoid[iz]),
                         qw_FB[iz] * kVoid[iz])

    return qw, regime, TCHF


# ============================================================================
# RHS function
# ============================================================================


def _rhs(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    """Right-hand side of the ODE system."""
    nz, nf, nc = p["nz"], p["nf"], p["nc"]
    dz0 = p["dz0"]

    state = _unpack(y, p)
    fuel_T = state["fuel_T"]     # (nz, nf)
    clad_T = state["clad_T"]     # (nz, nc)
    cool_h = state["cool_h"]     # (nz,)
    clad_epsPeff = state["clad_epsPeff"]  # (nz, nc)
    clad_epsP = state["clad_epsP"]        # (3, nz, nc)

    # ======================================================================
    # THERMAL HYDRAULICS
    # ======================================================================

    # Interpolate boundary conditions
    vel0 = _interp_bc(p["vel0_table"], t)
    T0 = _interp_bc(p["T0_table"], t)
    p0 = _interp_bc(p["p0_table"], t)
    p0 = max(p0, 0.05)  # floor to avoid sub-atmospheric singularities

    # Water properties at each node -- use previous-step pressure estimate
    # For the first call, use outlet pressure as estimate
    cool_props = []
    Lsat_arr = []
    Vsat_arr = []

    # First pass: estimate pressure from outlet BC
    p_est = np.full(nz, p0)

    for iz in range(nz):
        pro, Lsat, Vsat = h2o_properties(max(p_est[iz], 0.05), cool_h[iz])
        cool_props.append(pro)
        Lsat_arr.append(Lsat)
        Vsat_arr.append(Vsat)

    # Compute pressure algebraically
    cool_p, cool_vel, cool_mdot, cool_Re = _compute_pressure(
        cool_h, cool_props, vel0, T0, p0, p
    )

    # Recompute water properties with corrected pressure
    cool_props.clear()
    Lsat_arr.clear()
    Vsat_arr.clear()
    for iz in range(nz):
        pro, Lsat, Vsat = h2o_properties(max(cool_p[iz], 0.05), cool_h[iz])
        cool_props.append(pro)
        Lsat_arr.append(Lsat)
        Vsat_arr.append(Vsat)

    # Recompute velocity and Re with corrected density
    rho_corr = np.array([pro.rho for pro in cool_props])
    nu_corr = np.array([pro.nu for pro in cool_props])
    cool_vel = cool_mdot / rho_corr / p["a_flow"]
    cool_Re = p["d_hyd"] * np.abs(cool_vel) / nu_corr

    # ======================================================================
    # FUEL STRAIN (simplified -- only thermal expansion)
    # ======================================================================
    fuel_Tavg = np.sum(p["fuel_v0"] * fuel_T, axis=1) / np.sum(p["fuel_v0"], axis=1)
    fuel_epsT = matpro.uo2_thexp(fuel_Tavg)  # (nz,)
    fuel_r = p["fuel_r_out"] * (1.0 + fuel_epsT)  # (nz,) deformed outer radius
    fuel_dz = dz0 * (1.0 + fuel_epsT)  # (nz,) deformed node height

    # ======================================================================
    # CLAD STRAINS AND STRAIN RATES
    # ======================================================================
    clad_Tavg = np.sum(p["clad_v0"] * clad_T, axis=1) / np.sum(p["clad_v0"], axis=1)

    # Thermal strain
    clad_epsT = matpro.zry_thexp(clad_T)  # (nz, nc)

    # We need to solve for stresses to compute elastic strains and geometry.
    # First, compute inner gas pressure.

    # ======================================================================
    # GAP AND INNER GAS
    # ======================================================================
    gap_T = 0.5 * (fuel_T[:, -1] + clad_T[:, 0])  # (nz,)
    # Gas gap volume using fabricated clad inner radius (deformation is small)
    v_gas_gap = fuel_dz * np.pi * (p["clad_r_in_fab"] ** 2 - fuel_r ** 2)
    v_gas_gap = np.maximum(v_gas_gap, 1e-15)  # prevent negative volumes

    ingas_Tplenum = np.array([pro.T for pro in cool_props])[-1]

    ingas_p = (
        p["mu_He0"] * 8.31e-6
        / (p["v_gas_plenum"] / ingas_Tplenum + np.sum(v_gas_gap / gap_T))
    )

    # Clad failure: inner gas pressure = coolant pressure
    if p["clad_fail"]:
        ingas_p = cool_p[-1]

    # ======================================================================
    # CLAD STRESS (algebraic)
    # ======================================================================
    stress = _solve_clad_stress(clad_T, clad_epsP, cool_p, ingas_p, p)
    sig_r = stress["sig_r"]  # (nz, nc)
    sig_h = stress["sig_h"]
    sig_z = stress["sig_z"]

    # Store for event function access
    p["_last_stress"] = stress
    p["_last_ingas_p"] = ingas_p
    p["_last_cool_p"] = cool_p.copy()
    p["_last_clad_Tavg"] = clad_Tavg.copy()

    # ======================================================================
    # CLAD STRAIN COMPUTATION AND GEOMETRY UPDATE
    # ======================================================================
    sig_sum = sig_r + sig_h + sig_z

    clad_epsE = np.empty((3, nz, nc))
    clad_eps = np.empty((3, nz, nc))
    sigs = [sig_r, sig_h, sig_z]
    E_clad = matpro.zry_E(clad_T)  # (nz, nc)

    for i in range(3):
        clad_epsE[i] = (sigs[i] - matpro.ZRY_NU * (sig_sum - sigs[i])) / E_clad
        clad_eps[i] = clad_epsT + clad_epsE[i] + clad_epsP[i]

    # Von Mises stress
    sigVM = _von_mises(sig_r, sig_h, sig_z) + 1e-6

    # Effective plastic strain rate
    K_zry = matpro.zry_K(clad_T)
    m_zry = matpro.zry_m(clad_T)
    n_zry = matpro.zry_n(clad_T)

    fail_mult = 0.0 if p["clad_fail"] else 1.0
    rate_epsPeff = fail_mult * np.minimum(
        0.1,
        1e-3 * (sigVM * 1e6 / K_zry / np.abs(clad_epsPeff + 1e-6) ** n_zry) ** (1.0 / m_zry),
    )

    rate_epsP = []
    for i in range(3):
        rate_epsP.append(
            rate_epsPeff * 1.5 * (sigs[i] - sig_sum / 3.0) / sigVM
        )

    # Update clad geometry
    clad_eps_bnd = [_interp_between(clad_eps[i]) for i in range(3)]  # (nz, nc-1) each

    clad_r = np.tile(p["clad_r0"], (nz, 1)) * (1.0 + clad_eps[1])  # hoop -> radial position
    clad_r_bnd = _interp_between(clad_r)  # (nz, nc-1)
    clad_r_plus = np.column_stack([clad_r[:, 0], clad_r_bnd, clad_r[:, -1]])  # (nz, nc+1)

    clad_dz = np.tile(np.full(nc, dz0), (nz, 1)) * (1.0 + clad_eps[2])  # (nz, nc)
    clad_a_bnd_def = 2 * np.pi * clad_r_bnd * _interp_between(clad_dz)   # (nz, nc-1)
    clad_v = np.pi * np.diff(clad_r_plus ** 2, axis=1) * clad_dz  # (nz, nc)

    # Deformed clad heat transfer areas (nz, nc-1) -- used for heat conduction
    clad_a_bnd = clad_a_bnd_def  # computed on the line above

    # Store clad geometry for result extraction
    p["_last_clad_r"] = clad_r
    p["_last_fuel_r"] = fuel_r

    # ======================================================================
    # GAP GEOMETRY UPDATE
    # ======================================================================
    gap_dr = clad_r[:, 0] - fuel_r  # (nz,)
    gap_open = np.all(gap_dr > 0)

    # Deformed gap geometry
    gap_r_mid = 0.5 * (clad_r[:, 0] + fuel_r)  # (nz,)
    gap_a_bnd = 2 * np.pi * gap_r_mid * 0.5 * (clad_dz[:, 0] + fuel_dz)  # (nz,)

    gap_kGasMix = matpro.k_He(gap_T)  # (nz,)
    if gap_open:
        gap_h = gap_kGasMix / np.maximum(gap_dr, 1e-9)
    else:
        gap_h = gap_kGasMix / p["roughness"]

    # ======================================================================
    # ENGINEERING STRESS (for failure detection)
    # ======================================================================
    clad_sigI = (
        (ingas_p - cool_p)
        * (clad_r[:, -1] + clad_r[:, 0]) / 2.0
        / (clad_r[:, -1] - clad_r[:, 0])
    )
    p["_last_sigI"] = clad_sigI.copy()

    # ======================================================================
    # WALL HEAT EXCHANGE
    # ======================================================================
    Twall = clad_T[:, -1]  # outer clad surface temperature
    qw, regime, TCHF = _wall_heat_transfer(
        Twall, cool_props, Lsat_arr, Vsat_arr, cool_p, cool_h, cool_Re, p["d_hyd"],
    )

    # Store for result extraction
    p["_last_cool_props"] = cool_props
    p["_last_Lsat"] = Lsat_arr
    p["_last_Vsat"] = Vsat_arr
    p["_last_regime"] = regime.copy()
    p["_last_TCHF"] = TCHF.copy()
    p["_last_cool_vel"] = cool_vel.copy()

    # ======================================================================
    # COOLANT ENTHALPY RATE
    # ======================================================================
    h0_cool = h2o_enthalpy(max(p0, 0.05), T0)

    mdot_jnc = np.full(nz + 1, cool_mdot[0])  # uniform mass flow
    h_jnc = np.concatenate(([h0_cool], cool_h))  # upwind scheme

    area_HX = p["area_HX"]
    vol_flow = p["vol_flow"]
    rho = np.array([pro.rho for pro in cool_props])

    rate_cool_h = (
        -np.diff(mdot_jnc * h_jnc) + qw * area_HX
    ) / (rho * vol_flow)

    # ======================================================================
    # FUEL ROD POWER
    # ======================================================================
    pow_t = _interp_bc(p["power_table"], t)
    lhgr = pow_t / (dz0 * nz)
    fuel_qV = np.outer(
        lhgr * dz0 * p["kz"] / p["fuel_v0"].sum(axis=1),
        np.ones(nf),
    )  # (nz, nf)

    # ======================================================================
    # FUEL ROD THERMAL CALCULATIONS
    # ======================================================================
    # Temperature between fuel nodes (nz, nf-1)
    Tf_bnd = _interp_between(fuel_T)
    # Temperature between clad nodes (nz, nc-1)
    Tc_bnd = _interp_between(clad_T)

    # Gap heat flux (W/m2)
    qGap = gap_h * (fuel_T[:, -1] - clad_T[:, 0])

    # Fuel radial heat flux between nodes (nz, nf-1) (W/m2)
    fuel_dr0 = p["fuel_dr0"]
    qFuel_bnd = -matpro.uo2_k(Tf_bnd, p["burnup"], p["porosity"]) * np.diff(fuel_T, axis=1) / fuel_dr0

    # Fuel heat transfer with BCs: inner=0 (adiabatic), outer=gap
    # (nz, nf+1) array of heat flows at boundaries
    QFuel_bnd = np.column_stack([
        np.zeros(nz),
        qFuel_bnd * p["fuel_a_bnd"],  # (nz, nf-1) -- fuel area stays initial (MATLAB convention)
        qGap * gap_a_bnd,             # (nz,) -- deformed gap area
    ])  # (nz, nf+1)

    # Clad radial heat flux between nodes (nz, nc-1) (W/m2)
    clad_dr0_val = p["clad_dr0"]
    qClad_bnd = -matpro.zry_k(Tc_bnd) * np.diff(clad_T, axis=1) / clad_dr0_val

    # Clad heat transfer with BCs: inner=gap, outer=wall
    QClad_bnd = np.column_stack([
        qGap * gap_a_bnd,             # (nz,) -- deformed gap area
        qClad_bnd * clad_a_bnd,       # (nz, nc-1) -- deformed clad area
        qw * area_HX,                 # (nz,)
    ])  # (nz, nc+1)

    # Rate of fuel temperature
    rate_fuel_T = (-np.diff(QFuel_bnd, axis=1) + fuel_qV * p["fuel_v0"]) / (
        p["fuel_mass"] * matpro.uo2_cp(fuel_T)
    )

    # Rate of clad temperature
    rate_clad_T = -np.diff(QClad_bnd, axis=1) / (
        matpro.ZRY_RHO * matpro.zry_cp(clad_T) * p["clad_v0"]
    )

    return _pack_rates(rate_fuel_T, rate_clad_T, rate_cool_h, rate_epsPeff, rate_epsP)


# ============================================================================
# Cladding failure event
# ============================================================================


def _clad_failure_event(t: float, y: np.ndarray, p: dict) -> float:
    """Event function: returns sigB(Tavg) - sigI.

    When this crosses zero from above, cladding has failed (burst stress
    exceeded by engineering stress).
    """
    # Use cached values from last RHS evaluation
    clad_Tavg = p.get("_last_clad_Tavg")
    sigI = p.get("_last_sigI")

    if clad_Tavg is None or sigI is None:
        # Not yet computed -- return large positive value
        return 1e6

    sigB = matpro.zry_burst_stress(clad_Tavg)
    # Use outlet node (last axial level), matching MATLAB cladFailureEvent.m
    return float(sigB[-1] - sigI[-1])


_clad_failure_event.terminal = True
_clad_failure_event.direction = -1  # trigger when burst stress is reached (positive -> negative)


# ============================================================================
# Main solver
# ============================================================================


def solve_thermal_hydraulics(params: THParams | None = None) -> THResult:
    """Run the LOCA thermal-hydraulics simulation.

    Parameters
    ----------
    params : THParams, optional
        Simulation parameters. Uses defaults (PWR LOCA) if not provided.

    Returns
    -------
    THResult with time histories of all key variables.
    """
    if params is None:
        params = THParams()

    p, y0 = _initialize_th(params)

    time_end = params.time_end
    dt = params.time_step
    t_eval = np.arange(0, time_end + dt, dt)

    # Storage for results at each output step
    records = {
        "time": [], "fuel_T": [], "clad_T": [],
        "cool_T": [], "cool_Tsat": [], "cool_TCHF": [],
        "cool_p": [], "cool_h": [], "cool_vel": [],
        "cool_void": [], "cool_x": [], "cool_regime": [],
        "ingas_p": [], "fuel_r": [],
        "clad_r_in": [], "clad_r_out": [],
        "clad_sig_r": [], "clad_sig_h": [], "clad_sig_z": [],
        "clad_sig_I": [], "clad_sig_B": [],
        "clad_epsPeff": [],
    }

    def _record_snapshot(t_val, y_val):
        """Evaluate the RHS once to populate cached fields, then record."""
        _rhs(t_val, y_val, p)

        state = _unpack(y_val, p)
        records["time"].append(t_val)
        records["fuel_T"].append(state["fuel_T"].copy())
        records["clad_T"].append(state["clad_T"].copy())
        records["cool_h"].append(state["cool_h"].copy())
        records["clad_epsPeff"].append(state["clad_epsPeff"].copy())

        cool_props = p.get("_last_cool_props", [])
        records["cool_T"].append(np.array([pro.T for pro in cool_props]))
        records["cool_Tsat"].append(np.array([pro.Tsat for pro in cool_props]))
        records["cool_p"].append(p.get("_last_cool_p", np.zeros(params.nz)))
        records["cool_vel"].append(p.get("_last_cool_vel", np.zeros(params.nz)))
        records["cool_void"].append(np.array([pro.void for pro in cool_props]))
        records["cool_x"].append(np.array([pro.x for pro in cool_props]))
        records["cool_regime"].append(p.get("_last_regime", np.zeros(params.nz)))
        records["cool_TCHF"].append(p.get("_last_TCHF", np.zeros(params.nz)))
        records["ingas_p"].append(p.get("_last_ingas_p", 0.0))
        records["fuel_r"].append(p.get("_last_fuel_r", np.zeros(params.nz)))

        clad_r = p.get("_last_clad_r", np.zeros((params.nz, params.clad_nr)))
        records["clad_r_in"].append(clad_r[:, 0].copy())
        records["clad_r_out"].append(clad_r[:, -1].copy())

        stress = p.get("_last_stress", {})
        records["clad_sig_r"].append(stress.get("sig_r", np.zeros((params.nz, params.clad_nr))))
        records["clad_sig_h"].append(stress.get("sig_h", np.zeros((params.nz, params.clad_nr))))
        records["clad_sig_z"].append(stress.get("sig_z", np.zeros((params.nz, params.clad_nr))))
        records["clad_sig_I"].append(p.get("_last_sigI", np.zeros(params.nz)))

        clad_Tavg = p.get("_last_clad_Tavg", np.full(params.nz, 553.0))
        records["clad_sig_B"].append(matpro.zry_burst_stress(clad_Tavg))

    # --- Phase 1: integrate until clad failure or end ---
    clad_fail_time = np.nan

    t_eval_1 = t_eval[t_eval <= time_end]

    # Chunked integration with manual event detection (avoids scipy brentq
    # sign errors).  Each chunk spans one output interval (time_step).
    t_cur = 0.0
    y_cur = y0.copy()

    # Evaluate event at t=0
    _rhs(t_cur, y_cur, p)
    ev_prev = _clad_failure_event(t_cur, y_cur, p)

    for t_next in t_eval_1:
        if t_next <= t_cur:
            continue

        sol_chunk = solve_ivp(
            fun=lambda t, y: _rhs(t, y, p),
            t_span=(t_cur, t_next),
            y0=y_cur,
            method="BDF",
            rtol=1e-6,
            atol=1e-4,
            max_step=10.0,
        )

        if not sol_chunk.success:
            print(f"Warning: solver failed at t={t_cur:.2f} s: {sol_chunk.message}")
            break

        y_end = sol_chunk.y[:, -1]
        _rhs(t_next, y_end, p)
        ev_now = _clad_failure_event(t_next, y_end, p)

        if ev_prev > 0 and ev_now <= 0:
            # Clad failure detected in this chunk.  Use the chunk endpoint
            # (1 s resolution) as the failure state -- it is a fully converged
            # solver state, unlike dense-output or bisection interpolants
            # which can produce unphysical intermediate values.
            clad_fail_time = t_next
            y_fail = y_end.copy()
            print(f"Clad failure at time {clad_fail_time:.2f} s")

            # Record snapshot at failure time
            _record_snapshot(clad_fail_time, y_fail)
            break

        # No event -- record snapshot at this output time
        _record_snapshot(t_next, y_end)

        y_cur = y_end.copy()
        t_cur = t_next
        ev_prev = ev_now

    # --- Phase 2: continue from failure to end (failed-clad mode) ---
    if not np.isnan(clad_fail_time):
        p["clad_fail"] = True

        t_eval_2 = np.arange(
            clad_fail_time + dt,
            time_end + dt,
            dt,
        )
        t_eval_2 = t_eval_2[t_eval_2 <= time_end]

        # Chunked integration for post-failure: the Jacobian numerical
        # differencing can push water properties out of valid range, causing
        # NaN.  Integrate chunk-by-chunk and stop gracefully if a chunk fails.
        y_cur2 = y_fail.copy()
        t_cur2 = clad_fail_time
        for t_next2 in t_eval_2:
            try:
                sol2 = solve_ivp(
                    fun=lambda t, y: _rhs(t, y, p),
                    t_span=(t_cur2, t_next2),
                    y0=y_cur2,
                    method="BDF",
                    rtol=1e-6,
                    atol=1e-4,
                    max_step=1.0,
                )
                if not sol2.success:
                    print(f"Warning: post-failure solver stopped at t={t_cur2:.1f} s")
                    break
                _record_snapshot(sol2.t[-1], sol2.y[:, -1])
                y_cur2 = sol2.y[:, -1].copy()
                t_cur2 = sol2.t[-1]
            except ValueError:
                print(f"Warning: post-failure solver hit NaN at t={t_cur2:.1f} s, "
                      "stopping Phase 2")
                break

    # --- Build result dataclass ---
    nt = len(records["time"])
    nz = params.nz
    nf = params.fuel_nr
    nc = params.clad_nr

    result = THResult(
        time=np.array(records["time"]),
        fuel_T=np.array(records["fuel_T"]).transpose(1, 2, 0),    # (nz, nf, nt)
        clad_T=np.array(records["clad_T"]).transpose(1, 2, 0),    # (nz, nc, nt)
        cool_T=np.array(records["cool_T"]).T,                     # (nz, nt)
        cool_Tsat=np.array(records["cool_Tsat"]).T,               # (nz, nt)
        cool_TCHF=np.array(records["cool_TCHF"]).T,               # (nz, nt)
        cool_p=np.array(records["cool_p"]).T,                     # (nz, nt)
        cool_h=np.array(records["cool_h"]).T,                     # (nz, nt)
        cool_vel=np.array(records["cool_vel"]).T,                 # (nz, nt)
        cool_void=np.array(records["cool_void"]).T,               # (nz, nt)
        cool_x=np.array(records["cool_x"]).T,                     # (nz, nt)
        cool_regime=np.array(records["cool_regime"]).T,           # (nz, nt)
        ingas_p=np.array(records["ingas_p"]),                     # (nt,)
        fuel_r=np.array(records["fuel_r"]).T,                     # (nz, nt)
        clad_r_in=np.array(records["clad_r_in"]).T,               # (nz, nt)
        clad_r_out=np.array(records["clad_r_out"]).T,             # (nz, nt)
        clad_sig_r=np.array(records["clad_sig_r"]).transpose(1, 2, 0),  # (nz, nc, nt)
        clad_sig_h=np.array(records["clad_sig_h"]).transpose(1, 2, 0),
        clad_sig_z=np.array(records["clad_sig_z"]).transpose(1, 2, 0),
        clad_sig_I=np.array(records["clad_sig_I"]).T,             # (nz, nt)
        clad_sig_B=np.array(records["clad_sig_B"]).T,             # (nz, nt)
        clad_epsPeff=np.array(records["clad_epsPeff"]).transpose(1, 2, 0),  # (nz, nc, nt)
        clad_fail_time=clad_fail_time,
        params=params,
    )

    return result


# ============================================================================
# Plotting and main
# ============================================================================

OUTPUT = Path("07_results")


def main():
    print("=" * 70)
    print("THERMAL HYDRAULICS -- PWR LOCA TRANSIENT (600s)")
    print("=" * 70)

    result = solve_thermal_hydraulics()

    # fuel_T shape: (nz, nf, nt)
    print(f"\n  Time steps: {len(result.time)}")
    print(f"  Max fuel center T: {result.fuel_T[0, 0, :].max() - 273:.0f} C")
    print(f"  Max clad outer T: {result.clad_T[0, -1, :].max() - 273:.0f} C")
    if hasattr(result, 'clad_fail_time') and result.clad_fail_time is not None and not np.isnan(result.clad_fail_time):
        print(f"  Clad failure at: {result.clad_fail_time:.1f} s")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    nt = len(result.time)

    # Temperature evolution (first axial node)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.time, result.fuel_T[0, 0, :nt] - 273, "-r", label="Fuel center")
    ax.plot(result.time, result.clad_T[0, -1, :nt] - 273, "-b", label="Clad outer")
    ax.plot(result.time, result.cool_T[0, :nt] - 273, "-g", label="Coolant node 1")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "TH_01_temperatures.pdf")
    plt.close(fig)

    # Pressure
    fig, ax = plt.subplots()
    ax.plot(result.time, result.cool_p[0, :nt], "-b", label="Node 1")
    if result.cool_p.shape[0] > 1:
        ax.plot(result.time, result.cool_p[1, :nt], "--b", label="Node 2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (MPa)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "TH_02_pressure.pdf")
    plt.close(fig)

    # Void fraction
    fig, ax = plt.subplots()
    ax.plot(result.time, result.cool_void[0, :nt], "-b", label="Node 1")
    if result.cool_void.shape[0] > 1:
        ax.plot(result.time, result.cool_void[1, :nt], "--b", label="Node 2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Void fraction (-)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "TH_03_void.pdf")
    plt.close(fig)

    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
