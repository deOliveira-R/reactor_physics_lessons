"""Fuel behaviour under 1-D radial approximation.

Solves coupled radial heat conduction and thermo-elastic-plastic deformation
of a single fuel rod over its operational lifetime (default 6 years).

Port of MATLAB Module 08 (``fuelBehaviour.m``, ``initializeFuelRod.m``,
``funRHS.m``, ``gapClosureEvent.m``).

Key restructuring vs. MATLAB: stresses are NOT carried as DAE state
variables.  They are computed algebraically at each RHS evaluation by
solving the linear stress equilibrium + strain compatibility + boundary
condition system.  The ODE state vector contains only genuinely
time-evolving quantities (temperatures, fission density, swelling, creep
strains, and plastic strains).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from data.materials import matpro


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class FuelRodGeometry:
    """As-fabricated fuel rod geometry and operating parameters."""

    # --- fuel ---
    h0: float = 3.0  # fuel stack height (m)
    fuel_r_in: float = 0.0  # inner fuel radius (m)
    fuel_r_out: float = 4.12e-3  # outer fuel radius (m)
    fuel_nr: int = 30  # radial nodes in fuel
    porosity: float = 0.05  # fuel porosity
    fgr: float = 0.06  # fission gas release fraction
    q_lhgr: float = 200e2  # linear heat generation rate (W/m)

    # --- clad ---
    clad_r_in: float = 4.22e-3  # inner clad radius (m)
    clad_r_out: float = 4.75e-3  # outer clad radius (m)
    clad_nr: int = 5  # radial nodes in clad
    clad_T_out: float = 600.0  # outer clad temperature (K), BC
    fast_flux: float = 1e13  # fast neutron flux (1/cm2-s)

    # --- gap ---
    roughness: float = 6e-6  # effective roughness (m)

    # --- inner gas ---
    T_plenum: float = 600.0  # plenum temperature (K)
    p0: float = 1.0  # fabrication He pressure (MPa)
    v_gas_plenum: float = 10e-6  # plenum free volume (m3)

    # --- coolant ---
    cool_p: float = 16.0  # coolant pressure (MPa)

    # --- time ---
    time_end_years: float = 6.0
    time_step_days: float = 1.0


@dataclass
class FuelBehaviourResult:
    """Complete time-history of the fuel-rod calculation."""

    time: np.ndarray  # (nt,) seconds
    time_years: np.ndarray  # (nt,) years

    # radii at every output step -- (nr, nt) in metres
    fuel_r: np.ndarray
    clad_r: np.ndarray

    # temperatures (K) -- (nr, nt)
    fuel_T: np.ndarray
    clad_T: np.ndarray

    # stresses (MPa) -- (nr, nt)  indices: r, h, z
    fuel_sig_r: np.ndarray
    fuel_sig_h: np.ndarray
    fuel_sig_z: np.ndarray
    fuel_sig_vm: np.ndarray
    clad_sig_r: np.ndarray
    clad_sig_h: np.ndarray
    clad_sig_z: np.ndarray
    clad_sig_vm: np.ndarray

    # total strains (fraction) -- (nr, nt)
    fuel_eps_r: np.ndarray
    fuel_eps_h: np.ndarray
    fuel_eps_z: np.ndarray
    clad_eps_r: np.ndarray
    clad_eps_h: np.ndarray
    clad_eps_z: np.ndarray

    # component strains -- thermal, elastic, creep, swelling, plastic
    fuel_eps_T: np.ndarray  # (nr, nt) linear thermal
    fuel_eps_S: np.ndarray  # (nr, nt) volumetric swelling
    fuel_eps_rE: np.ndarray
    fuel_eps_hE: np.ndarray
    fuel_eps_zE: np.ndarray
    fuel_eps_rC: np.ndarray
    fuel_eps_hC: np.ndarray
    fuel_eps_zC: np.ndarray

    clad_eps_T: np.ndarray
    clad_eps_rE: np.ndarray
    clad_eps_hE: np.ndarray
    clad_eps_zE: np.ndarray
    clad_eps_rC: np.ndarray
    clad_eps_hC: np.ndarray
    clad_eps_zC: np.ndarray
    clad_eps_rP: np.ndarray
    clad_eps_hP: np.ndarray
    clad_eps_zP: np.ndarray

    # scalars per timestep
    gap_dr: np.ndarray  # gap width (m)
    gap_open: np.ndarray  # boolean flag
    gap_p_contact: np.ndarray  # contact pressure (MPa)
    ingas_p: np.ndarray  # inner gas pressure (MPa)
    fuel_dz: np.ndarray  # fuel height (m)
    clad_dz: np.ndarray  # clad height (m)
    burnup: np.ndarray  # (MWd/kgUO2)

    geometry: FuelRodGeometry = field(repr=False)


# ============================================================================
# Helpers
# ============================================================================


def _interp_between(y: np.ndarray) -> np.ndarray:
    """Interpolate values at node boundaries (midpoints between nodes).

    Given values at n nodes, returns n-1 values at the midpoints.
    Equivalent to MATLAB ``interp1((1:n)',y,(1.5:n-0.5)')``.
    """
    return 0.5 * (y[:-1] + y[1:])


def _von_mises(sig_r: np.ndarray, sig_h: np.ndarray, sig_z: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5 * (sig_r - sig_h) ** 2
        + 0.5 * (sig_r - sig_z) ** 2
        + 0.5 * (sig_h - sig_z) ** 2
    )


# ============================================================================
# Initialization
# ============================================================================


def _initialize_fuel_rod(
    geo: FuelRodGeometry,
) -> tuple[np.ndarray, dict]:
    """Build initial state vector and a *params* dict used by the RHS.

    Returns
    -------
    y0 : 1-D array
        Initial ODE state vector.
    params : dict
        Precomputed constants carried through the integration.
    """
    nf = geo.fuel_nr
    nc = geo.clad_nr

    # --- fuel mesh ---
    fuel_dr0 = (geo.fuel_r_out - geo.fuel_r_in) / (nf - 1)
    fuel_r0 = np.linspace(geo.fuel_r_in, geo.fuel_r_out, nf)
    fuel_r0_bnd = np.concatenate(
        ([geo.fuel_r_in], _interp_between(fuel_r0), [geo.fuel_r_out])
    )
    fuel_v0 = np.pi * np.diff(fuel_r0_bnd ** 2) * geo.h0
    fuel_mass = matpro.UO2_RHO * (1.0 - geo.porosity) * fuel_v0

    # power / fission rate
    fuel_qV = geo.q_lhgr * geo.h0 / fuel_v0.sum()
    fuel_dFdt = fuel_qV / (1.60217656535e-13 * 200.0)

    # --- clad mesh ---
    clad_dr0 = (geo.clad_r_out - geo.clad_r_in) / (nc - 1)
    clad_r0 = np.linspace(geo.clad_r_in, geo.clad_r_out, nc)
    clad_r0_bnd = np.concatenate(
        ([geo.clad_r_in], _interp_between(clad_r0), [geo.clad_r_out])
    )
    clad_v0 = np.pi * np.diff(clad_r0_bnd ** 2) * geo.h0

    # --- gap ---
    gap_dr0 = geo.clad_r_in - geo.fuel_r_out
    gap_r0_mid = 0.5 * (geo.clad_r_in + geo.fuel_r_out)

    # --- inner gas initial moles ---
    v_gap0 = geo.h0 * np.pi * (geo.clad_r_in ** 2 - geo.fuel_r_out ** 2)
    v_void0 = geo.h0 * np.pi * geo.fuel_r_in ** 2
    mu_He0 = (geo.p0 * 1e6) * (geo.v_gas_plenum + v_gap0 + v_void0) / (8.31 * 293.0)

    # --- pack params ---
    params = dict(
        nf=nf,
        nc=nc,
        h0=geo.h0,
        fuel_r0=fuel_r0,
        fuel_r0_bnd=fuel_r0_bnd,
        fuel_dr0=fuel_dr0,
        fuel_v0=fuel_v0,
        fuel_mass=fuel_mass,
        fuel_qV=fuel_qV,
        fuel_dFdt=fuel_dFdt,
        porosity=geo.porosity,
        fgr=geo.fgr,
        clad_r0=clad_r0,
        clad_r0_bnd=clad_r0_bnd,
        clad_dr0=clad_dr0,
        clad_v0=clad_v0,
        clad_T_out=geo.clad_T_out,
        gap_dr0=gap_dr0,
        gap_r0_mid=gap_r0_mid,
        roughness=geo.roughness,
        T_plenum=geo.T_plenum,
        v_gas_plenum=geo.v_gas_plenum,
        mu_He0=mu_He0,
        cool_p=geo.cool_p,
        fuel_r_in=geo.fuel_r_in,
        # gap state -- mutated upon closure event
        gap_open=True,
        gap_clsd=False,
        gap_depsh=0.0,
        gap_depsz=0.0,
    )

    # --- initial state vector ---
    # Layout: fuel_T(nf), clad_T(nc-1), F_scaled, swelling(nf),
    #         fuel_creep(nf*3), clad_creep(nc*3), clad_epsPeff(nc),
    #         clad_plastic(nc*3)
    n_ode = nf + (nc - 1) + 1 + nf + nf * 3 + nc * 3 + nc + nc * 3
    y0 = np.zeros(n_ode)
    # temperatures initialised to outer clad temperature
    y0[:nf] = geo.clad_T_out  # fuel temperatures
    y0[nf : nf + nc - 1] = geo.clad_T_out  # clad temperatures (excl. outer)
    # everything else starts at zero

    return y0, params


# ============================================================================
# State vector packing / unpacking
# ============================================================================


def _unpack_state(y: np.ndarray, params: dict) -> dict:
    """Unpack the flat ODE state vector into named arrays.

    Values are returned in *physical* units (strains as fractions, fission
    density in 1/m3).  The state vector stores fission density scaled by
    1e-27 and strains as percent to help the ODE solver with scaling.
    """
    nf = params["nf"]
    nc = params["nc"]
    idx = 0

    fuel_T = y[idx : idx + nf].copy()
    idx += nf

    clad_T_inner = y[idx : idx + nc - 1].copy()
    idx += nc - 1
    clad_T = np.append(clad_T_inner, params["clad_T_out"])

    F = y[idx] * 1e27  # 1/m3
    idx += 1

    fuel_epsS = y[idx : idx + nf] / 100.0  # fraction
    idx += nf

    fuel_epsC = np.empty((3, nf))
    for i in range(3):
        fuel_epsC[i] = y[idx : idx + nf] / 100.0
        idx += nf

    clad_epsC = np.empty((3, nc))
    for i in range(3):
        clad_epsC[i] = y[idx : idx + nc] / 100.0
        idx += nc

    clad_epsPeff = y[idx : idx + nc] / 100.0
    idx += nc

    clad_epsP = np.empty((3, nc))
    for i in range(3):
        clad_epsP[i] = y[idx : idx + nc] / 100.0
        idx += nc

    return dict(
        fuel_T=fuel_T,
        clad_T=clad_T,
        F=F,
        fuel_epsS=fuel_epsS,
        fuel_epsC=fuel_epsC,
        clad_epsC=clad_epsC,
        clad_epsPeff=clad_epsPeff,
        clad_epsP=clad_epsP,
    )


# ============================================================================
# Algebraic stress solver
# ============================================================================


def _solve_stress(state: dict, params: dict) -> dict:
    """Solve for stress components given current strains and temperatures.

    The system of equations is linear in the 3*(nf+nc) stress unknowns.

    Equations per region (fuel / clad):
      * (n-1) stress equilibrium:  d(sig_r)/dr = (sig_h - sig_r)/r
      * (n-1) strain compatibility: d(eps_h)/dr = (eps_r - eps_h)/r
      * (n-1) axial strain constant: eps_z(i) = eps_z(i+1)

    Plus 6 boundary conditions linking fuel and clad.

    For each node, strain = epsT + epsE + epsC + epsS/3 (fuel) or
    strain = epsT + epsE + epsC + epsP (clad), with epsE depending
    linearly on stress via Hooke's law.

    We build A*sigma = b and solve with numpy.
    """
    nf = params["nf"]
    nc = params["nc"]
    h0 = params["h0"]
    por = params["porosity"]

    fuel_T = state["fuel_T"]
    clad_T = state["clad_T"]
    F = state["F"]
    fuel_epsS = state["fuel_epsS"]
    fuel_epsC = state["fuel_epsC"]  # (3, nf)
    clad_epsC = state["clad_epsC"]  # (3, nc)
    clad_epsP = state["clad_epsP"]  # (3, nc)

    # --- material properties ---
    fuel_E = matpro.uo2_E(fuel_T, por)  # (nf,) MPa
    fuel_nu = matpro.UO2_NU
    clad_E = matpro.zry_E(clad_T)  # (nc,) MPa
    clad_nu = matpro.ZRY_NU

    # Thermal strains
    fuel_epsT = matpro.uo2_thexp(fuel_T)  # (nf,)
    clad_epsT = matpro.zry_thexp(clad_T)  # (nc,)

    # Inelastic strains (non-stress-dependent part of total strain)
    # fuel: eps_i = epsT + epsE_i + epsC_i + epsS/3
    # clad: eps_i = epsT + epsE_i + epsC_i + epsP_i
    fuel_epsNE = np.empty((3, nf))  # non-elastic part
    clad_epsNE = np.empty((3, nc))
    for i in range(3):
        fuel_epsNE[i] = fuel_epsT + fuel_epsC[i] + fuel_epsS / 3.0
        clad_epsNE[i] = clad_epsT + clad_epsC[i] + clad_epsP[i]

    # --- mesh geometry (use original, stress system is formulated on original
    #     geometry; deformed geometry is computed *after* stress is known) ---
    # Actually the MATLAB uses deformed geometry for the residuals, which
    # depends on eps{2} which depends on stress.  However, since we're solving
    # for stress from scratch, we need to iterate or use original geometry.
    # The MATLAB DAE solver has the luxury of having the previous-step stresses
    # as initial guesses.  For our algebraic solve, using original geometry is
    # a very good approximation (strains are O(1%) so errors are O(0.01%)).
    fuel_r = params["fuel_r0"]
    clad_r = params["clad_r0"]
    fuel_dr = params["fuel_dr0"]
    clad_dr = params["clad_dr0"]
    fuel_r_mid = _interp_between(fuel_r)
    clad_r_mid = _interp_between(clad_r)

    # --- inner gas pressure ---
    Bu = F * 200.0 * 1.60217656535e-13 / 8.64e10 / matpro.UO2_RHO
    frac_FG = np.array([0.01, 0.02, 0.23])  # He, Kr, Xe fractions
    mu_gen = F * params["fuel_v0"].sum() / 6.022e23 * frac_FG
    mu_rel = mu_gen * params["fgr"]
    mu = mu_rel + np.array([params["mu_He0"], 0.0, 0.0])
    gap_T = 0.5 * (fuel_T[-1] + clad_T[0])
    v_gap = h0 * np.pi * (clad_r[0] ** 2 - fuel_r[-1] ** 2)
    v_void = h0 * np.pi * fuel_r[0] ** 2
    ingas_p = (
        mu.sum()
        * 8.31e-6
        / (
            params["v_gas_plenum"] / params["T_plenum"]
            + v_void / fuel_T[0]
            + v_gap / gap_T
        )
    )

    cool_p = params["cool_p"]
    gap_open = params["gap_open"]
    gap_clsd = params["gap_clsd"]
    gap_depsz = params["gap_depsz"]

    gap_r_mid = 0.5 * (clad_r[0] + fuel_r[-1])

    # Total unknowns: sig_r(nf), sig_h(nf), sig_z(nf),
    #                 sig_r(nc), sig_h(nc), sig_z(nc)
    n_unknowns = 3 * (nf + nc)
    A = np.zeros((n_unknowns, n_unknowns))
    b = np.zeros(n_unknowns)

    # Helper: column offset for each stress component
    # fuel_sig_r: [0, nf)
    # fuel_sig_h: [nf, 2*nf)
    # fuel_sig_z: [2*nf, 3*nf)
    # clad_sig_r: [3*nf, 3*nf + nc)
    # clad_sig_h: [3*nf + nc, 3*nf + 2*nc)
    # clad_sig_z: [3*nf + 2*nc, 3*nf + 3*nc)
    fr = 0
    fh = nf
    fz = 2 * nf
    cr = 3 * nf
    ch = 3 * nf + nc
    cz = 3 * nf + 2 * nc

    # Compliance coefficients for Hooke's law:
    # eps_i = (sig_i - nu*(sig_j + sig_k)) / E + epsNE_i
    # => eps_i = sig_i/E * (1) + sig_j/E * (-nu) + sig_k/E * (-nu) + epsNE_i
    # For fuel node j:
    #   C_self = 1/E(j),  C_cross = -nu/E(j)

    fuel_C_self = 1.0 / fuel_E
    fuel_C_cross = -fuel_nu / fuel_E
    clad_C_self = 1.0 / clad_E
    clad_C_cross = -clad_nu / clad_E

    row = 0

    # ----- Fuel stress equilibrium: (nf-1) equations -----
    # diff(sig_r) / dr = (sig_h_ - sig_r_) / r_
    # => (sig_r[j+1] - sig_r[j]) / dr = (sig_h_ - sig_r_) / r_
    # where sig_h_ = 0.5*(sig_h[j] + sig_h[j+1]), sig_r_ = 0.5*(sig_r[j] + sig_r[j+1])
    for j in range(nf - 1):
        c_diff = 1.0 / fuel_dr
        c_avg = 0.5 / fuel_r_mid[j]
        # (sig_r[j+1] - sig_r[j])/dr - (sig_h_ - sig_r_)/r_ = 0
        A[row, fr + j] = -c_diff + c_avg  # sig_r[j] coeff: -1/dr + 0.5/r
        A[row, fr + j + 1] = c_diff + c_avg  # sig_r[j+1] coeff: 1/dr + 0.5/r
        A[row, fh + j] = -c_avg  # sig_h[j] coeff: -0.5/r
        A[row, fh + j + 1] = -c_avg  # sig_h[j+1] coeff: -0.5/r
        row += 1

    # ----- Fuel strain compatibility: (nf-1) equations -----
    # diff(eps_h)/dr - (eps_r_ - eps_h_)/r_ = 0
    # where Cs = 1/E, Cc = -nu/E, dC = Cs - Cc = (1+nu)/E:
    #   eps_h[k] = Cs[k]*sig_h[k] + Cc[k]*(sig_r[k]+sig_z[k]) + epsNE_h[k]
    #   eps_r[k] = Cs[k]*sig_r[k] + Cc[k]*(sig_h[k]+sig_z[k]) + epsNE_r[k]
    # (eps_r_ - eps_h_)/r_ depends only on dC*(sig_r-sig_h) + (epsNE_r-epsNE_h)
    for j in range(nf - 1):
        jp = j + 1
        Cs_j = fuel_C_self[j]
        Cs_jp = fuel_C_self[jp]
        Cc_j = fuel_C_cross[j]
        Cc_jp = fuel_C_cross[jp]
        r_ = fuel_r_mid[j]
        dr = fuel_dr

        # diff(eps_h)/dr terms:
        A[row, fh + j] += -Cs_j / dr
        A[row, fh + jp] += Cs_jp / dr
        A[row, fr + j] += -Cc_j / dr
        A[row, fr + jp] += Cc_jp / dr
        A[row, fz + j] += -Cc_j / dr
        A[row, fz + jp] += Cc_jp / dr

        # -(eps_r_ - eps_h_)/r_ terms:
        dC_j = Cs_j - Cc_j  # = (1 + nu)/E
        dC_jp = Cs_jp - Cc_jp
        A[row, fr + j] += -0.5 * dC_j / r_
        A[row, fr + jp] += -0.5 * dC_jp / r_
        A[row, fh + j] += 0.5 * dC_j / r_
        A[row, fh + jp] += 0.5 * dC_jp / r_
        # sig_z cancels in (eps_r_ - eps_h_) since both have same Cc*sig_z

        # RHS = non-elastic contributions
        dNE_h = fuel_epsNE[1, jp] - fuel_epsNE[1, j]  # hoop NE diff
        rhs_ne = (
            fuel_epsNE[0, j]
            - fuel_epsNE[1, j]
            + fuel_epsNE[0, jp]
            - fuel_epsNE[1, jp]
        )
        b[row] = -dNE_h / dr + 0.5 * rhs_ne / r_

        row += 1

    # ----- Fuel axial strain constant: (nf-1) equations -----
    # eps_z[j] = eps_z[j+1]  =>  eps_z[j] - eps_z[j+1] = 0
    # eps_z[k] = Cs[k]*sig_z[k] + Cc[k]*(sig_r[k]+sig_h[k]) + epsNE_z[k]
    for j in range(nf - 1):
        jp = j + 1
        A[row, fz + j] += fuel_C_self[j]
        A[row, fr + j] += fuel_C_cross[j]
        A[row, fh + j] += fuel_C_cross[j]
        A[row, fz + jp] += -fuel_C_self[jp]
        A[row, fr + jp] += -fuel_C_cross[jp]
        A[row, fh + jp] += -fuel_C_cross[jp]
        b[row] = -(fuel_epsNE[2, j] - fuel_epsNE[2, jp])
        row += 1

    # ----- Clad stress equilibrium: (nc-1) equations -----
    for j in range(nc - 1):
        c_diff = 1.0 / clad_dr
        c_avg = 0.5 / clad_r_mid[j]
        A[row, cr + j] = -c_diff + c_avg
        A[row, cr + j + 1] = c_diff + c_avg
        A[row, ch + j] = -c_avg
        A[row, ch + j + 1] = -c_avg
        row += 1

    # ----- Clad strain compatibility: (nc-1) equations -----
    for j in range(nc - 1):
        jp = j + 1
        Cs_j = clad_C_self[j]
        Cs_jp = clad_C_self[jp]
        Cc_j = clad_C_cross[j]
        Cc_jp = clad_C_cross[jp]
        r_ = clad_r_mid[j]
        dr = clad_dr

        # diff(eps_h)/dr terms
        A[row, ch + j] += -Cs_j / dr
        A[row, ch + jp] += Cs_jp / dr
        A[row, cr + j] += -Cc_j / dr
        A[row, cr + jp] += Cc_jp / dr
        A[row, cz + j] += -Cc_j / dr
        A[row, cz + jp] += Cc_jp / dr

        # -(eps_r_ - eps_h_)/r_ terms
        dC_j = Cs_j - Cc_j
        dC_jp = Cs_jp - Cc_jp
        A[row, cr + j] += -0.5 * dC_j / r_
        A[row, cr + jp] += -0.5 * dC_jp / r_
        A[row, ch + j] += 0.5 * dC_j / r_
        A[row, ch + jp] += 0.5 * dC_jp / r_

        dNE_h = clad_epsNE[1, jp] - clad_epsNE[1, j]
        rhs_ne = (
            clad_epsNE[0, j]
            - clad_epsNE[1, j]
            + clad_epsNE[0, jp]
            - clad_epsNE[1, jp]
        )
        b[row] = -dNE_h / dr + 0.5 * rhs_ne / r_
        row += 1

    # ----- Clad axial strain constant: (nc-1) equations -----
    for j in range(nc - 1):
        jp = j + 1
        A[row, cz + j] += clad_C_self[j]
        A[row, cr + j] += clad_C_cross[j]
        A[row, ch + j] += clad_C_cross[j]
        A[row, cz + jp] += -clad_C_self[jp]
        A[row, cr + jp] += -clad_C_cross[jp]
        A[row, ch + jp] += -clad_C_cross[jp]
        b[row] = -(clad_epsNE[2, j] - clad_epsNE[2, jp])
        row += 1

    # ----- Boundary conditions (6 equations) -----
    # BC1: fuel inner surface
    if params["fuel_r_in"] > 0:
        # sig_r(1) = -p_ingas
        A[row, fr + 0] = 1.0
        b[row] = -ingas_p
    else:
        # sig_r(1) = sig_h(1)  (symmetry at centre)
        A[row, fr + 0] = 1.0
        A[row, fh + 0] = -1.0
        b[row] = 0.0
    row += 1

    # BC2: clad outer surface
    # sig_r(nc) = -cool_p
    A[row, cr + nc - 1] = 1.0
    b[row] = -cool_p
    row += 1

    # BC3: fuel outer / clad inner surface (radial stress)
    if gap_open:
        # sig_r_fuel(nf) = -p_ingas
        A[row, fr + nf - 1] = 1.0
        b[row] = -ingas_p
    else:
        # Radial stress continuity at contact:
        #   sig_r_clad(1) = sig_r_fuel(nf)
        A[row, cr + 0] = 1.0
        A[row, fr + nf - 1] = -1.0
        b[row] = 0.0
    row += 1

    # BC4: fuel outer / clad inner surface (hoop strain compatibility or pressure)
    if gap_open:
        # sig_r_clad(1) = -p_ingas
        A[row, cr + 0] = 1.0
        b[row] = -ingas_p
    else:
        # Contact displacement constraint (closed gap):
        #
        #   r_clad_in(deformed) - r_fuel_out(deformed) = roughness
        #
        # Expanding with eps_h = (r - r0)/r0:
        #   clad_r0_in * (1 + eps_h_c1) - fuel_r0_out * (1 + eps_h_fn)
        #       = roughness
        #
        # Rearranging:
        #   clad_r0_in * eps_h_c1 - fuel_r0_out * eps_h_fn
        #       = roughness - gap_dr0
        #
        # Splitting eps_h into elastic + NE:
        #   clad_r0_in * [Cs_c1*sig_h_c1 + Cc_c1*(sig_r_c1+sig_z_c1)]
        # - fuel_r0_out * [Cs_fn*sig_h_fn + Cc_fn*(sig_r_fn+sig_z_fn)]
        #       = roughness - gap_dr0
        #         - clad_r0_in * epsNE_h_c1
        #         + fuel_r0_out * epsNE_h_fn
        #
        # This is the physical constraint that the deformed gap width
        # equals roughness.  It is LINEAR in the stresses and naturally
        # bounded (no 1/gap_dr amplification).  The MATLAB's BC4
        # residual with evolving gap.dr converges to the same result.
        Cs_c1 = clad_C_self[0]
        Cc_c1 = clad_C_cross[0]
        Cs_fn = fuel_C_self[nf - 1]
        Cc_fn = fuel_C_cross[nf - 1]

        r_c = clad_r[0]  # clad_r0_in
        r_f = fuel_r[-1]  # fuel_r0_out

        A[row, ch + 0] = r_c * Cs_c1
        A[row, cr + 0] = r_c * Cc_c1
        A[row, cz + 0] = r_c * Cc_c1
        A[row, fh + nf - 1] = -r_f * Cs_fn
        A[row, fr + nf - 1] = -r_f * Cc_fn
        A[row, fz + nf - 1] = -r_f * Cc_fn

        b[row] = (
            params["roughness"]
            - params["gap_dr0"]
            - r_c * clad_epsNE[1, 0]
            + r_f * fuel_epsNE[1, nf - 1]
        )
    row += 1

    # BC5: axial force balance / axial strain continuity
    # sig_z interpolated at midpoints, integrated over ring areas
    # sigzIntegral = sum( sig_z_mid * diff(r^2) )
    # sig_z_mid[j] = 0.5*(sig_z[j] + sig_z[j+1])
    # diff(r^2)[j] = r[j+1]^2 - r[j]^2
    fuel_dr2 = np.diff(fuel_r ** 2)
    clad_dr2 = np.diff(clad_r ** 2)

    if gap_open:
        # integral of axial stress in fuel = 0
        for j in range(nf - 1):
            A[row, fz + j] += 0.5 * fuel_dr2[j]
            A[row, fz + j + 1] += 0.5 * fuel_dr2[j]
        b[row] = 0.0
    else:
        # eps_z_clad(1) - eps_z_fuel(nf) - gap_depsz = 0
        A[row, cz + 0] += clad_C_self[0]
        A[row, cr + 0] += clad_C_cross[0]
        A[row, ch + 0] += clad_C_cross[0]
        A[row, fz + nf - 1] += -fuel_C_self[nf - 1]
        A[row, fr + nf - 1] += -fuel_C_cross[nf - 1]
        A[row, fh + nf - 1] += -fuel_C_cross[nf - 1]
        b[row] = -(clad_epsNE[2, 0] - fuel_epsNE[2, nf - 1] - gap_depsz)
    row += 1

    # BC6: axial force balance for clad (open) or fuel+clad (closed)
    if gap_open:
        # integral of clad axial stress = p_inner * r_inner^2 - p_outer * r_outer^2
        for j in range(nc - 1):
            A[row, cz + j] += 0.5 * clad_dr2[j]
            A[row, cz + j + 1] += 0.5 * clad_dr2[j]
        b[row] = ingas_p * clad_r[0] ** 2 - cool_p * clad_r[-1] ** 2
    else:
        # integral of (fuel + clad) axial stress = p_inner * r_inner_clad^2 - p_outer * r_outer_clad^2
        for j in range(nf - 1):
            A[row, fz + j] += 0.5 * fuel_dr2[j]
            A[row, fz + j + 1] += 0.5 * fuel_dr2[j]
        for j in range(nc - 1):
            A[row, cz + j] += 0.5 * clad_dr2[j]
            A[row, cz + j + 1] += 0.5 * clad_dr2[j]
        b[row] = ingas_p * clad_r[0] ** 2 - cool_p * clad_r[-1] ** 2
    row += 1

    assert row == n_unknowns, f"Expected {n_unknowns} equations, got {row}"

    # --- Solve the linear system ---
    sigma = np.linalg.solve(A, b)

    fuel_sig_r = sigma[fr : fr + nf]
    fuel_sig_h = sigma[fh : fh + nf]
    fuel_sig_z = sigma[fz : fz + nf]
    clad_sig_r = sigma[cr : cr + nc]
    clad_sig_h = sigma[ch : ch + nc]
    clad_sig_z = sigma[cz : cz + nc]

    return dict(
        fuel_sig_r=fuel_sig_r,
        fuel_sig_h=fuel_sig_h,
        fuel_sig_z=fuel_sig_z,
        clad_sig_r=clad_sig_r,
        clad_sig_h=clad_sig_h,
        clad_sig_z=clad_sig_z,
        ingas_p=ingas_p,
    )


# ============================================================================
# RHS function
# ============================================================================


def _rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """Right-hand side of the ODE system.

    Returns dy/dt for all ODE variables.  Stresses are solved algebraically
    inside this function and do not appear in the state vector.
    """
    nf = params["nf"]
    nc = params["nc"]
    h0 = params["h0"]
    por = params["porosity"]

    # --- unpack state ---
    state = _unpack_state(y, params)
    fuel_T = state["fuel_T"]
    clad_T = state["clad_T"]
    F = state["F"]
    fuel_epsS = state["fuel_epsS"]
    fuel_epsC = state["fuel_epsC"]  # (3, nf)
    clad_epsC = state["clad_epsC"]  # (3, nc)
    clad_epsPeff = state["clad_epsPeff"]  # (nc,)
    clad_epsP = state["clad_epsP"]  # (3, nc)

    # --- solve for stress ---
    stress = _solve_stress(state, params)
    fuel_sig_r = stress["fuel_sig_r"]
    fuel_sig_h = stress["fuel_sig_h"]
    fuel_sig_z = stress["fuel_sig_z"]
    clad_sig_r = stress["clad_sig_r"]
    clad_sig_h = stress["clad_sig_h"]
    clad_sig_z = stress["clad_sig_z"]
    ingas_p = stress["ingas_p"]

    # --- burnup ---
    Bu = F * 200.0 * 1.60217656535e-13 / 8.64e10 / matpro.UO2_RHO
    fuel_dFdt = params["fuel_dFdt"]

    # --- fuel strains (total) ---
    fuel_E = matpro.uo2_E(fuel_T, por)
    fuel_nu = matpro.UO2_NU
    fuel_epsT = matpro.uo2_thexp(fuel_T)

    fuel_sigSum = fuel_sig_r + fuel_sig_h + fuel_sig_z
    fuel_epsE = np.empty((3, nf))
    fuel_eps = np.empty((3, nf))
    for i, sig_i in enumerate([fuel_sig_r, fuel_sig_h, fuel_sig_z]):
        fuel_epsE[i] = (sig_i - fuel_nu * (fuel_sigSum - sig_i)) / fuel_E
        fuel_eps[i] = fuel_epsT + fuel_epsE[i] + fuel_epsC[i] + fuel_epsS / 3.0

    # fuel Von Mises stress (add small offset to avoid division by zero)
    fuel_sigVM = _von_mises(fuel_sig_r, fuel_sig_h, fuel_sig_z) + 1e-6

    # fuel creep rates
    fuel_eff_creep = matpro.uo2_creep_rate(fuel_sigVM, fuel_T)
    fuel_creep_rate = np.empty((3, nf))
    for i, sig_i in enumerate([fuel_sig_r, fuel_sig_h, fuel_sig_z]):
        fuel_creep_rate[i] = fuel_eff_creep * (sig_i - fuel_sigSum / 3.0) / fuel_sigVM

    # fuel swelling rate
    fuel_swel_rate = matpro.uo2_swelling_rate(fuel_dFdt, F, fuel_T)

    # --- clad strains (total) ---
    clad_E = matpro.zry_E(clad_T)
    clad_nu = matpro.ZRY_NU
    clad_epsT = matpro.zry_thexp(clad_T)

    clad_sigSum = clad_sig_r + clad_sig_h + clad_sig_z
    clad_epsE = np.empty((3, nc))
    clad_eps = np.empty((3, nc))
    for i, sig_i in enumerate([clad_sig_r, clad_sig_h, clad_sig_z]):
        clad_epsE[i] = (sig_i - clad_nu * (clad_sigSum - sig_i)) / clad_E
        clad_eps[i] = clad_epsT + clad_epsE[i] + clad_epsP[i] + clad_epsC[i]

    # clad Von Mises stress
    clad_sigVM = _von_mises(clad_sig_r, clad_sig_h, clad_sig_z) + 1e-6

    # clad creep rate
    clad_eff_creep = matpro.zry_creep_rate(clad_sigVM, clad_T)
    clad_creep_rate = np.empty((3, nc))
    for i, sig_i in enumerate([clad_sig_r, clad_sig_h, clad_sig_z]):
        clad_creep_rate[i] = (
            clad_eff_creep * 1.5 * (sig_i - clad_sigSum / 3.0) / clad_sigVM
        )

    # clad plastic strain rate
    eff_plastic_rate = 1e-3 * (
        clad_sigVM
        * 1e6
        / matpro.zry_K(clad_T)
        / np.abs(clad_epsPeff + 1e-6) ** matpro.zry_n(clad_T)
    ) ** (1.0 / matpro.zry_m(clad_T))

    clad_plastic_rate = np.empty((3, nc))
    plastic_active = eff_plastic_rate >= clad_sigVM / clad_E
    for i, sig_i in enumerate([clad_sig_r, clad_sig_h, clad_sig_z]):
        clad_plastic_rate[i] = (
            plastic_active
            * eff_plastic_rate
            * 1.5
            * (sig_i - clad_sigSum / 3.0)
            / clad_sigVM
        )

    # --- update geometry ---
    fuel_eps_mid = np.empty((3, nf - 1))
    clad_eps_mid = np.empty((3, nc - 1))
    for i in range(3):
        fuel_eps_mid[i] = _interp_between(fuel_eps[i])
        clad_eps_mid[i] = _interp_between(clad_eps[i])

    fuel_r0 = params["fuel_r0"]
    clad_r0 = params["clad_r0"]
    fuel_dr0 = params["fuel_dr0"]
    clad_dr0 = params["clad_dr0"]

    fuel_dr = fuel_dr0 * (1.0 + fuel_eps_mid[0])  # radial node thickness
    clad_dr = clad_dr0 * (1.0 + clad_eps_mid[0])

    fuel_r = fuel_r0 * (1.0 + fuel_eps[1])  # hoop => radius change
    clad_r = clad_r0 * (1.0 + clad_eps[1])

    fuel_r_mid = _interp_between(fuel_r)
    clad_r_mid = _interp_between(clad_r)

    fuel_r_bnd = np.concatenate(([fuel_r[0]], fuel_r_mid, [fuel_r[-1]]))
    clad_r_bnd = np.concatenate(([clad_r[0]], clad_r_mid, [clad_r[-1]]))

    fuel_dz = h0 * (1.0 + fuel_eps[2])
    clad_dz = h0 * (1.0 + clad_eps[2])

    fuel_a_mid = 2.0 * np.pi * fuel_r_mid * _interp_between(fuel_dz)
    clad_a_mid = 2.0 * np.pi * clad_r_mid * _interp_between(clad_dz)

    fuel_v = np.pi * np.diff(fuel_r_bnd ** 2) * fuel_dz
    clad_v = np.pi * np.diff(clad_r_bnd ** 2) * clad_dz

    # --- gap geometry ---
    gap_dr = clad_r[0] - fuel_r[-1]
    gap_r_mid = 0.5 * (clad_r[0] + fuel_r[-1])
    gap_a_mid = 2.0 * np.pi * gap_r_mid * 0.5 * (clad_dz[0] + fuel_dz[-1])

    # store gap width for event detection
    params["_gap_dr"] = gap_dr

    # --- gas pressure (recompute with deformed geometry) ---
    gap_T = 0.5 * (fuel_T[-1] + clad_T[0])
    dT_gap = fuel_T[-1] - clad_T[0]

    v_gap = fuel_dz[-1] * np.pi * (clad_r[0] ** 2 - fuel_r[-1] ** 2)
    v_void = fuel_dz[0] * np.pi * fuel_r[0] ** 2

    frac_FG = np.array([0.01, 0.02, 0.23])
    mu_gen = F * params["fuel_v0"].sum() / 6.022e23 * frac_FG
    mu_rel = mu_gen * params["fgr"]
    mu = mu_rel + np.array([params["mu_He0"], 0.0, 0.0])

    ingas_p_deformed = (
        mu.sum()
        * 8.31e-6
        / (
            params["v_gas_plenum"] / params["T_plenum"]
            + v_void / fuel_T[0]
            + v_gap / gap_T
        )
    )

    # --- gap conductance ---
    # Mole fractions [He, Kr, Xe]
    mu_total = mu.sum()
    mole_frac = mu / mu_total if mu_total > 0 else np.array([1.0, 0.0, 0.0])
    k_gas = matpro.gas_mixture_k(gap_T, mole_frac)

    gap_open = params["gap_open"]
    if gap_open:
        h_gap = k_gas / max(gap_dr, 1e-10)
    else:
        h_gap = k_gas / params["roughness"]

    # --- thermal calculations ---
    Tf_mid = _interp_between(fuel_T)
    Tc_mid = _interp_between(clad_T)

    q_gap = h_gap * dT_gap

    q_fuel_mid = -matpro.uo2_k(Tf_mid, Bu, por) * np.diff(fuel_T) / fuel_dr
    Q_fuel = np.concatenate(([0.0], q_fuel_mid * fuel_a_mid, [q_gap * gap_a_mid]))

    q_clad_mid = -matpro.zry_k(Tc_mid) * np.diff(clad_T) / clad_dr
    Q_clad = np.concatenate(([q_gap * gap_a_mid], q_clad_mid * clad_a_mid, [0.0]))

    rate_fuel_T = (
        -np.diff(Q_fuel) + params["fuel_qV"] * params["fuel_v0"]
    ) / (params["fuel_mass"] * matpro.uo2_cp(fuel_T))

    rate_clad_T = -np.diff(Q_clad) / (
        matpro.ZRY_RHO * matpro.zry_cp(clad_T) * params["clad_v0"]
    )
    # exclude outer clad node (fixed BC)
    rate_clad_T = rate_clad_T[: nc - 1]

    # --- pack output vector ---
    # Same order as state vector, same scaling
    dydt = np.concatenate(
        [
            rate_fuel_T,  # (nf,) K/s
            rate_clad_T,  # (nc-1,) K/s
            [fuel_dFdt / 1e27],  # scaled fission density rate
            fuel_swel_rate * 100.0,  # percent/s
            fuel_creep_rate[0] * 100.0,  # fuel creep r
            fuel_creep_rate[1] * 100.0,  # fuel creep h
            fuel_creep_rate[2] * 100.0,  # fuel creep z
            clad_creep_rate[0] * 100.0,  # clad creep r
            clad_creep_rate[1] * 100.0,  # clad creep h
            clad_creep_rate[2] * 100.0,  # clad creep z
            eff_plastic_rate * 100.0,  # clad effective plastic
            clad_plastic_rate[0] * 100.0,  # clad plastic r
            clad_plastic_rate[1] * 100.0,  # clad plastic h
            clad_plastic_rate[2] * 100.0,  # clad plastic z
        ]
    )
    return dydt


# ============================================================================
# Event function
# ============================================================================


def _gap_closure_event(t: float, y: np.ndarray, params: dict) -> float:
    """Returns gap_dr - roughness; terminal event when this crosses zero."""
    # Use the gap width cached by the last RHS call if available
    if "_gap_dr" in params:
        gap_dr = params.pop("_gap_dr")  # consume (one-shot cache)
        return gap_dr - params["roughness"]

    # Otherwise compute from scratch
    state = _unpack_state(y, params)
    try:
        stress = _solve_stress(state, params)
    except Exception:
        return 1.0  # large positive = gap open, skip this evaluation

    por = params["porosity"]
    fuel_T = state["fuel_T"]
    clad_T = state["clad_T"]

    fuel_E = matpro.uo2_E(fuel_T, por)
    clad_E = matpro.zry_E(clad_T)
    fuel_nu = matpro.UO2_NU
    clad_nu = matpro.ZRY_NU

    fuel_sigSum = stress["fuel_sig_r"] + stress["fuel_sig_h"] + stress["fuel_sig_z"]
    clad_sigSum = stress["clad_sig_r"] + stress["clad_sig_h"] + stress["clad_sig_z"]

    fuel_epsT = matpro.uo2_thexp(fuel_T)
    clad_epsT = matpro.zry_thexp(clad_T)

    fuel_eps_h = fuel_epsT + (
        (stress["fuel_sig_h"] - fuel_nu * (fuel_sigSum - stress["fuel_sig_h"])) / fuel_E
    ) + state["fuel_epsC"][1] + state["fuel_epsS"] / 3.0
    clad_eps_h = clad_epsT + (
        (stress["clad_sig_h"] - clad_nu * (clad_sigSum - stress["clad_sig_h"])) / clad_E
    ) + state["clad_epsP"][1] + state["clad_epsC"][1]

    fuel_r = params["fuel_r0"] * (1.0 + fuel_eps_h)
    clad_r = params["clad_r0"] * (1.0 + clad_eps_h)
    gap_dr = clad_r[0] - fuel_r[-1]

    return gap_dr - params["roughness"]


_gap_closure_event.terminal = True
_gap_closure_event.direction = -1  # trigger only when gap closes (positive -> negative)


# ============================================================================
# Snapshot collector (called after each solve_ivp segment)
# ============================================================================


def _collect_snapshot(
    t: float, y: np.ndarray, params: dict
) -> dict:
    """Evaluate all derived quantities at a single time/state and return a
    flat dict of scalars and arrays for result storage."""
    nf = params["nf"]
    nc = params["nc"]
    h0 = params["h0"]
    por = params["porosity"]

    state = _unpack_state(y, params)
    stress = _solve_stress(state, params)

    fuel_T = state["fuel_T"]
    clad_T = state["clad_T"]
    F = state["F"]
    fuel_epsS = state["fuel_epsS"]
    fuel_epsC = state["fuel_epsC"]
    clad_epsC = state["clad_epsC"]
    clad_epsP = state["clad_epsP"]

    fuel_sig_r = stress["fuel_sig_r"]
    fuel_sig_h = stress["fuel_sig_h"]
    fuel_sig_z = stress["fuel_sig_z"]
    clad_sig_r = stress["clad_sig_r"]
    clad_sig_h = stress["clad_sig_h"]
    clad_sig_z = stress["clad_sig_z"]

    fuel_E = matpro.uo2_E(fuel_T, por)
    clad_E = matpro.zry_E(clad_T)
    fuel_nu = matpro.UO2_NU
    clad_nu = matpro.ZRY_NU

    fuel_epsT = matpro.uo2_thexp(fuel_T)
    clad_epsT = matpro.zry_thexp(clad_T)

    fuel_sigSum = fuel_sig_r + fuel_sig_h + fuel_sig_z
    clad_sigSum = clad_sig_r + clad_sig_h + clad_sig_z

    fuel_epsE = np.empty((3, nf))
    fuel_eps = np.empty((3, nf))
    for i, sig_i in enumerate([fuel_sig_r, fuel_sig_h, fuel_sig_z]):
        fuel_epsE[i] = (sig_i - fuel_nu * (fuel_sigSum - sig_i)) / fuel_E
        fuel_eps[i] = fuel_epsT + fuel_epsE[i] + fuel_epsC[i] + fuel_epsS / 3.0

    clad_epsE = np.empty((3, nc))
    clad_eps = np.empty((3, nc))
    for i, sig_i in enumerate([clad_sig_r, clad_sig_h, clad_sig_z]):
        clad_epsE[i] = (sig_i - clad_nu * (clad_sigSum - sig_i)) / clad_E
        clad_eps[i] = clad_epsT + clad_epsE[i] + clad_epsP[i] + clad_epsC[i]

    fuel_r = params["fuel_r0"] * (1.0 + fuel_eps[1])
    clad_r = params["clad_r0"] * (1.0 + clad_eps[1])
    fuel_dz = h0 * (1.0 + fuel_eps[2])
    clad_dz = h0 * (1.0 + clad_eps[2])

    gap_dr = clad_r[0] - fuel_r[-1]
    gap_p_contact = params["gap_clsd"] * (-clad_sig_r[0])

    Bu = F * 200.0 * 1.60217656535e-13 / 8.64e10 / matpro.UO2_RHO

    # inner gas pressure (with deformed geometry)
    gap_T = 0.5 * (fuel_T[-1] + clad_T[0])
    v_gap = fuel_dz[-1] * np.pi * (clad_r[0] ** 2 - fuel_r[-1] ** 2)
    v_void = fuel_dz[0] * np.pi * fuel_r[0] ** 2
    frac_FG = np.array([0.01, 0.02, 0.23])
    mu_gen = F * params["fuel_v0"].sum() / 6.022e23 * frac_FG
    mu_rel = mu_gen * params["fgr"]
    mu = mu_rel + np.array([params["mu_He0"], 0.0, 0.0])
    ingas_p = (
        mu.sum()
        * 8.31e-6
        / (
            params["v_gas_plenum"] / params["T_plenum"]
            + v_void / fuel_T[0]
            + v_gap / gap_T
        )
    )

    fuel_sigVM = _von_mises(fuel_sig_r, fuel_sig_h, fuel_sig_z)
    clad_sigVM = _von_mises(clad_sig_r, clad_sig_h, clad_sig_z)

    return dict(
        fuel_r=fuel_r,
        clad_r=clad_r,
        fuel_T=fuel_T,
        clad_T=clad_T,
        fuel_sig_r=fuel_sig_r,
        fuel_sig_h=fuel_sig_h,
        fuel_sig_z=fuel_sig_z,
        fuel_sig_vm=fuel_sigVM,
        clad_sig_r=clad_sig_r,
        clad_sig_h=clad_sig_h,
        clad_sig_z=clad_sig_z,
        clad_sig_vm=clad_sigVM,
        fuel_eps_r=fuel_eps[0],
        fuel_eps_h=fuel_eps[1],
        fuel_eps_z=fuel_eps[2],
        clad_eps_r=clad_eps[0],
        clad_eps_h=clad_eps[1],
        clad_eps_z=clad_eps[2],
        fuel_eps_T=fuel_epsT,
        fuel_eps_S=fuel_epsS,
        fuel_eps_rE=fuel_epsE[0],
        fuel_eps_hE=fuel_epsE[1],
        fuel_eps_zE=fuel_epsE[2],
        fuel_eps_rC=fuel_epsC[0],
        fuel_eps_hC=fuel_epsC[1],
        fuel_eps_zC=fuel_epsC[2],
        clad_eps_T=clad_epsT,
        clad_eps_rE=clad_epsE[0],
        clad_eps_hE=clad_epsE[1],
        clad_eps_zE=clad_epsE[2],
        clad_eps_rC=clad_epsC[0],
        clad_eps_hC=clad_epsC[1],
        clad_eps_zC=clad_epsC[2],
        clad_eps_rP=clad_epsP[0],
        clad_eps_hP=clad_epsP[1],
        clad_eps_zP=clad_epsP[2],
        gap_dr=gap_dr,
        gap_open=float(params["gap_open"]),
        gap_p_contact=gap_p_contact,
        ingas_p=ingas_p,
        fuel_dz=fuel_dz[0],
        clad_dz=clad_dz[0],
        burnup=Bu,
    )


# ============================================================================
# Main entry point
# ============================================================================


def solve_fuel_behaviour(
    geo: FuelRodGeometry | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-4,
    verbose: bool = True,
) -> FuelBehaviourResult:
    """Simulate fuel behaviour over the rod lifetime.

    Parameters
    ----------
    geo : FuelRodGeometry, optional
        Geometry and operating conditions.  Uses defaults if *None*.
    rtol, atol : float
        Relative and absolute ODE tolerances.
    verbose : bool
        Print progress messages.

    Returns
    -------
    FuelBehaviourResult
    """
    if geo is None:
        geo = FuelRodGeometry()

    y0, params = _initialize_fuel_rod(geo)

    DAY = 86_400.0
    YEAR = 365.0 * DAY
    t_step = geo.time_step_days * DAY
    t_end = geo.time_end_years * YEAR
    t_eval = np.arange(0.0, t_end + t_step * 0.5, t_step)

    # --- Phase 1: open gap ---
    if verbose:
        print("Solving fuel behaviour (open gap)...")

    gap_event = lambda t, y: _gap_closure_event(t, y, params)  # noqa: E731
    gap_event.terminal = True
    gap_event.direction = -1

    sol1 = solve_ivp(
        fun=lambda t, y: _rhs(t, y, params),
        t_span=(0.0, t_end),
        y0=y0,
        method="Radau",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        events=gap_event,
    )

    # Collect Phase 1 snapshots while gap is still open
    snapshots: list[tuple[float, dict]] = []
    for i in range(len(sol1.t)):
        snap = _collect_snapshot(sol1.t[i], sol1.y[:, i], params)
        snapshots.append((sol1.t[i], snap))

    # --- Phase 2: gap closure continuation ---
    if sol1.t_events[0].size > 0:
        t_closure = sol1.t_events[0][0]
        y_closure = sol1.y_events[0][0]

        if verbose:
            print(
                f"Gap closure at t = {t_closure / YEAR:.4f} years. "
                "Continuing with closed gap..."
            )

        # Compute strain jumps at closure
        state_cl = _unpack_state(y_closure, params)
        stress_cl = _solve_stress(state_cl, params)

        nf = params["nf"]
        por = params["porosity"]
        fuel_T = state_cl["fuel_T"]
        clad_T = state_cl["clad_T"]

        fuel_E = matpro.uo2_E(fuel_T, por)
        clad_E = matpro.zry_E(clad_T)
        fuel_nu = matpro.UO2_NU
        clad_nu = matpro.ZRY_NU

        fuel_sigSum = (
            stress_cl["fuel_sig_r"]
            + stress_cl["fuel_sig_h"]
            + stress_cl["fuel_sig_z"]
        )
        clad_sigSum = (
            stress_cl["clad_sig_r"]
            + stress_cl["clad_sig_h"]
            + stress_cl["clad_sig_z"]
        )

        fuel_epsT = matpro.uo2_thexp(fuel_T)
        clad_epsT = matpro.zry_thexp(clad_T)

        # Compute total strains at closure
        fuel_eps_cl = np.empty((3, nf))
        for i, sig_i in enumerate(
            [stress_cl["fuel_sig_r"], stress_cl["fuel_sig_h"], stress_cl["fuel_sig_z"]]
        ):
            fuel_epsE_i = (sig_i - fuel_nu * (fuel_sigSum - sig_i)) / fuel_E
            fuel_eps_cl[i] = (
                fuel_epsT + fuel_epsE_i + state_cl["fuel_epsC"][i] + state_cl["fuel_epsS"] / 3.0
            )

        nc = params["nc"]
        clad_eps_cl = np.empty((3, nc))
        for i, sig_i in enumerate(
            [stress_cl["clad_sig_r"], stress_cl["clad_sig_h"], stress_cl["clad_sig_z"]]
        ):
            clad_epsE_i = (sig_i - clad_nu * (clad_sigSum - sig_i)) / clad_E
            clad_eps_cl[i] = (
                clad_epsT + clad_epsE_i + state_cl["clad_epsP"][i] + state_cl["clad_epsC"][i]
            )

        params["gap_open"] = False
        params["gap_clsd"] = True
        params["gap_depsz"] = clad_eps_cl[2, 0] - fuel_eps_cl[2, nf - 1]
        params["gap_depsh"] = clad_eps_cl[1, 0] - fuel_eps_cl[1, nf - 1]

        t_eval2 = np.arange(t_closure + t_step, t_end + t_step * 0.1, t_step)
        t_eval2 = t_eval2[t_eval2 <= t_end]  # ensure within span

        sol2 = solve_ivp(
            fun=lambda t, y: _rhs(t, y, params),
            t_span=(t_closure, t_end),
            y0=y_closure,
            method="Radau",
            t_eval=t_eval2,
            rtol=1e-4,  # relaxed tolerance for closed-gap phase (MATLAB does this)
            atol=atol,
        )

        for i in range(len(sol2.t)):
            snap = _collect_snapshot(sol2.t[i], sol2.y[:, i], params)
            snapshots.append((sol2.t[i], snap))

    if verbose:
        print("Collecting results...")

    # --- Build result arrays ---
    nt = len(snapshots)
    nf = params["nf"]
    nc = params["nc"]

    # Preallocate
    arrays_nf = [
        "fuel_r",
        "fuel_T",
        "fuel_sig_r",
        "fuel_sig_h",
        "fuel_sig_z",
        "fuel_sig_vm",
        "fuel_eps_r",
        "fuel_eps_h",
        "fuel_eps_z",
        "fuel_eps_T",
        "fuel_eps_S",
        "fuel_eps_rE",
        "fuel_eps_hE",
        "fuel_eps_zE",
        "fuel_eps_rC",
        "fuel_eps_hC",
        "fuel_eps_zC",
    ]
    arrays_nc = [
        "clad_r",
        "clad_T",
        "clad_sig_r",
        "clad_sig_h",
        "clad_sig_z",
        "clad_sig_vm",
        "clad_eps_r",
        "clad_eps_h",
        "clad_eps_z",
        "clad_eps_T",
        "clad_eps_rE",
        "clad_eps_hE",
        "clad_eps_zE",
        "clad_eps_rC",
        "clad_eps_hC",
        "clad_eps_zC",
        "clad_eps_rP",
        "clad_eps_hP",
        "clad_eps_zP",
    ]
    scalars = [
        "gap_dr",
        "gap_open",
        "gap_p_contact",
        "ingas_p",
        "fuel_dz",
        "clad_dz",
        "burnup",
    ]

    result_data: dict = {k: np.empty((nf, nt)) for k in arrays_nf}
    result_data.update({k: np.empty((nc, nt)) for k in arrays_nc})
    result_data.update({k: np.empty(nt) for k in scalars})

    for i, (ti, snap) in enumerate(snapshots):
        for k in arrays_nf:
            result_data[k][:, i] = snap[k]
        for k in arrays_nc:
            result_data[k][:, i] = snap[k]
        for k in scalars:
            result_data[k][i] = snap[k]

    time_arr = np.array([t for t, _ in snapshots])

    if verbose:
        print("Done.")

    return FuelBehaviourResult(
        time=time_arr,
        time_years=time_arr / YEAR,
        geometry=geo,
        **result_data,
    )
