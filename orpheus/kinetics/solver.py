"""Coupled 0-D reactor kinetics + 1-D thermal hydraulics + fuel/clad mechanics.

Simulates a Reactivity Insertion Accident (RIA) in a PWR-like subchannel:

1. **Steady-state** (0--100 s): constant power, establish equilibrium.
2. **Transient** (100--120 s): power excursion driven by an inlet-temperature
   perturbation, with Doppler and coolant temperature feedback.  Gap closure
   is tracked as a terminal event.

Port of MATLAB Module 10 (``reactorKinetics.m``, ``initializeNeutronics.m``,
``initializeCoolant.m``, ``initializeFuelRod.m``, ``funRHS.m``,
``gapClosureEvent.m``).

Key restructuring vs. MATLAB
-----------------------------
* Stresses are solved **algebraically** at every RHS evaluation (like
  Module 08) instead of being DAE state variables.  This eliminates the
  need for a mass matrix and lets us use a standard ODE solver.
* Pressure is also computed algebraically (not as an AE variable).
* The state vector therefore contains only genuinely time-evolving quantities:
  power, 6 delayed-neutron precursor densities, fuel & clad temperatures,
  coolant enthalpy, clad plastic effective strain, and clad plastic strain
  components.

.. seealso:: :ref:`theory-reactor-kinetics` — Key Facts, point kinetics, reactivity feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from orpheus.data.materials import matpro
from orpheus.data.materials.h2o_properties import h2o_density, h2o_enthalpy, h2o_properties


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class KineticsParams:
    """Physical parameters for the RIA simulation."""

    # --- axial mesh ---
    nz: int = 2
    dz0: float = 1.5  # axial node height (m)

    # --- fuel ---
    fuel_r_in: float = 0.0  # inner fuel radius (m)
    fuel_r_out: float = 4.12e-3  # outer fuel radius (m)
    fuel_nr: int = 20  # radial fuel nodes
    porosity: float = 0.05  # fuel porosity
    burnup: float = 0.0  # MWd/kgHM
    fgr: float = 0.06  # fission gas release fraction
    pow0: float = 69140.625  # initial fuel rod power (W)

    # --- clad ---
    clad_r_in: float = 4.22e-3  # inner clad radius (m)
    clad_r_out: float = 4.75e-3  # outer clad radius (m)
    clad_nr: int = 5  # radial clad nodes

    # --- gap ---
    roughness: float = 6e-6  # effective roughness (m)

    # --- inner gas ---
    p0_gas: float = 1.2  # fabrication He pressure (MPa)
    v_gas_plenum: float = 10e-6  # plenum free volume (m3)

    # --- coolant ---
    a_flow0: float = 8.914e-5  # channel flow area (m2)

    # --- kinetics ---
    beta_eff: tuple = (2.584e-4, 1.520e-3, 1.391e-3, 3.070e-3, 1.102e-3, 2.584e-4)
    decay_constants: tuple = (0.013, 0.032, 0.119, 0.318, 1.403, 3.929)
    prompt_lifetime: float = 20e-6  # prompt neutron lifetime (s)
    doppler_coeff: float = -2e-5  # (1/K)
    coolant_temp_coeff: float = -20e-5  # (1/K)

    # --- boundary conditions ---
    # inlet temperature: (time_s, T_K) pairs
    inlet_T_table: tuple = (
        (0.0, 553.0), (100.0, 553.0), (100.5, 300.0),
        (101.0, 300.0), (101.5, 553.0), (600.0, 553.0),
    )
    # outlet pressure (MPa): constant
    outlet_p_table: tuple = ((0.0, 15.5), (600.0, 15.5))
    # inlet velocity (m/s): constant
    inlet_vel_table: tuple = ((0.0, 4.8), (600.0, 4.8))

    # --- time control ---
    t_steady_end: float = 100.0  # end of steady state (s)
    t_transient_end: float = 120.0  # end of transient (s)
    dt_steady: float = 1.0  # output step, steady state (s)
    dt_transient: float = 0.1  # output step, transient (s)

    # --- solver ---
    rtol: float = 1e-6
    atol: float = 1e-4
    max_step_steady: float = 10.0
    max_step_transient: float = 1e-3


@dataclass
class KineticsResult:
    """Complete time-history of the RIA simulation."""

    time: np.ndarray  # (nt,)

    # kinetics
    power: np.ndarray  # (nt,) normalized P/P0
    reac_doppler: np.ndarray  # (nt,) pcm
    reac_coolant: np.ndarray  # (nt,) pcm
    reac_total: np.ndarray  # (nt,) pcm
    cDNP: np.ndarray  # (6, nt) precursor densities

    # inner gas
    ingas_p: np.ndarray  # (nt,) MPa

    # fuel temperatures (nz, nf, nt) in Celsius
    fuel_T: np.ndarray

    # clad temperatures (nz, nc, nt) in Celsius
    clad_T: np.ndarray

    # coolant
    cool_T: np.ndarray  # (nz, nt) Celsius
    cool_Tsat: np.ndarray  # (nz, nt) Celsius
    cool_TCHF: np.ndarray  # (nz, nt) Celsius
    cool_regime: np.ndarray  # (nz, nt)
    cool_p: np.ndarray  # (nz, nt) MPa
    cool_h: np.ndarray  # (nz, nt) kJ/kg
    cool_vel: np.ndarray  # (nz, nt) m/s
    cool_void: np.ndarray  # (nz, nt)

    # geometry
    fuel_r_out: np.ndarray  # (nz, nt) mm
    clad_r_in: np.ndarray  # (nz, nt) mm
    clad_r_out: np.ndarray  # (nz, nt) mm

    # stresses (nz, nc, nt) MPa
    clad_sig_r: np.ndarray
    clad_sig_h: np.ndarray
    clad_sig_z: np.ndarray
    clad_sig_eng: np.ndarray  # (nz, nt) engineering stress

    # gap
    gap_dr: np.ndarray  # (nz, nt) microns
    gap_h: np.ndarray  # (nz, nt) W/m2-K

    # plastic strain (nz, nc, nt) %
    clad_epsPeff: np.ndarray

    params: KineticsParams = field(repr=False)


# ============================================================================
# Helpers
# ============================================================================


def _interp_between(y: np.ndarray) -> np.ndarray:
    """Midpoint interpolation between adjacent nodes (works on last axis)."""
    return 0.5 * (y[..., :-1] + y[..., 1:])


def _interp_table(table: tuple, t: float) -> float:
    """Piecewise-linear interpolation of a (time, value) table."""
    times = np.array([row[0] for row in table])
    values = np.array([row[1] for row in table])
    return float(np.interp(t, times, values))


def _von_mises(sr: np.ndarray, sh: np.ndarray, sz: np.ndarray) -> np.ndarray:
    return np.sqrt(0.5 * (sr - sh) ** 2 + 0.5 * (sr - sz) ** 2 + 0.5 * (sh - sz) ** 2)


# ============================================================================
# Initialization
# ============================================================================


def _initialize(kp: KineticsParams) -> tuple[np.ndarray, dict]:
    """Build initial state vector and parameter dictionary.

    Returns
    -------
    y0 : initial flat ODE state vector
    p  : dict of precomputed constants carried through the integration
    """
    nz, nf, nc = kp.nz, kp.fuel_nr, kp.clad_nr
    dz0 = kp.dz0

    # --- fuel mesh ---
    fuel_dr0 = (kp.fuel_r_out - kp.fuel_r_in) / (nf - 1)
    fuel_r0 = np.linspace(kp.fuel_r_in, kp.fuel_r_out, nf)
    fuel_r0_bnd = np.concatenate(
        ([kp.fuel_r_in], _interp_between(fuel_r0), [kp.fuel_r_out])
    )
    # fuel_v0: (nz, nf)
    ring_areas = np.pi * np.diff(fuel_r0_bnd ** 2)  # (nf,)
    fuel_v0 = np.tile(ring_areas * dz0, (nz, 1))  # (nz, nf)
    fuel_mass = matpro.UO2_RHO * (1.0 - kp.porosity) * fuel_v0  # (nz, nf)

    # power distribution
    kz = np.ones(nz)
    LHGR = kp.pow0 / (dz0 * nz)
    fuel_qV = np.outer(LHGR * dz0 * kz / fuel_v0.sum(axis=1), np.ones(nf))  # (nz, nf)

    # fuel boundary cross-section areas (nf-1,) then tile to (nz, nf-1)
    fuel_r0_mid = _interp_between(fuel_r0)
    fuel_a_mid = np.tile(2 * np.pi * fuel_r0_mid * dz0, (nz, 1))  # (nz, nf-1)

    # --- clad mesh ---
    clad_dr0 = (kp.clad_r_out - kp.clad_r_in) / (nc - 1)
    clad_r0 = np.linspace(kp.clad_r_in, kp.clad_r_out, nc)
    clad_r0_bnd = np.concatenate(
        ([kp.clad_r_in], _interp_between(clad_r0), [kp.clad_r_out])
    )
    clad_v0 = np.tile(np.pi * np.diff(clad_r0_bnd ** 2) * dz0, (nz, 1))  # (nz, nc)
    clad_r0_mid = _interp_between(clad_r0)
    clad_a_mid = np.tile(2 * np.pi * clad_r0_mid * dz0, (nz, 1))  # (nz, nc-1)

    # --- gap ---
    gap_dr0 = kp.clad_r_in - kp.fuel_r_out
    gap_r0_mid = 0.5 * (kp.clad_r_in + kp.fuel_r_out)
    gap_a0 = np.full(nz, 2 * np.pi * gap_r0_mid * dz0)

    # --- inner gas ---
    v_gap0 = dz0 * nz * np.pi * (kp.clad_r_in ** 2 - kp.fuel_r_out ** 2)
    v_void0 = dz0 * nz * np.pi * kp.fuel_r_in ** 2
    mu_He0 = (kp.p0_gas * 1e6) * (kp.v_gas_plenum + v_gap0 + v_void0) / (8.31 * 293.0)

    # --- coolant ---
    cool_aFlow = np.full(nz, kp.a_flow0)
    cool_volFlow = cool_aFlow * dz0  # (nz,)
    cool_areaHX = np.full(nz, 2 * np.pi * kp.clad_r_out * dz0)  # (nz,)
    cool_dHyd = 4 * cool_volFlow / cool_areaHX  # (nz,)

    # initial coolant state
    T0 = kp.inlet_T_table[0][1]  # K
    p0 = kp.outlet_p_table[0][1]  # MPa
    h0 = h2o_enthalpy(p0, T0)

    # --- kinetics initial conditions ---
    beff = np.array(kp.beta_eff)
    lmb = np.array(kp.decay_constants)
    cDNP0 = beff * kp.pow0 / (kp.prompt_lifetime * lmb)

    # --- pack params ---
    p = dict(
        nz=nz, nf=nf, nc=nc, dz0=dz0,
        fuel_r0=fuel_r0, fuel_r0_bnd=fuel_r0_bnd, fuel_dr0=fuel_dr0,
        fuel_v0=fuel_v0, fuel_mass=fuel_mass, fuel_qV=fuel_qV,
        fuel_a_mid=fuel_a_mid, fuel_r0_mid=fuel_r0_mid,
        kz=kz, porosity=kp.porosity, burnup=kp.burnup,
        clad_r0=clad_r0, clad_r0_bnd=clad_r0_bnd, clad_dr0=clad_dr0,
        clad_v0=clad_v0, clad_a_mid=clad_a_mid, clad_r0_mid=clad_r0_mid,
        gap_dr0=gap_dr0, gap_r0_mid=gap_r0_mid, gap_a0=gap_a0,
        roughness=kp.roughness,
        mu_He0=mu_He0, v_gas_plenum=kp.v_gas_plenum,
        cool_aFlow=cool_aFlow, cool_aFlow0=kp.a_flow0,
        cool_volFlow=cool_volFlow, cool_areaHX=cool_areaHX,
        cool_dHyd=cool_dHyd,
        beff=beff, lmb=lmb, tLife=kp.prompt_lifetime,
        dopC=kp.doppler_coeff, coolTemC=kp.coolant_temp_coeff,
        pow0=kp.pow0,
        inlet_T_table=kp.inlet_T_table,
        outlet_p_table=kp.outlet_p_table,
        inlet_vel_table=kp.inlet_vel_table,
        # gap state (mutated on closure)
        gap_open=True, gap_clsd=False,
        gap_depsh=np.zeros(nz), gap_depsz=np.zeros(nz),
        # reactivity bias (set after steady-state)
        reac_dop_bias=0.0, reac_cool_bias=0.0,
        transient=False,
    )

    # --- initial state vector ---
    # Layout: power(1), cDNP(6), fuel_T(nz*nf), clad_T(nz*nc),
    #         cool_h(nz), clad_epsPeff(nz*nc), clad_epsP(nz*nc*3)
    n_ode = 1 + 6 + nz * nf + nz * nc + nz + nz * nc + nz * nc * 3
    y0 = np.zeros(n_ode)
    idx = 0
    y0[idx] = kp.pow0;  idx += 1
    y0[idx:idx + 6] = cDNP0;  idx += 6
    y0[idx:idx + nz * nf] = T0;  idx += nz * nf  # fuel T
    y0[idx:idx + nz * nc] = T0;  idx += nz * nc  # clad T
    y0[idx:idx + nz] = h0;  idx += nz  # coolant enthalpy
    # remaining (plastic strains) start at zero

    return y0, p


# ============================================================================
# State vector packing / unpacking
# ============================================================================


def _unpack(y: np.ndarray, p: dict) -> dict:
    """Unpack the flat ODE state vector into named 2-D arrays."""
    nz, nf, nc = p["nz"], p["nf"], p["nc"]
    idx = 0

    power = y[idx];  idx += 1
    cDNP = y[idx:idx + 6].copy();  idx += 6
    fuel_T = y[idx:idx + nz * nf].reshape(nz, nf).copy();  idx += nz * nf
    clad_T = y[idx:idx + nz * nc].reshape(nz, nc).copy();  idx += nz * nc
    cool_h = y[idx:idx + nz].copy();  idx += nz
    clad_epsPeff = y[idx:idx + nz * nc].reshape(nz, nc).copy();  idx += nz * nc
    clad_epsP = np.empty((3, nz, nc))
    for i in range(3):
        clad_epsP[i] = y[idx:idx + nz * nc].reshape(nz, nc)
        idx += nz * nc

    return dict(
        power=power, cDNP=cDNP,
        fuel_T=fuel_T, clad_T=clad_T, cool_h=cool_h,
        clad_epsPeff=clad_epsPeff, clad_epsP=clad_epsP,
    )


# ============================================================================
# Algebraic clad stress solver
# ============================================================================


def _solve_clad_stress(
    clad_T: np.ndarray,
    clad_epsP: np.ndarray,
    ingas_p: float,
    cool_p: np.ndarray,
    fuel_epsT: np.ndarray,
    p: dict,
) -> dict:
    """Solve for clad stress components at each axial level.

    Uses the same algebraic approach as Module 08 (fuel_behaviour.py),
    but only for the cladding (fuel stress is not tracked in Modules 09/10).
    """
    nz, nc = p["nz"], p["nc"]
    clad_r0 = p["clad_r0"]
    clad_dr0 = p["clad_dr0"]
    clad_r0_mid = p["clad_r0_mid"]
    gap_open = p["gap_open"]
    gap_clsd = p["gap_clsd"]

    clad_sig_r = np.zeros((nz, nc))
    clad_sig_h = np.zeros((nz, nc))
    clad_sig_z = np.zeros((nz, nc))

    for iz in range(nz):
        T_iz = clad_T[iz]  # (nc,)
        clad_E = matpro.zry_E(T_iz)
        clad_nu = matpro.ZRY_NU
        clad_epsT_iz = matpro.zry_thexp(T_iz)

        # Non-elastic strains
        epsNE = np.empty((3, nc))
        for i in range(3):
            epsNE[i] = clad_epsT_iz + clad_epsP[i, iz]

        C_self = 1.0 / clad_E
        C_cross = -clad_nu / clad_E

        # Build linear system: 3*nc unknowns
        # sig_r: [0, nc), sig_h: [nc, 2*nc), sig_z: [2*nc, 3*nc)
        n_unk = 3 * nc
        A = np.zeros((n_unk, n_unk))
        b = np.zeros(n_unk)
        sr, sh, sz = 0, nc, 2 * nc
        row = 0

        # --- Stress equilibrium: nc-1 equations ---
        for j in range(nc - 1):
            c_d = 1.0 / clad_dr0
            c_a = 0.5 / clad_r0_mid[j]
            A[row, sr + j] = -c_d + c_a
            A[row, sr + j + 1] = c_d + c_a
            A[row, sh + j] = -c_a
            A[row, sh + j + 1] = -c_a
            row += 1

        # --- Strain compatibility: nc-1 equations ---
        for j in range(nc - 1):
            jp = j + 1
            Cs_j, Cs_jp = C_self[j], C_self[jp]
            Cc_j, Cc_jp = C_cross[j], C_cross[jp]
            r_ = clad_r0_mid[j]
            dr = clad_dr0

            A[row, sh + j] += -Cs_j / dr
            A[row, sh + jp] += Cs_jp / dr
            A[row, sr + j] += -Cc_j / dr
            A[row, sr + jp] += Cc_jp / dr
            A[row, sz + j] += -Cc_j / dr
            A[row, sz + jp] += Cc_jp / dr

            dC_j = Cs_j - Cc_j
            dC_jp = Cs_jp - Cc_jp
            A[row, sr + j] += -0.5 * dC_j / r_
            A[row, sr + jp] += -0.5 * dC_jp / r_
            A[row, sh + j] += 0.5 * dC_j / r_
            A[row, sh + jp] += 0.5 * dC_jp / r_

            dNE_h = epsNE[1, jp] - epsNE[1, j]
            rhs_ne = epsNE[0, j] - epsNE[1, j] + epsNE[0, jp] - epsNE[1, jp]
            b[row] = -dNE_h / dr + 0.5 * rhs_ne / r_
            row += 1

        # --- Axial strain constant: nc-1 equations ---
        for j in range(nc - 1):
            jp = j + 1
            A[row, sz + j] += C_self[j]
            A[row, sr + j] += C_cross[j]
            A[row, sh + j] += C_cross[j]
            A[row, sz + jp] += -C_self[jp]
            A[row, sr + jp] += -C_cross[jp]
            A[row, sh + jp] += -C_cross[jp]
            b[row] = -(epsNE[2, j] - epsNE[2, jp])
            row += 1

        # --- BC1: sig_r at outer surface = -cool_p ---
        A[row, sr + nc - 1] = 1.0
        b[row] = -cool_p[iz]
        row += 1

        # --- BC2: sig_r at inner surface ---
        if gap_open:
            A[row, sr + 0] = 1.0
            b[row] = -ingas_p
        else:
            # Strain compatibility across closed gap
            gap_dr_bc = p["gap_dr0"]
            gap_r_bc = p["gap_r0_mid"]
            Cs_c1 = C_self[0]
            Cc_c1 = C_cross[0]

            A[row, sh + 0] += Cs_c1 / gap_dr_bc
            A[row, sr + 0] += Cc_c1 / gap_dr_bc
            A[row, sz + 0] += Cc_c1 / gap_dr_bc

            dC_c1 = Cs_c1 - Cc_c1
            A[row, sr + 0] += -0.5 * dC_c1 / gap_r_bc
            A[row, sh + 0] += 0.5 * dC_c1 / gap_r_bc

            ne_diff = epsNE[1, 0] - fuel_epsT[iz] - p["gap_depsh"][iz]
            # For the fuel side: eps_h_fuel ~ fuel_epsT (simplified)
            # RHS approximation: only clad NE contributes in the difference
            b[row] = (-ne_diff / gap_dr_bc
                      + 0.5 * (fuel_epsT[iz] - epsNE[1, 0]
                               + fuel_epsT[iz] - epsNE[0, 0]) / gap_r_bc)
        row += 1

        # --- BC3: axial force balance ---
        clad_dr2 = np.diff(clad_r0 ** 2)
        if gap_open:
            for j in range(nc - 1):
                A[row, sz + j] += 0.5 * clad_dr2[j]
                A[row, sz + j + 1] += 0.5 * clad_dr2[j]
            b[row] = ingas_p * clad_r0[0] ** 2 - cool_p[iz] * clad_r0[-1] ** 2
        else:
            # eps_z_clad(1) = fuel_epsT + gap_depsz
            A[row, sz + 0] += C_self[0]
            A[row, sr + 0] += C_cross[0]
            A[row, sh + 0] += C_cross[0]
            b[row] = -(epsNE[2, 0] - fuel_epsT[iz] - p["gap_depsz"][iz])
        row += 1

        sigma = np.linalg.solve(A[:row, :row], b[:row])
        clad_sig_r[iz] = sigma[sr:sr + nc]
        clad_sig_h[iz] = sigma[sh:sh + nc]
        clad_sig_z[iz] = sigma[sz:sz + nc]

    return dict(
        clad_sig_r=clad_sig_r, clad_sig_h=clad_sig_h, clad_sig_z=clad_sig_z,
    )


# ============================================================================
# Pressure and TH
# ============================================================================


def _compute_coolant(t: float, cool_h: np.ndarray, cool_p_prev: np.ndarray,
                     clad_T_wall: np.ndarray, p: dict) -> dict:
    """Compute coolant state, pressure, and wall heat flux.

    Returns a dict with coolant properties needed by the RHS.
    """
    nz = p["nz"]
    dz0 = p["dz0"]

    # Boundary conditions at current time
    T_inlet = _interp_table(p["inlet_T_table"], t)
    p_outlet = _interp_table(p["outlet_p_table"], t)
    vel_inlet = _interp_table(p["inlet_vel_table"], t)

    # --- mixture properties ---
    cool_T = np.empty(nz)
    cool_rho = np.empty(nz)
    cool_k = np.empty(nz)
    cool_mu = np.empty(nz)
    cool_nu = np.empty(nz)
    cool_cp = np.empty(nz)
    cool_void = np.empty(nz)
    cool_x = np.empty(nz)
    cool_Tsat = np.empty(nz)
    Lsat_rho = np.empty(nz)
    Lsat_h = np.empty(nz)
    Lsat_k = np.empty(nz)
    Lsat_sig = np.empty(nz)
    Vsat_rho = np.empty(nz)
    Vsat_h = np.empty(nz)
    Vsat_k = np.empty(nz)
    Vsat_cp = np.empty(nz)
    Vsat_nu = np.empty(nz)

    # Use previous pressure as approximation for property evaluation
    for i in range(nz):
        pro, Lsat, Vsat = h2o_properties(cool_p_prev[i], cool_h[i])
        cool_T[i] = pro.T
        cool_rho[i] = pro.rho
        cool_k[i] = pro.k
        cool_mu[i] = pro.mu
        cool_nu[i] = pro.nu
        cool_cp[i] = pro.c_p
        cool_void[i] = pro.void
        cool_x[i] = pro.x
        cool_Tsat[i] = pro.Tsat
        Lsat_rho[i] = Lsat.rho
        Lsat_h[i] = Lsat.h
        Lsat_k[i] = Lsat.k
        Lsat_sig[i] = Lsat.sig
        Vsat_rho[i] = Vsat.rho
        Vsat_h[i] = Vsat.h
        Vsat_k[i] = Vsat.k
        Vsat_cp[i] = Vsat.c_p
        Vsat_nu[i] = Vsat.nu

    # --- mass flowrate ---
    rho_inlet = h2o_density(cool_p_prev[0], T_inlet)
    mdot0 = vel_inlet * p["cool_aFlow0"] * rho_inlet
    mdot_junc = np.full(nz + 1, mdot0)
    mdot_node = 0.5 * (mdot_junc[:-1] + mdot_junc[1:])

    # --- velocity and Reynolds number ---
    cool_vel = mdot_node / cool_rho / p["cool_aFlow"]
    cool_re = p["cool_dHyd"] * np.abs(cool_vel) / cool_nu

    # --- pressure calculation (Churchill friction + gravity + acceleration) ---
    theta1 = -2.457 * np.log((7.0 / cool_re) ** 0.9)
    theta2 = 37530.0 / cool_re
    f_wall = 8.0 * ((8.0 / cool_re) ** 12 + 1.0 / (theta1 ** 16 + theta2 ** 16) ** 1.5) ** (1.0 / 12)

    dp_fric = f_wall * dz0 / (2 * p["cool_dHyd"]) * cool_rho * cool_vel ** 2 / 2 * 1e-6
    dp_grav = cool_rho * 9.81e-6 * dz0
    dp_accel = -np.diff(cool_rho * cool_vel ** 2) * 1e-6  # (nz-1,)

    # Junction pressures (from outlet upward)
    p_junc = p_outlet + np.concatenate(([0], np.cumsum((dp_grav + dp_fric)[::-1])))[::-1]
    # Node pressures
    p_node = 0.5 * (p_junc[:-1] + p_junc[1:])
    if nz > 1:
        p_node += np.concatenate(([0], np.cumsum(dp_accel[::-1])))[::-1]
    cool_p = p_node

    # --- wall heat transfer ---
    Twall = clad_T_wall
    dTw = Twall - cool_T
    dTwSat = Twall - cool_Tsat
    drhoSat = Lsat_rho - Vsat_rho
    dhLat = Vsat_h - Lsat_h
    dhSub = np.maximum(Lsat_h - cool_h, 0.0)
    Pr = cool_cp * cool_mu / cool_k
    kSub = 1.0 + 0.1 * (Lsat_rho / Vsat_rho) ** 0.75 * dhSub / dhLat
    kVoid = 1.0 - cool_void

    # Single phase (Dittus-Boelter)
    qw1ph = np.maximum(4.36, 0.023 * cool_re ** 0.8 * Pr ** 0.4) * (cool_k / p["cool_dHyd"]) * dTw

    # Nucleate boiling (Thom)
    qwNB = 2000.0 * np.abs(dTwSat) * dTwSat * np.exp(cool_p / 4.34)

    # Critical heat flux
    qwCHF = 0.14 * dhLat * (9.81 * Lsat_sig * Vsat_rho ** 2 * drhoSat) ** 0.25 * kSub * kVoid

    # Wall temperature at CHF
    TCHF = cool_Tsat + np.sqrt(qwCHF * np.exp(-cool_p / 4.34) / 2000.0)

    # Film boiling
    qwFB = 0.25 * (9.81 * Vsat_k ** 2 * Vsat_cp * drhoSat / Vsat_nu) ** 0.333 * kSub * dTwSat

    # --- flow regime selection ---
    qw = np.empty(nz)
    regime = np.empty(nz, dtype=int)
    for iz in range(nz):
        if Twall[iz] < cool_Tsat[iz]:
            regime[iz] = 1
            qw[iz] = qw1ph[iz]
        elif Twall[iz] < TCHF[iz]:
            regime[iz] = 2
            qw[iz] = max(qw1ph[iz], qwNB[iz])
        else:
            regime[iz] = 3
            qw[iz] = max(qw1ph[iz] * (1 - kVoid[iz]), qwFB[iz] * kVoid[iz])

    # --- coolant enthalpy rate ---
    h_inlet = h2o_enthalpy(p_outlet, T_inlet)
    h_junc = np.concatenate(([h_inlet], cool_h))
    rate_cool_h = (
        (-np.diff(mdot_junc * h_junc) + qw * p["cool_areaHX"])
        / (cool_rho * p["cool_volFlow"])
    )

    return dict(
        cool_T=cool_T, cool_p=cool_p, cool_rho=cool_rho,
        cool_vel=cool_vel, cool_re=cool_re,
        cool_void=cool_void, cool_x=cool_x, cool_Tsat=cool_Tsat,
        TCHF=TCHF, regime=regime, qw=qw,
        rate_cool_h=rate_cool_h,
        Lsat_rho=Lsat_rho, Lsat_h=Lsat_h,
        Vsat_h=Vsat_h,
    )


# ============================================================================
# RHS function
# ============================================================================


def _rhs(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    """Right-hand side for the ODE system."""
    nz, nf, nc = p["nz"], p["nf"], p["nc"]
    dz0 = p["dz0"]
    por = p["porosity"]
    Bu = p["burnup"]

    state = _unpack(y, p)
    power = state["power"]
    cDNP = state["cDNP"]
    fuel_T = state["fuel_T"]  # (nz, nf)
    clad_T = state["clad_T"]  # (nz, nc)
    cool_h = state["cool_h"]  # (nz,)
    clad_epsPeff = state["clad_epsPeff"]  # (nz, nc)
    clad_epsP = state["clad_epsP"]  # (3, nz, nc)

    # ---- Coolant TH (uses previous-step pressure from cool_h) ----
    # For the iterative pressure solve, use ideal gas law estimate
    # or a fixed initial guess based on outlet pressure
    p0_out = _interp_table(p["outlet_p_table"], t)
    cool_p_guess = np.full(nz, p0_out)

    cool = _compute_coolant(t, cool_h, cool_p_guess, clad_T[:, -1], p)
    cool_p = cool["cool_p"]
    cool_T = cool["cool_T"]

    # ---- Fuel strain (simplified: thermal expansion only) ----
    fuel_Tavg = np.sum(p["fuel_v0"] * fuel_T, axis=1) / np.sum(p["fuel_v0"], axis=1)  # (nz,)
    fuel_epsT = matpro.uo2_thexp(fuel_Tavg)  # (nz,)
    fuel_r = p["fuel_r0"][-1] * (1.0 + fuel_epsT)  # (nz,)
    fuel_dz = dz0 * (1.0 + fuel_epsT)  # (nz,)

    # ---- Inner gas pressure ----
    gap_T = 0.5 * (fuel_T[:, -1] + clad_T[:, 0])  # (nz,)
    v_gap = fuel_dz * np.pi * (p["clad_r0"][0] ** 2 - fuel_r ** 2)  # simplified
    ingas_Tplenum = cool_T[-1]
    ingas_p = (
        p["mu_He0"] * 8.31e-6
        / (p["v_gas_plenum"] / ingas_Tplenum + np.sum(v_gap / gap_T))
    )

    # ---- Clad stress (algebraic solve) ----
    stress = _solve_clad_stress(clad_T, clad_epsP, ingas_p, cool_p, fuel_epsT, p)
    clad_sig_r = stress["clad_sig_r"]  # (nz, nc)
    clad_sig_h = stress["clad_sig_h"]
    clad_sig_z = stress["clad_sig_z"]

    # ---- Clad strains ----
    clad_epsT = matpro.zry_thexp(clad_T)  # (nz, nc)
    clad_E = matpro.zry_E(clad_T)
    clad_nu = matpro.ZRY_NU
    sigSum = clad_sig_r + clad_sig_h + clad_sig_z

    clad_eps = np.empty((3, nz, nc))
    clad_eps_mid = np.empty((3, nz, nc - 1))
    for i, sig_i in enumerate([clad_sig_r, clad_sig_h, clad_sig_z]):
        epsE = (sig_i - clad_nu * (sigSum - sig_i)) / clad_E
        clad_eps[i] = clad_epsT + epsE + clad_epsP[i]
        clad_eps_mid[i] = _interp_between(clad_eps[i])

    # Von Mises stress
    clad_sigVM = _von_mises(clad_sig_r, clad_sig_h, clad_sig_z) + 1e-6

    # Effective plastic strain rate
    rate_epsPeff = np.minimum(
        0.1,
        1e-3 * (
            clad_sigVM * 1e6
            / matpro.zry_K(clad_T)
            / np.abs(clad_epsPeff + 1e-6) ** matpro.zry_n(clad_T)
        ) ** (1.0 / matpro.zry_m(clad_T))
    )

    # Plastic strain rate components
    rate_epsP = np.empty((3, nz, nc))
    for i, sig_i in enumerate([clad_sig_r, clad_sig_h, clad_sig_z]):
        rate_epsP[i] = rate_epsPeff * 1.5 * (sig_i - sigSum / 3.0) / clad_sigVM

    # ---- Update clad geometry ----
    clad_r = p["clad_r0"][np.newaxis, :] * (1.0 + clad_eps[1])  # (nz, nc)

    # ---- Clad deformed areas ----
    clad_r_mid = _interp_between(clad_r)  # (nz, nc-1)
    clad_dz_def = dz0 * (1.0 + clad_eps[2])  # (nz, nc)
    clad_a_mid_def = 2 * np.pi * clad_r_mid * _interp_between(clad_dz_def)  # (nz, nc-1)

    # ---- Gap geometry ----
    gap_dr = clad_r[:, 0] - fuel_r  # (nz,)
    p["_gap_dr"] = gap_dr  # cache for event detection

    # Deformed gap area
    gap_r_mid = 0.5 * (clad_r[:, 0] + fuel_r)  # (nz,)
    gap_a_def = 2 * np.pi * gap_r_mid * 0.5 * (clad_dz_def[:, 0] + fuel_dz)  # (nz,)

    gap_open = p["gap_open"]
    k_He = matpro.k_He(gap_T)
    if gap_open:
        gap_h = k_He / np.maximum(gap_dr, 1e-10)
    else:
        gap_h = k_He / p["roughness"]

    # ---- Power distribution ----
    LHGR = power / (dz0 * nz)
    fuel_qV = np.outer(LHGR * dz0 * p["kz"] / np.sum(p["fuel_v0"], axis=1), np.ones(nf))

    # ---- Reactivity ----
    reac_dop = p["dopC"] * np.mean(fuel_Tavg)
    reac_cool = p["coolTemC"] * np.mean(cool_T)

    if not p["transient"]:
        p["reac_dop_bias"] = reac_dop
        p["reac_cool_bias"] = reac_cool

    reac_dop -= p["reac_dop_bias"]
    reac_cool -= p["reac_cool_bias"]
    reac_total = reac_dop + reac_cool

    # Store for snapshot collection
    p["_reac_dop"] = reac_dop
    p["_reac_cool"] = reac_cool
    p["_reac_total"] = reac_total

    # ---- Point kinetics ----
    beta_total = np.sum(p["beff"])
    rate_power = (reac_total - beta_total) * power / p["tLife"] + p["lmb"] @ cDNP
    rate_cDNP = p["beff"] / p["tLife"] * power - p["lmb"] * cDNP

    # ---- Fuel thermal calculation ----
    # Temperature at fuel node boundaries (nz, nf-1)
    Tf_mid = _interp_between(fuel_T)
    # Temperature at clad node boundaries (nz, nc-1)
    Tc_mid = _interp_between(clad_T)

    # Gap heat flux
    qGap = gap_h * (fuel_T[:, -1] - clad_T[:, 0])  # (nz,)

    # Fuel radial heat flux
    qFuel_mid = -matpro.uo2_k(Tf_mid, Bu, por) * np.diff(fuel_T, axis=1) / p["fuel_dr0"]
    # Boundary conditions: zero flux at fuel center, gap flux at outer surface
    QFuel = np.concatenate([
        np.zeros((nz, 1)),
        qFuel_mid * p["fuel_a_mid"],           # fuel area stays initial (MATLAB convention)
        (qGap * gap_a_def)[:, np.newaxis],     # deformed gap area
    ], axis=1)  # (nz, nf+1)

    # Clad radial heat flux
    qClad_mid = -matpro.zry_k(Tc_mid) * np.diff(clad_T, axis=1) / p["clad_dr0"]
    QClad = np.concatenate([
        (qGap * gap_a_def)[:, np.newaxis],     # deformed gap area
        qClad_mid * clad_a_mid_def,            # deformed clad area
        (cool["qw"] * p["cool_areaHX"])[:, np.newaxis],
    ], axis=1)  # (nz, nc+1)

    rate_fuel_T = (
        (-np.diff(QFuel, axis=1) + fuel_qV * p["fuel_v0"])
        / (p["fuel_mass"] * matpro.uo2_cp(fuel_T))
    )

    rate_clad_T = (
        -np.diff(QClad, axis=1)
        / (matpro.ZRY_RHO * matpro.zry_cp(clad_T) * p["clad_v0"])
    )

    # ---- Pack output ----
    dydt = np.concatenate([
        [rate_power],
        rate_cDNP,
        rate_fuel_T.ravel(),
        rate_clad_T.ravel(),
        cool["rate_cool_h"],
        rate_epsPeff.ravel(),
        rate_epsP[0].ravel(),
        rate_epsP[1].ravel(),
        rate_epsP[2].ravel(),
    ])
    return dydt


# ============================================================================
# Event function
# ============================================================================


def _gap_closure_event(t: float, y: np.ndarray, p: dict) -> float:
    """Terminal event: returns min(gap_dr - roughness)."""
    if "_gap_dr" in p:
        val = float(np.min(p["_gap_dr"] - p["roughness"]))
        return val

    # Fallback: compute from state
    state = _unpack(y, p)
    fuel_Tavg = np.sum(p["fuel_v0"] * state["fuel_T"], axis=1) / np.sum(p["fuel_v0"], axis=1)
    fuel_epsT = matpro.uo2_thexp(fuel_Tavg)
    fuel_r = p["fuel_r0"][-1] * (1.0 + fuel_epsT)

    clad_epsT = matpro.zry_thexp(state["clad_T"])
    clad_r_in = p["clad_r0"][0] * (1.0 + clad_epsT[:, 0])

    gap_dr = clad_r_in - fuel_r
    return float(np.min(gap_dr - p["roughness"]))


_gap_closure_event.terminal = True
_gap_closure_event.direction = -1


# ============================================================================
# Snapshot collector
# ============================================================================


def _collect_snapshot(t: float, y: np.ndarray, p: dict) -> dict:
    """Evaluate all derived quantities at a single time/state."""
    nz, nf, nc = p["nz"], p["nf"], p["nc"]
    dz0 = p["dz0"]
    por = p["porosity"]
    Bu = p["burnup"]

    state = _unpack(y, p)
    power = state["power"]
    cDNP = state["cDNP"]
    fuel_T = state["fuel_T"]
    clad_T = state["clad_T"]
    cool_h = state["cool_h"]
    clad_epsPeff = state["clad_epsPeff"]
    clad_epsP = state["clad_epsP"]

    # Coolant
    p0_out = _interp_table(p["outlet_p_table"], t)
    cool_p_guess = np.full(nz, p0_out)
    cool = _compute_coolant(t, cool_h, cool_p_guess, clad_T[:, -1], p)

    # Fuel
    fuel_Tavg = np.sum(p["fuel_v0"] * fuel_T, axis=1) / np.sum(p["fuel_v0"], axis=1)
    fuel_epsT = matpro.uo2_thexp(fuel_Tavg)
    fuel_r = p["fuel_r0"][-1] * (1.0 + fuel_epsT)
    fuel_dz = dz0 * (1.0 + fuel_epsT)

    # Gas pressure
    gap_T = 0.5 * (fuel_T[:, -1] + clad_T[:, 0])
    v_gap = fuel_dz * np.pi * (p["clad_r0"][0] ** 2 - fuel_r ** 2)
    ingas_Tplenum = cool["cool_T"][-1]
    ingas_p = (
        p["mu_He0"] * 8.31e-6
        / (p["v_gas_plenum"] / ingas_Tplenum + np.sum(v_gap / gap_T))
    )

    # Stress
    stress = _solve_clad_stress(clad_T, clad_epsP, ingas_p, cool["cool_p"], fuel_epsT, p)
    clad_sig_r = stress["clad_sig_r"]
    clad_sig_h = stress["clad_sig_h"]
    clad_sig_z = stress["clad_sig_z"]

    # Clad total strain -> geometry
    clad_E = matpro.zry_E(clad_T)
    clad_nu = matpro.ZRY_NU
    sigSum = clad_sig_r + clad_sig_h + clad_sig_z
    clad_eps_h = matpro.zry_thexp(clad_T) + (
        (clad_sig_h - clad_nu * (sigSum - clad_sig_h)) / clad_E
    ) + clad_epsP[1]
    clad_r = p["clad_r0"][np.newaxis, :] * (1.0 + clad_eps_h)

    gap_dr = clad_r[:, 0] - fuel_r

    # Gap conductance
    k_He = matpro.k_He(gap_T)
    if p["gap_open"]:
        gap_hc = k_He / np.maximum(gap_dr, 1e-10)
    else:
        gap_hc = k_He / p["roughness"]

    # Engineering stress
    clad_sigI = (
        (ingas_p - cool["cool_p"])
        * (clad_r[:, -1] + clad_r[:, 0]) / 2.0
        / (clad_r[:, -1] - clad_r[:, 0])
    )

    # Reactivity
    reac_dop = p["dopC"] * np.mean(fuel_Tavg)
    reac_cool = p["coolTemC"] * np.mean(cool["cool_T"])
    reac_dop -= p["reac_dop_bias"]
    reac_cool -= p["reac_cool_bias"]
    reac_total = reac_dop + reac_cool

    return dict(
        power=power / p["pow0"],
        cDNP=cDNP,
        reac_dop=reac_dop * 1e5,  # pcm
        reac_cool=reac_cool * 1e5,
        reac_total=reac_total * 1e5,
        ingas_p=ingas_p,
        fuel_T=fuel_T - 273.15,  # Celsius
        clad_T=clad_T - 273.15,
        cool_T=cool["cool_T"] - 273.15,
        cool_Tsat=cool["cool_Tsat"] - 273.15,
        cool_TCHF=cool["TCHF"] - 273.15,
        cool_regime=cool["regime"],
        cool_p=cool["cool_p"],
        cool_h=cool_h / 1e3,  # kJ/kg
        cool_vel=cool["cool_vel"],
        cool_void=cool["cool_void"],
        fuel_r_out=fuel_r * 1e3,  # mm
        clad_r_in=clad_r[:, 0] * 1e3,
        clad_r_out=clad_r[:, -1] * 1e3,
        clad_sig_r=clad_sig_r,
        clad_sig_h=clad_sig_h,
        clad_sig_z=clad_sig_z,
        clad_sig_eng=clad_sigI,
        gap_dr=gap_dr * 1e6,  # microns
        gap_h=gap_hc,
        clad_epsPeff=clad_epsPeff * 100.0,  # percent
    )


# ============================================================================
# Main entry point
# ============================================================================


def solve_reactor_kinetics(
    params: KineticsParams | None = None,
    verbose: bool = True,
) -> KineticsResult:
    """Simulate a Reactivity Insertion Accident.

    Parameters
    ----------
    params : KineticsParams, optional
        Problem specification.  Uses defaults if *None*.
    verbose : bool
        Print progress messages.

    Returns
    -------
    KineticsResult
    """
    if params is None:
        params = KineticsParams()

    y0, p = _initialize(params)

    # ====================================================================
    # Phase 1: Steady state (0 -> t_steady_end)
    # ====================================================================
    t_eval_ss = np.arange(0.0, params.t_steady_end + 0.5 * params.dt_steady,
                          params.dt_steady)
    p["transient"] = False

    if verbose:
        print("Phase 1: steady state (0 -- {:.0f} s) ...".format(params.t_steady_end))

    sol_ss = solve_ivp(
        fun=lambda t, y: _rhs(t, y, p),
        t_span=(0.0, params.t_steady_end),
        y0=y0,
        method="BDF",
        t_eval=t_eval_ss,
        rtol=params.rtol,
        atol=params.atol,
        max_step=params.max_step_steady,
    )
    if not sol_ss.success:
        raise RuntimeError(f"Steady-state integration failed: {sol_ss.message}")

    # Freeze the reactivity bias from the end of steady state
    # (it was being updated on every RHS call; now lock it)
    if verbose:
        print("  Steady-state bias locked: "
              f"Doppler = {p['reac_dop_bias']:.6e}, "
              f"Coolant = {p['reac_cool_bias']:.6e}")

    # Collect steady-state snapshots
    snapshots: list[tuple[float, dict]] = []
    for i in range(len(sol_ss.t)):
        snap = _collect_snapshot(sol_ss.t[i], sol_ss.y[:, i], p)
        snapshots.append((sol_ss.t[i], snap))

    # ====================================================================
    # Phase 2: Transient with open gap (t_steady_end -> t_transient_end)
    # ====================================================================
    p["transient"] = True
    y_trans = sol_ss.y[:, -1].copy()

    t_eval_tr = np.arange(
        params.t_steady_end + params.dt_transient,
        params.t_transient_end + 0.5 * params.dt_transient,
        params.dt_transient,
    )

    if verbose:
        print("Phase 2: transient ({:.0f} -- {:.0f} s), gap open ...".format(
            params.t_steady_end, params.t_transient_end))

    # Chunked integration with manual event detection (avoids scipy brentq
    # sign errors).  Each chunk spans one output interval (dt_transient).
    t_cur = params.t_steady_end
    y_cur = y_trans.copy()

    # Evaluate event at the start of the transient
    _rhs(t_cur, y_cur, p)
    ev_prev = _gap_closure_event(t_cur, y_cur, p)
    t_closure = None
    y_closure = None

    for t_next in t_eval_tr:
        sol_chunk = solve_ivp(
            fun=lambda t, y: _rhs(t, y, p),
            t_span=(t_cur, t_next),
            y0=y_cur,
            method="BDF",
            rtol=params.rtol,
            atol=params.atol,
            max_step=params.max_step_transient,
            dense_output=True,
        )
        if not sol_chunk.success:
            raise RuntimeError(f"Transient integration failed at t={t_cur:.4f}: "
                               f"{sol_chunk.message}")

        y_end = sol_chunk.y[:, -1]
        _rhs(t_next, y_end, p)
        ev_now = _gap_closure_event(t_next, y_end, p)

        if ev_prev > 0 and ev_now <= 0:
            # Gap closure detected in this chunk — refine via bisection
            t_lo, t_hi = t_cur, t_next
            for _ in range(40):  # ~1e-12 precision
                t_mid = 0.5 * (t_lo + t_hi)
                y_mid = sol_chunk.sol(t_mid)
                _rhs(t_mid, y_mid, p)
                ev_mid = _gap_closure_event(t_mid, y_mid, p)
                if ev_mid > 0:
                    t_lo = t_mid
                else:
                    t_hi = t_mid

            t_closure = t_hi
            y_closure = sol_chunk.sol(t_closure)
            if verbose:
                print(f"  Gap closure at t = {t_closure:.6f} s")
            break

        # No event — record snapshot at this output time
        snap = _collect_snapshot(t_next, y_end, p)
        snapshots.append((t_next, snap))

        y_cur = y_end.copy()
        t_cur = t_next
        ev_prev = ev_now

    # ====================================================================
    # Phase 3: Transient with closed gap (if gap closure detected)
    # ====================================================================
    if t_closure is not None:
        # Compute strain jumps at closure
        state_cl = _unpack(y_closure, p)
        fuel_Tavg = (
            np.sum(p["fuel_v0"] * state_cl["fuel_T"], axis=1)
            / np.sum(p["fuel_v0"], axis=1)
        )
        fuel_epsT_cl = matpro.uo2_thexp(fuel_Tavg)

        clad_T_cl = state_cl["clad_T"]
        clad_epsT_cl = matpro.zry_thexp(clad_T_cl)

        # Compute clad total strain at closure for BC jump conditions
        cool_p_est = np.full(p["nz"], _interp_table(p["outlet_p_table"], t_closure))
        cool_cl = _compute_coolant(t_closure, state_cl["cool_h"], cool_p_est,
                                   clad_T_cl[:, -1], p)
        ingas_Tplenum = cool_cl["cool_T"][-1]
        fuel_r_cl = p["fuel_r0"][-1] * (1.0 + fuel_epsT_cl)
        fuel_dz_cl = p["dz0"] * (1.0 + fuel_epsT_cl)
        gap_T_cl = 0.5 * (state_cl["fuel_T"][:, -1] + clad_T_cl[:, 0])
        v_gap_cl = fuel_dz_cl * np.pi * (p["clad_r0"][0] ** 2 - fuel_r_cl ** 2)
        ingas_p_cl = (
            p["mu_He0"] * 8.31e-6
            / (p["v_gas_plenum"] / ingas_Tplenum + np.sum(v_gap_cl / gap_T_cl))
        )

        stress_cl = _solve_clad_stress(
            clad_T_cl, state_cl["clad_epsP"], ingas_p_cl, cool_cl["cool_p"],
            fuel_epsT_cl, p,
        )

        clad_E_cl = matpro.zry_E(clad_T_cl)
        sigSum_cl = stress_cl["clad_sig_r"] + stress_cl["clad_sig_h"] + stress_cl["clad_sig_z"]
        clad_eps_cl = np.empty((3, p["nz"], p["nc"]))
        for i, sig_i in enumerate([stress_cl["clad_sig_r"], stress_cl["clad_sig_h"],
                                   stress_cl["clad_sig_z"]]):
            epsE_i = (sig_i - matpro.ZRY_NU * (sigSum_cl - sig_i)) / clad_E_cl
            clad_eps_cl[i] = clad_epsT_cl + epsE_i + state_cl["clad_epsP"][i]

        p["gap_open"] = False
        p["gap_clsd"] = True
        p["gap_depsz"] = clad_eps_cl[2, :, 0] - fuel_epsT_cl
        p["gap_depsh"] = clad_eps_cl[1, :, 0] - fuel_epsT_cl

        t_eval_cl = np.arange(
            t_closure + params.dt_transient,
            params.t_transient_end + 0.5 * params.dt_transient,
            params.dt_transient,
        )
        t_eval_cl = t_eval_cl[t_eval_cl <= params.t_transient_end]

        if verbose:
            print("Phase 3: transient ({:.4f} -- {:.0f} s), gap closed ...".format(
                t_closure, params.t_transient_end))

        sol_cl = solve_ivp(
            fun=lambda t, y: _rhs(t, y, p),
            t_span=(t_closure, params.t_transient_end),
            y0=y_closure,
            method="BDF",
            t_eval=t_eval_cl,
            rtol=params.rtol,
            atol=params.atol,
            max_step=params.max_step_transient,
        )
        if not sol_cl.success:
            raise RuntimeError(f"Closed-gap integration failed: {sol_cl.message}")

        for i in range(len(sol_cl.t)):
            snap = _collect_snapshot(sol_cl.t[i], sol_cl.y[:, i], p)
            snapshots.append((sol_cl.t[i], snap))

    # ====================================================================
    # Assemble results
    # ====================================================================
    if verbose:
        print("Collecting results ...")

    nt = len(snapshots)
    nz, nf, nc = p["nz"], p["nf"], p["nc"]

    time_arr = np.array([s[0] for s in snapshots])
    power_arr = np.array([s[1]["power"] for s in snapshots])
    reac_dop = np.array([s[1]["reac_dop"] for s in snapshots])
    reac_cool = np.array([s[1]["reac_cool"] for s in snapshots])
    reac_total = np.array([s[1]["reac_total"] for s in snapshots])
    cDNP_arr = np.array([s[1]["cDNP"] for s in snapshots]).T  # (6, nt)
    ingas_p_arr = np.array([s[1]["ingas_p"] for s in snapshots])

    # 2-D arrays (nz, nt) or (nz, nr, nt)
    fuel_T_arr = np.empty((nz, nf, nt))
    clad_T_arr = np.empty((nz, nc, nt))
    cool_T_arr = np.empty((nz, nt))
    cool_Tsat_arr = np.empty((nz, nt))
    cool_TCHF_arr = np.empty((nz, nt))
    cool_regime_arr = np.empty((nz, nt), dtype=int)
    cool_p_arr = np.empty((nz, nt))
    cool_h_arr = np.empty((nz, nt))
    cool_vel_arr = np.empty((nz, nt))
    cool_void_arr = np.empty((nz, nt))
    fuel_r_out_arr = np.empty((nz, nt))
    clad_r_in_arr = np.empty((nz, nt))
    clad_r_out_arr = np.empty((nz, nt))
    clad_sig_r_arr = np.empty((nz, nc, nt))
    clad_sig_h_arr = np.empty((nz, nc, nt))
    clad_sig_z_arr = np.empty((nz, nc, nt))
    clad_sig_eng_arr = np.empty((nz, nt))
    gap_dr_arr = np.empty((nz, nt))
    gap_h_arr = np.empty((nz, nt))
    clad_epsPeff_arr = np.empty((nz, nc, nt))

    for i, (_, snap) in enumerate(snapshots):
        fuel_T_arr[:, :, i] = snap["fuel_T"]
        clad_T_arr[:, :, i] = snap["clad_T"]
        cool_T_arr[:, i] = snap["cool_T"]
        cool_Tsat_arr[:, i] = snap["cool_Tsat"]
        cool_TCHF_arr[:, i] = snap["cool_TCHF"]
        cool_regime_arr[:, i] = snap["cool_regime"]
        cool_p_arr[:, i] = snap["cool_p"]
        cool_h_arr[:, i] = snap["cool_h"]
        cool_vel_arr[:, i] = snap["cool_vel"]
        cool_void_arr[:, i] = snap["cool_void"]
        fuel_r_out_arr[:, i] = snap["fuel_r_out"]
        clad_r_in_arr[:, i] = snap["clad_r_in"]
        clad_r_out_arr[:, i] = snap["clad_r_out"]
        clad_sig_r_arr[:, :, i] = snap["clad_sig_r"]
        clad_sig_h_arr[:, :, i] = snap["clad_sig_h"]
        clad_sig_z_arr[:, :, i] = snap["clad_sig_z"]
        clad_sig_eng_arr[:, i] = snap["clad_sig_eng"]
        gap_dr_arr[:, i] = snap["gap_dr"]
        gap_h_arr[:, i] = snap["gap_h"]
        clad_epsPeff_arr[:, :, i] = snap["clad_epsPeff"]

    if verbose:
        print("Done.")

    return KineticsResult(
        time=time_arr,
        power=power_arr,
        reac_doppler=reac_dop,
        reac_coolant=reac_cool,
        reac_total=reac_total,
        cDNP=cDNP_arr,
        ingas_p=ingas_p_arr,
        fuel_T=fuel_T_arr,
        clad_T=clad_T_arr,
        cool_T=cool_T_arr,
        cool_Tsat=cool_Tsat_arr,
        cool_TCHF=cool_TCHF_arr,
        cool_regime=cool_regime_arr,
        cool_p=cool_p_arr,
        cool_h=cool_h_arr,
        cool_vel=cool_vel_arr,
        cool_void=cool_void_arr,
        fuel_r_out=fuel_r_out_arr,
        clad_r_in=clad_r_in_arr,
        clad_r_out=clad_r_out_arr,
        clad_sig_r=clad_sig_r_arr,
        clad_sig_h=clad_sig_h_arr,
        clad_sig_z=clad_sig_z_arr,
        clad_sig_eng=clad_sig_eng_arr,
        gap_dr=gap_dr_arr,
        gap_h=gap_h_arr,
        clad_epsPeff=clad_epsPeff_arr,
        params=params,
    )
