"""Collision probability method for a 1D slab geometry.

Solves the multi-group transport equation for a half-cell slab with
reflective boundary at x = 0 (centre) and white boundary at x = L
(cell edge) for an infinite lattice.  Uses E_3 exponential-integral
kernels — the slab analogue of the Ki_3/Ki_4 kernels used in
cylindrical geometry.

The slab geometry matches the SN exercise:
    fuel half-width  0.9 cm   (5 nodes × 0.2 cm)
    clad thickness   0.2 cm   (1 node × 0.2 cm)
    coolant width    0.7 cm   (4 nodes × 0.2 cm, with half at boundary)
    half-cell        1.8 cm   (= pitch/2)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.special import expn  # expn(3, x) = E_3(x)

from data.macro_xs.mixture import Mixture


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SlabGeometry:
    """1D slab half-cell with sub-regions from centre (x=0) to edge (x=L)."""

    n_fuel: int
    n_clad: int
    n_cool: int
    thicknesses: np.ndarray  # (N,) thickness of each sub-region (cm)
    mat_ids: np.ndarray      # (N,) material ID: 2=fuel, 1=clad, 0=cool
    N: int

    @classmethod
    def default_pwr(
        cls,
        n_fuel: int = 10,
        n_clad: int = 3,
        n_cool: int = 7,
        fuel_half: float = 0.9,
        clad_thick: float = 0.2,
        cool_thick: float = 0.7,
    ) -> SlabGeometry:
        """Build equal-thickness sub-regions for each material zone."""
        N = n_fuel + n_clad + n_cool

        thicknesses = np.empty(N)
        mat_ids = np.empty(N, dtype=int)

        if n_fuel > 0:
            thicknesses[:n_fuel] = fuel_half / n_fuel
            mat_ids[:n_fuel] = 2

        if n_clad > 0:
            thicknesses[n_fuel:n_fuel + n_clad] = clad_thick / n_clad
            mat_ids[n_fuel:n_fuel + n_clad] = 1

        if n_cool > 0:
            thicknesses[n_fuel + n_clad:] = cool_thick / n_cool
            mat_ids[n_fuel + n_clad:] = 0

        return cls(
            n_fuel=n_fuel, n_clad=n_clad, n_cool=n_cool,
            thicknesses=thicknesses, mat_ids=mat_ids, N=N,
        )

    @property
    def half_cell(self) -> float:
        return self.thicknesses.sum()


@dataclass
class SlabCPResult:
    """Results of a slab collision probability calculation."""

    keff: float
    keff_history: list[float]
    flux: np.ndarray           # (N, ng) scalar flux per sub-region
    flux_fuel: np.ndarray      # (ng,) thickness-averaged fuel flux
    flux_clad: np.ndarray
    flux_cool: np.ndarray
    geometry: SlabGeometry
    eg: np.ndarray
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Slab CP matrix (single energy group)
# ---------------------------------------------------------------------------

def _e3(x):
    """Vectorised E_3(x) = integral_0^1 mu exp(-x/mu) dmu."""
    return expn(3, np.maximum(x, 0.0))


def _compute_slab_cp_group(
    sig_t_g: np.ndarray,
    geom: SlabGeometry,
) -> np.ndarray:
    """Within-cell CP matrix for one energy group in slab geometry.

    Uses the E_3 second-difference formula (slab analogue of the Ki_4
    formula for cylindrical geometry).  The half-cell has a reflective
    centre boundary and a white (isotropic re-entry) outer boundary.

    Returns the infinite-lattice CP matrix P_inf (N, N).
    """
    N = geom.N
    t = geom.thicknesses
    tau = sig_t_g * t             # optical thicknesses

    # Cumulative optical path from centre (boundary 0)
    bnd_pos = np.zeros(N + 1)
    for k in range(N):
        bnd_pos[k + 1] = bnd_pos[k] + tau[k]

    # --- Reduced CP: rcp(i,j) = Σ_t,i · t_i · P_cell(i,j) ---
    # All contributions are divided by 2 for direction averaging:
    # each neutron goes right (prob 1/2) or left (prob 1/2).
    rcp = np.zeros((N, N))

    for i in range(N):
        sti = sig_t_g[i]
        tau_i = tau[i]
        if sti <= 0:
            continue

        # Self-same segment (collision within region i from source in i)
        # Going right: t_i - (1/Σ_t)(E_3(0) - E_3(τ_i))
        # Going left:  same by symmetry
        # Sum * Σ_t / 2 (direction average) → rcp contribution:
        self_same = 0.5 * sti * (2 * t[i] - (2.0 / sti) * (0.5 - _e3(tau_i)))
        rcp[i, i] += self_same

        for j in range(N):
            tau_j = tau[j]

            # Same-side path (right→right for j>i, left→left for j<i)
            if j > i:
                gap_d = bnd_pos[j] - bnd_pos[i + 1]
            elif j < i:
                gap_d = bnd_pos[i] - bnd_pos[j + 1]
            else:
                gap_d = None

            if gap_d is not None:
                gap_d = max(gap_d, 0.0)
                dd = (_e3(gap_d) - _e3(gap_d + tau_i)
                      - _e3(gap_d + tau_j) + _e3(gap_d + tau_i + tau_j))
            else:
                dd = 0.0

            # Reflected path (going left to x=0, reflecting, going right)
            gap_c = bnd_pos[i] + bnd_pos[j]
            dc = (_e3(gap_c) - _e3(gap_c + tau_i)
                  - _e3(gap_c + tau_j) + _e3(gap_c + tau_i + tau_j))

            rcp[i, j] += 0.5 * (dd + dc)

    # --- Convert to P_cell ---
    P_cell = np.zeros((N, N))
    for i in range(N):
        if sig_t_g[i] * t[i] > 0:
            P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * t[i])

    # --- Escape (through outer boundary at x = L) ---
    P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)

    # Surface-to-region (white BC, slab perimeter = 2 per unit area)
    # For the half-cell, the "surface" is at x = L.
    # Reciprocity: S · P_in(j) = Σ_t(j)·t(j)·P_out(j)
    # For a slab, S = 1 (unit area surface on each side).
    # But for the half-cell with one reflective and one open surface,
    # S = 1 (the open surface area per unit transverse area).
    P_in = sig_t_g * t * P_out  # reciprocity: P_in(j) * 1 = Σ_t(j)*t(j)*P_out(j)

    P_inout = max(1.0 - P_in.sum(), 0.0)

    # Infinite lattice
    P_inf = P_cell.copy()
    if P_inout < 1.0:
        P_inf += np.outer(P_out, P_in) / (1.0 - P_inout)

    return P_inf


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_slab_cp(
    materials: dict[int, Mixture],
    geom: SlabGeometry | None = None,
    max_outer: int = 500,
    keff_tol: float = 1e-6,
    flux_tol: float = 1e-5,
) -> SlabCPResult:
    """Run the slab collision probability transport calculation."""
    t_start = time.perf_counter()

    if geom is None:
        geom = SlabGeometry.default_pwr()

    _any_mat = next(iter(materials.values()))
    eg = _any_mat.eg
    ng = _any_mat.ng
    N = geom.N
    t = geom.thicknesses

    # --- Per-sub-region cross sections ---
    sig_t = np.empty((N, ng))
    sig_a = np.empty((N, ng))
    sig_p = np.empty((N, ng))
    chi = np.empty((N, ng))

    for k in range(N):
        m = materials[geom.mat_ids[k]]
        sig2_colsum = np.array(m.Sig2.sum(axis=1)).ravel()
        sig_a[k, :] = m.SigF + m.SigC + m.SigL + sig2_colsum
        sig_t[k, :] = sig_a[k, :] + np.array(m.SigS[0].sum(axis=1)).ravel()
        sig_p[k, :] = m.SigP
        chi[k, :] = m.chi

    # --- CP matrices for all energy groups ---
    print(f"  Computing slab CP matrices for {ng} groups, {N} regions ...")
    P_inf = np.empty((N, N, ng))

    for g in range(ng):
        P_inf[:, :, g] = _compute_slab_cp_group(sig_t[:, g], geom)
        if (g + 1) % 100 == 0:
            print(f"    group {g + 1}/{ng}")

    print("  CP matrices done.")

    # --- Power iteration ---
    print("  Starting power iteration ...")
    phi = np.ones((N, ng))
    keff = 1.0
    keff_history: list[float] = []

    scat_mats = {mid: materials[mid].SigS[0] for mid in materials}
    n2n_mats = {mid: materials[mid].Sig2 for mid in materials}

    for n_iter in range(1, max_outer + 1):
        phi_old = phi.copy()
        keff_old = keff

        Q = np.zeros((N, ng))

        fission_rate = np.sum(sig_p * phi, axis=1)
        for k in range(N):
            Q[k, :] += chi[k, :] * fission_rate[k] / keff

        for k in range(N):
            mid = geom.mat_ids[k]
            Q[k, :] += scat_mats[mid].T @ phi[k, :]
            Q[k, :] += 2.0 * (n2n_mats[mid].T @ phi[k, :])

        for g in range(ng):
            source = t * Q[:, g]  # t_j * Q_j,g
            phi[:, g] = P_inf[:, :, g].T @ source

        denom = sig_t * t[:, None]
        pos = denom > 0
        phi[pos] = phi[pos] / denom[pos]
        phi[~pos] = 0.0

        production = np.sum(sig_p * phi * t[:, None])
        absorption = np.sum(sig_a * phi * t[:, None])
        keff = production / absorption
        keff_history.append(keff)

        phi *= 1.0 / np.max(phi)

        delta_k = abs(keff - keff_old)
        delta_phi = np.max(np.abs(phi - phi_old)) / max(np.max(np.abs(phi)), 1e-30)

        if n_iter <= 5 or n_iter % 10 == 0:
            print(f"    iter {n_iter:4d}  keff = {keff:.6f}  "
                  f"dk = {delta_k:.2e}  dphi = {delta_phi:.2e}")

        if n_iter > 2 and delta_k < keff_tol and delta_phi < flux_tol:
            print(f"    iter {n_iter:4d}  keff = {keff:.6f}  Converged.")
            break

    # --- Post-processing ---
    vol_fuel = t[geom.mat_ids == 2].sum()
    vol_clad = t[geom.mat_ids == 1].sum()
    vol_cool = t[geom.mat_ids == 0].sum()

    flux_fuel = np.zeros(ng)
    flux_clad = np.zeros(ng)
    flux_cool = np.zeros(ng)

    for k in range(N):
        tk = t[k]
        mid = geom.mat_ids[k]
        if mid == 2:
            flux_fuel += phi[k, :] * tk / vol_fuel
        elif mid == 1:
            flux_clad += phi[k, :] * tk / vol_clad
        else:
            flux_cool += phi[k, :] * tk / vol_cool

    elapsed = time.perf_counter() - t_start
    print(f"  Elapsed: {elapsed:.1f}s")

    return SlabCPResult(
        keff=keff,
        keff_history=keff_history,
        flux=phi,
        flux_fuel=flux_fuel,
        flux_clad=flux_clad,
        flux_cool=flux_cool,
        geometry=geom,
        eg=eg,
        elapsed_seconds=elapsed,
    )
