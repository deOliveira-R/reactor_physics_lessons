#!/usr/bin/env python3
"""Collision probability (CP) method for a cylindrical PWR pin cell.

Physics
-------
This script solves the multi-group neutron transport equation using the
collision probability method with the Wigner-Seitz cylindrical cell
approximation and a white boundary condition for an infinite lattice.

In the CP method the integral transport equation is solved directly:
for each energy group g, the scalar flux in region i is

    Sigma_t,i * V_i * phi_i = sum_j P_{j->i} * V_j * Q_j

where P_{j->i} is the collision probability (the probability that a
neutron born uniformly and isotropically in region j has its next
collision in region i), and Q_j is the total source in region j
(fission + scattering).

The CP matrix is computed by numerical integration over chord heights y
through the Wigner-Seitz cylinder. Source-position averaging is done
analytically via Ki_4, the antiderivative of the Bickley-Naylor function
Ki_3.  Both direct and through-centre paths are included for each
source-target pair.

The infinite-lattice correction applies a white (isotropic re-entry)
boundary condition: neutrons escaping the cell re-enter uniformly.

The eigenvalue keff is found by outer power iterations on the fission
source. Cross-section data comes from the ORPHEUS data library.

Reference results:
    SN (slab geometry):   keff = 1.04188
    MC (slab geometry):   keff = 1.03484 +/- 0.00192
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.integrate import quad

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CPGeometry:
    """Wigner-Seitz cylindrical pin cell with annular sub-regions."""

    r_fuel: float
    r_clad: float
    r_cell: float
    n_fuel: int
    n_clad: int
    n_cool: int
    radii: np.ndarray    # (N,) outer radius of each sub-region
    volumes: np.ndarray  # (N,) area per unit height (cm^2)
    mat_ids: np.ndarray  # (N,) material ID: 2=fuel, 1=clad, 0=cool
    N: int

    @classmethod
    def default_pwr(
        cls,
        n_fuel: int = 10,
        n_clad: int = 3,
        n_cool: int = 7,
        r_fuel: float = 0.9,
        r_clad: float = 1.1,
        pitch: float = 3.6,
    ) -> CPGeometry:
        """Build equi-volume annular sub-regions for each material zone."""
        r_cell = pitch / np.sqrt(np.pi)
        N = n_fuel + n_clad + n_cool

        radii = np.empty(N)
        mat_ids = np.empty(N, dtype=int)

        if n_fuel > 0:
            for k in range(n_fuel):
                radii[k] = r_fuel * np.sqrt((k + 1) / n_fuel)
                mat_ids[k] = 2

        if n_clad > 0:
            for k in range(n_clad):
                radii[n_fuel + k] = np.sqrt(
                    r_fuel**2 + (k + 1) / n_clad * (r_clad**2 - r_fuel**2)
                )
                mat_ids[n_fuel + k] = 1

        if n_cool > 0:
            for k in range(n_cool):
                radii[n_fuel + n_clad + k] = np.sqrt(
                    r_clad**2 + (k + 1) / n_cool * (r_cell**2 - r_clad**2)
                )
                mat_ids[n_fuel + n_clad + k] = 0

        r_inner = np.zeros(N)
        r_inner[1:] = radii[:-1]
        volumes = np.pi * (radii**2 - r_inner**2)

        return cls(
            r_fuel=r_fuel, r_clad=r_clad, r_cell=r_cell,
            n_fuel=n_fuel, n_clad=n_clad, n_cool=n_cool,
            radii=radii, volumes=volumes, mat_ids=mat_ids, N=N,
        )


@dataclass
class CPParams:
    """Solver parameters for collision probability method."""

    max_outer: int = 500
    keff_tol: float = 1e-6
    flux_tol: float = 1e-5
    n_ki_table: int = 20000
    ki_max: float = 50.0
    n_quad_y: int = 64


@dataclass
class CPResult:
    """Results of a collision probability calculation."""

    keff: float
    keff_history: list[float]
    flux: np.ndarray
    flux_fuel: np.ndarray
    flux_clad: np.ndarray
    flux_cool: np.ndarray
    geometry: CPGeometry
    eg: np.ndarray
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Bickley-Naylor Ki_3 and Ki_4 tables
# ---------------------------------------------------------------------------

def _build_ki_tables(
    n_pts: int = 20000,
    x_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate Ki_3 and Ki_4 on a uniform grid.

    Ki_3(x) = int_0^{pi/2} exp(-x/sin t) sin t dt
    Ki_4(x) = int_x^inf Ki_3(t) dt   (antiderivative of -Ki_3)

    Returns (x_grid, ki3_vals, ki4_vals).
    """
    x_grid = np.linspace(0, x_max, n_pts)
    ki3_vals = np.empty(n_pts)
    ki3_vals[0] = 1.0

    for i in range(1, n_pts):
        ki3_vals[i], _ = quad(
            lambda t, xx=x_grid[i]: np.exp(-xx / np.sin(t)) * np.sin(t),
            0, np.pi / 2,
        )

    # Ki_4 via cumulative integration from right
    dx = x_grid[1] - x_grid[0]
    ki4_vals = np.cumsum(ki3_vals[::-1])[::-1] * dx
    ki4_vals[-1] = 0.0

    return x_grid, ki3_vals, ki4_vals


def _ki4_lookup(x, x_grid, ki4_vals):
    """Vectorised Ki_4 lookup."""
    return np.interp(x, x_grid, ki4_vals, right=0.0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _chord_half_lengths(radii, y_pts):
    """Half-chord lengths l_k(y) for each annular region.  Shape (N, n_y)."""
    N = len(radii)
    chords = np.zeros((N, len(y_pts)))
    r_inner = np.zeros(N)
    r_inner[1:] = radii[:-1]
    y2 = y_pts**2

    for k in range(N):
        r_out, r_in = radii[k], r_inner[k]
        outer = np.sqrt(np.maximum(r_out**2 - y2, 0.0))
        if r_in > 0:
            inner = np.sqrt(np.maximum(r_in**2 - y2, 0.0))
            mask = y_pts < r_in
            chords[k, mask] = outer[mask] - inner[mask]
        mask_p = (y_pts >= r_in) & (y_pts < r_out)
        chords[k, mask_p] = outer[mask_p]

    return chords


# ---------------------------------------------------------------------------
# CP matrix — chord-based Ki_4 formulation (single energy group)
# ---------------------------------------------------------------------------

def _compute_cp_group(
    sig_t_g: np.ndarray,
    geom: CPGeometry,
    chords: np.ndarray,
    y_pts: np.ndarray,
    y_wts: np.ndarray,
    ki_x: np.ndarray,
    ki4_v: np.ndarray,
) -> np.ndarray:
    """Within-cell CP matrix for one energy group.

    For each chord at height y the contribution to P(i->j)*Sig_t,i*V_i is
    computed by integrating Ki_3 over the source position analytically
    (giving Ki_4 second-differences) and summing over both the direct
    and the through-centre path.

    Returns the *infinite-lattice* CP matrix P_inf (N, N) after applying
    the white boundary condition.
    """
    N = geom.N
    V = geom.volumes

    # Optical half-thicknesses  tau_k(y) = Sig_t,k * l_k(y)
    tau = sig_t_g[:, None] * chords          # (N, n_y)

    n_y = len(y_pts)

    # Pre-compute cumulative optical path from the right outer boundary
    # (boundary N) going inward to each boundary k.
    cum_from_right = np.zeros((N + 1, n_y))
    for k in range(N - 1, -1, -1):
        cum_from_right[k, :] = cum_from_right[k + 1, :] + tau[k, :]

    # Determine innermost intersected region for each y-point.
    innermost = np.full(n_y, N, dtype=int)
    for k in range(N):
        mask = (chords[k, :] > 0) & (innermost == N)
        innermost[mask] = k

    # Build boundary positions from the deepest point outward.
    bnd_pos = np.zeros((N + 1, n_y))  # boundary index 0..N
    for k in range(N):
        bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]

    ki4_0 = _ki4_lookup(np.zeros(n_y), ki_x, ki4_v)  # Ki_4(0) vector

    # --- Compute the "reduced CP" r(i,j) = Sig_t,i*V_i*P_cell(i,j) ---
    rcp = np.zeros((N, N))  # reduced collision probability

    for i in range(N):
        tau_i = tau[i, :]  # (n_y,)
        sti = sig_t_g[i]
        if sti == 0:
            continue

        # --- Self-same-segment collision ---
        self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
            ki4_0 - _ki4_lookup(tau_i, ki_x, ki4_v)
        )
        rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

        for j in range(N):
            tau_j = tau[j, :]

            # --- Same-side path (right-half source -> right-half target) ---
            if j > i:
                gap_d = bnd_pos[j, :] - bnd_pos[i + 1, :]
            elif j < i:
                gap_d = bnd_pos[i, :] - bnd_pos[j + 1, :]
            else:
                gap_d = None

            if gap_d is not None:
                gap_d = np.maximum(gap_d, 0.0)
                dd = (_ki4_lookup(gap_d, ki_x, ki4_v)
                      - _ki4_lookup(gap_d + tau_i, ki_x, ki4_v)
                      - _ki4_lookup(gap_d + tau_j, ki_x, ki4_v)
                      + _ki4_lookup(gap_d + tau_i + tau_j, ki_x, ki4_v))
            else:
                dd = np.zeros(n_y)

            # --- Through-centre path (right-half source -> left-half target) ---
            gap_c = bnd_pos[i, :] + bnd_pos[j, :]

            dc = (_ki4_lookup(gap_c, ki_x, ki4_v)
                  - _ki4_lookup(gap_c + tau_i, ki_x, ki4_v)
                  - _ki4_lookup(gap_c + tau_j, ki_x, ki4_v)
                  + _ki4_lookup(gap_c + tau_i + tau_j, ki_x, ki4_v))

            # Integrate over y (factor 2 for left-half source by symmetry,
            # and 1/Sig_t,i from the x-integration of Ki_3 -> Ki_4)
            rcp[i, j] += 2.0 * np.dot(y_wts, dd + dc)

    # --- Convert reduced CP to P_cell ---
    P_cell = np.zeros((N, N))
    for i in range(N):
        if sig_t_g[i] * V[i] > 0:
            P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * V[i])

    # --- Escape, surface-to-region, surface-to-surface probabilities ---
    P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)

    S_cell = 2.0 * np.pi * geom.r_cell
    P_in = sig_t_g * V * P_out / S_cell

    P_inout = max(1.0 - P_in.sum(), 0.0)

    # --- Infinite-lattice CP (white boundary condition) ---
    P_inf = P_cell.copy()
    if P_inout < 1.0:
        P_inf += np.outer(P_out, P_in) / (1.0 - P_inout)

    return P_inf


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_collision_probability(
    materials: dict[int, Mixture],
    geom: CPGeometry | None = None,
    params: CPParams | None = None,
) -> CPResult:
    """Run the collision probability transport calculation."""
    t_start = time.perf_counter()

    if geom is None:
        geom = CPGeometry.default_pwr()
    if params is None:
        params = CPParams()

    _any_mat = next(iter(materials.values()))
    eg = _any_mat.eg
    ng = _any_mat.ng
    N = geom.N

    print("  Building Ki3/Ki4 lookup tables ...")
    ki_x, _, ki4_v = _build_ki_tables(params.n_ki_table, params.ki_max)

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

    # --- y-quadrature (composite Gauss-Legendre) ---
    breakpoints = np.concatenate(([0.0], geom.radii))
    gl_pts, gl_wts = np.polynomial.legendre.leggauss(params.n_quad_y)
    y_all, w_all = [], []
    for seg in range(len(breakpoints) - 1):
        a, b = breakpoints[seg], breakpoints[seg + 1]
        y_all.append(0.5 * (b - a) * gl_pts + 0.5 * (b + a))
        w_all.append(0.5 * (b - a) * gl_wts)
    y_pts = np.concatenate(y_all)
    y_wts = np.concatenate(w_all)

    chords = _chord_half_lengths(geom.radii, y_pts)

    # --- CP matrices for all energy groups ---
    print(f"  Computing CP matrices for {ng} groups, {N} regions ...")
    P_inf = np.empty((N, N, ng))

    for g in range(ng):
        P_inf[:, :, g] = _compute_cp_group(
            sig_t[:, g], geom, chords, y_pts, y_wts, ki_x, ki4_v,
        )
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

    for n_iter in range(1, params.max_outer + 1):
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
            source = geom.volumes * Q[:, g]
            phi[:, g] = P_inf[:, :, g].T @ source

        denom = sig_t * geom.volumes[:, None]
        pos = denom > 0
        phi[pos] = phi[pos] / denom[pos]
        phi[~pos] = 0.0

        production = np.sum(sig_p * phi * geom.volumes[:, None])
        absorption = np.sum(sig_a * phi * geom.volumes[:, None])
        keff = production / absorption
        keff_history.append(keff)

        phi *= 1.0 / np.max(phi)

        delta_k = abs(keff - keff_old)
        delta_phi = np.max(np.abs(phi - phi_old)) / max(np.max(np.abs(phi)), 1e-30)

        if n_iter <= 5 or n_iter % 10 == 0:
            print(f"    iter {n_iter:4d}  keff = {keff:.6f}  "
                  f"dk = {delta_k:.2e}  dphi = {delta_phi:.2e}")

        if n_iter > 2 and delta_k < params.keff_tol and delta_phi < params.flux_tol:
            print(f"    iter {n_iter:4d}  keff = {keff:.6f}  Converged.")
            break

    # --- Post-processing ---
    vol_fuel = geom.volumes[geom.mat_ids == 2].sum()
    vol_clad = geom.volumes[geom.mat_ids == 1].sum()
    vol_cool = geom.volumes[geom.mat_ids == 0].sum()

    flux_fuel = np.zeros(ng)
    flux_clad = np.zeros(ng)
    flux_cool = np.zeros(ng)

    for k in range(N):
        v = geom.volumes[k]
        mid = geom.mat_ids[k]
        if mid == 2:
            flux_fuel += phi[k, :] * v / vol_fuel
        elif mid == 1:
            flux_clad += phi[k, :] * v / vol_clad
        else:
            flux_cool += phi[k, :] * v / vol_cool

    elapsed = time.perf_counter() - t_start
    print(f"  Elapsed: {elapsed:.1f}s")

    return CPResult(
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Material colors
_MAT_COLORS = {2: "#d62728", 1: "#2ca02c", 0: "#1f77b4"}  # fuel, clad, cool
_MAT_LABELS = {2: "Fuel", 1: "Cladding", 0: "Coolant"}


def plot_cp_geometry(
    geom: CPGeometry,
    output_dir: Path,
    filename: str = "CP_01_geometry.pdf",
) -> None:
    """Plot concentric annular sub-regions colored by material."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    r_inner = np.concatenate(([0.0], geom.radii[:-1]))
    plotted_labels = set()

    # Draw from outermost to innermost so inner rings overlap outer
    for k in reversed(range(geom.N)):
        mat_id = geom.mat_ids[k]
        label = _MAT_LABELS[mat_id] if mat_id not in plotted_labels else None
        plotted_labels.add(mat_id)

        circle = Circle(
            (0, 0), geom.radii[k],
            facecolor=_MAT_COLORS[mat_id], edgecolor="k",
            linewidth=0.3, alpha=0.7, label=label,
        )
        ax.add_patch(circle)

    lim = geom.r_cell * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Wigner-Seitz cell: annular sub-regions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_cp_convergence(
    result: CPResult,
    output_dir: Path,
) -> None:
    """Plot keff convergence history."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    iters = range(1, len(result.keff_history) + 1)
    ax.plot(iters, result.keff_history, "-or", markersize=3)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("k-effective")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_02_keff.pdf")
    plt.close(fig)


def plot_cp_spectra(
    result: CPResult,
    output_dir: Path,
) -> None:
    """Plot neutron spectra per unit lethargy in fuel, cladding, coolant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    eg = result.eg
    eg_mid = 0.5 * (eg[:-1] + eg[1:])
    du = np.log(eg[1:] / eg[:-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(eg_mid, result.flux_fuel / du, "-r", label="Fuel")
    ax.semilogx(eg_mid, result.flux_clad / du, "-g", label="Cladding")
    ax.semilogx(eg_mid, result.flux_cool / du, "-b", label="Coolant")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Neutron flux per unit lethargy (a.u.)")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_03_flux_lethargy.pdf")
    plt.close(fig)


def plot_cp_radial_flux(
    result: CPResult,
    output_dir: Path,
) -> None:
    """Plot radial flux profile for thermal, resonance, and fast groups."""
    output_dir.mkdir(parents=True, exist_ok=True)
    geom = result.geometry

    # Mid-radius of each sub-region
    r_inner = np.concatenate(([0.0], geom.radii[:-1]))
    r_mid = 0.5 * (r_inner + geom.radii)

    # Group ranges (same convention as DO/MoC)
    FI_T = result.flux[:, :50].sum(axis=1)     # thermal < 1 eV
    FI_R = result.flux[:, 50:287].sum(axis=1)   # resonance < 0.1 MeV
    FI_F = result.flux[:, 287:].sum(axis=1)     # fast > 0.1 MeV

    fig, ax = plt.subplots()
    ax.plot(r_mid, FI_F, "-or", label="Fast", markersize=4)
    ax.plot(r_mid, FI_R, "-og", label="Resonance", markersize=4)
    ax.plot(r_mid, FI_T, "-ob", label="Thermal", markersize=4)

    # Mark material boundaries
    for r, lbl in [(geom.r_fuel, "Fuel/Clad"), (geom.r_clad, "Clad/Cool")]:
        ax.axvline(r, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Radius (cm)")
    ax.set_ylabel("Neutron flux (a.u.)")
    ax.set_title("Radial flux distribution")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_04_flux_radial.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("COLLISION PROBABILITY — PWR PIN CELL (Wigner-Seitz)")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections (same as SN)
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Set up Wigner-Seitz cylindrical geometry
    geom = CPGeometry.default_pwr(n_fuel=10, n_clad=3, n_cool=7)
    params = CPParams()

    print(f"\n  Geometry: r_fuel = {geom.r_fuel:.3f} cm, "
          f"r_clad = {geom.r_clad:.3f} cm, r_cell = {geom.r_cell:.3f} cm")
    print(f"  Sub-regions: {geom.n_fuel} fuel + {geom.n_clad} clad "
          f"+ {geom.n_cool} cool = {geom.N} total")
    print()

    # 3. Solve
    result = solve_collision_probability(materials, geom, params)

    # 4. Report
    print(f"\n  keff = {result.keff:.5f}  (SN slab reference: 1.04188)")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    output = Path("09_results")
    output.mkdir(parents=True, exist_ok=True)
    plot_cp_geometry(geom, output)
    plot_cp_convergence(result, output)
    plot_cp_spectra(result, output)
    plot_cp_radial_flux(result, output)
    print(f"\n  Plots saved to {output.resolve()}/")


if __name__ == "__main__":
    main()
