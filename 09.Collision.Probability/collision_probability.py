"""Collision probability (CP) method for a cylindrical PWR pin cell.

Solves the multi-group neutron transport equation using the collision
probability method with Wigner-Seitz cylindrical cell approximation
and white boundary condition for an infinite lattice.

The CP matrix is computed by numerical integration over chord heights,
with source-position averaging done analytically via Ki_4 (the
antiderivative of Ki_3).  Both direct and through-centre paths are
included for each source–target pair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from data.macro_xs.mixture import Mixture


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

    For each chord at height y the contribution to P(i→j)*Σ_t,i*V_i is
    computed by integrating Ki_3 over the source position analytically
    (giving Ki_4 second-differences) and summing over both the direct
    and the through-centre path.

    Returns the *infinite-lattice* CP matrix P_inf (N, N) after applying
    the white boundary condition.
    """
    N = geom.N
    V = geom.volumes

    # Optical half-thicknesses  tau_k(y) = Σ_t,k * l_k(y)
    tau = sig_t_g[:, None] * chords          # (N, n_y)

    # --- Build the chord layout for each y-point ---
    # The chord at height y passes through regions from the outermost down
    # to the innermost intersected region and back.  We represent this as
    # an ordered list of *segments*.  For efficiency the layout is encoded
    # in arrays rather than Python lists.
    #
    # Segment ordering (left to right along the chord):
    #   N-1 (left), N-2 (left), …, l (full or left), l (right), …, N-1 (right)
    # where l is the innermost intersected region (the one whose inner
    # boundary is below y).
    #
    # For a given (source_segment, target_segment) pair the contribution
    # to Σ_t,i*V_i*P(i→j) at this y is:
    #
    #   (1/Σ_t,i) * [Ki4(gap) − Ki4(gap+τ_src) − Ki4(gap+τ_tgt) + Ki4(gap+τ_src+τ_tgt)]
    #
    # where gap is the cumulative optical path of all segments strictly
    # between source and target.  τ_src and τ_tgt are the optical
    # thicknesses of the source and target segments.
    #
    # Because the geometry is symmetric about the chord midpoint the
    # contribution of (left_i, left_j) = (right_i, right_j) and
    # (left_i, right_j) = (right_i, left_j).  So we only compute from
    # the *right-half* source and double.

    # --- Compute the "reduced CP" r(i,j) = Σ_t,i*V_i*P_cell(i,j) ---
    rcp = np.zeros((N, N))  # reduced collision probability

    n_y = len(y_pts)

    # Pre-compute cumulative optical path from the right outer boundary
    # (boundary N) going inward to each boundary k.
    # cum[k, :] = sum_{m=k}^{N-1} tau[m, :]  (right-half optical path
    # from boundary k to boundary N)
    cum_from_right = np.zeros((N + 1, n_y))
    for k in range(N - 1, -1, -1):
        cum_from_right[k, :] = cum_from_right[k + 1, :] + tau[k, :]

    # Position of each boundary along the right half of the chord,
    # measured as cumulative optical path from the deepest point:
    # pos[k] = sum_{m=l}^{k-1} tau[m]  where l is innermost intersected.
    # pos[l] = 0  (the deepest point).
    #
    # For the LEFT half, the boundary positions are mirrored:
    # left_pos[k] = pos[k]  (same distance from centre)

    # Determine innermost intersected region for each y-point.
    # Region k is intersected if chords[k, :] > 0.  The innermost
    # intersected region l is the smallest k with chords[k] > 0.
    innermost = np.full(n_y, N, dtype=int)
    for k in range(N):
        mask = (chords[k, :] > 0) & (innermost == N)
        innermost[mask] = k

    # Build boundary positions from the deepest point outward.
    # bnd_pos[k, y] = cumulative tau from innermost region to boundary k.
    # bnd_pos[innermost, y] = 0 for the inner boundary of the innermost.
    # For k > innermost: bnd_pos[k] = sum_{m=innermost}^{k-1} tau[m].
    bnd_pos = np.zeros((N + 1, n_y))  # boundary index 0..N
    for k in range(N):
        bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]
    # Shift so that the innermost boundary is at 0:
    # inner boundary of innermost region l has index l (0-based).
    # For innermost = l, we want bnd_pos_shifted[l] = 0.
    # But inner boundary of region l has index l (the inner radius).
    # Actually, bnd_pos[0] corresponds to the centre (R=0), and
    # bnd_pos[k] = sum_{m=0}^{k-1} tau[m].
    # The deepest point of the chord at height y is *inside* region l.
    # On the right half the position goes from bnd_pos[l] (inner bnd
    # of region l on the right side) to bnd_pos[N] (cell boundary).
    # But for the innermost region the "inner boundary" is actually the
    # centre of the chord — all optical path below boundary l has tau=0
    # because chords=0 for those regions.
    #
    # So bnd_pos already has the correct property:
    # bnd_pos[l] = sum_{m=0}^{l-1} tau[m] = 0  (since tau=0 for m<l).

    # For each pair (i, j), the source is on the right half of region i
    # (from bnd_pos[i] to bnd_pos[i+1], optical thickness tau[i]).
    # Target is on either the right half or the left half of region j.
    #
    # RIGHT-HALF target j:
    #   If j > i (target outward of source):
    #     gap = bnd_pos[j] - bnd_pos[i+1]  = sum tau[k] for k=i+1..j-1
    #   If j < i (target inward of source):
    #     This path goes inward from source — the source needs to go LEFT
    #     through inner regions.  We handle this via the left-half target.
    #   If j == i (self): gap = 0.
    #
    # LEFT-HALF target j:
    #   Path goes from source (right half) leftward through centre and
    #   back out to region j on the left side.
    #   gap = bnd_pos[i] + bnd_pos[j]
    #       = (distance from source inner bnd to centre) +
    #         (distance from centre to target inner bnd on the left)
    #   The left half of region j has the same tau as the right half.

    ki4_0 = _ki4_lookup(np.zeros(n_y), ki_x, ki4_v)  # Ki_4(0) vector

    for i in range(N):
        tau_i = tau[i, :]  # (n_y,)
        sti = sig_t_g[i]
        if sti == 0:
            continue

        # --- Self-same-segment collision ---
        # A neutron at position x in the right-half segment of region i
        # can collide within that SAME segment before exiting it.
        # Going right:  1 − Ki_3(Σ_t(l_i − x))
        # Going left:   1 − Ki_3(Σ_t · x)
        # Integrating over x from 0 to l_i:
        #   2·l_i − (2/Σ_t)[Ki_4(0) − Ki_4(τ_i)]
        # Factor 2 for left-half source (by symmetry).
        self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
            ki4_0 - _ki4_lookup(tau_i, ki_x, ki4_v)
        )
        rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

        for j in range(N):
            tau_j = tau[j, :]

            # --- Same-side path (right-half source → right-half target) ---
            # j > i: going RIGHT (outward) from source.
            #        gap = bnd_pos[j] − bnd_pos[i+1]
            # j < i: going LEFT (inward) from source to an inner target
            #        on the same side.
            #        gap = bnd_pos[i] − bnd_pos[j+1]
            # j == i: handled by self-same term above.
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

            # --- Through-centre path (right-half source → left-half target) ---
            # Valid for ALL j (including j == i: right→left half of same region).
            gap_c = bnd_pos[i, :] + bnd_pos[j, :]

            dc = (_ki4_lookup(gap_c, ki_x, ki4_v)
                  - _ki4_lookup(gap_c + tau_i, ki_x, ki4_v)
                  - _ki4_lookup(gap_c + tau_j, ki_x, ki4_v)
                  + _ki4_lookup(gap_c + tau_i + tau_j, ki_x, ki4_v))

            # Integrate over y (factor 2 for left-half source by symmetry,
            # and 1/Σ_t,i from the x-integration of Ki_3 → Ki_4)
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
