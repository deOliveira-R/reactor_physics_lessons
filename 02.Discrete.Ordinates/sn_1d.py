"""1D Discrete Ordinates (SN) neutron transport solver.

Solves the multi-group neutron transport equation on a 1D slab mesh
with reflective boundary conditions using Gauss-Legendre angular
quadrature and diamond-difference spatial discretization.

The 1D transport equation for direction mu_n:
    mu_n * dpsi/dx + Sigma_t * psi = Q / 2

where the /2 is the 1D isotropic source normalization (integral over
[-1,1] of (1/2) dmu = 1).

The solver satisfies the ``EigenvalueSolver`` protocol from
``numerics.eigenvalue`` and can be used with the generic
``power_iteration`` function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse

from data.macro_xs.mixture import Mixture
from data.macro_xs.cell_xs import assemble_cell_xs
from numerics.eigenvalue import power_iteration


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GaussLegendreQuadrature:
    """Gauss-Legendre angular quadrature on [-1, 1].

    Weights sum to 2.0 (the measure of [-1, 1]).
    Points are symmetric about mu=0; the partner of index i is N-1-i.
    """

    mu: np.ndarray       # (N,) quadrature points
    weights: np.ndarray  # (N,) weights summing to 2.0
    N: int

    @classmethod
    def gauss_legendre(cls, n_ordinates: int) -> GaussLegendreQuadrature:
        """Build N-point Gauss-Legendre quadrature.

        Parameters
        ----------
        n_ordinates : int
            Number of quadrature points (must be even for SN).
        """
        mu, w = np.polynomial.legendre.leggauss(n_ordinates)
        return cls(mu=mu, weights=w, N=n_ordinates)


@dataclass
class Slab1DGeometry:
    """1D slab geometry for SN transport.

    A sequence of cells with specified widths and material IDs.
    Boundary conditions are reflective on both sides (infinite lattice).
    """

    cell_widths: np.ndarray   # (N_cells,) cell widths in cm
    mat_ids: np.ndarray       # (N_cells,) material ID per cell
    N: int                    # number of cells

    @classmethod
    def from_benchmark(
        cls,
        n_fuel: int,
        n_mod: int,
        t_fuel: float,
        t_mod: float,
    ) -> Slab1DGeometry:
        """Build a half-cell slab [fuel | moderator] for benchmark comparison.

        Reflective BCs on both sides give the infinite-lattice eigenvalue.
        """
        dx_fuel = t_fuel / n_fuel
        dx_mod = t_mod / n_mod
        widths = np.concatenate([
            np.full(n_fuel, dx_fuel),
            np.full(n_mod, dx_mod),
        ])
        mat_ids = np.concatenate([
            np.full(n_fuel, 2, dtype=int),
            np.full(n_mod, 0, dtype=int),
        ])
        return cls(cell_widths=widths, mat_ids=mat_ids, N=n_fuel + n_mod)

    @classmethod
    def from_regions(
        cls,
        thicknesses: list[float],
        mat_ids_per_region: list[int],
        n_cells_per_region: int = 10,
    ) -> Slab1DGeometry:
        """Build a multi-region slab from thicknesses and material IDs.

        Parameters
        ----------
        thicknesses : thickness of each region (innermost first).
        mat_ids_per_region : material ID for each region.
        n_cells_per_region : number of cells per region.
        """
        widths_list = []
        mids_list = []
        for t, mid in zip(thicknesses, mat_ids_per_region):
            dx = t / n_cells_per_region
            widths_list.append(np.full(n_cells_per_region, dx))
            mids_list.append(np.full(n_cells_per_region, mid, dtype=int))
        widths = np.concatenate(widths_list)
        mids = np.concatenate(mids_list)
        return cls(cell_widths=widths, mat_ids=mids, N=len(widths))

    @classmethod
    def homogeneous(
        cls,
        n_cells: int,
        total_width: float,
        mat_id: int = 2,
    ) -> Slab1DGeometry:
        """Build a homogeneous slab of a single material."""
        dx = total_width / n_cells
        return cls(
            cell_widths=np.full(n_cells, dx),
            mat_ids=np.full(n_cells, mat_id, dtype=int),
            N=n_cells,
        )


@dataclass
class SN1DResult:
    """Results of a 1D SN calculation."""

    keff: float
    keff_history: list[float]
    flux: np.ndarray          # (N_cells, ng) scalar flux per cell
    geometry: Slab1DGeometry
    eg: np.ndarray            # (ng+1,) energy group boundaries
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Solver class (satisfies EigenvalueSolver protocol)
# ---------------------------------------------------------------------------

class SN1DSolver:
    """1D SN eigenvalue solver with diamond-difference transport sweeps.

    The ``solve_fixed_source`` method performs inner scattering iterations:
    for a fixed fission source it sweeps all angles, recomputes the
    scattering source from the updated scalar flux, and repeats until
    the scattering source converges.
    """

    def __init__(
        self,
        materials: dict[int, Mixture],
        geom: Slab1DGeometry,
        quad: GaussLegendreQuadrature,
        keff_tol: float = 1e-7,
        flux_tol: float = 1e-6,
        max_inner: int = 200,
        inner_tol: float = 1e-8,
    ) -> None:
        self.geom = geom
        self.quad = quad
        self.keff_tol = keff_tol
        self.flux_tol = flux_tol
        self.max_inner = max_inner
        self.inner_tol = inner_tol

        _any_mat = next(iter(materials.values()))
        self.ng = _any_mat.ng
        self.nc = geom.N
        self.dx = geom.cell_widths

        # Per-cell cross sections
        xs = assemble_cell_xs(materials, geom.mat_ids)
        self.sig_t = xs.sig_t
        self.sig_a = xs.sig_a
        self.sig_p = xs.sig_p
        self.chi = xs.chi

        # Dense P0 scattering matrices per cell
        nc, ng = self.nc, self.ng
        self.sig_s0 = np.empty((nc, ng, ng))
        for i in range(nc):
            m = materials[geom.mat_ids[i]]
            S0 = m.SigS[0]
            self.sig_s0[i] = S0.toarray() if issparse(S0) else np.asarray(S0)

        # Pre-compute sweep coefficients (positive mu only)
        n_half = quad.N // 2
        self.n_half = n_half
        self.mu_pos = quad.mu[n_half:]
        self.w_pos = quad.weights[n_half:]

        two_mu_pos = 2.0 * self.mu_pos
        denom = two_mu_pos[:, None, None] + \
            self.dx[None, :, None] * self.sig_t[None, :, :]
        self.source_coeff = (0.5 * self.dx)[None, :, None] / denom
        self.stream_coeff = two_mu_pos[:, None, None] / denom

        # Persistent boundary fluxes for reflective BCs
        self.psi_bc_right = np.zeros((n_half, ng))
        self.psi_bc_left = np.zeros((n_half, ng))

    def initial_flux_distribution(self) -> np.ndarray:
        return np.ones((self.nc, self.ng))

    def compute_fission_source(
        self, flux_distribution: np.ndarray, keff: float,
    ) -> np.ndarray:
        fission_rate = np.sum(self.sig_p * flux_distribution, axis=1)
        return self.chi * fission_rate[:, np.newaxis] / keff

    def solve_fixed_source(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Transport sweep with inner scattering iterations."""
        phi = flux_distribution.copy()

        for n_inner in range(self.max_inner):
            phi_prev = phi.copy()

            Q_s = np.einsum('ijk,ij->ik', self.sig_s0, phi)
            Q = fission_source + Q_s

            phi = _transport_sweep_fast(
                Q, self.stream_coeff, self.source_coeff, self.w_pos,
                self.n_half, self.psi_bc_right, self.psi_bc_left,
                self.nc, self.ng,
            )

            norm = np.linalg.norm(phi)
            if norm > 0:
                inner_res = np.linalg.norm(phi - phi_prev) / norm
                if inner_res < self.inner_tol:
                    break

        return phi

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        dx = self.dx[:, np.newaxis]
        production = np.sum(self.sig_p * flux_distribution * dx)
        absorption = np.sum(self.sig_a * flux_distribution * dx)
        return float(production / absorption)

    def converged(
        self, keff: float, keff_old: float,
        flux_distribution: np.ndarray, flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        if iteration <= 2:
            return False
        keff_change = abs(keff - keff_old)
        flux_change = np.linalg.norm(flux_distribution - flux_old) / \
            max(np.linalg.norm(flux_distribution), 1e-30)
        return keff_change < self.keff_tol and flux_change < self.flux_tol


# ---------------------------------------------------------------------------
# Convenience wrapper (preserves existing call signature)
# ---------------------------------------------------------------------------

def solve_sn_1d(
    materials: dict[int, Mixture],
    geom: Slab1DGeometry,
    quad: GaussLegendreQuadrature | None = None,
    max_outer: int = 500,
    keff_tol: float = 1e-7,
    flux_tol: float = 1e-6,
    max_inner: int = 200,
    inner_tol: float = 1e-8,
) -> SN1DResult:
    """Solve the 1D multi-group SN eigenvalue problem.

    Parameters
    ----------
    materials : dict mapping material ID to Mixture.
    geom : Slab1DGeometry — cell widths and material assignment.
    quad : GaussLegendreQuadrature (default: S16, 16 ordinates).
    max_outer : int — maximum power iterations.
    keff_tol, flux_tol : convergence tolerances for outer iteration.
    max_inner, inner_tol : convergence parameters for inner scattering iteration.
    """
    t_start = time.perf_counter()

    if quad is None:
        quad = GaussLegendreQuadrature.gauss_legendre(16)

    _any_mat = next(iter(materials.values()))
    eg = _any_mat.eg

    solver = SN1DSolver(materials, geom, quad,
                        keff_tol=keff_tol, flux_tol=flux_tol,
                        max_inner=max_inner, inner_tol=inner_tol)
    keff, keff_history, phi = power_iteration(solver, max_iter=max_outer)

    elapsed = time.perf_counter() - t_start

    return SN1DResult(
        keff=keff,
        keff_history=keff_history,
        flux=phi,
        geometry=geom,
        eg=eg,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Transport sweep internals
# ---------------------------------------------------------------------------

def _transport_sweep_fast(
    Q: np.ndarray,
    stream_coeff: np.ndarray,
    source_coeff: np.ndarray,
    w_pos: np.ndarray,
    n_half: int,
    psi_bc_right: np.ndarray,
    psi_bc_left: np.ndarray,
    nc: int,
    ng: int,
) -> np.ndarray:
    """Vectorized diamond-difference transport sweep.

    Exploits GL symmetry: mu[N-1-n] = -mu[n], w[N-1-n] = w[n].
    Only sweeps positive directions; negative directions are the
    reverse sweep with the same |mu| and weight.

    The diamond-difference recurrence psi_out[i] = a[i]*psi_in[i] + b[i]*Q[i]
    is solved via cumulative products to avoid Python cell loops.
    """
    phi_new = np.zeros((nc, ng))
    bQ = source_coeff * Q[np.newaxis, :, :]

    for n in range(n_half):
        w = w_pos[n]
        a = stream_coeff[n]
        s = bQ[n]

        psi_fwd = _solve_recurrence_fwd(a, s, psi_bc_left[n])
        psi_bc_right[n, :] = _outgoing_fwd(a, s, psi_bc_left[n], nc)
        phi_new += w * psi_fwd

        a_rev = a[::-1]
        s_rev = s[::-1]
        psi_bwd_rev = _solve_recurrence_fwd(a_rev, s_rev, psi_bc_right[n])
        psi_bc_left[n, :] = _outgoing_fwd(a_rev, s_rev, psi_bc_right[n], nc)
        phi_new += w * psi_bwd_rev[::-1]

    return phi_new


def _solve_recurrence_fwd(
    a: np.ndarray, s: np.ndarray, psi0: np.ndarray,
) -> np.ndarray:
    """Solve diamond-difference recurrence and return cell-average fluxes.

    Recurrence: psi_out[i] = a[i]*psi_in[i] + s[i]
                psi_in[0] = psi0
                psi_in[i+1] = psi_out[i]
                psi_avg[i] = 0.5*(psi_in[i] + psi_out[i])

    Uses cumulative products for vectorization.
    """
    nc, ng = a.shape
    cp = np.cumprod(a, axis=0)
    s_over_cp = s / cp
    cs = np.cumsum(s_over_cp, axis=0)

    psi_in = np.empty((nc, ng))
    psi_in[0] = psi0
    if nc > 1:
        psi_in[1:] = cp[:-1] * (psi0[np.newaxis, :] + cs[:-1])

    psi_out = a * psi_in + s
    return 0.5 * (psi_in + psi_out)


def _outgoing_fwd(
    a: np.ndarray, s: np.ndarray, psi0: np.ndarray, nc: int,
) -> np.ndarray:
    """Compute the outgoing flux at the end of a forward sweep."""
    cp = np.cumprod(a, axis=0)
    s_over_cp = s / cp
    cs = np.cumsum(s_over_cp, axis=0)
    return cp[-1] * (psi0 + cs[-1])
