"""Collision probability (CP) method for neutron transport.

Solves the multi-group neutron transport equation using the collision
probability method with white boundary condition for an infinite lattice.

The geometry-specific kernel is encapsulated in :class:`CPMesh`, an
augmented geometry that wraps a :class:`~geometry.mesh.Mesh1D` and
provides :meth:`~CPMesh.compute_pinf_group`.  Three coordinate systems
are supported:

* **Cartesian** (slab) — E₃ exponential-integral kernel.
* **Cylindrical** (Wigner-Seitz pin) — Ki₃/Ki₄ Bickley-Naylor kernel.
* **Spherical** (shell) — exponential kernel with y-weighted quadrature.

All share the same power iteration via :class:`CPSolver`.

Two solver modes are available:

* **Jacobi** (default) — computes the scattering source from the
  previous iteration's flux for all groups simultaneously.  Simple,
  no inner iterations.
* **Gauss-Seidel** — sweeps group-by-group from fast to thermal,
  using the updated flux from already-processed groups.  Within each
  group, inner iterations converge the within-group scattering.
  Converges faster for problems with strong thermal self-scatter.

.. seealso:: :ref:`theory-collision-probability` — Key Facts, equations, gotchas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.special import expn

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.data.macro_xs.cell_xs import CellXS, assemble_cell_xs
from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.numerics.eigenvalue import power_iteration


# ═══════════════════════════════════════════════════════════════════════
# Parameters and result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CPParams:
    """Solver parameters for the collision probability method."""

    max_outer: int = 500
    keff_tol: float = 1e-6
    flux_tol: float = 1e-5
    n_ki_table: int = 20000
    ki_max: float = 50.0
    n_quad_y: int = 64
    solver_mode: str = "jacobi"    # "jacobi" or "gauss_seidel"
    inner_tol: float = 1e-8        # convergence for inner iterations (GS only)
    max_inner: int = 100           # max inner iterations per group (GS only)


@dataclass
class CPResult:
    """Results of a collision probability calculation."""

    keff: float
    keff_history: list[float]
    flux: np.ndarray
    flux_fuel: np.ndarray
    flux_clad: np.ndarray
    flux_cool: np.ndarray
    geometry: Mesh1D
    eg: np.ndarray
    elapsed_seconds: float
    residual_history: list[float] = field(default_factory=list)
    n_inner: np.ndarray | None = None  # (n_outer, ng) inner iteration counts


# ═══════════════════════════════════════════════════════════════════════
# Helper functions (module-level, used by CPMesh)
# ═══════════════════════════════════════════════════════════════════════

def _e3(x):
    """Vectorised E_3(x) = integral_0^1 mu exp(-x/mu) dmu."""
    return expn(3, np.maximum(x, 0.0))


def _build_ki_tables(
    n_pts: int = 20000,
    x_max: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate Ki_3 and Ki_4 on a uniform grid.

    Ki_3(x) = int_0^{pi/2} exp(-x/sin t) sin t dt
    Ki_4(x) = int_x^inf Ki_3(t) dt

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

    dx = x_grid[1] - x_grid[0]
    ki4_vals = np.cumsum(ki3_vals[::-1])[::-1] * dx
    ki4_vals[-1] = 0.0

    return x_grid, ki3_vals, ki4_vals


def _ki4_lookup(x, x_grid, ki4_vals):
    """Vectorised Ki_4 lookup."""
    return np.interp(x, x_grid, ki4_vals, right=0.0)


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


def _composite_gauss_legendre(breakpoints, n_quad):
    """Composite Gauss-Legendre quadrature over breakpoint intervals.

    Returns (pts, wts) concatenated across all intervals.
    """
    gl_pts, gl_wts = np.polynomial.legendre.leggauss(n_quad)
    y_all, w_all = [], []
    for seg in range(len(breakpoints) - 1):
        a, b = breakpoints[seg], breakpoints[seg + 1]
        y_all.append(0.5 * (b - a) * gl_pts + 0.5 * (b + a))
        w_all.append(0.5 * (b - a) * gl_wts)
    return np.concatenate(y_all), np.concatenate(w_all)


# ═══════════════════════════════════════════════════════════════════════
# CPMesh — augmented geometry for the collision probability method
# ═══════════════════════════════════════════════════════════════════════

class CPMesh:
    """Augmented geometry for the collision probability method.

    Wraps a :class:`~geometry.mesh.Mesh1D` and adds the CP-specific
    kernel, quadrature, and :meth:`compute_pinf_group` method.  The
    kernel is selected automatically based on the mesh coordinate system:

    * **Cartesian** — E₃ kernel (slab), scalar computation.
    * **Cylindrical** — Ki₄ kernel, y-quadrature with chord half-lengths.
    * **Spherical** — exp(-τ) kernel, y-quadrature with y-weighted chords.

    Parameters
    ----------
    mesh : Mesh1D
        Base geometry.
    params : CPParams, optional
        Solver parameters (Ki table size, quadrature order, etc.).
    """

    def __init__(self, mesh: Mesh1D, params: CPParams | None = None) -> None:
        self.mesh = mesh
        self.params = params or CPParams()

        match mesh.coord:
            case CoordSystem.CARTESIAN:
                self._setup_slab()
            case CoordSystem.CYLINDRICAL:
                self._setup_cylindrical()
            case CoordSystem.SPHERICAL:
                self._setup_spherical()
            case _:
                raise ValueError(f"CP not implemented for {mesh.coord}")

    # ── Setup methods ─────────────────────────────────────────────────

    def _setup_slab(self) -> None:
        """Slab: E₃ kernel, no y-quadrature."""
        self._chords = None
        self._y_wts = None
        self._kernel: Callable | None = None
        self._kernel_zero = None

    def _setup_radial_quadrature(self) -> None:
        """Shared: build y-quadrature and chord half-lengths for
        cylindrical or spherical meshes."""
        p = self.params
        radii = self.mesh.edges[1:]
        self._y_pts, self._y_wts = _composite_gauss_legendre(
            self.mesh.edges, p.n_quad_y,
        )
        self._chords = _chord_half_lengths(radii, self._y_pts)

    def _setup_cylindrical(self) -> None:
        """Cylindrical: Ki₄ kernel + y-quadrature."""
        p = self.params
        print("  Building Ki3/Ki4 lookup tables ...")
        ki_x, _, ki4_v = _build_ki_tables(p.n_ki_table, p.ki_max)

        self._setup_radial_quadrature()
        n_y = len(self._y_pts)
        self._kernel = lambda tau: _ki4_lookup(tau, ki_x, ki4_v)
        self._kernel_zero = _ki4_lookup(np.zeros(n_y), ki_x, ki4_v)

    def _setup_spherical(self) -> None:
        """Spherical: exp(-τ) kernel + y-weighted quadrature."""
        self._setup_radial_quadrature()
        # Kernel F(τ) = exp(-τ)
        self._kernel = lambda tau: np.exp(-tau)
        self._kernel_zero = np.ones(len(self._y_pts))
        # Spherical weight: extra factor of y in the quadrature
        self._y_wts = self._y_wts * self._y_pts

    # ── P_inf computation ─────────────────────────────────────────────

    def compute_pinf_group(self, sig_t_g: np.ndarray) -> np.ndarray:
        """Compute the infinite-lattice P_inf matrix for one energy group.

        Parameters
        ----------
        sig_t_g : ndarray, shape (N,)
            Total macroscopic cross section per cell for this group.

        Returns
        -------
        ndarray, shape (N, N)
            Infinite-lattice collision probability matrix.
        """
        match self.mesh.coord:
            case CoordSystem.CARTESIAN:
                rcp = self._compute_slab_rcp(sig_t_g)
            case _:
                rcp = self._compute_radial_rcp(sig_t_g)

        P_cell = self._normalize_rcp(rcp, sig_t_g)
        return self._apply_white_bc(P_cell, sig_t_g)

    def _compute_slab_rcp(self, sig_t_g: np.ndarray) -> np.ndarray:
        """Reduced collision probabilities for slab geometry (E₃ kernel).

        Returns rcp[i,j] = Σ_t(i) · V(i) · P_cell(i,j) (un-normalized).
        """
        N = self.mesh.N
        t = self.mesh.widths
        tau = sig_t_g * t

        # Optical boundary positions
        bnd_pos = np.concatenate(([0.0], np.cumsum(tau)))

        rcp = np.zeros((N, N))

        for i in range(N):
            sti = sig_t_g[i]
            tau_i = tau[i]
            if sti <= 0:
                continue

            # Self-same collision: Σ_t·V - (F(0) - F(τ_i))
            rcp[i, i] += sti * t[i] - (0.5 - _e3(tau_i))

            for j in range(N):
                tau_j = tau[j]

                # Direct path gap (optical distance between regions)
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

                # Reflected path (via centre symmetry plane)
                gap_c = bnd_pos[i] + bnd_pos[j]
                dc = (_e3(gap_c) - _e3(gap_c + tau_i)
                      - _e3(gap_c + tau_j) + _e3(gap_c + tau_i + tau_j))

                rcp[i, j] += 0.5 * (dd + dc)

        return rcp

    def _compute_radial_rcp(self, sig_t_g: np.ndarray) -> np.ndarray:
        """Reduced collision probabilities for radial geometries.

        Shared by cylindrical (Ki₄ kernel) and spherical (exp kernel).
        The kernel and y-quadrature weights are set during setup.

        Returns rcp[i,j] = Σ_t(i) · V(i) · P_cell(i,j) (un-normalized).
        """
        N = self.mesh.N
        chords = self._chords
        y_wts = self._y_wts
        kernel = self._kernel
        kernel_zero = self._kernel_zero
        n_y = len(y_wts)

        tau = sig_t_g[:, None] * chords  # (N, n_y)

        # Optical boundary positions at each y-line
        bnd_pos = np.zeros((N + 1, n_y))
        for k in range(N):
            bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]

        rcp = np.zeros((N, N))

        for i in range(N):
            tau_i = tau[i, :]
            sti = sig_t_g[i]
            if sti == 0:
                continue

            # Self-same collision
            self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
                kernel_zero - kernel(tau_i)
            )
            rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

            for j in range(N):
                tau_j = tau[j, :]

                # Direct path gap
                if j > i:
                    gap_d = bnd_pos[j, :] - bnd_pos[i + 1, :]
                elif j < i:
                    gap_d = bnd_pos[i, :] - bnd_pos[j + 1, :]
                else:
                    gap_d = None

                if gap_d is not None:
                    gap_d = np.maximum(gap_d, 0.0)
                    dd = (kernel(gap_d) - kernel(gap_d + tau_i)
                          - kernel(gap_d + tau_j)
                          + kernel(gap_d + tau_i + tau_j))
                else:
                    dd = np.zeros(n_y)

                # Reflected path
                gap_c = bnd_pos[i, :] + bnd_pos[j, :]
                dc = (kernel(gap_c) - kernel(gap_c + tau_i)
                      - kernel(gap_c + tau_j)
                      + kernel(gap_c + tau_i + tau_j))

                rcp[i, j] += 2.0 * np.dot(y_wts, dd + dc)

        return rcp

    def _normalize_rcp(
        self, rcp: np.ndarray, sig_t_g: np.ndarray,
    ) -> np.ndarray:
        """Normalize rcp to get P_cell: P_cell[i,:] = rcp[i,:] / (Σ_t·V)."""
        N = self.mesh.N
        V = self.mesh.volumes
        P_cell = np.zeros((N, N))
        for i in range(N):
            denom = sig_t_g[i] * V[i]
            if denom > 0:
                P_cell[i, :] = rcp[i, :] / denom
        return P_cell

    def _apply_white_bc(
        self, P_cell: np.ndarray, sig_t_g: np.ndarray,
    ) -> np.ndarray:
        """Apply white boundary condition to get P_inf.

        P_inf = P_cell + P_out ⊗ P_in / (1 - P_inout)

        Works for all coordinate systems because mesh.volumes and
        mesh.surfaces[-1] encode the geometry correctly.
        """
        V = self.mesh.volumes
        S_cell = self.mesh.surfaces[-1]

        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        P_in = sig_t_g * V * P_out / S_cell
        P_inout = max(1.0 - P_in.sum(), 0.0)

        P_inf = P_cell.copy()
        if P_inout < 1.0:
            P_inf += np.outer(P_out, P_in) / (1.0 - P_inout)

        return P_inf


# ═══════════════════════════════════════════════════════════════════════
# CP solver class (satisfies EigenvalueSolver protocol)
# ═══════════════════════════════════════════════════════════════════════

class CPSolver:
    """Geometry-independent CP eigenvalue solver.

    Once the infinite-lattice CP matrices P_inf are built (by
    :class:`CPMesh`), the eigenvalue iteration is identical for all
    geometries:

        φ_g = P_inf_g^T · (V · Q_g) / (Σ_t · V)

    where V is the cell volume array and Q is the total source
    (fission + scattering + n,2n).

    Two solver modes are supported:

    * **Jacobi** (default) — all groups updated from previous iteration's
      flux.  Simple, no inner iterations.
    * **Gauss-Seidel** — groups swept from fast to thermal; each group
      uses the latest flux from already-processed groups.  Inner
      iterations within each group converge the within-group scattering
      source.  Converges faster for thermal problems with strong
      self-scatter.
    """

    def __init__(
        self,
        P_inf: np.ndarray,
        xs: CellXS,
        volumes: np.ndarray,
        mat_ids: np.ndarray,
        materials: dict[int, Mixture],
        keff_tol: float = 1e-6,
        flux_tol: float = 1e-5,
        solver_mode: str = "jacobi",
        inner_tol: float = 1e-8,
        max_inner: int = 100,
    ) -> None:
        self.P_inf = P_inf        # (N, N, ng)
        self.xs = xs
        self.volumes = volumes    # (N,)
        self.mat_ids = mat_ids
        self.ng = xs.sig_t.shape[1]
        self.N = xs.sig_t.shape[0]
        self.keff_tol = keff_tol
        self.flux_tol = flux_tol
        self.solver_mode = solver_mode
        self.inner_tol = inner_tol
        self.max_inner = max_inner

        # Cache scattering and (n,2n) matrices per material
        self._scat_mats = {mid: materials[mid].SigS[0] for mid in materials}
        self._n2n_mats = {mid: materials[mid].Sig2 for mid in materials}

        # Convergence diagnostics (populated during iteration)
        self.residual_history: list[float] = []
        self.n_inner_history: list[np.ndarray] = []  # list of (ng,) arrays

    def initial_flux_distribution(self) -> np.ndarray:
        return np.ones((self.N, self.ng))

    def compute_fission_source(
        self, flux_distribution: np.ndarray, keff: float,
    ) -> np.ndarray:
        fission_rate = np.sum(self.xs.sig_p * flux_distribution, axis=1)
        return self.xs.chi * fission_rate[:, np.newaxis] / keff

    def solve_fixed_source(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        if self.solver_mode == "gauss_seidel":
            return self._solve_fixed_source_gs(fission_source, flux_distribution)
        return self._solve_fixed_source_jacobi(fission_source, flux_distribution)

    def _solve_fixed_source_jacobi(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Jacobi solver: all groups updated from previous iteration's flux."""
        N, ng = self.N, self.ng

        # Total source = fission + scattering + (n,2n)
        Q = fission_source.copy()
        for k in range(N):
            mid = self.mat_ids[k]
            Q[k, :] += self._scat_mats[mid].T @ flux_distribution[k, :]
            Q[k, :] += 2.0 * (self._n2n_mats[mid].T @ flux_distribution[k, :])

        # Apply CP matrices: φ_g = P_inf_g^T · (V · Q_g) / (Σ_t · V)
        phi = np.empty((N, ng))
        for g in range(ng):
            source = self.volumes * Q[:, g]
            phi[:, g] = self.P_inf[:, :, g].T @ source

        denom = self.xs.sig_t * self.volumes[:, np.newaxis]
        pos = denom > 0
        phi[pos] = phi[pos] / denom[pos]
        phi[~pos] = 0.0

        # Numerical conditioning (prevent overflow in subsequent iterations)
        phi *= 1.0 / np.max(phi)

        return phi

    def _solve_fixed_source_gs(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Gauss-Seidel solver with inner iterations per group.

        Sweeps groups from fast (g=0) to thermal (g=ng-1).  For each
        group, inner iterations converge the within-group scattering
        source by recomputing the scattering from the latest flux
        (which includes already-updated groups < g).

        The inner iteration count per group reveals which groups have
        strong self-scatter (typically thermal groups).
        """
        N, ng = self.N, self.ng
        phi = flux_distribution.copy()  # start from previous estimate
        n_inner_this_outer = np.zeros(ng, dtype=int)

        for g in range(ng):
            for n_in in range(1, self.max_inner + 1):
                phi_g_old = phi[:, g].copy()

                # Build total source for group g using latest phi
                Q_g = fission_source[:, g].copy()
                for k in range(N):
                    mid = self.mat_ids[k]
                    # Scattering into group g from all groups (using latest phi)
                    scat_row = np.asarray(
                        self._scat_mats[mid].T[g, :].todense()
                    ).ravel()
                    Q_g[k] += scat_row @ phi[k, :]
                    # (n,2n) into group g
                    n2n_row = np.asarray(
                        self._n2n_mats[mid].T[g, :].todense()
                    ).ravel()
                    Q_g[k] += 2.0 * (n2n_row @ phi[k, :])

                # Apply CP matrix for group g
                source_g = self.volumes * Q_g
                transported_g = self.P_inf[:, :, g].T @ source_g

                # New flux for group g
                denom_g = self.xs.sig_t[:, g] * self.volumes
                phi_g_new = np.zeros(N)
                pos = denom_g > 0
                phi_g_new[pos] = transported_g[pos] / denom_g[pos]

                phi[:, g] = phi_g_new

                # Inner convergence: relative change in group flux
                # Nonzero when within-group self-scatter Σ_s(g→g) causes
                # the source Q_g to depend on φ_g itself.
                phi_g_norm = np.linalg.norm(phi_g_new)
                res_in = (np.linalg.norm(phi_g_new - phi_g_old)
                          / max(phi_g_norm, 1e-30))

                if res_in < self.inner_tol:
                    break

            n_inner_this_outer[g] = n_in

        self.n_inner_history.append(n_inner_this_outer.copy())

        # Numerical conditioning
        phi *= 1.0 / np.max(phi)
        return phi

    def _compute_balance_residual(
        self, flux_distribution: np.ndarray, keff: float,
    ) -> float:
        """Compute the neutron balance residual ||Aφ - B1φ - (1/k)B2φ||₂.

        This is the L2 norm of the difference between the collision rate
        (Σ_t V φ) and the transported total source (P^T V Q_total) across
        all regions and groups.  At convergence, this should be zero.
        """
        N, ng = self.N, self.ng
        V = self.volumes

        # Collision rate: Σ_t(i,g) · V(i) · φ(i,g)
        collision = self.xs.sig_t * V[:, np.newaxis] * flux_distribution

        # Build total source Q = fission + scattering + (n,2n)
        fission_rate = np.sum(self.xs.sig_p * flux_distribution, axis=1)
        Q_fis = self.xs.chi * fission_rate[:, np.newaxis] / keff
        Q = Q_fis.copy()
        for k in range(N):
            mid = self.mat_ids[k]
            Q[k, :] += self._scat_mats[mid].T @ flux_distribution[k, :]
            Q[k, :] += 2.0 * (self._n2n_mats[mid].T @ flux_distribution[k, :])

        # Transported source: P^T · (V · Q) for each group
        transported = np.empty((N, ng))
        for g in range(ng):
            transported[:, g] = self.P_inf[:, :, g].T @ (V * Q[:, g])

        return float(np.linalg.norm(collision - transported))

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        v = self.volumes[:, np.newaxis]
        production = np.sum(self.xs.sig_p * flux_distribution * v)

        # Net removal = total - scattering - 2*(n,2n)
        # This is the denominator of the eigenvalue balance:
        #   (Σt - Σs - 2Σ₂)φV = (1/k) χ(νΣf·φ)V
        total = np.sum(self.xs.sig_t * flux_distribution * v)
        scatter = 0.0
        n2n = 0.0
        for k in range(self.N):
            mid = self.mat_ids[k]
            scatter += np.sum(
                (self._scat_mats[mid].T @ flux_distribution[k, :])
                * self.volumes[k]
            )
            n2n += np.sum(
                (self._n2n_mats[mid].T @ flux_distribution[k, :])
                * self.volumes[k]
            )
        net_removal = total - scatter - 2.0 * n2n

        return float(production / net_removal)

    def converged(
        self, keff: float, keff_old: float,
        flux_distribution: np.ndarray, flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        if iteration <= 2:
            return False

        delta_k = abs(keff - keff_old)
        delta_phi = np.max(np.abs(flux_distribution - flux_old)) / \
            max(np.max(np.abs(flux_distribution)), 1e-30)

        # Compute and store the neutron balance residual
        residual = self._compute_balance_residual(flux_distribution, keff)
        self.residual_history.append(residual)

        # Print diagnostics
        if iteration <= 5 or iteration % 10 == 0:
            msg = (f"    iter {iteration:4d}  keff = {keff:.6f}  "
                   f"dk = {delta_k:.2e}  dphi = {delta_phi:.2e}  "
                   f"res = {residual:.2e}")
            if (self.solver_mode == "gauss_seidel"
                    and self.n_inner_history):
                n_in = self.n_inner_history[-1]
                msg += f"  inner(max/mean) = {n_in.max()}/{n_in.mean():.1f}"
            print(msg)

        if delta_k < self.keff_tol and delta_phi < self.flux_tol:
            print(f"    iter {iteration:4d}  keff = {keff:.6f}  "
                  f"res = {residual:.2e}  Converged.")
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════════════

def _volume_averaged_fluxes(
    phi: np.ndarray,
    volumes: np.ndarray,
    mat_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume-averaged flux per material (fuel=2, clad=1, cool=0)."""
    ng = phi.shape[1]
    vol_fuel = volumes[mat_ids == 2].sum()
    vol_clad = volumes[mat_ids == 1].sum()
    vol_cool = volumes[mat_ids == 0].sum()

    flux_fuel = np.zeros(ng)
    flux_clad = np.zeros(ng)
    flux_cool = np.zeros(ng)

    for k in range(len(mat_ids)):
        v = volumes[k]
        mid = mat_ids[k]
        if mid == 2:
            flux_fuel += phi[k, :] * v / vol_fuel
        elif mid == 1:
            flux_clad += phi[k, :] * v / vol_clad
        else:
            flux_cool += phi[k, :] * v / vol_cool

    return flux_fuel, flux_clad, flux_cool


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def solve_cp(
    materials: dict[int, Mixture],
    mesh: Mesh1D | None = None,
    params: CPParams | None = None,
) -> CPResult:
    """Solve the CP eigenvalue problem for any supported geometry.

    The kernel is selected automatically based on ``mesh.coord``:
    Cartesian (slab E₃), Cylindrical (Ki₄), or Spherical (exp).

    Parameters
    ----------
    materials : dict[int, Mixture]
        Macroscopic cross sections keyed by material ID.
    mesh : Mesh1D, optional
        1-D mesh.  Defaults to a cylindrical PWR pin cell via
        :func:`geometry.factories.pwr_pin_equivalent`.
    params : CPParams, optional
        Solver parameters (tolerances, Ki table size, quadrature order).

    Returns
    -------
    CPResult
    """
    t_start = time.perf_counter()

    if mesh is None:
        from orpheus.geometry import pwr_pin_equivalent
        mesh = pwr_pin_equivalent()
    if params is None:
        params = CPParams()

    _any_mat = next(iter(materials.values()))
    eg = _any_mat.eg
    ng = _any_mat.ng
    N = mesh.N

    # Build augmented geometry (sets up kernel + quadrature)
    cp_mesh = CPMesh(mesh, params)

    xs = assemble_cell_xs(materials, mesh.mat_ids)

    # Build CP matrices for all energy groups
    print(f"  Computing CP matrices for {ng} groups, {N} regions "
          f"({mesh.coord.value}) ...")
    P_inf = np.empty((N, N, ng))
    for g in range(ng):
        P_inf[:, :, g] = cp_mesh.compute_pinf_group(xs.sig_t[:, g])
        if (g + 1) % 100 == 0:
            print(f"    group {g + 1}/{ng}")
    print("  CP matrices done.")

    # Eigenvalue solve
    print(f"  Starting power iteration (mode: {params.solver_mode}) ...")
    solver = CPSolver(
        P_inf, xs, mesh.volumes, mesh.mat_ids, materials,
        keff_tol=params.keff_tol, flux_tol=params.flux_tol,
        solver_mode=params.solver_mode,
        inner_tol=params.inner_tol, max_inner=params.max_inner,
    )
    keff, keff_history, phi = power_iteration(solver, max_iter=params.max_outer)

    flux_fuel, flux_clad, flux_cool = _volume_averaged_fluxes(
        phi, mesh.volumes, mesh.mat_ids)

    # Collect inner iteration diagnostics (GS mode only)
    n_inner = None
    if solver.n_inner_history:
        n_inner = np.array(solver.n_inner_history)  # (n_outer, ng)

    elapsed = time.perf_counter() - t_start
    print(f"  Elapsed: {elapsed:.1f}s")

    return CPResult(
        keff=keff, keff_history=keff_history, flux=phi,
        flux_fuel=flux_fuel, flux_clad=flux_clad, flux_cool=flux_cool,
        geometry=mesh, eg=eg, elapsed_seconds=elapsed,
        residual_history=solver.residual_history,
        n_inner=n_inner,
    )