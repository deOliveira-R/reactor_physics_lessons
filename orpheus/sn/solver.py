"""Unified SN (Discrete Ordinates) eigenvalue solver.

Supports 1D (ny=1, GL quadrature) and 2D (Lebedev quadrature) with
selectable inner solver strategy and scattering anisotropy order.

The transport sweep uses diamond-difference spatial discretization:
- 1D: cumulative-product recurrence (~ms)
- 2D: wavefront parallelism along anti-diagonals

Boundary conditions are reflective on all sides (infinite lattice).

.. seealso:: :ref:`theory-discrete-ordinates` — Key Facts, equations, gotchas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from orpheus.data.macro_xs.cell_xs import assemble_cell_xs
from orpheus.data.macro_xs.mixture import Mixture
from orpheus.geometry import Mesh1D, Mesh2D
from orpheus.numerics.eigenvalue import power_iteration
from .geometry import SNMesh
from .quadrature import AngularQuadrature
from .sweep import transport_sweep


@dataclass
class SNResult:
    """Results of an SN transport calculation.

    The primary output is the angular flux (the direct solution of the
    SN equations). Scalar flux is derived by quadrature integration.
    """

    keff: float
    keff_history: list[float]
    angular_flux: np.ndarray   # (N_ordinates, nx, ny, ng)
    scalar_flux: np.ndarray    # (nx, ny, ng) = Σ w_n ψ_n
    geometry: Mesh1D | Mesh2D
    quadrature: AngularQuadrature
    eg: np.ndarray             # (ng+1,) energy group boundaries
    elapsed_seconds: float


# ═══════════════════════════════════════════════════════════════════════
# Solver class (EigenvalueSolver protocol)
# ═══════════════════════════════════════════════════════════════════════

class SNSolver:
    """Unified SN eigenvalue solver satisfying the EigenvalueSolver protocol.

    Parameters
    ----------
    materials : dict mapping material ID to Mixture.
    sn_mesh : SNMesh — augmented geometry (wraps Mesh1D or Mesh2D with
        precomputed streaming stencil).
    inner_solver : "source_iteration" or "bicgstab".
    scattering_order : int — Legendre order for scattering (0 = P0).
    keff_tol, flux_tol : outer iteration convergence.
    max_inner, inner_tol : inner iteration parameters.
    """

    def __init__(
        self,
        materials: dict[int, Mixture],
        sn_mesh: SNMesh,
        inner_solver: str = "source_iteration",
        scattering_order: int = 0,
        keff_tol: float = 1e-7,
        flux_tol: float = 1e-6,
        max_inner: int = 200,
        inner_tol: float = 1e-8,
    ):
        self.sn_mesh = sn_mesh
        self.quad = sn_mesh.quad
        self.inner_solver = inner_solver
        self.scattering_order = scattering_order
        self.keff_tol = keff_tol
        self.flux_tol = flux_tol
        self.max_inner = max_inner
        self.inner_tol = inner_tol

        nx, ny = sn_mesh.nx, sn_mesh.ny
        _any_mat = next(iter(materials.values()))
        self.ng = _any_mat.ng

        # Per-cell cross sections
        xs = assemble_cell_xs(materials, sn_mesh.mat_map)
        self.sig_t = xs.sig_t.reshape(nx, ny, self.ng)
        self.sig_a = xs.sig_a.reshape(nx, ny, self.ng)
        self.sig_p = xs.sig_p.reshape(nx, ny, self.ng)
        self.chi = xs.chi.reshape(nx, ny, self.ng)

        # Scattering matrices per material — all available Legendre orders
        L = min(scattering_order, min(len(m.SigS) - 1 for m in materials.values()))
        self.scattering_order = L
        self.sig_s: dict[int, list[np.ndarray]] = {}
        self.sig2: dict[int, np.ndarray] = {}
        for mat_id, mix in materials.items():
            self.sig_s[mat_id] = [
                np.array(mix.SigS[l].todense()) for l in range(L + 1)
            ]
            self.sig2[mat_id] = np.array(mix.Sig2.todense())

        # Backward-compatible alias: sig_s0[mid] = sig_s[mid][0]
        self.sig_s0: dict[int, np.ndarray] = {
            mid: mats[0] for mid, mats in self.sig_s.items()
        }

        # Precompute spherical harmonics if anisotropic scattering requested
        if L > 0:
            self._Y = sn_mesh.quad.spherical_harmonics(L)  # (N, L+1, 2L+1)
        else:
            self._Y = None

        # Pre-group cells by material for vectorized source computation
        self._cells_by_mat: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for mat_id in materials:
            ix, iy = np.where(sn_mesh.mat_map == mat_id)
            self._cells_by_mat[mat_id] = (ix, iy)

        # Pre-computed sig2 row sums per material (for keff)
        self._sig2_sum: dict[int, np.ndarray] = {}
        for mat_id in materials:
            self._sig2_sum[mat_id] = np.asarray(
                self.sig2[mat_id].sum(axis=1)
            ).ravel()

        # Weight normalization (1/sum(w) — works for both GL and Lebedev)
        self.weight_norm = 1.0 / sn_mesh.quad.weights.sum()

        # Persistent boundary flux cache (passed to sweep)
        self._psi_bc: dict = {}

        # Volume array for keff computation
        self.volume = sn_mesh.volumes

    def initial_flux_distribution(self) -> np.ndarray:
        """Initial scalar flux guess: ones(nx, ny, ng)."""
        return np.ones((self.sn_mesh.nx, self.sn_mesh.ny, self.ng))

    def compute_fission_source(
        self, flux_distribution: np.ndarray, keff: float,
    ) -> np.ndarray:
        """Fission source: χ · (νΣ_f · φ) / k."""
        fission_rate = np.sum(self.sig_p * flux_distribution, axis=2)  # (nx, ny)
        return self.chi * fission_rate[:, :, None] / keff

    def solve_fixed_source(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Solve the within-group transport equation for given fission source.

        Returns updated scalar flux (nx, ny, ng).
        """
        if self.inner_solver == "source_iteration":
            return self._solve_source_iteration(fission_source, flux_distribution)
        elif self.inner_solver == "bicgstab":
            return self._solve_bicgstab(fission_source, flux_distribution)
        else:
            raise ValueError(f"Unknown inner solver: {self.inner_solver}")

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        """k = production / absorption (volume-weighted)."""
        vol = self.volume[:, :, None]
        production = np.sum(self.sig_p * flux_distribution * vol)
        # Add (n,2n) contribution — vectorized by material
        for mid, (ix, iy) in self._cells_by_mat.items():
            n2n = flux_distribution[ix, iy, :] @ self._sig2_sum[mid]
            production += 2.0 * np.dot(n2n, self.volume[ix, iy])
        absorption = np.sum(self.sig_a * flux_distribution * vol)
        return float(production / absorption)

    def converged(
        self, keff: float, keff_old: float,
        flux_distribution: np.ndarray, flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        if iteration <= 2:
            return False
        dk = abs(keff - keff_old)
        dphi = np.linalg.norm(flux_distribution - flux_old) / \
            max(np.linalg.norm(flux_distribution), 1e-30)
        return dk < self.keff_tol and dphi < self.flux_tol

    # ── Inner solver: source iteration ────────────────────────────────

    def _solve_source_iteration(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Scattering source iteration: sweep → update scatter → sweep → ..."""
        phi = flux_distribution.copy()
        angular = None  # no angular flux on first iteration

        for n_inner in range(self.max_inner):
            phi_prev = phi.copy()

            # Isotropic source = fission + P0 scattering + (n,2n)
            Q = fission_source.copy()
            self._add_scattering_source(Q, phi)
            self._add_n2n_source(Q, phi)

            # Anisotropic scattering (P1+ terms, None when L=0)
            Q_aniso = self._build_aniso_scattering(angular)

            # Transport sweep
            angular, phi = transport_sweep(
                Q, self.sig_t, self.sn_mesh,
                self._psi_bc, Q_aniso=Q_aniso,
            )

            norm = np.linalg.norm(phi)
            if norm > 0:
                res = np.linalg.norm(phi - phi_prev) / norm
                if res < self.inner_tol:
                    break

        return phi

    # ── Inner solver: BiCGSTAB ────────────────────────────────────────

    def _solve_bicgstab(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Direct Krylov solve of the angular transport equation.

        Solves  T·ψ = b  via BiCGSTAB where T = μ·∇ + Σ_t is the
        streaming + collision operator (formed explicitly via finite
        differences) and b = fission + scattering + (n,2n) sources.

        Returns the updated scalar flux (nx, ny, ng).
        """
        from scipy.sparse.linalg import bicgstab

        nx, ny, ng = self.sn_mesh.nx, self.sn_mesh.ny, self.ng
        sum_w = float(self.quad.weights.sum())
        phi = flux_distribution
        fission_src_norm = fission_source / sum_w

        if self.sn_mesh.curvature == "spherical":
            return self._solve_bicgstab_spherical(fission_src_norm, phi)
        elif self.sn_mesh.curvature == "cylindrical":
            return self._solve_bicgstab_cylindrical(fission_src_norm, phi)
        else:
            return self._solve_bicgstab_cartesian(fission_src_norm, phi)

    def _solve_bicgstab_cartesian(
        self, fission_src_norm: np.ndarray, phi: np.ndarray,
    ) -> np.ndarray:
        """BiCGSTAB for Cartesian geometry."""
        from scipy.sparse.linalg import bicgstab
        from .operator import (
            build_equation_map,
            build_transport_linear_operator,
            build_rhs,
            solution_to_angular_flux,
            angular_flux_to_scalar,
        )

        nx, ny, ng = self.sn_mesh.nx, self.sn_mesh.ny, self.ng

        if not hasattr(self, '_eq_map'):
            self._eq_map = build_equation_map(nx, ny, self.quad, ng)
            self._T_op = build_transport_linear_operator(
                self._eq_map, self.quad, self.sig_t,
                nx, ny, ng, self.sn_mesh.dx, self.sn_mesh.dy,
            )

        eq_map = self._eq_map
        T_op = self._T_op

        angular = None
        if self.scattering_order > 0 and hasattr(self, '_psi_solution'):
            angular = solution_to_angular_flux(
                self._psi_solution, eq_map, self.quad, nx, ny, ng,
            )

        rhs = build_rhs(
            fission_src_norm, phi, eq_map, self.quad,
            self.sig_s, self.sig2, self.sn_mesh.mat_map,
            nx, ny, ng,
            scattering_order=self.scattering_order,
            angular_flux=angular,
        )

        if hasattr(self, '_psi_solution'):
            x0 = self._psi_solution
        else:
            x0 = np.ones(eq_map.n_unknowns)

        solution, info = bicgstab(
            T_op, rhs, x0=x0,
            rtol=self.inner_tol, maxiter=self.max_inner,
        )
        self._psi_solution = solution

        fi = solution_to_angular_flux(solution, eq_map, self.quad, nx, ny, ng)
        return angular_flux_to_scalar(fi, self.quad, nx, ny, ng)

    def _solve_bicgstab_spherical(
        self, fission_src_norm: np.ndarray, phi: np.ndarray,
    ) -> np.ndarray:
        """BiCGSTAB for spherical 1D geometry."""
        from scipy.sparse.linalg import bicgstab
        from .operator import (
            build_equation_map_spherical,
            build_transport_linear_operator_spherical,
            build_rhs_spherical,
            solution_to_angular_flux_spherical,
            angular_flux_to_scalar,
        )

        nx, ng = self.sn_mesh.nx, self.ng

        if not hasattr(self, '_eq_map'):
            self._eq_map = build_equation_map_spherical(nx, self.quad, ng)
            self._T_op = build_transport_linear_operator_spherical(
                self._eq_map, self.quad, self.sig_t,
                nx, ng,
                self.sn_mesh.face_areas,
                self.sn_mesh.volumes,
                self.sn_mesh.alpha_half,
                self.sn_mesh.redist_dAw,
                self.sn_mesh.tau_mm,
            )

        eq_map = self._eq_map
        T_op = self._T_op

        rhs = build_rhs_spherical(
            fission_src_norm, phi, eq_map, self.quad,
            self.sig_s, self.sig2, self.sn_mesh.mat_map,
            nx, ng,
        )

        if hasattr(self, '_psi_solution'):
            x0 = self._psi_solution
        else:
            x0 = np.ones(eq_map.n_unknowns)

        solution, info = bicgstab(
            T_op, rhs, x0=x0,
            rtol=self.inner_tol, maxiter=self.max_inner,
        )
        self._psi_solution = solution

        fi = solution_to_angular_flux_spherical(
            solution, eq_map, self.quad, nx, ng,
        )
        return angular_flux_to_scalar(fi, self.quad, nx, 1, ng)

    def _solve_bicgstab_cylindrical(
        self, fission_src_norm: np.ndarray, phi: np.ndarray,
    ) -> np.ndarray:
        """BiCGSTAB for cylindrical 1D geometry."""
        from scipy.sparse.linalg import bicgstab
        from .operator import (
            build_equation_map_cylindrical,
            build_transport_linear_operator_cylindrical,
            build_rhs_cylindrical,
            solution_to_angular_flux_cylindrical,
            angular_flux_to_scalar,
        )

        nx, ng = self.sn_mesh.nx, self.ng

        if not hasattr(self, '_eq_map'):
            self._eq_map = build_equation_map_cylindrical(nx, self.quad, ng)
            self._T_op = build_transport_linear_operator_cylindrical(
                self._eq_map, self.quad, self.sig_t,
                nx, ng,
                self.sn_mesh.face_areas,
                self.sn_mesh.volumes,
                self.sn_mesh.alpha_per_level,
                self.sn_mesh.redist_dAw_per_level,
                self.sn_mesh.tau_mm_per_level,
            )

        eq_map = self._eq_map
        T_op = self._T_op

        rhs = build_rhs_cylindrical(
            fission_src_norm, phi, eq_map, self.quad,
            self.sig_s, self.sig2, self.sn_mesh.mat_map,
            nx, ng,
        )

        if hasattr(self, '_psi_solution'):
            x0 = self._psi_solution
        else:
            x0 = np.ones(eq_map.n_unknowns)

        solution, info = bicgstab(
            T_op, rhs, x0=x0,
            rtol=self.inner_tol, maxiter=self.max_inner,
        )
        self._psi_solution = solution

        fi = solution_to_angular_flux_cylindrical(
            solution, eq_map, self.quad, nx, ng,
        )
        return angular_flux_to_scalar(fi, self.quad, nx, 1, ng)

    # ── Source computation helpers ────────────────────────────────────

    def _add_scattering_source(self, Q: np.ndarray, phi: np.ndarray) -> None:
        """Add P0 scattering source to Q in-place (vectorized by material)."""
        for mid, (ix, iy) in self._cells_by_mat.items():
            Q[ix, iy, :] += phi[ix, iy, :] @ self.sig_s0[mid]

    def _build_aniso_scattering(
        self, angular_flux: np.ndarray | None,
    ) -> np.ndarray | None:
        """Build per-ordinate anisotropic scattering source (P1+ terms).

        Returns (N, nx, ny, ng) or None if scattering_order == 0 or
        no angular flux is available yet.
        """
        if self.scattering_order == 0 or angular_flux is None:
            return None

        N = self.quad.N
        nx, ny = self.sn_mesh.nx, self.sn_mesh.ny
        ng = self.ng
        L = self.scattering_order
        Y = self._Y  # (N, L+1, 2L+1)
        w = self.quad.weights

        # Compute Legendre moments: fiL[x, y, g, l, l+m] = Σ_n w_n ψ_n Y_l^m(n)
        fiL = np.zeros((nx, ny, ng, L + 1, 2 * L + 1))
        for l in range(L + 1):
            for m in range(-l, l + 1):
                # (N,) * (N, nx, ny, ng) summed over N → (nx, ny, ng)
                fiL[:, :, :, l, l + m] = np.einsum(
                    'n,nxyg->xyg', w * Y[:, l, l + m], angular_flux,
                )

        # Build anisotropic source: only l >= 1 terms (P0 is in Q_iso)
        Q_aniso = np.zeros((N, nx, ny, ng))
        for mid, (ix, iy) in self._cells_by_mat.items():
            n_cells = len(ix)
            sig_s_l = self.sig_s[mid]
            for l in range(1, L + 1):  # skip l=0 (handled by _add_scattering_source)
                # Σ_m fiL[..., l, m] * Y_l^m(n) → reconstruct angular moment at ordinate n
                for m in range(-l, l + 1):
                    moment = fiL[ix, iy, :, l, l + m]  # (n_cells, ng)
                    # (n_cells, ng) @ (ng, ng) → (n_cells, ng)
                    scattered = moment @ sig_s_l[l]  # Σ_s^l @ fiL_lm
                    for n in range(N):
                        Q_aniso[n, ix, iy, :] += (2 * l + 1) * Y[n, l, l + m] * scattered

        return Q_aniso

    def _add_n2n_source(self, Q: np.ndarray, phi: np.ndarray) -> None:
        """Add (n,2n) source to Q in-place (vectorized by material)."""
        for mid, (ix, iy) in self._cells_by_mat.items():
            Q[ix, iy, :] += 2.0 * (phi[ix, iy, :] @ self.sig2[mid])


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def solve_sn(
    materials: dict[int, Mixture],
    mesh: Mesh1D | Mesh2D,
    quadrature: AngularQuadrature,
    inner_solver: str = "source_iteration",
    scattering_order: int = 0,
    max_outer: int = 500,
    keff_tol: float = 1e-7,
    flux_tol: float = 1e-6,
    max_inner: int = 200,
    inner_tol: float = 1e-8,
) -> SNResult:
    """Solve the multi-group SN eigenvalue problem.

    Parameters
    ----------
    materials : dict mapping material ID to Mixture.
    mesh : Mesh1D or Mesh2D (base geometry).
    quadrature : AngularQuadrature (GaussLegendre1D or LebedevSphere).
    inner_solver : "source_iteration" (default) or "bicgstab".
    scattering_order : Legendre order for scattering (0 = P0).
    max_outer : maximum outer (power) iterations.
    keff_tol, flux_tol : outer convergence.
    max_inner, inner_tol : inner solver parameters.
    """
    t_start = time.perf_counter()

    # Build augmented geometry (precomputes streaming stencil)
    sn_mesh = SNMesh(mesh, quadrature)

    solver = SNSolver(
        materials, sn_mesh,
        inner_solver=inner_solver,
        scattering_order=scattering_order,
        keff_tol=keff_tol, flux_tol=flux_tol,
        max_inner=max_inner, inner_tol=inner_tol,
    )

    keff, keff_history, scalar_flux = power_iteration(solver, max_iter=max_outer)

    # Final sweep to get angular flux
    Q_final = solver.compute_fission_source(scalar_flux, keff)
    solver._add_scattering_source(Q_final, scalar_flux)
    solver._add_n2n_source(Q_final, scalar_flux)
    angular_flux, _ = transport_sweep(
        Q_final, solver.sig_t, sn_mesh, solver._psi_bc,
    )

    _any_mat = next(iter(materials.values()))
    elapsed = time.perf_counter() - t_start

    return SNResult(
        keff=keff_history[-1],
        keff_history=keff_history,
        angular_flux=angular_flux,
        scalar_flux=scalar_flux,
        geometry=mesh,
        quadrature=quadrature,
        eg=_any_mat.eg,
        elapsed_seconds=elapsed,
    )
