"""Unified SN (Discrete Ordinates) eigenvalue solver.

Supports 1D (ny=1, GL quadrature) and 2D (Lebedev quadrature) with
selectable inner solver strategy and scattering anisotropy order.

The transport sweep uses diamond-difference spatial discretization:
- 1D: cumulative-product recurrence (~ms)
- 2D: wavefront parallelism along anti-diagonals

Boundary conditions are reflective on all sides (infinite lattice).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from data.macro_xs.cell_xs import assemble_cell_xs
from data.macro_xs.mixture import Mixture
from numerics.eigenvalue import power_iteration
from sn_geometry import CartesianMesh
from sn_quadrature import AngularQuadrature
from sn_sweep import transport_sweep


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
    geometry: CartesianMesh
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
    mesh : CartesianMesh (ny=1 for 1D, ny>1 for 2D).
    quadrature : AngularQuadrature (GaussLegendre1D or LebedevSphere).
    inner_solver : "source_iteration" or "bicgstab".
    scattering_order : int — Legendre order for scattering (0 = P0).
    keff_tol, flux_tol : outer iteration convergence.
    max_inner, inner_tol : inner iteration parameters.
    """

    def __init__(
        self,
        materials: dict[int, Mixture],
        mesh: CartesianMesh,
        quadrature: AngularQuadrature,
        inner_solver: str = "source_iteration",
        scattering_order: int = 0,
        keff_tol: float = 1e-7,
        flux_tol: float = 1e-6,
        max_inner: int = 200,
        inner_tol: float = 1e-8,
    ):
        self.mesh = mesh
        self.quad = quadrature
        self.inner_solver = inner_solver
        self.scattering_order = scattering_order
        self.keff_tol = keff_tol
        self.flux_tol = flux_tol
        self.max_inner = max_inner
        self.inner_tol = inner_tol

        nx, ny = mesh.nx, mesh.ny
        _any_mat = next(iter(materials.values()))
        self.ng = _any_mat.ng

        # Per-cell cross sections
        xs = assemble_cell_xs(materials, mesh.mat_map)
        self.sig_t = xs.sig_t.reshape(nx, ny, self.ng)
        self.sig_a = xs.sig_a.reshape(nx, ny, self.ng)
        self.sig_p = xs.sig_p.reshape(nx, ny, self.ng)
        self.chi = xs.chi.reshape(nx, ny, self.ng)

        # Scattering matrices per material (dense, P0 for now)
        self.sig_s0: dict[int, np.ndarray] = {}
        self.sig2: dict[int, np.ndarray] = {}
        for mat_id, mix in materials.items():
            self.sig_s0[mat_id] = np.array(mix.SigS[0].todense())
            self.sig2[mat_id] = np.array(mix.Sig2.todense())

        # Weight normalization (1/sum(w) — works for both GL and Lebedev)
        self.weight_norm = 1.0 / quadrature.weights.sum()

        # Persistent boundary flux cache (passed to sweep)
        self._psi_bc: dict = {}

        # Volume array for keff computation
        self.volume = mesh.volume

    def initial_flux_distribution(self) -> np.ndarray:
        """Initial scalar flux guess: ones(nx, ny, ng)."""
        return np.ones((self.mesh.nx, self.mesh.ny, self.ng))

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
        # Add (n,2n) contribution to production
        for ix in range(self.mesh.nx):
            for iy in range(self.mesh.ny):
                mid = int(self.mesh.mat_map[ix, iy])
                sig2_sum = np.array(self.sig2[mid].sum(axis=1)).ravel()
                production += 2.0 * np.dot(sig2_sum, flux_distribution[ix, iy, :]) * self.volume[ix, iy]
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

        for n_inner in range(self.max_inner):
            phi_prev = phi.copy()

            # Total source = fission + scattering + (n,2n)
            Q = fission_source.copy()
            self._add_scattering_source(Q, phi)
            self._add_n2n_source(Q, phi)

            # Transport sweep
            _, phi = transport_sweep(
                Q, self.sig_t, self.mesh.dx, self.mesh.dy,
                self.quad, self._psi_bc,
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
        """BiCGSTAB solve of (T - S)φ = F.

        Uses the transport sweep as the matvec for the streaming+collision
        operator, with scattering on the RHS.
        """
        from scipy.sparse.linalg import LinearOperator, bicgstab

        phi = flux_distribution.copy()
        shape_flat = phi.size

        # RHS: fission + scattering + (n,2n) from current phi
        rhs = fission_source.copy()
        self._add_scattering_source(rhs, phi)
        self._add_n2n_source(rhs, phi)

        def matvec(x_flat):
            """Apply transport operator: T·φ (streaming + collision)."""
            x = x_flat.reshape(phi.shape)
            # Sweep with x as source, return Σ_t·φ_sweep (the LHS of the DD eq)
            _, phi_sweep = transport_sweep(
                x, self.sig_t, self.mesh.dx, self.mesh.dy,
                self.quad, self._psi_bc,
            )
            return phi_sweep.ravel()

        A_op = LinearOperator((shape_flat, shape_flat), matvec=matvec)
        rhs_flat = rhs.ravel()

        solution, info = bicgstab(
            A_op, rhs_flat, x0=phi.ravel(),
            rtol=self.inner_tol, maxiter=self.max_inner,
        )

        return solution.reshape(phi.shape)

    # ── Source computation helpers ────────────────────────────────────

    def _add_scattering_source(self, Q: np.ndarray, phi: np.ndarray) -> None:
        """Add P0 scattering source to Q in-place."""
        nx, ny = self.mesh.nx, self.mesh.ny
        for ix in range(nx):
            for iy in range(ny):
                mid = int(self.mesh.mat_map[ix, iy])
                Q[ix, iy, :] += self.sig_s0[mid].T @ phi[ix, iy, :]

    def _add_n2n_source(self, Q: np.ndarray, phi: np.ndarray) -> None:
        """Add (n,2n) source to Q in-place."""
        nx, ny = self.mesh.nx, self.mesh.ny
        for ix in range(nx):
            for iy in range(ny):
                mid = int(self.mesh.mat_map[ix, iy])
                Q[ix, iy, :] += 2.0 * (self.sig2[mid].T @ phi[ix, iy, :])


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def solve_sn(
    materials: dict[int, Mixture],
    mesh: CartesianMesh,
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
    mesh : CartesianMesh (ny=1 for 1D, ny>1 for 2D).
    quadrature : AngularQuadrature (GaussLegendre1D or LebedevSphere).
    inner_solver : "source_iteration" (default) or "bicgstab".
    scattering_order : Legendre order for scattering (0 = P0).
    max_outer : maximum outer (power) iterations.
    keff_tol, flux_tol : outer convergence.
    max_inner, inner_tol : inner solver parameters.
    """
    t_start = time.perf_counter()

    solver = SNSolver(
        materials, mesh, quadrature,
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
        Q_final, solver.sig_t, mesh.dx, mesh.dy,
        quadrature, solver._psi_bc,
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
