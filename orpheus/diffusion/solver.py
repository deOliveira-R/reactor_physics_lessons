"""1D two-group neutron diffusion solver for a PWR subassembly.

Solves the 1D neutron diffusion equation with vacuum boundary conditions
using a finite-difference discretization and BiCGSTAB inner iterations.

Port of MATLAB ``CORE1D.m`` + ``funCORE1D.m``.

.. seealso:: :ref:`theory-verification` — diffusion verification (buckling eigenvalue, Richardson).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab

from orpheus.numerics.eigenvalue import power_iteration


@dataclass
class CoreGeometry:
    """1D geometry for a PWR subassembly."""

    bot_refl_height: float = 50.0   # cm
    fuel_height: float = 300.0      # cm
    top_refl_height: float = 50.0   # cm
    f2f: float = 21.61              # flat-to-flat distance (cm)
    dz: float = 5.0                 # axial cell height (cm)

    @property
    def n_refl_bot(self) -> int:
        return int(self.bot_refl_height / self.dz)

    @property
    def n_fuel(self) -> int:
        return int(self.fuel_height / self.dz)

    @property
    def n_refl_top(self) -> int:
        return int(self.top_refl_height / self.dz)

    @property
    def n_cells(self) -> int:
        return self.n_refl_bot + self.n_fuel + self.n_refl_top

    @property
    def n_faces(self) -> int:
        return self.n_cells + 1

    @property
    def az(self) -> float:
        """Cross-sectional area of one subassembly (cm^2)."""
        return self.f2f**2

    @property
    def dv(self) -> float:
        """Volume of one cell (cm^3)."""
        return self.az * self.dz


@dataclass
class TwoGroupXS:
    """Two-group macroscopic cross sections for a material."""

    transport: np.ndarray    # (2,) transport XS
    absorption: np.ndarray   # (2,) absorption XS
    fission: np.ndarray      # (2,) fission XS
    production: np.ndarray   # (2,) production XS (nu*sigma_f)
    chi: np.ndarray          # (2,) fission spectrum
    scattering: np.ndarray   # (2,) down-scattering (fast->thermal)


@dataclass
class DiffusionResult:
    """Results of a 1D diffusion calculation."""

    keff: float
    flux: np.ndarray          # (2, n_cells) group fluxes normalized to power
    current: np.ndarray       # (2, n_faces) net currents
    z_cells: np.ndarray       # (n_cells,) cell center positions
    z_faces: np.ndarray       # (n_faces,) face positions
    geometry: CoreGeometry


def _default_xs() -> tuple[TwoGroupXS, TwoGroupXS]:
    """Default cross sections matching MATLAB CORE1D.m."""
    reflector = TwoGroupXS(
        transport=np.array([0.3416, 0.9431]),
        absorption=np.array([0.0029, 0.0933]),
        fission=np.array([0.0, 0.0]),
        production=np.array([0.0, 0.0]),
        chi=np.array([0.0, 0.0]),
        scattering=np.array([2.4673e-04, 0.0]),
    )
    fuel = TwoGroupXS(
        transport=np.array([0.2181, 0.7850]),
        absorption=np.array([0.0096, 0.0959]),
        fission=np.array([0.0024, 0.0489]),
        production=np.array([0.0061, 0.1211]),
        chi=np.array([1.0, 0.0]),
        scattering=np.array([0.0160, 0.0]),
    )
    return reflector, fuel


class DiffusionSolver:
    """1D two-group diffusion eigenvalue solver (EigenvalueSolver protocol).

    Wraps the finite-difference diffusion operator with vacuum boundary
    conditions and BiCGSTAB inner solves into the generic power iteration
    framework.
    """

    def __init__(
        self,
        geom: CoreGeometry,
        reflector_xs: TwoGroupXS,
        fuel_xs: TwoGroupXS,
        *,
        errtol: float = 1e-6,
        maxiter_inner: int = 2000,
    ) -> None:
        self.geom = geom
        self.ng = 2
        self.nc = geom.n_cells
        self.nf = geom.n_faces
        self.dz = geom.dz
        self.dv = geom.dv
        self.errtol = errtol
        self.maxiter_inner = maxiter_inner

        # Build cell-wise cross sections: (2, n_cells)
        def _tile(refl_val: np.ndarray, fuel_val: np.ndarray) -> np.ndarray:
            return np.hstack([
                np.tile(refl_val[:, None], geom.n_refl_bot),
                np.tile(fuel_val[:, None], geom.n_fuel),
                np.tile(refl_val[:, None], geom.n_refl_top),
            ])

        self.sig_t = _tile(reflector_xs.transport, fuel_xs.transport)
        self.sig_a = _tile(reflector_xs.absorption, fuel_xs.absorption)
        self.sig_f = _tile(reflector_xs.fission, fuel_xs.fission)
        self.sig_p = _tile(reflector_xs.production, fuel_xs.production)
        self.chi = _tile(reflector_xs.chi, fuel_xs.chi)
        self.sig_s = _tile(reflector_xs.scattering, fuel_xs.scattering)

        # Face-interpolated transport XS for diffusion coefficient
        sig_t_face = np.zeros((self.ng, self.nf))
        for ig in range(self.ng):
            sig_t_face[ig, 0] = self.sig_t[ig, 0]
            sig_t_face[ig, -1] = self.sig_t[ig, -1]
            sig_t_face[ig, 1:-1] = 0.5 * (self.sig_t[ig, :-1] + self.sig_t[ig, 1:])

        self.D = 1.0 / (3.0 * sig_t_face)  # (2, n_faces) diffusion coefficient

        # Linear operator A*x
        self.A_op = LinearOperator(
            shape=(self.ng * self.nc, self.ng * self.nc),
            matvec=self._matvec,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _matvec(self, solution: np.ndarray) -> np.ndarray:
        """Diffusion operator: -div(D grad phi) + removal - inscatter."""
        fi = solution.reshape(self.ng, self.nc)

        # Gradient with vacuum BCs (phi=0 at boundaries)
        dfidz = np.zeros((self.ng, self.nf))
        dfidz[:, 0] = fi[:, 0] / (0.5 * self.dz)
        dfidz[:, -1] = -fi[:, -1] / (0.5 * self.dz)
        dfidz[:, 1:-1] = np.diff(fi, axis=1) / self.dz

        J = -self.D * dfidz

        # A*fi = div(J)/dz + (SigA + SigS)*fi - SigS_flipped * fi_flipped
        Ax = (
            np.diff(J, axis=1) / self.dz
            + (self.sig_a + self.sig_s) * fi
            - self.sig_s[::-1, :] * fi[::-1, :]
        )
        return Ax.ravel()

    def _compute_current(self, fi: np.ndarray) -> np.ndarray:
        """Net current J = -D * dphi/dz with vacuum BCs."""
        dfidz = np.zeros((self.ng, self.nf))
        dfidz[:, 0] = fi[:, 0] / (0.5 * self.dz)
        dfidz[:, -1] = -fi[:, -1] / (0.5 * self.dz)
        dfidz[:, 1:-1] = np.diff(fi, axis=1) / self.dz
        return -self.D * dfidz

    # ------------------------------------------------------------------
    # EigenvalueSolver protocol
    # ------------------------------------------------------------------

    def initial_flux_distribution(self) -> np.ndarray:
        """Return flat initial guess, shape (ng, nc)."""
        return np.ones((self.ng, self.nc))

    def compute_fission_source(
        self,
        flux_distribution: np.ndarray,
        keff: float,
    ) -> np.ndarray:
        """Fission RHS: Q_f = chi * sum_g(nu_sig_f * phi * dV) / keff.

        Standard eigenvalue formulation: A*phi = (1/k)*F*phi.
        """
        fi = flux_distribution
        p_rate = (self.sig_p * fi * self.dv).sum(axis=0)  # (n_cells,)
        return self.chi * np.tile(p_rate / keff, (self.ng, 1))

    def solve_fixed_source(
        self,
        fission_source: np.ndarray,
        flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Solve A*phi = fission_source via BiCGSTAB.

        The solution is conditioned by dividing out ``max(|phi|)`` to
        prevent overflow in the absence of per-iteration power
        normalization.
        """
        guess = flux_distribution.ravel()
        solution, _info = bicgstab(
            self.A_op,
            fission_source.ravel(),
            x0=guess,
            rtol=self.errtol,
            maxiter=self.maxiter_inner,
        )
        fi = solution.reshape(self.ng, self.nc)
        # Numerical conditioning: keep flux O(1) to prevent overflow.
        fi /= np.abs(fi).max()
        return fi

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        """k_eff = production / (absorption + leakage)."""
        fi = flux_distribution
        p_rate = (self.sig_p * fi * self.dv).sum()
        a_rate = (self.sig_a * fi * self.dv).sum()

        J = self._compute_current(fi)
        leakage = (-J[:, 0] * self.geom.az + J[:, -1] * self.geom.az).sum()

        return p_rate / (a_rate + leakage)

    def converged(
        self,
        keff: float,
        keff_old: float,
        flux_distribution: np.ndarray,
        flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        """Check keff change and relative solution change."""
        if iteration <= 1:
            return False
        rel_change = (
            np.linalg.norm(flux_distribution - flux_old)
            / np.linalg.norm(flux_distribution)
        )
        print(
            f"  keff = {keff:9.6f}  #outer = {iteration:3d}"
            f"  rel_change = {rel_change:.2e}"
        )
        return rel_change < 1e-5


def solve_diffusion_1d(
    geom: CoreGeometry | None = None,
    reflector_xs: TwoGroupXS | None = None,
    fuel_xs: TwoGroupXS | None = None,
    power_W: float = 1.76752e7,
    errtol: float = 1e-6,
    maxiter: int = 2000,
) -> DiffusionResult:
    """Solve the 1D two-group diffusion eigenvalue problem.

    Parameters
    ----------
    geom : CoreGeometry (default: standard PWR SA).
    reflector_xs, fuel_xs : TwoGroupXS (default: from MATLAB CORE1D.m).
    power_W : float — target power for flux normalization (W).
    """
    if geom is None:
        geom = CoreGeometry()
    if reflector_xs is None or fuel_xs is None:
        reflector_xs, fuel_xs = _default_xs()

    solver = DiffusionSolver(
        geom, reflector_xs, fuel_xs,
        errtol=errtol, maxiter_inner=maxiter,
    )

    keff, _keff_history, flux = power_iteration(solver, max_iter=200)

    # Normalize flux to target power (post-processing, not part of eigensolve)
    e_per_fission = np.array([0.3213e-10, 0.3206e-10])
    f_rate = solver.sig_f * flux * solver.dv
    pow_calc = (f_rate.sum(axis=1) * e_per_fission).sum()
    flux *= power_W / pow_calc

    # Final current from normalized flux
    J = solver._compute_current(flux)

    dz = geom.dz
    nc = geom.n_cells
    nf = geom.n_faces
    z_cells = np.arange(dz / 2, dz / 2 + dz * nc, dz)
    z_faces = np.arange(0, dz * nf, dz)

    return DiffusionResult(
        keff=keff,
        flux=flux,
        current=J,
        z_cells=z_cells,
        z_faces=z_faces,
        geometry=geom,
    )
