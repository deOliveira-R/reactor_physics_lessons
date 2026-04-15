"""Generic eigenvalue solvers for neutron transport and diffusion.

The criticality eigenvalue problem A·φ = (1/k)·F·φ has a spectrum of
eigenvalues k_0 > k_1 > k_2 > ...  The ``power_iteration`` function
converges to the **dominant eigenvalue** k_0 (= k_eff) and its
eigenvector φ_0 — the **fundamental mode**.  This is the only
physically meaningful steady-state solution: by the Perron-Frobenius
theorem the fundamental mode is the unique non-negative eigenvector,
while all higher harmonics change sign in space.

The eigenvector is a **flux distribution** — its shape is determined
but its absolute scale is arbitrary.  Normalizing to absolute flux
(e.g. via total power or integral flux) is a separate post-processing
step.

Any deterministic solver that can express its physics in terms of the
``EigenvalueSolver`` protocol can be plugged into the generic
``power_iteration`` loop defined here.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EigenvalueSolver(Protocol):
    """Contract for a deterministic neutron transport eigenvalue solver.

    Power iteration structure (each outer iteration):

    1. ``compute_fission_source`` — build the fission RHS from the
       current flux distribution estimate and eigenvalue.
    2. ``solve_fixed_source`` — apply the transport (or diffusion)
       operator to the fission source, returning an updated flux
       distribution.  Scattering and (n,2n) sources are assembled
       **inside** this method because they couple to the transport
       solve (e.g. inner scattering iterations in SN).
    3. ``compute_keff`` — update the eigenvalue from the neutron
       production / loss balance.
    """

    def initial_flux_distribution(self) -> np.ndarray:
        """Return an initial guess for the flux distribution."""
        ...

    def compute_fission_source(
        self,
        flux_distribution: np.ndarray,
        keff: float,
    ) -> np.ndarray:
        """Fission source: Q_f = χ · (νΣ_f · φ) / k_eff."""
        ...

    def solve_fixed_source(
        self,
        fission_source: np.ndarray,
        flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Apply the transport operator and return an updated flux distribution.

        This method encapsulates the model-specific physics:

        * **Collision probability** — direct matrix multiplication with P_inf.
        * **Discrete ordinates (SN)** — diamond-difference sweep with inner
          scattering iterations.
        * **Method of characteristics** — ray-tracing sweep.
        * **Diffusion** — implicit solve with BiCGSTAB.
        * **Homogeneous** — sparse direct solve of the removal matrix.

        Scattering and (n,2n) sources must be assembled inside this method
        so that inner iteration schemes (e.g. source iteration in SN) can
        update them between sweeps.

        Numerical conditioning (e.g. dividing by max(φ) to prevent overflow)
        is an implementation detail of this method, not physics normalization.
        """
        ...

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        """Compute the eigenvalue from the neutron balance.

        k_eff = production / (absorption + leakage)

        For lattice models with reflective boundary conditions the leakage
        term is zero.  For whole-core models with vacuum boundary conditions
        (e.g. diffusion) it is non-zero.
        """
        ...

    def converged(
        self,
        keff: float,
        keff_old: float,
        flux_distribution: np.ndarray,
        flux_old: np.ndarray,
        iteration: int,
    ) -> bool:
        """Return True when the outer iteration has converged."""
        ...


def power_iteration(
    solver: EigenvalueSolver,
    max_iter: int = 500,
) -> tuple[float, list[float], np.ndarray]:
    """Converge to the dominant eigenvalue and fundamental mode.

    Power iteration converges to the largest eigenvalue k_0 (= k_eff)
    and its eigenvector φ_0 (the fundamental mode).  The convergence
    rate is governed by the dominance ratio :math:`|k_1 / k_0|`.

    Returns
    -------
    keff : float
        Dominant eigenvalue (k_eff).
    keff_history : list[float]
        Eigenvalue estimate at each outer iteration.
    flux_distribution : np.ndarray
        Fundamental mode (arbitrary normalization).
    """
    flux_distribution = solver.initial_flux_distribution()
    keff = 1.0
    keff_history: list[float] = []

    for n in range(1, max_iter + 1):
        flux_old = flux_distribution.copy()
        keff_old = keff

        fission_source = solver.compute_fission_source(flux_distribution, keff)
        flux_distribution = solver.solve_fixed_source(fission_source, flux_distribution)
        keff = solver.compute_keff(flux_distribution)
        keff_history.append(keff)

        if solver.converged(keff, keff_old, flux_distribution, flux_old, n):
            break

    return keff, keff_history, flux_distribution
