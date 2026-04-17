r"""Method of Manufactured Solutions (MMS) cases for SN verification.

MMS is a **closed-form** construction of an L1 spatial-convergence test:
we pick a smooth angular flux :math:`\psi_n(x)` that satisfies the
vacuum boundary conditions, substitute it into the transport operator,
and algebraically compute the external source :math:`Q^{\text{ext}}`
that would have produced it. The solver is then run on this source;
any deviation of the numerical flux from :math:`\psi_n` is pure
spatial-discretisation error and must decay at the design order of
the scheme (:math:`\mathcal{O}(h^{2})` for diamond difference).

**1D slab ansatz** (vacuum BCs on :math:`[0, L]`, 1 group):

.. math::

    \psi_n(x) = \frac{1}{W}\,A(x),
    \qquad A(x) = \sin\!\left(\tfrac{\pi x}{L}\right)

where :math:`W = \sum_n w_n = 2` for Gauss–Legendre. The flux is
isotropic in angle, so the scalar flux recovered by any quadrature
order equals :math:`\phi(x) = A(x)` exactly — the test isolates
spatial error from angular quadrature error.

**Manufactured source**. Substituting into

.. math::

    \mu_n\,\psi'_n + \Sigma_t\,\psi_n
    = \frac{1}{W}\!\left(\Sigma_s\,\phi + Q^{\text{ext}}_n\right)

and solving for :math:`Q^{\text{ext}}_n`:

.. math::

    Q^{\text{ext}}_n(x)
    = \mu_n\,A'(x) + \bigl(\Sigma_t - \Sigma_s\bigr)\,A(x)
    = \mu_n\,\frac{\pi}{L}\cos\!\left(\tfrac{\pi x}{L}\right)
      + \bigl(\Sigma_t - \Sigma_s\bigr)\sin\!\left(\tfrac{\pi x}{L}\right).

The :math:`W` factor cancels because the ansatz is already divided
by :math:`W`; the solver divides the isotropic and anisotropic source
slots by :math:`W` internally, so what we hand it is already the
full residual.

The BCs :math:`A(0)=A(L)=0` imply :math:`\psi_n=0` on both faces
for every ordinate — vacuum BCs are satisfied automatically, so no
inflow-flux bookkeeping is required by the caller.

.. seealso::

   - :doc:`/theory/discrete_ordinates` — MMS verification section
     with the full derivation and convergence-rate argument.
   - :func:`orpheus.sn.solve_sn_fixed_source` — consumer of the
     external source produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.geometry import Mesh1D, Mesh2D
from orpheus.geometry.coord import CoordSystem
from orpheus.geometry.mesh import BC
from orpheus.sn.quadrature import GaussLegendre1D, LebedevSphere, ProductQuadrature

from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)


@dataclass(frozen=True)
class SNSlabMMSCase:
    r"""Closed-form MMS fixed-source problem for 1D slab SN verification.

    Attributes
    ----------
    name : str
        Unique identifier, e.g. ``"sn_mms_slab_sin"``.
    sigma_t, sigma_s : float
        Total and isotropic scattering macroscopic cross sections
        (1-group, cm\ :sup:`-1`). The absorption ratio
        :math:`c = \\Sigma_s/\\Sigma_t` controls source-iteration
        convergence; :math:`c<1` is required.
    slab_length : float
        Physical length :math:`L` of the slab in cm.
    materials : dict[int, Mixture]
        Material map consumable by :class:`orpheus.sn.SNSolver`.
    mat_id : int
        Material ID assigned to every mesh cell.
    quadrature : GaussLegendre1D
        Angular quadrature (shared across mesh refinements so the
        convergence study isolates spatial error).
    tolerance : str
        Expected convergence order, e.g. ``"O(h^2)"``.
    equation_labels : tuple[str, ...]
        Sphinx labels exercised by tests built from this case.
    """

    name: str
    sigma_t: float
    sigma_s: float
    slab_length: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: GaussLegendre1D
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cartesian",
        "dd-cartesian-1d",
        "dd-slab",
    )

    # ── Manufactured solution ─────────────────────────────────────────

    def phi_exact(self, x: np.ndarray) -> np.ndarray:
        r"""Scalar flux :math:`\phi(x) = \sin(\pi x/L)`."""
        return np.sin(np.pi * np.asarray(x) / self.slab_length)

    def dphi_exact(self, x: np.ndarray) -> np.ndarray:
        r"""Derivative :math:`A'(x) = (\pi/L)\cos(\pi x/L)`."""
        L = self.slab_length
        return (np.pi / L) * np.cos(np.pi * np.asarray(x) / L)

    # ── Mesh + source construction ────────────────────────────────────

    def build_mesh(self, n_cells: int) -> Mesh1D:
        """Uniform Cartesian slab mesh with ``n_cells`` equal cells."""
        edges = np.linspace(0.0, self.slab_length, n_cells + 1)
        mat_ids = np.full(n_cells, self.mat_id, dtype=int)
        return Mesh1D(edges=edges, mat_ids=mat_ids)

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r"""Per-ordinate external source :math:`Q^{\text{ext}}_n` on ``mesh``.

        Evaluated at cell centres to match the diamond-difference
        cell-average convention. Returned shape is
        ``(N, nx, 1, 1)`` — per ordinate, per cell, one energy group.
        """
        x = mesh.centers                          # (nx,)
        A = self.phi_exact(x)                     # (nx,)
        Ap = self.dphi_exact(x)                   # (nx,)
        mu = self.quadrature.mu_x                 # (N,)
        N = len(mu)
        nx = len(x)

        streaming = mu[:, None] * Ap[None, :]     # (N, nx)
        removal = (self.sigma_t - self.sigma_s) * A[None, :]  # (1, nx)
        Q = streaming + removal                   # (N, nx)
        return Q[:, :, None, None]                # (N, nx, 1, 1)


# ═══════════════════════════════════════════════════════════════════════
# Case builders
# ═══════════════════════════════════════════════════════════════════════

def _make_1g_mixture(sigma_t: float, sigma_s: float) -> Mixture:
    """Build a minimal 1-group non-fissile mixture with capture = Σ_t − Σ_s.

    The solver builds sig_a internally from absorption_xs =
    SigC + SigL + SigF + Sig2_out. With no fission / (n,2n) / (n,α),
    setting SigC = Σ_t − Σ_s gives absorption = Σ_t − Σ_s (exactly
    the pure-absorber fraction that completes the Σ_t balance).
    """
    if sigma_s >= sigma_t:
        raise ValueError(
            f"Need Σ_s < Σ_t for a physical mixture (got "
            f"Σ_t={sigma_t}, Σ_s={sigma_s})."
        )

    ng = 1
    SigS0 = csr_matrix(np.array([[sigma_s]], dtype=float))
    Sig2 = csr_matrix(np.zeros((ng, ng)))
    return Mixture(
        SigC=np.array([sigma_t - sigma_s]),
        SigL=np.zeros(ng),
        SigF=np.zeros(ng),
        SigP=np.zeros(ng),
        SigT=np.array([sigma_t]),
        SigS=[SigS0],
        Sig2=Sig2,
        chi=np.zeros(ng),
        eg=np.array([1e-5, 2e7]),
    )


def build_1d_slab_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    slab_length: float = 5.0,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_slab_sin",
) -> SNSlabMMSCase:
    r"""Build the canonical 1D slab MMS case.

    Default parameters give :math:`c = \Sigma_s/\Sigma_t = 0.5`
    (source iteration converges in ~40 sweeps to 1e-12) and a slab
    about 5 mean free paths thick, which fits several wavelengths
    of the :math:`\sin(\pi x/L)` ansatz without being so optically
    thick that the manufactured source amplitude is uninteresting.
    """
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = GaussLegendre1D.create(n_ordinates=n_ordinates)
    return SNSlabMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        slab_length=slab_length,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


def all_cases() -> list[SNSlabMMSCase]:
    """Return every registered MMS case (currently just the default)."""
    return [build_1d_slab_mms_case()]


# ═══════════════════════════════════════════════════════════════════════
# Phase 2.1a — heterogeneous continuous-Σ 2-group MMS case
# ═══════════════════════════════════════════════════════════════════════
r"""
2-group heterogeneous SN MMS reference.

**Problem.** A vacuum-BC slab of length :math:`L` with
**spatially continuous** cross sections :math:`\Sigma_{t,g}(x)`
and :math:`\Sigma_{s,g\to g'}(x)`. Continuous Σ(x) is deliberate:
discontinuous (piecewise-constant) cross sections degrade diamond
difference from :math:`\mathcal O(h^{2})` to :math:`\mathcal O(h)`
at interfaces that do not coincide with cell faces, which would
contaminate the spatial-convergence measurement with interface
treatment artefacts rather than testing the multigroup operator
itself. This follows the Salari & Knupp recommendation (SAND2000-1444,
§6): use smooth cross sections when you want to measure the
operator's design order on a heterogeneous problem.

**Ansatz.** Keep the same isotropic-in-angle ansatz as the
homogeneous case, with a **per-group amplitude vector**:

.. math::

    \psi_{n,g}(x) \;=\; \frac{c_g}{W}\,A(x),
    \qquad A(x) \;=\; \sin\!\left(\frac{\pi x}{L}\right),

giving the scalar flux :math:`\phi_g(x) = c_g\,A(x)`. The ansatz
vanishes at :math:`x = 0` and :math:`x = L` for every group and
every ordinate, so vacuum BCs are automatic (no inflow
bookkeeping). Picking a non-trivial amplitude vector
:math:`\mathbf c = (c_1, c_2)` (e.g. :math:`(1.0, 0.3)`) makes the
two groups linearly independent at the reference level.

**Manufactured source derivation.** Substituting the ansatz into
the multigroup discrete-ordinates transport equation

.. math::

    \mu_n\,\frac{\partial\psi_{n,g}}{\partial x}
        + \Sigma_{t,g}(x)\,\psi_{n,g}
    \;=\; \frac{1}{W}\!\left(
        \sum_{g'}\Sigma_{s,g'\to g}(x)\,\phi_{g'}(x)
      + Q^{\text{ext}}_{n,g}(x)
    \right),

and solving algebraically for :math:`Q^{\text{ext}}_{n,g}`:

.. math::

    Q^{\text{ext}}_{n,g}(x) \;=\;
        \mu_n\,c_g\,A'(x)
      + \Sigma_{t,g}(x)\,c_g\,A(x)
      - \sum_{g'}\Sigma_{s,g'\to g}(x)\,c_{g'}\,A(x).

**This is the load-bearing equation.** The ``g=1`` source
involves only :math:`c_1` (no upscatter), but the ``g=2`` source
couples to :math:`c_1` through the downscatter term
:math:`\Sigma_{s,1\to 2}(x)\,c_1\,A(x)`, so the test exercises the
multigroup scatter assembly in the sweep. A bug that forgets to
accumulate in-scatter from other groups (or transposes the
scatter matrix) will produce an incorrect :math:`\phi_2` that
the O(h²) convergence test catches immediately.

**Precision floor.** The ansatz is smooth
(:math:`C^{\infty}`), the cross sections are smooth
(:math:`C^{\infty}`), the quadrature is exact for constant-in-:math:`\mu`
integrands (isotropic ansatz), so the ONLY remaining error at
convergence is the spatial diamond-difference truncation,
:math:`\mathcal O(h^{2})` exactly. The finest-mesh error floor
for the convergence study is the solver's own BiCGSTAB / source-
iteration convergence tolerance (observed ~1e-10 with
``inner_tol=1e-12``), well below the discretisation error at
the refinements used.

.. seealso::

    - :func:`build_1d_slab_heterogeneous_mms_case` — the builder
      that constructs a fully-specified instance.
    - :class:`SNSlab2GHeterogeneousMMSCase` — the dataclass that
      carries the continuous cross-section functions and the
      reference solution.
    - ``tests/sn/test_mms_heterogeneous.py`` — the L1 convergence
      consumer test.
    - ``docs/theory/discrete_ordinates.rst`` — the heterogeneous
      MMS verification section.
"""


@dataclass(frozen=True)
class SNSlab2GHeterogeneousMMSCase:
    r"""Continuous-Σ 2-group heterogeneous SN MMS reference.

    Unlike :class:`SNSlabMMSCase` this case carries the cross
    sections as **callables** :math:`\Sigma_{t,g}(x)`,
    :math:`\Sigma_{s,g\to g'}(x)` rather than scalars, so every
    cell gets a distinct material with cross sections evaluated
    at the cell centre. That per-cell material is built on
    demand by :meth:`build_materials`, so mesh refinements
    construct fresh materials without any caching machinery.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"sn_mms_slab_2g_hetero"``.
    slab_length : float
        Physical length :math:`L` of the slab in cm.
    c_spectrum : ndarray, shape (2,)
        Per-group amplitudes :math:`\mathbf c = (c_1, c_2)`. The
        ansatz scalar flux is :math:`\phi_g(x) = c_g\sin(\pi x/L)`.
    sigma_t_fn : callable
        ``sigma_t_fn(x, g) -> ndarray`` returning :math:`\Sigma_{t,g}(x)`
        evaluated at every point in ``x`` (shape preserved).
    sigma_s_fn : callable
        ``sigma_s_fn(x, g_from, g_to) -> ndarray`` returning
        :math:`\Sigma_{s,g_{\text{from}}\to g_{\text{to}}}(x)` on
        the same shape as ``x``.
    quadrature : GaussLegendre1D
        Fixed angular quadrature used across all mesh refinements
        (so the spatial convergence study isolates spatial error).
    n_groups : int
        Number of energy groups. Fixed at 2 for this class; a
        general-:math:`n_g` variant is a Phase-3 extension.
    tolerance : str
        Expected convergence order, e.g. ``"O(h²)"``.
    equation_labels : tuple[str, ...]
        Sphinx ``:label:`` IDs the test ``@pytest.mark.verifies(...)``
        should reference.
    """

    name: str
    slab_length: float
    c_spectrum: np.ndarray
    sigma_t_fn: "Callable[[np.ndarray, int], np.ndarray]"
    sigma_s_fn: "Callable[[np.ndarray, int, int], np.ndarray]"
    quadrature: GaussLegendre1D
    n_groups: int = 2
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cartesian",
        "dd-cartesian-1d",
        "dd-slab",
        "multigroup",
        "mg-balance",
        "sn-mms-hetero-psi",
        "sn-mms-hetero-qext",
    )

    # ── Reference scalar flux ────────────────────────────────────────

    def phi_exact(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r"""Reference :math:`\phi_g(x) = c_g\,\sin(\pi x/L)`."""
        x = np.asarray(x, dtype=float)
        return self.c_spectrum[g] * np.sin(np.pi * x / self.slab_length)

    def dphi_exact(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r"""Reference derivative
        :math:`\phi_g'(x) = c_g\,(\pi/L)\cos(\pi x/L)`."""
        x = np.asarray(x, dtype=float)
        L = self.slab_length
        return self.c_spectrum[g] * (np.pi / L) * np.cos(np.pi * x / L)

    # ── Mesh + materials construction ────────────────────────────────

    def build_mesh(self, n_cells: int) -> Mesh1D:
        """Uniform Cartesian slab mesh with ``n_cells`` cells and
        a unique material ID per cell."""
        edges = np.linspace(0.0, self.slab_length, n_cells + 1)
        mat_ids = np.arange(n_cells, dtype=int)
        return Mesh1D(edges=edges, mat_ids=mat_ids)

    def build_materials(self, mesh: Mesh1D) -> dict[int, Mixture]:
        r"""Build a per-cell material dictionary by sampling the
        continuous cross-section functions at each cell's centre.

        Each cell ``i`` gets a :class:`Mixture` whose
        :math:`\Sigma_t`, :math:`\Sigma_s` row, and absorption
        :math:`\Sigma_a = \Sigma_t - \Sigma_{s,\text{total}}` are
        set from the callables at :math:`x_i = (x_{i-1/2} + x_{i+1/2})/2`.

        This is exactly the midpoint rule for the cell-average
        cross section, which is :math:`\mathcal O(h^{2})`-accurate
        for smooth :math:`\Sigma(x)`. That accuracy matches the
        diamond-difference design order and does not degrade the
        measured convergence rate.
        """
        centers = mesh.centers  # (n_cells,)
        materials: dict[int, Mixture] = {}
        for i, x_i in enumerate(centers):
            sig_t = np.array([
                float(self.sigma_t_fn(np.array([x_i]), 0)[0]),
                float(self.sigma_t_fn(np.array([x_i]), 1)[0]),
            ])
            sig_s_row = np.zeros((2, 2))
            for g_from in range(2):
                for g_to in range(2):
                    sig_s_row[g_from, g_to] = float(
                        self.sigma_s_fn(np.array([x_i]), g_from, g_to)[0]
                    )
            sig_s_total_out = sig_s_row.sum(axis=1)  # row sum per from-group
            sig_a = sig_t - sig_s_total_out
            if np.any(sig_a <= 0):
                raise ValueError(
                    f"Cross sections at x={x_i}: Σ_t={sig_t}, "
                    f"Σ_s_total_out={sig_s_total_out}, Σ_a={sig_a}. "
                    "Need Σ_a > 0 everywhere for a physical mixture."
                )
            materials[i] = Mixture(
                SigC=sig_a,                        # pure absorber capture
                SigL=np.zeros(2),                  # no (n,α)
                SigF=np.zeros(2),                  # no fission
                SigP=np.zeros(2),                  # no production
                SigT=sig_t,
                SigS=[csr_matrix(sig_s_row)],      # P0 only
                Sig2=csr_matrix(np.zeros((2, 2))),  # no (n,2n)
                chi=np.zeros(2),
                eg=np.array([1e-5, 1e3, 2e7]),     # dummy energy boundaries
            )
        return materials

    # ── Manufactured source on the mesh ──────────────────────────────

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r"""Per-ordinate, per-cell, per-group external source.

        Shape ``(N_ord, n_cells, 1, n_groups)``, matching the
        convention expected by
        :func:`orpheus.sn.solve_sn_fixed_source`. The formula is

        .. math::

            Q^{\text{ext}}_{n,g}(x_i) \;=\;
                \mu_n\,c_g\,A'(x_i)
              + \Sigma_{t,g}(x_i)\,c_g\,A(x_i)
              - \sum_{g'}\Sigma_{s,g'\to g}(x_i)\,c_{g'}\,A(x_i),

        evaluated at cell centres. The first term (streaming) is
        per-ordinate; the removal and in-scatter terms are
        isotropic across ordinates.
        """
        x = mesh.centers
        L = self.slab_length
        # Compute the spatial ansatz shape A(x) directly rather than
        # dividing phi_exact(x, 0) by c_spectrum[0] — the latter would
        # divide-by-zero if a caller constructs a degenerate case
        # with c_0 = 0 (e.g. the multigroup-coupling regression test).
        A = np.sin(np.pi * x / L)
        Ap = (np.pi / L) * np.cos(np.pi * x / L)
        mu = self.quadrature.mu_x
        N = len(mu)
        nx = len(x)
        ng = self.n_groups

        Q = np.zeros((N, nx, 1, ng))
        for g in range(ng):
            c_g = self.c_spectrum[g]
            sig_t_g = np.asarray(self.sigma_t_fn(x, g), dtype=float)  # (nx,)
            streaming = mu[:, None] * c_g * Ap[None, :]               # (N, nx)
            removal = c_g * sig_t_g * A                               # (nx,)
            in_scatter = np.zeros_like(A)
            for g_from in range(ng):
                sig_s = np.asarray(
                    self.sigma_s_fn(x, g_from, g), dtype=float,
                )  # (nx,)
                in_scatter += sig_s * self.c_spectrum[g_from] * A
            Q[:, :, 0, g] = streaming + (removal - in_scatter)[None, :]
        return Q


def _default_hetero_xs_functions() -> tuple[
    "Callable[[np.ndarray, int], np.ndarray]",
    "Callable[[np.ndarray, int, int], np.ndarray]",
]:
    r"""Return the canonical smooth 2-group cross-section functions.

    Chosen so that :math:`\Sigma_{a,g}(x) > 0` everywhere on
    :math:`[0, L]` for any slab length (verified algebraically):

    - :math:`\Sigma_{t,1}(x) = 1.0 + 0.2\sin(\pi x/L)` →
      :math:`\Sigma_{a,1} = 0.5 + 0.05\sin(\pi x/L) > 0`.
    - :math:`\Sigma_{t,2}(x) = 2.0 + 0.3\cos(\pi x/L)` →
      :math:`\Sigma_{a,2} = 0.5 + 0.3\cos(\pi x/L) - 0.15\sin(\pi x/L)`
      which is bounded below by :math:`0.5 - \sqrt{0.3^2 + 0.15^2}
      \approx 0.165 > 0`.

    The scattering ratios :math:`c_g = \Sigma_{s,\text{tot},g}/\Sigma_{t,g}`
    stay around 0.5 for both groups, giving geometric source-
    iteration convergence at rate :math:`\sim 0.5^{n}` per sweep.
    """
    L_holder: dict[str, float] = {}  # filled by the builder

    def sigma_t_fn(x: np.ndarray, g: int) -> np.ndarray:
        L = L_holder["L"]
        s = np.sin(np.pi * np.asarray(x, dtype=float) / L)
        c = np.cos(np.pi * np.asarray(x, dtype=float) / L)
        if g == 0:
            return 1.0 + 0.2 * s
        if g == 1:
            return 2.0 + 0.3 * c
        raise ValueError(f"2-group case: g must be 0 or 1, got {g}")

    def sigma_s_fn(x: np.ndarray, g_from: int, g_to: int) -> np.ndarray:
        L = L_holder["L"]
        s = np.sin(np.pi * np.asarray(x, dtype=float) / L)
        if g_from == 0 and g_to == 0:
            return 0.3 + 0.1 * s
        if g_from == 0 and g_to == 1:
            return 0.2 + 0.05 * s
        if g_from == 1 and g_to == 1:
            return 1.5 + 0.15 * s
        if g_from == 1 and g_to == 0:
            return np.zeros_like(np.asarray(x, dtype=float))
        raise ValueError(
            f"2-group case: g_from, g_to must be 0 or 1, got "
            f"({g_from}, {g_to})"
        )

    return sigma_t_fn, sigma_s_fn, L_holder  # type: ignore[return-value]


def build_1d_slab_heterogeneous_mms_case(
    slab_length: float = 5.0,
    c_spectrum: tuple[float, float] = (1.0, 0.3),
    n_ordinates: int = 16,
    name: str = "sn_mms_slab_2g_hetero",
) -> SNSlab2GHeterogeneousMMSCase:
    r"""Build the canonical 2-group heterogeneous SN MMS case.

    Default parameters:

    - :math:`L = 5\,\text{cm}` — several mean free paths,
      enough wavelengths of the :math:`\sin(\pi x/L)` ansatz to
      exercise the streaming term non-trivially.
    - :math:`\mathbf c = (1.0, 0.3)` — non-trivial group ratio so
      the downscatter coupling is visible in the manufactured source.
    - :math:`N = 16` — S16 Gauss-Legendre quadrature, fixed across
      refinements so the convergence study isolates spatial error.

    The smooth cross-section profiles come from
    :func:`_default_hetero_xs_functions` and have
    :math:`\Sigma_a > 0` everywhere on :math:`[0, L]`.
    """
    sigma_t_fn, sigma_s_fn, L_holder = _default_hetero_xs_functions()
    L_holder["L"] = float(slab_length)
    quad = GaussLegendre1D.create(n_ordinates=n_ordinates)
    return SNSlab2GHeterogeneousMMSCase(
        name=name,
        slab_length=float(slab_length),
        c_spectrum=np.asarray(c_spectrum, dtype=float),
        sigma_t_fn=sigma_t_fn,
        sigma_s_fn=sigma_s_fn,
        quadrature=quad,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.1 — 2D Cartesian MMS (1-group, Lebedev quadrature)
# ═══════════════════════════════════════════════════════════════════════
r"""
2D Cartesian SN MMS reference.

**Problem.** A vacuum-BC rectangle :math:`[0, L_x] \times [0, L_y]`
with uniform cross sections, 1 energy group, Lebedev angular
quadrature. The MMS ansatz is separable and isotropic in angle:

.. math::

    \psi_n(x, y) = \frac{1}{W}\,A(x, y),
    \qquad A(x, y) = \sin\!\left(\frac{\pi x}{L_x}\right)
                      \sin\!\left(\frac{\pi y}{L_y}\right),

so the scalar flux equals :math:`\phi(x, y) = A(x, y)` for any
quadrature set — angular error is exactly zero.

**Manufactured source.** Substituting into the 2D transport equation
:eq:`transport-cartesian-2d`:

.. math::

    Q^{\text{ext}}_n(x, y) \;=\;
        \mu_{x,n}\,\frac{\partial A}{\partial x}
      + \mu_{y,n}\,\frac{\partial A}{\partial y}
      + (\Sigma_t - \Sigma_s)\,A(x, y).

The partial derivatives are:

.. math::

    \frac{\partial A}{\partial x} =
        \frac{\pi}{L_x}\cos\!\left(\frac{\pi x}{L_x}\right)
        \sin\!\left(\frac{\pi y}{L_y}\right), \qquad
    \frac{\partial A}{\partial y} =
        \sin\!\left(\frac{\pi x}{L_x}\right)
        \frac{\pi}{L_y}\cos\!\left(\frac{\pi y}{L_y}\right).

The ansatz vanishes on all four edges, so vacuum BCs are automatic.

.. seealso::

    - :doc:`/theory/discrete_ordinates` — 2D Cartesian MMS section.
    - :func:`orpheus.sn.solve_sn_fixed_source` — consumer.
"""


@dataclass(frozen=True)
class SN2DCartesianMMSCase:
    r"""Closed-form MMS fixed-source problem for 2D Cartesian SN verification.

    Attributes
    ----------
    name : str
        Unique identifier, e.g. ``"sn_mms_2d_cartesian_sin"``.
    sigma_t, sigma_s : float
        Total and isotropic scattering cross sections (1-group, cm⁻¹).
    length_x, length_y : float
        Physical dimensions of the rectangle in cm.
    materials : dict[int, Mixture]
        Material map consumable by the SN solver.
    mat_id : int
        Material ID assigned to every cell.
    quadrature : LebedevSphere
        Angular quadrature (fixed across mesh refinements).
    tolerance : str
        Expected convergence order.
    equation_labels : tuple[str, ...]
        Sphinx labels exercised by tests built from this case.
    """

    name: str
    sigma_t: float
    sigma_s: float
    length_x: float
    length_y: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: LebedevSphere
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cartesian-2d",
        "dd-cartesian-2d",
        "sn-mms-2d-psi",
        "sn-mms-2d-qext",
    )

    # ── Manufactured solution ─────────────────────────────────────────

    def phi_exact(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Reference scalar flux :math:`\phi(x,y) = \sin(\pi x/L_x)\sin(\pi y/L_y)`.

        Parameters
        ----------
        x, y : ndarray, shapes (nx,) and (ny,)
            Cell-centre coordinates.  Broadcast to ``(nx, ny)`` via
            outer product.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.sin(np.pi * x[:, None] / self.length_x) * \
               np.sin(np.pi * y[None, :] / self.length_y)

    # ── Mesh + source construction ────────────────────────────────────

    def build_mesh(self, nx: int, ny: int | None = None) -> Mesh2D:
        """Uniform Cartesian 2D mesh with ``nx × ny`` cells.

        If ``ny`` is None, uses ``ny = nx`` for a square mesh.
        """
        if ny is None:
            ny = nx
        edges_x = np.linspace(0.0, self.length_x, nx + 1)
        edges_y = np.linspace(0.0, self.length_y, ny + 1)
        mat_map = np.full((nx, ny), self.mat_id, dtype=int)
        return Mesh2D(edges_x=edges_x, edges_y=edges_y, mat_map=mat_map)

    def external_source(self, mesh: Mesh2D) -> np.ndarray:
        r"""Per-ordinate external source on a 2D mesh.

        Returns shape ``(N, nx, ny, 1)`` — per ordinate, per cell (x),
        per cell (y), one energy group.  Evaluated at cell centres.
        """
        cx = mesh.centers_x                          # (nx,)
        cy = mesh.centers_y                          # (ny,)
        Lx, Ly = self.length_x, self.length_y

        # Spatial ansatz and its partial derivatives
        sin_x = np.sin(np.pi * cx / Lx)              # (nx,)
        cos_x = np.cos(np.pi * cx / Lx)              # (nx,)
        sin_y = np.sin(np.pi * cy / Ly)              # (ny,)
        cos_y = np.cos(np.pi * cy / Ly)              # (ny,)

        A = sin_x[:, None] * sin_y[None, :]           # (nx, ny)
        dA_dx = (np.pi / Lx) * cos_x[:, None] * sin_y[None, :]  # (nx, ny)
        dA_dy = sin_x[:, None] * (np.pi / Ly) * cos_y[None, :]  # (nx, ny)

        mu_x = self.quadrature.mu_x                   # (N,)
        mu_y = self.quadrature.mu_y                   # (N,)
        N = len(mu_x)

        # streaming: mu_x * dA/dx + mu_y * dA/dy   → (N, nx, ny)
        streaming = (mu_x[:, None, None] * dA_dx[None, :, :]
                     + mu_y[:, None, None] * dA_dy[None, :, :])
        removal = (self.sigma_t - self.sigma_s) * A   # (nx, ny)
        Q = streaming + removal[None, :, :]            # (N, nx, ny)
        return Q[:, :, :, None]                        # (N, nx, ny, 1)


def build_2d_cartesian_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 5.0,
    lebedev_order: int = 17,
    mat_id: int = 1,
    name: str = "sn_mms_2d_cartesian_sin",
) -> SN2DCartesianMMSCase:
    r"""Build the canonical 2D Cartesian MMS case.

    Default parameters:

    - :math:`c = 0.5` — geometric source-iteration convergence.
    - :math:`L_x = L_y = 5\,\text{cm}` — square domain, several MFP.
    - Lebedev order 17 (110 ordinates) — consistent with existing
      2D eigenvalue tests.
    """
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = LebedevSphere.create(order=lebedev_order)
    return SN2DCartesianMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        length_x=length_x,
        length_y=length_y,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.2 — 2D Cartesian MMS (2-group, heterogeneous, Lebedev)
# ═══════════════════════════════════════════════════════════════════════
r"""
2-group heterogeneous 2D Cartesian SN MMS reference.

**Problem.** Combine the 2D Cartesian geometry from Phase 3.1 with
the 2-group smooth-:math:`\Sigma` heterogeneous approach from
Phase 2.1a. The cross sections are smooth 2D functions
:math:`\Sigma(x, y)` so the diamond-difference design order
:math:`\mathcal O(h^{2})` is preserved (no interface degradation).

**Ansatz.** Per-group amplitudes :math:`c_g` with the same 2D
separable shape:

.. math::

    \psi_{n,g}(x, y) = \frac{c_g}{W}\,A(x, y), \qquad
    A(x, y) = \sin(\pi x/L_x)\sin(\pi y/L_y).

**Manufactured source.** From the 2D multigroup transport equation:

.. math::

    Q^{\text{ext}}_{n,g}(x, y) =
        \mu_{x,n}\,c_g\,\partial_x A
      + \mu_{y,n}\,c_g\,\partial_y A
      + \Sigma_{t,g}(x, y)\,c_g\,A
      - \sum_{g'}\Sigma_{s,g'\to g}(x, y)\,c_{g'}\,A.
"""


@dataclass(frozen=True)
class SN2DCartesian2GHeterogeneousMMSCase:
    r"""2-group heterogeneous MMS case for 2D Cartesian SN verification.

    Cross sections are **callables** :math:`\Sigma(x, y, g)` evaluated
    at cell centres. Each cell gets a unique :class:`Mixture`.
    """

    name: str
    length_x: float
    length_y: float
    c_spectrum: np.ndarray
    sigma_t_fn: "Callable[[np.ndarray, np.ndarray, int], np.ndarray]"
    sigma_s_fn: "Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]"
    quadrature: LebedevSphere
    n_groups: int = 2
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cartesian-2d",
        "dd-cartesian-2d",
        "multigroup",
        "mg-balance",
        "sn-mms-2d-2g-psi",
        "sn-mms-2d-2g-qext",
    )

    # ── Reference scalar flux ────────────────────────────────────────

    def phi_exact(
        self, x: np.ndarray, y: np.ndarray, g: int = 0,
    ) -> np.ndarray:
        r"""Reference :math:`\phi_g(x,y) = c_g \sin(\pi x/L_x)\sin(\pi y/L_y)`.

        Returns shape ``(len(x), len(y))``.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.c_spectrum[g] * (
            np.sin(np.pi * x[:, None] / self.length_x)
            * np.sin(np.pi * y[None, :] / self.length_y)
        )

    # ── Mesh + materials construction ────────────────────────────────

    def build_mesh(self, nx: int, ny: int | None = None) -> Mesh2D:
        """Uniform 2D mesh with unique material ID per cell."""
        if ny is None:
            ny = nx
        edges_x = np.linspace(0.0, self.length_x, nx + 1)
        edges_y = np.linspace(0.0, self.length_y, ny + 1)
        mat_map = np.arange(nx * ny, dtype=int).reshape(nx, ny)
        return Mesh2D(edges_x=edges_x, edges_y=edges_y, mat_map=mat_map)

    def build_materials(self, mesh: Mesh2D) -> dict[int, Mixture]:
        """Build per-cell materials by sampling Σ(x,y) at cell centres."""
        cx = mesh.centers_x
        cy = mesh.centers_y
        ng = self.n_groups
        materials: dict[int, Mixture] = {}

        for i, xi in enumerate(cx):
            for j, yj in enumerate(cy):
                cell_id = mesh.mat_map[i, j]
                xa = np.array([xi])
                ya = np.array([yj])
                sig_t = np.array([
                    float(self.sigma_t_fn(xa, ya, g)[0]) for g in range(ng)
                ])
                sig_s_row = np.zeros((ng, ng))
                for g_from in range(ng):
                    for g_to in range(ng):
                        sig_s_row[g_from, g_to] = float(
                            self.sigma_s_fn(xa, ya, g_from, g_to)[0]
                        )
                sig_a = sig_t - sig_s_row.sum(axis=1)
                materials[cell_id] = Mixture(
                    SigC=sig_a,
                    SigL=np.zeros(ng),
                    SigF=np.zeros(ng),
                    SigP=np.zeros(ng),
                    SigT=sig_t,
                    SigS=[csr_matrix(sig_s_row)],
                    Sig2=csr_matrix(np.zeros((ng, ng))),
                    chi=np.zeros(ng),
                    eg=np.array([1e-5, 1e3, 2e7]),
                )
        return materials

    # ── Manufactured source ──────────────────────────────────────────

    def external_source(self, mesh: Mesh2D) -> np.ndarray:
        r"""Per-ordinate, per-cell, per-group external source.

        Shape ``(N_ord, nx, ny, n_groups)``.
        """
        cx = mesh.centers_x
        cy = mesh.centers_y
        Lx, Ly = self.length_x, self.length_y
        ng = self.n_groups

        sin_x = np.sin(np.pi * cx / Lx)
        cos_x = np.cos(np.pi * cx / Lx)
        sin_y = np.sin(np.pi * cy / Ly)
        cos_y = np.cos(np.pi * cy / Ly)

        A = sin_x[:, None] * sin_y[None, :]
        dA_dx = (np.pi / Lx) * cos_x[:, None] * sin_y[None, :]
        dA_dy = sin_x[:, None] * (np.pi / Ly) * cos_y[None, :]

        mu_x = self.quadrature.mu_x
        mu_y = self.quadrature.mu_y
        N = len(mu_x)
        nx, ny_ = len(cx), len(cy)

        # Evaluate cross sections on the 2D grid
        xx, yy = np.meshgrid(cx, cy, indexing="ij")  # (nx, ny)
        xx_flat = xx.ravel()
        yy_flat = yy.ravel()

        Q = np.zeros((N, nx, ny_, ng))
        for g in range(ng):
            c_g = self.c_spectrum[g]
            sig_t_g = self.sigma_t_fn(xx_flat, yy_flat, g).reshape(nx, ny_)
            streaming = (mu_x[:, None, None] * c_g * dA_dx[None, :, :]
                         + mu_y[:, None, None] * c_g * dA_dy[None, :, :])
            removal = c_g * sig_t_g * A
            in_scatter = np.zeros((nx, ny_))
            for g_from in range(ng):
                sig_s = self.sigma_s_fn(
                    xx_flat, yy_flat, g_from, g,
                ).reshape(nx, ny_)
                in_scatter += sig_s * self.c_spectrum[g_from] * A
            Q[:, :, :, g] = streaming + (removal - in_scatter)[None, :, :]
        return Q


def _default_hetero_2d_xs_functions(
    Lx: float, Ly: float,
) -> tuple[
    "Callable[[np.ndarray, np.ndarray, int], np.ndarray]",
    "Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]",
]:
    r"""Return smooth 2-group 2D cross-section functions.

    Same base profiles as the 1D heterogeneous case (Phase 2.1a),
    with an additional mild :math:`y`-dependent modulation to exercise
    both spatial dimensions:

    - :math:`\Sigma_{t,1}(x,y) = 1.0 + 0.2\sin(\pi x/L_x)
      + 0.1\cos(\pi y/L_y)`
    - :math:`\Sigma_{t,2}(x,y) = 2.0 + 0.3\cos(\pi x/L_x)
      + 0.1\sin(\pi y/L_y)`

    The scattering cross sections add a :math:`0.05\cos(\pi y/L_y)`
    modulation to the 1D profiles. All :math:`\Sigma_a > 0` bounds
    from the 1D case are preserved because the :math:`y`-modulation
    amplitudes (0.1, 0.05) are smaller than the 1D margins (~0.165).
    """
    def sigma_t_fn(x: np.ndarray, y: np.ndarray, g: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sx = np.sin(np.pi * x / Lx)
        cx = np.cos(np.pi * x / Lx)
        sy = np.sin(np.pi * y / Ly)
        cy = np.cos(np.pi * y / Ly)
        if g == 0:
            return 1.0 + 0.2 * sx + 0.1 * cy
        if g == 1:
            return 2.0 + 0.3 * cx + 0.1 * sy
        raise ValueError(f"2-group: g must be 0 or 1, got {g}")

    def sigma_s_fn(
        x: np.ndarray, y: np.ndarray, g_from: int, g_to: int,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sx = np.sin(np.pi * x / Lx)
        cy = np.cos(np.pi * y / Ly)
        if g_from == 0 and g_to == 0:
            return 0.3 + 0.1 * sx + 0.05 * cy
        if g_from == 0 and g_to == 1:
            return 0.2 + 0.05 * sx
        if g_from == 1 and g_to == 1:
            return 1.5 + 0.15 * sx + 0.05 * cy
        if g_from == 1 and g_to == 0:
            return np.zeros_like(x)
        raise ValueError(
            f"2-group: g_from, g_to must be 0 or 1, got ({g_from}, {g_to})"
        )

    return sigma_t_fn, sigma_s_fn


def build_2d_cartesian_heterogeneous_mms_case(
    length_x: float = 5.0,
    length_y: float = 5.0,
    c_spectrum: tuple[float, float] = (1.0, 0.3),
    lebedev_order: int = 17,
    name: str = "sn_mms_2d_cartesian_2g_hetero",
) -> SN2DCartesian2GHeterogeneousMMSCase:
    r"""Build the canonical 2D 2-group heterogeneous MMS case.

    Smooth :math:`\Sigma(x, y)` with 2D spatial variation and
    per-group amplitudes :math:`\mathbf c = (1.0, 0.3)`.
    """
    sigma_t_fn, sigma_s_fn = _default_hetero_2d_xs_functions(
        float(length_x), float(length_y),
    )
    quad = LebedevSphere.create(order=lebedev_order)
    return SN2DCartesian2GHeterogeneousMMSCase(
        name=name,
        length_x=float(length_x),
        length_y=float(length_y),
        c_spectrum=np.asarray(c_spectrum, dtype=float),
        sigma_t_fn=sigma_t_fn,
        sigma_s_fn=sigma_s_fn,
        quadrature=quad,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.5 — 1D Cartesian P1 anisotropic scattering MMS
# ═══════════════════════════════════════════════════════════════════════
r"""
P1 anisotropic scattering MMS reference.

**Problem.**  1-group vacuum-BC slab with linearly anisotropic
scattering (:math:`\ell = 0, 1`). The ansatz has weak
:math:`\mu`-dependence so the P1 scattering slot is exercised:

.. math::

    \psi_n(x) = \frac{1}{W}\bigl(A(x) + \alpha\,\mu_n\,B(x)\bigr)

with :math:`A(x) = \sin(\pi x/L)`, :math:`B(x) = \sin(\pi x/L)`,
and small :math:`\alpha`.  The scalar flux is
:math:`\phi(x) = A(x)` (isotropic term), the current is
:math:`J(x) = \alpha\,B(x)/3` (for Gauss–Legendre on :math:`[-1, 1]`
where :math:`\sum w_n\mu_n^2 = 2/3` and :math:`W = 2`).

**Manufactured source.** From the 1D transport equation with P1
scattering:

.. math::

    Q^{\text{ext}}_n(x) =
        \mu_n A'(x)
      + (\Sigma_t - \Sigma_s^{(0)})\,A(x)
      + \alpha\,\mu_n\,(\Sigma_t - \Sigma_s^{(1)})\,B(x)
      + \alpha\,\mu_n^2\,B'(x)
"""


@dataclass(frozen=True)
class SNP1AnisoMMSCase:
    r"""MMS case with P1 anisotropic scattering on a 1D Cartesian slab.

    Attributes
    ----------
    sigma_t : float
        Total cross section.
    sigma_s0 : float
        P0 (isotropic) scattering cross section.
    sigma_s1 : float
        P1 (linearly anisotropic) scattering cross section.
    alpha : float
        Strength of the μ-dependent term in the ansatz.
    """

    name: str
    sigma_t: float
    sigma_s0: float
    sigma_s1: float
    alpha: float
    slab_length: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: GaussLegendre1D
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cartesian",
        "dd-cartesian-1d",
        "dd-slab",
        "pn-scatter",
        "sn-mms-p1-psi",
        "sn-mms-p1-qext",
    )

    def phi_exact(self, x: np.ndarray) -> np.ndarray:
        r"""Scalar flux :math:`\phi(x) = A(x) = \sin(\pi x/L)`."""
        return np.sin(np.pi * np.asarray(x) / self.slab_length)

    def build_mesh(self, n_cells: int) -> Mesh1D:
        edges = np.linspace(0.0, self.slab_length, n_cells + 1)
        mat_ids = np.full(n_cells, self.mat_id, dtype=int)
        return Mesh1D(edges=edges, mat_ids=mat_ids)

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r"""Per-ordinate external source for the P1 MMS ansatz.

        .. math::

            Q_n(x) = \mu_n A' + (\Sigma_t - \Sigma_s^0) A
                    + \alpha\mu_n(\Sigma_t - \Sigma_s^1) B
                    + \alpha\mu_n^2 B'

        where :math:`A = B = \sin(\pi x/L)`, :math:`A' = B' = (\pi/L)\cos(\pi x/L)`.
        Shape ``(N, nx, 1, 1)``.
        """
        x = mesh.centers
        L = self.slab_length
        A = np.sin(np.pi * x / L)
        Ap = (np.pi / L) * np.cos(np.pi * x / L)
        mu = self.quadrature.mu_x
        N = len(mu)
        a = self.alpha

        # Each term: (N, nx)
        t1 = mu[:, None] * Ap[None, :]                              # μ A'
        t2 = (self.sigma_t - self.sigma_s0) * A[None, :]            # (Σ_t - Σ_s0) A
        t3 = a * mu[:, None] * (self.sigma_t - self.sigma_s1) * A[None, :]  # α μ (Σ_t - Σ_s1) B
        t4 = a * (mu[:, None] ** 2) * Ap[None, :]                   # α μ² B'
        Q = t1 + t2 + t3 + t4
        return Q[:, :, None, None]


def _make_1g_p1_mixture(
    sigma_t: float, sigma_s0: float, sigma_s1: float,
) -> Mixture:
    """Build a 1-group mixture with P0 and P1 scattering matrices."""
    if sigma_s0 >= sigma_t:
        raise ValueError(f"Need Σ_s0 < Σ_t (got {sigma_s0}, {sigma_t})")
    ng = 1
    SigS0 = csr_matrix(np.array([[sigma_s0]], dtype=float))
    SigS1 = csr_matrix(np.array([[sigma_s1]], dtype=float))
    Sig2 = csr_matrix(np.zeros((ng, ng)))
    return Mixture(
        SigC=np.array([sigma_t - sigma_s0]),
        SigL=np.zeros(ng),
        SigF=np.zeros(ng),
        SigP=np.zeros(ng),
        SigT=np.array([sigma_t]),
        SigS=[SigS0, SigS1],   # P0 and P1 scattering matrices
        Sig2=Sig2,
        chi=np.zeros(ng),
        eg=np.array([1e-5, 2e7]),
    )


def build_p1_aniso_mms_case(
    sigma_t: float = 1.0,
    sigma_s0: float = 0.5,
    sigma_s1: float = 0.2,
    alpha: float = 0.1,
    slab_length: float = 5.0,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_p1_aniso",
) -> SNP1AnisoMMSCase:
    r"""Build the canonical P1 anisotropic scattering MMS case.

    Default parameters:

    - :math:`\alpha = 0.1` — weak anisotropy, enough to exercise
      the P1 slot without making the source stiff.
    - :math:`\Sigma_s^{(1)} = 0.2` — moderate forward scattering
      (positive = forward-peaked in the lab frame).
    """
    materials = {mat_id: _make_1g_p1_mixture(sigma_t, sigma_s0, sigma_s1)}
    quadrature = GaussLegendre1D.create(n_ordinates=n_ordinates)
    return SNP1AnisoMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s0=sigma_s0,
        sigma_s1=sigma_s1,
        alpha=alpha,
        slab_length=slab_length,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.3 — 1D Spherical MMS (1-group, GL quadrature)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SNSphericalMMSCase:
    r"""Closed-form MMS fixed-source problem for 1D spherical SN verification.

    Ansatz: :math:`\psi_n(r) = A(r)/W` with :math:`A(r) = \sin(\pi r/R)`.
    Vanishes at :math:`r = 0` and :math:`r = R`.  Isotropic in angle,
    so the angular redistribution term vanishes and the manufactured
    source is :math:`Q_n = \mu_n A'(r) + (\Sigma_t - \Sigma_s) A(r)`.
    """

    name: str
    sigma_t: float
    sigma_s: float
    radius: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: GaussLegendre1D
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-spherical",
        "sn-mms-spherical-psi",
        "sn-mms-spherical-qext",
    )

    def phi_exact(self, r: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * np.asarray(r) / self.radius)

    def dphi_exact(self, r: np.ndarray) -> np.ndarray:
        R = self.radius
        return (np.pi / R) * np.cos(np.pi * np.asarray(r) / R)

    def build_mesh(self, n_cells: int) -> Mesh1D:
        edges = np.linspace(0.0, self.radius, n_cells + 1)
        mat_ids = np.full(n_cells, self.mat_id, dtype=int)
        return Mesh1D(
            edges=edges, mat_ids=mat_ids,
            coord=CoordSystem.SPHERICAL,
            bc_left=BC("reflective"),   # r = 0: symmetry
            bc_right=BC("vacuum"),      # r = R: vacuum
        )

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r = mesh.centers
        A = self.phi_exact(r)
        Ap = self.dphi_exact(r)
        mu = self.quadrature.mu_x
        N = len(mu)
        streaming = mu[:, None] * Ap[None, :]
        removal = (self.sigma_t - self.sigma_s) * A[None, :]
        Q = streaming + removal
        return Q[:, :, None, None]


def build_spherical_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    radius: float = 5.0,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_spherical_sin",
) -> SNSphericalMMSCase:
    r"""Build the canonical 1D spherical MMS case."""
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = GaussLegendre1D.create(n_ordinates=n_ordinates)
    return SNSphericalMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        radius=radius,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.4 — 1D Cylindrical MMS (1-group, Product quadrature)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SNCylindricalMMSCase:
    r"""Closed-form MMS fixed-source problem for 1D cylindrical SN verification.

    Ansatz: :math:`\psi_n(r) = A(r)/W` with :math:`A(r) = \sin(\pi r/R)`.
    Isotropic in angle; the azimuthal redistribution term vanishes.
    The radial direction cosine for cylindrical SN is :math:`\eta_n`,
    accessed as ``quadrature.mu_x[n]``.

    Manufactured source:
    :math:`Q_n = \eta_n A'(r) + (\Sigma_t - \Sigma_s) A(r)`.
    """

    name: str
    sigma_t: float
    sigma_s: float
    radius: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: ProductQuadrature
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cylindrical",
        "sn-mms-cylindrical-psi",
        "sn-mms-cylindrical-qext",
    )

    def phi_exact(self, r: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * np.asarray(r) / self.radius)

    def dphi_exact(self, r: np.ndarray) -> np.ndarray:
        R = self.radius
        return (np.pi / R) * np.cos(np.pi * np.asarray(r) / R)

    def build_mesh(self, n_cells: int) -> Mesh1D:
        edges = np.linspace(0.0, self.radius, n_cells + 1)
        mat_ids = np.full(n_cells, self.mat_id, dtype=int)
        return Mesh1D(
            edges=edges, mat_ids=mat_ids,
            coord=CoordSystem.CYLINDRICAL,
            bc_left=BC("reflective"),   # r = 0: symmetry
            bc_right=BC("vacuum"),      # r = R: vacuum
        )

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r = mesh.centers
        A = self.phi_exact(r)
        Ap = self.dphi_exact(r)
        # mu_x is the radial direction cosine (η) for cylindrical
        eta = self.quadrature.mu_x
        N = len(eta)
        streaming = eta[:, None] * Ap[None, :]
        removal = (self.sigma_t - self.sigma_s) * A[None, :]
        Q = streaming + removal
        return Q[:, :, None, None]


def build_cylindrical_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    radius: float = 5.0,
    n_mu: int = 4,
    n_phi: int = 8,
    mat_id: int = 1,
    name: str = "sn_mms_cylindrical_sin",
) -> SNCylindricalMMSCase:
    r"""Build the canonical 1D cylindrical MMS case."""
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = ProductQuadrature.create(n_mu=n_mu, n_phi=n_phi)
    return SNCylindricalMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        radius=radius,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


# ── Phase-0 ContinuousReferenceSolution wrapper ──────────────────────

def _build_heterogeneous_continuous_reference() -> ContinuousReferenceSolution:
    r"""Produce the Phase-0 :class:`ContinuousReferenceSolution`
    wrapper for the heterogeneous MMS case.

    The reference is a **fixed-source** problem with
    ``is_eigenvalue=False`` and ``k_eff=None``. The reference
    scalar flux is the continuous ansatz
    :math:`\phi_g(x) = c_g\,\sin(\pi x/L)`, which tests can
    evaluate at arbitrary :math:`x` via
    :meth:`~orpheus.derivations.ContinuousReferenceSolution.phi_on_mesh`.

    The external source and the per-cell materials are
    mesh-dependent constructions — a test that wants to actually
    run the solver pulls the concrete
    :class:`SNSlab2GHeterogeneousMMSCase` instance via
    ``problem.geometry_params["mms_case"]`` and uses its
    ``build_mesh`` / ``build_materials`` / ``external_source``
    methods. The ``ProblemSpec.materials`` field is populated
    with an empty dict because the materials are mesh-specific.
    """
    mms_case = build_1d_slab_heterogeneous_mms_case()

    def phi(x: np.ndarray, g: int = 0) -> np.ndarray:
        return mms_case.phi_exact(x, g)

    return ContinuousReferenceSolution(
        name=mms_case.name,
        problem=ProblemSpec(
            materials={},  # per-cell, built on demand by mms_case.build_materials(mesh)
            geometry_type="slab",
            geometry_params={
                "length": mms_case.slab_length,
                "mms_case": mms_case,  # concrete handle for test consumers
            },
            boundary_conditions={"left": "vacuum", "right": "vacuum"},
            external_source=None,  # constructed per-mesh by mms_case.external_source(mesh)
            is_eigenvalue=False,
            n_groups=mms_case.n_groups,
        ),
        operator_form="differential-sn",
        phi=phi,
        provenance=Provenance(
            citation=(
                "Salari & Knupp, SAND2000-1444 §6 (smooth-Σ MMS); "
                "Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals)"
            ),
            derivation_notes=(
                "2-group heterogeneous SN spatial-operator reference "
                "via the Method of Manufactured Solutions with "
                "continuous (smooth) cross-section functions. Ansatz "
                "ψ_{n,g}(x) = (c_g/W) sin(π x/L), giving φ_g(x) = "
                "c_g sin(π x/L) for all ordinates. Manufactured "
                "per-ordinate source Q_ext_{n,g}(x) = μ_n c_g A'(x) "
                "+ Σ_{t,g}(x) c_g A(x) - Σ_{g'} Σ_{s,g'→g}(x) c_{g'} "
                "A(x). Continuous Σ(x) avoids the O(h²) → O(h) "
                "degradation at material interfaces that does not "
                "coincide with cell faces (Salari & Knupp §6); the "
                "design order of diamond difference on a smooth "
                "problem is exactly O(h²). The g=2 source couples "
                "to c_1 through the downscatter term, which "
                "exercises the multigroup scatter assembly in the "
                "sweep — a bug that transposes the scatter matrix "
                "or drops a cross-group source term produces an "
                "incorrect φ_2 that the convergence test catches."
            ),
            sympy_expression=(
                r"Q^{\text{ext}}_{n,g}(x) = \mu_n c_g A'(x) "
                r"+ c_g \Sigma_{t,g}(x) A(x) "
                r"- \sum_{g'} \Sigma_{s,g' \to g}(x) c_{g'} A(x)"
            ),
            precision_digits=None,  # closed-form reference flux
        ),
        k_eff=None,
        psi=None,
        equation_labels=mms_case.equation_labels,
        vv_level="L1",
        description=(
            "2-group heterogeneous SN MMS — smooth Σ(x), vacuum BCs, "
            "downscatter-coupled manufactured source. "
            "Phase-2.1a continuous reference."
        ),
        tolerance="O(h^2)",
    )


def _build_2d_cartesian_continuous_reference() -> ContinuousReferenceSolution:
    r"""Produce the Phase-0 :class:`ContinuousReferenceSolution`
    wrapper for the 2D Cartesian MMS case.

    The reference is a **fixed-source** problem with
    ``is_eigenvalue=False``. The reference scalar flux is
    :math:`\phi(x,y) = \sin(\pi x/L_x)\sin(\pi y/L_y)`.
    """
    mms_case = build_2d_cartesian_mms_case()

    def phi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return mms_case.phi_exact(x, y)

    return ContinuousReferenceSolution(
        name=mms_case.name,
        problem=ProblemSpec(
            materials=mms_case.materials,
            geometry_type="cartesian-2d",
            geometry_params={
                "length_x": mms_case.length_x,
                "length_y": mms_case.length_y,
                "mms_case": mms_case,
            },
            boundary_conditions={
                "left": "vacuum", "right": "vacuum",
                "bottom": "vacuum", "top": "vacuum",
            },
            external_source=None,  # constructed per-mesh
            is_eigenvalue=False,
            n_groups=1,
        ),
        operator_form="differential-sn",
        phi=phi,
        provenance=Provenance(
            citation=(
                "Salari & Knupp, SAND2000-1444 §6 (MMS methodology); "
                "Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals)"
            ),
            derivation_notes=(
                "1-group 2D Cartesian SN spatial-operator reference "
                "via the Method of Manufactured Solutions. Ansatz "
                "ψ_n(x,y) = (1/W) sin(πx/Lx) sin(πy/Ly), giving "
                "φ(x,y) = sin(πx/Lx) sin(πy/Ly). Manufactured source "
                "Q_ext_n = μ_x ∂A/∂x + μ_y ∂A/∂y + (Σ_t − Σ_s) A. "
                "Isotropic-in-angle ansatz eliminates angular "
                "quadrature error; the only remaining error is the "
                "2D diamond-difference spatial truncation at O(h²). "
                "Vacuum BCs are satisfied because the separable "
                "sinusoidal ansatz vanishes on all four edges."
            ),
            sympy_expression=(
                r"Q^{\text{ext}}_n(x,y) = \mu_{x,n}\,\partial_x A "
                r"+ \mu_{y,n}\,\partial_y A "
                r"+ (\Sigma_t - \Sigma_s)\,A(x,y)"
            ),
            precision_digits=None,
        ),
        k_eff=None,
        psi=None,
        equation_labels=mms_case.equation_labels,
        vv_level="L1",
        description=(
            "1-group 2D Cartesian SN MMS — separable sinusoidal ansatz, "
            "vacuum BCs, Lebedev quadrature. Phase-3.1 continuous reference."
        ),
        tolerance="O(h^2)",
    )


def _build_2d_cartesian_2g_continuous_reference() -> ContinuousReferenceSolution:
    r"""Produce the :class:`ContinuousReferenceSolution` wrapper for
    the 2D 2-group heterogeneous MMS case (Phase 3.2)."""
    mms_case = build_2d_cartesian_heterogeneous_mms_case()

    def phi(x: np.ndarray, y: np.ndarray, g: int = 0) -> np.ndarray:
        return mms_case.phi_exact(x, y, g)

    return ContinuousReferenceSolution(
        name=mms_case.name,
        problem=ProblemSpec(
            materials={},  # per-cell, built per-mesh
            geometry_type="cartesian-2d",
            geometry_params={
                "length_x": mms_case.length_x,
                "length_y": mms_case.length_y,
                "mms_case": mms_case,
            },
            boundary_conditions={
                "left": "vacuum", "right": "vacuum",
                "bottom": "vacuum", "top": "vacuum",
            },
            external_source=None,
            is_eigenvalue=False,
            n_groups=mms_case.n_groups,
        ),
        operator_form="differential-sn",
        phi=phi,
        provenance=Provenance(
            citation=(
                "Salari & Knupp, SAND2000-1444 §6 (smooth-Σ MMS); "
                "Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals)"
            ),
            derivation_notes=(
                "2-group heterogeneous 2D Cartesian SN spatial-operator "
                "reference via MMS with smooth Σ(x,y). Ansatz "
                "ψ_{n,g}(x,y) = (c_g/W) sin(πx/Lx) sin(πy/Ly). "
                "Manufactured source has 2D streaming terms "
                "μ_x ∂A/∂x + μ_y ∂A/∂y and position-dependent "
                "removal Σ_{t,g}(x,y) and in-scatter Σ_{s,g'→g}(x,y). "
                "The g=2 source couples to c_1 through downscatter, "
                "exercising the 2D multigroup scatter assembly."
            ),
            sympy_expression=(
                r"Q^{\text{ext}}_{n,g}(x,y) = "
                r"\mu_{x,n} c_g \partial_x A + \mu_{y,n} c_g \partial_y A "
                r"+ c_g \Sigma_{t,g}(x,y) A "
                r"- \sum_{g'} \Sigma_{s,g' \to g}(x,y) c_{g'} A"
            ),
            precision_digits=None,
        ),
        k_eff=None,
        psi=None,
        equation_labels=mms_case.equation_labels,
        vv_level="L1",
        description=(
            "2-group heterogeneous 2D Cartesian SN MMS — smooth Σ(x,y), "
            "vacuum BCs, downscatter-coupled. Phase-3.2 continuous reference."
        ),
        tolerance="O(h^2)",
    )


def _build_spherical_continuous_reference() -> ContinuousReferenceSolution:
    """Phase-3.3 continuous reference for spherical MMS."""
    mms = build_spherical_mms_case()
    return ContinuousReferenceSolution(
        name=mms.name,
        problem=ProblemSpec(
            materials=mms.materials,
            geometry_type="sphere",
            geometry_params={"radius": mms.radius, "mms_case": mms},
            boundary_conditions={"inner": "reflective", "outer": "vacuum"},
            external_source=None, is_eigenvalue=False, n_groups=1,
        ),
        operator_form="differential-sn",
        phi=lambda r: mms.phi_exact(r),
        provenance=Provenance(
            citation="Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals)",
            derivation_notes=(
                "1-group spherical SN MMS with isotropic ansatz "
                "ψ_n(r) = (1/W) sin(πr/R). Angular redistribution "
                "vanishes for isotropic flux; manufactured source "
                "Q_n = μ_n A'(r) + (Σ_t − Σ_s) A(r)."
            ),
            sympy_expression=r"Q_n(r) = \mu_n A'(r) + (\Sigma_t - \Sigma_s) A(r)",
            precision_digits=None,
        ),
        k_eff=None, psi=None,
        equation_labels=mms.equation_labels,
        vv_level="L1",
        description="1-group spherical SN MMS — Phase 3.3 continuous reference.",
        tolerance="O(h^2)",
    )


def _build_cylindrical_continuous_reference() -> ContinuousReferenceSolution:
    """Phase-3.4 continuous reference for cylindrical MMS."""
    mms = build_cylindrical_mms_case()
    return ContinuousReferenceSolution(
        name=mms.name,
        problem=ProblemSpec(
            materials=mms.materials,
            geometry_type="cylinder",
            geometry_params={"radius": mms.radius, "mms_case": mms},
            boundary_conditions={"inner": "reflective", "outer": "vacuum"},
            external_source=None, is_eigenvalue=False, n_groups=1,
        ),
        operator_form="differential-sn",
        phi=lambda r: mms.phi_exact(r),
        provenance=Provenance(
            citation="Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals)",
            derivation_notes=(
                "1-group cylindrical SN MMS with isotropic ansatz "
                "ψ_n(r) = (1/W) sin(πr/R). Azimuthal redistribution "
                "vanishes for isotropic flux; manufactured source "
                "Q_n = η_n A'(r) + (Σ_t − Σ_s) A(r)."
            ),
            sympy_expression=r"Q_n(r) = \eta_n A'(r) + (\Sigma_t - \Sigma_s) A(r)",
            precision_digits=None,
        ),
        k_eff=None, psi=None,
        equation_labels=mms.equation_labels,
        vv_level="L1",
        description="1-group cylindrical SN MMS — Phase 3.4 continuous reference.",
        tolerance="O(h^2)",
    )


def _build_p1_aniso_continuous_reference() -> ContinuousReferenceSolution:
    """Phase-3.5 continuous reference for P1 anisotropic scattering MMS."""
    mms = build_p1_aniso_mms_case()
    return ContinuousReferenceSolution(
        name=mms.name,
        problem=ProblemSpec(
            materials=mms.materials,
            geometry_type="slab",
            geometry_params={"length": mms.slab_length, "mms_case": mms},
            boundary_conditions={"left": "vacuum", "right": "vacuum"},
            external_source=None, is_eigenvalue=False, n_groups=1,
        ),
        operator_form="differential-sn",
        phi=lambda x: mms.phi_exact(x),
        provenance=Provenance(
            citation="Oberkampf & Roy 2010, Ch. 6 (MMS fundamentals)",
            derivation_notes=(
                "1-group P1 anisotropic scattering SN MMS. Ansatz "
                "ψ_n(x) = (1/W)(A(x) + α μ_n B(x)) with A=B=sin(πx/L), "
                "α=0.1. Exercises the P1 scattering slot in "
                "_build_aniso_scattering. Manufactured source includes "
                "the standard streaming + removal terms plus α μ_n "
                "(Σ_t − Σ_s^1) B and α μ_n² B' from the P1 current."
            ),
            sympy_expression=(
                r"Q_n = \mu_n A' + (\Sigma_t - \Sigma_s^0) A "
                r"+ \alpha \mu_n (\Sigma_t - \Sigma_s^1) B "
                r"+ \alpha \mu_n^2 B'"
            ),
            precision_digits=None,
        ),
        k_eff=None, psi=None,
        equation_labels=mms.equation_labels,
        vv_level="L1",
        description="1-group P1 aniso scattering SN MMS — Phase 3.5 reference.",
        tolerance="O(h^2)",
    )


def continuous_cases() -> list[ContinuousReferenceSolution]:
    """Return the Phase-0 continuous references produced by this module."""
    return [
        _build_heterogeneous_continuous_reference(),
        _build_2d_cartesian_continuous_reference(),
        _build_2d_cartesian_2g_continuous_reference(),
        _build_spherical_continuous_reference(),
        _build_cylindrical_continuous_reference(),
        _build_p1_aniso_continuous_reference(),
    ]
