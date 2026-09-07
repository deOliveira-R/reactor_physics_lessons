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

   - :doc:`/theory/verification/sn` — MMS verification section
     with the full derivation and convergence-rate argument.
   - :func:`orpheus.sn.solve_sn_fixed_source` — consumer of the
     external source produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.geometry import Mesh1D, Mesh2D
from orpheus.geometry.coord import CoordSystem
from orpheus.geometry.mesh import BC
from orpheus.numerics.quadrature import Quadrature

from ...common.continuous_reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)

if TYPE_CHECKING:
    # ``sympy`` is imported lazily inside each symbolic builder (keeps the
    # heavy symbolic dependency off the module-load path); this type-only
    # alias makes the ``"sp.Expr"`` signature annotations resolvable without
    # eagerly importing it. ``TimedFullField`` is likewise a return-type-only
    # reference (constructed via its own lazy import at the call site).
    import sympy as sp

    from orpheus.transport.timed_full_field import TimedFullField


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
    quadrature : Quadrature
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
    quadrature: Quadrature
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
        r"""Per-ordinate-density external source :math:`Q^{\text{ext}}_n` on ``mesh``.

        Evaluated at cell centres to match the diamond-difference
        cell-average convention. Returned shape is
        ``(N, ng=1, nx)`` — per ordinate, one energy group, per cell
        (the principled (N, ng, *spatial) layout; no phantom ny axis).

        R-1 Step 4 A1 — returns **per-ordinate density** (already
        projected via ``/sum_w``).  The continuous derivation gives
        :math:`Q_n = \mu_n A'(x) + (\Sigma_t - \Sigma_s) A(x)` for an
        isotropic ansatz :math:`\psi_n = A(x)/W`; under the per-ord
        contract this is divided by :math:`\sum_n w_n` at the producer
        boundary (Pattern 7 of ``coding-elegance``).
        """
        x = mesh.centers                          # (nx,)
        A = self.phi_exact(x)                     # (nx,)
        Ap = self.dphi_exact(x)                   # (nx,)
        mu = self.quadrature.mu_x                 # (N,)
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu)
        nx = len(x)

        streaming = mu[:, None] * Ap[None, :]     # (N, nx)
        removal = (self.sigma_t - self.sigma_s) * A[None, :]  # (1, nx)
        Q = (streaming + removal) / sum_w         # (N, nx) per-ord density
        return Q[:, None, :]                       # (N, ng=1, nx)


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
    # Synthetic 1G mixture: no physical energy grid (Phase E).
    return Mixture(
        SigC=np.array([sigma_t - sigma_s]),
        SigL=np.zeros(ng),
        SigF=np.zeros(ng),
        SigP=np.zeros(ng),
        SigT=np.array([sigma_t]),
        SigS=[SigS0],
        Sig2=[Sig2],
        chi=np.zeros(ng),
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
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
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
    - ``docs/theory/verification/sn.rst`` — the heterogeneous
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
    quadrature : Quadrature
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
    quadrature: Quadrature
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
            # Synthetic MMS mixture: no physical energy grid (Phase E).
            materials[i] = Mixture(
                SigC=sig_a,                        # pure absorber capture
                SigL=np.zeros(2),                  # no (n,α)
                SigF=np.zeros(2),                  # no fission
                SigP=np.zeros(2),                  # no production
                SigT=sig_t,
                SigS=[csr_matrix(sig_s_row)],      # P0 only
                Sig2=[csr_matrix(np.zeros((2, 2)))],  # no (n,2n)
                chi=np.zeros(2),
            )
        return materials

    # ── Manufactured source on the mesh ──────────────────────────────

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r"""Per-ordinate, per-cell, per-group external source.

        Shape ``(N_ord, n_groups, n_cells)`` — the principled
        (N, ng, *spatial) layout (no phantom ny axis).  The formula is

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
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu)
        nx = len(x)
        ng = self.n_groups

        Q = np.zeros((N, ng, nx))
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
            Q[:, g, :] = streaming + (removal - in_scatter)[None, :]
        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q /= sum_w
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
    quad = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
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

    - :doc:`/theory/verification/sn` — 2D Cartesian MMS section.
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
    quadrature : Quadrature
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
    quadrature: Quadrature
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

        Returns shape ``(N, 1, nx, ny)`` (Issue #196 PR-INDEX-5 —
        principled).  Per ordinate, one energy group, per cell (x, y).
        Evaluated at cell centres.
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
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu_x)

        # streaming: mu_x * dA/dx + mu_y * dA/dy   → (N, nx, ny)
        streaming = (mu_x[:, None, None] * dA_dx[None, :, :]
                     + mu_y[:, None, None] * dA_dy[None, :, :])
        removal = (self.sigma_t - self.sigma_s) * A   # (nx, ny)
        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q = (streaming + removal[None, :, :]) / sum_w  # (N, nx, ny)
        return Q[:, None, :, :]                        # (N, 1, nx, ny)


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
    quadrature = Quadrature.lebedev(order=lebedev_order)
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
    quadrature: Quadrature
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
        return _build_per_cell_hetero_materials(
            mesh, self.sigma_t_fn, self.sigma_s_fn, self.n_groups,
        )

    # ── Manufactured source ──────────────────────────────────────────

    def external_source(self, mesh: Mesh2D) -> np.ndarray:
        r"""Per-ordinate, per-cell, per-group external source.

        Shape ``(N_ord, n_groups, nx, ny)`` (Issue #196 PR-INDEX-5 —
        principled).
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
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu_x)
        nx, ny_ = len(cx), len(cy)

        # Evaluate cross sections on the 2D grid
        xx, yy = np.meshgrid(cx, cy, indexing="ij")  # (nx, ny)
        xx_flat = xx.ravel()
        yy_flat = yy.ravel()

        Q = np.zeros((N, ng, nx, ny_))
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
            Q[:, g, :, :] = streaming + (removal - in_scatter)[None, :, :]
        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q /= sum_w
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


def _build_per_cell_hetero_materials(
    mesh: Mesh2D,
    sigma_t_fn: "Callable[[np.ndarray, np.ndarray, int], np.ndarray]",
    sigma_s_fn: "Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]",
    n_groups: int,
) -> dict[int, Mixture]:
    r"""Per-cell :class:`Mixture` map, sampling :math:`\Sigma(x,y)` at cell
    centres — one :class:`Mixture` per cell, with
    :math:`\Sigma_a = \Sigma_t - \sum_{g'}\Sigma_s[g, g']`.

    Shared by every 2-D Cartesian heterogeneous MMS case (the DD
    :class:`SN2DCartesian2GHeterogeneousMMSCase` and the LD
    :class:`SN2DCartesianLDStressMMSCase`): the per-cell material *assembly* is
    common mechanism (a synthetic MMS mixture — no fission, no physical energy
    grid), while the MMS independence lives in each case's ``external_source`` /
    ``phi_exact``.  Single-sourced here so a future :class:`Mixture` field change
    cannot silently diverge the two references (Cardinal Rule 2)."""
    cx = mesh.centers_x
    cy = mesh.centers_y
    materials: dict[int, Mixture] = {}
    for i, xi in enumerate(cx):
        for j, yj in enumerate(cy):
            cell_id = mesh.mat_map[i, j]
            xa = np.array([xi])
            ya = np.array([yj])
            sig_t = np.array([
                float(sigma_t_fn(xa, ya, g)[0]) for g in range(n_groups)
            ])
            sig_s_row = np.zeros((n_groups, n_groups))
            for g_from in range(n_groups):
                for g_to in range(n_groups):
                    sig_s_row[g_from, g_to] = float(
                        sigma_s_fn(xa, ya, g_from, g_to)[0]
                    )
            sig_a = sig_t - sig_s_row.sum(axis=1)
            materials[cell_id] = Mixture(
                SigC=sig_a,
                SigL=np.zeros(n_groups),
                SigF=np.zeros(n_groups),
                SigP=np.zeros(n_groups),
                SigT=sig_t,
                SigS=[csr_matrix(sig_s_row)],
                Sig2=[csr_matrix(np.zeros((n_groups, n_groups)))],
                chi=np.zeros(n_groups),
            )
    return materials


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
    quad = Quadrature.lebedev(order=lebedev_order)
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
# #240 D5b-S4 — 2-D Cartesian Linear-Discontinuous STRESS MMS
# (the multi-D bilinear UBLD slope-row verification — vv Mode-7 override)
# ═══════════════════════════════════════════════════════════════════════
r"""
2-D Cartesian LD stress MMS — activates the bilinear per-axis SPATIAL slope rows.

**Why this case exists (vv Mode-7 override).** The existing 2-D MMS
(:class:`SN2DCartesianMMSCase`, :class:`SN2DCartesian2GHeterogeneousMMSCase`) is
**isotropic-in-μ** (:math:`\psi = A(x,y)/W`, no :math:`\mu` dependence).  That
ansatz NULLS the per-axis slope rows of the bilinear (UBLD) Linear-Discontinuous
closure — the very machinery #240 D5b introduces (the :math:`2^d`-moment cell
:math:`\{\bar\psi, \hat\psi_x, \hat\psi_y, \hat\psi_{xy}\}`).  An isotropic
ansatz tests NOTHING about the multi-D slope coupling; the LD smoke gate
(``test_ld_2d_converges_second_order_smoke``) is a CONVERGENCE check, not a
flux-shape verification of the bilinear slope.  This stress ansatz is the
flux-shape verification (the ``@verifies("ld-cartesian-2d")`` claim).

**The trial solution** (per group :math:`g`):

.. math::

   \psi_{n,g}(x,y) = \frac{1}{W}\bigl[\,A_g(x,y)
       + \mu_{x,n}\,B_g(x,y) + \mu_{y,n}\,C_g(x,y)\,\bigr]

with the per-group spatial drivers (the strengthening = the
:math:`b_2,\,c_2` cross-harmonic terms in the SLOPE drivers, chosen so
:math:`B` and :math:`C` are NOT x↔y reflections of each other):

.. math::

   A_g &= a_{0,g} + a_{1,g}\sin\!\tfrac{\pi x}{L_x}\sin\!\tfrac{\pi y}{L_y}
        + a_{2,g}\cos\!\tfrac{2\pi x}{L_x}\cos\!\tfrac{3\pi y}{L_y}\\
   B_g &= b_{0,g} + b_{1,g}\sin\!\tfrac{2\pi x}{L_x}\sin\!\tfrac{\pi y}{L_y}
        + b_{2,g}\cos\!\tfrac{\pi x}{L_x}\cos\!\tfrac{2\pi y}{L_y}\\
   C_g &= c_{0,g} + c_{1,g}\sin\!\tfrac{\pi x}{L_x}\sin\!\tfrac{2\pi y}{L_y}
        + c_{2,g}\sin\!\tfrac{3\pi x}{L_x}\cos\!\tfrac{\pi y}{L_y}

**Activated / nulled-term declaration (vv Mode 7 — MANDATORY):**

============================  =====================================================
Term                          Activated by
============================  =====================================================
x-axis LD slope (UNKNOWN)     :math:`\mu_x B_g` with :math:`\partial_x B \neq 0`:
                              the bilinear x-slope row solves a genuinely
                              x-varying, :math:`\mu_x`-weighted field — the NEW
                              multi-D coupling DD cannot represent.
y-axis LD slope (UNKNOWN)     :math:`\mu_y C_g` with :math:`\partial_y C \neq 0`,
                              INDEPENDENT of x (the bilinearity: the two slopes
                              do not collapse).
cross-axis x↔y discrimination the :math:`b_2,c_2` cross-harmonics break the x↔y
                              reflection so a same-sign slope-row SIGN bug (both
                              rows share ``_LDCellTerms.slope``) CANNOT cancel in
                              the measured flux (the LM-1989 same-sign trap).
boundary closure (AVERAGE)    :math:`a_{0,g}>0` → :math:`\psi \neq 0` on all four
                              edges → the prescribed-inflow trace is exercised at
                              the FACE-AVERAGE moment (the production widens the
                              scalar trace onto the average moment).
group coupling                per-group ``a/b/c`` + 2G asymmetric downscatter
                              (mode #6 convention drift).
============================  =====================================================

**NULLED — the slope-SOURCE half of the LM-1989 trap, and the transverse
face-slope inflow moment** (see "Honest scope" below).

**The scalar flux** is :math:`\phi_g = \int\psi\,d\mu = A_g(x,y)` — the
:math:`\mu_x B + \mu_y C` terms integrate to zero over any symmetric quadrature
(:math:`\langle\mu_x\rangle=\langle\mu_y\rangle=0`).  The VALUE band checks the
converged scalar flux against :math:`A_g`; the per-ordinate streaming derivative
carries the full :math:`\mu`-weighted ansatz into the manufactured source.

**Quadrature exactness.** Streaming a :math:`\mu`-linear :math:`\psi` produces
:math:`\mu_x^2,\,\mu_x\mu_y,\,\mu_y^2` source terms.  The scalar-flux identity
:math:`\phi=A` needs :math:`\langle\mu_x\rangle=\langle\mu_y\rangle=
\langle\mu_x\mu_y\rangle=0` and :math:`\langle\mu_x^2\rangle=\langle\mu_y^2
\rangle=W/3` — exact for level-symmetric S4+ and Lebedev order≥5 on the full
sphere (probed; ``test_ld_2d.py`` selects ``level_symmetric(4)`` for the
headline gate, no pure-z ordinate; ``lebedev`` for the matvec-twin gate).

**Honest scope (cross-domain-attacker Frame 2 §224–236; the LM-1989 trap).**
The trap has TWO halves:

1. the slope-UNKNOWN sign (always exercised when the slope is non-trivially
   solved — VERIFIED here: :math:`B,\,C` drive non-trivial per-ordinate fields,
   the bilinear closure solves :math:`\hat\psi_x,\,\hat\psi_y` from the average +
   the scattering source); and
2. the slope-SOURCE sign :math:`\hat Q` (exercised ONLY when a non-zero
   slope-moment EXTERNAL source is supplied AND consumed).

The slope-SOURCE sign half (the EXTERNAL :math:`\hat Q`) is now VERIFIED
(**Leg A, #247**).  The public ``solve_sn_fixed_source`` entry accepts a typed
union of TWO bulk ranks — flat ``(N, ng, *spatial)`` (the slope rows
:math:`\hat Q` zeroed, the honest default) OR moment-resolved
``(N, ng, *spatial, per_axis**ndim)`` (the projected slope rows threaded
through) — and ``_lift_external_source_to_moments`` threads the moment-resolved
slope rows into the SI rhs alongside the iterate-driven SCATTERING source
:math:`\Sigma_s\cdot\hat\phi`.  The slope-SOURCE sign is pinned by the
per-moment structural gate + the mutation controls (M1–M4) in
``tests/sn/verification/mms/test_mms_ld_2d.py`` (the #247 block): flipping a
CONSUMED slope-source row changes the converged flux ≫ the inner tolerance,
while the FLAT scalar gate stays GREEN (the Mode-10 asymmetry that closed the
gap).  Per vv-principles Mode 10: the converged flux is only sub-floor sensitive
to the slope-source sign (an O(h²)-small forcing), so the teeth are STRUCTURAL
(the lift threads the projection through unchanged at machine precision; the
consumed flip moves the answer O(1) above tol), NOT a converged-value band.

The BOUNDARY transverse-face-slope (**Leg B**, **#251**) is now CARRIED end to
end: the boundary trace ``mesh.angular_trace`` carries the :math:`2^{d-1}`
transverse-moment axis (LD), ``_inflow_to_moments`` threads the projected
transverse face-slope onto slot-1, and — since **#257 S9** — this case's
:meth:`~SN2DCartesianLDStressMMSCase.prescribed_inflow` itself EMITS the
moment-resolved slot (``_project_inflow_to_face_moments`` projects the
manufactured inflow trace onto the bare transverse Legendre moments
``[bar, slope]``; DD/Step keeps the byte-identical scalar trace).

**S9 verdict (vv Mode-10, the boundary sub-case where the companion-isolating
value gate is UNAVAILABLE):** the transverse face-slope is GENUINELY consumed (a
flip moves the near-boundary flux ≫ tol) and threaded at machine precision, but
its converged-flux contribution is SUB-FLOOR — the cell-AVERAGE moment already
delivers O(h²) AT the boundary (the coherent promise: LD is 2nd-order everywhere
incl. the boundary, no asterisk).  The slope is an inflow-representation
refinement (O(h)→O(h²) on the face trace), NOT a deficiency repair; no
value/order gate is keyed on it (it would falsely RED a correct term).  Locked by
``tests/sn/verification/mms/test_ld_2d_boundary_promise.py`` (the coherent-promise
gate + the sub-floor verdict pins + the Mode-11 toggle sentinel).

CONSEQUENCE: this MMS now verifies the multi-D slope-UNKNOWN sign + the
slope-SOURCE sign (Leg A, #247) + the AVERAGE-moment boundary closure + the
transverse face-slope threading/consumption (Leg B, #251) + the boundary
coherent promise (S9, #257).  Per Frame 2 §232: the LM-1989 trap's bulk half is
closed by Leg A; the boundary half by Leg B + S9.  The reflective-BC
transverse-slope SIGN remains a vacuum-BC-blind follow-up (#252).

.. seealso::

   - ``.claude/skills/vv-principles/SKILL.md`` — Mode 7 (MMS simplification bias).
   - :class:`SN2DCartesianLDStressMMSCase` (Branch-2 numerical factory).
   - :func:`derive_2d_cartesian_ld_stress_mms` (Branch-1 SymPy gate).
"""


#: Spatial harmonic amplitudes of the LD stress drivers as ``(numerator,
#: denominator)`` pairs — the SINGLE source shared by Branch 1 (SymPy, via
#: :func:`sympy.Rational`) and Branch 2 (numpy, via the float ratio).  Rows are
#: ``(A_coeffs, B_coeffs, C_coeffs)``, each ``((n0,d0), (n1,d1), (n2,d2))``
#: for the constant / first-harmonic / cross-harmonic amplitudes.  The ``b2,
#: c2`` cross-harmonics (and the differing harmonic pairs) break the x↔y
#: reflection so a same-sign slope-row sign bug cannot cancel (Frame 2).
_LD2D_STRESS_COEFFS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((7, 10), (1, 2), (3, 10)),     # A: a0 (>0, non-vanishing edges), a1, a2
    ((2, 5), (1, 4), (1, 5)),       # B: b0, b1 (∂_x ≠ 0), b2 cross-harmonic
    ((3, 10), (7, 20), (3, 20)),    # C: c0, c1 (∂_y ≠ 0), c2 cross-harmonic
)


def _ld2d_stress_amplitudes() -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    r"""The Branch-2 float amplitude triples ``(A_amp, B_amp, C_amp)``.

    Reads the single-sourced :data:`_LD2D_STRESS_COEFFS` (the SAME pairs Branch 1
    feeds to :func:`sympy.Rational`) as exact floats, so the numpy evaluator and
    the SymPy algebra-of-record cannot drift on the amplitudes."""
    return tuple(
        np.array([n / d for (n, d) in row], dtype=float)
        for row in _LD2D_STRESS_COEFFS
    )


def _2d_cartesian_ld_stress_symbolic() -> "tuple":
    r"""Build the symbolic objects for the 2-D Cartesian LD stress MMS.

    Returns ``(x, y, mu_x, mu_y, Lx, Ly, Sigma_t, Sigma_s, W, A, B, C, psi,
    phi, Q_closed)``: the strengthened :math:`\mu`-bilinear ansatz and the
    closed-form per-ordinate source.  The drivers carry concrete (Rational)
    spatial harmonic coefficients so the substitution residual SIMPLIFIES to
    zero cleanly; the per-group amplitude :math:`c_g` and the cross sections
    enter the Branch-2 evaluator (this single-group symbolic layer pins the
    spatial + angular ALGEBRA — the group axis scales mechanically, the
    algebra-of-record minimal-SymPy + scaling-argument).

    The continuous 2-D within-group transport PDE (Cartesian — NO angular
    redistribution):

    .. math::

       \mu_x\,\partial_x\psi + \mu_y\,\partial_y\psi + \Sigma_t\,\psi
       = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\rm ext}\bigr),
       \qquad \phi = A(x,y).

    Solving for :math:`Q^{\rm ext}` (the scalar flux :math:`\phi=A` because
    :math:`\langle\mu_x\rangle=\langle\mu_y\rangle=0`) gives the closed form

    .. math::

       Q^{\rm ext}_n &= \mu_x\,\partial_x A + \mu_y\,\partial_y A
                         \quad(\text{streaming of the average})\\
            &+ \mu_x^2\,\partial_x B + \mu_x\mu_y\,\partial_y B
                         \quad(\text{streaming of the }\mu_x B\text{ slope})\\
            &+ \mu_x\mu_y\,\partial_x C + \mu_y^2\,\partial_y C
                         \quad(\text{streaming of the }\mu_y C\text{ slope})\\
            &+ (\Sigma_t-\Sigma_s)\,A
                         \quad(\text{removal} - \text{in-scatter, isotropic})\\
            &+ \Sigma_t\,(\mu_x B + \mu_y C)
                         \quad(\text{removal of the anisotropic part}).

    The :math:`\mu_x\mu_y` cross terms (from streaming :math:`B` along y and
    :math:`C` along x) are the genuinely-bilinear pieces — they exercise the
    cross moment :math:`\hat\psi_{xy}` the simplex-P1 closure lacks.
    """
    import sympy as sp  # local import: keep the symbolic dependency lazy

    x, y, mu_x, mu_y = sp.symbols("x y mu_x mu_y", real=True)
    Lx, Ly = sp.symbols("L_x L_y", positive=True, real=True)
    Sigma_t, Sigma_s, W = sp.symbols(
        "Sigma_t Sigma_s W", positive=True, real=True,
    )

    # Strengthened drivers (concrete spatial harmonics; Rational amplitudes so
    # simplify() closes cleanly).  A carries a mixed cross-harmonic
    # (cos2x·cos3y); the SLOPE drivers B, C ALSO carry cross-harmonics
    # (b2·cosx·cos2y, c2·sin3x·cosy) — chosen so B and C are NOT x↔y reflections
    # (the same-sign slope-row trap defence, Frame 2 §256–268).  The amplitudes
    # are the single-sourced :data:`_LD2D_STRESS_COEFFS` the Branch-2 numpy
    # evaluator reads too, so Branch 1 and Branch 2 descend from the SAME spatial
    # algebra (the L1 cross-check pins the two evaluators agree, NOT just the
    # symbolic identity).
    px, py = sp.pi / Lx, sp.pi / Ly
    (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = (
        tuple(sp.Rational(n, d) for (n, d) in row) for row in _LD2D_STRESS_COEFFS
    )
    A = a0 + a1 * sp.sin(px * x) * sp.sin(py * y) \
        + a2 * sp.cos(2 * px * x) * sp.cos(3 * py * y)
    B = b0 + b1 * sp.sin(2 * px * x) * sp.sin(py * y) \
        + b2 * sp.cos(px * x) * sp.cos(2 * py * y)
    C = c0 + c1 * sp.sin(px * x) * sp.sin(2 * py * y) \
        + c2 * sp.sin(3 * px * x) * sp.cos(py * y)

    psi = (A + mu_x * B + mu_y * C) / W
    phi = A  # <mu_x> = <mu_y> = 0 over a symmetric quadrature

    Q_closed = (
        mu_x * sp.diff(A, x) + mu_y * sp.diff(A, y)
        + mu_x**2 * sp.diff(B, x) + mu_x * mu_y * sp.diff(B, y)
        + mu_x * mu_y * sp.diff(C, x) + mu_y**2 * sp.diff(C, y)
        + (Sigma_t - Sigma_s) * A
        + Sigma_t * (mu_x * B + mu_y * C)
    )
    return (x, y, mu_x, mu_y, Lx, Ly, Sigma_t, Sigma_s, W, A, B, C,
            psi, phi, Q_closed)


def derive_2d_cartesian_ld_stress_mms() -> dict:
    r"""V_ld2d-stress — 2-D Cartesian LD stress-MMS source identity (Branch 1).

    Proves: substituting the strengthened :math:`\mu`-bilinear ansatz
    :math:`\psi_n = (A + \mu_x B + \mu_y C)/W` into the continuous 2-D
    Cartesian SN operator (NO angular redistribution)

    .. math::

       \mu_x\,\partial_x\psi + \mu_y\,\partial_y\psi + \Sigma_t\,\psi
       = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\rm ext}\bigr)

    yields the closed-form :math:`Q^{\rm ext}_n` (the substitution residual
    vanishes under :func:`sympy.simplify`).  This is the load-bearing
    algebra-of-record claim (L11): the source is the unique manufactured RHS
    consistent with the ansatz, derived from the CONTINUOUS PDE — it touches
    NONE of the LD cell-update code (not ``_LDCellTerms``, not ``_schur_terms``,
    not ``_ubld``), so a sign bug in the LD slope rows cannot also corrupt the
    reference (the defining property of a structurally-independent MMS).
    """
    import sympy as sp

    (x, y, mu_x, mu_y, Lx, Ly, Sigma_t, Sigma_s, W,
     A, B, C, psi, phi_, Q_closed) = _2d_cartesian_ld_stress_symbolic()

    LHS = mu_x * sp.diff(psi, x) + mu_y * sp.diff(psi, y) + Sigma_t * psi
    Q_subst = sp.simplify(W * LHS - Sigma_s * phi_)
    diff = sp.simplify(Q_subst - Q_closed)

    return {
        "name": "V_ld2d-stress: 2-D Cartesian LD stress MMS source identity",
        "psi": psi,
        "phi": phi_,
        "Q_substituted": Q_subst,
        "Q_closed": Q_closed,
        "diff": diff,
        "pass": diff == 0,
    }


@dataclass(frozen=True)
class SN2DCartesianLDStressMMSCase:
    r"""2-D Cartesian Linear-Discontinuous STRESS MMS (#240 D5b-S4).

    The vv-Mode-7 strengthened ansatz that activates the bilinear (UBLD) LD
    per-axis SPATIAL slope rows — the multi-D coupling #240 D5b introduced.
    Per group :math:`g`:

    .. math::

       \psi_{n,g}(x,y) = \frac{1}{W}\bigl[A_g(x,y)
           + \mu_{x,n} B_g(x,y) + \mu_{y,n} C_g(x,y)\bigr],

    with the shared spatial drivers (single-sourced
    :data:`_LD2D_STRESS_COEFFS`) scaled by a per-group amplitude :math:`c_g`.
    See the module docstring above for the full driver definitions, the
    activated/nulled-term declaration, and the HONEST SCOPE.  The slope-SOURCE
    half of the LM-1989 trap (the EXTERNAL :math:`\hat Q`) is now VERIFIED
    (**Leg A, #247**): the public entry accepts a moment-resolved external
    source and the lift threads the projected slope rows through (pinned by the
    per-moment structural gate + mutation controls in
    ``tests/sn/verification/mms/test_mms_ld_2d.py``).  The BOUNDARY
    transverse-face-slope (**Leg B, #251**) is now CARRIED end to end (the trace
    is moment-resolved, the cochain consumes slot-1) and — since **#257 S9** —
    :meth:`prescribed_inflow` EMITS the projected transverse face-slope (its
    converged-flux contribution is SUB-FLOOR; the AVERAGE moment delivers O(h²)
    at the boundary — the coherent promise, locked by
    ``test_ld_2d_boundary_promise.py``).

    Cross sections are **callables** :math:`\Sigma(x,y,g)` evaluated at cell
    centres (one :class:`Mixture` per cell); reuses
    :func:`_default_hetero_2d_xs_functions` (heterogeneous, :math:`\Sigma_a>0`,
    2G asymmetric downscatter).  The domain is NON-SQUARE (:math:`L_x \neq
    L_y`) — the x↔y-swap defence.  The mesh BCs are VACUUM; the non-vanishing
    inflow trace is injected via :meth:`prescribed_inflow` (the ``q.boundary``
    slot), NOT a mesh BC — exactly the :class:`SNSlabNonVacuumMMSCase` posture
    lifted to 2-D.
    """

    name: str
    length_x: float
    length_y: float
    c_spectrum: np.ndarray
    sigma_t_fn: "Callable[[np.ndarray, np.ndarray, int], np.ndarray]"
    sigma_s_fn: "Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]"
    quadrature: Quadrature
    n_groups: int = 2
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        # ``ld-cartesian-2d`` is a label D6 (archivist) minted (now homed
        # in docs/theory/verification/sn.rst) — carried here (the
        # verifies edge is written by Nexus; the label block is the
        # archivist's stub).
        "ld-cartesian-2d",
        "transport-cartesian-2d",
        "multigroup",
        "mg-balance",
    )

    # ── Spatial drivers (Branch 2 numpy; the SAME harmonics as Branch 1) ──

    def _drivers(
        self, x: np.ndarray, y: np.ndarray, g: int,
    ) -> "tuple[np.ndarray, ...]":
        r"""Return ``(A, dA_dx, dA_dy, B, dB_dx, dB_dy, C, dC_dx, dC_dy)`` on the
        2-D grid ``(len(x), len(y))``, scaled by the per-group amplitude
        :math:`c_g`.  Reads the single-sourced amplitudes
        (:func:`_ld2d_stress_amplitudes`), so the numpy drivers cannot drift
        from the SymPy algebra-of-record."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        Lx, Ly = self.length_x, self.length_y
        px, py = np.pi / Lx, np.pi / Ly
        (a, b, c) = _ld2d_stress_amplitudes()
        cg = float(self.c_spectrum[g])

        sx, cx = np.sin(px * x), np.cos(px * x)
        s2x, c2x = np.sin(2 * px * x), np.cos(2 * px * x)
        s3x, c3x = np.sin(3 * px * x), np.cos(3 * px * x)
        sy, cy = np.sin(py * y), np.cos(py * y)
        s2y, c2y = np.sin(2 * py * y), np.cos(2 * py * y)
        s3y, c3y = np.sin(3 * py * y), np.cos(3 * py * y)

        def outer(fx, fy):
            return fx[:, None] * fy[None, :]

        # A = a0 + a1 sin(px x) sin(py y) + a2 cos(2px x) cos(3py y)
        A = cg * (a[0] + a[1] * outer(sx, sy) + a[2] * outer(c2x, c3y))
        dA_dx = cg * (a[1] * px * outer(cx, sy)
                      - a[2] * 2 * px * outer(s2x, c3y))
        dA_dy = cg * (a[1] * py * outer(sx, cy)
                      - a[2] * 3 * py * outer(c2x, s3y))
        # B = b0 + b1 sin(2px x) sin(py y) + b2 cos(px x) cos(2py y)
        B = cg * (b[0] + b[1] * outer(s2x, sy) + b[2] * outer(cx, c2y))
        dB_dx = cg * (b[1] * 2 * px * outer(c2x, sy)
                      - b[2] * px * outer(sx, c2y))
        dB_dy = cg * (b[1] * py * outer(s2x, cy)
                      - b[2] * 2 * py * outer(cx, s2y))
        # C = c0 + c1 sin(px x) sin(2py y) + c2 sin(3px x) cos(py y)
        C = cg * (c[0] + c[1] * outer(sx, s2y) + c[2] * outer(s3x, cy))
        dC_dx = cg * (c[1] * px * outer(cx, s2y)
                      + c[2] * 3 * px * outer(c3x, cy))
        dC_dy = cg * (c[1] * 2 * py * outer(sx, c2y)
                      - c[2] * py * outer(s3x, sy))
        return A, dA_dx, dA_dy, B, dB_dx, dB_dy, C, dC_dx, dC_dy

    def phi_exact(
        self, x: np.ndarray, y: np.ndarray, g: int = 0,
    ) -> np.ndarray:
        r"""Reference scalar flux :math:`\phi_g(x,y) = A_g(x,y)` (the
        :math:`\mu_x B + \mu_y C` terms integrate to zero).  Shape
        ``(len(x), len(y))``."""
        return self._drivers(x, y, g)[0]

    # ── Mesh + materials construction (mirror the 2G het 2-D case) ────────

    def build_mesh(self, nx: int, ny: int | None = None) -> Mesh2D:
        """VACUUM-BC 2-D mesh with a unique material ID per cell.

        Non-square by default (``ny = max(1, round(nx · Ly/Lx))`` when ``ny``
        is None) — the x↔y-swap defence carries into the refinement ladder."""
        if ny is None:
            ny = max(1, round(nx * self.length_y / self.length_x))
        edges_x = np.linspace(0.0, self.length_x, nx + 1)
        edges_y = np.linspace(0.0, self.length_y, ny + 1)
        mat_map = np.arange(nx * ny, dtype=int).reshape(nx, ny)
        return Mesh2D(
            edges_x=edges_x, edges_y=edges_y, mat_map=mat_map,
            coord=CoordSystem.CARTESIAN,
            bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
            bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
        )

    def build_materials(self, mesh: Mesh2D) -> dict[int, Mixture]:
        """Per-cell materials sampling :math:`\\Sigma(x,y)` at cell centres
        (the shared 2-D heterogeneous-MMS material builder)."""
        return _build_per_cell_hetero_materials(
            mesh, self.sigma_t_fn, self.sigma_s_fn, self.n_groups,
        )

    # ── Manufactured source (Branch 2 numpy — the lambdified Branch 1) ────

    def external_source(self, mesh: Mesh2D) -> np.ndarray:
        r"""Per-ordinate, per-cell, per-group external source — shape
        ``(N_ord, ng, nx, ny)`` (the FLAT per-ordinate density; the widened
        public solve lifts this onto slot-0 with zero slope rows, the honest
        flat default).  The moment-resolved bulk slope source (Leg A, #247) is
        the test-side projection of this density fed to the widened lift — see
        the module docstring HONEST SCOPE.

        The manufactured source is the PDE residual

        .. math::

           Q^{\rm ext}_{n,g} = \mu_x\,\partial_x A_g + \mu_y\,\partial_y A_g
               + \mu_x^2\,\partial_x B_g + \mu_x\mu_y\,\partial_y B_g
               + \mu_x\mu_y\,\partial_x C_g + \mu_y^2\,\partial_y C_g
               + \Sigma_{t,g}\,A_g + \Sigma_{t,g}\,(\mu_x B_g + \mu_y C_g)
               - \sum_{g'}\Sigma_s[g', g]\,A_{g'},

        emitted as a per-ordinate density (divided by :math:`\sum_n w_n` at the
        producer boundary, Pattern 7 — the solver multiplies by the cell
        volume).  Bit-equal to the lambdified Branch-1 SymPy closed form
        (:func:`derive_2d_cartesian_ld_stress_mms`) on a sample cell.
        """
        cx = mesh.centers_x
        cy = mesh.centers_y
        ng = self.n_groups
        nx, ny_ = len(cx), len(cy)

        mu_x = self.quadrature.mu_x
        mu_y = self.quadrature.mu_y
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu_x)

        xx, yy = np.meshgrid(cx, cy, indexing="ij")   # (nx, ny)
        xx_flat, yy_flat = xx.ravel(), yy.ravel()

        # Per-group drivers + their derivatives on the (nx, ny) grid.
        drivers = [self._drivers(cx, cy, g) for g in range(ng)]

        Q = np.zeros((N, ng, nx, ny_))
        mxx = mu_x[:, None, None] ** 2
        myy = mu_y[:, None, None] ** 2
        mxy = (mu_x * mu_y)[:, None, None]
        mx = mu_x[:, None, None]
        my = mu_y[:, None, None]
        for g in range(ng):
            (A, dA_dx, dA_dy, B, dB_dx, dB_dy,
             C, dC_dx, dC_dy) = drivers[g]
            sig_t_g = self.sigma_t_fn(xx_flat, yy_flat, g).reshape(nx, ny_)

            # Streaming μ_x ∂_x ψ + μ_y ∂_y ψ, per ordinate.
            streaming = (
                mx * dA_dx[None] + my * dA_dy[None]
                + mxx * dB_dx[None] + mxy * dB_dy[None]
                + mxy * dC_dx[None] + myy * dC_dy[None]
            )
            # Removal Σ_t ψ = Σ_t (A + μ_x B + μ_y C), per ordinate.
            removal = sig_t_g[None] * (
                A[None] + mx * B[None] + my * C[None]
            )
            # In-scatter Σ_{g'} Σ_s[g', g] φ_{g'} = Σ_{g'} Σ_s[g', g] A_{g'}
            # (ORPHEUS SigS[g_from, g_to] — the transpose-active term).
            in_scatter = np.zeros((nx, ny_))
            for g_from in range(ng):
                sig_s = self.sigma_s_fn(
                    xx_flat, yy_flat, g_from, g,
                ).reshape(nx, ny_)
                in_scatter += sig_s * drivers[g_from][0]
            Q[:, g, :, :] = streaming + removal - in_scatter[None]
        # Pattern 7 — emit per-ordinate density.
        Q /= sum_w
        return Q

    def prescribed_inflow(self, sn_mesh):
        r"""The ``q.boundary`` prescribed-inflow term (a
        :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`).

        For each domain face and group :math:`g`, the inflow ordinate slots
        carry :math:`\gamma_-\psi = \psi_{n,g}(x_{\rm face}, \mu_n)/W`, the
        face-trace of the manufactured angular flux.  Because :math:`a_0>0` the
        trace is NON-zero on all four edges — the boundary closure is stressed
        at the FACE-AVERAGE moment.

        **Moment honesty (#257 S9).**  When the mesh is moment-resolved (the
        bilinear UBLD Linear-Discontinuous closure, ``face_moment_count > 1``),
        the inflow trace varies transversely ALONG each face, so this builds the
        FULL transverse moment slot ``(N, ng, n_t, face_moment_count)`` — slot 0
        the transverse cell AVERAGE, slot 1 the bare transverse :math:`P_1`
        slope — and feeds it to the producer's full-slot branch
        (:meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`).
        The slope is genuinely consumed by the LD boundary closure (its
        converged-flux contribution is sub-floor — a representation refinement,
        not a deficiency repair; see the theory page).  When the mesh is a
        cell-average closure (DD/Step, ``face_moment_count == 1``) it builds the
        SCALAR per-face trace ``(N, ng, n_t)`` exactly as before — byte-identical
        (the producer's scalar branch seeds slot 0, no moment axis exists).

        Materialised via the ergonomic
        :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
        generator (full ``(N, ng, n_face[, face_moment_count])`` per face; the
        generator keeps only the inflow ordinates).
        """
        from orpheus.numerics.moment_layout import face_moment_count
        from orpheus.transport.source_sinks import AngularBoundarySourceSink

        n_face_moments = face_moment_count(
            sn_mesh.scheme.spatial_basis_per_axis, sn_mesh.ndim,
        )
        W = float(self.quadrature.weights.sum())
        mu_x = self.quadrature.mu_x
        mu_y = self.quadrature.mu_y
        ng = self.n_groups
        mesh = sn_mesh.mesh
        cx = mesh.centers_x
        cy = mesh.centers_y
        ex = mesh.edges_x
        ey = mesh.edges_y
        Lx, Ly = self.length_x, self.length_y

        # Each face's transverse direction: x-faces vary in y (transverse edges
        # ey, transverse centres cy), y-faces vary in x (ex, cx).  The constant
        # axis carries the fixed face coordinate.  ``const_axis`` selects which
        # of (x, y) is fixed so ``_project_inflow_to_face_moments`` can evaluate
        # ψ_{n,g}(face, t) = (A + μ_x B + μ_y C)/W at transverse positions t.
        face_specs = {
            "xmin": ("x", 0.0, ey, cy),
            "xmax": ("x", Lx, ey, cy),
            "ymin": ("y", 0.0, ex, cx),
            "ymax": ("y", Ly, ex, cx),
        }
        face_values: dict[str, np.ndarray] = {}
        for face, (const_axis, const_val, t_edges, t_centres) in face_specs.items():
            n_t = len(t_centres)               # transverse cell count
            if n_face_moments == 1:
                # DD/Step — scalar per-face trace (cell-CENTRE eval), the
                # byte-identical path (the producer seeds slot 0).
                vals = np.empty((len(mu_x), ng, n_t))
                for g in range(ng):
                    if const_axis == "x":
                        A, _, _, B, _, _, C, _, _ = self._drivers(
                            np.array([const_val]), t_centres, g)
                    else:
                        A, _, _, B, _, _, C, _, _ = self._drivers(
                            t_centres, np.array([const_val]), g)
                    A_t, B_t, C_t = A.reshape(n_t), B.reshape(n_t), C.reshape(n_t)
                    vals[:, g, :] = (
                        A_t[None, :]
                        + mu_x[:, None] * B_t[None, :]
                        + mu_y[:, None] * C_t[None, :]
                    ) / W
                face_values[face] = vals
            else:
                # LD (moment-resolved) — full transverse moment slot
                # (N, ng, n_t, face_moment_count): slot 0 the transverse cell
                # AVERAGE, slot 1 the bare transverse P₁ slope.
                face_values[face] = self._project_inflow_to_face_moments(
                    const_axis, const_val, t_edges, n_face_moments,
                )
        return AngularBoundarySourceSink.prescribed_inflow(sn_mesh, face_values)

    def _project_inflow_to_face_moments(
        self, const_axis, const_val, t_edges, n_face_moments,
    ) -> np.ndarray:
        r"""Project the manufactured inflow trace :math:`\psi_{n,g}(\text{face},t)/W`
        onto the per-transverse-cell BARE Legendre moments — shape
        ``(N, ng, n_t, n_face_moments)``.

        Per transverse cell :math:`[t_L, t_R]` (mapped to :math:`\xi\in[-1,1]`),
        slot 0 is the cell AVERAGE :math:`\langle\psi,P_0\rangle/\langle P_0,P_0
        \rangle` and slot 1 the BARE transverse slope :math:`\langle\psi,P_1
        \rangle/\langle P_1,P_1\rangle`.  NO :math:`\theta`/:math:`h_t` weighting
        — the cochain's transverse mass ``diag(h_t, θ·h_t)`` applies them
        downstream (a θ- or h_t-weighted slope would double-apply the mass, a
        TRUE bug; #251 §1 / #257 S9 GATE C).

        **L11 structural independence:** descends ONLY from ``self._drivers``
        (the manufactured angular-flux harmonics) + ``numpy.polynomial.legendre.
        leggauss`` (the transparent trusted-library quadrature) — NEVER
        ``_inflow_to_moments``, ``_ubld``, any LD operator, or the test-side
        projectors.  The leggauss rule integrates the linear-in-:math:`t`
        :math:`P_0`/:math:`P_1` projections of the (smooth, non-polynomial) trace
        to quadrature-converged accuracy; ``n_face_moments == 2`` (d=2 LD) uses
        the ``[bar, slope]`` slots.

        **Collapse trigger (bounding the L11 twin):** this projector and the
        test-side ``_face_transverse_legendre`` are deliberately kept INDEPENDENT
        (GATE C pins their agreement; a shared import would make GATE C
        tautological and let a double-applied transverse-mass slip through).  The
        only thing that would force them to merge is a 3-D face
        (``n_face_moments = per_axis**(d-1) > 2``), which needs a genuine
        tensor-Legendre lift on BOTH sides — the rule-of-two→three trigger that
        folds into the #263 collocation seam.  Until then the parallel-but-
        independent split is correct, not duplication.
        """
        from numpy.polynomial.legendre import leggauss

        from orpheus.numerics.moment_layout import AVERAGE_MOMENT

        W = float(self.quadrature.weights.sum())
        mu_x = self.quadrature.mu_x
        mu_y = self.quadrature.mu_y
        ng = self.n_groups
        N = len(mu_x)
        n_t = len(t_edges) - 1

        # Transverse Gauss-Legendre rule on [-1, 1] (the BARE Legendre frame —
        # the same ξ∈[-1,1] basis {1, ξ} the cochain's transverse mass keys on).
        xi, wq = leggauss(6)
        W2 = float(wq.sum())
        mean_p1sq = float((wq * xi * xi).sum() / W2)   # mean(P₁²) = 1/3

        slot = np.zeros((N, ng, n_t, n_face_moments))
        for g in range(ng):
            for j in range(n_t):
                tL, tR = t_edges[j], t_edges[j + 1]
                tq = (tL + tR) / 2 + (tR - tL) / 2 * xi
                # ψ_{n,g}(face, t_q)/W per ordinate at the transverse nodes.
                if const_axis == "x":
                    A, _, _, B, _, _, C, _, _ = self._drivers(
                        np.array([const_val]), tq, g)
                else:
                    A, _, _, B, _, _, C, _, _ = self._drivers(
                        tq, np.array([const_val]), g)
                A_q, B_q, C_q = A.reshape(-1), B.reshape(-1), C.reshape(-1)
                # (N, q): per-ordinate trace at each transverse quadrature node.
                psi = (
                    A_q[None, :]
                    + mu_x[:, None] * B_q[None, :]
                    + mu_y[:, None] * C_q[None, :]
                ) / W
                # Slot 0 — transverse cell AVERAGE ⟨ψ,P₀⟩/⟨P₀,P₀⟩ (the slot-0
                # convention single-sourced via AVERAGE_MOMENT).
                slot[:, g, j, AVERAGE_MOMENT] = (wq[None, :] * psi).sum(axis=1) / W2
                # Slot 1 — bare transverse slope ⟨ψ,P₁⟩/⟨P₁,P₁⟩.
                if n_face_moments > 1:
                    slot[:, g, j, 1] = (
                        (wq[None, :] * xi[None, :] * psi).sum(axis=1)
                        / (W2 * mean_p1sq)
                    )
        return slot


def build_2d_cartesian_ld_stress_mms_case(
    length_x: float = 1.3,
    length_y: float = 0.9,
    c_spectrum: tuple[float, float] = (1.0, 0.4),
    level_symmetric_order: int = 4,
    quadrature: "Quadrature | None" = None,
    name: str = "sn_mms_2d_cartesian_ld_stress",
) -> SN2DCartesianLDStressMMSCase:
    r"""Build the canonical 2-D Cartesian LD stress MMS case (#240 D5b-S4).

    NON-SQUARE domain (``length_x=1.3 ≠ length_y=0.9`` — the x↔y-swap defence),
    heterogeneous 2G (``_default_hetero_2d_xs_functions``, :math:`\Sigma_a>0`,
    asymmetric downscatter), per-group amplitudes :math:`\mathbf c=(1.0, 0.4)`.

    The default quadrature is **level-symmetric S4** (N=24): it resolves the
    bilinear streaming moments :math:`\langle\mu_x^2\rangle=\langle\mu_y^2
    \rangle=W/3`, :math:`\langle\mu_x\mu_y\rangle=0` EXACTLY (so :math:`\phi=A`)
    and carries NO pure-z ordinate (the cheap headline-gate quadrature).  Pass
    a Lebedev quadrature for the matvec-twin gate (it carries the ±z poles, the
    ERR-062 pure-z habitat)."""
    sigma_t_fn, sigma_s_fn = _default_hetero_2d_xs_functions(
        float(length_x), float(length_y),
    )
    quad = quadrature or Quadrature.level_symmetric(level_symmetric_order)
    return SN2DCartesianLDStressMMSCase(
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
    quadrature: Quadrature
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
        Shape ``(N, ng=1, nx)``.
        """
        x = mesh.centers
        L = self.slab_length
        A = np.sin(np.pi * x / L)
        Ap = (np.pi / L) * np.cos(np.pi * x / L)
        mu = self.quadrature.mu_x
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu)
        a = self.alpha

        # Each term: (N, nx)
        t1 = mu[:, None] * Ap[None, :]                              # μ A'
        t2 = (self.sigma_t - self.sigma_s0) * A[None, :]            # (Σ_t - Σ_s0) A
        t3 = a * mu[:, None] * (self.sigma_t - self.sigma_s1) * A[None, :]  # α μ (Σ_t - Σ_s1) B
        t4 = a * (mu[:, None] ** 2) * Ap[None, :]                   # α μ² B'
        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q = (t1 + t2 + t3 + t4) / sum_w
        return Q[:, None, :]


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
    # Synthetic 1G P1 mixture: no physical energy grid (Phase E).
    return Mixture(
        SigC=np.array([sigma_t - sigma_s0]),
        SigL=np.zeros(ng),
        SigF=np.zeros(ng),
        SigP=np.zeros(ng),
        SigT=np.array([sigma_t]),
        SigS=[SigS0, SigS1],   # P0 and P1 scattering matrices
        Sig2=[Sig2],
        chi=np.zeros(ng),
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
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
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
    quadrature: Quadrature
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
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu)
        streaming = mu[:, None] * Ap[None, :]
        removal = (self.sigma_t - self.sigma_s) * A[None, :]
        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q = (streaming + removal) / sum_w
        return Q[:, None, :]


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
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
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
# Phase 3.4 — 1D Cylindrical MMS (1-group, σ_y-folded product quadrature
# since Q5.6's 6.3 flip — SNMesh(CYLINDRICAL) admits exactly the
# carrying rules, so the case builders default to folded_product)
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
    quadrature: Quadrature
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
        eta = self.quadrature.eta
        sum_w = float(self.quadrature.weights.sum())
        N = len(eta)
        streaming = eta[:, None] * Ap[None, :]
        removal = (self.sigma_t - self.sigma_s) * A[None, :]
        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q = (streaming + removal) / sum_w
        return Q[:, None, :]


def build_cylindrical_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    radius: float = 5.0,
    n_mu: int = 4,
    n_phi: int = 8,
    mat_id: int = 1,
    name: str = "sn_mms_cylindrical_sin",
) -> SNCylindricalMMSCase:
    r"""Build the canonical 1D cylindrical MMS case.

    ``n_mu``/``n_phi`` are the PARENT rule's counts — the σ_y fold
    keeps ``n_mu * n_phi / 2`` ordinates on the ξ > 0 half (``n_phi``
    must be even; :meth:`Quadrature.folded_product
    <orpheus.numerics.quadrature.Quadrature.folded_product>` refuses
    odd). The ansatz is ξ-independent, hence trivially in the
    quotient's ξ-even function space.
    """
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = Quadrature.folded_product(n_mu=n_mu, n_phi=n_phi)
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
                "α=0.1. Exercises the l=1 scattering emission (the "
                "ScatteringOperator's redistribution route, driven as the "
                "SI gain). Manufactured source includes "
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


# ═══════════════════════════════════════════════════════════════════════
# Phase 3.6 — Anisotropic curvilinear MMS (vv-principles failure mode #7)
# ═══════════════════════════════════════════════════════════════════════
r"""
Anisotropic curvilinear MMS — activates the angular-redistribution term.

**Why this case exists.** The ``vv-principles`` skill calls out
test-design failure mode #7 (MMS simplification bias): the existing
isotropic curvilinear MMS in :class:`SNSphericalMMSCase` /
:class:`SNCylindricalMMSCase` uses ansatz
:math:`\psi_n(r) = A(r)/W`. The angular redistribution term
(:math:`(1-\mu^2)/r \cdot \partial\psi/\partial\mu` for the sphere,
:math:`-(1/r)\partial(\xi\psi)/\partial\varphi` for the cylinder) is
**identically zero** for that ansatz, so the MMS test mathematically
cannot detect ERR-026-class bugs (curvilinear sweep WDD bug, where
the wrong fixed point emerges from a redistribution miscoupling).
This Phase 3.6 ansatz adds a :math:`\mu`-linear (sphere) or
:math:`\eta`-linear (cylinder) term so the redistribution path is
exercised under refinement.

**Spherical ansatz.** On a vacuum-BC sphere of radius :math:`R`:

.. math::

    \psi_n(r) = \frac{1}{W}\bigl(A(r) + B(r)\,\mu_n\bigr),
    \qquad
    A(r) = \sin\!\left(\frac{\pi r}{R}\right),
    \qquad
    B(r) = \frac{r}{R}\Bigl(1 - \frac{r}{R}\Bigr)
            \cos\!\left(\frac{\pi r}{R}\right).

:math:`A(r)` vanishes at :math:`r = 0` (symmetry BC) and :math:`r = R`
(vacuum BC). :math:`B(r)` vanishes at both endpoints by the
:math:`r(R-r)` factor (symmetry: :math:`B(0) = 0` keeps the centre
isotropic; vacuum: :math:`B(R) = 0` so every ordinate satisfies the
vacuum BC). The :math:`\mu_n` coefficient is non-trivial: one
ordinate's BC differs from another's by sign of :math:`\mu_n`, but
both vanish at :math:`r \in \{0, R\}`. The scalar flux is
:math:`\phi(r) = A(r)` because :math:`\sum_n w_n \mu_n = 0` for any
symmetric quadrature.

**Spherical manufactured source.** Substituting into

.. math::

    \mu_n\,\frac{\partial\psi_n}{\partial r}
        + \frac{1 - \mu_n^2}{r}\,\frac{\partial\psi}{\partial \mu}
        + \Sigma_t\,\psi_n
    = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\text{ext}}_n\bigr)

and solving for :math:`Q^{\text{ext}}_n`:

.. math::

    Q^{\text{ext}}_n(r) =
        \mu_n\,A'(r)
      + \mu_n^2\,B'(r)
      + (1 - \mu_n^2)\,\frac{B(r)}{r}
      + (\Sigma_t - \Sigma_s)\,A(r)
      + \Sigma_t\,\mu_n\,B(r).

The :math:`\mu_n^2 B'(r)` and :math:`(1-\mu_n^2)\,B/r` terms are the
load-bearing pieces — neither is present in the isotropic MMS source.
The redistribution term :math:`(1-\mu_n^2)\,B/r` is the angular
analytic of the sphere; ERR-026 would alter this term's coupling and
produce a flux that fails the :math:`\mathcal O(h^{2})` convergence
test under refinement.

**Cylindrical ansatz.** The radial direction cosine is :math:`\eta`;
azimuthal partner is :math:`\xi` with :math:`\eta^2 + \xi^2 + \mu^2
= 1`. Use:

.. math::

    \psi_n(r) = \frac{1}{W}\bigl(A(r) + B(r)\,\eta_n\bigr),

with the same :math:`A(r)` / :math:`B(r)` shapes. :math:`\sum_n w_n
\eta_n = 0` for ProductQuadrature, so :math:`\phi(r) = A(r)`.

**Cylindrical manufactured source.** Substituting into

.. math::

    \frac{\eta}{r}\,\frac{\partial(r\psi)}{\partial r}
        - \frac{1}{r}\,\frac{\partial(\xi\psi)}{\partial \varphi}
        + \Sigma_t\,\psi
    = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\text{ext}}\bigr)

at fixed :math:`(\theta, \varphi)` per ordinate, with
:math:`\eta = \sin\theta\cos\varphi`,
:math:`\xi = \sin\theta\sin\varphi`, gives

.. math::

    Q^{\text{ext}}_n(r) =
        \eta_n\,A'(r)
      + \eta_n^2\,B'(r)
      + \xi_n^2\,\frac{B(r)}{r}
      + (\Sigma_t - \Sigma_s)\,A(r)
      + \Sigma_t\,\eta_n\,B(r).

The :math:`\xi_n^2\,B/r` term is the cylindrical analog of the
sphere's :math:`(1-\mu_n^2)\,B/r`; both come from the angular
redistribution operator and both vanish for any isotropic ansatz.

**Verification chain.**

- Branch-1 SymPy: :func:`derive_spherical_anisotropic_mms` and
  :func:`derive_cylindrical_anisotropic_mms` substitute the
  ansatz into the continuous transport equation and confirm
  :math:`\mathrm{LHS} - \mathrm{RHS} = 0` algebraically — proves
  the closed form for :math:`Q^{\text{ext}}_n` is the unique
  source consistent with the ansatz.
- Branch-2 numerical: :class:`SNSphericalAnisotropicMMSCase` and
  :class:`SNCylindricalAnisotropicMMSCase` evaluate
  :math:`Q^{\text{ext}}_n(r)` at cell centres using vectorised
  numpy and feed the source to the SN solver.
- L1 cross-check (the gate): the symbolic substitution residual
  must be zero; the numerical :math:`Q^{\text{ext}}` (Branch 2)
  must agree with the SymPy-evaluated :math:`Q^{\text{ext}}`
  (Branch 1) to :math:`\sim 10^{-13}` on a sample mesh.

.. seealso::

   - ``.claude/skills/vv-principles/SKILL.md`` — failure mode #7
     ("MMS simplification bias").
   - :class:`SNSphericalMMSCase`, :class:`SNCylindricalMMSCase` —
     isotropic siblings retained for narrow-down diagnostics.
   - :cite:`BaileyMorelChang2010` for the spherical and cylindrical
     angular-redistribution operator structure.
"""


def _spherical_anisotropic_symbolic(
    A: "sp.Expr | None" = None,
    B: "sp.Expr | None" = None,
) -> "tuple[sp.Expr, ...]":
    r"""Build the symbolic objects for spherical anisotropic MMS.

    Returns ``(r, mu, R, Sigma_t, Sigma_s, W, A, B, psi, phi, Q)``:
    the symbolic ansatz and the closed-form Q^ext expression.
    Shared by the foundation test gate AND by the Branch-2 numerical
    factory (via lambdify) so both branches descend from the same
    SymPy ancestor.

    Parameters
    ----------
    A, B : sympy.Expr or None, optional
        The radial profiles of the ansatz
        :math:`\psi_n(r) = (A(r) + B(r)\mu_n)/W`.  When ``None`` (the
        default) the Phase 3.6 **vacuum** shapes are used —
        :math:`A=\sin(\pi r/R)`, :math:`B=(r/R)(1-r/R)\cos(\pi r/R)` —
        so the existing no-arg caller
        :func:`derive_spherical_anisotropic_mms` and all Phase 3.6
        tests are byte-unchanged.  Pass non-vanishing-at-:math:`R`
        shapes (with :math:`B(0)=0` for pole regularity, HAZARD H1) to
        build the Phase 4 / O.2b 4.6 **non-vacuum** case — both cases
        share the EXACT SAME closed form ``Q_closed`` derived below;
        only :math:`A,\,B` differ (Cardinal Rule 2 — single source of
        truth for the spherical transport-operator residual).
    """
    import sympy as sp  # local import: keep the symbolic dependency lazy

    r, mu, R = sp.symbols("r mu R", positive=True, real=True)
    Sigma_t, Sigma_s, W = sp.symbols(
        "Sigma_t Sigma_s W", positive=True, real=True,
    )
    if A is None:
        A = sp.sin(sp.pi * r / R)
    if B is None:
        B = (r / R) * (1 - r / R) * sp.cos(sp.pi * r / R)
    psi = (A + B * mu) / W
    # Scalar flux: int_{-1}^{1} (A + B mu) dmu / 2 = A. For any
    # symmetric quadrature, sum_n w_n mu_n = 0, so phi = A.
    phi = A

    # Closed-form Q^ext_n derived analytically; SymPy verifies in
    # the foundation test (derive_spherical_anisotropic_mms /
    # derive_nonvacuum_spherical_mms — the SAME closed form for both
    # A,B choices).
    Q_closed = (
        mu * sp.diff(A, r)
        + mu**2 * sp.diff(B, r)
        + (1 - mu**2) * B / r
        + (Sigma_t - Sigma_s) * A
        + Sigma_t * mu * B
    )
    return (r, mu, R, Sigma_t, Sigma_s, W, A, B, psi, phi, Q_closed)


def _cylindrical_anisotropic_symbolic() -> "tuple[sp.Expr, ...]":
    """Build the symbolic objects for cylindrical anisotropic MMS.

    Returns ``(r, theta, phi_az, R, Sigma_t, Sigma_s, W, A, B, eta,
    xi, psi, phi_scalar, Q)``: ``phi_az`` is the azimuthal angle
    :math:`\\varphi`; ``phi_scalar`` is the SymPy expression for the
    scalar flux (which equals :math:`A(r)` because :math:`\\sum_n
    w_n \\eta_n = 0` for ProductQuadrature).
    """
    import sympy as sp

    r, theta, phi_az, R = sp.symbols(
        "r theta phi_az R", positive=True, real=True,
    )
    Sigma_t, Sigma_s, W = sp.symbols(
        "Sigma_t Sigma_s W", positive=True, real=True,
    )
    eta = sp.sin(theta) * sp.cos(phi_az)
    xi = sp.sin(theta) * sp.sin(phi_az)

    A = sp.sin(sp.pi * r / R)
    B = (r / R) * (1 - r / R) * sp.cos(sp.pi * r / R)
    psi = (A + B * eta) / W
    phi_scalar = A  # symmetric quadrature => sum w_n eta_n = 0

    Q_closed = (
        eta * sp.diff(A, r)
        + eta**2 * sp.diff(B, r)
        + xi**2 * B / r
        + (Sigma_t - Sigma_s) * A
        + Sigma_t * eta * B
    )
    return (r, theta, phi_az, R, Sigma_t, Sigma_s, W, A, B,
            eta, xi, psi, phi_scalar, Q_closed)


def derive_spherical_anisotropic_mms() -> dict:
    r"""V_sph-aniso — spherical anisotropic-MMS source identity.

    Proves: substituting the ansatz
    :math:`\psi_n(r) = (A(r) + B(r)\mu_n)/W` (with
    :math:`A(r) = \sin(\pi r/R)`,
    :math:`B(r) = (r/R)(1 - r/R)\cos(\pi r/R)`) into the continuous
    spherical SN transport operator

    .. math::

       \mu \frac{\partial\psi}{\partial r}
       + \frac{1-\mu^2}{r}\frac{\partial\psi}{\partial \mu}
       + \Sigma_t\,\psi = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\rm ext}\bigr)

    yields, after :math:`Q^{\rm ext}` is solved for, the closed
    form

    .. math::

       Q^{\rm ext}_n(r) =
            \mu_n A'(r)
          + \mu_n^2 B'(r)
          + (1-\mu_n^2)\,\frac{B(r)}{r}
          + (\Sigma_t-\Sigma_s) A(r)
          + \Sigma_t\,\mu_n B(r),

    with the scalar flux :math:`\phi(r) = A(r)` (because
    :math:`\sum_n w_n \mu_n = 0` for any symmetric quadrature).

    The redistribution term :math:`(1-\mu^2)\,B/r` is what the
    isotropic MMS misses by construction; this identity is the
    foundation that makes :class:`SNSphericalAnisotropicMMSCase`
    a non-trivial probe of the curvilinear sweep.
    """
    import sympy as sp

    r, mu, R, Sigma_t, Sigma_s, W, A, B, psi, phi_, Q_closed = (
        _spherical_anisotropic_symbolic()
    )

    LHS = (
        mu * sp.diff(psi, r)
        + (1 - mu**2) / r * sp.diff(psi, mu)
        + Sigma_t * psi
    )
    Q_subst = sp.simplify(W * LHS - Sigma_s * phi_)
    diff = sp.simplify(Q_subst - Q_closed)

    return {
        "name": "V_sph-aniso: spherical anisotropic MMS source identity",
        "psi": psi,
        "phi": phi_,
        "Q_substituted": Q_subst,
        "Q_closed": Q_closed,
        "diff": diff,
        "pass": diff == 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 4 / O.2b 4.6 — NON-VACUUM prescribed-inflow MMS (slab + sphere)
# ═══════════════════════════════════════════════════════════════════════
r"""
Non-vacuum prescribed-inflow MMS reference (Phase 4 / O.2b 4.6).

**The single structural delta over Phase 3.5/3.6.** Every existing MMS
ansatz VANISHES at both boundaries (:math:`A(0)=A(R)=0`), making them
vacuum-automatic — :math:`\gamma_-\psi \equiv 0` on every ordinate, so
the prescribed-inflow source slot ``q.boundary`` is identically zero.
4.6 chooses :math:`A,\,B` **non-vanishing at the outer face** so
:math:`\gamma_-\psi = \psi_{\rm chosen}(x_{\rm face}, \mu_n) \neq 0`.
That is the entire novelty: the :math:`q.\text{boundary} \neq 0`
prescribed-inflow path that the existing catalog never exercises.

The form is the proven P1 element :math:`\psi_n = (A + \mu_n B)/W` —
linear-in-:math:`\mu` FULLY activates the curvilinear redistribution
:math:`(1-\mu^2)B/r` (the discrete closure is linear; see the
cross-domain-attacker frame memo). 4.6 changes ONLY the boundary-trace:

- **Slab** (no pole): :math:`A(x)=a_0+a_1\sin(k x)` with :math:`a_0>0`
  (non-zero at faces) and :math:`B(x)=b_0\cos(k x)` (non-zero at faces).
  The slab transport operator has NO angular-redistribution term, so it
  is a genuinely DIFFERENT operator from the sphere — it gets its own
  symbolic pair (:func:`_nonvacuum_slab_symbolic` /
  :func:`derive_nonvacuum_slab_mms`).
- **Sphere** (pole at :math:`r=0`): :math:`A(r)=a_0+a_1\sin(k r)` and
  :math:`B(r)=(r/R)[b_0+b_1\cos(k r)]`. The :math:`(r/R)` prefactor
  keeps :math:`B(0)=0` (HAZARD H1 — the redistribution
  :math:`(1-\mu^2)B/r \to \infty` at the pole otherwise) while leaving
  :math:`B(R)\neq 0` (non-vacuum outer inflow). It REUSES
  :func:`_spherical_anisotropic_symbolic` with these :math:`A,\,B`
  (Cardinal Rule 2 — the spherical-operator residual lives in ONE
  place).

The manufactured bulk source :math:`Q^{\rm ext}` is SymPy-derived
(Branch 1, State 1C); the reference scalar flux is
:math:`\phi_{\rm chosen}(x) = A(x)` (since :math:`\sum_n w_n \mu_n = 0`),
which is NON-ZERO at the boundary — the load-bearing property that lets
the converged-VALUE assertion catch a dropped ``q.boundary``.

.. seealso::

   - ``.claude/skills/vv-principles/SKILL.md`` failure mode #7 + Mode 9.
   - :class:`SNSlabNonVacuumMMSCase`, :class:`SNSphericalNonVacuumMMSCase`
     (Branch-2 numerical factories).
   - ``docs/theory/verification/sn.rst`` labels
     ``sn-mms-nonvacuum-psi``/``-qext`` (slab),
     ``sn-mms-nonvacuum-sph-psi``/``-qext`` (sphere).
"""


def _nonvacuum_slab_symbolic() -> "tuple[sp.Expr, ...]":
    r"""Build the symbolic objects for the NON-VACUUM slab MMS.

    Returns ``(x, mu, k, a0, a1, b0, Sigma_t, Sigma_s, W, A, B, psi,
    phi, Q)``.  The slab transport operator has NO angular
    redistribution, so this is a fresh symbolic pair (the sphere's
    :func:`_spherical_anisotropic_symbolic` cannot be reused — it
    carries the :math:`(1-\mu^2)/r\,\partial_\mu` term that the slab
    lacks).

    Ansatz: :math:`\psi_n(x) = (A(x) + \mu_n B(x))/W`, with
    :math:`A(x) = a_0 + a_1\sin(kx)` (**a0>0** → non-zero at faces, the
    4.6 novelty) and :math:`B(x) = b_0\cos(kx)` (non-zero at faces).
    The scalar flux is :math:`\phi(x) = A(x)` (since
    :math:`\sum_n w_n \mu_n = 0`).

    Closed-form source (slab — no redistribution):

    .. math::

       Q^{\rm ext}_n(x) = \mu_n A'(x) + \mu_n^2 B'(x)
                        + (\Sigma_t - \Sigma_s) A(x)
                        + \Sigma_t\,\mu_n B(x).
    """
    import sympy as sp

    x, mu, k = sp.symbols("x mu k", positive=True, real=True)
    a0, a1, b0 = sp.symbols("a0 a1 b0", real=True)
    Sigma_t, Sigma_s, W = sp.symbols(
        "Sigma_t Sigma_s W", positive=True, real=True,
    )
    A = a0 + a1 * sp.sin(k * x)
    B = b0 * sp.cos(k * x)
    psi = (A + B * mu) / W
    phi = A

    Q_closed = (
        mu * sp.diff(A, x)
        + mu**2 * sp.diff(B, x)
        + (Sigma_t - Sigma_s) * A
        + Sigma_t * mu * B
    )
    return (x, mu, k, a0, a1, b0, Sigma_t, Sigma_s, W, A, B, psi, phi, Q_closed)


def derive_nonvacuum_slab_mms() -> dict:
    r"""V_nonvac-slab — non-vacuum slab MMS source identity.

    Proves: substituting the ansatz
    :math:`\psi_n(x) = (A(x) + \mu_n B(x))/W` (with
    :math:`A = a_0 + a_1\sin(kx)`, :math:`B = b_0\cos(kx)`) into the
    continuous slab SN operator

    .. math::

       \mu\,\frac{\partial\psi}{\partial x} + \Sigma_t\,\psi
       = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\rm ext}\bigr)

    yields the closed form

    .. math::

       Q^{\rm ext}_n(x) = \mu_n A'(x) + \mu_n^2 B'(x)
                        + (\Sigma_t - \Sigma_s) A(x)
                        + \Sigma_t\,\mu_n B(x),

    with :math:`\phi(x) = A(x)`.  Unlike the existing
    :class:`SNP1AnisoMMSCase` (which uses :math:`A=B=\sin(\pi x/L)`,
    vanishing at the faces, and a P1 *scattering* concern), here
    :math:`a_0>0` makes :math:`A` — and hence :math:`\gamma_-\psi` —
    NON-zero at the boundary.  THAT is the 4.6 novelty (prescribed
    inflow), NOT the angular form (which is the proven P1 element).
    """
    import sympy as sp

    (x, mu, k, a0, a1, b0, Sigma_t, Sigma_s, W, A, B, psi, phi_, Q_closed) = (
        _nonvacuum_slab_symbolic()
    )

    LHS = mu * sp.diff(psi, x) + Sigma_t * psi
    Q_subst = sp.simplify(W * LHS - Sigma_s * phi_)
    diff = sp.simplify(Q_subst - Q_closed)

    return {
        "name": "V_nonvac-slab: non-vacuum slab MMS source identity",
        "psi": psi,
        "phi": phi_,
        "Q_substituted": Q_subst,
        "Q_closed": Q_closed,
        "diff": diff,
        "pass": diff == 0,
    }


def _nonvacuum_spherical_AB() -> "tuple[sp.Expr, sp.Expr]":
    r"""The Phase 4 / O.2b 4.6 spherical :math:`A(r),\,B(r)` shapes.

    :math:`A(r) = a_0 + a_1\sin(k r)` (a0>0 → :math:`A(R)\neq 0`, finite
    at the pole — A has no :math:`1/r` companion so :math:`A(0)=a_0` is
    fine).  :math:`B(r) = (r/R)[b_0 + b_1\cos(k r)]` (HAZARD H1: the
    :math:`(r/R)` prefactor forces :math:`B(0)=0` so the redistribution
    :math:`(1-\mu^2)B/r` is regular at the pole, while
    :math:`B(R)=b_0+b_1\cos(kR)` may be :math:`\neq 0` for the non-vacuum
    outer inflow).

    The numeric coefficients are baked into the SYMBOLIC shapes (not
    free symbols) so the substitution residual SIMPLIFIES to zero
    cleanly and the Branch-2 factory can lambdify against the same
    closed form.  The defaults match
    :func:`build_spherical_nonvacuum_mms_case`.
    """
    import sympy as sp

    r, R, k = sp.symbols("r R k", positive=True, real=True)
    a0, a1 = sp.Rational(1, 2), sp.Rational(1, 4)
    b0, b1 = sp.Rational(3, 10), sp.Rational(1, 5)
    A = a0 + a1 * sp.sin(k * r)
    B = (r / R) * (b0 + b1 * sp.cos(k * r))
    return A, B


def derive_nonvacuum_spherical_mms() -> dict:
    r"""V_nonvac-sph — non-vacuum spherical MMS source identity.

    Proves: substituting the ansatz
    :math:`\psi_n(r) = (A(r) + B(r)\mu_n)/W` (with
    :math:`A = a_0 + a_1\sin(kr)`, :math:`a_0>0`, and
    :math:`B = (r/R)[b_0 + b_1\cos(kr)]`, :math:`B(0)=0`) into the
    continuous spherical SN operator

    .. math::

       \mu\,\frac{\partial\psi}{\partial r}
       + \frac{1-\mu^2}{r}\,\frac{\partial\psi}{\partial \mu}
       + \Sigma_t\,\psi = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\rm ext}\bigr)

    yields the SAME closed form as the Phase 3.6 vacuum case
    (:func:`derive_spherical_anisotropic_mms`) — only :math:`A,\,B`
    differ:

    .. math::

       Q^{\rm ext}_n(r) = \mu_n A'(r) + \mu_n^2 B'(r)
                        + (1-\mu_n^2)\,\frac{B(r)}{r}
                        + (\Sigma_t-\Sigma_s) A(r)
                        + \Sigma_t\,\mu_n B(r).

    REUSES :func:`_spherical_anisotropic_symbolic` with the 4.6 shapes
    (Cardinal Rule 2): the spherical-operator residual is derived in ONE
    place; this function only swaps the boundary-trace-non-vanishing
    :math:`A,\,B`.  HAZARD H1: :math:`B(0)=0` keeps the redistribution
    regular at the pole.
    """
    import sympy as sp

    A_nv, B_nv = _nonvacuum_spherical_AB()
    r, mu, R, Sigma_t, Sigma_s, W, A, B, psi, phi_, Q_closed = (
        _spherical_anisotropic_symbolic(A=A_nv, B=B_nv)
    )

    LHS = (
        mu * sp.diff(psi, r)
        + (1 - mu**2) / r * sp.diff(psi, mu)
        + Sigma_t * psi
    )
    Q_subst = sp.simplify(W * LHS - Sigma_s * phi_)
    diff = sp.simplify(Q_subst - Q_closed)

    return {
        "name": "V_nonvac-sph: non-vacuum spherical MMS source identity",
        "psi": psi,
        "phi": phi_,
        "Q_substituted": Q_subst,
        "Q_closed": Q_closed,
        "diff": diff,
        "pass": diff == 0,
    }


def derive_cylindrical_anisotropic_mms() -> dict:
    r"""V_cyl-aniso — cylindrical anisotropic-MMS source identity.

    Proves: substituting the ansatz
    :math:`\psi_n(r) = (A(r) + B(r)\eta_n)/W` (with
    :math:`\eta = \sin\theta\cos\varphi`, same :math:`A,\,B` as the
    sphere) into the continuous cylindrical SN transport operator

    .. math::

       \frac{\eta}{r}\frac{\partial(r\psi)}{\partial r}
       - \frac{1}{r}\frac{\partial(\xi\psi)}{\partial\varphi}
       + \Sigma_t\,\psi = \frac{1}{W}\bigl(\Sigma_s\,\phi + Q^{\rm ext}\bigr)

    yields the closed form

    .. math::

       Q^{\rm ext}_n(r) =
            \eta_n A'(r)
          + \eta_n^2 B'(r)
          + \xi_n^2 \frac{B(r)}{r}
          + (\Sigma_t-\Sigma_s) A(r)
          + \Sigma_t\,\eta_n B(r),

    with :math:`\phi(r) = A(r)` (symmetric quadrature gives
    :math:`\sum_n w_n \eta_n = 0`).

    Note the :math:`\xi_n^2\,B/r` redistribution term: this is the
    cylindrical analog of the sphere's :math:`(1-\mu_n^2)\,B/r`;
    both originate from the angular-redistribution operator that
    the isotropic ansatz nullifies by construction.
    """
    import sympy as sp

    (
        r, theta, phi_az, R, Sigma_t, Sigma_s, W,
        A, B, eta, xi, psi, phi_scalar, Q_closed,
    ) = _cylindrical_anisotropic_symbolic()

    LHS = (
        (eta / r) * sp.diff(r * psi, r)
        - (1 / r) * sp.diff(xi * psi, phi_az)
        + Sigma_t * psi
    )
    Q_subst = sp.simplify(W * LHS - Sigma_s * phi_scalar)
    diff = sp.simplify(Q_subst - Q_closed)

    return {
        "name": "V_cyl-aniso: cylindrical anisotropic MMS source identity",
        "psi": psi,
        "phi": phi_scalar,
        "Q_substituted": Q_subst,
        "Q_closed": Q_closed,
        "diff": diff,
        "pass": diff == 0,
    }


@dataclass(frozen=True)
class SNSphericalAnisotropicMMSCase:
    r"""Anisotropic-ansatz MMS fixed-source problem for 1D spherical SN.

    **Activates the angular-redistribution term** that the isotropic
    sibling :class:`SNSphericalMMSCase` cancels by construction. See
    the module docstring for the load-bearing math; the closed-form
    :math:`Q^{\rm ext}_n(r)` is

    .. math::

        Q^{\rm ext}_n(r) =
            \mu_n A'(r) + \mu_n^2 B'(r)
          + (1-\mu_n^2)\,\frac{B(r)}{r}
          + (\Sigma_t - \Sigma_s) A(r) + \Sigma_t\,\mu_n B(r),

    with :math:`A(r) = \sin(\pi r/R)`,
    :math:`B(r) = (r/R)(1-r/R)\cos(\pi r/R)`. Both :math:`A` and
    :math:`B` vanish at :math:`r \in \{0, R\}`, so the symmetry +
    vacuum BCs hold for every ordinate.

    Optional P1 anisotropic-scattering coupling: when ``sigma_s1``
    is non-zero, the same :math:`B(r)\mu_n` term enters the P1
    scattering source; the field is reserved for a Phase-3.7
    extension and is **not** wired in this dataclass (which keeps
    P0 only, mirroring :class:`SNSphericalMMSCase`).
    """

    name: str
    sigma_t: float
    sigma_s: float
    radius: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: Quadrature
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-spherical",
        "sn-mms-spherical-aniso-psi",
        "sn-mms-spherical-aniso-qext",
    )

    # ── Reference solution shapes ────────────────────────────────────

    def A(self, r: np.ndarray) -> np.ndarray:
        r"""Radial profile :math:`A(r) = \sin(\pi r/R)`."""
        return np.sin(np.pi * np.asarray(r) / self.radius)

    def Ap(self, r: np.ndarray) -> np.ndarray:
        r"""Derivative :math:`A'(r) = (\pi/R)\cos(\pi r/R)`."""
        R = self.radius
        return (np.pi / R) * np.cos(np.pi * np.asarray(r) / R)

    def B(self, r: np.ndarray) -> np.ndarray:
        r"""Angular-coupling profile
        :math:`B(r) = (r/R)(1 - r/R)\cos(\pi r/R)`. Vanishes at
        :math:`r=0` (symmetry BC) and :math:`r=R` (vacuum BC)."""
        R = self.radius
        rr = np.asarray(r) / R
        return rr * (1.0 - rr) * np.cos(np.pi * rr)

    def Bp(self, r: np.ndarray) -> np.ndarray:
        r"""Derivative :math:`B'(r)`. Computed analytically:

        .. math::

           B'(r) = \frac{1}{R}\Bigl(1 - \frac{2r}{R}\Bigr)
                     \cos\!\left(\frac{\pi r}{R}\right)
                 - \frac{\pi}{R}\,\frac{r}{R}\Bigl(1 - \frac{r}{R}\Bigr)
                     \sin\!\left(\frac{\pi r}{R}\right).
        """
        R = self.radius
        rr = np.asarray(r) / R
        return (
            (1.0 - 2.0 * rr) * np.cos(np.pi * rr) / R
            - (np.pi / R) * rr * (1.0 - rr) * np.sin(np.pi * rr)
        )

    def phi_exact(self, r: np.ndarray) -> np.ndarray:
        r"""Reference scalar flux :math:`\phi(r) = A(r)`.

        For any symmetric quadrature on :math:`\mu \in [-1, 1]`
        (Gauss-Legendre satisfies this), :math:`\sum_n w_n \mu_n = 0`,
        so the :math:`\mu_n B(r)` term integrates to zero. The exact
        scalar flux is therefore identical to the isotropic case's
        :math:`\phi(r)`, but every individual angular flux
        :math:`\psi_n(r)` is non-trivially :math:`\mu_n`-dependent.
        """
        return self.A(r)

    def psi_exact(self, r: np.ndarray, mu_n: float) -> np.ndarray:
        r"""Reference angular flux for a given ordinate
        :math:`\psi_n(r) = (A(r) + B(r) \mu_n)/W`. Returned **without**
        the :math:`1/W` factor for caller convenience; the test
        consumer multiplies by :math:`1/W` if needed."""
        return self.A(r) + self.B(r) * mu_n

    # ── Mesh + source construction ───────────────────────────────────

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
        r"""Per-ordinate external source :math:`Q^{\rm ext}_n(r)` on
        ``mesh``. Shape ``(N, ng=1, nx)``.

        Closed-form evaluation (Branch 2 — vectorised numpy):

        .. math::

           Q^{\rm ext}_n(r) =
              \mu_n A'(r) + \mu_n^2 B'(r)
            + (1 - \mu_n^2)\,B(r)/r
            + (\Sigma_t - \Sigma_s) A(r) + \Sigma_t \mu_n B(r).

        Bit-equal to the SymPy expression
        :func:`derive_spherical_anisotropic_mms` returns (cross-checked
        in :file:`tests/derivations/test_sn_mms_anisotropic_symbolic.py`).
        """
        r = mesh.centers                              # (nx,)
        A_ = self.A(r)                                # (nx,)
        Ap_ = self.Ap(r)                              # (nx,)
        B_ = self.B(r)                                # (nx,)
        Bp_ = self.Bp(r)                              # (nx,)
        mu = self.quadrature.mu_x                     # (N,)
        sum_w = float(self.quadrature.weights.sum())

        streaming_iso = mu[:, None] * Ap_[None, :]               # μ A'
        streaming_aniso = (mu[:, None] ** 2) * Bp_[None, :]      # μ² B'
        redistribution = (1.0 - mu[:, None] ** 2) * (B_ / r)[None, :]  # (1-μ²) B/r
        removal_iso = (self.sigma_t - self.sigma_s) * A_[None, :]  # (Σ_t-Σ_s) A
        removal_aniso = self.sigma_t * mu[:, None] * B_[None, :]   # Σ_t μ B

        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q = (streaming_iso + streaming_aniso + redistribution
             + removal_iso + removal_aniso) / sum_w    # (N, nx)
        return Q[:, None, :]                     # (N, ng=1, nx)


def build_spherical_anisotropic_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    radius: float = 5.0,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_spherical_aniso",
) -> SNSphericalAnisotropicMMSCase:
    r"""Build the canonical anisotropic 1D spherical MMS case.

    Defaults match :func:`build_spherical_mms_case` so the two
    cases are paired one-to-one for narrow-down diagnostics: if the
    isotropic case passes :math:`\mathcal O(h^{2})` convergence and
    this case fails, the bug is in the angular-redistribution or
    P1-coupling code path.
    """
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
    return SNSphericalAnisotropicMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        radius=radius,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


# ═══════════════════════════════════════════════════════════════════════
# Phase 4 / O.2b 4.6 — NON-VACUUM prescribed-inflow MMS factories
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SNSlabNonVacuumMMSCase:
    r"""NON-VACUUM prescribed-inflow MMS for 1D slab SN (Phase 4 / O.2b 4.6).

    Per-group ansatz :math:`\psi_{n,g}(x) = (A_g(x) + \mu_n B_g(x))/W`
    with the shared shape scaled by a per-group amplitude :math:`c_g`:

    .. math::

       A_g(x) = c_g\,(a_0 + a_1\sin(kx)),\qquad B_g(x) = c_g\,b_0\cos(kx),
       \qquad a_0 > 0.

    The :math:`a_0>0` makes :math:`A_g` — and hence
    :math:`\gamma_-\psi = \psi_{n,g}(x_{\rm face}, \mu_n)/W` —
    NON-zero at the faces, the 4.6 novelty.  Unlike the vacuum-automatic
    :class:`SNSlabMMSCase` / :class:`SNP1AnisoMMSCase` (both vanish at
    the faces), the prescribed-inflow source slot ``q.boundary`` is
    exercised.

    The scalar flux is :math:`\phi_g(x) = A_g(x)` (since
    :math:`\sum_n w_n \mu_n = 0`), NON-zero at the boundary — the
    load-bearing property that lets the converged-VALUE assertion catch
    a dropped ``q.boundary`` (a vacuum solve converges cleanly to a
    boundary-zero limit).

    Multi-group: ``c_groups`` is the per-group amplitude vector
    (``(1.0,)`` for 1g; e.g. ``(1.0, 0.4)`` for the 2g row).
    ``sigma_t_g`` is the per-group total XS; ``sigma_s_matrix`` is the
    :math:`\Sigma_s[g_{\rm from}, g_{\rm to}]` scattering matrix
    (ORPHEUS convention: rows = source group, cols = sink group; the
    in-scatter source is :math:`(\Sigma_s^\top \phi)_g`).  An ASYMMETRIC
    downscatter-only matrix keeps the ERR-002 transpose hazard live.

    The mesh BCs are VACUUM — the prescribed inflow is injected via
    :meth:`prescribed_inflow` (the ``q.boundary`` slot), NOT a mesh BC.
    The Branch-2 numerical :meth:`external_source` is bit-equal to the
    lambdified Branch-1 SymPy (:func:`derive_nonvacuum_slab_mms`) on the
    1g shape.
    """

    name: str
    slab_length: float
    a0: float
    a1: float
    b0: float
    k: float
    c_groups: np.ndarray
    sigma_t_g: np.ndarray
    sigma_s_matrix: np.ndarray
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: Quadrature
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cartesian",
        "dd-cartesian-1d",
        "dd-slab",
        "sn-mms-nonvacuum-psi",
        "sn-mms-nonvacuum-qext",
    )

    @property
    def n_groups(self) -> int:
        return int(len(self.c_groups))

    # ── Reference solution shapes (Branch 2 numpy) ───────────────────

    def _shape(self, x: np.ndarray) -> np.ndarray:
        r"""Group-independent shape :math:`a_0 + a_1\sin(kx)`."""
        return self.a0 + self.a1 * np.sin(self.k * np.asarray(x))

    def _shape_p(self, x: np.ndarray) -> np.ndarray:
        r""":math:`a_1 k\cos(kx)`."""
        return self.a1 * self.k * np.cos(self.k * np.asarray(x))

    def _bshape(self, x: np.ndarray) -> np.ndarray:
        r""":math:`b_0\cos(kx)`."""
        return self.b0 * np.cos(self.k * np.asarray(x))

    def _bshape_p(self, x: np.ndarray) -> np.ndarray:
        r""":math:`-b_0 k\sin(kx)`."""
        return -self.b0 * self.k * np.sin(self.k * np.asarray(x))

    def A(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r""":math:`A_g(x) = c_g(a_0 + a_1\sin(kx))`."""
        return self.c_groups[g] * self._shape(x)

    def Ap(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r""":math:`A_g'(x) = c_g a_1 k\cos(kx)`."""
        return self.c_groups[g] * self._shape_p(x)

    def B(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r""":math:`B_g(x) = c_g b_0\cos(kx)`."""
        return self.c_groups[g] * self._bshape(x)

    def Bp(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r""":math:`B_g'(x) = -c_g b_0 k\sin(kx)`."""
        return self.c_groups[g] * self._bshape_p(x)

    def phi_exact(self, x: np.ndarray, g: int = 0) -> np.ndarray:
        r"""Reference scalar flux :math:`\phi_g(x) = A_g(x)`."""
        return self.A(x, g)

    def psi_exact(self, x: np.ndarray, mu_n: float, g: int = 0) -> np.ndarray:
        r"""Reference angular flux :math:`\psi_{n,g}(x) = A_g(x) + \mu_n
        B_g(x)` (WITHOUT the :math:`1/W` factor)."""
        return self.A(x, g) + self.B(x, g) * mu_n

    # ── Mesh + source construction ───────────────────────────────────

    def build_mesh(self, n_cells: int) -> Mesh1D:
        """VACUUM-BC slab mesh; prescribed inflow is the ``q.boundary`` slot."""
        edges = np.linspace(0.0, self.slab_length, n_cells + 1)
        mat_ids = np.full(n_cells, self.mat_id, dtype=int)
        return Mesh1D(
            edges=edges, mat_ids=mat_ids,
            coord=CoordSystem.CARTESIAN,
            bc_left=BC("vacuum"), bc_right=BC("vacuum"),
        )

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r"""Per-ordinate-density bulk source on ``mesh``. Shape
        ``(N, ng, nx, 1)``.

        .. math::

           Q^{\rm ext}_{n,g}(x) = \mu_n A_g'(x) + \mu_n^2 B_g'(x)
                            + \Sigma_{t,g}\,A_g(x)
                            + \Sigma_{t,g}\,\mu_n B_g(x)
                            - \sum_{g'}\Sigma_s[g', g]\,A_{g'}(x).

        The in-scatter source uses :math:`\Sigma_s[g', g]` (ORPHEUS
        ``SigS[g_from, g_to]`` convention — the transpose-active term
        the ERR-002 hazard lives in).  For 1g this reduces to
        :math:`(\Sigma_t - \Sigma_s) A`, bit-equal to the SymPy closed
        form (:func:`derive_nonvacuum_slab_mms`).  Divided by
        :math:`\sum_n w_n` at the producer boundary (Pattern 7).
        """
        x = mesh.centers                              # (nx,)
        mu = self.quadrature.mu_x                     # (N,)
        sum_w = float(self.quadrature.weights.sum())
        N = len(mu)
        nx = len(x)
        ng = self.n_groups

        Q = np.zeros((N, ng, nx))
        for g in range(ng):
            A_g = self.A(x, g)
            Ap_g = self.Ap(x, g)
            B_g = self.B(x, g)
            Bp_g = self.Bp(x, g)
            sig_t_g = float(self.sigma_t_g[g])

            streaming_iso = mu[:, None] * Ap_g[None, :]            # μ A_g'
            streaming_aniso = (mu[:, None] ** 2) * Bp_g[None, :]   # μ² B_g'
            removal_iso = sig_t_g * A_g[None, :]                   # Σt_g A_g
            removal_aniso = sig_t_g * mu[:, None] * B_g[None, :]   # Σt_g μ B_g
            # in-scatter: Σ_{g'} Σs[g', g] A_{g'}  (SigSᵀ — transpose-active)
            in_scatter = np.zeros(nx)
            for g_from in range(ng):
                in_scatter += self.sigma_s_matrix[g_from, g] * self.A(x, g_from)

            Q[:, g, :] = (
                streaming_iso + streaming_aniso
                + (removal_iso + removal_aniso)
                - in_scatter[None, :]
            )
        Q /= sum_w
        return Q

    def prescribed_inflow(self, sn_mesh):
        r"""The ``q.boundary`` prescribed-inflow term — a
        :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`.

        For each boundary face and group :math:`g`, the inflow ordinate
        slots carry :math:`\gamma_-\psi = \psi_{n,g}(x_{\rm face}, \mu_n)/W
        = (A_g(x_{\rm face}) + \mu_n B_g(x_{\rm face}))/W` (the affine-BC
        inhomogeneous term :math:`q`); both slab faces carry inflow because
        :math:`a_0>0`. Materialised via the ergonomic
        :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
        generator (full ``(N, ng)`` per face; the generator keeps only the
        inflow ordinates).
        """
        from orpheus.transport.source_sinks import AngularBoundarySourceSink

        W = float(self.quadrature.weights.sum())
        mu = self.quadrature.mu_x
        ng = self.n_groups
        N = len(mu)
        face_values: dict[str, np.ndarray] = {}
        for face, x_face in {"xmin": 0.0, "xmax": self.slab_length}.items():
            vals = np.empty((N, ng))
            for g in range(ng):
                vals[:, g] = (self.A(x_face, g) + mu * self.B(x_face, g)) / W
            face_values[face] = vals
        return AngularBoundarySourceSink.prescribed_inflow(sn_mesh, face_values)

def build_slab_nonvacuum_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    slab_length: float = 5.0,
    a0: float = 0.5,
    a1: float = 0.25,
    b0: float = 0.3,
    n_wavelengths: float = 1.5,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_slab_nonvacuum",
) -> SNSlabNonVacuumMMSCase:
    r"""Build the canonical 1-group non-vacuum slab MMS case (T1).

    :math:`k = 2\pi\,n_{\rm wavelengths}/L` so the slab spans several
    wavelengths (mixed-scale stress; ``vv-principles`` MMS operational
    rules).  Defaults give :math:`a_0=0.5>0` (non-vacuum), weak
    :math:`a_1, b_0` so the source is not stiff.
    """
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
    k = 2.0 * np.pi * n_wavelengths / slab_length
    return SNSlabNonVacuumMMSCase(
        name=name,
        slab_length=slab_length,
        a0=a0, a1=a1, b0=b0, k=k,
        c_groups=np.array([1.0]),
        sigma_t_g=np.array([sigma_t]),
        sigma_s_matrix=np.array([[sigma_s]]),
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


def _make_2g_asymmetric_mixture(
    sigma_t_g: np.ndarray, sigma_s_matrix: np.ndarray,
) -> "Mixture":
    r"""Build a homogeneous 2g non-fissile mixture with a DOWNSCATTER-only
    asymmetric :math:`\Sigma_s` (the 1-group-degeneracy rule — 2g is
    mandatory; the asymmetry keeps the ERR-002 ``SigSᵀ`` transpose hazard live).

    ``sigma_s_matrix[g_from, g_to]`` (ORPHEUS convention).  Absorption
    :math:`\Sigma_{a,g} = \Sigma_{t,g} - \sum_{g'}\Sigma_s[g, g']`
    (row-sum out-scatter) must be positive.
    """
    ng = sigma_t_g.shape[0]
    out_scatter = sigma_s_matrix.sum(axis=1)          # row sum per from-group
    sig_a = sigma_t_g - out_scatter
    if np.any(sig_a <= 0):
        raise ValueError(
            f"Need Σ_a > 0 per group: Σ_t={sigma_t_g}, "
            f"out-scatter={out_scatter}, Σ_a={sig_a}."
        )
    return Mixture(
        SigC=sig_a,
        SigL=np.zeros(ng),
        SigF=np.zeros(ng),
        SigP=np.zeros(ng),
        SigT=sigma_t_g,
        SigS=[csr_matrix(sigma_s_matrix)],   # P0 only
        Sig2=[csr_matrix(np.zeros((ng, ng)))],
        chi=np.zeros(ng),
    )


def build_slab_2g_nonvacuum_mms_case(
    slab_length: float = 5.0,
    a0: float = 0.5,
    a1: float = 0.25,
    b0: float = 0.3,
    c_groups: tuple[float, float] = (1.0, 0.4),
    n_wavelengths: float = 1.5,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_slab_2g_nonvacuum",
) -> SNSlabNonVacuumMMSCase:
    r"""Build the 2-group asymmetric-Σs non-vacuum slab MMS case (T2).

    DOWNSCATTER-only asymmetric :math:`\Sigma_s` (g0→g1 ≠ g1→g0=0) so
    the cross-group transfer is non-trivial and the ERR-002 transpose
    hazard is live (the 1-group-degeneracy rule — the MANDATORY ≥2g row).
    Per-group amplitudes :math:`\mathbf c = (1.0, 0.4)` make the
    group-coupling discriminating: the manufactured source for group 1
    carries a :math:`\Sigma_s[0,1] A_0` in-scatter term feeding from
    group 0's amplitude.
    """
    sigma_t_g = np.array([1.0, 1.5])
    # SigS[g_from, g_to]: g0→g0 self, g0→g1 downscatter, g1→g1 self;
    # g1→g0 upscatter = 0 (asymmetric, downscatter-only).
    sigma_s_matrix = np.array([
        [0.3, 0.2],   # g0 → {g0, g1}
        [0.0, 0.6],   # g1 → {g0, g1}
    ])
    materials = {mat_id: _make_2g_asymmetric_mixture(sigma_t_g, sigma_s_matrix)}
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
    k = 2.0 * np.pi * n_wavelengths / slab_length
    return SNSlabNonVacuumMMSCase(
        name=name,
        slab_length=slab_length,
        a0=a0, a1=a1, b0=b0, k=k,
        c_groups=np.asarray(c_groups, dtype=float),
        sigma_t_g=sigma_t_g,
        sigma_s_matrix=sigma_s_matrix,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


@dataclass(frozen=True)
class SNSphericalNonVacuumMMSCase:
    r"""NON-VACUUM prescribed-inflow MMS for 1D spherical SN (Phase 4 / O.2b 4.6).

    Ansatz :math:`\psi_n(r) = (A(r) + \mu_n B(r))/W` with

    .. math::

       A(r) = a_0 + a_1\sin(kr),\qquad B(r) = \frac{r}{R}\bigl[b_0 + b_1\cos(kr)\bigr].

    HAZARD H1: the :math:`(r/R)` prefactor forces :math:`B(0)=0` so the
    angular redistribution :math:`(1-\mu^2)B/r` is REGULAR at the pole,
    while :math:`B(R)\neq 0` gives the non-vacuum angular structure at
    the outer inflow face.  :math:`a_0>0` makes :math:`A(R)\neq 0`
    (non-vacuum) and :math:`A(0)=a_0` finite (A has no :math:`1/r`
    companion).

    Unlike the vacuum :class:`SNSphericalAnisotropicMMSCase`
    (:math:`B=(r/R)(1-r/R)\cos` → B(R)=0), THIS case is non-vacuum at
    r=R — lighting the prescribed-inflow ``q.boundary`` path on the
    curvilinear geometry (the Mode-7 mandatory companion to the slab).

    The scalar flux is :math:`\phi(r) = A(r)` (since
    :math:`\sum_n w_n \mu_n = 0`).  The closed-form source REUSES the
    spherical-operator residual of :class:`SNSphericalAnisotropicMMSCase`
    (Cardinal Rule 2) — only :math:`A,\,B` differ; SymPy re-proves it via
    :func:`derive_nonvacuum_spherical_mms`.

    Pole r=0 is the symmetry BC (not a face); the only boundary face is
    r=R (``xmax``).
    """

    name: str
    sigma_t: float
    sigma_s: float
    radius: float
    a0: float
    a1: float
    b0: float
    b1: float
    k: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: Quadrature
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-spherical",
        "sn-mms-nonvacuum-sph-psi",
        "sn-mms-nonvacuum-sph-qext",
    )

    # ── Reference solution shapes (Branch 2 numpy) ───────────────────

    def A(self, r: np.ndarray) -> np.ndarray:
        r""":math:`A(r) = a_0 + a_1\sin(kr)`."""
        return self.a0 + self.a1 * np.sin(self.k * np.asarray(r))

    def Ap(self, r: np.ndarray) -> np.ndarray:
        r""":math:`A'(r) = a_1 k\cos(kr)`."""
        return self.a1 * self.k * np.cos(self.k * np.asarray(r))

    def B(self, r: np.ndarray) -> np.ndarray:
        r""":math:`B(r) = (r/R)[b_0 + b_1\cos(kr)]`. B(0)=0 (pole-regular)."""
        rr = np.asarray(r)
        return (rr / self.radius) * (self.b0 + self.b1 * np.cos(self.k * rr))

    def Bp(self, r: np.ndarray) -> np.ndarray:
        r""":math:`B'(r)` by the product rule:

        .. math::

           B'(r) = \frac{1}{R}[b_0 + b_1\cos(kr)]
                 - \frac{r}{R}\,b_1 k\sin(kr).
        """
        rr = np.asarray(r)
        R = self.radius
        return (
            (self.b0 + self.b1 * np.cos(self.k * rr)) / R
            - (rr / R) * self.b1 * self.k * np.sin(self.k * rr)
        )

    def phi_exact(self, r: np.ndarray) -> np.ndarray:
        r"""Reference scalar flux :math:`\phi(r) = A(r)`."""
        return self.A(r)

    def psi_exact(self, r: np.ndarray, mu_n: float) -> np.ndarray:
        r"""Reference angular flux :math:`\psi_n(r) = A(r) + \mu_n B(r)`
        (WITHOUT the :math:`1/W` factor)."""
        return self.A(r) + self.B(r) * mu_n

    # ── Mesh + source construction ───────────────────────────────────

    def build_mesh(self, n_cells: int) -> Mesh1D:
        r"""Spherical mesh; r=0 symmetry (reflective), r=R VACUUM — the
        prescribed inflow at r=R is the ``q.boundary`` slot (NOT a mesh
        BC)."""
        edges = np.linspace(0.0, self.radius, n_cells + 1)
        mat_ids = np.full(n_cells, self.mat_id, dtype=int)
        return Mesh1D(
            edges=edges, mat_ids=mat_ids,
            coord=CoordSystem.SPHERICAL,
            bc_left=BC("reflective"),   # r=0 symmetry
            bc_right=BC("vacuum"),      # r=R: prescribed inflow via q.boundary
        )

    def external_source(self, mesh: Mesh1D) -> np.ndarray:
        r"""Per-ordinate-density bulk source on ``mesh``. Shape
        ``(N, ng=1, nx, 1)``.

        .. math::

           Q^{\rm ext}_n(r) = \mu_n A'(r) + \mu_n^2 B'(r)
                            + (1-\mu_n^2)\,\frac{B(r)}{r}
                            + (\Sigma_t-\Sigma_s) A(r)
                            + \Sigma_t\,\mu_n B(r).

        Bit-equal to the SymPy closed form
        (:func:`derive_nonvacuum_spherical_mms`), divided by
        :math:`\sum_n w_n` at the producer boundary (Pattern 7).
        """
        r = mesh.centers                              # (nx,)
        A_ = self.A(r)
        Ap_ = self.Ap(r)
        B_ = self.B(r)
        Bp_ = self.Bp(r)
        mu = self.quadrature.mu_x                     # (N,)
        sum_w = float(self.quadrature.weights.sum())

        streaming_iso = mu[:, None] * Ap_[None, :]               # μ A'
        streaming_aniso = (mu[:, None] ** 2) * Bp_[None, :]      # μ² B'
        redistribution = (1.0 - mu[:, None] ** 2) * (B_ / r)[None, :]  # (1-μ²)B/r
        removal_iso = (self.sigma_t - self.sigma_s) * A_[None, :]  # (Σt-Σs) A
        removal_aniso = self.sigma_t * mu[:, None] * B_[None, :]   # Σt μ B

        Q = (streaming_iso + streaming_aniso + redistribution
             + removal_iso + removal_aniso) / sum_w    # (N, nx)
        return Q[:, None, :]                     # (N, 1, nx, 1)

    def prescribed_inflow(self, sn_mesh):
        r"""The ``q.boundary`` prescribed-inflow at r=R — a
        :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`.

        The r=R face's inflow ordinate slots (μ<0) carry
        :math:`\gamma_-\psi = (A(R) + \mu_n B(R))/W`; r=0 is the symmetry
        BC, not a face. Materialised via the ergonomic
        :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
        generator (full ``(N, 1)``; the generator keeps only the inflow
        ordinates).
        """
        from orpheus.transport.source_sinks import AngularBoundarySourceSink

        W = float(self.quadrature.weights.sum())
        mu = self.quadrature.mu_x
        R = self.radius
        A_R = float(self.A(np.array([R]))[0])
        B_R = float(self.B(np.array([R]))[0])
        vals = ((A_R + mu * B_R) / W)[:, None]        # (N, ng=1)
        return AngularBoundarySourceSink.prescribed_inflow(sn_mesh, {"xmax": vals})

def build_sphere_nonvacuum_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    radius: float = 5.0,
    a0: float = 0.5,
    a1: float = 0.25,
    b0: float = 0.3,
    b1: float = 0.2,
    n_ordinates: int = 16,
    mat_id: int = 1,
    name: str = "sn_mms_sphere_nonvacuum",
) -> SNSphericalNonVacuumMMSCase:
    r"""Build the canonical non-vacuum spherical MMS case (T3 / T3g).

    :math:`k = \pi/(2R)` so :math:`A(R) = a_0 + a_1 = 0.75`,
    :math:`B(R) = b_0 = 0.3` (matching the symbolic
    :func:`_nonvacuum_spherical_AB` coefficients — the same a0,a1,b0,b1
    and ``kR = π/2`` baked into the SymPy shapes, so the L1
    cross-check holds).  HAZARD H1: :math:`B(0)=0` from the :math:`(r/R)`
    prefactor (pole-regular).
    """
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = Quadrature.gauss_legendre(n_ordinates=n_ordinates)
    k = np.pi / (2.0 * radius)
    return SNSphericalNonVacuumMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        radius=radius,
        a0=a0, a1=a1, b0=b0, b1=b1, k=k,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


def build_nonvacuum_fixed_source(case, sn_mesh) -> "TimedFullField":
    r"""The composite fixed-source RHS :math:`q = q_{\rm bulk} \oplus q_\partial`
    for a non-vacuum MMS ``case``.

    Bundles the manufactured bulk source (``case.external_source(mesh)``) and
    the prescribed-inflow boundary (``case.prescribed_inflow(sn_mesh)``) into
    the single :class:`~orpheus.transport.timed_full_field.TimedFullField`
    that :func:`~orpheus.sn.solver.solve_sn_fixed_source` consumes — the
    ergonomic one-call non-vacuum source (no manual operator-triple bypass).

    Generic over the ``(external_source(mesh), prescribed_inflow(sn_mesh))``
    protocol: ONE definition shared by every non-vacuum case
    (:class:`SNSlabNonVacuumMMSCase`, :class:`SNSphericalNonVacuumMMSCase`),
    rather than a per-case method twin (Cardinal Rule 2). ``sn_mesh.mesh``
    supplies the underlying mesh for the bulk source.
    """
    from orpheus.transport.source_sinks import AngularSourceSink
    from orpheus.transport.timed_full_field import TimedFullField

    return TimedFullField(
        interior=AngularSourceSink(values=case.external_source(sn_mesh.mesh), space=sn_mesh.angular_bulk_space),
        boundary=case.prescribed_inflow(sn_mesh),
    )


@dataclass(frozen=True)
class SNCylindricalAnisotropicMMSCase:
    r"""Anisotropic-ansatz MMS fixed-source problem for 1D cylindrical SN.

    Activates the azimuthal redistribution term
    :math:`-(1/r)\,\partial(\xi\psi)/\partial\varphi` that the
    isotropic sibling :class:`SNCylindricalMMSCase` cancels. The
    closed-form :math:`Q^{\rm ext}_n(r)` is

    .. math::

        Q^{\rm ext}_n(r) =
            \eta_n A'(r) + \eta_n^2 B'(r)
          + \xi_n^2\,\frac{B(r)}{r}
          + (\Sigma_t - \Sigma_s) A(r) + \Sigma_t\,\eta_n B(r),

    with the same :math:`A(r),\,B(r)` as the spherical case. The
    radial direction cosine for cylindrical 1D is :math:`\eta_n =
    \sin\theta_n\cos\varphi_n`; the partner :math:`\xi_n =
    \sin\theta_n\sin\varphi_n` enters the redistribution term.
    Both are exposed by the :class:`Quadrature` ``mu_x`` and ``mu_y``
    views — ``@property`` reads of the wrapped measure's node
    columns, not cached fields (the per-family ``ProductQuadrature``
    adapter that stored them was retired in R-1 Phase A detour-C) —
    family-agnostic, so they read identically off the σ_y-folded
    rule the builder defaults to since the 6.3 flip.

    The manufactured fields live in the quotient's function space:
    :math:`\psi = A + B\eta` carries no :math:`\xi`-odd content
    (:math:`\eta` is σ_y-invariant), and the source enters through
    :math:`\xi_n^2` only — both even under :math:`\xi \to -\xi`, so
    restriction to the folded rule loses nothing.
    """

    name: str
    sigma_t: float
    sigma_s: float
    radius: float
    materials: dict[int, "Mixture"]
    mat_id: int
    quadrature: Quadrature
    tolerance: str = "O(h^2)"
    equation_labels: tuple[str, ...] = (
        "transport-cylindrical",
        "sn-mms-cylindrical-aniso-psi",
        "sn-mms-cylindrical-aniso-qext",
    )

    # ── Reference solution shapes ────────────────────────────────────

    def A(self, r: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * np.asarray(r) / self.radius)

    def Ap(self, r: np.ndarray) -> np.ndarray:
        R = self.radius
        return (np.pi / R) * np.cos(np.pi * np.asarray(r) / R)

    def B(self, r: np.ndarray) -> np.ndarray:
        R = self.radius
        rr = np.asarray(r) / R
        return rr * (1.0 - rr) * np.cos(np.pi * rr)

    def Bp(self, r: np.ndarray) -> np.ndarray:
        R = self.radius
        rr = np.asarray(r) / R
        return (
            (1.0 - 2.0 * rr) * np.cos(np.pi * rr) / R
            - (np.pi / R) * rr * (1.0 - rr) * np.sin(np.pi * rr)
        )

    def phi_exact(self, r: np.ndarray) -> np.ndarray:
        r"""Reference scalar flux :math:`\phi(r) = A(r)`.

        :math:`\sum_n w_n \eta_n = 0` kills the :math:`B\eta` term:
        the roots-of-unity azimuthal circle integrates
        :math:`\cos\varphi` to zero, and the σ_y fold preserves the
        sum exactly (:math:`\eta` is mirror-invariant, each kept
        orbit carries the whole orbit weight :math:`2w`)."""
        return self.A(r)

    def psi_exact(self, r: np.ndarray, eta_n: float) -> np.ndarray:
        r"""Reference angular flux :math:`\psi_n(r) = A(r) + B(r)\,\eta_n`
        (without the :math:`1/W` factor)."""
        return self.A(r) + self.B(r) * eta_n

    # ── Mesh + source construction ───────────────────────────────────

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
        r"""Per-ordinate external source on ``mesh``. Shape
        ``(N, ng=1, nx)``.

        .. math::

           Q^{\rm ext}_n(r) =
              \eta_n A'(r) + \eta_n^2 B'(r)
            + \xi_n^2\,B(r)/r
            + (\Sigma_t - \Sigma_s) A(r) + \Sigma_t \eta_n B(r).

        :math:`\eta_n` is the radial direction cosine
        (``quadrature.mu_x``), :math:`\xi_n` is the azimuthal cosine
        (``quadrature.mu_y``). Bit-equal to the SymPy form derived
        in :func:`derive_cylindrical_anisotropic_mms`.
        """
        r = mesh.centers
        A_ = self.A(r)
        Ap_ = self.Ap(r)
        B_ = self.B(r)
        Bp_ = self.Bp(r)
        eta = self.quadrature.eta       # (N,) — radial cosine
        xi = self.quadrature.xi        # (N,) — azimuthal cosine
        sum_w = float(self.quadrature.weights.sum())

        streaming_iso = eta[:, None] * Ap_[None, :]              # η A'
        streaming_aniso = (eta[:, None] ** 2) * Bp_[None, :]     # η² B'
        redistribution = (xi[:, None] ** 2) * (B_ / r)[None, :]  # ξ² B/r
        removal_iso = (self.sigma_t - self.sigma_s) * A_[None, :]
        removal_aniso = self.sigma_t * eta[:, None] * B_[None, :]

        # R-1 Step 4 A1 — emit per-ordinate density (Pattern 7).
        Q = (streaming_iso + streaming_aniso + redistribution
             + removal_iso + removal_aniso) / sum_w
        return Q[:, None, :]


def build_cylindrical_anisotropic_mms_case(
    sigma_t: float = 1.0,
    sigma_s: float = 0.5,
    radius: float = 5.0,
    n_mu: int = 4,
    n_phi: int = 8,
    mat_id: int = 1,
    name: str = "sn_mms_cylindrical_aniso",
) -> SNCylindricalAnisotropicMMSCase:
    r"""Build the canonical anisotropic 1D cylindrical MMS case.

    Defaults match :func:`build_cylindrical_mms_case` (PARENT-count
    semantics; even ``n_phi`` required). Pairing both cases narrows
    down failures: a passing isotropic + failing anisotropic
    pinpoints the azimuthal redistribution path."""
    materials = {mat_id: _make_1g_mixture(sigma_t, sigma_s)}
    quadrature = Quadrature.folded_product(n_mu=n_mu, n_phi=n_phi)
    return SNCylindricalAnisotropicMMSCase(
        name=name,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        radius=radius,
        materials=materials,
        mat_id=mat_id,
        quadrature=quadrature,
    )


def _build_spherical_anisotropic_continuous_reference() -> ContinuousReferenceSolution:
    """Phase-3.6 continuous reference for spherical anisotropic MMS."""
    mms = build_spherical_anisotropic_mms_case()
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
            citation=(
                "Bailey-Morel-Chang 2010, NSE 165(2):149-169 (curvilinear SN "
                "angular differencing; 'Bailey 2009' here until "
                "2026-08-27 was the wrong-paper citation retracted "
                "at Issue #168 Phase B); Oberkampf & Roy 2010, "
                "Ch. 6 (MMS fundamentals); vv-principles failure "
                "mode #7 (MMS simplification bias)"
            ),
            derivation_notes=(
                "1-group spherical SN MMS with anisotropic ansatz "
                "ψ_n(r) = (A(r) + B(r) μ_n)/W, A(r)=sin(πr/R), "
                "B(r)=(r/R)(1-r/R)cos(πr/R). Activates the angular "
                "redistribution term (1-μ²)/r ∂ψ/∂μ that the "
                "isotropic ansatz cancels by construction. "
                "Manufactured source: Q_n = μ_n A' + μ_n² B' + "
                "(1-μ_n²) B/r + (Σ_t-Σ_s) A + Σ_t μ_n B. "
                "Detects ERR-026-class bugs (curvilinear sweep "
                "WDD wrong fixed point) under O(h²) refinement."
            ),
            sympy_expression=(
                r"Q^{\rm ext}_n(r) = \mu_n A'(r) + \mu_n^2 B'(r) "
                r"+ (1 - \mu_n^2)\,B(r)/r "
                r"+ (\Sigma_t - \Sigma_s) A(r) "
                r"+ \Sigma_t\,\mu_n B(r)"
            ),
            precision_digits=None,
        ),
        k_eff=None, psi=None,
        equation_labels=mms.equation_labels,
        vv_level="L1",
        description=(
            "1-group spherical SN MMS with μ-linear anisotropic "
            "ansatz — Phase 3.6 continuous reference. Activates "
            "the angular redistribution term."
        ),
        tolerance="O(h^2)",
    )


def _build_cylindrical_anisotropic_continuous_reference() -> ContinuousReferenceSolution:
    """Phase-3.6 continuous reference for cylindrical anisotropic MMS."""
    mms = build_cylindrical_anisotropic_mms_case()
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
            citation=(
                "Bailey-Morel-Chang 2010, NSE 165(2):149-169 (curvilinear SN "
                "angular differencing; 'Bailey 2009' here until "
                "2026-08-27 was the wrong-paper citation retracted "
                "at Issue #168 Phase B); Oberkampf & Roy 2010, "
                "Ch. 6 (MMS fundamentals); vv-principles failure "
                "mode #7 (MMS simplification bias)"
            ),
            derivation_notes=(
                "1-group cylindrical SN MMS with anisotropic ansatz "
                "ψ_n(r) = (A(r) + B(r) η_n)/W, η_n = sinθ_n cosφ_n. "
                "Activates the azimuthal redistribution term "
                "-(1/r) ∂(ξψ)/∂φ that the isotropic ansatz cancels. "
                "Manufactured source: Q_n = η_n A' + η_n² B' + "
                "ξ_n² B/r + (Σ_t-Σ_s) A + Σ_t η_n B. "
                "Detects ERR-026-class bugs in the cylindrical "
                "azimuthal sweep under O(h²) refinement."
            ),
            sympy_expression=(
                r"Q^{\rm ext}_n(r) = \eta_n A'(r) + \eta_n^2 B'(r) "
                r"+ \xi_n^2\,B(r)/r "
                r"+ (\Sigma_t - \Sigma_s) A(r) "
                r"+ \Sigma_t\,\eta_n B(r)"
            ),
            precision_digits=None,
        ),
        k_eff=None, psi=None,
        equation_labels=mms.equation_labels,
        vv_level="L1",
        description=(
            "1-group cylindrical SN MMS with η-linear anisotropic "
            "ansatz — Phase 3.6 continuous reference. Activates "
            "the azimuthal redistribution term."
        ),
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
        _build_spherical_anisotropic_continuous_reference(),
        _build_cylindrical_anisotropic_continuous_reference(),
    ]
