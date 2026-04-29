r"""Unified flat-source collision-probability construction.

Level-3 sibling of :mod:`~orpheus.derivations.peierls_geometry`.
See :doc:`/theory/peierls_unified` §§11-17 for the end-to-end
derivation of the three-tier kernel hierarchy and the factorisation

.. math::

    P_{ij}\,\Sigma_{t,i}\,V_i \;=\;
        2\!\int_{0}^{R}\! W(y)\,
        \Delta^{2}[\mathcal F_{d}]\bigl(\tau_{i},\tau_{j};\mathrm{gap}(y)\bigr)\,
        \mathrm d y

that isolates the per-geometry data into four primitives
(``kernel_F3``, ``kernel_F3_at_zero``, ``outer_y_weight``,
``surface_area``) and a single flag (``kind``). The
geometry-invariant operator

.. math::

    \Delta^{2}[\mathcal F](\tau_i,\tau_j;\mathrm{gap})
        \;=\;\mathcal F(\mathrm{gap})
           - \mathcal F(\mathrm{gap}+\tau_i)
           - \mathcal F(\mathrm{gap}+\tau_j)
           + \mathcal F(\mathrm{gap}+\tau_i+\tau_j)

is a module-level free function :func:`_second_difference` that takes
a callable kernel — by design it carries no geometry state, because
the identity holds for any second antiderivative of
:math:`\mathcal F''(\tau)`.

The per-geometry kernel choices are:

- ``slab`` — :math:`E_{3}(\tau)` via ``scipy.special.expn(3, x)``
  (double-precision, ~1e-16 accuracy).
- ``cylinder-1d`` — :math:`\mathrm{Ki}_{3}(\tau)` via a Chebyshev
  interpolant of :math:`e^{\tau}\,\mathrm{Ki}_{3}(\tau)` built from
  :func:`~._kernels.ki_n_mp` at 30 dps on 64 Chebyshev-Gauss nodes
  over :math:`[0, 50]`. Scaling by :math:`e^{\tau}` converts the
  exponentially-decaying tail into a slowly-varying function —
  reached by a degree-63 polynomial to ~:math:`10^{-6}` relative
  accuracy (compared with the legacy :class:`BickleyTables` at
  ~:math:`10^{-3}`). The :class:`BickleyTables` class is retired
  as of this commit (:issue:`94`).
- ``sphere-1d`` — :math:`e^{-\tau}` via ``np.exp(-tau)``
  (double-precision, machine-accurate).

The three pre-built :class:`FlatSourceCPGeometry` singletons
:data:`SLAB`, :data:`CYLINDER_1D`, :data:`SPHERE_1D` are consumed by
:func:`build_cp_matrix` and by the thin facade modules
:mod:`~orpheus.derivations.cp_slab`, :mod:`~.cp_cylinder`,
:mod:`~.cp_sphere`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from ._kernels import chord_half_lengths, e3_vec, ki_n_mp
from ._quadrature_recipes import chord_quadrature


# ═══════════════════════════════════════════════════════════════════════
# Geometry-invariant second-difference operator (§12 of peierls_unified)
# ═══════════════════════════════════════════════════════════════════════

def _second_difference(
    kernel: Callable[[np.ndarray], np.ndarray],
    gap: np.ndarray,
    tau_i: np.ndarray,
    tau_j: np.ndarray,
) -> np.ndarray:
    r"""The geometry-invariant operator
    :math:`\Delta^{2}[\mathcal F](\tau_i, \tau_j;\,\mathrm{gap})
    = \mathcal F(\mathrm{gap})
      - \mathcal F(\mathrm{gap}+\tau_i)
      - \mathcal F(\mathrm{gap}+\tau_j)
      + \mathcal F(\mathrm{gap}+\tau_i+\tau_j)`.

    See :eq:`cp-second-difference-operator` for the derivation."""
    return (kernel(gap)
            - kernel(gap + tau_i)
            - kernel(gap + tau_j)
            + kernel(gap + tau_i + tau_j))


# ═══════════════════════════════════════════════════════════════════════
# Per-geometry kernel callables
# ═══════════════════════════════════════════════════════════════════════

_KI3_TAU_MAX: float = 50.0
_KI3_DEG: int = 63          # 64 Chebyshev-Lobatto nodes
_KI3_DPS: int = 30


@lru_cache(maxsize=1)
def _ki3_scaled_cheb() -> "np.polynomial.Chebyshev":
    r"""Chebyshev interpolant of the SCALED kernel
    :math:`f(\tau) = e^{\tau}\,\mathrm{Ki}_3(\tau)` on
    :math:`[0, \tau_{\max}]`.

    Scaling by :math:`e^{\tau}` converts the exponentially-decaying
    tail of :math:`\mathrm{Ki}_3` into a slowly-varying function that
    a degree-63 polynomial fits to ~:math:`10^{-6}` relative accuracy.
    Built once at first access via :func:`lru_cache`; uses
    :func:`~._kernels.ki_n_mp` at 30 dps. Build time: ~0.3 s.
    """
    def func_scaled(tau: np.ndarray) -> np.ndarray:
        return np.array([
            float(ki_n_mp(3, float(t), _KI3_DPS)) * float(np.exp(t))
            for t in tau
        ])
    return np.polynomial.Chebyshev.interpolate(
        func_scaled, deg=_KI3_DEG, domain=[0.0, _KI3_TAU_MAX],
    )


def _ki3_mp(tau: np.ndarray) -> np.ndarray:
    r"""Double-precision :math:`\mathrm{Ki}_3(\tau)` via the Chebyshev
    interpolant of :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)`.

    Clamps :math:`\tau > \tau_{\max}` to 0 (:math:`\mathrm{Ki}_3(50)
    \approx 3\times 10^{-23}`, negligible)."""
    poly = _ki3_scaled_cheb()
    tau = np.asarray(tau, dtype=float)
    out = np.zeros_like(tau, dtype=float)
    mask = tau <= _KI3_TAU_MAX
    if np.any(mask):
        out[mask] = poly(tau[mask]) * np.exp(-tau[mask])
    return out


def _exp_kernel(tau: np.ndarray) -> np.ndarray:
    r"""Vectorised :math:`e^{-\tau}` for the sphere branch."""
    return np.exp(-np.asarray(tau, dtype=float))


# ═══════════════════════════════════════════════════════════════════════
# FlatSourceCPGeometry — Phase B.2 Level-3 abstraction (§14)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FlatSourceCPGeometry:
    r"""Level-3 flat-source CP geometry.

    One instance per geometry; four primitives + one flag fully
    specify the flat-source CP construction. Sibling to
    :class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
    (same family, different kernel level). The three pre-built
    singletons :data:`SLAB`, :data:`CYLINDER_1D`, :data:`SPHERE_1D`
    are the conventional consumption path."""

    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("slab", "cylinder-1d", "sphere-1d"):
            raise ValueError(f"Unsupported geometry kind {self.kind!r}")

    # ── Level-3 kernel F_3(τ) ─────────────────────────────────────────

    def kernel_F3(self, tau: np.ndarray) -> np.ndarray:
        r"""Level-3 kernel :math:`F_3(\tau)`: :math:`E_3` (slab),
        :math:`\mathrm{Ki}_3` (cyl), :math:`e^{-\tau}` (sph)."""
        if self.kind == "slab":
            return e3_vec(np.asarray(tau, dtype=float))
        if self.kind == "cylinder-1d":
            return _ki3_mp(tau)
        return _exp_kernel(tau)

    def kernel_F3_at_zero(self) -> float:
        r""":math:`F_3(0)`: :math:`1/2` (slab, :math:`E_3(0)`),
        :math:`\pi/4` (cyl, :math:`\mathrm{Ki}_3(0)`), :math:`1` (sph)."""
        if self.kind == "slab":
            return 0.5
        if self.kind == "cylinder-1d":
            return float(np.pi / 4.0)
        return 1.0

    # ── Outer-integration weight W(y) ────────────────────────────────

    def outer_y_weight(self, y_pts: np.ndarray) -> np.ndarray:
        r"""Weight in the :math:`\mathrm d y` outer integral.

        - Slab: n/a (trivial 1-point "y-quadrature" at :math:`y=0`;
          this method returns 1 to pass through unchanged).
        - Cylinder: 1 (weight is purely :math:`\mathrm d y`).
        - Sphere: :math:`y` (spherical area element
          :math:`2\pi y\,\mathrm d y` minus the :math:`2\pi` absorbed
          into :meth:`surface_area`).
        """
        y_pts = np.asarray(y_pts, dtype=float)
        if self.kind == "sphere-1d":
            return y_pts
        return np.ones_like(y_pts)

    # ── Cell surface area (white-BC closure normalisation) ───────────

    def surface_area(self, R: float) -> float:
        """White-BC closure divisor:

        - Slab: 1 (per unit transverse area).
        - Cylinder: :math:`2\\pi R` (lateral surface per unit z).
        - Sphere: :math:`4\\pi R^{2}`.
        """
        if self.kind == "slab":
            return 1.0
        if self.kind == "cylinder-1d":
            return 2.0 * np.pi * R
        return 4.0 * np.pi * R * R

    # ── y-quadrature presence flag ───────────────────────────────────

    @property
    def has_y_quadrature(self) -> bool:
        """False for slab (trivial 1-point rule), True for cyl/sph."""
        return self.kind != "slab"


# Singletons (convention: lower-case import path, upper-case constants)
SLAB = FlatSourceCPGeometry(kind="slab")
CYLINDER_1D = FlatSourceCPGeometry(kind="cylinder-1d")
SPHERE_1D = FlatSourceCPGeometry(kind="sphere-1d")


# ═══════════════════════════════════════════════════════════════════════
# Shared y-quadrature with breakpoints at each radius
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Unified flat-source CP matrix builder
# ═══════════════════════════════════════════════════════════════════════

def build_cp_matrix(
    geom: FlatSourceCPGeometry,
    sig_t_all: np.ndarray,
    radii_or_thicknesses: np.ndarray,
    volumes: np.ndarray,
    R_cell: float,
    n_quad_y: int = 64,
) -> np.ndarray:
    r"""Unified flat-source CP matrix constructor.

    Produces :math:`P_\infty` of shape ``(N_reg, N_reg, ng)`` via the
    second-difference formula with white-BC closure, for any of the
    three geometries.

    Parameters
    ----------
    geom : FlatSourceCPGeometry
        Selects the Level-3 kernel, outer weight, and surface-area
        normalisation (one of :data:`SLAB`, :data:`CYLINDER_1D`,
        :data:`SPHERE_1D`).
    sig_t_all : (N_reg, ng) ndarray
        Total cross section per region and group.
    radii_or_thicknesses : (N_reg,) ndarray
        Outer radii (cyl/sph) OR region thicknesses (slab).
    volumes : (N_reg,) ndarray
        Region volumes (or thicknesses for slab; V = t in 1-D).
    R_cell : float
        Outer radius (cyl/sph) OR total thickness (slab). Used for
        the white-BC surface normalisation.
    n_quad_y : int, default 64
        Gauss-Legendre order per panel for the outer y-quadrature.
        Ignored for the slab (trivial 1-point rule).

    Returns
    -------
    P_inf : (N_reg, N_reg, ng) ndarray
        Infinite-lattice collision probability matrix.
    """
    radii_or_thicknesses = np.asarray(radii_or_thicknesses, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    sig_t_all = np.asarray(sig_t_all, dtype=float)

    N_reg = len(radii_or_thicknesses)
    ng = sig_t_all.shape[1]

    # Outer quadrature setup
    if geom.has_y_quadrature:
        # Issue #134: route through `chord_quadrature` for kink-aware
        # subdivision at shell radii + visibility-cone-upper substitution
        # absorbing the √(r_k² − y²) chord-Jacobian into the per-panel
        # weights. With ``split_first_panel=True`` (default) the first
        # impact-parameter panel ``[0, r_1]`` is split at ``r_1/2`` to
        # dodge the upper-variant degeneracy at y=0; downstream callers
        # consume the (y_pts, y_wts) pair through `chord_half_lengths`,
        # which has no compensating 1/y singularity.
        q = chord_quadrature(radii_or_thicknesses, n_per_panel=n_quad_y)
        y_pts = q.pts
        y_wts = q.wts * geom.outer_y_weight(q.pts)
        chords = chord_half_lengths(radii_or_thicknesses, y_pts)
    else:
        # Slab: trivial 1-point rule at y=0 with weight 0.25 so that
        # the shared ``2 * <y_wts, dd + dc>`` reduces to the legacy
        # ``0.5 * (dd + dc)``. Chords are the thicknesses (constant).
        y_pts = np.zeros(1)
        y_wts = np.array([0.25])
        chords = radii_or_thicknesses[:, None].copy()  # (N_reg, 1)

    n_y = len(y_pts)
    kernel_zero = geom.kernel_F3(np.zeros(n_y))

    P_inf_g = np.empty((N_reg, N_reg, ng))

    for g in range(ng):
        sig_t_g = sig_t_all[:, g]
        tau = sig_t_g[:, None] * chords  # (N_reg, n_y)

        bnd_pos = np.zeros((N_reg + 1, n_y))
        for k in range(N_reg):
            bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]

        rcp = np.zeros((N_reg, N_reg))

        for i in range(N_reg):
            tau_i = tau[i, :]
            sti = sig_t_g[i]
            if sti == 0:
                continue

            # Self-same contribution: diagonal collision rate
            self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
                kernel_zero - geom.kernel_F3(tau_i)
            )
            rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

            for j in range(N_reg):
                tau_j = tau[j, :]

                # Same-side gap (skipped for self-same i==j)
                if j > i:
                    gap_d = np.maximum(bnd_pos[j, :] - bnd_pos[i + 1, :], 0.0)
                elif j < i:
                    gap_d = np.maximum(bnd_pos[i, :] - bnd_pos[j + 1, :], 0.0)
                else:
                    gap_d = None

                if gap_d is not None:
                    dd = _second_difference(
                        geom.kernel_F3, gap_d, tau_i, tau_j,
                    )
                else:
                    dd = np.zeros(n_y)

                # Through-centre (reflected / across-origin) branch —
                # present for all three geometries.
                gap_c = bnd_pos[i, :] + bnd_pos[j, :]
                dc = _second_difference(
                    geom.kernel_F3, gap_c, tau_i, tau_j,
                )

                rcp[i, j] += 2.0 * np.dot(y_wts, dd + dc)

        # Normalise to P_cell = rcp / (Σ_t · V)
        P_cell = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            if sig_t_g[i] * volumes[i] > 0:
                P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * volumes[i])

        # White-BC closure: P_out escapes, distributes via P_in
        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        S_cell = geom.surface_area(R_cell)
        P_in = sig_t_g * volumes * P_out / S_cell
        P_inout = max(1.0 - P_in.sum(), 0.0)

        P_inf = P_cell.copy()
        if P_inout < 1.0:
            P_inf += np.outer(P_out, P_in) / (1.0 - P_inout)
        P_inf_g[:, :, g] = P_inf

    return P_inf_g
