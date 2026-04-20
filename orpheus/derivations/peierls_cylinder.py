r"""Peierls integral equation reference for cylindrical CP verification.

Cylindrical specialisation of the unified polar-form Peierls Nyström
infrastructure in :mod:`orpheus.derivations.peierls_geometry`. This
module is a THIN FACADE: it owns the cylinder-specific API names
(kept stable for backward compatibility with the Phase-4.2 tests),
the chord-form :math:`\tau^{\pm}` walker from the C2 scaffold, the
``_build_peierls_cylinder_case`` case builder, and the
``continuous_cases`` registration. Everything else — volume-kernel
assembly, Lagrange basis, angular/radial composite quadrature,
white-BC rank-1 closure, eigenvalue power iteration — dispatches
through the unified
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
with ``kind = "cylinder-1d"``.

See :doc:`/theory/peierls_unified` for the end-to-end derivation of
the unified structure. The cylinder-specific narrative (polar form,
Jacobian cancellation, rank-1 white-BC limitations) remains in
:doc:`/theory/collision_probability`.

The cylindrical Peierls equation in observer-centred polar coords:

.. math::

   \Sigma_t(r)\,\varphi(r)
     \;=\; \frac{\Sigma_t(r)}{\pi}\!
       \int_{0}^{\pi}\!\mathrm d\beta\!
       \int_{0}^{\rho_{\max}(r,\beta)}\!\!
         \mathrm{Ki}_1\!\bigl(\tau(r,\rho,\beta)\bigr)\,
         q\bigl(r'(r,\rho,\beta)\bigr)\,\mathrm d\rho
     + S_{\rm bc}(r).

The prefactor :math:`1/\pi` absorbs the :math:`1/(2\pi)` of the 2-D
Green's function (from integrating the 3-D point kernel over the
infinite z-axis) and a factor of 2 from folding
:math:`\beta \in [0, 2\pi]` to :math:`[0, \pi]` via the radially-
symmetric :math:`\pm\beta` reflection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import peierls_geometry as _pg
from ._kernels import ki_n_mp  # noqa: F401 re-exported
from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)
from ._xs_library import LAYOUTS, get_mixture, get_xs


# ═══════════════════════════════════════════════════════════════════════
# Cylinder geometry singleton (binds the unified infrastructure)
# ═══════════════════════════════════════════════════════════════════════

# Production cylinder path: ``CYLINDER_1D`` with the natural
# :math:`\mathrm{Ki}_1` kernel evaluated via :func:`ki_n_mp` /
# :func:`ki_n_float`. The cylinder-polar variant (explicit out-of-plane
# :math:`\varphi`-quadrature, formerly the production path) was
# archived 2026-04-19 as mathematically equivalent — see
# :file:`derivations/archive/peierls_cylinder_polar_assembly.py`.
GEOMETRY = _pg.CYLINDER_1D


# ═══════════════════════════════════════════════════════════════════════
# Backward-compatible public API (re-exports from peierls_geometry)
# ═══════════════════════════════════════════════════════════════════════

composite_gl_r = _pg.composite_gl_r


def composite_gl_y(
    radii: np.ndarray,
    n_panels_per_region: int,
    p_order: int,
    dps: int = 30,
):
    """Alias retained for the Phase-4.2 C2 tests (which treated the
    radial grid as a y-grid). Identical to :func:`composite_gl_r`."""
    return _pg.composite_gl_r(radii, n_panels_per_region, p_order, dps=dps)


_lagrange_basis_on_panels = _pg.lagrange_basis_on_panels


def _rho_max(r_obs: float, cos_beta: float, R: float) -> float:
    """Ray-exit distance along angle :math:`\\beta`."""
    return GEOMETRY.rho_max(r_obs, cos_beta, R)


def _optical_depth_along_ray(
    r_obs: float,
    cos_beta: float,
    sin_beta: float,  # noqa: ARG001  (kept in signature for backward compat)
    rho: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
) -> float:
    """Optical depth along the ray from :math:`r_{\\rm obs}` in direction
    :math:`\\beta` for distance :math:`\\rho`.

    The ``sin_beta`` argument is unused (the walker only needs
    :math:`\\cos\\beta`) but is kept in the signature so the Phase-4.2
    C4 tests calling with the three-argument form continue to work.
    """
    return GEOMETRY.optical_depth_along_ray(r_obs, cos_beta, rho, radii, sig_t)


def _which_annulus(r: float, radii: np.ndarray) -> int:
    return GEOMETRY.which_annulus(r, radii)


def build_volume_kernel(
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_beta: int,
    n_rho: int,
    dps: int = 30,
) -> np.ndarray:
    """Cylindrical volume Nyström kernel. Backward-compatible signature
    (``n_beta`` names the angular-integration order)."""
    return _pg.build_volume_kernel(
        GEOMETRY, r_nodes, panel_bounds, radii, sig_t,
        n_angular=n_beta, n_rho=n_rho, dps=dps,
    )


def compute_P_esc(
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_beta: int = 32,
    dps: int = 25,
) -> np.ndarray:
    """Cylindrical uncollided escape probability."""
    return _pg.compute_P_esc(
        GEOMETRY, r_nodes, radii, sig_t,
        n_angular=n_beta, dps=dps,
    )


def compute_G_bc(
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_phi: int = 32,
    dps: int = 25,
) -> np.ndarray:
    """Cylindrical surface-to-volume Green's function."""
    return _pg.compute_G_bc(
        GEOMETRY, r_nodes, radii, sig_t,
        n_surf_quad=n_phi, dps=dps,
    )


def build_white_bc_correction(
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_beta: int = 32,
    n_phi: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Rank-1 white-BC correction for the cylindrical Peierls kernel.

    Returns :math:`K_{\rm bc}[i, j] = u[i]\,v[j]` with
    :math:`u[i] = \Sigma_t(r_i)\,G_{\rm bc}(r_i) / R` and
    :math:`v[j] = r_j\,w_j\,P_{\rm esc}(r_j)`.

    .. warning::

       **Approximation level.** The rank-1 closure is exact for the
       scalar partial-current balance (:math:`J^-=J^+` is a single
       scalar by radial symmetry) but assumes isotropic-Mark angular
       distribution for the re-entering current. At the pointwise-
       Nyström level this shows up as a spread in the row-sum
       identity :math:`(K_{\rm vol}+K_{\rm bc})\cdot\mathbf 1 \approx
       \Sigma_t` that grows with inverse cell size:

       =====  ======================
       R/MFP  max \|K_tot·1 − Σ_t\|
       =====  ======================
       0.5    0.32
       1.0    0.16
       2.0    0.20
       5.0    0.12
       10     < 0.04
       =====  ======================

       White-BC :math:`k_{\rm eff}` agrees with :math:`k_\infty`
       only asymptotically:

       =====  ==========  ==========
       R/MFP  k(white)    err vs k∞
       =====  ==========  ==========
       1.0    1.19        21 %
       2.0    1.40        7 %
       5.0    1.48        2 %
       10     1.49        1 %
       =====  ==========  ==========

       Parallel to sphere issue #100. Rigorous fix requires a
       higher-rank angular decomposition of the surface currents —
       deferred follow-up.
    """
    return _pg.build_white_bc_correction(
        GEOMETRY, r_nodes, r_wts, radii, sig_t,
        n_angular=n_beta, n_surf_quad=n_phi, dps=dps,
    )


# ═══════════════════════════════════════════════════════════════════════
# τ± chord walker (C2 scaffold — kept as a geometric utility)
# ═══════════════════════════════════════════════════════════════════════

def optical_depths_pm(
    r: float,
    r_prime: float,
    y_pts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Same-side :math:`\tau^+` and through-centre :math:`\tau^-`
    optical-path branches for a chord at impact parameter :math:`y`.

    For a chord at impact parameter :math:`y`, a point at radius
    :math:`\rho \ge y` sits at signed chord position :math:`\pm s_\rho`
    with :math:`s_\rho = \sqrt{\rho^{2}-y^{2}}`.

    - **Same-side (τ⁺)**: from :math:`s_r` to :math:`s_{r'}` on the
      positive branch.
    - **Through-centre (τ⁻)**: from :math:`s_r` to :math:`-s_{r'}`.

    This primitive dates from the Phase-4.2 C2 scaffold and is kept
    for backward compatibility. The polar-form kernel builder uses
    :meth:`CurvilinearGeometry.optical_depth_along_ray` instead,
    which integrates along a directed ray — a strictly more general
    primitive than the ±-branch chord walker.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    y_pts = np.asarray(y_pts, dtype=float)
    r = float(r)
    r_prime = float(r_prime)

    s_r = np.sqrt(np.maximum(r ** 2 - y_pts ** 2, 0.0))
    s_rp = np.sqrt(np.maximum(r_prime ** 2 - y_pts ** 2, 0.0))

    N = len(radii)
    s_breaks = np.zeros((N + 1, len(y_pts)))
    for k in range(N):
        s_breaks[k + 1] = np.sqrt(np.maximum(radii[k] ** 2 - y_pts ** 2, 0.0))

    def _on_pos(s_lo: np.ndarray, s_hi: np.ndarray) -> np.ndarray:
        tau = np.zeros_like(s_lo)
        for k in range(N):
            lo = np.maximum(s_lo, s_breaks[k])
            hi = np.minimum(s_hi, s_breaks[k + 1])
            overlap = np.maximum(hi - lo, 0.0)
            tau += sig_t[k] * overlap
        return tau

    s_lo = np.minimum(s_r, s_rp)
    s_hi = np.maximum(s_r, s_rp)
    tau_plus = _on_pos(s_lo, s_hi)
    zero = np.zeros_like(y_pts)
    tau_minus = _on_pos(zero, s_r) + _on_pos(zero, s_rp)
    return tau_plus, tau_minus


# ═══════════════════════════════════════════════════════════════════════
# Solution container — backward-compatible alias for PeierlsSolution
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeierlsCylinderSolution:
    """Result of a Peierls Nyström solve on a 1-D radial cylinder.

    Kept as a backward-compatible facade over
    :class:`~peierls_geometry.PeierlsSolution`. New code should use
    the unified :class:`~peierls_geometry.PeierlsSolution` directly.
    """

    r_nodes: np.ndarray
    phi_values: np.ndarray
    k_eff: float | None
    cell_radius: float
    n_groups: int
    n_quad_r: int
    n_quad_y: int
    precision_digits: int
    panel_bounds: list[tuple[float, float, int, int]] | None = None

    def phi(self, r: np.ndarray, g: int = 0) -> np.ndarray:
        r = np.asarray(r, dtype=float).ravel()
        out = np.empty_like(r)
        if self.panel_bounds is None:
            return np.interp(r, self.r_nodes, self.phi_values[:, g])
        for idx, r_eval in enumerate(r):
            L = _pg.lagrange_basis_on_panels(
                self.r_nodes, self.panel_bounds, float(r_eval),
            )
            out[idx] = float(np.dot(L, self.phi_values[:, g]))
        return out


def _soln_to_cylinder(sol: _pg.PeierlsSolution) -> PeierlsCylinderSolution:
    return PeierlsCylinderSolution(
        r_nodes=sol.r_nodes,
        phi_values=sol.phi_values,
        k_eff=sol.k_eff,
        cell_radius=sol.cell_radius,
        n_groups=sol.n_groups,
        n_quad_r=sol.n_quad_r,
        n_quad_y=sol.n_quad_angular,
        precision_digits=sol.precision_digits,
        panel_bounds=sol.panel_bounds,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1G eigenvalue drivers
# ═══════════════════════════════════════════════════════════════════════

def solve_peierls_cylinder_1g(
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    *,
    boundary: str = "vacuum",
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_beta: int = 24,
    n_rho: int = 24,
    n_phi: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> PeierlsCylinderSolution:
    """1G cylindrical Peierls k-eigenvalue driver (vacuum or white BC).

    Thin wrapper over :func:`peierls_geometry.solve_peierls_1g` with
    the cylinder geometry pre-bound.
    """
    sol = _pg.solve_peierls_1g(
        GEOMETRY, radii, sig_t, sig_s, nu_sig_f,
        boundary=boundary,
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_angular=n_beta,
        n_rho=n_rho,
        n_surf_quad=n_phi,
        dps=dps,
        max_iter=max_iter,
        tol=tol,
    )
    return _soln_to_cylinder(sol)


def solve_peierls_cylinder_1g_vacuum(
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    *,
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_beta: int = 24,
    n_rho: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> PeierlsCylinderSolution:
    """Vacuum-BC alias — kept for the C5 tests."""
    return solve_peierls_cylinder_1g(
        radii, sig_t, sig_s, nu_sig_f,
        boundary="vacuum",
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_beta=n_beta, n_rho=n_rho,
        dps=dps, max_iter=max_iter, tol=tol,
    )


# ═══════════════════════════════════════════════════════════════════════
# ContinuousReferenceSolution builder
# ═══════════════════════════════════════════════════════════════════════

_MAT_IDS_CYL = {1: [2]}


def _build_peierls_cylinder_case(
    ng_key: str,
    n_regions: int,
    n_panels_per_region: int = 3,
    p_order: int = 5,
    n_beta: int = 20,
    n_rho: int = 20,
    n_phi: int = 20,
    precision_digits: int = 20,
) -> ContinuousReferenceSolution:
    """Build a Peierls-cylinder reference matching a cp_cyl1D case."""
    if ng_key != "1g" or n_regions != 1:
        raise NotImplementedError(
            f"peierls_cylinder continuous reference currently supports "
            f"1G 1-region only; got ng_key={ng_key!r}, n_regions={n_regions}"
        )

    from .cp_cylinder import _RADII

    layout = LAYOUTS[n_regions]
    ng = int(ng_key[0])
    radii = np.array(_RADII[n_regions], dtype=float)

    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t = np.array([xs["sig_t"][0] for xs in xs_list])
    sig_s = np.array([xs["sig_s"][0, 0] for xs in xs_list])
    nu_sig_f = np.array([(xs["nu"] * xs["sig_f"])[0] for xs in xs_list])

    sol = solve_peierls_cylinder_1g(
        radii, sig_t, sig_s, nu_sig_f,
        boundary="white",
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_beta=n_beta, n_rho=n_rho, n_phi=n_phi,
        dps=precision_digits,
    )

    r_nodes = sol.r_nodes
    _, r_wts, _ = _pg.composite_gl_r(
        radii, n_panels_per_region, p_order, dps=precision_digits,
    )
    phi = sol.phi_values[:, 0]
    integral = 2.0 * np.pi * np.dot(r_nodes * r_wts, phi)
    if abs(integral) > 1e-30:
        phi_normed = phi / integral
        sol = PeierlsCylinderSolution(
            r_nodes=sol.r_nodes,
            phi_values=phi_normed[:, np.newaxis],
            k_eff=sol.k_eff,
            cell_radius=sol.cell_radius,
            n_groups=sol.n_groups,
            n_quad_r=sol.n_quad_r,
            n_quad_y=sol.n_quad_y,
            precision_digits=sol.precision_digits,
            panel_bounds=sol.panel_bounds,
        )

    def phi_fn(x: np.ndarray, g: int = 0) -> np.ndarray:
        return sol.phi(x, g)

    mat_ids = _MAT_IDS_CYL[n_regions]
    materials = {
        mat_ids[i]: get_mixture(region, ng_key)
        for i, region in enumerate(layout)
    }

    return ContinuousReferenceSolution(
        name=f"peierls_cyl1D_{ng}eg_{n_regions}rg",
        problem=ProblemSpec(
            materials=materials,
            geometry_type="cylinder-1d",
            geometry_params={
                "radius": float(radii[-1]),
                "radii": radii.tolist(),
                "mat_ids": mat_ids,
            },
            boundary_conditions={"outer": "white"},
            is_eigenvalue=True,
            n_groups=ng,
        ),
        operator_form="integral-peierls",
        phi=phi_fn,
        k_eff=sol.k_eff,
        provenance=Provenance(
            citation=(
                "Sanchez & McCormick 1982 (NSE 80) §IV.A; "
                "Hebert 2020 Ch. 3 §3.5"
            ),
            derivation_notes=(
                f"Polar (β, ρ) Nyström via the unified "
                f"CurvilinearGeometry(kind='cylinder-1d'). "
                f"{n_panels_per_region} panels × {p_order} GL points on "
                f"[0, R], n_β = {n_beta}, n_ρ = {n_rho}. Ki₁ kernel, "
                f"no singular Jacobian. White BC via rank-1 Schur "
                f"closure (radial symmetry collapses the general "
                f"N_β block to a single scalar J⁻ = J⁺)."
            ),
            sympy_expression=None,
            precision_digits=precision_digits,
        ),
        equation_labels=(
            "peierls-cylinder-equation",
            "peierls-cylinder-polar",
            "peierls-cylinder-ray-optical-depth",
            "peierls-unified",
        ),
        vv_level="L1",
        description=(
            f"{ng}G {n_regions}-region cylindrical Peierls "
            f"(Ki₁ polar Nyström via unified geometry, rank-1 white BC)"
        ),
        tolerance="O(h²)",
    )


def continuous_cases() -> list[ContinuousReferenceSolution]:
    """Peierls cylinder continuous references for the registry.

    Returns an empty list until the rank-1 white-BC closure is
    replaced by a higher-rank variant accurate at :math:`R < 5` MFP
    (the regime of the existing ``cp_cyl1D`` cases).
    """
    return []
