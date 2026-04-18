r"""Peierls integral equation reference for spherical CP verification.

Spherical specialisation of the unified polar-form Peierls Nyström
infrastructure in :mod:`orpheus.derivations.peierls_geometry`. This
module is a THIN FACADE that mirrors :mod:`peierls_cylinder`: it owns
the sphere-specific API names, the ``_build_peierls_sphere_case``
case builder, and the ``continuous_cases`` registration. Everything
else — volume-kernel assembly, Lagrange basis, angular/radial
composite quadrature, white-BC rank-1 closure, eigenvalue power
iteration — dispatches through the unified
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
with ``kind = "sphere-1d"``.

See :doc:`/theory/peierls_unified` for the end-to-end derivation of
the unified structure and :doc:`/theory/collision_probability` for
the sphere-specific narrative (3-D point kernel, :math:`\sin\theta`
angular measure, rank-1 white-BC limitations paralleling the
cylinder).

The spherical Peierls equation in observer-centred polar coords:

.. math::

   \Sigma_t(r)\,\varphi(r)
     \;=\; \frac{\Sigma_t(r)}{2}\!
       \int_{0}^{\pi}\!\sin\theta\,\mathrm d\theta\!
       \int_{0}^{\rho_{\max}(r,\theta)}\!\!
         e^{-\tau(r,\rho,\theta)}\,
         q\bigl(r'(r,\rho,\theta)\bigr)\,\mathrm d\rho
     + S_{\rm bc}(r).

The prefactor :math:`1/2` absorbs the :math:`1/(4\pi)` of the 3-D
Green's function and a factor of :math:`2\pi` from trivial azimuthal
integration (the source field is radially symmetric, so only the
polar angle :math:`\theta` matters):
:math:`1/(4\pi) \cdot 2\pi = 1/2`.

The :math:`\sin\theta` weight comes from the spherical solid-angle
element :math:`\mathrm d\Omega = \sin\theta\,\mathrm d\theta\,
\mathrm d\phi` — no :math:`\pm` folding is needed since
:math:`\sin\theta \ge 0` on :math:`[0, \pi]` and the integrand
already covers the full hemisphere of directions seen from the
observer.

.. note::

   The ray-geometry primitives :math:`\rho_{\max}(r,\theta)` and
   :math:`r'(r,\rho,\theta)` are IDENTICAL to the cylinder case —
   a 1-D radial domain bounded by a spherical shell of radius
   :math:`R` has the same chord algebra regardless of whether the
   surrounding field is 2-D-symmetric (cylinder) or 3-D-symmetric
   (sphere). The only geometry-specific ingredients are the kernel
   (:math:`e^{-\tau}` vs :math:`\mathrm{Ki}_1`) and the angular
   weight (:math:`\sin\theta` vs constant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import peierls_geometry as _pg
from ._reference import (
    ContinuousReferenceSolution,
    ProblemSpec,
    Provenance,
)
from ._xs_library import LAYOUTS, get_mixture, get_xs


# ═══════════════════════════════════════════════════════════════════════
# Sphere geometry singleton (binds the unified infrastructure)
# ═══════════════════════════════════════════════════════════════════════

GEOMETRY = _pg.SPHERE_1D


# ═══════════════════════════════════════════════════════════════════════
# Public API (mirrors peierls_cylinder, dispatches to peierls_geometry)
# ═══════════════════════════════════════════════════════════════════════

composite_gl_r = _pg.composite_gl_r
_lagrange_basis_on_panels = _pg.lagrange_basis_on_panels


def _rho_max(r_obs: float, cos_theta: float, R: float) -> float:
    """Ray-exit distance along polar angle :math:`\\theta`."""
    return GEOMETRY.rho_max(r_obs, cos_theta, R)


def _optical_depth_along_ray(
    r_obs: float,
    cos_theta: float,
    rho: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
) -> float:
    """Line-integrated optical depth along a ray from :math:`r_{\\rm obs}`
    in direction :math:`\\theta` for distance :math:`\\rho`.

    Identical walker to the cylinder case — the 1-D radial annulus
    crossings are geometry-agnostic."""
    return GEOMETRY.optical_depth_along_ray(r_obs, cos_theta, rho, radii, sig_t)


def _which_annulus(r: float, radii: np.ndarray) -> int:
    return GEOMETRY.which_annulus(r, radii)


def build_volume_kernel(
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_theta: int,
    n_rho: int,
    dps: int = 30,
) -> np.ndarray:
    """Spherical volume Nyström kernel.

    ``n_theta`` names the polar-angle integration order (the
    :math:`\\sin\\theta` weight is applied internally by the unified
    geometry dispatch)."""
    return _pg.build_volume_kernel(
        GEOMETRY, r_nodes, panel_bounds, radii, sig_t,
        n_angular=n_theta, n_rho=n_rho, dps=dps,
    )


def compute_P_esc(
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_theta: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Spherical uncollided escape probability.

    .. math::

       P_{\rm esc}(r_i) \;=\; \tfrac12 \int_0^\pi \sin\theta\,
         e^{-\tau(r_i,\,\rho_{\max}(r_i,\theta),\,\theta)}\,\mathrm d\theta.
    """
    return _pg.compute_P_esc(
        GEOMETRY, r_nodes, radii, sig_t,
        n_angular=n_theta, dps=dps,
    )


def compute_G_bc(
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_theta: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Spherical surface-to-volume Green's function.

    .. math::

       G_{\rm bc}(r_i) \;=\; R^2 \!\int_0^\pi \sin\theta\,
         \frac{e^{-\tau_{\rm surf}(r_i,\theta)}}{d(r_i,R,\theta)^2}
         \,\mathrm d\theta,
       \quad d = \sqrt{r_i^2 + R^2 - 2 r_i R \cos\theta}.

    The implementation lives in
    :func:`peierls_geometry.compute_G_bc`, which dispatches on
    ``geometry.kind == "sphere-1d"``.
    """
    return _pg.compute_G_bc(
        GEOMETRY, r_nodes, radii, sig_t,
        n_surf_quad=n_theta, dps=dps,
    )


def build_white_bc_correction(
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_theta: int = 32,
    n_phi: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Rank-1 white-BC correction for the spherical Peierls kernel.

    Returns :math:`K_{\rm bc}[i, j] = u[i]\,v[j]` with
    :math:`u[i] = \Sigma_t(r_i)\,G_{\rm bc}(r_i) / R` and
    :math:`v[j] = r_j^{2}\,w_j\,P_{\rm esc}(r_j)`.

    .. warning::

       **Approximation level.** The rank-1 closure assumes the
       re-entering angular distribution is Mark/isotropic. By
       radial symmetry the scalar partial-current balance
       :math:`J^- = J^+` is exact, but the angular moments beyond
       the zeroth are approximated. The resulting error in
       :math:`k_{\rm eff}` grows with inverse cell size, mirroring
       the cylindrical case — see Issue #100. Deferred higher-rank
       fix is tracked by Issue #103 (N1).
    """
    return _pg.build_white_bc_correction(
        GEOMETRY, r_nodes, r_wts, radii, sig_t,
        n_angular=n_theta, n_surf_quad=n_phi, dps=dps,
    )


# ═══════════════════════════════════════════════════════════════════════
# Solution container — backward-compatible alias for PeierlsSolution
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeierlsSphereSolution:
    """Result of a Peierls Nyström solve on a 1-D radial sphere.

    Kept as a sphere-flavoured facade over
    :class:`~peierls_geometry.PeierlsSolution`. Mirrors the
    :class:`~peierls_cylinder.PeierlsCylinderSolution` shape for
    symmetry — new code may use the unified
    :class:`~peierls_geometry.PeierlsSolution` directly.
    """

    r_nodes: np.ndarray
    phi_values: np.ndarray
    k_eff: float | None
    cell_radius: float
    n_groups: int
    n_quad_r: int
    n_quad_theta: int
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


def _soln_to_sphere(sol: _pg.PeierlsSolution) -> PeierlsSphereSolution:
    return PeierlsSphereSolution(
        r_nodes=sol.r_nodes,
        phi_values=sol.phi_values,
        k_eff=sol.k_eff,
        cell_radius=sol.cell_radius,
        n_groups=sol.n_groups,
        n_quad_r=sol.n_quad_r,
        n_quad_theta=sol.n_quad_angular,
        precision_digits=sol.precision_digits,
        panel_bounds=sol.panel_bounds,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1G eigenvalue drivers
# ═══════════════════════════════════════════════════════════════════════

def solve_peierls_sphere_1g(
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    *,
    boundary: str = "vacuum",
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_theta: int = 24,
    n_rho: int = 24,
    n_phi: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> PeierlsSphereSolution:
    """1G spherical Peierls k-eigenvalue driver (vacuum or white BC).

    Thin wrapper over :func:`peierls_geometry.solve_peierls_1g` with
    the sphere geometry pre-bound."""
    sol = _pg.solve_peierls_1g(
        GEOMETRY, radii, sig_t, sig_s, nu_sig_f,
        boundary=boundary,
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_angular=n_theta,
        n_rho=n_rho,
        n_surf_quad=n_phi,
        dps=dps,
        max_iter=max_iter,
        tol=tol,
    )
    return _soln_to_sphere(sol)


def solve_peierls_sphere_1g_vacuum(
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    *,
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_theta: int = 24,
    n_rho: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> PeierlsSphereSolution:
    """Vacuum-BC alias for the scaffold-level verification gate."""
    return solve_peierls_sphere_1g(
        radii, sig_t, sig_s, nu_sig_f,
        boundary="vacuum",
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_theta=n_theta, n_rho=n_rho,
        dps=dps, max_iter=max_iter, tol=tol,
    )


# ═══════════════════════════════════════════════════════════════════════
# ContinuousReferenceSolution builder
# ═══════════════════════════════════════════════════════════════════════

_MAT_IDS_SPH = {1: [2]}


def _build_peierls_sphere_case(
    ng_key: str,
    n_regions: int,
    n_panels_per_region: int = 3,
    p_order: int = 5,
    n_theta: int = 20,
    n_rho: int = 20,
    n_phi: int = 20,
    precision_digits: int = 20,
) -> ContinuousReferenceSolution:
    """Build a Peierls-sphere reference matching a cp_sph1D case."""
    if ng_key != "1g" or n_regions != 1:
        raise NotImplementedError(
            f"peierls_sphere continuous reference currently supports "
            f"1G 1-region only; got ng_key={ng_key!r}, n_regions={n_regions}"
        )

    from .cp_sphere import _RADII

    layout = LAYOUTS[n_regions]
    ng = int(ng_key[0])
    radii = np.array(_RADII[n_regions], dtype=float)

    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t = np.array([xs["sig_t"][0] for xs in xs_list])
    sig_s = np.array([xs["sig_s"][0, 0] for xs in xs_list])
    nu_sig_f = np.array([(xs["nu"] * xs["sig_f"])[0] for xs in xs_list])

    sol = solve_peierls_sphere_1g(
        radii, sig_t, sig_s, nu_sig_f,
        boundary="white",
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_theta=n_theta, n_rho=n_rho, n_phi=n_phi,
        dps=precision_digits,
    )

    r_nodes = sol.r_nodes
    _, r_wts, _ = _pg.composite_gl_r(
        radii, n_panels_per_region, p_order, dps=precision_digits,
    )
    phi = sol.phi_values[:, 0]
    # Spherical volume normalisation: ∫ 4π r² φ(r) dr
    integral = 4.0 * np.pi * np.dot(r_nodes * r_nodes * r_wts, phi)
    if abs(integral) > 1e-30:
        phi_normed = phi / integral
        sol = PeierlsSphereSolution(
            r_nodes=sol.r_nodes,
            phi_values=phi_normed[:, np.newaxis],
            k_eff=sol.k_eff,
            cell_radius=sol.cell_radius,
            n_groups=sol.n_groups,
            n_quad_r=sol.n_quad_r,
            n_quad_theta=sol.n_quad_theta,
            precision_digits=sol.precision_digits,
            panel_bounds=sol.panel_bounds,
        )

    def phi_fn(x: np.ndarray, g: int = 0) -> np.ndarray:
        return sol.phi(x, g)

    mat_ids = _MAT_IDS_SPH[n_regions]
    materials = {
        mat_ids[i]: get_mixture(region, ng_key)
        for i, region in enumerate(layout)
    }

    return ContinuousReferenceSolution(
        name=f"peierls_sph1D_{ng}eg_{n_regions}rg",
        problem=ProblemSpec(
            materials=materials,
            geometry_type="sphere-1d",
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
                "Case & Zweifel 1967 (bare-sphere critical-R); "
                "Hebert 2020 Ch. 3 §3.5 (curvilinear Peierls)"
            ),
            derivation_notes=(
                f"Polar (θ, ρ) Nyström via the unified "
                f"CurvilinearGeometry(kind='sphere-1d'). "
                f"{n_panels_per_region} panels × {p_order} GL points on "
                f"[0, R], n_θ = {n_theta}, n_ρ = {n_rho}. Exponential "
                f"kernel (no dimensional reduction needed — the 3-D "
                f"point kernel is already in 3-D), sin θ angular weight. "
                f"White BC via rank-1 Schur closure (radial symmetry "
                f"collapses the general N_θ block to a single scalar "
                f"J⁻ = J⁺)."
            ),
            sympy_expression=None,
            precision_digits=precision_digits,
        ),
        equation_labels=(
            "peierls-unified",
        ),
        vv_level="L1",
        description=(
            f"{ng}G {n_regions}-region spherical Peierls "
            f"(exp polar Nyström via unified geometry, rank-1 white BC)"
        ),
        tolerance="O(h²)",
    )


def continuous_cases() -> list[ContinuousReferenceSolution]:
    """Peierls sphere continuous references for the registry.

    Returns an empty list until the rank-1 white-BC closure is
    replaced by a higher-rank variant accurate at :math:`R < 5` MFP
    (the regime of the existing ``cp_sph1D`` cases). Tracked by
    Issue #103 (N1); see also Issue #100."""
    return []
