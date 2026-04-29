r"""Peierls integral equation reference for spherical CP verification.

Spherical specialisation of the unified polar-form Peierls Nyström
infrastructure in :mod:`orpheus.derivations.peierls_geometry`. This
module is a THIN FACADE that mirrors :mod:`peierls_cylinder`: it owns
the sphere-specific API names (``solve_peierls_sphere_{1g,mg}``
permanent wrappers per :ref:`theory-peierls-api-posture`), the
:class:`PeierlsSphereSolution` dataclass, and the
``_build_peierls_sphere_case`` registry constructors. Everything
else — volume-kernel assembly, Lagrange basis, angular/radial
composite quadrature, white-BC rank-1 closure, eigenvalue power
iteration — lives in
:mod:`~orpheus.derivations.peierls_geometry` and dispatches through
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
    inner_radius: float = 0.0,
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_theta: int = 24,
    n_rho: int = 24,
    n_phi: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> PeierlsSphereSolution:
    r"""1G spherical Peierls k-eigenvalue driver.

    Thin wrapper over :func:`peierls_geometry.solve_peierls_1g` with
    the sphere geometry pre-bound.

    Parameters
    ----------
    boundary
        - ``"vacuum"`` (default): vacuum BC on the outer surface.
        - ``"white"``: rank-1 Mark (isotropic) white BC. For
          :math:`r_0 = 0` (solid sphere) this is the baseline
          closure used throughout the Phase 4.x CP-vs-Peierls
          campaign. Scalar accuracy on solid spheres is bounded by
          ``build_white_bc_correction``'s rank-1 assumption (GitHub
          Issue #100 / #103).
        - ``"white_rank2"`` (**F.4 — Stamm'ler IV Eq. 34 = Hébert
          2009 Eq. 3.323**): scalar rank-2 per-face closure. Requires
          ``inner_radius > 0`` (a hollow sphere — F.4 reduces to
          rank-1 Mark on solid geometry because there is no
          second-face coupling). At default quadrature:
          :math:`r_0 / R = 0.1 \to 0.4\,\%`,
          :math:`r_0 / R = 0.2 \to 1.2\,\%`,
          :math:`r_0 / R = 0.3 \to 3.3\,\%`. The sphere's explicit
          3-D :math:`\theta` integration retains angular information
          that the cylinder's Ki\ :sub:`3` fold averages away,
          giving 3–10× tighter residuals at the same :math:`r_0/R`.
          Rank-N per-face refinement was falsified as a path to
          improve beyond F.4 — see
          :ref:`peierls-rank-n-per-face-closeout` and research-log
          lesson L21.

    inner_radius
        Inner cavity radius for hollow-sphere geometries (Phase F.5).
        Default ``0.0`` (solid sphere). Must be strictly less than
        ``radii[-1]``.
    """
    if inner_radius != 0.0:
        geometry = _pg.CurvilinearGeometry(
            kind="sphere-1d", inner_radius=float(inner_radius),
        )
    else:
        geometry = GEOMETRY
    sol = _pg.solve_peierls_1g(
        geometry, radii, sig_t, sig_s, nu_sig_f,
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


# ═══════════════════════════════════════════════════════════════════════
# Multi-group eigenvalue driver (Issue #104)
# ═══════════════════════════════════════════════════════════════════════


def solve_peierls_sphere_mg(
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    chi: np.ndarray,
    *,
    boundary: str = "vacuum",
    inner_radius: float = 0.0,
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_theta: int = 24,
    n_rho: int = 24,
    n_phi: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
) -> PeierlsSphereSolution:
    r"""Multi-group spherical Peierls k-eigenvalue driver.

    Thin wrapper over :func:`peierls_geometry.solve_peierls_mg` with
    the sphere geometry pre-bound. Semantics of ``boundary`` and
    ``inner_radius`` match :func:`solve_peierls_sphere_1g`.

    Parameters
    ----------
    sig_t
        Total cross section per region and group, shape
        ``(n_regions, ng)``.
    sig_s
        P\ :sub:`0` scattering matrix, shape ``(n_regions, ng, ng)``.
        Convention: ``sig_s[r, g_src, g_dst]`` = rate from ``g_src``
        to ``g_dst`` at region ``r``. See
        :func:`peierls_geometry.solve_peierls_mg` for the full
        convention note.
    nu_sig_f, chi
        ``(n_regions, ng)`` per-group arrays.
    """
    if inner_radius != 0.0:
        geometry = _pg.CurvilinearGeometry(
            kind="sphere-1d", inner_radius=float(inner_radius),
        )
    else:
        geometry = GEOMETRY
    sol = _pg.solve_peierls_mg(
        geometry, radii, sig_t, sig_s, nu_sig_f, chi,
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
    integral = GEOMETRY.shell_volume_integral(r_nodes, r_wts, phi)
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


def _build_peierls_sphere_hollow_f4_case(
    r0_over_R: float,
    ng_key: str = "1g",
    n_panels_per_region: int = 3,
    p_order: int = 5,
    n_theta: int = 24,
    n_rho: int = 24,
    n_phi: int = 24,
    precision_digits: int = 20,
) -> ContinuousReferenceSolution:
    r"""Build a hollow-sphere Peierls F.4 continuous reference.

    Uses Stamm'ler IV Eq. 34 = Hébert 2009 Eq. 3.323 (see
    :math:numref:`hebert-3-323`) — scalar rank-2 per-face closure
    assembled through
    :func:`~orpheus.derivations.peierls_geometry.compute_hollow_sph_transmission`
    (bare :math:`\exp(-\tau)` kernel with explicit :math:`\theta`
    integration, :math:`W_{oi} = (R/r_0)^2\,W_{io}` reciprocity —
    distinct from the cylinder's first-power form).

    Residuals vs :math:`k_\infty` at default quadrature and XS:

    - :math:`r_0 / R = 0.1` → 0.4 % err
    - :math:`r_0 / R = 0.2` → 1.2 % err
    - :math:`r_0 / R = 0.3` → 3.3 % err

    The sphere's 3-10× tighter residuals relative to the cylinder at
    the same :math:`r_0/R` (cylinder: 1.4 %, 5.4 %, 13 %) reflect
    the sphere's higher SO(3) symmetry capturing more angular
    structure at the scalar mode. Rank-N per-face refinement was
    falsified as a path to improve beyond F.4 — see
    :ref:`peierls-rank-n-per-face-closeout` and research-log L21.

    The reference is forward-looking: no production CP solver case
    exists at ``inner_radius > 0`` today. A future hollow
    ``cp_sph1D_hollow_*`` consumer can load this reference via
    :func:`orpheus.derivations.reference_values.continuous_get`.
    """
    from .cp_sphere import _RADII

    n_regions = 1
    ng = int(ng_key[0])
    R_out = float(_RADII[n_regions][-1])
    r0 = float(r0_over_R) * R_out
    if not (0.0 < r0 < R_out):
        raise ValueError(
            f"r0_over_R must be in (0, 1); got {r0_over_R}"
        )

    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(region, ng_key) for region in layout]
    # (n_regions, ng) per-region per-group arrays (n_regions == 1 here).
    sig_t = np.stack([np.asarray(xs["sig_t"], dtype=float) for xs in xs_list])
    sig_s = np.stack([np.asarray(xs["sig_s"], dtype=float) for xs in xs_list])
    nu_sig_f = np.stack([
        np.asarray(xs["nu"] * xs["sig_f"], dtype=float) for xs in xs_list
    ])
    chi = np.stack([np.asarray(xs["chi"], dtype=float) for xs in xs_list])

    radii = np.array([R_out])

    sol = solve_peierls_sphere_mg(
        radii, sig_t, sig_s, nu_sig_f, chi,
        boundary="white_f4",
        inner_radius=r0,
        n_panels_per_region=n_panels_per_region,
        p_order=p_order,
        n_theta=n_theta, n_rho=n_rho, n_phi=n_phi,
        dps=precision_digits,
    )

    r_nodes = sol.r_nodes
    _, r_wts, _ = _pg.composite_gl_r(
        radii, n_panels_per_region, p_order, dps=precision_digits,
        inner_radius=r0,
    )
    # Normalise each group's flux to unit shell-volume integral.
    phi_normed = np.empty_like(sol.phi_values)
    for g in range(ng):
        phi_g = sol.phi_values[:, g]
        integral_g = GEOMETRY.shell_volume_integral(r_nodes, r_wts, phi_g)
        phi_normed[:, g] = (
            phi_g / integral_g if abs(integral_g) > 1e-30 else phi_g
        )
    sol = PeierlsSphereSolution(
        r_nodes=sol.r_nodes,
        phi_values=phi_normed,
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

    r0_tag = f"{int(round(r0_over_R * 100)):02d}"
    return ContinuousReferenceSolution(
        name=f"peierls_sph1D_hollow_{ng}eg_{n_regions}rg_r0_{r0_tag}",
        problem=ProblemSpec(
            materials=materials,
            geometry_type="sphere-1d",
            geometry_params={
                "radius": R_out,
                "inner_radius": r0,
                "radii": radii.tolist(),
                "mat_ids": mat_ids,
            },
            boundary_conditions={"outer": "white_rank2"},
            is_eigenvalue=True,
            n_groups=ng,
        ),
        operator_form="integral-peierls",
        phi=phi_fn,
        k_eff=sol.k_eff,
        provenance=Provenance(
            citation=(
                "Stamm'ler & Abbate 1983 Ch. IV Eq. 34; "
                "Hébert 2009 Ch. 3 §3.8.4 Eq. 3.323"
            ),
            derivation_notes=(
                f"F.4 scalar rank-2 per-face closure on hollow sphere "
                f"(r_0/R = {r0_over_R:.2f}). Polar (θ, ρ) Nyström via "
                f"CurvilinearGeometry(kind='sphere-1d', "
                f"inner_radius={r0:.6f}). Transmission matrix built by "
                f"compute_hollow_sph_transmission (bare exp(-τ) kernel, "
                f"explicit θ integration), R_eff = (I - W)^(-1). "
                f"{n_panels_per_region} panels × {p_order} GL points on "
                f"[r_0, R], n_θ = {n_theta}, n_ρ = {n_rho}, "
                f"n_surf_quad = {n_phi}. White BC."
            ),
            sympy_expression=None,
            precision_digits=precision_digits,
        ),
        equation_labels=(
            "hebert-3-323",
            "peierls-unified",
        ),
        vv_level="L1",
        description=(
            f"{ng}G {n_regions}-region hollow spherical Peierls "
            f"(F.4 rank-2 per-face, exp kernel, r_0/R = {r0_over_R:.2f})"
        ),
        tolerance=f"O(h²) + scalar-mode residual ~{_F4_SPH_TOL[r0_over_R]}",
    )


# Measured baseline at default quadrature (see test_hollow_sph_rank2_beats_rank1_mark).
_F4_SPH_TOL = {0.1: "0.4 %", 0.2: "1.2 %", 0.3: "3.3 %"}


