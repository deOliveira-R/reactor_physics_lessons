r"""SN (Discrete Ordinates) eigenvalue derivations.

Homogeneous cases: derived analytically from the 1D SN transport equation
with spatially-flat, isotropic flux (reflective BCs).

Heterogeneous cases: Richardson-extrapolated reference from O(h²)
diamond-difference mesh convergence (S16 angular quadrature).
The reference is the converged limit of the SN equations themselves.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ._eigenvalue import kinf_homogeneous
from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_xs, get_mixture, XS

# Slab thicknesses and mat_ids (same convention as cp_slab)
_THICKNESSES = {
    1: [0.5],
    2: [0.5, 0.5],
    4: [0.4, 0.05, 0.1, 0.45],
}
_MAT_IDS = {
    1: [2],
    2: [2, 0],
    4: [2, 3, 1, 0],
}


def _derive_sn_homogeneous(ng_key: str) -> VerificationCase:
    """Derive SN eigenvalue for a homogeneous slab from the transport equation."""
    xs = get_xs("A", ng_key)
    ng = len(xs["sig_t"])

    k_val = kinf_homogeneous(
        sig_t=xs["sig_t"], sig_s=xs["sig_s"],
        nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
    )

    if ng == 1:
        latex = (
            r"From the 1D S\ :sub:`N` equation with "
            r":math:`\partial\psi_m/\partial x = 0` (homogeneous, reflective BCs):"
            "\n\n"
            r".. math::" "\n"
            rf"   k = \nu\Sigma_f / \Sigma_a = {k_val:.6f}"
        )
    else:
        latex = (
            r"Multi-group S\ :sub:`N` homogeneous: flat flux reduces to "
            r":math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`."
            "\n\n"
            r".. math::" "\n"
            rf"   k_\infty = {k_val:.10f}"
        )

    return VerificationCase(
        name=f"sn_slab_{ng}eg_1rg",
        k_inf=k_val,
        method="sn",
        geometry="slab",
        n_groups=ng,
        n_regions=1,
        materials={0: get_mixture("A", ng_key)},
        geom_params={},
        latex=latex,
        description=f"SN 1D reflective slab, {ng}G homogeneous — from transport equation",
        tolerance="< 1e-8",
    )


def _derive_sn_heterogeneous(ng_key: str, n_regions: int) -> VerificationCase:
    """Derive SN eigenvalue for a heterogeneous slab via Richardson extrapolation.

    Runs the SN solver at 4 mesh refinements with S16 GL quadrature,
    then extrapolates to h→0 assuming O(h²) convergence.
    Results are cached to avoid recomputation on subsequent test runs.
    """
    from ._richardson_cache import get_cached, store

    layout = LAYOUTS[n_regions]
    mat_ids = _MAT_IDS[n_regions]
    thicknesses = _THICKNESSES[n_regions]
    ng = int(ng_key[0])

    # Build materials dict
    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    case_name = f"sn_slab_{ng}eg_{n_regions}rg"
    cells_per_region = [5, 10, 20, 40]

    # Cache key includes all inputs that affect the result
    cache_params = dict(
        method="sn", ng_key=ng_key, n_regions=n_regions,
        thicknesses=thicknesses, mat_ids=mat_ids,
        cells_per_region=cells_per_region, n_angles=16,
        xs={r: get_xs(r, ng_key) for r in layout},
    )

    k_ref = get_cached(case_name, cache_params)
    if k_ref is None:
        from orpheus.geometry import CoordSystem, mesh1d_from_zones, Zone
        from orpheus.sn.quadrature import GaussLegendre1D
        from orpheus.sn.solver import solve_sn

        quad = GaussLegendre1D.create(16)
        keffs = []
        for n_per in cells_per_region:
            edge = 0.0
            zones = []
            for t, mid in zip(thicknesses, mat_ids):
                edge += t
                zones.append(Zone(outer_edge=edge, mat_id=mid, n_cells=n_per))
            mesh = mesh1d_from_zones(zones, coord=CoordSystem.CARTESIAN)
            result = solve_sn(
                materials, mesh, quad,
                max_outer=500, max_inner=500, inner_tol=1e-10,
                keff_tol=1e-8,
            )
            keffs.append(result.keff)

        # O(h²) Richardson extrapolation using two finest meshes (ratio 2)
        k_ref = keffs[-1] + (keffs[-1] - keffs[-2]) / 3.0
        store(case_name, cache_params, k_ref, keffs)

    latex = (
        rf"Richardson-extrapolated S\ :sub:`N` eigenvalue for {ng}G, "
        rf"{n_regions}-region slab (S16, diamond-difference O(h²))."
        "\n\n"
        rf"Mesh sequence: {cells_per_region} cells/region. "
        rf"Extrapolated :math:`k_\infty = {k_ref:.10f}`."
    )

    return VerificationCase(
        name=case_name,
        k_inf=k_ref,
        method="sn",
        geometry="slab",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=dict(thicknesses=thicknesses, mat_ids=mat_ids),
        latex=latex,
        description=f"SN 1D slab, {ng}G {n_regions}-region — Richardson extrapolation",
        tolerance="O(h²)",
    )


def all_cases() -> list[VerificationCase]:
    """Return analytical SN cases (homogeneous only).

    Heterogeneous cases require the SN solver and are returned by
    ``solver_cases()`` instead.
    """
    return [_derive_sn_homogeneous(ng_key) for ng_key in ["1g", "2g", "4g"]]


def solver_cases() -> list[VerificationCase]:
    """Return solver-computed SN cases (heterogeneous, Richardson extrapolation).

    These require the SN solver to be importable (via pythonpath).
    """
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        for n_regions in [2, 4]:
            cases.append(_derive_sn_heterogeneous(ng_key, n_regions))
    return cases
