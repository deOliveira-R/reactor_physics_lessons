r"""MOC (Method of Characteristics) eigenvalue derivations.

Homogeneous: derived analytically from the characteristic ODE.

Heterogeneous: Richardson-extrapolated reference from the MOC solver
at multiple mesh refinements. The reference is the converged limit
of the MOC equations themselves.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_xs, get_mixture

# Cylindrical radii and mat_ids (same convention as cp_cylinder)
_RADII = {
    1: [1.0],
    2: [0.5, 1.0],
    4: [0.4, 0.45, 0.55, 1.0],
}
_MAT_IDS = {
    1: [2],
    2: [2, 0],
    4: [2, 3, 1, 0],
}


def _derive_moc_homogeneous(ng_key: str) -> VerificationCase:
    """Derive MOC eigenvalue for a homogeneous pin cell."""
    xs = get_xs("A", ng_key)
    ng = len(xs["sig_t"])

    if ng == 1:
        nu_s, Sig_f_sym, Sig_a = sp.symbols('nu Sigma_f Sigma_a', positive=True)
        k_expr = nu_s * Sig_f_sym / Sig_a
        sig_a_val = xs["sig_c"][0] + xs["sig_f"][0]
        k_val = float(k_expr.subs({
            nu_s: xs["nu"][0], Sig_f_sym: xs["sig_f"][0], Sig_a: sig_a_val,
        }))
        latex = (
            r"From the MOC characteristic ODE with homogeneous, isotropic "
            r"source: :math:`\bar\psi = Q/(4\pi\Sigma_t)`, giving"
            "\n\n"
            r".. math::" "\n"
            rf"   k = {sp.latex(k_expr)} = {k_val:.6f}"
        )
    else:
        A_sym = sp.Matrix(np.diag(xs["sig_t"])) - sp.Matrix(xs["sig_s"].T)
        F_sym = sp.Matrix(np.outer(xs["chi"], xs["nu"] * xs["sig_f"]))
        M_sym = A_sym.inv() * F_sym
        eigs = M_sym.eigenvals()
        k_val = float(max(sp.re(e) for e in eigs.keys()))
        latex = (
            r"Multi-group MOC homogeneous: flat-source along every "
            r"characteristic reduces to the matrix eigenvalue."
            "\n\n"
            r".. math::" "\n"
            rf"   k_\infty = {k_val:.10f}"
        )

    return VerificationCase(
        name=f"moc_cyl1D_{ng}eg_1rg",
        k_inf=k_val,
        method="moc",
        geometry="cyl1D",
        n_groups=ng,
        n_regions=1,
        materials={0: get_mixture("A", ng_key)},
        geom_params={},
        latex=latex,
        description=f"MOC cylindrical pin cell, {ng}G homogeneous — from characteristic ODE",
        tolerance="< 1e-4",
    )


def _derive_moc_heterogeneous(ng_key: str, n_regions: int) -> VerificationCase:
    """Derive MOC eigenvalue via Richardson extrapolation of mesh refinement."""
    from method_of_characteristics import MoCGeometry, solve_moc

    layout = LAYOUTS[n_regions]
    mat_ids = _MAT_IDS[n_regions]
    radii = _RADII[n_regions]
    ng = int(ng_key[0])

    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    r_cell = radii[-1]
    pitch = r_cell * np.sqrt(np.pi) * 2  # large enough to contain the pin

    # Run at 3 mesh refinements (MOC is slow, use coarser sequence)
    n_cells_list = [8, 12, 16]
    keffs = []
    for nc in n_cells_list:
        geom = MoCGeometry.from_annular(radii, mat_ids, pitch=pitch, n_cells=nc)
        result = solve_moc(materials, geom, max_outer=300)
        keffs.append(result.keff)

    # Richardson extrapolation (O(h²), ratio from 12→16)
    h_ratio = n_cells_list[-2] / n_cells_list[-1]
    k_ref = keffs[-1] + (keffs[-1] - keffs[-2]) / (h_ratio**2 - 1)

    latex = (
        rf"Richardson-extrapolated MOC eigenvalue for {ng}G, "
        rf"{n_regions}-region cylindrical pin cell."
        "\n\n"
        rf"Mesh sequence: {n_cells_list} cells/side. "
        rf"Extrapolated :math:`k_\infty = {k_ref:.10f}`."
    )

    return VerificationCase(
        name=f"moc_cyl1D_{ng}eg_{n_regions}rg",
        k_inf=k_ref,
        method="moc",
        geometry="cyl1D",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=dict(radii=radii, mat_ids=mat_ids),
        latex=latex,
        description=f"MOC cylindrical, {ng}G {n_regions}-region — Richardson extrapolation",
        tolerance="< 1e-2",
    )


def all_cases() -> list[VerificationCase]:
    """Return analytical MOC cases (homogeneous only)."""
    return [_derive_moc_homogeneous(ng_key) for ng_key in ["1g", "2g", "4g"]]


def solver_cases() -> list[VerificationCase]:
    """Return solver-computed MOC cases (heterogeneous, Richardson extrapolation)."""
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        for n_regions in [2, 4]:
            cases.append(_derive_moc_heterogeneous(ng_key, n_regions))
    return cases
