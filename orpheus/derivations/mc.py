r"""Monte Carlo eigenvalue derivations.

Homogeneous: derived analytically from random walk probability theory.

Heterogeneous: the MC solver is run with the ConcentricPinCell geometry.
The reference eigenvalue comes from the CP cylinder derivation for the
same geometry (white BC approximation matches MC periodic BC for
sufficiently large cells). Statistical verification via z-score.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ._eigenvalue import kinf_homogeneous, kinf_from_cp
from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_xs, get_mixture

# Cylindrical radii and mat_ids (matching cp_cylinder)
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


def _derive_mc_homogeneous(ng_key: str) -> VerificationCase:
    """Derive MC eigenvalue from random walk probability theory."""
    xs = get_xs("A", ng_key)
    ng = len(xs["sig_t"])

    k_val = kinf_homogeneous(
        sig_t=xs["sig_t"], sig_s=xs["sig_s"],
        nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
    )

    if ng == 1:
        latex = (
            r"From random walk probability: "
            r":math:`k = \nu\Sigma_f/\Sigma_a`."
            "\n\n"
            r".. math::" "\n"
            rf"   k = \nu\Sigma_f / \Sigma_a = {k_val:.6f}"
        )
    else:
        latex = (
            r"Multi-group MC: collision kernel matrix gives "
            r":math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`."
            "\n\n"
            r".. math::" "\n"
            rf"   k_\infty = {k_val:.10f}"
        )

    return VerificationCase(
        name=f"mc_cyl1D_{ng}eg_1rg",
        k_inf=k_val,
        method="mc",
        geometry="cyl1D",
        n_groups=ng,
        n_regions=1,
        materials={0: get_mixture("A", ng_key)},
        geom_params={},
        latex=latex,
        description=f"MC pin cell, {ng}G homogeneous — from random walk probability",
        tolerance="z < 5\u03c3",
    )


def _derive_mc_heterogeneous(ng_key: str, n_regions: int) -> VerificationCase:
    """MC heterogeneous: reference from CP cylinder (same geometry, white BC).

    The CP semi-analytical eigenvalue serves as the reference.
    MC verification is statistical: z = |k_MC - k_ref| / σ < 5.
    """
    from .cp_cylinder import _cylinder_cp_matrix

    layout = LAYOUTS[n_regions]
    mat_ids = _MAT_IDS[n_regions]
    radii = np.array(_RADII[n_regions])
    ng = int(ng_key[0])

    # Annular volumes
    r_inner = np.zeros(n_regions)
    r_inner[1:] = radii[:-1]
    volumes = np.pi * (radii**2 - r_inner**2)
    r_cell = radii[-1]

    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t_all = np.vstack([xs["sig_t"] for xs in xs_list])

    P_inf_g = _cylinder_cp_matrix(sig_t_all, radii, volumes, r_cell)
    k_ref = kinf_from_cp(
        P_inf_g=P_inf_g,
        sig_t_all=sig_t_all,
        V_arr=volumes,
        sig_s_mats=[xs["sig_s"] for xs in xs_list],
        nu_sig_f_mats=[xs["nu"] * xs["sig_f"] for xs in xs_list],
        chi_mats=[xs["chi"] for xs in xs_list],
    )

    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    latex = (
        rf"MC {ng}G {n_regions}-region cylindrical pin cell. "
        r"Reference from CP cylinder (Ki₄ kernel, white BC). "
        r"Statistical verification: :math:`z = |k_{\rm MC} - k_{\rm ref}|/\sigma < 5`."
        "\n\n"
        r".. math::" "\n"
        rf"   k_{{\rm ref}} = {k_ref:.10f}"
    )

    return VerificationCase(
        name=f"mc_cyl1D_{ng}eg_{n_regions}rg",
        k_inf=k_ref,
        method="mc",
        geometry="cyl1D",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=dict(radii=_RADII[n_regions], mat_ids=mat_ids),
        latex=latex,
        description=f"MC cylindrical, {ng}G {n_regions}-region — ref from CP cylinder",
        tolerance="z < 5\u03c3",
    )


def all_cases() -> list[VerificationCase]:
    """Return all MC verification cases: {1,2,4}eg × {1,2,4}rg.

    Full matrix: 3 homogeneous + 6 heterogeneous = 9 cases.
    """
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        cases.append(_derive_mc_homogeneous(ng_key))
        for n_regions in [2, 4]:
            cases.append(_derive_mc_heterogeneous(ng_key, n_regions))
    return cases
