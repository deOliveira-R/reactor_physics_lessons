r"""MOC (Method of Characteristics) eigenvalue derivations.

Homogeneous: derived analytically from the characteristic ODE.

Heterogeneous: Richardson-extrapolated reference from the MOC solver
at multiple ray spacing refinements.  The reference is the converged limit
of the MOC equations themselves.
"""

from __future__ import annotations

import numpy as np

from ._eigenvalue import kinf_homogeneous
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

    k_val = kinf_homogeneous(
        sig_t=xs["sig_t"], sig_s=xs["sig_s"],
        nu_sig_f=xs["nu"] * xs["sig_f"], chi=xs["chi"],
    )

    if ng == 1:
        latex = (
            r"From the MOC characteristic ODE with homogeneous, isotropic "
            r"source: :math:`\bar\psi = Q/(4\pi\Sigma_t)`, giving"
            "\n\n"
            r".. math::" "\n"
            rf"   k = \nu\Sigma_f / \Sigma_a = {k_val:.6f}"
        )
    else:
        latex = (
            r"Multi-group MOC homogeneous: flat-source along every "
            r"characteristic reduces to the matrix eigenvalue."
            "\n\n"
            r".. math::" "\n"
            rf"   k_\infty = {k_val:.10f}"
        )

    labels: list[str] = ["characteristic-ode", "bar-psi", "isotropic-source"]
    if ng == 1:
        labels.append("one-group-kinf")
    else:
        labels += ["matrix-eigenvalue", "mg-balance"]

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
        vv_level="L1",
        equation_labels=tuple(labels),
    )


def all_cases() -> list[VerificationCase]:
    """Return analytical MOC cases (homogeneous only)."""
    return [_derive_moc_homogeneous(ng_key) for ng_key in ["1g", "2g", "4g"]]
