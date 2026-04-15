r"""SN (Discrete Ordinates) eigenvalue derivations.

Homogeneous cases: derived analytically from the 1D SN transport
equation with spatially-flat, isotropic flux (reflective BCs).

Heterogeneous SN verification is covered in two places:

- **Spatial operator** — the Method of Manufactured Solutions
  with continuous cross sections, in
  :mod:`orpheus.derivations.sn_mms`
  (:func:`~orpheus.derivations.sn_mms.build_1d_slab_heterogeneous_mms_case`).
  This is a Phase-2.1a L1 reference that replaces the earlier
  Richardson-extrapolated heterogeneous cases that were removed
  from this module.
- **Eigenvalue** — (Phase-2.1b, pending) Case singular-
  eigenfunction expansion of the multigroup SN transport
  equation on a two-region slab, providing a semi-analytical
  :math:`k_{\text{eff}}` reference without any self-crutch.

See :doc:`/verification/reference_solutions` and the heterogeneous
MMS section of :doc:`/theory/discrete_ordinates` for the full
campaign context.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ._eigenvalue import kinf_homogeneous
from ._types import VerificationCase
from ._xs_library import get_xs, get_mixture


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

    labels: list[str] = ["transport-cartesian", "reflective-bc"]
    if ng == 1:
        labels.append("one-group-kinf")
    else:
        labels += ["matrix-eigenvalue", "mg-balance", "multigroup"]

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
        vv_level="L1",
        equation_labels=tuple(labels),
    )


def all_cases() -> list[VerificationCase]:
    """Return analytical SN cases (homogeneous only).

    Heterogeneous SN verification is now covered by the
    Phase-2.1a MMS continuous reference in
    :mod:`orpheus.derivations.sn_mms`; this module no longer
    produces heterogeneous ``VerificationCase`` objects.
    """
    return [_derive_sn_homogeneous(ng_key) for ng_key in ["1g", "2g", "4g"]]
