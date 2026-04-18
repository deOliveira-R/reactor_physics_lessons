"""Semi-analytical cylindrical collision probability eigenvalues.

Thin facade over :mod:`~orpheus.derivations.cp_geometry` with
:data:`~.cp_geometry.CYLINDER_1D` pre-selected. Derives ``k_inf``
for {1, 2, 4} energy groups × {1, 2, 4} regions using the canonical
:math:`\\mathrm{Ki}_3` (Bickley-Naylor) kernel as the second-difference
anti-derivative.

See :doc:`/theory/peierls_unified` §§11-17 for the three-tier
integration hierarchy and the unified :math:`\\Delta^{2}` operator.

.. note::

   Phase B.4 retired the legacy :class:`~._kernels.BickleyTables`
   tabulation (Issue #94). The :math:`\\mathrm{Ki}_3` kernel is now
   evaluated through the Chebyshev interpolant of
   :math:`e^{\\tau}\\,\\mathrm{Ki}_3(\\tau)` in
   :func:`~.cp_geometry._ki3_mp`, built from
   :func:`~._kernels.ki_n_mp` at 30 dps. The solver
   (``orpheus.cp.solver``) shares the same kernel to guarantee
   bit-identical evaluations between solver ``keff`` and derivation
   ``k_inf``.
"""

from __future__ import annotations

import numpy as np

from . import cp_geometry as _cpg
from ._eigenvalue import kinf_from_cp
from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_xs, get_mixture


# ═══════════════════════════════════════════════════════════════════════
# Geometry singleton (binds the unified infrastructure)
# ═══════════════════════════════════════════════════════════════════════

GEOMETRY = _cpg.CYLINDER_1D


# ═══════════════════════════════════════════════════════════════════════
# Backward-compatible cylindrical CP matrix
# ═══════════════════════════════════════════════════════════════════════

def _cylinder_cp_matrix(
    sig_t_all: np.ndarray,
    radii: np.ndarray,
    volumes: np.ndarray,
    r_cell: float,
    n_quad_y: int = 64,
) -> np.ndarray:
    """Compute the infinite-lattice CP matrix for a cylindrical cell.

    Delegates to :func:`cp_geometry.build_cp_matrix` with the
    pre-bound :data:`~.cp_geometry.CYLINDER_1D` geometry.

    Returns P_inf : (N_reg, N_reg, ng).
    """
    return _cpg.build_cp_matrix(
        GEOMETRY,
        sig_t_all=sig_t_all,
        radii_or_thicknesses=np.asarray(radii, dtype=float),
        volumes=np.asarray(volumes, dtype=float),
        R_cell=float(r_cell),
        n_quad_y=n_quad_y,
    )


# ═══════════════════════════════════════════════════════════════════════
# Cylindrical geometry parameters
# ═══════════════════════════════════════════════════════════════════════

# Radii for each region count (innermost first)
_RADII = {
    1: [1.0],
    2: [0.5, 1.0],
    4: [0.4, 0.45, 0.55, 1.0],
}

# Material IDs per region count (innermost = highest)
_MAT_IDS = {
    1: [2],
    2: [2, 0],
    4: [2, 3, 1, 0],
}


# ═══════════════════════════════════════════════════════════════════════
# Case generation
# ═══════════════════════════════════════════════════════════════════════

def _build_case(ng_key: str, n_regions: int) -> VerificationCase:
    """Build a cylindrical CP verification case."""
    layout = LAYOUTS[n_regions]
    ng = int(ng_key[0])
    radii = np.array(_RADII[n_regions])

    # Annular volumes
    r_inner = np.zeros(n_regions)
    r_inner[1:] = radii[:-1]
    volumes = np.pi * (radii**2 - r_inner**2)

    r_cell = radii[-1]

    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t_all = np.vstack([xs["sig_t"] for xs in xs_list])

    P_inf_g = _cylinder_cp_matrix(sig_t_all, radii, volumes, r_cell)

    k_inf = kinf_from_cp(
        P_inf_g=P_inf_g,
        sig_t_all=sig_t_all,
        V_arr=volumes,
        sig_s_mats=[xs["sig_s"] for xs in xs_list],
        nu_sig_f_mats=[xs["nu"] * xs["sig_f"] for xs in xs_list],
        chi_mats=[xs["chi"] for xs in xs_list],
    )

    mat_ids = _MAT_IDS[n_regions]
    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    geom_params_out = dict(
        radii=radii.tolist(),
        mat_ids=mat_ids,
    )

    name = f"cp_cyl1D_{ng}eg_{n_regions}rg"
    dim = n_regions * ng

    latex = (
        rf"Cylindrical CP eigenvalue with {ng} groups, {n_regions} regions, "
        r"white boundary condition. "
        rf"The Ki₄-based CP matrix yields a {dim}×{dim} eigenvalue problem."
        "\n\n"
        r".. math::" "\n"
        rf"   k_\infty = {k_inf:.10f}"
    )

    labels: list[str] = ["collision-rate", "ki3-def", "chord-length", "self-cyl"]
    if n_regions > 1:
        labels += ["second-diff-cyl", "wigner-seitz"]
    if ng == 1 and n_regions == 1:
        labels.append("one-group-kinf")
    if ng > 1:
        labels += ["matrix-eigenvalue", "mg-balance"]

    return VerificationCase(
        name=name,
        k_inf=k_inf,
        method="cp",
        geometry="cyl1D",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=geom_params_out,
        latex=latex,
        description=f"{ng}G {n_regions}-region cylindrical CP (Ki₄ kernel, white BC)",
        tolerance="< 1e-5",
        vv_level="L1",
        equation_labels=tuple(labels),
    )


def all_cases() -> list[VerificationCase]:
    """Return all cylindrical CP verification cases: {1,2,4}eg × {1,2,4}rg."""
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        for n_regions in [1, 2, 4]:
            cases.append(_build_case(ng_key, n_regions))
    return cases
