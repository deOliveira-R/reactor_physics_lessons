"""Semi-analytical cylindrical collision probability eigenvalues.

Derives k_inf for {1,2,4} energy groups × {1,2,4} regions using the
canonical Ki₃ (Bickley-Naylor) kernel as the second-difference
anti-derivative. The CP matrix is computed numerically via
y-quadrature; the eigenvalue problem is a finite matrix solve.

.. note::

   Phase-4.2 resolved the naming discrepancy documented in #94:
   the legacy ``BickleyTables.ki4`` method computes the canonical
   Ki₃ (A&S convention).  This module now uses the canonical-named
   alias ``Ki3_vec`` for clarity.
"""

from __future__ import annotations

import numpy as np

from ._eigenvalue import kinf_from_cp
from ._kernels import bickley_tables
from ._types import VerificationCase
from ._xs_library import LAYOUTS, get_xs, get_mixture


# ═══════════════════════════════════════════════════════════════════════
# Cylindrical CP matrix from Ki₄ kernel
# ═══════════════════════════════════════════════════════════════════════

def _chord_half_lengths(radii: np.ndarray, y_pts: np.ndarray) -> np.ndarray:
    """Half-chord lengths l_k(y) for each annular region. Shape (N, n_y)."""
    N = len(radii)
    chords = np.zeros((N, len(y_pts)))
    r_inner = np.zeros(N)
    r_inner[1:] = radii[:-1]
    y2 = y_pts**2

    for k in range(N):
        r_out, r_in = radii[k], r_inner[k]
        outer = np.sqrt(np.maximum(r_out**2 - y2, 0.0))
        if r_in > 0:
            inner = np.sqrt(np.maximum(r_in**2 - y2, 0.0))
            mask = y_pts < r_in
            chords[k, mask] = outer[mask] - inner[mask]
        mask_p = (y_pts >= r_in) & (y_pts < r_out)
        chords[k, mask_p] = outer[mask_p]

    return chords


def _cylinder_cp_matrix(
    sig_t_all: np.ndarray,
    radii: np.ndarray,
    volumes: np.ndarray,
    r_cell: float,
    n_quad_y: int = 64,
) -> np.ndarray:
    """Compute the infinite-lattice CP matrix for a cylindrical cell.

    Returns P_inf : (N_reg, N_reg, ng).
    """
    N_reg = len(radii)
    ng = sig_t_all.shape[1]
    tables = bickley_tables()

    gl_pts, gl_wts = np.polynomial.legendre.leggauss(n_quad_y)
    breakpoints = np.concatenate(([0.0], radii))
    y_all, w_all = [], []
    for seg in range(len(breakpoints) - 1):
        a, b = breakpoints[seg], breakpoints[seg + 1]
        y_all.append(0.5 * (b - a) * gl_pts + 0.5 * (b + a))
        w_all.append(0.5 * (b - a) * gl_wts)
    y_pts = np.concatenate(y_all)
    y_wts = np.concatenate(w_all)
    chords = _chord_half_lengths(radii, y_pts)
    n_y = len(y_pts)
    ki4_0 = tables.Ki3_vec(np.zeros(n_y))

    P_inf_g = np.empty((N_reg, N_reg, ng))

    for g in range(ng):
        sig_t_g = sig_t_all[:, g]
        tau = sig_t_g[:, None] * chords

        bnd_pos = np.zeros((N_reg + 1, n_y))
        for k in range(N_reg):
            bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]

        rcp = np.zeros((N_reg, N_reg))

        for i in range(N_reg):
            tau_i = tau[i, :]
            sti = sig_t_g[i]
            if sti == 0:
                continue

            self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
                ki4_0 - tables.Ki3_vec(tau_i)
            )
            rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

            for j in range(N_reg):
                tau_j = tau[j, :]
                if j > i:
                    gap_d = np.maximum(bnd_pos[j, :] - bnd_pos[i + 1, :], 0.0)
                elif j < i:
                    gap_d = np.maximum(bnd_pos[i, :] - bnd_pos[j + 1, :], 0.0)
                else:
                    gap_d = None

                if gap_d is not None:
                    dd = (tables.Ki3_vec(gap_d)
                          - tables.Ki3_vec(gap_d + tau_i)
                          - tables.Ki3_vec(gap_d + tau_j)
                          + tables.Ki3_vec(gap_d + tau_i + tau_j))
                else:
                    dd = np.zeros(n_y)

                gap_c = bnd_pos[i, :] + bnd_pos[j, :]
                dc = (tables.Ki3_vec(gap_c)
                      - tables.Ki3_vec(gap_c + tau_i)
                      - tables.Ki3_vec(gap_c + tau_j)
                      + tables.Ki3_vec(gap_c + tau_i + tau_j))

                rcp[i, j] += 2.0 * np.dot(y_wts, dd + dc)

        P_cell = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            if sig_t_g[i] * volumes[i] > 0:
                P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * volumes[i])

        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        S_cell = 2.0 * np.pi * r_cell
        P_in = sig_t_g * volumes * P_out / S_cell
        P_inout = max(1.0 - P_in.sum(), 0.0)

        P_inf = P_cell.copy()
        if P_inout < 1.0:
            P_inf += np.outer(P_out, P_in) / (1.0 - P_inout)
        P_inf_g[:, :, g] = P_inf

    return P_inf_g


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
