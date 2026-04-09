"""Semi-analytical slab collision probability eigenvalues.

Derives k_inf for {1,2,4} energy groups × {1,2,4} regions using the E₃
exponential integral kernel. The CP matrix is computed numerically to
integrator precision; the eigenvalue problem is then a finite matrix solve.
"""

from __future__ import annotations

import numpy as np

from ._eigenvalue import kinf_from_cp
from ._kernels import e3
from ._types import VerificationCase
from ._xs_library import XS, LAYOUTS, get_xs, get_mixture, make_mixture


# ═══════════════════════════════════════════════════════════════════════
# Slab CP matrix from E₃ kernel (N regions, N_g groups)
# ═══════════════════════════════════════════════════════════════════════

def _slab_cp_matrix(
    sig_t_all: np.ndarray,
    t_arr: np.ndarray,
) -> np.ndarray:
    """Compute the infinite-lattice CP matrix for a slab.

    Uses the E₃ second-difference formula with white boundary condition.

    Parameters
    ----------
    sig_t_all : (N_reg, ng) — total XS per region and group
    t_arr : (N_reg,) — region thicknesses

    Returns
    -------
    P_inf : (N_reg, N_reg, ng) — collision probability matrix
    """
    N_reg = len(t_arr)
    ng = sig_t_all.shape[1]
    P_inf_g = np.zeros((N_reg, N_reg, ng))

    for g in range(ng):
        sig_t_g = sig_t_all[:, g]
        tau = sig_t_g * t_arr

        bnd_pos = np.zeros(N_reg + 1)
        for i in range(N_reg):
            bnd_pos[i + 1] = bnd_pos[i] + tau[i]

        rcp = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            rcp[i, i] += 0.5 * sig_t_g[i] * (
                2 * t_arr[i] - (2.0 / sig_t_g[i]) * (0.5 - e3(tau[i]))
            )

            for j in range(N_reg):
                tau_i, tau_j = tau[i], tau[j]

                if j > i:
                    gap_d = max(bnd_pos[j] - bnd_pos[i + 1], 0.0)
                elif j < i:
                    gap_d = max(bnd_pos[i] - bnd_pos[j + 1], 0.0)
                else:
                    gap_d = None

                dd = 0.0
                if gap_d is not None:
                    dd = (e3(gap_d) - e3(gap_d + tau_i)
                          - e3(gap_d + tau_j) + e3(gap_d + tau_i + tau_j))

                gap_c = bnd_pos[i] + bnd_pos[j]
                dc = (e3(gap_c) - e3(gap_c + tau_i)
                      - e3(gap_c + tau_j) + e3(gap_c + tau_i + tau_j))

                rcp[i, j] += 0.5 * (dd + dc)

        P_cell = np.zeros((N_reg, N_reg))
        for i in range(N_reg):
            P_cell[i, :] = rcp[i, :] / (sig_t_g[i] * t_arr[i])

        P_out = np.maximum(1.0 - P_cell.sum(axis=1), 0.0)
        P_in = sig_t_g * t_arr * P_out
        P_inout = max(1.0 - P_in.sum(), 0.0)
        P_inf_g[:, :, g] = P_cell + np.outer(P_out, P_in) / (1.0 - P_inout)

    return P_inf_g


# ═══════════════════════════════════════════════════════════════════════
# Slab geometry parameters
# ═══════════════════════════════════════════════════════════════════════

# Region thicknesses (innermost to outermost)
_THICKNESSES = {
    1: [0.5],                       # A only
    2: [0.5, 0.5],                  # A + B
    4: [0.4, 0.05, 0.1, 0.45],     # A + D + C + B
}

# Material IDs per region count, matching solver convention
# (innermost = highest ID)
_MAT_IDS = {
    1: [2],          # A → fuel(2)
    2: [2, 0],       # A → fuel(2), B → cool(0)
    4: [2, 3, 1, 0], # A → fuel(2), D → gap(3), C → clad(1), B → cool(0)
}


# ═══════════════════════════════════════════════════════════════════════
# Case generation
# ═══════════════════════════════════════════════════════════════════════

def _build_case(ng_key: str, n_regions: int) -> VerificationCase:
    """Build a slab CP verification case for given groups and regions."""
    layout = LAYOUTS[n_regions]
    ng = int(ng_key[0])
    t_arr = np.array(_THICKNESSES[n_regions])

    # Collect XS per region (innermost first)
    xs_list = [get_xs(region, ng_key) for region in layout]
    sig_t_all = np.vstack([xs["sig_t"] for xs in xs_list])

    P_inf_g = _slab_cp_matrix(sig_t_all, t_arr)

    k_inf = kinf_from_cp(
        P_inf_g=P_inf_g,
        sig_t_all=sig_t_all,
        V_arr=t_arr,
        sig_s_mats=[xs["sig_s"] for xs in xs_list],
        nu_sig_f_mats=[xs["nu"] * xs["sig_f"] for xs in xs_list],
        chi_mats=[xs["chi"] for xs in xs_list],
    )

    # Build materials dict with mat_ids matching the solver convention
    mat_ids = _MAT_IDS[n_regions]
    materials = {}
    for i, region in enumerate(layout):
        materials[mat_ids[i]] = get_mixture(region, ng_key)

    # Geometry params for building SlabGeometry in solver tests
    # Use thicknesses and mat_ids directly
    geom_params_out = dict(
        thicknesses=t_arr.tolist(),
        mat_ids=mat_ids,
    )

    name = f"cp_slab_{ng}eg_{n_regions}rg"
    dim = n_regions * ng

    latex = (
        rf"Slab CP eigenvalue with {ng} groups, {n_regions} regions, "
        r"white boundary condition. "
        rf"The E₃-based CP matrix yields a {dim}×{dim} eigenvalue problem."
        "\n\n"
        r".. math::" "\n"
        rf"   k_\infty = {k_inf:.10f}"
    )

    return VerificationCase(
        name=name,
        k_inf=k_inf,
        method="cp",
        geometry="slab",
        n_groups=ng,
        n_regions=n_regions,
        materials=materials,
        geom_params=geom_params_out,
        latex=latex,
        description=f"{ng}G {n_regions}-region slab CP (E₃ kernel, white BC)",
        tolerance="< 1e-6",
    )


def all_cases() -> list[VerificationCase]:
    """Return all slab CP verification cases: {1,2,4}eg × {1,2,4}rg."""
    cases = []
    for ng_key in ["1g", "2g", "4g"]:
        for n_regions in [1, 2, 4]:
            cases.append(_build_case(ng_key, n_regions))
    return cases
