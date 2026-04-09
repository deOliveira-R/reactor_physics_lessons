"""Shared eigenvalue solvers for analytical verification cases.

Two functions cover all derivation eigenvalue computations:

* :func:`kinf_homogeneous` — infinite medium, any group count.
* :func:`kinf_from_cp` — multi-region CP transport with P_inf matrices.

Both support an optional ``sig_2`` / ``sig_2_mats`` parameter for
(n,2n) reactions.  When omitted, (n,2n) is treated as zero.
"""

from __future__ import annotations

import numpy as np


def kinf_homogeneous(
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    chi: np.ndarray,
    sig_2: np.ndarray | None = None,
) -> float:
    r"""Infinite-medium eigenvalue for any number of energy groups.

    Solves :math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})` where

    .. math::

        \mathbf{A} = \text{diag}(\Sigma_t) - (\Sigma_s + 2\Sigma_2)^T

        \mathbf{F} = \chi \otimes (\nu\Sigma_f)

    For 1-group, this reduces to :math:`k = \nu\Sigma_f / \Sigma_a`.

    Parameters
    ----------
    sig_t : (ng,) total XS.
    sig_s : (ng, ng) P0 scattering matrix, ``[from, to]`` convention.
    nu_sig_f : (ng,) production XS (nu * sigma_f).
    chi : (ng,) fission spectrum.
    sig_2 : (ng, ng) or None — (n,2n) transfer matrix.
    """
    ng = len(sig_t)

    # Effective scattering: Σs + 2·Σ₂
    sig_s_eff = sig_s.copy()
    if sig_2 is not None:
        sig_s_eff = sig_s_eff + 2.0 * sig_2

    A = np.diag(sig_t) - sig_s_eff.T
    F = np.outer(chi, nu_sig_f)
    M = np.linalg.solve(A, F)
    return float(np.max(np.real(np.linalg.eigvals(M))))


def kinf_from_cp(
    P_inf_g: np.ndarray,
    sig_t_all: np.ndarray,
    V_arr: np.ndarray,
    sig_s_mats: list[np.ndarray],
    nu_sig_f_mats: list[np.ndarray],
    chi_mats: list[np.ndarray],
    sig_2_mats: list[np.ndarray] | None = None,
) -> float:
    r"""Multi-region CP eigenvalue via dense eigensolver.

    Builds the generalized eigenvalue problem
    :math:`\mathbf{A}\phi = \frac{1}{k}\mathbf{B}\phi` and returns
    :math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{B})`.

    The matrices are:

    .. math::

        A_{ig,jg'} = \delta_{ij}\delta_{gg'}\,\Sigma_{t,ig} V_i
                    - P_{ji,g}\, V_j\, (\Sigma_{s,j} + 2\Sigma_{2,j})_{g' \to g}

        B_{ig,jg'} = P_{ji,g}\, V_j\, \chi_{j,g}\, (\nu\Sigma_f)_{j,g'}

    Parameters
    ----------
    P_inf_g : (N_reg, N_reg, ng) — CP matrices per group.
    sig_t_all : (N_reg, ng) — total XS per region and group.
    V_arr : (N_reg,) — region volumes.
    sig_s_mats : list of (ng, ng) — scattering matrices per region.
    nu_sig_f_mats : list of (ng,) — production XS per region.
    chi_mats : list of (ng,) — fission spectrum per region.
    sig_2_mats : list of (ng, ng) or None — (n,2n) matrices per region.
    """
    N_reg = P_inf_g.shape[0]
    ng = P_inf_g.shape[2]
    dim = N_reg * ng

    A_mat = np.zeros((dim, dim))
    B_mat = np.zeros((dim, dim))

    for i_reg in range(N_reg):
        for g in range(ng):
            row = i_reg * ng + g
            A_mat[row, row] = sig_t_all[i_reg, g] * V_arr[i_reg]

            for j_reg in range(N_reg):
                for gp in range(ng):
                    col = j_reg * ng + gp
                    Pji = P_inf_g[j_reg, i_reg, g]

                    scat = sig_s_mats[j_reg][gp, g]
                    if sig_2_mats is not None:
                        scat = scat + 2.0 * sig_2_mats[j_reg][gp, g]

                    A_mat[row, col] -= Pji * V_arr[j_reg] * scat
                    B_mat[row, col] += (
                        Pji * V_arr[j_reg]
                        * chi_mats[j_reg][g]
                        * nu_sig_f_mats[j_reg][gp]
                    )

    M = np.linalg.solve(A_mat, B_mat)
    return float(np.max(np.real(np.linalg.eigvals(M))))
