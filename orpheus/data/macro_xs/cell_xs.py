"""Per-cell cross-section assembly for spatial transport solvers.

Extracts macroscopic cross sections from Mixture objects and maps them
onto a spatial mesh defined by material IDs per cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mixture import Mixture


@dataclass
class CellXS:
    """Per-cell macroscopic cross sections on a spatial mesh.

    All arrays have shape (N_cells, ng) where N_cells is the number
    of spatial cells and ng is the number of energy groups.
    """

    sig_t: np.ndarray    # total XS
    sig_a: np.ndarray    # absorption XS
    sig_p: np.ndarray    # production XS (nu * sigma_f)
    chi: np.ndarray      # fission spectrum


def assemble_cell_xs(
    materials: dict[int, Mixture],
    mat_ids: np.ndarray,
) -> CellXS:
    """Build per-cell XS arrays from a material map.

    Parameters
    ----------
    materials : dict mapping material ID to Mixture.
    mat_ids : (N_cells,) int array of material IDs per cell.

    Returns
    -------
    CellXS with arrays of shape (N_cells, ng).
    """
    flat = mat_ids.ravel()
    nc = len(flat)
    _any_mat = next(iter(materials.values()))
    ng = _any_mat.ng

    sig_t = np.empty((nc, ng))
    sig_a = np.empty((nc, ng))
    sig_p = np.empty((nc, ng))
    chi = np.empty((nc, ng))

    for i, mid in enumerate(flat):
        m = materials[mid]
        sig_t[i] = m.SigT
        sig_a[i] = m.absorption_xs
        sig_p[i] = m.SigP
        chi[i] = m.chi

    return CellXS(sig_t=sig_t, sig_a=sig_a, sig_p=sig_p, chi=chi)
