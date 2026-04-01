"""Interpolation of microscopic cross sections at computed sigma-zeros."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from data.micro_xs.isotope import NG, Isotope


def interp_xs_field(
    field: np.ndarray,
    iso: Isotope,
    sig0_row: np.ndarray,
) -> np.ndarray:
    """Interpolate a (n_sig0, NG) cross section field at given sigma-zeros.

    Parameters
    ----------
    field : (n_sig0, NG) array
        Tabulated XS values (e.g. iso.sigC, iso.sigF, etc.)
    iso : Isotope
        The isotope (for sigma-zero base points).
    sig0_row : (NG,) array
        Target sigma-zero for this isotope in each energy group.

    Returns
    -------
    (NG,) array of interpolated cross sections.
    """
    if iso.n_sig0 == 1:
        return field[0].copy()

    # sig0 is in decreasing order; reverse for np.interp (needs increasing xp)
    xp = np.log10(iso.sig0)[::-1]
    log_target = np.clip(np.log10(sig0_row), 0.0, 10.0)

    result = np.empty(NG)
    for ig in range(NG):
        result[ig] = np.interp(log_target[ig], xp, field[::-1, ig])
    return result


def interp_sig_s(
    iso: Isotope,
    legendre: int,
    sig0_row: np.ndarray,
) -> csr_matrix:
    """Interpolate a scattering matrix for one Legendre order.

    Parameters
    ----------
    iso : Isotope
        Source isotope with tabulated scattering matrices.
    legendre : int
        Legendre order index (0, 1, or 2).
    sig0_row : (NG,) array
        Target sigma-zero for this isotope in each energy group.

    Returns
    -------
    (NG, NG) sparse scattering matrix interpolated at the target sigma-zeros.
    """
    if iso.n_sig0 == 1:
        return iso.sigS[legendre][0].copy()

    matrices = iso.sigS[legendre]  # list of n_sig0 sparse matrices

    # All sigma-zero variants share the same sparsity pattern
    ref = matrices[0].tocoo()
    ifrom = ref.row
    ito = ref.col
    n_nz = ref.nnz

    # Extract values at known positions from each sigma-zero table
    val_table = np.array([mat.tocsr()[ifrom, ito].A1 for mat in matrices])

    # sig0 is in decreasing order; reverse for np.interp
    xp = np.log10(iso.sig0)[::-1]
    log_target = np.clip(np.log10(sig0_row), 0.0, 10.0)
    val_table_rev = val_table[::-1]

    # Interpolate each non-zero using the from-group's sigma-zero
    interp_vals = np.empty(n_nz)
    for i in range(n_nz):
        interp_vals[i] = np.interp(log_target[ifrom[i]], xp, val_table_rev[:, i])

    return coo_matrix((interp_vals, (ifrom, ito)), shape=(NG, NG)).tocsr()
