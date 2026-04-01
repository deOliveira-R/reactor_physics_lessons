"""HDF5 serialization for Isotope cross section data.

Each isotope+temperature is stored as an HDF5 group inside a per-element
file.  For example, ``H_001.h5`` contains groups ``/294K``, ``/350K``, etc.

Layout inside each temperature group::

    /{temp_K}K/
        aw          : scalar
        temp        : scalar
        eg          : (NG+1,)
        sig0        : (n_sig0,)
        sigC        : (n_sig0, NG)
        sigL        : (n_sig0, NG)
        sigF        : (n_sig0, NG)
        sigT        : (n_sig0, NG)
        nubar       : (NG,)
        chi         : (NG,)
        sig2/
            row     : (nnz,)
            col     : (nnz,)
            data    : (nnz,)
        sigS/
            L{j}_S{k}/
                row  : (nnz,)
                col  : (nnz,)
                data : (nnz,)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from .isotope import NG, Isotope


def save_isotope(iso: Isotope, h5file: h5py.File) -> None:
    """Write one Isotope into an open HDF5 file as a temperature group."""
    temp_K = int(round(iso.temp))
    grp = h5file.require_group(f"{temp_K}K")

    grp.attrs["aw"] = iso.aw
    grp.attrs["temp"] = iso.temp

    for name, data in [
        ("eg", iso.eg), ("sig0", iso.sig0),
        ("sigC", iso.sigC), ("sigL", iso.sigL),
        ("sigF", iso.sigF), ("sigT", iso.sigT),
        ("nubar", iso.nubar), ("chi", iso.chi),
    ]:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data, compression="gzip", compression_opts=4)

    # Sparse matrices: store as COO triplets
    _save_sparse(grp, "sig2", iso.sig2)

    sig_s_grp = grp.require_group("sigS")
    for j, legendre_mats in enumerate(iso.sigS):
        for k, mat in enumerate(legendre_mats):
            _save_sparse(sig_s_grp, f"L{j}_S{k}", mat)


def load_isotope_h5(path: Path, temp_K: int) -> Isotope:
    """Load an Isotope from an HDF5 file for a given temperature."""
    with h5py.File(path, "r") as f:
        grp = f[f"{temp_K}K"]

        aw = float(grp.attrs["aw"])
        temp = float(grp.attrs["temp"])
        eg = grp["eg"][:]
        sig0 = grp["sig0"][:]
        n_sig0 = len(sig0)

        sigC = grp["sigC"][:]
        sigL = grp["sigL"][:]
        sigF = grp["sigF"][:]
        sigT = grp["sigT"][:]
        nubar = grp["nubar"][:]
        chi = grp["chi"][:]

        sig2 = _load_sparse(grp, "sig2")

        # Reconstruct sigS structure: [legendre][sig0_idx]
        sig_s_grp = grp["sigS"]
        n_legendre = max(int(k.split("_")[0][1:]) for k in sig_s_grp.keys()) + 1
        sigS = [
            [_load_sparse(sig_s_grp, f"L{j}_S{k}") for k in range(n_sig0)]
            for j in range(n_legendre)
        ]

    name = path.stem + f"_{temp_K}K"
    return Isotope(
        name=name, aw=aw, temp=temp, eg=eg, sig0=sig0,
        sigC=sigC, sigL=sigL, sigF=sigF, sigT=sigT,
        nubar=nubar, chi=chi, sigS=sigS, sig2=sig2,
    )


def _save_sparse(parent: h5py.Group, name: str, mat: csr_matrix) -> None:
    """Save a sparse matrix as COO triplets."""
    grp = parent.require_group(name)
    coo = mat.tocoo()
    for key in ("row", "col", "data"):
        if key in grp:
            del grp[key]
    grp.create_dataset("row", data=coo.row.astype(np.int32), compression="gzip")
    grp.create_dataset("col", data=coo.col.astype(np.int32), compression="gzip")
    grp.create_dataset("data", data=coo.data, compression="gzip")


def _load_sparse(parent: h5py.Group, name: str) -> csr_matrix:
    """Load a sparse matrix from COO triplets."""
    grp = parent[name]
    row = grp["row"][:]
    col = grp["col"][:]
    data = grp["data"][:]
    return coo_matrix((data, (row, col)), shape=(NG, NG)).tocsr()
