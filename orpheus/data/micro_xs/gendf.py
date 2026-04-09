"""GENDF (GXS) file parser.

Reads the IAEA 421-group GENDF files directly — no CSV intermediary needed.
The format is fixed-width 80-column ENDF-style records:

    Columns  1-66 : 6 data fields, 11 characters each
    Columns 67-70 : MAT number
    Columns 71-72 : MF  number
    Columns 73-75 : MT  number
    Columns 76-80 : line sequence number

Data fields use a compact float notation where 'E' is omitted,
e.g. ``1.001000+3`` means ``1.001E+3``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .isotope import NG, Isotope

_GXS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Low-level GENDF parsing
# ---------------------------------------------------------------------------

def _parse_gendf_field(s: str) -> float:
    """Parse one 11-character GENDF data field into a float.

    Handles the compact notation where 'E' is omitted:
    ``' 1.001000+3'`` → ``1.001e+3``.
    """
    s = s.strip()
    if not s:
        return 0.0
    # Insert 'E' before +/- sign that follows a digit (but not at position 0)
    s = re.sub(r"(\d)([+-])", r"\1E\2", s)
    return float(s)


def _parse_gendf(path: Path) -> np.ndarray:
    """Read a GXS file into a numeric matrix (n_lines, 10).

    Columns: [data1..data6, MAT, MF, MT, line_seq].
    Mirrors what MATLAB's ``importdata('file.CSV', ';')`` produces.
    """
    rows = []
    with open(path) as f:
        for line in f:
            if len(line.rstrip("\n")) < 75:
                continue
            data = [_parse_gendf_field(line[i * 11 : (i + 1) * 11]) for i in range(6)]
            mat = int(line[66:70])
            mf = int(line[70:72])
            mt = int(line[72:75])
            seq = int(line[75:80])
            rows.append(data + [mat, mf, mt, seq])
    return np.array(rows)


# ---------------------------------------------------------------------------
# Cross section extraction (ports of MATLAB extract_mf3, extract_mf6)
# ---------------------------------------------------------------------------

def _extract_n_words(n: int, i_row: int, m: np.ndarray) -> tuple[np.ndarray, int]:
    """Read *n* consecutive data words starting at row ``i_row``.

    Words are packed 6 per row in columns 0-5.  Returns the extracted
    values and the row index of the last row read.
    """
    a = []
    row = i_row
    full_lines = n // 6
    for _ in range(full_lines):
        a.extend(m[row, :6])
        row += 1
    remainder = n - full_lines * 6
    if remainder > 0:
        a.extend(m[row, :remainder])
    else:
        row -= 1  # last full row was the final one
    return np.array(a), row


def _extract_mf3(mt: int, temp_idx: int, m: np.ndarray) -> np.ndarray | None:
    """Extract MF=3 cross sections for reaction *mt* at temperature index *temp_idx*.

    Returns (n_sig0, NG) array, or None if the reaction is absent.
    """
    n_row = m.shape[0]
    n_temp = 0
    i_row_found = 0

    for i in range(n_row):
        if m[i, 7] == 3 and m[i, 8] == mt and m[i, 9] == 1:
            n_temp += 1
            if n_temp == temp_idx:
                i_row_found = i + 1
                break

    if i_row_found == 0:
        return None

    n_sig0 = int(m[i_row_found - 1, 3])
    n_lgn = int(m[i_row_found - 1, 2])
    sig = np.zeros((n_sig0, NG))

    i = i_row_found + 1
    while i < n_row and m[i, 7] == 3 and m[i, 8] == mt:
        ig = int(m[i - 1, 5]) - 1  # 0-based group index
        n_words = n_sig0 * n_lgn * 2
        a, i_new = _extract_n_words(n_words, i, m)
        # The second half contains the XS values for each sigma-zero
        sig[:, ig] = a[n_sig0 * n_lgn : n_sig0 * n_lgn + n_sig0]
        i = i_new + 2
    return sig


def _extract_mf6(
    mt: int, temp_idx: int, m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]] | None:
    """Extract MF=6 transfer matrix for reaction *mt* at temperature index *temp_idx*.

    Returns (ifrom, ito, sig_dict) where sig_dict[(legendre, sig0_idx)] is
    a 1-D array of non-zero values.  ifrom/ito are 1-based group indices
    (matching MATLAB convention for later sparse matrix construction).
    Returns None if the reaction is absent.
    """
    n_row = m.shape[0]
    i = 0
    n_temp = 0
    ifrom_list: list[int] = []
    ito_list: list[int] = []
    sig: dict[tuple[int, int], list[float]] = {}

    while i < n_row and m[i, 6] != -1:
        if m[i, 7] == 6 and m[i, 8] == mt:
            if m[i, 9] == 1:  # first record of this MF/MT section
                n_lgn = int(m[i, 2])
                n_sig0 = int(m[i, 3])
                i += 1
                n_temp += 1

            ng2 = int(m[i, 2])      # number of secondary positions
            ig2lo = int(m[i, 3])    # lowest nonzero group (1-based)
            nw = int(m[i, 4])       # words to read
            ig = int(m[i, 5])       # current group (1-based)

            i += 1
            a, i_new = _extract_n_words(nw, i, m)
            i = i_new

            if n_temp == temp_idx:
                k = n_lgn * n_sig0  # skip flux words
                for i_to in range(ig2lo, ig2lo + ng2 - 1):
                    ifrom_list.append(ig)
                    ito_list.append(i_to)
                    for i_sig0 in range(n_sig0):
                        for i_lgn in range(n_lgn):
                            k += 1
                            sig.setdefault((i_lgn, i_sig0), []).append(a[k - 1])
                        if n_lgn == 1:
                            sig.setdefault((1, i_sig0), []).append(0.0)
                            sig.setdefault((2, i_sig0), []).append(0.0)
        i += 1

    if n_temp == 0:
        return None

    ifrom = np.array(ifrom_list)
    ito = np.array(ito_list)
    sig_arrays = {key: np.array(vals) for key, vals in sig.items()}
    return ifrom, ito, sig_arrays


# ---------------------------------------------------------------------------
# High-level: GXS → Isotope
# ---------------------------------------------------------------------------

_IG_THRESH = 95  # last group of thermal energy (E ≈ 4 eV)


def convert_gxs(name: str) -> list[Isotope]:
    """Convert a GXS file to a list of Isotope objects (one per temperature).

    Parameters
    ----------
    name : str
        Isotope identifier matching the GXS filename, e.g. ``"H_001"``,
        ``"U_235"``, ``"ZR090"``.

    Returns
    -------
    list[Isotope] — one entry per temperature found in the file.
    """
    path = _GXS_DIR / f"{name}.GXS"
    if not path.exists():
        raise FileNotFoundError(f"No GXS file: {path}")

    print(f"  Parsing {path.name}...", end=" ", flush=True)
    m = _parse_gendf(path)
    print(f"{m.shape[0]} records.", flush=True)

    # --- Header: temperatures, sigma-zeros, energy grid (MF=1, MT=451) ---
    temps: list[float] = []
    for i in range(m.shape[0]):
        if m[i, 7] == 1 and m[i, 8] == 451 and m[i, 9] == 2:
            temps.append(m[i, 0])

    n_sig0 = int(m[1, 3])
    header_words = 1 + n_sig0 + (NG + 1)
    a, _ = _extract_n_words(header_words, 3, m)
    sig0 = a[1 : 1 + n_sig0]
    eg = a[1 + n_sig0 : 1 + n_sig0 + NG + 1]
    aw = m[1, 1] * 1.008664916  # convert to amu

    isotopes = []
    for i_temp, temp in enumerate(temps, start=1):
        iso = _build_isotope(name, temp, i_temp, m, aw, eg, sig0, n_sig0)
        isotopes.append(iso)

    return isotopes


def _build_isotope(
    name: str,
    temp: float,
    i_temp: int,
    m: np.ndarray,
    aw: float,
    eg: np.ndarray,
    sig0: np.ndarray,
    n_sig0: int,
) -> Isotope:
    """Build one Isotope from parsed GENDF data for a given temperature."""
    # --- MF=3 reactions ---
    sigC = _extract_mf3(102, i_temp, m)  # radiative capture
    if sigC is None:
        sigC = np.zeros((n_sig0, NG))
    elif sigC.shape[0] == 1 and n_sig0 > 1:
        sigC = np.tile(sigC, (n_sig0, 1))

    sigL_raw = _extract_mf3(107, i_temp, m)  # (n,alpha)
    if sigL_raw is None:
        sigL = np.zeros((n_sig0, NG))
    elif sigL_raw.shape[0] == 1 and n_sig0 > 1:
        sigL = np.tile(sigL_raw, (n_sig0, 1))
    else:
        sigL = sigL_raw

    sigF_raw = _extract_mf3(18, i_temp, m)  # fission
    if sigF_raw is None:
        sigF = np.zeros((n_sig0, NG))
    else:
        sigF = sigF_raw

    nubar_raw = _extract_mf3(452, i_temp, m)  # nubar
    nubar = nubar_raw[0] if nubar_raw is not None else np.zeros(NG)

    # --- (n,2n) matrix: MF=6, MT=16 ---
    n2n = _extract_mf6(16, i_temp, m)
    if n2n is not None:
        ifrom2, ito2, sig2_data = n2n
        sig2 = coo_matrix(
            (sig2_data[(0, 0)], (ifrom2 - 1, ito2 - 1)), shape=(NG, NG)
        ).tocsr()
    else:
        sig2 = csr_matrix((NG, NG))

    # --- Scattering matrices: elastic + inelastic + thermal ---
    # Elastic: MF=6, MT=2
    elastic = _extract_mf6(2, i_temp, m)
    sigS = _init_scattering(elastic, n_sig0)

    # Inelastic: MF=6, MT=51..91
    for mt in range(51, 92):
        inel = _extract_mf6(mt, i_temp, m)
        if inel is not None:
            _accumulate_scattering(sigS, inel, n_sig0, sigma_zero_independent=True)

    # Thermal: MT=222 for H-in-water, MT=221 for free gas
    thermal_mt = 222 if name.startswith("H_001") else 221
    thermal = _extract_mf6(thermal_mt, i_temp, m)
    if thermal is not None:
        _accumulate_scattering(sigS, thermal, n_sig0, sigma_zero_independent=True)

    # --- Fission spectrum (chi): MF=6, MT=18 ---
    chi = _extract_chi(i_temp, m)

    # --- Total cross section (computed from components) ---
    sigT = np.zeros((n_sig0, NG))
    for i_sig0 in range(n_sig0):
        row_sums = np.array(sigS[0][i_sig0].sum(axis=1)).ravel()
        sigT[i_sig0] = sigC[i_sig0] + sigF[i_sig0] + sigL[i_sig0] + row_sums
        if sig2.nnz > 0:
            sigT[i_sig0] += np.array(sig2.sum(axis=1)).ravel()

    temp_K = int(round(temp))
    return Isotope(
        name=f"{name}_{temp_K}K",
        aw=aw,
        temp=temp,
        eg=eg,
        sig0=sig0,
        sigC=sigC,
        sigL=sigL,
        sigF=sigF,
        sigT=sigT,
        nubar=nubar,
        chi=chi,
        sigS=sigS,
        sig2=sig2,
    )



def _extract_chi(i_temp: int, m: np.ndarray) -> np.ndarray:
    """Extract the fission spectrum (chi) from MF=6, MT=18."""
    # Find MF=6, MT=18
    i = 0
    while i < m.shape[0]:
        if m[i, 7] == 6 and m[i, 8] == 18:
            break
        i += 1
    else:
        return np.zeros(NG)

    i += 1  # move to second record
    ig2lo = int(m[i, 3])
    nw = int(m[i, 4])
    i += 1
    a, _ = _extract_n_words(nw, i, m)

    chi = np.zeros(NG)
    for j in range(nw):
        chi[ig2lo - 1 + j] = a[j]
    total = chi.sum()
    if total > 0:
        chi /= total
    return chi


def _init_scattering(
    elastic: tuple | None, n_sig0: int
) -> list[list[csr_matrix]]:
    """Initialize the 3-Legendre × n_sig0 scattering matrix list from elastic data."""
    sigS: list[list[csr_matrix]] = [
        [csr_matrix((NG, NG)) for _ in range(n_sig0)] for _ in range(3)
    ]

    if elastic is None:
        return sigS

    ifrom, ito, data = elastic

    # Zero out thermal groups for elastic (they're handled by thermal scattering)
    thermal_mask = ifrom <= _IG_THRESH

    for j_lgn in range(3):
        for i_sig0 in range(n_sig0):
            key = (j_lgn, i_sig0)
            if key in data:
                vals = data[key].copy()
                vals[thermal_mask] = 0.0
                vals += 1e-30  # match MATLAB's +1e-30 to avoid exact zeros
                sigS[j_lgn][i_sig0] = coo_matrix(
                    (vals, (ifrom - 1, ito - 1)), shape=(NG, NG)
                ).tocsr()
    return sigS


def _accumulate_scattering(
    sigS: list[list[csr_matrix]],
    reaction: tuple,
    n_sig0: int,
    sigma_zero_independent: bool = False,
) -> None:
    """Add a scattering reaction (inelastic or thermal) into sigS."""
    ifrom, ito, data = reaction

    for j_lgn in range(3):
        for i_sig0 in range(n_sig0):
            # Inelastic/thermal: same for all sigma-zeros (use sig0=0 data)
            src_key = (j_lgn, 0) if sigma_zero_independent else (j_lgn, i_sig0)
            if src_key in data:
                vals = data[src_key] + 1e-30
                addition = coo_matrix(
                    (vals, (ifrom - 1, ito - 1)), shape=(NG, NG)
                ).tocsr()
                sigS[j_lgn][i_sig0] = sigS[j_lgn][i_sig0] + addition
