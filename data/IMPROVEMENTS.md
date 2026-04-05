# Data Package — Improvement Tracker

Central registry of improvements for cross-section data infrastructure.

## Tracking Number Format

`DA-YYYYMMDD-NNN` where DA = Data, YYYYMMDD = session date, NNN = sequence.

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## DONE — Implemented and Documented

### DA-20260405-001 — Mixture utility properties

**Status**: DONE  
**Commit**: 443c790  
**Sphinx**: docs/api/data.rst (autodoc)

Added properties to `Mixture` dataclass:
- `absorption_xs` — fission + capture + (n,alpha) + (n,2n) out
- `total_scattering_xs` — P0 row sum (in + out)
- `in_scattering_xs` — in-group elastic (P0 diagonal)
- `out_scattering_xs` — out-of-group (P0 off-diagonal sum)

Eliminates 6+ copies of the same absorption formula across solver modules.

### DA-20260405-002 — Shared per-cell XS assembly (CellXS)

**Status**: DONE  
**Commit**: e79a363  
**Sphinx**: docs/api/data.rst (autodoc)

Created `data/macro_xs/cell_xs.py` with:
- `CellXS` dataclass (sig_t, sig_a, sig_p, chi per cell)
- `assemble_cell_xs(materials, mat_ids)` function

Replaces identical per-cell XS extraction loops duplicated in CP, DO,
SN-1D, and MOC solvers.

### DA-20260405-003 — NG inferred from data (not global constant)

**Status**: DONE
**Commit**: fa5732c

Replaced the hardcoded `NG = 421` global constant with data-inferred
group count:

- `Isotope.ng` property: returns `len(self.eg) - 1`
- `Mixture.ng` property: returns `len(self.SigT)`
- HDF5 loader: infers sparse matrix shape from `eg` dataset
- ALL solver modules (01–04, 09) and plotting modules refactored to
  use `mix.ng` instead of importing NG
- Data pipeline (`sigma_zeros.py`, `interpolation.py`, `gendf.py`)
  retains `NG = 421` for GENDF-specific operations

This enabled synthetic benchmarks with arbitrary group counts (1, 2, 4)
and was the prerequisite for the formal verification system.

---

## OPEN — Not Yet Implemented

### DA-20260405-004 — Production-weighted fission spectrum

**Priority**: Low | **Effort**: Small

`compute_macro_xs` uses the first fissile isotope's chi spectrum.
Should use production-weighted average:
`chi_mix = sum_i(nu_i * Sigma_f_i * chi_i) / sum_i(nu_i * Sigma_f_i)`.
Currently acceptable for single-fissile problems (UO2) but incorrect
for MOX or mixed assemblies.

### DA-20260405-005 — Replace pyXSteam viscosity entirely with IAPWS

**Priority**: Low | **Effort**: Small

`data/materials/h2o_properties.py` line 38 has a TODO: consider using
`_iapws_viscosity()` for ALL viscosity calls instead of `_st.my_ph()`.
The IAPWS function is bit-identical to pyXSteam below 900 °C (validated
at 8 temperature points, ratio = 1.000000) and works above 900 °C where
pyXSteam returns NaN.  It also avoids pyXSteam's region dispatch overhead
since T and rho are already available at the call sites.  Currently used
only as a NaN fallback (TH-20260405-001).

### DA-20260405-006 — Scattering matrix Legendre order consistency

**Priority**: Low | **Effort**: Small

`Mixture.SigS` stores P0, P1, P2 Legendre orders but most solvers only
use P0.  The DO 2D solver uses up to PL.  No validation that the
requested order is available in the Mixture.


## IMPL — Implemented and Tested, Sphinx Documentation Pending

### DA-20260405-007 — GENDF (GXS) → HDF5 direct conversion pipeline

**Status**: IMPL
**Session**: 2026-04-05

Implemented a complete pipeline to convert IAEA GENDF files (`.GXS`)
directly to HDF5 (`.h5`), bypassing the MATLAB CSV intermediary:

1. **`gendf.py`** — fixed-width 80-column GENDF parser:
   - `_parse_gendf()`: reads the full GXS file into a numeric matrix
     (n_lines, 10) matching what MATLAB `importdata('file.CSV', ';')` produces
   - `_parse_gendf_field()`: handles compact Fortran float notation where
     'E' is omitted (`1.001000+3` → `1.001E+3`)
   - `_extract_mf3()`: extracts MF=3 cross sections (capture MT=102,
     fission MT=18, nubar MT=452, (n,alpha) MT=107, total MT=1)
   - `_extract_mf6()`: extracts MF=6 transfer matrices (elastic MT=2,
     inelastic MT=51-91, thermal MT=221/222, (n,2n) MT=16, chi MT=18)
   - `convert_gxs()`: high-level function building Isotope objects from GXS

2. **`hdf5_io.py`** — HDF5 serialization:
   - Dense arrays (sigC, sigF, sigT, etc.) gzip-compressed
   - Sparse scattering matrices stored as COO triplets (row/col/data)
   - One `.h5` per element, temperature groups inside

3. **`convert_gxs_to_hdf5.py`** — batch conversion script

4. **`__init__.py`** — auto-prefers HDF5 over legacy `.m` files

All 12 elements (54 temperature points) converted and validated:
- Aqueous reactor: k_inf = 1.03596 (exact MATLAB match)
- PWR-like mixture: k_inf = 1.01357 (exact MATLAB match)

**Sphinx needed**: theory page documenting the GENDF format, record
structure (MF/MT numbers), scattering assembly logic (thermal group
zeroing, inelastic accumulation), and the sigT consistency issue.

### DA-20260405-008 — sigT consistency between .m files and GENDF components

**Status**: IMPL
**Session**: 2026-04-05

**Discovery**: The `.m` files' stored `sigT` has a systematic offset of
10–30 barns (constant across all groups and sigma-zeros) compared to
`sigC + sigF + sigL + sigS_rowsum + sig2_rowsum` computed from the
same `.m` file's stored components.  For example, U-238 at 600K:

- `.m` file `sigT[0,0]` = 108.14
- Recomputed from `.m` components = 77.87
- GENDF MF=3 MT=1 total = 77.89

This means the MATLAB `convertCSVtoM.m` computed `sigT` from
full-precision intermediates, then independently truncated all
components to `%13.6e` for writing.  The `.m` file's `sigT` is the
once-truncated full-precision sum, while recomputing from the
already-truncated components gives a different value.

**Impact on sigma-zero iterations**: The sigma-zero solver interpolates
`sigT` as a function of sigma-zero.  Using the GXS-computed `sigT`
(77.87) instead of the `.m` file's `sigT` (108.14) gives slightly
different sigma-zeros, leading to different interpolated XS, and
ultimately a 0.4% shift in PWR k_inf (1.01778 vs 1.01357).

**Current approach**: The `gendf.py` computes `sigT` from components.
Exact MATLAB reproduction requires the `.m` file sigT values, which
can be loaded as an override when `.m` files are available.

### DA-20260405-009 — (n,3n) and (n,4n) reactions not included

**Status**: IMPL (intentionally excluded, matching MATLAB)
**Session**: 2026-04-05

The GENDF files for heavy isotopes (U-235, U-238) contain MF=6 entries
for MT=17 ((n,3n)) and MT=37 ((n,4n)) that are not extracted.  This
matches the original MATLAB `convertCSVtoM.m` which only processes
MT=51-91 for inelastic scattering.  The impact is negligible for
thermal reactor applications (these reactions have high thresholds,
~6-15 MeV) but would matter for fast reactor or fusion blanket analyses.

**Code location**: `gendf.py:282` — the inelastic loop covers MT=51-91 only.
