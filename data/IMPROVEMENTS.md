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
