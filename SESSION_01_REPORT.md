# Session 1 Report — MATLAB to Python Port

**Date:** 2026-03-31  
**Scope:** Microscopic/macroscopic cross sections, homogeneous infinite reactors, GXS→HDF5 pipeline

---

## What was accomplished

### 1. Project setup
- Python 3.14 venv (homebrew) with numpy, scipy, matplotlib, pyXSteam, h5py
- Clean package structure: `reactor_physics/` with `micro_xs/`, `macro_xs/`, `reactors/`

### 2. Microscopic cross section data pipeline

**Three data backends were built:**

| Backend | Source | Purpose |
|---------|--------|---------|
| `.m` parser | `01.Micro.XS.421g/micro_*.m` | Legacy fallback, regex-based |
| GENDF parser | `01.Micro.XS.421g/*.GXS` | Direct from IAEA GENDF files |
| HDF5 loader | `data/micro_xs/*.h5` | Production format, gzip-compressed |

The `load_isotope(name, temp_K)` function auto-selects HDF5 when available.

**Conversion pipeline:** `*.GXS` → `gendf.py` → `Isotope` dataclass → `hdf5_io.py` → `*.h5`

All 12 elements (H, O, B-10, B-11, Na, U-235, U-238, Zr-90/91/92/94/96) converted across 54 temperature points.

### 3. Macroscopic cross section computation
- **Sigma-zero solver** — iterative background XS calculation (port of `sigmaZeros.m`)
- **XS interpolation** — interpolation at converged sigma-zeros for all reaction types including sparse scattering matrices (port of `interpSigS.m`)
- **Mixture assembly** — `compute_macro_xs()` combines isotopes with number densities into macroscopic XS
- **Predefined recipes** — `aqueous_uranium()` and `pwr_like_mix()` matching MATLAB's `createH2OU` and `createPWR_like_mix`

### 4. Homogeneous infinite reactor solver
- Power iteration eigenvalue solver using `scipy.sparse.linalg.spsolve`
- Flux normalization and spectrum output (per-energy and per-lethargy)
- PDF plot generation

### 5. Validation results

| Case | Python k_inf | MATLAB k_inf | Match |
|------|-------------|-------------|-------|
| Aqueous reactor (H₂O + U-235) | 1.03596 | 1.03596 | **YES** |
| PWR-like mixture (UO₂ + Zry + H₂O+B) | 1.01357 | 1.01357 | **YES** |

---

## Key bugs found and fixed

1. **`np.interp` direction** — sigma-zero arrays are in decreasing order (1e10, 1e4, ..., 1); `np.interp` requires increasing `xp`. MATLAB's `interp1` handles both directions. Fixed by reversing arrays before interpolation.

2. **`repmat` in .m files** — B-10's `sigL` uses `s.sigL = repmat(s.sigL, nSig0, 1)` to replicate a single row. The parser initially missed this, producing zeros for rows 2+. Fixed by detecting the `repmat` pattern.

3. **sigT consistency** — The .m files' stored `sigT` has a systematic offset from what you'd recompute from the stored components (sigC + sigF + sigL + sigS_rowsum). This is because the MATLAB converter computed sigT from full-precision intermediates, then independently truncated all components to `%13.6e`. The HDF5 converter reads sigT from the .m files to maintain consistency with MATLAB's sigma-zero iterations.

---

## What remains to port

| # | Module | MATLAB folder | Complexity |
|---|--------|---------------|------------|
| 4 | Discrete Ordinates | `04.Discrete.Ordinates/` | Medium-high |
| 5 | Method of Characteristics | `05.Method.Of.Characteristics/` | High |
| 6 | Monte Carlo | `06.Monte.Carlo/` | Medium |
| 7 | Full Core 1D diffusion | `07.Full.Core.1D/` | Medium |
| 8 | Fuel Behaviour | `08.Fuel.Behaviour/` | Medium |
| 9 | Thermal Hydraulics | `09.Thermal.Hydraulics/` | Medium |
| 10 | Reactor Kinetics 0D | `10.Reactor.Kinetics.0D/` | Medium-high |

Modules 4-7 are neutronics solvers that consume the macro XS data already built.  
Modules 8-10 are thermo-mechanical/TH/kinetics that use `matpro.m` material properties and `XSteam`.

---

## How to run

```bash
cd python/
source .venv/bin/activate

# Convert GXS → HDF5 (only needed once)
python convert_gxs_to_hdf5.py

# Run homogeneous reactor calculations
python run_homogeneous.py
```
