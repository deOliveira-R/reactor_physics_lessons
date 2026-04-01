# Open-source Reactor Physics Education System

A Python reimplementation of the MATLAB-based educational system originally
developed by Konstantin Mikityuk (Paul Scherrer Institute, 2015-2021).

Provides practical tasks covering the main steps of nuclear reactor analysis:
cross section processing, neutron transport, diffusion, fuel behaviour,
thermal hydraulics, and reactor kinetics.

## Getting Started

### Prerequisites

- Python 3.12+ (3.14 recommended)
- git-lfs (for nuclear data files)

### Installation

```bash
git clone <repo-url>
cd ReactorPhysics_MATLAB
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Convert nuclear data

The repository ships GENDF (.GXS) cross section files via git-lfs.
Convert them to HDF5 before running any calculations:

```bash
cd data/micro_xs
python convert_gxs_to_hdf5.py
```

This reads the 12 `.GXS` files and produces one `.h5` file per isotope
containing all available temperatures.

## Modules

Each numbered folder is a self-contained lecture module. Run from inside
the folder:

```bash
cd 01.Homogeneous.Reactors
python run_homogeneous.py
```

| Module | Description | Entry script |
|--------|-------------|-------------|
| **00.Macro.XS** | Macroscopic cross section recipes (UO2, Zircaloy, borated water) | `recipes.py` |
| **01.Homogeneous.Reactors** | Infinite medium eigenvalue problem (421 groups) | `run_homogeneous.py` |
| **02.Discrete.Ordinates** | 2D SN transport with Lebedev quadrature (110 ordinates) | `run_discrete_ordinates.py` |
| **03.Method.Of.Characteristics** | 2D MoC transport with 8 ray directions | `run_moc.py` |
| **04.Monte.Carlo** | Monte Carlo with Woodcock delta tracking | `run_monte_carlo.py` |
| **05.Diffusion.1D** | Two-group 1D axial diffusion for a PWR subassembly | `run_diffusion_1d.py` |
| **06.Fuel.Behaviour** | 1D radial thermo-mechanical fuel rod analysis (6 years) | `run_fuel_behaviour.py` |
| **07.Thermal.Hydraulics** | Coupled TH + fuel mechanics under LOCA conditions (600 s) | `run_thermal_hydraulics.py` |
| **08.Reactor.Kinetics.0D** | Point kinetics + TH + fuel mechanics under RIA conditions | `run_reactor_kinetics.py` |

## Shared Data

The `data/` package provides infrastructure shared across all modules:

- `data/micro_xs/` — 421-group microscopic cross sections (GENDF/HDF5), isotope data model
- `data/macro_xs/` — sigma-zero iteration, XS interpolation, mixture assembly, material recipes
- `data/materials/` — MATPRO correlations (UO2, Zircaloy, gap gases) and water/steam properties (pyXSteam)

## Nuclear Data

Cross sections are in the IAEA 421-group GENDF format, downloaded from:
https://www-nds.iaea.org/ads/adsgendf.html

Isotopes included: H-1, B-10, B-11, O-16, Na-23, U-235, U-238,
Zr-90, Zr-91, Zr-92, Zr-94, Zr-96.

## Validation

Modules 01-07 are benchmarked against the original MATLAB results:

| Module | Quantity | Python | MATLAB | Match |
|--------|----------|--------|--------|-------|
| 01 Homogeneous (aqueous) | k_inf | 1.03596 | 1.03596 | exact |
| 01 Homogeneous (PWR) | k_inf | 1.01357 | 1.01357 | exact |
| 02 Discrete Ordinates | keff | 1.04190 | 1.04188 | 2e-5 |
| 03 Method of Characteristics | keff | 1.04923 | 1.04923 | exact |
| 04 Monte Carlo | keff | 1.038 +/- 0.002 | 1.035 +/- 0.002 | stochastic |
| 05 1D Diffusion | keff | 1.022170 | 1.022173 | 3e-6 |
| 06 Fuel Behaviour (t=1d) | Fuel center T | 1017.80 C | 1017.86 C | 0.006 C |

## Attribution

Based on the MATLAB educational system by Konstantin Mikityuk (PSI).
Python port by Rodrigo de Oliveira.
