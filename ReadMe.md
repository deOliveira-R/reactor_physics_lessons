# ORPHEUS — Open Reactor Physics Educational University System

Provides practical tasks covering the main steps of nuclear reactor analysis:
cross section processing, neutron transport, diffusion, fuel behaviour,
thermal hydraulics, and reactor kinetics.

## Getting Started

### Prerequisites

- Python 3.12+ (3.14 recommended)
- git-lfs (for nuclear data files)

### Installation

```bash
git clone git@github.com:deOliveira-R/reactor_physics_lessons.git
cd reactor_physics_lessons
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Convert nuclear data

The repository ships GENDF (.GXS) cross section files via git-lfs.
Convert them to HDF5 before running any calculations:

```bash
cd data/micro_xs
python convert_gxs_to_hdf5.py
```

### Run tests

```bash
pytest                  # non-slow tests (~90s, 56 tests)
pytest -m slow          # slow tests (~7min, 17 tests)
pytest -v               # all 73 tests
```

### Build documentation

```bash
python -m derivations.generate_rst
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
```

## Architecture

```
numerics/
    eigenvalue.py        EigenvalueSolver protocol + power_iteration()

data/
    micro_xs/            421-group microscopic XS (GENDF/HDF5), Isotope dataclass
    macro_xs/
        mixture.py       Mixture dataclass (SigT, SigS, absorption_xs, ...)
        cell_xs.py       CellXS dataclass + assemble_cell_xs()
        recipes.py       Material recipes (UO2, Zircaloy, borated water)
        sigma_zeros.py   Sigma-zero self-shielding iteration
        interpolation.py XS interpolation at converged background XS
    materials/           MATPRO correlations + water/steam properties (pyXSteam)

derivations/             SymPy analytical derivations (verification single source of truth)
    _xs_library.py       4 abstract regions × {1G,2G,4G} cross sections
    homogeneous.py       Infinite medium eigenvalues
    sn.py                SN transport equation derivation
    cp_slab.py           E₃ collision probability eigenvalues
    cp_cylinder.py       Ki₄ collision probability eigenvalues
    moc.py               Method of characteristics derivation
    mc.py                Monte Carlo random walk derivation
    diffusion.py         Diffusion buckling eigenvalue
    reference_values.py  Unified registry of all verification cases
    generate_rst.py      Generates RST fragments for Sphinx documentation

tests/                   pytest verification suite (73 tests)
    test_*_properties.py Unit tests (conservation, reciprocity, symmetry)
    test_*.py            Eigenvalue verification against analytical references

tools/
    plotting.py          Shared plotting utilities

01-08.*                  Lecture modules (one per physics domain)
09.Collision.Probability Collision probability solvers (slab + cylindrical)
```

All deterministic eigenvalue solvers satisfy the `EigenvalueSolver`
protocol defined in `numerics/eigenvalue.py` and share a generic
`power_iteration()` function.

## Modules

Each numbered folder is a self-contained lecture module with a demo script:

```bash
python 01.Homogeneous.Reactors/demo_homogeneous.py
```

| Module | Description | Solver | Demo script |
|--------|-------------|--------|-------------|
| **00.Demo** | Central Limit Theorem and spherical harmonics demos | — | `central_limit_theorem.py` |
| **01.Homogeneous.Reactors** | Infinite medium eigenvalue problem (421 groups) | `HomogeneousSolver` | `demo_homogeneous.py` |
| **02.Discrete.Ordinates** | Unified SN transport: 2D native (Lebedev), 1D degenerate (GL) | `SNSolver` | `demo_discrete_ordinates.py` |
| **03.Method.Of.Characteristics** | 2D MoC transport with 8 ray directions | `MoCSolver` | `demo_moc.py` |
| **04.Monte.Carlo** | Monte Carlo with Woodcock delta tracking | — | `demo_monte_carlo.py` |
| **05.Diffusion.1D** | Two-group 1D axial diffusion for a PWR subassembly | `DiffusionSolver` | `demo_diffusion_1d.py` |
| **06.Fuel.Behaviour** | 1D radial thermo-mechanical fuel rod analysis | — | `demo_fuel_behaviour.py` |
| **07.Thermal.Hydraulics** | Coupled TH + fuel mechanics under LOCA conditions | — | `demo_thermal_hydraulics.py` |
| **08.Reactor.Kinetics.0D** | Point kinetics + TH + fuel mechanics under RIA | — | `demo_reactor_kinetics.py` |
| **09.Collision.Probability** | CP method for slab and cylindrical geometries | `CPSolver` | `demo_cp_slab.py` |

### Discrete Ordinates

The unified SN solver (`02.Discrete.Ordinates/sn_solver.py`) supports:
- **1D slab** (ny=1, Gauss-Legendre quadrature) and **2D Cartesian** (Lebedev)
- Selectable inner solver: source iteration or BiCGSTAB
- Diamond-difference spatial discretization (wavefront sweep in 2D, cumprod in 1D)
- (n,2n) reactions and P_L scattering anisotropy

### Collision Probability

Module 09 provides two CP solvers sharing a single `CPSolver` class:
- **Slab** (`solve_cp_slab`) — 1D half-cell with E₃ exponential-integral kernel
- **Concentric** (`solve_cp_concentric`) — Wigner-Seitz cylindrical cell with Ki₃/Ki₄ Bickley-Naylor kernel

### Monte Carlo

Delta-tracking with `MCGeometry` protocol for pluggable geometry:
- `ConcentricPinCell` — concentric cylindrical regions
- `SlabPinCell` — 1D slab regions
- Future: constructive solid geometry (CSG)

## Verification

The verification suite uses SymPy-derived analytical references as the
single source of truth. Each solver has its own derivation from its own
equations — no cross-verification.

```bash
pytest tests/ -v    # run all verification tests
```

| Method | Geometry | Groups × Regions | Reference type |
|--------|----------|-------------------|---------------|
| Homogeneous | — | 1/2/4 × 1 | Analytical (matrix eigenvalue) |
| SN 1D | Slab | 1/2/4 × 1,2,4 | Analytical + Richardson O(h²) |
| CP Slab | Slab | 1/2/4 × 1,2,4 | Semi-analytical (E₃ eigenvalue) |
| CP Cylinder | Cyl1D | 1/2/4 × 1,2,4 | Semi-analytical (Ki₄ eigenvalue) |
| MOC | Cyl1D | 1/2/4 × 1,2,4 | Analytical + Richardson |
| MC | Cyl1D | 1/2/4 × 1,2,4 | Analytical + CP reference |
| Diffusion | Slab | 2 × 1,2 | Analytical (buckling) + Richardson |

Unit tests verify structural properties: CP conservation/reciprocity,
SN particle balance/flux symmetry, diffusion vacuum BCs.

## Nuclear Data

Cross sections are in the IAEA 421-group GENDF format, downloaded from:
https://www-nds.iaea.org/ads/adsgendf.html

Isotopes included: H-1, B-10, B-11, O-16, Na-23, U-235, U-238,
Zr-90, Zr-91, Zr-92, Zr-94, Zr-96.

## Attribution

Based on the MATLAB educational system by Konstantin Mikityuk (PSI).
Python port and augmentation by Rodrigo de Oliveira.
