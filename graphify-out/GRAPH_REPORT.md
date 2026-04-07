# Graph Report - docs/theory  (2026-04-06)

## Corpus Check
- Corpus is ~40,943 words - fits in a single context window. You may not need a graph.

## Summary
- 93 nodes · 117 edges · 8 communities detected
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Discrete Ordinates Method (SN)` - 15 edges
2. `Collision Probability Method` - 12 edges
3. `Method of Characteristics (MOC)` - 11 edges
4. `Thermal Hydraulics — Single-Channel LOCA` - 10 edges
5. `Cross-Section Data Pipeline (GENDF to HDF5)` - 10 edges
6. `Fuel Behaviour — 1D Radial Thermo-Mechanics` - 9 edges
7. `Monte Carlo Neutron Transport` - 9 edges
8. `Reactor Kinetics — 0D Point Kinetics + TH` - 9 edges
9. `Homogeneous Infinite-Medium Reactor` - 9 edges
10. `Verification Suite` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Multi-Group Neutron Balance` --semantically_similar_to--> `Collision Probability Method`  [INFERRED] [semantically similar]
  docs/theory/homogeneous.rst → docs/theory/collision_probability.rst
- `Gap Closure Event Detection (Reactor Kinetics)` --semantically_similar_to--> `Gap Closure Event Detection (Fuel Behaviour)`  [INFERRED] [semantically similar]
  docs/theory/reactor_kinetics.rst → docs/theory/fuel_behaviour.rst
- `Tabuchi-Yamamoto Polar Quadrature` --semantically_similar_to--> `Gauss-Legendre 1D Quadrature`  [INFERRED] [semantically similar]
  docs/theory/method_of_characteristics.rst → docs/theory/discrete_ordinates.rst
- `Multi-Group Neutron Balance` --semantically_similar_to--> `Discrete Ordinates Method (SN)`  [INFERRED] [semantically similar]
  docs/theory/homogeneous.rst → docs/theory/discrete_ordinates.rst
- `Flat-Source Approximation (MOC)` --semantically_similar_to--> `Flat-Source Approximation (CP)`  [INFERRED] [semantically similar]
  docs/theory/method_of_characteristics.rst → docs/theory/collision_probability.rst

## Hyperedges (group relationships)
- **Shared Two-Layer Mesh Pattern Across Deterministic Solvers** — cp_cpmesh, moc_mocmesh, sn_snmesh, mc_mcmesh, mesh1d_base_geometry, eigenvalue_solver_protocol [EXTRACTED 0.97]
- **Fuel-Clad Thermal-Mechanical Coupling (Gap, Stress, Gas Pressure)** — fb_gap_closure_event, fb_algebraic_stress_solver, fb_internal_gas_pressure, th_gap_conductance, th_algebraic_stress_solver, rk_gap_closure_event [EXTRACTED 0.93]
- **Single Source of Truth: Derivations -> Tests + Docs** — vv_derivations_single_source, vv_xs_library, vv_analytical_reference, vv_semi_analytical_reference, vv_richardson_extrapolation [EXTRACTED 0.95]

## Communities

### Community 0 - "Fuel Thermo-Mechanics"
Cohesion: 0.16
Nodes (17): Algebraic Stress Solver (_solve_stress), Prandtl-Reuss Creep Flow Rule, DAE-to-ODE Restructuring Design Decision, Displacement-Based Gap Constraint BC4 (FB-20260401-002), Fuel Behaviour — 1D Radial Thermo-Mechanics, Fuel Swelling Rate (MATPRO), Gap Closure Event Detection (Fuel Behaviour), Internal Gas Pressure Model (Fuel Behaviour) (+9 more)

### Community 1 - "Collision Probability"
Cohesion: 0.15
Nodes (17): Collision Probability Method, Flat-Source Approximation (CP), Infinite-Lattice CP Matrix P_inf, Collision Probability Matrix P_ij, CP Reciprocity Relation, White Boundary Condition (CP), EigenvalueSolver Protocol, ERR-017: Wrong Pitch Formula (24% Error + NaN) (+9 more)

### Community 2 - "Homogeneous & Eigenvalue"
Cohesion: 0.13
Nodes (17): Boltzmann Transport Equation, Homogeneous Infinite-Medium Reactor, Multi-Group Neutron Balance, One-Group k-infinity = nuSigF/SigA, Power Iteration Algorithm (Homogeneous), Removal Matrix A and Fission Matrix F, Scattering Matrix Convention (from-row, to-column), Sigma-Zero Self-Shielding Iteration (+9 more)

### Community 3 - "SN Transport"
Cohesion: 0.2
Nodes (11): Alpha Coefficients — Angular Redistribution Dome, Diamond Difference Discretization, Discrete Ordinates Method (SN), Gauss-Legendre 1D Quadrature, Lebedev Sphere Quadrature, Level-Symmetric SN Quadrature, Morel-Montry Closure Weights, Product Quadrature (GL x Equispaced Azimuthal) (+3 more)

### Community 4 - "Reactor Kinetics"
Cohesion: 0.22
Nodes (9): Reactivity Bias Locking (Steady State to Transient), Event Detection via Chunked Integration (BDF), Coolant Temperature Reactivity Feedback, Doppler Reactivity Feedback, Gap Closure Event Detection (Reactor Kinetics), Point Kinetics Equations (Power + Precursors), Reactor Kinetics — 0D Point Kinetics + TH, Reactivity Insertion Accident (RIA) Scenario (+1 more)

### Community 5 - "Monte Carlo"
Cohesion: 0.29
Nodes (8): Analog Absorption with Fission Weight Adjustment, Woodcock Delta-Tracking, Monte Carlo Neutron Transport, NeutronBank Population Management, Particle/Neutron Dataclass Hierarchy, Power Iteration MC (Inactive + Active Cycles), Precomputed XS Cache (_PrecomputedXS), Russian Roulette and Splitting (Population Control)

### Community 6 - "Verification & Kernels"
Cohesion: 0.25
Nodes (8): Ki3/Ki4 Bickley-Naylor Kernel (Cylinder), E3 Exponential-Integral Kernel (Slab), Analytical Reference Cases, Derivations Package as Single Source of Truth, Richardson Extrapolation Reference, Semi-Analytical Reference (E3/Ki4 Special Functions), Verification Suite, Synthetic Cross-Section Library (_xs_library.py)

### Community 7 - "Geometry & Meshing"
Cohesion: 0.33
Nodes (6): CPMesh Augmented Geometry, Rationale: Single Geometry Class vs Three Separate Solvers, MCMesh Augmented Geometry, Mesh1D Base Geometry, MOCMesh Augmented Geometry, SNMesh Augmented Geometry

## Knowledge Gaps
- **44 isolated node(s):** `Collision Probability Matrix P_ij`, `CP Reciprocity Relation`, `Infinite-Lattice CP Matrix P_inf`, `Rationale: Single Geometry Class vs Three Separate Solvers`, `Fuel Swelling Rate (MATPRO)` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Discrete Ordinates Method (SN)` connect `SN Transport` to `Collision Probability`, `Homogeneous & Eigenvalue`, `Geometry & Meshing`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `Homogeneous Infinite-Medium Reactor` connect `Homogeneous & Eigenvalue` to `Collision Probability`?**
  _High betweenness centrality (0.179) - this node is a cross-community bridge._
- **Why does `Collision Probability Method` connect `Collision Probability` to `Homogeneous & Eigenvalue`, `Verification & Kernels`, `Geometry & Meshing`?**
  _High betweenness centrality (0.166) - this node is a cross-community bridge._
- **What connects `Collision Probability Matrix P_ij`, `CP Reciprocity Relation`, `Infinite-Lattice CP Matrix P_inf` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Homogeneous & Eigenvalue` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._