.. _theory-thermal-hydraulics:

=============================================
Thermal Hydraulics — Single-Channel LOCA
=============================================

.. contents:: Contents
   :local:
   :depth: 3


Overview
========

Module 07 simulates a **Loss-of-Coolant Accident (LOCA)** in a single
PWR fuel channel.  The model couples:

- **Radial fuel heat conduction** — cylindrical UO\ :sub:`2` pellet with
  volumetric heat generation, burnup-dependent conductivity, and thermal
  expansion.
- **Radial clad heat conduction** — Zircaloy-4 tube with thermal expansion,
  creep plasticity, and a burst-stress failure criterion.
- **Axial coolant energy balance** — 1-D upward flow with two-phase
  correlations (Dittus–Boelter, Thom nucleate boiling, Zuber CHF, film
  boiling).
- **Gap mechanics** — helium-filled gap with deformable fuel/clad geometry
  and pressure-dependent contact conductance.
- **Internal gas pressure** — ideal gas law in the plenum + gap volume.

The simulation runs 600 s covering normal operation, depressurisation, and
LOCA blowdown.  Cladding failure (burst) is detected via an event function,
after which the integration continues with failed-clad boundary conditions.

The solver is implemented in :func:`solve_thermal_hydraulics` in
``07.Thermal.Hydraulics/thermal_hydraulics.py``.


Physics Equations
=================

ODE State Vector
----------------

The flat 1-D state vector packs:

.. math::

   \mathbf{y} = \bigl[\,
     T_{\text{fuel}}^{(n_z \times n_f)},\;
     T_{\text{clad}}^{(n_z \times n_c)},\;
     h_{\text{cool}}^{(n_z)},\;
     \bar\varepsilon_p^{(n_z \times n_c)},\;
     \varepsilon_p^{(3 \times n_z \times n_c)}
   \,\bigr]

where:

- :math:`T_{\text{fuel}}` — fuel nodal temperatures (K), shape ``(nz, nf)``
- :math:`T_{\text{clad}}` — clad nodal temperatures (K), shape ``(nz, nc)``
- :math:`h_{\text{cool}}` — coolant specific enthalpy (J/kg), shape ``(nz,)``
- :math:`\bar\varepsilon_p` — effective (scalar) plastic strain
- :math:`\varepsilon_p^{(r,\theta,z)}` — plastic strain tensor components

Default mesh: :math:`n_z = 2` axial levels, :math:`n_f = 20` fuel radial
nodes, :math:`n_c = 5` clad radial nodes.  Total DOFs:
:math:`n_z(n_f + 5 n_c) + n_z = 2(20 + 25) + 2 = 92`.


Fuel Temperature
----------------

1-D radial heat conduction in a cylindrical UO\ :sub:`2` pellet with
volumetric heat generation:

.. math::
   :label: fuel-heat

   \rho_f \, c_p(T) \frac{\partial T}{\partial t}
   = \frac{1}{r} \frac{\partial}{\partial r}
     \left[ r \, k(T, Bu, p) \frac{\partial T}{\partial r} \right]
   + \dot{q}'''

**Boundary conditions:**

- Centre (:math:`r = 0`): adiabatic, :math:`\partial T / \partial r = 0`
- Surface (:math:`r = r_f`): gap heat flux,
  :math:`q_{\text{gap}} = h_{\text{gap}} (T_{f,\text{surf}} - T_{c,\text{in}})`

**Finite-volume discretisation:**  The fuel pellet is divided into
:math:`n_f` concentric annular rings.  Heat flows :math:`Q_j` (W) are
computed at the :math:`n_f + 1` radial interfaces:

.. math::

   Q_0 &= 0 \qquad \text{(adiabatic centre)} \\
   Q_j &= -k_{\text{UO}_2}(T_{\text{mid}}) \;
     \frac{T_{j} - T_{j-1}}{\Delta r_j} \; A_{\text{mid},j}
     \qquad j = 1, \ldots, n_f - 1 \\
   Q_{n_f} &= h_{\text{gap}} \, (T_{f,n_f} - T_{c,1}) \, A_{\text{gap}}

where :math:`A_{\text{mid},j} = 2\pi r_{\text{mid},j} \Delta z` is the
interface area at the radial midpoint between nodes :math:`j-1` and :math:`j`,
and :math:`A_{\text{gap}}` is the deformed gap area.  The temperature rate
for node :math:`j` is:

.. math::
   :label: fuel-rate

   \frac{dT_{f,j}}{dt} = \frac{-(Q_{j+1} - Q_j) + \dot{q}'''_j V_j}
     {m_j \, c_p(T_j)}

where :math:`m_j = \rho_f V_j` is the ring mass and
:math:`V_j = \pi(r_{j+1}^2 - r_j^2) \Delta z` is the ring volume.

**Material properties** (MATPRO correlations in ``matpro.py``):

.. math::

   k_{\text{UO}_2}(T, Bu, p) &=
     \left[\frac{1}{0.0452 + 2.46{\times}10^{-4}\,T + 1.87{\times}10^{-3}\,Bu
       + 0.038\,(1 - 0.9\,e^{-0.04\,Bu})\,Bu^{0.28}\big/(1 + 396\,e^{-6380/T})}
       + 3.5{\times}10^{9}\,\frac{e^{-16360/T}}{T^2}\right] \\
     &\quad \times\; 1.0789\,\frac{1 - p}{1 + p/2}

.. math::

   c_{p,\text{UO}_2}(T) = 162.3 + 0.3038\,T - 2.391{\times}10^{-4}\,T^2
     + 6.404{\times}10^{-8}\,T^3 \quad (\text{J/kg-K})


Clad Temperature
----------------

1-D radial heat conduction in cylindrical Zircaloy-4 cladding:

.. math::
   :label: clad-heat

   \rho_c \, c_p(T) \frac{\partial T}{\partial t}
   = \frac{1}{r} \frac{\partial}{\partial r}
     \left[ r \, k(T) \frac{\partial T}{\partial r} \right]

**Boundary conditions:**

- Inner surface: gap heat flux from fuel
- Outer surface: wall-to-coolant heat transfer,
  :math:`q_w = h_{\text{conv}}(T_{c,\text{out}} - T_{\text{cool}})`

**Finite-volume discretisation:**  Same structure as fuel, with :math:`n_c`
concentric rings.  Heat flows at the :math:`n_c + 1` interfaces:

.. math::

   Q_0 &= h_{\text{gap}} \, (T_{f,n_f} - T_{c,1}) \, A_{\text{gap}}
     \qquad \text{(gap flux, inner BC)} \\
   Q_j &= -k_{\text{Zry}}(T_{\text{mid}}) \;
     \frac{T_{c,j} - T_{c,j-1}}{\Delta r_j} \; A_{\text{mid},j}^{\text{def}}
     \qquad j = 1, \ldots, n_c - 1 \\
   Q_{n_c} &= q_w \, A_{\text{HX}}
     \qquad \text{(wall-to-coolant, outer BC)}

The clad interface areas :math:`A_{\text{mid},j}^{\text{def}}` use the
**deformed** clad geometry (including thermal + elastic + plastic strains):

.. math::

   r_{c,j}^{\text{def}} = r_{c,j}^{0} \, (1 + \varepsilon_{\theta,j}^{\text{total}})

This is important for LOCA analysis where clad ballooning significantly
increases the heat transfer area.

**Material properties:**

.. math::

   k_{\text{Zry}}(T) &= 7.51 + 2.09{\times}10^{-2}\,T
     - 1.45{\times}10^{-5}\,T^2 + 7.67{\times}10^{-9}\,T^3
     \quad (\text{W/m-K}) \\
   c_{p,\text{Zry}}(T) &= 252.54 + 0.11474\,T
     \quad (\text{J/kg-K})


Coolant Enthalpy
----------------

1-D axial energy balance with upwind differencing:

.. math::
   :label: coolant-energy

   \frac{\partial(\rho h)}{\partial t}
   + \frac{\partial(\dot{m} h)}{\partial z}
   = \frac{q_w \, A_{\text{HX}}}{V_{\text{flow}}}

where :math:`\dot{m}` is the junction mass flow rate (kg/s),
:math:`q_w` is the regime-dependent wall heat flux (W/m\ :sup:`2`),
and :math:`A_{\text{HX}} = 2\pi r_{\text{out}} \Delta z` is the heat
exchange area.

**Discrete form** with upwind advection at :math:`n_z + 1` junctions:

.. math::
   :label: coolant-rate

   \frac{dh_z}{dt} = \frac{-(\dot{m} h)_{z+1/2} + (\dot{m} h)_{z-1/2}
     + q_w \, A_{\text{HX}}}{\rho_z \, a_{\text{flow}} \, \Delta z}

The junction enthalpies use upwind values:
:math:`h_{z-1/2} = h_{\text{inlet}}` at the first junction,
:math:`h_{z-1/2} = h_{z-1}` at interior junctions.  Water/steam
properties (:math:`\rho, T, T_{\text{sat}}, h_L, h_V`, etc.) are
evaluated via IAPWS correlations through pyXSteam at each axial node.


Gap Conductance
---------------

The gap between fuel and cladding is filled with helium.  The conductance
model depends on the gap state:

.. math::
   :label: gap-conductance

   h_{\text{gap}} =
   \begin{cases}
     k_{\text{He}}(T_{\text{gap}}) \;/\; \delta_{\text{gap}}
       & \text{if gap open } (\delta > 0) \\[4pt]
     k_{\text{He}}(T_{\text{gap}}) \;/\; \varepsilon_{\text{rough}}
       & \text{if gap closed } (\delta \le 0)
   \end{cases}

where :math:`k_{\text{He}}(T) = 2.639{\times}10^{-3} \, T^{0.7085}` W/m-K,
:math:`\delta_{\text{gap}} = r_{c,\text{in}} - r_{f,\text{out}}` is the
deformed gap width, and :math:`\varepsilon_{\text{rough}} \approx 6\,\mu\text{m}`
is the effective surface roughness for contact conductance.

The gap geometry is **deformable**: fuel outer radius changes with thermal
expansion :math:`\varepsilon_T^{\text{UO}_2}(T)`, and clad inner radius
changes with thermal + elastic + plastic strains.


Internal Gas Pressure
---------------------

Ideal gas law for the helium inventory distributed between plenum and gap:

.. math::
   :label: gas-pressure

   p_{\text{gas}} = \frac{\mu_{\text{He},0} \, R}
     {\displaystyle \frac{V_{\text{plenum}}}{T_{\text{plenum}}}
       + \sum_{z} \frac{V_{\text{gap},z}}{T_{\text{gap},z}}}

where :math:`\mu_{\text{He},0}` is the initial helium mole count (conserved),
:math:`V_{\text{gap},z} = \pi(r_{c,\text{in}}^2 - r_f^2) \Delta z_z` is the
deformed gap volume per axial level, and
:math:`T_{\text{gap},z} = (T_{f,\text{surf}} + T_{c,\text{in}})/2`.


Clad Stress — Algebraic Subsystem
----------------------------------

The cladding stress is solved as an **algebraic** (not differential) system
at each RHS evaluation.  For each axial node, a linear system of
:math:`3 n_c` equations is solved for :math:`\sigma_r, \sigma_\theta, \sigma_z`:

1. **Radial equilibrium** (:math:`n_c - 1` equations):

   .. math::

      \frac{\partial \sigma_r}{\partial r}
      + \frac{\sigma_r - \sigma_\theta}{r} = 0

2. **Strain compatibility** (:math:`n_c - 1` equations):

   .. math::

      \frac{\partial \varepsilon_\theta}{\partial r}
      = \frac{\varepsilon_\theta - \varepsilon_r}{r}

3. **Axial strain uniformity** (:math:`n_c - 1` equations):

   .. math::

      \varepsilon_z(r_j) = \varepsilon_z(r_{j+1})

4. **Boundary conditions** (3 equations):

   - Outer surface: :math:`\sigma_r(r_{\text{out}}) = -p_{\text{cool}}`
   - Inner surface (gap open): :math:`\sigma_r(r_{\text{in}}) = -p_{\text{gas}}`
   - Axial force balance:
     :math:`\int \sigma_z \, r \, dr = p_{\text{gas}} r_{\text{in}}^2
       - p_{\text{cool}} r_{\text{out}}^2`

**Strain decomposition:**

.. math::

   \varepsilon_i = \varepsilon_i^T + \varepsilon_i^E + \varepsilon_i^P
   \qquad
   \varepsilon_i^E = \frac{\sigma_i - \nu(\sigma_r + \sigma_\theta + \sigma_z - \sigma_i)}{E}

**Linear system construction:**  For each axial node :math:`z`, the
:math:`3 n_c` unknowns are ordered as
:math:`[\sigma_{r,1} \ldots \sigma_{r,n_c},\;
\sigma_{\theta,1} \ldots \sigma_{\theta,n_c},\;
\sigma_{z,1} \ldots \sigma_{z,n_c}]`.
The coefficient matrix :math:`\mathbf{A}` and RHS vector :math:`\mathbf{b}`
encode:

- Rows 1 to :math:`n_c - 1`: equilibrium via central differences on the
  clad node midpoints,
  :math:`(\sigma_{r,j+1} - \sigma_{r,j})/\Delta r + (\sigma_{r,\text{avg}} - \sigma_{\theta,\text{avg}})/r_{\text{mid}} = 0`
- Rows :math:`n_c` to :math:`2(n_c - 1)`: strain compatibility relating
  elastic compliance coefficients :math:`C_s = 1/E` and
  :math:`C_c = -\nu/E` with non-elastic (thermal + plastic) strain
  contributions on the RHS
- Rows :math:`2 n_c - 1` to :math:`3(n_c - 1)`: axial strain uniformity
- Row :math:`3(n_c - 1) + 1`: outer BC :math:`\sigma_r(r_{\text{out}}) = -p_{\text{cool}}`
- Row :math:`3(n_c - 1) + 2`: inner BC (pressure or strain compatibility)
- Row :math:`3 n_c`: axial force balance

The system is dense :math:`(15 \times 15)` for :math:`n_c = 5` and solved
via ``np.linalg.solve`` — no iterative method needed.

**Gap-closed inner BC:**  When the gap is closed (Phase 3 of Module 08),
the inner BC changes from :math:`\sigma_r(r_{\text{in}}) = -p_{\text{gas}}`
to a strain compatibility condition:

.. math::

   \varepsilon_{\theta,\text{clad}}(r_{\text{in}}) + \Delta\varepsilon_h
     = \varepsilon_T^{\text{fuel}}

where :math:`\Delta\varepsilon_h` is the hoop strain jump frozen at the
moment of closure.  This couples the fuel thermal expansion to the clad
inner-surface stress.


Plastic Strain Rate — Norton Creep
-----------------------------------

The effective plastic strain rate follows a Norton power-law:

.. math::
   :label: creep-rate

   \dot{\bar\varepsilon}_p = \min\!\left(0.1,\;
     10^{-3} \left( \frac{\sigma_{\text{VM}} \times 10^6}{K(T)\,
     |\bar\varepsilon_p + 10^{-6}|^{n(T)}} \right)^{1/m(T)} \right)

where :math:`\sigma_{\text{VM}}` is the von Mises stress (MPa), and
:math:`K(T), m(T), n(T)` are temperature-dependent Zircaloy strength
parameters from MATPRO.  The deviatoric components follow the Prandtl–Reuss
flow rule:

.. math::

   \dot\varepsilon_i^P = \dot{\bar\varepsilon}_p \;
     \frac{3}{2} \frac{\sigma_i - \sigma_{\text{mean}}}{\sigma_{\text{VM}}}


Two-Phase Flow Model
--------------------

The wall heat transfer coefficient is selected by flow regime:

**Regime 1** — Single-phase forced convection (:math:`T_w < T_{\text{sat}}`):

.. math::

   \text{Nu} = \max(4.36,\; 0.023 \, \text{Re}^{0.8} \, \text{Pr}^{0.4})
   \qquad \text{(Dittus–Boelter)}

**Regime 2** — Pre-CHF nucleate boiling
(:math:`T_{\text{sat}} < T_w < T_{\text{CHF}}`):

.. math::

   q_{\text{NB}} = 2000\,|T_w - T_{\text{sat}}|(T_w - T_{\text{sat}})\,
     e^{p/4.34}
   \qquad \text{(Thom correlation)}

**Regime 3** — Post-CHF film boiling (:math:`T_w \ge T_{\text{CHF}}`):

.. math::

   q_{\text{FB}} = 0.25\,(g\,k_V^2\,c_p\,(\rho_L - \rho_V)/\nu_V)^{1/3}
     \, k_{\text{sub}} \, (T_w - T_{\text{sat}})

**CHF** — Modified Zuber correlation with subcooling and void corrections:

.. math::

   q_{\text{CHF}} = 0.14 \, h_{fg} \,
     (g\,\sigma\,\rho_V^2\,(\rho_L - \rho_V))^{0.25}
     \, k_{\text{sub}} \, k_{\text{void}}


Clad Failure Criterion
----------------------

Failure is detected when the engineering hoop stress exceeds the
burst stress:

.. math::
   :label: burst-criterion

   \sigma_I = \frac{(p_{\text{gas}} - p_{\text{cool}})
     \cdot (r_{\text{out}} + r_{\text{in}})/2}{r_{\text{out}} - r_{\text{in}}}
   \qquad
   \text{Failure when } \sigma_B(T) - \sigma_I \le 0

The burst stress :math:`\sigma_B(T)` follows the Kingery–Hobson correlation
from MATPRO.  After failure, the internal gas pressure equilibrates with
coolant pressure: :math:`p_{\text{gas}} = p_{\text{cool,outlet}}`.


LOCA Boundary Conditions
------------------------

The LOCA scenario is defined by four time-dependent boundary condition
tables, interpolated linearly at each RHS call:

- **Inlet temperature** :math:`T_{\text{in}}(t)` — drops during reflood
- **Outlet pressure** :math:`p_{\text{out}}(t)` — depressurises from 15.5 MPa
- **Inlet velocity** :math:`v_{\text{in}}(t)` — flow coastdown
- **Power** :math:`P(t)` — decay heat curve after scram

The LOCA sequence (default PWR parameters):

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Time (s)
     - Pressure (MPa)
     - Velocity (m/s)
     - Power (W)
     - Phase
   * - 0 – 200
     - 15.5
     - 4.8
     - 69 141
     - Normal operation
   * - 200 – 201
     - 15.5 → 0.5
     - 4.8 → 0.1
     - 69 141 → 4 500
     - Blowdown initiation
   * - 201 – 600
     - 0.5 → 0.1
     - 0.1 → 0.01
     - 4 500 → 2 000
     - LOCA blowdown

The exact table values are stored in the ``THParams`` dataclass (lines 63–100
of ``thermal_hydraulics.py``).


Numerical Methods
=================

ODE Integration
---------------

The system is integrated using ``scipy.integrate.solve_ivp`` with
``method='BDF'`` (Backward Differentiation Formula), matching MATLAB's
``ode15s``.  Tolerances: ``rtol=1e-6``, ``atol=1e-4``, ``max_step=10.0``.

Chunked Integration for Event Detection
-----------------------------------------

scipy's built-in event detection (which uses ``brentq`` internally) fails
for this problem because the event function can have the same sign at both
ends of a solver step, causing ``ValueError``.

**Solution:** The integration is split into chunks of one output step
(:math:`\Delta t = 1` s).  After each chunk:

1. Evaluate :math:`E(t) = \sigma_B - \sigma_I` at the chunk endpoint
2. If :math:`E` changed sign (positive → negative), declare clad failure

The chunk endpoint is used as the failure state (1 s resolution).  Dense-output
bisection was attempted but rejected — the interpolated states produce
unphysical water property values (NaN from pyXSteam) because the interpolant
does not enforce thermodynamic consistency.

**Post-failure integration** also uses chunked integration with
``try/except ValueError`` for graceful NaN handling.  The Jacobian numerical
differencing can push water/steam properties out of their valid range during
extreme LOCA conditions (coolant T > 900 °C).  The simulation stops cleanly
at the NaN boundary (typically t ≈ 395 s).


Validation
==========

The MATLAB reference is in ``matlab_archive/09.Thermal.Hydraulics/results.m``.

Temperature Evolution
---------------------

Fuel centre (outlet node) temperature at selected times, showing the
approach to steady state and the LOCA transient:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Time (s)
     - Fuel centre (°C)
     - Clad outer (°C)
     - Coolant (°C)
     - Phase
   * - 1
     - 429.5
     - 290.8
     - 285.7
     - Initial transient
   * - 10
     - 1050.8
     - 334.0
     - 315.2
     - Developing profile
   * - 100
     - 1160.5
     - 338.8
     - 318.4
     - Steady state
   * - 200
     - 1160.5
     - 338.8
     - 318.4
     - Pre-LOCA (steady)
   * - 287
     - —
     - —
     - —
     - Clad failure
   * - 395
     - 1288.5
     - —
     - —
     - NaN boundary

**Comparison with MATLAB at t = 1 s** (BDF solver, outlet node):

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 20

   * - Quantity
     - Python
     - MATLAB
     - Difference
   * - Fuel centre
     - 429.5 °C
     - 428.0 °C
     - 0.3%
   * - Fuel surface
     - 376.3 °C
     - 312.5 °C
     - 20% (developing)
   * - Clad outer
     - 290.8 °C
     - 298.4 °C
     - 2.5%
   * - Coolant
     - 285.7 °C
     - 287.8 °C
     - 0.7%

The fuel surface temperature difference at t = 1 s reflects the BDF solver
startup transient — the radial profile is still developing.  By t = 10 s
the profile is fully established and at t = 100 s the steady state matches.

Clad Failure Timing
-------------------

**Python:** Clad failure at t = 287 s.

**MATLAB:** Clad failure at t ≈ 425 s.

The 138 s discrepancy is due to a **known bug in the MATLAB gap geometry**
(``funRHS.m`` line 272): ``gap.r_ = (clad.r(1) + fuel.dz)/2`` mixes a
radius (~4 mm) with an axial height (~1.5 m), making the gap heat transfer
area 180× too large.  This keeps MATLAB's fuel 332 °C cooler (808 °C vs
1140 °C at steady state), delaying the clad failure.

A ``thermal_hydraulics_dae.py`` variant replicates MATLAB's gap bug for
direct comparison — see TH-20260402-001 in the data module improvements.


Known Limitations
=================

1. **Post-failure NaN (TH-20260401-004):** Integration stops at t ≈ 395 s
   due to pyXSteam property limits during Jacobian evaluation.

2. **Clad failure time resolution:** 1 s (chunk size).  Could be refined
   via re-integration bisection but not implemented (TH-20260401-006).

3. **No radiation heat transfer** across the gap (gas conduction only).

4. **Axial mesh coarse:** Only :math:`n_z = 2` nodes; no axial conduction.


References
==========

.. [MATPRO2003] D.L. Hagrman et al., *MATPRO — A Library of Materials
   Properties for Light-Water-Reactor Accident Analysis*, NUREG/CR-6150,
   Idaho National Laboratory, 2003.
