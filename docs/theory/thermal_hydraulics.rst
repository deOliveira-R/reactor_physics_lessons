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

**Material properties** (MATPRO correlations [MATPRO2003]_ in ``matpro.py``):

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
   - Axial force balance: :math:`\int \sigma_z \, r \, dr = p_{\text{gas}} r_{\text{in}}^2 - p_{\text{cool}} r_{\text{out}}^2`

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
``try/except ValueError`` for graceful degradation if any residual NaN
issues arise.


IAPWS Viscosity Fallback
-------------------------

.. admonition:: TH-20260405-001

   **Problem:** During post-failure LOCA blowdown, coolant temperature in the
   outlet node exceeds 900 °C.  pyXSteam's viscosity function
   (``my_AllRegions_ph``) has an artificial cutoff:

   .. code-block:: matlab

      % XSteam.m line 3410
      if T > 900 + 273.15
          my_AllRegions_ph = NaN;
          return
      end

   This is a conservative validity bound, not a physical singularity.  The
   NaN viscosity propagates: ``nu = mu/rho`` → NaN → Churchill friction
   factor → NaN → ``_compute_pressure`` returns NaN pressures → solver crash.

   **Fix:** ``_iapws_viscosity(T_K, rho)`` in ``h2o_properties.py``
   implements the **same IAPWS 2008 correlation** without the cutoff.
   The formula has two parts:

   .. math::

      \mu = \mu_0(\bar{T}) \cdot \mu_1(\bar{T}, \bar{\rho}) \cdot 55.071 \times 10^{-6} \;\text{Pa·s}

   where :math:`\bar{T} = T / 647.226` and :math:`\bar{\rho} = \rho / 317.763`.

   **Dilute-gas contribution** (kinetic theory):

   .. math::

      \mu_0 = \frac{\bar{T}^{1/2}}{1 + 0.978197/\bar{T}
        + 0.579829/\bar{T}^2 - 0.202354/\bar{T}^3}

   **Finite-density correction:**

   .. math::

      \mu_1 = \exp\!\left(\bar{\rho} \sum_{i=0}^{6}\sum_{j=0}^{5}
        H_{ij}\,(1/\bar{T} - 1)^j\,(\bar{\rho} - 1)^i\right)

   The :math:`H_{ij}` coefficient matrix (7 × 6, mostly sparse) is from
   the IAPWS release [IAPWS2008]_.  At low pressures (< 1 MPa) and high
   temperatures, :math:`\bar{\rho} \ll 1` so :math:`\mu_1 \approx 1` and the
   viscosity is dominated by the dilute-gas term.

   **Validation:**

   .. list-table::
      :header-rows: 1
      :widths: 15 20 20 20

      * - T (°C)
        - pyXSteam μ (Pa·s)
        - IAPWS μ (Pa·s)
        - Ratio
      * - 500
        - 2.858e-5
        - 2.858e-5
        - 1.000000
      * - 700
        - 3.656e-5
        - 3.656e-5
        - 1.000000
      * - 899
        - 4.405e-5
        - 4.405e-5
        - 1.000000
      * - 900
        - 4.409e-5
        - 4.409e-5
        - 1.000000
      * - 950
        - NaN
        - 4.590e-5
        - — (fallback)
      * - 1000
        - NaN
        - 4.767e-5
        - — (fallback)
      * - 1200
        - NaN
        - 5.450e-5
        - — (fallback)

   Bit-identical below 900 °C; smooth, physically reasonable above.  The
   function could replace pyXSteam's viscosity entirely (DA-20260405-005).


Validation
==========

The MATLAB reference is in ``matlab_archive/09.Thermal.Hydraulics/results.m``.

.. _th-matlab-gap-bug:

MATLAB Gap Geometry Bug
-----------------------

.. admonition:: TH-20260405-002

   MATLAB's ``funRHS.m`` line 272 contains a bug:

   .. code-block:: matlab

      gap.r_ = (clad.r(1) + fuel.dz)/2;

   This adds a cladding inner radius (``clad.r(1)`` ≈ 4.22 mm) to the fuel
   axial node height (``fuel.dz`` ≈ 1.5 m), producing ``gap.r_`` ≈ 0.752 m
   instead of the correct mid-gap radius ≈ 4.17 mm.  The intended code was
   likely ``(clad.r(:,1) + fuel.r)/2``.

   The gap heat transfer area is then:

   .. math::

      A_{\text{gap}}^{\text{MATLAB}} = 2\pi \times 0.752 \times 1.5
        \approx 7.09 \;\text{m}^2

   versus the correct value:

   .. math::

      A_{\text{gap}}^{\text{correct}} = 2\pi \times 4.17{\times}10^{-3} \times 1.5
        \approx 0.039 \;\text{m}^2

   A factor of **180×** difference.  This massively enhances gap heat transfer,
   keeping the fuel 332 °C cooler at steady state (808 °C vs 1140 °C).

   **Consequence for validation:** Direct comparison of Python and MATLAB is
   only meaningful using ``thermal_hydraulics_dae.py`` which replicates the
   MATLAB gap geometry.  The "correct physics" version (``thermal_hydraulics.py``)
   has no MATLAB reference — it represents the physically correct solution.


Two Versions
------------

Two solver files are maintained for comparison:

.. list-table::
   :header-rows: 1
   :widths: 25 38 37

   * - Property
     - ``thermal_hydraulics.py``
     - ``thermal_hydraulics_dae.py``
   * - Gap geometry
     - Correct: :math:`r_{\text{mid}} = (r_{c,\text{in}} + r_f)/2`
     - MATLAB match: :math:`r_{\text{mid}} = (r_{c,\text{in}} + \Delta z_f)/2`
   * - Pressure
     - Algebraic (computed in RHS)
     - State variable with relaxation
   * - Fuel centre (SS)
     - 1140 °C
     - 808 °C (matches MATLAB)
   * - Clad failure
     - 287 s
     - 379 s (MATLAB: 425 s)
   * - Runs to t = 600 s
     - Yes
     - Yes


MATLAB Parity (DAE Version)
----------------------------

Comparison of ``thermal_hydraulics_dae.py`` against
``matlab_archive/09.Thermal.Hydraulics/results.m`` at key time points.
All values for outlet node (axial node 2):

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18 18

   * - Time (s)
     - Python h (kJ/kg)
     - MATLAB h (kJ/kg)
     - Python T_cool (°C)
     - MATLAB T_cool (°C)
     - Python fuel_T (°C)
   * - 1
     - 1299
     - 1300
     - 292.8
     - 292.9
     - 426.6
   * - 200
     - 1443
     - 1443
     - 318.4
     - 318.4
     - 808.2
   * - 210
     - 2357
     - 2357
     - 283.0
     - 283.0
     - 368.1
   * - 250
     - 3356
     - 3358
     - 438.9
     - 440.0
     - 462.4
   * - 287
     - 3528
     - 3527
     - 519.5
     - 518.9
     - 523.0

Pre-LOCA steady state and early blowdown match to within 1–2 kJ/kg in
coolant enthalpy and 1–2 °C in temperature.


Correct-Physics Steady-State Check
------------------------------------

The correct-physics version (``thermal_hydraulics.py``) has no MATLAB
reference, but the steady-state fuel centre temperature can be checked
against an analytical estimate.  For a solid cylindrical pellet with
uniform volumetric heating:

.. math::

   T_{\text{centre}} - T_{\text{surface}} = \frac{\dot{q}'''}{4\,k}
     = \frac{\text{LHGR}}{4\pi\,k}

With LHGR = 69 141 / 3.0 = 23 047 W/m and :math:`k_{\text{UO}_2}` ≈ 3.0 W/m-K
(average over the 300–1100 °C range), this gives:

.. math::

   \Delta T_{\text{fuel}} \approx \frac{23047}{4\pi \times 3.0} \approx 612\;°\text{C}

Adding the gap + clad + coolant resistance (~320 °C above coolant at 300 °C):

.. math::

   T_{\text{centre}} \approx 300 + 320 + 612 \approx 1232\;°\text{C}

The Python solver gives 1140 °C, which is lower than this simple estimate
because the actual :math:`k(T)` increases at lower temperatures (the
conductivity-averaged ΔT is smaller than the constant-k estimate).  The
agreement within ~8% confirms the correct-physics version is physically
reasonable (TH-20260405-006).

The clad failure timing difference (Python 379 s vs MATLAB 425.3 s) is
traced to MATLAB's additional indexing quirk in the gap temperature
calculation: ``gap.T = (fuel.T(fuel.nr) + clad.T(1))/2`` with
``fuel.nr = 20`` returns element 20 of the column-major flattened
``(2, 20)`` array — i.e., axial node 2, radial node 10 — not the fuel
surface temperature at each axial level.  This produces a different inner
gas pressure (MATLAB ~2.9 MPa vs Python ~4.5 MPa at t = 400 s), which
changes the engineering stress and delays failure.


Investigation History
=====================

The NaN at t ≈ 395 s during post-failure LOCA integration was investigated
in three phases over sessions 2026-04-01 and 2026-04-05.

Phase 1 — Initial Diagnosis (2026-04-01)
-----------------------------------------

- Identified that post-failure Phase 2 integration crashed with ``ValueError``
- Traced to pyXSteam returning NaN for water/steam properties
- **Hypothesis:** BDF Jacobian numerical differencing perturbs the state
  vector, pushing enthalpy/pressure outside pyXSteam's valid range
- **Mitigation:** Chunked integration with ``try/except`` for graceful stop
- **Result:** Simulation runs to t = 395 s, stops cleanly

Phase 2 — Root Cause Chain (2026-04-05)
-----------------------------------------

Systematic investigation revealed the full causal chain:

1. **Added diagnostics** to the RHS function to print state values at
   NaN occurrence:

   - Coolant enthalpy at failure: h = [3172, 4399] kJ/kg — **valid** for
     pyXSteam at this pressure
   - Pressure at failure: ``cool_p = [NaN, NaN]`` — the NaN comes from
     ``_compute_pressure``, not from pyXSteam directly

2. **Traced NaN through pressure computation:** First-pass water properties
   showed ``nu = [1.93e-5, NaN]``.  Node 2 kinematic viscosity was NaN.
   Density and temperature were valid.

3. **Identified pyXSteam viscosity cutoff:**

   .. code-block:: python

      p_bar, h_kJ = 3.31, 4399.0
      T = st.t_ph(p_bar, h_kJ)    # 900.7 °C — works fine
      rho = st.rho_ph(p_bar, h_kJ) # 0.611 kg/m³ — works fine
      mu = st.my_ph(p_bar, h_kJ)   # NaN — 900 °C cutoff!

   The cutoff is exactly 900 °C.  All other properties (T, ρ, k, c_p) work
   to well above 1000 °C.

4. **Why Python reaches 900 °C but MATLAB doesn't:** Compared fuel centre
   temperatures:

   - **Python:** 1140 °C at steady state (correct physics)
   - **MATLAB:** 808 °C at steady state

   The 332 °C difference comes from the gap geometry bug (see
   :ref:`th-matlab-gap-bug`).  The hotter fuel stores more energy; during
   LOCA blowdown this energy heats the coolant past 900 °C.

Phase 3 — Approaches Tried and Outcome
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 5 30 20 45

   * - #
     - Approach
     - Result
     - Why
   * - 1
     - Reduce ``max_step`` to 1e-3 (match MATLAB)
     - Failed
     - Coolant genuinely reaches >900 °C; smaller steps don't prevent it
   * - 2
     - DAE with pressure as state variable + relaxation
     - NaN persisted
     - Pressure relaxation doesn't prevent viscosity NaN
   * - 3
     - Replicate MATLAB gap geometry
     - NaN eliminated
     - Fuel stays at 808 °C; coolant never reaches 900 °C
   * - 4
     - IAPWS viscosity fallback
     - **NaN eliminated**
     - Addresses root cause; both versions run to 600 s

Approach 4 was adopted as the primary fix because it preserves correct
physics without replicating the MATLAB bug.  Approach 3 is retained
in ``thermal_hydraulics_dae.py`` for parity comparison.


Known Limitations
=================

1. **Clad failure time resolution:** 1 s (chunk size).  Could be refined
   via re-integration bisection but not implemented (TH-20260401-006).

2. **No radiation heat transfer** across the gap (gas conduction only).

3. **Axial mesh coarse:** Only :math:`n_z = 2` nodes; no axial conduction.

4. **DAE clad failure timing:** ``thermal_hydraulics_dae.py`` fails at
   379 s vs MATLAB's 425.3 s.  The remaining 46 s gap is due to MATLAB
   indexing quirks in gap temperature and inner gas pressure (not worth
   replicating further).


References
==========

.. [MATPRO2003] D.L. Hagrman et al., *MATPRO — A Library of Materials
   Properties for Light-Water-Reactor Accident Analysis*, NUREG/CR-6150,
   Idaho National Laboratory, 2003.

.. [IAPWS2008] IAPWS, *Release on the IAPWS Formulation 2008 for the
   Viscosity of Ordinary Water Substance*, 2008.  The correlation
   implemented in ``_iapws_viscosity`` and in XSteam/pyXSteam.
