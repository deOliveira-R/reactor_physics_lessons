.. _theory-fuel-behaviour:

===============================================
Fuel Behaviour — 1-D Radial Thermo-Mechanics
===============================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying the fuel behaviour solver.**

- 1D radial steady-state: fuel pellet + gap + cladding
- Temperature: :math:`-\frac{1}{r}\frac{d}{dr}(r k \frac{dT}{dr}) = q'''` with temperature-dependent conductivity
- Gap conductance: radiation + gas conduction (MATPRO correlations)
- Cladding stress: thick-wall cylinder (Lamé equations) with internal pressure + thermal stress
- Material properties: ``data/materials/matpro.py`` (UO2, Zircaloy, gap gases)
- IAPWS viscosity fallback: NaN at very high temperatures fixed via fallback (ERR-010)


Overview
========

Module 06 simulates the **long-term thermo-mechanical behaviour** of a
single PWR fuel rod under steady-state irradiation (6 years at 200 W/cm).
The model couples:

- **Radial heat conduction** — cylindrical UO\ :sub:`2` pellet (30 nodes)
  and Zircaloy cladding (5 nodes) with temperature-dependent properties.
- **Thermo-elastic-plastic deformation** — fuel and cladding strains
  including thermal expansion, elastic, creep, and (fuel only) swelling
  and (clad only) plasticity.
- **Fuel-cladding gap** — helium-filled annular gap with gas-conduction
  conductance, evolving geometry, and gap closure detection.
- **Internal gas pressure** — ideal gas law with fission gas release
  (He, Kr, Xe).
- **Cladding stress** — radial equilibrium, strain compatibility, and
  axial force balance solved as an algebraic linear system.

The simulation discovers the gap closure event (~2.85 years for the
default parameters) and continues with closed-gap boundary conditions
to end-of-cycle (6 years).

The solver is implemented in :func:`solve_fuel_behaviour` in
``06.Fuel.Behaviour/fuel_behaviour.py``.


Key Design Decision: DAE-to-ODE Restructuring
==============================================

The MATLAB original uses ``ode15s`` with a diagonal mass matrix to solve
a mixed ODE/AE (differential-algebraic) system.  Stress components are
algebraic variables (zeros on the mass matrix diagonal).  ``scipy.integrate.solve_ivp``
does not support mass matrices.  The BDF time integration approach is
shared with the thermal hydraulics module; see :ref:`bdf-integration`.

**Approach adopted:**  Remove stresses from the state vector entirely.
At every RHS evaluation, ``_solve_stress()`` constructs and solves the
:math:`3(n_f + n_c)` linear system for stress unknowns.  The ODE state
vector contains only time-evolving quantities:

.. math::

   \mathbf{y} = \bigl[\,
     T_{\text{fuel}}^{(n_f)},\;
     T_{\text{clad}}^{(n_c - 1)},\;
     F,\;
     \dot{V}_S^{(n_f)},\;
     \varepsilon_C^{(3 \times n_f)},\;
     \varepsilon_C^{(3 \times n_c)},\;
     \bar\varepsilon_P^{(n_c)},\;
     \varepsilon_P^{(3 \times n_c)}
   \,\bigr]

Default mesh: :math:`n_f = 30` fuel radial nodes, :math:`n_c = 5` clad
radial nodes.  Total ODE variables: :math:`30 + 4 + 1 + 30 + 90 + 15 + 5 + 15 = 190`.
(The MATLAB DAE has ~280 variables including stresses.)

**Why this works:**  The stress equilibrium, strain compatibility, and
boundary conditions form a **linear** system in the stresses when the
non-elastic strains (thermal, creep, swelling, plastic) are given.  The
elastic strains depend linearly on stress through Hooke's law:

.. math::

   \varepsilon_i^E = \frac{\sigma_i - \nu(\sigma_j + \sigma_k)}{E}
   = C_s \, \sigma_i + C_c (\sigma_j + \sigma_k)

where :math:`C_s = 1/E` (self compliance) and :math:`C_c = -\nu/E` (cross
compliance).  The total strain :math:`\varepsilon_i = \varepsilon_i^T +
\varepsilon_i^E + \varepsilon_i^C + \varepsilon_i^S/3` (fuel) or
:math:`\varepsilon_i = \varepsilon_i^T + \varepsilon_i^E + \varepsilon_i^P +
\varepsilon_i^C` (clad) is linear in the stresses.  Therefore, the equilibrium
and compatibility equations are linear in stress, and the system can be solved
with ``np.linalg.solve`` — no iteration needed.

**Tradeoff:**  The algebraic approach uses the **undeformed (fabrication)
geometry** for the node positions in the stress system.  Since strains are
O(1%), the geometry error is O(0.01%), which is negligible.  The MATLAB DAE
solver has the luxury of iterating geometry and stress simultaneously.  A
small transient settling difference occurs immediately after gap closure
(see FB-20260401-004 in the improvement tracker).


Physics Equations
=================

Fuel Temperature
----------------

Identical to the thermal hydraulics formulation (§ :ref:`theory-thermal-hydraulics`,
fuel temperature section), but with the full active fuel height (:math:`h_0 = 3` m)
rather than an axial mesh.  The fuel operates at a constant linear heat generation
rate of 200 W/cm = 20 000 W/m.

.. math::
   :label: fb-fuel-heat

   m_j \, c_p(T_j) \frac{dT_j}{dt}
   = -(Q_{j+1} - Q_j) + \dot{q}'''_j V_{j,0}

.. vv-status: fb-fuel-heat documented

The heat flows :math:`Q_j` at the :math:`n_f + 1` radial interfaces are:

.. math::

   Q_0 &= 0 \qquad \text{(adiabatic centre)} \\
   Q_j &= -k_{\text{UO}_2}(T_{\text{mid}}, Bu, p) \;
     \frac{T_j - T_{j-1}}{\Delta r} \; A_{\text{mid},j}
     \qquad j = 1, \ldots, n_f - 1 \\
   Q_{n_f} &= h_{\text{gap}} \, (T_{f,n_f} - T_{c,1}) \, A_{\text{gap}}

The outer clad surface temperature is **fixed** at :math:`T_{\text{out}} = 600` K
(no coolant model in this module — the TH coupling is added in Modules 07–08).


Strain Decomposition
--------------------

For each strain component :math:`i \in \{r, \theta, z\}`:

**Fuel:**

.. math::
   :label: fb-fuel-strain

   \varepsilon_i^{\text{fuel}} = \varepsilon^T(T) + \varepsilon_i^E(\boldsymbol{\sigma})
     + \varepsilon_i^C + \frac{\dot{V}_S}{3}

.. vv-status: fb-fuel-strain documented

where :math:`\varepsilon^T = \text{thExp}(T)` is the isotropic thermal expansion,
:math:`\varepsilon_i^E` is the elastic strain (Hooke's law), :math:`\varepsilon_i^C`
is the directional creep strain, and :math:`\dot{V}_S/3` is the isotropic volumetric
swelling divided equally among the three directions.

**Cladding:**

.. math::
   :label: fb-clad-strain

   \varepsilon_i^{\text{clad}} = \varepsilon^T(T) + \varepsilon_i^E(\boldsymbol{\sigma})
     + \varepsilon_i^P + \varepsilon_i^C

.. vv-status: fb-clad-strain documented


Fuel Swelling
-------------

The volumetric fuel swelling rate has two contributions (MATPRO [MATPRO2003]_):

.. math::
   :label: fb-swelling

   \dot{V}_S = \underbrace{2.5 \times 10^{-29} \dot{F}}_{\text{solid FP}} \;+\;
     \underbrace{8.8 \times 10^{-56} \dot{F} \, (2800 - T)^{11.73}
     e^{-0.0162(2800 - T)} e^{-8 \times 10^{-27} F}}_{\text{gaseous (T < 2800\,K)}}

.. vv-status: fb-swelling documented

where :math:`\dot{F}` is the fission rate density (fissions/m³/s) and :math:`F` is
the cumulative fission density (fissions/m³).  The gaseous contribution is strongly
temperature-dependent and vanishes above 2800 K (restructuring releases fission gases).


Fuel and Clad Creep
-------------------

The effective creep rate follows simplified MATPRO correlations:

.. math::

   \dot{\varepsilon}_{\text{eff}}^{\text{fuel}} &= 5 \times 10^5 \, \sigma_{\text{VM}}
     \, \exp\!\left(-\frac{4 \times 10^5}{8.314 \, T}\right) \\
   \dot{\varepsilon}_{\text{eff}}^{\text{clad}} &= 10^5 \, \sigma_{\text{VM}}
     \, \exp\!\left(-\frac{2 \times 10^5}{8.314 \, T}\right)

The deviatoric components follow the **Prandtl–Reuss flow rule**:

.. math::

   \dot{\varepsilon}_i^C = \dot{\varepsilon}_{\text{eff}}
     \frac{\sigma_i - \sigma_{\text{mean}}}{\sigma_{\text{VM}}}

where :math:`\sigma_{\text{mean}} = (\sigma_r + \sigma_\theta + \sigma_z)/3`
and :math:`\sigma_{\text{VM}} = \sqrt{\frac{1}{2}[(\sigma_r - \sigma_\theta)^2 +
(\sigma_r - \sigma_z)^2 + (\sigma_\theta - \sigma_z)^2]}`.


Clad Plastic Strain
-------------------

Same Norton power-law as the thermal hydraulics module
(§ :ref:`theory-thermal-hydraulics`, equation :eq:`creep-rate`):

.. math::

   \dot{\bar\varepsilon}_P = 10^{-3} \left( \frac{\sigma_{\text{VM}} \times 10^6}
     {K(T) \, |\bar\varepsilon_P + 10^{-6}|^{n(T)}} \right)^{1/m(T)}

with Prandtl–Reuss deviatoric components.  The :math:`K(T), m(T), n(T)` are
temperature-dependent Zircaloy strength parameters from MATPRO.


Algebraic Stress Solver
=======================

For each RHS evaluation, the function ``_solve_stress()`` constructs and
solves the linear system :math:`\mathbf{A} \boldsymbol{\sigma} = \mathbf{b}`:

System Structure
----------------

:math:`3(n_f + n_c)` unknowns ordered as:

.. math::

   \boldsymbol{\sigma} = [\,
     \sigma_r^{f,1} \ldots \sigma_r^{f,n_f},\;
     \sigma_\theta^{f,1} \ldots \sigma_\theta^{f,n_f},\;
     \sigma_z^{f,1} \ldots \sigma_z^{f,n_f},\;
     \sigma_r^{c,1} \ldots \sigma_r^{c,n_c},\;
     \sigma_\theta^{c,1} \ldots \sigma_\theta^{c,n_c},\;
     \sigma_z^{c,1} \ldots \sigma_z^{c,n_c}
   \,]

Equations per region (fuel / clad):

1. **Radial stress equilibrium** (:math:`n - 1` equations per region):

   .. math::

      \frac{d\sigma_r}{dr} = \frac{\sigma_\theta - \sigma_r}{r}

   Discretised at node boundary midpoints:

   .. math::

      \frac{\sigma_{r,j+1} - \sigma_{r,j}}{\Delta r_j}
      = \frac{(\sigma_{\theta,j} + \sigma_{\theta,j+1})/2
        - (\sigma_{r,j} + \sigma_{r,j+1})/2}{r_{\text{mid},j}}

2. **Strain compatibility** (:math:`n - 1` equations per region):

   .. math::

      \frac{d\varepsilon_\theta}{dr} = \frac{\varepsilon_r - \varepsilon_\theta}{r}

   Since :math:`\varepsilon_\theta = C_s \sigma_\theta + C_c(\sigma_r + \sigma_z) +
   \varepsilon_\theta^{\text{NE}}` (where NE = non-elastic), this becomes a linear
   equation in the stresses with the non-elastic strains on the RHS.

3. **Axial strain uniformity** (:math:`n - 1` equations per region):

   .. math::

      \varepsilon_z(r_j) = \varepsilon_z(r_{j+1})

4. **Boundary conditions** (6 equations):

   - **BC1** — Inner fuel surface: :math:`\sigma_r(0) = -p_{\text{gas}}` (central hole)
     or :math:`\sigma_r(0) = \sigma_\theta(0)` (solid cylinder, no central hole)
   - **BC2** — Outer clad surface: :math:`\sigma_r(r_{\text{out}}) = -p_{\text{cool}}`
   - **BC3** — Fuel/clad gap (radial stress): open gap:
     :math:`\sigma_r^f(r_f) = -p_{\text{gas}}`; closed gap: radial stress continuity
     :math:`\sigma_r^c(r_{\text{in}}) = \sigma_r^f(r_f)`
   - **BC4** — Fuel/clad gap (hoop strain): open gap:
     :math:`\sigma_r^c(r_{\text{in}}) = -p_{\text{gas}}`; closed gap:
     displacement constraint (see below)
   - **BC5** — Axial force balance (fuel): open gap:
     :math:`\int \sigma_z^f \, r \, dr = 0`; closed gap:
     axial strain continuity across gap
   - **BC6** — Axial force balance (clad or fuel+clad):
     :math:`\int \sigma_z \, r \, dr = p_{\text{in}} r_{\text{in}}^2 - p_{\text{out}} r_{\text{out}}^2`


Closed-Gap BC4 — Displacement-Based Constraint
-----------------------------------------------

.. admonition:: FB-20260401-002

   The original MATLAB formulation for BC4 (closed gap) uses a differential
   form:

   .. math::

      \frac{\varepsilon_\theta^c(r_{\text{in}}) - \varepsilon_\theta^f(r_f) - \Delta\varepsilon_h}
        {\delta_{\text{gap}}}
      = \frac{(\varepsilon_r^f + \varepsilon_r^c)/2 - (\varepsilon_\theta^f + \varepsilon_\theta^c)/2}
        {r_{\text{gap}}}

   This requires dividing by the gap width :math:`\delta_{\text{gap}}`, which is
   problematic for the algebraic solver:

   - The MATLAB DAE solver iterates and converges to the deformed :math:`\delta`
   - The algebraic solver uses fabrication geometry, where :math:`\delta = 100\,\mu\text{m}`
     instead of the physical ~6 μm (roughness)
   - This causes a 17× error in the stress gradient (ERR-013)

   **Fix adopted:**  Replace with a **displacement-based gap constraint**:

   .. math::
      :label: fb-bc4-displacement

      r_{c,\text{in}}^0 \, (1 + \varepsilon_\theta^c(r_{\text{in}}))
      - r_{f,\text{out}}^0 \, (1 + \varepsilon_\theta^f(r_f))
      = \varepsilon_{\text{rough}}

   .. vv-status: fb-bc4-displacement documented

   where :math:`\varepsilon_{\text{rough}} = 6\,\mu\text{m}`.  This formulation:

   - Is **physically transparent**: the deformed gap width equals the roughness
   - Is **linear in the stresses** (through the Hooke's law dependence of
     :math:`\varepsilon_\theta` on :math:`\sigma`)
   - **Avoids** the :math:`1/\delta_{\text{gap}}` amplification that made the
     differential form ill-conditioned
   - **Eliminates** the dependency on :math:`\Delta\varepsilon_h` (the hoop strain
     offset frozen at the moment of closure)

   **Result:** Contact pressure 40.7 MPa vs MATLAB 39.8 MPa (2.2% match).


Gap Closure Event
=================

The gap between fuel and cladding closes when the deformed gap width
:math:`\delta = r_{c,\text{in}} - r_{f,\text{out}}` decreases to the
surface roughness :math:`\varepsilon_{\text{rough}} = 6\,\mu\text{m}`:

.. math::

   E(t, \mathbf{y}) = \delta(t) - \varepsilon_{\text{rough}}

When :math:`E` crosses zero (direction :math:`-1`, terminal),
``solve_ivp`` stops Phase 1 and hands control back to the driver
so that BC3/BC4/BC5 can switch to closed-gap form.

**Event function implementation.** Computing :math:`\delta(t)`
from scratch inside the event function would require re-running
the entire ``_solve_stress()`` linear system — a waste, because
the same stress solve just happened at the latest RHS call.
``_gap_closure_event()`` therefore uses a **one-shot cache** of
the deformed gap width, keyed by ``(t_last, y_last)`` and updated
at the tail of every ``_rhs()`` call:

.. code-block:: python

   # At the end of each RHS evaluation:
   self._cache_gap = (t, y.copy(), delta_deformed)

   def _gap_closure_event(self, t, y):
       t_last, y_last, delta_last = self._cache_gap
       if t == t_last and np.array_equal(y, y_last):
           return delta_last - self.eps_rough  # cache hit
       try:
           # Cache miss: recompute from scratch
           sigma = self._solve_stress(t, y, closed=False)
           delta = self._deformed_gap(y, sigma)
           return delta - self.eps_rough
       except np.linalg.LinAlgError:
           # Ill-conditioned right at closure — return
           # the cached value (safe fallback)
           return delta_last - self.eps_rough

The try/except fallback is a defensive guard: in extremely tight
pre-closure states the stress linear system can become
ill-conditioned. Falling back to the cached value is safe because
(a) the cache is at most one RHS call stale, and (b) ``solve_ivp``
uses sign changes to trigger the event, not absolute magnitude —
a slightly stale :math:`E` is still monotonic across the root.

The ``direction=-1`` flag ensures the event fires **only** on
gap closing, not opening. Terminal event means ``solve_ivp``
stops immediately when detected, rather than continuing the
integration.

After closure, the solver:

1. Records the strain jumps :math:`\Delta\varepsilon_h` and :math:`\Delta\varepsilon_z`
   at the fuel/clad contact interface
2. Switches to closed-gap BCs (BC3, BC4, BC5 change — see above)
3. Relaxes tolerances from ``rtol=1e-6`` to ``rtol=1e-4`` (matching MATLAB)
4. Continues integration from the closure state to end-of-cycle


Internal Gas Pressure
=====================

The fuel rod contains an initial charge of helium at 1 MPa (Module 06; 1.2 MPa
for Modules 07–08).  Fission gas (He, Kr, Xe) is generated proportionally to
fission density and released at a constant fraction (FGR = 6%):

.. math::

   \mu_{\text{gen},g} = F \cdot V_{\text{fuel}} / N_A \cdot f_g
   \qquad
   \mu_{\text{rel},g} = \mu_{\text{gen},g} \cdot \text{FGR}

where :math:`f_g = (0.01, 0.02, 0.23)` for (He, Kr, Xe) as fractions of
fission products.

The gas pressure is computed from the ideal gas law with the gas distributed
between the plenum and the gap/void volumes:

.. math::

   p_{\text{gas}} = \frac{(\mu_{\text{He},0} + \sum_g \mu_{\text{rel},g}) \, R}
     {\displaystyle \frac{V_{\text{plenum}}}{T_{\text{plenum}}}
       + \frac{V_{\text{void}}}{T_{f,1}}
       + \frac{V_{\text{gap}}}{T_{\text{gap}}}}

The gap gas conductivity is computed using the Prandtl mixing rule for
He–Kr–Xe mixtures (MATPRO [MATPRO2003]_):

.. math::

   k_{\text{mix}} = \sum_g \frac{k_g \, x_g}
     {\sum_h \psi(k_g, k_h, M_g, M_h) \, x_h}

where :math:`\psi(k_1, k_2, M_1, M_2) = (1 + \sqrt{\sqrt{M_1/M_2} \, k_1/k_2})^2
/ \sqrt{8(1 + M_1/M_2)}`, :math:`x_g` are mole fractions, and :math:`M_g` are
molecular weights (He=2, Kr=54, Xe=36).


Validation
==========

Comparison with MATLAB at t = 1 day (open gap, pre-swelling):

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Quantity
     - Python
     - MATLAB
     - Diff
   * - Fuel centre T (°C)
     - 1017.80
     - 1017.86
     - 0.06
   * - Fuel outer T (°C)
     - 543.06
     - 543.03
     - 0.03
   * - Clad inner T (°C)
     - 349.66
     - 349.51
     - 0.15
   * - Clad outer T (°C)
     - 327.00
     - 326.85
     - 0.15
   * - Gap width (μm)
     - 71.19
     - 71.29
     - 0.10
   * - Gas pressure (MPa)
     - 2.4855
     - 2.4846
     - 0.04%
   * - Fuel outer r (mm)
     - 4.15197
     - 4.15200
     - 0.03 μm

Comparison at t = 4.69 years (after gap closure):

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Quantity
     - Python
     - MATLAB
     - Diff
   * - Fuel centre T (°C)
     - 1142
     - 1124
     - 1.6%
   * - Contact pressure (MPa)
     - 40.7
     - 39.8
     - 2.2%
   * - Gas pressure (MPa)
     - 6.923
     - 6.930
     - 0.1%
   * - Gap width (μm)
     - 6.0
     - 5.8
     - 3%

The open-gap phase achieves excellent parity (< 0.1%).  The closed-gap
phase shows 1–2% drift due to the algebraic stress solver's operator-split
treatment of the stiff stress-creep feedback (FB-20260401-004).


Known Limitations
=================

1. **1-D radial only** — no axial mesh, no axial conduction or fuel
   relocation.

2. **Algebraic stress solver uses undeformed geometry** — errors are
   O(0.01%) per strain level, negligible for PWR fuel but may matter
   for high-burnup fuels with large swelling.

3. **Closed-gap transient settling** — the operator-split approach
   resolves the stress-creep feedback more abruptly than the MATLAB
   DAE (~0.2 yr vs ~2 yr to reach positive hoop stresses after closure).

4. **Fixed outer clad temperature** — no coolant model.  The TH coupling
   is added in Modules 07 and 08.

5. **Simplified creep** — single-term Arrhenius correlations.  No
   irradiation creep, no stress relaxation.


References
==========

.. [MATPRO2003] D.L. Hagrman et al., *MATPRO — A Library of Materials
   Properties for Light-Water-Reactor Accident Analysis*, NUREG/CR-6150,
   Idaho National Laboratory, 2003.
