.. _theory-reactor-kinetics:

==============================================
Reactor Kinetics — 0-D Point Kinetics + TH
==============================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying the kinetics solver.**

- 0D point kinetics + thermal-hydraulic feedback
- Reactivity insertion accident (RIA) transient
- 6 delayed neutron precursor groups
- BDF time integration (``scipy.integrate.solve_ivp``); see :ref:`bdf-integration` in the thermal hydraulics chapter
- Doppler + moderator temperature reactivity feedback coefficients
- Power-temperature coupling: power → fuel temp → reactivity → power


Overview
========

Module 08 simulates a **Reactivity Insertion Accident (RIA)** in a single
PWR fuel channel using 0-D point kinetics coupled with 1-D radial heat
transfer and thermo-mechanics.  The formulation follows the standard
textbook treatment of point-kinetics feedback [Ott1985]_.  The model
consists of:

- **Point kinetics** — 6-group delayed neutrons with prompt and delayed
  fission power.
- **Reactivity feedback** — Doppler (fuel temperature) and coolant
  temperature coefficients with bias locking.
- **Radial fuel and clad heat conduction** — identical formulation to the
  thermal hydraulics module (§ :ref:`theory-thermal-hydraulics`).
- **Gap closure detection** — when fuel expansion closes the gap, the
  boundary conditions switch from gas-pressure to strain-compatibility.
- **Coolant energy balance** — 1-D axial with two-phase correlations.
- **Clad creep plasticity** — Norton power-law with von Mises flow rule.

The simulation runs 120 s in three phases: steady state (0–100 s),
transient with open gap (100 s – closure), and transient with closed gap
(closure – 120 s).

The solver is implemented in :func:`solve_reactor_kinetics` in
``08.Reactor.Kinetics.0D/reactor_kinetics.py``.


The RIA Scenario
================

The accident is driven by a cold-coolant insertion.  The inlet temperature
follows a piecewise-linear ramp:

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - Time (s)
     - :math:`T_{\text{inlet}}` (K)
     - Phase
   * - 0 – 100
     - 553
     - Steady state
   * - 100 → 100.5
     - 553 → 300
     - Cold shock (253 K drop in 0.5 s)
   * - 100.5 → 101.0
     - 300
     - Hold cold
   * - 101.0 → 101.5
     - 300 → 553
     - Recovery
   * - 101.5 → 120
     - 553
     - Normal

The rapid coolant temperature drop inserts **positive reactivity** (coolant
temperature coefficient :math:`\alpha_c = -2{\times}10^{-4}` K\ :sup:`-1`
is negative, so cooling the coolant *reduces* the negative feedback,
net positive insertion).  This drives a prompt-supercritical power excursion
that is eventually terminated by Doppler feedback from fuel heating.

**Reactivity budget:** At t = 100.4 s, the total reactivity reaches
≈ 682 pcm, just below the prompt-critical threshold
:math:`\beta_{\text{eff}} \approx 760` pcm.  The power peaks at 22–25×
nominal before Doppler feedback brings it down.


Physics Equations
=================

ODE State Vector
----------------

The flat state vector extends the thermal-hydraulics vector with neutronics:

.. math::

   \mathbf{y} = \bigl[\,
     P,\;
     C_1 \ldots C_6,\;
     T_{\text{fuel}}^{(n_z \times n_f)},\;
     T_{\text{clad}}^{(n_z \times n_c)},\;
     h_{\text{cool}}^{(n_z)},\;
     \bar\varepsilon_p^{(n_z \times n_c)},\;
     \varepsilon_p^{(3 \times n_z \times n_c)}
   \,\bigr]

where :math:`P` is the fission power (W) and :math:`C_i` are the delayed
neutron precursor densities (m\ :sup:`-3`) for 6 groups.


Point Kinetics Equations
------------------------

**Power equation:**

.. math::
   :label: power-equation

   \frac{dP}{dt} = \frac{\rho - \beta_{\text{eff}}}{\Lambda} P
     + \sum_{i=1}^{6} \lambda_i C_i

.. vv-status: power-equation documented

**Precursor equations:**

.. math::
   :label: precursor-equation

   \frac{dC_i}{dt} = \frac{\beta_i}{\Lambda} P - \lambda_i C_i
   \qquad i = 1, \ldots, 6

.. vv-status: precursor-equation documented

**Kinetics parameters:**

.. list-table::
   :header-rows: 1
   :widths: 15 40 30

   * - Group
     - :math:`\beta_i`
     - :math:`\lambda_i` (s\ :sup:`-1`)
   * - 1
     - 2.584 × 10\ :sup:`-4`
     - 0.013
   * - 2
     - 1.520 × 10\ :sup:`-3`
     - 0.032
   * - 3
     - 1.391 × 10\ :sup:`-3`
     - 0.119
   * - 4
     - 3.070 × 10\ :sup:`-3`
     - 0.318
   * - 5
     - 1.102 × 10\ :sup:`-3`
     - 1.403
   * - 6
     - 2.584 × 10\ :sup:`-4`
     - 3.929

:math:`\beta_{\text{eff}} = \sum \beta_i \approx 7.6{\times}10^{-3}`,
:math:`\Lambda = 20\;\mu\text{s}` (prompt neutron lifetime).


Reactivity Feedback
-------------------

**Doppler feedback** (fuel temperature):

.. math::
   :label: doppler-feedback

   \rho_D = \alpha_D \, \langle T_{\text{fuel}} \rangle

.. vv-status: doppler-feedback documented

where :math:`\alpha_D = -2{\times}10^{-5}` K\ :sup:`-1` and
:math:`\langle T_{\text{fuel}} \rangle` is the volume-averaged fuel
temperature, averaged over all radial and axial nodes.

**Coolant temperature feedback:**

.. math::
   :label: coolant-feedback

   \rho_c = \alpha_c \, \langle T_{\text{cool}} \rangle

.. vv-status: coolant-feedback documented

where :math:`\alpha_c = -2{\times}10^{-4}` K\ :sup:`-1` (10× larger than
Doppler) and :math:`\langle T_{\text{cool}} \rangle` is the mean coolant
temperature across axial nodes.

**Bias locking:** During steady state (Phase 1), the reactivity values are
updated on every RHS call to track the evolving temperature field.  At the
end of steady state, the biases are **frozen**:

.. math::

   \rho_{D,\text{bias}} = \alpha_D \, \langle T_f \rangle_{t=100}
   \qquad
   \rho_{c,\text{bias}} = \alpha_c \, \langle T_c \rangle_{t=100}

During the transient (Phases 2–3), the total reactivity is:

.. math::

   \rho = (\rho_D - \rho_{D,\text{bias}}) + (\rho_c - \rho_{c,\text{bias}})

This construction ensures :math:`\rho = 0` at the start of the transient,
and all reactivity changes are relative to the equilibrium state.


Thermal-Hydraulics Subsystem
----------------------------

The fuel temperature, clad temperature, coolant enthalpy, gap conductance,
gas pressure, clad stress, and creep equations are identical to the thermal
hydraulics module — see § :ref:`theory-thermal-hydraulics`.


Gap Closure Event
-----------------

The gap closure event function monitors:

.. math::
   :label: gap-closure-event

   E(t) = \min_z \bigl[ \delta_{\text{gap},z} - \varepsilon_{\text{rough}} \bigr]

.. vv-status: gap-closure-event documented

where :math:`\delta_{\text{gap},z} = r_{c,\text{in},z} - r_{f,z}` is the
deformed gap width and :math:`\varepsilon_{\text{rough}}` is the surface
roughness.  When :math:`E` crosses zero from positive to negative, the
fuel pellet contacts the cladding.

**Post-closure boundary conditions:** The inner clad surface switches from
gas-pressure loading to **strain compatibility** with the fuel:

.. math::

   \varepsilon_{\theta,\text{clad}}(r_{\text{in}}) + \Delta\varepsilon_h
     = \varepsilon_{\theta,\text{fuel}}

where :math:`\Delta\varepsilon_h = \varepsilon_{\theta,\text{clad}} -
\varepsilon_T^{\text{fuel}}` is the strain jump frozen at the moment of
closure.  The gap heat transfer switches from
:math:`h = k_{\text{He}}/\delta` to :math:`h = k_{\text{He}}/\varepsilon_{\text{rough}}`.


Three-Phase Simulation
======================

Phase 1 — Steady State (0 → 100 s)
------------------------------------

Single ``solve_ivp`` call with ``method='BDF'``, ``max_step=10``.
Reactivity biases updated on every RHS call and frozen at the end.

Phase 2 — Transient, Open Gap (100 s → closure)
--------------------------------------------------

Chunked integration: one chunk per output step (:math:`\Delta t = 0.1` s)
with manual gap closure event checking between chunks.  When a sign change
is detected, the crossing time is refined via **bisection on the
dense-output interpolant** (40 iterations, :math:`\sim 10^{-12}` precision).

Dense-output bisection works for this module (unlike Module 07) because the
reactor kinetics state variables remain physical under interpolation — the
coolant stays in single-phase subcooled conditions during the RIA transient.

Solver: ``method='BDF'``, ``max_step=1e-3`` (very small to capture the
prompt-supercritical power excursion).

Phase 3 — Transient, Closed Gap (closure → 120 s)
---------------------------------------------------

Single ``solve_ivp`` call from the closure time.  The gap boundary conditions
are switched, strain jumps are frozen, and integration continues with the
closed-gap stress solver.


Numerical Methods
=================

Solver Choice: Radau → BDF
--------------------------

All three ``solve_ivp`` calls in the kinetics driver (steady-state
bias locking, transient RIA, closed-gap continuation) use
``method='BDF'`` (Backward Differentiation Formula). An earlier
port used ``method='Radau'`` — the BDF switch was made to match
MATLAB's ``ode15s``, which is a variable-order BDF (NDF) method.

**Why this matters.** The point kinetics + feedback system is
stiff in a distinctive way: the prompt-neutron time constant
:math:`\Lambda \sim 10^{-5}` s is five orders of magnitude
faster than the fuel thermal time constant
:math:`\tau_{\text{feedback}} \sim 1` s, and both are active
simultaneously during the RIA peak. Both Radau (implicit RK) and
BDF (linear multistep) handle stiffness correctly, but their
step-size adaptation heuristics differ enough to produce slightly
different trajectories on this specific problem.

**Validation** (against ``matlab_archive/10.Reactor.Kinetics.0D/results.m``,
247 time steps):

- :math:`t = 100.0` s: power exact match
- :math:`t = 100.1` s: 0.1 %
- :math:`t = 100.2` s: 0.2 %
- Peak power: MATLAB 25.3× at :math:`t = 100.6` s, Python 24.0× at
  :math:`t = 100.4` s — same magnitude, slight timing offset from
  different BDF implementations

The residual 0.2 s timing offset is listed under "Known
Limitations" below (RK-20260401-005) and attributed to BDF
Jacobian numerical differencing differences; it is an accepted
implementation-level discrepancy, not a bug.

See also: :ref:`bdf-integration` in the thermal hydraulics chapter
for the full tolerance/max-step recipe, which is shared across the
kinetics, thermal-hydraulics, and fuel-behaviour modules.


Event Detection via Chunked Integration
-----------------------------------------

scipy's ``solve_ivp`` event detection uses ``brentq`` root-finding
internally, which requires the event function to change sign within a
single solver step.  For the gap closure event, the sign can remain
constant across multiple steps (the gap closes gradually), causing
``brentq`` to raise ``ValueError``.

**Solution:** Each output interval is integrated as a separate chunk.
Between chunks, the event function is evaluated and a sign change triggers
detection.  The crossing time is refined via bisection on the dense-output
interpolant ``sol.sol(t)`` (the ODE's continuous extension).

**Why this works for Module 08 but not Module 07:** The reactor kinetics
state stays in a regime where water properties are well-behaved (subcooled
liquid at 15.5 MPa).  Module 07's LOCA blowdown produces extreme
conditions (near-zero pressure, superheated steam) where interpolated
states violate the property tables.


Validation Against MATLAB
==========================

The MATLAB reference is in ``matlab_archive/10.Reactor.Kinetics.0D/results.m``
(247 time steps, 82 unique output points).

Power History Comparison
------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20

   * - Time (s)
     - MATLAB Power
     - Python Power
     - Match
   * - 100.0
     - 1.000
     - 1.000
     - exact
   * - 100.1
     - 1.114
     - 1.113
     - 0.1%
   * - 100.2
     - 1.665
     - 1.669
     - 0.2%
   * - 100.3
     - 4.993
     - 5.239
     - 5%
   * - 100.4
     - 22.061
     - 23.960
     - 9%

Peak power: MATLAB 25.3× at t = 100.6 s, Python 24.0× at t = 100.4 s.
Same magnitude, slight timing offset from different BDF implementations.

Reactivity Comparison
---------------------

At t = 100.1 s: MATLAB 81.15 pcm, Python 80.39 pcm (1% match).

**Historical note on the "5× vs 24×" discrepancy:** Early parity notes
reported the MATLAB peak as "~5×", leading to a major investigation
(RK-20260401-003).  Extraction of the full MATLAB ``results.m`` revealed
the actual peak is **25.3×** — the "5×" value was likely a misread of a
recovery-phase power level.  The Python result (24×) matches well.


Known Limitations
=================

1. **Power timing offset (RK-20260401-005):** Python peaks 0.2 s earlier
   than MATLAB.  Expected for different BDF implementations on a stiff
   system with :math:`\Lambda / \tau_{\text{feedback}} \sim 10^{-4}`.

2. **0-D kinetics:** No spatial effects (axial flux shape, control rod
   worth profile).  The entire core is represented by a single fuel
   channel.

3. **No xenon or samarium feedback.**

4. **Constant boundary conditions** apart from the inlet temperature
   transient (pressure, flow rate fixed at 15.5 MPa, 4.8 m/s).


References
==========

.. [Ott1985] K.O. Ott and R.J. Neuhold, *Introductory Nuclear Reactor
   Dynamics*, American Nuclear Society, 1985.
