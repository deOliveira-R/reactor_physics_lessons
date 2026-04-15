Reactor Kinetics — 0-D Point Kinetics + TH
=============================================

The :mod:`orpheus.kinetics` package simulates a Reactivity
Insertion Accident (RIA) in a PWR-like subchannel by coupling
0-D point kinetics to a 1-D thermal-hydraulic channel and a
radial fuel / clad mechanical model. It is the terminal course
module — every prior topic (cross sections, deterministic
transport, feedback coefficients) feeds into its inputs.

.. contents::
   :local:
   :depth: 2

.. seealso::

   :ref:`theory-reactor-kinetics` — point-kinetics derivation,
   reactivity feedback, and the RIA scenario.


Physical Scope
--------------

The scenario runs in two phases:

1. **Steady state** (:math:`0 \le t \le 100\ {\rm s}`) — constant
   rod power equilibrates fuel / clad temperatures, coolant
   enthalpy, and gap conductance.
2. **Transient** (:math:`100 \le t \le 120\ {\rm s}`) — an inlet
   temperature perturbation drives a power excursion. Doppler
   broadening of :math:`^{238}\mathrm{U}` absorption and coolant
   density feedback oppose the excursion; gap closure is tracked
   as a terminal event.

The fuel rod is discretised into ``nz`` axial nodes and each axial
slice carries a radial mesh of ``fuel_nr`` fuel nodes plus
``clad_nr`` clad nodes. At each RHS evaluation the solver
computes a new radial temperature profile, a gap conductance from
roughness plus gas conduction plus radiation, an algebraic clad
stress state (von Mises + plastic correction), and a one-group
reactivity from Doppler and coolant temperature coefficients
multiplied by their respective temperature changes.


Port of MATLAB Module 10
------------------------

This module is the SciPy port of the MATLAB Module 10 course
materials (``reactorKinetics.m`` and friends). Two structural
changes were made to make the solver compatible with a standard
stiff-ODE integrator:

* **Stresses are algebraic, not DAE state.** In the MATLAB port,
  radial, hoop, and axial stress were treated as DAE variables
  with a mass matrix. ORPHEUS solves them algebraically at every
  RHS evaluation (the same choice made in Module 08 for the fuel
  rod), which eliminates the mass matrix and lets us use
  :func:`scipy.integrate.solve_ivp` with LSODA / BDF.
* **Pressure is algebraic.** Likewise, ideal-gas pressure in the
  plenum is computed from the current gas volume and temperature
  at every RHS call rather than carried as an AE variable.

What remains in the time-evolving state vector is only genuinely
dynamic quantities: reactor power, the six delayed-neutron
precursor densities, fuel and clad nodal temperatures, coolant
enthalpy, and clad plastic strain (effective and componentwise).


Point-Kinetics Equations
------------------------

With six delayed-neutron precursor groups the 0-D point-kinetics
system is

.. math::

   \frac{dP}{dt} &=
   \frac{\rho(t) - \beta_{\rm eff}}{\Lambda}\,P
   + \sum_{i=1}^{6} \lambda_i\,C_i, \\
   \frac{dC_i}{dt} &=
   \frac{\beta_i}{\Lambda}\,P - \lambda_i\,C_i,
   \qquad i = 1, \dots, 6,

where :math:`P` is the normalised reactor power, :math:`C_i` the
:math:`i`-th precursor density, :math:`\Lambda` the prompt
neutron lifetime, and
:math:`\beta_{\rm eff} = \sum_i \beta_i` the effective delayed
fraction. The reactivity is

.. math::

   \rho(t) = \alpha_D\,\bigl(\bar T_f(t) - \bar T_{f,0}\bigr)
   + \alpha_C\,\bigl(\bar T_c(t) - \bar T_{c,0}\bigr),

with Doppler coefficient :math:`\alpha_D = -2\times 10^{-5}\ {\rm
K^{-1}}` and coolant coefficient :math:`\alpha_C = -20\times 10^{-5}\
{\rm K^{-1}}` as the defaults in
:class:`~orpheus.kinetics.solver.KineticsParams`.


Data Classes
------------

* :class:`~orpheus.kinetics.solver.KineticsParams` — every
  physical input (geometry, burnup, :math:`\beta_i`, :math:`\lambda_i`,
  boundary-condition tables, solver tolerances). All defaults
  correspond to the baseline PWR pin used in the MATLAB module.
* :class:`~orpheus.kinetics.solver.KineticsResult` — the full
  time history: power, reactivity breakdown, precursor densities,
  2-D temperature fields, coolant properties (temperature,
  saturation temperature, CHF, flow regime, pressure, enthalpy,
  velocity, void), fuel / clad geometry, clad stress components,
  gap conductance, and plastic strain.

**Entry point:**
:func:`~orpheus.kinetics.solver.solve_reactor_kinetics` builds the
parameter dictionary via ``_initialize``, runs
:func:`scipy.integrate.solve_ivp` with the ``_gap_closure_event``
terminal event, and unpacks the output into a ``KineticsResult``.


API Reference
-------------

.. automodule:: orpheus.kinetics.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
