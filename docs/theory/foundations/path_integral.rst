.. _path-integral:

=========================================================
The Transport Path Integral: One Object, Five Methods
=========================================================

.. contents:: Contents
   :local:
   :depth: 1


.. Machine header — the ``nexus-meta`` schema for this page.  Ingestion is
.. PENDING nexus#1 Phase 2 (the directive is not yet registered), so the
.. schema is rendered here as a collapsed dropdown and machine-consumed
.. later.

.. dropdown:: Machine header — ``nexus-meta`` schema (role · thesis · axes)
   :color: muted

   .. code-block:: yaml

      module: transport
      concept: path_integral
      status: "AUTHORED — Phase H (2026-07-22); the seven scaffold anchors are stable"
      role: "root of the transport corpus; parent of methods/index — the frame every method derives FROM"
      thesis: >
        the five transport methods are five discretizations of ONE object
        (the sum over neutron histories); the reaction operators C, S, F are
        method-invariant AT FIXED MULTIGROUP DATA; a method chooses how to
        realize the propagator (L+C)^-1 — with diffusion the one exception
        (a limit of the object, not a quadrature of it)
      taxonomy_axes:
        A1: "how (L+C)^-1 is realized — sweep-DAG / track / region-pair kernel / sampled / not-at-all (a limit)"
        A2: "where S is resummed      — outer Neumann (SI) / direct inverse / exact spectral (Case Λ) / in-process (MC)"
        A3: "angular representation   — ordinates / harmonics / continuous / Case ν-spectrum"
      axes_note: "the three axes are INDEPENDENT; their product is PARTIALLY POPULATED (MC has no A2 value; diffusion and Case have no A1 value)"
      shared_operators:
        all_three_consumers: [MultiplicationOperator, IsotropicFission]    # orpheus.transport.operators — [M] 2026-08-31 the only two classes instantiated in sn AND diffusion AND homogeneous
        iso_specializations: [IsotropicScattering, IsotropicN2N]           # diffusion + homogeneous
        angular_bindings: [ScatteringOperator, N2NOperator, FissionOperator]  # SN only — the frame's ℓ-conjugations of the energy bindings above
        shared_kernel: [ScatteringOperator]                                # SN routes the same package's anisotropic kernel
      eigenvalue_posing: "k and α are properties of the OPERATOR, posed before any discretization; every method inherits the posing"
      gates_resolved: "#298 + #299 fixed in-branch; Phase-I literature survey ingested (Larsen–Morel 2010, Adams–Larsen 2002)"
      delivered: [kinetic_ledger, name_chain, generator_splitting_table, girsanov_bridge, three_axis_method_table, pade_positivity_table, method_placement_map, eigenvalue_spectral_yield]
      algebra_of_record: orpheus.derivations.discrete.sn.sweep_acyclicity   # the A1 sweep-DAG claim: SCC decomposition of the (face, ordinate) trace digraph
      depends_on: [operator_algebra, discretization]
      parent_of: [methods/index]


This is the **root of the transport corpus** — the page every method
derives *from*, and the parent of the :doc:`transport-methods entry
</theory/methods/index>`. Where that entry states the differential
transport equation the deterministic methods discretize, this page answers
the prior question: *what is the one object that all of the methods
approximate, which part of it is shared, which part varies, on which axes
do methods differ, and where does each one land?*

The thesis, in one line: **the transport methods are not five different
subjects but five discretizations of one object — the sum over neutron
histories.** The operator algebra is powerful precisely because the
*reaction* operators are shared objects across methods — a shared-code
fact, not an analogy: the collision operator
:class:`~orpheus.transport.operators.MultiplicationOperator` and the
fission energy binding
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
are the *same Python classes* instantiated by S\ :sub:`N`, diffusion and
the infinite-medium solver, and all three draw their scattering from the
same :mod:`orpheus.transport.operators` package —
:class:`~orpheus.transport.operators.IsotropicScattering` and
:class:`~orpheus.transport.operators.IsotropicN2N` for the isotropic
consumers (diffusion, infinite-medium), the same package's anisotropic
:class:`~orpheus.transport.operators.ScatteringOperator` kernel for
S\ :sub:`N`. A method is a choice of how to realize what remains.

The five, by name: **discrete ordinates** (S\ :sub:`N`), the **method of
characteristics** (MoC), **collision probability** (CP), **Monte Carlo**
(MC), and **diffusion** — the production methods of this corpus, each
with its own book under the :doc:`transport-methods entry
</theory/methods/index>`. The taxonomy of this page also places two
families that sit *outside* the production five: **P**\ :sub:`N`
(diffusion's spherical-harmonics siblings, sharing its column of the
axis table) and **Case / F**\ :sub:`N` (the exact-spectral family,
realized in this corpus as :doc:`reference solvers
</theory/references/index>`, not as a production method) — which is why
the axis table of Section :ref:`5 <path-integral-axes>` carries six
columns for five methods.

One distinction keeps this thesis honest, and the reader should carry it
through the whole page: the *one object* is the **physical quantity** —
the expected outcome of the neutron histories. Two methods that
discretize the same object generally do **not** discretize the same
**operator**: each method poses its own operator equation, on its own
function space, and statements elsewhere in this corpus that "the two
architectures are not different discretisations of the same operator"
(the :doc:`Peierls–Nyström vs. trajectory-resolvent split
</theory/references/peierls>`) are exact and remain true. Section
:ref:`6 <path-integral-method-map>` states the reconciliation precisely.


.. _path-integral-one-object:

1. The one object
=================

Every ORPHEUS method computes the **first moment of one branching
stochastic process**. The process is easy to state: a neutron is released
into the medium by a source; it streams in a straight line, surviving a
path of :term:`optical thickness` :math:`\tau` with probability
:math:`e^{-\tau}`; at a collision it branches — into one neutron
(scattering, with a new direction and energy drawn from the differential
cross section), zero (capture), or :math:`\nu` (fission, each newborn
drawn from :math:`\chi`). The family of all such **histories** — every
flight, every branch, every generation — is the object. Each solved-for
field is a first moment of it: the :term:`angular flux`
:math:`\psi(\vec r, \hat\Omega, E)` is the expected path length traversed
per unit phase-space volume per unit time, and every physically measurable
response pairs against it,

.. math::
   :label: path-integral-track-length-moment

   \bigl\langle \Sigma_d,\; \phi \bigr\rangle
   \;=\;
   \mathbb{E}\!\left[\,
     \sum_{u \,\in\, \text{histories}}
     \int_{\text{flight of } u} \Sigma_d\bigl(\vec r_u(s)\bigr)\,
     \mathrm{d}s
   \right],

.. (vv-status rationale) Definitional identity: the track-length moment is
   the DEFINITION of the quantity all five transport methods compute
   (Lux–Koblinger 1991) — ⟨Σ_d,φ⟩ as the expected Σ_d-weighted track length
   over the history family. A literature-transcribed definition, not a solver
   claim; the flux it defines is verified per method downstream.
.. vv-status: path-integral-track-length-moment documented

the detector reading as the expected accumulation of
:math:`\Sigma_d`-weighted **track length** over the whole family. This
identity is not a Monte Carlo convention; it is the definition of what all
five methods compute, and the reason a track-length tally
:cite:`Lux1991` and a converged S\ :sub:`N` :term:`sweep` estimate
*the same number*.

Because the first moment is what every method targets, the branching tree
collapses. This is the **many-to-one lemma** (the *spine* decomposition)
of branching-process theory :cite:`LyonsPemantlePeres1995,HardyHarris2009`:
the expectation of a sum over all particles of a branching walk equals the
expectation of a **single** distinguished path — the spine — carrying a
multiplicative weight that books the mean branching encountered along it,

.. math::
   :label: path-integral-many-to-one

   \mathbb{E}\!\left[\,\sum_{u \in N_t} f\bigl(X_u(t)\bigr)\right]
   \;=\;
   \mathbb{E}\!\left[\;
     e^{\int_0^t (m(\xi_s) - 1)\,\beta(\xi_s)\,\mathrm{d}s}\,
     f(\xi_t)
   \right],

.. (vv-status rationale) Literature-transcribed theorem: the many-to-one
   (spine) lemma of branching-process theory (Lyons–Pemantle–Peres 1995,
   Hardy–Harris 2009). A classical identity that legitimises the first-moment
   collapse, not a solver claim.
.. vv-status: path-integral-many-to-one documented

with :math:`\xi` the spine, :math:`\beta` the collision rate and :math:`m`
the mean number of secondaries per collision. **Linearity is what makes
this collapse legal**: the transport equation evolves the first moment and
only the first moment, so the tree's full genealogy is redundant for the
flux. And **fission does not break the path reading — it makes the
multiplicative weight exceed one**: a multiplying medium is simply the
regime :math:`m > 1` along part of the spine, where the weight
:math:`e^{\int (m-1)\beta}` grows. (What *does* eventually break, and
exactly where, is the divergence of the generation sum in a supercritical
medium — the honest limit of the path reading, kept for Section
:ref:`7 <path-integral-eigenvalue>`.)

One naming discipline, before the readings: this page does **not** call
the object "the Feynman–Kac formula." For the pure propagator — a neutron
between collisions — the motion is *deterministic* straight-line flight
under exponential killing, and an "expectation" over it is an expectation
over a one-point measure: technically a Feynman–Kac representation,
vacuously so. The name earns its content only when the jump kernel rides
inside the stochastic process — the Monte Carlo splitting of Section
:ref:`4 <path-integral-generator-splitting>` — and that is the only place
this corpus uses it.

Where the object comes from, and why it is linear
-------------------------------------------------

The path sum is not postulated; it is what survives of two-species kinetic
theory after three deliberate reductions, and knowing the ledger is what
tells you *when the object stops being valid*
:cite:`BellGlasstone1970,Duderstadt1976,CaseZweifel1967`.

**The semiclassical hybrid.** Neutron transport is a hybrid description:
classical ballistics between interactions, quantum mechanics inside them.
The de Broglie wavelength of a thermal neutron is
:math:`\lambda_{\mathrm{dB}} = h/mv \approx 1.8\ \text{Å}` — eight orders
of magnitude below a centimetre-scale mean free path — so between
collisions the neutron is a classical point particle with a well-defined
:math:`(\vec r, \hat\Omega, E)`, and a phase-space density :math:`\psi` is
meaningful. At a collision the same wavelength is *comparable to nuclear
and interatomic scales*, so the interaction itself is irreducibly quantum
— resonances, interference between scattering centres (the Bragg cutoff in
crystalline moderators), spin statistics. All of that quantum content
enters the theory through one door: the **evaluated cross sections**. This
split is not a modelling accident — it is this page's thesis seen at the
microscopic level: *the quantum data live exactly in the invariant
operators* :math:`C, S, F` *(Section* :ref:`2 <path-integral-invariant>`\
*), and the classical ballistics live exactly in the propagator*
:math:`(L+C)^{-1}` *whose realization varies by method (Section*
:ref:`3 <path-integral-streaming>`\ *)*.

**The two-species ledger.** Done honestly, a reactor is a coupled
two-species kinetic system — neutrons with density :math:`f_n` and target
atoms with density :math:`f_A` — with **two self-interaction terms and one
shared event**. A neutron–atom collision is *one* physical event with two
bookkeeping faces: the term :math:`Q_{nA}[f_n, f_A]` records its effect on
the neutron field, and :math:`Q_{An}[f_A, f_n]` records the recoil's
effect on the atom field. The honest pair is

.. math::
   :label: path-integral-two-species-ledger

   \partial_t f_n + \vec v \cdot \nabla f_n
     &= Q_{nn}[f_n, f_n] \;+\; Q_{nA}[f_n, f_A],
   \\
   \partial_t f_A + \vec v_A \cdot \nabla f_A
     &= Q_{AA}[f_A, f_A] \;+\; Q_{An}[f_A, f_n],

.. (vv-status rationale) Literature-transcribed kinetic-theory ledger: the
   coupled two-species Boltzmann system (Bell–Glasstone 1970, Duderstadt–
   Hamilton 1976, Case–Zweifel 1967) from which linear neutron transport is
   recovered by three stated reductions. A governing-equation transcription,
   not a solver claim.
.. vv-status: path-integral-two-species-ledger documented

and every collision term is **bilinear** in its two arguments — the full
system is nonlinear. Neutron transport is this system after **three
switches**:

1. **Drop** :math:`Q_{nn}`. Neutron densities in a power reactor are
   :math:`\sim 10^{7}`–:math:`10^{9}\ \text{cm}^{-3}` against atomic
   densities of :math:`\sim 10^{22}`–:math:`10^{23}\ \text{cm}^{-3}`; with
   comparable (barn-scale) cross sections, neutron–neutron collisions are
   some fourteen orders of magnitude rarer than neutron–atom ones. This is
   a *density* argument, and it has a genuine failure regime: neutron-star
   matter, where the neutron fluid is the dense species.
2. **Freeze** :math:`f_A` **to a Maxwellian at the material temperature**
   :math:`T`. The atom–atom term is not *ignored* — it is **presumed
   complete**: interatomic interactions equilibrate the target medium on
   timescales far shorter than neutronic ones, so :math:`f_A` is pinned at
   the equilibrium :math:`Q_{AA}` enforces, and :math:`Q_{AA}` itself then
   has nothing left to do in the equation. The surviving imprint of the
   interatomic physics is *inside the cross sections*: thermal-scattering
   laws :math:`S(\alpha, \beta)`, and the Bragg cutoff — crystal binding
   appearing as structure in :math:`\sigma`, exactly per the semiclassical
   split above.
3. **Keep** :math:`Q_{nA}`, evaluated against the frozen :math:`f_A`. The
   bilinear form :math:`Q_{nA}[f_n, f_A^{\mathrm{Maxwell}}(T)]` collapses
   to a **linear** operator on :math:`f_n` — the :math:`C`, :math:`S`,
   :math:`F` of this corpus, with the material temperature as a parameter.

**Linearity requires switches 1 and 2 together.** Dropping
:math:`Q_{nn}` alone is not enough: :math:`Q_{nA}[f_n, f_A]` with a *live*
:math:`f_A` is still effectively nonlinear in the neutron field, because
:math:`f_n` drives :math:`f_A` through :math:`Q_{An}` and :math:`f_A`
feeds back through :math:`Q_{nA}`. The linear transport operator exists
because the neutron field is dilute *and* the target field is pinned.

**What the thermal treatment recovers — and what it deliberately does
not.** The fast-range simplification "the target is stationary"
(:math:`E \gg kT`) is an *energy* argument layered on top of the ledger,
and when it fails — in the thermal range — the evaluated data walk it
back: free-gas and :math:`S(\alpha,\beta)` kernels restore the
**target-motion face** of the collision, the channel by which atoms hand
energy *to* neutrons (upscattering). The fingerprint that this channel has
been restored *correctly* is **detailed balance**: with :math:`M(E)` the
Maxwellian flux spectrum at the target temperature, the thermal kernel
satisfies :math:`M(E)\,\Sigma_s(E \to E') = M(E')\,\Sigma_s(E' \to E)`
exactly — so in an infinite non-absorbing medium it drives the neutron
spectrum *toward the equilibrium of the frozen target and holds it there*.
That is equilibrium physics encoded in :math:`\sigma`, with :math:`f_A`
*still frozen*, so linearity survives the recovery. What is **never** re-admitted kinetically is the
**back-reaction face** :math:`Q_{An}` — neutrons and their reaction
products reshaping :math:`f_A` (heating). Its neglect is a
*density-and-timescale* argument, not an energy argument, and it returns
not inside the operator but **quasi-statically around it**: the material
temperature that parametrizes :math:`\Sigma(T)` (Doppler broadening,
thermal expansion) is updated by a thermal-hydraulics model *outside* the
transport solve — the architectural seat of multiphysics feedback, and
the reason a coupled steady state iterates transport against
:math:`\Sigma(T)` rather than solving a nonlinear kinetic system. The
failure regime of the quasi-static split is fast, strong heating — a
transient violent enough that :math:`f_A` cannot be presumed
re-equilibrated between neutronic time steps.

Three readings of one object
----------------------------

The same first moment :eq:`path-integral-track-length-moment` admits three
exact readings, and every method chapter in this corpus opens from one of
them:

- **A collision-order series** — group the histories by *how many
  collisions they have suffered*. The :math:`k`-th class is the flux of
  neutrons scattered exactly :math:`k` times, and the sum over classes is
  the **Neumann–Peierls collision-number expansion**
  :cite:`Peierls1939` — the series the operator algebra states as
  :eq:`apply-solve-source-iteration-series` and names "the Peierls
  collision-number expansion"; its per-term inverter is the
  sweep and its outer summation is source iteration. This is the reading
  the deterministic methods discretize, and the iteration literature
  leans on its physical meaning: with a zero initial guess, the
  :math:`\ell`-th source iterate *is* the flux of particles that have
  scattered at most :math:`\ell - 1` times :cite:`AdamsLarsen2002`.
- **A resolvent** — read the stationary problem as the time-integral of
  an evolution: the transport generator :math:`\mathcal{A}` advances an
  initial population, and the stationary flux is the accumulated exposure
  :math:`\int_0^\infty e^{t\mathcal{A}} q\, \mathrm{d}t = (-\mathcal{A})^{-1} q`
  — when the integral converges, which is the subcriticality condition
  Section :ref:`4 <path-integral-generator-splitting>` states precisely.
  This reading carries the functional-analytic content — which splittings
  of :math:`\mathcal{A}` converge, and why — and Section
  :ref:`4 <path-integral-generator-splitting>` gives its honest name
  chain.
- **A Monte Carlo expectation** — sample the histories and average the
  track-length functional :cite:`Lux1991`. No discretization of phase
  space occurs; the estimator error is statistical. Section
  :ref:`4 <path-integral-generator-splitting>` locates *which* stochastic
  process is simulated — there is more than one, and they are not
  interchangeable.

These are three readings of one object, **not** three approximations: each
is exact. Where methods differ is in *which reading they realize, and what
they then truncate* — the subject of the rest of this page.


.. _path-integral-invariant:

2. What is invariant — the reaction operators
=============================================

Collision (:math:`C`), scattering (:math:`S`) and fission (:math:`F`) are
**method-invariant reaction operators at fixed multigroup data**. The
reason is visible in the kinetic ledger of Section
:ref:`1 <path-integral-one-object>`: these operators *are* the evaluated
collision physics — the quantum data — and a numerical method never
touches them. A method chooses how a neutron *gets from one collision to
the next*; it has no say in *what happens when it arrives*.

Shared code, not shared analogy
-------------------------------

In ORPHEUS the invariance is demonstrated structurally, not asserted. The
reaction operators live in one package, :mod:`orpheus.transport.operators`,
and the methods consume them as follows:

- :class:`~orpheus.transport.operators.MultiplicationOperator` — the
  collision diagonal :math:`C = M[\Sigma_t]` — and the fission energy
  binding
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
  are the **same classes** instantiated by all three deterministic
  consumers (S\ :sub:`N` in ``orpheus/sn``, diffusion in
  ``orpheus/diffusion/solver.py``, the infinite-medium solver in
  ``orpheus/homogeneous/solver.py``).  ``[M]`` 2026-08-31, by AST over
  ``orpheus/``: those are the **only two** operator classes with a
  construction site in all three packages.
- Scattering **and fission** are **shared kernels with
  representation-matched faces** — the same pattern, stated once: the
  isotropic consumers (diffusion, infinite-medium) instantiate the
  *energy* bindings
  :class:`~orpheus.transport.operators.IsotropicScattering`,
  :class:`~orpheus.transport.operators.IsotropicN2N` and
  ``IsotropicFission``; S\ :sub:`N`, which resolves angle, routes the
  *same package's* angular bindings — the anisotropic
  :class:`~orpheus.transport.operators.ScatteringOperator` and
  :class:`~orpheus.transport.operators.n2n.N2NOperator` (two roles of
  ONE :class:`~orpheus.transport.operators.transfer.TransferOperator`
  since #426 step 2, differing in the yield alone), and
  :class:`~orpheus.transport.operators.fission.FissionOperator` — each
  of which is the harmonic frame's :math:`\ell`-conjugation of the
  corresponding energy binding and *retains it as its middle factor*
  (:ref:`sn-fission-binding-adjoint`).  ⚠ The :math:`\ell` range
  differs by channel and by *reason*: the two transfer gains conjugate
  at the solve's ``scattering_order``, fission at :math:`\ell = 0`
  because a fission spectrum IS isotropic.  Until 2026-09-04
  :math:`N_{2n}` conjugated at :math:`\ell = 0` too, and that was a
  model rather than a physical fact (ERR-082).  One package owns the reaction
  mathematics; each method takes the face matched to its angular
  representation, and the two faces of a channel cannot drift because
  one is built from the other.

  .. note::

     **This bullet said "**\ ``FissionOperator``\ **" until CS4c step 4
     (2026-08-30), and the correction sharpens the thesis rather than
     weakening it.**  Fission was the one reaction channel with a single
     class serving both a scalar and an angular consumer, which is why
     it read as the cleanest example of sharing — and why the *shape* of
     the sharing was invisible.  Step 4 split it into the two bindings
     of one datum that scattering and :math:`(n,2n)` already had, so the
     shared-code claim is now the same claim for all three channels
     instead of one claim for fission and another for scattering.
     ``[M]`` at HEAD: ``FissionOperator`` has **one** production
     construction site and it is in ``orpheus/sn``; ``IsotropicFission``
     has four, spanning ``diffusion``, ``homogeneous``, ``sn`` and
     ``transport``.

The payoff is the single-source-of-truth payoff everywhere in this
codebase: the collision physics is implemented once, verified once
(against the infinite-medium analytic anchor,
:doc:`/theory/foundations/infinite_medium`), and every method inherits
the verification. A fixed bug in :math:`S` is fixed for every solver
simultaneously; a divergence between two methods can never be hiding in
the reaction terms, because there is only one copy of them.

The canon contains one precedent for this factoring move. Hébert
:cite:`Hebert2009` factors the *streaming* operator out ahead of his
method chapters — deriving its Cartesian, cylindrical and spherical forms
once, before P\ :sub:`N`, CP, S\ :sub:`N` and MoC each consume it — the
only systematic de-duplication in the textbook literature. He factors
streaming by **geometry**; this page factors the reaction operators by
**method**. It is the same move, one level up: identify what is shared,
state it once, and let every downstream chapter *derive* rather than
*restate*.

The scope condition is load-bearing
-----------------------------------

"Invariant" is a statement **at fixed multigroup data**, and the scope
condition does real work. Multigroup condensation — collapsing a fine
group structure to a coarse one — is a **solution-weighted
Petrov–Galerkin projection** (:doc:`/theory/foundations/frame`): the
coarse-group cross sections are fine-group data weighted by a *flux*, and
the flux is the property of a *solution*, not of an operator. The
projection is therefore owned by no single operator and by no single
method. The consequence is the honest boundary of this section's claim:

  If a 2-group S\ :sub:`N` model and a 2-group diffusion model were each
  condensed with **their own** flux solution, their coarse-group
  :math:`S` operators would be **different operators** — same class, same
  kernel mathematics, different data. The invariance theorem compares
  methods *at the same data*, and ORPHEUS enforces exactly that: one
  cross-section pipeline (:doc:`/theory/foundations/cross_section_data`)
  produces the multigroup data, and every method consumes the identical
  :class:`~orpheus.data.macro_xs.mixture.Mixture` objects.

The distinction matters in practice because it is precisely where
"cross-method agreement" claims die when they are careless: two methods
disagreeing at *different* data are exhibiting a data difference, not a
method difference. The V&V discipline of this corpus
(:doc:`/theory/verification/cross_method`) pins every cross-method comparison to the
fixed-data regime for exactly this reason.

What remains once the invariant part is factored out is the subject of
the next two sections: the propagator, and the split of the generator
that defines it.


.. _path-integral-streaming:

3. What varies — realizing the propagator
=========================================

What a method actually chooses is **how to realize the propagator**
:math:`(L+C)^{-1}` — the inverse that carries a neutron from one collision
to the next. Here streaming carries its intuitive **Lagrangian** meaning:
attenuation along the flight path between interactions. (The complementary
**Eulerian** reading — that :math:`\hat\Omega\cdot\nabla\psi` is the
divergence of the angular current — is taken up on the :doc:`diffusion
page </theory/methods/diffusion_1d>`, where it becomes the continuity law
that Fick's law closes.)

Why the propagator is one bundled inverse
-----------------------------------------

The operator-algebra page establishes *that* inversion does not distribute
over the sum — :eq:`solve-does-not-distribute`, with the scalar
:math:`(3+5)^{-1} \ne 3^{-1} + 5^{-1}` anchor and the
resistors-in-parallel identity :eq:`apply-solve-parallel-identity` as the
algebraic illustrations. This page owes the reader the *analytic* reason
the split is drawn where it is — why the resolved unit is :math:`L + C`
and the perturbation is :math:`S` (and :math:`F`), never the other way
around. Two structural facts about the leaves decide it:

1. :math:`L = \hat\Omega \cdot \nabla` is a **differential operator**:
   unbounded, and with a **non-trivial kernel** — it annihilates every
   distribution constant along characteristics, so :math:`L` *alone* is
   not invertible at all. :math:`C = M[\Sigma_t]` is the multiplication
   by the total cross section: bounded, positive, trivially invertible.
2. :math:`S` and :math:`F` are **integral operators with evaluated-data
   kernels**: bounded, with norms controlled by the scattering and
   production cross sections.

Perturbation theory for semigroups and resolvents has one standing rule:
**the unbounded piece must ride inside the resolved unit, and the
perturbation must be the bounded part.** :math:`(L+C)^{-1}` exists as a
well-defined bounded operator precisely because :math:`C`'s absorption
regularizes :math:`L`'s kernel — a neutron flying forever along a
characteristic is exponentially killed — while a hypothetical split that
tried to "resolve" :math:`S` and perturb by :math:`L` would be
perturbing by an operator that no bounded series can control. The
familiar sweep-and-iterate structure of transport is therefore not one
option among many: every convergent splitting of the transport operator
resolves :math:`L + C` together and perturbs by reaction terms. Section
:ref:`4 <path-integral-generator-splitting>` makes this a theorem with a
name chain and a single convergence condition.

The realization is exact; the discretization is not
---------------------------------------------------

One two-level distinction must be kept sharp, because the whole of
Section :ref:`6 <path-integral-method-map>` stands on it:

- **At the discrete level, the propagator is realized exactly.** The
  S\ :sub:`N` sweep *is* :math:`(L+C)^{-1}` for the discretized
  operator — a direct triangular solve (forward substitution on the cell
  dependency order), not an iteration and not an approximation *of the
  discrete system*. The characteristic and collision-probability methods
  likewise apply their discrete propagators exactly.
- **What varies is the fidelity of the discrete propagator to the
  continuous one.** A spatial closure decides how faithfully one cell
  transmits the exponential attenuation :math:`e^{-\tau}` across itself
  — and Section :ref:`6 <path-integral-method-map>` shows the closures
  are literally *rational approximants of the exponential*, sitting in
  one Padé table.

Collapsing these two levels produces the false dichotomy this corpus once
flirted with — "deterministic methods approximate the generator, Monte
Carlo approximates the propagator." By the two-level reading, S\ :sub:`N`
is *propagator-side*: it realizes :math:`(L+C)^{-1}` exactly on its grid,
and its approximation error lives in the grid's fidelity, not in the
solve.

The partition itself is method-dependent
----------------------------------------

The realization of :math:`(L+C)^{-1}` is *not* the only thing that
changes across methods, and this page must not overclaim: **the split of
the generator into** :math:`L`, :math:`C`, :math:`S` **is itself
method-dependent.** The clean counterexample is diffusion's
transport-corrected removal. ORPHEUS computes
(:attr:`~orpheus.data.macro_xs.mixture.Mixture.transport_xs`, the
Stamm'ler outflow convention :cite:`Stamm1983`)

.. math::
   :label: path-integral-transport-correction

   \Sigma_{\mathrm{tr}} \;=\; \Sigma_t - \Sigma_{s1,\mathrm{out}},
   \qquad
   D \;=\; \frac{1}{3\,\Sigma_{\mathrm{tr}}},

.. (vv-status rationale) Literature-transcribed definition: the Stamm'ler
   outflow transport correction (Stamm'ler–Abbate 1983). Its terminal result
   is verified downstream under the diffusion method's ``diffusion-coefficient``
   label — ``tests/data/test_mixture_transport_xs.py`` pins Σ_tr = Σ_t −
   rowsum(Σ_s1) (foundation ``test_transport_xs_is_total_minus_p1_outscatter_row_sum``)
   and D = 1/(3Σ_tr) (L1 ``test_diffusion_coefficient_matches_definition``).
   Restated here to illustrate the method-dependent partition; not a separate claim.
.. vv-status: path-integral-transport-correction documented

which **relocates the** :math:`\ell = 1` **scattering moment into the
streaming term**: the anisotropy of scattering, a piece of :math:`S` in
the S\ :sub:`N` partition, lives inside diffusion's :math:`L` (through
:math:`D`). In diffusion, :math:`\Sigma_t` then appears in *both* the
streaming term and the removal term, whereas S\ :sub:`N`'s streaming leaf
is :math:`\sigma`-free. Same object, same physics — differently
partitioned generator. Statements of this page's thesis that claim "the
difference between methods is confined to streaming" are therefore
*false as stated*; the honest claim is the one Section
:ref:`2 <path-integral-invariant>` made — the reaction *data* are
invariant, while both the partition and the propagator realization are
the method's choice.

And **diffusion is the one genuine exception** to the propagator frame
altogether: it does not realize :math:`(L+C)^{-1}` *at all*. Diffusion is
a **limit of the object** — the leading order of the asymptotic expansion
in which the medium becomes optically thick and scattering-dominated
(:cite:`LarsenMorel2010` §1.4.4 carries the derivation this corpus
adopts) — rather than a :term:`quadrature` of it. That is why its solved
operator is elliptic and self-adjoint while every propagator-realizing
method solves characteristic-triangular systems, and why it earns the
"one principle, one exception" clause in the thesis. Section
:ref:`5 <path-integral-axes>` encodes this as diffusion having *no value
on the A1 axis*.


.. _path-integral-generator-splitting:

4. The branch point — how the generator is split
================================================

The deepest branch between methods — deeper than any discretization
choice — is a choice of **generator splitting**. Write the (source-free,
within-group) transport generator as

.. math::
   :label: path-integral-generator-splitting-eq

   \mathcal{A} \;=\; \mathcal{A}_0 \;+\; P,

.. (vv-status rationale) Structural/conceptual definition: the generator
   splitting into a resolved part and a perturbation — the branch point that
   organises the whole method family (killing / jump / majorised-jump). A
   framing definition, not a solver claim.
.. vv-status: path-integral-generator-splitting-eq documented

and choose which physics rides in the **resolved part**
:math:`\mathcal{A}_0` (the part whose evolution is computed in closed
form) and which becomes the **perturbation** :math:`P` (the part expanded
in a series, or realized as a jump kernel). Section
:ref:`3 <path-integral-streaming>` fixed the analytic constraint — the
unbounded streaming must ride inside :math:`\mathcal{A}_0` — but within
that constraint there are three genuinely different splittings, and the
whole method family organizes along them:

.. list-table::
   :header-rows: 1
   :widths: 16 22 22 22 18

   * - Splitting
     - :math:`\Sigma_t` sits in
     - :math:`S` sits in
     - Series
     - Methods
   * - **Killing**
     - the attenuation functional :math:`e^{-\int \Sigma_t\,\mathrm{d}s}`
     - the **source**
     - Dyson–Phillips = collision-order = source iteration
     - S\ :sub:`N`, MoC, CP, Peierls
   * - **Jump**
     - the jump **rate**
     - the jump **kernel**
     - none — the process *is* the answer
     - analog Monte Carlo
   * - **Majorized jump**
     - the majorant :math:`\Sigma_{\mathrm{maj}}`
     - the jump kernel + a virtual self-scatter
     - none
     - ORPHEUS MC (Woodcock :cite:`Woodcock1965`)

In the **killing split**, the neutron's free flight is deterministic
straight-line motion attenuated by the *whole* total cross section —
collision is read as removal — and everything a collision re-emits
(scattering, fission) is a source to be re-injected. Expanding in that
source yields the collision-order series of Section
:ref:`1 <path-integral-one-object>`: this is the splitting the
deterministic methods realize, term by term, sweep by sweep. In the
**jump split**, the collision is read not as removal-plus-source but as a
*transition*: the neutron jumps to a new direction and energy with rate
:math:`\Sigma_t v` and kernel :math:`\Sigma_s/\Sigma_t`. Nothing is
expanded; one simulates the process and takes the expectation — analog
Monte Carlo, and the one place this corpus speaks of the object as a
**Feynman–Kac expectation for a piecewise-deterministic Markov process**.
The **majorized-jump split** modifies the jump split for geometric
convenience: sample collisions at a spatially *constant* majorant rate
:math:`\Sigma_{\mathrm{maj}} \ge \Sigma_t`, and classify each candidate
collision as real (probability :math:`\Sigma_t/\Sigma_{\mathrm{maj}}`) or
as a **virtual self-scatter** that continues the flight unchanged — so
the tracking never needs to resolve where material boundaries fall along
a flight path :cite:`Woodcock1965,Lux1991`.

The delta-tracking bridge is a change of measure
------------------------------------------------

The jump and majorized-jump splittings are connected by an exact
**Girsanov-type change of measure** on path space, and stating it
precisely retires any impression that delta tracking is an ad-hoc trick.
Let :math:`\mathbb{P}_t` be the law of the analog process (jump rate
:math:`\Sigma_t` along the flight path) and
:math:`\mathbb{P}_{\mathrm{maj}}` the law of the majorized process (rate
:math:`\Sigma_{\mathrm{maj}}`). On a path with collision points
:math:`x_1, \dots, x_J`, the Radon–Nikodym derivative is

.. math::
   :label: path-integral-girsanov

   \frac{\mathrm{d}\mathbb{P}_t}{\mathrm{d}\mathbb{P}_{\mathrm{maj}}}
   \;=\;
   \exp\!\left(
     \int \bigl(\Sigma_{\mathrm{maj}} - \Sigma_t\bigr)\,\mathrm{d}s
   \right)
   \prod_{j=1}^{J}
     \frac{\Sigma_t(x_j)}{\Sigma_{\mathrm{maj}}},

.. (vv-status rationale) Literature-transcribed identity: the Girsanov /
   Radon–Nikodym change of measure on path space bridging the analog and
   majorised (delta-tracking) Monte-Carlo laws (Woodcock 1965, Lux–Koblinger
   1991). A change-of-measure definition for the MC reading, not a solver claim.
.. vv-status: path-integral-girsanov documented

with the integral running along the flight path. Read its two factors
against the algorithm: the **per-jump factor**
:math:`\Sigma_t/\Sigma_{\mathrm{maj}}` *is* the delta-tracking acceptance
probability, and the **exponential compensator** — the likelihood the
majorized process owes for jumping more often — is realized in
expectation by the rejected candidates themselves: thinning a rate-\
:math:`\Sigma_{\mathrm{maj}}` collision stream by the acceptance
probability reproduces a rate-:math:`\Sigma_t` stream exactly, with the
virtual scatters (whose kernel is the identity) supplying the
compensator. Delta tracking is therefore an **unbiased reweighting of
one splitting into another**, not an approximation — the acceptance test
is the Radon–Nikodym derivative evaluated jump by jump.

Different splittings, one value
-------------------------------

The three splittings must be compared with care, because two statements
that sound contradictory are both true:

- **Value-equivalence:** all three splittings are *exact representations
  of the same object*. Each evaluates the first moment
  :eq:`path-integral-track-length-moment` without error (before
  discretization or finite sampling). None is an approximation of
  another.
- **Structure-non-equivalence:** they are *not interchangeable readings
  of one process*. The killing split's object is a **series** whose terms
  a deterministic method computes; the jump splits' object is a
  **process** whose realizations a stochastic method samples. They
  produce different algorithmic anatomies, different error structures
  (truncation and discretization versus statistical variance), and
  different intermediate quantities — there is no per-collision-order
  flux inside an analog Monte Carlo run, and no sampled history inside a
  sweep.

The deterministic-versus-stochastic distinction — the traditional
top-level split of every textbook — is therefore *real but derivative*:
it falls out of the generator splitting, one level below where the
textbooks draw it. What the page denies is only the interchangeability of
the splittings, never the exactness of each.

One condition governs every series
----------------------------------

The killing split's convergence has a single honest statement, and it is
the same statement in the stationary and time-dependent readings. The
names, each with its precise role: **Hille–Yosida** is the *generation
theorem* — it certifies that the transport generator generates a
contraction semigroup at all, so :math:`e^{t\mathcal{A}}` exists to be
integrated. **Dynkin's formula** is the *probabilistic resolvent* — it
identifies :math:`(\lambda - \mathcal{A})^{-1}` with an expected
discounted time-integral over paths, the bridge between the operator and
the process readings. **Dyson–Phillips** is the *time-dependent
perturbation series* — the expansion of :math:`e^{t(\mathcal{A}_0 + P)}`
in time-ordered integrals of :math:`P` against
:math:`e^{t\mathcal{A}_0}`. Its stationary shadow is the
**Neumann–Peierls series** — the collision-order expansion the operator
algebra states as :eq:`apply-solve-source-iteration-series` and already
names "the Peierls collision-number expansion," after Peierls'
integral-equation form of the transport problem :cite:`Peierls1939`. And **Feynman–Kac for a PDMP** is the Monte Carlo reading
of the same resolvent. One condition governs both expansions:

.. math::
   :label: path-integral-subcriticality

   \rho\bigl[(L+C)^{-1} S\bigr] \;<\; 1
   \quad\Longleftrightarrow\quad
   \int_0^\infty e^{\,t\mathcal{A}}\,\mathrm{d}t
   \ \text{converges},
   \qquad
   \mathcal{A} = -(L + C - S),

.. (vv-status rationale) Mathematical identity: the subcriticality /
   series-convergence condition — the iteration spectral radius crossing 1
   coincides with convergence of the semigroup time-integral. A spectral-
   convergence statement, not a solver claim.
.. vv-status: path-integral-subcriticality documented

and when it holds, both sides construct the *same* operator
:math:`(L+C-S)^{-1}` — the stationary series sums it, the semigroup
time-integral accumulates it. (The equivalence leans on positivity: for
positivity-preserving operators — Krein–Rutman territory, taken up in
Section :ref:`7 <path-integral-eigenvalue>` — the spectral radius of the
iteration operator and the spectral bound of the generator cross their
thresholds together. It is *not* a generic operator identity.)

The hypothesis deserves its physics name: **sub-stochasticity of the
collision** — on average, a collision must not return more than one
neutron through the channels ridden by :math:`S`. The classical bound is
the :term:`scattering ratio` :math:`c`:
:math:`\rho[(L+C)^{-1}S] \le \max_g c_g` **when** :math:`S` **carries
scattering alone** — once :math:`(n,2n)` rides inside :math:`S`, as it
does in ORPHEUS, the operative per-collision count is the :math:`c^\ast`
of clause 2 below. Two honesty clauses attach:

1. **The bound is a supremum, not an identity.** :math:`\rho = c` is
   attained only in the infinite-homogeneous limit; any finite medium
   with leakage gives :math:`\rho < c` strictly — leakage shortens
   neutron lifetimes and hastens convergence, an effect that fades as the
   system grows optically thick :cite:`AdamsLarsen2002`.
2. **The hypothesis is checkable against ORPHEUS's own data — and it can
   fail before fission enters.** The data model's per-group balance
   (``orpheus/data/macro_xs/mixture.py``) is

   .. math::
      :label: path-integral-substochasticity-bound

      \Sigma_t
      \;=\;
      \Sigma_c + \Sigma_L + \Sigma_f
      + \Sigma_{s0} + \Sigma_{2n},

   .. (vv-status rationale) Literature-transcribed definition: the data-model
      per-group balance every Mixture carries (the same identity as
      :eq:`sigT-computed`); gated by ``Mixture.assert_balanced``. A data-layer
      definition, restated here to derive the sub-stochasticity check; not a
      solver claim.
   .. vv-status: path-integral-substochasticity-bound documented

   with :math:`\Sigma_c` capture, :math:`\Sigma_L` the
   :math:`(n,\alpha)`-family absorption, :math:`\Sigma_{s0}` the **full**
   P\ :sub:`0` scattering row sum — in-group scatter included, because
   the total cross section counts *every* collision — and
   :math:`\Sigma_{2n}` the :math:`(n,2n)` row sum, a channel emitting
   **two** neutrons per event (the factor of 2 is applied at the
   scattering-source assembly in
   :class:`~orpheus.transport.operators.IsotropicN2N` and the
   infinite-medium solver). The mean number of secondaries per collision,
   fission excluded, is
   :math:`c^\ast = (\Sigma_{s0} +
   2\,\Sigma_{2n})/\Sigma_t`, and the balance identity turns
   the sub-stochasticity check into a one-line criterion:

   .. math::
      :label: path-integral-n2n-criterion

      c^\ast > 1
      \quad\Longleftrightarrow\quad
      \Sigma_{2n} \;>\; \Sigma_c + \Sigma_L + \Sigma_f .

   .. (vv-status rationale) Derivation step: the (n,2n) super-stochasticity
      criterion c* > 1, obtained by rearranging the balance identity
      :eq:`path-integral-substochasticity-bound`. A data-checkable criterion,
      not a solver claim.
   .. vv-status: path-integral-n2n-criterion documented

   A group where :math:`(n,2n)` production outweighs all absorption is
   *locally* super-stochastic without any fission — the collision-order
   bound exceeds one, and convergence of the scattering series then rests
   on leakage and on the other groups, not on the per-collision bound.
   :math:`(n,2n)` breaks sub-stochasticity **before fission does**,
   because it rides inside :math:`S` where fission does not.

This is the honest form of the folklore "source iteration converges
because :math:`c < 1`": a *supremum* bound, with a *stated* hypothesis,
and a *data-checkable* violation criterion.


.. _path-integral-axes:

5. The three independent axes
=============================

Once the object is fixed and the splitting chosen, a method is located by
**three independent choices**, not by a single dichotomy:

- **A1 — how** :math:`(L+C)^{-1}` **is realized:** a sweep over a
  cell-dependency DAG (S\ :sub:`N`), exact exponential attenuation along
  tracks (method of characteristics :cite:`Askew1972`), a
  region-to-region kernel (collision probability), sampled histories
  (Monte Carlo), or **not realized at all** (diffusion — a limit; Case —
  an exact spectral solution). The canonical power-iteration engine in
  :mod:`orpheus.numerics.eigenvalue` documents exactly this axis as its
  "resolvent" layer: the S\ :sub:`N` sweep or Krylov solve, CP's
  ``P_inf`` collision-probability kernel — *that kernel is CP's*
  :math:`(L+C)^{-1}` — and diffusion's and the infinite-medium solver's
  eager LU. Every deterministic method ORPHEUS ships factors its loss
  operator; what varies on A1 is only the inner-solve *strategy and
  carrier*.
- **A2 — where** :math:`S` **is resummed:** an outer Neumann iteration
  (source iteration, summing :eq:`apply-solve-source-iteration-series`
  term by term), a direct inverse (assemble :math:`L+C-S` and factor it),
  or an **exact spectral resummation** — Case's dispersion function
  :math:`\Lambda(z)`, in which the whole collision series has been summed
  in closed form and the scattering ratio :math:`c` survives only
  as a *parameter* :cite:`Case1960,CaseZweifel1967`.
- **A3 — the angular representation:** :term:`discrete ordinates
  <ordinate>`, spherical harmonics, a continuous direction (Monte Carlo),
  or the Case discrete-plus-continuum :math:`\nu`-spectrum. The angular
  representation is also what fixes the **trace** of the boundary law:
  ORPHEUS factors :math:`B` as a **method-invariant law composed with a
  method-specific trace** (``orpheus/transport/method.py`` —
  ``realize_boundary_law`` per method, one generic
  ``resolve_boundary_conditions`` body), and the trace space is set by
  the *angular* representation — S\ :sub:`N`'s trace is angular
  (``SNMesh.angular_trace``), diffusion's is scalar
  (``DiffusionMesh.scalar_trace``) — **not** by how streaming is
  realized. Reflection, :term:`vacuum <vacuum boundary condition>` and
  :term:`albedo` *laws* are shared physics; what
  a method owns is only the space their traces act on
  (:doc:`/theory/foundations/boundary_conditions`).

Independent, and only partially populated
-----------------------------------------

The axes are **independent** — no choice on one determines the choice on
another — but the honest statement is stronger than "orthogonal," and it
is what the word "orthogonal" would hide: **the product space is only
partially populated.** Monte Carlo has *no* A2 value — its scattering is
never resummed anywhere, because in the jump splitting :math:`S` rides
inside the process itself. Diffusion and Case have *no* A1 value — no
propagator is realized, by limit in one case and by closed-form
resummation in the other. The map of Section
:ref:`6 <path-integral-method-map>` marks these cells empty rather than
forcing a value into them; a taxonomy that fills every cell is a taxonomy
that has stopped being true.

.. list-table::
   :header-rows: 1
   :widths: 14 16 15 15 13 14 13

   * - Axis
     - S\ :sub:`N`
     - MoC
     - CP
     - MC
     - Diffusion / P\ :sub:`N`
     - Case / F\ :sub:`N`
   * - **A1** — :math:`(L+C)^{-1}` realized as
     - cell-DAG sweep (rational-approximant transmission)
     - exact :math:`e^{-\tau}` along tracks
     - region-pair Peierls kernel
     - sampled flights
     - — (a limit)
     - — (closed form)
   * - **A2** — :math:`S` resummed by
     - outer Neumann (SI) or Krylov
     - outer Neumann (SI)
     - outer iteration on the kernel
     - — (in the process)
     - direct inverse (LU)
     - exact spectral: :math:`\Lambda(z)`
   * - **A3** — angle represented by
     - ordinates
     - ordinates along tracks
     - integrated out
     - continuous
     - harmonics
     - :math:`\nu`-spectrum

One classification error is common enough to refuse explicitly: **Case
and F**\ :sub:`N` **are not "spectral in angle."** The
:math:`\nu`-spectrum is not an angular basis choice (A3 alone); the
singular eigenfunctions diagonalize the **full within-group generator**
:math:`L + C - S`, which is why the collision series never appears — it
has been *summed exactly*, an A2 value ("exact spectral"), with the
:math:`\nu`-spectrum as the A3 representation that makes the
resummation closed-form. Reading Case as merely an exotic angular basis
misses what the method actually resolves, and misclassifies the one
family that occupies A2's most extreme value. (The classical taxonomy
review — Sanchez & McCormick :cite:`Sanchez1982` — splits the family
top-level into integro-differential and integral formulations; the
three-axis frame refines that split rather than contradicting it: in
these coordinates, the formulation dichotomy is a projection of A1.)


.. _path-integral-method-map:

6. Where each method lands
==========================

With the axes fixed, each method is a **point in their partially
populated product space** — and neighbours that a textbook keeps in
separate chapters turn out adjacent. The sharpest adjacency is the one
the canon most thoroughly obscures: S\ :sub:`N` and the method of
characteristics sit on the **same side** of every divide that matters.

One propagator, one Padé table
------------------------------

Fix a cell and an ordinate, and let :math:`\tau` be the
optical thickness of the traversal — the optical path length
:math:`\Sigma_t \Delta s` along the flight, the CP/MoC sense of the
symbol (the S\ :sub:`N` chapters' *closure weight*, the Morel–Montry
angular weight :cite:`MorelMontry1984`, is a different object that shares
the letter; the notation crosswalk keeps the ledger). The continuous
propagator transmits the cell with attenuation :math:`e^{-\tau}`. Every
spatial closure of the sweeping methods — step,
:term:`diamond difference`, characteristics — answers one question —
*what does the discrete cell transmit instead?* — and the answers line up
as entries of the **Padé table of the exponential**:

.. math::
   :label: path-integral-pade-table

   \underbrace{\frac{1}{1+\tau}}_{\text{step} \;=\; [0/1]}
   \qquad
   \underbrace{\frac{1-\tau/2}{1+\tau/2}}_{\text{diamond difference} \;=\; [1/1]}
   \qquad
   \underbrace{e^{-\tau}}_{\text{characteristics — exact}}

.. (vv-status rationale) Representational identity: the spatial-closure family
   read as entries of the Padé table of e^−τ (step = [0/1], diamond-difference
   = [1/1], characteristics = exact). The [1/1] entry is realised by the
   sweep-cache amplification coefficient a = (2|μ|−Σ_tV)/(2|μ|+Σ_tV)
   (``diamond.py`` / ``sn/sweep/cache.py``). A representational framing, not a
   solver claim.
.. vv-status: path-integral-pade-table documented

The identification is verbatim in the code, not an analogy: the sweep
cache's slab-neutral amplification coefficient is
:math:`a = (2|\mu| - \Sigma_t V)/(2|\mu| + \Sigma_t V)`
(``orpheus/transport/spatial/diamond.py``,
``orpheus/sn/sweep/cache.py``), which is
:math:`(2-\tau)/(2+\tau) = (1-\tau/2)/(1+\tau/2)` with
:math:`\tau = \Sigma_t V / |\mu|` — the :math:`[1/1]` Padé approximant of
:math:`e^{-\tau}`, character for character.

The table now yields the sign structure of the whole closure family as a
theorem rather than folklore:

- **Step never produces a negative flux from positive data, and the
  table says why**: the :math:`[0/1]` entry has a *constant numerator* —
  its only zero-free way of approximating a positive function is to be a
  positive function, and :math:`1/(1+\tau) > 0` for all
  :math:`\tau > 0`. The price is paid in order: matching
  :math:`e^{-\tau}` only through :math:`O(\tau)` makes step first-order.
- **Diamond difference buys second order at the price of a numerator
  zero.** The :math:`[1/1]` entry matches :math:`e^{-\tau}` through
  :math:`O(\tau^2)`, and its numerator :math:`1 - \tau/2` **changes sign
  at** :math:`\tau = 2`: an optically thick cell
  (:math:`\tau > 2`) transmits a positive inflow as a *negative*
  contribution. The infamous "negative flux in optically thick cells" is
  therefore the **numerator zero of the** :math:`[1/1]` **approximant** —
  a structural property of the table entry, not a scheme pathology and
  not an instability. (It is also *not a pole*: the :math:`[1/1]`
  denominator vanishes at :math:`\tau = -2`, which no physical cell can
  reach. The approximant stays bounded for all :math:`\tau > 0`; it
  merely crosses zero.)
- **Characteristics integrate the exponential exactly** — no zero, no
  pole, positivity for free, at the cost of carrying track-based
  geometry :cite:`Askew1972`.
- **Positivity-versus-order is a property of the table position, not a
  ladder.** The :math:`[2/2]` entry
  :math:`(1 - \tau/2 + \tau^2/12)/(1 + \tau/2 + \tau^2/12)` is
  *fourth*-order and **positive for every** :math:`\tau > 0` — its
  numerator's discriminant is :math:`1/4 - 1/3 < 0`, so it never crosses
  zero. Higher order does not intrinsically cost sign preservation;
  *which entry the closure realizes* does. (The classical scheme family
  simply never lands on :math:`[2/2]`: a two-point closure has one
  transmission ratio to spend, and diamond spends it on :math:`[1/1]`.)

This is the precise sense in which S\ :sub:`N`-with-a-closure and MoC are
one method family: **the closures are rational approximants of the one
propagator that characteristics integrate exactly.** The choice among
step, diamond and characteristics is a *choice of Padé entry* — order
against sign structure against geometric cost — made cell by cell against
the same :math:`e^{-\tau}`. The Larsen–Morel review's observation that
1-D step characteristics is itself expressible as a
:term:`weighted-diamond <weighted diamond difference>` scheme
:cite:`LarsenMorel2010` is this section's claim seen from inside
the S\ :sub:`N` family: one parametrized transmission, sliding along the
table.

The deterministic-grid form is not the universal form
-----------------------------------------------------

The operator algebra :math:`A = L + C - S - N_{2n} - B`
(:doc:`/theory/foundations/operator_algebra`) is the shape this frame
takes **on a deterministic angular–spatial grid**. Two families refuse
it, and their refusals are instructive rather than embarrassing:

- **Collision probability folds the boundary into its kernel.** CP's
  white-boundary re-entry closes *in closed form*: the production solver
  computes
  :math:`P_\infty = P_{\mathrm{cell}} + P_{\mathrm{out}} \otimes
  P_{\mathrm{in}} / (1 - P_{\mathrm{inout}})`
  (``orpheus/cp/solver.py``) — the Sherman–Morrison rank-1 update, in
  longhand, of the no-re-entry kernel. There is no separate :math:`B`
  operator in CP's posing; the boundary *is* a low-rank correction baked
  into :math:`(L+C)^{-1}` itself. The operator algebra names this the
  :ref:`low-rank exception <smw-low-rank-exception>`, and it cuts both
  ways: **CP solves in closed form the boundary cycle that S**\
  :sub:`N` **source iteration iterates**, and porting that Woodbury
  closure to the S\ :sub:`N` boundary — two sweeps and one scalar
  division per :term:`white <white boundary condition>`/albedo
  face, zero boundary iterations, the classical
  response-matrix move — is an open seam tracked as Issue #300.
- **Case / F**\ :sub:`N` **carry no separate** :math:`C`, :math:`S`,
  :math:`F` **at all.** The singular-eigenfunction machinery
  diagonalizes the full :math:`L + C - S` in one stroke; the scattering
  ratio :math:`c` enters the dispersion function :math:`\Lambda(z)` as a
  *number*, and no operator corresponding to :math:`S` is ever built or
  iterated :cite:`Case1960,CaseZweifel1967`.

Even inside the deterministic-grid form, the boundary's position is
subtler than the notation suggests: on the extended space the full
streaming operator and the boundary operator **occupy the same
trace–trace block with complementary triangle structure** —
:eq:`bc-extraction-block-matrix` and :eq:`bc-extraction-trace-blocks` —
with :math:`L_{\mathrm{full}}`'s unit-lower-triangular trace block
carrying the identity that makes the whole within-group operator
triangular (load-bearing twice over, Issue #298). The
:doc:`boundary-conditions page </theory/foundations/boundary_conditions>`
owns that story; this page needs only its conclusion: **where a method
puts the boundary is part of where the method lands.**

The sweep's existence is a theorem, certified per case
------------------------------------------------------

For the DAG-realizing corner of the map (S\ :sub:`N`), "the propagator is
a triangular solve" is not a slogan — it is a *theorem about a triple*,
and ORPHEUS treats it as one:

- **The theorem.** For a fixed ordinate on a structured Cartesian mesh
  with a Cartesian closure, the cell-dependency digraph is acyclic by a
  lattice product-order argument:
  :math:`\operatorname{sign}(\Omega_x)\, i +
  \operatorname{sign}(\Omega_y)\, j` is a strict potential that every
  dependency edge increases. Note what the proof uses: the **mesh's**
  lattice order — not the characteristics. It is a mesh theorem, and it
  is exactly as strong as its hypotheses.
- **The certification.** Acyclicity is a property of the (mesh, closure,
  boundary) **triple**, and ORPHEUS certifies it *per case* rather than
  assuming it: the assembly-mode gates
  (``tests/sn/sweep/test_assembly_mode.py``) assert the strictly-upper
  triangle of the assembled operator is exactly zero under the sweep
  ordering, and that the LAPACK dense solve and the sweep agree to
  :math:`\sim 6 \times 10^{-16}` — the sweep *is* the direct solve, to
  machine precision.
- **The falsification on record.** The certificate has teeth because it
  has fired: Issue #282 records a defensible-looking closure whose
  coupling **broke** acyclicity — a cold residual of :math:`5.2 \times
  10^{5}` in the operator-level probes, with the seed iteration
  diverging geometrically — exactly the back-edge the triangularity
  certification exists to catch. A theorem about a triple can be false
  for a new triple; certify, don't assume.
- **Reflection does not automatically force extraction.** The folklore
  "a reflective boundary creates a cycle, so the reflected coupling must
  be extracted from the sweep" is *false as stated*: ORPHEUS keeps the
  curvilinear :math:`r = 0` pole mirror — a specular
  :term:`reflective <reflective boundary condition>` coupling — **inside
  the walk** as a forward edge, because the sweep order visits
  :math:`\mu < 0` before :math:`\mu > 0` and the mirror feeds
  information only downstream
  (``orpheus/sn/loss_representation``, certified lower-triangular). A
  *single* reflecting face is acyclic; a cycle needs a *closed loop* —
  e.g. both faces of a slab reflecting. The honest extraction criterion
  is therefore not a boolean on the boundary type but an **SCC
  decomposition** of the (face, ordinate) dependency digraph: extract
  exactly the strongly connected components, sweep everything else.

  That criterion is now **executable**:
  :mod:`orpheus.derivations.discrete.sn.sweep_acyclicity` builds the
  trace digraph and computes its SCCs, and
  ``tests/sn/sweep/test_sweep_acyclicity.py`` gates the verdicts —
  ``reflective|vacuum`` and ``vacuum|reflective`` acyclic,
  ``reflective|reflective`` two mirror-pair SCCs with the closing edges
  named, ``periodic`` cyclic from a single law. The same module records
  a distinction easy to lose: acyclicity says *some* one-pass order
  exists, while triangularity is a property of an (operator, **order**)
  pair — a left-reflecting slab is one-pass in the :math:`\mu<0`-first
  order, a right-reflecting one needs :math:`\mu>0` first. (The
  S\ :sub:`N` grand report *proposes* names for the components —
  ``SweepStrongComponent``, ``ReflectiveSweepCycle`` — as a design
  direction; ORPHEUS does not yet ship them as production types.)

The object is one; the operators are many
-----------------------------------------

This page's thesis must end by paying its sharpest debt. Elsewhere in
this corpus stand statements that *sound* like its negation — the
reference-solver pages assert, of the Peierls–Nyström and
trajectory-resolvent architectures, that "the two architectures are
**not** different discretisations of the same operator"
(:doc:`/theory/references/peierls`,
:doc:`/theory/references/trajectory_resolvent`), and the
:ref:`three-meanings taxonomy <reference-solvers-three-meanings>` insists
that three different objects all called "the Green's function" must
never be conflated. **Those statements are exact, and this page changes
nothing about them**, because they are statements about **operators**
while the thesis is a statement about the **object**:

- The *object* — the first moment of the branching process,
  :eq:`path-integral-track-length-moment` — is one. Every method, every
  reference solver, every kernel on those pages computes it.
- The *operators* are many, **necessarily**: a generator splitting
  (Section :ref:`4 <path-integral-generator-splitting>`), an angular
  representation, a choice of kernel (angle-integrated versus
  angle-resolved) each produce a *different operator equation* for the
  same object, on a different function space, with different spectra,
  different singularity structure, different discretization behaviour.
  Two architectures targeting different operators that "share the same
  physical content" is not a qualification of this page's thesis — it
  is this page's thesis, stated at the operator level.

The slogan form: **"five discretizations of one object" is true; "five
discretizations of one operator" is false** — and the corpus's
reference-solver pages are precisely where the false version would have
done damage, which is why they deny it. The three meanings of "Green's
function" survive untouched as three *reference-kernel realizations* of
the object, and the map of this section is where each realization
lands.


.. _path-integral-eigenvalue:

7. Posing the eigenvalue problem
================================

The multiplication eigenvalue (:math:`k`) and the time eigenvalue
(:math:`\alpha`) are properties of the **operator**, posed *before any
discretization* — so every method inherits the same posing, and no method
owns it. The precedent is Bell & Glasstone :cite:`BellGlasstone1970`, who
pose criticality in their opening chapter, on the continuous operator,
before a single discretization appears — alone in the canon in doing so.
This corpus follows them structurally: the posing lives here, at the
root, and the method chapters inherit it.

Where the path reading ends
---------------------------

This section is also where the path reading of Section
:ref:`1 <path-integral-one-object>` **meets its honest limit and yields
to a spectral statement**. Group the histories by *fission generation*:
let :math:`A = L + C - S - N_{2n} - B` be the loss operator and :math:`F`
the
fission operator, so :math:`A^{-1} F` — the **mean-offspring operator**,
ORPHEUS's multiplication operator :math:`K` — maps one generation's
fission source to the next generation's. The all-generations flux is the
generation series

.. math::
   :label: path-integral-generation-series

   \psi
   \;=\;
   \sum_{n=0}^{\infty} \bigl(A^{-1} F\bigr)^{n}\, A^{-1} q,
   \qquad
   \rho\bigl(A^{-1} F\bigr) \;=\; k,

.. (vv-status rationale) Mathematical identity: the fission-generation Neumann
   series and the statement that its convergence radius IS the multiplication
   factor. Its terminal result k = ρ(A⁻¹F) is the ``matrix-eigenvalue`` claim
   verified downstream (:doc:`infinite_medium`). A derivation identity, not a
   separate solver claim.
.. vv-status: path-integral-generation-series documented

and its convergence is governed by the spectral radius of the
mean-offspring operator — which **is** the effective multiplication
factor. For a subcritical medium (:math:`k < 1`) the series converges and
the path reading holds to the last term: the flux *is* the sum over all
histories, fission branches included. For a supercritical medium
(:math:`k > 1`) **the naive sum over histories diverges** — each fission
generation contributes more than its predecessor, and no rearrangement
saves it. The sum-over-histories reading of a supercritical steady state
is *false*, not merely awkward.

The eigenvalue posing is exactly the rescue, and seeing it as such
explains its slightly odd shape. Dividing the fission term by :math:`k`
— posing :math:`A\psi = \frac{1}{k} F \psi` — rescales the mean
offspring per generation to exactly one, which is precisely the
condition for the generation series to sit on its boundary of
summability. But :math:`k` is the spectral radius itself, **not known a
priori** — so the "rescaled sum" cannot be evaluated as a sum at all; it
must be *posed as an eigenproblem* and solved for the pair
:math:`(k, \psi)` together. The :math:`1/k` in the k-eigenvalue equation
is not a modelling convention: it is the unique rescaling that makes the
history sum summable, discovered as an eigenvalue *because* it is not
known in advance.

What replaces the path reading at this boundary is positive-cone
spectral theory: **Krein–Rutman** — the infinite-dimensional
Perron–Frobenius theorem — applied to the mean-offspring operator. The
fission and scattering kernels are positivity-preserving, and
:math:`A^{-1}` inherits positivity from the sub-stochasticity of Section
:ref:`4 <path-integral-generator-splitting>`; Krein–Rutman then delivers
what every criticality solver silently relies on: the dominant
eigenvalue of :math:`A^{-1}F` is **real and positive**, its eigenvector
is the **unique non-negative mode** (the physically meaningful flux),
and all higher harmonics change sign somewhere in phase space. That
statement — not any path sum — is what the power iteration converges on.

The two eigenvalues, one posing table
-------------------------------------

ORPHEUS realizes the posing exactly at this level of generality. The
canonical engine (:mod:`orpheus.numerics.eigenvalue`) poses the
**generalized eigenproblem**
:math:`A_{\mathrm{loss}}\, \psi = \lambda\, M \psi` and solves it by
power iteration on the resolvent :math:`A_{\mathrm{loss}}^{-1} M`, whose
dominant eigenvalue :math:`\mu = 1/\lambda` is what the iteration
delivers; a *posing* is a row of assignments:

- **k-eigenvalue**: :math:`A_{\mathrm{loss}} = L + C - S - N_{2n} - B`,
  :math:`M = F`, :math:`k = \mu` — the multiplication factor *is* the
  resolvent's dominant eigenvalue, :math:`\rho(A^{-1}F)`, consistent
  with :eq:`path-integral-generation-series` (the pencil eigenvalue is
  its reciprocal, :math:`\lambda = 1/k`, matching the
  :math:`A\psi = \frac{1}{k}F\psi` posing above). The **static**
  eigenvalue: fission is rescaled, time never appears. :math:`k` answers
  a bookkeeping question — *by what factor must fission be diluted for a
  steady state to exist?*
- **α-eigenvalue**: :math:`A_{\mathrm{loss}} = L + C - S - N_{2n} - F - B`,
  :math:`M = 1/v`, :math:`\alpha = -1/\mu`. The **dynamic** eigenvalue:
  the full prompt operator is kept and the spectrum of the *free
  evolution* is asked for — :math:`\alpha` is the growth rate of the
  slowest-decaying (or fastest-growing) mode,
  :math:`\psi \sim e^{\alpha t}`. In branching-process language
  :math:`\alpha` is the **Malthusian parameter** of the neutron
  population — the same object that names exponential growth in every
  branching model — and its spectrum for multiplying slabs and spheres
  is classical territory :cite:`DahlSjostrand1979`. (In the engine this
  row is a *documented future seam* — the posing is stated, the solver
  wiring is not yet built.)

The two agree exactly at criticality — :math:`k = 1 \Leftrightarrow
\alpha = 0` — and answer different questions away from it: :math:`k`
compares generations, :math:`\alpha` compares instants. Both are posed
on the operator, so both are *method-portable by construction*: the
power iteration sees only a normalized-source fixed-point procedure over
an abstract resolvent, and each deterministic consumer contributes
exactly its **A1 value** (Section :ref:`5 <path-integral-axes>`) as the
inner solve — the S\ :sub:`N` sweep or Krylov solve, CP's ``P_inf``
kernel, diffusion's and the infinite-medium solver's eager LU. One
engine, one posing, one inner resolvent per consumer — the module's own
docstring now states precisely this layering, and the eigenvalue
chapters of the method books derive their solvers from this section
rather than re-posing the problem.

That closes the loop the page opened: the object is one; the invariant
operators carry the physics; the propagator realization, the generator
splitting and the angular representation locate each method; and the
eigenvalue — the question the whole discipline exists to answer — is
posed once, above all of them.


Development history
===================

.. dropdown:: Changelog — how this page reached its current form
   :color: muted

   - **Phase C′ (2026-07)** — created as a labelled scaffold: seven
     anchored sections with one-paragraph synopses, machine header, and
     stable anchors consumed by the methods entry, the S\ :sub:`N` slab
     chapter and the foundations index (Issue #231; corpus plan §3.6).
   - **Phase H audit (2026-07-22)** — the corpus-plan §3.6 specification
     was itself proved/falsified claim-by-claim before writing (user
     directive). One falsification: the scaffold's "negative flux at
     :math:`\tau > 2` is the pole of the :math:`[1/1]` Padé" — the pole
     sits at :math:`\tau = -2`, unreachable; the negativity is the
     **numerator zero** at :math:`\tau = 2`. The corrected sign story
     became the Padé-table treatment of Section 6. Sharpenings from the
     same audit: the three axes stated as *independent with a partially
     populated product* (not bare "orthogonal"); the delta-tracking
     bridge stated as the explicit Radon–Nikodym derivative; the
     one-condition convergence theorem stated with its positivity
     hypothesis and the :math:`(n,2n)` violation criterion; the
     object-versus-operator reconciliation with the reference-solver
     pages stated as a two-level distinction (Section 6), with no
     retirement of the operator-level denials.
   - **Phase H authoring (2026-07-22)** — full authoring from the audit
     contract, including the kinetic-theory ledger of Section 1 (the
     two-species reduction, the three switches, and the
     thermal-recovery / multiphysics-feedback split of the two collision
     faces — user contribution, audited).
