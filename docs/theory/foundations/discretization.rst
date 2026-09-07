.. _theory-discretization:

=================================================
Discretization: The Cell Balance and Its Closures
=================================================

.. contents:: Contents
   :local:
   :depth: 3


.. Machine header — the ``nexus-meta`` schema for this page.  Ingestion is
.. PENDING nexus#1 Phase 2: the ``nexus-meta`` directive is NOT yet
.. registered, so the schema is rendered here as a collapsed sphinx-design
.. dropdown and machine-consumed later.

.. dropdown:: Machine header — ``nexus-meta`` schema (invariant · pipeline · closures · conventions)
   :color: muted

   .. code-block:: yaml

      module: transport
      concept: discretization
      aliases: [cell balance, closure scheme, spatial closure, Step, Diamond Difference, DD, Linear Discontinuous, LD, upwind, central, box, Keller box, linear upwind, weighted diamond, WDD]
      governing_invariant: "sinks = sources  (neutron balance over any control volume)"
      pipeline: "continuous → semi-discrete (discretize ONE axis) → fully-discrete (cell balance + a closure)"
      cell_balance: "|μ|(ψ_out − ψ_in) + Σ_t h ψ̄ = q h    [ net face outflow + collision loss = source ]"
      closure_relation: "ψ̄ = (1−w)·ψ_in + w·ψ_out    [ w = the cell-average blend weight ]"
      closures:
        step:                "upwind / piecewise-constant;   w = 1;         O(h);   unconditionally positive; diffusive"
        diamond_difference:  "central / Keller box;          w = ½;         O(h²);  NOT positivity-preserving (negative flux in thick cells)"
        linear_discontinuous: "DG-P1 / linear upwind (2 moments); w = 1/(1+k) ∈ [½,1]; O(h²); diffusion-limit consistent (slope source threaded)"
      dimension_agnostic: "the cell may be a SPATIAL interval OR an ANGULAR interval (curvilinear redistribution) — SAME balance+closure"
      key_types: [DiscretizationScheme, DiamondDifference, LinearDiscontinuous, CellVisit, UpstreamState, CellResult]
      algebra_of_record: orpheus.derivations.discrete.sn.balance
      depends_on: [operator_algebra]
      consumed_by: [sn, cp, moc, diffusion]
      verification: [foundation, L1]     # SymPy algebra-of-record (foundation) + MMS convergence-order (L1)


.. _discretization-synopsis:

Synopsis
========

Every deterministic transport solver answers the same structural question:
**how does a continuous conservation law become a finite system of algebraic
equations that a computer can solve?** This chapter derives that pipeline
**once**, generically, so that the :term:`discrete-ordinates <ordinate>`, collision-probability,
and method-of-characteristics chapters can *apply* it rather than re-derive it.

The spine is a single principle: **a problem is posed by naming what it holds
invariant.** For the steady-state transport equation the invariant is
**sinks = sources** — neutron balance over any control volume. Discretization
is nothing more than *enforcing that same invariant on a finite structure*, and
re-posing a scheme (a new closure, a new geometry, a new axis) is *enforcing the
same invariant differently*. Three moves realize it:

- **Continuous → semi-discrete → fully-discrete.** Start pointwise in the full
  phase space :math:`(\mathbf r,\hat\Omega,E)`; discretize one axis at a time
  (angle → ordinates, energy → groups, space → cells); at each step the
  invariant migrates from "holds at every point" to "holds for every discrete
  direction / group / cell".
- **The cell balance.** Integrating the transport equation over one cell gives
  *(net face outflow) + (collision loss) = (source)* — the invariant made
  discrete. It is **one equation relating the two face fluxes to the cell-average
  flux**, hence has more unknowns than equations, and therefore **needs a
  closure**: a face ↔ average relation.
- **The closure schemes.** **Step** (upwind), **Diamond Difference** (central /
  box), and **Linear Discontinuous** (linear upwind) are three ways to supply
  that missing relation. They form a spectrum of in-cell flux representations —
  piecewise-constant → linear-via-shared-faces → full-linear — parameterized by
  a single **cell-average blend weight** :math:`w`.

The final, unifying observation is that the cell-balance-plus-closure is
**dimension-agnostic**: the "cell" may be a spatial interval *or* an angular
interval. In curvilinear :math:`S_N` the angular redistribution is literally the
same :term:`weighted-diamond <weighted diamond difference>` closure applied to an *angular* cell — space and angle are
discretized by one and the same machine. The chapter closes by connecting the
discretized balance to the operator algebra: the assembled cell balance **is**
the loss operator :math:`L+C`, and the closure is what fixes its matrix
structure (and, for the diamond family, what makes it invertible by a :term:`sweep`).

The derivations here are the presentation layer of the SymPy
**algebra-of-record** in
:mod:`orpheus.derivations.discrete.sn.balance`, which verifies each identity
symbolically; the concrete schemes live in
:mod:`orpheus.transport.spatial` — method-generic by construction, consumed by
:math:`S_N` first but owned by no method.

.. admonition:: Notation

   Throughout, :math:`q` is the **volumetric source** on a cell (the SymPy
   module writes it :math:`S`; we reserve :math:`S` for the scattering
   operator, per :doc:`/theory/foundations/operator_algebra`). In a real solve,
   :math:`q` gathers the in-scatter, fission, and external sources for one
   ordinate/group. :math:`|\mu|` is the **streaming coefficient** (the
   magnitude of the direction cosine along the swept axis — a "wave speed" in
   the advection–reaction reading below); :math:`\Sigma_t` the total /
   removal cross section (the "reaction rate"); :math:`h` the cell width;
   :math:`\bar\psi` the cell-average flux; :math:`\psi_{\rm in}`,
   :math:`\psi_{\rm out}` the upstream (inflow) and downstream (outflow) face
   fluxes. The sweep direction is **pre-resolved**, so "in" always means
   upstream and "out" always means downstream, in every geometry.


.. _discretization-invariant:

1. Pose by the invariant
========================

Before any numerics, a mathematical problem must be *posed*, and the act of
posing is the act of **naming what the problem holds invariant**. A conservation
law is precisely an invariant: some quantity, integrated over any control
volume, is neither created nor destroyed except by named sources and sinks.

For steady-state neutron transport the invariant is **balance**:

.. math::
   :label: discretization-invariant-eq

   \underbrace{\text{(neutrons leaving a control volume)}}_{\text{sinks}}
   \;=\;
   \underbrace{\text{(neutrons entering it + produced in it)}}_{\text{sources}}.

.. vv-status: discretization-invariant-eq documented
.. (vv-status rationale) discretization-invariant-eq states the governing
   conservation principle (sinks = sources) that every discrete balance in this
   chapter enforces; it is a definitional/structural statement, not a solver
   claim, so it lands in the matrix's Documented-only bucket. Its discrete
   realizations (:eq:`discretization-cell-balance-eq`,
   :eq:`discretization-curvilinear-balance`) are pinned by the SymPy
   algebra-of-record :mod:`orpheus.derivations.discrete.sn.balance`.

Everything that follows is a way of enforcing :eq:`discretization-invariant-eq`
on progressively coarser structures:

- **Continuous.** The invariant holds *pointwise*: at every point of phase
  space, the local streaming plus collision losses equal the local source.
- **Semi-discrete.** Discretize one axis (say angle). The invariant now holds
  *per discrete direction*: sinks equal sources for each ordinate.
- **Fully-discrete.** Close the balance on the last axis. The invariant becomes
  a *discrete cell equation*: sinks equal sources for each cell — a finite
  algebraic system.

This is not a slogan; it is an operational lever. **Re-posing** a construct
means naming the invariant it enforces and then enforcing *that same invariant*
by a different device.

.. tip:: **The generalization lever (used in** :ref:`discretization-reposing`
   **).** A construct that today is tied to one scheme is generalized by naming
   the invariant it represents and re-posing that invariant for the other
   schemes. The worked target is **radial characteristics** in curvilinear
   :math:`S_N`, which is written today for the Diamond-Difference closure; once
   we identify the invariant it enforces (the *angular-redistribution balance*),
   the identical condition can be posed for Step or Linear Discontinuous. Naming
   the invariant is what makes a scheme-specific construct method-generic.

The remainder of the chapter is the invariant :eq:`discretization-invariant-eq`,
made discrete and then closed.


.. _discretization-continuous-semi-full:

2. Continuous, semi-discrete, fully-discrete
============================================

The passage from a continuous equation to an algebraic system happens one axis
at a time. Phase space has three axes — space :math:`\mathbf r`, angle
:math:`\hat\Omega`, energy :math:`E` — and each is discretized by *enforcing the
balance invariant on the newly-discrete structure*. **Which** axis goes first is
a method choice; the **pipeline** is method-invariant.

Continuous — the pointwise invariant
------------------------------------

The steady-state linear Boltzmann (transport) equation states balance at every
point of phase space [Duderstadt & Hamilton 1976; Lewis & Miller 1984]:

.. math::
   :label: discretization-continuous

   \underbrace{\hat\Omega\cdot\nabla\psi(\mathbf r,\hat\Omega,E)}_{\text{streaming}}
   \;+\;
   \underbrace{\Sigma_t(\mathbf r,E)\,\psi(\mathbf r,\hat\Omega,E)}_{\text{collision loss}}
   \;=\;
   \underbrace{q(\mathbf r,\hat\Omega,E)}_{\text{source}},

.. vv-status: discretization-continuous documented
.. (vv-status rationale) literature-transcribed governing equation (the
   integro-differential transport equation), definitional not a solver claim.

where the source

.. math::
   :label: discretization-source

   q(\mathbf r,\hat\Omega,E)
   = \int_0^\infty\!\!\int_{4\pi}
     \Sigma_s(\mathbf r,E'\!\to\!E,\hat\Omega'\!\to\!\hat\Omega)\,
     \psi(\mathbf r,\hat\Omega',E')\,d\Omega'\,dE'
   \;+\;
   \frac{\chi(E)}{4\pi\,k}\int_0^\infty \nu\Sigma_f(\mathbf r,E')\,\phi(\mathbf r,E')\,dE'

.. vv-status: discretization-source documented
.. (vv-status rationale) literature-transcribed definition of the transport
   source (scattering + fission); definitional, not a solver claim.

collects in-scatter and fission. The **streaming** term
:math:`\hat\Omega\cdot\nabla\psi` is the **leakage** along the direction of
flight and the **collision** term :math:`\Sigma_t\psi` the local removal rate —
the two "sink" contributions. (What :math:`\hat\Omega\cdot\nabla\psi` *means* —
the directional derivative along the neutron's ray, equivalently the divergence
of the angular current :math:`\hat\Omega\psi`, and the proof the two readings
agree — is derived once at the :doc:`transport-methods entry
</theory/methods/index>`, upstream of every method; this page takes the
continuous equation as given and discretizes it.) Equation
:eq:`discretization-continuous` *is* :eq:`discretization-invariant-eq` written
differentially: streaming (net leakage) + collision (removal) = source. It holds
at every :math:`(\mathbf r,\hat\Omega,E)`.

Semi-discrete — the per-direction (or per-group) invariant
----------------------------------------------------------

Discretize **one** axis while leaving the others continuous. Two canonical
choices:

**Angle → discrete ordinates** (the :math:`S_N` move). Choose a :term:`quadrature` set
:math:`\{(\hat\Omega_m, w_m)\}_{m=1}^{M}` on the unit sphere and require
:eq:`discretization-continuous` to hold along each direction
:math:`\hat\Omega_m`:

.. math::
   :label: discretization-semidiscrete-angle

   \hat\Omega_m\cdot\nabla\psi_m(\mathbf r,E)
   + \Sigma_t(\mathbf r,E)\,\psi_m(\mathbf r,E)
   = q_m(\mathbf r,E),
   \qquad m = 1,\dots,M,

.. vv-status: discretization-semidiscrete-angle documented
.. (vv-status rationale) the discrete-ordinates semi-discretization step;
   structural/definitional, not a solver claim.

with :math:`\psi_m(\mathbf r,E)\equiv\psi(\mathbf r,\hat\Omega_m,E)` and the
scattering integral in :eq:`discretization-source` replaced by a quadrature sum
:math:`\int_{4\pi}(\cdot)\,d\Omega'\approx\sum_{m'} w_{m'}(\cdot)`. The result is
a **coupled set of** :math:`M` **equations, each still continuous in**
:math:`(\mathbf r,E)`. The invariant has migrated: instead of "balance at every
angle", it is now "balance for each of the :math:`M` discrete directions". This
is the defining feature of :math:`S_N` — it **retains the angular flux**
:math:`\psi_m` (contrast the collision-probability method, which integrates
angle out into a scalar-flux integral equation).

**Energy → multigroup.** Independently, partition the energy axis into groups
:math:`\{[E_g, E_{g-1}]\}_{g=1}^{G}` and integrate
:eq:`discretization-continuous` over each group:

.. math::
   :label: discretization-semidiscrete-energy

   \hat\Omega\cdot\nabla\psi_g(\mathbf r,\hat\Omega)
   + \Sigma_{t,g}(\mathbf r)\,\psi_g(\mathbf r,\hat\Omega)
   = q_g(\mathbf r,\hat\Omega),
   \qquad g = 1,\dots,G.

.. vv-status: discretization-semidiscrete-energy documented
.. (vv-status rationale) the multigroup semi-discretization step;
   structural/definitional, not a solver claim. The multigroup balance is
   the subject of :ref:`theory-cross-section-data` and :ref:`theory-homogeneous`.

Energy is *always* grouped in deterministic transport, so
:eq:`discretization-semidiscrete-energy` is a background move under every
method; the group-transfer :math:`\Sigma_s(E'\!\to\!E)\to\Sigma_{s,g'\to g}` and
the fission spectrum :math:`\chi_g` are treated in
:doc:`/theory/foundations/cross_section_data`. What matters here is the *shape*:
after semi-discretizing, the invariant is a per-ordinate (and per-group) balance,
still continuous in space.

Fully-discrete — the cell invariant
-----------------------------------

One axis remains — space. Integrate the semi-discrete equation
:eq:`discretization-semidiscrete-angle` over each spatial **cell** (control
volume). This produces the **cell balance** (:ref:`discretization-cell-balance`),
which relates face fluxes to the cell average and therefore requires a
**closure** (:ref:`discretization-closures`). Once the closure is chosen, the
per-ordinate, per-group, per-cell equations assemble into a finite linear system

.. math::
   :label: discretization-fully-discrete

   \mathbf A\,\boldsymbol\psi = \mathbf q,

.. vv-status: discretization-fully-discrete documented
.. (vv-status rationale) the fully-discrete algebraic system; the operator
   A = L + C − S − N_2n − B and its posing are the subject of
   :ref:`operator-algebra`. Structural, not a solver claim.

where :math:`\mathbf A` is the discretized loss operator and
:math:`\boldsymbol\psi` the vector of unknown fluxes (cell-averaged, per
ordinate, per group). The invariant :eq:`discretization-invariant-eq` is now a
row of :eq:`discretization-fully-discrete`: sinks equal sources for each cell.
The bridge from :eq:`discretization-fully-discrete` back to the operator algebra
:math:`\mathbf A = L + C - S - N_{2n} - B` is
:ref:`discretization-operator-bridge`.

.. note:: **The pipeline is method-invariant; the ordering is a method choice.**
   :math:`S_N` discretizes angle into ordinates and space into cells;
   collision-probability integrates angle out analytically and discretizes space
   into flat-source regions; method-of-characteristics discretizes angle into
   ordinates *and* traces along continuous rays. Each collapses the axes in a
   different order and with different closures — but each is the same three
   moves: pointwise invariant → per-structure invariant → discrete cell
   invariant. This is why the cell balance and its closures are **foundation**
   material, shared across all of :mod:`orpheus.transport`.


.. _discretization-cell-balance:

3. The cell balance — the invariant made discrete
=================================================

Take one ordinate of the semi-discrete equation
:eq:`discretization-semidiscrete-angle` and, to isolate the essential structure,
reduce to **one spatial dimension** — a generic cell :math:`i` occupying
:math:`x\in[x_{i-1/2}, x_{i+1/2}]` of width :math:`h`, with streaming
coefficient :math:`\mu` along the axis:

.. math::
   :label: discretization-1d-transport

   \mu\,\frac{\partial\psi}{\partial x}(x) + \Sigma_t\,\psi(x) = q(x).

.. vv-status: discretization-1d-transport documented
.. (vv-status rationale) the 1-D per-ordinate transport equation; the starting
   point of the SymPy derivation derive_cartesian_1d in the algebra-of-record.

Integrate over the cell, :math:`\int_{x_{i-1/2}}^{x_{i+1/2}}(\cdot)\,dx`. The
streaming term integrates exactly by the fundamental theorem of calculus, the
collision term produces the cell-average flux
:math:`\bar\psi\equiv\frac1h\int\psi\,dx`, and the source produces
:math:`\bar q\,h`:

.. math::
   :label: discretization-cell-balance-eq

   \underbrace{|\mu|\,\bigl(\psi_{\rm out} - \psi_{\rm in}\bigr)}_{\text{net face outflow (streaming)}}
   \;+\;
   \underbrace{\Sigma_t\,h\,\bar\psi}_{\text{collision loss}}
   \;=\;
   \underbrace{q\,h}_{\text{source in cell}}.

.. vv-status: discretization-cell-balance-eq documented
.. (vv-status rationale) the discrete cell-balance equation — the invariant
   sinks=sources made discrete on one cell. Derived and pinned by SymPy in
   :func:`orpheus.derivations.discrete.sn.balance.derive_cartesian_1d` (the
   balance line ``μ(ψ_out − ψ_in) + Σ_t Δx ψ_avg = S Δx``, face areas = 1 in
   slab), a foundation-level algebraic identity, not a solver claim.

Here :math:`\psi_{\rm in} = \psi(x_{i-1/2})` and
:math:`\psi_{\rm out} = \psi(x_{i+1/2})` are the **face fluxes** (the sweep
pre-resolves which physical face is upstream, so :math:`\psi_{\rm in}` is always
the inflow and :math:`\psi_{\rm out}` the outflow, with the sign of :math:`\mu`
absorbed into :math:`|\mu|`). Equation :eq:`discretization-cell-balance-eq` is
*exactly* :eq:`discretization-invariant-eq` for this control volume: **net
face outflow** (streaming across the two faces) **+ collision loss = source in
the cell**. Nothing has been approximated yet — it is the integral of an exact
equation.

The crux: one equation, two unknowns
------------------------------------

The inflow face flux :math:`\psi_{\rm in}` is **known** — it is the outflow of
the upstream cell, delivered by the sweep. That leaves
:eq:`discretization-cell-balance-eq` as **one equation in two unknowns**: the
cell average :math:`\bar\psi` and the outflow face :math:`\psi_{\rm out}`. The
balance alone cannot determine both. This deficit is not a defect of the
derivation — it is intrinsic: the cell balance is a *statement about integrals*
(the average and the fluxes through the boundary), and integrals do not pin the
in-cell *shape*.

.. important:: **The cell balance needs a closure.** The one missing equation is
   a **face ↔ average relation** — a rule that ties the outflow face flux
   :math:`\psi_{\rm out}` to the cell average :math:`\bar\psi` (and the known
   inflow :math:`\psi_{\rm in}`). Supplying that rule is choosing an assumed
   *in-cell flux shape*. Every difference scheme — Step, Diamond Difference,
   Linear Discontinuous, and their weighted variants — **is** a choice of this
   closure, and nothing else. This is the single most important structural fact
   in deterministic transport discretization.

Dividing :eq:`discretization-cell-balance-eq` through by :math:`h` and writing
:math:`g \equiv |\mu|/h` (the **streaming-over-volume**, a rate) casts the
balance in the form the closures will complete:

.. math::
   :label: discretization-cell-balance-divided

   g\,\bigl(\psi_{\rm out} - \psi_{\rm in}\bigr) + \Sigma_t\,\bar\psi = q .

.. vv-status: discretization-cell-balance-divided documented
.. (vv-status rationale) the volume-normalized cell balance; algebraic restatement
   of :eq:`discretization-cell-balance-eq`, foundation-level, not a solver claim.

.. note:: **Dimension-agnostic by construction.** The derivation used only that
   a cell is a control volume with an inflow face, an outflow face, a cell
   average, a reaction rate, and a source. It made **no** use of the axis being
   *space*. In :math:`d\ge2` spatial dimensions the same integral produces one
   net-outflow term per axis (:math:`\sum_a |\mu_a|(\psi^a_{\rm out} -
   \psi^a_{\rm in})`); in curvilinear geometry it produces an additional
   *angular* net-outflow term (:ref:`discretization-space-angle`). The
   structure — *net face outflow + collision = source, one balance needing a
   face ↔ average closure* — is identical in every case. That universality is
   why this chapter exists.


.. _discretization-closures:

4. The closure schemes: Step, DD, LD
====================================

A closure supplies the missing face ↔ average relation of
:eq:`discretization-cell-balance-divided` by positing an **in-cell flux shape**.
The three canonical single-axis closures form a spectrum of increasing
representational richness:

.. list-table:: The three closures, at a glance
   :header-rows: 1
   :widths: 18 26 12 14 30

   * - Scheme (ORPHEUS)
     - Standard name(s)
     - In-cell shape
     - Blend :math:`w`
     - Order / positivity
   * - **Step**
     - upwind, first-order upwind, donor-cell
     - constant
     - :math:`w=1`
     - :math:`O(h)`; **unconditionally positive**; diffusive
   * - **Diamond Difference**
     - central, Keller box, box
     - linear (shared faces)
     - :math:`w=\tfrac12`
     - :math:`O(h^2)`; **can go negative**; dispersive
   * - **Linear Discontinuous**
     - linear upwind, DG-P1-upwind
     - full linear (2 moments)
     - :math:`w=\tfrac1{1+k}`
     - :math:`O(h^2)`; milder negatives; diffusion-limit consistent

The unifying device is the **cell-average blend weight** :math:`w`, defined by

.. math::
   :label: discretization-blend

   \bar\psi = (1-w)\,\psi_{\rm in} + w\,\psi_{\rm out}.

.. vv-status: discretization-blend documented
.. (vv-status rationale) the cell-average blend-weight identity — the universal
   convex face blend that parameterizes every consistent affine closure. Pinned
   by the generic reconstruction staticmethods on
   :class:`orpheus.transport.spatial.scheme.DiscretizationSchemeBase`
   (``cell_average`` / ``outgoing_face_from_average`` / ``source_emission``)
   and their foundation tests; structural, not a solver claim.

**Exactness on a constant flux forces the two weights to sum to one.** If
:math:`\psi_{\rm in}=\psi_{\rm out}=\bar\psi` (a spatially flat flux under a
matched source), :eq:`discretization-blend` must reduce to
:math:`\bar\psi=\bar\psi`, which holds for *any* :math:`w` precisely because the
coefficients :math:`(1-w)` and :math:`w` sum to one. So :math:`w` is the **only**
free number in a consistent affine closure, and each scheme is a choice of it.
Inverting :eq:`discretization-blend` gives the outflow reconstruction the sweep
actually applies,

.. math::
   :label: discretization-outflow-reconstruction

   \psi_{\rm out} = \frac{\bar\psi - (1-w)\,\psi_{\rm in}}{w},

.. vv-status: discretization-outflow-reconstruction documented
.. (vv-status rationale) algebraic inverse of :eq:`discretization-blend`;
   the ``outgoing_face_from_average`` reconstruction, foundation-level.

which is implemented once, generically, as
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average`
— every scheme differs only in the value of :math:`w` it supplies.

.. admonition:: The advection–reaction reading (why these are CFD scheme names)

   Equation :eq:`discretization-1d-transport`,
   :math:`\mu\,\partial_x\psi + \Sigma_t\psi = q`, is a one-dimensional
   **linear advection–reaction equation** with wave speed :math:`\mu` and
   reaction rate :math:`\Sigma_t`. Under that reading the three closures are
   textbook finite-volume schemes: **Step = first-order upwind**, **Diamond
   Difference = central / Keller box**, **Linear Discontinuous =
   discontinuous-Galerkin P1 upwind**. The blend weight :math:`w` is the
   CFD **Péclet / κ-scheme blend** between central (:math:`w=\tfrac12`) and
   upwind (:math:`w=1`). This is not an analogy bolted on after the fact — it is
   the same mathematics, and the ORPHEUS scheme layer is written model-agnostic
   in the wave speed and reaction rate so a future diffusion solver reuses the
   identical primitives (:class:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase`,
   the ``reaction_xs`` parameter).


.. _discretization-step:

Step — piecewise-constant upwind
--------------------------------

**The closure.** Step assumes the in-cell flux is **constant**, equal to the
cell average. The outflow face therefore carries the cell's own value:

.. math::
   :label: discretization-step-closure

   \psi(x) \approx \bar\psi
   \quad\Longrightarrow\quad
   \psi_{\rm out} = \bar\psi
   \quad\Longleftrightarrow\quad
   w = 1 .

.. vv-status: discretization-step-closure documented
.. (vv-status rationale) the Step (upwind) closure relation ψ_out = ψ̄, i.e.
   the w=1 case of :eq:`discretization-blend`; definitional, not a solver claim.

Setting :math:`w=1` in :eq:`discretization-blend` gives
:math:`\bar\psi = 0\cdot\psi_{\rm in} + 1\cdot\psi_{\rm out} = \psi_{\rm out}`:
the cell average equals the **outflow** face and **forgets its inflow face**
entirely. This one-sided (donor-cell) construction is what "upwind" means —
information is taken only from the upstream side.

**The solve.** Substitute :math:`\psi_{\rm out}=\bar\psi` into the divided
balance :eq:`discretization-cell-balance-divided`:

.. math::
   :label: discretization-step-solve

   g\,(\bar\psi - \psi_{\rm in}) + \Sigma_t\,\bar\psi = q
   \quad\Longrightarrow\quad
   \boxed{\;\bar\psi = \frac{q + g\,\psi_{\rm in}}{g + \Sigma_t}\;},
   \qquad \psi_{\rm out} = \bar\psi .

.. vv-status: discretization-step-solve documented
.. (vv-status rationale) the Step cell update, solved from
   :eq:`discretization-cell-balance-divided` under :eq:`discretization-step-closure`;
   foundation-level algebra, not a solver claim.

**Positivity.** :math:`\bar\psi` is a ratio of manifestly non-negative
quantities: if :math:`q\ge0` and :math:`\psi_{\rm in}\ge0`, then
:math:`\bar\psi\ge0`, and hence :math:`\psi_{\rm out}=\bar\psi\ge0`. **Step is
unconditionally positivity-preserving** — for *any* cell width, cross section,
or source. This is its defining virtue.

**Accuracy.** Expand the true flux about the cell centre :math:`x_j`:
:math:`\psi_{\rm out}=\psi(x_j)+\tfrac{h}{2}\psi'(x_j)+O(h^2)`. Step sets
:math:`\psi_{\rm out}=\bar\psi=\psi(x_j)+O(h^2)`, discarding the
:math:`\tfrac{h}{2}\psi'` term. The face flux is therefore wrong at
**first order**, :math:`O(h)`. The leading error acts as an artificial
**numerical diffusion** of coefficient :math:`\propto h\,|\mu|` — it smears
gradients, which is why upwind schemes are *diffusive* and why very fine meshes
are needed for accuracy. **Step is exact only on a spatially constant flux**
(the :math:`w`-blend is exact on constants for any :math:`w`; Step adds no more).

.. note:: **ORPHEUS status.** Step is the designed-but-unbuilt arm of the
   spatial-scheme contract (Issue #158). Its traits are anticipated on
   :class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`
   (``is_positivity_preserving = True``,
   ``transverse_coupling_is_facewise = True``, the :math:`w=1` upwind blend), so
   the base docstring's ``class Step`` sketch registers under ``key="step"``
   when implemented. It is documented here on its mathematical merits; the
   diamond and linear-discontinuous arms below are shipped.


.. _discretization-dd:

Diamond Difference — the central / box scheme
---------------------------------------------

**The closure.** :term:`Diamond Difference <diamond difference>` (DD) assumes the in-cell flux is **linear**
and takes the cell average to be the **arithmetic mean of the two faces**:

.. math::
   :label: discretization-dd-closure

   \bar\psi = \tfrac12\bigl(\psi_{\rm in} + \psi_{\rm out}\bigr)
   \quad\Longleftrightarrow\quad
   \psi_{\rm out} = 2\bar\psi - \psi_{\rm in}
   \quad\Longleftrightarrow\quad
   w = \tfrac12 .

.. vv-status: discretization-dd-closure documented
.. (vv-status rationale) the Diamond-Difference closure — the w=½ symmetric
   diamond mean, the case of :eq:`discretization-blend`; definitional. The
   value _DD_W = 0.5 is the single source of truth in
   :mod:`orpheus.transport.spatial.diamond`.

The name is geometric: on the :math:`(x,\psi)` diagram the cell average sits at
the *centre* of the "diamond" whose corners are the two face fluxes. Unlike
Step, DD uses **both** faces symmetrically — it is the central-difference /
Keller-box scheme.

**The solve.** Substitute :math:`\psi_{\rm out}=2\bar\psi-\psi_{\rm in}` into the
divided balance :eq:`discretization-cell-balance-divided`:

.. math::
   :label: discretization-dd-solve

   g\,\bigl((2\bar\psi-\psi_{\rm in}) - \psi_{\rm in}\bigr) + \Sigma_t\bar\psi = q
   \;\Longrightarrow\;
   (2g + \Sigma_t)\,\bar\psi = q + 2g\,\psi_{\rm in}
   \;\Longrightarrow\;
   \boxed{\;\bar\psi = \frac{q + 2g\,\psi_{\rm in}}{2g + \Sigma_t}\;}.

.. vv-status: discretization-dd-solve documented
.. (vv-status rationale) the DD cell update, solved from
   :eq:`discretization-cell-balance-divided` under
   :eq:`discretization-dd-closure`. Pinned by SymPy in
   :func:`orpheus.derivations.discrete.sn.balance.derive_cartesian_1d`
   (``ψ_avg = (S + 2μ/Δx·ψ_in)/(Σ_t + 2μ/Δx)`` with g = μ/Δx), foundation-level.

This is exactly the identity the SymPy algebra-of-record verifies in
:func:`~orpheus.derivations.discrete.sn.balance.derive_cartesian_1d`. Written as
a face-to-face recurrence,
:math:`\psi_{\rm out} = a\,\psi_{\rm in} + b` with
:math:`a=(2g-\Sigma_t)/(2g+\Sigma_t)` and :math:`b=2q/(2g+\Sigma_t)` — the
cumulative-product form the fast 1-D sweep uses
(:func:`~orpheus.derivations.discrete.sn.balance.derive_cumprod_recurrence`).

**Accuracy.** The mean :math:`\tfrac12(\psi_{\rm in}+\psi_{\rm out})` is the
midpoint value of a linear function *exactly*, and one order beyond that for a
smooth flux. Expanding about :math:`x_j`,

.. math::
   :label: discretization-dd-truncation

   \tfrac12\bigl(\psi_{\rm in}+\psi_{\rm out}\bigr)
   = \tfrac12\Bigl[\bigl(\psi_j - \tfrac{h}{2}\psi'_j + \tfrac{h^2}{8}\psi''_j\bigr)
                 + \bigl(\psi_j + \tfrac{h}{2}\psi'_j + \tfrac{h^2}{8}\psi''_j\bigr)\Bigr]
   = \psi_j + \tfrac{h^2}{8}\psi''_j + O(h^4),

.. vv-status: discretization-dd-truncation documented
.. (vv-status rationale) Taylor truncation analysis of the DD closure; a
   pedagogical derivation-decomposition step, not a solver claim.

the odd (:math:`O(h)`) term **cancels by symmetry**, leaving second-order error,
:math:`O(h^2)`. **DD is exact on a linear-in-:math:`x` flux.** It is a genuine
accuracy gain over Step at the same mesh — but the symmetry that buys the extra
order also removes the upwind bias that damps oscillations.

**Positivity — the negative-flux failure mode.** From the reconstruction
:math:`\psi_{\rm out}=2\bar\psi-\psi_{\rm in}`, the outflow goes **negative**
whenever :math:`2\bar\psi<\psi_{\rm in}`. Take the sharpest case — a source-free
cell (:math:`q=0`) with positive inflow. Then
:math:`\bar\psi = 2g\,\psi_{\rm in}/(2g+\Sigma_t)` and

.. math::
   :label: discretization-dd-negative

   \psi_{\rm out}
   = 2\bar\psi - \psi_{\rm in}
   = \psi_{\rm in}\,\frac{2g - \Sigma_t}{2g + \Sigma_t}
   \;<\;0
   \quad\Longleftrightarrow\quad
   \Sigma_t > 2g
   \quad\Longleftrightarrow\quad
   \Sigma_t\,h > 2|\mu| .

.. vv-status: discretization-dd-negative documented
.. (vv-status rationale) the DD negative-flux threshold, derived algebraically
   from the DD reconstruction; the canonical positivity counter-example
   (Lewis & Miller §5.3). Pedagogical, not a solver claim.

So an **optically thick cell** — one whose :term:`optical thickness`
:math:`\Sigma_t h` exceeds :math:`2|\mu|` — produces a negative outflow face
flux from a perfectly physical positive inflow. This is the canonical DD
counter-example (Lewis & Miller 1984, §5.3), and the reason
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` declares
``is_positivity_preserving = False``. It is a *dispersive* error (unphysical
oscillation), the price of the central stencil's lack of upwind damping.

.. _discretization-negative-flux-example:

.. admonition:: Worked contrast — a thick cell (verified against the code)

   Take :math:`\Sigma_t = 1`, :math:`h = 2` (so optical thickness
   :math:`\Sigma_t h = 2`), :math:`|\mu| = \tfrac12` (so :math:`\Sigma_t h = 2 >
   2|\mu| = 1` — thick), zero source, and unit inflow :math:`\psi_{\rm in}=1`.
   Then :math:`g = |\mu|/h = \tfrac14`. The three closures give, evaluated
   against the live implementations
   (:meth:`DiamondDifference.update <orpheus.transport.spatial.diamond.DiamondDifference.update>`,
   :meth:`LinearDiscontinuous.update <orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous.update>`):

   .. list-table::
      :header-rows: 1
      :widths: 24 24 24 28

      * - Scheme
        - :math:`\bar\psi`
        - :math:`\psi_{\rm out}`
        - Verdict
      * - **Step** (:math:`w=1`)
        - :math:`\tfrac15 = 0.200`
        - :math:`\tfrac15 = 0.200`
        - positive ✓
      * - **Diamond Difference** (:math:`w=\tfrac12`)
        - :math:`\tfrac13 = 0.333`
        - :math:`-\tfrac13 = -0.333`
        - **strongly negative** ✗
      * - **Linear Discontinuous** (:math:`w=\tfrac7{10}`)
        - :math:`\tfrac{5}{19} = 0.263`
        - :math:`-\tfrac{1}{19} = -0.053`
        - mildly negative (6.3× milder than DD)

   Step stays positive; DD produces a large negative outflow; Linear
   Discontinuous produces a *much* smaller negative excursion (its upwind bias
   damps the DD oscillation, though it does not eliminate it — see below). The
   fractions are exact and reproduce the code to machine precision.


.. _discretization-ld:

Linear Discontinuous — the two-moment upwind scheme
---------------------------------------------------

**The closure.** Step carries one number per cell (the average) and posits a
constant; DD carries one number and posits a linear whose average is the face
mean. Linear Discontinuous (LD) carries **two** numbers per cell — the average
**and an independent slope** — and represents the in-cell flux as a *full*
linear function [Larsen & Morel 1989, JCP 83(1):212–236, Eqs. (4.1)]:

.. math::
   :label: discretization-ld-moments

   \bar\psi_j = \frac1{h}\!\int\psi\,dx,
   \qquad
   \hat\psi_j = \frac{6}{h^2}\!\int (x-x_j)\,\psi\,dx,
   \qquad
   \psi(x) \approx \bar\psi_j + \frac{2}{h}(x-x_j)\,\hat\psi_j ,

.. vv-status: discretization-ld-moments documented
.. (vv-status rationale) the LD linear representation (cell-average + slope
   moments on the Legendre basis {1, P₁}); literature-transcribed definition
   (Larsen & Morel 1989 Eq. 4.1), not a solver claim.

where :math:`\hat\psi_j` is the (scaled) :math:`P_1` Legendre moment — the
**slope**. Evaluating the linear at the downstream face
(:math:`x = x_j + h/2` in sweep coordinates, where :math:`\tfrac{2}{h}\cdot
\tfrac{h}{2}=1`) gives the outflow reconstruction

.. math::
   :label: discretization-ld-face

   \psi_{\rm out} = \bar\psi + \hat\psi .

.. vv-status: discretization-ld-face documented
.. (vv-status rationale) the LD upwind/discontinuous downstream-face
   reconstruction ψ_out = ψ̄ + ψ̂ (Larsen & Morel 1989 Eq. 4.3c); definitional.

The **"discontinuous"** in the name is the crux: the cell's *inflow* face is
**not** its own linear reconstruction :math:`\bar\psi-\hat\psi`; it is the
upwind neighbour's outflow :math:`\psi_{\rm in}`, taken with **no continuity
enforced** across the face. That upwind coupling is what gives LD the stability
Step has and DD lacks, while the slope moment gives it the accuracy Step lacks.

**The two-moment cell system.** LD requires *two* balance-type equations — the
zeroth (average, :math:`P_0`) moment and the first (slope, :math:`P_1`) moment
of :eq:`discretization-1d-transport` over the cell. With the upwind face closure
:eq:`discretization-ld-face` substituted, and the slope-moment weight
:math:`\theta = \tfrac13` (the :math:`S_N`-exact linear-discontinuous value,
Larsen & Morel 1989 Eq. (4.3b)), the per-cell system is a
:math:`2\times2` linear solve:

.. math::
   :label: discretization-ld-system

   \underbrace{\begin{bmatrix}
     \Sigma_t h + |\mu| & |\mu| \\[3pt]
     -\,|\mu| & \Sigma_t\theta h + |\mu|
   \end{bmatrix}}_{\mathbf A}
   \begin{bmatrix} \bar\psi \\[3pt] \hat\psi \end{bmatrix}
   =
   \begin{bmatrix}
     \bar q\,h + |\mu|\,\psi_{\rm in} \\[3pt]
     \theta\,\hat q\,h - |\mu|\,\psi_{\rm in}
   \end{bmatrix}.

.. vv-status: discretization-ld-system documented
.. (vv-status rationale) the slab-LD 2×2 moment system (Larsen & Morel 1989
   Eqs. 4.3a-b, θ=1/3). This is the exact natural (mass-weighted) matrix the
   SymPy algebra-of-record assembles — :math:`\mathbf A` and its RHS are the
   verbatim output of
   :func:`orpheus.derivations.discrete.sn.ld_ubld.assemble_ubld` /
   :func:`~orpheus.derivations.discrete.sn.ld_ubld.derive_d1_reduction_to_production`
   (which proves it equals the production ``_LDCellTerms`` 2×2 symbolically),
   not a hand-rearranged variant. Structural, not a solver claim.

The top row is the cell balance itself: expanding it,
:math:`(\Sigma_t h+|\mu|)\bar\psi + |\mu|\hat\psi = \bar q h + |\mu|\psi_{\rm in}`,
is :math:`|\mu|(\underbrace{\bar\psi+\hat\psi}_{\psi_{\rm out}}-\psi_{\rm in}) +
\Sigma_t h\,\bar\psi = \bar q h` — precisely
:eq:`discretization-cell-balance-eq`. The bottom row is the new information the
slope moment adds.

.. warning:: **The slope-row signs are a documented correctness trap.** The
   off-diagonal and right-hand-side signs of the slope (:math:`P_1`) row are
   internally inconsistent in the published boxed form (Larsen & Morel 1989,
   memo §1.4/§6). ORPHEUS therefore does **not** transcribe them: the
   :math:`2\times2` is regenerated symbolically (the SymPy algebra-of-record
   :mod:`orpheus.derivations.discrete.sn.ld_ubld`) and validated against the
   strongest available oracle — **LD is exact on a
   linear-in-:math:`x` flux**, so :math:`(\bar\psi,\hat\psi,\psi_{\rm out})` are
   recovered to machine precision for any :math:`\psi=a+bx`. The sign convention
   lives in exactly one place,
   :meth:`_LDCellTerms.slope <orpheus.transport.spatial.linear_discontinuous._LDCellTerms.slope>`,
   pinned by the foundation tests in
   ``tests/transport/spatial/test_linear_discontinuous.py``.

**The Schur-complement scalar form.** The slope :math:`\hat\psi` is *local* to
the cell — it appears in no other cell's equations — so eliminate it. The
**slope row** of :eq:`discretization-ld-system`,
:math:`-|\mu|\,\bar\psi + (\Sigma_t\theta h + |\mu|)\,\hat\psi
= \theta\,\hat q\,h - |\mu|\,\psi_{\rm in}`, solves for :math:`\hat\psi` in terms
of :math:`\bar\psi`, with the **slope denominator**
:math:`D_2' \equiv \Sigma_t\theta h + |\mu|` (the slope-row diagonal
:math:`\mathbf A_{22}`):

.. math::
   :label: discretization-ld-slope-elim

   \hat\psi = \frac{|\mu|\,\bar\psi + \theta\,\hat q\,h - |\mu|\,\psi_{\rm in}}{D_2'} .

.. vv-status: discretization-ld-slope-elim documented
.. (vv-status rationale) the slope-row elimination ψ̂(ψ̄) of the LD 2×2 — the ψ̂
   closed form produced by
   :func:`orpheus.derivations.discrete.sn.ld_ubld.derive_d1_reduction_to_production`;
   foundation-level algebra, not a solver claim.

Substituting into the **average row**
:math:`(\Sigma_t h + |\mu|)\,\bar\psi + |\mu|\,\hat\psi
= \bar q\,h + |\mu|\,\psi_{\rm in}` and collecting :math:`\bar\psi` collapses the
cell to a **scalar** balance — the Schur complement
:math:`S = \mathbf A_{11} - \mathbf A_{12}\mathbf A_{21}/\mathbf A_{22}` of the
:math:`2\times2` with respect to the slope row:

.. math::
   :label: discretization-ld-schur

   \underbrace{\Bigl[(\Sigma_t h + |\mu|) + \tfrac{|\mu|^2}{D_2'}\Bigr]}_{\displaystyle S}\,\bar\psi
   \;=\;
   \underbrace{\bar q\,h + |\mu|\,\psi_{\rm in}
     - \tfrac{|\mu|\,(\theta\,\hat q\,h - |\mu|\,\psi_{\rm in})}{D_2'}}_{\text{effective source}\,+\,\text{upstream}},
   \qquad
   S = \frac{|\mu|^2 + (\Sigma_t h + |\mu|)\,D_2'}{D_2'} ,

.. vv-status: discretization-ld-schur documented
.. (vv-status rationale) the Schur complement S = A₁₁ − A₁₂A₂₁/A₂₂ of the LD 2×2
   (the slope elimination above, substituted into the average row); expressed
   here from
   :func:`orpheus.derivations.discrete.sn.ld_ubld.derive_d1_reduction_to_production`,
   which proves S / D₂' / ψ̄ / ψ̂ equal the production
   :func:`orpheus.transport.spatial._ubld.d1_closed_form` symbolically. Not a
   solver claim.

so :math:`\bar\psi = \text{rhs}/S`, with :math:`S` a manifestly positive
denominator and the slope recovered locally from
:eq:`discretization-ld-slope-elim` afterward.

This has **exactly the shape of the DD balance**
:math:`(\text{denom})\,\bar\psi = \text{source} + \text{upstream}`, so LD fits
the *same* scalar cell-update contract as DD with no change — the slope is
reconstructed locally after :math:`\bar\psi` is known. That is why LD, despite
carrying two moments, is still *affine-scannable* (it supplies a single-upstream
recurrence :math:`\psi_{\rm out}=a\,\psi_{\rm in}+b` and rides the same fast 1-D
sweep as DD).

**Accuracy.** Because LD represents the flux as a full linear with an
independent slope, it is **exact on a linear-in-:math:`x` flux** and second
order, :math:`O(h^2)` — the same order as DD, but reached through an upwind
(rather than central) coupling.

**Positivity.** LD is **not** strictly positivity-preserving — it can produce
negative averages and edge fluxes in thin or steep-source cells (Lewis & Miller
1984, §5.3), so :class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
declares ``is_positivity_preserving = False``. But its negatives are **milder**
than DD's (the worked contrast above: :math:`-\tfrac1{19}` versus
:math:`-\tfrac13`), because the upwind bias damps the dispersive oscillation
that the central DD stencil has nothing to suppress. A positivity **fixup**
(set-negative-slope-to-zero, the Larsen limiter) makes LD non-negative at the
cost of linearity; the first ORPHEUS cut ships without a fixup, keeping the
clean linear :math:`2\times2`.

**The diffusion limit.** LD's decisive advantage is the **thick-diffusion
limit**. In optically thick, scattering-dominated media the transport equation
limits to a diffusion equation; a spatial scheme is *diffusion-limit consistent*
if its thick limit is a consistent diffusion discretization of the leading-order
:term:`scalar flux`. Step has **no** valid diffusion limit; DD has the leading-order
limit but can oscillate at cell edges; **full LD has all four diffusion limits**
(Larsen, Morel & Miller 1987, Table I; Larsen & Morel 1989, Part II) — provided
the *slope source* moment :math:`\Sigma_s\hat\phi` is threaded through the
scattering iterate. That thread is what lets a non-DD spatial scheme lift the
curvilinear pole-cell :math:`O(h)` accuracy floor (Issue #233). It is declared
``diffusion_limit_consistent = True``.

.. note:: **ORPHEUS scope.** :class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
   implements the **slab / Cartesian** LD; the curvilinear (sphere / cylinder)
   LD cell closure — where the angular weight :math:`\tau` enters the average
   moment and the radial :math:`(r-r_j)` weighting couples slope to curvature —
   is not yet implemented and raises ``NotImplementedError`` on a curvilinear
   visit (Issue #158, the curvilinear arm; the derivation is published —
   Adams--Martin 1992, NSE 111, App. A, with the weighted-diamond angular
   closure applied per spatial moment). This is the concrete instance of the
   re-posing problem in :ref:`discretization-reposing`.


.. _discretization-spectrum:

The closure spectrum — one blend weight
---------------------------------------

The three closures are a **spectrum of in-cell flux representations**, and the
cell-average blend weight :math:`w` of :eq:`discretization-blend` is the single
dial that orders them:

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 38

   * - Scheme
     - :math:`w`
     - In-cell flux
     - Face reconstruction
   * - Step (upwind)
     - :math:`1`
     - constant
     - :math:`\psi_{\rm out} = \bar\psi`
   * - Diamond Difference (central)
     - :math:`\tfrac12`
     - linear (shared faces)
     - :math:`\psi_{\rm out} = 2\bar\psi - \psi_{\rm in}`
   * - Linear Discontinuous
     - :math:`\dfrac1{1+k}`
     - full linear (2 moments)
     - :math:`\psi_{\rm out} = \bar\psi + \hat\psi`

For LD the effective blend is not fixed: eliminating the slope leaves
:math:`w = 1/(1+k)` with the **slope-elimination ratio**

.. math::
   :label: discretization-ld-blend

   k = \frac{g/\theta}{\,g/\theta + \Sigma_t\,},
   \qquad
   w = \frac{1}{1+k},
   \qquad g = \frac{|\mu|}{h},

.. vv-status: discretization-ld-blend documented
.. (vv-status rationale) the LD scale-free invariants k and w=1/(1+k) (the
   Péclet-adaptive blend); implemented in
   :func:`orpheus.transport.spatial._ubld.d1_closed_form`. Structural, not a
   solver claim.

so LD's effective face blend **adapts to the cell optical thickness**:

- **optically thin** (:math:`\Sigma_t\to0`, pure streaming): :math:`k\to1`,
  :math:`w\to\tfrac12` — LD's face blend approaches **Diamond Difference**
  (central).
- **optically thick** (:math:`\Sigma_t\to\infty`, diffusive): :math:`k\to0`,
  :math:`w\to1` — LD's face blend approaches **Step** (upwind).

This is precisely the CFD **Péclet / κ-scheme** adaptive upwinding: central
where the cell is streaming-dominated, upwind where it is
collision-dominated, blended in between — *per cell*, driven by the local
optical thickness :math:`\Sigma_t h / |\mu|`.

.. important:: **The blend weight is one axis; the moment count is another.** It
   is tempting to read "LD :math:`\to` Step in the thick limit" as "LD *becomes*
   Step". It does not. The face-blend weight :math:`w` governs only the
   average-to-faces relation; LD additionally carries the **slope moment**
   :math:`\hat\psi`, which Step and DD do not. Even as :math:`w\to1`, LD tracks
   :math:`\hat\psi`, and that retained moment is exactly what gives LD the
   correct diffusion limit that Step (which has no slope) cannot have. The
   spectrum is therefore two-dimensional: the **blend weight** (Step
   :math:`1` → DD :math:`\tfrac12` → LD adaptive) and the **moment count**
   (Step/DD: one moment; LD: two). The blend orders the *face reconstruction*;
   the moment count sets the *representational richness* — and the diffusion
   limit lives in the moment count, not the blend.


.. _discretization-transmission-ladder:

The transmission multiplier — the positivity ladder
---------------------------------------------------

:ref:`discretization-spectrum` orders the closures by *face
reconstruction*.  A second, independent ordering falls out of asking what
each closure does to the simplest cell there is: **source-free, with a
positive inflow**.  Set :math:`q = 0` and name the cell's optical depth
along the direction of travel

.. math::
   :label: discretization-optical-depth

   \tau_{\rm opt} \;\equiv\; \frac{\Sigma_t\,h}{|\mu|}
   \;=\; \frac{\Sigma_t}{g},
   \qquad
   g \;=\; \frac{|\mu|}{h} .

.. vv-status: discretization-optical-depth documented
.. (vv-status rationale) definition of the per-cell OPTICAL depth along the
   direction of travel — the scale-free variable a source-free cell update
   depends on. Definitional, not a solver claim: it is the ``Σ_t h/|μ|``
   already written verbatim in :eq:`discretization-dd-negative` and the
   quantity :eq:`discretization-ld-blend` calls "the cell optical thickness".

The exact answer is pure attenuation, :math:`\psi_{\rm out} =
e^{-\tau_{\rm opt}}\,\psi_{\rm in}`, so each source-free cell update of
:ref:`discretization-closures` is a **rational approximation to the
exponential**.  Write the
coefficient it produces as

.. math::
   :label: discretization-transmission-multiplier

   \psi_{\rm out} \;=\; a(\tau_{\rm opt})\,\psi_{\rm in},
   \qquad
   a(\tau_{\rm opt}) \;\approx\; e^{-\tau_{\rm opt}} ,

.. vv-status: discretization-transmission-multiplier documented
.. (vv-status rationale) definitional: the source-free specialisation of the
   affine face recurrence ψ_out = a·ψ_in + b that every affine-scannable
   scheme already supplies through ``affine_scan_coefficients``. Not a new
   solver claim — ``a`` is the shipped scan coefficient, and the per-scheme
   closed forms below are read off the already-verified cell updates
   :eq:`discretization-dd-solve` / :eq:`discretization-ld-schur`.

and call :math:`a` the **transmission multiplier**.  It is not a new
object: it is exactly the :math:`a` of the affine face recurrence
:math:`\psi_{\rm out} = a\psi_{\rm in} + b` that every affine-scannable
scheme hands to the fast 1-D scan through
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.affine_scan_coefficients`.

.. warning:: **The letter** :math:`\tau` **is overloaded four ways in this
   corpus, and every collision has cost real time.**  On THIS page — and
   in :eq:`discretization-optical-depth` — it is the *optical* one, and it
   is written :math:`\tau_{\rm opt}` for exactly that reason.

   .. list-table::
      :header-rows: 1
      :widths: 20 44 36

      * - spelling
        - what it is
        - where it lives
      * - :math:`\tau` (unqualified, on the :math:`S_N` pages)
        - the **Morel--Montry angular closure weight** — a dimensionless
          barycentric coordinate in :math:`[0,1]`, one per ordinate,
          :eq:`mm-weights`
        - :eq:`discretization-angular-closure`;
          :ref:`sn-curvilinear-one-group`;
          :func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`
      * - :math:`\tau_{\rm opt}`
        - **optical depth** :math:`\Sigma_t s` along a path
          — :eq:`discretization-optical-depth`
        - this section; the method-of-characteristics and Peierls pages;
          the spatial schemes
      * - :math:`\tau` (Morel--Wareing--Smith 1996, Eq. 75)
        - :math:`\sigma_t/\mu` on their **unit-thickness** model problem
          — the *same physical quantity* as :math:`\tau_{\rm opt}` with the
          cell width absorbed, so the hazard is a **missing** :math:`h`,
          not a different concept
        - quoted verbatim below
      * - :math:`\tau_{F_N}`
        - a **critical half-thickness** in mean free paths
        - the :math:`F_N`-method slab/reflector pages

   The code-side record of the first, second and fourth senses (and of the
   two different :math:`\beta`) is the module docstring of
   :mod:`orpheus.derivations.discrete.sn.angular_differencing`; the third
   is added here.

The ladder
~~~~~~~~~~

Substituting each closure's source-free cell update
(:eq:`discretization-step-solve`, :eq:`discretization-dd-solve`,
:eq:`discretization-ld-schur`) into :eq:`discretization-transmission-multiplier`
gives four closed forms — of which **ORPHEUS ships the first three**;
the lumped member is unimplemented (Issue #158) and appears here because
it is what the ladder's structure is *for* — and each one is a **Padé
approximant of the exponential** — Morel, Wareing & Smith 1996 name the last two as such in
so many words (their Eqs. (74) and (76), reproduced below):

.. math::
   :label: discretization-transmission-pade

   a_{\rm Step}(\tau_{\rm opt}) &= \frac{1}{1 + \tau_{\rm opt}} , \\[4pt]
   a_{\rm DD}(\tau_{\rm opt})   &= \frac{2 - \tau_{\rm opt}}{2 + \tau_{\rm opt}} , \\[4pt]
   a_{\rm LD}(\tau_{\rm opt})   &= \frac{1 - \tau_{\rm opt}/3}
                                       {1 + 2\tau_{\rm opt}/3 + \tau_{\rm opt}^{2}/6} , \\[4pt]
   a_{\rm LLD}(\tau_{\rm opt})  &= \frac{1}{1 + \tau_{\rm opt} + \tau_{\rm opt}^{2}/2} .

.. vv-status: discretization-transmission-pade documented
.. (vv-status rationale) the four source-free transmission multipliers, read
   off the already-verified cell updates by setting q = 0. The LD and lumped-LD
   forms are literature-transcribed (Morel-Wareing-Smith 1996 Eqs. 74/76,
   page-verified). Definitional/structural, not a solver claim — the cell
   updates they specialise are pinned by
   :func:`orpheus.derivations.discrete.sn.balance.derive_cartesian_1d` (DD) and
   :func:`orpheus.derivations.discrete.sn.ld_ubld.derive_d1_reduction_to_production`
   (LD); the lumped member is not implemented (Issue #158 / #408).

.. list-table:: the transmission ladder — accuracy against positivity
   :header-rows: 1
   :widths: 17 15 21 20 27

   * - scheme
     - Padé
     - order in :math:`\tau_{\rm opt}`
     - first sign change
     - :math:`a` as :math:`\tau_{\rm opt}\to\infty`
   * - **Step**
     - :math:`(0,1)`
     - 1st (error :math:`+\tau_{\rm opt}^{2}/2`)
     - **never**
     - :math:`\to 0^{+}` like :math:`1/\tau_{\rm opt}`
   * - **Diamond Difference**
     - :math:`(1,1)`
     - 2nd (error :math:`-\tau_{\rm opt}^{3}/12`)
     - **exactly** :math:`\tau_{\rm opt} = 2`
     - :math:`\to -1` — a full sign flip with **no decay at all**
   * - **bare Linear Discontinuous**
     - :math:`(1,2)`
     - 3rd (error :math:`-\tau_{\rm opt}^{4}/72`)
     - **exactly** :math:`\tau_{\rm opt} = 3`
     - :math:`\to 0^{-}` like :math:`-2/\tau_{\rm opt}`
   * - **lumped Linear Discontinuous**
     - :math:`(0,2)`
     - 2nd (error :math:`+\tau_{\rm opt}^{3}/6`)
     - **never**
     - :math:`\to 0^{+}` like :math:`2/\tau_{\rm opt}^{2}`

**The structure the table exposes, and it is one line.**  A sign change is
a **positive real root of the NUMERATOR**.  Step and lumped LD are
:math:`(0,n)` — constant numerator, so no root exists at any optical depth;
DD and bare LD are :math:`(1,n)` — a degree-1 numerator, so a positive root
exists and its location is forced by the requirement that the approximant
match :math:`e^{-\tau_{\rm opt}}` to the stated order.  **Lumping is the
operation that drops the numerator degree by one**, and the ladder says
precisely what that costs: one order of accuracy, in exchange for the
removal of the root.  Morel, Wareing & Smith (1996, printed p. 452) put it
in their own words, of the unlumped-versus-lumped pair above:

   *"The linear-discontinuous solution is third-order accurate in*
   :math:`\tau` *, but it is negative for* :math:`\tau > 3` *.  The lumped
   linear-discontinuous solution is only second-order accurate in*
   :math:`\tau` *, but it is strictly positive.  Thus the lumped scheme is
   indeed less accurate but more robust."*

⚠ The *denominators* matter too, and all four are safe: a positive real
root of a denominator would be a **pole**, which is a far worse failure
than a sign change.  None of the four has one — Step and DD have negative
roots (:math:`-1`, :math:`-2`), and both quadratics are definite
(discriminants :math:`4/9 - 2/3 = -2/9` for LD and :math:`1 - 2 = -1` for
lumped LD, both negative, so neither has a real root at all).

.. admonition:: The worked contrast above is this ladder at
   :math:`\tau_{\rm opt} = 4`

   :ref:`The thick-cell worked contrast <discretization-negative-flux-example>`
   is not a separate fact.
   Its cell is :math:`\Sigma_t = 1`, :math:`h = 2`, :math:`|\mu| =
   \tfrac12`, so :math:`\tau_{\rm opt} = \Sigma_t h/|\mu| = 4`, and the
   three multipliers of :eq:`discretization-transmission-pade` evaluate to
   :math:`a_{\rm Step} = \tfrac15`, :math:`a_{\rm DD} = -\tfrac13`,
   :math:`a_{\rm LD} = -\tfrac1{19}` — the three :math:`\psi_{\rm out}`
   values already tabulated there, exactly.  The lumped member would give
   :math:`a_{\rm LLD} = 1/(1+4+8) = \tfrac1{13} \approx +0.077`.  The reason LD's negative
   is "6.3× milder than DD's" is now readable off the ladder rather than
   observed: at :math:`\tau_{\rm opt} = 4` DD is already twice past its root
   while LD is barely past its own, and DD's multiplier is heading for
   :math:`-1` while LD's is heading for :math:`0`.

Values against the exact attenuation, so the two failure directions are
visible side by side.  All four columns are evaluated from
:eq:`discretization-transmission-pade`; the DD and bare-LD columns are
`[M]` reproduced by the **shipped** scan coefficients — a slab cell
(:math:`A_{\rm down}=1`, :math:`A_{\rm total}=2`, :math:`V=h`,
:math:`\Delta A/w = 0`) fed to
:meth:`DiamondDifference.affine_scan_coefficients
<orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients>`
and
:meth:`LinearDiscontinuous.affine_scan_coefficients
<orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous.affine_scan_coefficients>`
returns them with :math:`\max|a_{\rm shipped} - a_{\rm closed\ form}| =
1.1\times10^{-16}` (DD) and :math:`1.2\times10^{-16}` (LD) over the six
rows.  The lumped column has no shipped counterpart to compare against
(Issue #158):

.. list-table:: :math:`a(\tau_{\rm opt})` versus :math:`e^{-\tau_{\rm opt}}`
   :header-rows: 1
   :widths: 14 18 18 18 18 14

   * - :math:`\tau_{\rm opt}`
     - Step
     - DD
     - bare LD
     - lumped LD
     - :math:`e^{-\tau_{\rm opt}}`
   * - 0.5
     - ``0.666667``
     - ``0.600000``
     - ``0.606061``
     - ``0.615385``
     - ``0.606531``
   * - 1
     - ``0.500000``
     - ``0.333333``
     - ``0.363636``
     - ``0.400000``
     - ``0.367879``
   * - 2
     - ``0.333333``
     - ``0.000000``
     - ``0.111111``
     - ``0.200000``
     - ``0.135335``
   * - 3
     - ``0.250000``
     - ``-0.200000``
     - ``0.000000``
     - ``0.117647``
     - ``0.049787``
   * - 6
     - ``0.142857``
     - ``-0.500000``
     - ``-0.090909``
     - ``0.040000``
     - ``0.002479``
   * - 20
     - ``0.047619``
     - ``-0.818182``
     - ``-0.069959``
     - ``0.004525``
     - ``2.06e-09``

Read the :math:`\tau_{\rm opt} = 20` row twice.  DD returns
:math:`-0.818` where the physics says :math:`2\times10^{-9}`: not merely
the wrong sign but the wrong *magnitude by nine orders*, and it does not
improve as the cell gets thicker.  Bare LD returns :math:`-0.070` — still
the wrong sign, but decaying.  This is the sense in which LD's negatives
are "milder", made quantitative.

.. important:: **Where this bites hardest is the curvilinear
   starting direction**, because that solve has **nothing angular to damp
   it**: the redistribution coefficient vanishes at both ends of every
   angular march (:math:`\alpha_{1/2} = \alpha_{M+1/2} = 0`), so the
   starting-direction equation is a *purely spatial* advection--reaction
   solve that inherits its scheme's ladder row verbatim.  The measurement,
   on the shipped march, is at :ref:`sn-seed-cone-risk` in
   :doc:`/theory/methods/sn/curvilinear_one_group`.

Two things this ladder is NOT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**(1) It is not the multi-dimensional face-transmission spectrum — it is
its** :math:`d = 1` **case, and the two answer different questions.**
Driving a source-free cell one unit inflow at a time gives a face-to-face
transmission *matrix* :math:`\Sigma`, whose multi-D DD form is
:eq:`dd-face-transmission-spectrum`,
:math:`\Sigma = (2/D)\,\mathbf 1\mathbf w^{\mathsf T} - I`.  At
:math:`d = 1` that matrix is :math:`1\times1` and its single entry **is**
:math:`a_{\rm DD}(\tau_{\rm opt})`.  What changes at :math:`d \ge 2` is
that :math:`\Sigma` acquires a second eigenvalue family — the undamped
sawtooth :math:`-1` with multiplicity :math:`d-1` — which has no
:math:`d = 1` analogue.  So:

* the **sign** of the physical eigenvalue is the *positivity* question
  (this section), and it is already live at :math:`d = 1`;
* the **modulus** of the sawtooth eigenvalue is the *gauge* question (a
  singular loss operator on a multiply-reflective box), and it is
  :math:`d \ge 2` only.

:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.face_transmission_spectrum`
measures the second; it reports :math:`\rho = |a|` at :math:`d = 1` (`[M]`
``0.7391304347826089`` for DD on its own thinner probe cell, which is
:math:`(2 - \tau_{\rm opt})/(2 + \tau_{\rm opt})` at that cell's
:math:`\tau_{\rm opt} = 0.3`) and ``1.0`` at :math:`d = 2, 3`.  Neither
number is a positivity verdict.

**(2) It is not a full cone-preservation verdict, and reading it as one is
the mistake Issue #408 exists to prevent.**  ``is_positivity_preserving``
(:mod:`orpheus.transport.spatial.scheme`, documented as *"the
cone-preservation flag"*) conflates **three** distinct properties, and
lumping repairs only the first:

.. list-table::
   :header-rows: 1
   :widths: 26 34 13 13 14

   * - property
     - meaning
     - DD
     - bare LD
     - lumped LD
   * - transmission sign-preservation
     - :math:`a(\tau_{\rm opt}) > 0` — a non-negative inflow cannot
       produce a negative outflow
     - ``False``
     - ``False``
     - ``True``
   * - monotonicity (:math:`\mathbf A^{-1}\ge0`, the M-matrix property)
     - the whole cell solve maps the cone into the cone, for any
       non-negative nodal source too
     - ``False``
     - ``False``
     - ``False``
   * - per-component cone membership
     - ":math:`\psi_{\text{component}} \ge 0`" is a meaningful predicate
       at all
     - meaningful
     - **not meaningful**
     - **not meaningful**

The naive move when a lumped member lands is to set the flag ``True``,
since lumping is *the* positivity-restoring operation in the literature.
⛔ That would be wrong.  Morel, Wareing & Smith lump the mass matrices and
deliberately **not** the spatial gradient (their §4), so the lumped cell
matrix keeps :math:`\mathbf A_{LR} = +\tfrac12` — the wrong sign for an
M-matrix — and its inverse carries the explicit counterexample
:math:`\mathbf A^{-1}_{LR} = -2/(\tau_{\rm opt}^{2} + 2\tau_{\rm opt} + 2)
< 0`: a source concentrated on the **downstream** node drives the upstream
node negative.  That is the paper's own §8 caveat, *"quite robust, but not
strictly positive"*.  The third row is already modelled correctly
elsewhere and must not be duplicated onto the scheme — the spatial-moment
axis is **MODAL**, so a strictly positive function has legally signed slope
coefficients, and ``has_coordinate_cone`` reads ``False`` through
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`.

.. _discretization-lumped-family:

"Lumped LD" is a FAMILY, and the trade is a choice inside it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last row of the table above cannot be a single row, and the reason is
worth deriving because it names the knob the flag would have to be keyed
on.  Write the mass-lumped slab-LD cell in **nodal** coordinates
:math:`(\psi_L, \psi_R)` — the two Lagrange node values, of which
:math:`\psi_R` is the outflow face — with a general within-cell leakage
matrix :math:`\mathbf L` and the upwind inflow entering only the
:math:`L` row:

.. math::
   :label: discretization-lumped-family-cell

   \Bigl(\mathbf L + \tfrac{\tau_{\rm opt}}{2}\,\mathbf I\Bigr)
   \begin{bmatrix}\psi_L \\ \psi_R\end{bmatrix}
   = \begin{bmatrix}\psi_{\rm in} \\ 0\end{bmatrix},
   \qquad
   \mathbf L = \begin{bmatrix} 1-\lambda & \lambda \\ -\nu & \nu \end{bmatrix},
   \qquad
   a = \psi_R\big|_{\psi_{\rm in}=1} .

.. vv-status: discretization-lumped-family-cell documented
.. (vv-status rationale) the mass-lumped nodal slab-LD cell written with a
   one-parameter-per-row within-cell leakage — a re-parametrisation of the
   already-verified LD 2×2 :eq:`discretization-ld-system`, used here only to
   exhibit the family. Structural, not a solver claim; no member other than
   the standard upwind one is implemented (Issue #158 / #408).

The row sums of :math:`\mathbf L` are already fixed at :math:`(1, 0)` in
:eq:`discretization-lumped-family-cell` — that is the **infinite-medium
identity**, that a uniform flux with :math:`\psi_{\rm in} = \psi_L =
\psi_R` reproduce itself — so exactly **one free parameter per row**
survives, :math:`\lambda` and :math:`\nu`.  Two further conditions cut it
down, and both are one line of algebra:

* **Consistency.**  :math:`a(0) = 1` holds identically (that *is* the
  infinite-medium identity), but the *rate* does not:
  :math:`a'(0) = (\lambda - \nu - 1)/(2\nu)`, and matching the exponential's
  :math:`-1` forces

  .. math::
     :label: discretization-lumped-consistency

     \nu \;=\; 1 - \lambda .

  .. vv-status: discretization-lumped-consistency documented
  .. (vv-status rationale) the consistency condition on the family of
     :eq:`discretization-lumped-family-cell`, obtained by matching a'(0) to
     the exponential's slope. Derived, definitional; no shipped member other
     than λ = ν = ½ (Issue #158 / #408).

  ⟹ the family is **one**-parameter, not two.
* **Monotonicity.**  With :math:`\nu = 1-\lambda`,
  :math:`\mathbf A^{-1}_{LR} = -\lambda/\det` and
  :math:`\mathbf A^{-1}_{RL} = \nu/\det` with :math:`\det > 0` throughout,
  so :math:`\mathbf A^{-1} \ge 0` entrywise **iff** :math:`\lambda \le 0`.

The Morel--Wareing--Smith member is :math:`(\lambda,\nu) =
(\tfrac12,\tfrac12)` — consistent, second order, sign-preserving
transmission, and :math:`\lambda = +\tfrac12 > 0`, so **not monotone**,
which is the :math:`\mathbf A^{-1}_{LR} < 0` counterexample above.  The
nearest monotone member is :math:`(\lambda,\nu) = (0,1)`:

.. math::
   :label: discretization-lumped-monotone-member

   a_{(0,1)}(\tau_{\rm opt}) \;=\; \frac{1}{\bigl(1 + \tau_{\rm opt}/2\bigr)^{2}} ,
   \qquad
   \mathbf A^{-1} = \frac{1}{1+\tau_{\rm opt}/2}
   \begin{bmatrix} 1 & 0 \\ \frac{1}{1+\tau_{\rm opt}/2} & 1 \end{bmatrix}
   \;\ge\; 0 ,

.. vv-status: discretization-lumped-monotone-member documented
.. (vv-status rationale) the transmission and inverse of the (λ,ν) = (0,1)
   member of :eq:`discretization-lumped-family-cell` — the nearest monotone
   consistent member. Derived here; not implemented (Issue #158 / #408), so
   there is no code to gate.

strictly positive at every optical depth, monotone, and **first** order
(error :math:`+\tau_{\rm opt}^{2}/4`) — one order below MWS, which is the
honest price of the M-matrix property.

.. warning:: ⛔ **A monotone member is not automatically a scheme —
   check the SLOPE, not just the sign.**  The obvious lower-triangular
   candidate :math:`(\lambda,\nu) = (0,\tfrac12)`, with the attractive
   transmission
   :math:`a = 2/\bigl((1+\tau_{\rm opt})(2+\tau_{\rm opt})\bigr)`,
   is monotone and strictly positive — and it is **inconsistent**:
   :math:`\nu \ne 1-\lambda`, so :math:`a'(0) = -\tfrac32` rather than
   :math:`-1`, and the transmission across a *fixed* physical thickness
   :math:`X` tends to :math:`e^{-\frac32 \Sigma_t X/|\mu|}` however fine
   the mesh.  Measured on :math:`\Sigma_t X/|\mu| = 1`:

   .. list-table::
      :header-rows: 1
      :widths: 14 22 22 22 20

      * - cells
        - MWS :math:`(\tfrac12,\tfrac12)`
        - :math:`(0,\tfrac12)`
        - :math:`(0,1)`
        - bare LD
      * - 10
        - ``0.368449``
        - ``0.236690``
        - ``0.376889``
        - ``0.367874``
      * - 100
        - ``0.367886``
        - ``0.224521``
        - ``0.368797``
        - ``0.367879``
      * - 1000
        - ``0.367880``
        - ``0.223270``
        - ``0.367971``
        - ``0.367879``
      * - 10000
        - ``0.367879``
        - ``0.223144``
        - ``0.367889``
        - ``0.367879``
      * - exact
        - ``0.367879``
        - :math:`e^{-1} = 0.367879`
        - ``0.367879``
        - ``0.367879``

   The :math:`(0,\tfrac12)` column is converging cleanly — to
   :math:`e^{-3/2} = 0.223130`.  This is
   ``vv-principles`` anti-pattern #5 in its purest form: *a correct
   convergence rate to the wrong limit is still the wrong limit*, and the
   positivity and monotonicity properties are both perfectly true of it.
   ⟹ **admit a member to the family only after checking**
   :eq:`discretization-lumped-consistency`; sign-preservation and
   :math:`\mathbf A^{-1} \ge 0` are silent about it.

⟹ **the accuracy/positivity trade is a choice of** :math:`\lambda`
**inside** :eq:`discretization-lumped-consistency`, not a property of
"lumping"; "lumped LD" names a *family*; and any positivity flag must
therefore be **per member**.

.. _discretization-space-angle:

5. One closure, two axes: space and angle
=========================================

Everything above used the abstract cell of :ref:`discretization-cell-balance`,
which was deliberately built with **no reference to the axis being space**. The
payoff is now collected: in curvilinear geometry the *same* cell-balance +
closure machinery discretizes the **angular** axis, because curvilinear
streaming carries an angular derivative that produces an angular cell balance of
identical structure.

Why curvilinear streaming has an angular term
---------------------------------------------

In slab geometry the streaming operator is purely spatial,
:math:`\mu\,\partial_x\psi`. In spherical or cylindrical geometry the divergence
of the angular current, written in the natural coordinates, picks up an
**angular derivative** term — physically, as a neutron streams along a straight
line through curved coordinates, its direction cosine :math:`\mu` relative to
the local radial axis *changes*. Discretizing :math:`\mu` into the ordinate set
turns that :math:`\partial/\partial\mu` term into a balance across an **angular
cell** :math:`[\mu_{n-1/2}, \mu_{n+1/2}]` — the *angular redistribution*.

The angular cell balance is the same cell balance
-------------------------------------------------

Integrating the curvilinear per-ordinate equation over one **space × angle**
product cell gives the balance the SymPy algebra-of-record verifies in
:func:`~orpheus.derivations.discrete.sn.balance.derive_curvilinear_balance`:

.. math::
   :label: discretization-curvilinear-balance

   \underbrace{|\mu|\bigl(A_{\rm out}\psi_{\rm out} - A_{\rm in}\psi_{\rm in}\bigr)}_{\text{spatial net face outflow}}
   \;+\;
   \underbrace{\frac{\Delta A}{w_m}\bigl(\alpha_{\rm out}\psi^\theta_{\rm out} - \alpha_{\rm in}\psi^\theta_{\rm in}\bigr)}_{\text{angular net face outflow}}
   \;+\;
   \underbrace{\Sigma_t V\,\bar\psi}_{\text{collision}}
   \;=\;
   \underbrace{q\,V}_{\text{source}}.

.. vv-status: discretization-curvilinear-balance documented
.. (vv-status rationale) the curvilinear (spherical/cylindrical) cell balance
   with the ΔA/w angular-redistribution term; derived and pinned by SymPy in
   :func:`orpheus.derivations.discrete.sn.balance.derive_curvilinear_balance`.
   Foundation-level, not a solver claim. The full curvilinear S_N treatment is
   :ref:`theory-discrete-ordinates`.

Read the two streaming terms side by side. The first is the **spatial** net face
outflow: face fluxes :math:`\psi_{\rm in},\psi_{\rm out}` through spatial faces
of areas :math:`A_{\rm in}, A_{\rm out}`. The second is the **angular** net face
outflow: half-angle fluxes :math:`\psi^\theta_{\rm in},\psi^\theta_{\rm out}`
through the *angular* faces of the angular cell, with the coefficients
:math:`\alpha_{\rm in},\alpha_{\rm out}` playing the role of **angular face
areas** and the geometry factor :math:`\Delta A/w_m` normalizing them (here
:math:`w_m` is the **angular quadrature weight** of ordinate :math:`m` — *not*
the cell-average blend weight :math:`w` of :ref:`discretization-closures`; the
two are distinct quantities that share a letter in the literature, so this page
subscripts the quadrature weight). The two
terms have **identical structure** — net outflow across the two faces of a
one-dimensional cell — one on a *spatial* interval, one on an *angular*
interval. Collision and source are the same as before. Equation
:eq:`discretization-curvilinear-balance` is a cell balance on a product cell,
and it needs a closure **on each axis**.

The same weighted-diamond closure, on each axis
-----------------------------------------------

The closures of :ref:`discretization-closures` apply per axis. The Diamond
Difference closure on the **spatial** face is
:eq:`discretization-dd-closure`,
:math:`\psi_{\rm out}=2\bar\psi-\psi_{\rm in}`. The closure on the **angular**
face is the *same weighted-diamond form*, with the Morel–Montry angular weight
:math:`\tau` playing the role the spatial blend weight :math:`w` played:

.. math::
   :label: discretization-angular-closure

   \bar\psi = \tau\,\psi^\theta_{\rm out} + (1-\tau)\,\psi^\theta_{\rm in}
   \quad\Longleftrightarrow\quad
   \psi^\theta_{\rm out} = \frac{\bar\psi - (1-\tau)\,\psi^\theta_{\rm in}}{\tau} .

.. vv-status: discretization-angular-closure documented
.. (vv-status rationale) the Morel–Montry weighted-diamond ANGULAR closure —
   the angular-axis analogue of :eq:`discretization-blend`. Pinned by SymPy in
   :func:`orpheus.derivations.discrete.sn.balance.derive_wdd_solve`
   (τ=½ recovers the standard angular diamond). Structural, not a solver claim.

Compare :eq:`discretization-angular-closure` with the spatial blend
:eq:`discretization-blend`, :math:`\bar\psi=(1-w)\psi_{\rm in}+w\psi_{\rm out}`.
They are the **same equation**, with :math:`\tau` in the angular role of
:math:`w`. Setting :math:`\tau=\tfrac12` gives the angular *diamond*
:math:`\bar\psi=\tfrac12(\psi^\theta_{\rm out}+\psi^\theta_{\rm in})` — the exact
angular analogue of spatial DD. The Morel–Montry choice of :math:`\tau`
(unique exact-on-linear-in-:math:`\mu`; Bailey, Morel & Chang 2010, Eq. 43)
is the *weighted-diamond* generalization, exactly as a weighted spatial diamond
generalizes DD.

Substituting *both* closures (spatial DD and angular WDD) into the curvilinear
balance :eq:`discretization-curvilinear-balance` and solving for :math:`\bar\psi`
is
:func:`~orpheus.derivations.discrete.sn.balance.derive_wdd_solve`; it yields the
combined denominator constant :math:`c_{\rm out}=\alpha_{\rm out}/\tau` and
numerator constant :math:`c_{\rm in}=(1-\tau)/\tau\cdot\alpha_{\rm out} +
\alpha_{\rm in}`, which reduce to the standard-DD values
(:math:`c_{\rm out}=2\alpha_{\rm out}`,
:math:`c_{\rm in}=\alpha_{\rm out}+\alpha_{\rm in}`) at :math:`\tau=\tfrac12`.

The flat-flux consistency identity
----------------------------------

The angular closure carries the exact analogue of the spatial "exactness on a
constant flux". For a spatially- and angularly-flat flux the spatial and angular
net-outflow terms must **cancel per ordinate** (a constant flux has zero net
streaming). With the recursion
:math:`\alpha_{\rm out}-\alpha_{\rm in}=-w_m\mu` (verified in
:func:`~orpheus.derivations.discrete.sn.balance.prove_flat_flux_consistency`),

.. math::
   :label: discretization-flat-flux

   \underbrace{\mu\,\Delta A\,\psi_0}_{\text{spatial}}
   \;+\;
   \underbrace{\frac{\Delta A}{w_m}\,(\alpha_{\rm out}-\alpha_{\rm in})\,\psi_0}_{\text{angular}}
   \;=\;
   \mu\,\Delta A\,\psi_0 + \frac{\Delta A}{w_m}(-w_m\mu)\,\psi_0
   \;=\; 0 .

.. vv-status: discretization-flat-flux documented
.. (vv-status rationale) the per-ordinate flat-flux consistency identity — the
   spatial/angular net-outflow cancellation that the ΔA/w geometry factor
   guarantees. Pinned by SymPy in
   :func:`orpheus.derivations.discrete.sn.balance.prove_flat_flux_consistency`
   (and its counter-proof that dropping ΔA/w breaks the cancellation).
   Foundation-level, not a solver claim.

The :math:`\Delta A/w_m` geometry factor is **exactly** what makes this
cancellation hold *per ordinate* rather than only in the sum over ordinates —
dropping it leaves a residual :math:`(\mu\Delta A - w_m\mu)\psi_0\ne0` and
creates artificial angular anisotropy that worsens under mesh refinement (the
Morel–Montry flux dip). This is the angular-axis counterpart of the spatial
closure's exactness-on-constants, and it is the load-bearing correctness
property of the curvilinear discretization.

.. admonition:: The unification, stated plainly

   The cell balance + closure is **one machine**. In slab :math:`S_N` it
   discretizes space (the closures of :ref:`discretization-closures`). In
   curvilinear :math:`S_N` it discretizes space **and** angle — the angular
   redistribution :eq:`discretization-curvilinear-balance` is the *same* cell
   balance and the Morel–Montry closure :eq:`discretization-angular-closure` is
   the *same* weighted diamond, on the angular axis. A future higher-order
   angular scheme would be Step or LD applied to the angular cell, exactly as
   they apply to the spatial cell. The full curvilinear :math:`S_N`
   construction — the :math:`\alpha`-dome recursion, the geometry factor, the
   half-angle :term:`starting direction`, and radial characteristics — lives in
   :ref:`theory-discrete-ordinates`; this chapter supplies the shared skeleton
   it specializes.


.. _discretization-reposing:

6. Re-posing a scheme-specific construct via its invariant
==========================================================

The generalization lever of :ref:`discretization-invariant` now has a concrete
target. In curvilinear :math:`S_N`, **radial characteristics** — the exact
single-pass inverse that marches the half-angle starting direction
:math:`\psi_{1/2}` along the radial characteristic
(:class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator`,
:ref:`sn-direct-seed-solve`) — is written today for the
**Diamond-Difference** angular closure. That looks like a scheme-specific
construct. Naming its invariant shows it is not.

Name the invariant
------------------

Radial characteristics enforces the **angular-redistribution balance**: along
the starting-direction characteristic, the net angular-face outflow
:math:`\tfrac{\Delta A}{w_m}(\alpha_{\rm out}\psi^\theta_{\rm out} -
\alpha_{\rm in}\psi^\theta_{\rm in})` of
:eq:`discretization-curvilinear-balance` balances against the spatial streaming
and collision on the angular cell, closed by
:eq:`discretization-angular-closure`. That balance — *sinks = sources on the
angular cell* — is a special case of :eq:`discretization-invariant-eq`, and it
is **closure-independent**: it is a statement about the angular faces and the
collision, not about which face ↔ average rule is used.

Re-pose it for the other schemes
--------------------------------

Because the invariant is closure-independent, the *same* angular-redistribution
balance can be posed with any spatial/angular closure of
:ref:`discretization-closures`:

- **Diamond Difference** (today): close the angular cell with the weighted
  diamond :eq:`discretization-angular-closure` at the Morel–Montry :math:`\tau`;
  the starting direction marches through the DD-closed balance.
- **Step**: close the angular cell with the piecewise-constant relation
  :math:`\psi^\theta_{\rm out}=\bar\psi` (:eq:`discretization-step-closure` on
  the angular axis, :math:`\tau=1`); the radial-characteristic march is the same
  invariant with a constant angular reconstruction — unconditionally positive,
  first order in the angular cell.
- **Linear Discontinuous**: close the angular cell with the full-linear
  two-moment relation (:eq:`discretization-ld-moments`–:eq:`discretization-ld-face`
  on the angular axis); the march carries an angular slope moment, second order
  in the angular cell.

  ⛔ **This bullet used to add "with the diffusion limit that lifts the
  pole-cell** :math:`O(h)` **floor" — corrected 2026-08-26, because it
  attributed a SPATIAL cure to an ANGULAR device.** ERR-059's pole-cell
  floor is a defect of the *spatial* closure, and what lifts it is
  LD-in-**space** (Issue #158's curvilinear arm).  LD-in-**angle** is a
  different, separately published scheme (Walters & Morel 1991, described
  in Lathrop 2000 §III.D): it *dissolves* the starting-direction
  requirement rather than improving the spatial order, and Larsen & Morel
  record it as **less** accurate than weighted diamond.  The posing point
  the bullet makes is untouched — the invariant really is
  closure-device-agnostic on the angular axis; only the accuracy credit
  was misattributed.

The Step and LD radial-characteristic forms are **posable** in exactly this
sense — the invariant is fixed, only the closure device changes. (The
curvilinear LD closure itself is not yet implemented and is the open Issue #158
curvilinear arm; the point here is structural: identifying the invariant is what
converts "radial characteristics is DD-specific" into "radial characteristics is
the angular-redistribution balance, closable by any scheme".) This is the
operational content of *pose by the invariant*: a construct is generalized not
by rewriting its algebra but by naming the invariant beneath it and re-posing
that invariant with a different closure.


.. _discretization-operator-bridge:

7. Bridge to the operator algebra
=================================

The discretized cell balance is not merely a recurrence — assembled over all
cells and ordinates, it **is** the loss operator of the transport operator
algebra. This section is the bridge; the full treatment is
:doc:`/theory/foundations/operator_algebra`.

The balance is :math:`L + C`
----------------------------

Read the divided cell balance
:eq:`discretization-cell-balance-divided`,
:math:`g(\psi_{\rm out}-\psi_{\rm in}) + \Sigma_t\bar\psi = q`, as an operator
acting on the flux vector :math:`\boldsymbol\psi`:

- The streaming term :math:`g(\psi_{\rm out}-\psi_{\rm in})` assembles, over the
  cell chain, into the **streaming operator** :math:`L` — a first-difference
  (bidiagonal) operator that couples each cell's outflow to its inflow.
- The collision term :math:`\Sigma_t\bar\psi` assembles into the **collision /
  removal operator** :math:`C` — a **multiplication operator**, diagonal,
  acting pointwise as :math:`\Sigma_t\cdot` (its intrinsic type is
  :ref:`multiplication-operator-promotion`).

Their sum :math:`L+C` is the **within-group loss operator**; with the two
collision gains :math:`S` (scattering) and :math:`N_{2n}` (the :math:`(n,2n)`
emission — the same binding in a different role, :ref:`the two collision gains
<operator-algebra-two-gains>`), both projections / integral couplings, the
boundary law :math:`B`,
and the fission dyad :math:`F` (an :ref:`integral-kernel-category`), the full
operator is :math:`A = L + C - S - N_{2n} - B`, posed
:math:`A\psi=\tfrac1k F\psi`
(eigenvalue) or :math:`A\psi=q` (fixed source) — the algebra of
:ref:`operator-algebra`. The discretization of this chapter is what turns the
*continuous* operators :math:`L,C,S,N_{2n},B,F` into the *finite matrices*
:math:`\mathbf A,\mathbf F` of :eq:`discretization-fully-discrete`.

The closure sets the matrix structure
--------------------------------------

**Which closure is chosen determines the matrix structure of** :math:`L+C`,
and that structure is what makes it invertible. The Diamond-Difference
reconstruction :math:`\psi_{\rm out}=2\bar\psi-\psi_{\rm in}` makes each cell's
outflow depend only on its own average and its (already-known) inflow. Ordered
along the sweep (upwind) direction, :math:`L+C` is therefore **lower-triangular**
(bidiagonal): cell :math:`i` couples only to cell :math:`i-1`. A lower-triangular
system is solved by **forward substitution** — and forward substitution through
the cell chain is *exactly the transport sweep*. In operator language,
:math:`(L+C)^{-1}` **is** the sweep: the closure is what gives :math:`L+C` the
triangular structure that makes the sweep a genuine inverse.

.. note:: **The closure is the invertibility.** A general linear operator is
   inverted only by a global solve; the diamond (and Step, and LD-after-Schur)
   closure makes :math:`L+C` *unit-lower-triangular in sweep order*, so its
   inverse is a single forward pass — :math:`O(N)` cells, no matrix
   materialized. This is the deep reason the sweep exists, and it is a *property
   of the closure*, not of the physics: a different closure that broke the
   triangularity would forfeit the sweep and force a global solve. How the
   triangular :math:`(L+C)` becomes the strategy-encoding sweep operator (and
   the Krylov alternative that avoids materializing the matrix) is developed in
   :ref:`operator-algebra` and specialized for :math:`S_N` in
   :ref:`theory-discrete-ordinates`.


Literature
==========

The derivations on this page are the presentation layer of the SymPy
algebra-of-record :mod:`orpheus.derivations.discrete.sn.balance`, which verifies
each identity symbolically (``pytest`` foundation tests). References are given
in full so the page is self-contained (the equation and section numbers are the
load-bearing detail).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Reference
     - What it supplies
   * - **Lewis, E. E. & Miller, W. F. (1984).** *Computational Methods of
       Neutron Transport.* Wiley/ANS.
     - §5.3 — Diamond Difference, weighted-DD, Step, Linear Discontinuous, and
       the negative-flux failure mode (the canonical DD counter-example). §4.5 —
       the Morel–Montry angular closure. The general :math:`S_N` framework.
   * - **Hébert, A. (2009).** *Applied Reactor Physics.* Presses
       Internationales Polytechnique.
     - Ch. 3 §3.9.4 — the curvilinear :math:`S_N` cell balance and difference
       relations; the primary source for the geometry-factor form of
       :eq:`discretization-curvilinear-balance`.
   * - **Larsen, E. W. & Morel, J. E. (1989).** *Asymptotic solutions of
       numerical transport problems in optically thick, diffusive regimes II.*
       JCP 83(1):212–236.
     - §IV, Eqs. (4.1)–(4.3) — the slab-LD two-moment system
       :eq:`discretization-ld-system` and the full diffusion-limit proof
       (Part II); the slope-row sign trap.
   * - **Larsen, E. W., Morel, J. E. & Miller, W. F. (1987).** *Asymptotic
       solutions of numerical transport problems in optically thick, diffusive
       regimes.* JCP 69(2):283–324.
     - Table I p.287, §IX — the verdict that LD has all four diffusion limits,
       DD the leading-order limit, and Step none.
   * - **Bailey, T. S., Morel, J. E. & Chang, J. H. (2010).** *Asymptotic
       Diffusion-Limit Accuracy of :math:`S_N` Angular Differencing Schemes.*
       NSE 165(2):149–169 (LLNL-JRNL-420356).
     - Eq. 43 — the Morel–Montry angular-closure weight :math:`\tau` (unique
       exact-on-linear-in-:math:`\mu`) used in
       :eq:`discretization-angular-closure`.
   * - **Maginot, P. G., Ragusa, J. C. & Morel, J. E. (2016).** *Upstream
       Linear-Discontinuous discretization.* NSE 185(1):17–42.
     - The :math:`d`-dimensional Cartesian UBLD tensor-product structure that
       generalizes the 1-D LD system to :math:`d\ge2`
       (:mod:`orpheus.transport.spatial._ubld`).
   * - **Duderstadt, J. J. & Hamilton, L. J. (1976).** *Nuclear Reactor
       Analysis.* Wiley.
     - The steady-state transport equation :eq:`discretization-continuous` and
       the source :eq:`discretization-source`.

.. seealso::

   - :doc:`/theory/foundations/operator_algebra` — the operator algebra
     :math:`A = L + C - S - N_{2n} - B` this chapter's cell balance
     assembles into.
   - :doc:`/theory/foundations/infinite_medium` — the 0-D multigroup balance
     (:eq:`discretization-semidiscrete-energy` with all streaming dropped).
   - :doc:`/theory/foundations/cross_section_data` — the multigroup
     cross-section pipeline behind :math:`\Sigma_{t,g}`, :math:`\Sigma_{s,g'\to
     g}`, :math:`\chi_g`.
   - :doc:`/theory/methods/sn/index` — discrete-ordinates :math:`S_N`, which
     realizes this discretization in both space and angle.
   - :mod:`orpheus.transport.spatial` — the method-generic scheme layer
     (:class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`,
     :class:`~orpheus.transport.spatial.diamond.DiamondDifference`,
     :class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`).
   - :mod:`orpheus.derivations.discrete.sn.balance` — the SymPy
     algebra-of-record for the cell balance, DD/WDD, curvilinear balance, and
     the flat-flux identity (7 foundation-verified derivations).
   - :mod:`orpheus.derivations.discrete.sn.ld_ubld` — the SymPy
     algebra-of-record for the Linear-Discontinuous :math:`2\times2` system and
     its :math:`d`-dimensional Kronecker generalization.
