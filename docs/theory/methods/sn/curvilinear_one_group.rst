.. _sn-curvilinear-one-group:

Curvilinear one-group: angle enters the walk
============================================

This chapter opens Part B of the S\ :sub:`N` book: **curvilinear
geometry** — the 1-D sphere and the axially-infinite cylinder — at one
energy group.  Exactly one thing is new.  In Cartesian geometry a
neutron's direction is constant in flight, so :term:`ordinates <ordinate>` couple only
through the sources; on a curved coordinate frame the *same straight
flight line* changes its **local** direction coordinate as it advances
— the radial cosine (:math:`\mu` on the sphere, :math:`\eta` on the
cylinder) drifts toward its most-outward value as the ray passes its
point of closest approach.  The transport operator therefore gains an
**angular redistribution** term — a conservative derivative in the
direction coordinate — and ordinates become **sequentially coupled**,
along the full :math:`\mu` sequence (sphere) or within each
:math:`\mu`-level (cylinder).  The walk acquires an angular thread.

The chain of the book repeats on the new axis:

1. **the invariant** — sinks = sources on a cell that is now a shell
   :math:`\times` an angular cell: curvature moves neutrons between
   neighbouring ordinates without absorbing them, so on a flat
   isotropic flux the redistribution must cancel the streaming *per
   ordinate* → pose the conservative balance :eq:`conservative-form`
   and force the geometry factor :math:`\Delta A/w` from that
   invariant;
2. **the operator** — :math:`A = L + C - S - N_{2n} - B`
   (:eq:`sn-within-group-with-n2n`) keeps its honest shape
   and :math:`L+C` stays **lower-triangular**: :term:`sweep` order now threads
   the angular cascade (:math:`\alpha_{1/2} = 0` **seeds** the recursion at
   each level's :term:`starting direction`; :math:`\alpha_{M+1/2} = 0`
   **closes** it at the far edge — two different statements, and only the
   first is an axiom) through the
   radial march;
3. **the matrix picture** — the per-cell system gains one more
   upstream state (the angular half-flux :math:`\psi_{n-1/2}`) and
   one more closure weight (the Morel--Montry :math:`\tau`);
4. **the strategy encodings** — the cell-update contract
   (:ref:`cell-update-strategies`) is unchanged: the same Step/DD/LD
   algebra serves slab and curvilinear alike, only the populated
   fields of the streaming packet change.

What does *not* change: space stays 1-D radial (the multi-D walk of
:doc:`cartesian_multid` is not needed), the energy axis stays a single
group (multigroup returns in :doc:`curvilinear_multigroup`), and the
closure machinery is still the generic
:doc:`/theory/foundations/discretization`, cross-linked never
re-derived.

.. admonition:: Key Facts
   :class: tip

   * The curvilinear balance :eq:`balance-general` adds two structures
     to the slab balance: the **redistribution cascade**
     (:eq:`alpha-recursion`, :math:`\alpha_{1/2} = 0`, a non-negative
     dome peaking near :math:`\mu = 0`) and the **geometry factor**
     :math:`\Delta A_i/w_n` — forced by per-ordinate flat-flux
     consistency, not optional (without it the solver manufactures
     angular anisotropy that *worsens* under refinement near
     :math:`r = 0`; failure mode #3).
   * The dome **closes**, :math:`\alpha_{M+1/2} = 0`
     (:eq:`alpha-dome-closure`), on both arms — a *theorem about the
     quadrature*, not an axiom of the one-sided recursion, and therefore an
     **admission contract** a bad rule can violate.  It raises at
     construction (:ref:`sn-alpha-dome-closes`).
   * The **streaming-equilibrium identity** :eq:`streaming-equilibrium`
     (:math:`\phi = Q/(\Sigma_t(1-c))`, :math:`\psi_n = \phi/W`) is
     the canonical L0 gate — asserted **per ordinate**, never via
     particle balance (telescoping sums hide per-ordinate errors; vv
     anti-pattern #8).
   * The angular closure is **weighted diamond** :eq:`wdd-closure`
     with **Morel--Montry weights** :eq:`mm-weights`: :math:`\tau` is
     the barycentric coordinate of the ordinate inside its own angular
     cell (predicate P2), the unique weight exact for a flux affine in
     the radial cosine.  **No clamp, in either arm** — the sphere
     since W1 (2026-06-13), the cylinder since Q5.6.4 (2026-08-11).
     The :math:`[0, 1]` membership RAISES at the producer (predicate
     P3: an ordinate outside its own angular cell = an ill-posed
     march).  ⭐ :math:`\tau` carries **no geometry**; the geometry is
     in the **cell partition**
     (:eq:`angular-cell-partition`) — cumulative WEIGHT on the sphere,
     the midpoint in :math:`\omega` on the cylinder.  On the σ_y fold
     :math:`\tau \in [\tfrac14, \tfrac34]`
     (:eq:`morel-montry-folded-arc`).
   * :math:`\tau` is an **angular-scheme property the closure owns**
     (#236 Phase 2): produced solely by the angular closure, and the
     stateless spatial scheme never sees it.  ⛔ Until P4.9a
     (2026-08-28) it travelled to the scheme as
     :class:`~orpheus.transport.spatial.scheme.CellVisit` **data**
     (:math:`c_{\rm in}`, :math:`c_{\rm out}`, :math:`\tau`) stamped at
     one production site; those three fields and the stamping method are
     **retired** — what the scheme receives now is two already-multiplied
     contributions, ``angular_denom_term`` and ``angular_numer_upstream``
     (:ref:`sn-p49a-closure-owns-the-march`).  Either way, never a closure
     dependency — the spatial :math:`\otimes` angular separation.
   * That separation is a **theorem**, not a convention: the
     redistribution operator is the tensor product
     :eq:`sn-redistribution-tensor-product-eq`, so :math:`\tau`
     **keeps its per-ordinate arity under any spatial scheme**, and the
     diffusion-limit condition stays the identical angular scalar
     :eq:`sn-contamination-factor`
     (:ref:`sn-scheme-vs-angular-weight`).  ⛔ The risk a two-moment
     spatial scheme brings is not :math:`\tau` — it is the **seed**
     (:ref:`sn-seed-cone-risk`), which carries no angular redistribution
     at all and so inherits its scheme's positivity ladder undamped.
   * The error axes obey the **geometry-split law**
     :eq:`sn-space-angle-separability`: Cartesian **separates**
     (:math:`E \approx E_{\rm space} + E_{\rm angle}`), curvilinear
     **gates** (:math:`E \approx \max(E_{\rm space}, E_{\rm
     angle})`) — fine-:math:`h` accuracy cannot be harvested at
     coarse :math:`N`; pinned by the ST5 characterisation gate.
   * The solved per-cell form is :eq:`dd-solve` — the WDD closure
     substituted into the balance: one more upstream state
     (:math:`\psi_{n-\frac12}`) in the numerator and the
     :math:`(\Delta A_i/w_n)\,c_{\rm out}` shift in the
     denominator, relative to the slab update.
   * The sweep is **sequential in angle**: cell-by-cell,
     ordinate-by-ordinate — the spatial face flux propagates to the
     next cell, the angular face flux to the next ordinate on the
     same cell.  The production walk is
     :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk` +
     :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update`,
     with the vectorized CumprodScan fast path.
   * The pole/axis is **intrinsic geometry** (a coordinate-system
     singularity, not a BC): the angular closure is the strategy ABC
     :class:`~orpheus.sn.angular.closure.AngularClosureBase`,
     whose sole production strategy is the Morel--Montry **weighted**
     angular recurrence :eq:`pole-mm-recurrence` (:cite:`MorelMontry1984`;
     implemented form :cite:`BaileyMorelChang2010` Eqs. (42)/(43) —
     Hébert defines no :math:`\tau`, see
     :ref:`sn-tau-source-of-record`), **seeded at the marched**
     :math:`\psi_{1/2}` (the bullet below; the Phase-B
     :math:`\phi_{1/2} = 0` was a wrong-term initialisation, not a
     convention).
   * The march has **two** endpoint conditions, not one — the
     redistribution coefficient vanishes at both
     (:math:`\alpha_{1/2} = \alpha_{M+1/2} = 0`), so production ODE-solves
     both ends with one engine and the recurrence *also* predicts the far
     one.  Their disagreement is the named residual
     :math:`D` (:eq:`sn-angular-endpoint-defect-eq`,
     :ref:`sn-angular-endpoint-defect`): a reference-free **consistency**
     diagnostic, ⛔ **not** an error estimator and ⛔ **not** a voter on
     :math:`\tau`.
   * The apply matvec is **one sweep iteration semantically**
     (:eq:`phase-c-wdd-recurrence`): apply and solve consume the
     same three primitives — the WDD closure, the direction-keyed
     DAG walk, and the BC trace law at the boundary edge — so the
     two paths to the loss operator :math:`L+C` agree by
     construction.
   * The starting-direction flux :math:`\psi_{1/2}` is
     **first-class typed state** (System B): route (a) (#282)
     marches it directly from the true within-group source through
     ``RadialCharacteristicOperator.solve``, making the cold
     within-group solve a genuine single-pass exact inverse (sphere
     cold-start residual :math:`5.18\times 10^{5} \to 2.5\times
     10^{-16}`; seed-insensitivity :math:`\to 0` bitwise).

.. _balance-curvilinear:

Curvilinear Balance Equation (Spherical and Cylindrical)
=========================================================

Derivation from the Continuous PDE
------------------------------------

Start with the general 1D curvilinear transport equation.  In
conservative form for a coordinate :math:`r` with face area
:math:`A(r)` and volume element :math:`V`:

.. math::
   :label: conservative-form

   \frac{\mu_n}{V_i}
   \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   + \frac{1}{V_i}
   \bigl[\alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}\bigr]
   + \Sigt{} \psi_{n,i} = S_i

.. vv-status: conservative-form documented

where the streaming cosine is :math:`\mu_n` for spherical and
:math:`\eta_m` for cylindrical, and :math:`S_i = Q_i / W` is the
isotropic source density divided by the quadrature weight sum.

**Step 1: Integrate the PDE over a spatial cell.**

For spherical geometry, integrating :eq:`transport-spherical` over the
shell :math:`[r_{i-1/2}, r_{i+1/2}]` and using the divergence theorem
on the radial streaming gives:

.. math::

   \mu_n \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   + \int_{V_i} \frac{1-\mu^2}{r} \frac{\partial\psi}{\partial\mu}\, dV
   + \Sigt{} V_i \psi_{n,i} = S_i V_i

For cylindrical geometry, integrating :eq:`transport-cylindrical` over
the annular shell gives:

.. math::

   \eta_m \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   - \int_{V_i} \frac{1}{r} \frac{\partial(\xi\psi)}{\partial\varphi}\, dV
   + \Sigt{} V_i \psi_{m,i} = S_i V_i

**Step 2: Discretise the angular redistribution.**

The angular integral is discretised as a finite difference in the
ordinate index.  For spherical:

.. math::

   \int_{V_i} \frac{1-\mu^2}{r}\frac{\partial\psi}{\partial\mu}\, dV
   \;\approx\;
   \alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}

For cylindrical (per :math:`\mu`-level):

.. math::

   -\int_{V_i} \frac{1}{r}\frac{\partial(\xi\psi)}{\partial\varphi}\, dV
   \;\approx\;
   \alpha_{m+\frac12}\psi_{m+\frac12} - \alpha_{m-\frac12}\psi_{m-\frac12}

**Step 3: Apply the geometry factor** :math:`\Delta A / w`.

The raw discretisation above does NOT preserve per-ordinate flat-flux
consistency.  The correct form from :cite:`BaileyMorelChang2010` includes the
geometry factor :math:`\Delta A_i / w_n`:

.. math::
   :label: balance-general

   \mu_n
   \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   + \frac{\Delta A_i}{w_n}
   \bigl[\alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}\bigr]
   + \Sigt{} V_i \psi_{n,i} = S_i V_i

where :math:`\Delta A_i = A_{i+1/2} - A_{i-1/2}`.  This is the curvilinear
balance form of :cite:`BaileyMorelChang2010` for both spherical and cylindrical
geometry.

Note why :eq:`dd-cartesian-1d` has no :math:`\alpha` or :math:`\Delta A`
terms: in Cartesian geometry the face area is unity (:math:`A = 1`), so
:math:`\Delta A = 0`, and there is no curvature to redistribute angular
flux.

The Alpha Redistribution Coefficients
======================================

The :math:`\alpha` coefficients encode how the :term:`angular flux` redistributes
between neighbouring ordinates due to the geometry curvature.  They are
defined recursively:

.. math::
   :label: alpha-recursion

   \alpha_{n+\frac12} = \alpha_{n-\frac12} - w_n \mu_n

seeded at :math:`\alpha_{1/2} = 0`.  The recursion is **strictly
one-sided** — it marches from the seed and never consults the far end —
which is why the far end is a separate statement and not a corollary
(:ref:`sn-alpha-dome-closes`).

For **spherical** geometry, all :math:`N` ordinates form a single
sequence sorted by :math:`\mu` (most negative to most positive).
The :math:`\alpha` values form a **non-negative dome**: they rise while
:math:`\mu < 0`, peak near :math:`\mu = 0`, and fall back to zero at
:math:`\mu = 1`.  The endpoint condition
:math:`\alpha_{N+1/2} = 0` is guaranteed by Gauss--Legendre
antisymmetry: :math:`\sum_n w_n \mu_n = 0`.

For **cylindrical** geometry, each :math:`\mu`-level has its own
independent :math:`\alpha` sequence.  On level :math:`p`, the ordinates
are sorted by increasing :math:`\eta` (radial direction cosine), and the
recursion uses :math:`\eta` instead of :math:`\mu`:

.. math::
   :label: alpha-cylindrical

   \alpha_{p,m+\frac12} = \alpha_{p,m-\frac12} - w_m \eta_m

This is the :cite:`BaileyMorelChang2010` curvilinear :math:`\alpha`-recursion.
Each level's :math:`\alpha` values form an independent dome from
:math:`\eta = -\sin\theta` to :math:`\eta = +\sin\theta`.

**Dome shape properties:**

- :math:`\alpha_{n+1/2} \geq 0` for all :math:`n` (non-negative dome).
- The peak occurs near the ordinate where :math:`\mu_n` (or
  :math:`\eta_m`) crosses zero.
- The dome height scales with the quadrature weight sum: higher-order
  quadratures have narrower but taller domes.
- Non-negativity ensures the denominator of the DD equation is
  unconditionally positive, guaranteeing numerical stability.
- **The dome CLOSES**: :math:`\alpha_{M+1/2} = 0` at the far end of every
  level, on **both** arms — see :ref:`sn-alpha-dome-closes`, which is where
  that belongs, because it is an admission contract rather than a shape.

The code stores the dome on
:class:`~orpheus.sn.angular.redistribution.AngularRedistribution`, as
:attr:`~orpheus.sn.angular.redistribution.AngularRedistribution.alpha_per_level`
— **per** :math:`\mu`\ **-level on both charts**, each of shape
``(M_p + 1,)``, with the sphere as the one-level case and Cartesian
carrying the neutral zero dome.  It is reached as
``mesh.reduced.angular.alpha_per_level``.

.. note:: **Until 2026-08-26 the dome lived directly on the reduced
   streaming operator** as ``alpha_half`` (spherical, shape ``(N+1,)``) and
   ``alpha_per_level`` (cylindrical, a list), and the stated reason for
   keeping it there was that it is *"genuinely geometric"*.  That reason
   was wrong, and the factorization
   :eq:`sn-redistribution-tensor-product-eq` says why: the dome is the
   **angular** factor, not the spatial one — it is a function of
   ``(quadrature, coord)`` alone, with no cell, mesh or material in it.
   What stays on the streaming operator is the **spatial** chart data
   (``face_areas``, ``delta_A``).  See
   :ref:`sn-redistribution-tensor-product`; the six per-coordinate
   ``Optional`` fields the move retired are why Cartesian now spells the
   neutral element instead of ``None``.

One body computes them —
:func:`~orpheus.sn.angular.redistribution.alpha_dome` — called from the
single producer
:func:`~orpheus.sn.angular.redistribution.angular_redistribution` (once per
:math:`\mu`-level), and delegated to by the derivations-side name of the
same function; before ``bea6a367`` (2026-08-12) the recursion had **three**
spellings, and that is exactly why its closure contract could live on one
of them only.

.. _sn-alpha-dome-closes:

The dome closes — :math:`\alpha_{M+1/2} = 0` as an admission contract
----------------------------------------------------------------------

The recursion :eq:`alpha-recursion` is one-sided: given
:math:`\alpha_{1/2} = 0` and the marching cosines it produces *some*
:math:`\alpha_{M+1/2}`, whatever the rule happens to be.  Telescoping the
recursion over the level gives that value in closed form,

.. math::
   :label: alpha-dome-closure

   \alpha_{M+\frac12} \;=\; \alpha_{\frac12} \;-\; \sum_{m} w_m\,c_m
   \;=\; -\sum_{m} w_m\,c_m ,

where :math:`c_m` is the level's **marching cosine** (:math:`\mu` on the
sphere, :math:`\eta` on a cylinder level).  So the dome closes **iff the
marching cosines are antisymmetric about the level's centre**, i.e. iff the
measure's first moment in the marching coordinate vanishes.  Two readings
follow, and keeping them apart is the whole point:

* :math:`\alpha_{1/2} = 0` is an **axiom** — a definition that seeds the
  recursion, true by fiat in every published convention
  (:ref:`normalization-alpha-crosswalk`).
* :math:`\alpha_{M+1/2} = 0` is a **theorem about the quadrature**, and
  therefore a **contract the quadrature can violate**.  A non-zero endpoint
  does not mean the recursion is wrong; it certifies that the measure
  handed to it is inadmissible — mis-ordered, truncated, or duplicated
  ordinates upstream.

.. vv-status: alpha-dome-closure documented
   Rationale: the telescoped closed form of the one-sided Lathrop–Carlson
   recursion at the far endpoint — a representational identity of the
   recursion, not a solver claim with an L0..L3 ladder slot.  The verifiable
   content is the admission guard
   ``orpheus.sn.angular.redistribution._assert_alpha_dome_closes`` (a real
   raise, per level on the cylinder), gated by an explicit positive+negative
   pair in ``tests/geometry/test_reduced_operator.py`` (vv-principles #11):
   ``test_every_shipped_gauss_legendre_dome_closes`` and
   ``test_every_shipped_folded_product_dome_closes_on_every_level`` admit
   every shipped rule on both arms (and assert the PRODUCERS' ~1e-16 floor
   separately from the guard's 1e-12 admission band, vv-principles #16),
   while ``test_a_dome_that_does_not_close_is_refused`` requires the refusal
   under ``python -O`` — the row the retired bare ``assert`` could not carry.

**Why it must REFUSE rather than absorb.**  :math:`\alpha_{M+1/2}` is not
inert.  It is the last entry of the ``alpha_out`` slice, so it lands in
*both* cell-balance coefficients of the final ordinate —
:math:`c_{\rm out}[M-1] = \alpha_{M+1/2}/\tau` (a **denominator** term) and
:math:`c_{\rm in}[M-1] = \bigl((1-\tau)/\tau\bigr)\alpha_{M+1/2} +
\alpha_{M-1/2}` (an upstream-numerator term); see
:ref:`sn-closure-c-constants-owned`.  A closing dome makes that denominator
term vanish, which is what "angular redistribution stops at the level's top
edge" *means* mechanically.  A non-zero value instead redistributes flux
past the top edge, into nothing — a leak, not a small error.

.. warning:: **The contract existed on one arm and did not run
   (``bea6a367``, 2026-08-12).**

   Enforcement before that commit was: sphere — a bare
   ``assert abs(alpha[N]) < 1e-12``; cylinder — nothing at all.  The
   canonical test invocation in this project is ``python -O``, and ``-O``
   sets ``__debug__ = False`` and strips every ``assert`` **statement** at
   compile time, so the one check that existed did not run in the suite that
   matters.  `[M]` on the verbatim recursion: a measure closing at
   ``alpha[N] = +0.2000`` is REFUSED under plain ``python`` and **ACCEPTED**
   under ``python -O``.

   The reusable lesson is not "add a check to the cylinder arm" — that
   would have guarded a *duplicate*.  The recursion had three spellings,
   which is precisely how a contract comes to live on one of them; Cardinal
   Rule 2 first (one :func:`~orpheus.sn.angular.redistribution.alpha_dome`
   body), then the guard
   (:func:`~orpheus.sn.angular.redistribution._assert_alpha_dome_closes`, a
   real ``raise``, **per level** on the cylinder so the failure is
   locatable).  Recursion and contract stay separate functions deliberately:
   a derivation that wants to *study* a non-closing measure may call the
   recursion without being refused by it.  Sibling guard, same shape:
   ``_assert_tau_within_unit_interval`` in
   :mod:`orpheus.sn.angular.closure`.

   ⭐ **The argument completed on 2026-08-26.**  ``bea6a367`` gave the
   recursion one body; the guard still ran from whichever streaming
   factory remembered to call it.  Since the α-dome moved onto
   :class:`~orpheus.sn.angular.redistribution.AngularRedistribution` it
   has exactly **one producer**,
   :func:`~orpheus.sn.angular.redistribution.angular_redistribution`, and
   the contract is checked there — at the site that mints the value, for
   every chart, rather than on the arms that happen to consume it.  A
   guard on a *duplicate* and a guard on a *consumer* fail the same way;
   only a guard at the single source cannot be bypassed by a new caller.

   ⚠ Generalise it: **a numerical or domain contract expressed as a bare
   ``assert`` in** ``orpheus/`` **does not run under the canonical runner.**
   The discriminator is what the assert is *for* — type-narrowing for a type
   checker is fine to strip; an admission predicate must be a real
   ``raise``.  (``.claude/rules/coding-standards.md``; ``vv-principles``
   Mode 8.)

.. _sn-geometry-factor:

The Geometry Factor and Why It Is Needed
=========================================

The geometry factor :math:`\Delta A_i / w_n` in :eq:`balance-general`
is the key to correct curvilinear transport.  Without it, the balance
equation violates **per-ordinate flat-flux consistency**: for a spatially
uniform, isotropic flux :math:`\psi = \text{const}`, the streaming and
redistribution terms should cancel exactly for EACH ordinate
individually.

**Proof of consistency.**

Set :math:`\psi_{n,i} = \psi_{n+1/2} = \psi_{n-1/2} = \psi_0` (flat
in both space and angle) and :math:`\psi_{i+1/2} = \psi_{i-1/2} = \psi_0`
(flat in space).  The streaming term becomes:

.. math::

   \mu_n \bigl[A_{i+\frac12} - A_{i-\frac12}\bigr] \psi_0
   = \mu_n \,\Delta A_i\, \psi_0

The redistribution term with the :math:`\Delta A/w` factor becomes:

.. math::

   \frac{\Delta A_i}{w_n}
   \bigl[\alpha_{n+\frac12} - \alpha_{n-\frac12}\bigr] \psi_0
   = \frac{\Delta A_i}{w_n} (-w_n \mu_n) \psi_0
   = -\mu_n \,\Delta A_i\, \psi_0

where we used the recursion :eq:`alpha-recursion`:
:math:`\alpha_{n+1/2} - \alpha_{n-1/2} = -w_n \mu_n`.  The two terms
cancel exactly, giving :math:`\Sigt{} \psi_0 = S_0`, which is the
correct homogeneous solution.

**Without** the :math:`\Delta A/w` factor (i.e., using
:math:`[\alpha_{n+1/2}\psi_{n+1/2} - \alpha_{n-1/2}\psi_{n-1/2}]`
directly), the redistribution term for flat flux is
:math:`(-w_n \mu_n)\psi_0`, but the streaming term is
:math:`\mu_n \Delta A_i \psi_0`.  These differ by a factor of
:math:`\Delta A_i`, so consistency only holds in the limit
:math:`\Delta A_i \to 0` (i.e., at the origin or on an infinitely fine
mesh).

**Consequence of the missing factor:**  The solver creates artificial
angular anisotropy that *worsens* with mesh refinement near :math:`r = 0`
(where :math:`\Delta A_i` is smallest but non-zero).  This manifests as
a flux spike at the origin in fixed-source problems and as divergent
eigenvalues in heterogeneous eigenvalue problems.

**No shared store holds this factor — each consumer builds its own, from
the two factors.**  The **spatial** one is
:attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.delta_A`
(shape ``(nx,)``, on the streaming operator); the **angular** one is
:math:`1/w_n`, the measure's own weight, recovered through the operator's
bound angular axis —
:attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.angular_axis`,
whose :attr:`~orpheus.numerics.axis.Axis.generator` is the quadrature.
(The ``AngularRedistribution.quadrature`` courier that used to carry it
retired at the P4-remainder, 2026-08-29 — the angular factor is pure
α data now.)

⛔ This paragraph opened *"Nothing precomputes this factor.  Each consumer
forms it where it is used"* until 2026-08-29.  The first sentence became
false at P4.9a and the second at P4.7, and the two halves are worth
separating because only one of them was ever the point.  There are three
formers today and **two of them precompute**: the angular closure mints
``_dAw_per_level`` once at construction (P4.9a), the sweep's scan cache
interns a chain-ordered row at build (P4.7), and only the degenerate
cylindrical arm of the walk still forms it per ``(cell, ordinate)`` at
use.  What survives — and what the factorization argument below actually
claims — is that no *shared* array holds the product: the split is **by
factor**, each consumer owns its own fusion at its own lifetime, and
nobody reads a stored :math:`\Delta A \otimes 1/w` off an object that
owns neither factor.  "Not cached" was never the claim; "not cached
*somewhere neither consumer owns*" was.

.. note:: ⭐ **The fused product was retired on 2026-08-26, and the reason
   is instructive.**  Until then the geometry object cached
   ``redist_dAw`` (spherical, ``(nx, N)``) and ``redist_dAw_per_level``
   (cylindrical, a list of ``(nx, M)``) — the *product*
   :math:`\Delta A_i \otimes 1/w_n` of a geometric factor with a
   quadrature factor, stored on the geometry side and read by two
   consumers that each wanted **a different one of the two**.  So neither
   side owned the fusion, and the cache was a second spelling of a
   quantity that belongs to neither.  The factorization
   :eq:`sn-redistribution-tensor-product-eq` is the reason the split is
   the right one: :math:`\Delta A_i` is the :math:`(0,0)` corner of the
   *spatial* Gram :math:`R` and :math:`1/w_n` is part of the *angular*
   operator :math:`A_{\rm angular}`, and the two carry disjoint indices.
   The product still appears in the algebra — it is
   :eq:`balance-general`'s own coefficient — but as an expression at the
   point of use, not as state.

The Streaming-Equilibrium Identity (canonical L0 gate)
=======================================================

The flat-flux consistency proof above is the *per-cell* statement of a
*global* exact solution that the verification suite leans on harder
than any other: for a *homogeneous* medium with a *uniform isotropic*
source :math:`Q` per group and boundaries that sustain a flat flux
(reflective faces, or an infinite/periodic medium), the discrete
transport equation has the exact fixed point

.. math::
   :label: streaming-equilibrium

   \phi \;=\; \frac{Q}{\Sigma_t\,\bigl(1 - c\bigr)},
   \qquad
   \psi_n \;=\; \frac{\phi}{W}
   \quad \forall n,
   \qquad
   W \equiv \sum_n w_n,
   \quad
   c \equiv \frac{\Sigma_{s0}}{\Sigma_t},

per group. For a pure-attenuation configuration (no scattering in the
residual, :math:`c = 0`) this reduces to :math:`\phi = Q/\Sigma_t` with
the per-ordinate angular flux :math:`\psi_n = Q/(W\,\Sigma_t)`.

**Why this is the canonical L0 gate.** Substituting the flat
:math:`\psi_n` into the discrete balance :eq:`balance-general` nulls
the streaming and redistribution terms *per ordinate* (the proof of
consistency above), leaving the pure collision balance
:math:`\Sigma_t\,\psi_n = Q/W + \Sigma_{s0}\,\phi/W`. Every term that
a discretisation can get wrong — a missing :math:`\Delta A/w` factor
(failure mode #3, the flux spike at :math:`r=0`), a wrong
:math:`\alpha` recursion (mode #4), a face-index slip (mode #5), a
weight-normalisation drift (:math:`1/W` vs :math:`1/4\pi`) — breaks
the identity at machine precision, with no discretisation error to
hide behind. The assertion is **per-ordinate**, never
particle-balance: telescoping global balance holds by construction
even when per-ordinate balance is wrong (the canonical ERR-006 hide;
vv-principles anti-pattern #8).

The identity holds in every geometry ORPHEUS supports (slab, sphere,
cylinder, 2-D/3-D Cartesian) and at both algebraic access points —
the sweep (``solve``: given :math:`Q`, recover the flat
:math:`\psi`) and the matvec (``apply``: given the flat :math:`\psi`,
recover the residual :math:`Q` with no spurious boundary or pole
contribution). Tests declare it via
``@pytest.mark.verifies("streaming-equilibrium")``.

The Morel--Montry Flux Dip
============================

Even with the correct :math:`\Delta A/w` factor, the standard
:term:`diamond-difference <diamond difference>` closure (equal weight :math:`\tau = 0.5`) introduces
a flux error near :math:`r = 0` known as the **Morel--Montry flux dip**
:cite:`MorelMontry1984`.

The standard DD angular closure is:

.. math::

   \psi_{n,i} = \frac{1}{2}(\psi_{n-\frac12} + \psi_{n+\frac12})

This can be rewritten as:

.. math::

   \psi_{n+\frac12} = 2\psi_{n,i} - \psi_{n-\frac12}

The contamination factor :math:`\beta` (:cite:`BaileyMorelChang2010`) quantifies
the coupling between the leading-order :term:`scalar flux` and the first-order
current in the asymptotic diffusion limit.  For spherical geometry:

.. math::
   :label: sn-contamination-factor

   \beta = \frac{1}{2} \sum_{n=1}^{N} \mu_n
   \bigl[\alpha_{n+\frac12}\, \mu_{n+\frac12}
        - \alpha_{n-\frac12}\, \mu_{n-\frac12}\bigr]

where :math:`\mu_{n\pm 1/2}` are the angular cell-edge cosines.  For
cylindrical, the equivalent is a per-level sum using :math:`\eta` and
:math:`\eta_{m\pm 1/2}`.  When :math:`\beta \neq 0`, the discrete
S\ :sub:`N` equations satisfy a **contaminated** diffusion equation near
:math:`r = 0`, producing the artificial flux dip (or spike).

The module :mod:`orpheus.derivations.discrete.sn.angular_differencing`
computes :math:`\beta` for any quadrature and geometry
(:func:`~orpheus.derivations.discrete.sn.angular_differencing.contamination_beta`).
With the correct :math:`\Delta A/w` factor AND Morel--Montry weights,
:math:`\beta \sim 10^{-16}` (machine zero) for both spherical and
cylindrical.

.. _sn-tau-beta-diagnostic-blind:

.. warning:: **β is BLIND on a σ_y-folded cylindrical arc — never gate a
   cylinder partition on it** (`[M]` 2026-08-11, Q5.6.4).

   On the fold, β is a **symmetry identity**, not a measurement.  The
   fold makes the nodes antisymmetric (`[M]`
   :math:`\max|\eta + \eta_{[::-1]}| = 0`) and the α-dome symmetric
   (`[M]` 2.78e-17), so for ANY antisymmetric edge set the terms cancel
   pairwise.  Measured at :math:`n_\varphi = 16`, level 0: the production
   edges give :math:`+6.94\mathrm{e}{-18}`, edges scaled
   :math:`0.5\times` give :math:`+3.47\mathrm{e}{-18}`, edges **CUBED**
   give :math:`+1.73\mathrm{e}{-18}`, and a **random antisymmetrised**
   set gives :math:`-3.47\mathrm{e}{-18}`.  Only breaking antisymmetry
   moves it (one edge nudged: :math:`-3.53\mathrm{e}{-3}`).  ⟹ β sees
   *only* whether the edges are antisymmetric; garbage passes.  It also
   certified the cumulative-weight convention that `[M]` DIVERGES the
   solve.  The fold — the campaign's own achievement — annihilated the
   functional that would have judged what it enabled
   (``vv-principles`` Mode 12).  The instrument that DOES discriminate,
   solve-free, is the **ν-closure** residual: see
   :ref:`sn-tau-absorber-retirement`.

   ⚠ Also: :math:`\beta` is overloaded.  This one is BMC's *contamination*
   coefficient — ONE SCALAR per level, zero iff :math:`\tau` is the
   Morel--Montry weight.  Lathrop's :math:`\beta` is the *α-defect* — a
   SEQUENCE, zero iff :math:`\tau \equiv \tfrac12`, the diamond scheme.
   The two are near-opposites at different orders, and reading one as the
   other cost a full design cycle on 2026-08-11.  Both are implemented,
   side by side and named apart, in
   :mod:`orpheus.derivations.discrete.sn.angular_differencing`.

Weighted Diamond Difference (WDD) and Morel--Montry Weights
=============================================================

The Morel--Montry (M-M) angular closure replaces the equal-weight DD
with position-dependent weights :math:`\tau_n` :cite:`BaileyMorelChang2010` Eq. 43:

.. math::
   :label: wdd-closure

   \psi_{n,i} = (1 - \tau_n)\,\psi_{n-\frac12} + \tau_n\,\psi_{n+\frac12}

Solving for the angular face flux:

.. math::
   :label: wdd-face

   \psi_{n+\frac12}
   = \frac{\psi_{n,i} - (1 - \tau_n)\,\psi_{n-\frac12}}{\tau_n}

The M-M weights are defined as:

.. math::
   :label: mm-weights

   \tau_n = \frac{\mu_n - \mu_{n-\frac12}}{\mu_{n+\frac12} - \mu_{n-\frac12}}


.. implements:: mm-weights
   :by: orpheus.sn.angular.closure._assert_tau_within_unit_interval

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: mm-weights
   :by: orpheus.sn.angular.closure.angular_cell_edges_per_level

.. implements:: mm-weights
   :by: orpheus.sn.angular.closure.morel_montry_tau_per_level

where :math:`\mu_{n\pm 1/2}` are the angular cell edges.

**Spherical cell edges:**  :math:`\mu_{1/2} = -1`,
:math:`\mu_{N+1/2} = +1`, and interior edges by weight-sum:
:math:`\mu_{n+1/2} = \mu_{n-1/2} + w_n`.  This is exact for
Gauss--Legendre quadrature because the weights correspond to the
:math:`\mu`-space widths of the angular cells.

**Cylindrical cell edges:**  the **midpoint in** :math:`\omega`, with
the level's endpoints at :math:`\omega = \pi` and :math:`\omega = 0`
(hence :math:`\eta = \mp\sin\theta`):
:math:`\eta_{m+1/2} = \sin\theta\,\cos\bigl(\tfrac12(\omega_m +
\omega_{m+1})\bigr)`.  The azimuthal march is a march in
:math:`\omega`, arc by arc, so the cell boundary is the midpoint *in the
variable the march marches in*; on an equispaced-:math:`\omega` rule this
is exactly the half-angle boundary :math:`\omega_m \pm \Delta\omega/2`.
The weight-sum approach is NOT used for cylindrical because the
quadrature weights are uniform in :math:`\varphi`-space (not
:math:`\eta`-space): the product quadrature spaces :math:`\varphi`
equally, but :math:`\eta = \sin\theta\cos\varphi` is
cosine-distributed, so equal :math:`\varphi`-widths map to unequal
:math:`\eta`-widths — accumulating weights in :math:`\eta` therefore puts
ordinates outside their own cells and diverges under refinement.  Both
branches, their derivations and the measured refutation of the two
rejected conventions are at
:ref:`the partition's doctrine home <angular-cell-partition-section>`.

.. note:: **Retraction (2026-08-11, Q5.6.4).**  Until Q5.6.4 the
   cylindrical interior edges were taken at the **midpoint of
   consecutive** :math:`\eta` **values**,
   :math:`\eta_{m+1/2} = (\eta_m + \eta_{m+1})/2` — the *chord* midpoint.
   That is the partition above **with its end cells stretched**: every
   interior chord edge equals :math:`\cos(\Delta\omega/2) \times` the arc
   edge (`[M]` to :math:`10^{-16}`) while the endpoints stay unscaled, so
   the implied :math:`\omega`-width spread converges to
   :math:`\approx 17.45\,\%` against a quadrature whose own cells are
   bit-exactly equal.  :math:`\alpha` meanwhile used the *real*
   half-angle edge — one object, two derivations, in disagreement.  That
   :math:`O(1)` inconsistency is what the retired :math:`[\tfrac12, 1]`
   absorber was compensating for; full account at
   :ref:`sn-tau-absorber-retirement`.

   The paragraph that followed it here is also retracted.  It read:
   *"ordinates come in pairs with the same* :math:`|\eta|` *but opposite*
   :math:`\xi` … *the midpoint between paired ordinates equals their
   shared* :math:`\eta`, *creating zero-width angular cells.  The
   resulting* :math:`\tau` *alternates between 0.5 (DD) and 1.0 (step)…
   This alternating pattern is correct."*  It described the **full-circle
   product rule**, which is **inadmissible** at cylindrical ``SNMesh``
   since Q5.6.3 and is now refused *by the partition producer itself* — a
   full-circle level carries :math:`\omega` of both signs (the σ_y double
   cover), so "the midpoint in :math:`\omega`" is undefined for it and
   the rule is rejected by name rather than producing an alternating
   :math:`\tau`.  GitHub Issue #1's *"smooth it with a Gauss-type
   azimuthal rule with distinct* :math:`\eta`" remains a live idea for
   the residual azimuthal floor, but it is no longer needed to avoid
   zero-width cells.

The M-M weights force the contamination factor :math:`\beta` to **machine
zero** (verified: :math:`\beta \sim 10^{-16}`), completely eliminating
the Morel--Montry flux dip — ⚠ subject to the blindness warning above:
on a σ_y-folded arc that reading is a symmetry identity and certifies
nothing about the partition.

**No clamp, in either arm** (sphere since W1, 2026-06-13; cylinder since
Q5.6.4, 2026-08-11).  The weight :eq:`mm-weights` is the unique weight
exact for a flux affine in the radial cosine
(Bailey-Morel-Chang 2010 Eq. 43), admissible range
:math:`\tau \in [0, 1]` — predicate **P3**, enforced since Q5.5: the
producer RAISES on :math:`\tau \notin [0, 1]`, because on a well-posed
monotone march an out-of-range value certifies an ill-posed march:
mis-ordered members (T22's ω-ordered mis-ordering measured
:math:`\tau = 1.079`, which the pre-Q5.5 absorption silently
laundered into a finite wrong answer) or a quadrature incompatible
with the arm's edge convention (a raw 3-D ``level_symmetric`` rule on
the 1-D spherical arm — 23 of 24 :math:`\tau` outside, previously
consumed *unclamped*; #336, and the measured detail at the guard's
doctrine home in :doc:`/theory/foundations/structured_geometry`).

* **Sphere** — unclamped since W1.  On Gauss--Legendre quadrature
  :math:`\tau_n \in [0.39, 0.61]` (never 0), so the closure
  is positive without a clamp.  The former :math:`[0.5, 1.0]` clamp was
  an over-conservative, mis-cited positivity floor that re-floored the
  anisotropic solution; the full vindication is at
  :ref:`sn-curvilinear-aniso-norm-reconciliation`.
* **Cylinder** — unclamped since Q5.6.4.  T27 (2026-08-02) had already
  adjudicated the clip as TWO welded objects: the :math:`[0, 1]`
  membership (promoted to the raising guard at Q5.5) and a
  :math:`[\tfrac12, 1]` **absorption** that is NOT a range statement
  (the sphere runs outside the box, correct).  Its stated purpose was a
  division block for the edge-node march start
  (``on_edge_node`` ⟹ :math:`\tau_0 = 0` bit-exact, Q5.4); Q5.6.3
  (``1689faf4``) made that march start inadmissible, and Q5.6.4 found
  the absorber's real function was compensating the retracted chord
  partition above.  `[M]` no source prescribes any limiter on
  :math:`\tau`, and at :math:`S_8` Gauss--Legendre four of eight M-M
  :math:`\tau` sit below :math:`\tfrac12` — the box was never the
  admissible range in either arm.  On the fold
  :math:`\tau \in [\tfrac14, \tfrac34]` with the reversal identity
  :math:`\tau_m + \tau_{M-1-m} = 1` to 64 ULP.  Derivation,
  measurements, and gates: :eq:`morel-montry-folded-arc` and
  :ref:`sn-tau-absorber-retirement` in
  :doc:`/theory/foundations/structured_geometry`.

:math:`\tau` lives **inside the angular closure**, in
:func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`
— one geometry-free body reading the single partition producer
:func:`~orpheus.sn.angular.closure.angular_cell_edges_per_level`.
The closure exposes the resulting weight per global ordinate through
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`
(spherical: a single :math:`(N,)` array; cylindrical: the per-level
weights gathered to the global ordinate order).  Issue #236 Phase 2
retired the parallel geometry-side :math:`\tau` producer that formerly
baked these weights onto :class:`StreamingTerms`; :math:`\tau` is now
produced **solely** by the closure (see :ref:`sn-tau-c-on-cellvisit-live`).

.. _sn-tau-closure-owned:

τ is an angular-scheme property — the closure owns it
------------------------------------------------------

.. todo:: Archivist expansion needed.

   The Morel--Montry weight :math:`\tau` :eq:`mm-weights` is a function of
   the quadrature :math:`(\mu, w, \text{levels})` ALONE — an
   ANGULAR-scheme property, not a geometry one.  Issue #236 Phase 2
   (Step A) relocates :math:`\tau` PRODUCTION onto the angular closure:
   :func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`
   produces :math:`\tau` from the quadrature the closure is handed —
   since the P4-remainder (2026-08-29) that is the generator of the
   :attr:`~orpheus.numerics.axis.Axis` its constructor takes as a third
   operand, recovered by a typed narrow rather than read off a courier
   field on the angular factor — and
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
   consumes its OWN :math:`\tau` for the matvec contribution (P0) instead
   of reading it back from the streaming-geometry factory
   (:func:`~orpheus.sn.mesh.reduced_operator.spherical_streaming` /
   :func:`~orpheus.sn.mesh.reduced_operator.cylindrical_streaming`).

   Step A was BIT-IDENTICAL: the producer was a 0-ULP line-for-line
   replica of the factory arithmetic *as it stood then* (sphere
   unclamped, cylinder clamped to :math:`[\tfrac12, 1]` — the clamp
   retired at Q5.6.4 and the geometry-side producer at Step C, so
   neither side of that comparison exists any more), so the geometry
   factory still baked an
   IDENTICAL :math:`\tau` for the sweep path while the carve de-risked by
   parallel-run-and-compare.  The producer-equivalence gate (Leg 1)
   ``tests/sn/sweep/curvilinear/test_tau_producer_equivalence.py`` pins
   the closure-produced :math:`\tau` to (a) the geometry-factory value
   (0-ULP) AND (b) an independent reference — at the time
   ``contamination.morel_montry_weights``; see the
   :ref:`reference-migration note <sn-tau-reference-migration>` for what
   the arms compare against now.  The
   Cartesian :class:`~orpheus.sn.angular.closure.IdentityAngularClosure`
   supplies the neutral :math:`\tau = 1` WITHOUT a geometry branch — the
   closure TYPE is the dispatch.

   Step B (a later dispatch) retires the geometry-side
   :math:`\tau` producer and consolidates the four-site
   ``c_in``/``c_out`` duplication onto the closure.

.. _sn-closure-c-constants-owned:

c_in / c_out are angular-closure constants — Step B1 (one site folded)
-------------------------------------------------------------------------

.. todo:: Archivist expansion needed.

   The :term:`weighted-diamond <weighted diamond difference>` constants

   .. math::

      c_{\rm out}[m] &= \frac{\alpha_{m+1/2}}{\tau_m}, \\
      c_{\rm in}[m]  &= \frac{1-\tau_m}{\tau_m}\,\alpha_{m+1/2}
                        + \alpha_{m-1/2}

   are an ANGULAR-closure property: a function of the closure's own
   :math:`\alpha`-dome and :math:`\tau` weight :eq:`mm-weights`
   :eq:`dd-mm-closure-constants`.  Issue #236 Phase 2 Step B consolidates
   the FOUR independent inline rebuilds of this pair onto the closure,
   which already computes it once at construction (per :math:`\mu`-level,
   :math:`(M_p,)` arrays in ``_c_in_per_level`` / ``_c_out_per_level``).

   Step B1 (this dispatch) folds the ONE free seam — the
   :class:`~orpheus.sn.sweep.cache.StreamingCoefficientCache` populator
   (:meth:`~orpheus.sn.sweep.cache.StreamingCoefficientCache.from_mesh_and_quad`),
   which held ``sn_mesh`` and so read
   :attr:`~orpheus.sn.angular.closure.AngularClosureBase.c_out_per_ordinate`
   /
   :attr:`~orpheus.sn.angular.closure.AngularClosureBase.c_in_per_ordinate`
   with zero plumbing.  (Since P4.9b the populator was **handed** its
   closure — ``from_mesh_and_quad(sn_mesh, angular_closure)`` — so the
   mesh supplied geometry and the caller the method; see
   :ref:`sn-p49b-operator-poses-with-closures`.  Since P4b, 2026-08-29,
   the populator takes NO closure at all: the table shed the closure
   block, so every field derives from ``sn_mesh`` alone —
   ``from_mesh_and_quad(sn_mesh)`` — and the walk / σ-build read the
   constants through their handed closure.)  The accessor pair is
   PUBLIC and polymorphic on the base
   :class:`~orpheus.sn.angular.closure.AngularClosureBase`:
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
   returns its precomputed per-level :math:`c` gathered to the
   :math:`(N,)` global-ordinate order; the Cartesian
   :class:`~orpheus.sn.angular.closure.IdentityAngularClosure`
   returns the NEUTRAL zeros (:math:`\alpha=0,\ \tau=1 \Rightarrow c=0`).
   The dispatch is by closure TYPE, not by a ``coord ==`` branch in the
   cache.

   Step B1 is BIT-IDENTICAL: the closure computes :math:`c` from
   closure-:math:`\tau` (0-ULP equal to geometry-:math:`\tau`, pinned by
   the Step-A Leg-1 gate) and the SAME :math:`\alpha` the populator read,
   so the closure's per-level :math:`c` equals the inline :math:`c`
   bit-for-bit; the per-level :math:`\to (N,)` gather is a pure
   permutation (no arithmetic).  The anchor gate
   ``tests/sn/sweep/core/test_cache.py::test_cache_populator_matches_cell_balance_terms``
   pinned the cache ``denom`` (which carries :math:`(\Delta A/w)\,c_{\rm out}`)
   to ``cell_balance_terms`` at
   ``rtol=1e-14`` (both renamed / retired at P4.9a — the gate is now
   ``test_cache_populator_matches_cell_balance_for_streaming``), and the curvilinear regression snapshots stay unmoved.

   The remaining THREE inline ``c`` rebuild sites (they need CellVisit
   threading) are later dispatches (Step B2 / B3 / C).  See
   :mod:`orpheus.sn.angular.closure` for the canonical
   accessor and :mod:`orpheus.sn.sweep.cache` for the folded
   consumer.

.. _sn-closure-c-on-cellvisit:

c_in / c_out reach the stateless DD scheme as CellVisit data — Step B2
-------------------------------------------------------------------------

.. todo:: Archivist expansion needed.

   Step B2 folds the SECOND of the four inline ``c_in`` / ``c_out``
   rebuild sites — the matvec-twin residual
   :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual`
   (formerly rebuilding :math:`c_{\rm out} = \alpha_{\rm out}/\tau`,
   :math:`c_{\rm in} = (1-\tau)/\tau\,\alpha_{\rm out} + \alpha_{\rm in}`
   inline from the geometry-owned :class:`StreamingTerms`).

   The architectural crux: :class:`~orpheus.transport.spatial.diamond.DiamondDifference`
   is deliberately STATELESS — it reads only the
   :class:`~orpheus.transport.spatial.scheme.CellVisit` packet + the
   :class:`~orpheus.transport.spatial.scheme.UpstreamState`, never the mesh or
   the angular closure.  So the closure-owned :math:`c` cannot reach
   ``DD.residual`` by coupling DD to the closure object (that would break
   the spatial :math:`\otimes` angular separation — the SPATIAL scheme
   must not see the ANGULAR closure's type).  Instead the constants travel
   as DATA: the :class:`~orpheus.transport.spatial.scheme.CellVisit` gains two
   angular-closure-owned fields
   (``CellVisit.c_in`` /
   ``CellVisit.c_out``, distinct in
   provenance from the geometry-owned
   :attr:`~orpheus.transport.spatial.scheme.CellVisit.streaming_terms`), and the
   single production site
   ``SNMesh._make_cell_visit`` — through which
   ALL four ``dag_walk`` yield paths funnelled (Pattern 2, no per-site
   divergence) — stamps them from
   :attr:`~orpheus.sn.angular.closure.AngularClosureBase.c_in_per_ordinate`
   /
   :attr:`~orpheus.sn.angular.closure.AngularClosureBase.c_out_per_ordinate`
   indexed by the GLOBAL ordinate (``direction_idx`` for slab / sphere,
   ``level_indices[p][m]`` for cylinder — mirroring
   :meth:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.streaming_terms`).
   ``DD.residual`` then reads ``visit.c_in`` / ``visit.c_out``; the
   :math:`(\Delta A/w)`-scaled assembly that follows is byte-unchanged —
   only the SOURCE of :math:`c` moved.

   Step B2 also completes the matvec's typed-consumer binding (Issue
   #226): the unified SN matvec reads ``sn_mesh.pole_angular_closure``
   typed against the
   :class:`~orpheus.sn.angular.closure.AngularClosureBase`
   ABC and drives the angular path through
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.precompute_psi_state`,
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.cell_contribution`,
   and
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.angular_adjoint`.
   These were declared ``@abstractmethod`` on the ABC (matching the
   precedent where
   :class:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase` declares
   ``update`` / ``residual`` abstract so ``mesh.scheme`` consumers see the
   full contract) — making the ABC the COMPLETE strategy contract instead
   of declaring only ``__call__``.

   ⛔ **Both halves of that sentence have since moved, and the
   B2-era spellings are kept only as history.**  The attribute is
   ``angular_closure`` (P4.9b dropped "pole" from the family), and the
   matvec no longer reads it off the mesh **at all**: the walk consumes
   the closure pair the operator was **posed** with, so the hub route
   carries only two space facts.  See
   :ref:`sn-p49b-operator-poses-with-closures`.  What survives verbatim
   is the *typing* claim this step actually made — the ABC is the
   complete strategy contract, whoever hands the instance over.

   Step B2 is BIT-IDENTICAL: ``visit.c_in`` / ``visit.c_out``
   (closure-sourced) equal the former inline values 0-ULP — the closure
   computes :math:`c` from closure-:math:`\tau` (0-ULP equal to
   geometry-:math:`\tau`, Step-A Leg-1) and the SAME geometry-:math:`\alpha`,
   and the per-level :math:`\to (N,)` gather is a pure permutation.  The
   matvec residual path (fed by ``DD.residual``) stays bit-for-bit on the
   ``tests/sn/sweep/curvilinear/test_unified_matvec_{sphere,cylinder}.py``
   twin and on the DriftWarning-escalating
   ``tests/sn/sweep/core`` + ``tests/sn/solve`` snapshots.  The remaining
   TWO ``c`` rebuild sites
   (``cell_balance_terms`` for the
   ``DD.update`` solve path; the geometry-side :math:`\tau` producer) were
   Step B3 / C.  See :mod:`orpheus.transport.spatial.scheme` for the CellVisit
   fields and :mod:`orpheus.sn.mesh.augmented_mesh` for the production stamp.

   B2 review fixes (finishing pass).  THREE follow-ups landed after the
   carve, all bit-identical (0-ULP):

   * **Per-ordinate gather cached (L16).** The public accessors
     :attr:`~orpheus.sn.angular.closure.AngularClosureBase.c_in_per_ordinate`
     /
     :attr:`~orpheus.sn.angular.closure.AngularClosureBase.c_out_per_ordinate`
     re-ran the full :math:`(N,)` per-level :math:`\to` global gather on
     EVERY access, so the per-visit stamp made the visit-producing loop
     :math:`O(N^2\,n_x)`.  The gather is a pure permutation of immutable
     per-level data, so it is now computed ONCE in each mesh-bound
     ``__init__`` (shared
     :meth:`~orpheus.sn.angular.closure.AngularClosureBase._build_per_ordinate_cache`,
     called by both ``MorelMontryAngularSweep`` and
     ``IdentityAngularClosure``) and the accessors return the read-only
     cache (``setflags(write=False)`` guards the shared :math:`(N,)` view
     consumers hold — until P4b the ``StreamingCoefficientCache``
     populator; since P4b the walk, which binds the arrays per
     sweep).  Measured on a
     ``sphere N=32 nx=200`` walk: :math:`\sim 32\,\text{ms} \to \sim
     22\,\text{ms}` per sweep (:math:`\sim 1.46\times`), value-identical.

   * **Committed production-stamp catcher (vv L11 Mode 11).** The original
     carve had NO committed test exercising ``_make_cell_visit``'s c-stamp
     — the matvec twin reads the closure's ``cell_contribution`` directly
     (never ``DD.residual``), and the diamond / cell-balance fixtures stamp
     visits with a SURROGATE.  A wrong global-ordinate map (a
     ``c_in``:math:`\leftrightarrow`\ ``c_out`` swap, a mis-scattered
     cylinder level block) would ship silently.
     ``tests/sn/sweep/core/test_cell_visit_c_stamp.py`` walked a REAL
     production ``dag_walk`` (sphere + multi-level cylinder + slab) and
     asserts every ``visit.c_in`` / ``visit.c_out`` equals the constants
     recomputed INLINE from that visit's OWN
     :class:`~orpheus.transport.spatial.scheme.StreamingTerms` at 0-ULP
     (the hand-transcribed independent reference, not the closure's own
     ``c`` — vv L11).  Mutation-verified: the ``c_in``\ /\ ``c_out`` swap
     reddens the sphere + cylinder cases.

   * **Test-surrogate dedup (Pattern 2).** The byte-identical
     ``_c_from_streaming_terms`` (``test_diamond.py``) and ``_visit_c``
     (``test_cell_balance_for_streaming.py``) hand-recomputes were unified
     into one shared ``tests/sn/sweep/core/_c_surrogate.py`` consumed by
     both files and the new catcher.

.. _sn-tau-c-on-cellvisit-live:

The live sweep + scan consume closure-owned τ / c — Step B3
---------------------------------------------------------------

Step B1 folded the one free seam (the cache populator); Step B2 carried
the redistribution constants :math:`c_{\rm in}` / :math:`c_{\rm out}`
onto the :class:`~orpheus.transport.spatial.scheme.CellVisit` so the
*apply-direction* residual could read them as data.  Step B3 is the
step that makes the **live** paths consume the closure-owned weight:
the per-cell sweep solve, the matvec solve, and the CumprodScan
fast path now all read the angular weight :math:`\tau` :eq:`mm-weights`
(and the derived constants :eq:`dd-mm-closure-constants`) off the
closure rather than off the geometry-owned ``StreamingTerms.tau_mm``.
After B3 there is **no live reader** of ``StreamingTerms.tau_mm``
anywhere in the sweep, scan, or matvec — precisely the precondition
that let Step C delete the geometry-side :math:`\tau` producer (the two
parallel producers could be reduced to one only once nothing live
depended on the soon-to-be-deleted one; that retirement has now landed
— see the close-out at the end of this section).

This is the third of the four c-fold sites (after the cache populator
in B1 and the residual twin in B2) and the fifth :math:`\tau` consumer.
Like its predecessors it is **bit-identical (0-ULP)**: the carve moves
the *source* of an already-correct number, it does not change the
number.  The sections below derive why :math:`\tau` belongs to the
angular closure, why the constants must travel as visit *data* rather
than as a closure *reference*, why the field default is :math:`1.0`
(and not the more obvious :math:`0.0`), how the three live consumers
share one operator, and what makes the fold provably bit-identical and
therefore a safe regression floor for Step C.

τ is an angular-scheme property the closure owns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Morel--Montry weight

.. math::

   \tau_n = \frac{\mu_n - \mu_{n-\frac12}}{\mu_{n+\frac12} - \mu_{n-\frac12}}

(:eq:`mm-weights`, Bailey--Morel--Chang 2010 Eq. 43) is built **entirely**
from the angular quadrature: the ordinate cosine :math:`\mu_n`, the
neighbouring angular-cell edges :math:`\mu_{n\pm 1/2}`, and — for the
cylinder — the :math:`\mu`-level partition that groups ordinates.  Not
one of those inputs is a property of the *spatial* streaming geometry:
:math:`\tau` does not depend on the cell volume :math:`V_i`, the face
areas :math:`A_{i\pm 1/2}`, the surface-curvature redistribution area
:math:`\Delta A_i`, or the radial mesh at all.  It is a number attached
to an **ordinate**, not to a **cell**.

That :math:`\tau` had historically lived on
:class:`~orpheus.transport.spatial.scheme.StreamingTerms` (as
``tau_mm``) was an accident of where the curvilinear sweep was first
assembled — the streaming-geometry factory happened to be the object
in scope when the weighted-diamond closure was wired in, so it baked
the angular weight in alongside the genuinely geometric
:math:`\alpha`-dome and face areas.  The architectural correction
(Issue #236 Phase 2) is to give the weight back to its owner: the
**angular closure**
(:class:`~orpheus.sn.angular.closure.AngularClosureBase`),
which already binds the quadrature and already computes :math:`\tau`
from it in :func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`.
The closure exposes it through one public, polymorphic accessor,
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`,
a read-only :math:`(N,)` array indexed by the global ordinate.  The
two concrete strategies answer it differently *by type*, with no
``coord ==`` branch anywhere:

* :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
  returns its own per-level :math:`\tau` gathered to the global order;
* :class:`~orpheus.sn.angular.closure.IdentityAngularClosure`
  (Cartesian) returns the neutral :math:`\tau \equiv 1` — there is no
  angular redistribution in slab geometry, so the M-M weight reduces to
  its identity element (see below).

This is the same both-sites mint B1 applied to the :math:`c`-accessors:
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`
is declared on the
:class:`~orpheus.sn.angular.closure.AngularClosureBase`
ABC, so the contract is complete for every consumer.  (Phase B
originally declared these accessors on **both** the
``@runtime_checkable`` ``PoleAngularClosure`` Protocol and the ABC, to
serve structural-typing and nominal-inheritance consumers alike;
Issue #236 Phase 2 B2 retyped every consumer onto the ABC and Issue
#248 deleted the now-orphaned Protocol, so the ABC is the single
declaration site.)  The gather itself is a pure permutation
of the immutable per-level data, hoisted once into each mesh-bound
``__init__`` via the shared
:meth:`~orpheus.sn.angular.closure.AngularClosureBase._build_per_ordinate_cache`
(renamed from ``_build_c_per_ordinate_cache`` now that it gathers three
constants — :math:`c_{\rm in}`, :math:`c_{\rm out}`, and :math:`\tau`
— rather than two); the accessor returns the cached read-only view, so
the per-visit lookup is :math:`O(1)`.

The spatial ⊗ angular separation forbids coupling DD to the closure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If :math:`\tau` is owned by the angular closure but consumed in the
spatial cell update, the obvious move would be to hand the closure
object to the cell-update scheme so it can ask for
:math:`\tau`.  That move is **forbidden by design**, and the reason is
the load-bearing architectural fact of the SN sweep.

:class:`~orpheus.transport.spatial.diamond.DiamondDifference` is a **stateless
spatial discretization scheme**.  It reads only the per-cell
:class:`~orpheus.transport.spatial.scheme.CellVisit` packet and the
sweep-resolved :class:`~orpheus.transport.spatial.scheme.UpstreamState`; it
never sees the mesh, the quadrature, or the angular closure.  The whole
point of the spatial :math:`\otimes` angular product is that the
spatial scheme is interchangeable (diamond difference, linear
discontinuous, ...) without knowing *which* angular treatment sits on
the other axis of the tensor product, and the angular closure is
interchangeable (Morel--Montry, identity, a future Carlson variant)
without knowing the spatial scheme.  Coupling
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` to
:class:`~orpheus.sn.angular.closure.AngularClosureBase`
would collapse that product into a Cartesian-vs-curvilinear conditional
inside the spatial scheme — exactly the geometry dispatch the unified
body was built to delete.

So the constants travel as **data**, not as a **dependency**.

.. note:: **The conclusion survives P4.9a; the mechanism below does
   not (2026-08-28).**  *Data, not dependency* is still exactly right,
   and :ref:`sn-p49a-closure-owns-the-march` names the contract that
   forces it (``FORBIDDEN_EDGES["transport"]`` — an L2 package may not
   import an L3 one).  What changed is *which* data: the visit no longer
   carries the closure's constants, because the caller now multiplies
   them out and passes the two **products** the balance actually needs.
   That is strictly less coupling — a stamped ``c_in`` still let a
   scheme spell a Morel--Montry quantity, and one did.  The B3
   mechanism is recorded below as it then was.

The ``CellVisit`` packet — which the
orchestrator already populates per cell and per ordinate — carried the
angular-closure-owned numbers as plain ``float`` fields: ``c_in`` and
``c_out`` (added in B2) and then ``tau`` (B3).  They were stamped
at exactly **one** production site, ``SNMesh._make_cell_visit``, through
which all four ``dag_walk`` yield paths (slab, sphere, cylinder,
cylindrical pure-azimuthal degenerate) funnelled — Pattern 2, no
per-site divergence.  That site read the closure's per-global-ordinate
accessors and stamped:

.. code-block:: python

   closure = self.pole_angular_closure
   return CellVisit(
       cell_idx=cell_idx,
       streaming_terms=st,
       face_area_downstream=face_area_downstream,
       c_in=float(closure.c_in_per_ordinate[global_ordinate]),
       c_out=float(closure.c_out_per_ordinate[global_ordinate]),
       tau=float(closure.tau_per_ordinate[global_ordinate]),
   )

where ``global_ordinate`` was the global ordinate index resolved the
same way :meth:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.streaming_terms`
resolves it (``direction_idx`` for slab / sphere,
``level_indices[mu_level_idx][m]`` for cylinder).  The spatial scheme
downstream saw only ``visit.tau`` / ``visit.c_in`` / ``visit.c_out``;
it had no idea a closure produced them.  The provenance was recorded in
the field docstrings (the constants being distinct in origin from the
geometry-owned
:attr:`~orpheus.transport.spatial.scheme.CellVisit.streaming_terms`),
but the *type system* never let the spatial axis reach across to the
angular axis.  P4.9a kept that property and removed the three fields:
the scheme's signature now names the two assembled contributions
directly, so there is no closure-owned number on the packet to record
the provenance of.

Why τ = 1 is the slab value, and 0 would be a landmine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subsection was titled *"Why the* ``CellVisit.tau`` *default is
1.0, not 0.0"* until 2026-08-28, when P4.9a retired that field.  The
**argument is unchanged and still live**, one layer over: :math:`\tau`
is the identity closure's value on the slab, supplied through
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`,
and the choice is forced by the angular recurrence the value feeds —
which is why :meth:`~orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half`
returns :math:`\bar\psi` exactly under the identity closure.

:math:`\tau = 1` is the **neutral element** of the Morel--Montry weight.
With :math:`\tau_n = 1` the WDD angular closure
:eq:`wdd-closure` becomes
:math:`\psi_{n,i} = \tau_n\,\psi_{n+\frac12} + (1-\tau_n)\,\psi_{n-\frac12}
= \psi_{n+\frac12}`, i.e. the step (fully-outgoing) closure, and the
outgoing-face recurrence

.. math::

   \psi^a_{\rm out} = \frac{\bar\psi - (1-\tau)\,\psi^a_{\rm in}}{\tau}
   \;\xrightarrow{\;\tau = 1\;}\;
   \frac{\bar\psi - 0}{1} = \bar\psi

reduces to the **identity** in :math:`\bar\psi` — exactly what slab
geometry needs, where there is no angular redistribution and the
"angular-out" state is just the cell average.  Likewise the
denominator constant :math:`c_{\rm out} = \alpha_{n+1/2}/\tau` and the
scan split :math:`1/\tau`, :math:`(1-\tau)/\tau` are all well-defined
and reduce to their slab values (:math:`0`, :math:`1`, :math:`0`
respectively, since :math:`\alpha = 0` on the slab).

A :math:`0` value, by contrast, is a **divide-by-zero landmine**.
Every consumer of :math:`\tau` divides by it:
:func:`~orpheus.sn.angular.closure.march_psi_half_step` divides
:math:`(\bar\psi - (1-\tau)\psi^a_{\rm in})` by :math:`\tau`, and
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_inv_per_ordinate`
mints :math:`1/\tau` for the scan.  A :math:`\tau = 0` reaching either
would produce a silent ``inf``/``nan`` rather than a loud error, while
:math:`\tau = 1` computes the **identity** transformation — the safe
no-op, correct for the slab and a benign fallback that surfaces a
mis-wired closure as a *wrong-but-finite* answer a regression snapshot
catches rather than a ``nan`` that propagates.  ⛔ Until 2026-08-28 the
consumers named here were ``DiamondDifference.update``'s inline angular
thread and the cache's derived ``tau_inv``; both were re-homed onto the
closure by P4.9a, and :math:`\tau` is now guarded at its own producer
(``_assert_tau_within_unit_interval``) rather than by a field default.

The three live consumers and the L21 framing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: **Two of these three consumers were re-homed on 2026-08-28
   (P4.9a); the section is kept because the reasoning that produced them
   is what made the re-home legible.**  Read consumers (1) and (2) as the
   Step-B3 state; each carries its own dated correction below, and the
   full account is :ref:`sn-p49a-closure-owns-the-march`.

Three live paths read the closure-owned :math:`\tau` (or the constants
derived from it) after B3.  The first two are the **solve** and
**apply** directions of the same per-cell linear system; the third is
the vectorized scan form of the same recurrence — the apply / sweep
duality this page calls the **L21 twin-path** (two applications of the
*same* operator; cf. :ref:`sn-sweep-frame-apply-matvec` for the
apply-direction matvec that is the twin of the curvilinear sweep).

**(1) The scalar solve helper.**  ``cell_balance_terms`` — the
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update` solve
direction — no longer rebuilt :math:`c_{\rm in}` / :math:`c_{\rm out}`
from ``st.alpha_* / st.tau_mm``.  Its signature took them as
keyword inputs, and ``update`` supplied them straight off the visit::

   terms = cell_balance_terms(
       visit.streaming_terms, visit.face_area_downstream, total_xs,
       upstream_state, c_in=visit.c_in, c_out=visit.c_out,
   )

The helper read :math:`\tau` **not at all** — it consumed only the
already-derived :math:`c` constants, which is the right factoring: the
cell-balance denominator :eq:`dd-solve` needs
:math:`(\Delta A_i / w_n)\,c_{\rm out}`, and the upstream numerator
needs :math:`(\Delta A_i / w_n)\,c_{\rm in}\,\psi_{n-\frac12}`, neither
of which references :math:`\tau` once :math:`c` is in hand.

⛔ **Superseded 2026-08-28 (P4.9a).**  ``cell_balance_terms`` is
retired; the surviving helper is
:func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`,
reached from ``update`` and ``residual`` through one ``n_mask = 1``
conversion.  B3's factoring insight *survives and is carried further*:
having got :math:`\tau` out of the helper, the next step was to get
:math:`c` out too.  The survivor takes neither — it receives
:math:`(\Delta A/w)\,c_{\rm out}` and
:math:`(\Delta A/w)\,c_{\rm in}\,\psi_{n-\frac12}` **already
multiplied**, as ``angular_denom_term`` / ``angular_numer_upstream``, so
the transport layer names no Morel--Montry quantity at all.  The code
block above is retained as the Step-B3 spelling; it will not run today.

**(2) The angular recurrence.** The other half of
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update` — the
Morel--Montry outgoing-angular-face thread — *does* need the raw
:math:`\tau`:

.. math::
   :label: dd-mm-angular-recurrence

   \psi^a_{\rm out}
   = \frac{\bar\psi - (1 - \tau)\,\psi^a_{\rm in}}{\tau}

.. (vv-status rationale) Representational identity: the scalar cell-update
   spelling of the Morel–Montry angular recurrence.  It is algebraically the
   SAME recurrence STEP as :eq:`pole-mm-recurrence` (its seed line is a
   separate statement — this spelling carries no seed; genuinely wired to
   ``test_compute_psi_half_per_level.py::TestRecurrenceFormula``) and
   :eq:`dd-mm-scan-split`, applied — until P4.9a — in the
   ``DiamondDifference.update`` frame and now in the closure's own; the
   bit-identity across the consumer frames is the L21 twin-path content,
   pinned by ``tests/sn/sweep/core/test_wavefront_cumprod_equivalence.py``.
.. vv-status: dd-mm-angular-recurrence documented

and read it from ``CellVisit.tau`` (stamped by
``SNMesh._make_cell_visit`` from
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`)
rather than from ``visit.streaming_terms.tau_mm``.  That was the line
the :math:`1.0` default protected.

.. implements:: dd-mm-angular-recurrence
   :by: orpheus.sn.angular.closure.march_psi_half_step

   **Implemented by** 2 sites, both in the angular closure since
   2026-08-28.  ``march_psi_half_step`` is the relation itself — Form A
   of :eq:`sn-p49a-march-forms`, subtract-then-divide, the operation
   order being load-bearing for bit-identity.
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half`
   is the per-cell entry that supplies the closure's own :math:`\tau` at
   an ordinate and delegates.  ⛔ Before P4.9a this equation's
   implementer was ``DiamondDifference.update``, which evaluated the
   relation inline because its package may not import the closure's
   (:ref:`sn-p49a-closure-owns-the-march`); declaring the owners is what
   stands the inference down — the label previously attracted **32**
   guesses, every one matched on the shared token *angular*, i.e. a
   membership list of the angular package rather than a set of
   implementers.

.. implements:: dd-mm-angular-recurrence
   :by: orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half

⛔ **Superseded 2026-08-28 (P4.9a).**  The equation is unchanged and
still labelled; what moved is who evaluates it.  The stamp, the
``CellVisit.tau`` field and ``_make_cell_visit`` are all retired: a
scheme that does not apply the march has no use for :math:`\tau`, so the
mesh stopped being a second home for a closure-owned number.  The
:math:`1.0` default retired with the field it defended — see the
subsection below, which is kept for the argument rather than for the
field.

**(3) The CumprodScan split.** The vectorized fast path — the cumulative
product that replaces the per-cell Python loop for the curvilinear
sweep — needs the same recurrence in a form amenable to a forward scan.
:class:`~orpheus.sn.sweep.cache.StreamingCoefficientCache` sourced
:math:`\tau` from
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`
and precomputed the split

.. math::
   :label: dd-mm-scan-split

   \texttt{tau\_inv} = \frac{1}{\tau},
   \qquad
   \texttt{mm\_a\_in\_coeff} = \frac{1 - \tau}{\tau},

.. (vv-status rationale) Representational identity: the precomputed-split
   (vectorized-scan) spelling of the same Morel–Montry recurrence STEP as
   :eq:`dd-mm-angular-recurrence` / :eq:`pole-mm-recurrence` (the seed is a
   separate statement, carried only by the latter) — a
   perform-once-at-construction hoist (L16) of 1/τ and (1−τ)/τ, algebraically
   identical to the scalar recurrence.  The scalar↔scan bit-identity is pinned
   by ``tests/sn/sweep/core/test_wavefront_cumprod_equivalence.py``.
.. vv-status: dd-mm-scan-split documented

consumed at the loss-representation scan recurrence (in
:mod:`orpheus.sn.loss_representation`) as

.. math::

   \psi^a_{\rm out}
   = \texttt{tau\_inv}\cdot\bar\psi
     - \texttt{mm\_a\_in\_coeff}\cdot\psi^a_{\rm in}
   = \frac{\bar\psi}{\tau} - \frac{1-\tau}{\tau}\,\psi^a_{\rm in},

which is algebraically identical to :eq:`dd-mm-angular-recurrence` — the
same operator, applied in the vectorized frame — but **not bitwise
equal** to it, which is the whole reason the two spellings are kept
apart deliberately rather than collapsed
(:ref:`sn-p49a-two-forms`).

.. note:: **The hoisting argument survived P4.9a; the ownership half of
   it did not (2026-08-28).**

   This paragraph read: *"Precomputing* :math:`1/\tau` *and*
   :math:`(1-\tau)/\tau` *is a legitimate perform-once-at-construction
   hoist (L16): the closure exposes only the* **primitive**
   :math:`\tau` *(Pattern 5 — build the primitive, not the product),
   and each consumer derives the trivial* :math:`1/\tau`,
   :math:`(1-\tau)/\tau`, :math:`\alpha_{\rm out}/\tau` *algebra it
   needs at its own definition site.  The scan derivation lives in the
   cache; the recurrence consumes it."*

   The **hoist** is still right and still where it was — the cache
   computes these once per solve, not once per iteration.  What was
   wrong is *"each consumer derives"*.  The pairing of :math:`1/\tau`
   with :math:`(1-\tau)/\tau` is not a convenience product of a
   primitive: it is the :math:`\bar\psi` and :math:`\psi^a_{\rm in}`
   coefficient **pair** of the closure's own update written
   scan-normally — relation knowledge, and a cache deriving it was the
   scheme's inline-march smell one notch down.  The closure now
   **mints** both
   (:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_inv_per_ordinate`,
   :attr:`~orpheus.sn.angular.closure.AngularClosureBase.march_a_in_coeff_per_ordinate`)
   and the cache stores them.  Pattern 5 is unviolated by this: a
   coefficient of one's own relation is not a "product" a consumer
   should be assembling.  :math:`\alpha_{\rm out}/\tau` is unaffected —
   it is :math:`c_{\rm out}`, which the closure has minted since B1.

.. implements:: dd-mm-scan-split
   :by: orpheus.sn.angular.closure.AngularClosureBase.tau_inv_per_ordinate

   **Implemented by** 2 sites — one per constant, both **minted by the
   closure** since 2026-08-28.  ⛔ Until P4.9a the split was *derived by
   the cache* (``StreamingCoefficientCache.from_mesh_and_quad``), and
   that was the same smell as the scheme's inline march, one notch down:
   the pairing of :math:`1/\tau` with :math:`(1-\tau)/\tau` is not a
   convenience product, it is the :math:`\psi^a_{\rm in}` coefficient of
   the closure's own update written scan-normally — relation knowledge,
   which the cache had no business spelling.  L16's hoisting half now
   lives at the owner (P4b, 2026-08-29): the closure caches both arrays
   read-only at construction, the geometry table stores no copy, and the
   walk binds them per sweep.  ⚠ ``march_a_in_coeff_per_ordinate`` is
   spelled ``(1 - tau) / tau``, never the algebraically-equal
   ``tau_inv - 1.0`` — `[M]` the two differ by 1–2 ULP, and the
   closure-side spelling gate
   (``TestMintedScanConstants::test_minted_constants_pin_their_spelling``)
   pins it with ``array_equal``.

.. implements:: dd-mm-scan-split
   :by: orpheus.sn.angular.closure.AngularClosureBase.march_a_in_coeff_per_ordinate

Why the fold is bit-identical, and the regression floor for Step C
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step B3 is **bit-identical (0-ULP)**.  The argument has two legs, both
already established by the earlier carve steps:

#. **Closure-:math:`\tau` is 0-ULP equal to geometry-:math:`\tau`.** The
   closure's
   :func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`
   is a line-for-line replica of the geometry factory's :math:`\tau`
   arithmetic (Step A), pinned by the **Leg-1 producer-equivalence
   gate** ``tests/sn/sweep/curvilinear/test_tau_producer_equivalence.py``
   to the geometry-factory value (0-ULP) *and* to an independent
   reference (then ``contamination.morel_montry_weights``, a different
   code path to the same BMC-2010-Eq. 43 weight; see the
   :ref:`reference-migration note <sn-tau-reference-migration>`).
   Reading :math:`\tau` from the closure therefore yields exactly the
   same ``float64`` bits the former ``st.tau_mm`` read carried.

#. **The per-level :math:`\to (N,)` gather is a pure permutation.** No
   arithmetic happens between the closure's per-level :math:`\tau` and
   the global-ordinate :math:`(N,)` view the accessor returns — only a
   reindex.  So every derived quantity (:math:`c_{\rm in}`,
   :math:`c_{\rm out}`, ``tau_inv``, ``mm_a_in_coeff``) is bit-for-bit
   what the inline rebuilds produced.

This was confirmed not only by the regression snapshots but **in
process**: at sphere (single :math:`\mu`-level) and cylinder
(multi-level) configurations,
:math:`\lvert\texttt{visit.tau} - \texttt{st.tau\_mm}\rvert`,
:math:`\lvert\texttt{closure.tau\_per\_ordinate} - \texttt{st.tau\_mm}\rvert`,
and ``np.array_equal`` on the cache's ``tau_inv`` / ``mm_a_in_coeff``
against the geometry-:math:`\tau`-derived split were **all exactly
zero**.  The DriftWarning-escalating ``tests/sn/sweep/core``,
``tests/sn/sweep``, and ``tests/sn/solve`` snapshots stayed unmoved
(588 + 60 green), with zero drift escalation.

The bit-identity guarantee is what makes B3 the **regression floor for
Step C**.  Two committed catchers pin the new live paths so that the
retirement of the parallel geometry-:math:`\tau` producer (now landed,
Step C) cannot silently break them:

* The **Leg-1 producer-equivalence gate** pins
  ``closure.tau_per_ordinate`` to the BMC-2010-Eq. 43 reference, so the
  closure remains the correct sole producer after the geometry one is
  deleted.

* The **production-stamp catcher**
  ``tests/sn/sweep/core/test_cell_visit_c_stamp.py`` walked a real
  production ``dag_walk`` (sphere, multi-level cylinder, slab) and — in
  its dedicated :math:`\tau` arm — asserted every ``visit.tau`` equals
  the **independently recomputed** Morel--Montry weight for that visit's
  ordinate at 0-ULP: the test pins the stamp's *ordinate map*, the
  complement of what Leg-1 pins (the producers' *values*).  Before
  Step C this catcher used the *geometry-produced* ``st.tau_mm`` as its
  independent reference; when Step C deleted that field the oracle was
  re-pointed onto ``morel_montry_weights`` (with the cylinder clamp
  replicated), keeping it geometry-:math:`\tau`-free and, at the time,
  independent of the closure under test.  See the
  :ref:`reference-migration note <sn-tau-reference-migration>` for the
  Q5.6.4 re-posing.  This arm was added specifically because B3
  made the
  ``CellVisit.tau`` stamp **live** while
  the existing named twins never call the rewired reader (vv L11
  Mode 11): a mutation stamping ``tau = ... * 1.1`` drifts the converged
  cylinder scalar flux by :math:`\sim 0.2\,\%` with **no** other test
  red, so the dedicated arm is the only committed catcher of a
  :math:`\tau`-stamp ordinate-map error.  Mutation-verified RED on the
  :math:`\times 1.1` stamp across sphere + cylinder + slab; GREEN clean.

  ⛔ **The stamp is retired (P4.9a, 2026-08-28) and the catcher was
  re-derived, not deleted.**  The ordinate-map hazard it existed for did
  not disappear — it moved one producer up.  With the mesh no longer
  copying closure data onto visits, a wrong per-level :math:`\to` global
  gather *inside the closure's own construction* would now reach every
  consumer of the :math:`(N,)` accessors: the cache populator, the walk's
  degenerate assembly, and the march itself.  The successor
  ``tests/sn/sweep/core/test_closure_constant_map.py`` pins exactly that
  map, against the same independent surrogate (:math:`\tau` from a
  different code path to the BMC-2010 weight, :math:`\alpha` from the
  operator's surviving dome) at 0-ULP, over every (cell, global ordinate)
  of a sphere, a multi-level cylinder and a slab.

* The **seam-6 scan catcher**
  ``tests/sn/sweep/core/test_affine_carve_baseline.py`` reddens on a
  corruption of the CumprodScan :math:`\tau` split, pinning the third
  live consumer.

With these in place, **Step C has now deleted** the geometry-side
:math:`\tau` producer.  The :math:`\tau` blocks inside
:func:`~orpheus.sn.mesh.reduced_operator.spherical_streaming`
(the ``mu_edge`` weight-sum loop)
:math:`\,/\,`
:func:`~orpheus.sn.mesh.reduced_operator.cylindrical_streaming` (the
per-level ``eta_edge`` loop) and the slab synthetic were excised, and
the now-orphaned ``StreamingTerms.tau_mm``, ``StreamingTerms.alpha_in``
/ ``alpha_out`` (whose sole readers were the c-rebuild sites B1--B3 just
retired), ``ReducedStreamingOperator.tau_mm``, and
``ReducedStreamingOperator.tau_mm_per_level`` dataclass fields were
dropped — confident that nothing live depended on them.
See :mod:`orpheus.sn.angular.closure` for the
``tau_per_ordinate`` accessor and the three-constant cache,
:mod:`orpheus.transport.spatial.scheme` for the ``CellVisit.tau`` field
and ``SNMesh._make_cell_visit`` for the single production stamp (both
retired at P4.9a — :ref:`sn-p49a-closure-owns-the-march`), and
:mod:`orpheus.sn.sweep.cache` for the scan split.

.. _sn-tau-step-c-closeout:

Step C close-out — the geometry-side τ producer is retired
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retirement is a **surgical field excision plus a test-oracle
migration**, not a blind delete: the τ producer was already dead in
*production* (B3 left zero live readers), but it remained load-bearing
as a *test oracle* — the regression-floor catchers used the
geometry-produced ``st.tau_mm`` as their structurally-independent
reference.  Migrate-then-delete preserved the floor:

#. **The oracles were re-pointed first.** The production-stamp catcher
   and the surviving producer-equivalence legs were pointed at
   ``contamination.morel_montry_weights`` — a different code path to the
   same BMC-2010-Eq. 43 weight, unclamped on both geometries, with the
   cylinder clamp :math:`\mathrm{clip}(\cdot, \tfrac12, 1)` replicated in
   the test surrogate.  This kept the floor green *while
   geometry-:math:`\tau` was still present*, proving the migration
   faithful before any deletion.  The two
   ``*_equals_geometry_factory_0ulp`` legs of
   ``test_tau_producer_equivalence.py`` (which compared closure-:math:`\tau`
   against the soon-to-be-deleted factory output) became vacuous and were
   retired; the independent-reference and clamp-difference legs survived
   at that point — both were themselves re-posed at Q5.6.4, next.

.. _sn-tau-reference-migration:

.. note:: **The τ oracle moved again at Q5.6.4 (2026-08-11) — and the
   independence CLASS changed with it.**

   ``derivations/discrete/sn/contamination.py`` is **retired**; its
   successor is
   :mod:`orpheus.derivations.discrete.sn.angular_differencing`, where
   ``morel_montry_weights`` survived by name for a while but
   **DELEGATED to the production producer** — a
   deliberate choice, so that a "reference" can never silently drift into
   a second definition of the angular cell.  That is exactly how the old
   module's cylinder arm went wrong: it kept building the retired
   η-midpoint edges, so `[M]` its :math:`\tau` disagreed with production
   by up to :math:`6.8\mathrm{e}{-2}`.

   ⛔ **It was therefore no longer an independent reference for**
   :math:`\tau`, and any sentence above crediting it as one describes the
   state before Q5.6.4.  Comparing against it would have been comparing
   :math:`\tau` with itself through a wrapper — green forever, and unable
   to detect the drift its name advertises (``coding-standards``: a rewire
   can demote a gate's claim class without touching one line of the
   body).  Both arms of ``test_tau_producer_equivalence.py`` were
   re-pointed at **hand-authored** references instead:

   * **Sphere** — an inline cumulative-weight expression (BMC Eq. 12,
     then P2), authored in the test.  `[M]` *not* bit-exact, and that is
     the evidence of independence: ``np.cumsum`` sums pairwise where the
     producer accumulates sequentially, giving 0 → 16 → 59 → 55 → 2024
     ULP at :math:`N = 4/8/16/32/64`.  Asserted at ``atol=1e-13``; a new
     row at :math:`N \ge 64` must widen it.
   * **Cylinder** — the **analytic closed form**
     :math:`\tau_m = \tfrac12 + \tfrac12\cot\omega_m\tan(\Delta\omega/4)`,
     hand-derived from the arc geometry: *structural* independence, a
     strictly stronger footing than the procedural twin it replaces.  It
     carries a **negative control** (``vv-principles`` #19): the closed
     form must DIFFER from the retired chord convention, or the row would
     pass equally against the partition the carve replaced and could not
     be evidence about the partition choice.

   The clamp-difference leg is gone with the clamp — its thesis
   (*"closure τ == clip(reference, ½, 1), and must DIFFER where the clamp
   bites"*) became vacuous when there was no clamp left to be the only
   difference.

#. **The producers were excised surgically.** The τ blocks *were*
   interleaved with outputs that Step C left alone:
   :func:`~orpheus.sn.mesh.reduced_operator.spherical_streaming` shared
   its ``mu_edge`` array with the starting direction ``mu_start`` (the
   Hébert §3.9.4 :math:`\mu_{1/2} = -1.0`), and
   :func:`~orpheus.sn.mesh.reduced_operator.cylindrical_streaming`
   shared its per-level loop with ``mu_start_per_level``.  A
   whole-function deletion would have been wrong; only the τ statements
   were removed, and Step C left the :math:`\alpha`-dome, the
   redistribution factor, the face areas and the starting-direction edges
   where they were, on the geometry operator, on the reasoning that they
   are geometric.

   ⛔ **That last reasoning was superseded on 2026-08-26, and three of
   those four objects have since moved.**  The dome
   (``alpha_half`` / ``alpha_per_level``) and the starting-direction edges
   (``mu_start`` / ``mu_start_per_level``) are the **angular** factor —
   functions of ``(quadrature, coord)`` alone — and now live on
   :class:`~orpheus.sn.angular.redistribution.AngularRedistribution` with
   one producer; the fused ``redist_dAw`` cache was retired outright,
   because it was a *product* of a geometric with a quadrature factor that
   neither consumer owned.  Only ``face_areas`` and ``delta_A`` — genuinely
   spatial chart data — remain on the streaming operator.

   ✅ **REMEDIED later the same day (2026-08-26, P1).**  This paragraph
   originally ended: *"the per-direction extraction packet*
   :class:`~orpheus.transport.spatial.scheme.StreamingTerms` *still
   carries a* ``mu_start`` *field, now read from the angular factor
   rather than from a field of the streaming operator's own."*  True
   when written, and repealed hours later: that field was the middle
   link of a three-link **dead chain** —
   ``AngularRedistribution.mu_start_per_level`` →
   ``StreamingTerms.mu_start`` → ``StreamingCoefficientCache.mu_start`` →
   nothing.  `[M]` the terminal had **zero readers of any kind**, so the
   packet's only consumer was the write into it.  Both downstream links
   are retired; the owner stays, and every consumer reads it.  Step C's
   *surgical* judgment stands unchanged; what it got wrong was the reason
   it gave for the residue, and :ref:`sn-redistribution-tensor-product`
   is where the correct split is derived.

#. **The deletion was proven inert.** The bit-identity regression gates
   (run under an escalated ``DriftWarning``) showed **zero** failures
   across the sweep / scan / matvec suites, and the test-count delta
   reconciled exactly to the four retired ``*_equals_geometry_factory``
   legs and the two retired ``test_reduced_operator.py``
   τ-bit-identical tests — no silent test loss.  After deletion the
   re-pointed catcher was **mutation-verified RED** (a :math:`\times 1.1`
   stamp and a ``c_in`` :math:`\leftrightarrow` ``c_out`` swap both
   reddened it against the independent oracle), confirming the migrated
   catcher is a real catcher reading the independent reference, not a
   tautology against the closure.

.. note::

   The legacy ``__call__``-argument ``tau_mm`` on the unbound
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
   path (``MorelMontryAngularSweep(sn_mesh=None)``, where :math:`\tau` was
   passed as a runtime argument because the closure is not mesh-bound) was
   a **separate surface** that **survived Step C** unchanged — it was the
   closure's own runtime parameter, not the geometry-side field the carve
   retired.  It was subsequently retired under
   `Issue #248 <https://github.com/deOliveira-R/ORPHEUS/issues/248>`_
   (landed in this same re-staging, which deleted the strategy
   ``__call__`` bundle entirely; see the contract-evolution note at
   :ref:`sn-pole-angular-closure-protocol`).

.. _sn-p49a-closure-owns-the-march:

P4.9a — the closure owns its march, and the scheme closes one axis
--------------------------------------------------------------------

Steps B1–C above moved the Morel--Montry *data* to its owner: first the
weight :math:`\tau`, then the derived constants :math:`c_{\rm in}` /
:math:`c_{\rm out}`, until the geometry side produced none of it.  What
they did **not** move is the *relation* — the march
:eq:`dd-mm-angular-recurrence` itself, which
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update`
went on evaluating inline, on data it had been handed.  P4.9a
(2026-08-28) closes that arc: a spatial discretization scheme now closes
the **spatial** axis and nothing else, and the angular axis is closed by
its own closure, applied at the site that composes the two.

Why the twin existed: a layer contract, not an oversight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inline copy was **forced**, and naming the forcing is the point of
this subsection — a reader who takes it for carelessness will
re-introduce it.

The project declares an import-layer contract and gates it.  `[M]` by
AST over ``tests/test_layer_imports.py``: ``transport`` is **L2**,
``sn`` is **L3**, and ``FORBIDDEN_EDGES["transport"] = L3_PACKAGES`` —
enforced per module by a ``@pytest.mark.foundation`` parametrized gate.
So :mod:`orpheus.transport.spatial.diamond` **may not import**
:mod:`orpheus.sn.angular.closure`.  A scheme that wanted to apply the
march therefore could not *call* its owner; it could only re-spell the
relation.  That is the shape of the defect: not a copy someone forgot to
delete, but a Pattern-2 twin that the architecture *manufactured* the
moment the angular relation was made the scheme's responsibility.

The repair is correspondingly not "delete the copy".  It is to move the
**responsibility** up one level, to the site that already sees both
packages — the SN walk today, and the ``StreamingOperator`` once P4.9b
lands.  The scheme keeps only what a spatial scheme can own; it receives
the angular axis's effect on the balance as two already-multiplied
numbers, and never sees a closure object, a :math:`\tau`, or a
half-angle thread.

.. _sn-p49a-two-forms:

Two arithmetic forms, and why they are welded rather than unified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The march is written two ways in this tree, and the difference is not
cosmetic:

.. math::
   :label: sn-p49a-march-forms

   \underbrace{\psi^a_{\rm out} =
     \frac{\bar\psi - (1-\tau)\,\psi^a_{\rm in}}{\tau}}_{\textbf{Form A} \;-\;
     \text{subtract, then divide}}
   \qquad\text{vs.}\qquad
   \underbrace{\psi^a_{\rm out} =
     \tau^{-1}\,\bar\psi \;-\; \frac{1-\tau}{\tau}\,\psi^a_{\rm in}}
     _{\textbf{Form B} \;-\; \text{scan-normal, precomputed constants}}

.. (vv-status rationale) Representational: the two floating-point
   spellings of ONE relation (:eq:`dd-mm-angular-recurrence` /
   :eq:`pole-mm-recurrence` state the relation itself; this states the
   pair).  Not a solver claim — what is verifiable is the arithmetic
   distance between them, measured below, and the per-path gates named
   there.
.. vv-status: sn-p49a-march-forms documented

Form A is the owner's spelling and the one
:func:`~orpheus.sn.angular.closure.march_psi_half_step` computes.  Form B
is the scan-normal spelling the vectorized fast path needs, built from
the two constants the closure now *mints*,
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_inv_per_ordinate`
and
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.march_a_in_coeff_per_ordinate`.
They are algebraically identical and **not bitwise equal**.

Reproduce it as follows: take :math:`\tau` from the shipped producer on a
shipped rule, evaluate both forms on the same random
:math:`(\bar\psi, \psi^a_{\rm in})`, and compare with ``==``:

.. code-block:: python

   import numpy as np
   from orpheus.numerics.quadrature.directional import Quadrature
   from orpheus.sn.angular.closure import morel_montry_tau_per_level
   from orpheus.geometry.coord import CoordSystem

   tau = np.concatenate(morel_montry_tau_per_level(
       Quadrature.folded_product(4, 6), CoordSystem.CYLINDRICAL))   # (12,)

   form_a = lambda t, p, a: (p - (1.0 - t) * a) / t
   form_b = lambda t, p, a: (1.0 / t) * p - ((1.0 - t) / t) * a

   frac = []
   for seed in range(200):                       # 200 seeds x 12 x 200 draws
       r = np.random.default_rng(seed)
       p = r.uniform(0.1, 2.0, (tau.size, 200))
       a = r.uniform(0.1, 2.0, (tau.size, 200))
       frac.append((form_a(tau[:, None], p, a) == form_b(tau[:, None], p, a)).mean())

   assert 0.46 <= min(frac) and max(frac) <= 0.52     # never close to 1.0
   assert (tau == 0.5).sum() == 2                     # of 12 — see below

`[M]` 2026-08-28, on that fixture:

.. list-table:: Form A vs Form B on ``folded_product(4, 6)``, cylindrical τ
   :header-rows: 1
   :widths: 40 60

   * - quantity
     - measured
   * - bit-equal fraction (200 seeds × 2400 evaluations)
     - :math:`46.21\,\%` – :math:`51.42\,\%`, mean :math:`48.66\,\%`
   * - :math:`\max|{\rm A}-{\rm B}|` (4.8 × 10\ :sup:`5` evaluations)
     - :math:`1.776\times 10^{-15}`
   * - :math:`\max` ULP gap, over the *same* 200 seeds
     - :math:`113` – :math:`91\,839`
   * - ordinates with :math:`\tau` **exactly** :math:`\tfrac12`
     - 2 of 12 → those are bit-equal :math:`100\,\%` of the time

.. warning:: **Two of those four rows are stable statistics and two are
   not — and the tree currently publishes one of the unstable ones.**

   :math:`\max|{\rm A}-{\rm B}| = 1.776\times 10^{-15}` reproduces
   exactly and is the number to quote.  The **bit-equal fraction** is a
   property of the *draw*: it must be published as the band above, not
   as a single figure.  The **ULP gap** is worse than draw-dependent —
   it spans three orders of magnitude over the same 200 seeds, because
   the ULP metric explodes wherever the two terms nearly cancel while
   the absolute difference stays at the round-off floor.

   ⚠ The docstring of
   :func:`~orpheus.sn.angular.closure.march_psi_half_step` records this
   as *"bit-equal 59 % on real τ, max 204 ULP"*, from the phase's own
   single-draw probe.  Neither figure reproduces here: 59 % lies outside
   the 200-seed band, and 204 sits at the bottom of the ULP range.  The
   *conclusion* those numbers were offered for is unaffected and in fact
   strengthened — the forms are not interchangeable at the bit level —
   but the two numbers should not be re-quoted as if they were
   properties of the fixture.

The mechanism behind the last table row is exact and needs no
statistics.  When :math:`\tau` is **bitwise** :math:`\tfrac12`,
:math:`1/\tau = 2.0` and :math:`(1-\tau)/\tau = 1.0` are both exact, so
Form B degenerates to Form A's arithmetic and the two agree on every
input.  `[M]` on ``folded_product(4, 6)`` the twelve ordinates carry
**six distinct** ``float64`` :math:`\tau` values — the three nominal
levels :math:`\{0.2679\ldots,\ \tfrac12,\ 0.7320\ldots\}`, each
appearing as a pair one ULP apart — and only **2 of 12** are exactly
:math:`\tfrac12`.  A bit-identity claim validated on one ordinate of one
rule is therefore reading a coin (``vv-principles`` #31, #13).

Which is why the design **welds** the two forms by gate rather than
unifying them by spelling.  Both are correct; each is the right shape
for its consumer's reduction tree; and they partition the ordinate set,
so no single input is ever evaluated both ways:

.. list-table:: The two forms partition the work — neither is a live twin of the other
   :header-rows: 1
   :widths: 22 26 52

   * - path
     - form
     - where
   * - degenerate cylindrical-axis ordinates (per-cell)
     - **A**
     - :func:`~orpheus.sn.angular.closure.march_psi_half_step`, reached
       from the walk through
       :meth:`~orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half`
   * - the batch half-angle grid
     - **A**
     - ``_psi_half_grid_single_level``, whose loop body *delegates* to the
       same function — so the delegation is bit-neutral by shared body
   * - non-degenerate ordinates (vectorized scan, forward **and** adjoint)
     - **B**
     - the scan fast path in :mod:`orpheus.sn.loss_representation`,
       consuming the closure's two minted constants — `[M]` three sites,
       the forward thread and the two transpose arms

`[M]` those are the whole population: ``march_psi_half_step`` has exactly
**two** callers in ``orpheus/`` — the per-cell entry and the batch
kernel's loop body — and the minted constants are read only on the scan
path.  No input reaches both forms.

.. important:: **The honest scope, so the done-when is not read as
   stronger than it is.**

   P4.9a does **not** achieve *"the Morel--Montry relation has exactly
   one spelling in the tree"*.  What it achieves is that the relation
   has exactly one spelling **inside** ``orpheus/transport/`` — namely
   none — and one **owner**, :mod:`orpheus.sn.angular.closure`, which
   now emits both forms from one place instead of one form here and one
   form in a package that could not name it.  Form B survives, as the
   closure's own scan-normal representation; the difference from before
   is that a reader who changes :math:`\tau`'s meaning now edits one
   module, and the two forms are pinned against each other by gate
   rather than by hope.

What moved, concretely
~~~~~~~~~~~~~~~~~~~~~~~

* **The march.** ``DiamondDifference.update`` no longer evaluates it.
  :func:`~orpheus.sn.angular.closure.march_psi_half_step` is the one
  production spelling of Form A; the batch kernel delegates to it, and
  the per-cell entry is
  :meth:`~orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half`.
* **The scalar cell-balance twin.** ``cell_balance_terms`` and its
  ``CellBalanceTerms`` record are retired onto
  :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`,
  which the per-cell solve and apply directions now reach through a
  single ``n_mask = 1`` conversion.
* **The visit family.** ``CellVisit`` lost the closure stamp
  (``tau`` / ``c_in`` / ``c_out``) and with it the mesh-side
  ``SNMesh._make_cell_visit``; ``UpstreamState`` lost
  ``angular_upstream``; ``CellResult`` lost ``outgoing_angular_state``.
  What the scheme receives instead are two keyword arguments carrying
  **assembled** contributions — ``angular_denom_term`` and
  ``angular_numer_upstream`` — whose slab values are the neutral
  elements of the sums they enter.
* **The scan constants.** The cache no longer *derives*
  :math:`1/\tau` and :math:`(1-\tau)/\tau`; the closure mints them and
  the cache stores them.  L16's hoisting argument survives — the cache
  is still where they are computed once per solve — but the *pairing* of
  those two coefficients is relation knowledge, and a cache deriving it
  was the same smell as the scheme one notch up.
* **A guard, re-keyed onto a stronger signal.** Linear-Discontinuous
  refused curvilinear visits by testing
  ``upstream_state.angular_upstream is not None``.  That field's
  retirement would have made the guard silently unreachable — a defaulted
  presence-test with nothing left to detect (``vv-principles`` #28's
  temporal twin).  It is re-keyed onto two **value** signals:
  ``face_area_inner != face_area_outer`` (`[M]` exactly ``False`` on
  every Cartesian cell, ``True`` on every constructible curvilinear one)
  and a non-neutral assembled angular contribution.  A value-keyed guard
  is reachable by calling the scheme directly, so its witness needs no
  mesh and no earlier guard can preempt it.

.. _sn-space-angle-separability-section:

Space ⊗ angle separability — the (spatial ⊗ angular) product capstone
----------------------------------------------------------------------

This section closes the Issue #236 *(spatial* :math:`\otimes` *angular)
product* narrative on the theory page.  The campaign had three phases:

* **Phase 1 — pairing validity.**  The spatial closure (the diamond /
  weighted-diamond cell update of :eq:`dd-curvilinear-scalar`) and the
  angular closure (the Morel--Montry weight :math:`\tau` of
  :eq:`mm-weights`, the redistribution dome
  :eq:`alpha-recursion`) are two distinct, independently-selectable
  axes — a genuine tensor product, with separate injection points on
  :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` (``scheme=`` for the
  spatial closure, ``angular_closure=`` for the angular one; the two
  keywords were ``cell_update=`` and ``pole_angular_closure=`` when this
  phase landed).  Since P4.9b the *operator* is posed with both, and the
  hub's two slots are what
  :meth:`~orpheus.sn.operators.streaming.StreamingOperator.pose` reads —
  the product is now spelled in the operator's own signature
  (:ref:`sn-p49b-operator-poses-with-closures`).
* **Phase 2 — :math:`\tau`-ownership carve.**  The angular weight
  :math:`\tau` was moved off the geometry operator and onto the angular
  closure, so the angular axis literally *owns* its own discretisation
  knob (:ref:`sn-tau-closure-owned` through
  :ref:`sn-tau-step-c-closeout`).  That carve made the product
  *structural in the type system*, not merely conceptual.
* **Phase 3 — separability characterisation (this section).**  Having
  established the product and given each axis its own knob, the final
  question is: *how do the two error contributions combine?*  The answer
  is geometry-dependent, and it is the campaign's headline claim.

The decomposition is pinned permanently by the L1 MMS characterisation
gate :mod:`tests.sn.verification.mms.test_space_angle_separability`,
which carries ``@pytest.mark.verifies("sn-space-angle-separability")``
against :eq:`sn-space-angle-separability` below.

The space–angle error decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write the total SN discretisation error of the scalar flux as a function
of the two refinement parameters — the spatial mesh size
:math:`h \sim 1/n_{\rm cells}` and the angular (quadrature) order
:math:`N` (the ordinate count, or the azimuthal order :math:`n_\varphi`
in the cylinder).  Let :math:`E_{\rm space}(h)` be the error of the
spatial closure at infinite quadrature order and :math:`E_{\rm angle}(N)`
the error of the angular closure on an exactly-resolved spatial mesh.
The campaign's headline result is the geometry-split law

.. math::
   :label: sn-space-angle-separability

   E(h, N) \;\approx\;
   \begin{cases}
     E_{\rm space}(h) \,+\, E_{\rm angle}(N),
       & \text{Cartesian (slab / 2-D / 3-D): \textbf{separates}},\\[1.2ex]
     \max\!\bigl(E_{\rm space}(h),\, E_{\rm angle}(N)\bigr),
       & \text{curvilinear (sphere / cylinder): \textbf{gates}}.
   \end{cases}

The two regimes are distinguished operationally by the **mixed second
difference** — the discrete :math:`\partial^2 E / \partial h\,\partial N`
evaluated on a two-quadrature error table over a coarse/fine pair of
mesh sizes:

.. math::
   :label: sn-space-angle-cross-term

   M \;=\; E[h_1, N_1] - E[h_1, N_2] - E[h_2, N_1] + E[h_2, N_2],
   \qquad
   \frac{|M|}{\max(\Delta E_h,\, \Delta E_N)} \;
   \begin{cases}
     \ll 1, & \text{separable (additive),}\\
     = \mathcal{O}(1), & \text{gated (coupled).}
   \end{cases}

For an additively separable error, :math:`E[h,N] = f(h) + g(N)` exactly,
so the cross-term telescopes to zero: :math:`M = f(h_1) + g(N_1) -
f(h_1) - g(N_2) - f(h_2) - g(N_1) + f(h_2) + g(N_2) = 0`.  A non-zero
:math:`M` is therefore a *direct, mechanism-anchored* measurement that
the two axes interact — that the second mixed partial of the error
surface does not vanish.  This is the quantity the ST5 gate measures.

.. (V&V scope note) Characterisation claim, now tested: both the
   law :eq:`sn-space-angle-separability` and its discriminator
   :eq:`sn-space-angle-cross-term` describe the STRUCTURE of the
   discretisation-error surface (the regime discrimination), not a
   solver eigenvalue or flux VALUE.  This is an L1 MMS-convergence-
   structure (math) claim per vv-principles — MMS does not reach the
   eigenvalue layer, so neither label is, or ever becomes, an
   eigenvalue / flux-value claim.  The ST5 characterisation gate
   ``test_space_angle_separability.py`` now carries the
   ``verifies`` markers for both labels, so each is ``documented`` AND
   ``tested``: the verifying ``tests`` edge is the characterisation
   gate (an L1 MMS gate), not a closed-form / semi-analytical value
   reference.  What each gate leg verifies:
     * :eq:`sn-space-angle-separability` (the geometry-split decomposition
       law) is pinned by all six legs as a *positive* signature — the
       Cartesian legs assert separability (mixed-second-difference
       :math:`\to 0`, N-independent spatial rate); the curvilinear legs
       assert gating (N-gated spatial rate).  The marker is FILE-level.
     * :eq:`sn-space-angle-cross-term` (the mixed-second-difference
       discriminator :math:`M`) is the gate's measured instrument: the
       three legs that assert directly on :math:`|M|/\max`
       (``test_cartesian_slab_iso_space_angle_separable``,
       ``test_cartesian_slab_p1_aniso_floor_n_independent``,
       ``test_sphere_cross_term_large_discriminates_from_cartesian``)
       carry the per-test ``verifies`` marker.  The discriminator is a
       quantity the gate *measures against a declared threshold*, not a
       passive derivation step, so the ``tested`` edge is a real
       coverage claim.
   The posture mirrors the pole-cell characterisation gate this gate is
   modelled on (#233).

Why the two axes factorize: LMM-1987 (spatial) × BMC-2010 (angular)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decomposition is not an empirical accident — it is forced by the
structure of the asymptotic *diffusion-limit-consistency* literature,
which is **literally split into a spatial paper and an angular paper**.
This split is the strongest possible evidence that the consistency
conditions live on two separate axes.

* **The spatial condition** — Larsen, Morel & Miller, "Asymptotic
  solutions of numerical transport problems in optically thick,
  diffusive regimes," *Journal of Computational Physics*
  **69(2):283--324 (1987)**, DOI
  `10.1016/0021-9991(87)90170-7 <https://doi.org/10.1016/0021-9991(87)90170-7>`_
  (and Part II, Larsen & Morel, JCP **83(2):212--236 (1989)**, DOI
  10.1016/0021-9991(89)90229-5).  LMM analyse the *spatial* differencing
  scheme's diffusion limit (cells scaled so they are not optically
  thin): a scheme whose discrete spatial limit is itself a valid
  diffusion discretisation (linear-discontinuous, weighted-diamond with
  the right closure) is "substantially more accurate" than one without
  (bare diamond difference).  This is a condition **on the spatial axis
  alone** — the angular order does not enter.

* **The angular condition** — Bailey, Morel & Chang,
  :cite:`BaileyMorelChang2010`, "The asymptotic diffusion-limit accuracy of
  S\ :sub:`N` angular differencing schemes," *Nuclear Science and
  Engineering* **165(2):149--169 (2010)**, DOI
  `10.13182/NSE08-66 <https://doi.org/10.13182/NSE08-66>`_.  BMC analyse
  the SN equations **discretised only in angle, with space kept
  continuous** (their analysis deliberately removes spatial differencing
  to isolate the angular error).  They prove that the angular axis
  carries its *own* diffusion-limit condition, independent of the
  spatial one.  Their p. 151 statement is the separability fact in the
  authors' own words: the spatial half "has been shown by Larsen, Morel,
  and Miller," while "retaining full first-order consistency can be
  important for **angular** discretisations" — the angular contribution
  they introduce.

The two conditions factorise.  In the leading-order (:math:`\varepsilon^0`)
diffusion limit, *any* weighted-diamond angular weight (step, diamond,
Morel--Montry) preserves consistency — BMC Eqs. (23)--(25).  The
**first-order** (:math:`\varepsilon^1`) limit carries a contamination
term :math:`\beta` (BMC Eq. 40), a **purely angular** functional of the
redistribution coefficients and quadrature,
:math:`\beta = \sum_m \mu_m\bigl[\alpha_{m+1/2}\mu_{m+1/2} -
\alpha_{m-1/2}\mu_{m-1/2}\bigr]`, which vanishes *only* for the
Morel--Montry weights (BMC Eq. 43, the weight of :eq:`mm-weights`).
Because :math:`\beta` depends on no spatial quantity, the angular
condition is provable on its own axis — exactly as the spatial
condition is provable on its own axis.  The diffusion limit needs
**both**:

.. math::

   \text{accurate diffusion limit}
   \;\;\Longleftrightarrow\;\;
   \underbrace{(\text{LMM spatial condition})}_{\text{depends on the spatial scheme only}}
   \;\;\wedge\;\;
   \underbrace{(\text{BMC angular condition},\ \beta = 0)}_{\text{depends on the angular weights only}}.

This conjunction of two single-axis conditions is *why* the Cartesian
error separates additively (each axis contributes its own consistency
defect, and the two defects add) and *why* a bad pairing can still break
the limit (independence of *selection* is not independence of
*consequence* — both conditions must hold simultaneously).

.. note::

   The literature's double-use of the name "linear-discontinuous" (LD)
   is itself evidence of the two-axis structure: LMM and every
   spatial-scheme paper list LD as a *spatial* scheme, while Lathrop
   (2000) lists "linear-discontinuous" among his *angular* differencing
   schemes.  The same trial-space name applies on either axis; the
   ORPHEUS registries disambiguate by axis (a spatial cell-update vs an
   angular closure), never a single ``LD`` enum.  This is the #158
   (spatial scheme) vs #6 (LD *angular* finite elements) distinction.

Cartesian separates, curvilinear gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The geometry split of :eq:`sn-space-angle-separability` follows
mechanically from *where the angular redistribution term lives*.

**Cartesian — additive separation.**  In slab / 2-D / 3-D Cartesian
geometry the curvilinear angular-redistribution term
:math:`\frac{1-\mu^2}{r}\,\partial_\mu\psi` is **absent**
(:math:`r \to \infty`; there is no :math:`1/r`).  The angular closure is
the :class:`~orpheus.sn.angular.closure.IdentityAngularClosure`,
which contributes no redistribution: each ordinate's spatial sweep is
fully independent of every other ordinate.  The Cartesian cell update
(:eq:`dd-2d-balance-form` / the slab balance) consumes only the per-axis
streaming ratios and :math:`\Sigma_t` — **no** :math:`\tau`, **no**
angular state.  The spatial error and the angular (quadrature) error are
generated by disjoint mechanisms, so they add:
:math:`E(h,N) \approx E_{\rm space}(h) + E_{\rm angle}(N)`, and the
mixed partial vanishes.  The operational signature is that the spatial
convergence **rate** is the same at every quadrature order
(N-independent O(h\ :sup:`2`)).

**Curvilinear — multiplicative gating.**  In the sphere / cylinder the
Morel--Montry angular thread

.. math::

   \psi_{n+\frac12} \;=\;
       \frac{\overline{\psi}_n - (1-\tau_n)\,\psi_{n-\frac12}}{\tau_n}

couples the ordinates *sequentially within a* :math:`\mu`-*level*, and
the coupling enters the **shared cell-balance denominator** of
:eq:`dd-curvilinear-scalar`: the redistribution divisor
:math:`(\Delta A_i / w_n)\,c_{\rm out}` (with
:math:`c_{\rm out} = \alpha_{\rm out}/\tau_n`) sits in the *same*
denominator that produces the spatial cell average
:math:`\overline{\psi}_{n,i}` the spatial closure then uses.  The
angular interpolation error of the :math:`\tau`-thread therefore
**caps** the accuracy the spatial closure can deliver: at a coarse
quadrature, refining :math:`h` cannot drive the cell average below the
angular floor, because the angular term contaminates the denominator
the spatial refinement acts through.  Hence the error *gates*:
:math:`E(h,N) \approx \max(E_{\rm space}(h), E_{\rm angle}(N))`.  You
cannot harvest fine-:math:`h` accuracy at coarse :math:`N`; both axes
must advance together.  The mechanism is documented in detail at
:ref:`sn-tau-c-on-cellvisit-live` (why :math:`\tau` is an angular
property that nonetheless flows through the spatial denominator) and the
shared-denominator algebra is :eq:`dd-curvilinear-scalar`.

.. warning::

   The gating is a property of *today's* curvilinear closure (the 1-D
   :math:`\eta`-march Morel--Montry thread), not a law of nature.  A
   future 2-D angular closure (#229) that resolves the
   :math:`(\eta,\varphi)` azimuthal variation the 1-D march cannot
   thread, or a higher-order spatial scheme (#158 / #6), would *lift*
   the gating — at which point the curvilinear error would begin to
   separate.  The ST5 gate is designed so that lifting the floor reddens
   the gating assertions (the coarse-N saturated h-ratio rises toward
   the O(h\ :sup:`2`) value), signalling that the regime changed; that
   redding is the intended signal to *re-tune* the gate to the new,
   better regime, **not** a regression.

Measured cross-term evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decomposition was established empirically by the four
``diag_sep_*`` probes (reproduced bit-for-bit after the Phase-2
:math:`\tau` carve) and is now pinned by the ST5 gate.  The measured
mixed-second-difference :math:`|M|/\max` spans **three orders of
magnitude** between the separable Cartesian and the gated sphere — a
clean discrimination band, not a brittle exact number.

.. list-table:: Measured space–angle error structure (2026-06-18,
   ``nx ∈ {20, 40, 80}``)
   :header-rows: 1
   :widths: 18 14 30 24 14

   * - Geometry
     - Regime
     - Scalar-flux L2 ladder (coarse → fine quadrature)
     - Spatial h-ratios (coarse-N / fine-N)
     - :math:`|M|/\max`
   * - Slab (isotropic)
     - **separates**
     - N=4 ``[5.42e-4, 1.35e-4, 3.38e-5]`` · N=16 ``[5.40e-4, 1.35e-4, 3.37e-5]``
     - ``[4.01, 4.00]`` / ``[4.01, 4.00]`` (N-independent O(h²))
     - **0.0047**
   * - Slab (P1 aniso)
     - **separates**
     - N=4 floor ``6.80e-3`` · N=16 floor ``6.79e-3`` (flat, angular floor)
     - flat at both N (floor N-independent to <0.3 %)
     - **0.0038**
   * - Cylinder
     - **gates**
     - :math:`n_\varphi`\ =8 ``[1.95e-2, 1.91e-2, 1.90e-2]`` · :math:`n_\varphi`\ =16 ``[8.05e-3, 7.47e-3, 7.37e-3]``
     - ``[1.02, 1.00]`` (saturated); azimuthal floor drops 2.58× at :math:`n_\varphi` 8→16
     - **0.019** (small only because :math:`E \approx E_{\rm angle}` swamps)
   * - Sphere
     - **gates**
     - N=8 ``[1.47e-2, 5.40e-3, 4.69e-3]`` · N=32 ``[1.50e-2, 3.71e-3, 9.29e-4]``
     - ``[2.71, 1.15]`` (saturates) / ``[4.04, 4.00]`` (O(h²) recovers)
     - **0.411**

The reading of the table:

* **Slab (both rows): separable.**  The spatial h-ratio is :math:`\approx
  4` (O(h\ :sup:`2`)) at *every* quadrature order — the spatial rate is
  blind to :math:`N`.  The isotropic row has a genuine O(h\ :sup:`2`)
  window; the P1-anisotropic row sits at a flat MMS/angular floor that
  is the *same* at every :math:`N`.  Both have :math:`|M|/\max \le
  0.005` — the cross-term vanishes whether or not the angular axis is
  active.  The P1 row is the load-bearing control: separability survives
  an *active* angular term, so it is not an artefact of the isotropic
  degeneracy.

* **Sphere: gating, the discriminator.**  At coarse N=8 the finest
  spatial h-ratio collapses to **1.15** — refinement saturates at the
  angular floor.  At fine N=32 the *same* spatial ladder recovers
  :math:`\approx 4.00` (O(h\ :sup:`2`)).  The spatial rate *depends on*
  :math:`N` — the defining gating fact — and the cross-term
  :math:`|M|/\max = 0.411` sits three orders above the Cartesian
  ceiling.

* **Cylinder: the extreme of gating.**  There is no pre-floor
  O(h\ :sup:`2`) window at any practical azimuthal order — the
  :math:`(\eta,\varphi)` variation a 1-D :math:`\eta`-march cannot
  thread exactly (#229) — so :math:`E \approx E_{\rm angle}(n_\varphi)`
  and the spatial h-ratio is :math:`\approx 1` at fixed :math:`n_\varphi`.
  The positive signature is the floor's azimuthal scaling: it drops
  :math:`2.58\times` when :math:`n_\varphi` doubles.  (The cylinder's
  small :math:`|M|/\max` is *not* evidence of separability — it is small
  because the angular floor so dominates that the spatial delta
  :math:`\Delta E_h` in the denominator is itself near zero; the gating
  is read from the *saturation* and the *azimuthal scaling*, not the
  cross-term magnitude.)

.. note::

   The scalar (weight-summed) L2 of the table is, by construction, blind
   to a *wrong angular closure* — the Morel--Montry :math:`\alpha`-dome
   telescopes under :math:`\sum_n w_n \psi_n` (vv-principles L27 / the
   per-ordinate-flat-flux discipline).  Because the curvilinear gating is
   *itself* an angular-closure phenomenon, the ST5 gate adds a
   **per-ordinate** leg
   (``test_curvilinear_gating_per_ordinate_not_blind``) that reproduces
   the sphere gating signature from the max-over-ordinates per-ordinate
   L2 (N=8 finest h-ratio 1.16 saturates; N=32 recovers ≈3) — so the
   gate cannot be telescoped blind to a future angular-closure
   regression.  That leg corrects a measured 1/W normalisation trap:
   ``case.psi_exact(r, μ_n)`` returns :math:`A(r) + B(r)\mu_n` *without*
   the :math:`1/W` factor by its own contract, while the solver stores
   the per-ordinate flux *with* it — the reference must be divided by
   :math:`W = \sum_n w_n` before comparison, else a 2× mismatch swamps
   the metric.

The pole-cell (#233) × azimuthal-floor interference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gating law has a concrete consequence for the two curvilinear
defect families.  The spatial pole-cell :math:`\mathcal{O}(h)` order (#233,
documented at :ref:`sn-pole-cell-spatial-closure` and minted as ERR-059
in :ref:`sn-curvilinear-aniso-norm-reconciliation`) and the azimuthal
angular floor (#229) are **not independent contributors** to the
curvilinear error — they *interfere through the gating*.

The mechanism: the angular thread's interpolation error sets the floor
that spatial refinement saturates at.  So the pole-cell spatial defect
(#233) is only *visible* — only the dominant error — once the angular
floor (#229) has been pushed below it by refining :math:`N`.  At a
coarse quadrature the #229 angular floor *masks* the #233 pole-cell
order entirely (the spatial ladder saturates before the
:math:`\mathcal{O}(h)` pole-cell term emerges); only at a fine
quadrature does the spatial ladder run long enough for the pole-cell
order to surface.  This is precisely why the sphere N=8 ladder saturates
at 1.15 while N=32 recovers O(h\ :sup:`2`): the same spatial closure,
read through two different angular floors.

This interference is the reason the two issues must be characterised
*together* rather than as separate spatial and angular bugs, and the
reason a fix to one cannot be validated in isolation: lifting the #229
angular floor (a 2-D angular closure) would *expose* the #233 pole-cell
order that the floor currently masks.  The gating law
:eq:`sn-space-angle-separability` makes this dependency explicit — the
curvilinear error is :math:`\max(E_{\rm space}, E_{\rm angle})`, so
whichever defect is larger *hides* the other.

The permanent pin: the ST5 characterization gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The decomposition is pinned permanently by
:mod:`tests.sn.verification.mms.test_space_angle_separability` (Issue
#236 Phase 3, sub-task ST5), an **L1** MMS characterisation gate modelled
on the pole-cell characterisation gate
``test_curvilinear_pole_cell_characterization.py`` (#233).  It carries
``@pytest.mark.verifies("sn-space-angle-separability")`` against
:eq:`sn-space-angle-separability`.  Its six legs pin both regimes as
*positive* signatures (never as an xfail-pending-fix):

* **Cartesian, separable** — ``test_cartesian_slab_iso_space_angle_separable``
  (N-independent O(h\ :sup:`2`) spatial rate, :math:`|M|/\max < 0.05`)
  and ``test_cartesian_slab_p1_aniso_floor_n_independent`` (the active-
  angular-axis control: the P1 floor is N-independent and the cross-term
  stays :math:`\approx 0`).
* **Curvilinear, gating** —
  ``test_sphere_spatial_rate_is_quadrature_gated`` (the discriminator:
  coarse-N saturates, fine-N recovers O(h\ :sup:`2`); the proven
  ``@catches("ERR-026")`` catcher via the fine-N O(h\ :sup:`2`)-recovery
  assertion), ``test_sphere_cross_term_large_discriminates_from_cartesian``
  (:math:`|M|/\max > 0.15`),
  ``test_cylinder_spatial_saturates_at_azimuthal_floor`` (spatial
  saturation + azimuthal floor scaling), and
  ``test_curvilinear_gating_per_ordinate_not_blind`` (the L27
  angular-aware per-ordinate leg, also ``@catches("ERR-026")``).

The gate is *characterisation*, not calcification: if a future 2-D
angular closure (#229) or higher-order spatial scheme (#158 / #6) lifts
the curvilinear gating, the gating assertions are designed to redden so
the regime change is *signalled* and the gate is re-tuned to the new
(better) regime — they are not xfails awaiting a fix.  The ``@slow``
mark reflects that the curvilinear solves dominate the ~2 s wall-clock,
not that the gate is optional.

.. _sn-scheme-vs-angular-weight:

The spatial scheme and the angular weight — the redistribution factorizes
=========================================================================

Everything above derives the curvilinear machinery with **one** spatial
moment per cell: the balance :eq:`balance-general` carries a scalar
:math:`\bar\psi_{n,i}`, the geometry factor is the scalar
:math:`\Delta A_i/w_n`, and :math:`\tau` is one number per ordinate
(:eq:`mm-weights`).  That is diamond difference, and it is what ships.

A two-moment spatial scheme — linear discontinuous
(:ref:`discretization-ld`), which carries a cell average **and an
independent slope** — makes every one of those objects wider.  Before such
a member can be built, one question has to be answered, and it is not a
question about the spatial scheme at all:

   **Is the flux-dip-eliminating weight** :math:`\tau` **a property of the
   angular scheme alone, or does it acquire a spatial-cell index once the
   cell carries two moments?**

The stakes are a public contract.  The angular closure delivers
:math:`\tau` through
:attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`
— an :math:`(N,)` array, **one weight per ordinate, no cell index**.  If
the answer were *"it depends on the cell"*, that accessor would have to
widen to per-:math:`(\text{ordinate}, \text{cell})`, and :math:`c_{\rm in}`
/ :math:`c_{\rm out}` with it.

The answer is **no**, and this chapter is the reason.  It is a *theorem*
about the structure of the redistribution operator, not a measurement on a
fixture — which matters, because :ref:`sn-two-questions-two-instruments`
shows the instrument that settles it is **structurally incapable** of
settling the question that actually turns out to be dangerous.

.. note:: **Four symbol overloads, local to this chapter, kept because
   they are the spellings the code and the literature use.**

   * :math:`R` alone is the **redistribution Gram**
     (:eq:`sn-redistribution-gram-eq`) — Palmer & Adams's own
     :math:`R_k`, and the code's ``R_spatial``.  It is **not** the
     sphere's outer radius, which this page writes only in the phrase
     :math:`r = R` and never bare.
   * :math:`A_{\rm angular}` (always subscripted) is the **angular factor**
     of :eq:`sn-redistribution-tensor-product-eq`.  It is neither the loss
     operator :math:`A = L + C - S - N_{2n} - B` nor a face area; face areas keep
     their own subscripts, :math:`A_{i\pm1/2}`, abbreviated
     :math:`A_{\pm}` inside a single cell.
   * :math:`P` is the **spatial moment projection** in
     :ref:`sn-tau-arity-theorem`.  Adams & Martin's :math:`P` in the
     literature table below is *their* normalised slope coordinate
     :math:`2(r-r_k)/\Delta r_k` — this chapter writes that one
     :math:`\xi`.
   * :math:`\tau` is the Morel--Montry **angular** weight
     (:eq:`mm-weights`) everywhere on this page.  Where the seed section
     needs an **optical** depth it writes :math:`\tau_{\rm opt}`
     (:eq:`discretization-optical-depth`), and the four-way overload table
     is at :ref:`discretization-transmission-ladder`.

.. admonition:: Key facts of this chapter
   :class: tip

   * The curvilinear angular-redistribution operator is a **tensor
     product** :eq:`sn-redistribution-tensor-product-eq`,
     :math:`\mathcal{R} = R_{\rm spatial} \otimes
     A_{\rm angular}(\tau,\alpha,w)`: the geometry-and-basis factor
     carries only the *moment* index, the angular factor only the
     *ordinate* index.  Since 2026-08-26 that is also the tree's own
     structure — the spatial factor on
     :attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.delta_A`,
     the member-independent angular factor on
     :class:`~orpheus.sn.angular.redistribution.AngularRedistribution`,
     and the fused :math:`\Delta A_i \otimes 1/w_n` cache that used to
     straddle both **retired**.
   * ⟹ the diffusion-limit contamination condition is the **identical**
     angular scalar :eq:`sn-contamination-factor`, whose free symbols are
     :math:`\{\mu, w, \tau\}` — no spatial symbol appears — and
     :math:`\beta = 0` annihilates the contamination in **every**
     spatial-moment row, for an **arbitrary symmetric** :math:`R`.
   * ⟹ :math:`\tau` keeps its arity (:ref:`sn-tau-arity-theorem`).  A
     scalar convex combination commutes with every linear map, so both of
     :math:`\tau`'s defining conditions are *the same scalar statement in
     every moment component*; a per-cell :math:`\tau` is an
     **overdetermined** system whose every row returns the same
     per-ordinate value.  **None of the hypotheses mentions a basis** —
     that is the whole content.
   * The geometry-and-basis factor is the **one-measure-down Gram**
     :eq:`sn-redistribution-gram-eq`, and the shipped scalar
     :math:`\Delta A_i` is exactly its :math:`(0,0)` corner.  It is
     non-diagonal on the **sphere** (:math:`R_{01}/R_{00} = h/6r_c`,
     rising to :math:`\tfrac13` at the pole cell) and **diagonal** on the
     cylinder — so a per-moment-row flat-flux gate is a real gate on the
     sphere and reads :math:`0 = 0` on the cylinder.
   * ⛔ The risk is **not** :math:`\tau`; it is the **seed**
     (:ref:`sn-seed-cone-risk`).  The starting-direction equation carries
     no angular redistribution at all, so it inherits the *spatial*
     scheme's positivity ladder verbatim — and a starting-cosine error of
     **1.6 %** (:math:`S_4`) falling to **0.05 %** (:math:`S_{32}`)
     reproduces the *entire* diamond-scheme contamination that
     :math:`\tau` exists to remove.
   * ⛔ Morel--Montry's own summary rule — *"…as long as the starting flux
     is not seriously* **under** *estimated"* — is refuted for
     :math:`N \ge 4` (:ref:`sn-morel-montry-summary-rule-refuted`): the
     **safe direction inverts** between :math:`S_2` and :math:`S_4`.

.. _sn-redistribution-tensor-product:

The factorization theorem
-------------------------

**The derivation, from the conservative form.**  Take the 1-D spherical
transport equation :eq:`transport-spherical` and weight it by a cell basis
function :math:`b_k(r)` against the volume measure :math:`dV = 4\pi r^2\,dr`
over one cell :math:`[r_-, r_+]`.  Write the *m-measure* inner product

.. math::

   \langle f, g\rangle_m \;\equiv\; 4\pi\!\int_{r_-}^{r_+} f\,g\,r^{m}\,dr ,

so the volume measure is :math:`m = 2`.  Three of the four terms behave as
one would expect — the streaming integrates by parts to a face-area-weighted
trace minus a volume-measure weak gradient, and collision and source carry
the volume Gram :math:`\langle b_k, \cdot\rangle_2`.  The **angular
redistribution term does not**:

.. math::
   :label: sn-one-measure-down

   \int_{r_-}^{r_+}\! b_k(r)\,\frac{1}{r}\,
     \frac{\partial}{\partial\mu}\bigl[(1-\mu^2)\psi\bigr]\,4\pi r^2\,dr
   \;=\;
   \frac{\partial}{\partial\mu}\Bigl[(1-\mu^2)\,
     \underbrace{4\pi\!\int_{r_-}^{r_+}\! b_k\,\psi\,r\,dr}_{\textstyle
       \langle b_k,\psi\rangle_1}\Bigr] .

.. vv-status: sn-one-measure-down documented
.. (vv-status rationale) the one-measure-down identity: the 1/r of the
   curvilinear angular-redistribution term cancels one power of the volume
   measure, so the half-angle fluxes enter moment row k through the m = 1
   inner product. A derivation-decomposition step of the already-verified
   balance :eq:`balance-general` (whose one-moment case IS the shipped ΔA/w
   term); not a solver claim.

**The** :math:`1/r` **of the redistribution term eats one power of the
volume measure.**  So the half-angle fluxes reach moment row :math:`k`
through the *one-measure-down* functional, not through the volume Gram
every other term uses.  With :math:`\psi` linear in the cell,
:math:`\langle b_k,\psi\rangle_1 = \tfrac12 (R\,\vec\psi)_k`, and
discretising :math:`\partial_\mu` the Lathrop--Carlson way — the
:math:`\alpha` recursion :eq:`alpha-recursion` — gives the moment-row
redistribution

.. math::
   :label: sn-redistribution-moment-row

   \text{Redist}_k \;=\; \frac{1}{w_n}\Bigl[
     \alpha_{n+\frac12}\bigl(R\,\vec\psi_{n+\frac12}\bigr)_k
     \;-\;
     \alpha_{n-\frac12}\bigl(R\,\vec\psi_{n-\frac12}\bigr)_k \Bigr] .

.. vv-status: sn-redistribution-moment-row documented
.. (vv-status rationale) the multi-moment generalisation of the shipped
   scalar redistribution term of :eq:`balance-general`; at one moment it IS
   that term, since R₀₀ = ΔA. Structural, not a solver claim — no
   multi-moment curvilinear member is implemented (Issue #158).

At one moment this is :math:`(\Delta A_i/w_n)\bigl[\alpha_{n+\frac12}
\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}\bigr]` — the shipped
term of :eq:`balance-general`, verbatim, because
:math:`R_{00} = \Delta A_i` exactly.

**The theorem.**  Read :eq:`sn-redistribution-moment-row` as an operator on
the joint (moment :math:`\times` ordinate) index and it factors:

.. math::
   :label: sn-redistribution-tensor-product-eq

   \mathcal{R} \;=\; R_{\rm spatial} \;\otimes\;
                     A_{\rm angular}(\tau,\ \alpha,\ w) ,
   \qquad
   \bigl(\mathcal{R}\,\psi\bigr)_{k,n}
   \;=\; \bigl[R_{\rm spatial}\bigr]_{kj}\;
         \bigl[A_{\rm angular}\bigr]_{n n'}\;\psi_{j,n'} .

.. vv-status: sn-redistribution-tensor-product-eq documented
.. (vv-status rationale) the factorization of the curvilinear
   angular-redistribution operator into a moment-index factor R_spatial
   (pure geometry × basis, μ-independent) and an ordinate-index factor
   A_angular (pure quadrature × closure, r-independent). A structural
   statement about :eq:`sn-redistribution-moment-row`; its one-moment case
   is the shipped operator and is gated by :eq:`streaming-equilibrium`. Not
   a solver claim — no multi-moment curvilinear member exists (Issue #158).

:math:`R_{\rm spatial}` is **pure geometry and basis** and carries no
:math:`\mu`; :math:`A_{\rm angular}` is **pure quadrature and closure** and
carries no :math:`r`.  They act on **disjoint index sets**.  That is the
whole theorem.  Write :math:`R \equiv R_{\rm spatial}` from here on, which
is also Palmer & Adams's own notation for it (their Eq. (9),
":math:`R_k` = angular redistribution matrix").  Three consequences follow
immediately.

**(1) Every angular functional of the redistribution factors as**
:math:`R\times`\ **(the one-moment angular scalar).**  In particular, redo
the Bailey--Morel--Chang first-order asymptotic diffusion-limit expansion
with the moment *vector* carried, and the first angular moment of the
redistribution comes out as

.. math::
   :label: sn-ld-contamination-vector

   \sum_n \mu_n\Bigl[
     \alpha_{n+\frac12}\,R\,\vec\psi^{(1)}_{n+\frac12}
     - \alpha_{n-\frac12}\,R\,\vec\psi^{(1)}_{n-\frac12}\Bigr]
   \;=\;
   R\,\Bigl(-\,W_2\,\tfrac12\vec\phi^{(1)}
            \;+\; \beta\,\vec g \;+\; \beta_e\,\vec e\Bigr) ,

.. vv-status: sn-ld-contamination-vector documented
.. (vv-status rationale) the first angular moment of the multi-moment
   redistribution, with the geometry Gram factored out — a consequence of
   :eq:`sn-redistribution-tensor-product-eq` and the classical BMC
   expansion. Its one-moment case is :eq:`sn-contamination-factor`, which is
   the object the shipped instruments compute. Structural, not a solver
   claim; the multi-moment member is unimplemented (Issue #158).

with :math:`\beta` the **identical** contamination scalar of
:eq:`sn-contamination-factor`, :math:`\vec g` the leading-order gradient
vector, and :math:`\beta_e` the coefficient of the *seed* defect
:math:`\vec e` that :ref:`sn-seed-cone-risk` is about.  The geometry matrix
:math:`R` factors out of **every** angular sum, at every moment count,
because of :eq:`sn-redistribution-tensor-product-eq`.

**(2)** :math:`\beta` **cannot acquire spatial content.**  Its free symbols
are :math:`\{\mu, w, \tau\}` — a quadrature and a closure weight.  No
spatial symbol appears anywhere in it, and no redefinition of the spatial
operators can reach it, because they enter :eq:`sn-ld-contamination-vector`
only through :math:`\vec g`, which is :math:`\tau`-free.

**(3)** :math:`\beta = 0` **kills every moment channel at once.**  Setting
:math:`\beta = 0` annihilates the whole vector :math:`\beta\,R\,\vec g` in
both rows, for an **arbitrary symmetric** :math:`R` — the moment structure
plays no part.  And the converse holds: :math:`R` is positive-definite on
every admissible cell (below), so :math:`\det R \ne 0` and
:math:`\beta R \vec g = 0` for all :math:`\vec g` **iff** :math:`\beta = 0`.
The multi-moment problem therefore generates **no weaker condition** on
:math:`\tau` than the one-moment problem does — not a different one, and
not an additional one.

.. admonition:: The factorization is the tree's structure, not only a
   theorem about it
   :class: important

   :eq:`sn-redistribution-tensor-product-eq` was derived to answer the
   arity question; on **2026-08-26** it became the shape of the code, and
   the two halves now have separate owners:

   .. list-table::
      :header-rows: 1
      :widths: 22 34 44

      * - factor
        - carries
        - where it lives
      * - :math:`R_{\rm spatial}`
        - the **moment** index — geometry :math:`\times` basis, no
          :math:`\mu`
        - :attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.delta_A`
          on the streaming operator (its :math:`(0,0)` corner is all a
          one-moment scheme needs), beside ``face_areas``
      * - :math:`A_{\rm angular}`
        - the **ordinate** index — quadrature :math:`\times` closure, no
          :math:`r`
        - split by ownership: the member-**independent** part (the
          :math:`\alpha` dome and :math:`\mu_{\rm start}`, per level) on
          :class:`~orpheus.sn.angular.redistribution.AngularRedistribution`,
          from the single producer
          :func:`~orpheus.sn.angular.redistribution.angular_redistribution`;
          the member's own :math:`\tau` and the derived
          :math:`c_{\rm in}` / :math:`c_{\rm out}` on the angular closure
          (:ref:`sn-tau-closure-owned`)

   Three consequences of the split are worth reading off, because each was
   a defect before it:

   * **Cartesian is the NEUTRAL element, not a special case.**  A slab has
     no curvature, so its dome is identically zero and its starting
     direction is the diameter ray.  Spelling those values instead of
     ``None`` is what let a six-field per-coordinate ``Optional`` union
     die: "no redistribution" is no longer separately representable.
   * **The fused** :math:`\Delta A_i \otimes 1/w_n` **cache had to go**,
     and the factorization says why — it welded a factor from each side, so
     neither side could own it.  See the note under
     :ref:`sn-geometry-factor` above.
   * ⭐ **The angular factor's own split is load-bearing for a second
     member.**  :class:`~orpheus.sn.angular.redistribution.AngularRedistribution`
     deliberately does **not** carry :math:`\tau`: the dome and the
     starting direction are shared by every angular-closure member, while
     :math:`\tau` is the member's *choice* (Morel--Montry's barycentric
     weight, plain diamond's :math:`\tfrac12`, the neutral
     :math:`\tau \equiv 1`).  A shared object holding :math:`\tau` would
     forbid a second member by construction.

.. warning:: **The correct argument is this one, and the tempting one is
   invalid.**  It is tempting to argue "the shipped
   :func:`~orpheus.derivations.discrete.sn.angular_differencing.contamination_beta`
   takes no spatial argument, therefore :math:`\beta` is
   spatial-scheme-independent."  ⛔ That inference is **not valid**, and it
   has a name: ``vv-principles`` Mode 8, the **SIGNATURE-tautological**
   class.  The signature admits no spatial argument because the *analysis
   that produced* :math:`\beta` held space **continuous** — Morel--Montry
   1984 state their Eq. (1) is "the :math:`S_n` equations discretized in
   angle only" (printed p. 617) and :cite:`BaileyMorelChang2010`
   Eqs. (30)--(41) carry :math:`\partial/\partial r` symbolically
   throughout (printed pp. 154--155).  A claim is *unfalsifiable* through
   that function at every quadrature order and every mesh, because the
   varying input cannot physically reach the object; a green reading there
   carries **zero** information.  Had the spatial scheme introduced a
   dependence, the answer would have been a *new* function with a spatial
   argument.  ⟹ **a type signature is evidence about an author's
   assumptions, never about a theorem.**

.. _sn-redistribution-gram:

The Gram, and what five independent primaries say about it
-----------------------------------------------------------

The geometry-and-basis factor of
:eq:`sn-redistribution-tensor-product-eq` is the **one-measure-down Gram**
of the spatial scheme's own basis.  On the Legendre pair
:math:`b_0 = 1`, :math:`b_1 = \xi \equiv 2(r-r_c)/h` with
:math:`r_c = \tfrac12(r_-+r_+)` and :math:`h = r_+-r_-`:

.. math::
   :label: sn-redistribution-gram-eq

   R \;=\; \Delta A_i
   \begin{bmatrix}
     1 & \dfrac{h}{6 r_c} \\[8pt]
     \dfrac{h}{6 r_c} & \dfrac13
   \end{bmatrix}
   \quad(\text{sphere}),
   \qquad
   R \;=\; \Delta A_i
   \begin{bmatrix} 1 & 0 \\ 0 & \tfrac13 \end{bmatrix}
   \quad(\text{cylinder}) .

.. vv-status: sn-redistribution-gram-eq documented
.. (vv-status rationale) the one-measure-down Gram of the {1, ξ} cell basis,
   normalised so R₀₀ = ΔA. Literature-cross-confirmed (Adams-Martin 1992
   Eq. A.1a/A.1b magnitudes; Machorro 2007 Eq. 3.5; Hill 1975 Table V) and
   derived independently. Its (0,0) corner IS the shipped ΔA of
   :eq:`balance-general`, gated by :eq:`streaming-equilibrium`; the 2×2 is
   unimplemented (Issue #158), so this is definitional, not a solver claim.

One spelling serves both arms.  Writing :math:`d = 3` for the sphere,
:math:`d = 2` for the cylinder and :math:`d = 1` for the slab (where
:math:`\Delta A = 0` and the whole term vanishes),

.. math::
   :label: sn-redistribution-gram-uniform

   R_{kj}(\text{cell}) \;=\; \Delta A_i\;
     \frac{\langle b_k, b_j\rangle_{d-2}}{\langle b_0, b_0\rangle_{d-2}} ,

.. vv-status: sn-redistribution-gram-uniform documented
.. (vv-status rationale) the geometry-uniform spelling of
   :eq:`sn-redistribution-gram-eq` — one body, parameterised by the measure
   exponent d−2, which also absorbs the geometry-dependent factor of two in
   the ORPHEUS α normalisation (:ref:`sn-alpha-normalization`).
   Definitional; unimplemented (Issue #158).

which also absorbs the geometry-dependent factor of two in the ORPHEUS
:math:`\alpha` normalisation (:ref:`sn-alpha-normalization`): the sphere
needs :math:`R = 2\langle\cdot,\cdot\rangle_1` to land on :math:`\Delta A`,
the cylinder needs :math:`R = 1\cdot\langle\cdot,\cdot\rangle_0`.

Four properties, each of which does work later:

#. :math:`R_{00} = \Delta A_i` **exactly** — the shipped scalar geometry
   factor is the :math:`(0,0)` corner of the matrix, so a two-moment member
   is a strict widening of the shipped one, not a re-derivation.
#. :math:`R_{11}/R_{00} = \tfrac13` **exactly, in both geometries** — the
   same :math:`\theta = \tfrac13` as the slab-LD mass
   (:eq:`discretization-ld-system`).  The *diagonal* of the redistribution
   Gram is free.
#. **The sphere's off-diagonal is not.**
   :math:`R_{01}/R_{00} = h/(6 r_c) = h/\bigl(3(r_-+r_+)\bigr)`, which is
   :math:`O(h/r)` in the bulk and **exactly** :math:`\tfrac13` at the pole
   cell (:math:`r_- = 0`) — its maximum over all admissible cells, since
   :math:`r_-+r_+ \ge h`.  On the test mesh
   :math:`r \in \{0,\,0.7,\,1.9,\,2.05,\,5.0\}` it reads
   ``[0.3333, 0.1538, 0.0127, 0.1395]``.  So **the new coupling is
   strongest exactly where the flux dip lives.**  It also makes :math:`R`
   positive-definite everywhere: :math:`\det R = \Delta A^2(\tfrac13 - x^2)`
   with :math:`x = R_{01}/R_{00} \le \tfrac13 < 1/\sqrt3`.
#. **The cylinder's is zero.**  One measure down from :math:`r\,dr` is the
   *flat* :math:`dr`, on which the Legendre basis is orthogonal — so the
   cylinder's redistribution Gram is diagonal and the average/slope
   coupling does not exist there at all.

.. important:: **The per-moment-row flat-flux identity is a real gate on
   the sphere and reads** :math:`0 = 0` **on the cylinder.**  The
   canonical curvilinear L0 gate (:eq:`streaming-equilibrium`) asserts
   per-ordinate flat-flux consistency.  With two moments it must be
   asserted **per moment row**, and row 1 is where the off-diagonal earns
   its keep.  Under a flat flux :math:`\vec\psi = (\psi_0, 0)` the slope
   row's streaming leaves the residue :math:`\mu_n\bigl[A_+ + A_- -
   2V/h\bigr]\psi_0`, and the only thing available to cancel it is
   :math:`R_{10}`:

   .. list-table::
      :header-rows: 1
      :widths: 16 42 42

      * - geometry
        - :math:`A_+ + A_- - 2V/h` (streaming residue)
        - :math:`R_{10}` (redistribution)
      * - **sphere**
        - :math:`\tfrac{4\pi}{3}\,h^{2}`
        - :math:`\tfrac{4\pi}{3}\,h^{2}` — a genuine cancellation of two
          non-zero terms
      * - **cylinder**
        - :math:`0` (:math:`A_+ + A_- = 2V/h` exactly, the area being
          linear in :math:`r`)
        - :math:`0`

   ⟹ **a per-moment-row flat-flux gate written only on the cylinder is
   structurally blind**: it passes for *any* off-diagonal whatsoever,
   including a wrong one and including none.  The gate must run on the
   **sphere**.  Equivalently: lumping :math:`R` to its Legendre diagonal
   breaks the slope row on the sphere and is a no-op on the cylinder.
   (This is the same shape as the ``vv-principles`` Mode-12 blindness the
   :math:`\sigma_y`-fold produces for :math:`\beta` — see the warning at
   :ref:`the β-blindness warning <sn-tau-beta-diagnostic-blind>`.)

The published record — and a published typo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:eq:`sn-redistribution-gram-eq` is **not** new mathematics, and the
corpus said otherwise until 2026-08-25.  The complete 1-D spherical
two-moment system has been in print since 1992:

.. list-table:: the curvilinear LD record, page-verified
   :header-rows: 1
   :widths: 27 33 40

   * - primary
     - what it carries
     - the Gram, in its own notation
   * - **Adams & Martin (1992)**, NSE **111**\ (2):145--167, App. A,
       printed pp. 160--161 (:cite:`AdamsMartin1992`)
     - Eqs. (A.1a)/(A.1b) — the complete spherical two-moment balance;
       (A.2a)/(A.2b) — the **weighted-diamond angular closure applied per
       spatial moment**; (A.3) — the upwind face closure; (A.4a--d) — the
       mass integrals :math:`V_k, W_k, X_k = \int r^2\{1,P,P^2\}dr`
     - :math:`\{r_k\Delta r_k,\; \Delta r_k^{2}/6,\; r_k\Delta r_k/3\}`
       ⟹ ratios :math:`\Delta r/(6r_k)` and :math:`\tfrac13`
   * - **Machorro (2007)**, JCP **223**:67--81, Eqs. (3.3)/(3.5),
       printed p. 70
     - a discontinuous-Galerkin weak form in which the angular term is a
       single :math:`\mu`-boundary integral with **one sign**
     - :math:`\int r\,dr,\;\int r(r-r_i)dr,\;\int r(r-r_i)^2 dr`
       ⟹ the same two ratios
   * - **Hill (1975)**, LA-5990-MS (ONETRAN), Eqs. (33b)/(33c)/(35a),
       Table V, printed pp. 10--12
     - the cylindrical **and** spherical LD systems, with the geometric
       coefficients tabulated per geometry
     - :math:`\Delta A_i` and :math:`z_5`, both **positive**, both
       entering (35a) with :math:`+`
   * - **Palmer & Adams (1993)**, UCRL-JC-111847, Eqs. (9)/(14),
       printed pp. 2--4, 8
     - names the object: :math:`R_k` = "angular redistribution matrix",
       with the half-angle fluxes as **vectors**
     - the r-z BLD :math:`R_{i,j} = \tfrac{\Delta r\Delta z}{36}
       [[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]]` — manifestly symmetric,
       all-positive
   * - **Wu, Xie & Fischer (1999)**, NSE **133**\ (3):350--357, Eq. (27)
       (:cite:`WuXieFischer1999`)
     - a nodal method that applies the angular diamond to the whole
       spatial-moment **vector**
     - kernels carrying :math:`(r_i + a_i r')^{j-1}/W_m` against
       :math:`P_n(r')`

⚠ **The printed minus signs in Adams--Martin (A.1a)/(A.1b) are a typo.**
As printed, the two :math:`\psi^x`-coupled redistribution terms carry
:math:`-`, so the four weights read
:math:`\bigl[\begin{smallmatrix} r_k\Delta r & -\Delta r^2/6 \\
+\Delta r^2/6 & -r_k\Delta r/3\end{smallmatrix}\bigr]` — **not symmetric,
and with a negative** :math:`(2,2)` **entry**.  The magnitudes match
:eq:`sn-redistribution-gram-eq` exactly; only those two signs differ.  Three
reasons the all-plus reading is the correct one, and two confirmations:

#. **Symmetry.**  Their Sec. III.B sets weight functions equal to basis
   functions (Galerkin, :math:`v = b`), so the coupling matrix must be
   symmetric.
#. **Positive-definiteness.**  :math:`\int r P^2\,dr > 0` strictly (a
   positive weight over the cell), so the :math:`(2,2)` entry cannot be
   negative.  It is a Gram matrix.
#. **Their own sibling block.**  The removal terms in the *same two
   equations* are :math:`\sigma_{tk}[V_k\psi + W_k\psi^x]` and
   :math:`\sigma_{tk}[W_k\psi + X_k\psi^x]` — the symmetric
   :math:`[[V,W],[W,X]]`.  Only the redistribution block breaks the
   pattern.
#. Machorro's weak form carries the angular term as **one** integral with
   **one** sign; no :math:`\pm` alternation between the average and slope
   rows exists anywhere in it.
#. ONETRAN's :math:`\Delta A_i` and :math:`z_5` are both positive and both
   enter Eq. (35a) with :math:`+`; and Palmer--Adams's fully-lumped
   :math:`R_k` entries are exactly the **row sums** of the all-positive
   Gram, which row-sum lumping of a sign-alternating matrix could not
   produce.

⛔ **And the trap that makes this worth a warning rather than a footnote:
the sign error is invisible to a conservation check.**  Both redistribution
terms telescope over :math:`m` (they are :math:`\sum_m` of a difference),
so global particle balance holds for **either** sign.  Discriminate with
the Galerkin symmetry, or with a slope-exciting fixture — never with a
balance test.  (This is ``vv-principles`` anti-pattern #8 in the
literature rather than in the tree.)

.. _sn-redistribution-rank-contradiction:

⛔ The rank contradiction — two published families, and ORPHEUS must choose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primaries agree on the Gram's *entries* and disagree on **what the
angular closure acts on** — which changes the **rank** of a production
operator:

.. list-table::
   :header-rows: 1
   :widths: 24 20 22 34

   * - family
     - angular device
     - applied to
     - redistribution coupling
   * - Adams--Martin (A.2a)/(A.2b); Palmer--Adams Eq. (9);
       Wu--Xie--Fischer Eq. (27)
     - weighted / plain diamond
     - **every** spatial moment
     - the full symmetric :math:`2\times2` Gram
   * - **ONETRAN** (Hill 1975, Eq. (32), from the plain angular diamond
       Eq. (30))
     - plain diamond
     - the spatial **average only** — one scalar half-angle flux per
       space-angle cell
     - **rank 1**: :math:`(\alpha_{m+\frac12}/w_m)\,
       [\Delta A_i;\, z_5]\otimes[1,\,1]`

Chronology explains part of it (ONETRAN 1975 predates
:cite:`MorelMontry1984`), but the per-moment-versus-average choice is
independent of the weighted-versus-plain choice.  **Both are published and
both shipped in production codes; ORPHEUS must pick one consciously rather
than inherit one by accident.**

⭐ The two published options are not two answers to one question — they
are **two different values of an index the factorization had silently
conflated**.  :eq:`sn-redistribution-tensor-product-eq` is sharpened by
splitting it:

* :math:`n_{\rm mom}` — how many spatial moments the **scheme** carries
  (DD 1, LD 2);
* :math:`n_{\rm thread}` — how much of the spatial representation the
  **angular device** propagates through its half-angle recurrence
  (ONETRAN 1, Adams--Martin 2).

:math:`R` is then the **rectangular pairing** of the two bases under the
one-measure-down geometry,

.. math::
   :label: sn-redistribution-gram-rectangular

   R_{kj} \;=\; \Delta A_i\;
     \frac{\bigl\langle b^{\rm scheme}_k,\; b^{\rm thread}_j
           \bigr\rangle_{d-2}}
          {\bigl\langle b_0, b_0\bigr\rangle_{d-2}} ,
   \qquad \text{shape } (n_{\rm mom},\, n_{\rm thread}) ,

.. vv-status: sn-redistribution-gram-rectangular documented
.. (vv-status rationale) the rectangular generalisation of
   :eq:`sn-redistribution-gram-uniform`, pairing the SCHEME's spatial basis
   with the ANGULAR THREAD's spatial basis. Its 1×1 case is the shipped ΔA;
   the 2×1 case reproduces ONETRAN's own printed [ΔA_i ; z_5] column and the
   2×2 case Adams-Martin's printed magnitudes. Structural, not a solver
   claim; unimplemented (Issue #158).

so that :math:`R` is owned by **neither side alone**:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - (scheme, angular thread)
     - shape
     - :math:`R`, and whether ORPHEUS ships it
   * - DD, one-moment thread
     - :math:`1\times1`
     - :math:`[\Delta A_i]` — ✅ **this is what ships**; it is
       :attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.delta_A`
   * - LD, per-moment thread (Adams--Martin)
     - :math:`2\times2`
     - :eq:`sn-redistribution-gram-eq` — ⛔ **not built** (Issue #158);
       derived here and cross-checked against the literature above
   * - LD, average-only thread (ONETRAN)
     - :math:`2\times1`
     - :math:`\bigl[\Delta A_i;\; \Delta A_i\,h/(6r_c)\bigr]
       = \bigl[\Delta A_i;\; \tfrac{4\pi}{3}h^2\bigr]`, matching ONETRAN's
       own :math:`[\Delta A_i;\, z_5]` — ⛔ **not built**

The factorization survives the sharpening: **each axis contributes exactly
one index to** :math:`R`.  What changes is that a hook exposing :math:`R`
cannot be a pure property of the spatial scheme — it must take the angular
thread's basis as an argument.

.. note:: The corpus asserted until 2026-08-25 that the curvilinear LD cell
   closure was *unpublished and must be derived*.  That was **wrong for the
   sphere** (Adams--Martin 1992 has carried the full moment balance since
   1992, in a paper that was sitting in the local literature folder) and
   **overstated for the cylinder** (ONETRAN writes the cylindrical system
   at report level in 1975).  The false negative traces to a one-query
   literature search whose effective denominator was near zero.  ⛔ Do not
   read "published" as "safe to transcribe": Palmer & Adams 1993 conclude
   that **plain** spherical LD *fails* the thick diffusion limit (a
   three-point removal term, unphysical boundary conditions, interior
   scalar flux low by :math:`\sim2\times` on their two-cell test), with
   only the fully-lumped and corner-balance forms passing.  Their
   organising principle is stated on printed p. 6: *"The main ingredient is
   'locality' of operators."*  The positivity half of that story, and the
   family it opens, is at :ref:`discretization-transmission-ladder` in
   :doc:`/theory/foundations/discretization`.

.. _sn-tau-arity-theorem:

τ is a per-ordinate scalar — a theorem, not a measurement
----------------------------------------------------------

   **Theorem.**  Let :math:`P : \mathcal F(\text{cell}) \to
   \mathbb R^{n_{\rm mom}}` be the spatial moment projection and let the
   angular closure be :eq:`wdd-closure`,
   :math:`\psi_n = (1-\tau_n)\psi_{n-\frac12} + \tau_n\psi_{n+\frac12}`,
   with :math:`\tau_n` independent of :math:`r`.  Then :math:`\tau` is a
   **per-ordinate scalar**, in both of its defining conditions, for
   **every** spatial representation.

**Proof, in one line.**  A scalar convex combination **commutes with every
linear map**, so :math:`P(\text{blend}) = \text{blend}(P)`.  Both defining
conditions are then *the same scalar statement in every component*:

* **Condition 1 — cone membership,** :math:`\tau \in [0,1]` (the shipped
  predicate P3, raised by
  :func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`).
  A convex combination of two elements of a convex set lies in that set.
  That statement mentions **no basis**, so no change of spatial
  representation can give it spatial content.
* **Condition 2 — the barycentric condition,** exactness on
  :math:`\operatorname{span}\{1,\mu\}`, which is what :eq:`mm-weights`
  *is*.  Solved independently in each moment component it returns
  :math:`\tau = (\mu_n - \mu_{n-\frac12})/(\mu_{n+\frac12} -
  \mu_{n-\frac12})` **every time** — which is a consequence of linearity,
  not a survey, and was checked symbolically at **4 of 4** moment
  components.  A per-:math:`(\text{ordinate}, \text{cell})` :math:`\tau` is
  therefore an **overdetermined system whose every row returns the same
  per-ordinate value**: row 0 alone determines :math:`\tau` and row 1 then
  holds identically.

**Check the hypotheses, and notice what is absent.**  :math:`\tau` is
:math:`r`-independent by construction — it is an angular weight, a
barycentric coordinate in the direction cosine.  :math:`P` is linear — an
:math:`L^2` projection onto a fixed basis.  The positive cone :math:`K` is
convex — an intersection of half-spaces, one per point of the cell.
**None of the three mentions a basis.**  That is the entire content of the
theorem, and it is why no expansion is required to settle the arity: the
asymptotic corroboration of :ref:`sn-redistribution-tensor-product` adds
magnitudes, not the decision.

⟹ :attr:`~orpheus.sn.angular.closure.AngularClosureBase.tau_per_ordinate`
**keeps its arity.**  So do :math:`c_{\rm in}` and :math:`c_{\rm out}`: the
derivative of the redistribution *vector* with respect to the ordinate
moment *vector* is exactly :math:`c_{\rm out}\,R/w_n` — **one scalar times
the geometry matrix**, with no row-dependent :math:`c` and no
moment-dependent :math:`\tau`.  What a two-moment scheme changes is only
*what those scalars multiply*.

.. warning:: ⚠ **The theorem certifies the CLOSURE.  It does not certify
   the MARCH.**  :math:`\tau \in [0,1]` says the *specification* is a convex
   combination.  The *algorithm* runs it backwards
   (:eq:`wdd-face`): the inverted closure
   :math:`\psi_{n+\frac12} = \bigl(\psi_n - (1-\tau_n)\psi_{n-\frac12}
   \bigr)/\tau_n` has coefficients :math:`\bigl(1/\tau,\,
   -(1-\tau)/\tau\bigr)`, the second **negative for every**
   :math:`\tau < 1`.  It is an *extrapolation*, not a convex combination,
   and a defect on the half-angle thread is amplified by

   .. math::
      :label: sn-halfangle-march-amplification

      \rho_m \;=\; \prod_{j\le m} \Bigl(-\frac{1-\tau_j}{\tau_j}\Bigr),
      \qquad |\rho| > 1 \ \text{ wherever } \ \tau < \tfrac12 .

   .. vv-status: sn-halfangle-march-amplification documented
   .. (vv-status rationale) the defect-amplification factor of the INVERTED
      angular closure :eq:`wdd-face` — a direct product of its own
      coefficients, definitional rather than a solver claim. The τ values it
      is evaluated on are produced by ``morel_montry_tau_per_level``, whose
      [0,1] membership is separately gated (predicate P3).

   `[M]` on Gauss--Legendre: at :math:`S_8`, **4 of 8** Morel--Montry
   :math:`\tau` lie below :math:`\tfrac12` (range
   :math:`[0.3923, 0.6077]`) and :math:`\max|\rho| = 2.0159`; at
   :math:`S_{32}`, **16 of 32** (range :math:`[0.3898, 0.6102]`) and
   :math:`\max|\rho| = 3.1457`.  The march is stable in practice because
   its input is *solved*, not inherited — but the amplification is the
   mechanism by which a seed error becomes an angular-thread error, which
   is what :ref:`sn-seed-cone-risk` quantifies.

.. _sn-two-questions-two-instruments:

What the asymptotic expansion cannot settle
--------------------------------------------

The instrument that settles the arity question is a **first-order
asymptotic diffusion-limit expansion**: scale
:math:`\sigma_t \to \sigma/\varepsilon`, :math:`\sigma_s \to
\sigma/\varepsilon - \varepsilon\sigma_a`, :math:`Q \to \varepsilon Q`,
write :math:`\vec\psi = \vec\psi^{(0)} + \varepsilon\vec\psi^{(1)} +
\varepsilon^2\vec\psi^{(2)} + \dots`, and read off the
:math:`O(\varepsilon)` balance.  That instrument is **structurally
incapable** of settling positivity, and saying so is not a caveat — it is
the reason the next two sections exist.

**The mechanism.**  The expansion's ansatz writes each order as a *smooth
function of the cell*, so a **sign-alternating cell-to-cell mode is
excluded by the ansatz itself**.  No order of the expansion can see it:
not a higher order, not a finer mesh, not a tighter tolerance.  This is
``vv-principles`` **Mode 12** — the measured functional's invariance group
contains the entire error class — and it is why Palmer & Adams carry

   *"the solution of the discretized transport equation must limit to the
   solution of a* **stable and reasonable** *discretization of the diffusion
   equation"*

as a **separate** acceptance criterion (printed p. 1, their criterion 1),
and why their verdict against bare LD is phrased as *"a three-point removal
term, which is* **known to cause oscillations**\ *"* (printed p. 3) rather
than as a failure of the expansion.

⟹ **Two questions, two instruments**, and neither substitutes for the
other:

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - question
     - instrument
     - what it CANNOT see
   * - Is the closure **consistent** in the diffusion limit — and does the
       weight :math:`\tau` acquire a cell index?
     - the asymptotic expansion; :math:`\beta = 0`
       (:eq:`sn-contamination-factor`)
     - any sign-alternating mode — **excluded by the ansatz**
   * - Does the scheme keep the flux in the **positive cone** :math:`K`?
     - the transmission multiplier's sign
       (:ref:`discretization-transmission-ladder`)
     - the diffusion-limit consistency of the *converged* answer

**And a limit on** :math:`\beta = 0` **itself, which the expansion does
show.**  The cancellation that makes :math:`\beta` vanish at the
Morel--Montry weight is **not** term by term — it is a cancellation
*between the two sweep halves*.  `[M]` at the Morel--Montry :math:`\tau`,
the half-range partial sums are individually large and exactly opposite:

.. list-table:: half-range partial sums of the contamination scalar
   :header-rows: 1
   :widths: 16 28 28 28

   * - :math:`N`
     - :math:`\beta^{-}` (:math:`\mu<0`)
     - :math:`\beta^{+}` (:math:`\mu>0`)
     - :math:`\beta^{-}+\beta^{+}`
   * - 4
     - ``+0.101808``
     - ``-0.101808``
     - ``+1.4e-17``
   * - 8
     - ``+0.119111``
     - ``-0.119111``
     - ``-4.2e-17``
   * - 16
     - ``+0.123476``
     - ``-0.123476``
     - ``-2.8e-17``
   * - 32
     - ``+0.124610``
     - ``-0.124610``
     - ``-6.9e-17``

So :math:`\beta = 0` is a **global angular identity** that relies on
:math:`\vec\psi^{(1)}` being affine in :math:`\mu` **across**
:math:`\mu = 0`.  Under a two-moment spatial scheme that is a *theorem
only where the leading-order cell-edge jumps vanish* — which the
:math:`O(\varepsilon^0)` solvability condition forces in the interior, and
which fails in an unresolved boundary layer, at a source discontinuity, or
in the first cell off a vacuum face.  There the contamination reads
:math:`\beta^-R\vec g^- + \beta^+R\vec g^+` with :math:`\vec g^+ \ne
\vec g^-`, and **no choice of** :math:`\tau` **cleans it**: the residue is
:math:`\approx 0.12\times` the directional slope discrepancy, which is not
a small coefficient.  This is a *spatial* effect, and it is the same family
as Morel--Montry's own coarse-mesh caveat (printed p. 630: the
correspondence "is lost with a coarse spatial mesh").  ⟹ it is an argument
for gating a two-moment curvilinear member on a **resolved-layer** fixture,
not for a per-cell :math:`\tau`.

.. _sn-seed-cone-risk:

Where the risk actually lives — the starting direction
--------------------------------------------------------

The arity question was never the risk.  The risk is the **seed**, and the
reason is one line of geometry that this chapter has already used twice.

**The starting-direction equation carries no angular redistribution at
all.**  The redistribution coefficient vanishes at both ends of every
angular march (:math:`\alpha_{1/2} = \alpha_{M+1/2} = 0`,
:ref:`the dome-closure contract <sn-alpha-dome-closes>`), and the
continuous statement is the same
one: :math:`(1-\mu^2) = 0` at :math:`\mu = \mp1`, so the transport equation
collapses to :eq:`sn-direct-seed-pole-straight-characteristic`, a pure
radial ODE with **no coupling to any other ordinate**
(:ref:`sn-direct-seed-solve`).

⟹ **the seed is a purely spatial advection--reaction solve, and it
inherits the spatial scheme's positivity ladder verbatim — in the one place
where there is nothing angular to damp it.**  The ladder is at
:ref:`discretization-transmission-ladder` in
:doc:`/theory/foundations/discretization`; the row that applies is the
spatial scheme's own.

**What ships, measured.**  ORPHEUS marches the seed with diamond
difference at :math:`|\mu| = 1`
(:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`,
the Hébert (3.434)/(3.435) recurrence; the engine is cosine-agnostic and
takes **path** widths, so a folded cylinder level's optical depth is
:math:`\Sigma_t\,\Delta r/|\eta_{\rm start}|`).  Its source-free
transmission is therefore DD's Padé(1,1) row, and it **changes sign at
optical path depth 2** and heads for :math:`-1`.  `[M]` on the shipped
function, source-free, unit entry face, eight equal cells at cell optical
depth 3:

.. list-table:: the shipped seed march, :math:`\Sigma_t\,\Delta r = 3` per cell
   :header-rows: 1
   :widths: 12 30 30 28

   * - cell
     - :math:`\psi_{1/2}` (marched)
     - exact :math:`e^{-\tau_{\rm opt}}` at the cell centre
     - ratio to the previous cell
   * - 1
     - ``+0.400000``
     - ``0.223130``
     - —
   * - 2
     - ``-0.080000``
     - ``0.011109``
     - :math:`-0.2`
   * - 3
     - ``+0.016000``
     - ``0.000553``
     - :math:`-0.2`
   * - 4
     - ``-0.003200``
     - ``0.000028``
     - :math:`-0.2`
   * - 5
     - ``+0.000640``
     - :math:`1.4\times10^{-6}`
     - :math:`-0.2`
   * - 6
     - ``-0.000128``
     - :math:`6.9\times10^{-8}`
     - :math:`-0.2`

The per-cell ratio is :math:`(2-3)/(2+3) = -\tfrac15` exactly, so the seed
**alternates in sign, cell to cell**, and in the thick diffusion limit
(:math:`\tau_{\rm opt} = \Sigma_t h \to \infty`) it does so with a ratio
tending to :math:`-1` — no decay at all.  A two-moment (bare LD) seed would
move the onset from :math:`\tau_{\rm opt}=2` to :math:`\tau_{\rm opt}=3`
and the far-field ratio to :math:`-2/\tau_{\rm opt}`; a *lumped* member
would remove it entirely (:ref:`discretization-transmission-ladder`).

.. _sn-beta-eff-seed-sensitivity:

How much seed error undoes the Morel--Montry weight entirely
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contamination scalar :eq:`sn-contamination-factor` is evaluated on the
cell edges the closure *implies* — the :math:`\tau`-implied march
:math:`\tilde\mu_{n+\frac12} = \bigl(\mu_n -
(1-\tau_n)\tilde\mu_{n-\frac12}\bigr)/\tau_n`, the same recursion the
solve-free :math:`\nu`-closure diagnostic marches (see
:ref:`the absorber retirement <sn-tau-absorber-retirement>` in
:doc:`/theory/foundations/structured_geometry`; there it is written
:math:`\nu` and asked to land on the level's far endpoint, here it is
written :math:`\tilde\mu` and summed).  Its **anchor** is the starting
cosine, :math:`\tilde\mu_{1/2} = -1`.

Because that recursion is affine in its anchor and the sum is linear, the
contamination is **exactly affine in the effective starting cosine**:

.. math::
   :label: sn-beta-eff-affine

   \beta_{\rm eff}(\mu_s) \;=\; \beta \;+\; (\mu_s + 1)\,\beta_e ,
   \qquad
   \beta_e \;\equiv\; \frac{\partial\beta_{\rm eff}}{\partial\mu_s}
   \ \ \text{(a constant)} .

.. vv-status: sn-beta-eff-affine documented
.. (vv-status rationale) the exact affinity of the contamination scalar in
   the anchor of the τ-implied edge march — a consequence of that march
   being affine in its anchor and the sum being linear. Structural. The
   β_eff(−1) = β leg IS the shipped
   :func:`orpheus.derivations.discrete.sn.angular_differencing.morel_montry_beta`
   (= 3β/2 in its Σw = 1 normalisation), gated on both legs by
   ``tests/sn/sweep/curvilinear/test_angular_beta_identity.py``; the affinity
   in μ_s has no gate because no production path perturbs the anchor.

`[M]` verified at :math:`\mu_s = -1.4,\,-1.0,\,-0.6,\,+0.3` and
:math:`N = 2 \dots 32`, to :math:`\le 3\times10^{-11}` absolute.  So **one
number decides how much a seed error costs**, and it can be read against
the contamination the Morel--Montry weight was invented to remove — namely
:math:`\beta` at the *diamond* weight :math:`\tau \equiv \tfrac12`:

.. list-table:: the starting-cosine error equivalent to the whole diamond-scheme dip
   :header-rows: 1
   :widths: 10 22 22 22 24

   * - :math:`N`
     - :math:`\beta` at :math:`\tau\equiv\tfrac12`
     - :math:`\beta_e` at the M-M :math:`\tau`
     - :math:`\beta_e` at :math:`\tau\equiv\tfrac12`
     - :math:`|\mu_s+1|` equivalent
   * - 2
     - ``+1.031e-01``
     - ``+9.107e-01``
     - ``+6.667e-01``
     - ``0.1132``
   * - 4
     - ``-1.786e-03``
     - ``-1.111e-01``
     - ``-4.222e-02``
     - ``0.0161``
   * - 8
     - ``-2.287e-05``
     - ``-4.276e-03``
     - ``-1.999e-03``
     - ``0.0053``
   * - 16
     - ``-3.776e-07``
     - ``-2.596e-04``
     - ``-1.256e-04``
     - ``0.0015``
   * - 32
     - ``-6.266e-09``
     - ``-1.662e-05``
     - ``-8.107e-06``
     - ``0.0004``

(Gauss--Legendre, sphere; :math:`\beta` at the Morel--Montry :math:`\tau`
is machine zero throughout, :math:`\le 8.6\times10^{-17}`.  The last column
is :math:`|\beta_{\tau=1/2}| / |\beta_e|`, the starting-cosine error whose
contamination equals the diamond scheme's.)

⟹ **from** :math:`S_4` **on, a starting-cosine error of ~1.6 %, falling to
0.05 % at** :math:`S_{32}`\ **, already gives back everything the
Morel--Montry weight removed — and the leverage GROWS with angular
order.**  :math:`\tau` and the seed are not two independent knobs; on any
production quadrature the seed dominates.  A sign-alternating seed march is
that error in its most extreme form, and it closes the loop with an
independent published diagnosis: Walters & Morel found an origin error in a
discontinuous spherical scheme under fine-radial / coarse-angular meshes
and attributed it to *insufficient starting-direction information*
(reported by Machorro 2007, printed p. 79); both they and Machorro repair
it with **quadratic-in-angle** functions in the cells bordering
:math:`\mu = -1`, and Lathrop 2000 §III.D adopts the same hybrid
independently.

.. important:: **The design consequence, stated once.**  Choose the seed
   march's *spatial* closure for its **cone** behaviour, and gate that
   choice on the cone — never on the flat-flux identity
   :eq:`streaming-equilibrium`, which is exact for every scheme in this
   family and therefore blind to the choice.  The gate that discriminates
   is the sign of the transmission multiplier
   (:ref:`discretization-transmission-ladder`), and the flag that should
   carry it is being split for exactly this reason (Issue #408 — the
   current ``is_positivity_preserving`` conflates three properties, and the
   one the seed needs is the *first*).

.. _sn-morel-montry-summary-rule-refuted:

⛔ Morel--Montry's own summary rule, refuted for N ≥ 4
-------------------------------------------------------

:cite:`MorelMontry1984` close with a rule of thumb that is quoted far more
often than the analysis behind it (printed p. 630):

   *"…the flux dip can be expected to be eliminated with any spatial
   differencing scheme as long as the starting flux is not seriously*
   **under** *estimated relative to the weighted fluxes."*

The affine law :eq:`sn-beta-eff-affine` makes that rule checkable in one
line, because its whole content is the **sign of** :math:`\beta_e`:
underestimating the starting flux moves :math:`\mu_s` in one direction, and
whether that drives :math:`\beta` positive or negative is decided by
:math:`\operatorname{sign}(\beta_e)`.

⛔ `[M]` **the sign flips between** :math:`N = 2` **and** :math:`N \ge 4`.
From the table above, at the Morel--Montry :math:`\tau`:

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - :math:`N`
     - :math:`\beta_e`
     - which direction of seed error is dangerous
   * - 2
     - ``+9.107e-01``
     - **under**\ estimating drives :math:`\beta < 0` — exactly as
       published
   * - 4
     - ``-1.111e-01``
     - **over**\ estimating does
   * - 8
     - ``-4.276e-03``
     - **over**\ estimating
   * - 16
     - ``-2.596e-04``
     - **over**\ estimating
   * - 32
     - ``-1.662e-05``
     - **over**\ estimating

The same inversion holds at the diamond weight
(:math:`+6.667\times10^{-1}` at :math:`N=2`, negative from :math:`N=4`).

**The cause, and it is a familiar one.**  :math:`S_2` — Gauss
:math:`S_2` — is the case Morel & Montry actually computed.  The rule is a
**universal generalised from a single sample**, and the sample happens to
be the one member of the family on the other side of a sign change.  It is
``vv-principles`` anti-pattern #13 in the literature rather than in the
tree, and the 1984 paper is not careless: the rule is *true* of everything
its authors measured.

⚠ **Read the refutation with its own honest caveat.**  :math:`|\beta_e|`
falls **five orders of magnitude** from :math:`N=2` to :math:`N=32`, so
what inverts is the *direction*, while the *stakes* collapse.  And
Morel--Montry's "effective starting cosine" is itself an
:math:`S_2`/:math:`S_4`-class construct — it presumes :math:`\psi` affine
in :math:`\mu`.  ⟹ the durable lesson is **not** "the rule points the other
way now".  It is: **do not use a direction heuristic at all — evaluate**
:math:`\beta_{\rm eff}` **directly** from the solve's own half-angle
thread, through the :math:`\tilde\mu`-implied edge set that
:func:`~orpheus.derivations.discrete.sn.angular_differencing.morel_montry_beta`
already marches.  The instrument exists and costs no solve; the heuristic
was only ever a substitute for it.

.. note:: **What is asserted rather than measured here.**  Every numerical
   leg of this chapter is **spherical**.  The theorems are
   geometry-independent — they need only that :math:`R` be
   :math:`\mu`-independent and that the cone be convex, both of which hold
   on the cylinder, where :math:`R` is moreover *diagonal* so the
   :math:`R_{01}` channel does not exist at all — but the cylinder's
   numbers have not been taken.  ⛔ When they are, the instrument must be
   :func:`~orpheus.derivations.discrete.sn.angular_differencing.nu_closure_residual`
   and **never** :math:`\beta`: on a :math:`\sigma_y`-folded arc
   :math:`\beta` is a symmetry identity rather than a measurement, and
   garbage passes it (:ref:`the β-blindness warning <sn-tau-beta-diagnostic-blind>`).  Also
   unquantified: how much *scalar-flux* error a sign-alternating seed
   produces end to end.  That needs a pole-resolved fixed-source sphere at
   :math:`\Sigma_t h > 3` on the seed's cells, with a fixture chosen
   **outside** :math:`\operatorname{span}\{1,\mu\}` — the closure is exact
   on that span, so a fixture inside it lies in the closure's own kernel
   and cannot rank anything (``vv-principles`` #24(d)/(e)).

Substituting the WDD Closure into the Balance Equation
=======================================================

Combining the balance equation :eq:`balance-general` with the WDD
angular closure :eq:`wdd-closure` and the standard spatial DD
(:math:`\psi_{n,i} = \frac{1}{2}(\psi_{\rm in}^s + \psi_{\rm out}^s)`,
:math:`\psi_{\rm out}^s = 2\psi_{n,i} - \psi_{\rm in}^s`), define:

.. math::

   c_{\rm out} &= \frac{\alpha_{n+\frac12}}{\tau_n} \\[6pt]
   c_{\rm in}  &= \frac{(1-\tau_n)}{\tau_n}\,\alpha_{n+\frac12}
                 + \alpha_{n-\frac12}

The cell-average angular flux is then:

.. math::
   :label: dd-solve

   \psi_{n,i} = \frac{
       S_i V_i
       + |\mu_n|(A_{\rm in} + A_{\rm out})\,\psi_{\rm in}^s
       + \dfrac{\Delta A_i}{w_n}\, c_{\rm in}\, \psi_{n-\frac12}
   }{
       2|\mu_n|\, A_{\rm out}^s
       + \dfrac{\Delta A_i}{w_n}\, c_{\rm out}
       + \Sigt{} V_i
   }

where the superscript :math:`s` denotes spatial face fluxes, and
:math:`A_{\rm in}`, :math:`A_{\rm out}` are the cell face areas in the
direction of neutron travel (see :ref:`sweep-algorithm` for their
definition).  This is the equation the per-cell
update solves —
:func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`,
consumed by
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update` —
and, in vectorized scan form, the CumprodScan fast path
(:ref:`sn-tau-c-on-cellvisit-live`).  ⛔ The helper named here was the
scalar twin ``cell_balance_terms`` until P4.9a retired it onto the
vectorized one (2026-08-28).  Note what the survivor does **not**
take: neither :math:`c_{\rm in}` nor :math:`c_{\rm out}` nor
:math:`\Delta A/w` reaches it — the caller multiplies the two
redistribution products out and passes them assembled
(:ref:`sn-p49a-closure-owns-the-march`), so the equation above is the
*balance* the helper closes, not the argument list it takes.

Geometry Comparison
====================

.. list-table::
   :header-rows: 1
   :widths: 15 28 28 29

   * - Aspect
     - Cartesian
     - Spherical
     - Cylindrical
   * - Streaming cosine
     - :math:`\mu`
     - :math:`\mu`
     - :math:`\eta` (radial)
   * - Face area :math:`A`
     - 1 (per unit area)
     - :math:`4\pi r^2`
     - :math:`2\pi r`
   * - Volume :math:`V`
     - :math:`\Delta x`
     - :math:`\tfrac{4}{3}\pi(r_{\rm out}^3 - r_{\rm in}^3)`
     - :math:`\pi(r_{\rm out}^2 - r_{\rm in}^2)`
   * - :math:`\Delta A`
     - 0
     - :math:`4\pi(r_{\rm out}^2 - r_{\rm in}^2)`
     - :math:`2\pi(r_{\rm out} - r_{\rm in})`
   * - Redistribution
     - None
     - :math:`+(\Delta A/w)\,[\alpha\psi]`
     - :math:`+(\Delta A/w)\,[\alpha\psi]`
   * - :math:`\alpha` scope
     - N/A
     - Global (all :math:`N` ordinates)
     - Per :math:`\mu`-level
   * - :math:`\alpha` recursion variable
     - N/A
     - :math:`\mu`
     - :math:`\eta`
   * - Quadrature required
     - GL or Lebedev
     - GL
     - Product or Level-Sym

Curvilinear 1D: Sequential Ordinate Sweep
==========================================

For spherical and cylindrical geometries, the angular redistribution
couples successive ordinates, preventing vectorisation across the
ordinate dimension.  The sweep proceeds cell-by-cell,
ordinate-by-ordinate:

**Spherical:** Ordinates are processed from most negative :math:`\mu` to
most positive (a single global sequence).  Negative-:math:`\mu` ordinates
sweep **inward** (outer boundary to centre); positive-:math:`\mu`
ordinates sweep **outward** (centre to outer boundary).

**Cylindrical:** For each :math:`\mu`-level, azimuthal ordinates are
processed from most-inward (:math:`\eta = -\sin\theta`) to most-outward
(:math:`\eta = +\sin\theta`).  Negative-:math:`\eta` ordinates sweep
inward; :math:`\eta \approx 0` ordinates have no radial streaming
(pure redistribution); positive-:math:`\eta` ordinates sweep outward.

At each cell, the sweep solves :eq:`dd-solve` for :math:`\psi_{n,i}`,
then updates:

1. **Spatial face flux:**
   :math:`\psi_{\rm out}^s = 2\psi_{n,i} - \psi_{\rm in}^s`
2. **Angular face flux:**
   :math:`\psi_{n+1/2} = (\psi_{n,i} - (1-\tau_n)\psi_{n-1/2})/\tau_n`
   (using M-M weights)
3. **Scalar flux accumulation:**
   :math:`\phi_i \mathrel{+}= w_n \psi_{n,i}`

The spatial face flux propagates to the next cell; the angular face flux
propagates to the next ordinate on the same cell.

Implemented today by the unified walk, and since P4.9a by **two owners,
one per axis**: the spatial face flux comes out of the per-cell
:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update` over
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk` visits, while
the angular face flux is advanced by the closure's own
:func:`~orpheus.sn.angular.closure.march_psi_half_step`, applied by the
walk that composes them.  The vectorized
:class:`~orpheus.sn.loss_representation.CumprodScan` fast path
(:doc:`loss_representation`) runs the same two closures in the
scan-normal spelling (:ref:`sn-p49a-two-forms`).

.. _sn-pole-angular-closure-protocol:

The angular closure
===================

.. note:: **Contract evolution (Issue #236 Phase 2 B2 → Issue #248).**
   This subsection originally introduced the angular-closure contract
   as a ``@runtime_checkable`` ``PoleAngularClosure`` **Protocol**
   (structural typing, Phase B). Issue #236 Phase 2 B2 retyped every
   production consumer (matvec / sweep / geometry / scheme /
   cell-balance) onto the
   :class:`~orpheus.sn.angular.closure.AngularClosureBase`
   **ABC** and made the three strategy methods
   (:meth:`~orpheus.sn.angular.closure.AngularClosureBase.precompute_psi_state`,
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.cell_contribution`,
   ``angular_adjoint``) ``@abstractmethod`` on it. That left the
   Protocol orphaned and divergent — it carried the ``c_*``/``tau``
   accessors and a legacy ``__call__`` bundle but **not** the three
   strategy methods — so **Issue #248 deleted it** along with the dead
   ``__call__`` bundle (and its ``tau_mm`` argument) and the orphaned
   recurrence helpers. The ABC is now the **sole** angular-closure
   contract; production consumes
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.precompute_psi_state`
   + :meth:`~orpheus.sn.angular.closure.AngularClosureBase.cell_contribution`,
   never ``__call__``. The narrative below preserves Phase B's
   reasoning; read "strategy ABC" wherever the original said "strategy
   Protocol". The section anchor ``sn-pole-angular-closure-protocol``
   is retained (it is cross-referenced from
   :doc:`/theory/foundations/boundary_conditions` and elsewhere); only the human
   label "protocol" is now loose — the contract is an ABC.

   ⭐ **Second name correction (P4.9b, 2026-08-28).** The word "pole" has
   now gone from the family too. A cylinder has no pole in the sense a
   sphere does; what the two closures actually are is one **spatial** and
   one **angular**, so the base is
   :class:`~orpheus.sn.angular.closure.AngularClosureBase`, the hub
   attribute is ``angular_closure``, and the operator and the
   representation carry one symmetric, greppable pair of slot names
   (:ref:`sn-p49b-operator-poses-with-closures`). Member names
   (:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`,
   ``IdentityAngularClosure``) are untouched — only the family-defining
   spellings moved — and genuine poles keep the word: the sphere's polar
   cap, the :math:`\mu = -1` starting direction, and Hébert's Carlson
   *coupled-pole* seed are all named correctly. The anchor stays, for the
   same reason as before: an anchor is an address, not a claim, and
   `[M]` it carries three **cross-document** references
   (:doc:`/theory/foundations/boundary_conditions`, :doc:`history`,
   :doc:`/theory/verification/sn`) plus three intra-document ones — and a
   cross-document ``:ref:`` that misses its target renders as plain text
   with no warning at any build severity, so a rename there is a silent
   break rather than a caught one. Both spellings therefore appear below;
   the old one only in past-tense history.

Phase B addresses **Defect 3** of Issue #168 — the angular-redistribution
truncation gap on angularly-varying :math:`\psi`.  The pre-Phase-B
operator carried inline τ-symmetric interpolation
:math:`\psi_{n+1/2} \approx \tau_n\,\psi_{n+1} + (1-\tau_n)\,\psi_n`,
which is the **flat-flux collapse** of the curvilinear cell balance
(:cite:`Hebert2009` Eq. 3.428) closed with the Morel--Montry weighted
angular recurrence (:cite:`MorelMontry1984`; :cite:`BaileyMorelChang2010`
Eqs. (42)/(43)) — exact when :math:`\psi` is constant in
:math:`\mu`, but only :math:`\mathcal{O}(1)` accurate on
angularly-varying :math:`\psi`.  Phase B lifts this evaluation into
a :class:`~orpheus.sn.angular.closure.AngularClosureBase`
strategy ABC — analogous to Phase A's
``BoundaryFaceFlux`` —
and shipped **three concrete strategies** trading off bit-identity,
flat-flux invariance, and asymptotic accuracy:

.. note:: **Superseded strategy set (PR-TYPED-6c Step 7, 2026-05-18; #248,
   2026-06-18).** Of the three Phase-B strategies below, only
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
   survives — it is now the **sole production curvilinear closure and the
   default**. The two ablation strategies
   ``LegacyTauSymmetricInterpolation`` (the pre-Phase-B inlined
   :math:`\tau`-symmetric form) and ``BaileyFlatFluxRedist`` (the
   algebraic flat-flux collapse) were retired once
   ``MorelMontryAngularSweep`` became the default (no production
   consumer remained), and the divergent
   ``test_spherical_flat_flux_legacy_matches_bailey_collapse`` gate went
   with them. The three-strategy exploration is preserved below as the
   design record of *why* the M-M weighted-DD recurrence is the right
   closure.

* ``LegacyTauSymmetricInterpolation``
  — bit-for-bit reproduction of the pre-Phase-B inlined math
  (the :math:`\tau`-symmetric form).  Was the **default** through
  Phase B, preserving the
  curvilinear regression-snapshot bit-identity contract and the
  per-ordinate flat-flux invariant the ERR-026 evidence test relied on.
  Carried Defect 3 by design — the truncation gap was *reproducible*
  so verification probes could cross-check against the
  documented behaviour.

* ``BaileyFlatFluxRedist``
  — the algebraic flat-flux collapse
  :math:`R_{n,i,g} = (\Delta A/w)\,(\alpha_{n+1/2} - \alpha_{n-1/2})\,
  \psi_{n,i,g} / V_i = -\mu_n\,\Delta A_i\,\psi_{n,i,g} / V_i`
  (using :eq:`alpha-dome-recursion`).  Equivalent to the legacy form
  on flat :math:`\psi` (a now-retired flat-flux-identity test pinned
  this), and used as a structurally simpler bridge to the
  flat-flux invariant in unit tests.

* :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
  — the canonical per-cell Morel--Montry **weighted**-DD angular
  recurrence (:cite:`MorelMontry1984`; the implemented form is
  :cite:`BaileyMorelChang2010` Eqs. (42)/(43) — see
  :ref:`sn-tau-source-of-record`, and note that Hébert ships the *plain*
  angular diamond, not this):

  .. math::
     :label: pole-mm-recurrence

     \phi_{1/2,i,g} &= \psi_{1/2,i,g}, \\
     \phi_{n+1/2,i,g} &= \frac{\phi_{n,i,g}
                              \;-\; (1 - \tau_n)\,\phi_{n-1/2,i,g}}{\tau_n},
     \qquad n = 1, \ldots, M_p .

  .. implements:: pole-mm-recurrence
     :by: orpheus.sn.angular.closure.march_psi_half_step

     **Implemented by** 5 sites, all inside the angular closure since
     P4.9a (2026-08-28).  Every symbol that executes this equation's
     arithmetic is declared, not only the canonical one: a test is
     adjudicated against the transcription it actually ran, and this
     equation's five ``verifies`` gates call ``compute_psi_half_per_level``
     rather than the step function underneath it.  The relation
     itself is ``march_psi_half_step`` — the second line, and the single
     production spelling of its subtract-then-divide form
     (:ref:`sn-p49a-two-forms`).  ``_psi_half_grid_single_level`` is the
     batch kernel that writes the **seed** line and loops the step,
     delegating the body so the delegation is bit-neutral;
     ``compute_psi_half_per_level`` is its public exposure and
     ``MorelMontryAngularSweep._psi_half_grid_for_level`` the mesh-bound
     wrapper that reads :math:`\tau_n` from the strategy;
     ``AngularClosureBase.advance_psi_half`` is the per-cell entry the
     degenerate cylindrical-axis path uses.  ⛔ Before these declarations
     the equation carried exactly one *inferred* implementer,
     ``_OneDimScanWalk._ensure_pole_mirror``, matched on the shared token
     *pole* — a method that mirrors pole **faces** and marches nothing.

  .. implements:: pole-mm-recurrence
     :by: orpheus.sn.angular.closure._psi_half_grid_single_level

  .. implements:: pole-mm-recurrence
     :by: orpheus.sn.angular.closure.compute_psi_half_per_level

  .. implements:: pole-mm-recurrence
     :by: orpheus.sn.angular.closure.MorelMontryAngularSweep._psi_half_grid_for_level

  .. implements:: pole-mm-recurrence
     :by: orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half

  The march runs over the level's own :math:`M_p` ordinates — the sphere is
  the single-level case :math:`M_p = N`, the cylinder loops it per
  :math:`\mu`-level, each with its own :math:`\alpha`-dome,
  :math:`\Delta A/w` and :math:`\tau`.

  **The seed** :math:`\psi_{1/2,i,g}` is the level's **starting-direction
  flux**: the value of the field on the closed ray at the level's
  most-inward angular edge :math:`\mu_{\rm start}` (sphere: :math:`\mu =
  -1`).  It is not free and it is not a closure choice — it solves a
  transport problem of its own, the plain radial ODE the balance collapses
  to where the redistribution coefficient vanishes.  Since #282 route (a)
  production **marches** it directly from the within-group source
  :math:`\bar q_{1/2}` and carries it as first-class typed state; the full
  account, including why reading it off the iterate was a walk-order back
  edge, is :ref:`sn-direct-seed-solve`.

  .. note:: **⛔ The seed was** :math:`\phi_{1/2,i,g} = 0` **until Phase D,
     and that zero was a bug — not a convention.**

     Phase B hardcoded it, justified as "the unique choice consistent with
     :math:`\alpha_{1/2} = 0`".  The justification is wrong twice over: the
     product :math:`\alpha_{1/2}\,\phi_{1/2}` vanishes for *any*
     :math:`\phi_{1/2}` precisely because :math:`\alpha_{1/2} = 0`, so that
     argument constrains nothing — and the seed *also* enters the
     denominator-propagation chain, reaching every downstream half-face
     through the :math:`(1-\tau_n)` weight.  It is a wrong-term
     initialisation, ``vv-principles`` failure Mode 3, catalogued as part of
     ERR-026 and dissected at :ref:`sn-phase-d-carlson-coupled-pole-sweep`
     ("The bug Phase B baked in").

     ⚠ **One live path still seeds at zero, legitimately, and it is not this
     one.**  The ray-**decoupled** :math:`(A,A)` diagonal block substitutes a
     zero seed by construction — there is no fold entry on that block, so the
     M-M thread has nothing to read and its cotangent propagates nowhere
     (:ref:`sn-loss-rep-ray-decoupled-block`).  The kernel
     :func:`~orpheus.sn.angular.closure.compute_psi_half_per_level`
     therefore keeps ``psi_half_seed=None`` → zero as its API default, and
     that default — **not** the shipped scheme — is what the equation's
     ``verifies`` gates in
     :file:`tests/sn/sweep/curvilinear/test_compute_psi_half_per_level.py`
     exercise.  On the coupled production path the seed is the marched
     state.

  At :math:`\tau_n = 1/2` the recurrence
  reduces to pure DD angular :math:`\phi_{n+1/2,i,g} = 2\,\phi_{n,i,g}
  - \phi_{n-1/2,i,g}` — which is exactly what
  :cite:`Hebert2009` Eqs. 3.437/3.439 (sphere) and Eqs. 3.412/3.414
  (cylinder) write, and what :cite:`BaileyMorelChang2010` Eq. 53 names
  "the diamond scheme".  It is a **different method**: it is the
  pointwise truncation-order optimum (:cite:`ReedLathrop1970`
  Eqs. (15)/(16) — second order iff :math:`\tau = \tfrac12 + O(w)`), but
  BMC prove it is diffusion-limit consistent only to LEADING order,
  whereas the *weighted* diamond is the only member of the family
  correct through **first** order — and that first-order consistency is
  what removes the flux dip in general.  Any
  :math:`\tau \in [0, 1] \setminus \{\tfrac12\}` gives
  weighted-DD; the admissible range is :math:`[0, 1]` and both arms run
  the derived weight unclamped (see the geometry-split subsection above —
  :math:`\tfrac12` is *not* a lower bound and never was).  The same
  recurrence runs
  inside :class:`~orpheus.transport.spatial.diamond.DiamondDifference` (the
  sweep's cell update).

  ⛔ **REFUTED by Phase C (2026-05; this bullet said the opposite until
  2026-08-13).**  The text here read: *"applying this strategy in the apply
  matvec brings the apply and sweep to the same angular closure, but the*
  **spatial** *closures still differ (apply uses arithmetic averages + DD
  extrapolation; sweep uses WDD).  Full ERR-026 closure on the apply matvec
  requires aligning the spatial closure also."*  That was true **of Phase
  B**, and it is exactly what Phase C then shipped: the matvec was rewritten
  into the sweep frame, so ``apply`` and ``solve`` now consume the **same
  three primitives** — the WDD closure :eq:`phase-c-wdd-recurrence`, the
  direction-keyed DAG walk, and the BC trace law at the boundary edge — and
  agree on what :math:`L + C` *is* by construction
  (:ref:`sn-apply-sweep-equivalence`).  No arithmetic-average face closure
  survives anywhere in ``orpheus``.  The sentence is preserved because it
  records *why* the canonical angular form could not be made the default in
  Phase B alone: an angular fix paired with a mismatched spatial closure
  measurably made the operator **worse** on flat :math:`\psi`
  (:ref:`sn-superseded-arithmetic-spatial-closure`).
  :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
  was therefore **opt-in** in Phase B (the default was then
  ``LegacyTauSymmetricInterpolation``); it has since become the sole
  production strategy and the default (see the supersession note above).

.. _sn-alpha-normalization:

α-recursion normalisation
-------------------------

Hébert Eq. 3.424 reads :math:`\alpha^{H}_{n+1/2} = \alpha^{H}_{n-1/2}
- 2\,\mathcal{W}_n\,\mu_n` with the corresponding redistribution
divisor :math:`\Delta S_i / (2\,\mathcal{W}_n)` in Eq. 3.428.  The
ORPHEUS arrays carry :math:`\alpha^{O} = \alpha^{H}/2`, absorbing the
factor of 2 into the recurrence; the redistribution divisor reads
:math:`\Delta A_i / w_n` correspondingly.  Both forms are
mathematically equivalent.  This normalisation is documented in
:mod:`orpheus.sn.angular.redistribution` (which carried the
:math:`\alpha` mathematics out of the dissolved
``geometry/reduced_operator.py`` at P4.2) and re-stated explicitly in
:mod:`orpheus.sn.angular.closure` so the Hébert canonical
form's connection to the ORPHEUS arrays is transparent.  The full
cross-source picture — the same recursion spelled four ways across
three texts, plus the review literature's :math:`\beta` spelling — is
tabulated at :ref:`normalization-alpha-crosswalk`.

.. _sn-citation-corrections:

Citation corrections
--------------------

Two corrections, three years apart in the literature they touch.  Both
are the same disease — *a scheme wearing another paper's name* (the #327
family) — and the second one produced a wrong cylinder :math:`\tau`
before it was caught, so the record is kept in full.

Correction 1 — the wrong Bailey paper (Issue #168 Phase B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-Phase-B the codebase cited "Bailey, T. S., Adams, M. L., Yang, B., & Zika, M. R.
(2009). *A piecewise linear finite element discretization of the
diffusion equation for arbitrary polyhedral grids*. JCP 227, 3738-3757"
for the curvilinear S\ :sub:`N` :math:`\alpha`-recursion.  This is the
**wrong Bailey paper** — Bailey-Adams-Yang-Zika is a piecewise-linear FE
diffusion paper unrelated to S\ :sub:`N`.  The intended reference is
:cite:`BaileyMorelChang2010`, NSE 165(2):149-169, "The Asymptotic
Diffusion-Limit Accuracy of :math:`S_N` Angular Differencing Schemes"
(LLNL preprint LLNL-JRNL-420356; OA at
https://www.osti.gov/servlets/purl/1020346).  Phase B corrected the
citations in ``orpheus.geometry.reduced_operator`` (since dissolved; its
reference apparatus lives in :mod:`orpheus.sn.angular.redistribution`),
``orpheus.sn.loss_representation`` (the dissolved ``sweep.py``),
:mod:`orpheus.transport.spatial.diamond`, and the new
:mod:`orpheus.sn.angular.closure` module.

.. _sn-tau-source-of-record:

Correction 2 — Hébert is not the source of the weighted :math:`\tau`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`[M]` all three claims below verified against the rendered pages,
2026-08-11.  Sources are listed by WHAT they are the authority for,
because the weighted :math:`\tau`, the
:math:`\alpha`-recursion and the sweep mechanics come from three
different places and conflating them is the error class this page has
already paid for:

.. list-table:: Source of record, per object
   :header-rows: 1
   :widths: 26 34 40

   * - Object
     - Authority
     - What it says
   * - The weighted :math:`\tau` — **primary**
     - :cite:`MorelMontry1984`
     - *Analysis and Elimination of the Discrete-Ordinates Flux Dip*,
       TTSP 13(5):615-633.  The scheme this closure implements, and the
       reason the family is named "Morel--Montry".
   * - The weighted :math:`\tau` — **the form we implement**
     - :cite:`BaileyMorelChang2010` Eqs. (42)/(43)
     - :math:`\tau_m` is the barycentric coordinate of the ordinate
       between its own cell's two edges, in the radial direction cosine
       (:eq:`morel-montry-closure`).  Their Eq. (41) is the first-order
       diffusion-limit condition :math:`\beta = 0`, and forcing
       :math:`\beta = 0` is what **determines** these weights — so
       :math:`\tau` is derived, not chosen.
   * - The same condition, 40 years earlier
     - :cite:`ReedLathrop1970` Eq. (13c)
     - Identical to BMC Eq. (43).  Their Eqs. (15)/(16) add the sharpest
       accuracy criterion available on :math:`\tau` — see
       :ref:`sn-tau-pointwise-second-order` below.
   * - The :math:`\alpha` recursion
       :math:`\alpha_{m+1/2} = \alpha_{m-1/2} - \mu_m w_m`
     - Lathrop, K., & Carlson, B. (1966), *J. Comp. Phys.* 1:173
     - Cited by :cite:`ReedLathrop1970` (their ref. 7) as "a requirement
       commonly invoked to define the :math:`\alpha` coefficients".
       Hébert credits the cylindrical :math:`\eta_{p,q\pm1/2}`
       construction to Alcouffe, R. E., & O'Dell, R. D. (1986),
       *Transport Calculations for Nuclear Reactors*, CRC Handbook of
       Nuclear Reactors Calculations Vol. I (Y. Ronen, ed.).
       ⚠ **Neither is in the local library and neither has been read**;
       both are recorded as attributions, not as consulted sources.
   * - The cell balance, the sweep ordering, the Carlson starting
       direction
     - :cite:`Hebert2009` **§3.9.3** (cylinder, printed pp. 137-141) and
       **§3.9.4** (sphere, printed pp. 141-144)
     - Authority for the cell-balance layout, the :math:`\Delta A/w`
       factor, the sweep ordering and the :math:`\alpha_{1/2} = 0`
       initialisation — and **not** for :math:`\tau`.

The three specific claims that were false, each verified against the
rendered pages on 2026-08-11:

#. **§3.9.4 is the SPHERE.** Hébert's cylinder is **§3.9.3**; the whole
   Eq. 3.418-3.439 range is spherical.  A cylinder claim carrying a
   §3.9.4 citation is citing the wrong geometry.
#. **Eqs. 3.437/3.439 are not weighted.**  Eq. 3.439 reads
   :math:`\phi_{n+1/2,i} = 2\phi_{n,i} - \phi_{n-1/2,i}` — Eq. 3.431
   rearranged for the sweep, i.e. :math:`\tau \equiv \tfrac12` exactly.
   The cylinder's azimuthal counterparts, Eqs. 3.412/3.414, have the
   identical shape.  (Citing 3.437/3.439 for the :math:`\tau = \tfrac12`
   *reduction* is therefore correct; citing them for :math:`\tau` itself
   is not.)
#. **Hébert defines no** :math:`\tau` **anywhere in chapter 3**, in
   either geometry.  He ships the *plain* angular diamond.

.. warning:: **Never cite Hébert against BMC here, and never call BMC
   "auxiliary".**  Until 2026-08-11 the paragraph above read *"Hébert
   (2009) §3.9.4 is the primary source for the curvilinear S*\
   :sub:`N` *discretization in this codebase; Bailey-Morel-Chang 2010 is
   the auxiliary justification for the M-M weighted-diamond* :math:`\tau`
   *itself"* — and an earlier revision still said *"the M-M
   weighted-diamond* :math:`\tau` *clamp"*.  Both are retracted.

   :cite:`BaileyMorelChang2010` is not auxiliary: it is the form the code
   implements, and they prove (their Eq. 53 and §I) that the plain
   diamond Hébert ships is diffusion-limit consistent only to LEADING
   order, while the *weighted* diamond is the only member of the family
   correct through **first** order — and first-order consistency is what
   removes the flux dip in general.  Their Eq. 53 explicitly names
   :math:`\tau = \tfrac12` "the diamond scheme".  BMC also prescribe
   **no clamp**: the admissible range they state is :math:`[0, 1]`, and
   their own :math:`S_2` example gives
   :math:`\tau_1 = 1 - 1/\sqrt3 \approx 0.4226 < \tfrac12` (their
   Eq. 47), which the retired :math:`[\tfrac12, 1]` absorber would have
   clipped.  The absorber's own provenance is at
   :ref:`sn-tau-absorber-retirement`.

.. _bmc-equation-map:

Correction 3 — a fictitious record, and the BMC equation-number map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **Retracted 2026-08-27.**  The quadrature package cited a *second*
"Bailey 2009" record, distinct from the one Correction 1 retracted:

   *"Bailey, T. S., Adams, M. L., Yang, B., Zika, M. R. (2009).  'A
   piecewise linear discontinuous finite element spatial discretization
   of the transport equation.'  Annals of Nuclear Energy 35,
   1929-1936."*

**That publication does not exist.**  Every field of it traces to a
different real one, which is exactly why it read as credible and
survived two prior citation audits:

.. list-table:: Where each field of the fictitious record came from
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Actual origin
   * - the title
     - Bailey, Adams & **Chang** (2008), *"…in 2D Cylindrical
       Geometry"*, LLNL-CONF-407632 (a conference paper; OSTI 952424)
   * - the author list (…Yang, Zika)
     - the record Correction 1 already retracted — JCP 227:3738-3757
   * - *Ann. Nucl. Energy* **35**, 1929-1936
     - Zio & Zoia (2008), *"Bayesian inference of BWR model parameters
       by Markov chain Monte Carlo"*, doi
       ``10.1016/j.anucene.2008.03.007``
   * - the year 2009
     - nothing — all three sources above are **2008**

`[M]` CrossRef over the whole run of *Annals of Nuclear Energy*
(``journals/0306-4549/works?query.author=Bailey``) returns
``total-results = 4``, none a finite-element paper and none in vol. 35.
Volume 35 is 2008, so the (author, year, volume) triple was
self-refuting before any lookup — a cheap check worth running on any
citation that carries all three.

⭐ **But the equation numbers were RIGHT.**  They belong to
:cite:`BaileyMorelChang2010` — already in ``docs/refs.bib``, already
the authority Correction 1 re-pointed to — so this was a re-point, not
a deletion.  All four verified on the rendered scan:

.. list-table:: The BMC 2010 equation-number map
   :header-rows: 1
   :widths: 14 26 60

   * - Equation
     - Object
     - What it says, and the ORPHEUS reading
   * - **Eq. (11)**
     - the :math:`\alpha` dome recursion, **sphere**
     - :math:`\alpha_{m+1/2} = \alpha_{m-1/2} - 2\mu_m w_m`.  The
       factor of 2 is theirs, because the sphere normalises
       :math:`\sum w = 2`.
   * - **Eq. (50)**
     - the :math:`\alpha` dome recursion, **R-Z** (printed p. 156)
     - the same recursion **without** the factor of 2, because R-Z
       normalises :math:`\sum\sum w = 4\pi`.  Seeded at, and closing
       back to, zero.  This is :eq:`alpha-recursion` /
       :eq:`alpha-cylindrical` in ORPHEUS letters.
   * - **Eq. (52)**
     - the per-level **edge-cosine accumulation** (printed p. 157)
     - the level's edge cosines are built by accumulating the level's
       weights from :math:`-\sin\theta` to :math:`+\sin\theta` — so the
       ordinates of a level run **ascending in the radial cosine**.
       ⚠ Two separable halves: see the scoping note below.
   * - **Eq. (74)**
     - the Morel--Montry :math:`\tau` (printed p. 160)
     - the same weight Correction 2 records at Eqs. (42)/(43); Eq. (74)
       is its R-Z statement.

.. warning:: **Eq. (50)'s printed right-hand side is self-referential —
   a published journal typo**, confirmed on the rendered scan and
   corrected here against the correctly-printed spherical twin
   Eq. (11).  A reader checking the paper will otherwise conclude that
   ORPHEUS has transcribed it wrongly.  The recursion ORPHEUS
   implements is the one Eq. (11) prints, less the factor of 2 that
   the R-Z weight normalisation removes.

.. important:: **Scoping Eq. (52): ORPHEUS shares its ORDERING and
   deliberately does NOT use its PARTITION.**  Eq. (52) states two
   things at once, and only the first transfers.

   * The **ordering** — a level's ordinates ascend in the radial
     cosine, from :math:`-\sin\theta` to :math:`+\sin\theta`.  ORPHEUS
     shares this; it is what the quadrature layer's per-level index
     lists are sorted by, and it is the only component the
     :math:`\alpha` march's arithmetic reads.
   * The **partition** — that each cell's radial-cosine measure equals
     that ordinate's weight.  ORPHEUS **refutes** this on its own rule:
     it is a property of *their* quadrature, not a law, and imposing it
     here violates the cell-membership predicate and diverges.  The
     measurement and the widening-mismatch table are at
     :ref:`sn-tau-absorber-provenance`, and the replacement partition
     (midpoint in :math:`\omega`) is defined at
     :ref:`angular-cell-partition-section` — both on
     :doc:`/theory/foundations/structured_geometry`.

   ⟹ cite Eq. (52) for the **order of a level**, never for how the
   level is **cut**.

.. _sn-tau-pointwise-second-order:

The pointwise accuracy criterion on :math:`\tau` (Reed & Lathrop)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cite:`ReedLathrop1970` reach BMC Eq. (43) as their Eq. (13c) forty
years earlier, and go one step further.  They additionally impose their
Eq. (13b), which turns the pair into a quadratic for the **ordinate**
(edges in, ordinates out) — a different branch that fixes the
*quadrature*, and not the one this codebase takes.  What transfers
directly is their **Eqs. (15)/(16)**: the angular truncation error is
second order **iff the ordinate is the** :math:`\mu`-**midpoint of its
own angular cell to** :math:`O(w^2)`, i.e. iff
:math:`\tau = \tfrac12 + O(w)`.

Two properties make this the sharpest criterion available on
:math:`\tau`:

* It is **pointwise**, not integrated.  Unlike BMC's :math:`\beta`
  (a diffusion-limit functional obtained by summing over the level), it
  is a statement about one ordinate in one cell — so a
  :math:`\sigma_y`-folded cylinder level, whose symmetry annihilates
  :math:`\beta`, does **not** annihilate it.  (That annihilation is the
  Mode-12 hazard recorded in
  :ref:`the β-blindness warning <sn-tau-beta-diagnostic-blind>`.)
* It explains why *both* shipped arms are legitimate without either
  being "the accurate one": the sphere's Gauss--Legendre
  :math:`\tau \in [0.39, 0.61]` and the folded cylinder's
  :math:`\tau \in [\tfrac14, \tfrac34]` are both
  :math:`\tfrac12 + O(w)`, while the retired :math:`[\tfrac12, 1]`
  absorber's one-sided box is not centred on :math:`\tfrac12` at all.

ERR-026 closure status after the pole closure
---------------------------------------------

.. note:: **Read this subsection as the PHASE-B-era status (2026-05).**
   Every present-tense claim below was true then and has since been
   discharged: Phase C aligned the apply matvec's spatial closure with the
   sweep's WDD form (:ref:`sn-apply-sweep-equivalence`),
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
   duly became the default and then the **sole** production closure, the
   four ``xfail-strict`` curvilinear MMS tripwires were retired, and
   ERR-058 (Issue #195, CLOSED 2026-06-12) showed the residual wrong fixed
   point was the *closure-seed* family rather than a boundary-truncation
   order.  It is preserved because the reason Phase B **declined** to flip
   the default — an angular fix paired with a mismatched spatial closure is
   worse than neither — is the load-bearing lesson, and it is the same
   pairing argument the seed correction later needed.

Phase B ships the architectural infrastructure for closing Defect 3
(the :class:`~orpheus.sn.angular.closure.AngularClosureBase`
Protocol + canonical
:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`)
without flipping the default — the canonical form in isolation does
not produce an :math:`\mathcal{O}(h^2)` MMS rate because the apply
matvec's spatial closure still differs from the sweep's WDD form.
The four ``xfail-strict`` curvilinear MMS tripwires therefore stay
xfail through Phase B; ERR-026 stays at **PARTIAL CLOSURE** (Phase A
closed Defects 1+2 spatial, Phase B ships Defect 3 architectural
scaffolding).  The full closure requires a Phase C follow-up that
aligns the apply matvec's spatial closure with the sweep's WDD
form, at which point
:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
becomes the natural default.

.. _sn-pole-closure-compute-psi-half:

Half-angle grid exposure
------------------------

.. todo:: Archivist expansion needed.

   The Issue #197 PR-TYPED-6b dispatch added the public surface
   :func:`~orpheus.sn.angular.closure.compute_psi_half_per_level`
   exposing the M-M recurrence's half-angle grid
   :math:`\phi_{m\pm 1/2,i,g}` for one level.  (Originally an instance
   method on ``MorelMontryAngularSweep``, served by the unbound
   ``sn_mesh=None`` legacy mode; the C5 retirement of that mode,
   2026-07-03, moved it to module level — the surface takes all data
   via arguments and the seed strategy as a keyword, so it never
   needed an instance.)  It is the intermediate exposure that lets the
   unified SN matvec
   (:class:`~orpheus.sn.operators.streaming.StreamingOperator`) consume
   :math:`\phi_{m\pm 1/2}` as
   :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`'s
   angular-upstream argument — closing the apply-vs-sweep
   twin path on the curvilinear angular branch.  (That parameter was
   named ``psi_angular_upstream`` until PR-TYPED-6.5 Phase 2.11 made
   the helper closure-blind; it is ``angular_numer_upstream`` today and
   carries :math:`(\Delta A/w)\,c_{\rm in}\,\phi_{m-1/2}` already
   multiplied out, not :math:`\phi_{m-1/2}` itself.)

   Pattern 2 (Single source of truth).  The :eq:`pole-mm-recurrence`
   recurrence body lives once, in the module-level kernel
   ``_psi_half_grid_single_level`` of
   :mod:`orpheus.sn.angular.closure`.  Both the
   public ``compute_psi_half_per_level`` AND the live production path
   route through it: the matvec/sweep call
   :meth:`~orpheus.sn.angular.closure.AngularClosureBase.precompute_psi_state`,
   which dispatches per level through ``_psi_half_grid_for_level``,
   which calls the same kernel.  (Phase B / Phase C drove the
   redistribution through a single ``__call__`` bundle that also routed
   through this kernel; Issue #248 deleted the bundle, so the
   single-source-of-truth invariant now binds
   ``compute_psi_half_per_level`` and ``precompute_psi_state``
   directly.)

   Test gate:
   :file:`tests/sn/sweep/curvilinear/test_compute_psi_half_per_level.py`
   — foundation + L0 tests pinning function existence, shape contract,
   the verbatim Hébert recurrence formula
   :math:`\phi_{m+1/2} = (\phi_m - (1-\tau_m)\phi_{m-1/2})/\tau_m`,
   Carlson-context seed contract, the Pattern-2 round-trip
   (``compute_psi_half_per_level`` against the
   ``_psi_half_grid_single_level`` kernel), and linearity. After
   Issue #248 these gates drive the recurrence through the **live**
   surface the matvec consumes, rather than the retired ``__call__``
   bundle.

   Closeout memo:
   ``.claude/agent-memory/method-implementer/issue_197_pr_typed_6b_closeout.md``.

.. _sn-sweep-frame-apply-matvec:

Sweep-frame apply matvec
========================

.. admonition:: Key Facts
   :class: important

   * Phase C (commits ``eae6f05`` → ``d445a8f``, 2026-05-12) rewrote
     the then-production ``transport_operator_matvec_spherical``
     and ``_cylindrical`` matvecs (the whole per-geometry family
     since deleted — #197 / #280 campaigns) as **one sweep iteration
     semantically**.
     The WDD diamond closure
     :math:`\psi^{\text{face}}_{\text{out}} = 2\,\psi^{\text{cell}}
     - \psi^{\text{face}}_{\text{in}}` propagates the face flux
     cell-by-cell along the direction's DAG; the BC trace law owns
     the boundary edge per :ref:`affine-bc-form`.
   * The pole-face initial condition is
     :math:`\psi^{\text{face}}_{\text{in}}(\text{pole}) =
     \psi^{\text{cell}}[0]` (Lewis–Miller §4.5 Carlson seed), **not**
     :math:`0`. The Carlson seed is the unique anchor that preserves
     the per-ordinate flat-flux invariant under the WDD recurrence.
   * Phase A's
     ``BoundaryFaceFlux``
     Protocol (415 LOC + 21 foundation tests) **retires entirely**
     — the boundary-face closure is now inside the WDD propagation
     chain, owned by the BC trace law at the boundary edge.
   * Empirical Gate 1.1 (per-ordinate flat-flux residual on
     reflective curvilinear) finds **spherical** MMS with
     ``MorelMontryAngularSweep`` FAILS, **cylindrical** MMS PASSES.
     The default flip to ``MorelMontryAngularSweep`` is therefore
     DEFERRED to Phase D (`Issue #192
     <https://github.com/deOliveira-R/ORPHEUS/issues/192>`_); the
     four ERR-026 ``xfail-strict`` curvilinear MMS tripwires stay
     xfail; ERR-026 remains at PARTIAL CLOSURE.
   * The structural-frame name for the rewrite is the **sweep /
     wavefront frame** (cross-domain-attacker 2026-05-12 analysis):
     the "ghost cell" idiom is realised as a typed
     :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` vector
     defined by the realised BC operator, not extrapolated from
     interior cell centres.

Phase C resumes the spatial-closure alignment that Phase B
identified as the load-bearing precondition for the
curvilinear-default flip (see
:ref:`sn-pole-angular-closure-protocol` for Phase B's full
:class:`~orpheus.sn.angular.closure.AngularClosureBase`
closure-contract narrative; this subsection picks up at the point
Phase B left for follow-up). Three unblockers landed before Phase C
resumed:

1. The trajectory_resolvent (Peierls Variant α Green's-function)
   campaign shipped cylinder MR Phase 1b (commits ``37e3e29``,
   ``cf662a6``, ``604f380``, ``e10c33c``), explicitly built to close
   the cylinder-2G ERR-026 gap. trajectory_resolvent now covers
   5 of the 6 deleted curvilinear regression snapshots at
   machine-precision-class precision; the 6th (P1 anisotropic)
   routes to a shape-independent :math:`k_\infty` closed form.
2. The cross-domain-attacker analysis on 2026-05-12 identified a
   **structural inconsistency** between the pre-Phase-C apply
   matvec's boundary closure and the :ref:`affine-bc-form` (§16A.3).
   The apply matvec was passing **cell-centre** values to
   ``bc_outer.apply``, whereas §16A.3 requires the boundary
   face TRACE :math:`\gamma_+ \psi`. The cross-domain-attacker's
   "sweep / wavefront frame" naming gave the rewrite its
   architectural shape: the apply matvec is one sweep iteration
   over the cell-visit DAG, with the BC trace law at the boundary
   edge owning the inflow trace.
3. The Phase A
   ``BoundaryFaceFlux``
   Protocol — built to second-order-accuratise the curvilinear
   outer face — was re-classified as **a patch on top of the wrong
   architecture**. Phase A's
   ``DDExtrapolation``
   produces a face-flux extrapolant
   :math:`\psi^{\text{face}}_{N-1/2} = \tfrac{3}{2}\,\psi_{N-1} -
   \tfrac{1}{2}\,\psi_{N-2}` that ignores the BC entirely; Phase B's
   :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
   produces angular closure that is inconsistent with the apply
   matvec's spatial closure (arithmetic interior average + DD
   extrapolation outer face). The fix at all three sites is the
   **same** — make every face closure consume the WDD-propagated
   face value, and let the BC trace law own the boundary edge.
   "Two paths to the same discrete operator over different storage
   conventions" (cross-domain-attacker Smell 16) is the trigger for
   the unification.

.. _sn-superseded-arithmetic-spatial-closure:

The superseded arithmetic spatial closure
-----------------------------------------

Pre-Phase-C, the matvec interior face values were computed as the
arithmetic average of cell-centre values
:math:`\psi^{\text{face}}_{i+1/2} = \tfrac{1}{2}(\psi^{\text{cell}}_i
+ \psi^{\text{cell}}_{i+1})`. This is **second-order accurate** on
smooth fields when the cell-centre values are themselves the
analytical values, but it does **NOT** match the sweep's WDD
recurrence which evaluates
:math:`\psi^{\text{face}}_{i+1/2} = 2\,\psi^{\text{cell}}_i -
\psi^{\text{face}}_{i-1/2}`. The two are equivalent only when
:math:`\psi` is constant on a cell; for any angular or spatial
variation the two values diverge by an :math:`\mathcal{O}(\Delta r)`
amount, and the operators are **not the same operator**.

This is the empirical content of Phase B's diagnosis (see
:ref:`sn-pole-angular-closure-protocol` "ERR-026 closure status"):
pairing the canonical
:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
angular closure with arithmetic-average spatial closure produces a
**worse** operator on flat :math:`\psi` than the legacy
:math:`\tau`-symmetric interpolation. Under the **Phase-B zero seed** the
canonical M–M form produces half-angle face fluxes oscillating as
:math:`0, 2c, 0, 2c, \ldots` on flat :math:`\psi` — read
:eq:`pole-mm-recurrence` at :math:`\tau = \tfrac12` from
:math:`\phi_{1/2} = 0` and the alternation is immediate — and the
arithmetic spatial
average then combines these oscillating angular face fluxes with
interior-averaged spatial face fluxes into garbage. Phase B's
empirical test ``test_apply_spherical_constant_flux_under_morel_montry_canonical_form``
saw :math:`\phi` range across [0.6, 1.004] on a flat-:math:`\psi`
input that analytical balance demands give exactly :math:`1.0`.

.. note:: **Read this diagnosis with its seed attached.**  The alternation
   above is a property of the *zero* seed, not of the M-M closure: a seed
   consistent with the field it marches gives :math:`\phi_{m+1/2} = c` at
   every face on flat :math:`\psi` (the recurrence's flat fixed point), and
   no alternation to combine with anything.  Phase D measured the two
   candidate injection points separately and found the **angular** seed was
   the binding one — replacing it closed the flat-:math:`\psi` residual to
   :math:`1.78\times10^{-15}` while replacing the WDD spatial seed changed
   nothing (the ``[A]``/``[B]``/``[C]``/``[D]`` sweep at
   :ref:`sn-phase-d-carlson-coupled-pole-sweep`).  Both closures were
   nevertheless wrong, and both were fixed: Phase C aligned the spatial
   closure, Phase D and then #282 route (a) fixed the seed.

The fix is to align the spatial closure with the sweep: use the
WDD recurrence
:math:`\psi^{\text{face}}_{\text{out}} = 2\,\psi^{\text{cell}}
- \psi^{\text{face}}_{\text{in}}` per cell, propagating face flux
in DAG order. Phase C ships this alignment.

The sweep-frame matvec algebra
--------------------------------

The rewritten matvec is **one sweep iteration semantically**.
For each bulk direction :math:`d \in \{+1, -1\}`, the per-cell
WDD recurrence walks the face flux along the DAG:

.. math::
   :label: phase-c-wdd-recurrence

   \psi^{\text{face}}_{\text{out}}(i) \;=\; 2\,\psi^{\text{cell}}(i)
   \;-\; \psi^{\text{face}}_{\text{in}}(i),
   \qquad
   \psi^{\text{face}}_{\text{in}}(i+1) \;=\;
   \psi^{\text{face}}_{\text{out}}(i),

.. (vv-status rationale) Representational identity: the per-cell WDD diamond
   recurrence the apply matvec walks in DAG order — the apply-direction spelling
   of the loss operator L+C.  Its verifiable content is the apply↔sweep
   structural equivalence and the apply linearity/reciprocity/determinism
   foundation gates (``tests/sn/sweep/core/test_phase_c_gates.py`` Gates
   1.2/1.3/1.4, ``@pytest.mark.foundation``, unwired — the label stays
   ``documented`` with the gates named here).
.. vv-status: phase-c-wdd-recurrence documented

evaluated cell-by-cell across the direction's DAG order yielded by
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk` invoked with
``direction_sign``. The
per-cell streaming term consumes both the inflow and outflow face
values along with the cell volume and face areas (Hébert §3.9.4
balance, ORPHEUS-normalised):

.. math::
   :label: phase-c-streaming-spherical

   S_{n,i,g} \;=\; \frac{\mu_n}{V_i}\,
   \bigl[ A_{i+1/2}\,\psi^{\text{face}}_{n,i+1/2,g}
        - A_{i-1/2}\,\psi^{\text{face}}_{n,i-1/2,g} \bigr],

.. (vv-status rationale) Literature-transcribed definition: the conservative
   radial streaming term (Hébert §3.9.4 balance, ORPHEUS-normalised), a
   constituent of the apply-matvec cell update :eq:`phase-c-cell-update`.  Not
   a standalone claim; its correctness is exercised through the per-ordinate
   flat-flux / streaming-equilibrium gate and the apply↔sweep equivalence
   foundation gates (``tests/sn/sweep/core/test_phase_c_gates.py``).
.. vv-status: phase-c-streaming-spherical documented

with the redistribution term provided by the Phase B
:class:`~orpheus.sn.angular.closure.AngularClosureBase`
strategy and the collision term :math:`\Sigma_t \psi^{\text{cell}}_{n,i,g}`
unchanged. The full per-cell update is

.. math::
   :label: phase-c-cell-update

   (T\psi)_{n,i,g} \;=\; S_{n,i,g} \;+\; R_{n,i,g} \;+\;
   \Sigma_t(i,g)\,\psi^{\text{cell}}_{n,i,g},

.. (vv-status rationale) Representational identity: the full per-cell
   apply-matvec output (streaming + redistribution + collision) — the
   definition of the (L+C)ψ action, not a separate solver claim.  Its
   constituents are foundation-gated by the apply linearity / reciprocity /
   determinism gates and the apply↔sweep structural equivalence
   (``tests/sn/sweep/core/test_phase_c_gates.py``).
.. vv-status: phase-c-cell-update documented

where :math:`R_{n,i,g}` is the strategy's redistribution output
(Bailey :math:`\Delta A_i / w_n` redistribution factor with the
strategy-specific :math:`\alpha_{n+1/2}\psi_{n+1/2} -
\alpha_{n-1/2}\psi_{n-1/2}` evaluation). In current production this
is produced by
:meth:`~orpheus.sn.angular.closure.AngularClosureBase.cell_contribution`,
consuming the per-level state that
:meth:`~orpheus.sn.angular.closure.AngularClosureBase.precompute_psi_state`
stamps once per sweep. (Phase B / Phase C shipped this redistribution
through a single ``__call__`` bundle on the strategy; that legacy
bundle — and its ``tau_mm`` argument — was retired in Issue #248, and
the two strategy methods are now the sole production surface.) See
:eq:`alpha-dome-recursion` for the :math:`\alpha` recurrence and
:eq:`pole-mm-recurrence` for the M–M angular DD form.

Ordinate vectorisation
-----------------------

Per the user's hard architectural directive (and the precedent at
``orpheus/sn/angular_operator.py:183``), the rewritten matvec
carries **no** ``for n in range(quad.N)`` loop. The per-cell update
operates on whole ordinate subsets via boolean masks:

.. code-block:: python

   eps = 1e-15
   outgoing_mask = quad.mu_x > +eps   # μ > 0 ordinates
   incoming_mask = quad.mu_x < -eps   # μ < 0 ordinates
   mu_out = quad.mu_x[outgoing_mask]
   mu_in  = quad.mu_x[incoming_mask]

A single per-cell statement updates the full outward ordinate
subset:

.. code-block:: python

   psi_cell = fi[:, outgoing_mask, i, 0]               # (ng, n_out)
   psi_face_out = 2.0 * psi_cell - psi_face_in         # WDD diamond
   streaming = (
       mu_out[None, :]
       * (A[i + 1] * psi_face_out - A[i] * psi_face_in)
       / V[i]
   )

This is the canonical "vectorise across ordinates" pattern that
the cross-method ordinate-anti-pattern audit
(`Issue #191 <https://github.com/deOliveira-R/ORPHEUS/issues/191>`_)
tracks systemically. Phase C's contribution is to introduce no new
per-ordinate loop inside any new code; the 14 existing sites stay
untouched as separate work.

The cylindrical case adds an outer loop over the :math:`\mu`-levels
(the per-level azimuthal-DAG topology is intrinsic to cylindrical's
structure), but each level still operates on whole within-level
ordinate subsets via the same masking pattern. A third pass handles
pure-azimuthal degenerate ordinates (:math:`|\eta_n| < 10^{-15}`)
whose streaming term is zero by construction but whose
redistribution + collision contributions still must be scattered to
the equation map.

The new APIs
------------

Two new APIs surface what the existing infrastructure already knew:

* :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk`
  (``dag_walk(*, ordinate_idx=..., direction_sign=..., mu_level_idx=None)``)
  — Issue #196 Phase G
  Step 2.6 (Q3) canonicalised this as the **single iteration
  primitive** for 1-D sweeps, replacing the legacy pair of
  ordinate-keyed / direction-keyed methods.  Exactly one of
  ``ordinate_idx`` or ``direction_sign`` is supplied (XOR):
  ``ordinate_idx=n`` for the sweep driver's per-ordinate march,
  ``direction_sign=±1`` for the apply matvec's whole-subset walk
  keyed by the **bulk sweep direction sign**. The existing
  cell-visit graph's per-quadrant ``_diag_cache`` is already keyed
  by :math:`(\mathrm{sign}(\mu_x), \mathrm{sign}(\mu_y))`; the
  direction-keyed branch surfaces that sign-only view as a
  first-class API. A foundation test
  (:file:`tests/sn/sweep/core/test_dag_walk.py`) pins
  bit-identity between the two invocation modes across sphere /
  slab / cylindrical for every representative ordinate. For
  cylindrical the per-level
  ``mu_level_idx`` is required (the within-level DAG topology
  differs per level).

* ``EquationMap.unknowns_at_cell_for_mask(cell_idx, ordinate_mask)``
  — a precomputed inverse lookup ``(cell, ordinate) → k``. Lazily
  builds an ``(nx, N) int`` table with :math:`-1` sentinels for
  absent ``(ordinate, cell)`` slots; subsequent calls are O(1) per
  ``(cell, mask)`` pair. Replaces the per-equation O(n_eq) linear
  scan the legacy scatter pattern used. The eq_map still iterates
  ``(spatial outer, ordinate inner)`` at construction time; the
  helper just precomputes the inverse for the sweep-frame matvec.
  (The whole packed-vector ``EquationMap`` codec was subsequently
  retired at D-J — 2026-05-30 — when the typed :class:`FullField`
  composite replaced the packed-vector convention; this bullet
  records the Phase G design as it stood.)

What retires
------------

Phase A's
``BoundaryFaceFlux``
Protocol — five symbols, the
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` field, and the 21 foundation
tests — retires entirely. The architectural reasoning is "two paths
to the same operator → unify after the second instance" (per the
:doc:`/development` agent memory ``Unify after two instances``
directive). Phase A was the first instance of a face-closure
strategy; Phase B was the second (angular closure). With Phase C
the **third** instance (the BC at the boundary edge) the unification
has cleaner shape: every face value comes from the WDD recurrence
or the BC trace law applied at the boundary edge, never from an
algebraic extrapolation of cell centres. The retired symbols are:

* ``orpheus.sn.spatial.boundary_face_flux.BoundaryFaceFlux`` (Protocol)
* ``orpheus.sn.spatial.boundary_face_flux.BoundaryFaceFluxBase`` (ABC)
* ``orpheus.sn.spatial.boundary_face_flux.DDExtrapolation`` (default strategy)
* ``orpheus.sn.spatial.boundary_face_flux.CellCenter`` (ablation strategy)
* The ``boundary_face_flux`` constructor field +
  :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` attribute
* The ``boundary_face_flux_closure`` keyword argument from
  ``transport_operator_matvec_spherical`` and ``_cylindrical`` (the
  matvec family since deleted — #197 / #280 campaigns)
* :file:`tests/sn/sweep/test_boundary_face_flux.py` (232 LOC,
  21 foundation tests)

Three additional simplifications shipped with the rewrite:

* the then-``solution_to_angular_flux_spherical`` codec
  (and its cylindrical alias) returned a single ``fi`` array
  ``(ng, N, nx, 1)`` instead of the Phase A
  ``(fi, boundary_face_flux)`` tuple. Inward-at-boundary cell-centre
  slots ``fi[:, n_inward, -1, 0]`` are filled with the
  **reflected-partner cell-centre value** as an analytical
  extension: the equation map excludes these from unknowns (the BC
  determines them), but the WDD recurrence on flat :math:`\psi`
  requires the cell-centre to be consistent so the per-ordinate
  flat-flux invariant holds.
* :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` no longer accepts the
  ``boundary_face_flux=`` keyword (a regression test pins the
  field retirement).
* :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply` dispatch
  drops the ``boundary_face_flux_closure`` plumbing.

What stays
----------

* The Phase B
  :class:`~orpheus.sn.angular.closure.AngularClosureBase`
  closure contract (the ABC, since Issue #248) stays. The sphere
  centre / cylinder axis is **intrinsic geometry** (a coordinate-system
  singularity, not an external BC), so a *strategy* keyed on the
  coordinate system — rather than a boundary law — is the right shape.
  (Phase C wrote "the **three**-strategy angular closure … only the
  default is under question, and that is the Phase D decision point".
  ⛔ Both clauses are spent: Phase D flipped the default, and the two
  ablation strategies were retired at PR-TYPED-6c Step 7 / #248, leaving
  :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`
  for curvilinear and
  :class:`~orpheus.sn.angular.closure.IdentityAngularClosure`
  for Cartesian — two members because there are two geometries, not
  three because there were three candidate schemes.  The shape claim
  survives; the census does not.)
* The :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply_transpose`
  machinery via dense-probe construction stays. Linearity of the
  rewritten :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply`
  (Gate 1.4, pinned to ``rtol=1e-13``) guarantees the transpose is
  correctly tracked.
* The
  :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` +
  :class:`~orpheus.sn.mesh.method_space.SNMethodSpace` +
  :class:`~orpheus.numerics.operator.LinearOperator`-1-arg
  ``apply`` substrate (Issues #186 + #176 + #188, Waves 0–12) — the
  BC trace law's realised 1-arg ``apply(outflow) → inflow``
  contract is exactly what the matvec consumes at the boundary
  edge.

The pole-face initial condition — Carlson seed, then mirror continuation
--------------------------------------------------------------------------

.. attention:: **⛔ The Carlson cell-centre seed derived in this subsection
   is RETIRED** (⛔ REFUTED 2026-06-12 — ERR-058 manifestation (a),
   Issue #195).

   Everything from "The single largest architectural deviation" down to the
   cylindrical analogue is the **Phase C** reasoning, preserved because it is
   *why* anyone reached for a cell-centre pole seed, and because its
   flat-:math:`\psi` algebra is precisely the blindness that let the wrong
   seed survive every flat-flux gate in the tree.  The **shipped** pole-face
   initial condition is the :math:`r = 0` characteristic continuation
   :math:`\psi(0, +\mu) = \psi(0, -\mu)`
   (:eq:`sn-err-058-coupled-pole-continuity`), stated as a contract under
   :ref:`sn-pole-face-mirror-continuation` below.  Read this subsection for
   the history; read that one for the current contract.

   ⚠ **Two different objects share the name "Carlson" on this page, and only
   one of them was retired.**  The *spatial* pole-face seed below — an
   initial value for the outward **radial** WDD march — is retired.  The
   *angular* starting-direction solve — the Hébert §3.9.4
   :eq:`hebert-3-434`–:eq:`hebert-3-435` inward march that produces
   :math:`\psi_{1/2}` and seeds the M-M **angular** recurrence
   :eq:`pole-mm-recurrence` — is production, and is the subject of
   :ref:`sn-direct-seed-solve`.  Phase D's own diagnostic separated them:
   *the seed lives in the M-M angular recurrence, not in the WDD spatial
   pole-face IC* (the ``[A]``/``[B]`` intervention sweep,
   :ref:`sn-phase-d-carlson-coupled-pole-sweep`).

The single largest architectural deviation between the Phase C plan
and the shipped code is the **pole-face initial condition** for the
outward WDD sweep. The plan's pseudocode wrote

.. code-block:: python

   psi_face_in = np.zeros((ng, n_out))   # plan's pseudocode

with the comment "Pole face: :math:`\psi^{\text{face}} = 0` by
symmetry (also multiplied by :math:`A_0 = 0`)". The first claim
(symmetry) is wrong; the second (annihilation by zero face area) is
correct but does not help propagation. Empirically, this initial
condition combined with the WDD recurrence on flat :math:`\psi`
produces oscillating face fluxes
:math:`0, 2c, 0, 2c, \ldots` that break the per-ordinate flat-flux
invariant on **all three** Phase B pole-closure strategies:

.. math::
   :label: phase-c-wdd-oscillation

   \psi^{\text{face}}_0 = 0, \quad
   \psi^{\text{cell}}_0 = c \;\Longrightarrow\;
   \psi^{\text{face}}_1 = 2c - 0 = 2c,

   \psi^{\text{cell}}_1 = c \;\Longrightarrow\;
   \psi^{\text{face}}_2 = 2c - 2c = 0,

   \psi^{\text{cell}}_2 = c \;\Longrightarrow\;
   \psi^{\text{face}}_3 = 2c - 0 = 2c, \ldots

.. (vv-status rationale) Derivation / negative result: the oscillating face
   sequence 0, 2c, 0, 2c the WRONG (symmetry-zero) pole seed produces — a
   documented failure mode, not a shipped code path.  There is no code to test;
   the label preserves WHY the naive seed breaks per-ordinate flat flux so a
   future session does not reintroduce it.  The correct Carlson seed is
   exercised by the curvilinear streaming-equilibrium and #282 direct-seed
   gates.
.. vv-status: phase-c-wdd-oscillation documented

Phase C's answer was the **Carlson starting-direction seed** of
:cite:`LewisMiller1984` §4.5 (paraphrased in Hébert §3.9.4
Eqs. 3.432–3.435 for the angular analogue):
:math:`\psi^{\text{face}}_{\text{in}}(\text{pole}) =
\psi^{\text{cell}}(\text{first cell})`.

⛔ **REFUTED 2026-06-12 (ERR-058 manifestation (a), Issue #195); retired
from production.**  Reading the innermost *cell centre* as if it were the
*face* value is a half-cell offset — the cell centre sits at
:math:`r = \Delta r/2`, not at :math:`r = 0` — so the seed is
:math:`\mathcal{O}(h)`-wrong on every non-flat radial profile, and the DD
face chain then propagates that seed error outward as an undamped odd–even
alternation which the area weighting amplifies as :math:`A/V \sim 1/r` near
the pole (:ref:`sn-err-058-manifestation-a`).  The flat-:math:`\psi` algebra
below is *exactly* why that survived: on flat :math:`\psi` the offset is
invisible, so every flat-flux gate in the tree stayed green.  What shipped is
:ref:`sn-pole-face-mirror-continuation`.

For true flat :math:`\psi` the Phase C seed yields

.. math::

   \psi^{\text{face}}_0 = c, \quad
   \psi^{\text{face}}_1 = 2c - c = c, \quad
   \psi^{\text{face}}_2 = 2c - c = c, \quad \ldots

at every cell, so the streaming term
:math:`\mu_n (A_{i+1}\psi^{\text{face}}_{i+1} -
A_i\psi^{\text{face}}_i)/V_i` cancels the redistribution term per
ordinate on flat :math:`\psi` and the per-ordinate flat-flux
invariant holds. The pole-face streaming contribution is still
multiplied by :math:`A_0 = 0` (the pole face has zero area in both
spherical and cylindrical 1-D), so the Carlson seed introduces no
spurious source there; it only **anchors the recurrence** for the
cell-by-cell WDD propagation across the interior.

Why Lewis–Miller §4.5 is the canonical reference: at the spherical
centre :math:`r=0` and the cylindrical axis the angular dependence
of :math:`\psi` becomes **structurally singular** in the
transport-theory sense (:cite:`Pomraning1989` p. 339; see also
:ref:`sn-phase-d-pomraning-structural-singularity` and earlier
treatments in :cite:`LewisMiller1984` §4.5) — the angular flux is not a
separable function of
:math:`(\mu, r)` in any neighbourhood of the singular point because
the inward and outward ordinate cones meet there. Lewis–Miller's
"starting direction" handles this by introducing a half-step inward
sweep at :math:`\mu = -1` that initialises the
:math:`\alpha`-cascade and propagates to the outward sweep; Phase C
took the Carlson cell-centre read to be the natural anchor for that
half-step in the cell-by-cell WDD formulation. The same logic applies
to the cylindrical axis: the per-level azimuthal-DAG topology has a
half-step inward-zero-weight ordinate at :math:`\mu_x = -1` that
anchors each level's :math:`\alpha`-cascade.

⚠ **Which half of that paragraph survived.**  Lewis–Miller's
*starting direction* — a half-step solve at the pole cosine that
initialises the :math:`\alpha`-cascade — is **production**, and
:ref:`sn-direct-seed-solve` is its modern form (marched directly from the
within-group source, promoted to first-class state).  What did **not**
survive is the last clause: the *spatial* WDD pole-face anchor is not the
Carlson cell-centre read; it is the mirror continuation.  Lewis–Miller
never asked the starting direction to supply a radial face value — that
conflation was Phase C's, not the literature's.

Phase C's cylindrical analogue used the identical Carlson seed per level:
:math:`\psi^{\text{face}}_{\text{in,level}}(\text{pole}) =
\psi^{\text{cell}}(\text{first cell at level})`. For a
level-symmetric quadrature the cylinder tolerates a wrong seed via
the **dead first-ordinate weight** (:math:`c_{\rm in}[m_0]=0`) in a
way the spherical case does not — see the Gate 1.1 finding below
(and its #280 Phase 2.5b correction: this is level-symmetric-only,
NOT :math:`\alpha`-dome telescoping, and false for a product
quadrature).  Historical mechanics: since Q5.6.3 a cylindrical
``SNMesh`` refuses both of those rule classes at construction
(:ref:`sn-direct-seed-r12a`), so the tolerate-a-wrong-seed regime is
unconstructible on the live tree.

.. _sn-pole-face-mirror-continuation:

What shipped instead — the r = 0 mirror continuation
-----------------------------------------------------

The shipped pole-face initial condition is not a closure at all.  It is an
**identity of the geometry**, and recognising that is what makes it exact
where every extrapolant is not.

**The physics.**  A straight flight line does not terminate at the origin;
it passes through and re-emerges on the far side.  In the 1-D reduced
coordinate the ray that arrives at :math:`r = 0` with radial cosine
:math:`-\mu` leaves with :math:`+\mu` — the inward and outward
characteristics are *one* characteristic, cut in half by the coordinate
chart rather than by the physics.  Hence

.. math::

   \psi(0,\,+\mu) \;=\; \psi(0,\,-\mu),

which is :eq:`sn-err-058-coupled-pole-continuity`, derived in full at
:ref:`sn-err-058-manifestation-a`.  It is the :math:`r = 0` quotient's
:math:`\sigma_x` **deck transformation** — the same mirror motion the
boundary tier realizes for a specular face — so the codebase derives its
ordinate pairing from one source
(:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`
applied to the x-mirror motion) rather than from a hand-written table.

**What the outward sweep therefore reads.**  Run the inward
(:math:`-\mu`) sweep FIRST; its pole-face *outflow*, gathered at the
mirror ordinate, IS the outward (:math:`+\mu`) sweep's pole-face inflow.
The production line is one gather —
``pole_face_seed = outflow_at_inner.T[self._ensure_pole_mirror()]`` in
:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`, with
the SI sweep twin reading ``pole_outflow[mirror[global_n]]``.  The
per-ordinate invariant the gather relies on — the partner must be the
*intra-level* :math:`\mu_x` sign-flip, :math:`\mu_y` and :math:`\mu_z`
held — is stated and gated at
:ref:`sn-coupled-pole-mu-level-invariant`.

Three properties, and each one is the answer to a way the Phase C seed
failed:

* **Exact, not :math:`\mathcal{O}(h)`.**  There is no truncation to make:
  the value is *already computed*, at the face, in the same solve.  The
  Carlson cell-centre read had to guess :math:`\psi(0)` from
  :math:`\psi(\Delta r/2)`; the continuation does not guess.
* **Lower-triangular — it is data, not a self-reference.**  The inward leg
  completes before the outward leg opens, so the pole handoff is a DAG edge
  in cell-visit order and the operator stays forward-substitutable.  This is
  the "inward-determines-outward" pole condition deferred at Phase C
  (`Issue #192 <https://github.com/deOliveira-R/ORPHEUS/issues/192>`_) and
  landed by ERR-058.
* **Realizability is enforced, not assumed.**  A quadrature not closed under
  the :math:`\sigma_x` mirror has no bijective weight-preserving pairing, so
  the continuation is unrealizable on it;
  :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._ensure_pole_mirror`
  raises at first use with that diagnosis rather than silently mispairing.

.. warning::

   **No flat-flux gate can ever see this.**  On a flat :math:`\psi` the two
   candidate seeds coincide exactly — :math:`\psi^{\text{cell}}[0] =
   \psi(0,-\mu) = c` — so the per-ordinate flat-flux invariant, the
   streaming-equilibrium identity :eq:`streaming-equilibrium`, and every
   flat-:math:`\psi` L0 anchor in the tree are **structurally blind** to the
   difference (a Mode-12 stabiliser, not a tolerance question).  That is not
   a historical curiosity: it is why the wrong seed shipped through Phases
   B, C, D and F.  A gate that is to constrain the pole face must run a
   **non-flat radial profile** — the manufactured :math:`A(r) = \sin(\pi
   r/R)` ansatz, or a heterogeneous vacuum problem.

.. _sn-apply-sweep-equivalence:

apply ↔ sweep structural equivalence
-------------------------------------

Pre-Phase-C, the matvec's :meth:`apply` and the sweep's :meth:`solve`
were structurally distinct paths to the **same** discrete loss
operator :math:`L+C`: :meth:`apply` walked cell-centre storage with arithmetic
face averages, :meth:`solve` walked face storage with WDD
asymmetric propagation. The cross-domain-attacker frame analysis
(Smell 16, ``.claude/agent-memory/cross-domain-attacker/issue_168_phase_c_sweep_frame.md``)
flagged this as the elegance-smell trigger that Phase C resolves:

   *Two paths to the same discrete operator over different storage
   conventions, with order-degradation at boundaries on one path.
   FRAME: Sweep / wavefront — recover the boundary as a DAG edge
   consumed via the trace law, not as a cell-centre algebraic
   closure.*

Under the sweep-frame architecture both paths consume the **same
three primitives**:

#. The WDD diamond closure :eq:`phase-c-wdd-recurrence` per cell.
#. The direction-keyed cell-visit DAG via
   :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk` invoked with
   ``direction_sign=±1``.
#. The BC trace law applied **once** at the boundary edge per
   :ref:`affine-bc-form`.

The face-flux propagation identity therefore holds **by
construction** post-Phase-C: extracting the implicit face fluxes
from ``apply`` (by inverting the cell-balance equation) recovers
the same WDD recurrence the sweep walks. The structural-frame
identity is the load-bearing acceptance criterion for
preconditioned-Krylov stability — when ``apply`` is the loss
operator :math:`L+C` and the sweep is :math:`(L+C)^{-1}`
(approximately), they must agree on what :math:`L+C` **is**. The Phase-C
gate set lives in :file:`tests/sn/sweep/core/test_phase_c_gates.py`:

* **Gate 1.2** — ``apply(ψ) == apply(ψ)`` bit-identical across two
  invocations of the composite, on ``interior`` **and** ``boundary``.
* **Gate 1.3** (apply ↔ apply_transpose reciprocity)
  :math:`\langle (L+C)\psi, \phi \rangle = \langle \psi, (L+C)^T\phi \rangle`
  to ``rtol=1e-12, atol=1e-13``. Free if Gate 1.4 (linearity)
  passes.
* **Gate 1.4** (apply linearity) :math:`(L+C)(\alpha\psi + \beta\phi)
  = \alpha (L+C)\psi + \beta (L+C)\phi` to ``rtol=1e-13``.
  **Precondition** for Gates 1.2 + 1.3 + the dense-probe
  ``apply_transpose`` construction.

.. warning:: **Gate 1.2 does NOT pin the face-flux propagation identity —
   it pins determinism**, and the distinction matters because the identity
   is the load-bearing claim of this subsection.  Its test is named for the
   identity and its own docstring concedes the shape: *"The structural
   identity is built by construction.  We pin it via a deterministic
   input."*  A repeat-call comparison runs the same code twice, so it is
   structurally incapable of reddening for a wrong shared closure
   (``vv-principles`` #23 — an invariance gate's coverage is exactly the set
   of lines that read the varied knob, and nothing here varies).  What
   actually holds the two paths together is **single-sourcing**: both call
   :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`
   and :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.dag_walk`, so there is
   no second closure to drift — the identity is prevented rather than
   detected (``coding-standards``: single-sourcing a duplicate demotes every
   gate that compared its copies, and that is the *correct* trade).  The
   independent value evidence for the pair is the SI ≡ Krylov equivalence
   gate (:ref:`sn-issue-196-eigenvalue-equivalence`), which compares two
   genuinely different *solvers* over the one operator.

.. _bc-trace-contract-respected-by-matvec:

BC trace contract respected by matvec
-------------------------------------

.. note::

   **Superseded by Wave O steps O.4a.2 + O.4b (Issue #208).** This
   subsection documents the **Phase C** matvec, where the boundary law
   was applied *inside* the sweep at the boundary edge
   (``inflow_full = bc_outer.apply(outflow_at_boundary.T)``, the
   "keystone"). That intra-sweep BC re-apply has been **deleted** for
   **every geometry**: O.4a.2 made the **1-D** path bare, and O.4b
   made the **2-D Cartesian wavefront** path bare. The boundary law
   :math:`B` is now a first-class sibling :math:`-B` operator and the
   sweeps read the seeded inflow trace directly. The Phase C
   trace-contract *insight* — the BC must consume the WDD-propagated
   outflow face vector, not cell centres — is **preserved and
   strengthened** by the extraction: the outflow trace is now an
   explicit solved unknown :math:`\psi.\text{outflow}` rather than a
   local variable, and :math:`B` reads it as
   :math:`B\,\psi.\text{outflow}`. See :ref:`bare-sweep-extraction`
   below and the canonical algebra at :ref:`bc-extraction` in
   :doc:`/theory/foundations/boundary_conditions`.

The :doc:`/theory/foundations/boundary_conditions` :ref:`affine-bc-form` (§16A.3) reads

.. math::

   \gamma_- \psi \;=\; R\,G\,\gamma_+\psi \;+\; q,

requiring the BC operator to consume the **boundary face trace**
:math:`\gamma_+\psi`, not an interior cell-centre approximation
of the trace. The pre-Phase-C ``operator.py:533`` site read

.. code-block:: python

   outgoing = fi[:, :, -1, 0].T              # ← cell-centres
   incoming = bc_outer.apply(outgoing)

silently violating the contract. The contamination was invisible
for **specular reflection at** :math:`\alpha=1` because the
reflection permutation commutes with cell-centre fills (the same
permutation pattern applies to whatever value sits at the boundary
slot), but it surfaces for every other BC: vacuum (:math:`\alpha=0`),
albedo (:math:`0 < \alpha < 1`), prescribed inflow (any nonzero
:math:`q`), and white BC. Each of these is a regime where
higher-order spatial accuracy should appear — and where the
cell-centre approximation degrades the operator to first-order
boundary truncation.

The Phase C matvec honours the contract by construction: the BC
operator's input is the **WDD-propagated outflow face vector**, not
cell centres. The boundary-edge sequence is:

.. code-block:: python

   # ── Phase 1: outward sweep (μ > 0), i = 0 → nx-1 ─────────────
   # Carlson seed at pole: ψ_face_in = ψ_cell[0].
   for visit in sn_mesh.dag_walk(direction_sign=+1):
       i = visit.cell_idx
       psi_cell = fi[:, outgoing_mask, i, 0]
       psi_face_out = 2.0 * psi_cell - psi_face_in          # WDD
       # ... streaming + redistribution + collision scatter ...
       psi_face_in = psi_face_out                            # walk
   # The last cell's ψ_face_out is the boundary outflow face.
   outflow_at_boundary[:, outgoing_mask] = psi_face_out

   # ── BC trace law at boundary edge ────────────────────────────
   # bc_outer is the realised BC (BoundaryTraceLaw) from SNMethodSpace
   # via SNBoundaryRealizer.realize() — a 1-arg LinearOperator
   # whose apply maps Γ_+ → Γ_-, per the affine-bc-form contract.
   inflow_full = bc_outer.apply(outflow_at_boundary.T)

   # ── Phase 2: inward sweep (μ < 0), i = nx-1 → 0 ──────────────
   psi_face_in = inflow_full[incoming_mask, :].T            # BC-set
   for visit in sn_mesh.dag_walk(direction_sign=-1):
       i = visit.cell_idx
       psi_cell = fi[:, incoming_mask, i, 0]
       psi_face_out = 2.0 * psi_cell - psi_face_in          # WDD
       # ... walk ...

.. note::

   **The sketch above is the Wave-8-era shape, before two changes.**
   It is kept because it shows the *sequence* — sweep, apply the trace
   law at the boundary edge, seed the inward leg — which is what this
   section is about. Two things about it are no longer literal:

   * the sweep is **bare** since Wave O steps O.4a.2 / O.4b: it reads
     the given inflow trace instead of calling ``bc_outer.apply``, and
     the reflective coupling is delivered by the sibling :math:`-B`
     (:ref:`bc-extraction`);
   * the realized law's **domain is** :math:`\Gamma_+` since campaign
     phase B3.2, so it is fed ``γ₊.apply(face_slot)`` — not the whole
     face — and its image *is* :math:`\Gamma_-`, scattered back by
     :math:`\iota_-` (:ref:`bc-domain-narrowing`).

Under that typing there are no "outflow slots in the output" left to
be unspecified: the emission has :math:`|\Gamma_-|` rows and the rows
the old slice-write discarded are not in the operator's domain at all.
What survives unchanged is the **idiom**: the inflow face trace is the
user's "ghost cell for higher-order boundary closure", realised as a
typed
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
vector defined by the realised BC operator — not extrapolated from
interior cell centres.

**Gate 1.5** (foundation,
:func:`tests.sn.sweep.core.test_phase_c_gates.test_bc_trace_contract_respected_by_matvec_vacuum_sphere` /
:func:`tests.sn.sweep.core.test_phase_c_gates.test_bc_trace_contract_respected_by_matvec_reflective_sphere`)
pins this contract: for each
:class:`~orpheus.geometry.boundary.BoundaryTraceLaw` concrete kind
(``VacuumInflow`` / ``ReflectiveBoundary`` / ``WhiteBoundary`` /
``AlbedoBoundary`` / ``PrescribedInflow``), the apply matvec's BC
integration consumes the WDD-propagated outflow face value (not
cell centres) and produces the inflow face value consistent with
``bc.realize().apply(outflow_at_boundary)``. The assertion is
bit-identical across 5 random :math:`\psi^{\text{cell}}` inputs per
BC kind × geometry × ordinate count. ``apply(0) = 0`` is pinned
separately under vacuum + reflective BCs; under vacuum BC a flat
cell-centre :math:`\psi` produces a **non-zero** residual (the BC
physically removes the inflow), and that asymmetry is itself a
load-bearing acceptance criterion: a vacuum BC that left flat-flux
residual at zero would be the pre-Phase-C cell-centre contamination
returning silently.


.. _bare-sweep-extraction:

The bare sweep
--------------

Wave O step O.4a.2 (Issue #208, commits ``d7e1316`` / ``4c0ff96`` /
``2bdc66d``, 2026-06-03) **removed the boundary law from the 1-D
sweep entirely**. The boundary-edge ``inflow_full =
bc_outer.apply(outflow_at_boundary.T)`` line shown above (the
"keystone") is **deleted**; the 1-D ``transport_sweep`` entry (then the
production sweep, since retired at step 6) read the *seeded* inflow trace
directly:

.. code-block:: python

   # PRE-O.4a.2 (bc-in-sweep, the deleted keystone):
   inflow_full = bc_outer.apply(outflow_at_boundary.T)  # re-apply bc
   psi_face_in = inflow_full[incoming_mask, :].T        # backward seed

   # POST-O.4a.2 (1-D bare sweep):
   inflow_full = bc_outer    # incoming-ordinate slots = SEEDED inflow
   psi_face_in = inflow_full[incoming_mask, :].T        # backward seed

The seeded inflow trace is delivered by the caller as the sibling
:math:`-B` source term, in one of two ways depending on the iteration
path:

* **Driver paths** (SI / Krylov): the seed rides in
  :math:`\text{rhs.boundary}` (the boundary source
  :math:`q.\text{boundary} + B\,\psi.\text{outflow}`), delivered by
  :math:`B` as a separate coupling gain to the variadic driver (Wave O
  step O.2a; see :ref:`bc-extraction-variadic-driver`). The bare sweep's
  :meth:`StreamingCollisionOperator._solve_timed_full_field <orpheus.sn.operators.streaming.StreamingCollisionOperator._solve_timed_full_field>`
  seeds its boundary buffer from :math:`\text{rhs.boundary}`, **not**
  from the iterate ``initial_guess.boundary`` (the retired
  partner-flux carrier).
* **Direct loops** — ⛔ **there are none left in production.**  This
  bullet named two (the direct fixed-source SI loop and the final
  eigenvalue reconstruction sweep), each of which filled the inflow slots
  with :math:`B\,\psi.\text{outflow}` in place before the sweep via the
  canonical :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`.
  The first moved onto the variadic driver at Wave O O.2a; the second
  became one step of that same driven map at #448 (2026-09-06,
  :ref:`sn-finalize-one-step`), in which :math:`B` is a gain like every
  other.  The whole-trace helper survives as the sweep-tier gates'
  inter-sweep reflect (``tests/sn/_test_helpers.py``), and the
  face-restricted reflect the scheduled sweep uses is a different verb
  (:meth:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace`).

Both routes called the **identical**
:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` — :math:`B`
is single-sourced. For vacuum :math:`B = 0`, so the bare sweep reads a
zero inflow seed and the result is **bit-identical** to the
pre-extraction ``bc.apply`` of a vacuum law.

The full block-matrix derivation, the three design corrections (keep
the outflow defect; project :math:`B` to the inflow row; seed from
:math:`\text{rhs.boundary}`), the two delivery routes, and the O.2
forcing function all live at :ref:`bc-extraction` in
:doc:`/theory/foundations/boundary_conditions` — the canonical home for
the BC-extraction algebra.

Step **O.4b** extended the bare sweep to the **2-D Cartesian
wavefront** path (both :func:`~orpheus.sn.loss_representation._sweep_jacobi`
and the 2-D matvec
:meth:`StreamingOperator._apply_2d_cartesian <orpheus.sn.operators.streaming.StreamingOperator>`):
the intra-octant ``bc.apply`` is gone there too, and the
octant-incoming edge is seeded from the given inflow trace. The
``sn_mesh.reduced is not None`` predicate that guards the dispatch now
selects the **fold shape** (1-D parallel-prefix scan vs 2-D wavefront
DAG), **not** a bare-vs-bc-in-sweep distinction — both folds are bare,
so the sweep body and the helper-guard sites cannot drift. The 2-D
interior face fluxes both folds propagate are the interior 1-cochain
:math:`C^1_{\rm int}`, with the boundary seed/absorb the typed trace
operators :math:`\iota_*` / :math:`\iota^*` (#205 Phase 5; the
``WavefrontFlux`` carrier that named the cochain is retired at
S6.4(f), the cochain now living in ``_MovingFrontier`` /
``_octant_face_cochain`` — see :ref:`wavefront-flux-cochain` in
:doc:`/theory/foundations/wavefront_cochain`).

.. _sn-mms-spherical-aniso-spatial-convergence:

Spherical anisotropic-ansatz MMS convergence
--------------------------------------------

.. note:: **Retraction (2026-06-13, Issues #229 / #195).**

   The Phase-C/D attribution that follows — "the legacy angular
   closure default leaks :math:`\mathcal{O}(h^{1.3})` shape errors"
   and the rate is held back "until the angular default flips to
   ``MorelMontryAngularSweep``" — was **falsified** by the W1–W4
   root-cause program (2026-06-13).  ERR-058 (Issue #195) was the
   terminal closure-seed fix: the curvilinear *isotropic* MMS now
   converges clean :math:`\mathcal{O}(h^2)` under the existing
   default WITHOUT any default flip (the default-flip and the
   "Phase D pole-face refinement" framing below were never the
   lever).  The residual *anisotropic* floor is the
   angular half-angle-thread INTERPOLATION floor (the floor measured in
   #229), which scales with the angular **quadrature** and is
   independent of the spatial closure and of the default.  ⛔ **The
   original sentence also said "and of the** :math:`\tau`
   **-clamp"; that clause is REFUTED** (`[M]` 2026-08-11, Q5.6.4):
   removing the cylinder's :math:`[\tfrac12, 1]` absorber moves the
   anisotropic floor by :math:`1.8`--:math:`3.4\times` on the retired
   chord partition and by :math:`\sim 1.8`--:math:`2\times` on the
   shipped ω partition, so the clamp contributed a real constant to it.
   What survives is the *quadrature-scaling* attribution.  The
   numbers and the structural-asymmetry reasoning preserved below are
   historical evidence; the *interpretation* is superseded by the
   comprehensive treatment at
   :ref:`sn-curvilinear-aniso-norm-reconciliation`.  W3 (Issue #229)
   removed the xfail markers and migrated the
   :eq:`sn-mms-spherical-psi` / :eq:`sn-mms-spherical-qext` labels to
   green tests; this Gate 3.1 spherical spatial-convergence test was
   **retired** (its claim re-homed on the S32 full-ladder gate — see
   the reconciliation section).

Gate 3.1 (plan §5) is the L1 MMS verification of the spatial
convergence rate on the angularly-non-trivial ansatz

.. math::

   \psi_{\text{chosen}}(r, \mu) \;=\; \frac{A(r) + B(r)\,\mu}{W},
   \qquad
   A(r) = \cos(\pi r / (2R)), \quad B(r) = r/R,

with :math:`W` a normalisation constant. The ansatz **activates**
the angular redistribution term (the linear-:math:`\mu` content
:math:`B(r)\,\mu/W` is the curvilinear sweep's hardest math), in
explicit avoidance of Mode 7 of the ``vv-principles`` skill — an
isotropic-by-construction MMS that would null the redistribution
path by ansatz design and silently miss ERR-026. The companion
isotropic ansatz :math:`\psi = A(r)/W` would still test the
spatial closure but is **insufficient** in isolation; Phase C
ships both as separate test cases (per the Phase B closeout's
"every multi-dim MMS must declare which terms it activates AND
which it nulls" rule).

With the curvilinear default
``LegacyTauSymmetricInterpolation``
the rate stays at the pre-Phase-C
:math:`\mathcal{O}(h^{1.3})` profile. This is the diagnostic that
**the spatial-closure alignment is necessary but not sufficient**
for :math:`\mathcal{O}(h^2)` convergence; the underlying ERR-026
flux-shape drift survives Phase C's WDD sweep-frame alignment
when the angular closure default does not flip to the canonical
M–M form. Gate 3.1 is therefore marked
``@pytest.mark.xfail(strict=False)`` pending Phase D's pole-face
spatial-closure refinement (see
:ref:`sn-curvilinear-trajectory-resolvent-crosscheck-section` for the
Phase D scope summary).

The xfail is intentionally **not strict** at this gate (in
contrast to the four pre-existing ERR-026 ``xfail(strict=True)``
tripwires at the same labels). The non-strict marker reflects an
**empirical** test of an architectural prediction: if Phase C's
sweep-frame matvec accidentally moved the convergence rate past
1.9 on the legacy default (which would be the unexpected outcome
that demands a fresh investigation), the marker would flip to
``xpass`` rather than fail strictly. The strict markers stay on
the four canonical ERR-026 tripwires
(:file:`tests/sn/verification/mms/test_mms_curvilinear.py` and the L1 aniso file)
because those tests cover the closure status that Phase D will
actually close.

.. _sn-mms-cylindrical-aniso-spatial-convergence:

Cylindrical anisotropic-ansatz MMS convergence
----------------------------------------------

.. note:: **Retraction (2026-06-13, Issue #229).**

   The claim below that "the Phase D fix is expected to produce a
   clean :math:`\mathcal{O}(h^2)` cylindrical MMS rate" was
   **falsified**.  The cylinder has **NO** pre-floor
   :math:`\mathcal{O}(h^2)` window at any practical quadrature: the
   angular half-angle-thread interpolation floor dominates the spatial
   error before second order can establish (measured: even
   :math:`n_\mu = 16` reaches only order 1.80 on the coarsest segment).
   The cylinder floor scales with the **azimuthal** quadrature
   :math:`n_\varphi` (NOT the polar :math:`n_\mu`), is structurally
   blocked by duplicate azimuthal :math:`\eta` that a 1-D
   :math:`\eta`-thread cannot represent, and would need a 2-D
   :math:`(\eta, \varphi)` closure (out of scope).  W3 (Issue #229)
   replaced the Gate 3.1/3.2 cylindrical spatial-rate claim with a
   verified floor-scaling test
   (``test_cyl_aniso_floor_scales_with_quadrature``) and migrated the
   :eq:`sn-mms-cylindrical-psi` / :eq:`sn-mms-cylindrical-qext` labels
   to green.  See :ref:`sn-curvilinear-aniso-norm-reconciliation` for
   the full treatment; the reasoning preserved below is history.

Gate 3.2 is the cylindrical analogue of Gate 3.1 — same ansatz
structure (linear :math:`\mu_x` content + cosine radial profile)
adapted to the cylindrical level-DAG, parametrised across LS-4 and
Product 2×4 quadratures to surface any quadrature-family-dependent
constants (Signature 4 / ERR-004). Same xfail rationale:
spatial-closure alignment is necessary but not sufficient for
:math:`\mathcal{O}(h^2)` convergence until the angular default
flips to ``MorelMontryAngularSweep``. Cylindrical Gate 1.1
**passes** under the canonical M–M angular closure (see the
Empirical Gate 1.1 finding below), so the Phase D fix is expected
to produce a clean :math:`\mathcal{O}(h^2)` cylindrical MMS rate
without requiring the spherical pole-face refinement — but the
default must flip in unison across both geometries for the
``catches("ERR-026")`` story to be coherent. Phase D will ship
both.

Gate 3.3 (angular convergence at fixed ``nx=80``, varying
``n_ordinates``) **passes** under Phase C — the spatial closure
alignment is sufficient to expose the angular discretisation as
the limiting error when the spatial discretisation is held fine.
This is the inverse signature of Gate 3.1: holding the angular
closure fixed and refining spatially saturates at the angular
discretisation floor; holding the spatial closure fixed and
refining angularly saturates at the spatial discretisation floor;
the legacy default does not produce :math:`\mathcal{O}(h^2)`
spatial because the legacy angular closure leaks
:math:`\mathcal{O}(h^{1.3})` shape errors that the spatial
discretisation cannot resolve away.

.. _sn-curvilinear-homogeneous-kinf-recovery-section:

Homogeneous-reflective k\ :sub:`∞` recovery
--------------------------------------------

Gate 4.1 verifies the eigenvalue claim using a closed-form
reference: the 2-group homogeneous reflective sphere recovers the
analytical infinite-medium eigenvalue

.. math::
   :label: sn-curvilinear-homogeneous-kinf-recovery

   \kinf
        \;=\; \rho\bigl(\mathbf{\Sigma}_a^{-1}
              \,\boldsymbol{\chi}\,\boldsymbol{\nu\Sigma_f}^{\top}\bigr)
        \;\stackrel{1\text{G}}{=}\;
        \frac{\nu\Sigma_f}{\Sigma_a}\,,

i.e., the dominant eigenvalue of the multi-group production /
removal transfer matrix on the homogeneous infinite medium
(Lewis--Miller §3.2; reduces to :math:`\nSigf/\Sigma_a` in 1-group).
The reference is computed in closed form by
:func:`~orpheus.derivations.common.eigenvalue.kinf_homogeneous`
without any spatial or angular discretisation choice; it is the
State 1A closed-form pillar in the ``algebra-of-record`` taxonomy.
The Phase C sweep-frame matvec recovers it to ``rtol ≤ 5e-4`` on the
2-group homogeneous reflective sphere; pinned at
:func:`tests.sn.verification.analytical.test_phase_c_crosscheck.test_sn_spherical_homogeneous_kinf_recovery_2g`.

The clean :math:`k_\infty` recovery is **not** a contradiction of
ERR-026 staying at PARTIAL CLOSURE. The eigenvalue is shape-
independent: for a homogeneous reflective problem,
:math:`\kinf` is a material-property ratio
(:math:`\nSigf / \Sigma_a` over the volume-weighted average flux),
and the same ratio falls out of any discretisation that preserves
volume-weighted particle balance. Phase C's WDD spatial closure
preserves balance by construction (the streaming term telescopes
to surface area times average flux on a uniform mesh, and the
redistribution term integrates to zero against the volume weights
across an :math:`\mathcal{R}^4` cell). The shape-dependent ERR-026
flux-shape bug therefore drops out of the eigenvalue but persists
in the **flux shape** — exactly what
:ref:`sn-curvilinear-trajectory-resolvent-crosscheck-section` will
measure in Phase D. Gate 4.1 is therefore the **necessary** but
**not sufficient** evidence chain (per ``vv-principles`` 1-group
degeneracy rule); the sufficient chain requires structurally-
independent flux-shape evidence from Phase D.

.. _sn-curvilinear-trajectory-resolvent-crosscheck-section:

Trajectory-resolvent cross-check
--------------------------------

Gate 4.2 is the **flux-shape cross-check** against the
structurally-independent trajectory_resolvent Green's-function
reference (the Peierls Variant α State 1B semi-analytical pillar
in the ``algebra-of-record`` taxonomy).  The contractual claim is

.. math::
   :label: sn-curvilinear-trajectory-resolvent-crosscheck

   \bigl\|\phi^{\,\text{SN}}_h(r)
        \;-\; \phi^{\,\text{traj.res.}}(r)\bigr\|_{\infty}
        \;\le\; 5\times 10^{-4}
        \quad
        \text{on the 5 P0 curvilinear snapshots,}

with :math:`\phi^{\,\text{SN}}_h` the SN flux at the snapshot's
:math:`n_x` and :math:`\phi^{\,\text{traj.res.}}` the
trajectory-resolvent reference flux at the same radii.  The bare
function entry points cover the 5 P0 deleted curvilinear regression
snapshots:

.. list-table:: trajectory_resolvent reference coverage
   :header-rows: 1
   :widths: 35 50 15

   * - Snapshot
     - Bare entry point
     - Precision
   * - ``sphere_2g_homogeneous_dd_n20``
     - :func:`~orpheus.derivations.continuous.trajectory_resolvent.greens_function.solve_greens_function_sphere_mg`
     - :math:`k_\infty` exact via V_α1 identity
   * - ``sphere_2g_3reg_dd_n40``
     - :func:`~orpheus.derivations.continuous.trajectory_resolvent.greens_function.solve_greens_function_sphere_mr`
     - MR↔MG reduction ``rtol=1e-9``
   * - ``cyl_1g_homogeneous_LS4_dd_n20``
     - :func:`~orpheus.derivations.continuous.trajectory_resolvent.greens_function_cylinder.solve_greens_function_cylinder`
     - V_α1_cyl exact; Sood Ua-1-O-CY vacuum ``8.5e-6``
   * - ``cyl_1g_homogeneous_product_dd_n20``
     - same as above (different SN quadrature)
     - same
   * - ``cyl_2g_3reg_LS4_dd_n40``
     - :func:`~orpheus.derivations.continuous.trajectory_resolvent.greens_function_cylinder.solve_greens_function_cylinder_mr`
     - MR↔MG K=3 2G ``rtol=1e-9``

The 6th snapshot (``sphere_2g_p1_aniso_dd_n20``) routes to Gate 4.1
because :math:`P_1` anisotropic eigenvalue is still
shape-independent for a homogeneous reflective problem.

The cross-check placeholder landed as the Phase D test
:func:`tests.sn.verification.analytical.test_phase_c_crosscheck.test_phase_d_trajectory_resolvent_crosscheck`
(after the pole-face spatial-closure refinement). It is
**structurally important**: it pins
the names of the bare entry points so the reader knows
exactly where the reference comes from. The structurally-
independent cross-check is the load-bearing flux-shape evidence
for ERR-026 → CLOSED — without it the closure narrative would rest
on :math:`k_\infty` agreement alone, which is degenerate
(``vv-principles`` 1-group degeneracy rule applied to homogeneous
multi-group: any discretisation that preserves balance gets
:math:`k_\infty` right, so :math:`k_\infty` alone is not flux-shape
evidence). The Phase D L1 acceptance criterion is rtol :math:`\le
5 \times 10^{-4}` against the trajectory_resolvent reference on
each of the 5 P0 snapshots — relaxed from rtol :math:`\le 10^{-9}`
because SN nx-discretisation dominates the error budget at the
practical mesh refinement levels.

Phase B's ``pole-mm-recurrence`` label (:eq:`pole-mm-recurrence`)
**gains a tests edge transitively** through the Phase D fix: once
``MorelMontryAngularSweep`` becomes the default and Gates 3.1 / 3.2
xpass, the canonical Morel--Montry angular recurrence is exercised
by the apply matvec and pinned by an L1 test chain. Through Phase C
the label remains tested only via the Phase B foundation suite
(:file:`tests/sn/sweep/curvilinear/test_angular_closure.py`); the L1
upgrade is Phase D's responsibility.

Empirical Gate 1.1 finding: spherical-vs-cylindrical structural asymmetry
--------------------------------------------------------------------------

Gate 1.1 (the canonical curvilinear bug-class L0 diagnostic) is
the per-ordinate flat-flux residual probe: for :math:`\psi` constant
in space and per-ordinate on a reflective-BC homogeneous curvilinear
problem, the apply matvec must produce :math:`(T\psi)_{n,i,g} =
\Sigma_t \cdot \psi_{n,i,g}` per ordinate to ``rtol=1e-12``
(:math:`\Sigma_t = 0` reduces this to bit-zero to ``atol=1e-13``).
Parametrisation: spherical + cylindrical × 4 quadrature variants ×
2 group counts × 3 nx values × 2 :math:`\Sigma_t` values × 3
pole-closure strategies — 288 combinations under the strict
specification.

The empirical outcome decides the **default flip** per the user's
explicit constraint 7 (the "do not flip without empirical
evidence" sequencing). The decisive subset is the (geometry,
pole-closure) crosstab on the canonical Morel--Montry angular closure
strategy
(:class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`):

.. list-table:: Empirical Gate 1.1 outcome (Phase C, 2026-05-12)
   :header-rows: 1
   :widths: 20 25 25 30

   * - Geometry
     - Pole closure
     - :math:`\Sigma_t = 0`
     - :math:`\Sigma_t = 0.5`
   * - Sphere
     - ``LegacyTauSymmetricInterpolation``
     - PASS
     - PASS
   * - Sphere
     - ``BaileyFlatFluxRedist``
     - PASS
     - PASS
   * - Sphere
     - ``MorelMontryAngularSweep``
     - **FAIL**
     - **FAIL**
   * - Cylinder
     - ``LegacyTauSymmetricInterpolation``
     - PASS
     - PASS
   * - Cylinder
     - ``BaileyFlatFluxRedist``
     - PASS
     - PASS
   * - Cylinder
     - ``MorelMontryAngularSweep``
     - **PASS**
     - **PASS**

The asymmetry is **structural**: spherical-MMS fails;
cylindrical-MMS passes. The mechanism is the interaction between
the pole-face WDD initial condition (the Carlson seed
:math:`\psi^{\text{face}}_{\text{in}}(\text{pole}) =
\psi^{\text{cell}}[0]`) and the canonical Hébert §3.9.4 angular
recurrence's half-angle face flux at the pole:

* **Cylindrical case** has **per-level :math:`\alpha`-dome
  telescoping**. Each :math:`\mu`-level has its own
  :math:`\alpha_{n+1/2}` recurrence with its own pair of
  starting-direction face fluxes (:math:`\mu_x = -1` inward zero
  weight + :math:`\mu_x = +1` outward zero weight). The
  half-angle face flux discrepancy from the M–M recurrence's
  Carlson seed is absorbed by the level's own :math:`\alpha`-dome
  closure across its azimuthal ordinates — the level integrates to
  zero in the angular flux moment that drives the redistribution,
  and the level-to-level coupling at the level boundaries is
  through pole-azimuth-degenerate ordinates that carry no spatial
  flow. The cancellation is automatic.

  .. warning::

     **Level-symmetric-only (corrected #280 Phase 2.5b).**  This
     "the cylinder absorbs the seed discrepancy" mechanism holds
     ONLY for a level-symmetric quadrature, where the first-swept
     ordinate's seed weight is exactly zero
     (:math:`c_{\rm in}[m_0]=(1-\tau)/\tau=0` at raw :math:`\tau=1`)
     — a **dead** seed annihilated at source, not a cancellation
     across the azimuthal cascade.  For a **product** quadrature
     the starting direction coincides with the first-swept ordinate
     (:math:`t=0`, #229), so :math:`c_{\rm in}[m_0]\ne 0`, the seed
     is a **live self-coupling**, and the cold cylinder
     ``(L+C).solve`` was seed-**lagged** until the #280 2.5b
     direct-seed fold.  See the ERR-026 crosstab correction note.
     Since Q5.6.3 (``1689faf4``) the whole regime is historical:
     cylindrical ``SNMesh`` admission refuses every non-carrying
     rule (:ref:`sn-direct-seed-r12a`), and the 2.5b fold — whose
     only subjects were exactly these refused configurations — was
     retired with them.

* **Spherical case** has **no equivalent telescoping**. The
  spherical pole-face is a single point (the centre :math:`r=0`),
  not a level boundary, and the entire :math:`\alpha`-cascade
  meets there. The half-angle face flux discrepancy from the M–M
  recurrence accumulates across the full ordinate set rather than
  cancelling per level. The Carlson seed at the pole-**face**
  resolves the **outer** sweep direction; the M–M angular
  recurrence's starting-direction face flux at :math:`\mu = -1` is
  a separate seed that must be consistent with the spatial
  closure. The two seeds are not jointly consistent under Phase C
  alone — that is the Phase D scope.

The structural asymmetry is one of the **load-bearing
intellectual findings** of Phase C. The plan §1 had predicted
"sweep-frame architecture more likely to make MMS angular closure
viable (because spatial closure is now WDD throughout, matching
what MMS expects)", but the empirical probe revealed that the
spherical pole is **doubly singular** in a sense the cylindrical
case is not: both the angular :math:`\alpha`-cascade and the
spatial WDD recurrence converge to the same singular point, and
both need consistent starting-direction seeds. The Phase D
follow-up (a Carlson-style **coupled** pole sweep where the
outward-ordinate pole-face initial condition is determined by the
inward-ordinate pole-face propagation, not chosen independently)
is the architectural prescription. This is the symmetry condition
at :math:`r=0` written into the SN discretisation.

Per the user's explicit constraint 7, the default flip to
``MorelMontryAngularSweep`` is **DEFERRED to Phase D**
(`Issue #192 <https://github.com/deOliveira-R/ORPHEUS/issues/192>`_).
Cylindrical-MMS Gate 1.1 PASS is the strong positive signal for
the Phase D fix: the cylindrical structure is already shape-
correct under the canonical Hébert closure with Phase C's
sweep-frame architecture; the Phase D additional refinement
targets the spherical pole-face only and inherits cylindrical
behaviour for free.

ERR-026 closure status after the matvec alignment
-------------------------------------------------

Phase C ships the architectural alignment — sweep-frame matvec
with WDD spatial closure + BC trace law at the boundary edge +
retired Phase A
``BoundaryFaceFlux``
Protocol — but per the empirical Gate 1.1 finding above the
curvilinear default stays
``LegacyTauSymmetricInterpolation``
and the four ``xfail-strict`` curvilinear MMS tripwires STAY xfail.
ERR-026 remains at **PARTIAL CLOSURE** through Phase C — the
spatial-closure architecture is aligned (the load-bearing Phase B
precondition); the pole-face spatial-closure refinement is the
Phase D scope.

Verification gate summary
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Phase C verification gates
   :header-rows: 1
   :widths: 8 50 22 20

   * - Gate
     - Description
     - Status
     - Pinned at
   * - 1.1
     - Per-ordinate flat-flux residual
     - PASS (Legacy + BFF on both geometries); PASS (MMS on cyl); xfail (MMS on sphere)
     - ``test_phase_c_gates.py``
   * - 1.2
     - apply determinism via ``np.array_equal``
     - PASS
     - ``test_phase_c_gates.py``
   * - 1.3
     - apply ↔ apply_transpose reciprocity
     - PASS (``rtol=1e-12``)
     - ``test_phase_c_gates.py``
   * - 1.4
     - apply linearity (precondition)
     - PASS (``rtol=1e-13``)
     - ``test_phase_c_gates.py``
   * - 1.5
     - BC trace contract honoured by matvec
     - PASS
     - ``test_phase_c_gates.py``
   * - 2.1
     - 5 Cartesian regression snapshots bit-identical
     - PASS (``rtol=1e-12``)
     - ``test_dd_regression.py``
   * - 2.2
     - Phase B 28 foundation tests
     - PASS
     - ``test_angular_closure.py``
   * - 2.3
     - Phase B 5 L1 flat-flux-identity tests
     - PASS
     - ``test_pole_closure_flat_flux_identity.py``
   * - 2.4
     - 21 Phase A ``BoundaryFaceFlux`` tests retired
     - DONE
     - (file deleted)
   * - 3.1
     - Spherical anisotropic MMS spatial convergence
     - xfail (ERR-026 PARTIAL)
     - ``test_phase_c_mms.py``
   * - 3.2
     - Cylindrical anisotropic MMS spatial convergence
     - xfail (ERR-026 PARTIAL)
     - ``test_phase_c_mms.py``
   * - 3.3
     - Angular convergence at fixed nx
     - PASS
     - ``test_phase_c_mms.py``
   * - 4.1
     - :math:`k_\infty` recovery on 2G reflective sphere
     - PASS (``rtol < 5e-4``)
     - ``test_phase_c_crosscheck.py``
   * - 4.2
     - trajectory_resolvent flux-shape cross-check
     - SKIP (Phase D)
     - ``test_phase_c_crosscheck.py``

Phase D scope (shipped 2026-05-12 — Carlson coupled-pole seed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: **Read this Phase D / Phase F narrative as HISTORY (Issue
   #195 CLOSED 2026-06-12).**  The Carlson coupled-pole *seed concept*
   below is correct and survives; but two of Phase D's terminal
   decisions were later reverted by the ERR-058 fix: (i) the
   curvilinear ``inner_solver`` default flip to ``"krylov"`` was undone
   (curvilinear now defaults to ``"source_iteration"``, SI
   :math:`\equiv` Krylov bit-identical post-unification); and (ii) the
   "magnitude scope OPEN / pre-asymptotic transient" framing was
   falsified — the error PLATEAUED, the dominant defect was the
   *angular* closure seed, and ERR-058 made the isotropic MMS a clean
   :math:`\mathcal{O}(h^2)` ladder.  The
   ``CarlsonInwardSweep`` *half-angle* seed described here is also
   superseded as the default by
   ``AngularEdgeExtrapolation``.
   The production resolution is at
   :ref:`sn-err-058-closure-seed-closeout`; this section's per-step
   claims are tombstoned inline where they bear on those decisions.

Tracked at `Issue #192
<https://github.com/deOliveira-R/ORPHEUS/issues/192>`_; the
Hébert §3.9.4 inward sweep implementation is `Issue #193
<https://github.com/deOliveira-R/ORPHEUS/issues/193>`_; the
``@pytest.mark.verifies(...)`` wiring for the new equation
labels is `Issue #194
<https://github.com/deOliveira-R/ORPHEUS/issues/194>`_; the
remaining pre-asymptotic-magnitude open question is `Issue #195
<https://github.com/deOliveira-R/ORPHEUS/issues/195>`_. The
shipped deliverables flip ERR-026's **identity-and-rate** scope to
CLOSED while keeping the **magnitude** scope open per
:ref:`sn-phase-d-err-026-closure-narrative`:

1. **M-M half-angle seed refinement.** A canonical Hébert §3.9.4
   Eqs. (3.432)–(3.435) inward :math:`\mu = -1` sweep seeds the
   M-M angular recurrence's ``psi_half_left`` — replacing the
   hardcoded zero that Phase B had baked in. See
   :ref:`sn-phase-d-carlson-coupled-pole-sweep` for the full
   derivation, including the diagnostic finding that **the seed
   lives in the M-M angular recurrence, not in the WDD spatial
   pole-face IC**.
2. **Default flip.** The hub's angular-closure default (the constructor
   argument ``pole_angular_closure=`` at the time, ``angular_closure=``
   since P4.9b)
   ``LegacyTauSymmetricInterpolation``
   → :class:`~orpheus.sn.angular.closure.MorelMontryAngularSweep`;
   :class:`~orpheus.sn.solver.SNSolver` curvilinear default
   ``"source_iteration"`` → ``"krylov"``. See
   :ref:`sn-phase-d-default-flips`.
3. **Gate 1.1 sphere MMS PASS.** All 4 ``MorelMontryAngularSweep``
   parametrised cases (sphere × cylinder × :math:`\Sigma_t \in
   \{0, 0.5\}`) **xpass** on the per-ordinate flat-flux residual
   probe. See :ref:`sn-phase-d-gate-1-1-empirical`.
4. **Snapshot regeneration deferred.** The 11 DD regression
   snapshots remain bit-identical under Phase D — the SI/sweep
   path is the snapshot generator, and the SI path uses the
   sweep (not the apply matvec), so the Phase D default flip
   does NOT disturb them. Regeneration under a Phase D
   :meth:`StreamingCollisionOperator.apply`-driven Krylov path is
   carried at Issue #195.
5. **Gate 1.5 strengthened (capture-and-compare).** The §16A.3
   BC trace contract now has a stricter parametrised test that
   independently reconstructs the WDD-propagated outflow trace
   and asserts the captured BC apply input matches to
   ``rtol=1e-14``. See
   :ref:`sn-phase-d-gate-1-5-capture-and-compare` and the BC
   companion section :ref:`bc-two-bc-applies-per-matvec`.
6. **Marker partial removal — deferred.** The 4 ``xfail-strict``
   ERR-026 tripwires stay through Phase D Step 3 — they will
   ``xpass`` under the new default but require the Step 5
   marker-removal commit. ERR-026 stays at PARTIAL CLOSURE
   pending Issue #195 (pre-asymptotic-magnitude convergence).

.. _sn-direct-seed-solve:

The direct starting-direction ψ½ solve
======================================

.. admonition:: Status banner
   :class: important

   **Issue #282 — CLOSED by route (a)** (#280 Phase 2.5d, committed
   ``a29ab2d`` on branch ``refactor/sn-walk-unification``, 2026-07-04).
   The lagged Morel–Montry half-angle pole seed — a two-point
   extrapolation of the *previous* source-iteration iterate — was a
   **walk-order back edge**: it made the spherical within-group SOLVE a
   *non-direct* inverse.  Route (a) promotes the starting-direction flux
   :math:`\psi_{1/2}` to **first-class typed state** — System B's own
   :class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`
   composite (a :math:`V_{\rm cell}`-state-metric ``interior ⊕ boundary``
   of ψ½ role leaves on the split
   :class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicInteriorSpace`
   / :class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicBoundarySpace`)
   — and marches it **directly** from the true within-group source
   :math:`\bar q_{1/2}` through System B's named resolvent
   :meth:`RadialCharacteristicOperator.solve <orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve>`
   (the Hébert §3.9.4 :eq:`hebert-3-434`–:eq:`hebert-3-435` recurrence
   :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`).

   **Keystone:** the sphere cold-start residual
   :math:`\lVert A\cdot\mathrm{solve}(b) - b\rVert_\infty / \lVert b\rVert_\infty`
   (:math:`A` here and throughout this section: the **augmented loss**
   :math:`L+C` on the System A :math:`\oplus` System B composite — the
   swept sub-composite of the honest :math:`A = L+C-S-N_{2n}-B`)
   collapses from :math:`5.18\times10^{5}` to
   :math:`2.5\times10^{-16}`, and the seed-insensitivity
   :math:`\Delta` from :math:`4.57\times10^{-2}` to :math:`0` **bitwise**.
   The cold solve is now a genuine single-pass exact inverse — the
   posture the DSA program (#2) and the curvilinear Krylov
   preconditioner (#200) require, and the deliverable that lets the
   #280 unified walk build a spherical ``sweep_transpose`` against a
   triangular forward operator.

   This section is the **resolution chapter** of the seed-strategy saga
   (:ref:`sn-phase-d-carlson-coupled-pole-sweep`,
   :ref:`sn-phase-f-carlson-sweep-path-backport`,
   :ref:`sn-err-058-closure-seed-closeout`).  Those sections are
   preserved as the record of *what was tried and why it fell short*;
   the ``PsiHalfAngleSeed`` strategy family they built is **retired**
   here (see :ref:`sn-direct-seed-strategy-zoo`).

The lagged pole seed was a walk-order back edge
-----------------------------------------------

The curvilinear Morel–Montry angular recurrence :eq:`pole-mm-recurrence`
marches half-angle face fluxes :math:`\phi_{n+1/2,i}` up the
:math:`\alpha`-cascade from a **seed** at the level's starting direction
:math:`\mu_{\rm start}` (sphere: :math:`\mu = -1`).  The seed
:math:`\phi_{1/2,i} \equiv \psi_{1/2,i}` is the input field's value at
that closed ray.  Through Phase D–F (:ref:`sn-err-058-manifestation-b`)
the seed was produced by extrapolating the field **linearly in**
:math:`\mu` through the level's two most-inward ordinates —
operator-consistent for the forward *apply*, but structurally poisonous
for the *solve*.

The poison is a **directed-graph** fact.  A within-group solve
:math:`A\psi = q` is a single-pass direct inverse **iff** the operator is
triangular in the sweep-order permutation — every unknown depends only on
already-computed unknowns (:ref:`loss-rep-three-modes`; the #284
object-level discharge).  The two-point extrapolation seed reads
ordinate columns that the sweep visits **later** in the walk (the
inward-marching :math:`\mu<0` sweep needs the seed *before* it has swept
the :math:`\mu` neighbours the extrapolation samples).  In the sweep-order
permutation that places one entry **above the diagonal** — a back edge.
Consequences, all measured on a 4-cell homogeneous sphere:

* the spherical solve is **not** single-pass: the cold-start residual
  :math:`\lVert A\cdot\mathrm{solve}(b) - b\rVert_\infty/\lVert b\rVert_\infty`
  sits at :math:`5.18\times10^{5}` (a direct inverse must be
  :math:`\sim 10^{-16}`);
* the solve is **sensitive to the initial guess** — two random guesses
  give solutions differing by :math:`\Delta = 4.57\times10^{-2}` — because
  the seed reads the *previous iterate*, a lag that is harmless **at** the
  scattering fixed point but pollutes the cold, no-outer-iteration solve;
* a coarse pure-absorber (:math:`c = 0`, no scattering outer loop to mask
  the lag) returns ``NaN``; a coarse S\ :sub:`8` 16-cell fixed source
  returns ``NaN`` under source iteration and **negative flux** under
  Krylov.

The **#282 back edge is spherical-only**: the lagged two-point
extrapolation reads *later* ordinate columns only on the sphere's
Gauss–Legendre cascade (:math:`\tau_{{\rm raw},0}\in(0,1)`, the R12a
trichotomy below).  On a cylinder the starting direction carries **no
independent state** — a *dead* first-ordinate weight on a level-symmetric
rule (:math:`\tau_{{\rm raw},0}=1`, :math:`c_{\rm in}[m_0]=0`), a
:math:`\psi_0` rank-duplicate on a product rule
(:math:`\tau_{{\rm raw},0}=0`) — so no seed row lands **above** the
diagonal (the #282 "0.0-bit" row).  Route (a) therefore touches only the
sphere.

.. note::

   **Not** ':math:`\alpha`-dome telescoping' (#280 Phase 2.5b).  The
   cylinder's seed-insensitivity is the *dead first-ordinate weight* of
   the level-symmetric rule, a level-symmetric-only artefact (see
   :ref:`sn-phase-d-gate-1-1-empirical`).  On a **product** rule the seed
   :math:`\psi_0` is a **live self-coupling** on the :math:`m_0` diagonal
   (:math:`c_{\rm in}[m_0]\ne 0`), and the cold product-cylinder
   ``(L+C).solve`` was itself seed-**lagged** (cold error :math:`\approx
   0.57`) until the #280 Phase 2.5b direct-seed fold
   (:math:`c_{\rm out}\to c_{\rm out}-c_{\rm in}`) folded it onto the
   diagonal, making it a single-pass direct inverse — resolved by the
   SAME forward substitution the sphere route (a) certifies, not by any
   telescoping.  The fold itself was **retired at Q5.6.3**
   (``1689faf4``): cylindrical ``SNMesh`` admission now refuses every
   non-carrying rule (:ref:`sn-direct-seed-r12a`), so the self-coupled
   seed the fold absorbed is unconstructible — every admitted cylinder
   level carries a genuine independent seed, resolved by the same
   route-(a) forward substitution with no folded correction.

The pole is a straight characteristic — the physics beneath the direct solve
----------------------------------------------------------------------------

The direct solve is not a numerical trick — it is a **physical property
of the closed rays**, and framing it that way (not as a storage choice)
is what makes the rest of route (a) inevitable.  In curvilinear geometry
the streaming operator carries an angular-redistribution term whose
strength is the factor :math:`(1-\mu^2)`:

.. math::

   \Omega\cdot\nabla\psi
   \;=\; \mu\,\frac{\partial\psi}{\partial r}
       \;+\; \frac{1-\mu^2}{r}\,\frac{\partial\psi}{\partial\mu}
   \qquad(\text{sphere}).

At the poles :math:`\mu = \pm 1` the coefficient :math:`(1-\mu^2)`
vanishes.  These are the **radial** directions: a particle at
:math:`\mu = \pm 1` streams straight through the origin, never changing
angular cell.  The redistribution term switches off — equivalently the
:math:`\alpha`-dome endpoints are :math:`\alpha_{1/2} = \alpha_{N+1/2} =
0` — and the transport equation at the pole collapses to a **pure 1-D
spatial ODE in radius alone**:

.. math::
   :label: sn-direct-seed-pole-straight-characteristic

   \mu\,\frac{d\psi_{1/2}}{dr} \;+\; \sigma_t(r)\,\psi_{1/2}(r)
   \;=\; \bar q_{1/2}(r),
   \qquad \mu = \mp 1,

with **no coupling to any other ordinate** (Hébert §3.9.4, the Carlson
inward march :eq:`hebert-3-434`–:eq:`hebert-3-435`).  The pole flux is
therefore computable *by itself*, before and independently of the angular
cascade it goes on to seed.

.. (vv-status rationale) Literature-transcribed derivation identity: the
.. sphere transport equation restricted to the straight-characteristic
.. pole μ=∓1, where (1-μ²)=0 kills angular redistribution.  Not a solver
.. claim — the verifiable content is the C(i) direct-solve residual
.. collapse and the block-triangular A_sb=0 certificate (§16.C), tabled
.. under the route-(a) evidence below.
.. vv-status: sn-direct-seed-pole-straight-characteristic documented

This is the physics that makes route (a) possible and the augmented
operator triangular.  The seed rows of :eq:`sn-direct-seed-block-triangular` are
self-contained (:math:`A_{\rm sb} = 0`) **because** the pole ODE reads no
bulk and no trace unknown — a straight characteristic couples to nothing
downstream.  The representation choice — promote :math:`\psi_{1/2}` to
first-class state — is *downstream* of the physics: any storage would
inherit the same decoupling.  What the lagged seed got wrong was never
the physics; it was reading the *iterate* for a quantity the pole ODE can
solve **directly** from the source (:ref:`sn-direct-seed-source-fold`).  The
deeper structural reason :math:`\mu = \pm 1` is the *only* admissible
starting direction in any curvilinear geometry is set out under
:ref:`sn-phase-d-pomraning-structural-singularity`.

.. _sn-angular-endpoint-defect:

Both ends of the march are straight characteristics — and the defect between
-----------------------------------------------------------------------------

Read :eq:`sn-direct-seed-pole-straight-characteristic` again and notice the
sign it carries: :math:`\mu = \mp 1`.  The argument that decouples the pole
ODE is :math:`(1-\mu^2) = 0`, and that is true at **both** poles.  The
:math:`\alpha`-dome says the same thing in the discrete language —
:math:`\alpha_{1/2} = \alpha_{M+1/2} = 0` (:ref:`sn-alpha-dome-closes`) — so
the cell balance decouples into a plain radial DD ODE at each end of the
level's angular march, not only at the start.

Production solves both, with one engine.
:meth:`RadialCharacteristicOperator.solve <orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve>`
calls
:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
**twice** per carrying level: the inward :math:`\mu = -1` leg, whose cells
become the M-M recurrence's seed, and — after the pole continuation
:math:`\psi^{+}_{1/2}(0) = \psi^{-}_{1/2}(0)` — the outward :math:`\mu = +1`
leg, stored as ``cells(p, +1)``.

That makes the march **over-determined**, and the over-determination had
gone unnamed.  The recurrence, marched from the seed across all :math:`M`
ordinates, arrives at the far edge and predicts :math:`\psi_{M+1/2}` a
*second* time — the slice
:attr:`~orpheus.sn.angular.closure._MMHalfGrid.trailing_face`.
Only one of the two is imposed.  The difference is a first-class quantity:

.. math::
   :label: sn-angular-endpoint-defect-eq

   D_p \;:=\; \psi_{M+\frac12}\big|_p \;-\; \psi^{\text{marched}}_p(+1)

per carrying level, computed by
:meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.angular_endpoint_defect_per_level`.
Non-carrying levels have no :math:`D` at all — their seed is an
*interpolation* (:meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.edge_extrapolated_seed`),
not a solved endpoint, so there is no second computation to compare against
and :math:`D` is **undefined rather than zero**.

**Why the far end carries no coefficient, and why that is correct.**  Nothing
consumes :attr:`trailing_face <orpheus.sn.angular.closure._MMHalfGrid.trailing_face>`
in the balance.  The M-M closure is substituted *into* the cell balance —
that substitution is where :math:`c_{\rm in}` / :math:`c_{\rm out}` come
from (:ref:`sn-closure-c-constants-owned`) — so a half-angle face appears
only as some ordinate's **upstream** datum, and the last ordinate's outgoing
coefficient is :math:`c_{\rm out}[M-1] = \alpha_{M+1/2}/\tau = 0` because the
dome closes.  The face is computed and then annihilated.  The adjoint agrees
independently: ``angular_adjoint`` seeds ``psi_half_bar[:, :M, :]`` only,
leaving index :math:`M` at zero, so the last ordinate has no path at all
through the angular channel.  This is why
:attr:`~orpheus.sn.angular.closure._MMHalfGrid.upstream_per_ordinate`
returns ``faces[:, :-1, :]``: the trailing slice is real, correct, and not
anyone's upstream.

.. warning:: **⛔** :math:`D` **is a consistency residual.  It is NOT an
   error estimator, and it may not vote on** :math:`\tau`.

   This is a measurement, not a caution.  `[M]` 2026-08-12, against the
   **analytic** anisotropic cylindrical MMS (an analytic reference, not a
   second ORPHEUS solver), the Pearson correlation of :math:`\log D` with
   :math:`\log` of the true MMS error across four :math:`\tau` variants runs
   :math:`+0.7515 / +0.2608 / +0.0630` at :math:`n_\varphi = 8 / 16 / 32`,
   with :math:`2/4 \to 0/4 \to 0/4` rank agreement — it **degrades
   monotonically to zero** as angle refines.  Structurally it must:
   :math:`D = e_1 - e_2` is a *difference of two truncation errors*, hence
   small exactly when both are large and equal.  :math:`D` does rank the
   shipped Q5.6.4 angular cell partition first, by 2.6–45× over garbage
   :math:`\tau` — and that ranking is **not** evidence for the partition,
   because the instrument is uncorrelated with accuracy (``vv-principles``
   #24(b): a metric in rank correlation with a mechanism nobody is debating
   cannot adjudicate the one they are).  The campaign still has **no**
   reference-free instrument that can rank :math:`\tau`; any future
   :math:`D`-based :math:`\tau` argument must cite that first.

   ⛔ :math:`D` **must also not be used to CORRECT the seed.**  The march's
   linear part is exactly :math:`(-1)^M I` — it follows from
   :math:`\prod_m (1-\tau_m)/\tau_m = 1`, gated on both arms — and BOTH
   endpoint values come from physics.  Imposing both is an
   over-determination, i.e. a constraint on the interior solution, not an
   equation for a free parameter; zeroing :math:`D` would merely force the
   marched endpoint onto the directly-marched one with no evidence the
   latter is the better of the two.

ψ½ as first-class state — the augmented composite (System A ⊕ System B)
--------------------------------------------------------------------------

Route (a) kills the back edge by making the seed a **state variable the
solve computes**, not a functional of the iterate it reads.  The
within-group phase space carries a starting-direction subspace beyond the
bulk and trace:

.. math::
   :label: sn-direct-seed-augmented-composite

   V \;=\; V_{\rm bulk} \,\oplus\, V_{\rm trace} \,\oplus\, V_{\rm sd},
   \qquad
   V_{\rm sd} \;=\;
   \bigoplus_{p\,\in\,\mathcal{P}_{\rm carry}}
   \bigl(\underbrace{V_{1/2,p}^{-}}_{\mu=-1\ \rm leg}
         \oplus\, V_{1/2,p}^{+}\bigr),

one starting-direction block :math:`V_{1/2,p}^{\pm}` per **carrying**
:math:`\mu`-level :math:`p` (R12a; the GL sphere carries — one level —
and, since Q5.6, every level of a σ_y-folded cylinder carries — its
arcs start genuinely off-node, T22b).  Each block holds the level's half-angle flux at
every radial cell (the **interior**
:class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicInteriorSpace`)
plus its two :math:`r = R` corner slots (the **boundary**
:class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicBoundarySpace`)
— two flat backing buffers with typed ``(level, sign)`` views, mirroring the
:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
trace layout.  The single unified ψ½ buffer (which then held the
``RadialCharacteristicSpace`` name) split into this interior ⊕ boundary
pair at 4e-e1.  Like every typed phase-space quantity in this codebase
the seed is realised by a **role family** — here a quadruple of roles,
each split into an ``interior ⊕ boundary`` locus pair (4e-e1) and composed
by System B's ``RadialCharacteristicField`` (role-erased slots — role
identity lives on the members):

.. list-table:: The ψ½ role family — four roles × two split loci, composed by ``RadialCharacteristicField`` (#282 route (a); split at 4e-e1)
   :header-rows: 1
   :widths: 34 30 36

   * - Role — the ``interior ⊕ boundary`` leaf pair
     - Realises
     - Forced by
   * - **flux** —
       :class:`~orpheus.transport.fields.radial_characteristic_interior_flux.RadialCharacteristicInteriorFlux`
       ⊕ :class:`~orpheus.transport.fields.radial_characteristic_boundary_flux.RadialCharacteristicBoundaryFlux`
     - the ψ½ state :math:`\psi_{1/2}` itself (the marched cells ⊕ the
       :math:`r = R` corner)
     - the carrier promotion (2.5d d1)
   * - **source/sink** —
       :class:`~orpheus.transport.source_sinks.radial_characteristic_interior_source_sink.RadialCharacteristicInteriorSourceSink`
       ⊕ :class:`~orpheus.transport.source_sinks.radial_characteristic_boundary_source_sink.RadialCharacteristicBoundarySourceSink`
     - the q½ source block :math:`\bar q_{1/2}` (the fold below) and any
       operator ``.apply`` output on the seed rows
     - the augmented source composite
   * - ⛔ **displacement** — ``RadialCharacteristicInteriorDisplacement``
       ⊕ ``RadialCharacteristicBoundaryDisplacement``
     - the affine displacement between two ψ½ states (minted per block by ⊖).
       **RETIRED 2026-08-19** (campaign-1 CS3): flux lives in the vector
       space :math:`V`, so the difference of two ψ½ states is the ψ½ **flux**
       composite carrying signed values, and the displacement family
       retired with the ontology — :ref:`cone-role-grid`.
     - the composite torsor algebra (2.5d d1) — retired with the row
   * - **residual** —
       :class:`~orpheus.transport.residuals.radial_characteristic_interior_residual.RadialCharacteristicInteriorResidual`
       ⊕ :class:`~orpheus.transport.residuals.radial_characteristic_boundary_residual.RadialCharacteristicBoundaryResidual`
     - System B's typed residual members :math:`r_B = (A\psi)_B - q_B`
     - :func:`~orpheus.sn.solver.evaluate_residual`'s coupled arm (B.2d)

The historical **unified** single-buffer leaf — cells ⊕ corner interleaved
on one ``FaceField[(level, sign, part)]`` — carried the
``RadialCharacteristicField`` name until 4e; that name was reminted onto
System B's composite at 4e-e1b (see the "Where ψ½ lives" note below), and
the eight split role leaves above are the only ψ½ representation.

**Where ψ½ lives — System B, not a block on the bulk composite (the B.2d
eviction).**  The role family above is a **separate composite**, never a
third block bolted onto the bulk ⊕ trace field.  System A — the ordinary
bulk ⊕ trace SN state — is the pure **2-block**
:class:`~orpheus.transport.full_field.FullField`
(``Composite[BulkField, BoundaryField]``); the ψ½ ray is **System B**, its
own 2-block
:class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`
(the marched interior cells ⊕ the :math:`r = R` corner, carrying the same
flux / source-sink / displacement / residual role family), and on a
carrying sphere the driver iterate is the **coupled pair**
:class:`CoupledField[ψ_A, ψ_B] <orpheus.numerics.coupled_system.CoupledField>`
threaded through the within-group 2×2 loss grid (blocks :math:`A_{AA}`,
:math:`A_{AB}`, :math:`A_{BA}`, :math:`A_{BB}` — the :math:`A = M - N`
splitting built by
:func:`~orpheus.sn.coupled_system.build_within_group_system`; the grid
algebra is on :doc:`/theory/foundations/operator_algebra`).  The transitional 2.5d interim —
ψ½ as an **optional third block** on ``FullField`` with a mesh-keyed
*mixed-presence law* and runtime presence pins — is **retired**: a
live-ray :math:`\psi_A` is now **unrepresentable** (the type system is the
guard, not a runtime branch), so the B.2c dead-slot double-count hazard
(the welded seed feed **and** an explicit :math:`A_{AB}` block both firing
on one carrier) dissolved **structurally**.  The coupled flat dimension is
the **honest two-system sum** — no dead padding — which is why the ERR-053
``restart`` sizing (:ref:`sn-direct-seed-gotchas`) reads the true count off the
coupled ravel.  The converged ψ½ state is returned as
:attr:`Solution.radial_characteristic <orpheus.sn.solution.Solution.radial_characteristic>`
— System B's **own typed member**, ``None`` exactly when the mesh carries
no seed level (presence validated as a **biconditional** at construction)
— while :attr:`Solution.angular_flux <orpheus.sn.solution.Solution.angular_flux>`
stays the honest 2-block System-A composite.

**The state metric.**  The inner-product weight of :math:`V_{\rm sd}`
is the SPD **state metric** :math:`G_{\rm sd} = V_{\rm cell}` — the
radial cell-volume measure, mirroring the bulk metric
:math:`G_{\rm bulk} = V_{\rm cell}\,w_n` (the SAME spatial measure,
restricted to the single :math:`\mu = \pm 1` ray, without the angular
factor :math:`w_n`).  ψ½ is a **first-class radial state field**, not a
face trace: its operator self-block :math:`A_{\rm ss}` is a *banded
radial transport operator* :math:`\mu\,\partial_r + \sigma_t` (Hébert
Eqs. 3.434–3.435), so — like any state — its Hilbert metric is set by its
**operator role**, not by an integration weight.

Three pole-vanishing quantities were historically conflated into one
"ghost" zero; keep them apart — only the operator coefficient is zero:

.. list-table:: The three pole-vanishing quantities at :math:`\mu = \pm 1`
   :header-rows: 1
   :widths: 8 34 58

   * - Tag
     - Quantity
     - Where it lives / what it governs
   * - **M1**
     - moment / output weight :math:`= 0`
     - the *open* Gauss–Legendre rule has **no node** at the pole, so ψ½
       carries zero weight in :math:`\phi = \sum_n w_n\psi_n` — it lives
       in the **moment reducer** and correctly excludes ψ½ from the
       scalar flux.
   * - **M2**
     - angular through-flux coefficient
       :math:`(1-\mu^2)\big|_{\mu=\pm 1} = 0` (the :math:`\alpha`-dome
       endpoints :math:`\alpha_{1/2} = \alpha_{N+1/2} = 0`)
     - an **operator coefficient inside** :math:`A` — the
       angular-redistribution strength that makes the pole a straight
       characteristic (:eq:`sn-direct-seed-pole-straight-characteristic`).
       Correctly zero.
   * - **M3**
     - **state metric** :math:`G_{\rm sd} = V_{\rm cell} \neq 0`
     - *this block's* inner product — governs the G-adjoint reciprocity
       :math:`\langle A\psi,\chi\rangle_G =
       \langle\psi, A^{\dagger}\chi\rangle_G`.

The retired **"ghost metric" bug** installed **M2** (an operator
coefficient) as **M3** (the state metric): it read the angular
through-flux weight :math:`(1-\mu^2)|_{\rm pole} = 0` as the Hilbert
metric and set :math:`G_{\rm sd} \equiv 0`.  Because
:meth:`~orpheus.numerics.operator.SupportsAdjoint.apply_transpose` is the
*exact* Euclidean transpose (:math:`\lVert T - A^{\mathsf T}\rVert =
3.6\times10^{-16}`), the relation :math:`A^{\mathsf T}G = G A^{\dagger}`
behind :math:`A^{\dagger} = G^{-1}A^{\mathsf T}G` holds for **every** SPD
:math:`G_{\rm sd}` (the reciprocity is gauge-free among SPD choices), and
:math:`0` is the **one forbidden value** — it puts the seed rows in
:math:`\ker G`, severing the seed :math:`\to` bulk coupling
:math:`A_{\rm bs}` from :math:`A^{\dagger}` (a wrong adjoint the instant
the seed carries data — a production reciprocity defect of
:math:`1.3\times10^{-2}`, green only on a present-but-zero seed).  We
gauge-fix to :math:`V_{\rm cell}` (dropping the angular :math:`w` — a
single :math:`\mu=\pm 1` ray has no canonical quadrature weight) so the
adjoint's seed block is the physical **backward radial march** and all
bulk/trace observables (:math:`\phi^{\dagger}`, adjoint reaction rates)
are bitwise **gauge-invariant** (the block-upper-triangular
:math:`A^{\dagger}` seats the seed at the top, so only
:math:`\phi^{\dagger}_{\rm seed}` moves with the gauge).  Consequently
:meth:`~orpheus.numerics.space.FunctionSpace.apply_metric` **scales** the
block by :math:`V_{\rm cell}`, its inverse **divides** (empty null
space), and the block contributes :math:`\sum V_{\rm cell}\,x\,y` to the
composite inner product.  This closes a sharp V&V gap — see
:ref:`sn-direct-seed-gotchas` (Mode 12, ERR-067).  The gauge derivation of record
is
``.claude/agent-memory/numerics-investigator/radial_characteristic_metric_gauge_derivation.md``.

.. (vv-status rationale) Structural / representational identity: the
.. named-field-typed decomposition of the augmented within-group phase
.. space.  Not a solver claim — the verifiable content is the
.. RadialCharacteristicSpace layout / role-quadruple class-identity
.. arithmetic (Field Layer-1 gate) + the §16.A carrier gates.
.. vv-status: sn-direct-seed-augmented-composite documented

.. _sn-direct-seed-pole-state-metric:

The through-flux coefficient is not the state metric
----------------------------------------------------------------

The M1/M2/M3 table above turns on one structural fact worth spelling out,
because getting it wrong is exactly what produced the retired ghost
metric.  A codim-1 face carries a **through-flux** coefficient (M2) — the
normal component of the streaming flux across the face, which vanishes
when the characteristic runs *tangent* to it.  Both codim-1 traces have
one, and for one of them the through-flux coincides with the state
metric while for the other it does not:

.. list-table:: The through-flux coefficient (M2) is an OPERATOR coefficient on both faces
   :header-rows: 1
   :widths: 26 22 28 24

   * - Codim-1 face
     - Through-flux M2
     - Vanishes when…
     - State metric M3
   * - **spatial** :math:`r`-face — the boundary trace
       (:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`)
     - :math:`\lvert\Omega\cdot n\rvert\,w`
     - :math:`\Omega` is **tangent** to the surface
       (:math:`\lvert\Omega\cdot n\rvert = 0`, grazing incidence)
     - **equals** the through-flux
       :math:`\lvert\Omega\cdot n\rvert\,w`
   * - **angular** :math:`\mu = \pm 1` edge — the ψ½ seed
       (System B's
       :class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`)
     - :math:`(1-\mu^2)\,w = 0`
     - the ray is a **straight characteristic**
       (:math:`(1-\mu^2) = 0`, the pole)
     - :math:`V_{\rm cell}` (radial cell volume) — **NOT** the
       through-flux

For the pole, :math:`(1-\mu^2)\,w \equiv 0` correctly captures the M2
through-flux: the :math:`\mu = \pm 1` angular face is *entirely grazing*,
so no flux streams *across* it.  That is the straight-characteristic
physics of :eq:`sn-direct-seed-pole-straight-characteristic`, and it is exactly
why the augmented operator is triangular.  **What is wrong is reading that
through-flux coefficient as the block's Hilbert STATE metric.**

The through-flux coefficient equals the state metric only when the face's
**operator self-block is trivial**.  The spatial trace's self-block
:math:`A_{\rm tt}` is a pure restriction / reflection map (measured
off-diagonal norm :math:`\approx 2` on a 6-cell sphere, diagonal in
:math:`[-1, 1]` — no interior dynamics), so there the through-flux, the
state metric, and the partial current all coincide.  The pole's
self-block :math:`A_{\rm ss}` is a **banded radial transport operator**
:math:`\mu\,\partial_r + \sigma_t` (off-diagonal norm :math:`\approx 71`,
about :math:`35\times` the trace's, diagonal in :math:`[-1, 3.65]`) with
genuine interior radial dynamics — so its through-flux (:math:`0`) and its
state metric (:math:`V_{\rm cell}`) are *different objects*.  Installing
the M2 through-flux as the M3 state metric is a **category error** — an
operator coefficient placed where the inner product belongs — and that is
precisely the retired ghost :math:`G_{\rm sd} \equiv 0`.  The state metric
is the radial cell volume :math:`V_{\rm cell}` (derived, gauged, and
V&V-consequential above).

.. important::

   **Three measures at the pole; do not conflate them.**  M1, M2, M3 (the
   first table) answer three different questions, and only the operator
   coefficient M2 is zero:

   * **M1 — the scalar-flux moment** :math:`\int\psi\,d\mu`: "how much
     does this ray contribute to :math:`\phi`?"  *Rule-dependent* — under
     the sphere's *open* Gauss–Legendre rule
     (:ref:`sn-direct-seed-circle-vs-interval`) the pole has **no interior node**,
     so its moment weight is zero and the seed is a pure auxiliary DOF;
     under a pole-*including* rule (Gauss–Lobatto,
     :ref:`sn-direct-seed-lobatto-study`) it would carry a genuine nonzero moment
     weight.
   * **M2 — the angular through-flux** :math:`(1-\mu^2)`: "how much
     streams *across* this angular face?"  Zero at the pole **always**
     (independent of the quadrature) — an operator coefficient, never a
     state metric.
   * **M3 — the state metric** :math:`G_{\rm sd} = V_{\rm cell}`: "what
     is the inner product on the ψ½ state?"  Nonzero **always** — set by
     the operator role.

   The retired bug conflated **M2 with M3** (an operator coefficient read
   as the Hilbert metric).  The moment-vs-through-flux distinction (**M1
   vs M2**) is a *separate* trap — both are quadrature/operator facts, and
   neither is the state metric.

The walk triple — solve marches, apply reads, transpose reverses
----------------------------------------------------------------

The augmented loss operator :math:`A = L + C` acts on the seed block
through three orientation-coherent paths of the **one** 1-D scan/loop
walk (:class:`~orpheus.sn.loss_representation._OneDimScanWalk`; the #280
unification).  Ordering the unknowns **seed⁻ ≺ seed⁺ ≺ ordinate legs**
makes the augmented operator block-lower-triangular:

.. math::
   :label: sn-direct-seed-block-triangular

   A \;=\;
   \begin{bmatrix} A_{\rm ss} & 0 \\[2pt] A_{\rm bs} & A_{\rm bb} \end{bmatrix},
   \qquad
   \begin{aligned}
   A_{\rm ss} &: \text{the seed rows (Hébert DD residual + corner rows),} \\
   A_{\rm bs} &: \text{the seed} \to \text{bulk M-M recurrence coupling,} \\
   A_{\rm bb} &: \text{the bulk} (L+C)\ \text{walk (}A_{\rm sb} = 0\text{).}
   \end{aligned}

The zero upper-right block :math:`A_{\rm sb} = 0` **is** the death of the
back edge: the seed rows are *self-contained* in the seed state (they read
no bulk or trace unknown), and the seed → bulk coupling is one-directional
(the M-M recurrence reads :math:`\psi_{1/2}` into the bulk rows'
``angular_numer``, never the reverse).  The three orientations:

* **SOLVE** — routed through System B's named resolvent
  :meth:`RadialCharacteristicOperator.solve <orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve>`
  (the **4e-e2 un-weave**: the walk constructs ``A_BB`` over its own
  :math:`\sigma_t` and calls it **once, up front** — System B is
  iterate-independent — instead of inlining the two-leg march).  The two
  ψ½ legs are solved **directly** from the true q½ source: march inward
  from the seeded :math:`r = R` inflow corner (vacuum ⇒ 0; reflective ⇒
  the mirror outflow corner) via the Hébert
  :eq:`hebert-3-434`–:eq:`hebert-3-435` DD recurrence
  (:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`,
  the single-sourced DD **engine** — the only place the march lives),
  pole-continue :math:`\psi^{+}_{1/2}(0) = \psi^{-}_{1/2}(0)` (the inward
  march's exit face **is** the pole datum), then march the :math:`\mu = +1`
  leg outward to the :math:`r = R` outflow corner on the **reversed** cell
  data (orientation is carried by the data, never a flag — the 2.5a
  discipline).  The walk then reads the marched inward cells off the
  carrier **as** the M-M recurrence seed; the iterate plays no role.
* **APPLY** (the matvec).  Reads the *given* ψ½ carrier and **emits** the
  seed-block rows: per leg the Hébert (3.434) residual
  :math:`m_{1/2,i} = \sigma_i\psi_{1/2,i} + (2/\Delta r_i)(\psi_{1/2,i} - f_{{\rm in},i})`
  reconstructed from the stored cells (the DD face chain replays the
  solve's arithmetic), plus the inflow-corner *identity* row and the
  outflow-corner *streamed − stored* defect row.  Because the apply
  replays the solve's ops in the same order, ``apply ∘ solve`` closes the
  corner defect to :math:`0` bit.
* **TRANSPOSE** (``loss_action_transpose``).  The exact reverse-mode of
  the forward straight-line program: reverse the corner rows, the +
  leg's chain (descending), the pole continuation, the − leg's chain
  (ascending); the reverse M-M recurrence
  (:meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.angular_adjoint`)
  **stops** at the seed cotangent and lands it on the output composite's
  seed block, rather than scattering it back onto the bulk.

The triangularity is not asserted — it is **certified** by a probed
``triu == 0`` check on the assembled augmented block (the transpose
analogue of the #284 object-level discharge): the sweep is LAPACK forward
substitution on the source subspace.

.. note:: **The ray solve is un-woven from the walk (Cardinal Rule 2 — 4e-e2).**

   The two ψ½ **orchestrations** — the forward SOLVE and its reverse-scan
   solve-transpose — no longer live inline in the walk.  Since 4e-e2 both
   route through System B's named resolvent
   :meth:`RadialCharacteristicOperator.solve <orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve>`
   / :meth:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve_transpose`
   (``A_BB``, constructed inside the walk over its own :math:`\sigma_t`),
   which the coupled 2×2 grid also exposes at its ``(B, B)`` slot.  The
   two-leg **orchestration** (read source views → inward leg →
   pole-continue → reversed outward leg → write flux views) now lives in
   exactly **one** place —
   :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator`
   — and the DD **engine** it drives in exactly one other
   (:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
   / :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_transpose`).
   The walk's ``carlson_inward_sweep_*`` references went **8 → 0**; a
   source-scan tripwire (``test_4e_unweave_walk_source_has_no_carlson_reference``)
   pins that the walk holds zero references to the engines, and the
   :ref:`Mode-11 wrap-sentinels <sn-direct-seed-numerical-evidence>` re-aim onto
   the operator's ``solve`` / ``solve_transpose``.  The **forward matvec**
   and its transpose were single-sourced earlier (step 4b) onto
   :func:`~orpheus.sn.sweep.psi_half_angle_seed.radial_characteristic_forward_residual`,
   so ``apply`` and the walk's seed rows already share one kernel.

   **H1-narrow.**  Only the ``A_BB`` *solve* legs were extracted; the
   ``A_AB`` seed → bulk coupling stays **fused** in the within-group
   :math:`M` (the DP-splitting ruling).  In the transpose the reversed
   ordinate loop accumulates the Morel–Montry thread cotangent onto a copy
   of the seed cotangent — *that augmentation is* the fused
   :math:`A_{AB}^{\mathsf T}` feed — and only then does one
   ``A_BB.solve_transpose`` call return the seed-source cotangent.

.. (vv-status rationale) Structural / representational identity: the
.. block-triangular normal form of the augmented (L+C) in the augmented
.. walk order.  The verifiable content is the triu==0 triangularity
.. certificate + the apply∘solve corner-defect=0-bit gate (§16.C), not a
.. flux/eigenvalue claim.
.. vv-status: sn-direct-seed-block-triangular documented

.. _sn-direct-seed-source-fold:

The starting-direction source fold — why ALL Legendre moments (R14)
-------------------------------------------------------------------

The direct solve needs the true within-group source **at the starting
direction**, :math:`\bar q_{1/2}(\mu = \pm 1)` — the value the anisotropic
source :math:`q(r,\mu)` takes at the closed ray, reconstructed from **all**
its Legendre moments (Hébert Eq. (3.432); the "R14 full fold",
:func:`~orpheus.numerics.spaces.radial_characteristic_space.fold_moments_to_radial_characteristic`):

.. math::

   \bar q_{1/2}(\mu = \pm 1)
   \;=\; \sum_{\ell} \frac{2\ell+1}{2}\, q_\ell\,(\pm 1)^\ell,
   \qquad
   q_\ell(r) \;=\; \sum_n w_n\,P_\ell(\mu_n)\,q_n(r),

the exact 1-D addition-theorem weight :math:`(2\ell+1)/2` with
:math:`P_\ell(\pm 1) = (\pm 1)^\ell` — this is Eq. :eq:`hebert-3-432-source`
kept at **full order**, not collapsed to :math:`L = 0`.  The single q½
fold factory
:meth:`RadialCharacteristicField.source_from_angular <orpheus.transport.radial_characteristic_field.RadialCharacteristicField.source_from_angular>`
folds every moment the level resolves; the solver cold-start, the
fixed-source right-hand side, and the within-group joint march (the
System-B q½ source fed to the resolvent grid) all route through
it (Pattern 2 — the :math:`P_1(-1)` sign is spelled **once**).

**The full fold is load-bearing, and this is the subtle part.**  It is
tempting to fold only :math:`\ell = 0` (the apply matvec's isotropic
scattering reach is P0).  That is *wrong for the source*, because
**streaming manufactures angular structure the flux does not have**.
Take an isotropic trial flux :math:`\psi = A(r)` (no :math:`\mu`
dependence).  Applying the spherical streaming–collision operator
(:math:`\Omega\cdot\nabla + \sigma_t`, with
:math:`\Omega\cdot\nabla\psi = \mu\,\partial_r\psi + \tfrac{1-\mu^2}{r}\partial_\mu\psi`
and :math:`\partial_\mu\psi = 0`) gives a source that is **linear in**
:math:`\mu`:

.. math::
   :label: sn-direct-seed-anisotropic-source

   q(r,\mu) \;=\; \mu\,A'(r) \;+\; \sigma_t(r)\,A(r),
   \qquad
   q(r,\mu = -1) \;=\; \sigma_t A - A'.

.. (vv-status rationale) Literature-grounded derivation identity: the
.. spherical streaming-collision operator applied to an isotropic trial
.. flux A(r).  Not a solver claim — it is the analytical basis for the
.. full-fold requirement, whose verifiable content is the fold's ℓ=0
.. collapse to ½q₀ bit-identity (isotropic paths unchanged) + the
.. anisotropic-MMS O(h²) convergence gate.
.. vv-status: sn-direct-seed-anisotropic-source documented

The value at :math:`\mu = -1` carries the :math:`-A'` term — and that
term lives **entirely in the** :math:`\ell = 1` **moment**.  Working the
fold explicitly on a Gauss–Legendre level
(:math:`\sum_n w_n\mu_n = 0`, :math:`\sum_n w_n = 2`,
:math:`\sum_n w_n\mu_n^2 = 2/3`):

.. math::

   q_0 = 2\sigma_t A, \quad q_1 = \tfrac{2}{3}A',
   \qquad
   \underbrace{\tfrac12 q_0 P_0(-1)}_{\sigma_t A}
   + \underbrace{\tfrac32 q_1 P_1(-1)}_{-A'}
   \;=\; \sigma_t A - A' \;\checkmark

An :math:`\ell = 0`-only fold keeps only the :math:`\tfrac12 q_0 = \sigma_t A`
piece and **drops** the :math:`-A'`.  That dropped term is exactly what
**floored the anisotropic curvilinear MMS**: with the ℓ=0-only seed the
manufactured-solution error refused to converge; with the full fold the
sphere MMS converges :math:`\mathcal{O}(h^2)`-to-exact.

**For an isotropic source the full fold is a no-op.**  The higher moments
vanish (:math:`q_\ell = 0` for :math:`\ell \ge 1`), the fold collapses to
:math:`\tfrac12 q_0` **bit-exactly**, and the eigenvalue and
isotropic-fixed-source paths are **unchanged** by route (a).  The full
fold only "activates" when a >linear-in-:math:`\mu` source is present
(the companion anisotropic-MMS gate); in current production runs — all
isotropic sources — :math:`\ell \ge 1` is manufactured-before-needed and
identically zero (the isotropic-snapshot-blindness discipline: the
machinery is correct for the case the snapshots cannot exercise).

.. _sn-direct-seed-r12a:

Which levels carry a ψ½ block — the R12a predicate
--------------------------------------------------

A :math:`\mu`-level carries an **independent** starting-direction state
block **iff** the M-M half-angle recurrence genuinely *consumes* it.
Since Q5.4 (campaign ruling T26) the predicate is posed on **two named
structural facts** about the level's march-start edge
(:class:`~orpheus.sn.angular.closure.MarchStart`, produced by
:func:`~orpheus.sn.angular.closure.march_start_structure_per_level`,
read by
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.radial_characteristic_levels`)
— each a bit-exact identity on the level's own realization, never a
derived float:

.. math::
   :label: sn-direct-seed-r12a-predicate

   \text{level } p \text{ carries a ψ½ block}
   \quad\Longleftrightarrow\quad
   \neg\,\underbrace{\bigl(\eta_0 \text{ on the start edge}\bigr)}_{
   \texttt{on\_edge\_node}:\ \xi_0 = 0}
   \;\wedge\;
   \neg\,\underbrace{\bigl(\eta_0 = \eta_1\bigr)}_{
   \texttt{degenerate}:\ \text{double-cover tie}}

.. (vv-status rationale) Structural predicate: the two bit-exact facts keying
   which μ-levels carry an independent starting-direction block. Verified by
   ``tests/sn/sweep/test_march_start_structure.py`` — per-family
   classification over ten configurations (NODE_ALIGNED even/odd, STAGGERED
   full, level-symmetric, sphere-GL, and both σ_y-folded variants) carrying
   this label's marker, plus the bit-exact theorem gate demoting the former
   τ_raw trichotomy to a consequence (0 / 1 / strict-interior with NO
   epsilon). The terminal consequence (route (a) is a genuine single-pass
   exact inverse: sphere cold-start residual → 2.5×10⁻¹⁶, cylinder
   seed-sensitivity 0.0 bit) is exercised by
   ``tests/sn/sweep/curvilinear/test_282_direct_seed_fixed_point.py``
   and the ``@pytest.mark.foundation`` ``test_radial_characteristic_metric.py``
   suite.
.. vv-status: sn-direct-seed-r12a-predicate documented

.. note:: **Notation, 2026-08-11.**  The :math:`\tau_{\rm raw}` spelling
   throughout this subsection is **historical**: Q5.6.4 retired the
   :math:`[\tfrac12, 1]` absorber and with it the raw-vs-clamped
   distinction, so there is now one :math:`\tau`
   (:eq:`morel-montry-closure`) produced by one geometry-free body.  Read
   every :math:`\tau_{\rm raw}` below as that :math:`\tau`; the numbers
   quoted for the *chord* partition are flagged where they moved.

The two conjuncts are DISTINCT degeneracies — until Q5.4 one float
(the first-ordinate Morel–Montry weight, then produced by a separate
``morel_montry_tau_raw_per_level``, retired Q5.6.4 into
:func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`)
conflated them as the interval test :math:`\tau_{{\rm raw},0} \in (0,1)`
exclusive, deciding a structural question on derived float arithmetic
(with an FP-noise guard whose need was itself a symptom). The
trichotomy survives as a bit-exact gated **theorem** about the closure's
edge arithmetic — ``on_edge_node`` :math:`\Rightarrow \tau_{{\rm raw},0}
= 0` exactly, ``degenerate`` :math:`\Rightarrow \tau_{{\rm raw},0} = 1`
exactly, neither :math:`\Rightarrow` strict interior — on the production
quadratures:

.. list-table:: The R12a carrying-level trichotomy
   :header-rows: 1
   :widths: 18 30 52

   * - :math:`\tau_{{\rm raw},0}`
     - Rule
     - Why the seed is (not) independent state
   * - :math:`= 0`
     - cylinder **product** rules, NODE_ALIGNED **even**
       :math:`n_\varphi` (``on_edge_node``) — *refused at cylindrical*
       ``SNMesh`` *admission since Q5.6.3*
     - the starting direction coincides with the first ordinate
       (:math:`\eta_0 = \eta_{1/2} = -\sin\theta` bit-exactly, the #229
       clamp fact) — the seed is a rank-duplicate of :math:`\psi_0`.
       **No block.**
   * - :math:`= 1`
     - cylinder **level-symmetric** rules; product rules at **odd**
       :math:`n_\varphi` and **full STAGGERED** rules (``degenerate``)
       — *refused at cylindrical* ``SNMesh`` *admission since Q5.6.3*
     - duplicate-:math:`\eta` nodes collapse the midpoint edge onto
       :math:`\eta_0` — hemisphere partners on level-symmetric rules,
       the mirror pair straddling :math:`\varphi = \pi` on odd/staggered
       products (bit-exact by roots-of-unity conjugacy, Q5.E/E3) — so
       the seed's only consumption path, the recurrence weight
       :math:`(1-\tau_0)`, vanishes.  Dead state.  **No block.** (This
       is why the measured cylinder-LS seed sensitivity is :math:`0.0`
       bit.)
   * - :math:`\in (0,1)`
     - sphere **Gauss–Legendre**; every level of a
       :math:`\sigma_y`-**folded** product rule (neither fact)
     - the recurrence consumes the seed with a genuine weight
       (sphere-GL :math:`\tau_{{\rm raw},0} \approx 0.39\text{–}0.42`;
       folded staggered :math:`\approx 0.26` at :math:`n_\varphi = 8`,
       :math:`\to \tfrac14` from inside — `[M]` 2026-08-11 on the ω
       partition; the retired chord partition read
       :math:`\approx 0.22 \to \tfrac15`).  **Carries.**

**Since Q5.6.3 (``1689faf4``) the predicate is not only a classifier —
it is the cylindrical admission law.**  ``SNMesh`` construction on a
cylindrical mesh calls
:func:`~orpheus.sn.angular.closure.assert_carrying_quadrature`
(offender positions from
:func:`~orpheus.sn.angular.closure.non_carrying_levels`),
which raises on the first non-carrying level, naming exactly the facts
true on it and the remedy
(:meth:`Quadrature.folded_product
<orpheus.numerics.quadrature.Quadrature.folded_product>`).  The
decision reads the **structure** (the two ``MarchStart`` facts on the
rule's own realization), never a provenance tag — a hand-built
σ_y-quotient with the right arrays admits; a full-circle rule refuses
no matter how it was made.  So on production meshes **every admitted
level carries the block**: the sphere's GL levels and every level of a
folded cylinder rule alike ride route (a)'s forward substitution with
a genuine independent seed.  The first two trichotomy rows survive as
quadrature-level classifications (and as the admission's refusal
messages), not as constructible meshes; the 2-point angular-edge
extrapolation
(:meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.edge_extrapolated_seed`)
that non-carrying cylinder levels used to inline is no longer reachable
through any ``SNMesh``.  R12a **refines** the
earlier R12 letter ("μ_start ∉ the level's μ-nodes"), whose claimed
equivalence to :math:`\tau_{\rm raw} \ne 0` is empirically **false** on
level-symmetric cylinder rules (μ_start ∉ nodes there, yet
:math:`\tau_{\rm raw} = 1` — dead).  The clamp
:math:`0 \mapsto \tfrac12` erased exactly the 0-vs-(0,1) distinction the
predicate needs, which is why an unabsorbed producer was first-class —
and since Q5.6.4 there is only one producer, unabsorbed by construction.

.. _sn-direct-seed-circle-vs-interval:

Who pays for the pole, and why — circle vs interval
-----------------------------------------------------

The R12a trichotomy raises a deeper question: *why* does a level need an
independent seed at all, and what decides which rules do?  The answer is
neither "sphere vs cylinder" nor "curvilinear vs Cartesian" — it is the
**topology of the redistribution axis**, and it is the single most
clarifying fact about the whole ψ½ apparatus.

.. note:: **⛔ This section was titled "Why the sphere pays for the pole
   and the cylinder does not" and opened "why does the sphere carry an
   independent seed while the cylinder never does?" until 2026-08-13.
   Both were false at the time of writing** — refuted by the section's own
   closing paragraph and by the Q5.6 fold two screens earlier.  A
   :math:`\sigma_y`-**folded** product rule — the *only* cylinder family a
   cylindrical :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` admits since
   Q5.6.3 — carries an independent seed on **every** level, because folding
   turns the azimuthal circle into an arc, i.e. an interval, and puts the
   cylinder in exactly the sphere's position.  What is true, and what the
   section actually argues, is that a **full-circle** cylinder does not pay:
   the payment is owed by an *interval* axis, and the fold is the moment the
   cylinder chose to start paying.  The framing is kept as history because
   the full-circle rule was the existence proof that "seed = a bulk
   edge-ordinate" can work, and that is why the trade the fold makes is
   worth stating at all.

Two orthogonal questions must be kept apart:

#. *Does the geometry have an angular-redistribution term*
   :math:`\tfrac{1-\mu^2}{r}\,\partial_\mu`? — this is
   **curvilinear vs Cartesian** (sphere **and** cylinder: yes;
   Cartesian: no).
#. *Does the angular sweep need a separate, off-node starting DOF?* — this
   is the march-start predicate (:ref:`sn-direct-seed-r12a`), which is
   **quadrature-structural, not geometric**.

ψ½ answers question 2, and the deciding fact is what the redistribution
axis *is*.

**Cylinder — the redistribution axis is a circle.**  At a fixed polar
cosine :math:`\mu_z = \cos\theta`, the cylinder redistributes across the
**azimuthal angle** :math:`\varphi`, which lives on a **circle**
:math:`[0, 2\pi)` — a *periodic* domain.  The full-circle parent rule
(:func:`~orpheus.numerics.quadrature.rules_product.product_mu_phi`;
since Q5.6.3 the *parent* of the admitted cylinder family, no longer
itself admissible on a cylindrical ``SNMesh``) is
Gauss–Legendre in :math:`\mu_z` **×** the *periodic trapezoid* in
:math:`\varphi`
(:func:`~orpheus.numerics.quadrature.rules_circle.periodic_trapezoid`,
NODE_ALIGNED — nodes as roots of unity since Q5.E/E3).  The trapezoid
on a circle is **spectrally accurate** for smooth periodic integrands
(its error decays faster than any power of :math:`1/n_\varphi`) —
there is no accuracy penalty for the choice.  And crucially, for
**even** :math:`n_\varphi` the grid hits :math:`\varphi = \pi` exactly
(partition node :math:`k = n_\varphi/2`), where

.. math::

   \mu_x = \sin\theta\cos\pi = -\sin\theta, \qquad
   \mu_y = \sin\theta\sin\pi = 0,

i.e. the **most-inward radial direction** of that level.  The
starting-edge ordinate :math:`\eta_0 = -\sin\theta` therefore lands
*exactly on a quadrature node* — ``on_edge_node``,
:math:`\tau_{{\rm raw},0} = 0` — and the seed is a **bulk ordinate for
free, at no accuracy cost**.  This is the structural content of the
floor measured in #229: the cylinder's edge-inclusion is a property of
the *circle*, **not** of the (now retired) :math:`[\tfrac12, 1]`
Morel–Montry clamp — that absorption was a separate object, and the R12a
facts are read off the level's realization directly.

An odd azimuthal count misses :math:`\varphi = \pi` — but the cylinder
does **not** then carry a seed.  The mirror pair *straddling*
:math:`\pi` shares :math:`\eta` bit-exactly (roots-of-unity conjugacy,
:math:`\cos\varphi_k = \cos\varphi_{n_\varphi - k}`), so the level is
``degenerate`` instead: the seed's :math:`(1-\tau_0)` thread weight
vanishes and the state would be dead.  *(This corrects an earlier
version of this page, which claimed an odd count would carry — that
claim described the pre-E3 ``linspace``+cos realization, whose
5.6e-16 tie-breaking round-off flipped the float predicate; campaign
ruling T26.)*  The two parities fail through the two DIFFERENT facts —
a node ON the mirror plane at even :math:`n_\varphi`, a tie ACROSS it
at odd — which is the sharp form of the circle principle: **on a full
circle, the** :math:`\sigma_y` **mirror closes the march at every
parity; only the quotient — the folded arc (T22b) — opens a genuine
off-node start.**  A :math:`\sigma_y`-folded product rule carries on
every level (`[M]` 2026-08-11, folded staggered
:math:`\tau_{{\rm raw},0} \approx 0.26` at :math:`n_\varphi = 8`,
:math:`\to \tfrac14` from inside, strictly interior).

**Sphere — the redistribution axis is an interval.**  The sphere
redistributes across the **polar cosine** :math:`\mu \in [-1, 1]`, an
**interval** whose two endpoints :math:`\mu = \pm 1` *are* the physical
poles.  The optimal rule on an interval with a smooth integrand is
Gauss–Legendre — but Gauss–Legendre is an **open** rule: it places *no*
node at the endpoints.  So the sphere structurally *cannot* put an
ordinate on :math:`\mu = \pm 1`; the starting edge falls strictly between
nodes (:math:`\tau_{{\rm raw},0} \approx 0.39\text{–}0.42`), and a
**separate off-node seed DOF is unavoidable**.

.. list-table:: The redistribution axis decides the seed
   :header-rows: 1
   :widths: 16 26 24 17 17

   * - Geometry
     - Redistribution axis
     - Optimal rule
     - Edge-inclusive?
     - Seed?
   * - **Cylinder, full circle** (*the parent — refused at*
       ``SNMesh`` *admission since Q5.6.3*)
     - azimuth :math:`\varphi` — a **circle** (periodic)
     - equispaced (trapezoidal, spectral)
     - **yes** (even :math:`n_\varphi` hits :math:`\varphi=\pi`)
     - no — :math:`\tau_{\rm raw}=0` (or a dead :math:`\tau_{\rm raw}=1`
       tie at odd/staggered parity)
   * - **Cylinder, folded arc** (*the admitted production family,
       Q5.6.3*)
     - arc angle :math:`\omega` — an **interval** :math:`[0, \pi]`
       (the :math:`\sigma_y` quotient)
     - staggered midpoints (≡ Gauss–Chebyshev-1 in :math:`\cos\omega`)
     - **no** (midpoint rule, no endpoint node)
     - yes — :math:`\tau\in[\tfrac14,\tfrac34]`, every level
   * - **Sphere**
     - polar :math:`\mu` — an **interval** :math:`[-1,1]`
     - Gauss–Legendre (open)
     - **no** (open rule, no endpoint node)
     - yes — :math:`\tau_{\rm raw}\in(0,1)`

The principle in one line: **a periodic redistribution axis gives
edge-inclusion for free; an interval axis makes you pay for it with a
separate seed.**  The full-circle cylinder was the existence proof that
"seed = a bulk edge-ordinate" *works* — it worked there precisely
because the axis is a circle.  But the free edge-inclusion came bundled
with what the circle also forces: the singular set :math:`\Sigma` on
the mirror, the double-cover η-ties (#326), and the :math:`\tau = 0`
division block the :math:`[\tfrac12, 1]` absorption existed to hide.
The Q5.6 fold **deliberately renounces the free edge-inclusion**: the
:math:`\sigma_y` quotient turns the axis into an interval (the arc), so
the folded cylinder *joins the sphere* in paying one independent seed
per level — the price route (a) resolves exactly, with
:math:`\Sigma = \varnothing` and the reversal identity
:math:`\tau_m + \tau_{M-1-m} = 1` (to 64 ULP since the Q5.6.4 partition
fix; bit-exact on the retired chord partition, whose symmetric end-cell
stretch cancelled itself) as what the payment buys.  The sphere pays
because its axis has physical
endpoints and the best interior rule refuses to stand on them; the
folded cylinder pays because standing on the edge was never free — it
was the degeneracy.

.. _sn-direct-seed-lobatto-study:

Could the sphere put a node at the pole? — the Gauss–Lobatto study
------------------------------------------------------------------

The circle-vs-interval framing suggests an obvious question: could the
sphere *buy* the cylinder's free edge-inclusion by switching from
Gauss–Legendre to **Gauss–Lobatto** — a rule that *does* place nodes at
the interval endpoints :math:`\mu = \pm 1` (at the cost of exactness
:math:`2n-3` versus GL's :math:`2n-1`)?  A pole node would give
:math:`\tau_{{\rm raw},0} = 0`, making the seed a bulk ordinate exactly as
on the cylinder — dissolving the whole ψ½ block.  A dedicated empirical
study (scratch, uncommitted — see the note below) answered both halves of
the question.

**Affordable.**  At resolved angular order (:math:`N \ge 8`, and
:math:`N > L` for a :math:`P_L` scattering source) Gauss–Lobatto tracks
Gauss–Legendre at a bounded :math:`\sim 1.2\times` error penalty —
:math:`\sim 1.3\text{–}1.4\times` at S\ :sub:`8`, tightening to
:math:`\sim 1.2\times` at S\ :sub:`16` — i.e. **one to two extra
ordinates** to match GL.  The penalty is **not amplified by anisotropy**
(P0 through P5 all sit at :math:`\sim 1.2\text{–}1.3\times`) and is
**insensitive to the scattering ratio** :math:`c`.  The eigenvalue offset
is :math:`\sim 30\text{–}140` pcm at S\ :sub:`16`, and fine-:math:`N` GL
and GLob agree to :math:`< 6` pcm — the two rules converge to the **same**
:math:`N \to \infty` transport limit (the pole weighting is unbiased; the
straight-characteristic pole handling is redistribution-consistent, with a
per-ordinate flat-flux residual :math:`\sim 10^{-15}`).  Only the
under-resolved :math:`N \lesssim L` corner breaks, and there GL is
rank-deficient too.

**But not a drop-in.**  A pole node lands on the level's lower edge, so
the first-ordinate weight :math:`\tau_{{\rm raw},0} = 0` — and the
production Morel–Montry recurrence :eq:`pole-mm-recurrence` **divides its
first step by that weight**, so the recurrence is *singular*; separately,
the R12a ``on_edge_node`` fact fires (the march start IS an ordinate),
so the level classifies **non-carrying** — the same class the Q5.6.3
cylindrical admission refuses, now arising on the *sphere* side.
Adopting a pole-node quadrature is
therefore **not** a quadrature swap — it *requires* the
seed→bulk-ordinate restructure the pre-fold full-circle cylinder used
(make the pole
node the seed, straight-characteristic-solved, and start the recurrence
*from* it), and it must be reconciled with the admission machinery that
now encodes "non-carrying ⟹ refuse" for cylinders — the interaction is
tracked as `Issue #338
<https://github.com/deOliveira-R/ORPHEUS/issues/338>`_.

**Ruling: affordable but architecturally declined.**  The fold-in was
**not** adopted, and the reason is architectural, not numerical.  The bulk
is **cell-centred**; a pole ordinate would make it a *mixed* field — an
inert, zero-through-flux, straight-characteristic-solved,
redistribution-special passenger that *every* bulk consumer
(homogenization, condensation, moment extraction, every
``for ordinate in bulk`` loop) would have to know about and skip.  That is
the Cardinal-Rule-2 smell of two concepts in one type forcing a demux
downstream; the zero weight prevents *numerical* corruption but not the
*conceptual* pollution.  The value of the study is precisely that it makes
keeping ψ½ a **separate** object a *chosen* architecture — the pole seed
is kept out of the bulk **because the bulk stays clean**, not because a
pole-node scheme is infeasible.  The clean-bulk / ``FaceField``
architecture this decides is set out on the loss-operator page
(:ref:`loss-rep-facefield-codim1`).

.. note::

   The Gauss–Lobatto study is a set of scratch diagnostics
   (``scratch/experimental/glob_sphere_study/`` and
   ``derivations/diagnostics/diag_glob_0{1..5}_*.py`` — 33 green
   diagnostics covering moment integration, per-ordinate consistency,
   end-to-end penalty, the :math:`\tau_0 = 0` recurrence break, and the
   :math:`k_\infty` anchor).  They are **uncommitted** and are promotion
   targets *only if* a pole-node scheme is ever adopted; do not promote
   them otherwise.  The durable synthesis is
   ``.claude/plans/archive/facefield_codim1_design.md`` §3.5.

.. _sn-direct-seed-strategy-zoo:

What was tried and failed — the seed-strategy zoo (retired)
-----------------------------------------------------------

Three swappable ``PsiHalfAngleSeed`` strategies preceded route (a).  All
three are **retired** (2026-07-04); the module
:mod:`~orpheus.sn.sweep.psi_half_angle_seed` shrank from 851 lines to
161 — one engine.  The history matters because it prevents a future
session from re-attempting a known-dead seed:

.. list-table:: The retired seed strategies and why each fell short
   :header-rows: 1
   :widths: 26 40 34

   * - Retired strategy (literal — class deleted)
     - What it did
     - Why it failed
   * - ``ZeroSeed``
     - hardcoded :math:`\psi_{1/2} = 0`
     - the pre-ERR-026 term-initialisation bug (vv-principles failure
       Mode 3, a missing term); wrong off flat flux.
   * - ``CarlsonInwardSweep``
     - this module's Hébert march driven by the **proxy** source
       :math:`\bar Q = \sigma_t\phi_0/\!\sum w`
     - exact only at the flat-flux equilibrium; :math:`\mathcal{O}(1)`
       wrong off equilibrium — **floored the curvilinear MMS at ≈ 0.04**
       L2 independent of mesh (ERR-058b, Issue #195).
   * - ``AngularEdgeExtrapolation``
     - the operator-consistent 2-point angular extrapolation of the
       **iterate**
     - fixed the forward *apply* (#195) but left the *solve* seeded from
       the **previous iterate** — the #282 walk-order **back edge**
       (sphere cold residual :math:`5.18\times10^5`, seed sensitivity
       :math:`4.57\times10^{-2}`).

Route (a) retired the whole family.  The **arithmetic survives** in two
places, both correct on their own:

* :func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
  — the Hébert :eq:`hebert-3-434`–:eq:`hebert-3-435` recurrence, now
  driven by the **true** q½ source (:ref:`sn-direct-seed-source-fold`) instead of
  the falsified proxy, and used as the SOLVE engine (not a strategy);
* :meth:`~orpheus.sn.angular.closure.MorelMontryAngularSweep.edge_extrapolated_seed`
  — the 2-point angular-edge extrapolation, inlined **verbatim** for the
  non-carrying cylinder levels (where the R12a trichotomy makes it
  bit-identical to the retired default: :math:`t = 0` exact on product
  rules, dead seed weight on level-symmetric rules).  Since Q5.6.3 no
  ``SNMesh``-admitted **cylinder** has a non-carrying level, so this inline
  is unreachable *on that chart*.  It is **not** dead code: the spherical
  arm calls no admission gate, so a :math:`\mu = -1`-noded (Gauss–Lobatto)
  sphere rule builds a production ``SNMesh`` and reaches it — `[M]`
  2026-08-26, at 6 of 11 orders, over 75 reachable non-carrying levels
  (the census is recorded in ``_edge_seed_stencil``'s reachability note;
  the sphere-side interaction is `Issue #338
  <https://github.com/deOliveira-R/ORPHEUS/issues/338>`_ and the missing
  empty-seed gate witness `Issue #415
  <https://github.com/deOliveira-R/ORPHEUS/issues/415>`_).  Retirement is
  therefore **off the table** — this is the only seed path such a rule has.

.. _sn-direct-seed-numerical-evidence:

Numerical evidence — the lag death
----------------------------------

The acceptance gates live in
:file:`tests/sn/sweep/curvilinear/test_282_direct_seed_fixed_point.py`
(the §16.C fixed-point classifiers); every gate measures the **full
coupled state** (System A's bulk ⊕ trace *and* System B's ψ½), because a
bulk-only norm would be blind to any seed error (a Mode-12
functional-invariance point, independent of the metric closure in the
gotcha below).

.. list-table:: #282 route-(a) acceptance evidence
   :header-rows: 1
   :widths: 16 44 40

   * - Gate
     - Measurement
     - Before → after (route a)
   * - **C(i)** cold residual
     - :math:`\lVert A\cdot\mathrm{solve}(b)-b\rVert_\infty/\lVert b\rVert_\infty`
       on a cold start (the keystone: the solve is a single-pass exact
       inverse)
     - sphere :math:`5.18\times10^{5} \to 2.5\times10^{-16}`; slab & cyl
       already :math:`< 10^{-11}` (must **stay**)
   * - **C(ii)** seed-insensitivity
     - two random ``initial_guess`` seeds → bitwise-identical sphere
       solve (the lag signature)
     - :math:`\Delta = 4.57\times10^{-2} \to 0` **bitwise**
   * - **C(ii)** Probe-6
     - :math:`\psi_0` arbitrary, :math:`b = A\psi_0`, cold
       :math:`\mathrm{solve}(b)` recovers :math:`\psi_0`
     - pre-fix only the **warm** sphere solve recovered it; now the cold
       solve does (``rtol`` :math:`10^{-11}`)
   * - **C(iii)** coarse physicality
     - S\ :sub:`8` 16-cell sphere fixed source, finite **and**
       non-negative on **both** inner drivers
     - SI → ``NaN`` / Krylov → negative flux :math:`\to` finite + positive
       on both
   * - **C(iv)** pure absorber
     - :math:`c = 0` sphere (no scattering outer loop to mask the lag) —
       the cold solve **is** the answer
     - ``NaN`` :math:`\to` single-pass exact inverse
       (:math:`< 10^{-11}`, finite, positive)

Three teeth pin that the evidence is not vacuous: **Mode 11** —
a **class-level** wrap-sentinel confirms the sphere cold solve *executes*
:meth:`RadialCharacteristicOperator.solve <orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve>`
while the slab does **not** (no carrying levels), the transpose analogue
wraps :meth:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator.solve_transpose`,
and a source-scan tripwire
(``test_4e_unweave_walk_source_has_no_carlson_reference``) pins that the
walk holds **zero** ``carlson_inward_sweep_*`` references — the 4e-e2
un-weave (the marches live only behind the operator, so the sentinel wraps
the very method the walk routes through); **Mode 10** — zeroing
the q½ source block **moves** the sphere solve (the carrier is not inert);
**Mode 12** — with the :math:`V_{\rm cell}` state metric the G-reciprocity
gate now *catches* a seed-row flip (the closure, ERR-067; gotcha below).

The far endpoint — what the two-ended march is measured by
------------------------------------------------------------

Every gate in the table above measures the **near** end: whether the seed
is right, whether the solve that produces it is a single pass, whether the
carrier is inert.  None of them looks at the far end, and until 2026-08-13
nothing did — the second endpoint condition
:eq:`sn-angular-endpoint-defect-eq` was computed on every curvilinear solve
since route (a) landed and **compared by nothing**
(:ref:`sn-angular-endpoint-defect`).

Its gates live in
:file:`tests/sn/sweep/curvilinear/test_angular_endpoint_defect.py` — six
``foundation`` rows on a **heterogeneous, two-group, vacuum** curvilinear
problem, on both arms.  The fixture discipline is structural, not
decoration:

* a **flat-in-angle** fixture is *provably* blind, because the recurrence's
  flat fixed point makes the two endpoints coincide exactly.  That is the
  module's positive control (``vv-principles`` #11) *and* the reason no flat
  gate anywhere in the tree could ever have caught the omission;
* a **1-group / homogeneous** fixture nulls the redistribution terms the
  march is made of (``vv-principles`` #3/#4);
* a **reflective** outer face is NOT blind — `[M]` ``max|D|`` under
  reflection is comparable to vacuum.  What reflection removes is the
  *divergence* of :math:`L^\infty(D)` (Issue #360), not the defect.

What the rows pin, in the module's own words rather than repeated numbers
(``plan-authoring`` §9 — the gate re-measures its ladder, so it owns it):
that ``trailing_face`` is the slice
:attr:`upstream_per_ordinate <orpheus.sn.angular.closure._MMHalfGrid.upstream_per_ordinate>`
drops; that :math:`D` is that slice minus the marched outward leg; that
:math:`D` is exactly zero when the endpoints agree; that :math:`D` falls at
**~2nd order in** :math:`n_\varphi` on the cylinder; that on the sphere its
apparent non-convergence is a **spatial** floor whose turnover moves out
with ``nx``; and that it responds to the angular cell *partition*, not
merely to the product :math:`\prod_m (1-\tau_m)/\tau_m` that the reversal
identity pins.

.. note:: **Two of the six survive two of the three mutations, and that is
   reported rather than hidden** (``vv-principles`` #19/#20 — a row that
   cannot see a property must not be counted as covering it).  The
   subtraction row reads ``trailing_face`` on *both* sides, so it is a
   tautology with respect to *which slice* the property returns and can only
   ever test the subtraction; the slice itself is covered by the identity
   row.  The flat-flux zero row survives a face-index error and a
   seed-vs-far-endpoint swap because a flat flux collapses every face to the
   same constant — which is the same blindness that row exists to document,
   now measured rather than argued.

.. warning:: **The sphere's :math:`D` looked like it did not converge in
   angle.  It does — twice-measured and twice-wrong before that was
   established.**  At a coarse spatial grid :math:`D` falls from
   :math:`N = 4` to :math:`N = 8` and then *rises*; refine the mesh and the
   same ladder falls monotonically, the turnover having moved out past
   :math:`N = 32`.  A turnover point that moves out with ``nx`` is the
   textbook signature of a **spatially-set floor**, not of an angular
   inconsistency — and reading the coarse ladder alone gives exactly the
   opposite conclusion.  Any future claim that the curvilinear angular march
   fails to converge in :math:`N` must first vary ``nx``.

The eigenvalue re-pose — an N-sweep, not h→0
--------------------------------------------

Route (a) changed the ψ½ **angular closure**, so it re-posed the sphere
:math:`k`-eigenvalue — by :math:`\sim 4.66\times10^{-4}` at :math:`n = 40`
cells.  Judging whether such a re-pose is **principled** requires the
right discriminator, and this is a genuinely teachable subtlety.

**The** :math:`h \to 0` **continuum test is invalid here.**  A seed *is*
an angular closure: it changes the :math:`\mathcal{O}(N)` **angular
truncation** of the discrete operator.  So the mesh-refinement
(:math:`h \to 0`) limits *at fixed angular order* :math:`N` genuinely
differ between the old and new seed — that difference is not an error, it
is two consistent-but-distinct closures converging to two distinct fixed
angular truncations.  Refining :math:`h` cannot tell them apart.

**The correct discriminator is an angular-order** :math:`N`-**sweep at
fixed mesh** (gate
``test_heterogeneous_1g_angular_order_consistency``).  The retired
edge-extrapolation and the new direct Carlson seed **both converge to the
same transport eigenvalue as** :math:`N \to \infty`: they differ by
:math:`\sim 1.7\times10^{-3}` at Gauss–Legendre order 8 but agree to
:math:`\sim 10^{-6}` by GL32.  A seed that did **not** converge in
:math:`N` would be an *inconsistent* closure (a genuine regression); both
do, so the re-pose is principled.

.. warning::

   **Route (a) is NOT "more accurate" — it is justified structurally.**
   At the *low* angular orders the tests use (GL8), the retired seed is
   actually **closer** to the :math:`N`-limit.  Do **not** frame the
   re-baseline as an accuracy improvement.  Route (a) is justified by
   **structure** — the honest single-pass direct inverse (cold residual
   :math:`5.18\times10^5 \to 2.5\times10^{-16}`) required by the DSA (#2),
   curvilinear-Krylov (#200), and unified-walk (#280) programs — not by
   angular accuracy.

   And the **MMS is blind to the seed**.  Every curvilinear manufactured
   solution in this codebase is :math:`\le` linear-in-:math:`\mu`, which
   is exactly the seed's *exact* regime (:eq:`sn-direct-seed-anisotropic-source`
   is the boundary of what the seed can get wrong; vv-principles Mode 7).
   So the MMS-:math:`\mathcal{O}(h^2)` convergence does **not** certify
   the seed — only the :math:`N`-sweep gate does.  (Eigenvalue claims
   need a closed-form or semi-analytical reference regardless; MMS is a
   flux-shape / convergence-order pillar, never an eigenvalue one.)

.. _sn-direct-seed-gotchas:

Gotchas
-------

**The Krylov restart must be sized from the composite, not the bulk
(ERR-053 family).**  A carve that grows the Krylov composite (adds a
block) **must** resize the GMRES ``restart`` / ``n_dof`` from the
composite ``to_flat().size``, not the bulk :math:`N\cdot n_g\cdot n_x`.
Route (a)'s coupled System B (the ψ½ ray) pushed the coupled ``to_flat``
past the bulk count, so a bulk-sized ``restart`` re-truncated GMRES on the
trace **and** seed degrees of freedom — the restarted subspace could not
represent the coupled iterate and the sphere within-group inner **stalled**
(wrong
:math:`k` under an outer cap, :math:`\sim 868` s).  Fixed at **both**
solver Krylov drivers by sizing ``n_dof = initial_guess.to_flat().size``.
Distinct from #200 (the identity preconditioner); this is a pure
sizing bug.

**The product-cylinder solve consumed the iterate through the
edge-extrapolation stencil — that data flow had to stay bit-exact.**
(Historical since Q5.6.3, ``1689faf4``: the product cylinder is no longer
constructible.  The constraint is recorded because the *pattern* recurs
wherever a non-carrying level survives — today only on the sphere side,
under a :math:`\mu = -1`-noded rule.)  On a non-carrying level the seed is
the 2-point extrapolation of the *iterate* (:math:`t = 0` on product
rules: the stencil reads the first ordinate's iterate column).  That is a
*formal* lag, harmless at the fixed point, and the retirement of the
strategy zoo had to keep the non-carrying data flow **byte-identical** to
the pre-2.5d path — a diverging cylinder solve would have tripped the
§16.D cylinder-unmoved baseline.

**G-reciprocity catches a seed-row error (Mode 12 — CLOSED, ERR-067).**
Under the *retired* **ghost** :math:`G_{\rm sd} = 0` the seed block
carried zero metric weight, so its rows lay **inside** the
metric-weighted G-adjoint reciprocity functional's **invariance group**:
a sign flip on the seed rows left
:math:`\langle A\psi,\chi\rangle_G = \langle\psi, A^{\dagger}\chi\rangle_G`
**exactly** unchanged, at every tolerance, in every regime — a false
green (the classic Mode-12 instance).  The state metric
:math:`G_{\rm sd} = V_{\rm cell}` moves the seed rows **out** of that
invariance group.  With a nonzero ψ½ seed, a seed-row
(:math:`A_{\rm ss}`) sign flip on the forward operator — but not on
:math:`A^{\dagger}`'s independently-coded reverse mode — perturbs
:math:`\langle A\psi,\chi\rangle_G` by
:math:`\sum V_{\rm cell}\,(A_{\rm ss}\psi_{\rm seed})\,\chi_{\rm seed}`
that the unflipped adjoint cannot match, so **reciprocity now REDs** (the
Mode-12-closure gate
``test_mode12_g_reciprocity_catches_a_seed_row_flip``).  The gate carries
**both legs**: a control leg — the *unmutated* nonzero-seed reciprocity
holds :math:`< 10^{-12}`, proving the baseline is the honest
:math:`V_{\rm cell}` adjoint, so the mutated RED is attributable to the
flip (a reverted ghost :math:`G_{\rm sd} = 0` would *also* leave a defect
:math:`\approx 0.107`, a broken baseline mimicking "caught") — and the
mutated leg (:math:`> 10^{-6}`).  This metric-level catch **complements**,
and does not replace, the **object-level** pins that fix the seed
*coefficients*: the C(i) cold residual (forward) and the 2.5b Euclidean
:math:`M^{\mathsf T}` oracle (transpose).  The lesson stands, now
positively — a Mode-12 blindness closes either by gating the object OR by
repairing the functional's **metric** so the error class leaves its
invariance group; here the metric *was itself the bug*, so the correctness
fix and the Mode-12 closure are one and the same.
