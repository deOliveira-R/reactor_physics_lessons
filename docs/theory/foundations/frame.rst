.. _galerkin-projection:

=========================================
The Discrete Frame — projection machinery
=========================================

Every method in ORPHEUS that transitions a function between a **fine**
representation and a **coarse** one — discrete-ordinate :term:`angular flux` to
spherical-harmonic moments, fine-energy fluxes to broad-group cross
sections, regional flux to homogenised cross sections — does so via a
**(reconstruction, analysis)** pair of linear operators :math:`(R, M)`.
In harmonic-analysis language this pair is exactly the two operational
**faces of a discrete frame**: the analysis operator :math:`M = T`
(sampled values → coefficients) and the reconstruction :math:`R`
(coefficients → values, the canonical-dual synthesis). This page is the
canonical home for that pair, for the discrete-**frame** abstraction that
realises it (:class:`~orpheus.numerics.frame.FrameBase` and its
discipline subclasses), and for the **discipline** — Galerkin vs
Petrov-Galerkin, test space equal to vs different from trial space — that
distinguishes its variants.

The page is organised **general case first**. The Petrov-Galerkin frame
(test :math:`\ne` trial) is the general object; the Galerkin frame (test
:math:`=` trial) is its symmetric specialisation — a structure the code
mirrors exactly, since :class:`~orpheus.numerics.frame.GalerkinFrame`
**is-a** :class:`~orpheus.numerics.frame.PetrovGalerkinFrame`
(:class:`~orpheus.numerics.frame.FrameBase` →
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` →
:class:`~orpheus.numerics.frame.GalerkinFrame`). After the abstract frame
theory come the two disciplines and their concrete consumers — spatial
homogenisation and energy condensation (Petrov-Galerkin), spherical-
harmonic scattering projection (Galerkin) — and then the **advanced**
material: eigenbasis ownership, the cross-method consumer catalog, and
the adjoint-weighted seam. Binding a
:class:`~orpheus.numerics.basis.Basis` to a
:class:`~orpheus.numerics.measure.DiscreteMeasure` through a single
frame puts one mechanism in front of every consumer, instead of a
separate projection / reconstruction operator class per method.

.. contents::
   :local:
   :depth: 2

.. note:: **What changed (2026-06-24, Issue #268).** Two earlier
   framings on this page were **reversed** by the P1 discipline-type
   carve (``refactor/operator-inverse-algebra``), and the page now
   reflects the shipped architecture:

   1. **Discipline is a TYPE, not a property and not an operator
      marker.** The earlier draft alternately (a) carried the
      discipline as marker ABCs ``GalerkinProjection`` /
      ``PetrovGalerkinProjection`` *on the operator role* and (b)
      proposed collapsing it to a derived *property* of the frame
      (``measure == basis.canonical_measure``). Both are retired. The
      discipline is now a genuine **kind of frame**, carried by the
      frame TYPE:
      :class:`~orpheus.numerics.frame.FrameBase` →
      :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` →
      :class:`~orpheus.numerics.frame.GalerkinFrame`. The
      :mod:`orpheus.numerics.projection` module keeps only the two
      abstract operator **roles**
      (:class:`~orpheus.numerics.projection.AnalysisOperator`,
      :class:`~orpheus.numerics.projection.ReconstructionOperator`),
      which the frame faces subclass.

   2. **Homogenisation and condensation are Petrov-Galerkin.** An
      intermediate draft argued they were "Galerkin in the natural
      :math:`L^2(\phi V)` (resp. spectrum) metric". That reading folds
      the solution (the flux :math:`\phi`) into the *metric* — it is
      legitimate **only** for forward-flux, reaction-rate-only
      reduction, and breaks under the eigenvalue-consistent
      (adjoint-weighted) homogenisation reactor physics requires
      (where the test weight is the adjoint :math:`\varphi^*`, not the
      forward flux). The solution-weighting therefore lives on the
      **test side = a distinct test basis = the frame type**, never as
      a weight smuggled onto the measure: **the measure carries the
      axis and the fixed** :math:`L^2` **metric, never the
      discipline**.

   The homogenization derivation — the headline Petrov-Galerkin
   consumer of this hierarchy — lives in
   :ref:`sn-homogenization-petrov-galerkin-frame` (below); it was
   rewritten to this same
   Petrov-Galerkin framing under Issue #268 (the earlier
   ":math:`L^2(\phi V)`-Galerkin" reading is retired there, with the
   forward-flux metric-fold shown to be the Galerkin *degenerate* of the
   eigenvalue-consistent adjoint-weighted case).

.. note:: **What shipped since (P3 / P5 / P7).** The
   :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` base was empty of
   concrete consumers at the P1 carve; the **forward** (reaction-rate,
   :math:`\varphi^* = \varphi`) homogenisation (P3) and energy
   condensation (P5) have since shipped as concrete instances
   (:meth:`Solution.homogenize
   <orpheus.sn.solution.Solution.homogenize>`,
   :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`).
   This page (P7) is the capstone that ties the discipline-type
   hierarchy, the composed-operator verbs
   (:ref:`frame-composed-verbs`), the three-way Gram-structure gate
   (:ref:`frame-least-squares-discipline`), and the eigenbasis-ownership
   ruling (:ref:`frame-eigenbasis-ownership`) into one narrative, and
   reconciles its consumer table with the shipped frames. The one
   remaining frame discipline that is **theory-documented but not built**
   is the **least-squares** frame over a dense cross-Gram
   (:ref:`frame-least-squares-discipline`); the eigenvalue-consistent
   (adjoint-weighted) projection now **ships** (P6, #281;
   :ref:`frame-adjoint-weighted-seam`).

Key Facts
=========

- A **frame** binds a :class:`~orpheus.numerics.basis.Basis` (the
  synthesis / trial side — the functions :math:`\{e_k\}` and their
  convention) to a :class:`~orpheus.numerics.measure.DiscreteMeasure`
  (the domain — the sample points and the fixed quadrature-weight
  :math:`L^2` metric). The
  :class:`~orpheus.numerics.frame.FrameBase` so formed emits two
  operational faces:

  * the **analysis** face :math:`M = T` — sampled values →
    coefficients (``frame.analysis``), measured against the **test**
    basis;
  * the **reconstruction** face :math:`R` — coefficients → values,
    the canonical-dual synthesis (``frame.reconstruction``), purely
    trial-side.

  Together :math:`(R, M)` define a Galerkin-style discretisation of
  any :math:`A : V \to V` as :math:`A_h = M A R : W \to W`
  (Brenner & Scott 2008, §3.4).

- The **discipline** — whether the *test* functions equal the *trial*
  functions — is carried by the frame **type**, a genuine
  Liskov-correct hierarchy (Issue #268):

  .. code-block:: text

     FrameBase                 abstract; the discipline-FREE mechanics
     │                         (table, spaces, reconstruction face, the
     │                         analysis-face wiring — none depend on the
     │                         test side)
     └─ PetrovGalerkinFrame    explicit TEST basis (test ≠ trial); the
        │                      analysis measures against test functions
        │                      χ_k that need NOT equal the trial φ_k, so
        │                      M* ≠ R (an oblique dual)
        └─ GalerkinFrame       test IS trial — STRENGTHENS the promise to
                               a CANONICAL dual: M* = S₀∘G⁻¹ re-synthesises
                               on the TRIAL basis, so it is R rescaled per
                               mode by 1/(d_ℓ G_ℓ) — one scalar 1/W for the
                               SH frame. The angular spherical-harmonic
                               projection is the canonical pure-Galerkin
                               frame.

  :class:`~orpheus.numerics.frame.GalerkinFrame` *is-a*
  :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` with
  ``test is trial``: it strengthens (never weakens) the base promise.

  ⚠ **The Galerkin promise is** :math:`M^* \propto R`\ **, never**
  :math:`M^* = R`. Being a canonical dual fixes *which basis* the
  adjoint re-synthesises on (the trial one); it does not remove the
  metric. `[M]` under the no-prefactor SH convention the constant is
  :math:`1/W` (:eq:`frame-square-closure-sh`), and asserting the bare
  :math:`M^* = R` is precisely the ERR-039 / ERR-051 family. The
  indicator frame is the standing counter-example where the factor is
  not even one scalar
  (:ref:`frame-square-closure-section`).

- **The frame owns the codomain METRIC, and it is the INVERSE
  discrete Gram** (:ref:`frame-parseval-metric`). For a band-limited
  field :math:`\psi = S_0 c`, analysis returns
  :math:`\varphi = M\psi = Gc` **identically**
  (:eq:`frame-analysis-is-the-gram`, pure algebra — no exactness
  hypothesis), i.e. the *covariant* moments, not the coefficients. So
  the inner product under which analysis is an isometry onto its image
  is :math:`G^{-1}`, and
  :attr:`FrameBase.basis_space
  <orpheus.numerics.frame.FrameBase.basis_space>` dresses the basis's
  space with it. Consequences: (i) the metric is a property of the
  **pairing**, never a basis constant — the same SH basis on a slab
  Gauss–Legendre measure has :math:`W = 2`, not :math:`4\pi`;
  (ii) with it, each face's ``.H`` is the physical Hilbert adjoint, on
  **every** frame; (iii) when the measured Gram is NOT diagonal the
  metric is the matrix pseudo-inverse :math:`G^{+}`, installed as a
  :class:`~orpheus.numerics.metric.DenseMetric` since campaign 1 P7
  (2026-08-30), and Parseval is then a *theorem* for any Gram, singular
  or not (:ref:`frame-parseval-dense-arm`). ⛔ Until 2026-08-23 the
  frame exposed the basis's **continuum** Gram :math:`g_C` instead — the
  wrong side, off by :math:`(4\pi/(2\ell+1))^2` per :math:`\ell`; and
  until P7 a ``DENSE`` verdict *refused* the dressing altogether, so a
  face's ``.H`` there was the stored-metric sandwich rather than the
  physical adjoint. See :ref:`frame-parseval-what-was-wrong`.

- ⚠ **The frame square's scalar closure is a SEPARATE property from
  the metric, and only the DIAGONAL verdict implies it.** For the SH
  frame on a degree-exact sphere cubature the square closes on the
  single scalar :math:`W` — :math:`M^* = R/W`, :math:`R^* = W\,M`
  (:eq:`frame-square-closure-sh`) — because each live :math:`\ell`
  block of :math:`G` is one constant. A ``DENSE`` frame carries the
  right metric and need not satisfy that collapse, so ``DIAGONAL`` is
  *sufficient* and ``DENSE`` is *undecided*. ⛔ This bullet named the
  slab GL(8) :math:`L{=}2` frame as one of "the four ``DENSE``
  angular frames measured … three break it and one does not", until
  2026-09-02; that frame is now ``DIAGONAL`` and its closure HOLDS
  (:ref:`frame-g0-descent-arrow`). `[M]` 2026-09-02, 200 seeds each,
  the statement is unchanged and better witnessed: ``product(4,4)``
  :math:`L{=}2` (:math:`7.6\times10^{-4}`–:math:`0.47`),
  ``level_symmetric(4)`` :math:`L{=}3`
  (:math:`4.3\times10^{-2}`–:math:`0.15`) and
  ``folded_product(2,4)`` :math:`L{=}3` (:math:`0.26`–:math:`1.65`)
  break it, while ``folded_product(4,6)`` :math:`L{=}3` and — a NEW
  witness, from the other basis family — ``gauss_legendre(2)``
  :math:`L{=}2` do **not** (:math:`\le 1.9\times10^{-15}`), both
  because their disagreement lives entirely in :math:`\ker Y`. Never
  quote the closure as a frame law
  (:ref:`spaces-metric-frame-square` on
  :doc:`/theory/foundations/spaces`).

- **The measure never carries the discipline.** The
  :class:`~orpheus.numerics.measure.DiscreteMeasure` carries the axis
  and a fixed :math:`L^2` metric (the :term:`quadrature` weights). The
  solution-weighting (forward flux :math:`\phi`, adjoint
  :math:`\varphi^*`) that distinguishes a Petrov-Galerkin instance is
  a first-class **test basis** — the test *space* — not a metric on
  the measure. This is the load-bearing rule the homogenisation /
  condensation consumers obey.

- The :mod:`orpheus.numerics.projection` module carries only the two
  abstract operator **roles**:
  :class:`~orpheus.numerics.projection.AnalysisOperator`
  (:math:`M : V \to W`) and
  :class:`~orpheus.numerics.projection.ReconstructionOperator`
  (:math:`R : W \to V`), which the two frame faces subclass. The
  discipline is the frame's type, never a marker on these roles.

- **Two families of concrete frame ship today.** The
  **spherical-harmonic frame**
  :meth:`Quadrature.angular_frame(L)
  <orpheus.numerics.quadrature.Quadrature.angular_frame>` — the
  :class:`~orpheus.numerics.basis.SphericalHarmonicBasis` of order
  :math:`L` bound to an :math:`S^2` cubature — is a
  :class:`~orpheus.numerics.frame.GalerkinFrame` (``test is trial``)
  and a **4π-tight frame**. The forward (reaction-rate)
  **Petrov-Galerkin** consumers —
  :meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`
  (space) and
  :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`
  (energy) — ship as concrete
  :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` instances with an
  explicit flux- / spectrum-weighted
  :class:`~orpheus.numerics.basis.WeightedIndicatorBasis` test basis
  (landed P3 / P5; full derivations in
  :ref:`sn-homogenization-petrov-galerkin-frame` and
  :ref:`sn-energy-condensation`, below). The
  **eigenvalue-consistent** (adjoint-weighted,
  :math:`\varphi^* \ne \varphi`) case **also ships** (P6, #281): the
  ``adjoint=`` parameter on both verbs weights the test basis by the
  bilinear pair :math:`\varphi^*\!\odot\varphi` (with the exact angular
  and per-pair refinements) so the coarse :math:`\keff` is first-order
  stationary (:eq:`sn-homogenization-bilinear`), carrying its own
  full-taxonomy L0/L1/L2 gate battery
  (:ref:`frame-adjoint-weighted-seam`).

- **The frame is also the production COMPOSER, not only the two faces.**
  Beyond ``analysis`` / ``reconstruction``, a
  :class:`~orpheus.numerics.frame.FrameBase` emits the composed-operator
  verbs a consumer uses directly — *define a frame, compose, done*:
  :meth:`conjugate <orpheus.numerics.frame.FrameBase.conjugate>`
  (:math:`R\circ A\circ M`, the scattering kernel) and
  :meth:`project <orpheus.numerics.frame.FrameBase.project>`
  (:math:`G^{-1}M`, the homogenise / condense verb). These are typed
  operator products, not hand-rolled numpy chains
  (:ref:`frame-composed-verbs`).

- **Three disciplines, gated by the trial basis's Gram structure.** The
  built frames cover the *row-sum-collapsible* Gram cases —
  :class:`~orpheus.numerics.frame.GalerkinFrame` (diagonal Gram,
  ``test is trial``) and the forward
  :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` (diagonal *or*
  partition-of-unity Gram). The third discipline — a least-squares
  frame over a **dense** cross-Gram needing the real
  :math:`(MR)^{-1}M` solve — is **designed but not built**:
  :meth:`FrameBase.project <orpheus.numerics.frame.FrameBase.project>`
  *refuses* a :class:`~orpheus.numerics.basis.GramStructure`
  ``DENSE`` trial (raising
  :class:`~orpheus.numerics.operator.NotInvertible`) rather than
  return a silently-wrong coarsening (:ref:`frame-least-squares-discipline`).

- Every concrete :class:`~orpheus.numerics.frame.GalerkinFrame`
  satisfies the **idempotency-on-coefficients** invariant on a
  sufficiently-exact quadrature:

  .. math::
     :label: galerkin-frame-idempotency

     M \, R \;=\; c_{V}\,I_{W},

  .. (vv-status rationale) Structural invariant: the general Galerkin
     idempotency-on-coefficients schema M R = c_V I. Its SN concrete instance
     (c_V = 4π) is :eq:`pi-r-equals-4pi-i`, the L1-verified form — the canonical
     pin ``tests/numerics/test_spherical_harmonic_space.py`` ``verifies("pi-r-equals-4pi-i")``
     constructs Π R = 4π I at multiple L / Lebedev orders. Not a separate solver claim.
  .. vv-status: galerkin-frame-idempotency documented

  where :math:`c_V` is a scalar that depends on the inner-product
  convention of :math:`V`. For the SN spherical-harmonic frame on a
  Lebedev quadrature, :math:`c_V = 4\pi` — this is the **L1
  idempotency** identity :eq:`pi-r-equals-4pi-i` verified at multiple
  :math:`L` against multiple Lebedev orders. (A 4π-tight frame is one
  whose frame operator :math:`S = T^*T` is :math:`4\pi` times the
  identity; the tightness constant IS this :math:`c_V`.)

- **The frame is the single source of the COEFFICIENT SPACE too, not
  only of the faces** (:ref:`frame-moment-space-single-home`, #429
  tracker 2.5). A consumer reads the space off the bound basis
  (:eq:`moment-space-read-off-the-frame`); it never re-mints it from the
  truncation order, because an integer does not say *which family* — the
  full harmonics on a full-sphere rule, the σ-even restriction on a
  folded one, the Legendre basis on :math:`S^2/O(2)_a` on a 1-D one.
  `[M]` 2026-09-02 the angular moment space had **eight** homes (one
  basis + seven ``from_L(L)`` re-mints) and now has one. ⚠ Two spellings
  live on a frame and they are different objects: ``basis.space``
  carries the CONTINUUM Gram :math:`g_C` and is the basis's own
  coefficient space; ``basis_space`` carries the Parseval metric
  :math:`G^{-1}` — a positioned :math:`G^{+}` where the discrete Gram is
  dense — and is the ANALYSIS FACE's codomain
  (:ref:`frame-parseval-metric`). ⛔ Until CS4c step 6 item 6.2c-ii
  (2026-09-08) this bullet ended *"``==`` cannot tell them apart"* and
  called the continuum one *"the end an* :math:`\ell`\ *-diagonal operator
  wants"*. Both halves are now false. The harmonic heads are AXIS-BUILT,
  so the measure is part of the identity: `[M]` over the 33 shipped
  (rule, :math:`L`) rows ``frame.basis_space == frame.basis.space`` is
  **0 of 33** (it was 33 of 33). And ruling **R-6.2c-1** binds the ONE
  moment space the tree carries — the carrier's cached mint, every
  moment field, every operator end — to ``basis_space``, because *the
  carrier's norm is the field's energy*
  (:ref:`frame-moment-space-single-home`).


The discrete frame — analysis, synthesis, and the frame operator
================================================================

The :math:`(R, M)` pair is the language of **frame theory**
(Christensen 2016, *An Introduction to Frames and Riesz Bases*). A
discrete frame is a countable family :math:`\{e_k\}` in a Hilbert
space :math:`V` for which two operators are defined:

* the **analysis operator** :math:`T : V \to W`,
  :math:`(T f)_k = \langle e_k, f \rangle_V` — it *analyses* a
  function into its coefficients against the frame elements;
* the **synthesis operator** :math:`T^* : W \to V`,
  :math:`T^* c = \sum_k c_k\,e_k` — the formal adjoint of
  :math:`T`, the bare expansion with NO weighting and NO dual
  factor.

Their composition is the **frame operator**
:math:`S = T^* T : V \to V`. A frame is **tight with constant
:math:`c`** when :math:`S = c\,I` — the frame elements then behave
like an orthonormal basis up to the scalar :math:`c`, and the
inversion is trivial: :math:`f = c^{-1} T^* T f`.

In the ORPHEUS algebra, the analysis operator IS the **analysis face**
:math:`M = T` (measured against the test basis), and the
**reconstruction** :math:`R` is the **canonical-dual synthesis** —
:math:`T^*` weighted by the dual frame's Gram-inverse so that
:math:`M R` recovers the band-limited identity (up to tightness). The
bare :math:`T^* = S_0` (the *naked synthesis*) is the shared
:meth:`~orpheus.numerics.basis.Basis.synthesize` primitive on the
:class:`~orpheus.numerics.basis.Basis`; the analysis face :math:`M`
and the reconstruction face :math:`R` are each :math:`S_0`
post-multiplied by exactly one diagonal weight family (the quadrature
weight :math:`w_n` for analysis; the addition-theorem factor
:math:`2\ell+1` for reconstruction).

Given a fine space :math:`V` (e.g. :math:`L^2(S^2)` for the angular
flux) and a coarse coefficient space :math:`W` (e.g. polynomials of
degree :math:`\le L` on :math:`S^2`), a Galerkin-style discretisation
splits as:

.. math::
   :label: galerkin-pair

   R \;:\; W \to V, \qquad
   M \;:\; V \to W,
   \qquad
   M \, R \;=\; c_V \, I_W

where :math:`c_V` is the frame's tightness constant
(:math:`c_V = 1` in the fully-orthonormal case;
:math:`c_V = 4\pi` for the no-prefactor real spherical harmonics).

.. vv-status: galerkin-pair documented

The frame is fully determined by **three** ingredients — and the
third, the discipline, is the type, not a fourth parameter:

1. The **domain** :math:`V` and its inner product
   :math:`\langle \cdot, \cdot \rangle_V` — fixed by the
   :class:`~orpheus.numerics.measure.DiscreteMeasure` (the sample
   nodes and the fixed quadrature-weight :math:`L^2` metric). For SN
   angular flux, :math:`V = L^2(S^2)` and the inner product is the
   W-weighted discrete sum
   :math:`\langle f, g \rangle_W = \sum_n w_n f_n g_n` on the
   angular cubature.
2. The **trial basis** of :math:`W` — fixed by the
   :class:`~orpheus.numerics.basis.Basis` (the synthesis / trial
   side; it owns the reconstruction). For SN scattering, the basis is
   the real spherical harmonics :math:`\{Y_\ell^m\}_{\ell \le L}`.
3. The **test basis** — the analysis (measured) side, fixed by the
   frame **type**. A :class:`~orpheus.numerics.frame.GalerkinFrame`
   uses the trial basis itself (``test is trial``); a
   :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` carries an
   explicit, generally different test basis (e.g. the indicator
   weighted by the within-group spectrum, or by the region flux).

Once the basis and the measure are bound and the frame type is
chosen, :math:`M` and :math:`R` are uniquely determined up to the
:math:`c_V` normalisation.


.. _frame-parseval-metric:

The frame owns its codomain metric — the Parseval theorem
=========================================================

The three ingredients above fix :math:`M` and :math:`R`. They also fix
something that is easy to miss and was got **wrong in this project for
months**: the **inner product on the coefficient space** :math:`W`.
That metric is not a free choice, not a property of the basis, and —
in general — not even diagonal. It is *induced* by the pairing, and
this section derives it, states when it exists, and records what the
frame does when it does not.

The stakes are exactly one thing: an adjoint is metric-relative
(:math:`\langle Af, g\rangle_{\rm codomain} = \langle f, A^*g
\rangle_{\rm domain}` defines :math:`A^*` only once *both* inner
products are named), so a face's ``.H`` is the **physical** Hilbert
adjoint if and only if the metric the codomain carries is the right
one. Compose two faces and the interior metrics cancel — which is why
this defect could sit in the tree, unfalsified, behind a wall of green
gates.

.. _frame-analysis-is-the-gram-section:

The theorem: analysis returns the Gram, so the metric is its inverse
--------------------------------------------------------------------

.. warning:: **Three different things on this page are called** :math:`W`.
   The page's own convention (used above and below) writes the coarse
   **coefficient space** as :math:`W` and the quadrature-weighted nodal
   inner product as :math:`\langle\cdot,\cdot\rangle_W`; the corpus and
   the code additionally use the **scalar total weight**
   :math:`W = \sum_n w_n` (the scattering operator's :math:`1/W`
   prefactor, :ref:`normalization-prefactor`). All three are kept —
   they are the established spellings and renaming any of them here
   would mint a doc/code twin. To keep the algebra unambiguous this
   section therefore writes the weight **matrix** as
   :math:`\mathrm{diag}(w)`, never as :math:`W`.

Let the trial basis be :math:`\{\phi_k\}`, tabulated at the measure's
nodes as :math:`Y_{nk} = \phi_k(x_n)`
(:attr:`FrameBase.table <orpheus.numerics.frame.FrameBase.table>`), and
let :math:`\mathrm{diag}(w)` be the measure's weights. Define the
**discrete trial Gram** — computed once per frame from the cached
table, :math:`O(N K^2)`, as
:attr:`FrameBase.discrete_gram
<orpheus.numerics.frame.FrameBase.discrete_gram>`:

.. math::
   :label: frame-discrete-gram

   G_{jk} \;=\; \sum_n w_n\,\phi_j(x_n)\,\phi_k(x_n)
   \;=\; \bigl(Y^{\mathsf T}\,\mathrm{diag}(w)\,Y\bigr)_{jk}

.. (vv-status rationale) Definition of the frame's discrete trial Gram —
   a named intermediate, not a solver claim. Its shipped computation is
   FrameBase.discrete_gram (einsum over the CACHED table); the value is
   pinned per family by ``test_parseval_dressing_installed_on_diagonal_frames``
   in ``tests/numerics/test_frame.py`` (which reads the diagonal back out
   of it) and, for the SH basis, by the closed-form
   :eq:`real-sh-discrete-orthogonality` gate.
.. vv-status: frame-discrete-gram documented


.. implements:: frame-discrete-gram
   :by: orpheus.numerics.frame.FrameBase.discrete_gram

   **Implemented by** 5 sites — the frame's cached operational copy
   (``FrameBase.discrete_gram``, an ``einsum`` over the cached table),
   the basis-side diagnostic contract (``Basis.mass_matrix``), and its
   three concrete overrides, which compute the same object in each
   basis's own index layout. Declaring only the frame's copy would
   refute the tests that exercise a basis's own ``mass_matrix``.

.. implements:: frame-discrete-gram
   :by: orpheus.numerics.basis.base.Basis.mass_matrix

.. implements:: frame-discrete-gram
   :by: orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.mass_matrix

.. implements:: frame-discrete-gram
   :by: orpheus.numerics.basis.indicator_basis.IndicatorBasis.mass_matrix

.. implements:: frame-discrete-gram
   :by: orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis.mass_matrix

Now take a field that is **band-limited** in the frame, i.e. exactly
representable as :math:`\psi = S_0 c` for some coefficient vector
:math:`c`, and push it through the analysis face
:math:`M = Y^{\mathsf T}\,\mathrm{diag}(w)`. Then

.. math::
   :label: frame-analysis-is-the-gram

   \varphi \;=\; M\psi
   \;=\; Y^{\mathsf T}\,\mathrm{diag}(w) \bigl(Y c\bigr)
   \;=\; \bigl(Y^{\mathsf T}\,\mathrm{diag}(w)\,Y\bigr) c
   \;=\; G\,c .

.. (vv-status rationale) Structural identity — three matrix products
   re-associated; it is ALGEBRA, holding for every basis, every measure
   and every quadrature order, with no exactness hypothesis. There is no
   separate implementing symbol to gate: it is the reason the Parseval
   metric is G⁻¹, and it is what the isometry gate
   ``test_parseval_analysis_is_an_isometry_onto_its_image``
   (``tests/numerics/test_frame.py``, 6 sphere families) measures the
   consequence of.
.. vv-status: frame-analysis-is-the-gram documented


**Read what that says.** The analysis face does **not** return the
coefficients :math:`c`. It returns :math:`Gc` — the field's
**covariant** moments, the pairings of :math:`\psi` against each basis
function. The two agree only when :math:`G = I`, i.e. for a basis that
is *orthonormal on this measure*, which the no-prefactor spherical
harmonics are not (:math:`G_\ell = 4\pi/(2\ell+1)`) and the indicators
are not (:math:`G_{RR} = m_R`, the region mass).

:eq:`frame-analysis-is-the-gram` is pure algebra: three matrix products
re-associated. It needs **no** quadrature-exactness hypothesis, holds at
every order, and is exact in floating point up to round-off
(`[M]` :math:`\le 1.8\times10^{-14}` across every shipped sphere family,
2026-08-23).


.. no-implementation:: frame-analysis-is-the-gram
   :kind: identity

   **Nothing implements this.** It is an identity between two
   quantities that ARE each computed — :math:`M\psi` by the analysis
   face, :math:`Gc` by
   :attr:`~orpheus.numerics.frame.FrameBase.discrete_gram` — and the
   identity itself is never evaluated anywhere: it is the *reason* the
   Parseval metric is :math:`G^{-1}`, established by re-associating
   three matrix products, not by any line of code. Declaring either
   side would assert that one of them *is* the identity. What the
   suite does instead is measure the identity's consequence (the
   isometry, :eq:`frame-parseval-isometry`) and, in the design probe,
   the residual :math:`\|M\psi - Gc\|_\infty` directly
   (:ref:`frame-parseval-numerical-evidence`).

The metric follows immediately. For the analysis face to be an
**isometry onto its image** — Parseval — we need a coefficient-space
inner product :math:`\langle\cdot,\cdot\rangle_\star` with

.. math::
   :label: frame-parseval-isometry

   \|\varphi\|_\star^2
   \;=\; \|\psi\|_W^2
   \;=\; (Yc)^{\mathsf T}\,\mathrm{diag}(w)\,(Yc)
   \;=\; c^{\mathsf T} G\, c .

.. (vv-status rationale) The Parseval isometry — a representational
   identity that FOLLOWS from :eq:`frame-analysis-is-the-gram` by
   substitution, defining the codomain metric rather than asserting a
   solver result. Shipped by FrameBase.basis_space (the dressing) and
   gated directly by ``test_parseval_analysis_is_an_isometry_onto_its_image``
   over the seven Parseval-capable frames (the six DIAGONAL sphere
   families plus, since P7, the DENSE slab) plus
   ``test_indicator_frame_parseval_metric_is_the_inverse_region_mass``
   and the DENSE arm's own four-mechanism dressing gate
   ``test_dense_frames_are_dressed_with_the_pseudo_inverse_gram``, with
   two loaded-not-blind negative legs — the diagonal arm's
   ``test_parseval_reds_under_the_pre_repair_continuum_metric`` and the
   dense arm's
   ``test_the_dense_dressing_reds_under_the_diagonal_and_the_pre_repair_metrics``
   (``tests/numerics/test_frame.py``).
.. vv-status: frame-parseval-isometry documented


.. implements:: frame-parseval-isometry
   :by: orpheus.numerics.frame.FrameBase.basis_space

   **Implemented by** 1 site, deliberately. This equation *determines*
   the coefficient-space inner product, and exactly one symbol makes
   the choice it determines:
   :attr:`~orpheus.numerics.frame.FrameBase.basis_space`, which installs
   :math:`G^{-1}` (zero on dead slots) as the codomain's metric — since
   CS4c step 6 item 6.2c-ii by RE-WEIGHTING the coefficient space's single
   head axis, or, on a ``DENSE`` frame, by emptying that axis's measure and
   POSITIONING :math:`G^{+}` as a
   :class:`~orpheus.numerics.metric.DenseMetric` on the space's derived
   metric object; before it, as a plain ``inner_product_weights`` array.
   The *norm* on the left is then evaluated
   by the generic
   :meth:`FunctionSpace.inner_product
   <orpheus.numerics.space.FunctionSpace.inner_product>`, which is NOT
   declared here: it computes the pairing of every space in the corpus,
   so naming it would attribute this equation's content to a symbol that
   knows nothing about it.

Substituting :math:`\varphi = Gc` gives
:math:`\varphi^{\mathsf T} G^{-1} \varphi = c^{\mathsf T} G c`, so

.. math::

   \boxed{\;\langle\cdot,\cdot\rangle_\star
   \;=\; \langle\cdot,\cdot\rangle_{G^{-1}}\;}

— **the Parseval metric is the INVERSE of the discrete trial Gram.**
Not the Gram; its inverse. The distinction is the whole of step F-0.

**Dead slots and the pseudo-inverse.** :math:`G` need not be
invertible: a storage layout can carry slots no basis function occupies
(the :math:`|m| > \ell` padding of the SH ``(L+1, 2L+1)`` array), a
folded rule can annihilate whole columns (its :math:`\sigma`-odd
harmonics), and a region can be empty (:math:`m_R = 0`). Every such
slot has :math:`G_{kk} = 0` and — on a ``DIAGONAL`` frame, by the
verdict's own definition — no coupling into it, so :math:`G^{-1}` is
read as the **Moore–Penrose** pseudo-inverse: :math:`1/G_{kk}` on live
slots, **exactly** :math:`0.0` on dead ones. That is not a
convenience. A dead column annihilates its coefficient in
:math:`\psi = S_0c`, so it contributes nothing to :math:`\|\psi\|_W`;
it also zeroes that slot's moment; and the zero metric entry then
zeroes its contribution to :math:`\|\varphi\|_\star`. Parseval
therefore holds **whatever garbage sits in a dead slot**, which is
exactly the convention
:meth:`FrameBase.project <orpheus.numerics.frame.FrameBase.project>`
already uses.

.. _frame-metric-is-induced-not-a-constant:

The metric is INDUCED by the pairing, not a constant of the basis
-----------------------------------------------------------------

:math:`G = Y^{\mathsf T}\,\mathrm{diag}(w)\,Y` names *both* factors: the
basis (through :math:`Y`) and the measure (through
:math:`\mathrm{diag}(w)`). So the Parseval metric
is a property of the **pairing**, and the frame — the object that IS
the pairing — is its only correct owner. Three consequences, each of
which the pre-F-0 design got wrong by locating the metric on the basis:

**(1) A basis constant cannot be the metric.** The clean witness is the
slab. The same :class:`~orpheus.numerics.basis.SphericalHarmonicBasis`
bound to a 1-D Gauss–Legendre measure has total weight
:math:`W = \sum_n w_n = 2`, not :math:`4\pi`; every basis-level
constant carries :math:`4\pi` in it, so no basis-level constant can be
right for both. (`[M]` 2026-08-23, ``gauss_legendre(8).angular_frame(2)``.)

**(2) The metric depends on the ORDER, not only on the family.** Two
frames over the same basis with different measures have different
:math:`G`, hence different metrics. The frame recomputes it; nothing is
inherited.

**(3) There is no guarantee the metric is diagonal.** :math:`G` is
symmetric positive semi-definite by construction (a Gram), but nothing
makes it diagonal unless the basis is orthogonal *on that measure*.
When it is not, the Parseval metric is a genuine **matrix** — which the
legacy ``inner_product_weights`` (an elementwise diagonal) cannot
express, and which since campaign 1 P7 the space carries as a
:class:`~orpheus.numerics.metric.DenseMetric` object instead. See the
dense arm below.

.. _frame-declared-vs-measured-gram:

Two Gram facts, and they are INDEPENDENT: declared vs measured
---------------------------------------------------------------

The codebase carries two properties with ``gram`` in the name and one
shared vocabulary (:class:`~orpheus.numerics.basis.base.GramStructure`).
They answer different questions and they routinely disagree — reading
one for the other is the trap this subsection exists to close.

.. list-table:: The two Gram-structure facts
   :header-rows: 1
   :widths: 22 39 39

   * -
     - :attr:`Basis.gram_structure
       <orpheus.numerics.basis.base.Basis.gram_structure>`
     - :attr:`FrameBase.discrete_gram_structure
       <orpheus.numerics.frame.FrameBase.discrete_gram_structure>`
   * - Kind
     - **DECLARED** by the basis author
     - **MEASURED** from :eq:`frame-discrete-gram`
   * - About which matrix
     - The *cross* Gram :math:`MR` — what
       :meth:`~orpheus.numerics.frame.FrameBase.project` inverts via its
       row-sum probe
     - The *trial* Gram :math:`G = Y^{\mathsf T}WY` on **this** measure
   * - Depends on the measure?
     - No — a statement about the basis family
     - Yes — it is a measurement
   * - What it gates
     - :meth:`FrameBase.project
       <orpheus.numerics.frame.FrameBase.project>` (a
       :attr:`~orpheus.numerics.basis.base.GramStructure.DENSE`
       declaration is refused)
     - The Parseval dressing on
       :attr:`~orpheus.numerics.frame.FrameBase.basis_space`

Two measured disagreements ship today, one in each direction:

* the SH basis **declares** ``DIAGONAL`` (it is continuum-orthogonal)
  and **measures** ``DENSE`` on an under-resolved sphere rule —
  ``product(4,4)`` at :math:`L = 2`, say. ⛔ This bullet read *"on the
  slab Gauss–Legendre measure"* until 2026-09-02, when the fused
  ERR-080 commit stopped a 1-D rule binding that basis at all; `[M]` the
  slab GL(8) :math:`L{=}2` frame now binds
  :class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis` and
  **measures** ``DIAGONAL``, off-diagonal :math:`8.8\times10^{-17}`,
  diagonal exactly :math:`2/(2\ell+1) = [2,\ 2/3,\ 2/5]`. ⭐ The
  disagreement was never a property of the basis or of the measure
  alone: it was the *pairing*, and the pairing was ill-posed;
* an :class:`~orpheus.numerics.basis.overlap_basis.OverlapBasis`
  **declares** ``PARTITION_OF_UNITY`` — true, and what ``project``
  needs — while its trial Gram **measures** ``DENSE``, because a
  straddling row gives two columns shared support.

The verdict is a measurement with a stated threshold, not a heuristic:
``DIAGONAL`` iff no diagonal entry is negative (a negative-weight
quadrature can make :math:`G` indefinite, and an indefinite form is not
a metric) **and** every live off-diagonal is below
:math:`10^{-10}` of the Cauchy–Schwarz scale
:math:`\sqrt{G_{jj}G_{kk}}`. The measured separation is wide open: `[M]` 2026-08-23, the six
shipped sphere families sit at
:math:`\le 2.7\times10^{-16}` and the slab at :math:`0.93`, so **any**
threshold across fifteen orders of magnitude draws the same verdict
(the shipped one is :math:`10^{-10}`, leaving six orders of headroom
for round-off accumulation at high mode counts). **Structurally dead slots** (:math:`G_{kk}=0`:
layout padding, a folded rule's :math:`\sigma`-odd columns, an empty
indicator region) are exempt from the off-diagonal test but any
coupling *into* a dead slot is ``DENSE``.

.. _frame-parseval-dense-arm:

When no diagonal metric exists — the dense arm, and the slab witness
---------------------------------------------------------------------

.. note::

   This section carried the anchor ``frame-parseval-dense-refusal``
   and the title *"the refusal arm"* until campaign 1 P7 (2026-08-30).
   The subject did not move — it is still *what the frame does when no
   diagonal metric exists* — but the answer did, from a refusal to a
   matrix dressing, so the anchor was renamed with its one cross-page
   citer. A stale pointer to the old name renders as plain text at
   every build severity; if you meet one, it predates P7.

If the measured verdict is ``DENSE``,
:attr:`FrameBase.basis_space
<orpheus.numerics.frame.FrameBase.basis_space>` dresses the basis's
space with the **matrix** Parseval metric — the Moore–Penrose
pseudo-inverse :math:`G^{+}` of the measured (symmetrized) Gram,
installed as a :class:`~orpheus.numerics.metric.DenseMetric` at the
metric module's pinned cutoff, with the exact symmetrized Gram kept as
the inverse face — and **strips** the basis's continuum weights, since
the dressing replaces the metric on this arm exactly as the diagonal
arm overwrites it. Parseval is then a theorem for **any** Gram,
singular or not (:eq:`spaces-pseudo-inverse-parseval` on
:doc:`/theory/foundations/spaces`), and each face's ``.H`` is the
physical Hilbert adjoint on every frame rather than only the diagonal
ones.

⛔ **Until P7 this arm was a REFUSAL, and the record is worth keeping
because the refusal was correct at the time.** ``basis_space`` returned
the basis's own space **undressed**; the verdict property was the loud
record; the Parseval gate skipped such a frame with a named reason
rather than silently passing; and the cost, stated plainly then, was
that on a ``DENSE`` frame *Parseval is unavailable* and each face's
``.H`` was the stored-metric sandwich — a perfectly well-defined
operator, and **not** the physical Hilbert adjoint. The honest matrix
home was recorded as a debt against the CS4c Riesz-leg machinery, and
the reason it was a refusal rather than a bug is the same reason it was
repairable: a diagonal metric is not merely *unavailable* on a dense
frame, it is **provably insufficient**, and nothing in the space layer
could express the alternative. P7 built the alternative. What survives
unchanged from the refusal era is the *diagnosis* below — the slab
table, and the impossibility argument it supports.

⚠ **What still does not follow from a correct metric.** The
spherical-harmonic frame square's collapse onto one scalar
(:eq:`frame-square-closure-sh`) is a *different* property, and dressing
a ``DENSE`` frame does not buy it: `[M]` on the slab, under the correct
:math:`G^{+}`, the Parseval isometry reads :math:`1.000000000000` while
:math:`M^{*}` vs :math:`R/W` is :math:`O(1)` apart, because the live
:math:`\ell = 2` Gram diagonal :math:`[0.4,\,0.8,\,0.8]` is not a
per-:math:`\ell` scalar and no :math:`G_\ell` exists to collapse. The
full three-way split — and the shipped ``DENSE`` frame that *does*
close, for a reason worth knowing — is
:ref:`spaces-metric-frame-square`.

.. note::

   ⛔ **Retraction (2026-09-02, #429 / ERR-080). The table below is
   HISTORY: its numbers are exactly right and the frame it measures no
   longer exists.** The slab GL(8) :math:`L{=}2` frame was ``DENSE``
   *because* its basis was fabricating azimuthal columns — the rows
   "live slots per degree :math:`[1,1,3]`" and "diagonal :math:`0.8` on
   the two surviving :math:`\ell{=}2`, :math:`m \ne 0` slots" ARE the
   fabrication, tabulated. Since the fused commit a 1-D rule binds the
   Legendre basis its orbit space admits, and `[M]` 2026-09-02 the same
   frame reads:

   .. list-table:: The same frame, after the repair
      :header-rows: 1
      :widths: 40 60

      * - Quantity
        - Measured
      * - Total weight :math:`W`
        - :math:`2` exactly — unchanged, it is the rule's
      * - Live slots per degree, :math:`\ell = 0,1,2`
        - :math:`[\,1,\ 1,\ 1\,]` — a FLAT head of :math:`L+1`
          coefficients; there are no :math:`m` slots to fabricate
      * - Diagonal
        - :math:`2/(2\ell+1) = [\,2,\ 2/3,\ 2/5\,]`, exact
      * - Largest off-diagonal
        - :math:`8.8\times10^{-17}`; relative to
          :math:`\sqrt{G_{jj}G_{kk}}`, **0.0** under the verdict's
          threshold
      * - Verdict
        - ``DIAGONAL`` — the dense arm is not reached, and the frame
          square's scalar closure now HOLDS on it
          (`[M]` :math:`\le 5.1\times10^{-16}` over 200 seeds)

   ⭐ **The impossibility argument below is unaffected and its
   conclusion is unchanged** — a diagonal metric still cannot undo a
   coupling between two slots, and that is what makes the dense arm
   necessary. What changed is which frames exercise it. `[M]`
   2026-09-02 the shipped ``DENSE`` angular frames are
   ``product(4,4)`` :math:`L{=}2`, ``level_symmetric(4)`` :math:`L{=}3`,
   ``folded_product(2,4)`` :math:`L{=}2`\ /:math:`3`,
   ``folded_product(4,6)`` :math:`L{=}3`, and — from the *other* basis
   family — ``gauss_legendre(2)`` at :math:`L \ge 2`, whose dense
   verdict is the **dead-slot theorem** rather than a fabrication:
   :math:`P_n` vanishes identically at ``GL_n``'s own nodes, so a 1-D
   Gauss frame is diagonal-and-exact for :math:`L \le n-1` and
   rank-deficient at :math:`\ell = n` (`[M]` 12 of 12 rows). ⟹ **no
   1-D Gauss–Legendre frame can be both dense and full-rank**, which is
   worth knowing before anyone reads a slab dense arm as a defect
   again.

The witness WAS the slab. `[M]` 2026-08-23,
``Quadrature.gauss_legendre(8).angular_frame(2)``, **before the ERR-080
repair**:

.. list-table:: The slab GL(8) discrete Gram at :math:`L = 2` — why no diagonal metric works
   :header-rows: 1
   :widths: 34 66

   * - Quantity
     - Measured
   * - Total weight :math:`W = \sum_n w_n`
     - :math:`2` exactly (**not** :math:`4\pi`)
   * - Live slots per degree
       (:math:`G_{kk} > 0`), :math:`\ell = 0,1,2`
     - :math:`[\,1,\ 1,\ 3\,]` — every node has
       :math:`\mu_y = \mu_z = 0` (verified for all 8), so the rule has
       **no azimuthal resolution at all**: the :math:`m\neq0` columns
       are not independently sampled
   * - Diagonal on the :math:`m=0` slots
     - :math:`W/(2\ell+1) = 2,\ 2/3,\ 2/5` — exact
   * - Diagonal on the two surviving :math:`\ell=2`,
       :math:`m\neq0` slots
     - :math:`0.8 = 2\times(2/5)` — **twice** the
       :math:`W/(2\ell+1)` value, so even the diagonal is not a
       :math:`W/(2\ell+1)` law
   * - Off-diagonal couplings
     - three, all genuine:
       :math:`(\ell{=}0,m{=}0)\!\leftrightarrow\!(2,{+}2) = +1.1547`,
       :math:`(1,0)\!\leftrightarrow\!(2,{+}1) = +0.6826`,
       :math:`(2,0)\!\leftrightarrow\!(2,{+}2) = -0.2309`
   * - Largest off-diagonal, relative to
       :math:`\sqrt{G_{jj}G_{kk}}`
     - :math:`0.9347` — **ten** orders above the :math:`10^{-10}`
       verdict threshold; relative to the largest diagonal it is
       :math:`0.5774`
   * - Verdict
     - ``DENSE`` ⟹ the metric is the matrix :math:`G^{+}` (a
       :class:`~orpheus.numerics.metric.DenseMetric`); the basis's
       continuum weights are stripped. *(Until P7: dressing refused,
       continuum metric retained.)*

A diagonal metric can only rescale each coefficient slot; it cannot
undo a coupling between two slots. With off-diagonals at :math:`0.93`
of the Cauchy–Schwarz scale, **no diagonal candidate satisfies
Parseval on this frame** — this is a structural impossibility, not a
tolerance to be tightened. And "impossible" is measured rather than
argued: `[M]` 2026-08-30, on one band-limited :math:`\psi`
(``default_rng(1234)``) the Parseval ratio reads :math:`25.53` under the
undressed continuum metric, :math:`1.806` under the best diagonal
candidate :math:`1/\operatorname{diag}(G)`, and
:math:`0.999999999999999` under the matrix :math:`G^{+}`. The middle
reading is the load-bearing one — it is the *only* evidence class that
can adjudicate a metric at all, since the Hilbert-adjoint identity
:math:`A^{\dagger} \equiv G^{-1}A^{\mathsf T}G` holds for every
invertible :math:`G` and therefore proves loadedness without ever
proving *choice*.

.. note::

   **Recorded debt (CS4c) — the matrix-metric half is DISCHARGED; the
   legs are not.** The note below stood from 2026-08-23 until campaign 1
   P7 (2026-08-30), which landed the matrix metric it was waiting for
   (:ref:`spaces-metric-object`). Two halves of the debt remain open and
   are the compatibility target P7 deliberately built toward:

   - the **legs themselves** —
     :math:`\mathrm{riesz\_raise}` / :math:`\mathrm{riesz\_lower}`
     becoming space-minted *operators* rather than elementwise
     diagonals, at which point
     :math:`A^{*} = A.\mathrm{domain.riesz\_raise} \circ
     A.\mathrm{dual}() \circ A.\mathrm{codomain.riesz\_lower}` is a
     definition that a full matrix satisfies as easily as a diagonal;
     and
   - **retiring** ``AdjointOperator`` into that leg composition.

   Both are still CS4c's, and neither method exists in the tree today.
   What P7 changed is that they now have exactly one metric arithmetic
   to wrap: the :class:`~orpheus.numerics.metric.HilbertMetric` family's
   two faces are what the two legs will be, so the retirement needs no
   third spelling of the metric. Tracked in
   ``.claude/plans/frame_square_recarve.md`` (recorded debts).

.. _frame-square-closure-section:

The frame square, and the one scalar that closes it
----------------------------------------------------

With the Parseval metric installed, both faces' ``.H`` fall out of the
generic metric-aware adjoint wrapper with no bespoke code. For the
analysis face :math:`M` (domain = ``measure_space`` with metric
:math:`\mathrm{diag}(w)`, codomain = ``basis_space`` with metric
:math:`G^{-1}`) the sandwich
:math:`M^* = \mathrm{diag}(w)^{-1}\,M^{\mathsf T}\,G^{-1}` collapses,
because :math:`M^{\mathsf T} = Y\,\mathrm{diag}(w)` carries the very
weights the domain's inverse metric removes:

.. math::

   M^* \;=\; \mathrm{diag}(w)^{-1}\,\bigl(Y\,\mathrm{diag}(w)\bigr)\,G^{-1}
   \;=\; Y\,G^{-1}
   \;=\; S_0 \circ G^{-1} .

For the reconstruction face :math:`R = Y\,\mathrm{diag}(d)` (with
:math:`d` the trial-side synthesis weights — for the SH basis the
addition-theorem factor :math:`d_\ell = 2\ell+1`) the domain is now the
dressed coefficient space, so its *inverse* metric is :math:`G` and

.. math::

   R^* \;=\; G\,\mathrm{diag}(d)\,Y^{\mathsf T}\,\mathrm{diag}(w)
   \;=\; \bigl(G\,\mathrm{diag}(d)\bigr)\,M .

Both are general. The **spherical-harmonic** frame then does something
special: its two per-:math:`\ell` diagonals are reciprocal up to one
constant,

.. math::
   :label: frame-square-closure-sh

   d_\ell\,G_\ell
   \;=\; (2\ell+1)\cdot\frac{4\pi}{2\ell+1}
   \;=\; 4\pi \;=\; W
   \qquad\text{for every }\ell,

.. (vv-status rationale) The SH degree-exactness identity d_ℓ·G_ℓ = W —
   a property of the (spherical-harmonic basis ⊗ degree-exact sphere
   cubature) pairing, transcribed from the no-prefactor convention's own
   two constants (:eq:`real-sh-discrete-orthogonality` and
   :eq:`sh-addition-theorem-reconstruction`), not a solver claim. It is
   what collapses the general adjoints to M* = R/W and R* = W·M, and it
   is measured directly (max relative deviation over live ℓ) by
   ``test_parseval_frame_square_closes`` in ``tests/numerics/test_frame.py``,
   whose ``verifies`` marker targets
   :eq:`hilbert-adjoint-equals-metric-times-S0`.
.. vv-status: frame-square-closure-sh documented

so the whole per-:math:`\ell` dressing collapses to the single scalar
:math:`W`:

.. math::

   M^* \;=\; \frac{R}{W},
   \qquad
   R^* \;=\; W\,M .

**The frame square closes with one scalar — and that scalar is already
in the code.** It is the :math:`1/W` the scattering operator applies
once (:eq:`scattering-aniso-composite`,
:doc:`/theory/foundations/operator_algebra`); the prefactor ledger's
"unification the canon misses",
:math:`(2\ell+1)/W` (:ref:`normalization-prefactor`,
:doc:`/theory/conventions/normalization`), **is** the Parseval metric
:math:`G^{-1}` written out.


.. no-implementation:: frame-square-closure-sh
   :kind: identity

   **Nothing implements this.** Both of its factors ship —
   :math:`d_\ell` as
   :attr:`SphericalHarmonicBasis.addition_theorem_factor
   <orpheus.numerics.basis.SphericalHarmonicBasis.addition_theorem_factor>`
   and :math:`G_\ell` as the diagonal of
   :attr:`~orpheus.numerics.frame.FrameBase.discrete_gram` — but their
   *product* is never formed in production: nothing multiplies them,
   because the whole point of the identity is that the code does not
   have to. It is what lets the shipped scattering kernel carry one
   :math:`1/W` scalar instead of a per-:math:`\ell` table. Declaring
   either factor would attribute the identity to a quantity that is
   merely one of its sides. It is *measured* (as
   :math:`\max_\ell|d_\ell G_\ell/W - 1|`) by the frame-square gate and
   by the table at :ref:`frame-parseval-numerical-evidence`.

.. warning::

   :eq:`frame-square-closure-sh` is **SH-specific**; Parseval is not.
   Parseval needs only a diagonal :math:`G` — with *any* values — while
   :math:`M^* = R/W` additionally needs :math:`d_\ell G_\ell` to be the
   same number for every mode. The indicator frame is the standing
   counter-example: it satisfies Parseval exactly and does **not**
   satisfy the closure (`[M]` :math:`d = 1` and :math:`G_{RR} = m_R`,
   so :math:`d\,G` is the region-mass vector, not a constant; on the
   4-node / 3-region fixture ``M.H`` reads
   :math:`[0.5,\,0.5,\,0.667,\,0.667]` where :math:`R/W` reads
   :math:`[0.2,\,0.2,\,0.4,\,0.4]`). Never quote the closure as a frame
   law. The **single-region** indicator frame — the degenerate
   :math:`K = 1` case of the same counter-example — is what mints the
   axis collapse pair, where the Gram is :math:`1\times1` and its entry
   is the axis's total mass: see
   :ref:`spaces-collapse-pair-frame`.

.. _frame-parseval-numerical-evidence:

Numerical evidence
------------------

`[M]` 2026-08-23, measured against the tree at HEAD. **The
construction, so the table regenerates from this page**: build the
frame (``Quadrature.<family>(...).angular_frame(L)``, or
``GalerkinFrame(SphericalHarmonicBasis(L=2), lebedev_sphere(13))``);
draw :math:`c \sim` ``default_rng(1234).standard_normal(frame.basis_space.shape)``
*unmasked* (garbage in the dead slots is deliberate — see the
pseudo-inverse paragraph above); form
:math:`\psi = \texttt{frame.basis.synthesize}(c, \texttt{frame.table})`
and :math:`\varphi = \texttt{frame.analysis.apply}(\psi)`; then read
the five residuals off ``frame.discrete_gram``,
``frame.basis_space.inner_product``, ``frame.measure_space.inner_product``,
``frame.analysis.H``, ``frame.reconstruction.H`` and
:math:`W = \texttt{frame.measure.weights.sum()}`. Columns 3, 5 and 6
are max-absolute residuals; the Parseval column is the *ratio*
:math:`\|\varphi\|^2_\star / \|\psi\|^2_W` under whichever metric the
frame installs (:math:`G^{-1}` on a ``DIAGONAL`` frame, the matrix
:math:`G^{+}` on a ``DENSE`` one), whose exact value is :math:`1`;
column 7 is over the live :math:`\ell` only. Columns 5 and 6 need two
further draws — :math:`y` on the coefficient space and :math:`v` on the
node set — which the rows below do not record, because on a
``DIAGONAL`` frame both residuals sit at round-off and the draw is
immaterial. On a ``DENSE`` frame it is **not** immaterial, which is why
the slab row leaves those two cells unfilled and says so.

.. list-table:: The theorem, Parseval, and the closure, per shipped angular frame
   :header-rows: 1
   :widths: 22 12 15 15 12 12 12

   * - Frame
     - Verdict
     - :math:`\|M\psi - Gc\|_\infty`
     - Parseval ratio
     - :math:`\|M^*y - R y/W\|_\infty`
     - :math:`\|R^*v - W M v\|_\infty`
     - :math:`\max_\ell |d_\ell G_\ell/W - 1|`
   * - LS\ :sub:`4`, :math:`L=1`
     - ``DIAGONAL``
     - 3.6e-15
     - 1.000000000
     - 2.2e-16
     - 3.6e-15
     - 3.3e-16
   * - LS\ :sub:`4`, :math:`L=2`
     - ``DIAGONAL``
     - 1.3e-15
     - 1.000000000
     - 2.2e-16
     - 1.4e-14
     - 6.7e-16
   * - LS\ :sub:`8`, :math:`L=2`
     - ``DIAGONAL``
     - 1.2e-14
     - 1.000000000
     - 7.8e-16
     - 2.5e-14
     - 1.8e-15
   * - product :math:`8\times8`, :math:`L=2`
     - ``DIAGONAL``
     - 1.8e-14
     - 1.000000000
     - 2.2e-16
     - 1.8e-14
     - 5.6e-16
   * - folded :math:`8\times8`, :math:`L=2`
     - ``DIAGONAL``
     - 7.1e-15
     - 1.000000000
     - 1.7e-16
     - 2.1e-14
     - 2.2e-16
   * - Lebedev-13, :math:`L=2`
     - ``DIAGONAL``
     - 2.7e-15
     - 1.000000000
     - 1.7e-16
     - 8.9e-15
     - 3.3e-16
   * - GL(8) slab, :math:`L=2`
     - ``DENSE``
     - 4.4e-16
     - 1.000000000
     - *(O(1) — no collapse)*
     - *(O(1) — no collapse)*
     - *(no* :math:`G_\ell` *exists)*

.. note::

   **The slab row, added at P7 (2026-08-30), and why two of its cells
   are not numbers.** Until P7 the whole row read *"refused"*: the
   ``DENSE`` verdict withheld the dressing, so columns 4–7 had nothing
   to measure. With the matrix :math:`G^{+}` installed, columns 3 and 4
   are ordinary readings — the theorem
   :eq:`frame-analysis-is-the-gram` never depended on the metric, and
   Parseval now holds there as it does everywhere
   (:eq:`spaces-pseudo-inverse-parseval`).

   Columns 5 and 6 are deliberately *not* filled with a number. They
   measure the SH scalar collapse, which the slab does not satisfy at
   any metric — `[M]` the relative residual
   :math:`\|M^{*}y - Ry/W\|_\infty / \|Ry/W\|_\infty` ranges over
   :math:`0.30`–:math:`10.2` across 200 random :math:`y`, so any single
   figure printed here would be one draw's reading rather than a
   property of the frame. The *structural* statement is column 7's: the
   live :math:`\ell = 2` diagonal is :math:`[0.4,\,0.8,\,0.8]`, three
   different numbers, so there is no :math:`G_\ell` for
   :math:`d_\ell G_\ell = W` to be about. See
   :ref:`spaces-metric-frame-square` for the full three-way split,
   including the shipped ``DENSE`` frame whose closure *does* hold.

Every shipped sphere family is degree-exact at these :math:`L`,
including the level-symmetric rules: :math:`\max_\ell |d_\ell G_\ell/W
- 1| \le 1.8\times10^{-15}` throughout — which is worth stating
explicitly, because a pre-F-0 test-module comment asserted the
opposite: *"at* :math:`L=2` *,* :math:`LS_8` *has a 24 % diagonal
error and no LS order makes it exact"*. ⛔ **Refuted** by that column.
*Hypothesis, not measured:* the claim probably predates GitHub #327
(*"level_symmetric quadrature is degree-3 at EVERY order — advertises
N-1, over-claims by up to 12"*, CLOSED), whose repair moved the
level-symmetric node placement; nobody has bisected it.

The **indicator** frame instantiates the same theorem with completely
different numbers, which is the point — nothing here is spherical.
`[M]` on a 3-region / 4-node fixture with one empty region:
:math:`G = \mathrm{diag}(m_R) = \mathrm{diag}(2,\,3,\,0)` (the region
masses), the dressed metric is
:math:`[\,0.5,\ 1/3,\ 0.0\,]`, and Parseval is exact
(:math:`\|\varphi\|^2_{1/m} = \|f\|^2_V = 2900` on a region-wise
constant field). The empty region's metric slot is **exactly**
:math:`0.0`, matching
:meth:`~orpheus.numerics.frame.FrameBase.project`'s Moore–Penrose
convention: a dead slot annihilates its coefficient in :math:`S_0c`,
zeroes its moment, and zeroes its metric entry, so the identity holds
whatever garbage sits in that slot.

.. _frame-parseval-what-was-wrong:

What was wrong before, why nothing caught it, and what does
------------------------------------------------------------

Before step F-0 the frame exposed ``basis.space`` unchanged, so the
coefficient codomain carried the basis's **continuum** Gram
:math:`g_C = 4\pi/(2\ell+1)`
(:eq:`sh-space-metric`, :ref:`spherical-harmonics`). That is the Gram,
not its inverse — the **wrong side** for the covariant moments
:eq:`frame-analysis-is-the-gram` says analysis returns.

`[M]` the damage, on an LS\ :sub:`4` rule: the Parseval ratio read
:math:`81.4` at :math:`L=1` and :math:`65.2` at :math:`L=2` instead of
:math:`1`, and ``frame.analysis.H`` was off the physical adjoint by
exactly :math:`(4\pi/(2\ell+1))^2` per :math:`\ell` — a factor of
:math:`157.9` on the scalar moment, :math:`17.5` on the current,
:math:`6.3` at :math:`\ell = 2`. (The ratio is a moment-energy-weighted
average of those per-:math:`\ell` factors, so its numeric value depends
on the coefficient draw; what is draw-independent is that it lies
between the extreme factors present at that :math:`L`
(:math:`[17.5,\,157.9]` at :math:`L=1`, :math:`[6.3,\,157.9]` at
:math:`L=2`) and can therefore never be :math:`1`. The per-:math:`\ell`
factors themselves are exact, not approximate: `[M]` the ratio of the
pre-F-0 ``analysis.H`` to the shipped one on a single-:math:`\ell`
unit input reproduces :math:`(4\pi/(2\ell+1))^2` to
:math:`\le 2.8\times10^{-16}` relative at every :math:`\ell`.)

**Why every gate stayed green.** Three independent reasons, and each
one is a lesson worth carrying:

1. **Consistency is not correctness.** The machinery was
   self-consistent throughout — the defining adjoint identity
   :math:`\langle M\psi, c\rangle_{g_C} = \langle \psi, M^*c\rangle_W`
   held at the round-off floor (`[M]` 2026-08-23, LS\ :sub:`4`,
   ``default_rng(42)`` draws: relative residual
   :math:`9.5\times10^{-16}` at :math:`L=1`, **exactly** :math:`0.0`
   at :math:`L=2`) — because ``.H`` was *built from* the stored
   metric. A sandwich always reproduces the pairing it was
   assembled from, whatever that pairing is, so the identity is true
   for **every** symmetric positive-definite metric and carries zero
   information about which one is installed. The instrument that CAN
   fail is the isometry :math:`\|M\psi\|_\star = \|\psi\|_W`, which
   compares the codomain metric against something outside it — the
   field's own norm.
2. **Composed chains are immune.** Interior metrics cancel in a
   product, so the production kernel :math:`R\Lambda M` — which is
   where every angular moment in this code actually goes — never reads
   a face's ``.H`` at all. The 0-ULP anisotropic-scattering canary is
   green before and after F-0 by construction.
3. **Only END-of-chain adjoints are exposed, and there were none.**
   `[M]` ``grep -rn "analysis\.H\|reconstruction\.H" orpheus/`` returns
   exactly one hit and it is a docstring: **no production consumer of a
   face's** ``.H`` **existed**. The consumers arrive with the S6 adjoint
   gates, which is precisely why the metric had to be right before they
   land.

**What catches it now.** ``tests/numerics/test_frame.py`` grew a
``test_parseval_*`` family: the diagonal dressing pin (the installed
metric is :math:`1/G_{kk}` on live slots and exactly :math:`0` on dead
ones), the isometry over the six ``DIAGONAL`` sphere families, the
closure :math:`M^*=R/W` / :math:`R^*=W\,M`, the indicator
:math:`1/m_R` pin, the declared-vs-measured witness — and, the
load-bearing one, a **loaded-not-blind negative leg** that re-installs
the pre-F-0 continuum metric in-process and asserts the ratio is
:math:`\gg 1`. Without that leg the isometry gate's green would be
compatible with a gate that is merely *blind* to the metric
(``vv-principles`` #19: only the wrong-structure reading discriminates
loaded from blind).

Campaign 1 P7 (2026-08-30) extended the same discipline to the
``DENSE`` arm, which until then was pinned only by its *refusal*: the
four-mechanism dressing gate (a slab measure, a coarse product, a
coarse level-symmetric, and a non-angular partition-of-unity basis),
the isometry gate's new slab row, a second loaded-not-blind leg for
the dense dressing, and — the one that carries the phase's whole
argument — the **wrong-metric discriminator**, which reads the same
:math:`\psi` under three metrics and shows the best *diagonal*
candidate at :math:`1.806` where the matrix :math:`G^{+}` reads
:math:`1.000000000000`. See
:ref:`frame-parseval-dense-arm`.


The Petrov-Galerkin frame
=========================

The **general** discrete frame is Petrov-Galerkin: the *test* functions
that measure the residual need not equal the *trial* functions that
reconstruct it. The code mirrors this generality — the Galerkin case is a
*subclass* (:class:`~orpheus.numerics.frame.GalerkinFrame` **is-a**
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` with
``test is trial``) — so the Petrov-Galerkin frame is presented first, as
the general object, and the Galerkin frame (:ref:`the special case
<frame-galerkin-frame>`) as its symmetric specialisation.

In general
----------

The Petrov-Galerkin discipline is characterised by **test space
differs from trial space** — the
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` case. The
:math:`(R, M)` pair is built from two distinct bases —
:math:`\{e_k\}` for the trial space (the reconstruction basis, owned
by the :class:`~orpheus.numerics.basis.Basis`) and :math:`\{f_k\}`
for the test space (the explicit ``test_basis`` carried by the
frame):

.. math::
   :label: petrov-galerkin-construction

   (M g)_k \;=\; \langle f_k, g \rangle_V, \qquad
   R \, c \;=\; \sum_k c_k\,e_k.

.. vv-status: petrov-galerkin-construction documented

The pair satisfies :math:`M R = I_W` (the coefficient extraction
:math:`G^{-1} M` uses the *cross* Gram
:math:`G_{kj} = \langle f_k, e_j \rangle`), but :math:`M^* \ne R` —
the distinct test space makes the Hilbert adjoint distinct from the
reconstruction (an oblique, not canonical, dual).

The canonical Petrov-Galerkin pairs in reactor physics:

.. list-table:: Petrov-Galerkin pairs
   :header-rows: 1
   :widths: 22 24 28 26

   * - Use
     - Trial basis :math:`\{e_k\}`
     - Test basis :math:`\{f_k\}`
     - Reference
   * - Energy condensation
     - Indicator on broad group :math:`G`
     - Within-group spectrum
       :math:`\phi_g \cdot \mathbf{1}_{g \in G}`
     - Hébert 2009, §6.2
   * - Spatial homogenisation (reaction-rate)
     - Indicator on region :math:`R`
     - Region forward flux
       :math:`\phi \cdot \mathbf{1}_{i \in R}`
     - Smith 1986; Hébert 2009 §13
   * - Spatial homogenisation (eigenvalue-consistent)
     - Indicator on region :math:`R`
     - Region **bilinear pair**
       :math:`(\varphi^*\!\odot\varphi)\cdot\mathbf{1}_{i \in R}`
     - B&G 1970 §6.4h; Hébert 2009 §13
   * - Stochastic Galerkin
     - Polynomial-chaos basis (Hermite, Legendre)
     - Same basis under PCE inner product
     - Xiu & Karniadakis 2002

In each case the test basis encodes a **physical weighting** — the
importance of the fine-space slot from the solver's perspective — so
that the coarse coefficients faithfully preserve reaction rates /
flux-volume integrals / variance moments.

.. _petrov-galerkin-not-weighted-metric:

Posing the adjoint
------------------

It is tempting to absorb the test weight (:math:`\phi`,
:math:`\varphi^*`) into the *measure* and call the result an orthogonal
(Galerkin) projection in an :math:`L^2(\phi V)` metric. That fold is
legitimate **only** for the forward-flux, reaction-rate-only row of the
Petrov-Galerkin table above: there the test weight and the integrand
multiplier coincide (both are the forward flux :math:`\phi`), and the two
readings are the *same* map — the **Galerkin degenerate**. It **breaks**
for the eigenvalue-consistent row, whose preserved functional is the
**bilinear** form :math:`\langle \varphi^*, \Sigma\,\phi\rangle`: the
test weight is the **adjoint** :math:`\varphi^*`, the integrand is the
**forward** flux :math:`\phi` — different functions — so no single metric
on the measure reproduces it (folding either one into the metric
mis-weights the other). The discipline must therefore live on the **test
side** (the frame type), and the measure stays a fixed :math:`L^2`
metric; only then does the adjoint fall out naturally — the adjoint
problem swaps the test basis to :math:`\varphi^*`, and the oblique dual
:math:`M^* \ne R` (not the canonical dual) is exactly what that swap
needs.

This is why the architecture is as it is. The frame was in fact first
posed as a *Galerkin* projection with the flux folded into the volume
measure; it was the adjoint-weighted requirement — the need to keep
:math:`\keff` stationary, which first-order perturbation theory ties to
the **adjoint**-weighted residual, not the forward-weighted one — that
forced the re-posing as **Petrov-Galerkin**, with the weighting on an
explicit test basis rather than smuggled onto the measure. Folding the
solution into the metric is precisely the mistake the #268 ruling
forbids: *the measure carries the axis and the fixed* :math:`L^2`
*metric, never the discipline.*

The forward (Galerkin-degenerate) Petrov-Galerkin frames have since
shipped (P3 / P5): :meth:`Solution.homogenize
<orpheus.sn.solution.Solution.homogenize>` and
:meth:`Solution.condense <orpheus.sn.solution.Solution.condense>` build a
concrete :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` with an
explicit flux- / spectrum-weighted
:class:`~orpheus.numerics.basis.WeightedIndicatorBasis` test basis — the
first concrete instances landing exactly as a ``test_basis`` choice on
the existing mechanism, not a new mechanism. The forward-flux derivation
is worked in full in :ref:`sn-homogenization-petrov-galerkin-frame`
(:ref:`§2c <sn-spatial-homogenization>`). The non-degenerate
(:math:`\varphi^* \ne \varphi`) eigenvalue-consistent case **now ships
too** (P6, #281): the bilinear derivation
(:eq:`sn-homogenization-bilinear`,
:ref:`sn-homogenization-why-petrov-galerkin`) is realized under the
``adjoint=`` parameter, landing — as promised — as a ``test_basis``
weight (the bilinear pair :math:`\varphi^*\!\odot\varphi`) on the same
mechanism, its full taxonomy at
:ref:`frame-adjoint-weighted-seam`. This subsection fixes the
*architecture* of where the weighting lives.

.. _sn-spatial-homogenization:

Applied to spatial homogenization
---------------------------------

Once a fine-mesh solution :math:`\phi_{i,g}` is in hand, a coarse-mesh
model that **reproduces every reaction rate** of the fine model can be
built by collapsing the fine cross sections onto the coarse cells. This
is *spatial homogenization* — a domain operation on the solution, not a
solver step. It is the spatial sibling of energy *condensation* (group
collapse); the two together are the classical "smear the detail you
have resolved into effective constants for a coarser calculation" move
(Hébert, *Applied Reactor Physics*, §13 for space, §6.2 for energy).

.. note::

   This is the **space-only** slice (the spatial sibling of energy
   condensation). It is **dimension-agnostic** — 1-D and 2-D fine meshes
   flow through the one frame body, because the coarse cell-indicator
   basis and the fine volume measure are n-D (see
   :ref:`sn-homogenization-petrov-galerkin-frame`). Energy is *not*
   condensed — the group structure (:math:`eg`) carries through unchanged
   — and the coarse mesh must share the fine mesh's outer boundary with
   internal coarse edges aligned to fine-cell edges (each coarse cell is
   a contiguous union of fine cells). The asymmetry between
   homogenization and condensation, and *why* they return different
   types, is the subject of :ref:`sn-condense-homogenize-asymmetry`.

The defining property: reaction-rate preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Homogenization is defined by *what it must preserve*, not by an
averaging recipe chosen for convenience. The physical quantity a
transport calculation actually consumes is the **volume-integrated
reaction rate** in each region and group,

.. math::
   :label: sn-homogenization-fine-rate

   r_{R,g} \;=\; \sum_{i \in R} V_i\,\Sigma_{i,g}\,\phi_{i,g},

.. (vv-status rationale) Definitional identity: the fine-mesh
   volume-integrated reaction rate. A derivation-decomposition step of
   the rate-preservation claim (:eq:`sn-homogenization-rate-preservation`),
   which is the verifiable solver claim (L0 gate
   ``tests.sn.test_homogenization``); this is its premise, not a separate
   claim.
.. vv-status: sn-homogenization-fine-rate documented

the sum over the fine cells :math:`i` contained in coarse cell
:math:`R`, with :math:`V_i` the fine-cell volume,
:math:`\phi_{i,g}` the converged fine :term:`scalar flux`, and
:math:`\Sigma_{i,g}` the fine cross section for whatever channel (total,
capture, fission, …) is in question. The coarse model carries one
effective cross section :math:`\Sigma_{R,g}` and one region flux
:math:`\Phi_{R,g}` per cell; for it to reproduce the fine rate it must
satisfy

.. math::
   :label: sn-homogenization-rate-preservation

   \Sigma_{R,g}\,\Phi_{R,g}
   \;=\;
   \sum_{i \in R} V_i\,\Sigma_{i,g}\,\phi_{i,g}.

Equation :eq:`sn-homogenization-rate-preservation` is the **only**
constraint homogenization imposes; everything below is derived from it.

The coarse region flux is fixed first, by the requirement that the
*production-free* particle inventory match — the region flux is the
flux integrated over the region,

.. math::
   :label: sn-homogenization-region-flux

   \Phi_{R,g} \;=\; \sum_{i \in R} V_i\,\phi_{i,g}.

.. (vv-status rationale) Representational identity: the coarse region
   flux is the flux integrated over the region. A definition consumed by
   the rate-preservation claim (:eq:`sn-homogenization-rate-preservation`),
   not an independent solver claim.
.. vv-status: sn-homogenization-region-flux documented

Substituting :eq:`sn-homogenization-region-flux` into
:eq:`sn-homogenization-rate-preservation` and solving for the effective
cross section gives the **flux·volume-weighted average**

.. math::
   :label: sn-homogenization-vector-collapse

   \Sigma_{R,g}
   \;=\;
   \frac{\sum_{i \in R} V_i\,\phi_{i,g}\,\Sigma_{i,g}}
        {\sum_{i \in R} V_i\,\phi_{i,g}}
   \;=\;
   \frac{\sum_{i \in R} w_{i,g}\,\Sigma_{i,g}}
        {\sum_{i \in R} w_{i,g}},
   \qquad
   w_{i,g} \equiv V_i\,\phi_{i,g}.

.. (vv-status rationale) Derivation-decomposition step: the
   flux·volume-weighted collapse obtained by solving the
   rate-preservation identity (:eq:`sn-homogenization-rate-preservation`)
   for the effective cross section. The verifiable content is the L0
   rate-preservation gate; this is the algebraic rearrangement, not a
   separate claim.
.. vv-status: sn-homogenization-vector-collapse documented

The weight :math:`w_{i,g} = V_i\,\phi_{i,g}` is the **flux·volume**
of the fine cell — the same quantity that appears in both the numerator
(rate) and the denominator (flux integral), so the average is a genuine
convex combination of the fine values: :math:`\Sigma_{R,g}` is bracketed
by the region's fine-cell extremes
:math:`\min_{i\in R}\Sigma_{i,g} \le \Sigma_{R,g} \le \max_{i\in R}\Sigma_{i,g}`.
This is *not* a separate design choice — it falls straight out of
:eq:`sn-homogenization-rate-preservation`. Choosing any other weight
(volume-only, unweighted) would break rate preservation at material
interfaces, which is exactly the regime homogenization exists to handle.

The flux·volume weight :math:`w_{i,g}` is the operation's whole signal,
and it is *not* a free parameter: it is the **test weighting** that rate
preservation forces. That is what the next subsection makes precise —
homogenization is the coefficient extraction of a **Petrov-Galerkin**
frame whose *test* basis is the flux-weighted cell indicator and whose
*trial* basis is the plain cell indicator, and ORPHEUS realises it by
routing through the one discrete
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`, not a bespoke
membership matmul (see :ref:`sn-homogenization-petrov-galerkin-frame`).

The vector channels — total, capture, leakage-loss
(:math:`\Sigma_L`), fission, and production
(:math:`\nu\Sigma_f`) — each collapse through
:eq:`sn-homogenization-vector-collapse` with the *same* per-:math:`(R,g)`
weight. A group with zero region flux
(:math:`\Phi_{R,g} = 0`) has no reaction rate to preserve, so its
effective cross section is set to zero — the :math:`0/0` of
:eq:`sn-homogenization-vector-collapse` resolved by the only physically
meaningful value.

The matrix channels weight by the *source* group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattering matrices :math:`\Sigma_{s,\ell}[g',g]` (one per Legendre
order :math:`\ell`) and the :math:`(n,2n)` matrix
:math:`\Sigma_{2n}[g',g]` carry **two** group indices, stored
``[g_from, g_to]`` (the ORPHEUS scattering convention — see
:ref:`theory-cross-section-data`). A naïve reuse of the scalar weight
would be wrong: the reaction rate that an out-scatter channel
:math:`g' \to g` actually produces is

.. math::
   :label: sn-homogenization-scatter-rate

   r_{R}^{\,g'\to g}
   \;=\;
   \sum_{i \in R} V_i\,\phi_{i,g'}\,\Sigma_{s,\ell,i}[g',g],

.. (vv-status rationale) Definitional identity: the out-scatter rate of
   a matrix channel, driven by the source-group flux. The premise that
   forces the source-group weighting; the verifiable claim is the L0
   matrix-channel rate-preservation gate
   (``test_rate_preservation_scattering_and_n2n``).
.. vv-status: sn-homogenization-scatter-rate documented

driven by the population of the **source** group :math:`g'` — the group
whose flux scatters *out*. The number of :math:`g'\to g` events scales
with how many particles are *in* :math:`g'`, i.e. with
:math:`\phi_{i,g'}`, not with the sink-group flux
:math:`\phi_{i,g}`. Rate preservation
:eq:`sn-homogenization-rate-preservation` therefore demands that the
effective matrix entry be weighted by the source-group flux·volume:

.. math::
   :label: sn-homogenization-matrix-collapse

   \Sigma_{s,\ell,R}[g',g]
   \;=\;
   \frac{\sum_{i \in R} V_i\,\phi_{i,g'}\,\Sigma_{s,\ell,i}[g',g]}
        {\sum_{i \in R} V_i\,\phi_{i,g'}}
   \;=\;
   \frac{\sum_{i \in R} w_{i,g'}\,\Sigma_{s,\ell,i}[g',g]}
        {\Phi_{R,g'}},

.. (vv-status rationale) Derivation-decomposition step: the
   source-group-weighted matrix collapse — the matrix-channel analogue
   of :eq:`sn-homogenization-vector-collapse`. The verifiable content is
   the L0 matrix-channel rate-preservation gate (which catches a
   g_from↔g_to swap, vv Mode 2); this is the algebraic form, not a
   separate claim.
.. vv-status: sn-homogenization-matrix-collapse documented

so the weight :math:`w_{i,g'} = V_i\,\phi_{i,g'}` rides the **first**
(``g_from``) matrix axis. In the code this falls out of the *test side*
for free: the :class:`~orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis`
carries the per-group flux :math:`\phi` as its test weight, and its
**leading-aligned broadcast** aligns that weight's group axis to the
*first* trailing (``g_from``) axis of whatever field it analyses — a
vector channel weights elementwise, a ``[g_from, g_to]`` matrix channel
weights by its source group — *before* the region integral. The
:math:`1/\Phi_{R,g'}` normalisation is the frame's diagonal Gram
(:meth:`FrameBase.project <orpheus.numerics.frame.FrameBase.project>`'s
:meth:`FunctionSpace.apply_inverse_metric
<orpheus.numerics.space.FunctionSpace.apply_inverse_metric>`), whose
:math:`\Phi_{R,g'}` rides the ``g_from`` axis and broadcasts over the
trailing ``g_to`` axis. The :math:`(n,2n)` channel collapses identically
— same source-group weighting on its ``[g_from, g_to]`` layout. Both ride
the *same* ``sigma_frame`` because
:meth:`MaterialXSField.project_through
<orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through>`
routes every rate-bearing channel through it. The mechanism that carries
this — the discrete :class:`~orpheus.numerics.frame.PetrovGalerkinFrame`
— is derived in :ref:`sn-homogenization-petrov-galerkin-frame`.

.. warning::

   The source-group weighting is the subtle point of the whole
   operation and a textbook variable-swap trap (vv-principles failure
   **Mode 2**, ``SigS`` vs ``SigS^T``). Weighting the matrix collapse by
   the **sink** group :math:`g` instead of the **source** group
   :math:`g'` produces an effective scattering matrix that does *not*
   preserve the out-scatter rate — a bug that is invisible on a
   single-material or flat-flux region (where every group's weight is
   proportional) and only bites on a heterogeneous, multi-group region
   with an asymmetric flux spectrum. The regression gate
   (:ref:`sn-homogenization-verification`) catches it precisely because
   its reference loop weights by ``g_from`` *explicitly*.

The fission spectrum is production-weighted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The emission spectrum :math:`\chi_g` is **not** a reaction rate — it is
a probability distribution (a simplex,
:math:`\sum_g \chi_g = 1`; see :eq:`emission-spectrum-simplex` in
:ref:`theory-cross-section-data`). Flux·volume-weighting it would not
preserve anything physical and could leave the simplex. The
rate-preserving choice is to weight :math:`\chi` by each fine cell's
**fission production rate**

.. math::
   :label: sn-homogenization-production-weight

   p_i \;=\; \sum_g \nu\Sigma_{f,i,g}\,\phi_{i,g}\,V_i,

.. (vv-status rationale) Definitional identity: the per-cell fission
   production rate used as the χ-mixing weight. A premise of the
   χ-collapse, not a separate solver claim; the simplex/null-law content
   it feeds is verified by the data-layer ``Mixture`` invariant tests.
.. vv-status: sn-homogenization-production-weight documented

so that the homogenized spectrum is the production-weighted convex
average

.. math::
   :label: sn-homogenization-chi-collapse

   \chi_{R,g}
   \;=\;
   \frac{\sum_{i \in R} p_i\,\chi_{i,g}}
        {\sum_{i \in R} p_i}.

.. (vv-status rationale) Representational identity: the
   production-weighted convex average of the fine emission spectra — the
   spatial analogue of the multi-fissile χ_mix
   (:eq:`emission-spectrum-chi-mix`). The simplex/null law it must
   satisfy is the data-layer invariant verified by
   ``test_homogenized_chi_is_simplex`` + ``Mixture.__post_init__``; this
   label is the mixing formula, not a separate solver claim.
.. vv-status: sn-homogenization-chi-collapse documented

Because each fine :math:`\chi_i` is a probability simplex and the
weights :math:`p_i \ge 0`, :math:`\chi_R` is a **convex combination of
simplices, hence itself a simplex** — it is exactly the spatial analogue
of the production-weighted multi-fissile mixing
:math:`\chi_{\rm mix}` of :eq:`emission-spectrum-chi-mix` (where the
weights are per-isotope production rather than per-cell). The simplex /
null law is re-validated when the homogenized
:class:`~orpheus.data.macro_xs.mixture.Mixture` is constructed
(:meth:`Mixture.__post_init__
<orpheus.data.macro_xs.mixture.Mixture.__post_init__>`); a coarse cell
with no fissile fine cells (:math:`\sum_i p_i = 0`) gets
:math:`\chi_R = 0`, the null-law branch.

Balance is preserved cell-by-cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The definitional total-XS balance every :class:`Mixture` carries,

.. math::
   :label: sn-homogenization-balance

   \Sigma_t
   \;=\;
   \Sigma_c + \Sigma_L + \Sigma_f
   + \operatorname{rowsum}(\Sigma_{s0})
   + \operatorname{rowsum}(\Sigma_{2n}),

.. (vv-status rationale) Literature-transcribed definition: the total-XS
   balance identity every Mixture carries (the same identity as
   :eq:`sigT-computed`). Its preservation under homogenization is gated
   by ``test_homogenized_materials_balance`` (which calls
   ``Mixture.assert_balanced``); the identity itself is a data-layer
   definition, not an SN solver claim.
.. vv-status: sn-homogenization-balance documented

survives the collapse **cell-by-cell** when the fine materials balance.
The argument is one line once the weighting is understood: fix a coarse
cell :math:`R` and group :math:`g`. Every *removal* channel on the
left- and right-hand sides of :eq:`sn-homogenization-balance` —
:math:`\Sigma_t,\ \Sigma_c,\ \Sigma_L,\ \Sigma_f`, and the row-sums
:math:`\sum_{g'}\Sigma_{s0}[g,g']`, :math:`\sum_{g'}\Sigma_{2n}[g,g']`
— is a *removal from group* :math:`g`, so each collapses with the
**same** weight :math:`w_{i,g} = V_i\phi_{i,g}` (the row-sum of a
``[g_from, g_to]`` matrix over its sink index :math:`g'` is a removal
*from* the source group :math:`g`, weighted by :math:`g`'s flux — the
source-group weighting of :eq:`sn-homogenization-matrix-collapse`
restricted to a diagonal-of-the-source row). Because every term shares
the one weight, the homogenized balance is the *same convex average* of
the fine balances:

.. math::
   :label: sn-homogenization-balance-preservation

   \Sigma_{t,R,g}
   - \Big(\Sigma_{c,R,g} + \Sigma_{L,R,g} + \Sigma_{f,R,g}
   + \operatorname{rowsum}(\Sigma_{s0,R})_g
   + \operatorname{rowsum}(\Sigma_{2n,R})_g\Big)
   \;=\;
   \frac{\sum_{i\in R} w_{i,g}\,\big(\text{fine balance residual}_{i,g}\big)}
        {\sum_{i\in R} w_{i,g}}
   \;=\; 0,


.. implements:: sn-homogenization-balance-preservation
   :by: orpheus.data.macro_xs.mixture.Mixture.assert_balanced

   **Implemented by** 4 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: sn-homogenization-balance-preservation
   :by: orpheus.data.macro_xs.mixture.Mixture.balance_residual

.. implements:: sn-homogenization-balance-preservation
   :by: orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through

.. implements:: sn-homogenization-balance-preservation
   :by: orpheus.derivations.common.homogenization.derive_balance_tradeoff

since each fine residual is zero. No separate "rebalance the homogenized
total" step is needed — preservation is automatic, and the homogenized
``Mixture`` passes :meth:`Mixture.assert_balanced
<orpheus.data.macro_xs.mixture.Mixture.assert_balanced>`. (Had the
vector channels and the matrix row-sums collapsed with *different*
weights, the balance would break — which is another way of seeing why
the source-group weighting of :eq:`sn-homogenization-matrix-collapse`
is forced, not chosen.)

.. _sn-homogenization-petrov-galerkin-frame:

Homogenization is a Petrov-Galerkin projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above derives the flux·volume average from rate
preservation. This subsection takes the second view — the one that
fixes *what kind of operator* homogenization is, and therefore *how it
is implemented*. The answer is a single sentence that the rest of the
subsection unpacks:

  Homogenization is the coefficient extraction :math:`G^{-1} M` of a
  **Petrov-Galerkin** frame: the *trial* basis is the plain coarse-cell
  indicator :math:`\mathbf{1}_R`, the *test* basis is the
  flux-weighted indicator :math:`\chi_R = \phi\,\mathbf{1}_R`, and the
  measure is the bare geometric volume measure :math:`\mu_V`.

This is not decoration. It is the reason the production code routes
:meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`
through the *same* discrete :class:`~orpheus.numerics.frame.FrameBase`
abstraction that carries SN anisotropic-scattering moment projection —
one mechanism for every fine→coarse change of representation (Cardinal
Rule 2, single source of truth), instead of a bespoke membership
matmul per method. It is the consumer the discrete-frame theory page
points at as the headline **Petrov-Galerkin** instance
(Issue #268); the test functions differ
from the trial functions (:math:`\chi_R = \phi\,\mathbf{1}_R \ne
\mathbf{1}_R`), so the discipline is genuinely Petrov-Galerkin, carried
by the frame **type**
(:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`).

.. warning::

   **This corrects an earlier draft of this section.** A previous
   version argued homogenization was the ":math:`L^2(\phi V)`-orthogonal
   **Galerkin** projection" — that the flux multiplier could be folded
   into the *measure* (read :math:`\langle\Sigma,\phi\,\mathbf{1}_R
   \rangle_{\mathrm{d}V} = \langle\Sigma,\mathbf{1}_R\rangle_{\phi V}`),
   making test and trial the *same* span in a flux-weighted metric. That
   reading is **forward-flux, reaction-rate-only**, and it
   structurally breaks for the eigenvalue-consistent homogenization
   reactor physics actually requires (see
   :ref:`sn-homogenization-why-petrov-galerkin` below). Folding the
   solution into the metric is precisely the mistake the #268 ruling
   forbids: *the measure carries the axis and the fixed* :math:`L^2`
   *metric, never the discipline.* The flux is a **test-weighting the
   solution emits**, living on the test side — the frame type — not on
   the geometry's measure.

The trial space, the test space, and the cross-Gram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\Sigma(x)` be the fine cross-section field — a function on
the spatial domain, piecewise-constant on the *fine* cells. The coarse
model can only carry one value per coarse cell :math:`R`, so the
**trial** space — where the answer lives — is

.. math::
   :label: sn-homogenization-coarse-space

   W \;=\; \operatorname{span}\{\mathbf{1}_R\}_R,
   \qquad
   \mathbf{1}_R(x) =
   \begin{cases} 1 & x \in R, \\ 0 & \text{otherwise,} \end{cases}

.. (vv-status rationale) Structural/representational identity: names the
   coarse trial space as the span of the coarse-cell indicators (the P0
   space). The implementing object is
   :class:`~orpheus.numerics.basis.IndicatorBasis`; the verifiable content
   is the membership-table / Gram bit-identity gated by
   ``tests.numerics.test_indicator_basis``, not a solver claim.
.. vv-status: sn-homogenization-coarse-space documented

the span of the **coarse-cell indicators** (the piecewise-constant /
P0 / box space; Brenner & Scott 2008 §3.4). A Galerkin projection would
*test* the residual against these same trial functions. Homogenization
does not: rate preservation forces the residual to be tested against the
**flux-weighted** indicators

.. math::
   :label: sn-homogenization-test-functions

   \chi_R(x) \;=\; \phi(x)\,\mathbf{1}_R(x),

.. (vv-status rationale) Structural/representational identity: names the
   Petrov-Galerkin TEST functions as the flux-weighted coarse-cell
   indicators χ_R = φ·1_R. The implementing object is
   :class:`~orpheus.numerics.basis.WeightedIndicatorBasis`; the verifiable
   content is the weighted-analysis bit-identity gated by
   ``tests.numerics.test_weighted_indicator_basis`` and the Mode-11
   routing sentinel, not a solver claim.
.. vv-status: sn-homogenization-test-functions documented

— a genuinely different basis from the trial :math:`\mathbf{1}_R`. With
test :math:`\ne` trial, the projection is **Petrov-Galerkin**: the
coarse coefficients are the solution of the Petrov-Galerkin normal
equations (test the residual against every test function, in the bare
geometric metric :math:`\mu_V` = weight :math:`V`),

.. math::
   :label: sn-homogenization-normal-equations

   \big\langle \chi_R,\; \Sigma - \Sigma_W \big\rangle_{V}
   \;=\; 0
   \quad \forall R
   \;\;\Longleftrightarrow\;\;
   c_R \;=\;
   \frac{\langle \chi_R,\, \Sigma \rangle_{V}}
        {\langle \chi_R,\, \mathbf{1}_R \rangle_{V}}
   \;=\;
   \frac{\sum_{i\in R} V_i\,\phi_{i,g}\,\Sigma_{i,g}}
        {\sum_{i\in R} V_i\,\phi_{i,g}},

.. (vv-status rationale) Derivation-decomposition step: the
   Petrov-Galerkin normal equations for the P0 projection with a
   flux-weighted test basis, whose solution IS the flux·volume collapse
   (:eq:`sn-homogenization-vector-collapse`). The verifiable content is
   the L0 rate-preservation gate plus the φV-vs-dV discriminator
   (``test_homogenization_is_flux_weighted_not_volume_weighted``); this
   is the projection-theoretic reading, not a separate claim.
.. vv-status: sn-homogenization-normal-equations documented

where :math:`\langle \chi_R, f\rangle_V = \int_R \phi\,f\,\mathrm{d}V`
is the flux-weighted region integral the test functions induce. Because
the indicators (trial *and* test) have **disjoint support**, the
**cross-Gram**

.. math::
   :label: sn-homogenization-cross-gram

   G_{RS} \;=\; \langle \chi_R,\, \mathbf{1}_S \rangle_{V}
   \;=\; \delta_{RS}\,\sum_{i\in R} V_i\,\phi_{i,g}
   \;=\; \delta_{RS}\,\Phi_{R,g}

.. (vv-status rationale) Structural identity: the cross-Gram of the
   homogenisation Petrov-Galerkin frame is diagonal (disjoint indicator
   supports), its diagonal being the region flux integral
   :eq:`sn-homogenization-region-flux`. The diagonality is exercised by the
   diagonal-Gram fast path in the L0 rate-preservation gate
   (``test_homogenization_is_flux_weighted_not_volume_weighted`` + the Mode-11
   ``apply_inverse_metric`` routing sentinel). A derivation-decomposition
   structural identity, not a separate solver claim.
.. vv-status: sn-homogenization-cross-gram documented

is **diagonal**, so the normal equations decouple cell-by-cell and each
coefficient is exactly the flux·volume average
:eq:`sn-homogenization-vector-collapse`. The denominator is the region
mass :math:`m_R = G_{RR} = \Phi_{R,g}` — the region flux integral
:eq:`sn-homogenization-region-flux` *is* the diagonal of the
cross-Gram. (Contrast the spherical-harmonic
:class:`~orpheus.numerics.frame.GalerkinFrame`, whose Gram is the
*symmetric* :math:`\langle Y_k, Y_j\rangle = \delta_{kj}/(2\ell+1)`
because there test :math:`=` trial; here the two factors of the Gram are
*different* bases, but disjoint support still diagonalizes it.)

**The test weighting is derived, not chosen.** Had the residual been
tested against the *plain* indicators :math:`\mathbf{1}_R` (the Galerkin
choice, test :math:`=` trial in the bare :math:`\mu_V` metric) the
projection would have been the **volume average**
:math:`\sum_i V_i \Sigma_i / \sum_i V_i`, which does *not* preserve the
reaction rate across a flux gradient. Matching rate preservation
:eq:`sn-homogenization-rate-preservation` is what *forces* the test
functions to be flux-weighted (:math:`\chi_R = \phi\,\mathbf{1}_R`)
rather than the plain :math:`\mathbf{1}_R`. This is the load-bearing
discriminator the regression gate
``test_homogenization_is_flux_weighted_not_volume_weighted`` pins: a
coarse region spanning a vacuum→reflective flux tilt over two materials
makes the flux-weighted and volume-only effective :math:`\Sigma_t`
numerically distinct, and production *must* match the flux-weighted one.

.. _sn-homogenization-why-petrov-galerkin:

Why Petrov-Galerkin and not Galerkin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The flux-weighted projection *can* be written as a Galerkin projection
in a flux-weighted metric — fold :math:`\phi` from the test function
into the measure,

.. math::
   :label: sn-homogenization-metric-fold

   \big\langle \phi\,\mathbf{1}_R,\; \Sigma \big\rangle_{V}
   \;=\;
   \big\langle \mathbf{1}_R,\; \Sigma \big\rangle_{\phi V},

.. (vv-status rationale) Structural identity: the metric-fold that
   re-expresses the forward (φ*=φ) Petrov-Galerkin projection as a
   Galerkin projection in the L²(φV) metric. It is exact for the
   forward-flux case ONLY and is the convenience the #268 ruling rejects
   as the general framing (it folds the solution into the metric).
   Reframes the operator type; the numerical content is the same
   rate-preservation gate, not a new solver claim.
.. vv-status: sn-homogenization-metric-fold documented

making test and trial the *same* span :math:`\{\mathbf{1}_R\}` in the
:math:`L^2(\phi V)` metric. **This is a forward-only convenience, not
the structure.** It works here only because the test weight equals the
trial-side solution (:math:`\phi^* = \phi`) — the *forward* degenerate.
The homogenization reactor physics actually requires is
**eigenvalue-consistent**: the effective cross sections must keep the
multiplication factor :math:`\keff` stationary, and by first-order
perturbation theory :math:`\keff` is stationary with respect to the
**adjoint-weighted** residual. The functional that must be preserved is
therefore the **bilinear** form

.. math::
   :label: sn-homogenization-bilinear

   \big\langle \varphi^*,\, \Sigma\,\varphi \big\rangle,
   \qquad
   \Sigma_R \;=\;
   \frac{\int_R \varphi^*\,\Sigma\,\varphi\;\mathrm{d}V}
        {\int_R \varphi^*\,\varphi\;\mathrm{d}V},


.. implements:: sn-homogenization-bilinear
   :by: orpheus.data.macro_xs.mixture.Mixture.condense

   **Implemented by** 7 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: sn-homogenization-bilinear
   :by: orpheus.numerics.frame.PetrovGalerkinFrame

.. implements:: sn-homogenization-bilinear
   :by: orpheus.sn.solution.Solution.condense

.. implements:: sn-homogenization-bilinear
   :by: orpheus.sn.solution.Solution.homogenize

.. implements:: sn-homogenization-bilinear
   :by: orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through_bilinear

.. implements:: sn-homogenization-bilinear
   :by: orpheus.derivations.common.homogenization.collapse_rules

.. implements:: sn-homogenization-bilinear
   :by: orpheus.derivations.common.homogenization.vector_bilinear_rule

.. (Wired P6, #281 — no vv-status sentinel.) This bilinear identity —
   the eigenvalue-consistent (adjoint-weighted) effective cross section
   that keeps k_eff first-order stationary — is now a VERIFIED solver
   claim, not documented-only. Solution.homogenize / Solution.condense
   build the collapse under the ``adjoint=`` parameter, and the
   full-taxonomy discriminator gates C1 (tests.sn.test_homogenization)
   and C4 (tests.sn.test_condensation) stack
   verifies("sn-homogenization-bilinear") against structurally-
   independent per-region hand rules. The label is covered by tests, so
   it carries no ``documented`` sentinel.

with **test** functions :math:`\varphi^*\cdot\mathbf{1}_R` and
**trial** functions :math:`\varphi\cdot\mathbf{1}_R` that are now
genuinely distinct (:math:`\varphi^* \ne \varphi` away from a
self-adjoint problem). There is **no metric in which test equals
trial** — the map is irreducibly Petrov-Galerkin, :math:`M^* \ne R`.
The forward homogenization this slice ships is the **Galerkin
degenerate** of that map (:math:`\varphi^* = \varphi`, the flux is its
own adjoint weighting): it is a *legal* Galerkin reading because of the
coincidence :math:`\varphi^* = \varphi`, but the coincidence is *not*
the structure, so the honest framing — the one that survives the lift to
:math:`\varphi^* \ne \varphi` — is Petrov-Galerkin. ORPHEUS therefore
builds it as a :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` with
an explicit flux-weighted test basis, *not* a
:class:`~orpheus.numerics.frame.GalerkinFrame` with a flux-weighted
measure. The adjoint-weighted (:math:`\varphi^* \ne \varphi`) case
:eq:`sn-homogenization-bilinear` **now ships** (P6, #281): the ratified
``homogenize(..., adjoint=...)`` / ``condense(..., adjoint=...)``
parameter reads the role-typed ``AdjointSolution`` that
:func:`~orpheus.sn.solver.solve_sn_adjoint` returns (#276 A4/A5; see
:ref:`sn-adjoint`) and builds the eigenvalue-consistent collapse, with
its full-taxonomy gate battery. This section sets it up as the
non-degenerate sibling the forward case descends from; the landed
taxonomy — the per-channel collapse rules, the balance trade-off, the
exact angular pairing, and the Bell & Glasstone energy-axis convention —
is the capstone seam :ref:`frame-adjoint-weighted-seam`.

.. note::

   The **forward** degenerate (:math:`\varphi^* = \varphi`) is
   metric-fold invisible: forward homogenization produces the same
   numbers whether read as Petrov-Galerkin or as Galerkin in the
   :math:`L^2(\phi V)` metric (:eq:`sn-homogenization-metric-fold` is an
   exact identity when :math:`\varphi^* = \varphi`). The
   **adjoint-weighted** case (:math:`\varphi^* \ne \varphi`), now live,
   is *not* invisible — it produces genuinely different effective cross
   sections (the C1 / C4 discriminator gates assert the bilinear collapse
   differs from the forward one on **every** channel of a
   tilted-importance fixture). What remains purely an **architecture**
   distinction is the *implementation*: writing the discipline on the
   frame *type* (an explicit test basis) rather than on the *measure*
   (a flux-folded metric) is what let the adjoint arm land as a change of
   the test *weight* — the bilinear pair :math:`\varphi^*\!\odot\varphi`
   (and its exact angular / per-pair refinements) — rather than a
   re-derivation. The Mode-11 routing sentinels (C3 / C5) pin that the
   derived weight genuinely reaches the test side, so a regression that
   silently re-folded it into the metric, or swapped it for a bare
   :math:`\varphi^*`, would be caught even where it barely moves a number.

The measure carries the axis, never the discipline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The implementation key is that the flux weight rides the **test basis**,
and the :class:`~orpheus.numerics.measure.DiscreteMeasure` carries only
the bare geometric volume :math:`\mu_V`. The frame factors the test
functional into a geometric measure and a solution multiplier,

.. math::
   :label: sn-homogenization-radon-nikodym

   \langle \chi_R, f\rangle_V \;=\; \int_R \phi\,f\;\mathrm{d}\mu_V,
   \qquad
   \chi_R \;=\; \phi\cdot\mathbf{1}_R,

.. (vv-status rationale) Structural identity: the test functional splits
   into the geometric base measure μ_V and the flux density φ carried on
   the test basis. It is the design rationale for carrying φ as the test
   weight (an integrand multiplier) rather than as ng separate measures
   or as a metric on the measure; the verifiable content is the measure /
   weighted-basis bit-identity gates and the Mode-11 routing sentinel,
   not a solver claim.
.. vv-status: sn-homogenization-radon-nikodym documented

i.e. the **geometric base measure** :math:`\mu_V` (group-independent,
the coarse/fine mesh's :attr:`volume_measure
<orpheus.transport.mesh.material_mesh.MaterialMesh.volume_measure>` — a
:class:`~orpheus.numerics.measure.DiscreteMeasure`) is multiplied at
integration time by the flux :math:`\phi` *carried on the test basis*.
The code carries exactly this split: the
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` binds the trial
:class:`~orpheus.numerics.basis.IndicatorBasis` and the *bare*
group-independent :math:`\mu_V`, and the flux enters through the
**test basis**
:class:`~orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis`,
whose :meth:`analyze
<orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis.analyze>`
folds the per-group flux into the integrand on a trailing tensor axis —
``test.analyze(phi * channel_fine, …)`` — so the whole group structure
rides one frame.

This is *why* the test weight is **not** smuggled onto the measure.
:class:`~orpheus.numerics.measure.DiscreteMeasure`'s ``weights`` array
stays **1-D** (one mass per atom) and group-independent; a per-group
:math:`\mu_{\phi V}` would be :math:`n_g` distinct measures, and — worse
— a *measure*-borne flux weight is exactly the metric-fold the #268
ruling forbids as the general framing: it works for forward homogenization
and breaks under :math:`\varphi^* \ne \varphi`. Keeping :math:`\phi` on
the test basis instead of the measure forces the correct reading:
:math:`\phi` is a test-weighting the *solution* emits, not a property of
the geometry. The geometry (the mesh) owns one measure :math:`\mu_V`; the
solution owns the flux :math:`\phi`; the *frame type* (Petrov-Galerkin,
with its explicit test basis) carries the discipline. The
:class:`~orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis`
is **test-only** by construction — its :meth:`evaluate
<orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis.evaluate>`
is the *weight-free* geometric membership (the weight is an *analysis*
weight, not a tabulation), and its synthesis-side operations *raise*
(the Petrov-Galerkin reconstruction is purely trial-side; building a
weighted synthesis before a consumer exists would make a half-consistent
basis).

The mesh yields the basis; it does not inherit it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The coarse trial space :eq:`sn-homogenization-coarse-space` is realised
by a **new** concrete numerics basis,
:class:`~orpheus.numerics.basis.IndicatorBasis` — the second concrete
:class:`~orpheus.numerics.basis.Basis` after
:class:`~orpheus.numerics.basis.SphericalHarmonicBasis`, and the
piecewise-constant (P0 / characteristic-function) analogue of it. The
coarse :class:`~orpheus.geometry.mesh.Mesh1D` **yields** this view via
:meth:`coarse.indicator_basis() <orpheus.geometry.mesh.Mesh1D.indicator_basis>`,
exactly symmetric with how it already yields
:meth:`coarse.volume_measure <orpheus.geometry.mesh.Mesh1D.volume_measure>`.

The mesh is **not** a :class:`~orpheus.numerics.basis.Basis` subclass,
and the reason is a clean role separation: a
:class:`~orpheus.numerics.basis.Basis` is the **measure-free** half of a
frame, while a mesh **carries the volume measure**. A mesh that *were* a
basis would conflate the two roles of the frame pair (the
discipline-free trial side and the measured test side) into one object.
So the mesh yields *both* views — its measure-free indicator basis and
its volume measure — and the
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` binds them together
with the flux-weighted test basis. The yielded
:class:`~orpheus.numerics.basis.IndicatorBasis` is **geometry-free**: it
holds only the per-axis edge arrays, so :mod:`orpheus.numerics` carries
no dependency on :mod:`orpheus.geometry`. Its
:meth:`evaluate <orpheus.numerics.basis.IndicatorBasis.evaluate>` builds
the :math:`(n_{\rm fine} \times n_{\rm coarse})` one-hot **membership
table** :math:`T[i,R] = \mathbf{1}_R(x_i)` by a per-axis
``searchsorted`` followed by :func:`numpy.ravel_multi_index` in ``"ij"``
order — the *same* flat-cell ordering the volume measure uses for its
nodes, so the table column index and the measure node index agree by
construction in any dimension (no 1-D special case in the membership
machinery). This is what makes homogenization **dimension-agnostic**:
:meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`
flattens its ``(ng, *spatial)`` flux to ``(n_fine, ng)`` in the same
``"ij"`` order and a 1-D or 2-D mesh flows through the one frame body
(pinned end-to-end by ``test_homogenize_2d_rate_preservation``).

The coefficient-extraction verb and its normalisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The homogenization map is the frame's **coefficient-extraction verb**
:meth:`frame.project <orpheus.numerics.frame.FrameBase.project>`
= :math:`G^{-1} M`:

.. math::
   :label: sn-homogenization-frame-projector

   \Sigma_R \;=\; \big(G^{-1} \circ M\big)\,\Sigma,
   \qquad
   M = \text{analysis} \;\big(\textstyle\int_R \phi\,\cdot\;\mathrm{d}V\big),\;\;
   G^{-1} = \operatorname{diag}(1/\Phi_R),

.. (vv-status rationale) Structural/representational identity: the
   homogenization map as the inverse-Gram ∘ analysis coefficient
   extraction (``FrameBase.project``) of the Petrov-Galerkin frame.
   Each factor is a Frame primitive whose bit-identity is gated by
   ``tests.numerics.test_indicator_basis`` /
   ``tests.numerics.test_weighted_indicator_basis`` /
   ``tests.numerics.test_frame``; the Mode-11 sentinel
   ``test_homogenize_routes_through_the_petrov_galerkin_frame`` pins that
   ``homogenize`` actually calls them. Not a separate solver claim.
.. vv-status: sn-homogenization-frame-projector documented

read right-to-left:

#. **analysis** :math:`M` = ``frame.analysis.apply``, which delegates to
   the *test* basis's weighted analysis: the flux-weighted region
   integral :math:`(M\Sigma)_R = \sum_{i\in R} V_i\,\phi_i\,\Sigma_i =
   \int_R \phi\,\Sigma\,\mathrm{d}V` — the region reaction rate. The
   diagonal of the cross-Gram is recovered by the *same* face applied to
   the constant field, :math:`(M\,\mathbf 1)_R = \sum_{i\in R}
   V_i\,\phi_i = \Phi_R` (a single ``analysis ∘ reconstruction`` probe
   of the all-ones coefficient vector; the off-diagonals are
   structurally zero, so the row-sum IS the diagonal — see
   :attr:`frame.gram_inverse
   <orpheus.numerics.frame.FrameBase.gram_inverse>`, whose
   :attr:`~orpheus.numerics.frame.CrossGramInverse.diagonal` IS that
   probe).
#. **inverse Gram** :math:`G^{-1} = \operatorname{diag}(1/\Phi_R)` =
   :meth:`FunctionSpace.apply_inverse_metric
   <orpheus.numerics.space.FunctionSpace.apply_inverse_metric>` on a
   coarse coefficient space whose installed metric is the diagonal Gram
   :math:`\Phi_R = M\,\mathbf 1`. The normalisation :math:`1/\Phi_R` is
   **measure-dependent** (the region mass changes with the flux weight),
   *unlike* the spherical-harmonic :math:`2\ell+1` factor which is
   analytic and measure-free. A measure-dependent factor **cannot** live
   on the measure-free :class:`~orpheus.numerics.basis.Basis`, so the
   trial :meth:`reconstruct
   <orpheus.numerics.basis.IndicatorBasis.reconstruct>` stays the plain
   (identity-dual) broadcast and the :math:`1/\Phi_R` normalisation is
   applied **separately** by the coefficient space's metric. The metric
   is a **Moore–Penrose pseudo-inverse**: a coarse cell with zero region
   flux (:math:`\Phi_R = 0`) is in the metric's null space and gets
   effective :math:`\Sigma_R = 0` — the :math:`0/0` branch of
   :eq:`sn-homogenization-vector-collapse` resolved for free, with no
   special-casing in :meth:`Solution.homogenize`.

For the **matrix channels** the same verb runs with the source-group
flux as the test weight: :math:`\phi_{g'}` rides the ``g_from`` axis (the
leading axis the test weight aligns to — see
:meth:`WeightedIndicatorBasis._weighted
<orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis>`'s
leading-aligned broadcast), and ``apply_inverse_metric`` broadcasts the
per-region Gram :math:`\Phi_R[:, g_{\rm from}]` over the trailing
``g_to`` axis — so the source-group normalisation of
:eq:`sn-homogenization-matrix-collapse` falls out of the
**trailing-axis metric-broadcast** rather than needing its own code
path. The :math:`\chi` channel uses the *identical* machinery in a
*separate* frame with a *different* test weight — the per-cell production
density :math:`p_i = \sum_g \nu\Sigma_{f,i,g}\,\phi_{i,g}\,V_i`
(:eq:`sn-homogenization-production-weight`) — so its Gram becomes the
region production :math:`\sum_{i\in R} V_i\,p_i` and the projection is the
production-weighted convex average :eq:`sn-homogenization-chi-collapse`.

Two test weightings, two frames — one conserved rate each
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The vector and matrix :math:`\Sigma` channels and the emission spectrum
:math:`\chi` do **not** share a frame, because they preserve **two
different conserved rates**, and a Petrov-Galerkin frame carries exactly
one test weighting:

.. list-table::
   :header-rows: 1
   :widths: 22 30 26 22

   * - Frame
     - Channels
     - Conserved rate
     - Test weight :math:`w`
   * - ``sigma_frame``
     - :math:`\Sigma_t,\Sigma_c,\Sigma_L,\Sigma_f,\nu\Sigma_f` (vectors);
       :math:`\Sigma_{s,\ell},\Sigma_{2n}` (``[g_from, g_to]`` matrices)
     - **reaction rate** :math:`\sum_i V_i\phi_{i,g}\Sigma_{i,g}`
     - per-group flux :math:`\varphi` (matrices weight by the
       **source** group)
   * - ``emission_frame``
     - :math:`\chi` (emission spectrum)
     - **emission rate** :math:`\sum_i p_i\chi_{i,g}`
     - production density :math:`p=\sum_g\nu\Sigma_{f,g}\varphi_g`

Both frames bind the **same** trial
:class:`~orpheus.numerics.basis.IndicatorBasis` and the **same**
geometric measure :math:`\mu_V`; they differ *only* in the test basis
they carry. :meth:`Solution.homogenize` builds both and hands them to
:meth:`MaterialXSField.project_through
<orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through>`,
which owns the **channel → frame** taxonomy: it routes every rate-bearing
:math:`\Sigma` channel through ``sigma_frame`` and :math:`\chi` through
``emission_frame``, projecting the *whole* cross-section field as one
object and returning one effective
:class:`~orpheus.data.macro_xs.mixture.Mixture` per coarse cell. The
caller owns the flux, so the caller builds the test weightings; the field
owns *which* weighting each channel needs.

.. note::

   That homogenization actually *executes* through these Frame readers —
   :meth:`IndicatorBasis.evaluate
   <orpheus.numerics.basis.IndicatorBasis.evaluate>` (the trial
   membership), :meth:`WeightedIndicatorBasis.analyze
   <orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis.analyze>`
   (the **test-side** flux-weighted reader),
   :meth:`FrameBase.project <orpheus.numerics.frame.FrameBase.project>`
   (the :math:`G^{-1}M` verb), and
   :meth:`FunctionSpace.apply_inverse_metric
   <orpheus.numerics.space.FunctionSpace.apply_inverse_metric>` — rather
   than a green rate gate riding a buggy refactor that recomputes
   membership inline or quietly re-folds :math:`\phi` into the metric, is
   pinned by the **Mode-11 sentinel**
   ``test_homogenize_routes_through_the_petrov_galerkin_frame``
   (vv-principles **Mode 11**, gate-never-executes-the-rewired-path): it
   monkeypatch-counts each reader and asserts the counter is positive
   after a ``homogenize`` run. The load-bearing count is
   ``WeightedIndicatorBasis.analyze`` — a bit-identity-preserving
   regression that kept the *old* Galerkin metric-fold (folding
   :math:`\phi` into the coefficient-space metric, test = plain trial
   indicator) would produce **identical numbers** yet never construct the
   weighted test basis, leaving that counter at zero. The
   rate-preservation identity :eq:`sn-homogenization-rate-preservation`
   remains THE correctness claim (the L0 gate); the sentinel makes the
   *Petrov-Galerkin structure* load-bearing for *this* implementation.

Why route through the Frame at all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The membership-matmul the prior slice shipped was correct; the carve to
the :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` is an
**architecture** move, not a correctness fix (Cardinal Rule 2). Three
payoffs justify it:

* **One mechanism, not one per method.** The angular-flux →
  spherical-harmonic-moment projection of SN anisotropic scattering (a
  :class:`~orpheus.numerics.frame.GalerkinFrame`) and the fine → coarse
  cross-section projection of homogenization (a
  :class:`~orpheus.numerics.frame.PetrovGalerkinFrame`) are the *same*
  mechanism — a discrete frame's analysis/reconstruction pair —
  differing only in *which* :class:`~orpheus.numerics.basis.Basis` pair
  (trial / test) is bound and *which discipline type* carries them.
  Routing both through the :class:`~orpheus.numerics.frame.FrameBase`
  hierarchy collapses a twin path (coding-elegance anti-pattern 1) into
  one body.
* **Energy condensation becomes the same shape.** The deferred
  ``.condense`` sibling is the identical Petrov-Galerkin frame with the
  spatial :class:`~orpheus.numerics.basis.IndicatorBasis` replaced by a
  *spectral* group-indicator basis and the measure replaced by
  :math:`L^2(\text{spectrum})`. Establishing the frame routing here means
  condensation lands as a no-op extension through the same body, not a
  third arm.
* **The pseudo-inverse handles the empty-region edge case for free.** The
  :math:`0/0` of a flux-free coarse cell is the metric's null space, so
  it needs no guard in :meth:`Solution.homogenize` — the projection
  algebra absorbs it.

.. _sn-condense-homogenize-asymmetry:

The condense / homogenize asymmetry law
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Homogenization (space) and condensation (energy) are siblings, but they
are **not** symmetric — they return different types, and the asymmetry
is structural, not incidental:

.. list-table::
   :header-rows: 1
   :widths: 18 28 28 26

   * - Operation
     - Collapses
     - Mesh coupling
     - Return type
   * - **homogenize**
     - space (fine cells → coarse cells)
     - **mesh-COUPLED** — the effective materials *are* the coarse
       cells; geometry and materials are born together
     - :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`
       (geometry + materials)
   * - **condense**
     - energy (fine groups → coarse groups)
     - **mesh-DECOUPLED** — the condensed cross sections are
       group-structure data, independent of where they sit
     - ``dict[int, Mixture]`` (portable materials, *no* geometry)

Spatial homogenization is **mesh-coupled**: a homogenized material has
no meaning apart from *the coarse cell it homogenizes*, because the
flux·volume weights :math:`w_{i,g}` are tied to specific fine cells
inside a specific coarse region. The natural product is therefore the
coarse geometry *carrying* its materials — a
:class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`, the
mesh+materials data carrier minted for exactly this purpose. (This is
why ``MaterialMesh`` exists as the middle type between a bare
:class:`~orpheus.geometry.mesh.Mesh1D` and a method phase space: a
homogenized model is *materials-and-geometry-together but not yet
method-specific* — it has no quadrature until
:meth:`SNMesh.from_material_mesh
<orpheus.sn.mesh.augmented_mesh.SNMesh.from_material_mesh>` promotes it.)

Energy condensation is **mesh-decoupled**: a condensed cross-section set
is just a coarser :class:`Mixture` — group-structure data that can be
attached to *any* geometry. Its natural product is a portable
``dict[int, Mixture]`` keyed by material id, with no geometry attached.

The asymmetry has a clean **frame-theoretic** reading once
homogenization is seen as the Petrov-Galerkin frame projection of
:ref:`sn-homogenization-petrov-galerkin-frame`. Both operations are the
*same* frame mechanism — analysis ∘ inverse-Gram — and differ *only* in
their trial basis :math:`K` (the
:class:`~orpheus.numerics.basis.Basis` bound into the frame):

* **homogenize** binds a **geometric** :math:`K` — the spatial
  :class:`~orpheus.numerics.basis.IndicatorBasis` (cell indicators
  :math:`\{\mathbf{1}_R\}`). Its coefficients *are* the coarse cells, so
  the result is **mesh-coupled** and the natural product is geometry +
  materials (:class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`).
* **condense** binds a **spectral** :math:`K` — a group-indicator basis
  on the energy axis (broad-group indicators :math:`\{\mathbf{1}_G\}`)
  under the :math:`L^2(\text{spectrum})` measure. Its coefficients are
  *group-structure data*, carrying no spatial identity, so the result is
  **mesh-decoupled** and the natural product is a portable
  ``dict[int, Mixture]``.

The return-type asymmetry is therefore not incidental: it is the
:math:`K`-axis (space vs energy) of the *one* projection mechanism
showing through. A geometric trial basis births geometry; a spectral
trial basis births portable group data.

The condensation half is realised by :meth:`Solution.condense
<orpheus.sn.solution.Solution.condense>`; its theory — the per-material
rate-preserving collapse, the fractional-overlap re-binning that lifts
the nesting requirement, and the same Petrov-Galerkin discipline — is
:ref:`sn-energy-condensation` below, the energy-axis transpose of this
section.

.. _sn-homogenization-verification:

Verification
~~~~~~~~~~~~

The gate is :mod:`tests.sn.test_homogenization` (level **L0**, term
verification — it checks the defining identity term-by-term, not a
solver claim). Its load-bearing test asserts the rate-preservation
identity :eq:`sn-homogenization-rate-preservation` directly:

* **Vector channels** (``test_rate_preservation_vector_channels``) —
  for every channel
  (:math:`\Sigma_t,\ \Sigma_c,\ \Sigma_L,\ \Sigma_f,\ \nu\Sigma_f`),
  every coarse region, and every group, assert
  :math:`\Sigma_{R,g}\,\Phi_{R,g} = \sum_{i\in R} V_i\,\Sigma_{i,g}\,
  \phi_{i,g}` to machine precision.
* **Matrix channels** (``test_rate_preservation_scattering_and_n2n``) —
  the same identity for every Legendre order of
  :math:`\Sigma_{s,\ell}` and for :math:`\Sigma_{2n}`, with the
  reference loop weighting by the **source** group ``g_from``
  *explicitly* — which is what makes it catch a ``g_from``↔``g_to``
  swap (vv-principles **Mode 2**).
* **n-D** (``test_homogenize_2d_rate_preservation``) — the same
  rate-preservation identity end-to-end through a *real* 2-D
  ``solve_sn``, exercising the n-D membership
  (:func:`numpy.ravel_multi_index` ``"ij"``) and the
  ``(ng, nx, ny) → (n_fine, ng)`` flatten the dropped 1-D guard opens, a
  flux tilt keeping :math:`\phi` non-flat so the flux-weighting is
  genuinely activated.

The reference these are checked against is a **structurally-independent**
explicit per-region Python loop over the fine cells — *not* a re-call of
the production frame projection (vv-principles **L11**: a cross-check
must be structurally independent, not merely procedurally independent;
a frame-vs-loop comparison sharing the *same* region reduction would
share any bug in the reduction). Companion invariants pin the rest of
the contract:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Test
     - What it pins
   * - ``test_homogenized_materials_balance``
     - Balance :eq:`sn-homogenization-balance` survives the collapse —
       every removal channel shares the per-:math:`(R,g)` weight.
   * - ``test_homogenized_chi_is_simplex``
     - :math:`\chi_R` is a probability simplex (convex average of
       producing simplices, :eq:`sn-homogenization-chi-collapse`).
   * - ``test_chi_is_production_weighted``
     - :math:`\chi_R` uses the **production** weight
       :eq:`sn-homogenization-production-weight`, not a flux- or
       volume-weight — the simplex test is blind to *which* convex weight,
       so this pins the weight choice directly.
   * - ``test_homogenization_is_flux_weighted_not_volume_weighted``
     - The load-bearing **flux-weighted-test** guard: over a
       vacuum→reflective flux tilt the flux-weighted and volume-only
       effective :math:`\Sigma_t` are numerically distinct, and
       production MUST match the flux-weighted one
       (:eq:`sn-homogenization-normal-equations`) — reds a regression
       that drops :math:`\phi` from the test weight (Galerkin /
       volume-only averaging).
   * - ``test_homogenize_routes_through_the_petrov_galerkin_frame``
     - **Mode-11 sentinel**: ``homogenize`` actually calls
       :meth:`IndicatorBasis.evaluate
       <orpheus.numerics.basis.IndicatorBasis.evaluate>` (trial
       membership), :meth:`WeightedIndicatorBasis.analyze
       <orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis.analyze>`
       (the **test-side** flux-weighted reader),
       :meth:`FrameBase.project
       <orpheus.numerics.frame.FrameBase.project>`, and
       :meth:`FunctionSpace.apply_inverse_metric
       <orpheus.numerics.space.FunctionSpace.apply_inverse_metric>` — the
       Petrov-Galerkin routing
       (:eq:`sn-homogenization-frame-projector`) is on the gate's call
       graph, not bypassed by an inline recompute *or a silent re-fold of*
       :math:`\phi` *into the metric* (which would keep the numbers and
       leave ``WeightedIndicatorBasis.analyze`` at zero calls).
   * - ``test_effective_xs_bracketed_by_fine_extremes``
     - :math:`\Sigma_{t,R}` is bracketed by the region's fine-cell
       extremes — a physical sanity check independent of the rate gate.
   * - ``test_identity_homogenization_recovers_per_cell_materials``
     - Homogenizing onto the *same* fine mesh recovers each cell's
       original material (degenerate limit: one fine cell per coarse).
   * - ``test_single_material_region_recovers_material``
     - A coarse cell containing only material :math:`m` homogenizes
       back to :math:`m` (the flux weight cancels).
   * - ``test_outer_boundary_mismatch_raises``
     - The guard: a coarse mesh whose outer boundary differs from the
       fine mesh raises ``ValueError``.

The :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` data
contract itself — the ``ng``-consistency check, the volume measure, the
XS-field build, and the ``SNMesh(MaterialMesh)`` data/behavior split
(including a bit-identity check that ``SNMesh``'s inherited data block
matches a standalone ``MaterialMesh``) — is gated separately by
:mod:`tests.transport.test_material_mesh`.


.. _sn-energy-condensation:

Applied to energy condensation
------------------------------

Spatial homogenization (:ref:`sn-spatial-homogenization`) collapses the
*space* axis; **energy condensation** is its **energy-axis transpose** —
it collapses a fine-group cross-section set onto a coarser group
structure, **spectrum-weighted**, so that each coarse group reproduces
the fine reaction rate. The two are the classical pair of "smear the
detail you have resolved into effective constants for a coarser
calculation" moves (Hébert, *Applied Reactor Physics* :cite:`Hebert2009`,
§13 for space, §3.5 for energy).

In ORPHEUS condensation lives at two layers, mirroring how
homogenization splits between :meth:`Solution.homogenize` (the
orchestration) and :meth:`MaterialXSField.project_through
<orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through>`
(the per-channel collapse):

* :meth:`Mixture.condense
  <orpheus.data.macro_xs.mixture.Mixture.condense>` — the per-material
  channel collapse (the data layer). Given a coarse target
  :class:`~orpheus.data.energy_grid.EnergyGrid` and a representative
  spectrum, ``mix.condense(target, spectrum)`` builds the fine→coarse
  fractional-overlap trial internally
  (:meth:`mix.energy_grid.overlap_to(target)
  <orpheus.data.energy_grid.EnergyGrid.overlap_to>`) and returns the
  condensed (coarse-group)
  :class:`~orpheus.data.macro_xs.mixture.Mixture`. It is **data-native** —
  every object it touches (the grid, the overlap factory, the
  Petrov-Galerkin frame) lives in ``data`` / ``numerics``, with **no**
  transport dependency.
* :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>` —
  the orchestration (the SN layer). It derives each material's
  representative spectrum from the solved flux and returns a
  **portable** ``dict[int, Mixture]`` keyed by material id.

.. note::

   This is the **energy-only** slice (the energy sibling of spatial
   homogenization). Geometry is **not** touched — the result is portable
   few-group cross sections, not a mesh. The asymmetry between the two
   operations, and *why* they return different types
   (``dict[int, Mixture]`` vs
   :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`), is
   :ref:`sn-condense-homogenize-asymmetry` above. Energies obey the
   canonical fast-first convention (group ``0`` = fastest, descending
   boundaries; :ref:`canonical-group-convention`).

The defining property: reaction-rate preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Condensation, like homogenization, is defined by *what it must
preserve*, not by an averaging recipe. The quantity a transport
calculation consumes is the **reaction rate** in each group, and the
group-collapse cross section is *defined* so as to preserve it (Hébert
:cite:`Hebert2009` Eq. 3.103; Stamm'ler & Abbate, *Methods of Steady-State
Reactor Physics in Simplified Geometry* (1983), Eq. VI(6b) — two
independent authoritative textbooks state it identically). Fix a coarse
group :math:`G`, made up of fine groups :math:`g \in G`. The fine-group
reaction rate of any vector channel is

.. math::
   :label: energy-condensation-fine-rate

   r_G \;=\; \sum_{g \in G} \varphi_g\,\Sigma_g,

.. (vv-status rationale) Definitional identity: the fine-group reaction
   rate, a plain group sum because ORPHEUS φ_g is already
   group-integrated (the bin width is inside φ_g; see
   :eq:`energy-condensation-counting-measure`). A
   derivation-decomposition premise of the rate-preservation claim
   (:eq:`energy-condensation-rate-preservation`), which is the verifiable
   claim (L1 gate ``tests.data.test_mixture_condense``).
.. vv-status: energy-condensation-fine-rate documented

with :math:`\varphi_g` the per-material representative flux (the test
weight, fixed below) and :math:`\Sigma_g` the fine cross section for
whatever channel (total, capture, fission, …) is in question. The sum
has **no** :math:`\mathrm{d}E` or :term:`lethargy` width because ORPHEUS's
:math:`\varphi_g` is already the group-integrated flux
:math:`\int_g \phi\,\mathrm{d}E` — the bin width is baked into the flux
(:eq:`energy-condensation-counting-measure`). The coarse model carries
one effective cross section :math:`\Sigma_G` and one coarse-group flux
:math:`\Phi_G` per group; for it to reproduce the fine rate it must
satisfy

.. math::
   :label: energy-condensation-rate-preservation

   \Sigma_G\,\Phi_G
   \;=\;
   \sum_{g \in G} \varphi_g\,\Sigma_g,

This is the **only** constraint condensation imposes on the vector
channels; everything below is derived from it. It is the energy-axis
copy of :eq:`sn-homogenization-rate-preservation` (with the fine-cell
volume·flux weight :math:`V_i\phi_{i,g}` replaced by the fine-group flux
:math:`\varphi_g`, and the spatial region :math:`R` replaced by the
energy group :math:`G`). The coarse-group flux is fixed first, by the
production-free inventory match — the coarse flux is the fine flux
summed over the group,

.. math::
   :label: energy-condensation-coarse-flux

   \Phi_G \;=\; \sum_{g \in G} \varphi_g.

.. (vv-status rationale) Representational identity: the coarse-group flux
   is the fine flux summed over the group (the diagonal Gram Φ_G of the
   PG frame). A definition consumed by the rate-preservation claim
   (:eq:`energy-condensation-rate-preservation`), not an independent
   solver claim.
.. vv-status: energy-condensation-coarse-flux documented

Substituting :eq:`energy-condensation-coarse-flux` into
:eq:`energy-condensation-rate-preservation` and solving for the
effective cross section gives the **spectrum-weighted average**

.. math::
   :label: energy-condensation-vector-collapse

   \Sigma_G
   \;=\;
   \frac{\sum_{g \in G} \varphi_g\,\Sigma_g}
        {\sum_{g \in G} \varphi_g},

.. (vv-status rationale) Derivation-decomposition step: the
   spectrum-weighted collapse obtained by solving the rate-preservation
   identity (:eq:`energy-condensation-rate-preservation`) for the
   effective cross section — Hébert Eq. 3.103. The verifiable content is
   the L1 rate-preservation gate; this is the algebraic rearrangement,
   not a separate claim.
.. vv-status: energy-condensation-vector-collapse documented

the flux-weighted reaction-rate-preserving average (Hébert
:cite:`Hebert2009` Eq. 3.103 ≡ Stamm'ler VI(6b)). Because the weight
:math:`\varphi_g` appears in both the numerator (rate) and denominator
(flux), :math:`\Sigma_G` is a genuine convex combination of the fine
values: it is bracketed by the group's fine extremes
:math:`\min_{g\in G}\Sigma_g \le \Sigma_G \le \max_{g\in G}\Sigma_g`.
This is *not* a separate design choice — it falls straight out of
:eq:`energy-condensation-rate-preservation`. Choosing any other weight
(width-only, unweighted) would break rate preservation wherever the
spectrum varies across the coarse group, which is exactly the regime
condensation exists to handle. The vector channels — total, capture,
leakage-loss (:math:`\Sigma_L`), fission, and production
(:math:`\nu\Sigma_f`) — each collapse through
:eq:`energy-condensation-vector-collapse` with the *same* weight; a
coarse group with zero flux (:math:`\Phi_G = 0`) has no reaction rate to
preserve, so its effective cross section is zero — the :math:`0/0`
resolved by the only physically meaningful value (the frame's
Moore–Penrose Gram, below).

The counting measure: why the weight is :math:`\varphi_g`, not :math:`\Delta u_g\,\varphi_g`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A subtle but load-bearing point distinguishes the energy axis from the
spatial axis. In Hébert's continuous formulation the flux-weighted
average of a *distribution* (a reaction rate) is a plain lethargy
integral (Eq. 3.96, :math:`\langle X\rangle_g = \int_g X\,\mathrm{d}u`)
while the average of a *function* (the flux) carries a :math:`1/\Delta
u_g` lethargy-width normalisation (Eq. 3.97). The discrete weight that
preserves the rate is therefore

.. math::
   :label: energy-condensation-counting-measure

   w_g \;=\; 1
   \qquad\text{(counting), not}\qquad
   w_g \;=\; \Delta u_g \;=\; \ln(E_{g}^{\rm upper}/E_{g}^{\rm lower}),

.. (vv-status rationale) Structural identity: the energy-axis measure is
   COUNTING (w=1), because ORPHEUS φ_g is already group-integrated, so
   the discrete rate is a plain group sum. Design rationale for the
   measure choice; the verifiable content is the rate-preservation gate
   (a Δu weight breaks it) plus the
   :class:`~orpheus.numerics.measure.DiscreteMeasure` bit-identity, not a
   solver claim.
.. vv-status: energy-condensation-counting-measure documented

because ORPHEUS stores :math:`\varphi_g = \int_g \phi\,\mathrm{d}E`
already integrated over the bin ("eV-free"). The discrete rate is then a
plain group **sum** :math:`r_G = \sum_{g\in G}\varphi_g\Sigma_g`
(:eq:`energy-condensation-fine-rate`) — the bin width is already inside
:math:`\varphi_g`. Verified against the frame algebra:
:math:`\Sigma_G\cdot\Phi_G = \sum_g w_g\,\varphi_g\Sigma_g` equals the
physical rate **iff** :math:`w_g = 1`; introducing a :math:`\Delta u_g`
weight would double-count the width and break rate preservation. This is
the energy-axis analogue of the spatial case, where :math:`\phi_i` *is*
a density and therefore *does* need the geometric volume measure
:math:`V_i` (:eq:`sn-homogenization-fine-rate`); here the measure is
:math:`w_g = 1` because the integration is already done. Lethargy is the
node *coordinate*, never the *weight* (it reappears below as the
within-group flux model, :eq:`energy-condensation-overlap-fraction`,
which sets a fine group's *split* across coarse groups — a basis datum —
not the measure). The
:class:`~orpheus.numerics.measure.DiscreteMeasure` the condensation
frame binds is therefore a **counting** measure
(:attr:`weights = ones <orpheus.numerics.measure.DiscreteMeasure>`,
``support="energy"``).

The matrix channels: a two-axis collapse (sink summed, source averaged)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattering matrices :math:`\Sigma_{s,\ell}[g',g]` (one per Legendre
order :math:`\ell`) and the :math:`(n,2n)` matrix
:math:`\Sigma_{2n}[g',g]` carry **two** group indices, stored
``[g_from, g_to]`` (the ORPHEUS scattering convention — see
:ref:`theory-cross-section-data`). They collapse by a **two-axis** rule
that has *no spatial precedent* and is the energy-condensation analogue
that most differs from homogenization. The in-scatter rate from coarse
group :math:`G` into coarse group :math:`G'` that must be preserved is

.. math::
   :label: energy-condensation-scattering-collapse

   \Phi_G\,\Sigma_{s,\ell,G\to G'}
   \;=\;
   \sum_{g \in G}\sum_{g' \in G'}
   \varphi_g\,\Sigma_{s,\ell}[g, g'],

.. (vv-status rationale) Definitional identity: the in-scatter rate of a
   matrix channel that the 2-axis collapse must preserve — the SINK axis
   g' summed (every fine scatter into any fine group of G' is a scatter
   into G'), the SOURCE axis g flux-averaged. Hébert Eq. 3.104,
   Stamm'ler VI(6c). The verifiable claim is the L1 scattering-collapse
   gate (which catches a g_from↔g_to swap, vv Mode 2).

driven by the population of the **source** group :math:`G` (the group
whose flux scatters *out* — the number of :math:`G\to G'` events scales
with how many particles are *in* :math:`G`). Decompose the collapse into
its two axes:

#. **Sink axis** :math:`g'` (the destination group) is **summed**: any
   scatter into *any* fine group :math:`g'` of coarse :math:`G'` is a
   scatter into :math:`G'`. The destination has no rate to average — it
   is an accumulation. In matrix form the sink-sum is a right-multiply
   by the membership table :math:`T` (below):
   :math:`\Sigma^{\rm sink}[g, G'] = \sum_{g'} \Sigma_{s,\ell}[g, g']\,T[g', G'] = (\Sigma_{s,\ell}\,T)[g, G']`.
#. **Source axis** :math:`g` (the origin group) is **flux-averaged**, by
   the *same* :eq:`energy-condensation-vector-collapse` rule applied to
   the sink-summed matrix:
   :math:`\Sigma_{s,\ell,G\to G'} = \big(\text{project}\,(\Sigma_{s,\ell}\,T)\big)[G, G']`.

So the matrix collapse is

.. math::
   :label: energy-condensation-matrix-collapse

   \Sigma_{s,\ell,G\to G'}
   \;=\;
   \frac{\sum_{g \in G} \varphi_g
         \big(\sum_{g'\in G'} \Sigma_{s,\ell}[g, g']\big)}
        {\sum_{g \in G} \varphi_g}
   \;=\;
   \operatorname{project}\!\big(\Sigma_{s,\ell}\,T\big),

.. (vv-status rationale) Derivation-decomposition step: the two-axis
   matrix collapse — sink summed (@T), source flux-averaged (project) —
   the matrix-channel form of :eq:`energy-condensation-scattering-collapse`.
   The verifiable content is the L1 scattering-collapse gate plus its
   three mutation probes (swap axes / sum both / project both); this is
   the algebraic form, not a separate claim.
.. vv-status: energy-condensation-matrix-collapse documented

with :math:`\operatorname{project}` the source-group flux average
(:meth:`frame.project <orpheus.numerics.frame.FrameBase.project>`) and
:math:`T` the fine→coarse membership table. The :math:`(n,2n)` channel
collapses identically. In the code
(:meth:`Mixture.condense
<orpheus.data.macro_xs.mixture.Mixture.condense>`) this reads exactly
``frame.project(mat @ T)`` per matrix channel.

.. warning::

   The sink-sum / source-average asymmetry is the subtle point of the
   whole operation, and it is **the structural difference from spatial
   homogenization**, which flux-weights *both* matrix axes
   (:eq:`sn-homogenization-matrix-collapse` runs ``project`` on a single
   axis with the source weight, but the spatial collapse never
   *sums* a sink axis — there is no spatial sink-summation because
   homogenization keeps the group structure). Three wrong collapses each
   produce a numerically distinct — and rate-breaking — coarse matrix,
   and each is a textbook variable-swap / missing-factor trap
   (vv-principles failure **Mode 2** and **Mode 3**):

   * **swap the axes** (flux-weight the *sink*, sum the *source*,
     ``project(SigS.T @ T)``-style) — wrong source/sink roles;
   * **sum both axes** (``T.T @ SigS @ T``) — drops the source
     flux-weight entirely;
   * **project both axes** (flux-weight the sink too) — this is exactly
     the ``homogenize`` behaviour, which is *wrong* for condensation
     (it would guard against "the implementer copied ``homogenize``
     verbatim").

   The regression gate
   ``tests.data.test_mixture_condense::TestG3ScatteringTwoAxisCollapse``
   reds on all three because its in-scatter-rate reference loop sums the
   sink and flux-averages the source *explicitly* (a hand-coded double
   ``for`` over fine groups — structurally independent of the production
   ``project``).

The fission spectrum is a pure birth-group sum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The emission spectrum :math:`\chi_g` is **not** a reaction rate — it is
a probability distribution (a simplex, :math:`\sum_g \chi_g = 1`; see
:eq:`emission-spectrum-simplex` in :ref:`theory-cross-section-data`).
Flux-weighting it would not preserve anything physical and could leave
the simplex. The rate-preserving choice is the **pure birth-group sum**

.. math::
   :label: energy-condensation-chi-collapse

   \chi_G \;=\; \sum_{g \in G} \chi_g
   \;=\; \chi \,@\, T,

.. (vv-status rationale) Representational identity: the χ collapse is a
   pure birth-group sum (the @T contraction = the table's birth-group-sum
   role), NOT flux-weighted — χ is a probability mass, so the coarse
   probability is the sum of the fine masses landing in G (Hébert
   Eq. 3.112, Stamm'ler VI(6a)). The simplex/null law it must satisfy is
   the data-layer ``Mixture`` invariant; this label is the collapse
   formula, not a separate solver claim. Since CS4c step 4 the COUPLING
   with the νΣf channel is additionally gated end-to-end by G-F1
   (``tests/transport/test_kernels.py::TestFissionCondensationGF1``) —
   see the admonition below.
.. vv-status: energy-condensation-chi-collapse documented

.. important::

   **The** :math:`\chi` **and** :math:`\nu\Sigma_f` **channels condense
   through DIFFERENT morphisms, and the pairing is now gated as one
   law.**  Fission is the only channel whose two factors leave through
   opposite doors of this section: the **sink** factor :math:`\chi`
   takes the mass-preserving birth-group sum just derived
   (:eq:`energy-condensation-chi-collapse`), while the **source** factor
   :math:`\nu\Sigma_f` is an ordinary vector channel and takes the
   rate-preserving :math:`\varphi`-weighted average
   (:eq:`energy-condensation-vector-collapse`).  Read separately each is
   uncontroversial; read together they say the condensed fission dyad
   must satisfy

   .. math::
      :label: energy-condensation-fission-dyad

      \operatorname{dyad}\bigl(\mathcal{C}(K)\bigr)
      \;=\;
      \bigl|\,\textstyle\sum_{g\in G}\chi_g\,\bigr\rangle
      \bigl\langle\,
        \bigl(\textstyle\sum_{g\in G}\varphi_g\,\nu\Sigma_{f,g}\bigr)
        \big/ \bigl(\textstyle\sum_{g\in G}\varphi_g\bigr)
      \,\bigr| ,

   .. (vv-status rationale) Structural identity: the χ↔νΣf-coupled
      condensation of the fission kernel — the composition of two
      morphisms this section already derives separately, stated as one
      law because the PAIRING is what a wrong-morphism swap breaks and
      neither factor's own gate can see. Not a solver claim (no
      eigenvalue, no flux). Its verifiable content is the L1 gate
      ``tests/transport/test_kernels.py::TestFissionCondensationGF1::test_law_ruled_morphism_pair``
      (rtol 1e-14) with THREE measured wrong-morphism negative controls
      and an ASSERTED activation precondition; see the prose below.
   .. vv-status: energy-condensation-fission-dyad documented

   with :math:`\mathcal{C}` the condensation and
   :math:`K = (\chi, \nu\Sigma_f)` the per-material
   :class:`~orpheus.transport.kernels.FissionKernel`.  The asymmetry is
   **by design and is the finding**: swapping either factor onto the
   other's morphism produces a coarse dyad that still looks physical
   (positive, right shape, :math:`\chi` still on a simplex under the
   sum) and is wrong.  ``[M]`` 2026-08-30, on the shipped
   4-group :math:`\to` 2-group fixture, the three wrong pairs sit
   :math:`6.4\times10^{-1}` (average/average),
   :math:`1.7\times10^{0}` (marginalize/marginalize) and
   :math:`7.1\times10^{-2}` (average/marginalize) in relative max-norm
   from the correct dyad, against a law row asserted at
   ``rtol = 1e-14`` — so any of the three reds the gate by more than ten
   orders of magnitude.

   Two design properties of that gate are worth carrying into any future
   condensation work on this page.  First, the morphisms in the gate are
   **hand-built in the test body** from the partition and :math:`\varphi`
   — never a second ``frame.project`` call — so the reference is
   structurally independent of the machinery under test rather than a
   procedural rearrangement of it (``vv-principles`` L11).  Second, the
   gate **asserts its own activation precondition**: a target with one
   fine group per coarse group makes ``average ≡ marginalize`` and every
   control go silent (``vv-principles`` Mode 12 at the fixture), so an
   identity condensation is *refused as a fixture* with its own red row
   rather than silently accepted.  A future fixture edit that flattens
   :math:`\varphi` or coarsens one-to-one therefore fails loudly instead
   of quietly de-fanging the law.

   ⚠ **Branch scope.**  This pins the FORWARD branch
   (``adjoint_spectrum is None``).  The bilinear branch folds the
   adjoint carrier into the sink factor and obeys a *different* law; it
   is not covered here.

— the probability mass of the fine birth groups landing in coarse group
:math:`G`, summed (Hébert :cite:`Hebert2009` Eq. 3.112; Stamm'ler VI(6a)).
This **differs from spatial homogenization**, whose :math:`\chi`
collapse is a *production-weighted convex average across cells*
(:eq:`sn-homogenization-chi-collapse`): there are many fine cells
contributing different spectra to one coarse cell, so they must be
*mixed*; here there is one material whose birth groups are merely
*re-binned*, so the coarse spectrum is the partial sum of the fine
probability mass. The sum **preserves the simplex**:

.. math::
   :label: energy-condensation-chi-simplex-preservation

   \sum_G \chi_G
   \;=\;
   \sum_G \sum_{g\in G} \chi_g
   \;=\;
   \sum_g \chi_g
   \;=\; 1,

because the partition is a partition of unity over coarse groups (every
fine group's mass is counted exactly once;
:eq:`energy-condensation-partition-of-unity`). A flux-weighted
projection would give :math:`\sum_G\chi_G \ne 1`, destroying the
simplex — which is why :math:`\chi` is routed through the *table*
contraction ``χ @ T``, **not** through :meth:`frame.project
<orpheus.numerics.frame.FrameBase.project>`. The simplex / null law is
re-validated when the condensed
:class:`~orpheus.data.macro_xs.mixture.Mixture` is constructed
(:meth:`Mixture.__post_init__
<orpheus.data.macro_xs.mixture.Mixture.__post_init__>`).

.. note::

   Post fast-first flip (:ref:`canonical-group-convention`), coarse
   group ``0`` is the fastest, so a fission spectrum — which peaks in the
   fast range — is **fast-peaked**: the bulk of :math:`\chi_G` sits in
   the low coarse-group indices. (On the production 421-group grid the
   χ peak energy ≈ 1.15 MeV lands a few coarse groups in, not at index
   0, because the 20-MeV grid ceiling puts a small high-energy tail
   above the fission peak — so the physically-correct invariant the
   real-data gate pins is *cumulative* fast-half mass
   :math:`\sum_{G<G_{\max}/2}\chi_G > 0.5`, not ``argmax == 0``.)

Balance is preserved group-by-group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The definitional total-XS balance every
:class:`~orpheus.data.macro_xs.mixture.Mixture` carries (the same
identity :eq:`sigT-computed` / :eq:`sn-homogenization-balance`),

.. math::
   :label: energy-condensation-balance

   \Sigma_t
   \;=\;
   \Sigma_c + \Sigma_L + \Sigma_f
   + \operatorname{rowsum}(\Sigma_{s0})
   + \operatorname{rowsum}(\Sigma_{2n}),

.. (vv-status rationale) Literature-transcribed definition: the total-XS
   balance identity every Mixture carries. Its preservation under
   condensation is the energy-axis copy of the homogenization argument —
   every removal channel collapses with the SAME per-coarse-group flux
   weight Φ_G. Gated by ``Mixture.assert_balanced`` on the condensed
   mixture (foundation invariant); not an SN solver claim.
.. vv-status: energy-condensation-balance documented

survives the collapse **group-by-group** when the fine material
balances, by the *same* one-line argument as homogenization. Fix a
coarse group :math:`G`. Every *removal* channel —
:math:`\Sigma_t,\ \Sigma_c,\ \Sigma_L,\ \Sigma_f`, and the row-sums
:math:`\sum_{g'}\Sigma_{s0}[g,g']`, :math:`\sum_{g'}\Sigma_{2n}[g,g']` —
is a *removal from group* :math:`g`, so each collapses with the **same**
source-group weight :math:`\varphi_g`. Crucially the matrix row-sum
:math:`\sum_{G'}\Sigma_{s0,G\to G'}` equals the *source-flux-average of
the fine row-sum*: the sink-sum :math:`\Sigma_{s0}\,T` then a row-sum
over coarse :math:`G'` telescopes (partition of unity) back to the fine
total out-scatter
:math:`\sum_{g'}\Sigma_{s0}[g,g']`, which then source-averages by the
same :math:`\varphi_g` as the vector channels. Because every term shares
the one weight :math:`\varphi_g`, the condensed balance is the *same
flux-weighted average* of the fine balances:

.. math::
   :label: energy-condensation-balance-preservation

   \Sigma_{t,G}
   - \Big(\Sigma_{c,G} + \Sigma_{L,G} + \Sigma_{f,G}
   + \operatorname{rowsum}(\Sigma_{s0,G})
   + \operatorname{rowsum}(\Sigma_{2n,G})\Big)
   \;=\;
   \frac{\sum_{g\in G} \varphi_g\,\big(\text{fine balance residual}_g\big)}
        {\sum_{g\in G} \varphi_g}
   \;=\; 0,

since each fine residual is zero. No "rebalance the condensed total"
step is needed — preservation is automatic, and the condensed
``Mixture`` passes :meth:`Mixture.assert_balanced
<orpheus.data.macro_xs.mixture.Mixture.assert_balanced>` whenever the
fine one does. This is the operation's correctness **oracle**: the
balance identity is a free, structurally-independent regression guard on
every condense. (Had the vector channels and the matrix row-sums
collapsed with *different* weights — e.g. had the matrix been projected
on both axes — the balance would break, another way of seeing why the
sink-sum / source-average asymmetry of
:eq:`energy-condensation-matrix-collapse` is *forced*, not chosen.)

.. _sn-condensation-fractional-overlap:

The non-nested problem: fractional-overlap re-binning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above assumed a fine→coarse membership table :math:`T[g,G]`.
For a **nested** coarse structure — coarse boundaries a subset of the
fine boundaries — that table is one-hot (:math:`T[g,G]\in\{0,1\}`, each
fine group wholly inside one coarse group) and the collapse reduces to
the exact group-sum of :eq:`energy-condensation-vector-collapse`. But the
production case is **not nested**, and the table must generalise.

Why one-hot containing-interval fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ORPHEUS condenses a **421-group** library onto the WIMS-D 69- and
172-group structures (:cite:`WIMSD`). These structures were defined
*independently* of the 421-group grid, so their boundaries do **not**
align with the fine grid (the draft boundary-mismatch report flags 19
non-coincident boundaries for 172→69 alone), and — the harder part —
**the coarse grid is locally finer than the fine grid** in narrow
resonance and thermal bands. A naïve "assign each fine group wholly to
the coarse group its representative energy falls in" (a one-hot
``searchsorted`` containing-interval rule) then leaves **empty coarse
groups**: 3 empty for 421→69, 22 empty for 421→172 — a coarse group
narrower than the fine spacing receives no fine group's representative
energy at all. An empty coarse group has zero total cross section, which
is unphysical and breaks a downstream solve.

This is a well-known stage distinction in reactor-physics data
processing. There are **two** group-averaging stages, and only the
second has the nesting constraint:

.. list-table::
   :header-rows: 1
   :widths: 26 40 34

   * - Stage
     - What it averages
     - Boundary constraint
   * - **pointwise → multigroup**
       (NJOY/GROUPR, OpenMC ``mgxs``, MC²-3)
     - the *continuous-energy* :math:`\sigma(E)\phi(E)` directly over any
       boundaries
     - **none** — the actual cross-section shape inside each group is the
       truth, so any structure is trivially integrable
   * - **multigroup → fewer-group**
       (AMPX/MALOCS — *the stage ORPHEUS is in*)
     - *already-discretised* fine groups via a fine→coarse map
     - **nesting** — a fine group cannot be split, because the input
       carries only the fine-group *average*, not the within-group shape

The collapse-stage codes (AMPX's MALOCS module) **require** the coarse
boundaries to be a subset of the fine boundaries: their input is a
fine→coarse *correspondence array* (e.g. "the first 4 fine groups → broad
group 1"), which structurally cannot express a fine group straddling a
broad boundary. ORPHEUS deliberately goes **beyond** MALOCS — it lifts
the nesting requirement with conservative fractional re-binning, a
capability the production deterministic-library codes mostly lack (they
sidestep it by re-integrating the continuum, which ORPHEUS cannot do from
a pre-grouped 421-library).

The fix: a fractional partition of unity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fine group :math:`g` that *straddles* a coarse boundary apportions a
**fraction** of its rate to each coarse group it overlaps. The membership
table becomes **fractional**,

.. math::
   :label: energy-condensation-overlap-fraction

   T[g,G] \;=\; f_{g,G}
   \;=\; \frac{\int_{g \cap G} w(E)\,\mathrm{d}E}
              {\int_g w(E)\,\mathrm{d}E},

.. (vv-status rationale) Literature-transcribed definition: the
   flux-weighted overlap fraction of a straddling fine group — the
   fraction of its within-group flux falling in coarse group G (the
   standard reactor-physics conservative re-binning, Q2 of the P5
   literature pull). The verifiable content is the partition-of-unity
   bit-identity (:eq:`energy-condensation-partition-of-unity`) and the
   1/E lethargy-ratio gate (``TestF4WithinGroupModelOracle``); a
   definition, not a solver claim.
.. vv-status: energy-condensation-overlap-fraction documented

the fraction of fine group :math:`g`'s
(:math:`w`-weighted) interval lying in coarse group :math:`G`, with
:math:`w(E)` an assumed **within-group flux model**. By the integral's
additivity the table is a **partition of unity**:

.. math::
   :label: energy-condensation-partition-of-unity

   \sum_G T[g,G] \;=\;
   \frac{\int_g w(E)\,\mathrm{d}E}{\int_g w(E)\,\mathrm{d}E} \;=\; 1
   \qquad \forall g.

.. (vv-status rationale) Structural identity: the membership table's rows
   sum to 1 (every fine group's rate is partitioned, not duplicated or
   dropped) — the property that makes the collapse conservative and the χ
   sum simplex-preserving. Gated by ``TestF2PartitionOfUnity``
   (``table.sum(axis=1) == 1``); a representational invariant, not a
   solver claim.
.. vv-status: energy-condensation-partition-of-unity documented

so each fine group's rate is *partitioned* (counted once), never
duplicated or dropped — the collapse stays conservative, and the χ-sum
stays a simplex. The general fractional collapse

.. math::
   :label: energy-condensation-fractional-collapse

   \Sigma_G
   \;=\;
   \frac{\sum_g f_{g,G}\,\varphi_g\,\Sigma_g}
        {\sum_g f_{g,G}\,\varphi_g}

reduces **exactly** to the one-hot group-sum
:eq:`energy-condensation-vector-collapse` when the structure is nested
(:math:`f_{g,G}\in\{0,1\}`) — so the nested case is the *regression-safe
degenerate*, not a separate code path.

The within-group flux model :math:`w(E)` is **selectable**
(:class:`~orpheus.data.energy_grid.WithinGroupSpectrum`, a strategy
Protocol). The default — built first — is **1/E (flat in lethargy)**,
:class:`~orpheus.data.energy_grid.InverseEnergySpectrum`:

.. math::
   :label: energy-condensation-lethargy-overlap

   \int_{lo}^{hi} \frac{\mathrm{d}E}{E} \;=\; \ln\!\frac{hi}{lo},
   \qquad
   f_{g,G} \;=\;
   \frac{\ln(hi_{g\cap G}/lo_{g\cap G})}{\ln(hi_g/lo_g)},

.. (vv-status rationale) Literature-transcribed definition: the 1/E
   (flat-in-lethargy) overlap fraction is a lethargy ratio — the
   asymptotic slowing-down spectrum, NJOY IWT=3, the standard first
   choice for condensation. The verifiable content is the
   ``InverseEnergySpectrum.integrated_weight`` = ln(hi/lo) bit-identity
   and the ``TestF4`` 1/E-vs-flat-energy discriminator; a definition, not
   a solver claim.
.. vv-status: energy-condensation-lethargy-overlap documented

the lethargy-overlap ratio (the asymptotic slowing-down spectrum; the
standard first choice for condensation, NJOY ``IWT=3``). Flat-in-energy
(NJOY ``IWT=2``) and the library weighting spectrum (fission + 1/E +
Maxwellian, NJOY ``IWT=4``) are future options on the same strategy
seam. The model is the **only new numerics surface** the non-nested case
adds — everything else (the Petrov-Galerkin frame, :meth:`frame.project
<orpheus.numerics.frame.FrameBase.project>`, the diagonal Gram, rate
preservation) is unchanged.

.. _sn-condensation-petrov-galerkin-frame:

Condensation is a Petrov-Galerkin projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like homogenization (:ref:`sn-homogenization-petrov-galerkin-frame`),
condensation is the coefficient extraction :math:`G^{-1}M` of a
**Petrov-Galerkin** frame — the *energy-axis instance* of the *same*
mechanism, which is exactly why the numerics core is reused verbatim:

  Condensation is :meth:`frame.project
  <orpheus.numerics.frame.FrameBase.project>` of a
  :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` whose *trial*
  basis is the fractional group-overlap indicator
  :math:`\mathbf{1}_G` (carried by
  :class:`~orpheus.numerics.basis.OverlapBasis`), whose *test* basis is
  the spectrum-weighted indicator :math:`\varphi\,\mathbf{1}_G` (carried
  by :class:`~orpheus.numerics.basis.WeightedIndicatorBasis`), and whose
  measure is the **counting** measure :math:`\mu` (weight 1).

This is not decoration: it is why :meth:`Mixture.condense
<orpheus.data.macro_xs.mixture.Mixture.condense>` routes through the
*same* discrete :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` that
carries SN anisotropic-scattering moment projection and spatial
homogenization — one mechanism for every fine→coarse change of
representation (Cardinal Rule 2, single source of truth), not a bespoke
membership matmul per axis. The frame projection machinery
(:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`,
:class:`~orpheus.numerics.basis.OverlapBasis`,
:class:`~orpheus.numerics.basis.WeightedIndicatorBasis`,
:meth:`FunctionSpace.apply_inverse_metric
<orpheus.numerics.space.FunctionSpace.apply_inverse_metric>`) was built
for homogenization anticipating energy as the second consumer; the data
layer reaches it because ``data → numerics`` is a permitted layer edge.

The trial / test / measure separation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Three distinct objects carry the three roles, cleanly separated — the
campaign's three-way trial/test/measure split holding verbatim
on the energy axis:

.. list-table::
   :header-rows: 1
   :widths: 20 30 28 22

   * - Role
     - Object
     - Carries
     - On the energy axis
   * - **trial** :math:`K`
     - :class:`~orpheus.numerics.basis.OverlapBasis`
     - the fractional membership :math:`T[g,G]` (partition geometry +
       within-group split)
     - the *only* new surface — the fractional table
   * - **test** :math:`M^*`
     - :class:`~orpheus.numerics.basis.WeightedIndicatorBasis`
     - the spectrum :math:`\varphi_g` as an analysis weight
     - the flux is the test weight, **never** the measure
   * - **measure** :math:`\mu`
     - :class:`~orpheus.numerics.measure.DiscreteMeasure`
     - the counting metric (:math:`w_g=1`), group-independent
     - fixed :math:`L^2`, **never** the discipline

The test functions :math:`\varphi_g\,\mathbf{1}_G` differ from the trial
functions :math:`\mathbf{1}_G` (:math:`\varphi_g\,\mathbf{1}_G \ne
\mathbf{1}_G`), so the projection is genuinely **Petrov-Galerkin**,
carried by the frame **type**
(:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`). With trial and
test indicators of (fractional) partition-of-unity support, the
**cross-Gram** is **diagonal** — the all-ones probe through the frame
gives :math:`(M\,\mathbf 1)_G = \sum_g \varphi_g\,T[g,G] = \Phi_G`, the
coarse-group flux :eq:`energy-condensation-coarse-flux`, which *is* the
diagonal of the Gram — so the normal equations decouple group-by-group
and :meth:`frame.project <orpheus.numerics.frame.FrameBase.project>`
returns the spectrum-weighted average
:eq:`energy-condensation-vector-collapse` with a per-group
**reciprocal**, not a linear solve.

.. note::

   For a *fractional* (straddling) table two coarse columns share a fine
   row, so the **off-diagonal** cross-Gram
   :math:`G_{GG'} = \sum_g \varphi_g\,T[g,G]\,T[g,G']` is **not**
   structurally zero. This is correct and harmless:
   :meth:`frame.project <orpheus.numerics.frame.FrameBase.project>` uses
   only the **diagonal** :math:`G_{GG} = \Phi_G` (each coarse group's
   rate is its *own* functional — one DOF per group, the P0 / rank-0
   space), so it ignores the off-diagonals by construction. The
   :meth:`OverlapBasis.mass_matrix
   <orpheus.numerics.basis.indicator_basis.IndicatorBasis.mass_matrix>`
   inherits a docstring claiming a *diagonal* Gram (true for the one-hot
   parent, false for a fractional table) — that claim is **latent**: no
   consumer calls ``mass_matrix``, and the frame's row-sum probe never
   forms the full Gram. A *future* least-squares consumer needing the
   dense Gram (a non-indicator, richer coarse basis — which cross
   sections never want, a P1 coarse XS is not rate-meaningful) must
   compute it for the fractional case, not trust the inherited diagonal
   claim. See :ref:`frame-least-squares-discipline` for why this is
   **not** a ``LeastSquaresFrame`` (its trigger — test = :math:`A`·trial,
   a dense SPD Gram needing a real solve — is absent here; that
   discipline is designed-but-not-built, gated by
   :class:`~orpheus.numerics.basis.GramStructure` ``DENSE``).

Why the spectrum is the test weight, not a measure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The natural-seeming alternative — "treat the spectrum as a measure and
project the cross section onto a coarse basis in the
:math:`L^2(\text{spectrum})` metric" — is the energy-axis form of the
metric-fold (:eq:`sn-homogenization-metric-fold`), and it is **refused**
for the *same* reason it is refused for homogenization
(:ref:`sn-homogenization-why-petrov-galerkin`, the #268 ruling: *the
measure carries the axis and the fixed* :math:`L^2` *metric, never the
discipline*). Three structural grounds, identical to the spatial case:

#. **For a P0 (indicator) coarse basis the least-squares fit
   *coincides* with the flux-weighted average** — with disjoint /
   fractional partition-of-unity indicators the normal-equations Gram is
   diagonal and the least-squares solution is
   :math:`\Sigma_G = \sum_g w_g\Sigma_g / \sum_g w_g`, the flux-weighted
   average verbatim when :math:`w=\varphi`. So "least-squares" does not
   select a *new* frame — it re-derives the same Petrov-Galerkin average
   under a different name.
#. **Folding :math:`\varphi` into the measure breaks under
   adjoint-weighting.** The eigenvalue-consistent condensation reactor
   physics ultimately requires preserves the **bilinear**
   :math:`\langle\varphi^*,\Sigma\varphi\rangle` with
   :math:`\varphi^*\ne\varphi` (the same
   :eq:`sn-homogenization-bilinear` structure on the energy axis), where
   test :math:`= \varphi^*\mathbf{1}_G \ne` trial :math:`=
   \varphi\,\mathbf{1}_G` and **no single metric** reproduces the
   two-sided weighting. Forward-flux reaction-rate-only condensation is
   the degenerate :math:`\varphi^*=\varphi` case where the fold happens
   to work; the *type* (an explicit test basis) encodes the general
   case, so the adjoint-weighted lift (**landed** in P6, #281) enters
   as the bilinear **pair** test weight :math:`\varphi^*\!\odot\varphi`
   — the product of adjoint and forward spectra, *not* a bare
   :math:`\varphi \to \varphi^*` swap (a different rule that does not
   zero the worth; :ref:`frame-adjoint-weighted-seam`) — not a
   re-derivation.
#. **The discipline is a property the** *type* **carries, never the
   measure.** Keeping :math:`\varphi` on the
   :class:`~orpheus.numerics.basis.WeightedIndicatorBasis` test side
   forces the correct reading: :math:`\varphi` is a test-weighting the
   *solution* emits, not a property of the energy axis. The energy axis
   (the grid) owns one counting measure; the solution owns the spectrum
   :math:`\varphi`; the frame *type* (Petrov-Galerkin, with its explicit
   test basis) carries the discipline.

The flux-weighted average is the **rank-0 moment** of **Generalized
Energy Condensation** (Rahnema, Douglass & Forget 2008 :cite:`Rahnema2008`):
GEC expands the within-coarse-group flux in orthogonal functions
:math:`\varphi(E)\approx\sum_n\varphi_{n,G}P_n(E)`, and the zeroth
moment (the constant / piecewise-constant basis function on :math:`G`)
*recovers the standard flux-weighted multigroup average exactly* — "the
zeroth moment generates the standard few-group equation". The higher
moments (:math:`n\ge1`) add the within-coarse-group spectral detail the
simple average discards; that is faithful within-group reconstruction
(honest upscaling), and it is **deferred** — it would be a richer trial
basis on the *same* frame (`GitHub #275
<https://github.com/deOliveira-R/ORPHEUS/issues/275>`_), no architectural
change.

.. _sn-condensation-downsampling:

Downsampling only: condensation loses information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The load-bearing semantic — a design ruling, not an implementation
detail — is that **condensation is a one-way, lossy, downsampling
operation.** Collapsing fine groups into coarse groups *discards* the
within-coarse-group spectral structure (that is the rank-:math:`>0` GEC
content above). The continuous-projection view *could* fabricate detail
(a "64 → 200 group" upscaling would invent sub-group structure the data
never carried), but the group-collapse stage ORPHEUS is in **cannot**:
the input is only the fine-group *averages*. The asymmetry is encoded in
three places:

* **A global upscaling guard.**
  :meth:`EnergyGrid.overlap_to
  <orpheus.data.energy_grid.EnergyGrid.overlap_to>` (the binary
  mismatch factory) **refuses** a coarse target with *more* groups than
  the source — :math:`n_{\rm coarse} > n_{\rm fine}` raises
  ``ValueError`` ("condensation only DOWNSAMPLES; a finer target would
  fabricate sub-group structure the data does not contain"). Reconstructing
  a finer structure from group-integrated data is fabrication; the guard
  makes it unrepresentable.
* **The within-group model is the *explicit* assumption.** Where the
  coarse grid is *locally* finer than the fine grid (the resonance /
  thermal bands), the unavoidable *local* interpolation is done by the
  within-group flux model :math:`w(E)`
  (:eq:`energy-condensation-overlap-fraction`) — a bounded, named,
  selectable assumption, not a silent invention.
* **The provenance report.**
  :attr:`OverlapBasis.fractional_columns
  <orpheus.numerics.basis.overlap_basis.OverlapBasis.fractional_columns>`
  (carried on the trial basis :meth:`overlap_to
  <orpheus.data.energy_grid.EnergyGrid.overlap_to>` returns)
  lists the coarse-group indices whose data leaned on :math:`w(E)`
  (the columns that received a *fractional* — strictly between 0 and 1 —
  contribution). It is **empty** for a nested condensation (pure
  rate-preserving collapse, no assumption) and non-empty exactly where
  the coarse grid is locally finer than the fine grid. This is the
  data-vs-assumption provenance: a caller can see precisely which coarse
  groups are pure collapse and which lean on the spectral model. (The
  companion :attr:`OverlapBasis.dominant_column
  <orpheus.numerics.basis.overlap_basis.OverlapBasis.dominant_column>` —
  the ``argmax`` containing-coarse map — is the former
  ``GroupCondensation.coarse_of_fine``.)

.. warning::

   Faithful reconstruction / honest *upscaling* (recovering
   within-coarse-group detail via the rank-:math:`>0` GEC moments) is
   **not** what this slice ships, and the upscaling guard deliberately
   prevents accidentally posing it as a fine-target ``condense``. Do
   **not** read the within-group model :math:`w(E)` as faithful
   reconstruction — it is a bounded local-interpolation assumption used
   *only* to apportion a straddling fine group's already-known rate, not
   to invent new spectral structure. Honest upscaling is a future
   capability (`GitHub #275 <https://github.com/deOliveira-R/ORPHEUS/issues/275>`_).

.. _sn-condensation-grid-frame-axis:

The grid is a frame axis: dual measure / basis views
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`~orpheus.data.energy_grid.EnergyGrid` is the energy analogue
of a coarse :class:`~orpheus.geometry.mesh.Mesh1D`, and — exactly like a
mesh — it is a **frame axis** that yields *both* halves of a discrete
frame, the two roles an axis plays in a projection:

.. list-table::
   :header-rows: 1
   :widths: 22 30 26 22

   * - View
     - Method
     - Role
     - Spatial twin
   * - **measure** :math:`\mu`
     - :meth:`EnergyGrid.as_measure
       <orpheus.data.energy_grid.EnergyGrid.as_measure>`
     - the **source** — the axis you project *from* (a counting
       :class:`~orpheus.numerics.measure.DiscreteMeasure`, :math:`w_g=1`,
       ``support="energy"``)
     - :meth:`Mesh1D.volume_measure
       <orpheus.geometry.mesh.Mesh1D.volume_measure>`
   * - **basis** :math:`\mathbf{1}`
     - :meth:`EnergyGrid.as_basis
       <orpheus.data.energy_grid.EnergyGrid.as_basis>`
     - the **target** — the axis you project *to* (the group-indicator
       :class:`~orpheus.numerics.basis.IndicatorBasis`, one-hot)
     - :meth:`Mesh1D.indicator_basis
       <orpheus.geometry.mesh.Mesh1D.indicator_basis>`

So a *nested* condensation is just ``fine.as_measure()`` →
``coarse.as_basis()`` — the two unary views suffice. The non-nested
production case needs **one more thing**, and it is irreducibly
**binary**: the fractional membership table reads *both* grids' edges at
once (a fine group straddles a *coarse* boundary — neither grid alone
knows the straddle), so it cannot be a unary view of either. That is
:meth:`EnergyGrid.overlap_to
<orpheus.data.energy_grid.EnergyGrid.overlap_to>`, the
``(fine, coarse) → OverlapBasis`` factory, with the containment

.. math::
   :label: energy-condensation-nested-subset

   \texttt{coarse.as\_basis()}\ \text{(nested, one-hot)}
   \ \subset\
   \texttt{fine.overlap\_to(coarse)}\ \text{(non-nested, fractional)}.

.. (vv-status rationale) Structural identity: the nested one-hot target
   view (``as_basis``) is the degenerate of the binary fractional
   ``overlap_to`` — the same containment the partition-of-unity table
   collapses to when every straddle fraction is 0 or 1. A
   representational relationship between the two views, gated by the
   ``TestF3NestedDegeneracy`` bit-identity (a nested ``overlap_to`` table
   equals the ``searchsorted`` one-hot); not a solver claim.
.. vv-status: energy-condensation-nested-subset documented

The returned trial is an :class:`~orpheus.numerics.basis.OverlapBasis`,
which **IS-A** :class:`~orpheus.numerics.basis.IndicatorBasis` carrying
the fractional table — so the nested one-hot view and the non-nested
fractional view are the *same type* of object, the degenerate and the
general case, never two code paths.

.. _sn-condensation-no-frame-subclass:

Why no ``CondensationFrame`` — the data-native shape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two design decisions shape where this machinery lives, and both are the
kind of "what was tried and rejected" a future session needs spelled
out so it does not re-litigate them.

**Condensation is data-native (no transport dependency).** An earlier
plan put a ``CondensationFrame`` in :mod:`orpheus.transport` (a
``transport/frames/`` package, symmetric with the angular
:class:`~orpheus.numerics.frame.GalerkinFrame`). That was **overturned**.
Condensation's carrier is the :class:`~orpheus.data.macro_xs.mixture.Mixture`
— a *data* type — and every object the collapse touches (the
:class:`~orpheus.data.energy_grid.EnergyGrid`, the
:meth:`overlap_to <orpheus.data.energy_grid.EnergyGrid.overlap_to>`
factory, and the :class:`~orpheus.numerics.frame.PetrovGalerkinFrame`)
lives in ``data`` / ``numerics``. The layering forbids the move (``data``
must **not** depend on ``transport``), and nothing in the operation needs
transport: it is a pure cross-section *re-binning*. So the collapse verb
is :meth:`Mixture.condense
<orpheus.data.macro_xs.mixture.Mixture.condense>` (data), reaching the
frame through the permitted ``data → numerics`` edge; only the SN-layer
*orchestration* (deriving the per-material spectrum from a solved flux)
lives above, in :meth:`Solution.condense
<orpheus.sn.solution.Solution.condense>`.

**There is no frame subclass at all** — no ``CondensationFrame``, no
``HomogenizationFrame``. The frame is a plain
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`; the "condensation"
identity is **not** a new kind of frame, it lives in *two* ordinary
places:

#. the **binary overlap factory** :meth:`EnergyGrid.overlap_to
   <orpheus.data.energy_grid.EnergyGrid.overlap_to>` (which builds the
   energy-specific fractional trial), and
#. the **collapse verb** :meth:`Mixture.condense
   <orpheus.data.macro_xs.mixture.Mixture.condense>`, which shares its
   channel-assembly assembler :meth:`Mixture.from_dense_channels
   <orpheus.data.macro_xs.mixture.Mixture.from_dense_channels>` with
   spatial homogenization's :meth:`MaterialXSField.project_through
   <orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through>`
   — one home for "assemble a ``Mixture`` from coarsened dense channels",
   not one per verb (Cardinal Rule 2).

Minting a ``CondensationFrame`` would be wrong on two counts. First it is
**false symmetry** with homogenization: homogenization is intrinsically a
*two-frame* operation (a flux-weighted frame for the rate channels **and**
a production-weighted frame for :math:`\chi` — :meth:`project_through
<orpheus.transport.mesh.material_xs_field.MaterialXSField.project_through>`
takes both), so even there a single "HomogenizationFrame" is the wrong
shape. Second it is **unjustified type-minting** (the project's
type-vs-property rule): a new frame *type* is earned only by a new frame
*morphism*, and condensation introduces none — its analysis,
reconstruction, Gram, and :meth:`project
<orpheus.numerics.frame.FrameBase.project>` are the *unchanged*
Petrov-Galerkin operations. The only genuinely new surface is the
fractional trial *basis* (:class:`~orpheus.numerics.basis.OverlapBasis`),
which is exactly where the novelty belongs — a basis, not a frame.

.. _sn-condensation-verification:

Verification
~~~~~~~~~~~~

The gates are :mod:`tests.data.test_energy_grid`,
:mod:`tests.data.test_mixture_condense`, and
:mod:`tests.sn.test_condensation`. The two solver-facing claims carry
``@pytest.mark.verifies`` markers tying them to the equations above:

* ``energy-condensation-rate-preservation``
  (:eq:`energy-condensation-rate-preservation`) — the vector-channel
  rate-preservation identity, asserted for every channel
  (:math:`\Sigma_t,\ \Sigma_c,\ \Sigma_L,\ \Sigma_f,\ \nu\Sigma_f`) and
  every coarse group: :math:`\Sigma_G\,\Phi_G = \sum_{g\in G}
  \varphi_g\,\Sigma_g` to one ULP.
* ``energy-condensation-scattering-collapse``
  (:eq:`energy-condensation-scattering-collapse`) — the two-axis matrix
  collapse, asserted as the preserved in-scatter rate :math:`\Phi_G\,
  \Sigma_{s,\ell,G\to G'} = \sum_{g\in G}\sum_{g'\in G'}\varphi_g\,
  \Sigma_s[g,g']` for every Legendre order and the :math:`(n,2n)` matrix.

These are **L1** (equation) claims against a **closed-form** reference,
*not* eigenvalue claims — condensation is a data-reduction operation, not
a solve, so there is no :math:`k` to verify; the pillar question is "does
the reduction preserve the rate functional", answered by closed-form
hand-summation. The correctness oracle is a **structurally-independent**
explicit per-group Python loop over the fine groups — *not* a re-call of
the production :meth:`frame.project
<orpheus.numerics.frame.FrameBase.project>` (vv-principles **L11**: a
cross-check must be structurally independent, not merely procedurally
independent; a frame-vs-frame comparison would share any reduction bug).
The scattering rate is one ULP, not bit-identical, because the
``@ T`` matmul reduction tree differs from the explicit group-by-group
sum — FP-non-associativity, principled-equivalent per the `vv-principles`
bit-identity-vs-principled-equivalence criteria (drift = reduction-depth
× ULP); the gate uses ``np.testing.assert_allclose(rtol=1e-12)``, never
exact ``==``.

The discriminator and the companion invariants:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Test
     - What it pins
   * - ``TestG1RatePreservationVectors`` /
       ``TestF1StraddleRatePreservation``
     - The **rate-preservation anchor**
       (:eq:`energy-condensation-rate-preservation`, the
       ``energy-condensation-rate-preservation`` verifies-target):
       :math:`\Sigma_G\,\Phi_G = \sum_{g\in G}\varphi_g\,\Sigma_g` for
       every vector channel and every coarse group, against the hand-sum
       oracle — nested (G1) and straddling-fractional (F1).
   * - ``TestG2WithinGroupVaryingDiscriminator`` /
       ``TestF4ModelDiscriminatorSUT``
     - The load-bearing **flux-weighting discriminator** (vv Mode 7): a
       fine spectrum that *varies within* each coarse group (e.g.
       :math:`\varphi=[1,4,2,0.5]`) makes the flux-weighted and
       arithmetic-average collapses numerically distinct, and rate
       preservation MUST match the flux-weighted one — reds a regression
       that drops :math:`\varphi`. A *flat* spectrum would null the
       weighting (the Mode-7 trap), so the fixture is asserted
       within-group-varying.
   * - ``TestG3ScatteringTwoAxisCollapse`` (three mutations)
     - The sink-sum / source-average asymmetry
       (:eq:`energy-condensation-matrix-collapse`) — each of swap-axes /
       sum-both / project-both produces a *numerically different* coarse
       matrix → each reds (vv Mode 2 / Mode 3; the project-both mutation
       guards against copying ``homogenize`` verbatim).
   * - ``TestChiBirthGroupSum`` (χ sum-vs-project guard)
     - :math:`\chi_G = \chi\,@\,T` preserves :math:`\sum\chi=1`
       (:eq:`energy-condensation-chi-collapse`); a flux-*projected* χ
       sums to :math:`\ne 1`, destroying the simplex — pinned separately
       from the projected channels.
   * - ``TestF2PartitionOfUnity`` / ``TestF3NestedDegeneracy``
     - The table is a partition of unity
       (:eq:`energy-condensation-partition-of-unity`,
       ``rows.sum == 1``), and the fractional table reduces
       **bit-identically** to the one-hot ``searchsorted`` table for a
       nested structure (the regression-safe degenerate).
   * - ``TestF4WithinGroupModelOracle``
     - The 1/E overlap fraction is the lethargy ratio
       :math:`\ln(hi_{\cap}/lo_{\cap})/\ln(hi_g/lo_g)`
       (:eq:`energy-condensation-lethargy-overlap`), and 1/E ≠
       flat-energy on a straddling group (the model is load-bearing, not
       cosmetic).
   * - ``TestF5UpscalingGuard``
     - :math:`n_{\rm coarse} > n_{\rm fine}` raises ``ValueError`` (the
       downsampling-only guard), with a positive control that a valid
       downsample succeeds.
   * - ``TestF6LocalInterpolationReport``
     - :attr:`OverlapBasis.fractional_columns
       <orpheus.numerics.basis.overlap_basis.OverlapBasis.fractional_columns>`
       is empty for a nested condensation and lists the straddle columns
       for a non-nested one (the data-vs-assumption provenance).
   * - ``TestG4BalanceRegression`` (positive + negative)
     - A condensed balanced ``Mixture`` passes
       :meth:`Mixture.assert_balanced` (:eq:`energy-condensation-balance`);
       a hand-built rate-broken condensed ``Mixture`` raises (vv #11:
       the negative leg pins the *invariant*, not merely the raising).
   * - ``TestG5WimsDerivationValidation`` (Table 11.3)
     - The containing-interval partition derived by the rule reproduces
       the published ``CONDENSE_172_TO_69`` (:cite:`WIMSD` Table 11.3) on the
       coincident-boundary groups, collecting the known 19 non-coincident
       boundaries as expected (failing only on a *new* mismatch).
   * - ``test_real_pwr_421_to_wims69_condensation_succeeds``
     - **L2 integration**: a *real* 421-group production mixture
       (:func:`~orpheus.data.macro_xs.recipes.pwr_like_mix`) condenses to
       WIMS-69 with **no empty coarse groups** (:math:`\Sigma_t>0` ∀
       coarse — the one-hot empty-group bug *gone*), balance preserved,
       χ fast-half-mass :math:`>0.5`, and a non-empty
       ``locally_interpolated``.
   * - Mode-11 routing sentinel
     - ``Mixture.condense`` actually calls :meth:`frame.project
       <orpheus.numerics.frame.FrameBase.project>` and
       :meth:`WeightedIndicatorBasis.analyze
       <orpheus.numerics.basis.weighted_indicator_basis.WeightedIndicatorBasis.analyze>`
       (the **test-side** spectrum reader) — the Petrov-Galerkin routing
       is on the gate's call graph, not bypassed by an inline matmul (vv
       **Mode 11**).

The intrinsic-property tests pin the new value objects' defining laws:
:class:`~orpheus.data.energy_grid.EnergyGrid` (strictly **descending**
boundaries — the #265 monotonicity slice — all-positive energies,
partition completeness, with positive *and* negative legs), and the
fractional-overlap trial
:class:`~orpheus.numerics.basis.OverlapBasis` (the partition-of-unity
law, ``rows.sum == 1``; and the containing-interval law on its
:attr:`~orpheus.numerics.basis.overlap_basis.OverlapBasis.dominant_column`:
every fine group → exactly one *dominant* coarse group, contiguous,
**fast-first** ordering — the orientation pin that catches the silent
descending-edge column-reversal trap, vv Mode 6).

≥2 groups throughout (69-, 172-, and 421-group cases — never the
degenerate 1-group case). Every sentinel / negative leg uses
``np.testing.*`` / ``pytest.raises`` / ``pytest.fail``, never a bare
``assert`` (vv **Mode 8**: ``-O`` strips bare asserts).

.. _frame-galerkin-frame:

The Galerkin frame
==================

The **Galerkin** frame is the special case ``test is trial`` — the
:class:`~orpheus.numerics.frame.GalerkinFrame` specialisation of the
Petrov-Galerkin base above. It strengthens the base promise
:math:`M R = I_W` (up to tightness) to the self-dual :math:`M^* = R`
(under an orthonormal trial basis), and its canonical instance is the
angular spherical-harmonic frame.

.. _frame-galerkin-in-general:

In general
----------

The Galerkin discipline is characterised by **test space equals
trial space** — the :class:`~orpheus.numerics.frame.GalerkinFrame`
case. The defining identity is

.. math::
   :label: galerkin-self-adjoint

   M^* \;=\; R
   \quad \text{(under the } V \text{ inner product, orthonormal basis)}.

.. vv-status: galerkin-self-adjoint documented

i.e. the analysis face's Hilbert adjoint is its reconstruction. This is
why a single basis :math:`\{e_k\}` produces both :math:`M` and
:math:`R`:

.. math::
   :label: galerkin-construction

   (M f)_k &\;=\; \langle e_k, f \rangle_V, \\
   R \, c     &\;=\; \sum_k c_k\,e_k.

.. vv-status: galerkin-construction documented

.. warning::

   The identity :math:`M^* = R` holds when the basis
   :math:`\{e_k\}` is orthonormal in :math:`V`. When the basis is
   only orthogonal — the case for the no-:math:`4\pi/(2\ell+1)`-
   prefactor real spherical harmonics ORPHEUS uses — the Hilbert
   adjoint :math:`M^*` and the addition-theorem reconstruction face
   :math:`R` differ by a **diagonal-in-:math:`\ell` scaling**, and
   *which* diagonal depends on the coefficient-space metric
   (:ref:`frame-parseval-metric`). Under the frame's shipped
   **Parseval** metric :math:`G^{-1}`, **on a frame whose measured Gram
   is** ``DIAGONAL``, that diagonal collapses to a single scalar, the
   total weight :math:`W = \sum_n w_n`:

   .. math::
      :label: galerkin-strict-adjoint-vs-reconstruction

      (M^* c)_n
      &\;=\; \sum_\ell \frac{2\ell+1}{4\pi}
             \sum_m Y_\ell^m(\hat\Omega_n)\,c_\ell^m
        \;=\; \frac{(R\,c)_n}{W}
        \quad\text{(Hilbert adjoint, Parseval metric)}, \\
      (R c)_n
      &\;=\; \sum_\ell (2\ell+1)\,\sum_m Y_\ell^m(\hat\Omega_n)\,
             c_\ell^m
        \quad\text{(with factor — addition-theorem)}.

   ⚠ The scalar collapse is a property of the *pairing*, not of the
   metric being correct. Since campaign 1 P7 a ``DENSE`` frame also
   carries the right Parseval metric — the matrix :math:`G^{+}` — and
   the collapse still need not hold there, because it additionally
   requires each live :math:`\ell` block of :math:`G` to be **one
   number**. `[M]` on the slab GL(8) frame at :math:`L=2` that block is
   :math:`[0.4,\,0.8,\,0.8]`, so no :math:`G_\ell` exists and
   :math:`M^{*} \ne R/W` at any metric whatsoever. See
   :ref:`spaces-metric-frame-square` on
   :doc:`/theory/foundations/spaces` for the decidable form of the
   condition.

   .. (vv-status rationale) Representational identity: distinguishes the
      Hilbert adjoint M* from the reconstruction face R (with 2ℓ+1) — the
      ERR-039 distinction, re-keyed at F-0 (2026-08-23) to the Parseval
      metric, under which the two differ by the single scalar W. Each face
      is verified under its own label: M* = S_0∘G⁻¹ = R/W by
      :eq:`hilbert-adjoint-equals-metric-times-S0`, the (2ℓ+1) synthesis by
      :eq:`sh-addition-theorem-reconstruction`, and the collapsing identity
      d_ℓ·G_ℓ = W by :eq:`frame-square-closure-sh`. A face-distinction
      framing, not a separate solver claim.
   .. vv-status: galerkin-strict-adjoint-vs-reconstruction documented

   .. no-implementation:: galerkin-strict-adjoint-vs-reconstruction
      :kind: identity

      **Nothing implements this**, because it is a *contrast* between
      two operators, each of which is implemented and declared under
      its own label —
      :eq:`hilbert-adjoint-equals-metric-times-S0` (8 declared sites)
      and :eq:`sh-addition-theorem-reconstruction`. No symbol computes
      "the difference between :math:`M^*` and :math:`R`"; the whole
      content of the equation is that the difference is the scalar
      :math:`W`, which :eq:`frame-square-closure-sh` states. Declaring
      a site here would attribute a face-distinction to one of the two
      faces it distinguishes. (Before this declaration the graph
      inferred two implementers from shared name tokens, one of them
      ``solve_sn_adjoint`` — an SN solver entry point that never
      touches a spherical-harmonic face.)

   ⛔ **Corrected 2026-08-23 (F-0).** This equation's first line read
   :math:`(M^*c)_n = \sum_{\ell,m} Y_\ell^m c_\ell^m` — "*the strict
   adjoint is the naked synthesis, no factor*" — while the prose
   beneath it said :math:`M^* = g_C\,S_0`. **Both were published
   simultaneously and they contradict each other**, which is the
   diagnostic: each is the adjoint under a *different* coefficient
   metric (Euclidean for the first, the continuum Gram :math:`g_C`
   for the second), and neither block said which. Naming the metric
   is what makes the statement well-posed; the frame carries
   :math:`G^{-1}`, so the shipped answer is the third one,
   :math:`M^* = R/W`.

   The analysis face's representation transpose
   :meth:`frame.analysis.apply_transpose
   <orpheus.numerics.basis.Basis.analyze_transpose>` is
   :math:`w_n\,S_0` (the *naked* synthesis weighted by the
   quadrature weight) — that one is metric-free and unchanged; its
   metric-aware Hilbert adjoint ``frame.analysis.H`` is
   :math:`M^* = S_0 \circ G^{-1} = R/W`; and
   :meth:`frame.reconstruction.apply
   <orpheus.numerics.basis.Basis.reconstruct>` returns :math:`R`
   (with the :math:`(2\ell+1)` factor). The adjoint-face dishonesty
   that conflated the bare transpose with the Hilbert adjoint was
   caught by QA review and corrected as ERR-039 (see the project's
   L0 error catalog); F-0 is that entry's second chapter — right
   Gram, wrong side. :math:`R` and :math:`M^*` are both useful and
   coexist as distinct frame faces; they differ by exactly
   :math:`W`, so the docstrings and this page name them explicitly.

The Galerkin invariant :eq:`galerkin-pair` is then a consequence of
the basis being orthogonal in :math:`V`-inner-product. Concretely,
for the spherical-harmonic Galerkin pair:

.. math::

   (M R)_{\ell m, \ell' m'}
   &\;=\; \sum_n w_n\,Y_\ell^m(\hat\Omega_n)\,
                       Y_{\ell'}^{m'}(\hat\Omega_n) \\
   &\;=\; \frac{4\pi}{2\ell+1}\,
          \delta_{\ell\ell'}\,\delta_{mm'},

so :math:`M R = \mathrm{diag}(4\pi/(2\ell+1))`. Composing with the
reconstruction face's :math:`(2\ell+1)` factor (the addition-theorem
weight) yields :math:`M R = 4\pi I` — the L1 identity that the
test
``tests/numerics/test_spherical_harmonic_space.py``
verifies at :math:`L = 2,\,3,\,4` (see :eq:`pi-r-equals-4pi-i` in
:ref:`spherical-harmonics`). This :math:`4\pi` is precisely the
frame's **tightness constant** :math:`c_V`: the frame operator
:math:`S = T^*T` equals :math:`4\pi\,I`, so the spherical-harmonic
frame is a 4π-tight frame.

.. note::

   The identity :math:`M R = 4\pi I` is **not** identity-on-the-
   nose because the no-prefactor convention pushes the
   :math:`4\pi/(2\ell+1)` factor onto the orthogonality. A
   spherical-harmonic frame with a strict :math:`M R = I`
   invariant could be built by dividing the analysis weights by
   :math:`4\pi`, but the project chose to absorb the factor at the
   reconstruction face (the :math:`(2\ell+1)` weight) so the
   addition theorem reads cleanly. See :ref:`spherical-harmonics`
   for the convention rationale.


.. _frame-spherical-harmonic-galerkin:

Applied to spherical-harmonics projection
-----------------------------------------

The spherical-harmonic angular frame is the canonical
:class:`~orpheus.numerics.frame.GalerkinFrame`. Its Galerkin discipline
is *forced*, not chosen: the anisotropic scattering source is a rotation-
invariant (zonal) integral kernel whose eigenbasis — by Funk–Hecke — is
the spherical harmonics, and the eigenbasis of a self-adjoint operator is
orthogonal. Every term of the frame is named below: the analysis face
:math:`M` (the quadrature-weight :math:`w_n` projection onto the harmonic
moments), the reconstruction face :math:`R` (the :math:`(2\ell+1)`
addition-theorem synthesis), the diagonal eigenvalue operator
:math:`\Lambda` (the Legendre moments :math:`\Sigma_{s,\ell}`), and the
tightness constant :math:`c_V = 4\pi` for which :math:`M R = 4\pi I`
(:eq:`pi-r-equals-4pi-i`, derived in the
:ref:`general treatment of the Galerkin frame <frame-galerkin-in-general>`
above). The whole kernel is the spectral theorem
:math:`S = R\circ\Lambda\circ M = U\Sigma U^*` written out.

The anisotropic scattering source operator is an angular **integral
kernel** (:ref:`integral-kernel-category`, :doc:`/theory/foundations/operator_algebra`):
the source at :term:`ordinate` :math:`\hat\Omega` reads the flux at *every*
ordinate, weighted by the scattering kernel

.. math::
   :label: scattering-zonal-kernel

   (S_{\rm aniso}\,\psi)(\hat\Omega)
   \;=\; \int_{4\pi}
         \Sigma_s(\hat\Omega \cdot \hat\Omega')\,
         \psi(\hat\Omega')\;d\hat\Omega',

where the kernel depends on the incoming and outgoing directions
**only through their cosine** :math:`\hat\Omega \cdot \hat\Omega'`. A
kernel of this form is called a **zonal** kernel on the sphere
:math:`S^2` (it is invariant under a simultaneous rotation of both
arguments). Two classical theorems pin its spectrum.

.. vv-status: scattering-zonal-kernel documented
   The zonal-kernel form of the anisotropic scattering source is the
   literature-standard transport definition (Bell & Glasstone 1970
   §1.6; Lewis & Miller 1993 §4.7); it is a transcription, not a
   solver claim. The implementing kernel R∘Λ∘M is pinned by the
   0-ULP windowed-vs-full crosscheck
   ``tests/sn/operators/test_scattering_kernel_crosscheck.py`` and the
   addition-theorem identity :eq:`real-sh-addition-theorem`.

**Funk–Hecke theorem.** For any zonal kernel
:math:`k(\hat\Omega \cdot \hat\Omega')` on :math:`S^2`, the spherical
harmonics are eigenfunctions of the integral operator
:math:`(T_k f)(\hat\Omega) = \int_{S^2} k(\hat\Omega\cdot\hat\Omega')\,
f(\hat\Omega')\,d\hat\Omega'`, with an eigenvalue that depends on
:math:`\ell` **only** (not on :math:`m`):

.. math::
   :label: funk-hecke-eigenvalue

   T_k\,Y_\ell^m \;=\; \lambda_\ell\,Y_\ell^m,
   \qquad
   \lambda_\ell \;=\; 2\pi \int_{-1}^{+1} k(t)\,P_\ell(t)\,dt.

.. vv-status: funk-hecke-eigenvalue documented
   The Funk–Hecke eigenvalue formula is a classical result (Müller
   1966, *Spherical Harmonics*, Lecture Notes in Mathematics 17,
   §"Funk-Hecke"); transcribed here as the structural ground for the
   ownership ruling. The eigenvalues realised in code (the per-ℓ
   Legendre moments Σ_{s,ℓ}) are the diagonal of
   :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer` Λ.

Applied to the scattering kernel
:math:`k = \Sigma_s(\hat\Omega\cdot\hat\Omega')`, the eigenvalues are
exactly the **Legendre moments of the differential scattering cross
section**, :math:`\lambda_\ell = \Sigma_{s,\ell}` — which are
precisely the per-:math:`\ell` block entries of the diagonal operator
:math:`\Lambda` =
:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
(:eq:`scattering-as-tensor-product-sum`, :doc:`/theory/foundations/operator_algebra`). The
spherical harmonics are therefore not *a* convenient basis for
scattering — they are *the* eigenbasis, forced by the rotational
invariance of the kernel.

**The kernel factorisation is the spectral theorem written out.** A
self-adjoint operator :math:`A` on a finite-dimensional space has the
spectral decomposition :math:`A = U\,\Sigma\,U^*`, with :math:`U` the
unitary whose columns are the eigenvectors and :math:`\Sigma` the
diagonal of eigenvalues. The discrete ORPHEUS scattering kernel is
*literally* this decomposition:

.. math::
   :label: scattering-spectral-theorem

   S_{\rm aniso}
   \;=\;
   \underbrace{R}_{=\,U}\;\circ\;
   \underbrace{\Lambda}_{=\,\Sigma}\;\circ\;
   \underbrace{M}_{=\,U^*},

.. (vv-status rationale) Representational identity: the anisotropic scattering
   kernel written as the spectral theorem S = R∘Λ∘M = UΣU*. The implementing
   kernel R∘Λ∘M is pinned by the 0-ULP windowed-vs-full crosscheck
   ``tests/sn/operators/test_scattering_kernel_crosscheck.py`` and the
   addition-theorem identity :eq:`real-sh-addition-theorem` (same kernel as the
   sibling :eq:`scattering-zonal-kernel`). A spectral-theorem framing, not a
   separate solver claim.
.. vv-status: scattering-spectral-theorem documented

with

* :math:`M` (the frame's **analysis** face, ``frame.analysis``) =
  :math:`U^*` — the change of basis *into* the eigenbasis (project the
  flux onto its harmonic moments :math:`\phi_\ell^m`);
* :math:`\Lambda` (=
  :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`) =
  :math:`\Sigma` — the diagonal multiply by the spectrum
  :math:`\Sigma_{s,\ell}`, one scalar per :math:`\ell`-block;
* :math:`R` (the frame's **reconstruction** face,
  ``frame.reconstruction``) = :math:`U` — the synthesis *out of* the
  eigenbasis (rebuild the per-ordinate source).

The **addition theorem** :eq:`real-sh-addition-theorem` —
:math:`\sum_m Y_\ell^m(\hat\Omega)\,Y_\ell^m(\hat\Omega') =
P_\ell(\hat\Omega\cdot\hat\Omega')` — is exactly the *spectral
resolution* of the zonal kernel: it expresses the rank-:math:`(2\ell+1)`
projector onto the degree-:math:`\ell` eigenspace as an outer product
of harmonics. Reading :math:`S = R\circ\Lambda\circ M` as
:math:`U\Sigma U^*` is what makes the conjugation
:math:`S = \texttt{frame.conjugate}(\Lambda)` (the scattering **2-cell**
of the operator-algebra double category, :doc:`/theory/foundations/operator_algebra`) a
*spectral* statement and not merely a convenient bracketing.

**Schur's lemma fixes the block structure and the weights.** The
function space :math:`L^2(S^2)` decomposes into the
:math:`SO(3)`-irreducible subspaces
:math:`L^2(S^2) = \bigoplus_\ell V_\ell`, where
:math:`V_\ell = \mathrm{span}\{Y_\ell^m\}_{m=-\ell}^{\ell}` is the
degree-:math:`\ell` eigenspace of dimension :math:`2\ell+1`. The
scattering source operator commutes with every rotation (it is built
from a zonal kernel), so it lies in the :math:`SO(3)`-**commutant**.
By **Schur's lemma**, any operator in the commutant acts as a *scalar*
on each irreducible block — which is the :math:`m`-independence of the
Funk–Hecke eigenvalue, now derived from symmetry rather than computed
from an integral. The block dimensions :math:`\dim V_\ell = 2\ell+1`
are the origin of:

* the :math:`(2\ell+1)` reconstruction factor on the frame's
  reconstruction face (:eq:`sh-addition-theorem-reconstruction`,
  :ref:`spherical-harmonics`) — the irrep dimension; and
* the **continuum** Gram diagonal
  :math:`g_C = \mathrm{diag}(4\pi/(2\ell+1))`
  (:eq:`sh-space-metric`) — the :math:`SO(3)` Plancherel weight on
  each block. A degree-exact cubature reproduces it as the *discrete*
  Gram :math:`G`, whose **inverse** is the metric the frame's
  coefficient codomain carries (:ref:`frame-parseval-metric`); the
  two are reciprocal, so keep the direction straight.

So the entire numerical apparatus of the spherical-harmonic frame —
the per-:math:`\ell` block structure, the :math:`(2\ell+1)` factor,
the :math:`4\pi`-tightness — is the representation theory of
:math:`SO(3)` acting on a rotationally-invariant kernel. The frame is
Galerkin **because** the eigenbasis of a self-adjoint (here
:math:`SO(3)`-invariant) operator is orthogonal: ``test is trial`` is
forced by symmetry, not chosen.


.. _frame-moment-space-single-home:

The coefficient space has ONE home — the bound basis, never the integer
------------------------------------------------------------------------

Everything above derives the spherical-harmonic frame's *structure* from
the symmetry of the kernel. This subsection settles a different question,
and it is the one a consumer gets wrong: **where does the coefficient
space come from?** The frame is a Stage-2 generator — binding a basis to a
measure induces the two faces *and* the space those faces land in, at one
site. A consumer that keeps the faces and re-derives the space has
retained the induced part, and the copy is silently a different object the
moment the family changes.

Landed 2026-09-02, #429 tracker 2.5, as the pre-step the ERR-080 repair
needs.

The defect: eight homes for one space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`[M]` 2026-09-02, ``git grep -n "SphericalHarmonicSpace.from_L" HEAD --
orpheus/`` over the pre-step tree returns **13** lines, of which **8 are
executable calls** and 5 are docstring mentions. So the angular
coefficient space had **eight** homes. One is legitimate —
:attr:`SphericalHarmonicBasis.space
<orpheus.numerics.basis.SphericalHarmonicBasis.space>`, where a basis
answers *what do my coefficients live in?* — and **seven** were production
consumers re-minting it from the integer :math:`L`:

.. list-table:: `[M]` The seven re-mint sites, at the pre-2.5 tree
   :header-rows: 1
   :widths: 40 34 26

   * - Site
     - What it was minting
     - Now reads
   * - ``LegendreMomentScattering.from_material_xs``
     - the endomorphic ends of :math:`\Lambda`
     - ``basis.space`` (the tier-2 classmethod takes the basis)
   * - ``N2NMomentOperator.from_material_xs``
     - the same, for the :math:`(n,2n)` channel
     - ``basis.space``
   * - ``ScatteringOperator._sh_space``
     - the ends of the internally-minted :math:`\Lambda`
     - ``_moment_space`` = ``frame.basis.space``
   * - ``fission.py``'s ``_sh_space_l0``
     - the :math:`\ell = 0` ends of :math:`F`
     - ``self.frame.basis.space`` (helper retired)
   * - ``N2NOperator.full_n2n_kernel``
     - the :math:`\ell = 0` ends of the :math:`(n,2n)` dyad
     - ``self.frame.basis.space``
   * - ``MomentField._space_for_mesh_and_L``
     - the moment field's angular HEAD factor
     - ``SNMesh.moment_space(L)`` — the hub, which reads
       ``mesh.quad.angular_frame(L).basis.space``
   * - ``HarmonicMomentFlux.truncate``
     - the head of the truncated space
     - ``head.truncated(L_new)`` — the head's OWN family

.. note:: **2026-09-07, CS4c step 6 item 6.2b — the field row's "now reads"
   is one hop longer, because the PRODUCT moved to the hub.** Tracker 2.5
   made the moment field's angular *head* a read of the frame; it left the
   *product* — head :math:`\otimes` cell group — being re-minted on every
   call. Item 6.2b gives that product to the carrier:
   :meth:`SNMesh.moment_space
   <orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space>` is a cache keyed
   on ``(L, spatial_moments)`` holding **one object per key**, and the
   moment family is now entirely a set of CONSUMERS of it — the factories
   (``from_mesh_and_L``, ``zeros_for_mesh_and_L``), the ``space_on``
   admission reference, and the sweep's iterate wrap all hold the SAME
   instance (``is``, not merely ``==``). The head is still read off the
   frame at ``quad.angular_frame(L).basis.space``, so the row's third
   column stays true one hop in; what changed is *who owns the product*.
   `[M]` until 6.2b the field-side mint ran **113 of the 118** ``*``
   products per 2-D windowed solve — 58 from the boundary leaf's guard, 55
   from the sweep's iterate wrap. The refusal survives the move and changes
   its reason: a carrier that owns no moment space (a bare
   :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`, which has
   no quadrature) is refused by name at the same typed door, now spelled
   against the hub's surface rather than against ``quad``.

Two further narrowings guarded the frame itself: both
:class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`
doors (the constructor and
:meth:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame.from_galerkin`)
tested ``isinstance(basis, SphericalHarmonicBasis)`` and refused anything
else.

⛔ **Why this is a defect and not merely a duplication.** An integer does
not say *which family*. Every one of the seven copies silently chose the
full-sphere real spherical harmonics — which is right on a full-sphere
rule and wrong on a 1-D one, where the surviving harmonics are the
trivial isotypic component :math:`\{Y_\ell^0\} \cong \{P_\ell\}` of
:math:`S^2/O(2)_a` (:ref:`manifold-s2-so2`, ERR-080). The day the
quadrature binds that Legendre basis, each copy disagrees with the frame
at the ``(name, shape)`` composability guard, and — because fission and
:math:`(n,2n)` mint their ends at :math:`\ell = 0` on *every* solve — the
**first** disagreement is at :math:`L = 0`, on an isotropic problem, in a
channel that has nothing to do with anisotropic scattering. That is a
seven-site blast radius sitting between the ERR-080 repair and a green
tree; tracker 2.5 removes it before the repair, not with it.

`[M]` after the step the same command returns **6** lines, of which
**exactly one** is an executable call — ``spherical_harmonic_basis.py:403``,
the basis's own ``space`` property — and the other five are docstring
mentions. (``SphericalHarmonicSpace.truncated`` called
``type(self).from_L`` inside the space's own module, which is where a
family is *entitled* to name itself; since 2026-09-08 it delegates to
:func:`~orpheus.numerics.spaces.moment_head.truncated_head`, which re-mints
through the head axis's generator instead — see the truncation paragraph
below.)

Which space, though — ``basis.space`` or ``basis_space``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

   **Answered twice. Tracker 2.5 (2026-09-02) chose the basis's
   continuum ``space``; CS4c step 6 item 6.2c-ii (2026-09-08, ruling
   R-6.2c-1) OVERTURNED that and binds the frame's Parseval-dressed
   ``basis_space``.** The whole of this sub-subsection is preserved as
   the original argument — it is *why anyone chose the continuum end*,
   and a future reader owes it — with the re-measurement that refuted it
   set beside each claim. The one-sentence ruling is *the carrier's norm
   is the field's energy*: the moment vector a solver holds is what the
   analysis face emitted, so the inner product it carries must be the one
   under which its norm IS the band-limited field's :math:`L^2(\mathrm
   d\Omega)` energy — and that is Parseval's :math:`G^{-1}`, a theorem
   (:eq:`frame-parseval-isometry`), not a convention.

A frame carries **two** spellings of the coefficient space and they are
different objects. The basis's own
:attr:`Basis.space <orpheus.numerics.basis.Basis.space>` carries the
**continuum** Gram :math:`g_C = 4\pi/(2\ell+1)`
(:eq:`sh-space-metric`); the frame's
:attr:`~orpheus.numerics.frame.FrameBase.basis_space` **replaces** that
with the Parseval metric :math:`G^{-1}` — the inverse *discrete* Gram —
because the analysis face returns covariant moments
(:ref:`frame-parseval-metric`, and
:ref:`frame-parseval-what-was-wrong` for what the pre-F-0 continuum
reading cost).

*(Written 2026-09-02, and preserved:)* ⚠ So the fork is a live hazard for
a reader of this page: the corpus stamps the continuum Gram with a ⛔
*"the wrong side for covariant moments"* (:ref:`spherical-harmonics`, the
three-metric table), and tracker 2.5 binds the operator ends to exactly
that space. **It is not a regression of F-0, because it is not the same
object.** F-0's verdict is about the **analysis face's codomain**, where
the value IS a covariant moment vector. :math:`\Lambda`'s ends are the
domain and codomain of an **endomorphism** on the coefficient index set —
an object whose adjoint, under any metric diagonal in :math:`\ell`, is
its transpose.

⛔ **REFUTED 2026-09-08 (item 6.2c-ii, ruling R-6.2c-1) — the argument is
sound and its premise was never checked.** The reasoning above is
*conditional*: it says the continuum end is ADMISSIBLE for
:math:`\Lambda`, because an :math:`\ell`-diagonal metric commutes with a
per-:math:`\ell` scalar. It never established that the dressed end is
INADMISSIBLE, and the sentence it was defended with — *"the dressed end
would move* :math:`\Lambda`\ *'s Hilbert adjoint on 10 of 33 rows (the
dense-Gram rows)"* — does not survive re-measurement. Three findings,
each with its denominator
(``tests/transport/frames/test_moment_metric_fork_premise.py``, `[M]`
2026-09-07 on ``main`` @ ``79d2944a``, probes ``scratch/_step6_2c/``
``p3_scan_161.py`` / ``p4_lambda_adjoint.py`` / ``p8_parseval.py`` /
``p9_on_range.py``):

#. **The count and its attribution are both wrong.** Applied to an
   arbitrary head draw, :math:`\Lambda^{*}` differs from
   :math:`\Lambda^{\top}` under the dressed end on **5 of 33** rows, and
   **3 of the 5 are DIAGONAL-Gram** rows. The mechanism is not Gram
   density: it is the Parseval metric's Moore–Penrose **projection** of
   the slots a folded rule cannot see (on a σ-folded rule the σ-odd
   harmonics vanish at every node, so :math:`\operatorname{diag} G = 0`
   there and :math:`G^{+}` zeroes them, while :math:`g_C` does not).
   :math:`\Lambda^{*}` is then :math:`P\Lambda^{\top}P`, not
   :math:`\Lambda^{\top}`.
#. ⭐ **On the only inputs production can hold, the objection does not
   bite at all.** A moment vector a solver holds is a COVARIANT
   :math:`\varphi = M\psi`, never a raw coefficient draw. Re-run on
   :math:`\varphi = M\psi` for a random angular field: :math:`\Lambda^{*}
   \ne \Lambda^{\top}` on **0 of 33** rows (all :math:`\le
   1.9\times10^{-16}`). The slots the pseudo-inverse zeroes are exactly
   the slots whose moment is identically zero for every field the rule
   can see — `[M]` 41 %…79 % of a raw draw's mass is off the range of
   :math:`M` on the folded and level-symmetric rows.
#. **The companion figure reproduces under no statistic.** The same gate
   docstring recorded *"the dressed metric would move ``apply_metric`` by
   96–161 %"*, naming no statistic, no fixture and no denominator. `[M]`
   the draw-free per-element movement :math:`|p_\ell/g_\ell - 1|` spans
   **0.5 %…100.0 %** over the 33 rows and 0.5 %…222 % over 60 rows at
   :math:`L \le 4`; **no row lands in** :math:`[155\,\%, 170\,\%]`. Where
   ``161 %`` came from is unrecoverable from the tree.

And the price of the *other* arm was never on the table. `[M]` the
Parseval ratio :math:`\lVert M\psi\rVert^2_{\Phi_L} /
\lVert\psi\rVert^2_W` for a band-limited :math:`\psi`:

.. list-table:: `[M]` What each end costs — re-measured 2026-09-08,
               33 shipped (rule, :math:`L`) rows
   :header-rows: 1
   :widths: 26 22 24 28

   * - end bound to :math:`\Phi_L`
     - Parseval holds
     - :math:`\Lambda^{*} = \Lambda^{\top}` on a physical :math:`\varphi`
     - what it costs
   * - dressed ``basis_space`` (:math:`G^{-1}` / :math:`G^{+}`) —
       **shipped since 6.2c-ii**
     - **33 / 33**, ratio :math:`1.0000000000`
     - **33 / 33**
     - nothing on any production input; 5 / 33 off the range of
       :math:`M`, which no solver can reach
   * - continuum ``basis.space`` (:math:`g_C`) — tracker 2.5's choice
     - **0 / 33**, ratio :math:`3.41\ldots157.91`
     - 33 / 33
     - the frame square's isometry, i.e. the F-0 / P7 repair itself
       (`[M]` 148 battery reds if reverted)

The upper end of that ratio is exact rather than sampled: on a
degree-exact full-sphere rule :math:`G = g_C`, so at :math:`\ell = 0` the
ratio is :math:`(4\pi)^2 = 157.9137`. (Re-measured independently for this
page at one draw per row: **0 of 33** at :math:`1`, range
:math:`4.07\ldots157.9137`; the lower end is draw-dependent, the upper
end is not.)

⟹ the fork was never a correctness question about the converged flux
(`[M]` ``scalar_flux`` is ``array_equal``, the residual trajectory
bit-identical and ``n_inner`` unchanged under either end), and never a
:math:`\Lambda`-adjoint question (no physical input separates them). It
is: **does the moment carrier's norm mean the field's energy?** Under
:math:`G^{-1}` it does, exactly; under :math:`g_C` the recorded number is
:math:`\lVert G\Delta c\rVert_{g_C}`, a quantity with no reading in the
transport equation and differing from the un-windowed arm's
:math:`\lVert\Delta\psi\rVert` by :math:`\approx 4\pi`.

The identity the step installs:

.. math::
   :label: moment-space-read-off-the-frame

   \Phi_L \;\equiv\; \bigl(\text{the frame } q \text{ binds at } L
   \bigr).\texttt{basis\_space},
   \qquad
   \Lambda,\,F_0,\,N_0 : \Phi_L \to \Phi_L ,
   \qquad
   \text{head}(\phi) \;=\; \Phi_L ,

.. (vv-status rationale) Named-field-typing identity: the angular moment
   space is the coefficient space the quadrature's frame carries, read
   off the frame, and it is the shared end of every moment-space operator
   and the head factor of every moment field. Not a solver claim — no
   eigenvalue, no flux. The verifiable content is the foundation gate
   ``tests/transport/frames/test_moment_space_is_read_off_the_frame.py``
   (the ROUTE / METRIC / DOOR trio) plus the composability guard
   ``A.domain == B.codomain`` it leans on, and — since the head became
   axis-built — the hub/frame agreement gate
   ``tests/sn/mesh/test_hub_and_frame_agree_on_the_moment_space.py``.
.. vv-status: moment-space-read-off-the-frame documented

.. note::

   ⛔ **The right-hand side read** ``(the basis q's frame binds at
   L).space`` **until CS4c step 6 item 6.2c-ii (2026-09-08).** The
   *reader* half of the identity — nobody re-mints from :math:`L` — is
   untouched and is what this subsection is about. What moved is WHICH of
   the frame's two coefficient spellings the tree binds: ruling
   **R-6.2c-1** replaced the basis's continuum ``space`` with the frame's
   Parseval-dressed ``basis_space``, so :math:`\Phi_L` now carries
   :math:`G^{-1}` (or a positioned :math:`G^{+}`) instead of
   :math:`g_C`. The subsection below preserves the original argument and
   sets the re-measurement beside it.

with :math:`q` the quadrature, so that *which family* is the quadrature's
decision (:meth:`Quadrature._harmonic_basis
<orpheus.numerics.quadrature.Quadrature._harmonic_basis>` derives it from
the point set the measure lives on — the σ-even restriction on a folded
rule, the full harmonics otherwise) and every consumer is a reader.

**Measured, on the shipped rules.** Over **33 rows** — eleven rule
constructions drawn from **all five** shipped ``Quadrature`` classmethod
factories (`[M]` ``vars(Quadrature)``: ``gauss_legendre`` 2/8/16,
``level_symmetric`` 4/8, ``lebedev`` 11/17, ``product`` (4,6)/(8,8),
``folded_product`` (2,4)/(4,8)) × :math:`L \in \{0,1,2\}`, each frame
built as ``HarmonicFrame.from_galerkin(q.angular_frame(L))``:

.. list-table:: `[M]` 2026-09-02 — the two spellings against the ``from_L(L)`` mint they replace
   :header-rows: 1
   :widths: 40 20 40

   * - Comparison
     - Rows
     - Reading
   * - ``frame.basis.space == from_L(L)``
     - 33 / 33
     - ``(name, shape)``-equal
   * - ``inner_product_weights`` ``array_equal`` to ``from_L(L)``'s
     - 33 / 33
     - metric-identical — so nothing downstream can move
   * - ``frame.basis.space is from_L(L)``
     - 0 / 33
     - content-equal, never the same object
   * - ``frame.basis_space == frame.basis.space``
     - 33 / 33
     - ``(name, shape)``-equal — **equality was metric-blind**
   * - ``frame.basis_space`` metric ≠ ``frame.basis.space`` metric
     - 33 / 33
     - the fork a ``==`` gate structurally cannot see

The last two rows are the reason the gate for this step asserts the metric
ARRAY and not the space: a ``==`` comparison would have passed under
*either* choice (``vv-principles`` #19 — the reading that discriminates is
the wrong-structure one).

.. note::

   ⛔ **Row 4 is HISTORY since CS4c step 6 item 6.2c-ii (2026-09-08).**
   The numbers above are unchanged as a record of the 2026-09-02 tree, and
   rows 1–3 still reproduce; row 4 does not. Both harmonic heads became
   AXIS-BUILT — one
   :class:`~orpheus.numerics.axis.HarmonicAxis` (rectangular) or
   :class:`~orpheus.numerics.axis.LegendreAxis` (flat) whose MEASURE is
   the head's metric — and an axis-built space's identity IS its axis
   tuple, weights bytes included (:ref:`spaces-identity-bridge`). So the
   metric entered the identity and the two heads separated: `[M]`
   re-measured for this page on the same 33 rows,
   ``frame.basis_space == frame.basis.space`` is **0 of 33**, and every
   one of the 33 dressed heads reports ``axes is not None``. Row 5 stands
   and is now *implied* by row 4 rather than hidden from it. ⭐ The
   consequence for gate design runs the other way too: the ``==``
   assertion that could adjudicate nothing on the 2026-09-02 tree is,
   on this one, exactly the discriminating instrument — which is why the
   hub/frame agreement gate can assert ``==`` and mean it
   (``tests/sn/mesh/test_hub_and_frame_agree_on_the_moment_space.py``).

The size of the fork is **draw-free and exact**, not a sampled number. On
a degree-exact full-sphere rule the discrete Gram reproduces the continuum
one, :math:`G = g_C`, so the two metrics are exact reciprocals and their
ratio is

.. math::

   \frac{(\texttt{basis\_space})_{\ell}}{(\texttt{basis.space})_{\ell}}
   \;=\; \Bigl(\frac{2\ell+1}{4\pi}\Bigr)^{2}
   \;=\; 6.332574\times10^{-3},\;
         5.699317\times10^{-2},\;
         1.583143\times10^{-1}
   \quad (\ell = 0,1,2),

i.e. the same :math:`157.9\,/\,17.5\,/\,6.3` per-:math:`\ell` factors
:ref:`frame-parseval-what-was-wrong` already records for the pre-F-0
metric, read in the other direction. On a ``gauss_legendre`` rule, whose
weights sum to :math:`2` rather than :math:`4\pi`, the discrete Gram is
:math:`2/(2\ell+1)` and the ratio is :math:`(2\ell+1)^2/8\pi` —
:math:`3.978874\times10^{-2}` and :math:`3.580986\times10^{-1}` at
:math:`\ell = 0, 1`. (That :math:`\Sigma w = 2` is itself the slab's
signature: a 1-D rule integrates over :math:`\mu`, not over the sphere.)

Why the continuum space was chosen as the end for :math:`\Lambda`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **This sub-subsection is the tracker-2.5 argument, PRESERVED. Its
conclusion was overturned on 2026-09-08 (item 6.2c-ii, R-6.2c-1); read it
with the three-point refutation above and the closing note below.** The
heading read *"Why the continuum space is the right end for*
:math:`\Lambda`\ *"* until then.

:math:`\Lambda` is a scalar per :math:`\ell` (Funk–Hecke, above), so it
**commutes** with any metric that is diagonal in :math:`\ell` and constant
within each :math:`\ell` block. Its Hilbert adjoint under such a metric,
:math:`\Lambda^{*} = M_g^{-1}\Lambda^{\top}M_g`, therefore collapses to
:math:`\Lambda^{\top}`. The continuum Gram is such a metric **by
construction and on every rule** — it is a property of the harmonics, not
of the sampling. The Parseval dressing is not: it inherits whatever
structure the *discrete* Gram has.

`[M]` 2026-09-02, on a two-group four-cell slab binding. **Measured at
the OPERATOR level, not with a probe vector**: :math:`\Lambda^{*}` is
built column by column by pushing each :math:`e_k` through both arms, so
the numbers below are draw-free (a single random application is a
one-draw reading of the same quantity, and its ULP tail is a property of
the reduction order rather than of the operator). Relative Frobenius of
the matrix difference:

.. list-table:: `[M]` Where the fork is observable in :math:`\Lambda^{*}`
   :header-rows: 1
   :widths: 30 14 56

   * - Rows
     - Count
     - Relative movement of :math:`\Lambda^{*}` under the dressed end
   * - inert
     - 23 / 33
     - :math:`\le 1.045\times10^{-16}` — the 18 full-sphere degree-exact
       rows (``level_symmetric``, ``lebedev``, ``product`` at every
       :math:`L`) plus the five :math:`L = 0` rows, where there is one
       mode and nothing to commute with
   * - moves, ``DIAGONAL`` dressing
     - 6 / 33
     - :math:`9.699\times10^{-2}` … :math:`1.372\times10^{-1}` —
       ``gauss_legendre`` (2, 8, 16) at :math:`L = 1` and
       ``folded_product`` (2,4)/(4,8) at :math:`L \ge 1`
   * - moves, ``DENSE`` dressing
     - 4 / 33
     - :math:`1.082\times10^{-1}` … :math:`1.5839` —
       ``gauss_legendre`` (2, 8, 16) and ``folded_product`` (2,4) at
       :math:`L = 2`, where ``basis_space`` installs the matrix
       pseudo-inverse (:ref:`frame-parseval-dense-arm`) and
       ``inner_product_weights`` is ``None`` altogether

⭐ **Read the third column by IDENTITY, not by size.** Every row on which
the fork is observable is a ``gauss_legendre`` rule or a σ\ :sub:`y`-folded
one at :math:`L \ge 1` — precisely the two families whose discrete Gram is
:math:`m`-dependent, i.e. the forged-azimuth rules ERR-080 is about and
the fold whose sub-basis is a genuine quotient. On the six full-sphere
degree-exact rules the choice is inert on all 18 rows. So binding the
frame's dressed space would have moved :math:`\Lambda`'s adjoint by up to
:math:`158\,\%` on exactly the rules this campaign is repairing, and been
invisible everywhere a full-sphere regression fixture would have looked —
which is the shape of a change that ships green and is wrong later.

.. warning::

   ⛔ **Retraction (2026-09-08, item 6.2c-ii).** The table's *numbers* are
   a matrix-level measurement and stand as recorded; its *reading* does
   not. Two things were missing, and each alone reverses the conclusion.

   **(a) The statistic ranges over vectors production cannot produce.**
   The table is a relative Frobenius norm of the whole
   :math:`\Lambda^{*}` matrix, i.e. it pushes every basis vector
   :math:`e_k` through both arms. A solver never holds an arbitrary
   :math:`e_k`; it holds a covariant :math:`\varphi = M\psi`, and the
   columns where the two arms differ are precisely the ones OFF the range
   of :math:`M` — the σ-odd slots a folded rule cannot see, whose moment
   is identically zero for every field. `[M]` re-run on
   :math:`\varphi = M\psi`: the two arms agree on **33 of 33** rows
   (:math:`\le 1.9\times10^{-16}`). The draw-free matrix statistic is the
   *stronger* instrument for "do these operators differ" and the *wrong*
   one for "can this difference reach the solve" — vv-principles Mode 12
   at the level of the measured functional's DOMAIN rather than its
   invariance group.

   **(b) The comparison has one leg.** ":math:`158\,\%` on the rules this
   campaign is repairing" prices the dressed end and never prices the
   continuum one. `[M]` the continuum end breaks Parseval on **33 of 33**
   rows by a factor :math:`3.41\ldots157.91` — the F-0/P7 repair, on
   every shipped rule, including all six full-sphere degree-exact
   families the paragraph above calls "inert". The row that reads *inert*
   in this table is not inert in the other column; it was simply never
   measured.

   ⟹ the sentence *"would have moved* :math:`\Lambda`\ *'s adjoint by up
   to 158 %"* is retained above because it is a true statement about a
   matrix; it is not evidence about the fork, and it must not be
   re-quoted as though it were. The gate docstring that carried its
   companion figures (*"10 of 33 (the dense-Gram rows)"*, *"96–161 %"*)
   was rewritten with its statistic in 6.2c-ii's commit.

With the continuum end, :math:`\Lambda^{*} = \Lambda^{\top}` holds on all
33 rows **exactly**: `[M]` the column-built adjoint matrix and the
transpose of the column-built forward matrix are ``array_equal``,
:math:`\lVert\Lambda^{*} - \Lambda^{\top}\rVert = 0.0`, because the
:math:`g_C \Lambda^{\top} g_C^{-1}` sandwich is a per-mode scalar times
its own reciprocal. (Applied to a random vector instead, the same
identity reads :math:`\le 1.82\times10^{-16}` relative — the reduction
order, not the algebra.)

**Nothing moved** *(true of tracker 2.5, which replaced a* ``from_L(L)``
*mint with a metric-IDENTICAL read; item 6.2c-ii is the step that does
move the metric — see* :ref:`frame-the-one-moment-space` *)*\ **.** The two
ends' metrics are ``array_equal`` (table above), so ``.H`` is the *same
float program* under either — `[M]`
2026-09-02, ``Λ.H`` applied to a fixed draw is ``array_equal`` between
the read space and the ``from_L(L)`` mint it replaces on **33 of 33**
rows — and the forward ``apply`` / ``apply_transpose`` never read an end
metric at all. End-to-end on the ERR-080 gate's own fixture — one-group infinite
medium, ``gauss_legendre(8)``, four cells, reflective/reflective, uniform
per-ordinate source, Krylov inner at ``inner_tol=1e-13``,
``max_inner=5000`` — the converged scalar flux is ``np.array_equal``
between the pre-step tree and the post-step tree at :math:`L = 0, 1, 2`
**and** :math:`3`, ``max|Δ| = 0.0`` on all four. It is bit-identical even
at the orders where the answer is *wrong*, which is the property a
pre-step owes an ``xfail(strict=True)`` gate: it must not perturb the
defect it is clearing the way for.

The door asks for a SURFACE, not for a class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two ``isinstance(basis, SphericalHarmonicBasis)`` narrowings are
replaced by one demand, expressed as a ``runtime_checkable``
:class:`~orpheus.numerics.basis.base.TruncatedBasis` ``Protocol`` with two
members — ``L`` and ``space`` — which are exactly what the frame's mints
and the operator ends read.

A ``Protocol`` rather than a class list, for the same reason
:ref:`manifold-invariance-lower-bound` gives for deriving a basis's
symmetry from its ``domain`` rather than from its subclass: *a class list
is a closed enumeration of today's members, and the point of this step is
that tomorrow's member arrives.* The σ-even restriction and the Legendre
basis on :math:`S^2/O(2)_a` are as much harmonic-family members as the
full harmonics; a door naming one class refuses them, and refuses them
with an ``AttributeError`` three frames later rather than at the door.

`[M]` 2026-09-02, after #429's fused commit: the tree ships **six**
:class:`~orpheus.numerics.basis.Basis` subclasses and **three** satisfy
the surface — :class:`~orpheus.numerics.basis.SphericalHarmonicBasis`,
its σ-even restriction ``MirrorEvenSphericalHarmonicBasis``, and
:class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis`. The other
three — :class:`~orpheus.numerics.basis.IndicatorBasis`,
:class:`~orpheus.numerics.basis.OverlapBasis` and
:class:`~orpheus.numerics.basis.WeightedIndicatorBasis` — do not, and an
indicator trial is refused at both doors with a message naming the
*truncation order* rather than a class.

⛔ **Until later the same day this paragraph read** "five subclasses and
two satisfy the surface", **and ended** "The third member the surface
exists FOR — the Legendre basis on :math:`S^2/O(2)_a` — is tracker 3.4
and **does not ship**: 2.5 is a capability, and ERR-080 stays open."
Both halves were true when written; tracker 3.4 landed
the third member hours afterwards and CLOSED ERR-080. The step's design
is what made that landing a no-op at the doors — which is the point the
paragraph above makes, now with its own witness.

The field's head, and truncation inside the family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A moment field is not built from a frame, it is built from a mesh and an
order — so its space has to be *found*. It is found on the carrier:
:meth:`SNMesh.moment_space
<orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space>`, behind a small
``_CarriesMomentSpace`` Protocol so that a bare material mesh, which owns
no moment space (it has no quadrature, so there is no angular head to
read), is refused with a message that says so instead of failing later on
a shape. The public factory keeps its ``(values, mesh, L)`` signature: the
mesh carries the quadrature, so the head is frame-derived even though the
caller passes an integer.

.. note:: **Dated 2026-09-07, CS4c step 6 item 6.2b.** Until this item the
   paragraph above read *"It is found the same way:*
   ``mesh.quad.angular_frame(L).basis.space``\ *, behind a small*
   ``_CarriesQuadrature`` *Protocol"* — accurate for tracker 2.5's tree,
   where each consumer re-derived the product from the head it had just
   read. The head is still read exactly there; the difference is that the
   read now happens **once per key on the carrier** and the field asks the
   carrier, not the quadrature. The Protocol moved with it: the surface
   demanded is ``moment_space``, not ``quad``, so the refusal is keyed on
   *owning the space* rather than on *carrying a quadrature* — which is
   the honest predicate, because a carrier could in principle carry a
   quadrature and still not own a moment space.

⚠ **Gotcha, and it is older than this step: the FACE's moment codomain
and the FIELD's space are metric-different and compare EQUAL.**
⛔ **RESOLVED 2026-09-08 — read this paragraph and the one after it as
history, and the resolution at** :ref:`frame-the-one-moment-space`. Item
6.2c-ii gave the two the SAME metric (the frame's Parseval one) and made
the heads axis-built, so they are now metric-identical and compare equal
*because* they are the same space rather than in spite of not being. The
account is preserved because it is the clearest statement of the seam,
and because the *shape* of the defect — two producers, one equality
relation that cannot see what they disagree about — is the reusable part.
:meth:`HarmonicFrame.moment_space_on
<orpheus.transport.frames.harmonic_frame.HarmonicFrame.moment_space_on>`
builds the analysis face's codomain from the frame's **dressed**
``basis_space`` — correctly, because that is where a covariant moment
vector lands (:ref:`frame-parseval-metric`) — while
``HarmonicMomentFlux.zeros_for_mesh_and_L`` builds the field's head from
the basis's **continuum** space. `[M]` 2026-09-02, on a two-group slab
carrier at :math:`L = 0, 1, 2`: ``face.codomain == field.space`` is
``True`` at every order and the two heads' metrics differ at every order
(at :math:`L = 2` the face's head has no ``inner_product_weights`` at
all — the ``DENSE`` arm's matrix metric — while the field's carries
:math:`4\pi/(2\ell+1)`). Nothing in the tree can tell them apart, because
identity is ``(name, shape)`` (:ref:`spaces-metric-not-on-the-axis`). The
asymmetry is **unchanged** by tracker 2.5: before it the field's head was
``from_L(L)``, which carries the same continuum Gram the bound basis's
space does. It is recorded here because a reader who has just been told
*"the space is read off the frame"* will otherwise assume the field
inherits the face's metric, and it does not.

⛔ **And it is unchanged by CS4c step 6 item 6.2b (2026-09-07), which is
worth saying because that item looks like it should have closed it.**
Giving the product to the hub changes *which object* the field side holds
— one cached space per ``(L, spatial_moments)`` instead of one fresh mint
per call, `[M]` 113 field-side re-mints in a single 2-D windowed solve
before the item — and changes nothing about its **metric**: the hub reads
the same ``quad.angular_frame(L).basis.space``, so the head still carries
the continuum Gram. The seam therefore sharpens rather than closes: it now
separates **one hub-owned field space** from the frame's Parseval-dressed
codomain, two objects that are ``(name, shape)``-equal and
metric-different, where before it separated a *population* of
content-identical field mints from that codomain. The two-space /
two-metric design of #429 Landing A survives intact, and the
METRIC-IDENTITY gate keeps pinning the continuum metric on the field's
space with the dressed space as its negative control. Item **6.2c**, which
makes the head axis-built and so promotes its weights into the identity
(:ref:`spaces-identity-bridge`), is where that seam is decided — 6.2b
deliberately leaves the ruling open.


.. _frame-the-one-moment-space:

✅ The seam is CLOSED — one moment space, one metric (2026-09-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Item **6.2c-ii** decided it, and closed it in the direction the two
paragraphs above left open. The mechanism is that the harmonic heads
became **axis-built**: each carries ONE MODAL head axis
(:class:`~orpheus.numerics.axis.HarmonicAxis` for the rectangular family,
:class:`~orpheus.numerics.axis.LegendreAxis` for the flat one) whose
MEASURE *is* the head's metric, and an axis-built space's identity is its
axis tuple — weights bytes included. So the metric stopped being invisible
to ``==``, the two heads stopped being interchangeable, and one of them
had to be chosen. Ruling **R-6.2c-1** chose the frame's dressed one.

What that changes, stated as the tree now reads:

.. list-table:: The moment space after item 6.2c-ii
   :header-rows: 1
   :widths: 30 34 36

   * - Question
     - Before (tracker 2.5 → 6.2b)
     - After (6.2c-ii)
   * - the head's metric
     - continuum :math:`g_C = 4\pi/(2\ell+1)`
     - Parseval :math:`G^{-1}`, or a positioned
       :class:`~orpheus.numerics.metric.DenseMetric` :math:`G^{+}` where
       the discrete Gram is dense
   * - who owns the head
     - the basis (``basis.space``)
     - the FRAME (``basis_space``) — and the frame is the head axis's
       :attr:`~orpheus.numerics.axis.Axis.generator`, the object that can
       re-dress it at another order
   * - hub vs. face codomain
     - ``==`` (metric-blind), metric-different
     - ``==`` and metric-IDENTICAL — one space, two owners (ruling O-5);
       ``is`` *within* each owner, ``==`` *across* them, because the frame
       cannot see the carrier
   * - ``head == basis.space``
     - ``True`` (`[M]` 33 / 33)
     - ``False`` (`[M]` 0 / 33) — the continuum head survives only as the
       basis's own coefficient space
   * - what admits a windowed operand
     - ``(name, shape)`` — a metric-blind seam
     - structural identity; there is no seam left to be blind

⭐ **The two owners, and why "one space" is stated as ``==`` and not as
``is``.** The carrier's cached
:meth:`SNMesh.moment_space
<orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space>` and the frame's
:meth:`HarmonicFrame.moment_space_on
<orpheus.transport.frames.harmonic_frame.HarmonicFrame.moment_space_on>`
both build ``<head> ⊗ <cell axes>`` and both read the head off
``frame.basis_space``. They cannot share an *instance*: the frame is
constructed from ``(basis, measure)`` and has no reference to the carrier,
so no cache lookup can reach across. What they share is *identity*, which
under structural equality is the whole of what a composability guard asks.
Within each owner the object is interned (``is``), across them it is
``==`` — gated in
``tests/sn/mesh/test_hub_and_frame_agree_on_the_moment_space.py``.

⚠ **The refusal that had to be re-keyed, and why it is a hazard worth
naming (H-6).** ``moment_space_on`` used to refuse a non-per-ordinate
space by testing ``axes is None`` — which happened to catch a MOMENT space
too, because a moment space was axes-less. Once the head is axis-built a
moment space HAS axes, so the old guard silently lost its subject: `[M]`
handed one, it returned a plausible-looking product instead of raising.
The refusal is now keyed on the generator channel — the leading axis must
narrow to a :class:`~orpheus.numerics.quadrature.directional.Quadrature`
— so it names its subject again. This is the standing lesson that a guard
must be re-derived, not merely re-read, when the thing it inspects changes
shape.

✅ **And the spatial-moment TAIL closed the same day — item 6.2c-iii,
2026-09-08.** This block read, until then: *"What is NOT closed: the
spatial-moment TAIL. A widened moment space (LD,* ``spatial_moments > 1``\ *)
still appends the Euclidean, axes-less* ``SpatialMomentSpace``\ *, so that
product is axes-less overall even though its head is not — and the widened
angular space already carries the scheme's mass-weighted* ``moment_axis``\ *,
i.e. the same factor spelled twice."* The diagnosis was the fix. The hub now
composes its cell group through the fields' own composer,
:meth:`BulkField.compose_spatial_moments
<orpheus.transport.fields._bases.BulkField.compose_spatial_moments>` — the
same one the angular and scalar mints ride, which appends the discretization
scheme's own MODAL
:meth:`moment_axis
<orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis>` — and
``moment_space_on`` stopped dropping the angular space's tail axis, so it
threads that axis object through instead of re-appending a class beside it.
`[M]` 2026-09-08: on a widened 2-D LD carrier the frame's derived tail axis is
the angular space's own axis (``is``-identical), and the hub's cached space and
the frame's derivation are ``==`` with equal hashes at width 2 exactly as at
width 1 — ruling O-5 at both widths. So the tail's metric is no longer a gap:
it is the scheme's cell mass, the same measure the *angular* field's tail
carries, and the moment field's norm is its energy on every factor (ruling
R-6.2c-1's ONE-space principle, applied to the tail). Gated in
``tests/numerics/test_spatial_moment_tail_is_the_schemes_axis.py``.

Truncation is the same rule applied to a *lower* order.
:meth:`HarmonicMomentFlux.truncate
<orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux.truncate>`
asks the current head for **its own family** one order down
(``head.truncated(L_new)``, a second small Protocol) and slices the kept
block by the new head's own shape, so a spherical-harmonic head truncates
to a spherical-harmonic head and a Legendre head to a Legendre head. The
implementation on
:meth:`SphericalHarmonicSpace.truncated
<orpheus.numerics.spaces.SphericalHarmonicSpace.truncated>` delegates to
:func:`~orpheus.numerics.spaces.moment_head.truncated_head`, which re-mints
the family at ``L_new`` under **this** head's own name, so identity
survives the order change; a head that had been renamed cannot be handed
back the default name, which is the tell of an integer mint.

⛔ **That delegation is item 6.2c-ii's, and it replaced a
``from_L(L_new)`` call — a change of MECHANISM, not of spelling (ruling
O-3: re-mint AND re-axis, never slice).** Once the head carries its metric
as an axis MEASURE, "the same family one order down" stops being a pure
function of :math:`L`: the metric belongs to whichever object DRESSED the
head, and only that object can re-derive it at the new order.
:func:`~orpheus.numerics.spaces.moment_head.truncated_head` therefore
dispatches on the head axis's
:attr:`~orpheus.numerics.axis.Axis.generator` — a
:class:`~orpheus.numerics.basis.base.TruncatedBasis` re-spans the continuum
family and hands back ITS space; a
:class:`~orpheus.numerics.frame.GalerkinFrame` re-poses itself through
:meth:`~orpheus.numerics.frame.GalerkinFrame.at_order` and hands back its
dressed space at that order. ⚠ And a *slice* of the parent's dressing is
not merely inelegant, it is **undefined**: the discrete Gram's diagonality
verdict can FLIP with :math:`L` — `[M]` ``folded_product(2,4)`` is DENSE at
:math:`L = 2` and DIAGONAL at :math:`L = 1`, so the parent's metric is a
matrix :math:`G^{+}` and the child's a diagonal :math:`G^{-1}`, and no
sub-block of the first is the second. A truncated moment field therefore
lands on exactly the space its carrier mints at ``L_new`` (structurally
equal, ruling O-5), never on a second-metric twin. ⚠ The kept block is
the head's leading corner in each of its own axes — a lower order keeps
the low-index modes of every head layout — which is what makes the slice
layout-agnostic rather than hard-coded to the harmonics'
:math:`(L{+}1, 2L{+}1)` rectangle.

The gates, and the input each one rejects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/transport/frames/test_moment_space_is_read_off_the_frame.py``,
``@pytest.mark.foundation``. Three gates, each with a shipped-or-constructible
input it rejects (``plan-authoring`` §6c — a gate that lands with no case
to catch is green by construction):

.. list-table:: The three gates and their witnesses
   :header-rows: 1
   :widths: 22 40 38

   * - Gate
     - What it asserts
     - The input it rejects
   * - **ROUTE**
     - a FOREIGN truncated basis — carrying ``L``, *not* a
       spherical-harmonic subclass, with a renamed coefficient space —
       bound into the quadrature's own frame cache makes every operator
       end and the moment field's head MOVE with it
     - an end minted from :math:`L` alone: it fails the composability
       guard ``A.domain == B.codomain`` under
       :meth:`frame.conjugate <orpheus.numerics.frame.FrameBase.conjugate>`,
       which is the red a reverted producer produces
   * - **METRIC**
     - *(as written 2026-09-02:)* the bound end carries the basis's own
       CONTINUUM Gram, bit-for-bit equal to the ``from_L(L)`` mint, on
       every (rule, :math:`L`) row. ⛔ **Item 6.2c-ii FLIPPED this leg**
       (R-6.2c-1): the bound end is the frame's Parseval-dressed
       ``basis_space``, and the assertion moved with the binding
     - *(2026-09-02:)* the frame's Parseval-dressed ``basis_space``,
       asserted ``(name, shape)``-equal and metric-DIFFERENT. **Since
       6.2c-ii** the negative control is the CONTINUUM head, which is now
       also structurally UNEQUAL — so the leg discriminates twice over,
       on the metric array AND on ``==``
   * - **DOOR**
     - both doors demand the ``TruncatedBasis`` surface, typed, with a
       message naming the truncation order
     - an indicator trial (refused at both doors) — and, in the other
       direction, the foreign truncated basis, which the old
       ``isinstance`` door refused and which is now admitted

⚠ The ROUTE gate's mutant is *unconstructible* before the door widens, so
the door and the seven producers are one step, not two
(``plan-authoring`` §6b).

One existing gate was **demoted** by the step and is retained with a
narrower description:
``tests/transport/frames/test_harmonic_frame.py::test_moment_codomain_content_equals_the_carrier_mint``
compared a face's moment codomain against the carrier's own mint, and
both sides now derive from one source, so no input can make them
disagree (``coding-standards``, the single-sourcing demotion). It keeps
the discovery-path ``is``-identity it still tests, and the shape claim it
used to carry moved to a new external pin against a hand-written literal.

.. _frame-g0-descent-arrow:

G0 — the frame's two halves must name ONE orbit space
-------------------------------------------------------

The subsection above settles where the *coefficient space* comes from.
This one settles the question one level below it, and it is the check
whose absence was :doc:`ERR-080 </theory/verification/error_catalog>`:
**what makes a (basis, measure) pairing admissible at all?**

Landed 2026-09-02, #429 tracker 2.2, inside the fused commit that
repaired ERR-080. The point-set derivation is
:ref:`manifold-g0-descent-arrow` on
:doc:`/theory/foundations/manifolds`; what follows is the frame's side
of it.

The predicate, and why it is ONE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A frame binds functions on ``basis.domain`` to a rule on
``measure.support``. That is well-posed exactly when the functions can
be *evaluated at the rule's nodes* — i.e. when there is a map from the
rule's point set to the basis's:

   **admissible iff a quotient map** ``measure.support -> basis.domain``
   **EXISTS; and the frame's table is the basis pulled back along it.**

:func:`~orpheus.numerics.manifold.quotient_onto` returns that arrow or
``None``. Three cases are honest: the identity (equality is the special
case :math:`K = H`), the entry's own
:attr:`~orpheus.numerics.manifold.Quotient.quotient_map` when the target
is a quotient of the source, and the induced :math:`M/K \to M/H` when
both are quotients of one base with :math:`K \subseteq H`.

⭐ **The registry's stage 0 reads the SAME arrow, since 2026-09-02.**
#429 tracker 2.2b replaced
:meth:`AngularSymmetry.admits_domain
<orpheus.numerics.quadrature.registry.AngularSymmetry.admits_domain>`'s
support EQUALITY with the same
:func:`~orpheus.numerics.manifold.quotient_onto` question, plus a
containment on the group a rule's orbit space was quotiented by — which
#434 R3 (2026-09-03) re-posed against what the geometry's solution keeps
UNSPENT rather than what a reflecting face is owed
(:ref:`manifold-gamma-slot`; ERR-081). So a frame and a
selector now ask the point-set layer one question with two consumers
rather than two questions with one answer each, and `[M]` the shipped
cylindrical fold — refused at both selection stages until that date — is
admitted at both (see
:doc:`/theory/foundations/manifolds`,
:ref:`manifold-gamma-slot`). Nothing on the FRAME side moved: `[M]`
2026-09-02, ``GalerkinFrame(LegendreBasis(L), folded_product(4, 8).measure)``
still constructs at :math:`L \in \{0, 2, 4, 6\}` with a
:math:`(16, L+1)` table, exactly as it did before.

⭐ **This subsumes the containment the pairing was first stated as, and
it admits two pairings the containment cannot express.** The lattice
verdict :math:`G_{\text{spent}} \subseteq G_{\text{have}}`
(:ref:`manifold-invariance-pairing`) is precisely the third case. The
first two are not lattice statements at all — and the second of them is
a binding somebody actually wants: a Legendre basis
:math:`P_\ell(\Omega\cdot\hat e_a)` on a **full-sphere** rule, which is
a perfectly good expansion on a Lebedev or level-symmetric node set and
which a bare containment test refuses. Asking for the ARROW is the same
question asked in the category, and it answers all three.

`[M]` 2026-09-02, every shipped pairing constructed and run:

.. list-table:: G0 on the shipped pairings
   :header-rows: 1
   :widths: 26 26 10 38

   * - rule (``measure.support``)
     - basis (``domain``)
     - G0
     - arrow, or reason
   * - slab GL — :math:`S^2/O(2)_x`
     - Legendre on :math:`S^2/O(2)_x`
     - ✅
     - identity — **what the repair binds**
   * - sphere rule — :math:`S^2`
     - full harmonics on :math:`S^2`
     - ✅
     - identity
   * - sphere rule — :math:`S^2`
     - Legendre on :math:`S^2/O(2)_x`
     - ✅
     - the entry's :math:`\pi` — `[M]` ``lebedev(11)`` gives a
       :math:`(50, 3)` table at :math:`L = 2`, ``level_symmetric(8)`` an
       :math:`(80, 3)` one
   * - :math:`\sigma_y` fold — :math:`S^2/\sigma_y`
     - :math:`\sigma`-even harmonics on the same entry
     - ✅
     - identity
   * - slab GL — :math:`S^2/O(2)_x`
     - full harmonics on :math:`S^2`
     - ⛔
     - **ERR-080.** No map :math:`S^2/O(2)_x \to S^2` exists; the arrow
       runs the other way
   * - :math:`\sigma_y` fold
     - full harmonics on :math:`S^2`
     - ⛔
     - same shape — a fold cannot carry the unfolded family
   * - :math:`\sigma_y` fold
     - Legendre on :math:`S^2/O(2)_x`
     - ✅
     - the induced :math:`S^2/\sigma_y \to S^2/O(2)_x`, since
       :math:`\sigma_y \in O(2)_x` — `[M]` ``folded_product(4, 8)``
       gives a :math:`(16, 3)` table at :math:`L = 2`. ⛔ This row read
       **⛔ ⚠ mathematically admissible; over-refused (GitHub #432)**
       until 2026-09-02
   * - :math:`\sigma_y` fold
     - Legendre on :math:`S^2/O(2)_y`
     - ⛔
     - the NEGATIVE leg of the row above: :math:`\sigma_y \notin O(2)_y`
       — a mirror in the :math:`y`-plane flips :math:`\hat e_y` — so no
       arrow exists

The message names both point sets, both groups, and
:meth:`Quadrature.angular_frame
<orpheus.numerics.quadrature.directional.Quadrature.angular_frame>` as
the surface that derives the right basis — so a caller who trips it is
told what to do rather than what happened.

.. warning::

   ⛔ **This warning read as follows until 2026-09-02, and its
   diagnosis was correct — the DECLARATION was too weak, not the
   pairing:**

      *⚠ The last row is a known over-refusal and it is inert today.*
      :math:`P_\ell(\Omega\cdot\hat e_x)` *is invariant under the full*
      :math:`O(2)_x`, :math:`\sigma_y` *included, but*
      ``Basis.invariance_group`` *is DERIVED as* ``SO2('x')``, *a strict
      lower bound, and no axis-parameterised* :math:`O(2)` *member
      exists to declare instead.* `[M]` *no dispatch selects that
      pairing — the fold binds its* :math:`\sigma`-*even harmonics — so
      nothing shipped reaches it. Tracked at* **#432**.

   ✅ **#432 landed 2026-09-02.** The missing member ships as
   :class:`~orpheus.numerics.symmetry.O2`, the pointwise stabiliser of
   the axis, and an orbit space is now NAMED by its stabiliser — so
   ``invariance_group``, still derived from the domain, is the full
   :math:`O(2)_x` the warning already knew the functions had
   (:ref:`manifold-orbit-space-stabiliser`). `[M]` 2026-09-02:
   ``GalerkinFrame(LegendreBasis(L=L, axis="x"),
   Quadrature.folded_product(4, 8).measure)`` constructs at
   :math:`L = 0, 2, 4, 6` with a :math:`(16, L{+}1)` table, and an
   isotropic field's moments through it read :math:`4\pi =
   12.566370614359172` at :math:`\ell = 0` — bit-identical to
   ``measure.weights.sum()`` — and :math:`\le 1.42\times10^{-15}` at
   :math:`\ell \ge 1`, so the fold aliases nothing into the retained
   degrees (the azimuthal rule is exact to trigonometric degree 7 and
   the fold is :math:`\sigma_y`-even).

   ⚠ **The admission is not blanket, and the negative legs are the
   evidence of that.** `[M]` on the same :math:`\sigma_y` fold,
   ``LegendreBasis(axis="y")`` is still REFUSED — :math:`\sigma_y`
   flips :math:`\hat e_y`, so it is in no :math:`O(2)_y` — while
   ``axis="z"`` is admitted; and on a :math:`\sigma_x`-folded
   ``product(4, 8)`` the verdicts swap, ``axis="x"`` refused and
   ``axis="z"`` admitted with a :math:`(20, 3)` table. The predicate
   is the arrow, and the arrow is the lattice.

Where it fires, and why in three places
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FrameBase.__post_init__`` checks the **trial** half at construction, so
an inadmissible frame is unspellable rather than merely wrong.
:attr:`~orpheus.numerics.frame.FrameBase.test_descent` checks the
**test** half on first use, because the Petrov-Galerkin subclass binds
the test basis and it is not available in the base's ``__post_init__``
(on a Galerkin frame ``test is basis``, and the same cached arrow is
returned). And :class:`~orpheus.numerics.frame.GalerkinFrame`'s
hand-written ``__init__`` calls the helper explicitly — it bypasses the
dataclass ``__init__`` that would otherwise run ``__post_init__``, so a
gate installed only on the dataclass path would have been inert on the
one constructor every angular frame in the tree goes through.

⭐ **The arrow is not merely a gate; it is what the table is built
with.** :attr:`FrameBase.table <orpheus.numerics.frame.FrameBase.table>`
evaluates ``self.basis.evaluate(self.descent(self.measure.nodes))``, and
:attr:`~orpheus.numerics.frame.FrameBase.test_table` the same through
``test_descent``. So the check and the tabulation read ONE object: a
frame that passed G0 cannot then tabulate through a different map, which
is the failure mode a separate validator would have left open. On every
identity arm the map is ``np.asarray(points)`` — a bit-preserving
no-op — which is why the repair is **exactly inert** on the full-sphere
and folded rules, as a theorem rather than as a sample: same basis
class, same measure object, identity map, therefore the same float
program.

The moment head — a carrier reads its layout, never assumes one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

G0 admits two basis families onto angular frames, and they have
different coefficient LAYOUTS. The real harmonics' space is the
rectangular :math:`(L+1, 2L+1)` table with the addition-theorem-shifted
:math:`[\ell + m]` column and zero padding outside :math:`|m| \le \ell`;
:class:`~orpheus.numerics.spaces.legendre_space.LegendreSpace` is
**FLAT**, :math:`(L+1,)`, one coefficient per degree. A moment field's
space is ``<head> ⊗ cells`` either way, so the *rank* of the leading
factor is now a variable.

:class:`~orpheus.numerics.spaces.moment_head.MomentHead` is the
``runtime_checkable`` ``Protocol`` that makes the layout the head's to
say: ``L``, ``shape``, ``isotropic_slot``, ``degree_block(l)`` and
``truncated(L_new)``. Both space classes satisfy it structurally, and a
consumer holding ``space.factors[0]`` narrows with ``isinstance`` — the
same key-on-what-it-declares idiom as
:class:`~orpheus.numerics.basis.base.TruncatedBasis` on the basis side.

⛔ **Why it is a repair and not decoration: on a flat head the old reads
returned the wrong array and raised NOTHING.** Every carrier read that
indexed ``values[0, 0]`` or sliced ``values[l, :2l+1]`` was spelling the
first family's layout as if it were the contract. On a
:math:`(L+1, n_g, n_x)` tensor, ``values[0, 0]`` is *group 0's spatial
slice* — a well-shaped array of the wrong thing. `[M]` the sites:
``scalar_flux``, ``isotropic_part``, ``anisotropic_part``, ``l_block``,
the fission :math:`\ell = 0` dyad, ``ng`` (which located the group axis
at a hard-coded index 2), ``zeros_for_mesh_and_L``, and — the one no
census had listed — the material field's per-degree group contraction,
which spelled the :math:`m` axis into its ``einsum`` spec
(``"mfc...,fg->mgc..."``) and its slicing. That last one would have
contracted the GROUP axis as if it were :math:`m`. All of them now read
the head:

.. list-table:: What the head says, per family
   :header-rows: 1
   :widths: 30 35 35

   * - Question
     - real harmonics
     - Legendre (a 1-D rule)
   * - ``shape``
     - :math:`(L+1,\ 2L+1)`
     - :math:`(L+1,)`
   * - ``isotropic_slot``
     - ``(0, 0)``
     - ``(0,)``
   * - ``degree_block(l)``
     - ``(l, 0:2l+1)``
     - ``(l,)``
   * - rank (``len(shape)``) ⟹ the group axis
     - 2 ⟹ ``values.shape[2]``
     - 1 ⟹ ``values.shape[1]``
   * - :math:`\Lambda`'s block contraction
     - ``"mfc...,fg->mgc..."``
     - ``"fc...,fg->gc..."``

`[M]` 2026-09-02, built through the production carrier: a
``gauss_legendre(8)`` phase space gives
``HarmonicMomentFlux.zeros_for_mesh_and_L(sn, 2).values.shape ==
(3, 1, 4)`` with head ``legendre_space(S^2/O2_x)``, while
``level_symmetric(8)`` and ``folded_product(4,8)`` give ``(3, 5, 1, 4)``
with head ``spherical_harmonic_space``. The :math:`\Lambda` specs are
the former inline ones **verbatim** on the rank-2 rows, so the harmonic
path is bit-identical by construction; the rank-1 rows are new, and a
head of any other rank is refused by name rather than contracted
wrongly.


.. _frame-composed-verbs:

The frame's composed-operator verbs
===================================

A consumer does **not** hand-roll the analysis / reconstruction faces.
The point of binding a basis to a measure through a
:class:`~orpheus.numerics.frame.FrameBase` is that the frame then emits
the **composed operators** the method actually applies — *define a
frame, compose, done* (Cardinal Rule 2: the composition **is** the
production path, not a parallel "semantic" reading layered over a
hand-rolled numpy chain). Three composed verbs cover every consumer:

.. list-table:: The frame's composed-operator verbs
   :header-rows: 1
   :widths: 30 20 50

   * - Verb
     - Composition
     - Consumer
   * - :meth:`conjugate(A) <orpheus.numerics.frame.FrameBase.conjugate>`
     - :math:`R \circ A \circ M`
     - SN anisotropic scattering
       :math:`S_{\ell\ge 1} = R\,\Lambda\,M` (project to moments,
       multiply by the spectrum :math:`\Sigma_{s,\ell}`, reconstruct
       the per-ordinate source)
   * - :meth:`reconstruct_after(A) <orpheus.numerics.frame.FrameBase.reconstruct_after>`
     - :math:`R \circ A`
     - inputs **already** in coefficient space — the angular-windowed
       SN moment iterate, whose bulk is already :math:`M\psi`, so only
       :math:`R\,\Lambda` remains (wiring it to ``conjugate`` would
       double-project)
   * - :meth:`project(f) <orpheus.numerics.frame.FrameBase.project>`
     - :math:`G^{-1} M`
     - the **homogenise / condense** coefficient extraction —
       :meth:`Solution.homogenize
       <orpheus.sn.solution.Solution.homogenize>`,
       :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`

Each returns a **typed**
:class:`~orpheus.numerics.operator.OperatorProduct` (or, for
``project``, the inverse-Gram ∘ analysis chain), whose ``apply`` runs
exactly the numpy contraction a hand-rolled
``reconstruction.apply(A.apply(analysis.apply(x)))`` would — now as
**one named operator** with the
:class:`~orpheus.numerics.operator.OperatorProduct` space-compatibility
guard enforcing that :math:`A` composes between the faces (its
``domain`` is the analysis codomain, its ``codomain`` the
reconstruction domain).

:meth:`conjugate <orpheus.numerics.frame.FrameBase.conjugate>` is the
**2-cell** of the (Representation × Role) carrier double category
(:ref:`operator-algebra`): a coefficient-space Role-morphism :math:`A`
conjugated by the horizontal Representation-adjoint pair
:math:`(M, R)`. When the frame is the operator's *eigenbasis* — the SH
angular frame is the scattering kernel's, by Funk–Hecke —
:math:`R\circ\Lambda\circ M` **is** the spectral theorem
:math:`U\Sigma U^*` written out, and the frame is then *owned* by that
operator (:ref:`frame-eigenbasis-ownership`). The
coefficient-extraction verb :meth:`project
<orpheus.numerics.frame.FrameBase.project>` is the Petrov-Galerkin
:math:`G^{-1}M` derived term-by-term for the homogenisation consumer in
:ref:`sn-homogenization-petrov-galerkin-frame`; its diagonal-Gram normalisation
is the typed ARROW :attr:`gram_inverse
<orpheus.numerics.frame.FrameBase.gram_inverse>`, whose action is the
row-sum probe of :ref:`frame-least-squares-discipline` and whose ends
are the two faces' coefficient ends (:ref:`frame-gram-inverse-arrow`).


.. _frame-gram-inverse-arrow:

The projection normalisation is an ARROW, not a metric-twin space
==================================================================

:meth:`project <orpheus.numerics.frame.FrameBase.project>` is
:math:`G^{-1}M`: analyse against the test functions, then divide by the
cross Gram :math:`G_{kj} = \langle\chi_k, \phi_j\rangle_W = (MR)_{kj}`.
The frame computes :math:`G` with a single ``analysis ∘ reconstruction``
probe of the all-ones coefficient vector — the **row sum** of :math:`MR`
— and inverts it as a masked reciprocal
(:ref:`frame-least-squares-discipline` states the two structural
conditions under which that probe IS the normalisation).

The question this section settles is *what kind of object* that
reciprocal is. Until CS4c step 6 item 6.2c-ii (2026-09-08) it was a
**space**: ``FrameBase.gram`` returned the TEST space re-dressed with the
probe diagonal as its ``inner_product_weights``, and the
:math:`G`-orthogonal projector was spelled

.. code-block:: python

   frame.conjugate(InverseMetricOperator(frame.gram))     # retired

i.e. as an endomorphism of that re-dressed twin. Since 6.2c-ii it is a
typed **arrow**,
:class:`~orpheus.numerics.frame.CrossGramInverse`, exposed as
:attr:`FrameBase.gram_inverse
<orpheus.numerics.frame.FrameBase.gram_inverse>`, and the projector is

.. code-block:: python

   frame.conjugate(frame.gram_inverse)                    # = R G⁻¹ M

⭐ **Two independent reasons, and the second is the one that generalises.**

**(1) The twin stopped type-checking.** A metric-twin is *the same space
with a different metric*, which is only expressible while identity is
metric-blind. Item 6.2c-ii made the harmonic heads axis-built, so the
measure entered the identity (:ref:`spaces-identity-bridge`) and
``replace(test_space, inner_product_weights=probe)`` became a **different
space** from ``test_space``. The composition
``conjugate(InverseMetricOperator(twin))`` then failed
:class:`~orpheus.numerics.operator.OperatorProduct`'s own
space-compatibility guard on every harmonic frame — the guard doing
exactly its job, on a spelling that had been relying on the seam.

**(2) The twin was a costume, and it carried a real hazard.**
:math:`G^{-1}` maps the analysis face's covariant output (in
:attr:`~orpheus.numerics.frame.FrameBase.test_space`) to the trial side's
contravariant coefficients (in
:attr:`~orpheus.numerics.frame.FrameBase.basis_space`). Those ARE the two
faces' coefficient ends — it was an arrow all along, wearing an
endomorphism's clothes. Spelled as one it composes by construction, and a
whole class of defect becomes **unspellable**: the arrow's action reads
the probe diagonal and NO space's metric, so a dressed test space can
never leak its Parseval metric into the normalisation. That leak was not
hypothetical — `[M]` 2026-08-30, the pre-P7 spelling handed the probe a
space whose ``apply_inverse_metric`` applied its own matrix instead, a
**162 %** projection error on the overlap frame (:math:`\Sigma_R`
``[7.0, 11.0]`` for a true :math:`[8/3, 16/3]`). P7 fixed it by teaching
``gram`` to *strip* the dressing — a guard. The arrow removes the state
the guard was guarding.

.. list-table:: :class:`~orpheus.numerics.frame.CrossGramInverse` at a glance
   :header-rows: 1
   :widths: 24 76

   * - Slot
     - What it is
   * - ``domain``
     - :attr:`~orpheus.numerics.frame.FrameBase.test_space` — the analysis
       codomain, where a covariant moment lands
   * - ``codomain``
     - :attr:`~orpheus.numerics.frame.FrameBase.basis_space` — the
       reconstruction domain, the trial coefficient space. One space on a
       :class:`~orpheus.numerics.frame.GalerkinFrame`, two on a
       :class:`~orpheus.numerics.frame.PetrovGalerkinFrame`
   * - ``diagonal``
     - the probe :math:`(MR\,\mathbf 1)` — the cross Gram's row sums, in
       the analysis codomain's layout
   * - action
     - the Moore–Penrose reciprocal of that diagonal, one spelling: the
       :class:`~orpheus.numerics.metric.DiagonalMetric`'s own inverse
       arithmetic, zero on dead slots, broadcast against the coefficient
       array's leading block
   * - ``is_adjointable``
     - ``True`` — a real diagonal is self-adjoint, and its two ends share
       one shape

The consumers moved with it: :meth:`project
<orpheus.numerics.frame.FrameBase.project>` is now literally
``gram_inverse.apply(analysis.apply(field))``, and the SN loss-kernel
gauge spells its :math:`G`-orthogonal projector
``frame.conjugate(frame.gram_inverse)``. The ``DENSE``-trial refusal is
unchanged and still lives on this property
(:ref:`frame-least-squares-discipline`).

⭐ **The pattern worth carrying: when a "metric-twin space" appears in a
composition, ask whether the two ends are actually different spaces.**
Under structural identity a twin is a different object, so the twin
spelling was always either a type error or a claim that the metric is not
part of identity. Both halves of that dilemma are answered the same way —
give the map its own type and let its ends be the ends it actually has.

.. _frame-at-order:

A frame can re-pose itself at another order
============================================

:meth:`GalerkinFrame.at_order <orpheus.numerics.frame.GalerkinFrame.at_order>`
returns THIS frame over the same measure with its trial family cut at
``L_new``: same class, same measure, the trial basis's own
:meth:`~orpheus.numerics.basis.base.TruncatedBasis.at_order`. It exists
because the frame is the head's :attr:`generator
<orpheus.numerics.axis.Axis.generator>` (:ref:`frame-the-one-moment-space`)
and a head must be re-dressable at every order — the metric is the frame's
to install, and the discrete Gram's verdict can change with :math:`L`
(:ref:`frame-moment-space-single-home`, the truncation paragraph). It
refuses a trial that carries no truncation order, by name; and the frame it
returns is structurally equal to the quadrature's own interned frame at
that order, so truncation does not fork the frame cache.


.. _frame-least-squares-discipline:

The least-squares discipline — designed, not built
==================================================

The discipline split is carried one level deeper than the
Galerkin / Petrov-Galerkin *type* by the **trial basis's Gram
structure** — the declaration
:class:`~orpheus.numerics.basis.GramStructure` that decides whether the
coefficient extraction :meth:`project
<orpheus.numerics.frame.FrameBase.project>` can use a cheap row-sum
probe or needs a full dense solve. ``project`` normalises by the
cross-Gram :math:`G = MR`; the frame computes it with a single
``analysis(reconstruction(ones))`` **row-sum probe**, but that probe
equals the required normalisation only under one of two structural
conditions:

.. list-table:: The trial-basis Gram structure decides the projection machinery
   :header-rows: 1
   :widths: 22 30 26 22

   * - ``GramStructure``
     - Trial Gram :math:`MR`
     - Projection normalisation
     - Built?
   * - ``DIAGONAL``
     - diagonal (orthogonal harmonics; disjoint / nested cell / group
       indicators)
     - the row sum **is** the diagonal — a reciprocal
     - **yes** (Galerkin SH; forward homogenisation)
   * - ``PARTITION_OF_UNITY``
     - not diagonal, but membership rows sum to 1
       (:class:`~orpheus.numerics.basis.OverlapBasis`)
     - :math:`R\mathbf 1 = \mathbf 1` collapses the probe to the
       per-region weight — still a reciprocal
     - **yes** (forward condensation)
   * - ``DENSE``
     - neither (a tapered weight, a higher-rank GEC moment)
     - needs the real :math:`(MR)^{-1}M` least-squares solve
     - **no** — :meth:`project` *refuses* (#275)

The first two rows are the **built** frames: a
:class:`~orpheus.numerics.frame.GalerkinFrame` (diagonal Gram,
``test is trial``) and the forward
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` (diagonal *or*
partition-of-unity). For both, :attr:`FrameBase.gram_inverse
<orpheus.numerics.frame.FrameBase.gram_inverse>` is a single row-sum probe and
:meth:`project <orpheus.numerics.frame.FrameBase.project>` is a per-cell
reciprocal (a Moore–Penrose pseudo-inverse, so an empty / zero-flux
region maps to :math:`\Sigma_R = 0` for free), **not** a linear solve.

The third row is the **third discipline** — a least-squares frame over a
**dense** cross-Gram. It is the natural sibling of
:class:`~orpheus.numerics.frame.GalerkinFrame` under the Petrov-Galerkin
base (the designed hierarchy is ``FrameBase → PetrovGalerkinFrame →
{GalerkinFrame, LeastSquaresFrame}``): its trigger is a trial basis
whose :math:`MR` is genuinely dense — ``test`` :math:`= A\cdot`\ ``trial``
for some non-identity :math:`A`, a dense SPD Gram needing a real solve —
for which the row-sum probe is **wrong**. It is **designed but not
built**: the base :class:`~orpheus.numerics.basis.Basis` defaults to
``GramStructure.DENSE`` (the safe refusal), and
:meth:`FrameBase.project <orpheus.numerics.frame.FrameBase.project>`
raises :class:`~orpheus.numerics.operator.NotInvertible` on a
``DENSE`` trial rather than return a silently-wrong coarsening. The
known future consumer is **higher-rank Generalized Energy Condensation**
(within-coarse-group spectral moments :math:`n \ge 1`; Rahnema,
Douglass & Forget 2008) — a richer coarse basis than the rank-0 P0
indicator — deferred to `GitHub #275
<https://github.com/deOliveira-R/ORPHEUS/issues/275>`_. No
``LeastSquaresFrame`` type exists today; the name marks the seam, not a
shipped class.

.. note::

   Cross sections never need this dense seam at rank 0: a P0 (indicator)
   coarse cross section is the only rate-meaningful one, and its
   partition-of-unity Gram is row-sum-collapsible. The dense
   :math:`(MR)^{-1}M` solve becomes load-bearing only for a
   non-indicator coarse basis — exactly the GEC :math:`n \ge 1`
   moments — so the refusal is a forward-looking guard, not a
   present-day gap.


.. _frame-eigenbasis-ownership:

Why the discipline splits — an operator owns its frame iff the frame is its eigenbasis
======================================================================================

The page so far has *asserted* the discipline of each consumer
axis-by-axis: the angular spherical-harmonic frame is Galerkin
(:eq:`galerkin-pair`), energy condensation and spatial homogenisation
are Petrov-Galerkin (the consumer table). This section supplies the
**root cause** that decides which axis lands in which discipline, and
with it the deeper architectural fact behind the whole Frame campaign:

  **An operator owns its frame if and only if the frame is its
  eigenbasis — i.e. the basis that diagonalises the operator by a
  symmetry of the phase space.**

For the angular axis the symmetry is the rotation group :math:`SO(3)`,
the eigenbasis is the spherical harmonics, the diagonalisation is a
*theorem* (Funk–Hecke + Schur), and the owner is the **scattering
operator** — hence a :class:`~orpheus.numerics.frame.GalerkinFrame`
(orthogonal eigenbasis, ``test is trial``). For the energy and spatial
axes there is **no such symmetry**, no eigenbasis, and therefore no
operator that owns the frame: the projection is solution-*weighted*
and lives on the test side — hence a
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`. The two
disciplines are *one* structural cause read at two axes, not two
independent conventions.

The **structural leg** — that Funk–Hecke makes the spherical harmonics
scattering's eigenbasis, so the frame is Galerkin — is worked in
:ref:`frame-spherical-harmonic-galerkin` above; the subsections below
read its consequence for *ownership*: the asymmetry that assigns the
frame to scattering, the literature corroboration, and the unifying
principle.

The asymmetry that fixes ownership — streaming does not diagonalise
-------------------------------------------------------------------

If both transport operators were diagonalised by the spherical
harmonics, "scattering owns the frame" would be a coin toss. They are
not, and the asymmetry is what assigns ownership.

The **streaming** operator :math:`\hat\Omega\cdot\nabla` carries the
direction :math:`\hat\Omega`, which is itself the :math:`\ell = 1`
vector irrep of :math:`SO(3)`. It does **not** commute with rotations
(rotating the frame rotates the gradient direction), so it is **not**
in the commutant and **not** diagonalised by the harmonics. By the
Clebsch–Gordan rule
:math:`V_1 \otimes V_\ell = V_{\ell-1}\oplus V_\ell\oplus V_{\ell+1}`,
multiplication by a component of :math:`\hat\Omega` couples
:math:`V_\ell` to :math:`V_{\ell\pm 1}`:

.. math::
   :label: streaming-pn-recurrence

   \mu\,Y_\ell^m
   \;=\;
   \frac{\ell+1}{2\ell+1}\,Y_{\ell+1}^{m}
   \;+\;
   \frac{\ell}{2\ell+1}\,Y_{\ell-1}^{m}

— the **Pℓ moment recurrence**, the block-**tridiagonal** coupling that
makes the PN coefficient matrix tridiagonal in :math:`\ell` rather
than diagonal (Fletcher 1983, Eq. 5; Hébert 2009, §3.6–3.7). Streaming
in the harmonic basis is *tolerated*, not *diagonalised*: the basis is
chosen to make collision diagonal, and streaming is then expressed
(awkwardly, tridiagonally) in those same coordinates.

.. vv-status: streaming-pn-recurrence documented
   The μ·Y_ℓ recurrence is the standard Legendre/spherical-harmonic
   recurrence (Fletcher 1983 NSE 84:33 Eq. 5; Hébert 2009 §3.6); a
   transcribed structural identity, not a solver claim. ORPHEUS does
   not yet ship a PN solver — the recurrence is documented here as the
   asymmetry that fixes frame ownership, not as a verified code path.

The ownership conclusion is then forced:

.. list-table:: Which operator the spherical-harmonic basis diagonalises
   :header-rows: 1
   :widths: 26 26 28 20

   * - Operator
     - Symmetry of the kernel
     - Action in the SH basis
     - Diagonalised?
   * - Scattering :math:`\Sigma_s(\hat\Omega\cdot\hat\Omega')`
     - :math:`SO(3)`-invariant (zonal)
     - **diagonal** per :math:`\ell`-block (Funk–Hecke + Schur)
     - **yes** — its eigenbasis
   * - Streaming :math:`\hat\Omega\cdot\nabla`
     - :math:`\ell=1` tensor; **not** invariant
     - block-**tridiagonal** :math:`\ell\!\leftrightarrow\!\ell\pm 1`
       (Clebsch–Gordan)
     - no — merely tolerated

Because the spherical harmonics are the eigenbasis of *scattering* and
nothing else in the transport operator, the frame is **scattering's**
frame.  ⚠ Read "owns" in that sentence mathematically — *which basis
diagonalises which operator* — because the **constructional** reading
was true until 2026-08-30 and is not any more.  It read, verbatim:
*"*
:class:`~orpheus.transport.operators.scattering.ScatteringOperator`
*holds the frame as a cached property and binds its order to the
scattering order,* ``quadrature.angular_frame(self.scattering_order)``
*(the canonical constructor + the L-binding) … the constructor
ownership sits on scattering."*  At CS4c step 3 the frame became a
shared object **constructed outside and handed in**, interned on the
hub (:ref:`frame-eigenbasis-relocation-tripwire` records which
predicted trigger fired and which did not).  What did not change: the
frame *object* lives in the method-agnostic
:class:`~orpheus.numerics.frame.GalerkinFrame` hierarchy, and the
reason the harmonic frame is the one scattering uses is still the
Funk–Hecke diagonalisation above — a mathematical fact no refactor can
relocate.

Literature corroboration — no falsifier across six transport families
---------------------------------------------------------------------

The structural argument is corroborated by every documented transport
method: in SN, MoC, CP, PN, first-collision-source/ray-effect, and
random-ray, the flux→spherical-harmonic-moment projection
:math:`M = \int Y_\ell^m\,\psi\,d\hat\Omega` is invoked **solely** to
evaluate the anisotropic scattering source. A falsifier would be a
documented, structurally-independent *non-scattering* use of the flux
moment projection; none exists.

.. list-table:: Literature: the flux→SH-moment projection is anisotropic-scattering-rooted
   :header-rows: 1
   :widths: 22 30 48

   * - Reference
     - Equation / section
     - What it establishes
   * - Hébert 2009, *Applied Reactor Physics*
     - §3.3, Eq. (3.55) [the projection :math:`M`]; Eq. (3.54)
       [its sole use]; Eq. (3.42); Eq. (3.57)
     - :math:`M` (Eq. 3.55) is used **only** in the scattering source
       (Eq. 3.54). The integral / characteristic form natively wants
       **isotropic** sources (Eq. 3.42); fission is isotropic
       (Eq. 3.57). Only anisotropic scattering forces the moments.
   * - Brockmann 1981, NSE **77** (4), 377
     - Eq. (47) [the Legendre flux moment]
     - Introduces :math:`\Phi_\ell(x,E) = 2\pi\int P_\ell(\mu)\,\Phi\,d\mu`
       expressly "considering the problem of anisotropic neutron
       scattering"; the *same* projection is reused across SN, FEM,
       and orders-of-scattering.
   * - Fletcher 1983, NSE **84**, 33
     - Eq. (7) [moment equation]; Eq. (5) [streaming recurrence]
     - The moment equation is **diagonal** in :math:`\ell` "because of
       the orthogonality of spherical harmonics" (scattering); the
       streaming term produces the :math:`\ell\!\leftrightarrow\!\ell\pm1`
       **tridiagonal** coupling. PN's moments exist because the basis
       diagonalises scattering.
   * - Ahrens 2014, *Lagrange Discrete Ordinates*, arXiv:1405.3968
     - Eq. (7); abstract
     - The **negative-space proof**: LDO *removes* :math:`M` —
       "no spherical harmonic moments are needed" — precisely by
       reformulating the scattering source. An authority stating the
       only reason standard SN computes :math:`M` is the scattering
       source.
   * - External / boundary sources (Hébert 2009, Eq. 3.30/3.168)
     - —
     - An anisotropic external source is **specified** in spherical
       harmonics as input data :math:`Q_\ell^m`; it is never
       *projected* from the flux. Not a flux→moment projection.
   * - Anisotropic BCs (albedo / white / specular)
     - Hébert 2009, Eqs. 3.43–3.47
     - Handled in **ordinate (direction) space**, not via moments.
       Only PN expresses BCs in moments — because moments are PN's
       native unknowns, which exist to diagonalise scattering.

The single recurring exception, PN, is the method that makes the
moments the unknowns *in order to* diagonalise scattering — so even
there the root cause is scattering. The convergence of five
independent references on the same conclusion, plus Ahrens' explicit
removal of :math:`M`, leaves the claim "the spherical-harmonic angular
projection is intrinsically a scattering concern" with **zero
cross-validation against any non-scattering flux-moment use**.

The unifying principle — symmetry decides Galerkin vs Petrov-Galerkin
---------------------------------------------------------------------

The eigenbasis criterion is the *single structural cause* of the
discipline split this page documents. Reading it across all three
reduction axes:

.. list-table:: Symmetry decides the discipline
   :header-rows: 1
   :widths: 22 24 28 26

   * - Reduction axis
     - Phase-space symmetry
     - Eigenbasis?
     - Discipline (frame type)
   * - **Angular scattering**
     - :math:`SO(3)` rotational (zonal kernel)
     - **yes** — spherical harmonics, *orthogonal* (Funk–Hecke + Schur)
     - **Galerkin**; scattering's eigenbasis (ownership here is
       mathematical — the frame object is constructed and interned
       outside the operator since CS4c step 3)
   * - **Energy condensation**
     - none (general :math:`G\times G` group-transfer matrix)
     - no — no Funk–Hecke, no clean spectrum
     - **Petrov-Galerkin**, solution-weighted, owned by no operator
   * - **Spatial homogenisation**
     - none (arbitrary heterogeneous geometry)
     - no
     - **Petrov-Galerkin**, solution-weighted, owned by no operator

When a symmetry of the phase space diagonalises the operator, the
eigenbasis is *forced* — and because the eigenbasis of a self-adjoint
operator is orthogonal, the projection is **Galerkin** (``test is
trial``, :math:`M^* = R`) and the operator **owns** the frame (the
frame's order is the operator's order). When there is no symmetry —
the energy group-transfer matrix and the spatial homogenisation kernel
have no rotational or other structure to exploit — there is no
eigenbasis, no operator that naturally owns the basis, and the
coarse-graining must instead **weight** the projection by the solution
(the within-group spectrum :math:`\phi_g`, the region flux
:math:`\phi`, or the adjoint :math:`\varphi^*`). That weighting is a
*test* basis distinct from the trial basis, so the projection is
**Petrov-Galerkin** (:math:`M^* \ne R`) and lives on the test side,
owned by the *projection verb*, never by an operator. This is also why
**fission does not own an angular frame**: fission's concern is the
energy axis (the :math:`\chi\nu\Sigma_f` group-transfer), which has no
eigenbasis — its angular emission is isotropic
(:math:`\ell = 0`-only), so there is no harmonic structure for it to
own.

The architectural payoff is that the Galerkin-vs-Petrov-Galerkin
type split (:class:`~orpheus.numerics.frame.GalerkinFrame` vs
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame`) is *derived*
from a single physical question — *does a symmetry diagonalise the
operator?* — rather than asserted axis-by-axis. The
:ref:`homogenisation note above <petrov-galerkin-not-weighted-metric>`
(why folding the solution into the metric breaks for the
eigenvalue-consistent case) is the *converse* of the same principle:
absent an eigenbasis, the solution-weighting cannot be hidden in a
fixed :math:`L^2` metric — it is irreducibly a distinct test space.

.. _frame-eigenbasis-relocation-tripwire:

The relocation tripwire — when scattering stops owning the constructor
----------------------------------------------------------------------

.. admonition:: ✅ The tripwire FIRED on 2026-08-30 — and not on any
                trigger this section named
   :class: important

   This subsection was written as a **prediction**, and it is preserved
   below in its original tense because the prediction's structure was
   right and its enumeration was not — which is the more useful thing
   to record.

   *What it got right.*  Constructor ownership did relocate, it did
   relocate onto the discipline-neutral
   :meth:`Quadrature.angular_frame(L)
   <orpheus.numerics.quadrature.Quadrature.angular_frame>` factory, the
   factory did already exist and did already anticipate it, and the
   eigenbasis ruling did survive untouched.

   *What it got wrong.*  Both enumerated triggers were about an
   :math:`L` **foreign to** ``scattering_order`` — a detector of
   higher anisotropic order, a P\ :sub:`N` flux expansion — and neither
   has landed.  The trigger that actually fired is a **third** one the
   section did not consider: *sharing at the SAME* :math:`L`.  Fission
   and the angular-windowing method both want the frame that
   :math:`S` was minting, at :math:`S`'s own order, over :math:`S`'s
   own measure — and a frame minted *inside* :math:`S` is unreachable
   to them except by passing :math:`S` around.  So the relocation was
   forced by **who may reach the object**, not by whose :math:`L`
   parametrises it.

   *Where it went.*  Not merely to the factory: to a hub with two
   interning tiers —
   :meth:`HarmonicFrame.for_space
   <orpheus.transport.frames.harmonic_frame.HarmonicFrame.for_space>`
   over ``Quadrature.angular_frame(L)`` — so that "one frame per (axis
   content, :math:`L`)" is an object identity and every consumer
   receives the same cached projection table.  The operator retains the
   *products* (its two minted faces) and forgets the factory, keeping
   only a provenance accessor.  Details:
   :ref:`scattering-binding-cs4c`.

   ⭐ The transferable half: a relocation tripwire predicted from
   *parametrisation* (whose :math:`L`?) missed a trigger that came from
   *reachability* (who may construct it?).  When writing the next such
   tripwire, enumerate both axes.

"Scattering owns the frame" was true **until 2026-08-30** because
scattering is
the *only* consumer of the spherical-harmonic frame whose truncation
order :math:`L` is physically meaningful. The constructor ownership
:meth:`ScatteringOperator.frame
<orpheus.transport.operators.scattering.ScatteringOperator.frame>` bound the frame
order to ``self.scattering_order``. This binding **relocates** to the
discipline-neutral factory
:meth:`Quadrature.angular_frame(L)
<orpheus.numerics.quadrature.Quadrature.angular_frame>` the moment a
**second** consumer arrives with an :math:`L` *independent of*
``scattering_order``:

* an **output detector / response functional** of anisotropic order
  :math:`L_d > L_{\rm scatter}` — a flux moment projection whose
  truncation is set by the *detector*, not the scattering kernel
  (structurally independent); or
* a **PN / SPN flux** expansion of order
  :math:`L_{\rm flux} > L_{\rm scatter}` — needing
  :math:`\max(L_{\rm flux}, L_{\rm scatter})`, not ``scattering_order``.

No such consumer exists **even now**: the only output functional
ORPHEUS
computes is the :math:`\ell = 0` scalar flux (via the
:class:`~orpheus.sn.solver.SNSolver`'s angular integration), which does
**not** use the frame. The factory
:meth:`~orpheus.numerics.quadrature.Quadrature.angular_frame` already
exists and already anticipates this relocation — its naming tripwire
names the permanent *angular axis*, not the contingent
spherical-harmonic basis, so a second consumer is a signature change
(``angular_frame(basis=...)``), not a rename. ⚠ The sentence that
stood here — *"until that second consumer lands, the canonical
constructor home is the scattering operator"* — is what the banner
above corrects: the constructor home moved without either enumerated
consumer arriving, because the pressure came from *reachability*
rather than from a foreign :math:`L`.  The clause it justified itself
with is untouched: scattering is still the operator whose eigenbasis
the frame *is*.

A second, structurally distinct trigger is **cross-method use of**
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` (`#261`). The
operator is method-agnostic in principle — every transport method
scatters — but a
:class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame` needs an
angular **measure** (a :class:`~orpheus.numerics.quadrature.Quadrature`)
to exist at all, and CP / MoC carry none (CP integrates angle away into
collision probabilities; MoC uses a track quadrature, not an
:math:`S^2` ordinate set). So the moment scattering is consumed by a
measure-free method, the frame cannot live *as a field on* the shared
operator. Two resolutions were open (deferred to `#261`; user,
2026-06-25): **(a) relocate** the frame to where the angular measure
lives (the
:meth:`~orpheus.numerics.quadrature.Quadrature.angular_frame` factory —
the original W-E idea), or **(b) specialize**
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` per method — an SN
subclass that holds the frame (it carries the :math:`S^2` measure) over
a measure-free cross-method base. Where the independent-:math:`L`
triggers above make the *order* foreign, this one makes the *measure*
absent. The eigenbasis ruling still holds — scattering owns the frame
*wherever an angular measure exists to build it* — it merely stops being
expressible as a field on a single method-agnostic operator.

✅ **Resolution (a) landed on 2026-08-30, ahead of its own trigger.**
CS4c step 3 relocated the frame to the measure's side and removed it
as a field on the operator: the frame is reached through
:meth:`HarmonicFrame.for_space
<orpheus.transport.frames.harmonic_frame.HarmonicFrame.for_space>` over
``Quadrature.angular_frame(L)``, minted at the tier-2 classmethod,
and the operator retains only the two **faces** it minted from it plus
a provenance accessor.  Note the ordering, because it is the reusable
lesson: the resolution was chosen for a reason (sharing between
:math:`S`, :math:`F` and the windowing method) that has nothing to do
with the measure-free-method pressure this paragraph describes.

⚠ **It does not, by itself, make the operator constructible without a
measure** — the two minted faces are mandatory fields, so an
:math:`S^2`-less method still cannot build *this* binding.  What the
step did buy for `#261` is the separation the cross-method question
actually needs: the **representation-free datum** (the per-material
:class:`~orpheus.transport.kernels.TransferKernel` map paired with a
material layout) is now a first-class object of its own, held by the
binding rather than derived inside it.  A CP or MoC binding of the
same physics shares that datum and mints whatever
representation-bound arrows *its* angular treatment admits — which is
what option (b) was reaching for, without the subclass.  The open half
of `#261` is therefore what those arrows are, not where the frame
lives.


Cross-method consumer table
===========================

The frame pair is **infrastructure**, not SN-only. Every method that
lifts an angular / energy / spatial axis between fine and coarse
representations builds on these primitives, always through a frame of
the appropriate discipline type.

.. list-table:: Where the frame pair is consumed
   :header-rows: 1
   :widths: 22 22 22 16 18

   * - Consumer
     - Frame type
     - Pair
     - Status
     - Reference
   * - **SN aniso scattering**
     - GalerkinFrame
     - ``frame.analysis`` /
       ``frame.reconstruction``
       (``Quadrature.angular_frame(L)``)
     - **Live** (Wave 1)
     - §9 (Grand Report v3 line 1230)
   * - PN solver
     - GalerkinFrame
     - Same SH ``frame.analysis``
       on the moment-space side
     - Pending (PN solver not implemented)
     - §10 (lines 1299–1305)
   * - Spatial homogenisation (forward / reaction-rate)
     - PetrovGalerkinFrame
     - Flux-weighted test basis, indicator trial basis
       (:meth:`Solution.homogenize
       <orpheus.sn.solution.Solution.homogenize>`)
     - **Live** (P3)
     - :ref:`sn-homogenization-petrov-galerkin-frame`; Hébert 2009 §13
   * - Energy condensation (forward / reaction-rate)
     - PetrovGalerkinFrame
     - Spectrum-weighted test basis, fractional-overlap trial basis
       (:meth:`Solution.condense
       <orpheus.sn.solution.Solution.condense>`)
     - **Live** (P5)
     - :ref:`sn-energy-condensation`; Hébert 2009 §6.2
   * - Homogenisation / condensation (eigenvalue-consistent)
     - PetrovGalerkinFrame
     - **Bilinear pair** test weight
       :math:`\varphi^*\!\odot\varphi`, indicator /
       overlap trial basis
     - **Live** (P6, #281)
       (:ref:`frame-adjoint-weighted-seam`)
     - §18; B&G 1970 §6.4h
   * - Stochastic Galerkin
     - GalerkinFrame
     - Polynomial-chaos basis on the random-input axis
     - Pending
     - §15A.7
   * - MC adjoint moments
     - GalerkinFrame
     - Same SH ``frame.analysis``
       used as a variance-reduction estimator
     - Pending
     - Lewis & Miller 1993 §10
   * - SN sensitivity
     - GalerkinFrame (adjoint)
     - Same pair, applied to the adjoint flux
     - Pending
     - Cacuci 2003

The two architectural payoffs:

* **One mechanism per discipline type**, not one per consumer. The
  spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`
  emits the same ``analysis`` / ``reconstruction`` faces whether SN
  uses them for scattering or PN uses them for streaming — the
  difference is which axis the face is wrapped onto via the
  :class:`~orpheus.numerics.operator.TensorProductOperator`
  algebra (see :ref:`operator-algebra` and the tensor-product
  section there).
* **One V&V chain per discipline**. The Galerkin idempotency tests
  in :file:`tests/numerics/test_spherical_harmonic_space.py` cover
  every :class:`~orpheus.numerics.frame.GalerkinFrame` consumer, not
  just SN. The forward Petrov-Galerkin frames now carry their own
  rate-preservation **L0** gates (:mod:`tests.sn.test_homogenization` —
  the per-channel rate identity, the φV-vs-dV discriminator, and the
  Mode-11 routing sentinel; :ref:`sn-homogenization-verification`). The adjoint-weighted
  (:math:`\varphi^* \ne \varphi`) collapse now carries its own
  full-taxonomy discriminator battery (C1–C5, Cχ; landed P6;
  :ref:`frame-adjoint-weighted-seam`).


.. _frame-adjoint-weighted-seam:

Adjoint-weighted collapse — the eigenvalue-consistent taxonomy
==============================================================

.. important:: **Status — landed (P6, #281).** The adjoint-weighted
   (eigenvalue-consistent) collapse **ships**. Both
   :meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`
   (space) and
   :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`
   (energy) take an optional ``adjoint=`` keyword: pass the role-typed
   ``AdjointSolution`` that
   :func:`~orpheus.sn.solver.solve_sn_adjoint` returns (#276 A4/A5;
   :ref:`sn-adjoint`, whose importance carrier is
   :ref:`sn-adjoint-carrier`) and the collapse becomes worth-exact; omit
   it and the forward (:math:`\varphi^* = \varphi`) degenerate runs
   unchanged — **bit-identical** to the no-arg call (the §4.0 degenerate
   pins). The full algebra-of-record is
   :mod:`orpheus.derivations.common.homogenization` (theorems T0–T6,
   each an exact SymPy identity proof-welded to the production builder);
   the gate battery is the adjoint verification slice
   :ref:`sn-adjoint-verification-slice`.

The forward Petrov-Galerkin frames
(:meth:`Solution.homogenize <orpheus.sn.solution.Solution.homogenize>`,
:meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`)
weight the test functions by the **forward** flux
:math:`\chi_R = \varphi\,\mathbf{1}_R`. That is the
**Galerkin-degenerate** (:math:`\varphi^* = \varphi`) case of the
projection reactor physics ultimately wants. The general,
**eigenvalue-consistent** projection weights the test functions by the
**bilinear pair** :math:`\varphi^*\!\odot\varphi`, preserving the
functional

.. math::
   :label: sn-homogenization-adjoint-weighted

   \Sigma_R \;=\;
   \frac{\int_R \varphi^*\,\Sigma\,\varphi\;\mathrm{d}V}
        {\int_R \varphi^*\,\varphi\;\mathrm{d}V},

.. (Wired P6, #281 — no vv-status sentinel.) The eigenvalue-consistent
   (adjoint-weighted, φ*≠φ) collapse is BUILT: Solution.homogenize /
   Solution.condense implement it under ``adjoint=``. C1
   (tests.sn.test_homogenization) and C4 (tests.sn.test_condensation)
   stack verifies("sn-homogenization-adjoint-weighted") against
   structurally-independent per-region hand rules, and C2 pins the
   first-order-stationary keff signature. Covered by tests — no
   ``documented`` sentinel.

so that the multiplication factor :math:`\keff` stays stationary under
the collapse. The mechanism is the first-order eigenvalue-shift identity
(theorem T0, the keystone): for the generalized eigenproblem
:math:`A\varphi = \tfrac{1}{k}F\varphi` with left eigenvector
:math:`\varphi^*`, a perturbation :math:`(\delta A, \delta F)` shifts the
eigenvalue by

.. math::

   \delta\mu \;=\;
   \frac{\langle\varphi^*,(\delta A - \mu\,\delta F)\,\varphi\rangle}
        {\langle\varphi^*, F\varphi\rangle}
   \;+\; \mathcal O(\delta^2),
   \qquad \mu = \frac{1}{\keff}.

Replacing the fine per-cell cross sections by region-collapsed constants
**on the same fine mesh** is exactly such a perturbation — the
*XS-collapse worth*. A collapse rule that zeroes each region's worth term
therefore kills the first-order eigenvalue error of the collapse itself.
The remaining coarse re-solve error (the coarse *discretization* of
streaming) is shared by every weighting and is untouched by these rules —
the honest scope of the C2 gate (:ref:`sn-adjoint-verification-slice`).

Why this is *irreducibly* Petrov-Galerkin — :math:`M^* \ne R`, there is
**no** metric in which test :math:`= \varphi^*\mathbf{1}_R` equals trial
:math:`= \varphi\,\mathbf{1}_R` when :math:`\varphi^* \ne \varphi` — is
:eq:`sn-homogenization-bilinear` in
:ref:`sn-homogenization-why-petrov-galerkin`, and it applies *verbatim*
on the energy axis for condensation (:ref:`sn-energy-condensation`).

The per-channel collapse rules
------------------------------

Each cross-section channel has its own worth-zeroing rule — a distinct
theorem in :mod:`orpheus.derivations.common.homogenization`, each proved
as an exact SymPy identity and welded to the production builder it
mirrors (the display equations below are *generated from* that module on
every build, so the documented math and the verified math cannot drift).
With the **bilinear pair weight** :math:`w^{\pm}_{i,g} =
\varphi^*_{i,g}\varphi_{i,g}` on fine cell :math:`i`, group :math:`g`:

.. include:: ../../_generated/homogenization_collapse_rules.inc.rst

Three structural facts distinguish these from a naive "swap
:math:`\varphi` for :math:`\varphi^*`":

* **Vector channels (T1)** use the *product* :math:`\varphi^*\!\odot
  \varphi`, never a bare :math:`\varphi^*`. Solving the P0 reaction worth
  :math:`W_g = \sum_i V_i\varphi^*_{i,g}(\Sigma_{R,g} -
  \Sigma_{i,g})\varphi_{i,g} = 0` for :math:`\Sigma_{R,g}` returns the
  pair-weighted ratio as its **unique** solution; the
  bare-:math:`\varphi^*` rule leaves :math:`W_g \ne 0` (first-order in
  the adjoint tilt). This is the trap the C3 / C5 capture sentinels
  catch.
* **Matrix channels (T2)** need the **per-pair** weight
  :math:`\varphi^*_{i,g}\varphi_{i,g'}` — sink adjoint × source flux, one
  weight per :math:`(g'\!\to\!g)` entry. The source-product weight
  :math:`(\varphi^*\varphi)_{g'}` that the forward per-(cell, group)
  plumbing would broadcast does **not** zero the off-diagonal worth
  (this is why the ``adjoint=`` arm needed real per-pair plumbing, not a
  weight-array swap; B&G 1970 (6.136) :cite:`BellGlasstone1970`).
* **The fission dyad (T3)** :math:`\chi\otimes\nu\Sigma_f` collapses per
  pair into something that is not rank-1, but a
  :class:`~orpheus.data.macro_xs.mixture.Mixture` stores the *factors*.
  The **mixed-fold** rule — numerator folded by the fine emission
  importance :math:`\iota_i = \sum_g\varphi^*_{i,g}\chi_{i,g}`,
  denominator by the collapsed :math:`\tilde\iota_i` — zeroes the
  **total** fission worth for *any* simplex :math:`\chi_R` (all the
  eigenvalue needs; T0 contracts the dyad against one scalar). The
  canonical :math:`\chi_R` is the adjoint-weighted-emission convex
  average, a simplex by construction that degenerates to the
  production-weighted forward rule at :math:`\varphi^* = \varphi`.

The exact angular collision rule (T1b)
--------------------------------------

The collision term of the transport pencil acts on the **full angular
flux**, so its worth pairs the *angular* fluxes, not their scalar
moments. The exact :math:`\Sigma_t` rule (theorem T1b, user-ruled into
production) weights by

.. math::

   \rho_{i,g} \;=\; \sum_n w_n\,\psi^*_{i,g,n}\,\psi_{i,g,n},
   \qquad
   \Sigma_{t,R,g} \;=\;
   \frac{\sum_i V_i\,\rho_{i,g}\,\Sigma_{t,i,g}}
        {\sum_i V_i\,\rho_{i,g}},

the unique weight that zeroes the angular collision worth. The scalar
pair :math:`\varphi^*\!\odot\varphi` is its **P0 (isotropic)
truncation** — the two coincide *identically* on isotropic angular
shapes (:math:`\psi = \varphi/W`, :math:`\psi^* = \varphi^*/W`), so the
classical scalar prescription is exactly the isotropic limit. Because
both the forward and adjoint solutions carry the angular flux
:math:`\psi`, ORPHEUS implements the exact angular rule for
:math:`\Sigma_t` (the user ruling, P6 option 2) rather than the P0
truncation; C1's ``test_sigma_t_matches_angular_pairing_rule`` pins it
against the scalar-pair alternative on an anisotropic fixture.

The moment-resolved refinement for the **anisotropic** scattering orders
(:math:`\Sigma_{s,\ell}` pairing the :math:`\ell`-moments
:math:`\varphi^*_{\ell m}\varphi_{\ell m}`; :math:`\ell = 0` is exactly
T2's scalar pair, and Parseval makes :math:`\rho` the all-moment sum)
stays a **documented seam** until an anisotropic-collapse consumer
exists.

The balance trade-off — worth-exact constants do not balance (T4)
-----------------------------------------------------------------

Worth-exactness comes at a price that is **essential, not a bug**: a
worth-exact collapse **breaks** the definitional total-cross-section
balance

.. math::

   \Sigma_t \;=\; \Sigma_c + \Sigma_L + \Sigma_f
     + \mathrm{rowsum}(\Sigma_{s0}) + \mathrm{rowsum}(\Sigma_{2n}),

because the per-pair matrix rowsums (T2) re-weight each sink term by a
different denominator than the vector-channel collapse (T1) uses for
:math:`\Sigma_t`. Theorem T4 proves the two properties are **mutually
exclusive** away from :math:`\varphi^* = \text{const}`: restoring balance
by *defining* :math:`\Sigma_t :=` sum-of-parts re-introduces exactly that
mismatch as a first-order collision worth. This is the classical
**reactivity-preserving, not rate-preserving** property of
bilinear-weighted constants — B&G 1970 p. 308 :cite:`BellGlasstone1970`
notes the same trade-off — so the bilinear system preserves the
:math:`\keff` functional to second order, not the channel-wise reaction
rates. At :math:`\varphi^* = \varphi` both properties hold: the forward
collapse is the degenerate that enjoys both.

.. warning:: **Never call**
   :meth:`~orpheus.data.macro_xs.mixture.Mixture.assert_balanced` **on an
   adjoint-collapsed** :class:`~orpheus.data.macro_xs.mixture.Mixture`.
   The imbalance is the *derived* worth-exact property, not an accident;
   asserting balance would falsely red a correct collapse. The C1 gate
   ``test_worth_exact_collapse_breaks_balance_as_derived`` pins the
   imbalance as expected (and confirms the *forward* collapse still
   balances) — it is the committed catcher for a wiring that silently
   reverted to a balance-restoring (worth-nonzeroing) rule.

The energy axis — the Bell & Glasstone convention (T6)
------------------------------------------------------

Energy condensation has **no streaming carve** (streaming does not couple
groups), so on the energy axis the adjoint-weighted collapse is *pure
projection*: the coarse pencil is an exact left-diagonal rescaling of the
Petrov-Galerkin projection of the fine pencil (test
:math:`\varphi^*\mathbf{1}_G`, trial :math:`\varphi\mathbf{1}_{G'}`). The
convention is **Bell & Glasstone Ch. 6** (§6.4h
:cite:`BellGlasstone1970`), reconciled against the ORPHEUS carriers:

* the coarse **flux carrier** is the plain condensed flux
  :math:`\Phi_G = \sum_{g\in G}\varphi_g` (B&G (6.125) — a group
  *integral*, exactly the forward convention);
* the coarse **adjoint carrier** is the flux-weighted group average
  :math:`\Psi^{\dagger}_G = \langle\varphi^*\varphi\rangle_G/\Phi_G`
  (B&G (6.126)–(6.128)) — the choice that makes the classical *diagonal*
  bilinear vector row (B&G (6.135)) row-consistent with the plain flux
  carrier and preserves the duality pairing
  :math:`\langle\varphi^*\varphi\rangle_G = \Psi^{\dagger}_G\,\Phi_G`
  exactly;
* the matrix channels take per-block sink×source constants (B&G (6.136)),
  and fission stays **factored** — a flux-weighted
  :math:`\nu\Sigma_{f,G'}` and an adjoint-contracted emission
  :math:`\chi^{\dagger}_G`, with the rank-1 simplex rescale (the
  :class:`~orpheus.data.macro_xs.mixture.Mixture` law).

Four facts pin the convention (theorem T6):

* **T6a** — the condensed pencil's :math:`k` equals the fine :math:`k`
  *exactly* at the true spectra (a rational identity; unlike the spatial
  fission dyad T3, the factored fission survives condensation). The
  data-level pin is ``test_t6a_true_spectra_reproduce_fine_k_exactly``
  (0-D ∞-medium, :math:`\varphi = A^{-1}\chi`, :math:`\varphi^* =
  A^{-\mathsf T}\nu\Sigma_f`, to :math:`10^{-12}`).
* **T6r** — *row-scaling freedom*: rescaling every channel of a coarse
  row by a common factor preserves the pencil's spectrum, so
  :math:`k`-exactness **alone cannot pin the constants' values**. What
  pins them is the classical carrier normalization (B&G (6.125)/(6.126))
  — a Mode-12 argument one level up (see the eigenvalue-stabiliser
  accounting in :ref:`sn-adjoint-verification-slice`).
* **T6c** — *mixing* carriers (a diagonal-bilinear vector rule paired
  with a plain-sum adjoint fold in the matrix denominators) is not
  row-consistent and loses exactness: carrier consistency is
  load-bearing, not cosmetic.
* **T6b** — condensing with a *perturbed* spectrum pair leaves
  :math:`k_C(\varepsilon) - k = \mathcal O(\varepsilon^2)` for the
  bilinear convention (B&G (6.90): the flux-only error term is *first*
  order) versus :math:`\mathcal O(\varepsilon)` for the forward rule —
  the energy-axis instance of the C2 comparative order signature.

Hébert's plain lethargy-average adjoint carrier (§3.5) is the
flat-within-group approximation of B&G's flux-weighted-average carrier;
the two agree only in the many-group / flat-adjoint limit
(:cite:`Hebert2009`).

It generalises a degenerate that already ships
----------------------------------------------

The forward reaction-rate functional ORPHEUS already computes —
:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`,
:math:`\int_R \langle\Sigma_x, \varphi\rangle\;\mathrm{d}V` — is the
:math:`\varphi^* = 1` degenerate of the bilinear
:math:`\langle\varphi^*, \Sigma\varphi\rangle`. The eigenvalue-consistent
collapse is the **same** Petrov-Galerkin frame with the forward test
weight :math:`\varphi` replaced by the bilinear pair
:math:`\varphi^*\!\odot\varphi` on the test basis — a change of the
``test_basis`` *weight*, **not** a re-derivation. Writing the discipline
on the frame **type** (an explicit test basis) rather than on the measure
(a flux-folded metric) is precisely what bought this: the adjoint arm
landed as a weight construction the frame already supports, and the
C3 / C5 capture sentinels pin that the weight the frame receives is the
pair product — not a bare :math:`\varphi^*`, not the forward
:math:`\varphi`.

With :math:`\varphi^*` landed, the adjoint-weighted collapse is the
**first production consumer** of the importance field; the others —
perturbation theory and generalised perturbation theory — are catalogued
in :ref:`sn-adjoint-consumers`.


.. _frame-discipline-as-a-type:

Discipline as a type, not a property or an operator marker
==========================================================

The discipline — Galerkin vs Petrov-Galerkin — is a genuine **kind of
object**, so it is carried by the frame **type** (Issue #268). Two
rejected alternatives clarify why.

**Rejected (a): discipline as a marker ABC on the operator role.** An
earlier draft put the discipline on the analysis operator via marker
mixins ``GalerkinProjection`` / ``PetrovGalerkinProjection``:

.. code-block:: python

   # RETIRED: discipline declared on the operator role
   class GalerkinProjection(AnalysisOperator, ABC): ...
   class PetrovGalerkinProjection(AnalysisOperator, ABC): ...

This declares a discipline on the *role* (:math:`M`) when the
discipline is really a fact about the *frame* — the relationship
between the test and trial spaces, which an analysis operator in
isolation cannot express. The marker ABCs were retired; the
:mod:`orpheus.numerics.projection` module now carries only the two
discipline-free operator roles
(:class:`~orpheus.numerics.projection.AnalysisOperator`,
:class:`~orpheus.numerics.projection.ReconstructionOperator`).

**Rejected (b): discipline as a derived property of the frame.** An
intermediate draft proposed collapsing the distinction to a boolean
property (``measure == basis.canonical_measure``), on the theory that
homogenisation is "really Galerkin in a weighted metric". That reading
folds the solution into the metric and breaks for the
eigenvalue-consistent (adjoint-weighted) case (see the Petrov-Galerkin
note above). A property cannot encode a genuinely different object;
the discipline is a TYPE.

**Shipped: discipline as a Liskov-correct type hierarchy.** The frame
type names the discipline, and the user's pedantic naming rule is
satisfied — **a reader of a type name knows its properties without
reading the docstring**:

.. code-block:: python

   from orpheus.numerics.frame import (
       FrameBase, PetrovGalerkinFrame, GalerkinFrame,
   )

   # Galerkin: test IS trial, Π* ∝ R (a CANONICAL dual: 1/W for SH)
   sh_frame = quad.angular_frame(L)            # -> GalerkinFrame

   # Petrov-Galerkin: explicit test basis, M* ≠ R (an oblique dual)
   homog = PetrovGalerkinFrame(
       basis=mesh.indicator_basis(),
       measure=mesh.volume_measure,
       test_basis=adjoint_weighted_indicators,  # the discipline
   )

A reader of ``PetrovGalerkinFrame(...)`` immediately knows the
analysis measures against a distinct test basis, so :math:`M^*` is an
**oblique** dual — it re-synthesises on the test functions and is not
proportional to :math:`R` at all; a reader of
:class:`~orpheus.numerics.frame.GalerkinFrame` knows the strengthened
**canonical**-dual promise holds, :math:`M^* = S_0\circ G^{-1}`, which
re-synthesises on the trial basis and is therefore :math:`R` up to the
per-mode Gram factor (one scalar :math:`1/W` for the SH frame —
:eq:`frame-square-closure-sh`; never the bare :math:`M^* = R`, which is
the ERR-039 claim). The hierarchy answers the
discipline question without reading prose, and a
:class:`~orpheus.numerics.frame.GalerkinFrame` that is handed a
distinct ``test_basis`` raises (the contradiction is unrepresentable).


Numerical evidence
==================

The L1-tagged tests in
:file:`tests/numerics/test_spherical_harmonic_space.py` verify the
Galerkin discipline's invariants on the spherical-harmonic
:class:`~orpheus.numerics.frame.GalerkinFrame`'s ``analysis`` /
``reconstruction`` faces:

1. **Idempotency** (4π-tightness):
   :math:`M R c = 4\pi c` on band-limited
   coefficient input, verified at :math:`L = 2,\,3,\,4` against
   Lebedev orders :math:`7,\,13,\,17`. See
   :eq:`pi-r-equals-4pi-i` in :ref:`spherical-harmonics`.
2. **Adjoint pairing**:
   :math:`\langle M \psi, c \rangle_{G^{-1}} =
   \langle \psi, M^* c \rangle_W`
   with :math:`M^* = S_0 \circ G^{-1} = R/W` — the F-0 Parseval
   metric on the coefficient side (:ref:`frame-parseval-metric`) —
   verified to ``rtol=1e-12`` on a Lebedev order-13 grid at
   :math:`L = 3`. (⛔ Pre-F-0 this line read :math:`M^* = g_C\,S_0`,
   the continuum-metric adjoint; see
   :ref:`frame-parseval-what-was-wrong`.)
3. **Parseval and the frame square**: the isometry
   :math:`\|M\psi\|_{G^{-1}} = \|\psi\|_W` on band-limited input and
   the closure :math:`M^* = R/W`, :math:`R^* = W\,M`, verified over
   six sphere quadrature families in
   :file:`tests/numerics/test_frame.py` — with a loaded-not-blind
   negative leg that re-installs the pre-F-0 metric and measures the
   ratio it produces. Table:
   :ref:`frame-parseval-numerical-evidence`.

The tests verify mathematical identities of the operator algebra
(V&V level L1 — equation verification by analytical reference). The
companion L0/foundation shape and predicate tests verify software
invariants (frame face spaces, the ``is_invertible`` / ``is_adjointable``
predicates) and are tagged accordingly. The **forward** Petrov-Galerkin frames now ship their own
**L0** numerical evidence — the per-channel rate-preservation identity,
the φV-vs-dV (flux- vs volume-weighting) discriminator, the simplex /
production-weight :math:`\chi` gates, and the Mode-11 routing sentinel —
in :mod:`tests.sn.test_homogenization`
(:ref:`sn-homogenization-verification`),
together with the condensation gates of :ref:`sn-energy-condensation`.
The adjoint-weighted (:math:`\varphi^* \ne \varphi`) collapse now ships
its own full-taxonomy discriminator battery (C1–C5, Cχ; landed P6, #281;
:ref:`frame-adjoint-weighted-seam`).


Implementation map
==================

* :class:`~orpheus.numerics.frame.FrameBase` — the abstract discrete
  frame: binds a :class:`~orpheus.numerics.basis.Basis` to a
  :class:`~orpheus.numerics.measure.DiscreteMeasure` and emits the
  ``analysis`` (:math:`M = T`) and ``reconstruction`` (:math:`R`)
  faces. Carries the discipline-free mechanics (table, the two
  spaces, the reconstruction face, the analysis-face wiring); the
  single mechanism for every choice-dependent change-of-basis. Also
  emits the **composed-operator verbs**
  (:ref:`frame-composed-verbs`):
  :meth:`conjugate <orpheus.numerics.frame.FrameBase.conjugate>`
  (:math:`R\circ A\circ M`),
  :meth:`reconstruct_after <orpheus.numerics.frame.FrameBase.reconstruct_after>`
  (:math:`R\circ A`), and
  :meth:`project <orpheus.numerics.frame.FrameBase.project>`
  (:math:`G^{-1}M`) with its :attr:`gram_inverse
  <orpheus.numerics.frame.FrameBase.gram_inverse>` cross-Gram arrow
  (:class:`~orpheus.numerics.frame.CrossGramInverse`,
  ``test_space → basis_space``).
* :meth:`GalerkinFrame.at_order
  <orpheus.numerics.frame.GalerkinFrame.at_order>` — THIS frame over the
  same measure with its truncated trial family cut at ``L_new``; the verb
  a frame-dressed moment head truncates through
  (:func:`~orpheus.numerics.spaces.moment_head.truncated_head`).
* :class:`~orpheus.numerics.basis.GramStructure` — the trial basis's
  projection-validity declaration (``DIAGONAL`` / ``PARTITION_OF_UNITY``
  / ``DENSE``) that decides whether :meth:`project
  <orpheus.numerics.frame.FrameBase.project>` uses the row-sum probe or
  refuses (:ref:`frame-least-squares-discipline`).
* :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` — the general
  discipline: an explicit ``test_basis`` distinct from the trial
  basis, so :math:`M^* \ne R`. The base for homogenisation and
  condensation.
* :class:`~orpheus.numerics.frame.GalerkinFrame` — the Galerkin
  specialisation (``test is trial``, :math:`M^* = R`). The angular
  spherical-harmonic frame is the canonical instance.
* :class:`~orpheus.numerics.basis.Basis` — the synthesis (trial)
  side ABC: tabulate, naked synthesis :math:`S_0`, the three
  weighted contractions, and the discrete Gram.
* :class:`~orpheus.numerics.basis.SphericalHarmonicBasis` — the
  first concrete basis (real spherical harmonics); carries the
  no-prefactor convention and the
  :attr:`~orpheus.numerics.basis.SphericalHarmonicBasis.addition_theorem_factor`
  :math:`(2\ell+1)`.
* :class:`~orpheus.numerics.projection.AnalysisOperator` — the
  abstract fine→coarse operator role :math:`M : V \to W`; the
  ``analysis`` face subclasses it.
* :class:`~orpheus.numerics.projection.ReconstructionOperator` —
  the abstract coarse→fine operator role :math:`R : W \to V`; the
  ``reconstruction`` face subclasses it.
* :meth:`Quadrature.angular_frame(L)
  <orpheus.numerics.quadrature.Quadrature.angular_frame>` — builds
  the order-:math:`L` spherical-harmonic
  :class:`~orpheus.numerics.frame.GalerkinFrame` on a quadrature; the
  single home of the :math:`S^2` embedding.

The full-space projector — the operator that projects the SN
:math:`(N, n_x, n_y, n_g)` angular flux onto the
:math:`(\text{head}, n_x, n_y, n_g)` moment field, the head being
:math:`(L+1, 2L+1)` on a rule that binds the harmonics and
:math:`(L+1,)` on a 1-D rule since 2026-09-02
(:ref:`frame-g0-descent-arrow`) — is built as a
**tensor product** of the angular-axis analysis face :math:`M`
and identity operators on the spatial / energy axes:

.. code-block:: python

   from orpheus.numerics.operator import IdentityOperator

   frame = quad.angular_frame(L)
   M = frame.analysis
   M_full = M & IdentityOperator() & IdentityOperator() & IdentityOperator()

The ``&`` dunder constructs the
:class:`~orpheus.numerics.operator.TensorProductOperator`. See
:ref:`operator-algebra` and the **Tensor product algebra** section
there for the relationship between this operator-algebra type and
the underlying numpy primitives (``np.einsum``, ``np.tensordot``,
``np.kron``).


History — from operator classes to the discipline-type frame
============================================================

The spherical-harmonic projection and reconstruction were first
shipped (Wave 0 of the SN performance plan) as standalone operator
classes ``HarmonicMomentProjection`` / ``HarmonicMomentReconstruction``
under a three-level inheritance
(``ProjectionOperator`` → ``GalerkinProjection`` → concrete). Two
naming-audit corrections then established the discipline-must-be-
typed pedagogy.

The Frame/Basis carve (``refactor/operator-inverse-algebra``)
took the next step: the projection :math:`M = Y^*W` and the
addition-theorem reconstruction :math:`R = (2\ell+1)\,S_0` are NOT
two unrelated operator classes — they are the **two faces of one
discrete frame** binding the SH basis to the angular measure. The
standalone operator classes were retired into the frame faces:

* ``HarmonicMomentProjection`` → ``frame.analysis``
  (:attr:`FrameBase.analysis <orpheus.numerics.frame.FrameBase.analysis>`,
  the analysis face :math:`M = T`);
* ``HarmonicMomentReconstruction`` → ``frame.reconstruction``
  (:attr:`FrameBase.reconstruction
  <orpheus.numerics.frame.FrameBase.reconstruction>`, the
  reconstruction face :math:`R`);
* the :math:`(2\ell+1)` factor moved onto
  :attr:`SphericalHarmonicBasis.addition_theorem_factor
  <orpheus.numerics.basis.SphericalHarmonicBasis.addition_theorem_factor>`
  (one home for the SH convention).

The P1 discipline-type carve (Issue #268) took the final step: the
discipline, which an earlier draft had carried as marker ABCs on the
operator role and a later draft proposed collapsing to a frame
*property*, became the frame **type**
(:class:`~orpheus.numerics.frame.FrameBase` →
:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` →
:class:`~orpheus.numerics.frame.GalerkinFrame`). The architectural
payoff: one mechanism (the frame), the discipline visible in the type,
and the eigenvalue-consistent homogenisation case correctly typed as
Petrov-Galerkin (test = adjoint-weighted indicator) rather than
mis-folded into a weighted measure.

**2026-08-23 — step F-0, the metric truth.** Every step above moved
*operators*; none had asked whether the **metric** the coefficient
codomain carries is the right one. It was not. The frame exposed
``basis.space`` unchanged, so the codomain carried the basis's
continuum Gram :math:`g_C` — the Gram, where the covariant moments
analysis emits need its *inverse*
(:eq:`frame-analysis-is-the-gram`). ``FrameBase`` gained
:attr:`~orpheus.numerics.frame.FrameBase.discrete_gram` (the cached
:math:`K\times K` trial Gram),
:attr:`~orpheus.numerics.frame.FrameBase.discrete_gram_structure` (a
MEASURED diagonality verdict, distinct from the basis's DECLARED
:attr:`~orpheus.numerics.basis.base.Basis.gram_structure`), and a
:attr:`~orpheus.numerics.frame.FrameBase.basis_space` that dresses the
space with :math:`G^{-1}` on a diagonal frame and refuses on a dense
one. Nothing about the design was wrong; **what was stored** was. Full
account, with the derivation, the slab witness, the family-wide
residual table, and the three reasons no gate could see it:
:ref:`frame-parseval-metric`. Recorded debt at the time: a
matrix-valued metric needs the CS4c Riesz-leg machinery
(``.claude/plans/frame_square_recarve.md``). ⭐ That half of the debt
was discharged by campaign 1 P7 — see the 2026-08-30 entry below; the
dense **refusal** described here is therefore history, and only the
*diagnosis* it rested on still stands.

**2026-08-23 — step F-1, the mint: the faces ARE the bound operators.**
With the metric right, the remaining asymmetry was ownership. A frame
is not an operator; it is an operator **factory**, and it is shared —
the scattering operator, the windowed accumulation, DSA's
:math:`\ell=1` row and the loss-kernel gauge all read one frame. F-1
made the transport-level analysis and reconstruction **bound**
operators minted BY the frame (domain and codomain are the two full
field spaces), so the pre-F-1 "not yet typed" ``codomain is None`` debt
on :class:`~orpheus.sn.operators.windowing.BulkAnalysisOperator` died
with the mint and the composition guards now check that end.

**2026-08-24 — step S6.0b, the rank-one instance.** The same
generator, at :math:`K = 1`: a **single-region indicator** frame over
an axis's index set induces the axis **collapse pair** — the
retraction :math:`R = \pi_*` and its section :math:`E` — whose
normalisation divisor IS this page's Parseval metric, the
:math:`1\times1` :attr:`~orpheus.numerics.frame.FrameBase.discrete_gram`
entry. The frame is built at the mint, read for its induced data, and
**discarded** (the forgetful-map half of the stage-2 generator
discipline), with a tightness gate standing in for instance sharing.
Full account, including why the pair is not lifted out of the harmonic
frame: :ref:`spaces-collapse-pair` on
:doc:`/theory/foundations/spaces`.


**2026-08-30 — campaign 1 phase P7, the metric becomes an object, and
the dense arm is DRESSED.** F-0 put the metric on the right side; P7
made it a first-class thing. A space's metric stops being an array that
is *multiplied* into the element and becomes a typed
:class:`~orpheus.numerics.metric.HilbertMetric` that is *applied*, of
which the Hadamard weight is the diagonal special case
(:ref:`spaces-metric-object` on
:doc:`/theory/foundations/spaces` is the doctrine's home). The frame is
its founding consumer: ``basis_space``'s ``DENSE`` arm now installs
``DenseMetric.inverse_of(discrete_gram)`` — the Moore–Penrose
pseudo-inverse at a pinned cutoff, with the exact symmetrized Gram kept
as the inverse face — and strips the basis's continuum weights, so
**Parseval becomes a theorem on every frame** rather than a property of
the diagonal ones (:eq:`spaces-pseudo-inverse-parseval`). ``pinv``
rather than ``inv`` is forced, not stylistic: `[M]` the flagship slab
Gram is :math:`15\times15` with 5 live slots and **rank 4**, so no
inverse exists, while :math:`G G^{+} G = G` holds to
:math:`1.6\times10^{-15}` and is exactly what the theorem needs. Two
consequences the phase had to own rather than announce. (i) The change
is *behavioural*: `[M]` 10 of the 30 shipped angular frame
constructions (nine quadrature families :math:`\times` :math:`L \le 3`,
plus Lebedev-13) measure ``DENSE`` and are newly dressed, as is the
non-angular overlap frame; the scattering operator builds one of them
in production; and `[M]` the analysis adjoint moves by
:math:`98\,\%` in Frobenius relative on all three ``DENSE`` angular
frames measured — the recorded F-0 limitation repaired. The phase's
design pre-flight injected the dressing on unmodified production over
four test trees (4371 tests) and reddened **two** gates, neither of
which observes the adjoint, so the change landed with the gate that
does. (ii) ``FrameBase.gram`` — the accessor CS4c step 6 item 6.2c-ii
retired in favour of the :attr:`gram_inverse
<orpheus.numerics.frame.FrameBase.gram_inverse>` arrow — had to learn to
**strip** the dressing: the row-sum probe is CROSS-Gram machinery and
must never inherit the trial-side Parseval metric, `[M]` on pain of a
:math:`162\,\%` projection error on the overlap frame. The correctness
evidence is the wrong-metric discriminator, not reciprocity — the
Hilbert-adjoint identity holds for every invertible :math:`G` and can
never adjudicate one (:ref:`frame-parseval-dense-arm`).

**2026-09-02 — #429 tracker 2.5: the frame is the single source of the
COEFFICIENT SPACE, not only of the faces.** F-1 made the *faces* bound
operators minted by the frame; the *space* those faces land in was still
being re-derived by their consumers. `[M]` the angular coefficient space
had **eight** homes — the bound basis plus **seven** production sites
re-minting ``SphericalHarmonicSpace.from_L(L)`` — and two
``isinstance(basis, SphericalHarmonicBasis)`` doors on
:class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame` that
would have refused the very basis ERR-080's repair needs. Both doors now
ask for the :class:`~orpheus.numerics.basis.base.TruncatedBasis`
**surface** and every consumer READS ``basis.space``
(:eq:`moment-space-read-off-the-frame`), so *which family* is the
quadrature's decision and propagates by construction. The fork the step
had to settle is this page's own: the operator ends bind the basis's
**continuum** Gram, not the frame's Parseval-dressed ``basis_space``,
because an :math:`\ell`-diagonal metric commutes with a per-:math:`\ell`
scalar and the dressing does not — `[M]` binding the dressed space would
have moved :math:`\Lambda^{*}` by up to :math:`158\,\%`, on exactly the
1-D and folded rules the campaign is repairing and on none of the
full-sphere ones. Nothing moved: `[M]` metric-identical to the mint it
replaces on **33 of 33** (rule, :math:`L`) rows and the slab flux
``array_equal`` pre/post at :math:`L = 0` … :math:`3`. A capability, not
a repair — ERR-080 stays open. Full account:
:ref:`frame-moment-space-single-home`.

⛔ **The fork half of that entry was OVERTURNED on 2026-09-08 (item
6.2c-ii, ruling R-6.2c-1).** The *single-source* half stands untouched and
is what the entry is about. What did not survive is the choice of END:
`[M]` the :math:`158\,\%` is a whole-matrix statistic over columns off the
range of :math:`M` (on a physical :math:`\varphi = M\psi` the two ends
agree on **33 of 33** rows), and the leg it never priced is that the
continuum end breaks Parseval on **33 of 33** rows by
:math:`3.41\ldots157.91`. See :ref:`frame-the-one-moment-space`.

**2026-09-08 — CS4c step 6 item 6.2c: the head becomes AXIS-BUILT and
there is ONE moment space.** Two commits. **6.2c-i** (``db5be2ec``) made a
space's metric a DERIVED object: an axis-built space's metric is one
:class:`~orpheus.numerics.metric.FactoredMetric` over its axes — a
:class:`~orpheus.numerics.metric.DiagonalMetric` per weighted axis on its
own block, nothing on a counting axis — with an explicit metric object
admitted beside the axes only as the positioned **overlay** of forms (one
entry per axis, in order, a form only where the axis carries no measure),
MERGED into the derived entries rather than substituted for them. That is
the home a dense Gram needed (ruling R-6.2c-2: *a Gram is a FORM, never
on* ``Axis.weights``); the inline ``_apply_axes_weights`` loop retired into
:meth:`DiagonalMetric.apply_block
<orpheus.numerics.metric.DiagonalMetric.apply_block>`, bit-identically.
**6.2c-ii** then made both harmonic heads axis-built — one
:class:`~orpheus.numerics.axis.HarmonicAxis` or
:class:`~orpheus.numerics.axis.LegendreAxis`, ``MODAL``, whose measure IS
the head's metric — which puts the metric into the identity and forces the
fork. Ruled: the ONE moment space carries **Parseval**, so the carrier's
cached mint, every moment field, every operator end and the frame's own
faces read ``frame.basis_space``; the continuum head survives as the
basis's own coefficient space. Three consequences this page owns: the
metric-blind ``(name, shape)`` seam is **gone**
(:ref:`frame-the-one-moment-space`); the projection normalisation became
the typed arrow :class:`~orpheus.numerics.frame.CrossGramInverse`, because
its metric-twin spelling stopped type-checking under structural identity
(:ref:`frame-gram-inverse-arrow`); and truncation re-mints through the
head axis's generator, with :meth:`GalerkinFrame.at_order
<orpheus.numerics.frame.GalerkinFrame.at_order>` as the frame-side verb
(:ref:`frame-at-order`). `[M]` the converged flux, the residual trajectory
and ``n_inner`` are unchanged; what moves is the SI increment DIAGNOSTIC —
``‖Δφ‖`` by 91.6 % and ρ by 3.85 % relative — pinned for the first time by
``tests/sn/solve/test_windowed_si_diagnostic_trajectory.py``.

References
==========

* Brenner, S. C. and Scott, L. R. (2008). *The Mathematical Theory
  of Finite Element Methods*, 3rd ed. Springer. §3.4 (Galerkin /
  Petrov-Galerkin general framework — test vs trial space).
* Christensen, O. (2016). *An Introduction to Frames and Riesz
  Bases*, 2nd ed. Birkhäuser. (The analysis operator :math:`T`, the
  synthesis operator :math:`T^*`, the frame operator
  :math:`S = T^*T`, tight frames, and the canonical dual — the
  harmonic-analysis foundation of the
  :class:`~orpheus.numerics.frame.FrameBase` abstraction.)
* Bell, G. I. and Glasstone, S. (1970). *Nuclear Reactor Theory*.
  Van Nostrand Reinhold. §1.6 (spherical-harmonic moment
  projection in transport).
* Lewis, E. E. and Miller, W. F. Jr. (1993). *Computational Methods
  of Neutron Transport*. ANS. §4.7 (Pℓ Galerkin reconstruction with
  the :math:`(2\ell+1)` factor).
* Müller, C. (1966). *Spherical Harmonics*. Lecture Notes in
  Mathematics **17**, Springer. (The Funk–Hecke theorem: spherical
  harmonics are the eigenfunctions of any zonal kernel on
  :math:`S^2`, with eigenvalue
  :math:`\lambda_\ell = 2\pi\int_{-1}^{1} k(t) P_\ell(t)\,dt` — the
  structural ground for "the SH frame is scattering's eigenbasis".)
* Hébert, A. (2009). *Applied Reactor Physics*. Polytechnique. §3.3
  (the flux→SH-moment projection :math:`M`, Eq. 3.55, used **only**
  in the scattering source Eq. 3.54; fission isotropic Eq. 3.57;
  integral form natively isotropic Eq. 3.42), §3.6–3.7 (the streaming
  :math:`\ell\!\leftrightarrow\!\ell\pm1` recurrence), §6.2 (energy
  condensation as a Petrov-Galerkin projection), §13
  (eigenvalue-consistent / adjoint-weighted spatial homogenisation).
* Brockmann, H. (1981). *Treatment of anisotropic scattering in
  numerical neutron transport theory*. Nucl. Sci. Eng. **77** (4),
  377–414. Eq. (47) — the Legendre flux moment
  :math:`\Phi_\ell = 2\pi\int P_\ell(\mu)\Phi\,d\mu` is introduced
  expressly for the anisotropic-scattering source and reused across
  SN / FEM / orders-of-scattering.
* Fletcher, J. K. (1983). *The solution of the multigroup neutron
  transport equation using spherical harmonics*. Nucl. Sci. Eng.
  **84**, 33–46. Eq. (7) — the moment equation is diagonal in
  :math:`\ell` "because of the orthogonality of spherical harmonics"
  (scattering); Eq. (5) — the streaming term produces the
  block-tridiagonal :math:`\ell\!\leftrightarrow\!\ell\pm1` coupling.
* Ahrens, C. D. (2014). *Lagrange Discrete Ordinates: a new angular
  discretization for the three-dimensional transport equation*.
  arXiv:1405.3968. Eq. (7) and abstract — the negative-space proof:
  LDO **removes** the SH moment projection ("no spherical harmonic
  moments are needed") precisely by reformulating the scattering
  source.
* Cacuci, D. G. (2003). *Sensitivity and Uncertainty Analysis,
  Volume I*. CRC Press. (Adjoint flux moments and the Galerkin
  pair on the adjoint side.)
* Xiu, D. and Karniadakis, G. E. (2002). *The Wiener-Askey
  polynomial chaos for stochastic differential equations*. SIAM J.
  Sci. Comput. 24 (2), 619–644. (Stochastic Galerkin on the random
  input axis.)
* Grand Report v3 §5.7 (line 664), §17 (line 3935), §32.7 — the
  catalog entries that drove the placement of these primitives in
  :mod:`orpheus.numerics.projection`.
