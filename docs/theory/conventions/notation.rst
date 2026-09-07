.. _theory-conventions-notation:

========
Notation
========

**One concept, one spelling.** This page is the corpus's symbol table
— the single place where every load-bearing symbol is tied to its code
spelling and its canonical page — and the literature crosswalk that
maps ORPHEUS conventions onto the published canon. The part index
(:ref:`theory-conventions`) carries the evidence for *why* a crosswalk
must exist: the canon disagrees with itself, silently, and the failure
class it breeds is code that runs, converges, and is wrong by a
constant.

The internal-consistency doctrine is the stronger half of the rule.
Before any external mapping, ORPHEUS itself must express each concept
with exactly one spelling across code, tests, and prose; where two
layers must genuinely differ (the honest solver-level operator versus
the numerics-layer resolvent operand — the last crosswalk row below),
the binding is stated explicitly at **both** ends, so the difference
is a documented bridge and not a drift. Every row of the symbol table
is checkable: symbol → code object → canonical page, and where a
convention has teeth, the row names the test that bites.

.. _notation-symbol-table:

The ORPHEUS symbol table
========================

The columns are the discipline: a symbol without a code spelling is
not yet realized; a code object without a documented symbol is not yet
articulated. Rows whose *normalization* carries the trap point at
:doc:`normalization`.

Phase space and fields
----------------------

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Symbol
     - Code spelling
     - Meaning
   * - :math:`\psi`
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
     - :term:`Angular flux <angular flux>` on the ``(N, ng, nx, ny)`` layout
       (:ref:`theory-sn-index-convention`).
   * - :math:`\phi_{\ell m}`, :math:`\phi`
     - :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
     - Harmonic flux moments; :math:`\phi \equiv \phi_{00}` is the
       :term:`scalar flux`. Moments carry **no** :math:`4\pi` prefactor —
       the :math:`\Sigma w` normalization does the work (see
       :doc:`normalization`).
   * - :math:`\hat\Omega_m`, :math:`w_m`
     - :class:`~orpheus.numerics.quadrature.directional.Quadrature`
       — ``.mu_x`` / ``.mu_y`` / ``.mu_z``, ``.weights``
     - :term:`Ordinate <ordinate>` directions and weights. The accessors are derived
       views over the one discrete measure
       (:doc:`/theory/foundations/discrete_measures`) — the
       canonical dim-agnostic form is ``axis_cosines(i)`` — and
       there is no separate ordinate storage to drift.
   * - :math:`W`
     - ``quadrature.weights.sum()``
     - The weight sum — **the** normalization pivot:
       :math:`W = 2` for Gauss–Legendre
       (:meth:`~orpheus.numerics.quadrature.directional.Quadrature.gauss_legendre`),
       :math:`W = 4\pi` for the sphere rules (``lebedev``,
       ``level_symmetric``, ``product``). Enforced in
       ``tests/numerics/test_quadrature_directional.py``.
   * - :math:`\mu`
     - ``mu_x`` (slab); ``mu_z`` (axial cosine)
     - Direction cosines. The 1-D slab sweeps along :math:`x`
       (:math:`\mu > 0` flows :math:`+x`). The cylinder's **axial**
       cosine is ``mu_z`` — spelled :math:`\mu` in the accessor's
       own docstring, :math:`\xi` in much of the cylinder
       literature. The harmonic basis's polar axis is
       :math:`\mu_x`
       (:doc:`/theory/foundations/spherical_harmonics`).

Materials
---------

All on :class:`~orpheus.data.macro_xs.mixture.Mixture`:

.. list-table::
   :header-rows: 1
   :widths: 14 18 68

   * - Symbol
     - Attribute
     - Meaning
   * - :math:`\Sigma_{\mathrm{t}}`
     - ``SigT``
     - Total cross section, per group.
   * - :math:`\Sigma_{\mathrm{s},\ell}`
     - ``SigS[l]``
     - The :math:`\ell`-th Legendre scattering transfer matrix,
       stored ``[g_from, g_to]`` (source-row) — see
       :eq:`sigs-convention` and the crosswalk below.
   * - :math:`\nu\Sigma_{\mathrm{f}}`
     - ``SigP``
     - Production cross section (:math:`\nu`-weighted fission).
   * - :math:`\chi`
     - ``chi``
     - Fission emission spectrum. A probability simplex for
       producing mixtures, null otherwise — enforced at
       construction, so an illegal spectrum is unrepresentable.
   * - :math:`g, g'`
     - ``ng`` groups
     - Group indices, **fast → thermal** (``g = 0`` is fastest).
       Downscatter therefore fills the ``g_to >= g_from`` triangle.

Operators
---------

The six leaves and their composites
(:doc:`/theory/methods/sn/solver`,
:doc:`/theory/foundations/operator_algebra`):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`L`
     - Streaming, **bulk only** (:math:`\hat\Omega\cdot\nabla`). The
       boundary law is the sibling :math:`B`, never folded into
       :math:`L`.
   * - :math:`C`
     - Collision / removal (:math:`\Sigma_{\mathrm{t}}`).
   * - :math:`S`
     - Scattering gain. A kernel with the Funk–Hecke factorization
       :math:`S = R \circ \Lambda \circ M` — not a projection. The
       in-scatter contraction uses the **transpose** of the stored
       matrix (crosswalk row 1).
   * - :math:`N_{2n}`
     - The :math:`(n,2n)` emission gain — the **same** binding as
       :math:`S` in a different role, over the mixture's ``Sig2``
       Legendre stack and with the yield :math:`\nu_{2n} = 2` inside
       :math:`\Lambda`.  First-class since CS4c step 3; anything stated
       for :math:`S` holds for it with :math:`y = 2`
       (:ref:`the two collision gains <operator-algebra-two-gains>`).
   * - :math:`B`
     - The boundary law as a first-class sibling operator
       (reflective / vacuum / white), every geometry.
   * - :math:`F`
     - Fission production — rank-1 in energy,
       :math:`|\chi\rangle\langle\nu\Sigma_{\mathrm{f}}|`.
   * - :math:`A = L + C - S - N_{2n} - B`
     - **The honest within-group operator.** Page-wide, the bare
       letter :math:`A` means exactly this composite;
       the four-term :math:`L+C-S-B` is legitimate only where a method
       **fuses** the two collision gains at its composition site (the
       1-D diffusion solver) or where the fixture has
       :math:`\Sigma_{2n} \equiv 0`;
       :math:`(L+C)^{-1}` is the transport :term:`sweep` — the inner kernel
       of :math:`A^{-1}`, never "the sweep is :math:`A^{-1}`". Any
       local rebinding of :math:`A` must be declared where it is
       used (crosswalk row 8).
   * - :math:`k`, :math:`K = A^{-1}F`
     - The multiplication eigenvalue and the multiplication
       operator whose dominant eigenvalue it is.
   * - :math:`q_{\mathrm{ext}}`
     - External (inhomogeneous) source.

Discretization factors
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`\alpha_{m \pm 1/2}`
     - Curvilinear angular-redistribution coefficients.
       :math:`\alpha_{1/2} = 0` seeds the recursion in **every**
       published convention — the recursion's *spelling* varies by
       source (crosswalk row 5 and
       :ref:`normalization-alpha-crosswalk`).  The far end
       :math:`\alpha_{M+1/2} = 0` also holds in every spelling, but as a
       *consequence* of the measure's antisymmetry rather than a seed — an
       admission contract, not an axiom (:ref:`sn-alpha-dome-closes`).
   * - :math:`\tau`
     - In the S\ :sub:`N` sweep: the :term:`weighted-diamond <weighted diamond difference>` **closure
       weight** (:cite:`BaileyMorelChang2010` Eq. (15);
       :doc:`/theory/methods/sn/curvilinear_one_group`). In CP /
       MoC: the **optical path length** :math:`\Sigma_{\mathrm{t}}
       s` — the sense the corpus root page
       (:doc:`/theory/foundations/path_integral`) adopts page-wide,
       naming the closure weight in words there. Two
       canonical-in-their-literature objects sharing one letter;
       module context disambiguates (crosswalk row 7).
   * - :math:`\beta`
     - The angular-closure parametrization on the S\ :sub:`N`
       machine header (:doc:`/theory/methods/sn/index`); the
       Morel–Montry choice :cite:`MorelMontry1984` is the unique
       exact-on-linear-in-:math:`\mu` member
       (:cite:`BaileyMorelChang2010` Eq. (43)). ⚠ Collides with the
       Larsen–Morel review's :math:`\beta` = sphere
       *redistribution* coefficient (crosswalk row 7).
   * - :math:`\Delta A / w`
     - The **geometry factor** — the named object guaranteeing
       per-ordinate flat-flux consistency in curvilinear balance
       (:doc:`/theory/methods/sn/curvilinear_one_group`). The canon
       uses it everywhere and names it nowhere.
   * - ``(N, ng, nx, ny)``
     - The array layout: ordinates, groups, space. 1-D keeps
       ``ny = 1`` as a singleton, never squeezed
       (:ref:`theory-sn-index-convention`).

.. _notation-crosswalk:

The literature crosswalk
========================

The mapping discipline: **never map letters; map definitions.** Each
row is one convention axis on which the canon differs — from ORPHEUS,
from each other, or from itself.

.. list-table::
   :header-rows: 1
   :widths: 16 42 42

   * - Axis
     - ORPHEUS
     - The canon
   * - 1 — scattering arrow
     - Storage is ``[g_from, g_to]`` — **source-first** rows
       (:eq:`sigs-convention`). The balance applies the
       **transpose**: in-scatter into :math:`g` is
       :math:`(\boldsymbol{\Sigma}_{\mathrm{s}}^T
       \boldsymbol{\phi})_g`.
     - Hébert :cite:`Hebert2009` writes **destination-first**
       :math:`\Sigma_{\mathrm{s}}(E \leftarrow E', \Omega \leftarrow
       \Omega')`; Bell & Glasstone :cite:`BellGlasstone1970`
       (:math:`\sigma(x, E' \to E)`) and Stacey :cite:`Stacey2007`
       (:math:`\Sigma_{\mathrm{s}}(\Omega' \to \Omega)`) are
       **source-first**. Importing a destination-first matrix
       requires a transpose relative to ORPHEUS storage;
       source-first maps index-for-index.
   * - 2 — triangularity
     - With fast → thermal ordering, the **stored** downscatter
       matrix is **upper-triangular**; the **acting** transpose is
       lower-triangular. Enforced:
       ``tests/data/test_gendf_canonical_order.py::``
       ``test_downscatter_is_upper_triangular``.
     - A text's "the scattering matrix is lower-triangular" is
       meaningless until its row convention and group ordering are
       fixed — the *same physics* flips triangle under either
       choice. State both before importing any triangular-solve
       argument.
   * - 3 — weight sum :math:`W`
     - :math:`W = 2` (Gauss–Legendre) and :math:`W = 4\pi` (sphere
       rules), enforced by test. Moment definitions divide by
       :math:`W` explicitly — no silent unit-measure assumption.
     - Hébert §3.9.1 normalizes 1-D Gauss–Legendre to
       :math:`\Sigma w = 2`; five pages later, Eqs. (3.363)–(3.364)
       take :math:`\Sigma w = 1` over the positive octant — same
       symbol :math:`w_n`, two normalizations, no note. Every
       imported weight-bearing formula must be read against its
       page-local :math:`\Sigma w`.
   * - 4 — the :math:`(2\ell+1)` prefactor
     - The **no-prefactor** harmonic basis: :math:`(2\ell+1)` sits
       in the expansion, outside the basis functions
       (:doc:`/theory/foundations/spherical_harmonics`).
     - Hébert carries :math:`4\pi` in (3.30) but :math:`2` in
       (3.425) — the same object, silently tied to dimensionality.
       ERR-039 and ERR-051 were this class; the catchers live in
       ``tests/numerics/`` (:math:`\Pi R = 4\pi I`, not :math:`I`).
   * - 5 — the :math:`\alpha` recursion
     - One spelling, derived in
       :doc:`/theory/methods/sn/curvilinear_one_group`.
     - **Four spellings in three texts**, none acknowledging
       another exists — Stacey (9.213), Hébert sphere (3.424),
       Hébert cylinder (3.399, sign-flipped against its own sphere
       four pages earlier), Bell & Glasstone (5.21, with
       :math:`\Delta A` folded inside). The full crosswalk table is
       tabulated at :ref:`normalization-alpha-crosswalk`.
   * - 6 — the operator letters
     - :math:`L` = bulk streaming **only**;
       :math:`A = L+C-S-N_{2n}-B` is the honest within-group operator.
     - Adams & Larsen :cite:`AdamsLarsen2002` define
       :math:`A \equiv I - L^{-1}S` (their Eq. (1.27)) — a
       **sweep-preconditioned fixed-point map** (ORPHEUS's *Krylov
       system* operator, not its :math:`A`); their eigenvalue
       chapter's :math:`L` in :math:`A = L^{-1}P` (Eq. (8.5)) is
       the **full loss operator** — ORPHEUS's :math:`A` with the
       boundary folded into BCs — and their :math:`P` is ORPHEUS's
       :math:`F`. Same letters, three meanings.
   * - 7 — same-symbol collisions
     - :math:`\alpha` = redistribution; :math:`\beta` = closure
       parametrization; :math:`\tau` = closure weight
       (S\ :sub:`N`) vs optical path (CP / MoC); :math:`\mu` = a
       **cosine**; Fourier :math:`\lambda` carries length units.
     - The Larsen–Morel review :cite:`LarsenMorel2010` writes the sphere
       redistribution coefficient as :math:`\beta` (their
       Eq. (1.23b) — identical to :cite:`BaileyMorelChang2010`'s
       :math:`\alpha` with the sign absorbed) and uses
       :math:`\alpha` for the **spatial** weighted-diamond weight
       (their Eq. (1.30)). Adams & Larsen's 1-D cylinder
       :math:`\mu` is an **angle** ("the radial-plane projection of
       the direction of particle flight"), not a cosine; and their
       Fourier wave number :math:`\lambda` is dimensionless in
       **mean-free-path units** (the phase is
       :math:`\Sigma_{\mathrm{t}}\lambda x`).
   * - 8 — ORPHEUS-internal bindings
     - The numerics iteration layer poses
       :math:`(A - \sum_i g_i)\,\psi = q_{\mathrm{ext}}` with
       :math:`A` the **invertible resolvent operand**;
       :class:`~orpheus.numerics.iteration.SourceIteration` receives
       it pre-inverted (``A_inv``, then the variadic ``*gains``).
     - Not a canon row — the one place ORPHEUS itself carries two
       bindings of :math:`A`, by design. The S\ :sub:`N` binding
       hands :math:`A = L + C` with gains :math:`(S, N_{2n}, B)`,
       composing to the same honest :math:`L+C-S-N_{2n}-B`; the bridge
       is stated at
       both ends (the module heads of
       :mod:`orpheus.numerics.iteration` and
       :mod:`orpheus.numerics.operator`, and the solver page).
       Fission is **never** a gain in the eigenvalue posing — it
       stays on the right-hand side under :math:`1/k`.

       **The bridge, both ends (2026-07-28).** The solver-layer record
       :class:`~orpheus.sn.coupled_system.WithinGroupSystem` names the
       splitting by its *role*:
       :attr:`~orpheus.sn.coupled_system.WithinGroupSystem.implicit_operator`
       is :math:`M` (solved implicitly — inverted — each step) and
       :attr:`~orpheus.sn.coupled_system.WithinGroupSystem.explicit_gains`
       is :math:`N` (evaluated explicitly from the lagged iterate). The
       numerics layer keeps its own vocabulary — ``A`` / ``*gains`` on
       :class:`~orpheus.numerics.iteration.SourceIteration` and
       :class:`~orpheus.numerics.iteration.KrylovAcceleration` —
       deliberately, because that layer's ``A`` is the *resolvent
       operand* of the row above. So the crosswalk is
       ``implicit_operator`` :math:`\leftrightarrow` ``A`` and
       ``explicit_gains`` :math:`\leftrightarrow` ``*gains``.

       The record's field was called ``resolvent`` until 2026-07-28.
       That was a **misnomer**: it holds :math:`M`, the *un-inverted
       forward* operator, whereas a resolvent is inverse-like. The word
       is now reserved for its two honest uses — the corpus
       :math:`K_{\rm pm} = A_{\rm loss}^{-1}M` of
       :ref:`eigenvalue-posing`, and the future
       ``A.resolvent(z) = (A - zI).inverse()`` factory.
   * - 9 — the term *multiplication operator*
     - Two senses, kept apart by context and by label namespace:
       :math:`K = A^{-1}F` is the reactor-physics **multiplication
       operator** (the symbol table above — its dominant eigenvalue
       is :math:`k`); :math:`M[f]\psi = f\,\psi` is the
       functional-analysis **multiplier embedding** behind the
       collision operator :math:`C = M[\Sigma_{\mathrm{t}}]`
       (:doc:`/theory/foundations/operator_algebra`).
     - Not a canon row — a domain-inherent collision: both
       vocabularies own the term natively and the corpus
       deliberately spans both, so neither side is renamed.
       Equation labels keep the namespaces disjoint:
       ``mg-multiplication-operator`` (the :math:`K` eigenvalue
       posing) vs ``multiplication-operator-embedding`` /
       ``-action`` (the :math:`M[f]` algebra).

.. _notation-import-boundary:

The import boundary
===================

Every equation imported from the literature crosses into ORPHEUS
through three questions, answered **at the import site** (the
docstring or theory-page passage that quotes the source), before any
symbol is transcribed:

1. **Which way does the scattering arrow point?** Destination-first
   sources (Hébert) enter transposed; source-first (Bell & Glasstone,
   Stacey) enter index-for-index (crosswalk row 1).
2. **What does** :math:`\Sigma w` **equal on the source's page?** Not
   in the source's chapter — on its *page*; the canon switches
   normalization mid-chapter without notice (row 3).
3. **Where does the source put** :math:`(2\ell+1)` **and**
   :math:`4\pi`\ **?** Prefactor placement is convention, not physics
   (row 4).

ERR-025 (a missing :math:`1/W`, masked for homogeneous problems by a
compensating factor error) and ERR-039 (the harmonic prefactor) are
the recorded cost of skipping these questions; both catchers are
pinned in the test suite. The per-method machine headers (each
method's index page opens with its conventions block — sign,
scattering, quadrature norm, layout) restate the local answers so a
reader entering through any method page hits the conventions first.
