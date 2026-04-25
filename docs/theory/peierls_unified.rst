.. _theory-peierls-unified:

======================================================================
Unified polar-form Peierls Nyström architecture (slab · cyl · sph)
======================================================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying any Peierls Nyström reference solver,
or before extending the architecture to a new geometry.**
For "what references do we ship for problem X?" see
:ref:`theory-peierls-capabilities` (the capability matrix is the
index). For terminology that has historical collisions ("F.4",
``boundary`` strings, ``n_bc_modes`` vs ``n_surfaces``) see
:ref:`theory-peierls-naming`. For the **active slab-polar adaptive
mpmath.quad path** (and the two retired predecessors — τ-Laguerre
and moment-form), see :ref:`theory-peierls-slab-polar`. For the
**multi-group extension** (Issue #104, 2026-04-24), which added
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg` and
reduced :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
to a thin wrapper, see :ref:`theory-peierls-multigroup`. For the
**rank-:math:`N` falsifications on both topology classes** —
Class A (hollow) closed by L21 and Class B (solid) under Issue #132
re-derivation — see :ref:`peierls-rank-n-per-face-closeout` and
:ref:`peierls-rank-n-class-b-mr-mg-falsification`.

- **Primary organizing principle (2026-04-23): topology, not shape.**
  The cases this module ships partition into two **topological
  classes**, each with a distinct set of applicable closures:

  - **Class A — two-surface (F.4 applies).** Members: slab
    (two parallel faces), hollow annular cylinder (inner + outer
    ring), hollow sphere (inner + outer shell). Shared closure
    class: F.4 scalar rank-2 per-face (Stamm'ler Eq. 34 =
    Hébert 2009 Eq. 3.323 = :math:numref:`hebert-3-323`).
  - **Class B — one-surface compact (rank-1 Mark only).** Members:
    solid cylinder, solid sphere. F.4 structurally collapses here
    (no second face to couple to); only rank-1 Mark is shipped.
    Rank-:math:`N` Marshak (``n_bc_modes ≥ 2``) is callable but
    **unsafe in MR** — :ref:`peierls-rank-n-class-b-mr-mg-falsification`
    (Issue #132) documents the +57 % sphere 1G/2R sign-flip
    catastrophe and the mode-0 / mode-:math:`n \ge 1` normalisation
    mismatch in :func:`build_closure_operator`.

  The ``CurvilinearGeometry.topology`` property returns
  ``"two_surface"`` or ``"one_surface_compact"`` and is the canonical
  runtime discriminator. This supersedes shape-keyed dispatch
  (``kind="cylinder-1d"``) as the primary axis of organization —
  shape is a sub-axis *within* a topology class.

- All three 1-D Peierls integral equations (slab, cylinder, sphere)
  are *instances of one equation*
  :eq:`peierls-unified`, written in
  **observer-centred polar coordinates**. Only the angular measure
  :math:`\mathrm d\Omega_d`, the pre-integrated point kernel
  :math:`\kappa_d`, and the scalar ray-boundary distance
  :math:`\rho_{\max}` change between geometries.
- The standard textbook forms (:math:`E_1` slab kernel,
  :math:`\mathrm{Ki}_1` cylinder kernel, bare :math:`e^{-\tau}` sphere
  kernel) differ only by which symmetry directions have been
  integrated out of the **same** 3-D point kernel
  :math:`e^{-\Sigma_t R}/(4\pi R^{2})`
  (:eq:`peierls-point-kernel-3d`).
- In the polar form the volume element :math:`\rho^{d-1}\,\mathrm
  d\Omega\,\mathrm d\rho` exactly cancels the :math:`1/|r-r'|^{d-1}`
  of the pre-integrated kernel (:eq:`peierls-polar-jacobian-cancellation`).
  For curvilinear geometries (:math:`d=2,3`) the integrand becomes
  smooth; the slab (:math:`d=1`) retains a mild :math:`-\ln\rho`
  singularity because there is no :math:`\rho^{d-1}` factor to cancel.
- The resulting Nyström operator has an identical scaffolding for
  all three geometries: angular quadrature, per-ray ρ-quadrature,
  optical-depth walker, Lagrange interpolation of the source on a
  composite-panel radial grid, kernel evaluation, assembly.
- Only four functions are geometry-specific — the angular measure,
  the kernel :math:`\kappa_d`, the ray-boundary distance
  :math:`\rho_{\max}(r,\Omega)`, and the source position
  :math:`r'(r,\rho,\Omega)`. Everything else (optical-depth walker,
  Lagrange basis, composite GL, power iteration, and the *rank-1*
  white-BC closure) is shared.
- **Gotcha**: the cylinder and sphere share **identical**
  :math:`\rho_{\max}` and :math:`r'` closed forms. The only
  differences are the kernel (:math:`\mathrm{Ki}_1` vs :math:`e^{-\tau}`)
  and the angular measure (:math:`\mathrm d\beta` vs
  :math:`\sin\theta\,\mathrm d\theta`).
- **Gotcha**: The white-BC *rank-1* closure is exact at the
  flat-source / region-averaged level (as used by
  :mod:`orpheus.cp.solver`), but is only an *approximation* at the
  pointwise Nyström level. The error is bounded by the rank-1
  scaling of ``build_white_bc_correction`` in
  :mod:`orpheus.derivations.peierls_cylinder`; it is the same
  phenomenon reported for the sphere in GitHub Issue #100.
- **Phase F (Issue #110, 2026-04-21).** ``CurvilinearGeometry``
  carries an ``inner_radius`` field (hollow-core support), a
  per-face ``n_surfaces`` property (2 for slab and hollow cyl/sph,
  1 for solid), and per-surface escape/response primitives
  :func:`~orpheus.derivations.peierls_geometry.compute_P_esc_outer`
  /
  :func:`~orpheus.derivations.peierls_geometry.compute_P_esc_inner`
  /
  :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_outer`
  /
  :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_inner`.
  :func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
  with ``reflection="white"`` builds a **rank-2 per-face** closure
  on Class-A geometries (slab today; hollow cyl/sph land in F.4)
  with :math:`R = (I - W)^{-1}` from the surface-to-surface
  transmission
  :func:`~orpheus.derivations.peierls_geometry.compute_slab_transmission`.
  Rank-2 white closes the Wigner-Seitz identity
  :math:`k_{\rm eff} = k_\infty` up to the outer-GL quadrature
  order (O(h²) due to the :math:`E_2` endpoint log singularity)
  — a 10³-10⁴× improvement over rank-1 Mark at L ≲ 1 MFP. The
  tensor form
  :math:`K_{\rm bc} = G \cdot R \cdot P` reproduces the legacy
  :mod:`peierls_slab` :math:`E_2` / :math:`E_3` bilinear form
  bit-exactly (rtol 1e-13) — see
  ``tests/derivations/test_peierls_rank2_bc.py``. Solid cyl/sph
  with ``reflection="white"`` collapse to the rank-1 Mark form
  (:math:`R = (I - 0)^{-1} = 1`), preserving bit-exact regression.
  **Phase F.4 (hollow cylinder).** Extended rank-2 white to
  hollow cylindrical annuli via
  :func:`~orpheus.derivations.peierls_geometry.compute_hollow_cyl_transmission`
  — Lambert-emission chord decomposition with the out-of-plane
  :math:`\theta` fold into the :math:`\mathrm{Ki}_3` Bickley
  function: :math:`W_{\rm oo} = (4/\pi)\!\int_{\alpha_c}^{\pi/2}
  \!\cos\alpha\,\mathrm{Ki}_3(2\Sigma_t R\cos\alpha)\,\mathrm d\alpha`
  (grazing past cavity),
  :math:`W_{\rm io} = (4/\pi)\!\int_0^{\alpha_c}\!\cos\alpha\,
  \mathrm{Ki}_3(\Sigma_t\ell_{\rm annulus}(\alpha))\,\mathrm d\alpha`
  (hitting inner shell), reciprocity
  :math:`W_{\rm oi} = (R/r_0)\,W_{\rm io}`, and convex-cavity
  constraint :math:`W_{\rm ii} = 0`. **The rank-1-per-face
  (scalar) closure does NOT close the Wigner-Seitz identity
  exactly on curved hollow surfaces** — unlike slab, where every
  position on a face maps to the same angular geometry, the
  cylinder's inward J⁻ at :math:`r_0` carries non-trivial
  angular moments that the scalar mode omits. For
  :math:`r_0 = 0.1 R` the rank-2 residual is 1.4 % (vs rank-1
  Mark's 25 %); for :math:`r_0 = 0.3 R` it is 13 %. The numerical
  residual does **not** vanish under plain Legendre rank-N
  refinement — see :ref:`Phase F.5 close-out
  <peierls-rank-n-per-face-closeout>` for the five-reference
  synthesis, the :math:`c_{\rm in}` structural obstruction, the
  falsification of a novel per-mode Villarino-Stamm'ler extension,
  and the production decision to keep F.4 rank-2 scalar as the
  closure.
  **Hollow sphere** uses the same chord decomposition
  (:func:`~orpheus.derivations.peierls_geometry.compute_hollow_sph_transmission`)
  with the bare :math:`e^{-\tau}` kernel (no Bickley fold) and
  reciprocity :math:`W_{\rm oi} = (R/r_0)^2\,W_{\rm io}` on
  spherical surface areas. The sphere's higher symmetry yields
  smaller scalar-mode residuals than cylinder:
  :math:`r_0 = 0.1 R` → 0.4 % (70× over rank-1 Mark),
  :math:`r_0 = 0.2 R` → 1.2 % (24×),
  :math:`r_0 = 0.3 R` → 3.3 % (10×).
- **Historical note.** Standard references
  ([Sanchez1982]_, [Hebert2020]_, [Stamm1983]_, [BellGlasstone1970]_,
  [CaseZweifel1967]_) present each geometry in **chord coordinates**
  :math:`(y, r')` because those coordinates admit closed-form flat-source
  annular integrals that collapse to the :math:`\mathrm{Ki}_n` /
  :math:`E_n` second-difference formulae used by flat-source CP. The
  chord form is not wrong — it is *specialised* to flat-source region
  averaging. For pointwise Nyström with general polynomial sources the
  observer-centred polar form is the cleaner choice, and this page is
  the written record of that choice.
- **Production status (2026-04-20).** The slab K matrix is now
  assembled by the **unified verification primitive**
  :func:`~orpheus.derivations.peierls_geometry.K_vol_element_adaptive`
  — adaptive ``mpmath.quad`` over the polar form, polymorphic across
  all three geometries via :class:`CurvilinearGeometry`. This is the
  single ground-truth K-element computation; production-tier
  curvilinear paths (cylinder-1d, sphere-1d) verify against it at
  spectral-but-finite-rate convergence. The slab moment-form Nyström
  architecture (closed-form polynomial moments via
  Lewis–Miller / Hébert / Stamm'ler integration-by-parts) was archived
  on 2026-04-19 to
  :file:`derivations/archive/peierls_slab_moments_assembly.py`
  and tracked under
  `GitHub Issue #117 <https://github.com/deOliveira-R/ORPHEUS/issues/117>`_
  for future application to a higher-order production discrete CP
  solver. The cylinder-polar variant (explicit out-of-plane
  :math:`\varphi`-quadrature) was likewise archived to
  :file:`derivations/archive/peierls_cylinder_polar_assembly.py` —
  cylinder-1d (with :math:`\mathrm{Ki}_1` evaluated directly) is the
  natural-kernel form and the active production path.
- **Vacuum-BC verification milestone (2026-04-20).** Machine-precision
  (rel tol :math:`10^{-10}`) analytical references for the uniform-source
  vacuum-BC flux exist for all three geometries:
  :func:`~orpheus.derivations.peierls_reference.slab_uniform_source_analytical`
  (closed-form :math:`E_2` differences),
  :func:`~orpheus.derivations.peierls_reference.cylinder_uniform_source_analytical`
  (one ``mpmath.quad`` over in-plane azimuth with :math:`\mathrm{Ki}_2`
  absorbing the polar integral), and
  :func:`~orpheus.derivations.peierls_reference.sphere_uniform_source_analytical`
  (one ``mpmath.quad`` over :math:`\mu = \cos\Theta`). The row-sum
  identity :math:`\sum_j K_{ij} \cdot 1 = \Sigma_t\,\varphi_d(r_i)` is
  gated against these references by
  ``TestSlabKernelRowSum``, ``TestCylinderKernelRowSum``, and
  ``TestSphereKernelRowSum`` in :mod:`tests.derivations.test_peierls_reference`.
  Full derivations live at :ref:`peierls-vacuum-bc-analytical-references`.
  Because vacuum BC is :math:`R = 0` in the boundary-closure tensor
  network :math:`K_{\rm bc} = G\cdot R\cdot P` (see
  :class:`~orpheus.derivations.peierls_geometry.BoundaryClosureOperator`),
  this closes the volume-kernel verification at the flux level; the BC
  tensor-network expansion (Mark / Marshak DP_N / albedo) proceeds on a
  verified foundation.


.. _theory-peierls-capabilities:

Capabilities at a glance — what references this module ships
=============================================================

**For readers trying to answer "what continuous references do we have
for geometry X at closure Y?"** this section is the canonical index.
The table below enumerates every shipped Peierls continuous reference
case with its production status, accuracy class, and the test label
that gates regressions. The table body is auto-generated at Sphinx
build time by
:mod:`tools.verification.generate_peierls_matrix` from the registry
function
:func:`orpheus.derivations.peierls_cases.capability_rows` — if this
table diverges from ``continuous_all()`` filtered to
``operator_form == "integral-peierls"``, the capability-matrix
cross-check test will fail.

.. include:: _peierls_capability_matrix.inc.rst

All rows carry ``vv_level = "L1"``, ``equation_labels`` include
``peierls-unified`` and — for F.4 cases —
:math:numref:`hebert-3-323`. Regressions land in
``tests/derivations/test_peierls_rank2_bc.py`` (F.4 cases) and
``tests/derivations/test_peierls_reference.py`` (slab rank-2).

Class A — two-surface (F.4 applies)
------------------------------------

Members: slab, hollow annular cylinder, hollow sphere. Shared closure
class (F.4 scalar rank-2 per-face, Stamm'ler Eq. 34 =
:math:numref:`hebert-3-323`) and shared L19 stability-protocol
coverage. Implementation of F.4 lives in
:func:`~orpheus.derivations.peierls_geometry._build_closure_operator_rank2_white`
for cylinder and sphere, and in
:mod:`~orpheus.derivations.peierls_slab` (native E₁ Nyström) for
slab. As of 2026-04-23 slab **also** has a unified-path reference via
:data:`~orpheus.derivations.peierls_geometry.SLAB_POLAR_1D` and
:func:`~orpheus.derivations.peierls_geometry.K_vol_element_adaptive`
(adaptive tanh-sinh :func:`mpmath.quad` with a forced :math:`\mu = 0`
breakpoint — see :ref:`theory-peierls-slab-polar`). The two
implementations are independent cross-checks: they agree at machine
precision on the K matrix but follow different numerical routes
(polar-form adaptive quadrature vs classical :math:`E_1` Nyström with
singularity subtraction + product integration). The **closure class**
is the same across all Class A members, which is what makes Class A
a coherent group.

.. list-table:: Class A — closures shipped per shape
   :header-rows: 1
   :widths: 34 22 22 22

   * - Closure
     - Slab
     - Hollow cylinder
     - Hollow sphere
   * - ``vacuum``
     - ✅ E₁ Nyström (native) + unified adaptive (see :ref:`theory-peierls-slab-polar`)
     - ✅ Ki₁ via unified
     - ✅ exp(-τ) via unified
   * - ``white_rank1_mark`` (dep. alias: ``white``)
     - ✅ scalar collapse
     - ✅ production
     - ✅ production
   * - ``white_f4`` (dep. alias: ``white_rank2``)
     - ✅ E₂/E₃ bilinear — Wigner-Seitz exact
     - ✅ Ki₃ fold, R/r_0 reciprocity
     - ✅ bare exp(-τ), (R/r_0)² reciprocity
   * - White rank-N Marshak per-face
     - n/a
     - 🚫 ``NotImplementedError`` per L21; primitives retained (load-bearing for rank-1 path + conservation tests)
     - 🚫 same as cylinder

Class B — one-surface compact (rank-1 Mark only)
-------------------------------------------------

Members: solid cylinder, solid sphere (both ``inner_radius == 0``).
F.4 structurally collapses (no second face to couple to); the only
shipped closure is rank-1 Mark. Requesting ``closure="white_f4"`` on
a Class B geometry emits a ``DeprecationWarning`` and silently
collapses to rank-1 Mark (future release: ``ValueError``). Class B
has **zero** shipped continuous references in the registry today —
the rank-1 Mark floor (21 % err at R=1 MFP for cylinder per Issue
#103) is too loose to serve as an L1 reference for the ``cp_cyl1D``
/ ``cp_sph1D`` solver tests. Lifting the floor is the scope of
Issue #103 (rank-N DP_N on the single outer face, subject to the
L19 stability protocol) or Issue #101 (chord-based Ki₁ analytical
Nyström).

.. list-table:: Class B — closures shipped per shape
   :header-rows: 1
   :widths: 40 30 30

   * - Closure
     - Solid cylinder
     - Solid sphere
   * - ``vacuum``
     - ✅ Ki₁ via unified
     - ✅ exp(-τ) via unified
   * - ``white_rank1_mark`` (dep. alias: ``white``)
     - ✅ production
     - ✅ production
   * - ``white_f4`` (dep. alias: ``white_rank2``)
     - ⚠️ silent collapse + ``DeprecationWarning`` (no second face)
     - ⚠️ silent collapse + ``DeprecationWarning``
   * - White rank-N DP\ :sub:`N` on outer face
     - 🚫 Issue #103 open; reachable but unsafe in MR (Issue #132, see :ref:`peierls-rank-n-class-b-mr-mg-falsification`)
     - 🚫 Issue #100 / #103 open; reachable but unsafe in MR (Issue #132)
   * - Periodic / albedo / specular
     - 🚫 not shipped
     - 🚫 not shipped

The "✅ production" mark means *reachable through the shipped public
API* (``solve_peierls_*_1g`` with the topology-appropriate parameters),
*tested under the L19 stability protocol where applicable*, and
*documented in this page*. "🚫" means the path raises, is absent,
or is known to be too loose to ship as a reference. "⚠️" means the
call succeeds but with a documented collapse behavior.

**What the table does NOT cover** (separate tables / pages):
``VerificationCase`` registry (discrete solver test cases) —
see :doc:`/api/derivations`. Discrete CP, MOC, MC, SN, and
diffusion test cases live there with their own naming conventions.
Peierls continuous references bridge those two worlds for L1
flux / k_eff verification.

Known infrastructure gaps:

- **Multi-group parity benchmarks** vs the discrete ``cp_cylinder`` /
  ``cp_sphere`` native 2G solvers. Issue #104 commit 2
  (2026-04-24) added the 6 2G hollow cyl/sph reference cases above
  and a Phase G.5 tie-back test against the native slab MG driver,
  but the parity gate vs ``cp_cylinder`` / ``cp_sphere`` (1 %
  target per Issue #104 AC) is deferred to a follow-up session.
- **Multi-region Peierls references for cylinder and sphere** (the
  ``cp_{cyl,sph}1D_{2,4}rg`` solver cases have no matching
  continuous reference). Requires either F.4-per-internal-interface
  or a different closure class.
- **Solid cylinder / sphere rank-N DP\ :sub:`N` reference** at
  thin R (Issue #103). Blocks registration of solid-geometry
  ``peierls_{cyl,sph}1D_NeG_MrG`` matching the CP solver tests.


.. _theory-peierls-naming:

Terminology glossary — unambiguous names, despite the history
=============================================================

Several names in this module have historical collisions. A fresh
reader must disambiguate them before reading any dispatch code.

**Topology** is the primary organizing principle (see Key Facts
above). Two classes:

- ``"two_surface"``: slab, hollow cylinder, hollow sphere. F.4 applies.
- ``"one_surface_compact"``: solid cylinder, solid sphere. Only
  rank-1 Mark shipped.

Queryable at runtime as
:attr:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.topology`.
For slab (which does not use :class:`CurvilinearGeometry`) the
module-level constant ``orpheus.derivations.peierls_slab.TOPOLOGY``
carries the same label. This property **supersedes**
:attr:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.n_surfaces`
as the user-facing identifier: ``n_surfaces`` remains an internal
integer count, ``topology`` is the semantic label for dispatch +
documentation + test filtering. The plan that introduced this
concept lives at
:file:`.claude/plans/topology-based-consolidation.md`.

**"F.4"** is **overloaded**. Two distinct referents:

- **Phase F.4** (development subphase) — the 2026-04-21 Phase F
  rollout subphase that added hollow-cylinder support to
  :func:`~orpheus.derivations.peierls_geometry.compute_hollow_cyl_transmission`
  and the cylinder branch of
  :func:`~orpheus.derivations.peierls_geometry._build_closure_operator_rank2_white`.
  Historical label only; not part of the runtime API.
- **F.4 (closure formula)** — Stamm'ler & Abbate 1983 Ch. IV
  Eq. 34 = Hébert 2009 §3.8.4 Eq. 3.323 (see
  :math:numref:`hebert-3-323`), the scalar rank-2 per-face
  white-BC closure. **This** is the production closure, the thing
  the L21 research program confirmed is optimal within the rank-N
  white-BC paradigm on 1D curvilinear hollow cells.

When reading session notes, research logs, or commit messages,
interpret "F.4" by context: development timeline → Phase F.4;
closure math → Stamm'ler Eq. 34.

**Scattering matrix convention (sig_s).** ``sig_s[r, g_src, g_dst]``
with first index = **source** group, second = **destination**. See
:ref:`peierls-scattering-convention` in the multi-group driver
section for the authoritative statement and its cross-check against
the physical fixture in :mod:`orpheus.derivations._xs_library`.

**Boundary string → closure semantics**. The ``boundary=`` (a.k.a.
``closure=``) argument to :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
accepts three values today, each meaning a specific closure:

.. list-table:: ``boundary`` / ``closure`` string semantics (Stage 5 landed)
   :header-rows: 1
   :widths: 22 28 22 28

   * - Canonical name
     - Deprecated alias
     - Routes to
     - Physical meaning
   * - ``"vacuum"``
     - —
     - no BC correction added
     - Zero re-entering flux; outgoing rays escape.
   * - ``"white_rank1_mark"``
     - ``"white"`` (emits ``DeprecationWarning``)
     - :func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction_rank_n`
     - Rank-1 Mark (isotropic) re-emission. Accurate for flat-source
       region-averaged CP (exact at the limit) but loose at pointwise
       Nyström for curvilinear geometries.
   * - ``"white_f4"``
     - ``"white_rank2"`` (emits ``DeprecationWarning``)
     - :func:`~orpheus.derivations.peierls_geometry._build_closure_operator_rank2_white`
     - F.4 scalar rank-2 per-face. Distinct scalar moment per face,
       coupled via the :math:`(I - W)^{-1}` transmission operator.
       **On solid geometry (``n_surfaces == 1``) emits a
       ``DeprecationWarning`` and silently collapses to rank-1 Mark**
       — the collapse will become a ``ValueError`` in a future
       release. Use ``closure="white_rank1_mark"`` for solid.

Both the deprecated aliases and the new canonical names are accepted
by :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
(and the per-geometry thin wrappers). Existing code continues to
work unchanged; the canonical names are preferred for new callers.

**Guard terminology**: the ``NotImplementedError`` raised by
:func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
when ``reflection="white"`` AND ``n_bc_modes > 1`` AND
``geometry.n_surfaces == 2`` guards **rank-N Marshak per-face**
(Phase F.5's attempted and falsified enrichment), NOT F.4. F.4
itself always uses ``n_bc_modes = 1``. The guard is load-bearing
per research-log L21; removing it is the Stage 5 archive action,
not a bug fix.

**``n_bc_modes`` vs ``n_surfaces``**:

- ``n_surfaces`` is a geometry property (1 for solid slab, cylinder,
  sphere; 2 for hollow cylinder / sphere / slab-polar with two face
  constants). Determined by
  :attr:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.n_surfaces`.
- ``n_bc_modes`` is a closure parameter specifying the number of
  angular moments retained per face. ``n_bc_modes = 1`` = scalar per
  face (F.4's formulation and the only shipped mode). Higher
  ``n_bc_modes`` is currently guarded per L21.

**"Hollow"** means ``inner_radius > 0`` on ``cylinder-1d`` or
``sphere-1d``. A hollow cell has 2 surfaces (inner + outer); a
solid cell has 1 surface (outer only). This is the same as
``n_surfaces`` semantically but is more user-facing. The shipped
1-region solver cases (``cp_cyl1D_1eg_1rg`` etc.) are **solid**;
the shipped hollow Peierls continuous references are **1-region
shell-only**. Multi-region hollow cells are not yet shipped.


.. _theory-peierls-slab-polar:

Slab-polar in the unified framework — the active adaptive-mpmath path
======================================================================

**This section documents the slab path that is active in the shipping
code.** It supersedes two retired predecessors (archived below):

- The **τ-Laguerre prototype** (commit 4395cb8, 2026-04-07). Retired
  because Gauss–Laguerre on the substitution :math:`v = -\ln|\mu|`
  converges only algebraically on an integrand that is polynomial in
  :math:`e^{-v}` rather than polynomial in :math:`v`. See
  :ref:`theory-peierls-moment-form-failed-polar` for the full defect
  analysis and the diagnostic tables.
- The **moment-form fast-assembly** (2026-04-19). Archived on
  2026-04-20 per
  `Issue #117 <https://github.com/deOliveira-R/ORPHEUS/issues/117>`_
  because the verification side of the CP module does not need a fast
  slab-K assembly. The math is preserved in
  :ref:`theory-peierls-moment-form` for a future production discrete
  CP solver; it is not on the verification-reference hot path.

The active path — implemented in
:func:`~orpheus.derivations.peierls_geometry.K_vol_element_adaptive`
and dispatched via
:func:`~orpheus.derivations.peierls_geometry.build_volume_kernel` when
``geometry.kind == "slab-polar"`` — computes each :math:`K[i, j]`
element with **two nested adaptive** :func:`mpmath.quad` **calls**
sharing breakpoint hints with the cylinder and sphere paths. The
result is machine-precision :math:`K` uniformly across all three
geometries, paid for by a higher cost-per-element than a tuned
fast-assembly would give.


Subsection — The slab polar-form equation (recap)
--------------------------------------------------

Observer at :math:`x = x_i`, direction cosine :math:`\mu \in [-1, 1]`,
ray length :math:`\rho \in [0, \rho_{\max}(x_i, \mu)]`. The unified
Peierls equation :eq:`peierls-unified` specialises to

.. math::
   :label: peierls-slab-polar

   \Sigma_t(x_i)\,\varphi(x_i)
     \;=\;\frac{1}{2}
     \int_{-1}^{1}\!\mathrm d\mu
     \int_{0}^{\rho_{\max}(x_i,\mu)}
       e^{-\int_0^\rho \Sigma_t(x_i + s\mu)\,\mathrm ds}\,
       q\bigl(x_i + \rho\mu\bigr)\,\mathrm d\rho,

with the chord range

.. math::

   \rho_{\max}(x_i, \mu) \;=\;
     \begin{cases}
       (L - x_i)/\mu     & \mu > 0, \\[2pt]
       -x_i/\mu = x_i/|\mu| & \mu < 0, \\[2pt]
       \infty            & \mu = 0 \text{ (rays parallel to faces)}.
     \end{cases}

The prefactor :math:`1/2` is the slab's
:attr:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.prefactor`
value (already folded in the code path). The source
:math:`q(x') = \Sigma_s(x')\,\varphi(x') + \chi\,\nu\Sigma_f(x')\,\varphi(x')/k`
follows the unified scattering + fission convention.

This is **the same form** that
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g` solves
for cylinder-1d and sphere-1d. The only per-geometry differences are
already factored behind the :class:`CurvilinearGeometry` primitives
(``ray_direction_cosine``, ``rho_max``, ``source_position``,
``volume_kernel_mp``, ``angular_weight``). Slab inherits the
*architecture* for free.


Subsection — Why the outer :math:`\mu`-integral is stiff at :math:`\mu = 0`
----------------------------------------------------------------------------

For non-grazing :math:`\mu`, the inner integrand along the ray decays
as :math:`e^{-\Sigma_t\,\rho}`: essentially all the contribution comes
from the first few mean free paths, and the integrand is smooth in
both :math:`\rho` and :math:`\mu`.

Near :math:`\mu = 0` the ray becomes nearly parallel to the faces:
:math:`\rho_{\max}(x_i, \mu) \sim 1/|\mu| \to \infty`, and the
exponential decay along the ray measures optical depths proportional
to :math:`\Sigma_t \cdot L / |\mu|`. Concretely, substituting
:math:`v = -\ln|\mu|`, the outer integrand acquires a factor

.. math::

   \exp\!\bigl(-\Sigma_t\,L\,e^{v}\bigr),

which is **super-exponentially** suppressed in :math:`v`. This is the
*grazing-ray stiffness*: a smooth envelope everywhere except a thin
region near :math:`\mu = 0` where the integrand collapses to zero on
a scale set by the material.

**Why ordinary Gauss–Laguerre fails.** Gauss–Laguerre on :math:`v`
is polynomial-exact against the weight :math:`e^{-v}`. Our integrand
is polynomial-in-:math:`e^{-v}` (equivalent to polynomial-in-:math:`\mu`),
not polynomial-in-:math:`v`, so Gauss–Laguerre converges only
algebraically. The full defect — including generalised-Laguerre
:math:`\alpha`-sweep tables showing no Laguerre flavour helps — is
:ref:`theory-peierls-moment-form-failed-polar` below. **Do not
re-attempt the exp-stretched substitution** on the active path; the
failure is structural, not a tuning issue.


Subsection — Strategy: adaptive tanh-sinh with a forced breakpoint at :math:`\mu = 0`
--------------------------------------------------------------------------------------

The active path integrates :math:`\mu` on its native range
:math:`[-1, 1]` with :func:`mpmath.quad` and passes a forced
breakpoint at :math:`\mu = 0`:

.. code-block:: python

   outer_breaks = [mpmath.mpf(-1), mpmath.mpf(0), mpmath.mpf(1)]
   omega_integral = mpmath.quad(outer_integrand, outer_breaks)

:func:`mpmath.quad` uses tanh-sinh (double-exponential) quadrature by
default and subdivides recursively from each breakpoint outward until
the estimated error falls below the working precision. Two properties
make this well-suited to the grazing-ray problem:

1. **Tanh-sinh is endpoint-robust.** The nodes pile up at the
   breakpoints, which is exactly where the stiffness lives (:math:`\mu
   \to 0^{\pm}`). No weight pre-matching is required — the rule
   handles :math:`e^{-c\,e^{v}}`-style super-exp decay by pure node
   concentration, not by being polynomial-exact against it.
2. **The µ = 0 breakpoint splits the outer integrand into two
   branches**, each smooth all the way to the endpoint. Without the
   breakpoint the integrand has a weak (non-:math:`C^\infty`)
   transition at :math:`\mu = 0` where the ray direction flips
   (:math:`\rho_{\max}` switches branches). Recursive subdivision
   from the breakpoint keeps both branches' tanh-sinh estimates
   monotonically accurate in the node count.

The inner :math:`\rho`-integral uses the same adaptive
:func:`mpmath.quad` with breakpoints inserted at panel-boundary
crossings of :math:`r'(\rho)` along the ray (the Lagrange basis has
kinks there). Panel-boundary-crossing subdivision is shared with the
cylinder and sphere paths (Issue #114); the only slab-specific
subdivision is the :math:`\mu = 0` split.


Subsection — What this path costs and where it is called
---------------------------------------------------------

**Cost model.** Each :math:`K[i, j]` element costs one outer adaptive
:math:`\mu` integral over two breakpoint-segments, each of which
contains one inner adaptive :math:`\rho` integral. Assembly of the
:math:`N \times N` matrix therefore scales as

.. math::

   C_{\rm assembly}
     \;=\; N^2 \cdot \bigl\langle n_\mu \bigr\rangle
           \cdot \bigl\langle n_\rho \bigr\rangle
           \cdot c_{\rm eval},

with :math:`\langle n_\mu \rangle, \langle n_\rho \rangle` the
adaptive node counts and :math:`c_{\rm eval}` the cost of evaluating
the inner integrand (kernel + Lagrange basis). On a vacuum slab with
:math:`N = 8` (two panels, :math:`p = 4`) at ``dps = 25`` a
representative wall-time is minutes, not seconds. **This is the
verification reference**, not a production K-assembly.

**Where it is called.** Every path that requests a slab-polar K
matrix eventually hits
:func:`~orpheus.derivations.peierls_geometry.build_volume_kernel`,
which dispatches to
:func:`~orpheus.derivations.peierls_geometry.build_volume_kernel_adaptive`
for ``kind == "slab-polar"``. This includes
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g` when
called on :data:`SLAB_POLAR_1D`. The legacy
:func:`orpheus.derivations.peierls_slab.solve_peierls_eigenvalue`
retains its **classical** :math:`E_1` Nyström (singularity-subtraction
+ product-integration weights) as an independent cross-check
implementation — see :ref:`theory-peierls-slab-polar-retirement` for
the rationale.


Subsection — Verification status
--------------------------------

The active slab-polar path is gated by two L1 tests in
:mod:`tests.derivations.test_peierls_reference`:

- :class:`TestSlabPolarReferenceEquivalence.test_adaptive_polar_matches_E1_reference`
  — pointwise agreement :math:`K_{\rm polar}[i,j] = \Sigma_t\,K_{E_1}[i,j]`
  to relative tolerance :math:`10^{-10}` on four representative
  :math:`(i, j)` indices at :math:`N = 8` (two panels × :math:`p=4`).
  The :math:`\Sigma_t` prefactor is the unified-operator convention
  (:math:`\Sigma_t\varphi = Kq`) — the classical :math:`E_1` Nyström
  absorbs it into :math:`K` directly.
- :class:`TestSlabPolarBuildVolumeKernel.test_matches_E1_reference_at_machine_precision`
  — full :math:`N \times N` assembly via
  :func:`build_volume_kernel(SLAB_POLAR_1D, \dots)` at :math:`N = 2`
  (one panel × :math:`p = 2`), element-wise relative tolerance
  :math:`10^{-10}`.
- :class:`TestSlabPolarBuildVolumeKernel.test_row_sum_identity` —
  :math:`K \cdot \mathbf{1} = \Sigma_t\,\varphi_{\rm uniform}(x_i)`
  against :func:`slab_uniform_source_analytical` at relative
  tolerance :math:`10^{-10}`.

These close the volume-kernel verification for slab-polar at machine
precision. The white-BC closure side is separately covered by
:class:`tests.derivations.test_peierls_rank2_bc` which exercises
:func:`~orpheus.derivations.peierls_geometry._build_closure_operator_rank2_white`'s
slab branch against the Wigner–Seitz-exact
:math:`E_2/E_3` bilinear form.


.. _theory-peierls-slab-polar-retirement:

Subsection — Retention of :mod:`peierls_slab` (the native E₁ Nyström)
----------------------------------------------------------------------

The legacy module
:mod:`orpheus.derivations.peierls_slab` is **retained indefinitely**,
not retired, as an independent cross-check implementation. Rationale:

1. **Independent verification.** Two implementations that compute the
   same answer via different numerical routes (polar-form adaptive
   mpmath.quad vs classical :math:`E_1` Nyström with singularity
   subtraction + product integration) catch bugs that either
   implementation alone would miss.
2. **Archaeological reference.** The 697-line
   singularity-subtraction / product-integration machinery in
   :func:`~orpheus.derivations.peierls_slab._basis_kernel_weights` is
   a documented instance of the classical Nyström-on-E₁ technique
   useful for future readers studying slab-specific numerics.
3. **Low cost of retention.** The Phase G.5 routing switch
   (Issue #130, 2026-04-24) now **defaults to the unified path**
   (:data:`_SLAB_VIA_UNIFIED = True`) after Issue #131 resolved the
   multi-region closed-form gap; see the following subsection. The
   ``ORPHEUS_SLAB_VIA_E1=1`` env-var override routes to the native
   path for bisection. Both paths now agree bit-exactly on the
   shipped reference, so retention is purely for cross-check
   robustness rather than a correctness backstop.
4. **L0 error-catalog references.** The entries at
   ``tests/l0_error_catalog.md`` lines 1168 and 1221 cite
   :func:`~orpheus.derivations.peierls_slab._build_kernel_matrix`
   explicitly. Retiring the module orphans those catalog entries.

**Exception.** If a future session discovers a bug in the native E₁
path that is unfixable without major rework, the module can be moved
to :file:`derivations/archive/` (reversible via ``git mv``) at that
time. Do not delete.


.. _theory-peierls-slab-polar-g5-routing:

Subsection — Phase G.5 slab routing switch (Issue #130)
--------------------------------------------------------

**Status (2026-04-24): activated, bit-exact parity.**

Phase G.5 routes the shipped slab continuous reference
(``peierls_slab_2eg_2rg`` and any future ``peierls_slab_{ng}eg_{nr}rg``)
through the unified
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg` +
:data:`SLAB_POLAR_1D` path instead of the native
:func:`~orpheus.derivations.peierls_slab.solve_peierls_eigenvalue`.
The routing is controlled by:

- :data:`~orpheus.derivations.peierls_cases._SLAB_VIA_UNIFIED` —
  module-level boolean, **defaults to True**. The
  :func:`~orpheus.derivations.peierls_cases.build_two_surface_case`
  dispatcher picks between native and unified based on its value.
- ``ORPHEUS_SLAB_VIA_E1=1`` — environment-variable override that
  forces the native path at import time (for bisection / testing).
- :func:`~orpheus.derivations.peierls_cases._build_peierls_slab_case_via_unified`
  — the unified-path case builder (symmetric with the native
  ``_build_peierls_slab_case`` but uses ``solve_peierls_mg``).

The two routes now agree **bit-exactly** on the shipped
``peierls_slab_2eg_2rg`` fixture:

.. list-table:: Phase G.5 parity benchmark (post-Issue-#131 fix,
   N=12, dps=20)
   :header-rows: 1

   * - Path
     - :math:`k_{\rm eff}`
     - Wall time
   * - Native E₁ Nyström (``boundary="white"``)
     - 1.226 530 511 976
     - 1.70 s
   * - Unified ``solve_peierls_mg`` (``boundary="white_f4"``)
     - 1.226 530 511 976
     - 942.72 s
   * - **Relative disagreement**
     - **5.4 × 10\ :sup:`-16`**
     - **553 × cost**

The cost ratio (~550 ×) is expected — the unified adaptive-mpmath
path is a verification primitive, not a production K-assembly. The
native E₁ Nyström remains the fastest shipped slab solver; the
unified path is now the default for the **reference** build
because it exercises the same machinery used for the curvilinear
references (cylinder, sphere) and makes slab testable through the
same ``solve_peierls_mg`` API surface.


.. _theory-peierls-slab-polar-g5-diagnosis:

Subsubsection — How the 1.5 % gap was diagnosed (Issue #131)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original Phase G.5 benchmark (recorded in commit ``aa6ebf0``)
showed a 1.5 % disagreement on k_eff. Single-region 1G and
single-region 2G parity tests (:class:`TestSlabPolarVsNativeE1KEff`,
:class:`TestMGSlabPolarMatchesNativeSlabMG`) had shown 1e-8
agreement, so the gap was specific to multi-region slab.

The
`numerics-investigator agent <https://github.com/deOliveira-R/ORPHEUS/issues/131>`_
ran a cascade of isolation probes under
:file:`derivations/diagnostics/diag_slab_issue131_probe_*.py`:

- **Probe A** (1G 2-region vacuum) — isolated multi-region from
  MG / closure. Already showed the gap — ruling out MG-only bugs.
- **Probe B** (2G 2-region vacuum) — isolated from F.4 closure.
  Gap persisted → bug is in the volume kernel, not the closure.
- **Probe D** (multi-region P_esc quadrature) — pinned down
  the cause: ``compute_P_esc_outer`` / ``compute_P_esc_inner`` /
  ``compute_G_bc_outer`` / ``compute_G_bc_inner`` had separate
  branches for ``len(radii) == 1`` (closed-form
  :math:`\tfrac12 E_2(\tau)`) and ``len(radii) > 1`` (finite-N GL
  over the µ-integral). The multi-region branches converged only
  to ~4 × 10\ :sup:`-3` at N=24 — a quadrature artifact.

**The fix.** For a slab with piecewise-constant
:math:`\Sigma_t(x)`, the angular integral

.. math::

   P_{\rm esc}^{\rm outer}(x_i)
     \;=\; \frac{1}{2}\!\int_0^1\!
       \exp\!\bigl[-\tau_{\rm outer}(x_i)/\mu\bigr]\,\mathrm d\mu
     \;=\; \frac{1}{2}\,E_2\!\bigl(\tau_{\rm outer}(x_i)\bigr),

is closed-form **regardless of the number of regions**, because
:math:`\tau_{\rm outer}(x_i) = \sum_k \Sigma_{t,k}\,
(r_k - \max(r_{k-1}, x_i))^+` is independent of µ. The GL quadrature
branch was therefore wasteful (and underconvergent). The fix adds
two helpers :func:`_slab_tau_to_outer_face` and
:func:`_slab_tau_to_inner_face` in
:mod:`orpheus.derivations.peierls_geometry` that piecewise-integrate
:math:`\Sigma_t` across region boundaries, and routes
**all slab-polar calls** (any ``n_regions``) through the closed-form
branch.

**Result.** The shipped 2eg_2rg parity drops from rel_diff = 1.5 %
to rel_diff = 5.4 × 10\ :sup:`-16` — **bit-exact to machine
epsilon**. Same for the 1-region case (already bit-exact before,
untouched by the fix). The parity-gate test
:class:`tests.derivations.test_peierls_multigroup.TestSlabViaUnifiedDiscrepancyDiagnostic`
now asserts ``rel_diff < 10^{-10}`` (5 orders of margin above the
current measurement) and flips from a diagnostic recording test
into a regression guard.


Subsection — Related open questions
------------------------------------

- **OQ — Can the cost per K element be reduced?** The active adaptive
  path is a verification reference, not a production K-assembly. If a
  future user needs slab K at scale, candidates include sinh-sinh
  quadrature (a super-exp variant of tanh-sinh tuned for the
  specific :math:`e^{-c\,e^v}` decay near :math:`\mu = 0`) and the
  archived moment-form architecture in
  :ref:`theory-peierls-moment-form`. Neither is blocking the
  verification story.
- **OQ — Multi-group / multi-region slab via the unified path?** The
  shipped ``peierls_slab_2eg_2rg`` continuous reference is 2-group;
  :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g` is
  1-group by name and implementation. The unified-path equivalent
  requires
  `Issue #104 <https://github.com/deOliveira-R/ORPHEUS/issues/104>`_
  (N2: multi-group Peierls extension) before the slab reference can
  be routed through the unified path. Until then the routing of
  :func:`~orpheus.derivations.peierls_cases.build_two_surface_case`
  for ``shape="slab"`` stays on the legacy
  :mod:`~orpheus.derivations.peierls_slab` module. Phase G.3
  established 1-group parity; multi-group parity is blocked.
- **OQ — Planar-limit cross-check against hollow cylinder?** The
  naive claim "hollow cylinder at :math:`r_0 \to R` reduces to a
  slab of thickness :math:`L = R - r_0`" does **not** hold at the
  :math:`10^{-8}` level that Phase G.4 originally hoped for. Probed
  empirically at :math:`r_0 = 0.999\,R`, :math:`R = 1`,
  :math:`L = 0.001`, :math:`\Sigma_t = 1`, :math:`c = 0.4`,
  :math:`\nu\Sigma_f = 0.6`: unified slab gives
  :math:`k_{\rm eff} = 0.002\,355` and unified cylinder gives
  :math:`k_{\rm eff} = 0.001\,825`, a **22 %** relative disagreement.
  The physical reason is that the cylinder's :math:`\mathrm{Ki}_1`
  has already integrated the axial direction analytically; the
  remaining in-plane chord-length distribution in a thin annular
  shell scales as :math:`\sqrt{2\,R\,L}` for tangential rays
  (:math:`\approx 0.045` for the probe's parameters), not
  :math:`L/|\mu|` as in a slab. The two kernels therefore see
  different optical-depth spectra even in the thin-shell limit, so
  there is no simple geometric equivalence at matched
  :math:`(\Sigma_t, \Sigma_s, \nu\Sigma_f, L)`. A meaningful
  planar limit needs either a ray-distribution-matched comparison
  or a curvature-over-thickness expansion — both are future work.
  Phase G.4 as specified in the plan is filed as a
  GitHub Issue for future physics investigation rather than a
  shipping test.


.. _theory-peierls-api-posture:

API posture — permanent vs transitional wrappers
=================================================

**This section is the authoritative posture statement for every
public function in the Peierls reference framework.** It codifies
which functions are *permanent* (never retired), which are
*transitional* (candidates for retirement once callers migrate), and
which are *verification-of-verification* (only invoked when the
unified path itself is under test).

The posture comes from Lesson L104a (2026-04-24 meta-review):
*"Wrappers are temporary. Whatever a wrapper does needs to become
part of the standard machinery — unless it is really necessary for
some reason."* The "really necessary" reasons are enumerated below.

Permanent public APIs
---------------------

These never get retired. New code should use them freely. Removing
them breaks the public contract.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Why permanent
   * - :func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg`
     - The canonical multi-group driver. All other entry points
       eventually hit this function.
   * - :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
     - Permanent 1G convenience. Rationale: 1-group problems have
       scalar ``sig_t``, scalar ``sig_s``, scalar ``nu_sig_f`` — no
       ``chi`` axis — and forcing callers to reshape to
       ``(n_regions, 1)`` + synthesise ``chi=1`` for every 1G
       problem has no pedagogical or maintenance benefit. The
       wrapper is 25 lines and is bit-exact regression-gated by
       :class:`~tests.derivations.test_peierls_multigroup.TestMGNg1BitMatch1G`.
   * - :func:`~orpheus.derivations.peierls_cylinder.solve_peierls_cylinder_mg`
     - Permanent shape-specific MG API. Binds the geometry, renames
       ``n_angular`` → ``n_beta`` to match the cylinder's
       ``β``-angular variable convention, and returns a
       :class:`PeierlsCylinderSolution` (shape-specific dataclass
       with ``n_quad_y`` etc.). The parameter renaming is the
       ergonomic justification.
   * - :func:`~orpheus.derivations.peierls_sphere.solve_peierls_sphere_mg`
     - Mirror of the cylinder — ``n_angular`` → ``n_theta``, returns
       :class:`PeierlsSphereSolution`.
   * - :func:`~orpheus.derivations.peierls_cylinder.solve_peierls_cylinder_1g`
     - Shape-specific 1G convenience (``solve_peierls_cylinder_mg``
       on scalar XS). Same "permanent" rationale as
       ``solve_peierls_1g``.
   * - :func:`~orpheus.derivations.peierls_sphere.solve_peierls_sphere_1g`
     - Mirror for sphere.

Retirement candidates
---------------------

All retirement candidates have been retired as of 2026-04-23; this
subsection is intentionally empty as a placeholder for future
transitional wrappers.

Verification-of-verification
-----------------------------

These are the **independent cross-check** implementations. They
must not be imported by production code paths. They exist so that
the unified reference pipeline can be verified against an
independently-developed algorithm.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Permitted callers
   * - :func:`~orpheus.derivations.peierls_slab.solve_peierls_eigenvalue`
     - Only (a)
       :class:`~tests.derivations.test_peierls_multigroup.TestSlabViaUnifiedDiscrepancyDiagnostic`
       — the parity gate that keeps unified honest, and (b) anyone
       who sets ``ORPHEUS_SLAB_VIA_E1=1`` (explicit bisection /
       testing). Production references go through
       :func:`~orpheus.derivations.peierls_cases.build_two_surface_case`
       which dispatches on ``_SLAB_VIA_UNIFIED`` (default ``True``).
       If you find yourself importing ``solve_peierls_eigenvalue``
       from non-test code, you are writing something that should be
       refactored onto the unified path.

**No sig_s convention leaks here.** The native slab module's own
sig_s convention (``sig_s[src, dst]``) matches the canonical
convention documented in :ref:`peierls-scattering-convention`, so
verification-of-verification does not need translation when
exercised against the unified path.


.. _theory-peierls-multigroup:

Multi-group Peierls eigenvalue driver (Issue #104)
==================================================

**This section documents the multi-group extension
(:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg`) that
landed on 2026-04-24**, generalising the 1-group driver
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g` to
:math:`n_g \ge 1` energy groups with downscatter/upscatter coupling
and :math:`\chi`-weighted fission. :func:`solve_peierls_1g` is now a
25-line thin wrapper over :func:`solve_peierls_mg` with
:math:`n_g = 1` and synthesised :math:`\chi = 1`; a bit-exact
regression test (:class:`tests.derivations.test_peierls_multigroup.TestMGNg1BitMatch1G`)
enforces that the ng=1 MG path reproduces every legacy 1G k_eff and
flux value to numerical zero.

The shape wrappers have matching multi-group entry points:

- :func:`~orpheus.derivations.peierls_cylinder.solve_peierls_cylinder_mg`
  — cylinder with ``inner_radius`` for hollow cells
- :func:`~orpheus.derivations.peierls_sphere.solve_peierls_sphere_mg`
  — sphere with ``inner_radius`` for hollow cells

Slab's shipped ``peierls_slab_2eg_2rg`` continuous reference now
routes through the unified
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg` +
:data:`SLAB_POLAR_1D` path by default, after Phase G.5 activation
(Issue #130 + Issue #131, 2026-04-24). The two routes agree
bit-exactly on k_eff (rel_diff ≈ 5 × 10\ :sup:`-16`); the native
:func:`~orpheus.derivations.peierls_slab.solve_peierls_eigenvalue`
E\ :sub:`1` Nyström path is retained as an independent cross-check
and can be forced via ``ORPHEUS_SLAB_VIA_E1=1``. See
:ref:`theory-peierls-slab-polar-g5-routing` for the benchmark
record and
:ref:`theory-peierls-slab-polar-g5-diagnosis` for the
investigation that closed the original 1.5 % discrepancy.


Subsection — The multi-group operator form
-------------------------------------------

For observer at radial coordinate :math:`r_i` in region :math:`k`,
group :math:`g_{\rm out}`, the unified multi-group Peierls equation
is

.. math::
   :label: peierls-mg-operator

   \Sigma_{t,g_{\rm out}}(r_i)\,\varphi_{g_{\rm out}}(r_i)
     \;=\;\sum_{j}\!K^{(g_{\rm out})}_{ij}\!\!
         \sum_{g_{\rm in}}\!\Biggl[\,
           \Sigma_{s,\,g_{\rm in} \to g_{\rm out}}(r_j)\,
           \varphi_{g_{\rm in}}(r_j)
           + \frac{1}{k}\,\chi_{g_{\rm out}}(r_i)\,
             \nu\Sigma_{f,g_{\rm in}}(r_j)\,
             \varphi_{g_{\rm in}}(r_j)
         \,\Biggr],

recast as the generalised eigenvalue problem
:math:`\tilde A\,\varphi = (1/k)\,\tilde B\,\varphi` with

.. math::

   \tilde A_{i,g_{\rm out};\,j,g_{\rm in}}
     &\;=\;\Sigma_{t,g_{\rm out}}(r_i)\,\delta_{ij}\,\delta_{g_{\rm out},g_{\rm in}}
           \;-\;K^{(g_{\rm out})}_{ij}\,\Sigma_{s,\,g_{\rm in} \to g_{\rm out}}(r_j), \\[4pt]
   \tilde B_{i,g_{\rm out};\,j,g_{\rm in}}
     &\;=\;K^{(g_{\rm out})}_{ij}\,\chi_{g_{\rm out}}(r_i)\,
           \nu\Sigma_{f,g_{\rm in}}(r_j).

Row indexing is **node-major**: the flattened index is
:math:`i \cdot n_g + g`, matching the convention in
:func:`orpheus.derivations.peierls_slab._build_system_matrices`
(the native slab driver predates Issue #104 and has always used this
pattern). The solve uses the same fission-source power iteration
as the 1G path, acting on a vector of dimension :math:`N \cdot n_g`
instead of :math:`N`.

.. _peierls-scattering-convention:

Canonical ``sig_s`` convention (project-wide single source of truth)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sig_s[r, g_src, g_dst]`` = scatter rate **from** ``g_src`` **into**
``g_dst`` at region ``r``. First index = source group, second =
destination. Downscatter (fast → thermal with group 0 = fast) sits
in the **upper-triangular** entries; the physical fixture
``_A_2G["sig_s"] = [[0.38, 0.10], [0.00, 0.90]]`` in
:mod:`orpheus.derivations._xs_library` has ``sig_s[0, 1] = 0.10``
(fast-to-thermal downscatter) and ``sig_s[1, 0] = 0`` (no
upscatter), which is the physical sanity-check this convention
must pass. All Peierls drivers
(:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg`,
:func:`~orpheus.derivations.peierls_cylinder.solve_peierls_cylinder_mg`,
:func:`~orpheus.derivations.peierls_sphere.solve_peierls_sphere_mg`,
and :func:`~orpheus.derivations.peierls_slab.solve_peierls_eigenvalue`)
follow this convention — see their docstrings and the
:func:`~orpheus.derivations._xs_library.get_xs` module note. The
2G slab parity test (:class:`TestMGSlabPolarMatchesNativeSlabMG`)
is the definitive cross-check: if the two drivers agree on a 2G
eigenvalue with genuinely directional (non-symmetric) scatter, the
convention is correct across the stack. **Terminology-glossary
cross-reference**: the :ref:`theory-peierls-naming` page links
here as the authoritative statement.


Subsection — The per-group K matrix
------------------------------------

The only group-local loop in :func:`solve_peierls_mg` rebuilds the
volume+closure K matrix once per group with that group's
:math:`\Sigma_{t,g}` trace:

.. code-block:: python

   for g in range(ng):
       K_per_group[g] = (
           build_volume_kernel(geometry, r_nodes, panels, radii,
                               sig_t[:, g], ...)
           + (closure correction from sig_t[:, g] if boundary != "vacuum")
       )

The closure primitives
(:func:`~orpheus.derivations.peierls_geometry.build_closure_operator`,
:func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction_rank_n`)
are **group-local by construction**: each primitive consumes a
per-region scalar :math:`\Sigma_t` and emits per-face escape /
response / transmission primitives that couple only within that
group. No cross-group coupling passes through the reflection
operator — this is verified by reading
:func:`~orpheus.derivations.peierls_geometry._build_closure_operator_rank2_white`,
and was the primary de-risking step of Issue #104's scoping pass.


Subsection — Why the 1G path remains a wrapper
-----------------------------------------------

Approximately 30 downstream callers use
:func:`solve_peierls_1g` (direct tests, rank-N diagnostics, the
shape-specific ``solve_peierls_*_1g`` wrappers, and the case
builders at
:mod:`~orpheus.derivations.peierls_cylinder` and
:mod:`~orpheus.derivations.peierls_sphere`). Lifting every caller
to the MG signature in a single commit would have forced a large
multi-module diff that obscures the core refactor. Instead, Issue
#104 commit 1 added the MG path and reduced :func:`solve_peierls_1g`
to a wrapper that coerces 1-D :math:`\Sigma_t`, 1-D scatter, 1-D
:math:`\nu\Sigma_f`, and synthesised :math:`\chi = 1` into the
:math:`(n_r, n_g = 1)`-shaped arrays the MG path expects.

The wrapper is verified bit-exact against the pre-refactor 1G
behaviour on **every** ``(boundary, geometry)`` combination in the
regression suite — not a tolerance, numerical zero difference on
k_eff and on every flux value. This guarantees existing callers
see no behavioural change.


Subsection — Shipped multi-group references
---------------------------------------------

As of 2026-04-24, :func:`~orpheus.derivations.peierls_cases._class_a_cases`
registers the following 2-group hollow cells alongside the legacy
1G entries:

.. list-table:: 2-group hollow cells (added in Issue #104)
   :header-rows: 1
   :widths: 50 20 30

   * - Reference name
     - :math:`r_0 / R`
     - Closure
   * - ``peierls_cyl1D_hollow_2eg_1rg_r0_10``
     - 0.1
     - F.4 scalar rank-2 per-face (Ki\ :sub:`3` fold)
   * - ``peierls_cyl1D_hollow_2eg_1rg_r0_20``
     - 0.2
     - same
   * - ``peierls_cyl1D_hollow_2eg_1rg_r0_30``
     - 0.3
     - same
   * - ``peierls_sph1D_hollow_2eg_1rg_r0_10``
     - 0.1
     - F.4 scalar rank-2 per-face (bare :math:`e^{-\tau}` kernel)
   * - ``peierls_sph1D_hollow_2eg_1rg_r0_20``
     - 0.2
     - same
   * - ``peierls_sph1D_hollow_2eg_1rg_r0_30``
     - 0.3
     - same

All six use the single-region fuel composition (``LAYOUTS[1] =
["A"]``, ``get_xs("A", "2g")``) with region-A's 2G XS. Registry
is lazy (built on first access to
:func:`~orpheus.derivations.reference_values.continuous_all` or
:func:`~orpheus.derivations.reference_values.continuous_get`); each
reference builds on demand at ~1–2 min wall time at default
quadrature, so the first access after import is expensive.

The L1 residuals vs an independent MG reference solver have not been
benchmarked as part of Issue #104 commit 2 — the parity gate against
``cp_cylinder`` / ``cp_sphere`` 2G native solvers is planned as a
follow-up (1 % target per Issue #104 AC, matching the 1G hollow
residual class). The registered cases are **buildable and
reproducible** as established by the smoke tests in
:class:`tests.derivations.test_peierls_multigroup.TestMG2GHollowRegistration`.


Subsection — Cost characteristics
----------------------------------

For :math:`n_g` groups, :math:`n_r` radial nodes, the MG driver's
cost scales as:

- **K build**: :math:`n_g \cdot n_r^2 \cdot c_K` where :math:`c_K`
  is the cost of one adaptive-quad element call. For the
  verification-primitive curvilinear path at ``dps = 25``,
  :math:`c_K` is :math:`\sim 100\,\mathrm{ms}`; for slab-polar
  adaptive :math:`c_K` is :math:`\sim 1\,\mathrm{s}` per element.
- **Closure build**: :math:`n_g \cdot c_{\rm BC}` — group-local
  per construction, scales linearly in :math:`n_g`.
- **Eigenvalue solve**: :math:`O((n_r n_g)^3)` for one LU per
  iteration × :math:`O(100)` power-iteration iterations.

In the verification regime (N < 50 nodes, ``dps = 25``), the K
build dominates: the 2G hollow cyl/sph smoke tests at tight
quadrature run in :math:`\sim 60\,\mathrm{s}` each; at default
quadrature :math:`\sim 2\,\mathrm{min}`. This is acceptable for an
L1 reference primitive but not for a production hot path.


.. _theory-peierls-moment-form:

Moment-form Nyström assembly (closed-form polynomial moments) — ARCHIVED
========================================================================

.. note::

   **Status (2026-04-20):** The implementation described in this
   section has been **archived** to
   :file:`derivations/archive/peierls_moments.py` and
   :file:`derivations/archive/peierls_slab_moments_assembly.py`,
   tracked under
   `GitHub Issue #117 <https://github.com/deOliveira-R/ORPHEUS/issues/117>`_.

   The verification side of the CP module no longer needs a fast slab
   K assembly — it uses adaptive ``mpmath.quad`` per element via
   :func:`~orpheus.derivations.peierls_geometry.K_vol_element_adaptive`,
   the single unified verification primitive across all geometries.
   The moment-form architecture is preserved here as the **expected
   production path** for a future higher-order discrete CP solver
   (where performance matters); the math, derivations, and
   conditioning analysis below are the architectural reference for
   that future work.

This section documents the **moment-form Nyström architecture** as
implemented and verified in commit history ``investigate/peierls-solver-bugs``
(2026-04-19). It supersedes the τ-coordinate Gauss–Laguerre approach
of the original "slab-polar" prototype (commit 4395cb8); the historical
content of that prototype is preserved in :ref:`theory-peierls-moment-form-failed-polar`
below as a cautionary record.

The architecture rests on a single observation:

   The polar-form Peierls integral equation
   :eq:`peierls-unified` collapses, after one substitution, to a
   contraction between a vector of **polynomial coefficients of the
   panel basis** and a vector of **polynomial moments of the kernel**.
   Both factors admit closed-form expressions whose joint cost scales
   linearly in the panel order :math:`p` and incurs **zero inner
   quadrature** for the slab. The cylinder and sphere reuse the same
   contraction at every Gauss–Legendre :math:`u`-node when the chord
   :math:`r'(\rho)` is non-polynomial in :math:`u`.

The remainder of this section presents the unified statement, the
slab specialisation in full, the cylinder/sphere variant, the
investigation history that pinned down the polynomial-in-:math:`e^{-v}`
defect of the predecessor τ-Laguerre quadrature, and the numerical
evidence backing the production gates.


Subsection — The unified moment-form statement
----------------------------------------------

For an observer at radial coordinate :math:`r_i`, the unified-form K
matrix entry is (Section 4 / :eq:`peierls-unified`)

.. math::
   :label: peierls-moment-K-source-form

   K[i, j] \;=\; \Sigma_t(r_i)\,C_d
   \int_{\mathrm \Omega_d} W_\Omega(\Omega)\,\mathrm d\Omega
   \int_0^{\rho_{\max}(r_i,\Omega)}
       \frac{1}{\Sigma_t(r'(s,\Omega,r_i))}\,
       \kappa_d\!\bigl(\tau(s)\bigr)\,L_j\!\bigl(r'(s,\Omega,r_i)\bigr)\,\mathrm ds

with :math:`C_d` the geometry prefactor, :math:`\kappa_d \in \{E_1,
\mathrm{Ki}_1, e^{-u}\}`, and :math:`\tau(s) =
\int_0^{s}\Sigma_t(r'(t,\Omega,r_i))\,\mathrm dt` the cumulative
optical depth from the observer.

Substitute the **optical-depth coordinate** :math:`u = \tau(s)`. On
each homogeneous segment of the ray :math:`u` is linear in :math:`s`
with slope :math:`\Sigma_t`, so :math:`\mathrm ds = \mathrm du /
\Sigma_t`. The inner integral over a single segment with optical-depth
range :math:`[u_a, u_b]` becomes

.. math::
   :label: peierls-moment-segment

   \int_{u_a}^{u_b}\,\kappa_d(u)\,L_j\!\bigl(r'(u)\bigr)\,
   \frac{\mathrm du}{\Sigma_{t,\text{seg}}^{\,2}},

after one factor of :math:`1/\Sigma_t` from the :math:`1/\Sigma_t(r'_{ikm})`
already in :eq:`peierls-moment-K-source-form` and one from :math:`\mathrm
ds = \mathrm du / \Sigma_{t,\text{seg}}`. (The unified-form prefactor
:math:`\Sigma_t(r_i)` absorbs the second :math:`1/\Sigma_t` so the
matrix entry stays dimensionless.)

If the panel basis :math:`L_j` is polynomial of degree :math:`p-1`
in :math:`u`, expand it in monomials,

.. math::

   L_j(r'(u)) \;=\; \sum_{m=0}^{p-1} c^{(j)}_m\,u^m,

so the segment integral collapses to the inner product

.. math::
   :label: peierls-moment-contraction

   \boxed{\;
   \int_{u_a}^{u_b}\!\kappa_d(u)\,L_j(r'(u))\,\mathrm du
   \;=\;
   \sum_{m=0}^{p-1} c^{(j)}_m\,\bigl[J_m^{\kappa_d}(u_b) - J_m^{\kappa_d}(u_a)\bigr]
   \;=\;
   \langle\,\mathbf c^{(j)}\,,\,\mathbf M^{(\kappa_d)}\,\rangle.
   \;}

The two factors play distinct roles:

- :math:`\mathbf c^{(j)}` — **basis coefficients**, geometry-specific
  through the chord map :math:`r'(u)` but kernel-independent. Built
  once per (observer, segment) pair.
- :math:`\mathbf M^{(\kappa_d)}_m = J_m^{\kappa_d}(u_b) - J_m^{\kappa_d}(u_a)`
  — **kernel moments**, geometry-independent functions of the optical-depth
  endpoints :math:`(u_a, u_b)` only. The cumulative moment
  :math:`J_m^{\kappa_d}(z)\equiv\int_0^z u^m\,\kappa_d(u)\,\mathrm du`
  admits a closed-form recursion for every kernel of interest
  (see Subsection — *Closed-form moment recursions* below).

This is the **unified moment-form statement**. Each geometry instantiates
the kernel choice (and, for curvilinear, the way the basis coefficients
are obtained — see Subsection — *Curvilinear status* below). The
overall K-matrix assembly then reads, in pseudocode:

.. code-block:: text

   for each observer i:
       for each source panel p:
           build segments of the ray from x_i intersecting panel p
           for each segment with [u_a, u_b]:
               M       = kernel_moments(u_a, u_b, p_order)         # closed form
               c[a,m]  = basis_coeffs_in_u(panel_nodes, segment)   # Vandermonde solve
               K[i, panel_node_a] += prefactor · c[a,:] · M

with no inner quadrature. The Vandermonde solve is the subject of the
slab specialisation that follows.

When the chord :math:`r'(u)` is non-polynomial in :math:`u` (cylinder,
sphere), :math:`L_j(r'(u))` is no longer a polynomial in :math:`u`,
and the closed-form contraction requires *interpolation* to recover
polynomial coefficients. This recovers the standard Gauss–Legendre
Nyström — but applied at the kernel-moment-weighted level rather than
the kernel-evaluation level. The Subsection — *Curvilinear status*
explains the trade-off in detail.


Subsection — Closed-form moment recursions
------------------------------------------

The cumulative kernel moments :math:`J_m^{\kappa_d}(z)` are derived
once per kernel by integration by parts. The derivations are pure
calculus and the implementations live in
:mod:`orpheus.derivations.peierls_moments` with element-wise gates
against :func:`mpmath.quad` to :math:`10^{-15}` relative
(:mod:`tests.derivations.test_peierls_moments`).

**Slab — :math:`E_1` moments.** From :math:`E_1'(u) = -e^{-u}/u`
and :math:`(u^{m+1})' = (m+1)\,u^m`, integration by parts of
:math:`\int_0^z u^m E_1(u)\,\mathrm du` gives

.. math::
   :label: peierls-moment-J-E1

   J_m^{E_1}(z) \;\equiv\; \int_0^z u^m\,E_1(u)\,\mathrm du
   \;=\; \frac{z^{m+1}\,E_1(z) \;+\; \gamma(m+1, z)}{m+1},

where :math:`\gamma(a, z) = \int_0^z t^{a-1} e^{-t}\,\mathrm dt` is
the lower incomplete gamma function. The intermediate step is

.. math::

   \int_0^z u^m E_1(u)\,\mathrm du
   = \left[\frac{u^{m+1}}{m+1}\,E_1(u)\right]_0^z
     - \frac{1}{m+1}\int_0^z u^{m+1}\cdot\!\left(-\frac{e^{-u}}{u}\right)\!\mathrm du
   = \frac{z^{m+1} E_1(z)}{m+1} + \frac{1}{m+1}\int_0^z u^m e^{-u}\,\mathrm du.

The boundary term at :math:`u=0` vanishes because
:math:`u^{m+1}E_1(u) \sim -u^{m+1}\ln u \to 0` for :math:`m \ge 0`.
This identity is [LewisMiller1984]_ Appendix C (slab CP polynomial
sources), [Hebert2020]_ §3.2–3.3 (general slab polynomial-source CP),
and [AbramowitzStegun1964]_ §5.1.32 (the underlying recursion of the
:math:`E_n` family). Implemented in
:func:`~orpheus.derivations.peierls_moments.e_n_cumulative_moments`
(float-out) and
:func:`~orpheus.derivations.peierls_moments.slab_segment_moments_mp`
(mpmath-out for chaining into the linear solve below).

**Cylinder — :math:`\mathrm{Ki}_1` moments.** From :math:`\mathrm{Ki}_n'(u) =
-\mathrm{Ki}_{n-1}(u)` and :math:`\int\mathrm{Ki}_n(u)\,\mathrm du =
-\mathrm{Ki}_{n+1}(u)`, repeated integration by parts of
:math:`\int_0^z u^m \mathrm{Ki}_1(u)\,\mathrm du` gives

.. math::
   :label: peierls-moment-J-Ki1

   J_m^{\mathrm{Ki}_1}(z) \;=\; -\sum_{q=0}^{m}
     \frac{m!}{(m-q)!}\,z^{m-q}\,\mathrm{Ki}_{q+2}(z)
     \;+\; m!\,\mathrm{Ki}_{m+2}(0).

Each integration by parts trades one power of :math:`u` for one
increment of the Bickley index, telescoping :math:`u^m\mathrm{Ki}_1`
all the way down to :math:`m!\,\mathrm{Ki}_{m+2}` in the boundary terms.
The endpoint constants :math:`\mathrm{Ki}_n(0) = (\sqrt{\pi}/2)\,
\Gamma(n/2)/\Gamma((n+1)/2)` come from Wallis' formula evaluated at the
explicit definition :math:`\mathrm{Ki}_n(0) = \int_0^{\pi/2}
\cos^{n-1}\theta\,\mathrm d\theta`. The recursion itself is implicit in
[Stamm1983]_ Chapters 4–6 (higher-order CP cylinder spatial expansions)
and is restated in modern notation by [Hebert2020]_ §3.4–3.5; the
underlying Bickley identities are [AbramowitzStegun1964]_ §11.2 and
[Bickley]_. Implemented in
:func:`~orpheus.derivations.peierls_moments.ki_n_cumulative_moments`
and :func:`~orpheus.derivations.peierls_moments.cylinder_segment_moments_mp`.

The :math:`\mathrm{Ki}_n(0)` constants are computed at the caller's
mpmath ``workdps`` precision via the Wallis closed form (helper
:func:`~orpheus.derivations.peierls_moments._ki_n_at_zero_mp`)
because the small-:math:`z` regime exhibits cancellation between the
boundary :math:`m!\,\mathrm{Ki}_{m+2}(0)` term and the sum of
:math:`u^{m-q}\mathrm{Ki}_{q+2}(z)` terms, amplifying any roundoff
in the boundary constants. A float-precision constant would limit the
moment recursion to :math:`\sim10^{-9}` relative at small :math:`z`;
the mpmath-native constant gives full :math:`\mathrm{dps}` precision.

**Sphere — exponential moments.** :math:`\int_0^z u^m e^{-u}\,\mathrm du
= \gamma(m+1, z)` directly; no integration by parts needed. Implemented
in :func:`~orpheus.derivations.peierls_moments.exp_cumulative_moments`
and :func:`~orpheus.derivations.peierls_moments.sphere_segment_moments_mp`.

Per-segment differences (used by the K assembly):

.. math::

   \mathbf M^{(\kappa)}_m \;\equiv\;
   \int_{u_a}^{u_b}\!u^m\,\kappa(u)\,\mathrm du
   \;=\; J_m^{\kappa}(u_b) - J_m^{\kappa}(u_a),
   \qquad m = 0, 1, \dots, p-1.

For the slab, computing :math:`\mathbf M^{(E_1)}` requires exactly two
:math:`E_1` evaluations and :math:`2p` lower-incomplete-gamma
evaluations — entirely closed-form, no quadrature.


Subsection — The slab specialisation (production)
-------------------------------------------------

The slab is the "easy" geometry because the chord map is linear:
:math:`r'(s) = x_i + s\,\sigma`, where :math:`\sigma = \pm 1` is the
ray-direction sign. With :math:`u = \Sigma_{t,\text{seg}}\,s` (single
homogeneous segment along the ray), :math:`r'(u) = x_i +
\sigma\,u/\Sigma_{t,\text{seg}}` is **linear in** :math:`u`. A panel
Lagrange basis :math:`L_a(r')` of degree :math:`p-1` in :math:`r'`
remains degree :math:`p-1` in :math:`u` — exactly the polynomial form
that :eq:`peierls-moment-contraction` requires.

The implementation
:func:`~orpheus.derivations.peierls_geometry._build_volume_kernel_slab_moments`
processes each (observer, source panel) pair as follows:

**Step 1 — Walk the ray.** For each observer :math:`x_i`, walk the ray
to the source panel and accumulate the cumulative-:math:`\tau` offset
:math:`\Delta = \Sigma_{t}\,(x_l - x_i)` (right-going) or :math:`\Sigma_{t}\,(x_i - x_r)`
(left-going), using a piecewise-linear cumulative-tau table over the
material regions of the slab:

.. math::

   \Delta \;=\; \int_{x_i}^{\text{ray-entry into panel}}
                  \Sigma_t(\xi)\,\mathrm d\xi.

This handles heterogeneous material boundaries between observer and
source panel without unnecessary subdivision: the cumulative-tau table
is built once per K assembly. (Source panels do not span material
boundaries by construction, since the composite-GL panel grid respects
material regions.)

**Step 2 — Self-panel splitting.** When the observer sits inside the
source panel (:math:`x_l \le x_i \le x_r`), the ray-from-observer
naturally leaves the panel in two opposite directions. Split into two
ray pieces:

.. math::

   \text{piece A: } [x_i, x_r],\ \sigma=+1,\ \Delta = 0; \qquad
   \text{piece B: } [x_l, x_i],\ \sigma=-1,\ \Delta = 0.

Each piece is C\ :sup:`∞` in :math:`u`; no log-singularity subtraction
is needed because :math:`E_1(u)` itself is :math:`\sim -\ln u + O(1)`
near the origin and our **moment** recursion :eq:`peierls-moment-J-E1`
already absorbs that singularity into the closed form (the boundary
term :math:`z^{m+1}E_1(z)/(m+1)` carries the log analytically). This
is the central reason the slab-moment form succeeds where the legacy
:math:`E_1` Nyström needed Atkinson's product-integration recipe
([Atkinson1997]_): the closed form integrates the singularity in
*symbolic* form, so the residual integrand the quadrature would have
seen is already exactly :math:`\gamma(m+1, z) / (m+1)` — finite and
smooth.

**Step 3 — Build the per-segment moment vector.** With segment
optical-depth range :math:`[u_a = \Delta,\,u_b = \Delta + \Sigma_{t,\text{panel}}\,
(x_r - x_l)\bigr]` (or piece-specific endpoints for self-panel halves),
compute

.. math::

   \mathbf M_m \;=\; J_m^{E_1}(u_b) - J_m^{E_1}(u_a),
   \qquad m = 0, 1, \dots, p-1,

via the closed form :eq:`peierls-moment-J-E1` evaluated at mpmath ``dps``
precision (default :math:`30`).

**Step 4 — Vandermonde solve for cardinal Nyström weights.** The panel
nodes :math:`\xi_a` (for :math:`a = 0, 1, \dots, p-1`) define cardinal
Lagrange polynomials :math:`L_a(r') = \prod_{b\neq a}(r' - \xi_b)/(\xi_a -
\xi_b)`. In the segment's optical-depth coordinate, each node sits at

.. math::

   u_k \;=\; \Delta + \Sigma_{t,\text{panel}}\,\sigma\,(\xi_k - x_{\text{a,seg}})

(with :math:`x_{\text{a,seg}} = x_l` for :math:`\sigma=+1`,
:math:`x_{\text{a,seg}} = x_r` for :math:`\sigma=-1` — i.e. the
ray-entry point of the segment). We want the contribution of each
basis function :math:`L_a` to the segment integral, i.e. weights
:math:`w_a` such that

.. math::

   \int_{u_a}^{u_b} \kappa_d(u)\,L_a(r'(u))\,\mathrm du \;=\; w_a.

By the cardinal property and :eq:`peierls-moment-contraction`,

.. math::
   :label: peierls-moment-vandermonde

   \mathbf V^\top\,\mathbf w \;=\; \mathbf M,
   \qquad \mathbf V[k, m] = u_k^m,

where :math:`\mathbf V` is the :math:`p\times p` Vandermonde matrix at
the panel-node :math:`u`-coordinates and :math:`\mathbf w =
(w_0, w_1, \dots, w_{p-1})`. The transpose comes from the dual
cardinal-vs-monomial relation: the monomial expansion
:math:`L_a(r'(u)) = \sum_m c^{(a)}_m\,u^m` has coefficients
:math:`c^{(a)} = (\mathbf V^{-1})_{a,:}` (cardinal at :math:`u_k`),
so :math:`w_a = \sum_m (\mathbf V^{-1})_{a,m}\,M_m =
(\mathbf V^{-\top}\mathbf M)_a`.

The system :eq:`peierls-moment-vandermonde` is solved at mpmath ``dps``
precision with :func:`mpmath.lu_solve`. The Vandermonde matrix is
notoriously ill-conditioned in floating point — for distant source
panels (large :math:`\Delta`) the panel-node :math:`u`-coordinates
cluster (relative spread :math:`(\xi_p - \xi_0)\,\Sigma_t /
\Delta` becomes small), and a float-precision solve loses 6-10 digits.
Solving at :math:`\mathrm{dps}=30` gives single-digit ULP loss in the
final K entry — well below the :math:`10^{-12}` test gate.

**Step 5 — Accumulate into K.** With the Nyström weights in hand,

.. math::
   :label: peierls-moment-K-assembly

   K[i, j_{\text{start}+a}] \;\mathrel{+}=\;
   \frac{\Sigma_t(r_i)\,C_d}{\Sigma_{t,\text{panel}}}\;w_a,
   \qquad a = 0, 1, \dots, p-1,

where :math:`C_d = 1/2` for the slab. The :math:`1/\Sigma_{t,\text{panel}}`
absorbs the :math:`\mathrm ds = \mathrm du / \Sigma_t` Jacobian; the
:math:`\Sigma_t(r_i)` is the unified-form left-hand-side prefactor
from :eq:`peierls-unified` and is **observer-side** (different from the
segment :math:`\Sigma_{t,\text{panel}}` for heterogeneous slabs).

The full algorithm is :math:`O(N \cdot N_{\text{panels}} \cdot p^3)` —
linear in number of (observer, source-panel) pairs and cubic in panel
order through the LU. For typical :math:`N=24, p=4` it builds in
:math:`\sim 60` ms at :math:`\mathrm{dps}=30`; at :math:`p=6` and
:math:`N=24`, :math:`\sim 200` ms.

.. note::

   The slab moment-form is **exact in the inner integration**: there is
   no inner quadrature, no inner discretization error, no inner
   convergence sweep to perform. The only sources of error are (a) the
   panel basis order :math:`p-1` (truncation of the source expansion;
   converges spectrally for smooth solutions), (b) the mpmath
   precision of the moment evaluations and Vandermonde solve (default
   :math:`\mathrm{dps}=30` gives :math:`\sim10^{-28}` relative; well
   below any practical gate). Compare with the legacy :math:`E_1`
   Nyström, which used adaptive :math:`\texttt{mpmath.quad}` in the
   inner — also "exact" but at a much higher cost (see Subsection —
   *Performance characteristics* below).


.. _theory-peierls-moment-form-failed-polar:

Subsection — Why the predecessor τ-Laguerre polar form failed
-------------------------------------------------------------

The first attempt at unifying the slab into the polar-form Nyström
framework (commit 4395cb8, "feat(derivations): add slab-polar as
first-class CurvilinearGeometry kind") used a **τ-coordinate
Gauss–Laguerre** outer integration in the substitution :math:`v =
-\ln|\mu|`. **That implementation has been retired** because it does
not converge to machine precision — it plateaus at relative error
:math:`\sim 9 \times 10^{-4}` regardless of how many quadrature nodes
are added. Diagnostic scripts
:file:`derivations/diagnostics/diag_slab_polar_outer_mu_structure.py`
and :file:`derivations/diagnostics/diag_slab_polar_glaguerre.py`
isolate the cause; this subsection records the result so future
sessions do not re-attempt the failed approach.

**The slab-polar formulation.** With observer at :math:`x_i` and
direction-cosine :math:`\mu`, the polar-form slab K is

.. math::

   K[i, j] \;=\; \tfrac{1}{2}\,\Sigma_t(r_i)
   \int_{-1}^{1}\!\mathrm d\mu
     \int_0^{\rho_{\max}(x_i,\mu)}
       \frac{e^{-\Sigma_t \rho}}{\Sigma_t(r')}\,L_j(x_i + \rho\mu)\,\mathrm d\rho.

Substitute :math:`v = -\ln|\mu|` so :math:`\mu = \pm e^{-v}` with
Jacobian :math:`|\mathrm d\mu/\mathrm dv| = e^{-v}`. The outer integral
on each branch becomes :math:`\int_0^\infty g(e^{-v})\,e^{-v}\,\mathrm dv`,
where :math:`g(\mu)` is the inner ρ-integral as a function of :math:`\mu`.
Apply Gauss–Laguerre to the outer :math:`v`.

**The defect.** Gauss–Laguerre is *spectrally* accurate when the
integrand on :math:`(0, \infty)` has the form
:math:`p(v)\cdot e^{-v}` with :math:`p` polynomial in :math:`v` —
that is the rule's definition of "polynomial-exact." Our integrand
has the form

.. math::

   g(e^{-v})\,e^{-v},

where :math:`g(\mu)` admits the small-:math:`\mu` Laplace expansion

.. math::

   g(\mu) \;=\; \frac{L_j(x_i)}{\Sigma_t}
              + \mu\,\frac{L_j'(x_i)}{\Sigma_t^2}
              + \mu^2\,\frac{L_j''(x_i)}{\Sigma_t^3}
              + \dots

(from differentiating the absorption integral by parts in :math:`\mu`),
so :math:`g(\mu)` is a polynomial of degree :math:`p-1` *in*
:math:`\mu = e^{-v}`. The full integrand is therefore a sum of terms
:math:`e^{-kv}` for :math:`k = 1, 2, \dots, p`, **not** a polynomial
in :math:`v`.

Gauss–Laguerre :math:`n`-point integrates :math:`p(v)\,e^{-v}` exactly
for :math:`\deg p \le 2n - 1`; on a generic :math:`e^{-kv}` integrand
it converges only **algebraically** at rate :math:`O(n^{-2})` from the
remainder bound for non-polynomial integrands (the relevant stationary-phase
analysis: Gauss-Laguerre is asymptotic in :math:`1/n` for any
:math:`e^{-kv}` with :math:`k\neq 1`). Our integrand mixes :math:`k=1`
through :math:`k=p`, so the leading-order error is set by the worst
:math:`k`.

**Diagnostic confirmation** (table from
:file:`derivations/diagnostics/diag_slab_polar_outer_mu_structure.py`,
:math:`L=1`, :math:`\Sigma_t=1`, two-panel :math:`p=4`, exact-inner
:func:`mpmath.quad` so the outer is isolated):

.. list-table:: Outer-only convergence with EXACT inner. Each scheme
   uses :math:`n_v` outer nodes; relative error in :math:`K[0,0]`
   against the adaptive reference :math:`K[0,0] = 1.107\times10^{-1}`.
   :header-rows: 1

   * - :math:`n_v`
     - v-Laguerre
     - v-GL on :math:`[0, 30]`
     - GL on :math:`[-1,0]\cup[0,1]`
   * - 8
     - 6.6e-3
     - 5.2e-3
     - 1.1e-3
   * - 16
     - 1.6e-3
     - 1.3e-3
     - 7.4e-5
   * - 32
     - 4.0e-4
     - 3.4e-4
     - 6.2e-6
   * - 64
     - 1.0e-4
     - 8.7e-5
     - 7.1e-4

Two readings of this table are essential:

1. The **v-Laguerre and v-GL** rates are both :math:`O(n^{-2})` —
   confirming the polynomial-in-:math:`e^{-v}` defect is
   substitution-independent (any rule chosen for the polynomial-in-:math:`v`
   weight :math:`e^{-v}` shares the same exactness class).

2. The **standard GL** convergence is *non-monotonic*: it drops to
   :math:`6\times 10^{-6}` at :math:`n=32` then *jumps back up*
   to :math:`7\times 10^{-4}` at :math:`n=64`. This is the smoking
   gun of an outer-integrand kink that the GL node placement straddles
   differently at each :math:`n`. The kink has not been identified
   analytically but is the predicted signature of the dyadic-:math:`\tau`-cap
   on the inner subdivision (see hypothesis H3 in the diagnostic
   docstring) — a discrete change in the number of inner sub-intervals
   creates a discontinuity in :math:`g(\mu)` as :math:`\mu` moves past
   the cap-trigger threshold. The moment-form sidesteps this entirely
   because the inner integral is closed-form; no cap is needed.

**Generalised Laguerre does not help.** The diagnostic
:file:`diag_slab_polar_glaguerre.py` sweeps the :math:`\alpha`
parameter of the generalised Laguerre weight :math:`v^\alpha\,e^{-v}`
(intuition: shifting nodes toward small :math:`v` should help because
the integrand is concentrated there). Result (relative error in
:math:`K[0,0]`, :math:`n_v=32`, exact inner):

.. list-table:: Generalised Laguerre with :math:`\alpha`-sweep. The
   :math:`v^{-\alpha}` factor is folded into the integrand to
   compensate for the weight; quadrature acts on
   :math:`v^\alpha\,e^{-v}` directly.
   :header-rows: 1

   * - :math:`n_v`
     - :math:`\alpha = -0.5`
     - :math:`\alpha = 0.0`
     - :math:`\alpha = +0.5`
     - :math:`\alpha = +1.0`
     - :math:`\alpha = +2.0`
     - :math:`\alpha = +5.0`
   * - 8
     - 6.5e-3
     - 6.6e-3
     - 7.1e-3
     - 8.1e-3
     - 1.1e-2
     - 2.5e-2
   * - 16
     - 1.6e-3
     - 1.6e-3
     - 1.8e-3
     - 2.2e-3
     - 3.4e-3
     - 9.5e-3
   * - 32
     - 4.0e-4
     - 4.0e-4
     - 4.5e-4
     - 5.7e-4
     - 1.0e-3
     - 3.6e-3
   * - 64
     - 1.0e-4
     - 1.0e-4
     - 1.1e-4
     - 1.5e-4
     - 2.8e-4
     - 1.1e-3

Every value of :math:`\alpha` is in the same algebraic-convergence
class as standard Laguerre, with :math:`\alpha > 0` strictly worse
(node clustering toward large :math:`v` is precisely the wrong place
for an integrand concentrated at small :math:`v`) and :math:`\alpha < 0`
giving no measurable improvement. The theoretical reason is that **all
Laguerre flavours, regardless of** :math:`\alpha`, **are spectral only
for polynomial-in-**:math:`v` **integrands** — the weight :math:`v^\alpha
e^{-v}` does not change the rule's polynomial-exactness class. Our
integrand is polynomial in :math:`e^{-v}`, not in :math:`v`. There is
no Laguerre flavour that fixes this.

**The architectural realisation.** The legacy :math:`E_1` Nyström
(:func:`~orpheus.derivations.peierls_slab._basis_kernel_weights`) does
not exhibit this convergence pathology because it integrates the
angular variable **analytically** — the classical slab equation
:eq:`peierls-equation` already has :math:`E_1(\Sigma_t|x-x'|)` as the
kernel, and :math:`E_1` *is* the analytical result of
:math:`\int_0^\infty e^{-u}/u\,\mathrm du` along the angular direction.
The polar form was undoing that analytical integration and trying to
recover it numerically.

The unified moment form re-uses :math:`E_1` *as the kernel* (the way
the classical slab Nyström does) but re-writes the angular-then-radial
integration as a single optical-depth-coordinate integral with a
polynomial source. The result is the polynomial-coefficient × kernel-moment
contraction :eq:`peierls-moment-contraction` — exact, closed-form,
no quadrature.

Reading the diagnostic scripts will reproduce these tables and confirm
the analysis end-to-end.


Subsection — Curvilinear status (cylinder and sphere)
-----------------------------------------------------

For the cylinder and sphere geometries the chord map :math:`r'(\rho) =
\sqrt{r_i^2 + \rho^2 - 2 r_i \rho \cos\Omega}` is non-linear in
:math:`\rho`, hence non-linear in the optical-depth coordinate
:math:`u = \Sigma_t\,\rho` even within a homogeneous segment. A panel
basis :math:`L_j(r')` polynomial of degree :math:`p-1` in :math:`r'`
is **not** polynomial in :math:`u` once composed with the chord map.
Equation :eq:`peierls-moment-contraction` therefore does not give a
closed-form K entry; the polynomial coefficients of :math:`L_j(r'(u))`
are not a finite list.

The way the unified architecture handles this — implemented in
:func:`~orpheus.derivations.peierls_geometry._build_volume_kernel_curvilinear_moments`
— is to **interpolate** :math:`L_j(r'(u))` at :math:`n_\rho`
Gauss–Legendre nodes inside each segment, recover the polynomial
coefficients :math:`c^{(j)}_m` from a Vandermonde solve at those
:math:`u`-nodes, and contract against the closed-form
:math:`\mathrm{Ki}_n` / :math:`e^{-u}` moments
:eq:`peierls-moment-J-Ki1`:

.. math::

   K[i, j] \;\mathrel{+}=\;
   \frac{\Sigma_t(r_i)\,C_d}{\Sigma_{t,\text{seg}}}\,
   \sum_{k=0}^{n_\rho-1} w_k\,L_j\!\bigl(r'(u_k)\bigr),

with :math:`w_k = \langle (\mathbf V^{-\top})_{k,:}, \mathbf
M^{(\kappa)}\rangle` the Nyström weight at node :math:`u_k`, computed
once per segment from the closed-form moment vector and the
GL-:math:`u`-node Vandermonde.

The relationship to the slab form is precise: **same architecture, same
moment recursions, same Vandermonde solve**, but the polynomial
coefficients are obtained by interpolation rather than from the
panel basis directly. We call this the *kernel-natural moment form*
because the kernel evaluation reduces to closed-form moments at segment
endpoints — but the source expansion is GL-based, not panel-cardinal.

**Why this is not a regression.** The legacy curvilinear path
(:func:`~orpheus.derivations.peierls_geometry._build_volume_kernel_curvilinear_moments`
*precursor* — the GL-inner-then-:math:`\kappa_d`-evaluation path
documented in Section 6) evaluates the kernel :math:`\kappa_d(u_k)` at
each Gauss–Legendre node and quadrature-sums. The kernel evaluation at
isolated nodes is just a numerical approximation of the closed-form
moment :math:`\int_{u_a}^{u_b}\!u^m \kappa_d(u)\,\mathrm du` accurate
to the GL polynomial-exactness class :math:`(2 n_\rho - 1)` in :math:`u`.
The natural-kernel moment form **uses the closed form directly** — it
is one architectural step *more* accurate per node, and reduces the
required :math:`n_\rho` for a given precision. Otherwise the spectral
convergence in :math:`n_\rho` is identical (set by the smoothness of
:math:`L_j(r'(u))`).

In summary, all three geometries instantiate the same architecture:

.. list-table:: Architecture instantiation by geometry.
   :header-rows: 1

   * - Geometry
     - Kernel :math:`\kappa_d`
     - Moment recursion
     - Polynomial coeffs
     - Inner quadrature
   * - Slab
     - :math:`E_1`
     - :eq:`peierls-moment-J-E1`
     - panel-cardinal (closed)
     - **none**
   * - Cylinder
     - :math:`\mathrm{Ki}_1`
     - :eq:`peierls-moment-J-Ki1`
     - GL-:math:`u`-cardinal
     - :math:`n_\rho` GL on :math:`L_j(r'(u))`
   * - Sphere
     - :math:`e^{-u}`
     - :math:`\gamma(m+1, z)`
     - GL-:math:`u`-cardinal
     - :math:`n_\rho` GL on :math:`L_j(r'(u))`


Subsection — Numerical evidence
-------------------------------

The slab moment-form K matrix is gated against three independent
references in :mod:`tests.derivations.test_peierls_slab_moments`
(L1 equivalence, all marked
``@pytest.mark.verifies("peierls-equation")``). All gates pass to the
indicated tolerances on commit ``investigate/peierls-solver-bugs``.

**Gate 1 — moment vs legacy E\ :sub:`1` Nyström, homogeneous slabs.**
Element-wise relative error :math:`< 10^{-12}` across five
parametrizations:

.. list-table:: Homogeneous-slab moment vs legacy-:math:`E_1` Nyström
   gate (:func:`test_slab_moments_match_legacy_E1`).
   Tolerance :math:`10^{-12}` element-wise relative.
   :header-rows: 1

   * - :math:`L`
     - :math:`\Sigma_t`
     - :math:`n_{\text{panels}}`
     - :math:`p`
     - status
   * - 1.0
     - 1.0
     - 2
     - 4
     - pass
   * - 1.0
     - 1.0
     - 4
     - 6
     - pass
   * - 2.0
     - 0.5
     - 3
     - 4
     - pass
   * - 1.0
     - 5.0
     - 2
     - 4
     - pass (optically thick, :math:`\Sigma_t L = 5`)
   * - 0.5
     - 2.0
     - 4
     - 4
     - pass (short slab)

The :math:`p=6` parametrisation is particularly informative: the
moment recursion :eq:`peierls-moment-J-E1` is exercised at
:math:`m=0,1,\dots,5` and the Vandermonde
:eq:`peierls-moment-vandermonde` is the :math:`6\times 6` system
hardest to condition in float arithmetic. The mpmath-:math:`\mathrm{dps}=30`
solve preserves :math:`\sim 28` digits, comfortably below the
:math:`10^{-12}` test gate.

**Gate 2 — moment vs legacy E\ :sub:`1` Nyström, heterogeneous slabs.**
Same gate, three two-region parametrizations:

.. list-table:: Heterogeneous-slab moment vs legacy gate
   (:func:`test_slab_moments_heterogeneous_match_legacy`).
   Two regions, panel grid respects material boundary.
   :header-rows: 1

   * - region thicknesses
     - :math:`\Sigma_t` regions
     - :math:`n_{\text{panels per region}}`
     - :math:`p`
     - status
   * - [1.0, 1.0]
     - [1.0, 0.5]
     - 2
     - 4
     - pass
   * - [0.5, 1.5]
     - [2.0, 0.3]
     - 3
     - 4
     - pass
   * - [1.0, 0.5]
     - [0.8, 4.0]
     - 4
     - 4
     - pass

The third parametrisation is the most demanding: a thin
optically-thick region (:math:`\Sigma_t L = 4 \cdot 0.5 = 2`) directly
adjacent to a thicker optically-thin region (:math:`\Sigma_t L = 0.8`).
The cumulative-tau walker (:math:`\Delta` from Step 1 of the slab
specialisation above) must compose the two regions correctly to give
the right segment-:math:`u` endpoints; a sign-flip or material-region
indexing bug would surface here as a per-cross-region :math:`K[i,j]`
discrepancy. All :math:`(i, j)` entries pass :math:`10^{-12}`.

**Gate 3 — moment vs adaptive polar reference, element-wise.** Spot-check
five entries against
:func:`~orpheus.derivations.peierls_reference.slab_polar_K_vol_element`,
which performs the polar-form integral with nested adaptive
:math:`\texttt{mpmath.quad}` (:func:`test_slab_moments_element_matches_polar_reference`,
:math:`L=1`, :math:`\Sigma_t=1`, 2 panels :math:`p=4`):

.. list-table:: Moment-form K vs adaptive polar reference,
   element-wise. Tolerance :math:`10^{-10}`.
   :header-rows: 1

   * - :math:`(i, j)`
     - role
     - status
   * - :math:`(0, 0)`
     - leftmost-observer self-contribution
     - pass
   * - :math:`(0, N-1)`
     - leftmost observer to rightmost source
     - pass
   * - :math:`(N/2, N/2)`
     - middle-of-slab self-contribution
     - pass
   * - :math:`(N-1, 0)`
     - rightmost observer to leftmost source
     - pass
   * - :math:`(1, 3)`
     - cross-panel arbitrary entry
     - pass

The :math:`10^{-10}` floor (vs :math:`10^{-12}` in Gates 1–2) reflects
the adaptive :math:`\texttt{mpmath.quad}` reference's own discretisation
limit; the moment form itself is closer to mpmath ULP.

**Gate 0 — moment recursion vs mpmath.quad (term verification).**
Underneath all of the above sits :mod:`tests.derivations.test_peierls_moments`,
which verifies the closed-form moment recursions term-by-term against
:func:`mpmath.quad` of the same integrand:

.. list-table:: L0 term-verification gates for the closed-form
   moment recursions. 32 parametrisations total
   (:math:`z \in \{10^{-3}, 10^{-2}, 0.1, 0.5, 1, 2.5, 5, 10, 25\}`,
   :math:`m \in \{0, 1, \dots, 6\}`).
   :header-rows: 1

   * - moment family
     - tolerance
     - basis
   * - :math:`J_m^{E_1}` (slab)
     - :math:`10^{-13}`
     - integration by parts → :eq:`peierls-moment-J-E1`
   * - :math:`J_m^{\mathrm{Ki}_1}` (cylinder)
     - :math:`10^{-12}`
     - repeated IBP → :eq:`peierls-moment-J-Ki1`
   * - :math:`J_m^{e^{-u}}` (sphere)
     - :math:`10^{-15}`
     - direct via :math:`\gamma(m+1, z)`

The slightly looser :math:`\mathrm{Ki}_1` tolerance reflects the
small-:math:`z` cancellation in :eq:`peierls-moment-J-Ki1` between the
boundary :math:`m!\,\mathrm{Ki}_{m+2}(0)` term and the sum of
:math:`\mathrm{Ki}_{q+2}(z)` terms; even with mpmath-native :math:`\mathrm{Ki}_n(0)`
constants, the cancellation reduces working precision by 2-3 digits
at :math:`z = 10^{-3}`. This does not affect the K-matrix gate because
the segment endpoint optical depths in any practical slab/cylinder
geometry are :math:`> 10^{-3}` by orders of magnitude.

Together these four gates establish that the moment-form K matrix
**equals** the legacy adaptive references at the :math:`10^{-12}`
level, with the underlying recursions verified at :math:`10^{-13}` or
better. The :math:`10^{-12}` gate is well below the tolerance of the
power-iteration eigenvalue solver (:math:`10^{-10}` typical) so the
moment form is a drop-in replacement.


Subsection — Performance characteristics
----------------------------------------

The moment-form slab K assembly is dominated by two costs:

1. The closed-form moment vector evaluation
   :func:`~orpheus.derivations.peierls_moments.slab_segment_moments_mp`,
   which performs :math:`2 (p+1)` evaluations of
   :math:`\mathrm{mpmath.gammainc}` per segment plus two
   :math:`\mathrm{mpmath.expint}(1, \cdot)` evaluations.
2. The :math:`p \times p` Vandermonde LU solve at mpmath ``dps`` precision.

For a representative :math:`N = 24, p = 4, n_{\text{panels}} = 6`
homogeneous problem at :math:`\mathrm{dps} = 30`, the K assembly takes
:math:`\sim 60` ms wall-clock; for :math:`p = 6` the same problem takes
:math:`\sim 200` ms. By comparison the legacy :math:`E_1` Nyström at
the same precision and same K size takes :math:`\sim 800` ms, dominated
by adaptive :math:`\texttt{mpmath.quad}` calls (each adaptive call
spends :math:`\sim 50` integrand evaluations at :math:`\mathrm{dps}=25`).
The moment form is therefore both **simpler** (closed form vs
adaptive) and **faster** (no adaptive sub-grid management).

A future optimisation is to perform the Vandermonde solve in
**float precision** when the panel-node :math:`u`-spread is wide
enough that the system is well-conditioned (i.e. for self-panels and
near-observer source panels). The conditioning is
:math:`\kappa_2(\mathbf V) \sim ((\xi_p - \xi_0)\,\Sigma_t / \langle u \rangle)^{-(p-1)}`
roughly; for :math:`\langle u \rangle \lesssim 5` (within a few mean
free paths) the condition number is :math:`\le 10^{8}` at :math:`p=4`
and a float solve is safe. For more distant source panels the mpmath
solve is required; a hybrid switch could shave another :math:`\sim 3\times`
off the wall-clock without compromising precision.

This optimisation is **not** currently scheduled — the :math:`200`-ms
build at :math:`p=6` is a one-shot cost amortised across many
power-iteration steps, and rank-:math:`N` BC-mode K builds reuse the
same volume kernel. Profile first if K-build time becomes a bottleneck.


Subsection — References
-----------------------

The moment-form architecture and its implementation rest on the
following primary sources:

- [LewisMiller1984]_ Appendix C — slab CP polynomial-source
  integration-by-parts identity for :math:`\int_0^z u^m E_1(u)\,\mathrm du`.
  Closest single-source statement of the slab moment recursion
  :eq:`peierls-moment-J-E1`. See also Chapter 5 of the same volume for the
  "exponential integral identity" used here at :math:`m=0`.
- [Hebert2020]_ §3.2-3.5 — modern restatement of the slab
  (§3.2-3.3) and cylindrical (§3.4-3.5) polynomial-source CP recursions
  in the language of integration-by-parts moment expansions. Hébert is
  the most accessible textbook reference for a reader implementing the
  moment form from scratch.
- [Stamm1983]_ Chapters 4-6 — the canonical derivation of the
  cylinder Bickley-recursion family, including the Wallis closed form
  :math:`\mathrm{Ki}_n(0) = (\sqrt\pi/2)\,\Gamma(n/2)/\Gamma((n+1)/2)`
  used to seed the boundary terms in :eq:`peierls-moment-J-Ki1`.
- [AbramowitzStegun1964]_ §5.1.32 — the underlying recursion
  :math:`E_n'(u) = -E_{n-1}(u)` and the boundary behaviour
  :math:`u^{m+1} E_1(u) \to 0` as :math:`u \to 0` that justifies dropping
  the lower limit in the integration by parts. §11.2 covers the
  Bickley-Naylor :math:`\mathrm{Ki}_n` identities used in
  :eq:`peierls-moment-J-Ki1`.
- [Bickley]_ — original 1935 introduction of the Bickley-Naylor family
  with closed-form differentiation and integration rules. The identities
  :math:`\mathrm{Ki}_n'=-\mathrm{Ki}_{n-1}` and
  :math:`\int\mathrm{Ki}_n\,\mathrm du = -\mathrm{Ki}_{n+1}` come from
  this paper.

Implementations and tests:

- :mod:`orpheus.derivations.peierls_moments` — closed-form moment
  vector implementations (:func:`~orpheus.derivations.peierls_moments.e_n_cumulative_moments`
  / :func:`~orpheus.derivations.peierls_moments.ki_n_cumulative_moments`
  / :func:`~orpheus.derivations.peierls_moments.exp_cumulative_moments`)
  and per-segment differences (:func:`~orpheus.derivations.peierls_moments.slab_segment_moments_mp`
  / :func:`~orpheus.derivations.peierls_moments.cylinder_segment_moments_mp`
  / :func:`~orpheus.derivations.peierls_moments.sphere_segment_moments_mp`).
- :func:`~orpheus.derivations.peierls_geometry._build_volume_kernel_slab_moments` —
  slab K assembly via :eq:`peierls-moment-K-assembly`.
- :func:`~orpheus.derivations.peierls_geometry._build_volume_kernel_curvilinear_moments` —
  cylinder/sphere K assembly via the kernel-natural moment form.
- :mod:`tests.derivations.test_peierls_moments` — L0 gates for the
  closed-form moment recursions (32 parametrisations).
- :mod:`tests.derivations.test_peierls_slab_moments` — L1 equivalence
  gates for the slab K matrix (8 parametrisations + element-wise spot
  check).
- :file:`derivations/diagnostics/diag_slab_polar_outer_mu_structure.py`
  and :file:`derivations/diagnostics/diag_slab_polar_glaguerre.py` —
  diagnostic scripts confirming the polynomial-in-:math:`e^{-v}` defect
  of the predecessor τ-Laguerre approach.


Motivation and scope
====================

The Peierls integral equation appears three times in this project —
once per 1-D geometry supported by the collision-probability module
(slab, cylinder, sphere). Each instance is a Nyström reference
solver whose role is to verify the corresponding flat-source CP
solver (:mod:`orpheus.cp.solver`) against the full, un-integrated
integral transport equation. The three instances are currently
housed in:

- :mod:`orpheus.derivations.peierls_slab` — slab :math:`E_1` reference
  (Phase 4.1, shipped).
- :mod:`orpheus.derivations.peierls_cylinder` — cylinder
  :math:`\mathrm{Ki}_1` reference (Phase 4.2, shipped; companion
  theory page in :doc:`collision_probability`).
- (Planned) ``orpheus.derivations.peierls_sphere`` — sphere
  :math:`e^{-\tau}` reference (Phase 4.3, deferred on the white-BC
  closure; see GitHub Issue #100).

Read naively, these three modules implement three different
equations with three different kernels (:math:`E_1`,
:math:`\mathrm{Ki}_1`, :math:`e^{-\tau}`), three different singular
structures (logarithmic, integrable, smooth), and three different
chord parametrisations. The literature
([Sanchez1982]_ §IV, [Hebert2020]_ Chapter 3, [Stamm1983]_ Chapter 6,
[CaseZweifel1967]_ Chapter 2) reinforces this impression by
presenting each geometry with its own derivation, its own
nomenclature, and its own table of special functions.

**This impression is superficial.** The three equations are
*the same integral transport equation* written in three different
coordinate systems, discretised with three different pre-integrations
of the common 3-D point kernel. Once that pre-integration has been
performed and the remaining integral is written in polar coordinates
centred on the observer, the difference between the geometries
reduces to a choice of *angular measure* and a choice of *kernel
function*. The spatial integrand is smooth and uniformly treatable
by a single Nyström scaffolding.

This page records that unification in detail sufficient to:

1. Teach a future reader enough theory to implement a Peierls
   Nyström reference for a new geometry (e.g., the spherical case
   when Issue #100 is resolved, or a hypothetical 1-D Cartesian
   *box* with rectangular cross-section) by plugging four primitives
   (:math:`\mathrm d\Omega_d`, :math:`\kappa_d`, :math:`\rho_{\max}`,
   :math:`r'`) into the common scaffolding.
2. Explain why the chord form dominates textbook treatments even
   though it is awkward for pointwise Nyström — so future readers
   do not waste a cycle attempting the chord-form approach the
   Phase-4.2 investigation already ruled out for the cylinder.
3. Catalogue the row-sum identity, escape-probability deficit, and
   white-BC closure once across all three geometries rather than
   three separate times.

Nothing on this page replaces the geometry-specific pages in
:doc:`collision_probability`. Those pages document the full
derivation of each solver including its geometry-specific
investigation history (the cylinder's chord-form dead-end, the
slab's :math:`\Sigma_t`-LHS debugging, and so on). This page is
the *common scaffold* abstraction that sits behind all three
solvers.


Section 1 — The 3-D point kernel
================================

The starting point is the steady-state, monoenergetic (one-group),
isotropic-emission transport equation for the angular flux
:math:`\psi(\mathbf r,\mathbf\Omega)` inside a possibly
heterogeneous medium with total cross-section :math:`\Sigma_t(\mathbf r)`
and isotropic source :math:`q(\mathbf r)`:

.. math::

   \mathbf\Omega\cdot\nabla\psi(\mathbf r,\mathbf\Omega)
     + \Sigma_t(\mathbf r)\,\psi(\mathbf r,\mathbf\Omega)
     \;=\; \frac{q(\mathbf r)}{4\pi}.

Integrate along the characteristic
:math:`\mathbf r(s) = \mathbf r - s\,\mathbf\Omega` against the
integrating factor :math:`\exp[\int_0^s \Sigma_t(\mathbf r - t\mathbf\Omega)
\,\mathrm dt]`:

.. math::

   \psi(\mathbf r,\mathbf\Omega)
     \;=\; \frac{1}{4\pi}
       \int_{0}^{\infty}\!
         e^{-\tau(\mathbf r,\mathbf r-s\mathbf\Omega)}\,
         q(\mathbf r - s\mathbf\Omega)\,\mathrm ds,

where :math:`\tau(\mathbf r_1,\mathbf r_2) =
\int_{\mathbf r_2}^{\mathbf r_1} \Sigma_t(\mathbf s)\,\mathrm d\ell`
is the line-integrated optical path between two points. The scalar
flux :math:`\varphi(\mathbf r) = \int_{4\pi}\psi(\mathbf r,\mathbf\Omega)
\,\mathrm d\mathbf\Omega` is obtained by integrating
:math:`\psi` over :math:`\mathbf\Omega`. Using the substitution
:math:`\mathbf r' = \mathbf r - s\mathbf\Omega`, with Jacobian
:math:`\mathrm d^{3}r' = s^{2}\,\mathrm ds\,\mathrm d\mathbf\Omega`
(a 3-D *polar* parametrisation of :math:`\mathbf r'` centred on
:math:`\mathbf r`):

.. math::

   \varphi(\mathbf r)
     \;=\; \frac{1}{4\pi}\int_{\mathbb R^{3}}
       \frac{e^{-\tau(\mathbf r,\mathbf r')}}{|\mathbf r - \mathbf r'|^{2}}\,
       q(\mathbf r')\,\mathrm d^{3}r'.

This is the **3-D Peierls integral equation**, and identifies the
fundamental (pre-integration) kernel:

.. math::
   :label: peierls-point-kernel-3d

   G_{3\mathrm D}\bigl(|\mathbf r-\mathbf r'|,\tau\bigr)
     \;=\; \frac{e^{-\tau(\mathbf r,\mathbf r')}}
                 {4\pi\,|\mathbf r-\mathbf r'|^{2}}.

.. vv-status: peierls-point-kernel-3d documented

The two factors have distinct physical meanings that should never
be confused:

- The :math:`1/(4\pi|\mathbf r-\mathbf r'|^{2})` is the
  **inverse-square flux falloff** of a 3-D isotropic point emitter
  of unit strength: the total fluence passing through any spherical
  surface around the emitter is :math:`4\pi R^{2}\cdot(1/(4\pi R^{2}))
  = 1`, independent of radius. This reflects area dilution, not
  material attenuation.
- The :math:`e^{-\tau}` is the **uncollided attenuation** along
  the line-of-sight path, determined by the optical-path integral
  of :math:`\Sigma_t`. This is pure material absorption + out-scatter;
  it knows nothing about geometry.

Everything that follows — the :math:`E_1` slab kernel, the
:math:`\mathrm{Ki}_1` cylinder kernel, the bare :math:`e^{-\tau}`
sphere kernel — is a *dimensional reduction* of
:eq:`peierls-point-kernel-3d` obtained by integrating out one or two
symmetry directions. There is no independent derivation of the three
geometry kernels; they are projections of one master kernel onto three
symmetry-reduced submanifolds.


Section 2 — Dimensional reduction: :math:`E_1`, :math:`\mathrm{Ki}_1`, :math:`e^{-\tau}`
========================================================================================

We now specialise :eq:`peierls-point-kernel-3d` to each of the three
geometries. In each case we assume the emission density :math:`q`
depends only on the radial / axial coordinate of its geometry. This
lets us integrate out the symmetry directions *once and for all*,
producing a lower-dimensional Peierls equation in which :math:`q(r')`
and :math:`\Sigma_t(r')` appear at the natural radial coordinate.

Slab geometry — the :math:`E_1` kernel
---------------------------------------

A 1-D slab has translational symmetry in :math:`y` and :math:`z`.
If :math:`q` depends only on :math:`x`, integrate
:math:`G_{3\mathrm D}(|\mathbf r-\mathbf r'|)` over the two transverse
directions *at fixed* :math:`(x, x')`. Let :math:`\Delta x = x - x'`
and :math:`\rho_\perp = \sqrt{(y-y')^{2} + (z-z')^{2}}`. Writing
:math:`R^{2} = \Delta x^{2} + \rho_\perp^{2}` and letting
:math:`\tau = \Sigma_t |\Delta x|\cdot (R/|\Delta x|)` (valid because
a homogeneous slice of :math:`\Sigma_t` in :math:`x` depends only
on the projection):

.. math::

   G_{\rm slab}(|\Delta x|)
     \;=\; \int_{\mathbb R^{2}}\!\frac{e^{-\Sigma_t R}}{4\pi R^{2}}\,
           \mathrm dy'\,\mathrm dz'
     \;=\; \int_{0}^{\infty}\!\frac{e^{-\Sigma_t R}}{4\pi R^{2}}\,
           2\pi\rho_\perp\,\mathrm d\rho_\perp.

Substitute :math:`R = |\Delta x|\,t` with
:math:`\rho_\perp = |\Delta x|\,\sqrt{t^{2} - 1}`,
:math:`\rho_\perp\,\mathrm d\rho_\perp = |\Delta x|^{2}\,t\,\mathrm dt`,
:math:`t \in [1,\infty)`:

.. math::

   G_{\rm slab}(|\Delta x|)
     \;=\; \frac{1}{2}\int_{1}^{\infty}\!
           \frac{e^{-\Sigma_t |\Delta x| t}}{t}\,\mathrm dt
     \;=\; \frac{1}{2}\,E_1\!\bigl(\Sigma_t |\Delta x|\bigr),

using the Abramowitz–Stegun 5.1.4 definition of the exponential
integral:

.. math::
   :label: peierls-e1-derivation

   E_1(z) \;=\; \int_{1}^{\infty}\!\frac{e^{-zt}}{t}\,\mathrm dt
          \;=\; \int_{0}^{1}\!\frac{1}{\mu}\,e^{-z/\mu}\,\mathrm d\mu,

.. vv-status: peierls-e1-derivation documented

the second form coming from :math:`\mu = 1/t`. The slab scalar-flux
Peierls equation is therefore

.. math::

   \varphi(x) \;=\; \frac{1}{2}\int_{0}^{L}
     E_1\!\bigl(\tau(x,x')\bigr)\,q(x')\,\mathrm dx'
     \;+\; \varphi_{\rm bc}(x).

This is the form used by :mod:`orpheus.derivations.peierls_slab` and
documented in the :eq:`peierls-equation` section of
:doc:`collision_probability`. The singularity structure of the kernel
comes from the small-:math:`z` asymptote

.. math::

   E_1(z) \;=\; -\ln z - \gamma + z - \tfrac{z^{2}}{4} + \cdots,
   \qquad z \to 0^{+},

(A&S 5.1.11, [AbramowitzStegun1964]_), which is the log singularity
handled in :mod:`peierls_slab` by the singularity-subtraction /
product-integration recipe [Atkinson1997]_.

Cylinder geometry — the :math:`\mathrm{Ki}_1` kernel
----------------------------------------------------

An infinite cylinder has translational symmetry only in :math:`z`.
If :math:`q` depends only on the transverse radius :math:`r =
\sqrt{x^{2}+y^{2}}`, integrate
:math:`G_{3\mathrm D}(|\mathbf r-\mathbf r'|)` over the one axial
direction at fixed :math:`\mathbf r_\perp`:

.. math::

   G_{\rm cyl}(|\mathbf r_\perp-\mathbf r'_\perp|)
     \;=\; \int_{-\infty}^{\infty}\!
       \frac{e^{-\Sigma_t R}}{4\pi R^{2}}\,\mathrm dz',
     \qquad R \;=\; \sqrt{|\mathbf r_\perp-\mathbf r'_\perp|^{2} + z'^{2}}.

Let :math:`\rho = |\mathbf r_\perp-\mathbf r'_\perp|` and set
:math:`z' = \rho\sinh u`, :math:`R = \rho\cosh u`,
:math:`\mathrm dz' = \rho\cosh u\,\mathrm du`:

.. math::

   G_{\rm cyl}(\rho)
     \;=\; \frac{1}{4\pi\rho}\int_{-\infty}^{\infty}\!
           \frac{e^{-\Sigma_t \rho\cosh u}}{\cosh u}\,\mathrm du
     \;=\; \frac{1}{2\pi\rho}\int_{0}^{\infty}\!
           \frac{e^{-\Sigma_t \rho\cosh u}}{\cosh u}\,\mathrm du.

Using the A&S 11.2.1 definition in the form
:math:`\mathrm{Ki}_1(z) = \int_{0}^{\pi/2}\!e^{-z/\cos\theta}\,\mathrm d\theta`
and the substitution :math:`\cos\theta = 1/\cosh u`
(:math:`\mathrm d\theta = -\mathrm du/\cosh u`, ranges
:math:`\theta\in[0,\pi/2]\leftrightarrow u\in[0,\infty)`),

.. math::

   \int_{0}^{\infty}\!\frac{e^{-z\cosh u}}{\cosh u}\,\mathrm du
     \;=\; \int_{0}^{\pi/2}\!e^{-z/\cos\theta}\,\mathrm d\theta
     \;=\; \mathrm{Ki}_1(z),

so that

.. math::
   :label: peierls-ki1-derivation

   G_{\rm cyl}(\rho) \;=\; \frac{\mathrm{Ki}_1(\Sigma_t\rho)}{2\pi\rho}.

.. vv-status: peierls-ki1-derivation documented

The spot-check
:math:`\int_{0}^{\infty}\mathrm{Ki}_1(x)\,\mathrm dx = 1` follows
from the Fubini swap

.. math::

   \int_{0}^{\infty}\!\mathrm{Ki}_1(x)\,\mathrm dx
     \;=\; \int_{0}^{\pi/2}\!\!\int_{0}^{\infty}\!
       e^{-x/\cos\theta}\,\mathrm dx\,\mathrm d\theta
     \;=\; \int_{0}^{\pi/2}\!\cos\theta\,\mathrm d\theta
     \;=\; 1,

which is the :math:`\mathrm{Ki}_1` counterpart of the
:math:`\int_{0}^{\infty}E_1(x)\,\mathrm dx = 1` (A&S 5.1.32) that
underwrites the slab row-sum identity. The corresponding cylinder
scalar-flux Peierls equation is

.. math::

   \varphi(\mathbf r_\perp)
     \;=\; \frac{1}{2\pi}\!\iint_{\rm disc}
       \frac{\mathrm{Ki}_1\!\bigl(\tau(\mathbf r_\perp,\mathbf r'_\perp)\bigr)}
            {|\mathbf r_\perp-\mathbf r'_\perp|}\,q(\mathbf r'_\perp)\,
       \mathrm d^{2}r'_\perp
     \;+\; \varphi_{\rm bc}(\mathbf r_\perp),

identical in structure to the 3-D expression but one dimension
lower. It is the form used by
:mod:`orpheus.derivations.peierls_cylinder` before the polar-form
pivot.

Sphere geometry — the bare exponential kernel
---------------------------------------------

A 3-D sphere with only radial symmetry has *no* translational symmetry
to integrate out. The 3-D point kernel
:eq:`peierls-point-kernel-3d` therefore enters the Peierls equation
directly, with no pre-integration:

.. math::

   G_{\rm sph}(|\mathbf r-\mathbf r'|)
     \;=\; \frac{e^{-\tau(\mathbf r,\mathbf r')}}{4\pi\,|\mathbf r-\mathbf r'|^{2}},

and the Peierls equation is

.. math::

   \varphi(\mathbf r) \;=\;
     \iiint_{\rm ball}\!\frac{e^{-\tau}}{4\pi\,|\mathbf r-\mathbf r'|^{2}}
       \,q(\mathbf r')\,\mathrm d^{3}r' \;+\; \varphi_{\rm bc}(\mathbf r).

This is the sphere's "bare exponential" kernel. It looks more
singular than the slab's :math:`E_1` (which has only a log
divergence) because the :math:`1/R^{2}` blow-up is a *volume*
singularity rather than a line singularity — but it is still
integrable against the radial volume element
:math:`4\pi r^{2}\,\mathrm dr` as long as the measurement point
sits in the interior.

The three summary kernels are tabulated below. Note that the
progression of dimensional reductions is *monotone in the number
of symmetry directions integrated out*: two for the slab
(translation in :math:`y,z`), one for the cylinder (translation in
:math:`z`), zero for the sphere (only rotation about the centre,
which does not correspond to a translation and therefore cannot be
used to reduce the point-kernel dimension).

.. list-table:: Dimensionally-reduced point kernels
   :header-rows: 1
   :widths: 12 18 28 28 14

   * - Geometry
     - Native :math:`d`
     - Pre-integrated kernel
     - Singularity at :math:`R\to0^{+}`
     - A&S ref
   * - Slab
     - 1
     - :math:`\tfrac{1}{2}E_1(\Sigma_t|\Delta x|)`
     - :math:`-\tfrac12\ln|\Sigma_t\Delta x|`
     - 5.1.4
   * - Cylinder
     - 2
     - :math:`\mathrm{Ki}_1(\Sigma_t\rho)/(2\pi\rho)`
     - :math:`1/(2\pi\rho)` times
       :math:`\mathrm{Ki}_1(0)=\pi/2`
     - 11.2
   * - Sphere
     - 3
     - :math:`e^{-\Sigma_t R}/(4\pi R^{2})`
     - :math:`1/(4\pi R^{2})`
     - (none — native kernel)


Section 3 — Observer-centred polar form and Jacobian cancellation
=================================================================

The dimensionally-reduced kernels above all share a common
algebraic structure:

.. math::

   G_d(R)
     \;=\; \frac{\text{(geometry-specific factor)}\cdot
                 \text{(kernel function)}(\Sigma_t R)}{R^{d-1}}.

where :math:`R = |r-r'|` is the centre-to-centre distance in the
native geometry (:math:`R = |\Delta x|` for slab, :math:`R = \rho`
for cylinder, :math:`R = |\mathbf r - \mathbf r'|` for sphere).

The critical observation is that **the** :math:`1/R^{d-1}` **factor
exactly cancels the** :math:`\rho^{d-1}` **factor of the polar
volume element centred at the observer**. Write the volume element
in spherical (:math:`d=3`), polar (:math:`d=2`) or linear
(:math:`d=1`) coordinates centred at the observer:

.. list-table:: Polar volume elements by dimension
   :header-rows: 1
   :widths: 8 28 16 22 26

   * - :math:`d`
     - :math:`\mathrm d\Omega_d`
     - :math:`S_d \equiv \int \mathrm d\Omega_d`
     - Range
     - :math:`\mathrm dV'_d`
   * - 1
     - :math:`\mathrm d\mu`
     - 2
     - :math:`\mu \in [-1, 1]`
     - :math:`\mathrm d\mu\,\mathrm d\rho`
   * - 2
     - :math:`\mathrm d\beta`
     - :math:`2\pi`
     - :math:`\beta \in [0, 2\pi)`
     - :math:`\rho\,\mathrm d\rho\,\mathrm d\beta`
   * - 3
     - :math:`\sin\theta\,\mathrm d\theta\,\mathrm d\phi`
     - :math:`4\pi`
     - :math:`(\theta,\phi)\in[0,\pi]\times[0,2\pi)`
     - :math:`\rho^{2}\sin\theta\,\mathrm d\rho\,\mathrm d\theta\,\mathrm d\phi`

In all three cases, with :math:`R = \rho` (observer-centred
distance):

.. math::
   :label: peierls-polar-jacobian-cancellation

   G_d(\rho)\,\mathrm dV'_d
     \;=\; \frac{C_d\,\kappa_d(\Sigma_t\rho)}{\rho^{d-1}}\,
           \cdot\,\rho^{d-1}\,\mathrm d\Omega_d\,\mathrm d\rho
     \;=\; C_d\,\kappa_d(\Sigma_t\rho)\,
           \mathrm d\Omega_d\,\mathrm d\rho.

.. vv-status: peierls-polar-jacobian-cancellation documented

The :math:`\rho^{d-1}` of the volume element absorbs the
:math:`1/R^{d-1}` of the Green's function. After this cancellation,
the integrand of the Peierls equation contains only the scalar kernel
:math:`\kappa_d(\Sigma_t\rho)` times the source :math:`q(r')`
evaluated at :math:`r' = r'(\rho,\Omega,r)`, and a geometry-dependent
prefactor :math:`C_d`. Tabulating :math:`\kappa_d` and :math:`C_d`
against the three geometries:

.. list-table:: Polar-form kernel and prefactor by geometry
   :header-rows: 1
   :widths: 12 8 32 32 16

   * - Geometry
     - :math:`d`
     - Polar kernel :math:`\kappa_d(\tau)`
     - Prefactor :math:`C_d`
     - Smooth?
   * - Slab
     - 1
     - :math:`\tfrac12 E_1(\tau)`  *(a)*
     - :math:`1`
     - No — :math:`-\ln\tau`
   * - Cylinder
     - 2
     - :math:`\mathrm{Ki}_1(\tau) / (2\pi)` *(b)*
     - :math:`1`
     - Yes
   * - Sphere
     - 3
     - :math:`e^{-\tau} / (4\pi)`
     - :math:`1`
     - Yes

*(a)* For the slab the native variable is :math:`x' = x + \rho\mu`
with :math:`\rho \in [0,\rho_{\max}(x,\mu)]`; the kernel is taken along
a 1-D ray of cosine :math:`\mu` and therefore already carries a
:math:`1/|\mu|` factor from :math:`\mathrm dx' = |\mu|\,\mathrm d\rho`.
The form tabulated here absorbs that factor into the ρ-parametrisation:
see the unified equation tabulation in Section 4 for the
explicit :math:`1/|\mu|` bookkeeping.

*(b)* For the cylinder the formal prefactor reads
:math:`C_d\kappa_d = \mathrm{Ki}_1/(2\pi)`, but the azimuthal symmetry
:math:`\beta \to -\beta` is already exploited inside the module
:mod:`orpheus.derivations.peierls_cylinder`: the physical integration
range is folded from :math:`[0,2\pi]` to :math:`[0,\pi]` and the
remaining factor :math:`1/\pi` appears in front of the integral.
This is purely a halving of the work — mathematically equivalent to
integrating over the full :math:`[0,2\pi]` with :math:`1/(2\pi)`
prefactor.

The smoothness column is the single most important architectural
consequence. For :math:`d=2` and :math:`d=3` the integrand
:math:`\kappa_d(\Sigma_t\rho)\,q(r'(\rho,\Omega,r))` is **smooth on
its entire domain**: :math:`\kappa_d` is bounded and continuous,
:math:`r'` is smooth, :math:`q` is piecewise smooth (the only
kinks are the slope discontinuities at material boundaries, which
are handled by aligning radial panel breakpoints with annular
radii). Ordinary tensor-product Gauss–Legendre quadrature therefore
converges spectrally in :math:`(n_\Omega, n_\rho)`.

For :math:`d=1` the :math:`\rho^{d-1} = \rho^{0} = 1` volume factor
does *not* cancel a denominator because :math:`E_1(\tau)` has no
:math:`1/\rho` factor — the slab point kernel is already 1-D native.
The :math:`-\ln\tau` singularity of :math:`E_1` therefore remains,
and :mod:`peierls_slab` addresses it via product-integration
weights on the diagonal panel.

.. note::

   This does not mean the slab is "worse". It means the slab
   inherits its singularity from the native 1-D point kernel
   rather than from a Jacobian. The singularity structure is
   *mild* (logarithmic, integrable) precisely because the
   dimensional reduction from 3-D to 1-D has already stripped
   two :math:`1/R` factors out of the Green's function.


Section 4 — The unified Peierls equation
========================================

Collecting the preceding results gives the unified equation. Let
:math:`r` be the observer's radial coordinate (or :math:`x` in
Cartesian), :math:`\Omega` the angular variable on the unit
:math:`(d{-}1)`-sphere centred at the observer, and
:math:`\rho \ge 0` the ray distance from the observer. Define:

- :math:`r'(r,\rho,\Omega)`: the source position (projection of
  :math:`r - \rho\,\hat\Omega` back onto the geometry's native
  radial coordinate),
- :math:`\rho_{\max}(r,\Omega)`: the distance from the observer
  to the geometry boundary along the ray of direction :math:`\Omega`,
- :math:`\kappa_d(\tau)`: the polar-form kernel (Section 3 table),
- :math:`\widetilde\kappa(\rho)`: a :math:`\rho`-dependent factor that
  absorbs any non-cancelled :math:`\rho` dependence (needed only
  in Cartesian due to the :math:`\mu \leftrightarrow \rho` change
  of variables; equal to 1 for the curvilinear geometries),
- :math:`S_d`: the surface area of the unit :math:`(d{-}1)`-sphere
  (2, :math:`2\pi`, or :math:`4\pi`).

Then:

.. math::
   :label: peierls-unified

   \Sigma_t(r)\,\varphi(r)
     \;=\; \frac{\Sigma_t(r)}{S_d}
     \int_{\Omega_d}\!\mathrm d\Omega
     \int_{0}^{\rho_{\max}(r,\Omega)}\!
       \kappa_d(\Sigma_t\rho)\,\widetilde\kappa(\rho)\,
       q\bigl(r'(\rho,\Omega,r)\bigr)\,\mathrm d\rho
     \;+\; S_{\rm bc}(r).

.. vv-status: peierls-unified documented

The LHS is written in **identity-form** (coefficient on
:math:`\varphi` is :math:`\Sigma_t`, not the identity matrix) for
consistency with the generalised eigenvalue problem used by all
three Nyström drivers; see :eq:`peierls-equation` and the warning
at :ref:`peierls-conservation` for why this matters.

The three geometry-specific instantiations of :eq:`peierls-unified`
are:

.. list-table:: Geometry-specific pieces of the unified Peierls equation
   :header-rows: 1
   :widths: 10 8 16 24 22 20

   * - Geometry
     - :math:`d`
     - :math:`\int\mathrm d\Omega`
     - :math:`\kappa_d(\Sigma_t\rho)\,\widetilde\kappa`
     - :math:`r'(\rho,\Omega,r)`
     - :math:`\rho_{\max}(r,\Omega)`
   * - Slab *(a)*
     - 1
     - :math:`\int_{-1}^{1}\mathrm d\mu`
     - :math:`\tfrac12\, e^{-\Sigma_t\rho}/|\mu|`
     - :math:`x + \rho\mu`
     - :math:`(L-x)/\mu` (:math:`\mu>0`),
       :math:`x/|\mu|` (:math:`\mu<0`)
   * - Cylinder
     - 2
     - :math:`\int_{0}^{2\pi}\mathrm d\beta`
     - :math:`\mathrm{Ki}_1(\Sigma_t\rho)/(2\pi)`
     - :math:`\sqrt{r^{2}+2r\rho\cos\beta+\rho^{2}}`
     - :math:`-r\cos\beta + \sqrt{r^{2}\cos^{2}\beta + R^{2} - r^{2}}`
   * - Sphere
     - 3
     - :math:`\int_{0}^{\pi}\sin\theta\,\mathrm d\theta`
       (azimuth pre-integrated)
     - :math:`e^{-\Sigma_t\rho}/(4\pi)\cdot 2\pi`
     - :math:`\sqrt{r^{2}+2r\rho\cos\theta+\rho^{2}}`
     - :math:`-r\cos\theta + \sqrt{r^{2}\cos^{2}\theta + R^{2} - r^{2}}`

*(a)* For the slab, the unified form contains a
:math:`1/|\mu|` factor because the slab's native source coordinate
is :math:`x'`, not :math:`\rho`; the change of variables
:math:`\mathrm dx' = |\mu|\,\mathrm d\rho` is absorbed into
:math:`\widetilde\kappa`. The equivalent 1-D form with
:math:`\rho` replaced by the physical :math:`|\Delta x|`
recovers the familiar
:math:`\varphi(x) = \tfrac12\int E_1(\tau)\,q(x')\,\mathrm dx'`
after combining the :math:`|\mu|`-integral into :math:`E_1`
via :math:`E_1(z) = \int_0^1 \tfrac{1}{\mu}e^{-z/\mu}\,\mathrm d\mu`.
See the derivation at :eq:`peierls-e1-derivation`.

**Cylinder and sphere share identical radial closed forms.**
Inspect the :math:`r'(\rho,\Omega,r)` column: the cylinder and
sphere rows have *identical* formulae, differing only in whether
the inplane angle is called :math:`\beta` or :math:`\theta`. Same
for :math:`\rho_{\max}`. This is the architectural lever for code
reuse: the cylinder's ``_rho_max`` and ``r_prime`` routines port
to the sphere with zero algebraic modification — only the surrounding
angular quadrature and kernel call change. A future
``peierls_sphere.py`` module can share ``_optical_depth_along_ray``,
``_rho_max`` (via rename), ``composite_gl_r``,
``_lagrange_basis_on_panels`` with
:mod:`orpheus.derivations.peierls_cylinder` essentially verbatim;
only the angular quadrature (Gauss–Legendre on
:math:`[0,\pi]` weighted by :math:`\sin\theta`, replacing the
uniform :math:`\mathrm d\beta`) and the kernel
(:math:`e^{-\tau}` replacing :math:`\mathrm{Ki}_1(\tau)`) differ.

The single source of *geometric* content in both
:math:`r'(\rho,\Omega,r)` and :math:`\rho_{\max}(r,\Omega)` is the
**law of cosines**:

.. math::

   r'^{2} \;=\; r^{2} + \rho^{2} + 2r\rho\,\hat r \cdot \hat\Omega,

with :math:`\hat r\cdot\hat\Omega = \cos\beta` (cylinder) or
:math:`\cos\theta` (sphere), where the angle is measured from the
*outward* radial direction at the observer. The identity is in each
case the same; what differs is whether the angular integral runs over
a circle (cylinder) or a full 2-sphere (sphere), and whether the
azimuthal part of the full 2-sphere can be pre-integrated analytically
(yes — because the integrand depends on :math:`\theta` only, the
azimuthal :math:`\phi` integral contributes a factor of :math:`2\pi`).


Section 5 — Why the literature doesn't write it this way
========================================================

The unified form above is mathematically equivalent to the
standard chord-form presentations in [Sanchez1982]_, [Hebert2020]_,
[Stamm1983]_, [BellGlasstone1970]_, [Carlvik1966]_ — they are the
same integral equation in two different coordinate systems. The
literature's preference for the chord form is not an oversight;
it reflects the historical context in which integral transport
was developed.

Historical reason 1 — flat-source region averaging
--------------------------------------------------

The *flat-source* collision-probability method assumes
:math:`q(r')` piecewise constant. Under that assumption the
Peierls equation reduces to a :math:`P_{ij}` matrix whose elements
are *double integrals* over pairs of regions:

.. math::

   P_{ij} \;=\; \frac{1}{V_i}\int_{V_i}\!\mathrm dV
     \int_{V_j}\!G(|\mathbf r-\mathbf r'|)\,\mathrm dV'.

If both volumes are expressed in chord coordinates, the inner
integrand becomes a function of chord parameters only, and the
:math:`\mathrm{Ki}_n` and :math:`E_n` recurrence integrals collapse
the double integral to a **second-difference formula** in
:math:`\mathrm{Ki}_3` or :math:`E_3`:

.. math::

   P_{ij} \;\propto\; \Delta^{2}[\mathrm{Ki}_3]\bigl(\tau_{\rm gap},
                                              \tau_i, \tau_j\bigr),

as documented in :eq:`second-diff-general`, :eq:`self-slab`,
:eq:`second-diff-cyl`, and :eq:`second-diff-sph` of
:doc:`collision_probability`. This reduction is the *selling point*
of the flat-source CP method: all the angular and geometric
integration collapses to one evaluation per pair of regions
([Carlvik1966]_, [Stamm1983]_ §6.4). The polar form does not give
this closed-form reduction; it requires numerical quadrature
over :math:`(\rho,\Omega)`, which was prohibitive before modern
computing.

Historical reason 2 — pre-computer kernel tables
------------------------------------------------

[Bickley]_ and subsequent authors tabulated :math:`\mathrm{Ki}_n`
values at selected :math:`\tau` for hand-calculation of CP matrices.
The chord form permits the user to read :math:`\mathrm{Ki}_3(\tau)`
off a table and multiply by a small number of geometric factors
:math:`(V_i, V_j, y)` to produce :math:`P_{ij}`. The polar form
asks for :math:`\mathrm{Ki}_1` evaluated at :math:`O(n_\beta\cdot
n_\rho)` different :math:`\tau` values per observer — possible in a
computer, impossible by hand.

Historical reason 3 — the flat-source Jacobian singularity is
"invisible" under region averaging
-------------------------------------------------------------

The chord form carries a Jacobian
:math:`1/\sqrt{(r^{2}-y^{2})(r'^{2}-y^{2})}` (for the cylinder;
see the derivation at :eq:`peierls-cylinder-green-2d` in
:doc:`collision_probability`) with **two coincident
integrable singularities** at :math:`y=r` and :math:`y=r'`.
Pointwise this Jacobian is a numerical nightmare.  But when the
integrand is integrated over the full region (as it is in flat-source
CP), the Jacobian pairs with the region's :math:`y`-range to produce
finite :math:`\mathrm{Ki}_3` second-difference formulae. The
singularity is integrated out analytically before the pain can hit
numerics. No flat-source CP paper ever has to deal with a numerical
Jacobian singularity — the closed form eats it.

Why the polar form wins for pointwise Nyström
---------------------------------------------

The Peierls Nyström reference solvers in this project are *not*
flat-source. Their whole point is to verify the CP flat-source
discretisation *from outside* the flat-source assumption, with a
high-order polynomial source representation. The sequence of
arguments above therefore inverts:

- The second-difference closed form disappears (no region averaging),
  so there is no incentive to keep the chord coordinates. The
  :math:`\mathrm{Ki}_n` recurrence is bypassed entirely.
- The kernel is evaluated by direct numerical integration
  (:func:`~orpheus.derivations._kernels.ki_n_mp`) at mpmath precision;
  no tables needed.
- The Jacobian singularity at :math:`y=r` becomes a *pointwise*
  numerical obstacle rather than something that integrates out.

The polar form resolves all three. The :math:`1/R^{d-1}` of the kernel
cancels *before* integration (Section 3); the kernel
:math:`\kappa_d(\tau)` is a bounded, smooth function of a single
argument; and the geometry gets into the integrand only through
the (closed-form, algebraic) :math:`r'(\rho,\Omega,r)` and
:math:`\rho_{\max}(r,\Omega)`. Nothing is lost — the two forms are
mathematically equivalent — and the numerics becomes a simple
tensor-product Gauss–Legendre problem.

This is the root of the Phase-4.2 cylinder pivot (see the warning
admonition near :eq:`peierls-cylinder-green-2d` in
:doc:`collision_probability` for the specific Jacobian-factor
regression the chord form would have inflicted).


Section 6 — Unified Nyström architecture (code map)
===================================================

The preceding sections motivate the architectural claim that a
single Nyström scaffolding handles all three geometries. This
section makes the claim concrete by listing the abstract operations
and mapping each one to actual code in
:mod:`orpheus.derivations.peierls_cylinder`. The same map applies
to :mod:`orpheus.derivations.peierls_slab` (with kernel and
angular-quadrature substitutions) and to a future
``orpheus.derivations.peierls_sphere``.

Abstract operations
-------------------

1. **Angular quadrature** :math:`(\Omega_k, w_{\Omega,k})`.
   Gauss–Legendre on :math:`\mu \in [-1,1]` for the slab (cosine is
   uniform there); GL on :math:`\beta \in [0,\pi]` for the cylinder
   (:math:`\beta \to -\beta` folds :math:`[0,2\pi]` to :math:`[0,\pi]`);
   GL on :math:`\theta \in [0,\pi]` with explicit :math:`\sin\theta`
   weight for the sphere.

2. **Per-ray radial quadrature**
   :math:`(\rho_m, w_{\rho,m}(r_i,\Omega_k))`. Gauss–Legendre on
   :math:`[0,\rho_{\max}(r_i,\Omega_k)]` with per-(i, k) remap from
   the reference :math:`[-1,1]` interval. Uniform GL order
   :math:`n_\rho` gives uniform relative accuracy regardless of ray
   length; a single fixed :math:`\rho`-grid would over-resolve short
   radial rays and under-resolve long tangent-grazing rays.

3. **Source position** :math:`r'_{ikm} = r'(\rho_m,\Omega_k,r_i)`.
   One line of closed-form algebra per geometry (Section 4 table).

4. **Optical-depth walker**
   :math:`\tau(r_i,\Omega_k,\rho_m) = \int_{0}^{\rho_m}
   \Sigma_t(r'(s,\Omega_k,r_i))\,\mathrm ds`. Walks annular boundary
   crossings by solving a quadratic (curvilinear) or a linear
   equation (slab) for each :math:`r_k`, sorts the crossings in
   :math:`(0, \rho_m)`, accumulates :math:`\Sigma_{t,k}\,\Delta s`.
   This walker is **geometry-independent in structure** — all three
   geometries see the same sort-and-accumulate pattern — though the
   algebra for finding crossings differs (linear for the slab,
   quadratic for both curvilinear cases).

5. **Source interpolation**
   :math:`q(r'_{ikm}) = \sum_j L_j(r'_{ikm})\,q_j`. Lagrange basis
   on the panel containing :math:`r'_{ikm}`. **Geometry-independent**:
   the composite-GL panel structure is identical across geometries
   because the radial coordinate is one-dimensional in all three cases.

6. **Kernel assembly**

   .. math::

      K_{ij} \;=\; \frac{\Sigma_t(r_i)}{S_d}\sum_{k,m}
        w_{\Omega,k}\,w_{\rho,m}(r_i,\Omega_k)\,
        \kappa_d(\Sigma_t\rho_m)\,L_j(r'_{ikm}).

   The kernel :math:`\kappa_d` is the only geometry-specific
   *scalar* function. The prefactor :math:`1/S_d` absorbs the
   angular-measure normalisation (and the additional factor of 2
   from the cylinder's :math:`\beta \to -\beta` fold).

7. **Eigenvalue power iteration**. For a fixed scalar flux guess
   :math:`\varphi^{(n)}`, compute the fission source
   :math:`B\varphi^{(n)} = K\,\mathrm{diag}(\nu\Sigma_f)\,\varphi^{(n)}`,
   solve the within-group system
   :math:`A\,\varphi^{(n+1)} = B\varphi^{(n)}/k^{(n)}` where
   :math:`A = \mathrm{diag}(\Sigma_t) - K\,\mathrm{diag}(\Sigma_s)`,
   update :math:`k^{(n+1)}` from the fission-source norm ratio, and
   iterate. Completely geometry-agnostic.

8. **White-BC closure**. Rank-1 correction :math:`K_{\rm bc} =
   u\otimes v` that adds to :math:`K` before power iteration. The
   outer factor :math:`u_i` is proportional to
   :math:`\Sigma_t(r_i)\,G_{\rm bc}(r_i)`; the inner factor :math:`v_j`
   is proportional to :math:`r_j^{d-1}\,w_j\,P_{\rm esc}(r_j)`,
   where :math:`r_j^{d-1}` is the radial volume factor for the
   geometry (:math:`1` for slab, :math:`r_j` for cylinder,
   :math:`r_j^{2}` for sphere). See Section 8 for the derivation
   and the approximation-level caveat.

Which operations are geometry-specific?
---------------------------------------

The table below audits each operation against a "reusable across
all three geometries" criterion. An ✱ marks operations that are
geometry-specific; all other operations are *verbatim* portable.

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Operation
     - Slab
     - Cylinder
     - Sphere
     - Reusable?
   * - Composite-GL radial quadrature (``composite_gl_r``)
     - same
     - same
     - same
     - yes
   * - Lagrange basis on panels
     - same
     - same
     - same
     - yes
   * - Angular quadrature ✱
     - GL :math:`[-1,1]`
     - GL :math:`[0,\pi]` + fold
     - GL :math:`[0,\pi]`, :math:`\sin\theta` wt
     - no
   * - Kernel function :math:`\kappa_d` ✱
     - :math:`E_1(\tau)`
     - :math:`\mathrm{Ki}_1(\tau)`
     - :math:`e^{-\tau}`
     - no
   * - :math:`\rho_{\max}(r,\Omega)` ✱
     - linear in :math:`\mu`
     - quadratic
     - quadratic
     - cyl/sphere share
   * - :math:`r'(r,\rho,\Omega)` ✱
     - :math:`x + \rho\mu`
     - law of cosines
     - law of cosines
     - cyl/sphere share
   * - Optical-depth walker
     - sort crossings
     - sort crossings
     - sort crossings
     - yes (structure)
   * - Power iteration
     - same
     - same
     - same
     - yes
   * - Vacuum-BC closure
     - :math:`S_{\rm bc}=0`
     - :math:`S_{\rm bc}=0`
     - :math:`S_{\rm bc}=0`
     - yes
   * - White-BC rank-1 closure (approx) ✱
     - rank-2 (two faces)
     - rank-1
     - rank-1
     - no

Four non-trivial differences (angular quadrature, kernel, :math:`r'`,
:math:`\rho_{\max}`) and a slab-specific white-BC generalisation
(rank-2 because of two discrete faces). Everything else — the
optical-depth walker's structure, the Lagrange machinery, the
composite-GL panelling, the power iteration — is identical.

Proposed Protocol (future refactor)
-----------------------------------

A concrete way to factor the common scaffolding is a small
``Protocol`` that names the geometry-specific primitives:

.. code-block:: python

   from typing import Protocol
   import numpy as np

   class PeierlsGeometry(Protocol):
       """One-dimensional Peierls reference geometry.

       Implementations supply four primitives; the common Nyström
       scaffolding consumes them via this Protocol.
       """

       d: int         # intrinsic dimension: 1, 2, or 3
       S_d: float     # angular-measure normalisation: 2, 2π, or 4π

       def angular_quadrature(
           self, n: int
       ) -> tuple[np.ndarray, np.ndarray]:
           """Return (Ω-nodes, Ω-weights) matched to d and the
           kernel's azimuthal fold.
           """
           ...

       def rho_max(self, r: float, omega: float, R: float) -> float:
           """Distance from observer at r to the geometry boundary
           along angular direction omega.
           """
           ...

       def source_position(
           self, r: float, rho: float, omega: float,
       ) -> float:
           """Native radial coordinate of the source at distance rho
           from the observer at angular direction omega.
           """
           ...

       def kernel(self, tau: float) -> float:
           """Polar-form kernel κ_d(τ). Geometry-specific."""
           ...

A single scaffolding function

.. code-block:: python

   def build_peierls_kernel(
       geometry: PeierlsGeometry,
       r_nodes: np.ndarray,
       panel_bounds: list,
       radii: np.ndarray,
       sig_t: np.ndarray,
       n_omega: int, n_rho: int, dps: int,
   ) -> np.ndarray: ...

could then drive all three Peierls references, with geometry-specific
logic confined to three concrete ``PeierlsGeometry`` implementations.
The :func:`~orpheus.derivations.peierls_cylinder.build_volume_kernel`
already reads as such a scaffold minus one indirection — the
kernel :math:`\mathrm{Ki}_1` and the :math:`\rho_{\max}` formula
are hard-coded where they could accept injected callables. This
refactor is planned follow-up work; the Protocol sketch above
exists to guide the implementation rather than to commit to any
particular API shape.

.. note::

   The current code does *not* implement this Protocol. Both
   :mod:`peierls_slab` and :mod:`peierls_cylinder` inline the
   geometry-specific pieces. The refactor is a known improvement
   goal, tracked as a follow-up; the important point from this
   section is that the Protocol *exists conceptually* — the
   common scaffolding is real, and the cost of adding a third
   geometry is limited to the four primitives named above plus
   a couple of glue lines.


Section 7 — Row-sum identities, geometry-by-geometry
====================================================

The row-sum identity is the single most diagnostic
self-consistency check for any Peierls Nyström operator. It
isolates the prefactor :math:`1/S_d`, the kernel normalisation,
and the quadrature against the *multiplicative-factor class of
bugs* (wrong :math:`\pi` factor, wrong fold factor, wrong
angular-measure normalisation). Those bugs otherwise only show up
as a biased eigenvalue, at which point the combinatorial
search space of "which of seven prefactors did I miscount?" is
brutal.

The unified form gives the row-sum identity once, for all three
geometries.

Unified identity
----------------

Consider an infinite medium with constant :math:`\Sigma_t`. The
pure-scatter, spatially uniform flux solution is
:math:`\varphi \equiv 1`, which implies :math:`q = \Sigma_t\cdot 1
= \Sigma_t`. Substituting into :eq:`peierls-unified`:

.. math::

   \Sigma_t \cdot 1 \;=\; \frac{\Sigma_t}{S_d}
     \int_{\Omega_d}\!\mathrm d\Omega
     \int_{0}^{\infty}\!\kappa_d(\Sigma_t\rho)\,
     \underbrace{\Sigma_t}_{q(r')}\,\mathrm d\rho.

Change variables :math:`u = \Sigma_t\rho`,
:math:`\mathrm du = \Sigma_t\,\mathrm d\rho`:

.. math::

   1 \;=\; \frac{1}{S_d}\int_{\Omega_d}\!\mathrm d\Omega
     \int_{0}^{\infty}\!\kappa_d(u)\,\mathrm du.

For the curvilinear geometries, the integrand is
:math:`\Omega`-independent, so the :math:`\Omega`-integral yields
:math:`S_d` and the identity reduces to

.. math::

   \int_{0}^{\infty}\!\kappa_d(u)\,\mathrm du \;=\; 1
   \qquad (d=2, 3).

Plugging :math:`\kappa_2 = \mathrm{Ki}_1/(2\pi)` and
:math:`\kappa_3 = e^{-u}/(4\pi) \cdot 2\pi = e^{-u}/2` gives

- :math:`\int_0^\infty \mathrm{Ki}_1(u)/(2\pi)\,\mathrm du \cdot 2\pi
  = \int_0^\infty \mathrm{Ki}_1(u)\,\mathrm du = 1` ✓
  (Section 2 spot-check)
- :math:`\int_0^\infty e^{-u}/2\,\mathrm du \cdot 2 \cdot 2\pi / (4\pi)
  = \int_0^\infty e^{-u}\,\mathrm du = 1` ✓

For the slab, plugging the explicit angular integrand
:math:`e^{-\Sigma_t\rho}/(2|\mu|)` and using
:math:`\int_0^1(1/|\mu|)e^{-u/|\mu|}\,\mathrm d\mu\cdot\mathrm du = E_1(u)\,\mathrm du`:

.. math::

   \int_0^\infty\!\!\int_{-1}^{1}\!\frac{e^{-u}}{2|\mu|}\,\mathrm d\mu\,\mathrm du
     \;=\; \int_0^\infty E_1(u)\,\mathrm du \;=\; 1
   \quad\text{(A\&S 5.1.32)}\;\;✓

The infinite-medium identity therefore closes *for all three
geometries*:

.. math::

   \sum_{j=1}^{N} K_{ij}\cdot\Sigma_t(r_j) \;=\; \Sigma_t(r_i)
   \qquad (R \to \infty).

This is **not** the naive :math:`K\cdot\mathbf 1 = \Sigma_t`: see
the :ref:`peierls-cylinder-row-sum` analysis and the associated
warning at :eq:`peierls-cylinder-row-sum-identity` for the
multi-region case, where the change of variables :math:`u = \tau(\rho)`
picks up a :math:`1/\Sigma_t(r'(u))` factor that only cancels when
the source is :math:`q = \Sigma_t` (not :math:`q = 1`). The same
caution applies verbatim to sphere and slab: multi-region row sums
must be evaluated against :math:`q_j = \Sigma_t(r_j)`, not against
:math:`\mathbf 1`.

Finite-cell deficit
-------------------

For a finite geometry with vacuum BC, the row-sum identity picks
up a deficit equal to the **uncollided escape probability**:

.. math::

   \Sigma_t(r_i) - \sum_{j=1}^{N} K_{ij}\,\Sigma_t(r_j)
     \;=\; \Sigma_t(r_i)\,P_{\rm esc}(r_i),

with

.. math::

   P_{\rm esc}(r_i) \;=\; \frac{1}{S_d}
     \int_{\Omega_d}\!\mathrm d\Omega\,
     \widetilde\kappa_d\!\bigl(\tau(r_i,\Omega,\rho_{\max})\bigr),

where :math:`\widetilde\kappa_d` is the *once-integrated* kernel:

- slab: :math:`\widetilde\kappa_1 = E_2`  (from
  :math:`\int_0^\infty E_1(u)\,\mathrm du|_{u_0}^{\infty}= E_2(u_0)`
  in 1-D)
- cylinder: :math:`\widetilde\kappa_2 = \mathrm{Ki}_2/(2\pi)`
  (similarly, from
  :math:`\mathrm{Ki}_2(z) = \int_{z}^{\infty}\mathrm{Ki}_1(u)\,\mathrm du`)
- sphere: :math:`\widetilde\kappa_3 = e^{-\tau}/(4\pi)\cdot 2\pi
  = e^{-\tau}/2`

Physical interpretation: the deficit is the fraction of source
neutrons that leak through the boundary uncollided. For a bare
cylinder of :math:`R = 10` MFP this deficit is :math:`<10^{-3}`
at observer radii :math:`r_i \le R/2`
(see :eq:`peierls-cylinder-row-sum-identity` and the quantitative
table in :ref:`peierls-cylinder-row-sum`); the same scaling
(deficit :math:`\sim e^{-R}`) applies to slab and sphere.


.. _peierls-vacuum-bc-analytical-references:

Vacuum-BC analytical flux references (2026-04-20 milestone)
-----------------------------------------------------------

The row-sum deficit above is one diagnostic form. A stronger
diagnostic — and the one that closes the **machine-precision
vacuum-BC verification milestone** — is a closed-form evaluation of
the flux

.. math::
   :label: peierls-vacuum-bc-flux

   \varphi_d(r) \;=\; \int_{\mathcal V}\!K_d(r, r')\,q(r')\,\mathrm dV'

for :math:`q \equiv 1` on a bare pure-absorber cell with vacuum BC.
Because vacuum BC factorises the boundary closure operator as
:math:`K_{\rm bc} = G\cdot R\cdot P` with reflection operator
:math:`R = 0`
(see :class:`~orpheus.derivations.peierls_geometry.BoundaryClosureOperator`),
the operator reduces to the volume kernel alone, and
:eq:`peierls-vacuum-bc-flux` is the only ground truth needed to gate
the full K matrix assembly to machine precision:

.. math::
   :label: peierls-vacuum-bc-row-sum-gate

   \sum_{j=1}^{N} K_{ij}\cdot 1 \;\stackrel{!}{=}\; \Sigma_t(r_i)\,\varphi_d(r_i).

Slab
~~~~

For a slab of thickness :math:`L`, the exact result follows from
integrating :math:`(1/2)E_1(\Sigma_t|x - x'|)` over :math:`[0, L]`
and using the recurrence :math:`\int E_1(\alpha u)\,\mathrm du =
(1/\alpha)[1 - E_2(\alpha u)]`:

.. math::
   :label: peierls-vacuum-bc-slab

   \varphi_{\rm slab}(x) \;=\; \frac{1}{2\,\Sigma_t}
     \Bigl[\,2 - E_2(\Sigma_t\,x) - E_2(\Sigma_t\,(L - x))\,\Bigr].

Implemented in
:func:`~orpheus.derivations.peierls_reference.slab_uniform_source_analytical`.
Closed form — zero adaptive integration.

Cylinder (infinite axial extent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using observer-centred 3-D coordinates with in-plane azimuth
:math:`\theta'` and polar angle :math:`\psi \in (0, \pi)` from the
axis, the ray to the exit has in-plane projection

.. math::

   L_{2D}(r, \theta') \;=\; -r\cos\theta'
                            + \sqrt{R^{2} - r^{2}\sin^{2}\theta'}

and 3-D length :math:`\rho_{\max} = L_{2D}/\sin\psi`. The point
kernel :math:`e^{-\Sigma_t\rho}/(4\pi\rho^{2})` cancels the
:math:`\rho^{2}` Jacobian; the out-of-plane :math:`\psi`-integral
reduces to Bickley :math:`\mathrm{Ki}_2` via the substitution
:math:`\phi = \pi/2 - \psi` and the Bickley definition
:math:`\mathrm{Ki}_n(x) = \int_0^{\pi/2}\cos^{n-1}\!\phi\,
e^{-x/\cos\phi}\,\mathrm d\phi`.  Result:

.. math::
   :label: peierls-vacuum-bc-cylinder

   \varphi_{\rm cyl}(r) \;=\; \frac{1}{\pi\,\Sigma_t}\!\int_0^\pi\!
     \Bigl[\,1 - \mathrm{Ki}_2\!\bigl(\Sigma_t\,L_{2D}(r,\theta')\bigr)\,\Bigr]
     \,\mathrm d\theta'.

Implemented in
:func:`~orpheus.derivations.peierls_reference.cylinder_uniform_source_analytical`
as a single adaptive :func:`mpmath.quad` over :math:`\theta'`. The
integrand is smooth on :math:`[0, \pi]` — no breakpoints needed.

**Sanity limits.**

- :math:`r = 0`: :math:`L_{2D} \equiv R`, giving
  :math:`\varphi_{\rm cyl}(0) = (1 - \mathrm{Ki}_2(\Sigma_t R))/\Sigma_t`.
- :math:`\Sigma_t R \to 0`: :math:`\mathrm{Ki}_2(x) \approx 1 - (\pi/2)x`,
  so :math:`\varphi_{\rm cyl}(0) \to (\pi/2)\,R` — the 3-D
  mean-chord through the axis of an infinite cylinder
  (cf. `Cauchy's theorem <https://en.wikipedia.org/wiki/Mean_chord_length>`_).
- :math:`\Sigma_t R \to \infty`: :math:`\mathrm{Ki}_2 \to 0` and
  :math:`\varphi \to 1/\Sigma_t` — the infinite-medium limit.

Sphere
~~~~~~

For a sphere of radius :math:`R`, spherical symmetry collapses the
azimuth analytically. The chord length from the observer at radius
:math:`r` in direction :math:`\mu = \cos\Theta` to the spherical
surface is

.. math::

   L_{\rm chord}(r, \mu) \;=\; -r\mu + \sqrt{R^{2} - r^{2}(1 - \mu^{2})}.

The inner :math:`\rho`-integral of the point kernel (whose
:math:`\rho^{2}` Jacobian cancels the :math:`1/\rho^{2}` factor)
yields :math:`(1 - e^{-\Sigma_t L_{\rm chord}})/\Sigma_t`, and the
remaining :math:`\mu`-integral is:

.. math::
   :label: peierls-vacuum-bc-sphere

   \varphi_{\rm sph}(r) \;=\; \frac{1}{2\,\Sigma_t}\!\left[\,
       2 - \int_{-1}^{1}\!\exp\!\Bigl(
           -\Sigma_t\bigl[-r\mu + \sqrt{R^{2} - r^{2} + r^{2}\mu^{2}}\bigr]
         \Bigr)\,\mathrm d\mu\,\right].

Implemented in
:func:`~orpheus.derivations.peierls_reference.sphere_uniform_source_analytical`
as a single adaptive :func:`mpmath.quad` over :math:`\mu`.

**Sanity limits.**

- :math:`r = 0`: :math:`L_{\rm chord} \equiv R`, giving
  :math:`\varphi_{\rm sph}(0) = (1 - e^{-\Sigma_t R})/\Sigma_t`.
- :math:`\Sigma_t R \to 0`: :math:`\varphi_{\rm sph}(0) \to R` — the
  3-D mean chord through the centre of a sphere.
- :math:`\Sigma_t R \to \infty`: :math:`\varphi \to 1/\Sigma_t`.

Verification gate
~~~~~~~~~~~~~~~~~

The row-sum identity :eq:`peierls-vacuum-bc-row-sum-gate` against
these three closed-form / semi-analytical references is exercised
at machine precision (rel tol :math:`10^{-10}`) by

- ``TestSlabKernelRowSum`` (slab, N = 48 nodes, p_order = 6),
- ``TestSphereKernelRowSum`` and ``TestCylinderKernelRowSum``
  (curvilinear, small-N with the unified adaptive primitive
  :func:`~orpheus.derivations.peierls_geometry.build_volume_kernel_adaptive`),

in :mod:`tests.derivations.test_peierls_reference`. Agreement at
:math:`10^{-10}` across three independently-constructed references
(slab closed-form :math:`E_2`, cylinder Bickley-:math:`\mathrm{Ki}_2`
quadrature, sphere :math:`\mu`-quadrature) together with the unified
adaptive K primitive exhausts the failure modes available to the
volume-kernel layer. Vacuum-BC verification is complete at the K
level, and the BC tensor-network expansion (Mark / Marshak DP_N /
albedo closures) proceeds on a verified foundation.

**References.** [BellGlasstone1970]_ Ch. 2 (chord integration for
cylindrical and spherical cells with isotropic volume sources);
[CaseZweifel1967]_ Ch. 3 (point-kernel volume integration).


.. _peierls-slab-white-bc-analytical:

White-BC analytical flux — slab (Wigner-Seitz identity)
-------------------------------------------------------

For a homogeneous pure-absorber slab :math:`[0, L]` with uniform unit
source :math:`S = 1` and Mark white BC on both faces:

.. math::
   :label: peierls-white-bc-slab

   \varphi_{\rm white}(x) \;\equiv\; \frac{1}{\Sigma_t}
   \qquad (\text{for all } x, \text{ any } L).

This is the **Wigner-Seitz exact equivalence**: white BC models the
cell embedded in an infinite symmetric lattice, so the cell cannot
lose neutrons through the boundary. For a pure absorber with uniform
source the balance equation collapses to the pointwise equilibrium
:math:`\Sigma_t\,\varphi = S = 1`, giving :math:`\varphi = 1/\Sigma_t`
independent of :math:`L`.

**Self-consistent derivation.** The slab Peierls integral equation
with white BC is

.. math::

   \varphi(x) \;=\; \tfrac{1}{2}\!\int_0^L E_1\!\bigl(\Sigma_t|x-x'|\bigr)\,
                    S(x')\,\mathrm d x'
              \;+\; 2\,J^{-}\,\bigl[E_2(\Sigma_t x) + E_2(\Sigma_t(L - x))\bigr],

with :math:`\psi_{\rm in} = 2\,J^{-}` (the half-range-isotropic
angular flux per unit inward partial current). The partial-current
balance at :math:`x = L` for uniform :math:`S = 1`, using the
antiderivative identity
:math:`\int_0^{\tau_L} E_2(u)\,\mathrm du = E_3(0) - E_3(\tau_L)
= \tfrac{1}{2} - E_3(\tau_L)`:

.. math::

   J^{+}(L) \;=\; \tfrac{1}{2\Sigma_t}\bigl(\tfrac{1}{2} - E_3(\Sigma_t L)\bigr)
              + 2\,E_3(\Sigma_t L)\,J^{-}(0).

With symmetry :math:`J^{-}(0) = J^{-}(L) = J^{-}` and the Mark closure
:math:`J^{-} = J^{+}`:

.. math::

   J^{-} \;=\; \frac{\tfrac{1}{2} - E_3(\Sigma_t L)}
                    {2\,\Sigma_t\,(1 - 2\,E_3(\Sigma_t L))}
          \;=\; \frac{1}{4\,\Sigma_t}

independent of :math:`\tau_L = \Sigma_t L` (numerator and denominator
share the factor :math:`1 - 2 E_3(\tau_L)`). Substituting
:math:`2 J^{-} = 1/(2\Sigma_t)` back into the integral equation
collapses the :math:`E_2` terms and leaves :math:`\varphi \equiv
1/\Sigma_t`.

Implemented in
:func:`~orpheus.derivations.peierls_reference.slab_uniform_source_white_bc_analytical`.

**History of the algebra bug.** Commit ``2538cfe`` shipped an
incorrect closed form

.. math::

   \varphi_{\rm wrong}(x) \;=\; \tfrac{1}{2\Sigma_t}\!\left[
     2 + (2\beta - 1)\bigl(E_2(\Sigma_t x) + E_2(\Sigma_t(L - x))\bigr)
   \right],
   \qquad \beta = \tfrac{1 - E_3(\tau_L)}{1 - 2 E_3(\tau_L)},

derived with :math:`J^{+}(L)|_{\rm vol} = \tfrac{1}{2\Sigma_t}(1 -
E_3(\tau_L))` — which uses the wrong antiderivative identity
(:math:`\int E_2 \neq 1 - E_3`, it is :math:`\tfrac{1}{2} - E_3`). The
accompanying fixed-point diagnostic agreed with the wrong formula to
:math:`10^{-39}` because the fixed-point iteration had the *same
bug*. The error was caught when the first-order :math:`K_{\rm bc}`
row-sum disagreed with the published formula by a factor of
:math:`\sim 2.2` — a concrete example of how "two independent
derivations agreeing at 1e-39" is worthless if both share a
factor-of-two algebra mistake. Re-derivation showed the algebraic
simplification :math:`(\tfrac{1}{2} - E_3)/(1 - 2 E_3) = \tfrac{1}{2}`
collapses :math:`J^{-}` to :math:`1/(4\Sigma_t)`, giving
:math:`\varphi \equiv 1/\Sigma_t` exactly — the Wigner-Seitz
identity for the uniform cell.

**Testing leverage.** Because :math:`\varphi_{\rm white}` is spatially
constant, it supports two precise tests of the Peierls white-BC
tensor-network machinery (see
:mod:`tests.derivations.test_peierls_reference`):

1. **Factor-level closed forms** (machine-precision gates in
   ``TestSlabPescClosedForm`` and ``TestSlabGbcClosedForm``):

   .. math::

      P_{\rm esc}^{\rm slab}(x_i) \;&=\; \tfrac{1}{2}\bigl[E_2(\Sigma_t x_i)
                                  + E_2(\Sigma_t (L - x_i))\bigr], \\
      G_{\rm bc}^{\rm slab}(x_i) \;&=\; 2\bigl[E_2(\Sigma_t x_i)
                                  + E_2(\Sigma_t (L - x_i))\bigr].

2. **Rank-1 first-order row-sum** (``TestSlabKbcStructure``, gated at
   :math:`10^{-5}` owing to the algebraic GL convergence on the face
   log-singularities of :math:`P_{\rm esc}`):

   .. math::

      \sum_j K_{\rm bc}[i, j] \;=\;
        \bigl(\tfrac{1}{2} - E_3(\Sigma_t L)\bigr)\bigl[E_2(\Sigma_t x_i)
                                     + E_2(\Sigma_t (L - x_i))\bigr].

The rank-1 Mark closure — in slab exactly as for cylinder and sphere —
is **intentionally approximate**: it omits the
:math:`T \cdot J^{-} = 2\,E_3(\tau_L)\,J^{-}` self-feedback
transmission term in the partial-current balance. Consequently
:math:`k_{\rm eff}` converges to :math:`k_{\infty}` only in the
optically-thick limit :math:`\Sigma_t L \to \infty`, not exactly at
any finite :math:`L` — the same behaviour as the curvilinear cell
white-BC eigenvalue (see ``TestWhiteBCThickLimit`` for cylinder /
sphere).

**Slab-polar BC machinery fix (Issue #118).** The
:func:`~orpheus.derivations.peierls_geometry.compute_P_esc` and
:func:`~orpheus.derivations.peierls_geometry.compute_G_bc` functions
were originally implemented for curvilinear geometries only. Slab
support landed with these fixes:

1. :func:`compute_P_esc` replaces the hard-coded
   ``np.cos(omega_pts)`` with the polymorphic
   :meth:`CurvilinearGeometry.ray_direction_cosine` (identity for
   slab-polar, :math:`\cos\Omega` for curvilinear), and adds a
   homogeneous-slab closed-form branch that evaluates the
   :math:`(1/2)(E_2 + E_2)` expression directly via
   :func:`mpmath.expint`.

2. :func:`compute_G_bc` gains an explicit ``slab-polar`` branch with
   the closed-form :math:`2(E_2 + E_2)` expression for homogeneous
   single-region slabs and a GL :math:`\mu`-quadrature fall-through
   for multi-region slabs.

3. :meth:`CurvilinearGeometry.radial_volume_weight` returns ``1`` for
   slab-polar (no geometric factor in the Cartesian volume element);
   :meth:`CurvilinearGeometry.rank1_surface_divisor` returns ``2``
   (two unit-area faces).

4. Rank-N :math:`(n > 0)` modes in
   :func:`~orpheus.derivations.peierls_geometry.compute_P_esc_mode`
   and :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_mode`
   raise :class:`NotImplementedError` for slab: rank-N slab requires
   per-face mode decomposition (:math:`A = \mathbb{R}^{2N}` rather
   than :math:`\mathbb{R}^N`), which is a scope extension tracked
   separately.

**References.** [Davison1957]_ Ch. 5 (half-range isotropic re-entry
partial-current balance); [CaseZweifel1967]_ Ch. 6 (albedo-1 BC
analytical solutions); Wigner & Seitz lattice-cell approximation.


Section 8 — White-BC closure, geometry-by-geometry
==================================================

The white (albedo-1, isotropic-reflection) boundary condition is
the most physically interesting case: it models an infinite
lattice by imposing :math:`J^{-}(S) = J^{+}(S)` on every boundary
point. For a 1-D symmetric geometry this collapses to a scalar
partial-current balance.

Rank-1 closure in curvilinear geometry
--------------------------------------

For a radially-symmetric cylinder or sphere, the
outgoing and incoming currents on the boundary are uniform by
symmetry. The white-BC closure therefore contributes an additive
correction

.. math::

   S_{\rm bc}(r_i) \;=\; J^{-}\,G_{\rm bc}(r_i),

where :math:`G_{\rm bc}(r_i)` is the uncollided surface-to-volume
Green's function — the scalar-flux contribution at observer
:math:`r_i` from a unit uniform isotropic inward surface current —
and :math:`J^{-}` is the scalar (spatially uniform) re-entering
partial current. Closing :math:`J^{-}` against the volume source via
the partial-current balance

.. math::

   J^{+} \;=\; \frac{1}{A_d}\sum_j A_j\,\Sigma_t(r_j)\,\varphi_j\,
                P_{\rm esc}(r_j),

where :math:`A_d` is the surface area of the cell
(:math:`4\pi R^{2}` for the sphere, :math:`2\pi R` per unit :math:`z`
for the cylinder) and :math:`A_j` is the radial volume element of
the :math:`j`-th node (:math:`r_j^{d-1} w_j` up to normalisation),
gives the **rank-1** correction

.. math::

   K_{\rm bc}[i, j] \;=\; \frac{\Sigma_t(r_i)\,G_{\rm bc}(r_i)}{A_d}
       \,\cdot\, r_j^{d-1}\,w_j\,P_{\rm esc}(r_j)
     \;=\; u_i\,v_j.

This is the form implemented in
:func:`~orpheus.derivations.peierls_cylinder.build_white_bc_correction`:
:math:`u_i = \Sigma_t(r_i)\,G_{\rm bc}(r_i)/R`,
:math:`v_j = r_j\,w_j\,P_{\rm esc}(r_j)`, because :math:`A_d = 2\pi R`
and :math:`A_j \propto r_j`. A future sphere version would use
:math:`u_i = \Sigma_t(r_i)\,G_{\rm bc}(r_i)/(4\pi R^{2})` and
:math:`v_j = r_j^{2}\,w_j\,P_{\rm esc}(r_j)`; the architectural
pattern is identical.

Rank-2 closure in slab geometry
-------------------------------

The slab has **two discrete boundary faces** (:math:`x=0` and
:math:`x=L`), each with its own scalar partial current. The white
closure therefore carries two independent boundary unknowns
:math:`J^{-}_{0}, J^{-}_{L}`, each entering the volume via its
own :math:`G_{\rm bc}` kernel. The correction is rank-2 — an
outer product of 2-vectors — and the algebra is the geometric
series of multiple slab traversals, producing the
:math:`E_2\otimes E_2 + T\,E_2\otimes E_2` structure documented
at :eq:`peierls-white-bc`. It is not a special case of the rank-1
formula; it is what the rank-1 formula degenerates to when the
boundary manifold is 0-dimensional (two discrete points) rather
than a continuous manifold.

The pointwise rank-1 approximation — cylinder and sphere
---------------------------------------------------------

Rank-1 is the *correct* rank for the *partial-current* balance
alone (one scalar equation in one scalar unknown). But the rank-1
correction *assumes that the re-entering partial current
:math:`J^{-}` is isotropic in the incoming half-space*
(the "white" or "Mark" closure). For **flat-source** (region-averaged)
CP, that isotropy holds exactly: both :math:`J^{+}` and :math:`J^{-}`
are scalar quantities averaged over the full hemisphere, and the
rank-1 correction is exact. For **pointwise** Nyström with a
high-order polynomial representation of :math:`\varphi(r)`, the
outgoing angular distribution is anisotropic; imposing an
*isotropic* re-entering distribution breaks the pointwise balance
at the per-node level.

The deviation is a function of the cell's optical size. For a
homogeneous bare cylinder tested in
:func:`~orpheus.derivations.peierls_cylinder.build_white_bc_correction`:

.. list-table:: Rank-1 white-BC error (cylinder, 1-region, homogeneous)
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`R`/MFP
     - :math:`\max_i |K_{\rm tot}\cdot\mathbf 1 - \Sigma_t|`
     - :math:`k_{\rm white}(R)`
     - :math:`|k_{\rm white} - k_\infty|`
   * - 0.5
     - 0.32
     - –
     - (diverges — unphysical)
   * - 1.0
     - 0.16
     - 1.19
     - 21 %
   * - 2.0
     - 0.20
     - 1.40
     - 7 %
   * - 5.0
     - 0.12
     - 1.48
     - 2 %
   * - 10
     - :math:`<0.04`
     - 1.49
     - 1 %

The vacuum-BC driver remains bit-exact against the Sanchez tie-point
at any cell size. Tests that compare the Peierls cylinder reference
against CP (white BC) must use :math:`R \ge 5` MFP to keep the
closure error under 3 %.

Sphere — Issue #100 (retracted; historical record)
--------------------------------------------------

.. note::

   **2026-04-18 update — retraction.** The Phase-4.3 unified
   sphere Peierls implementation delivers **physically sensible
   rank-1 white-BC behaviour matching the cylinder**. The
   ":math:`k_{\rm eff} \approx 6.7`" datum and the
   "rank-1 fails structurally on the sphere" conclusion below
   are artefacts of the earlier attempt's missing :math:`R^{2}`
   surface divisor (the cylinder code was repurposed for the
   sphere without updating :math:`A_d = 2\pi R \to 4\pi R^{2}`).
   The corrected divisor is now dispatched by
   :meth:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.rank1_surface_divisor`.

   Current sphere rank-1 :math:`k_{\rm eff}` scan (bare sphere,
   :math:`\Sigma_t = 1`, :math:`\Sigma_s = 0.5`,
   :math:`\nu\Sigma_f = 0.75`, :math:`k_\infty = 1.5`):

   ======  ===================  ===============
   R/MFP   :math:`k_{\rm eff}`  err vs k_∞
   ======  ===================  ===============
   1.0     1.0963               26.9 %
   2.0     1.3914               7.2 %
   5.0     1.4897               0.7 %
   10.0    1.4957               0.3 %
   20.0    1.4945               0.4 %
   ======  ===================  ===============

   This **parallels the cylinder** (21 % at :math:`R=1` MFP,
   falling to 1 % at :math:`R=10` MFP): both geometries show the
   same inverse-cell-size growth of the rank-1 Mark closure
   error, which is a flat-source artefact reapplied at the
   pointwise level (Issue #103 / N1). The full retraction
   discussion and the R-vs-R² gotcha are archived in
   :doc:`collision_probability`, §
   :ref:`issue-100-retraction`. The text below is preserved as
   historical context — keeping the record of what was tried and
   why it failed prevents the same mistake from being made twice.

   Sphere Peierls is **shipped** as of Phase-4.3 (commits
   ``435c0b3``, ``9d03948``, ``cad2f0b``); the Peierls-vs-CP
   comparison at white-BC parity runs in
   ``tests/cp/test_peierls_sphere_flux.py``.

*Historical text (pre-correction):* The identical failure mode was
observed in the Phase-4.3 spherical Peierls attempt (GitHub Issue
#100). The sphere's uncollided escape probability
:math:`P_{\rm esc}(r)` varies from ~0.37 at the centre to ~0.68
at the surface, while the re-entry distribution
:math:`G_{\rm bc}(r)` varies from 0 at the centre (Davison's
:math:`u(0) = 0` constraint) to ~2.7 at the surface, and the
ratio is not constant — it varies by ~40 % across the sphere
radius. A rank-1 correction necessarily imposes a *constant*
ratio, so it over-shoots near the surface and under-shoots near
the centre, giving :math:`k_{\rm eff} \approx 6.7` for a 1-G
1-region case (expected :math:`k_\infty = 1.5`).

Both observations — the cylinder's size-dependent error and the
sphere's structural failure — are **the same phenomenon**: rank-1
is a flat-source result re-applied at the pointwise level. The
two paths forward (and open as of the session of this writing)
are:

(a) *Augmented Nyström system*: add the surface-current unknowns
    as additional degrees of freedom, promoting the
    :math:`(N\times N)` system to :math:`(N+n_{\rm surf})\times
    (N+n_{\rm surf})`. The rank of the white-BC block grows from 1
    to :math:`n_{\rm surf}`, which represents the angular
    resolution of the re-entering distribution.
(b) *Higher-rank angular decomposition*: resolve :math:`J^{-}` in
    a Mark-:math:`n`-like :math:`P_n` expansion of the re-entering
    hemisphere. Rank :math:`n+1` correction.

*Post-correction assessment (2026-04-18).* The "ratio varies by
40 %" argument above conflates two independent things: the rank-1
closure is an outer product :math:`u_i\,v_j` where :math:`u` and
:math:`v` can individually vary with radius. What the rank-1
closure approximates is the re-entering **angular distribution**
:math:`J^{-}(\Omega)` (treated as uniform isotropic by Mark),
**not** the :math:`(i, j)` coupling structure. A radius-dependent
ratio :math:`P_{\rm esc} / G_{\rm bc}` therefore does **not**
imply structural failure; it is absorbed into the outer-product
factorisation. What the rank-1 closure actually suffers from is
the Mark-closure error in the angular shape of :math:`J^{-}`, and
that error scales with cell optical thickness (thick cells
homogenise the angular distribution via multiple scattering, thin
cells do not). Path (a) and path (b) above are the correct
architectural fixes — Issue #103 (N1) tracks higher-rank
angular decomposition — but they apply **equally** to cylinder
and sphere. Neither is a sphere-specific blocker.


Rank-N (Marshak / Gelbard DP\ :sub:`N-1`) skeleton — WIP
--------------------------------------------------------

A rank-:math:`N` extension to the white-BC closure is under
construction at
:func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction_rank_n`.
The canonical form (Sanchez & McCormick 1982 §III.F.1
Eqs. 165–169; Gelbard 1968; Stepanek 1981) expands the half-range
surface angular flux in shifted Legendre polynomials
:math:`\tilde P_n(\mu_s) = P_n(2\mu_s - 1)` on
:math:`\mu_s \in [0, 1]` (inward hemisphere, orthogonality
:math:`\int_0^1 \tilde P_n \tilde P_m \,\mathrm d\mu =
\delta_{nm}/(2n+1)`), and imposes the Marshak per-mode equality
:math:`J^{-}_n = J^{+}_n` for :math:`n = 0, \ldots, N-1`. The
assembled correction is a sum of rank-1 outer products,

.. math::
   :label: peierls-rank-n-bc-closure

   K_{\rm bc} \;=\; \sum_{n=0}^{N-1} u_n \otimes v_n,
   \qquad
   u_n[i] = \frac{\Sigma_t(r_i)\,G_{\rm bc}^{(n)}(r_i)}{A_d},
   \qquad
   v_n[j] = (2n+1)\,r_j^{d-1}\,w_j\,P_{\rm esc}^{(n)}(r_j),

with :math:`P_{\rm esc}^{(n)}` and :math:`G_{\rm bc}^{(n)}` the
mode-:math:`n` Legendre-weighted versions of the rank-1
primitives. For :math:`n = 0`, :math:`\tilde P_0 \equiv 1` and
the mode-0 contribution is routed through the existing rank-1
:func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction`,
preserving bit-exact regression at
``rtol = 1e-14`` (gated by
``tests/derivations/test_peierls_rank_n_bc.py::test_rank1_bit_exact_recovery``).
The rank-1 **structural** decomposition
:math:`K_{\rm bc}(N) - K_{\rm bc}(N-1) = u_{N-1} \otimes v_{N-1}`
is also verified (``test_rank_n_cross_mode_diagonal``): each
successive rank increment is a single rank-1 outer product, which
is the content of Sanchez & McCormick's Eq. (167) reciprocity
after rotational symmetry collapses the surface-to-surface
mode-index matrix to diagonal.

**Current limitation** (Issue #112). The mode-:math:`n \ge 1`
*magnitudes* are geometry-dependent off from canonical:

- **Cylinder**: the current implementation weights the existing
  surface-centred :math:`\mathrm{Ki}_1/d` integrand by
  :math:`\tilde P_n(|\mu_{s,2D}|)` where
  :math:`|\mu_{s,2D}| = (R - r_i \cos\phi)/d` is the 2-D
  projected cosine. The canonical closure requires the 3-D
  cosine :math:`\mu_{s,3D} = \sin\theta_p \cdot \mu_{s,2D}` with
  the :math:`\theta_p` integration carried out *inside* the
  :math:`\tilde P_n`-weighted integrand (producing higher-order
  Bickley functions :math:`\mathrm{Ki}_{2+k}` per Knyazev 1993).
  The 2-D projection is exact for :math:`n = 0` (trivially
  :math:`\tilde P_0 \equiv 1`) but diverges from the 3-D
  canonical for :math:`n \ge 1`, making rank-:math:`N`
  non-monotone at thin :math:`R`. Thick-cell k\ :sub:`eff`
  drifts by 1–10 % for :math:`N \ge 2` at :math:`R = 10` MFP
  because the mode-:math:`n \ge 1` contributions are persistently
  O(0.2–0.6) of mode-0 magnitude rather than decaying as
  Lambertian predicts.

- **Sphere**: mode-1 is directionally correct (27 % → 15 %
  thin-cell error, mode magnitudes match Lambertian asymptotics
  ``|v_2 / v_0| ≈ 0.04``), but the convergence ladder plateaus at
  mode 2 rather than continuing. The current hypothesis is that
  the :math:`(2n+1)` factor sits on the wrong side of the u/v
  split, or an additional cosine weight is absorbed by the
  Gelbard basis that the naive transcription is missing. Sphere
  thick-cell also drifts by ~7 % at :math:`N = 2` for the same
  magnitude issue.

**Until Issue #112 lands** the 3-D cylinder quadrature and
sphere canonical DP\ :sub:`N` audit, the function is safe at the
default ``n_bc_modes = 1`` (byte-identical to the legacy
:func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction`).
Convergence-ladder tests
(``test_rank_n_row_sum_improves_thin_cell_*``,
``test_rank_n_thick_cell_unchanged``,
``test_rank_n_sphere_thin_cell_convergence``) carry ``xfail``
markers with diagnostic reasons referencing Issue #112 — they
flip to pass automatically when the full fix lands.

The :func:`~orpheus.derivations._kernels._shifted_legendre_eval`
utility (orthonormality and known-value tests in the same file)
is verified-correct and is the basis building block both the
cylinder 3-D quadrature and the sphere canonical DP\ :sub:`N`
audit will reuse unchanged.


.. _peierls-rank-n-jacobian-section:

Surface-to-observer Jacobian :math:`(\rho_{\max}/R)^2` — 2026-04-18 fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial rank-:math:`N` skeleton weighted the same angular
integrand :math:`\int W_\Omega(\Omega)\,e^{-\tau}\,\mathrm d\Omega`
by :math:`\tilde P_n(\mu_{\rm exit})` for BOTH
:math:`G_{\rm bc}^{(n)}` (the flux-per-unit-\ :math:`J^-_n`
response) and :math:`P_{\rm esc}^{(n)}` (the outgoing partial-
current moment per unit source). The numerics-investigator
uncovered an **identity** for the sphere,

.. math::

   G_{\rm bc}^{(n)}(r_i) \;=\; 4\,P_{\rm esc}^{(n)}(r_i)
   \qquad \text{(sphere only, every } r,\ n,\ R\text{)},

verified numerically to :math:`10^{-12}` precision across all
modes and all cell radii. Both primitives share the **same**
observer-centred integrand
:math:`\int_{4\pi} \tilde P_n(\mu_s)\,e^{-\tau}\,\mathrm d\Omega`;
the ratio :math:`4` is the geometry-prefactor convention
(:math:`2` for :math:`G_{\rm bc}` vs :math:`0.5` for
:math:`P_{\rm esc}`).

**Consequence.** Each rank-1 outer product
:math:`u_n \otimes v_n` of the form

.. math::

   u_n[i]\,v_n[j] \;=\;
     \text{(scalar)} \cdot \tilde P_n(\mu_{\rm exit}(r_i)) \,\cdot\,
                             \tilde P_n(\mu_{\rm exit}(r_j))

is a **symmetric** :math:`\tilde P_n(r_i)\,\tilde P_n(r_j)`
factorisation — the same function evaluated at both indices.
Summed over :math:`n`, this gives a rank-:math:`N` matrix in the
:math:`\tilde P_n` basis, but all modes share the **same** basis
functions and the same integrand structure. No amount of scalar
re-weighting by :math:`(2n+1)` vs :math:`1` vs :math:`1/(2n+1)`
can recover a proper Marshak closure — verified empirically over
three α-scans in
``derivations/diagnostics/diag_rank_n_{06,07}_*.py``.

**The fix**. Derive the canonical outgoing partial-current moment
per unit volumetric source at :math:`r_i` from first principles.
For a uniform source :math:`q = 1`:

.. math::

   J^{+}_n \;=\; \frac{1}{A_d}\,\int_S \mathrm d A_s
                 \int_{\text{outward}} |\mu_{\rm out}|\,
                 \tilde P_n(\mu_{\rm out})\,\psi^{+}\,\mathrm d\Omega.

Convert the surface-plus-outward-hemisphere integral to an
**observer-centred** integral at the source point :math:`r_i`
using the 3-D projection Jacobian

.. math::

   \mathrm d A_s\,\mathrm d\Omega_{\rm out}\,|\mu_s|
   \;=\; d^{\,2}\,\mathrm d\Omega_{r_i},
   \qquad d = \rho_{\max}(r_i,\Omega)

(standard optics; the outgoing direction :math:`\Omega_{\rm out}`
at the surface is the reverse of the observer-centred direction
:math:`\Omega_{r_i}`). The :math:`|\mu_s|` cancels the
:math:`|\mu_{\rm out}|` cosine weight of the partial-current
moment, and the :math:`\mathrm d A_s\,\mathrm d\Omega_{\rm out}`
pair becomes :math:`\mathrm d\Omega_{r_i}\,d^{\,2}`:

.. math::
   :label: peierls-rank-n-jacobian-derivation

   J^{+}_n(r_i) \;=\; \frac{1}{A_d}\,
                      \int_\Omega \tilde P_n(\mu_{\rm exit})\,
                      d^{\,2}(r_i,\Omega)\,\psi^{+}\,\mathrm d\Omega
                 \;=\; \frac{1}{A_d}\,
                      \int_\Omega \tilde P_n(\mu_{\rm exit})\,
                      \rho_{\max}^{\,2}(r_i,\Omega)\,e^{-\tau}\,
                      \mathrm d\Omega.

Dividing by the cell's characteristic :math:`R^2` (to match the
:math:`A_d^{\rm divisor}` convention in
:meth:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.rank1_surface_divisor`,
which is :math:`R` for cylinder and :math:`R^2` for sphere) gives
the dimensionless factor :math:`(\rho_{\max}/R)^2` now carried in
:func:`~orpheus.derivations.peierls_geometry.compute_P_esc_mode`:

.. math::
   :label: peierls-rank-n-P-esc-moment

   P_{\rm esc}^{(n)}(r_i) \;=\; C_d\!\int_0^\pi W_\Omega(\Omega)\,
                    \Bigl(\tfrac{\rho_{\max}(r_i,\Omega)}{R}\Bigr)^{\!2}\,
                    \tilde P_n\!\bigl(\mu_{\rm exit}\bigr)\,
                    K_{\rm esc}(\tau)\,\mathrm d\Omega.

The :math:`(\rho_{\max}/R)^2` factor is **trivially** :math:`1`
at :math:`r = 0` in the sphere (where every ray has the same
exit distance :math:`\rho_{\max} \equiv R`) and only matters
off-centre. This is why the old rank-N skeleton produced
apparently-correct mode-0 behaviour at :math:`r = 0`: the
Jacobian factor is :math:`1` there, hiding the bug. The
investigator's :math:`G_{\rm bc}^{(n)} = 4\,P_{\rm esc}^{(n)}`
identity only breaks for :math:`r > 0` once the Jacobian is
restored.

**Empirical impact** (bare homogeneous 1G 1-region white-BC,
:math:`k_\infty = 1.5`):

.. note::

   **Retraction (2026-04-25, Issue #132).** The 1R "rank-N
   converges" headline below is a **calibration artifact**, not
   evidence that the rank-:math:`N` Marshak closure is correct on
   Class B. The numerical values are reproducible — they were
   measured correctly at the BASE preset — but they are produced by
   a hybrid mode-0 / mode-:math:`n \ge 1` routing in
   :func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
   (peierls_geometry.py:3618-3642) where mode 0 uses the *legacy*
   :func:`compute_P_esc` / :func:`compute_G_bc` (no
   :math:`(\rho_{\max}/R)^2` Jacobian) while modes :math:`n \ge 1`
   use the *canonical* Jacobian-weighted
   :func:`compute_P_esc_mode` / :func:`compute_G_bc_mode`. The two
   normalisations live in **different partial-current expansion
   spaces**. The 1R rank-2 = -1.10 % sphere result lands near
   :math:`k_\infty` only because the legacy mode-0 was historically
   calibrated to make rank-1 Mark approximately right, and the
   residual error from adding a single mismatched mode-1 happens
   to be small for this configuration. **The same closure produces
   :math:`+57\,\%` k\ :sub:`eff` sign-flip catastrophes on Class B
   2R configurations** (sphere 1G, fuel-A inner /
   moderator-B outer). See
   :ref:`peierls-rank-n-class-b-mr-mg-falsification` below for the
   MR×MG falsification table, the Probe G LEGACY-vs-CANONICAL
   evidence, and the Issue #132 re-derivation paths. **Do not
   cite the 1G/1R rank-N table as evidence that the rank-:math:`N`
   Marshak closure works on Class B.**

.. list-table:: Rank-:math:`N` :math:`k_{\rm eff}` error, pre- and post-fix (1G/1R only — see retraction above)
   :header-rows: 1
   :widths: 10 10 15 15 15 15

   * - Geom
     - :math:`R`
     - N=1 err
     - N=2 pre-fix
     - N=2 post-fix
     - Improvement
   * - Sphere
     - 1 MFP
     - 26.9 %
     - 16 % (plateau)
     - **1.22 %**
     - :math:`22\times`
   * - Sphere
     - 10 MFP
     - 0.28 %
     - 6.64 % (worse!)
     - **0.17 %**
     - conserves
   * - Cylinder
     - 1 MFP
     - 20.9 %
     - 5.4 %
     - **8.3 %** (worse)
     - –
   * - Cylinder
     - 10 MFP
     - 1.14 %
     - 9.3 % (worse!)
     - **1.06 %**
     - conserves

Sphere is the clean win **at 1R only — see the 2026-04-25 retraction
note above**. Cylinder is partial — at thick :math:`R`
the fix prevents the rank-N degradation observed pre-fix (rank-2
no longer shoots up to 9.3 %), but the thin-cell rank-2 behaves
slightly worse than the old (no-Jacobian) form because the
cylinder's :func:`compute_G_bc_mode` still uses the 2-D projected
:math:`\mu_{s,2D}` in the surface-centred :math:`\mathrm{Ki}_1/d`
integrand rather than the 3-D :math:`\mu_{s,3D} = \sin\theta_p
\cdot \mu_{s,2D}`. That 3-D upgrade (Phase C of Issue #112) is
expected to flip the cylinder improvement too. Both observations
are conditional on the mode-0 normalisation hack remaining in
place, which Issue #132 (see
:ref:`peierls-rank-n-class-b-mr-mg-falsification`) tracks for
re-derivation.

**Conservation** (the quantitative smoking gun). For a
homogeneous pure absorber (:math:`\Sigma_s = \nu\Sigma_f = 0`)
with uniform :math:`q = 1`, the kernel identity
:math:`K\cdot\mathbf 1 = \Sigma_t\,\mathbf 1` must hold.
Pre-fix: rank-N **degraded** the defect by :math:`10\times` at
thick :math:`R`. Post-fix: rank-N **reduces** the defect
uniformly. The
``tests/derivations/test_peierls_rank_n_conservation.py``
fixtures gate this explicitly.


.. _peierls-rank-n-per-face-closeout:

Phase F.5 — Rank-N per-face closure: five-reference synthesis and structural close-out (Issue #119, CLOSED 2026-04-21)
======================================================================================================================

Status
------

**Issue #119 is CLOSED as of 2026-04-21.** The Phase F.4 *scalar
rank-2 per-face* closure is the production path for hollow 2-surface
cells under white BC. The :exc:`NotImplementedError` guard in
:func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
(``n_bc_modes > 1`` + ``reflection="white"`` on 2-surface cells)
**remains in place by design** — the guard is not a bug, it is the
documented refusal of a structurally unreachable closure. Five
independent authoritative references converge on scalar / DP-0 for
1D curvilinear interface-current methods; a novel per-mode
Villarino-Stamm'ler extension tested in this investigation does not
break the plateau; two research-tag pathways remain open for future
work. The Marshak/Sanchez rank-N primitives are retained as tested
dead code for that future research.

Motivation (preserved from the original open-issue section)
-----------------------------------------------------------

The scalar rank-2 per-face white BC closes the Wigner-Seitz identity
on slab exactly (via the legacy :math:`E_2` / :math:`E_3` bilinear
form) but leaves a 1–13 % residual on curved hollow cells. Extending
to rank-:math:`N` per face — so each surface's outgoing distribution
is resolved into more than one angular moment — was expected to drive
that residual toward machine precision. The single-surface rank-:math:`N`
Gelbard DP\ :sub:`N-1` path
(:func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction_rank_n`)
converges monotonically with :math:`N` for the solid cylinder /
sphere, so per-face generalisation — two surfaces coupled by a
:math:`(2N \times 2N)` transmission matrix :math:`W_N` — looked
inevitable. **It is not.** The remainder of this section is the
archival record of why.

Five-reference literature synthesis
-----------------------------------

Five authoritative reactor-physics references were scanned for an
independent derivation of the Sanchez-McCormick 1982 §III.F.1
Legendre-moment rank-:math:`N` ladder applied to curvilinear
(hollow cylindrical / spherical) cells. The surprising finding is
that **no such derivation exists in the canonical textbook corpus**.

.. list-table:: Curvilinear interface-current closures across five references
   :header-rows: 1
   :widths: 28 26 26 20

   * - Reference
     - Location
     - Closure form
     - Verdict
   * - Ligou (1982)
     - Ch. 8 §8.2.3, Eq. 8.47
     - Scalar :math:`j^-` with cosine return
     - Scalar (= F.4)
   * - Sanchez, Mao, Santandrea (2002)
     - NSE 140:23, Eqs. 37, 40
     - Piecewise-constant angular (PCA) sectors OR collocation-:math:`\delta_2`
     - Sector-based, not Legendre
   * - Stamm'ler & Abbate (1983)
     - Ch. IV §10, Eqs. 29–34
     - Scalar :math:`j^-` + cosine multi-reflection
     - Scalar (= F.4)
   * - Stacey (2007)
     - Ch. 9 §9.4–§9.5, Eqs. 9.66–9.110
     - DP-0 per face (isotropic hemisphere)
     - Cartesian-only DP-0
   * - Hébert (2009)
     - Ch. 3 §3.8.1 (abstract) and §3.8.4 Eq. 3.323 (1D cyl)
     - DP\ :sub:`N` machinery for 2D Cartesian pincells; scalar Eq. 3.323 for 1D curvilinear
     - Cartesian-only DP\ :sub:`N`; 1D = F.4

**The keystone fact.** Hébert 2009 Eq. 3.323 is the *modern textbook
statement* of the 1D cylindrical interface-current closure under
white BC:

.. math::
   :label: hebert-3-323

   \tilde P \;=\; P \;+\; \frac{\beta^+}{1 - \beta^+ P_{SS}}\,
      P_{iS}\, p_{Sj}^{\mathsf T}.

With :math:`\beta^+ = 1` (white BC), this is **Stamm'ler Eq. 34** in
modern notation, and it is **ORPHEUS F.4** in the rank-0 limit:
rank-0 scalar :math:`P_{SS}` (cylinder face return probability) with
the scalar geometric series :math:`(1 - P_{SS})^{-1}` is exactly the
:math:`(I - W)^{-1}` we ship at :math:`N = 1`. Three independent
textbook lineages (Ligou, Stamm'ler, Hébert) and one slab-response
textbook (Stacey) arrive at the same scalar closure. The rank-:math:`N`
Legendre ladder of Sanchez-McCormick 1982 §III.F.1 has **zero
cross-validation** among these five references.

The same equation is the production F.4 closure on BOTH 1D
curvilinear geometries shipped in ORPHEUS (:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`,
``kind`` ∈ :code:`{"cylinder-1d", "sphere-1d"}`), with the
geometry-specific transmission matrix :math:`W` supplied by
:func:`~orpheus.derivations.peierls_geometry.compute_hollow_cyl_transmission`
(cylinder, Ki\ :sub:`3` Bickley fold of the Lambertian emission) or
:func:`~orpheus.derivations.peierls_geometry.compute_hollow_sph_transmission`
(sphere, bare :math:`\exp(-\tau)` with explicit :math:`\theta`
integration). The two geometries differ in the :math:`W_{oi} /
W_{io}` reciprocity: cylinder uses circumference per unit length so
:math:`W_{oi} = (R / r_0)\,W_{io}` (FIRST power), sphere uses surface
area so :math:`W_{oi} = (R / r_0)^2\,W_{io}` (squared). The
zero-absorption conservation pair
:math:`W_{oo} + W_{io} = W_{oi} = 1`, :math:`W_{ii} = 0` holds in
both geometries and is regressed by
``test_hollow_{cyl,sph}_transmission_zero_absorption_conservation``
(see :file:`tests/derivations/test_peierls_rank2_bc.py`).
Residuals at default quadrature (2 panels × p_order=4, n_angular=24,
n_rho=24, n_surf_quad=24, dps=15):

.. list-table:: F.4 residual across :math:`r_0 / R` at :math:`k_\infty = 1.5` (:math:`\Sigma_t = 1`, :math:`\Sigma_s = 0.5`, :math:`\nu\Sigma_f = 0.75`)
   :header-rows: 1
   :widths: 15 20 20 22 23

   * - :math:`r_0 / R`
     - Cylinder F.4
     - Cylinder Mark
     - Sphere F.4
     - Sphere Mark
   * - 0.1
     - 1.4 %
     - 24.9 %
     - 0.4 %
     - ~ 25 %
   * - 0.2
     - 5.4 %
     - 28.5 %
     - 1.2 %
     - —
   * - 0.3
     - 13.2 %
     - 32.4 %
     - 3.3 %
     - —

Sphere residuals are 3–10× tighter than cylinder at the same
:math:`r_0/R` because the sphere's higher SO(3) symmetry captures
more angular structure at the scalar mode, while the cylinder's
out-of-plane :math:`\theta` fold into Ki\ :sub:`3` averages
irreversibly over that dimension. Neither is improvable via rank-N
refinement — see the Phase F.5 close-out below and research log
lesson L21.

See :file:`.claude/agent-memory/literature-researcher/rank_n_closure_four_references_synthesis.md`
for the Ligou / Sanchez 2002 / Stamm'ler IV / Stacey 9
side-by-side, and
:file:`.claude/agent-memory/literature-researcher/hebert_2009_ch3_interface_currents.md`
for the Hébert 2009 Ch. 3 extraction (5th reference, keystone).

The structural obstruction — the :math:`c_{\rm in}` remapping
----------------------------------------------------------------

Why do all five textbooks refuse rank-:math:`N` on curvilinear
cells? The answer is geometric. On a hollow spherical (or
cylindrical) cell of outer radius :math:`R` and inner radius
:math:`r_0`, a neutron leaving the outer surface with emission
cosine :math:`\mu_{\rm emit}` (relative to the outer outward normal)
strikes the inner surface with arrival cosine

.. math::
   :label: c-in-remapping

   c_{\rm in}(\mu_{\rm emit}) \;=\;
      \sqrt{1 - \left(\frac{R}{r_0}\right)^{\!2}\!
         \bigl(1 - \mu_{\rm emit}^2\bigr)},

relative to the inner outward normal. This is geometry, not
physics. The Legendre basis at the outer surface is
:math:`\{\tilde P_n(\mu_{\rm emit})\}_{n \ge 0}`; the Legendre basis
at the inner surface is :math:`\{\tilde P_n(c_{\rm in})\}_{n \ge 0}`.
The outer→inner transmission integrand therefore couples
:math:`\tilde P_n(\mu_{\rm emit})` at emission with
:math:`\tilde P_m(c_{\rm in}(\mu_{\rm emit}))` at arrival, with
:math:`c_{\rm in}` a non-trivial nonlinear function of
:math:`\mu_{\rm emit}` parameterised by :math:`R/r_0`.

At :math:`\sigma_t = 0` (pure geometry), the conservation identity
that F.4's physics requires at every mode is

.. math::
   :label: mode-conservation-target

   W_{\rm oo}[n, n] \;+\; W_{\rm io}[n, n] \;=\; \delta_{n, 0}
   \quad\text{(naive per-mode conservation)}.

Direct numerical evaluation of the shipped (Monte-Carlo-verified)
:func:`~orpheus.derivations.peierls_geometry.compute_hollow_sph_transmission_rank_n`
at :math:`r_0/R = 0.3`, :math:`\sigma_t = 0` gives:

.. list-table:: Per-mode conservation on hollow sphere (:math:`r_0/R = 0.3`, :math:`\sigma_t = 0`)
   :header-rows: 1
   :widths: 20 40 40

   * - Mode :math:`n`
     - :math:`W_{\rm oo}[n,n] + W_{\rm io}[n,n]`
     - Status
   * - 0
     - 1.000
     - F.4 identity — exact
   * - 1
     - 0.281
     - Structural failure
   * - 2
     - 0.134
     - Structural failure
   * - 3
     - 0.092
     - Structural failure

**Why mode 0 works**: :math:`\tilde P_0 = 1` is angle-independent.
The :math:`c_{\rm in}` remapping is *invisible* at mode 0 because
:math:`\tilde P_0(c_{\rm in}) = \tilde P_0(\mu_{\rm emit}) = 1`
trivially, so the integrand factorises and conservation holds.

**Why modes** :math:`n \ge 1` **fail**: the Legendre basis does
*not* diagonalise the nonlinear :math:`\mu_{\rm emit} \mapsto
c_{\rm in}` map. The :math:`W_{\rm io}` integrand couples
:math:`\tilde P_n(\mu_{\rm emit})` (emission) with
:math:`\tilde P_m(c_{\rm in}(\mu_{\rm emit}))` (arrival), and no
placement of :math:`(2n+1)` Gelbard factors, no half-range Gram
rescaling, no (Lambert ↔ Marshak) basis permutation can undo the
geometric distortion inside the kernel. This was confirmed by the
60-recipe empirical scan summarised in
:file:`.claude/agent-memory/numerics-investigator/peierls_rank_n_sanchez_closure_failed.md`:
every tested assembly reached the same 1.42 % plateau at
:math:`\sigma_t \cdot R = 5`, :math:`N \ge 2`.

Villarino-Stamm'ler per-mode extension (novel, falsified 2026-04-21)
--------------------------------------------------------------------

Hébert 2009 Ch. 3 explicitly warns (p. 129) that the rank-0 DP\
:sub:`N` primitives are *not* guaranteed conservative and recommends
an a-posteriori **Villarino-Stamm'ler normalisation**
(Eqs. 3.347–3.352). Villarino-Stamm'ler (V-S) is a 30-line
Gauss-Seidel fixed-point iteration that multiplies the symmetric
rank-0 :math:`t` matrix by an additive symmetric correction to
force row conservation while preserving reciprocity.

.. math::
   :label: hebert-3-350

   \hat{t}_{\ell m} \;=\; (z_{\ell} + z_{m})\, t_{\ell m},
   \qquad \ell,\, m \;=\; 1, \ldots, \Lambda + I.

Reciprocity is preserved by construction because the scalar factor
:math:`(z_{\ell} + z_{m})` is symmetric in :math:`\ell \leftrightarrow m`
and :math:`t_{\ell m}` is already symmetric (Hébert p. 129).

Hébert defines V-S only on rank-0 primitives. The novel extension
tested here is **per-diagonal-mode V-S on the rank-:math:`N` W**:
for each mode :math:`n \in \{0, \ldots, N{-}1\}`, extract the 2×2
diagonal block

.. math::

   W_{\rm sub}[n] \;=\;
   \begin{pmatrix}
      W[n,\, n] & W[n,\, N{+}n] \\[2pt]
      W[N{+}n,\, n] & W[N{+}n,\, N{+}n]
   \end{pmatrix},

solve the 2-unknown V-S system for :math:`(z_{\rm outer}^{(n)},
z_{\rm inner}^{(n)})`, and apply the additive symmetric correction
per mode. The target is the F.4 scalar row sum (mode-0 conserves
against F.4's target as a sanity check; :math:`n \ge 1` is forced
onto F.4's mode-0 target, which is the strongest possible demand
for per-mode conservation). Off-diagonal cross-mode blocks
:math:`n \ne m` are left untouched.

.. list-table:: Per-mode V-S residuals on hollow sphere (:math:`r_0/R = 0.3`, :math:`\sigma_t \cdot R = 5`)
   :header-rows: 1
   :widths: 10 22 22 22 22

   * - :math:`N`
     - µ-ortho raw
     - µ-ortho + V-S
     - Shipped Marshak raw
     - Shipped Marshak + V-S
   * - 1
     - 2.55 %
     - 2.17 %
     - 13.53 %
     - 13.53 %
   * - 2
     - **1.42 %**
     - **1.87 % WORSE**
     - 10.86 %
     - 10.83 %
   * - 3
     - 1.42 %
     - 1.87 % WORSE
     - 10.70 %
     - 10.66 %
   * - 4
     - 1.43 %
     - 1.88 % WORSE
     - 10.70 %
     - 10.66 %

Reciprocity is preserved at machine precision
(:math:`A_i\, W_{ij} - A_j\, W_{ji} \le 10^{-16}`) for every row of
the table, both pre- and post-V-S, confirming that the additive
symmetric form does what Hébert Eqs. 3.350 claim (p. 129). The V-S
scheme *also* hits its design target: at :math:`\sigma_t = 0` the
per-mode row sums are driven to the F.4 scalar targets for every
mode :math:`n` (e.g., at :math:`N = 3`: outer row sum
:math:`2.137 \to 1.910`, inner :math:`0.118 \to 0.090` for
:math:`n = 0`; outer :math:`1.191 \to 1.910`,
:math:`0.547 \to 1.910` for :math:`n = 1, 2`).

**The falsification is unambiguous.** Enforcing per-mode
conservation does not close the plateau — on the µ-ortho pipeline
it makes k\ :sub:`eff` **worse** (1.42 % → 1.87 %); on the shipped
Marshak pipeline it is essentially a no-op (10.86 % → 10.83 %).
**The plateau is a cross-mode coupling failure, not a
conservation-forcing failure.** Correcting the diagonal blocks
distorts the balance between diagonal and off-diagonal coupling,
shifting the closure further off rather than toward k\ :sub:`inf`.

Diagnostic at
:file:`derivations/diagnostics/diag_rank_n_villarino_stammler_per_mode.py`;
full memo at
:file:`.claude/agent-memory/numerics-investigator/peierls_villarino_stammler_per_mode.md`.

Production closure decision
---------------------------

**F.4 scalar rank-2 per-face is production.** Its residual across
optical thickness at :math:`r_0/R = 0.3` (from
:file:`derivations/diagnostics/diag_rank_n_closure_characterization.py`):

.. list-table:: F.4 scalar rank-2 residual, hollow sphere, :math:`r_0/R = 0.3`
   :header-rows: 1
   :widths: 20 20 60

   * - :math:`\sigma_t \cdot R`
     - :math:`|k_{\rm eff} - k_{\infty}| / k_{\infty}`
     - Regime
   * - 1
     - 3.27 %
     - Streaming limit (quadrature-dominated)
   * - 2.5
     - 0.55 %
     - Intermediate
   * - 5
     - **0.077 %**
     - Near quadrature floor (Mark DP\ :sub:`0` truncation ~0.04–0.1 %)
   * - 10
     - 0.26 %
     - Forward-peaked flux
   * - 20
     - 0.45 %
     - Strongly absorbing

The 0.077 % minimum at :math:`\sigma_t \cdot R = 5` coincides with
the Mark DP\ :sub:`0` truncation floor for this geometry, not a
pure radial-quadrature floor — refining the radial Gauss-Legendre
grid does not drive it below ~0.04 %. This is the expected
expression of Stacey 2007's warning on p. 329 that
:math:`[E_2(\Sigma)]^N \ne E_2(N\Sigma)` for DP-0 subdivision of
forward-peaked boundary fluxes: the error class is geometric, baked
into the DP-0 assumption, and cannot be removed without changing
the angular representation.

.. _peierls-f4-rank-1-gauge-why:

Why F.4 works at rank-1 but does not generalise (Direction Q, Issue #122)
-------------------------------------------------------------------------

F.4 scalar (Eq. 3.323 of Hébert 2009) pairs **Lambert-basis**
:math:`P_{\rm esc}, G_{bc}` (integrand has no :math:`\mu` weight on
the outgoing side) with **Marshak-basis** :math:`W` (the transmission
operator is defined on the :math:`\mu`-weighted half-range Gram). The
pairing is formally inconsistent — the escape / coupling primitives
and the transmission operator live in different inner products — yet
it is **empirically load-bearing**: every formally-consistent rank-N
closure we built (Marshak everywhere, Lambert everywhere, split
c\ :sub:`in`-aware basis with adaptive scale — see
``diag_cin_aware_split_basis_keff.py``) plateaus 10–100× above F.4's
residual on the hollow sphere at :math:`\sigma_t R \ge 5`.

Two sub-agent investigations closed Issue #122 with verdict **(B)**:
**F.4's Lambert/Marshak asymmetry is a lucky rank-1 algebraic
accident with a principled re-phrasing.** The algebraic bridge
between F.4 and the rank-N Marshak closure is the solid-harmonic
change-of-basis matrix between the two half-range L\ :sup:`2`
structures on :math:`[0, 1]`.

**Setup.** On the outgoing-:math:`\mu` half-line the two natural
inner products are

.. math::
   :label: peierls-half-range-inner-products

   \langle f, g \rangle_L \;=\; \int_0^1 f(\mu)\,g(\mu)\,\mathrm d\mu,
   \qquad
   \langle f, g \rangle_M \;=\; \int_0^1 f(\mu)\,g(\mu)\,\mu\,\mathrm d\mu.

Shifted-Legendre on :math:`[0, 1]` is orthogonal under
:math:`\langle \cdot, \cdot \rangle_L`; F.4 uses the
:math:`L`-orthonormal basis
:math:`\phi_n^L(\mu) = \sqrt{2n+1}\,P_n(2\mu-1)` for its :math:`P` and
:math:`G`. The ORPHEUS rank-N Marshak closure (helper
``_build_closure_operator_rank_n_white``, guarded behind
:exc:`NotImplementedError`) uses the :math:`M`-orthonormal basis
:math:`\psi_n^M` obtained by Gram-Schmidt of :math:`\{1, \mu, \mu^2, \ldots\}`
under :math:`\langle \cdot, \cdot \rangle_M` (equivalent to scaled
Jacobi :math:`P_n^{(0,1)}(2\mu-1)`).

**Change-of-basis matrix.** The matrix that expresses Lambert ONB
functions in Marshak ONB coordinates is

.. math::
   :label: peierls-change-of-basis

   M_{nm} \;=\; \langle \psi_n^M, \phi_m^L \rangle_M
          \;=\; \int_0^1 \psi_n^M(\mu)\,\phi_m^L(\mu)\,\mu\,\mathrm d\mu,
   \qquad
   \phi_m^L \;=\; \sum_n M_{nm}\,\psi_n^M.

The Marshak Gram matrix of the Lambert basis (already cited above for
the Phase F.5 rank-N primitives),
:math:`B^{\mu}_{mn} = \langle \phi_m^L, \phi_n^L \rangle_M`, equals
:math:`M^{\!\top} M` identically. This is the :math:`B^{\mu}` matrix
consumed by the existing ``_build_closure_operator_rank_n_white``
helper.

**Symbolic closed forms** (verified in
``derivations/diagnostics/diag_lambert_marshak_basis_change.py``;
runs in ~1 s).

At rank :math:`N = 1` the change-of-basis matrix is a **scalar**

.. math::
   :label: peierls-M-rank-1

   M^{(1)} \;=\; \tfrac{\sqrt{2}}{2} \;\approx\; 0.7071,
   \qquad
   (B^{\mu})^{(1)} \;=\; \tfrac{1}{2}.

At rank :math:`N = 2` the change-of-basis matrix is **upper bidiagonal**

.. math::
   :label: peierls-M-rank-2

   M^{(2)} \;=\;
   \begin{pmatrix}
     \tfrac{\sqrt{2}}{2} & \tfrac{\sqrt{6}}{6} \\[4pt]
     0                   & \tfrac{\sqrt{3}}{3}
   \end{pmatrix},
   \qquad
   (B^{\mu})^{(2)} \;=\;
   \begin{pmatrix}
     \tfrac{1}{2} & \tfrac{\sqrt{3}}{6} \\[4pt]
     \tfrac{\sqrt{3}}{6} & \tfrac{1}{2}
   \end{pmatrix}.

:math:`M^{(2)}` has singular values :math:`(0.460, 0.888)`. They
are **unequal**, so :math:`M^{(2)}` is not a scalar multiple of an
orthogonal matrix — it is a **genuine basis rotation**, not a scalar
gauge. The rank-3 and rank-4 change-of-basis matrices preserve this
upper-bidiagonal structure; the sum of magnitudes of strictly-off-
diagonal entries grows monotonically (0, 0.408, 0.855, 1.318 at
ranks 1–4).

**Implication for F.4.** At rank-1, the Lambert / Marshak mismatch is
the scalar :math:`M^{(1)} = \sqrt{2}/2`; it factors out of the closure
:math:`K_{bc} = G \beta (I - \beta W)^{-1} P` as a rescaling of
:math:`\beta`. F.4's "trick" is to effectively use the Marshak-side
:math:`\beta_{\rm eff} = \beta / (M^{(1)})^2 = 2\beta` without
explicitly saying so. The closure-level gauge factor
:math:`\alpha(\tau) \approx 0.38` measured in
``diag_lambert_marshak_symbolic.py`` (see §Experiment 7 of the
research log) is this basis-change scalar **times** the
:math:`\exp(-\sigma R)` attenuation integrated through the ray — a
single number, identifiable, and absorbable.

At rank :math:`N \ge 2`, :math:`M` ceases to commute with any
diagonal :math:`\operatorname{diag}(\beta_0, \beta_1, \ldots)`. There
is **no vector** :math:`\beta_{\rm eff}` that, pre-multiplied onto the
Marshak closure, reproduces F.4's Lambert behaviour. The off-diagonal
entries of :math:`M` mix Lambert mode-0 into Marshak mode-1 and vice
versa, and that mixing propagates through the closure operator
:math:`(I - \beta W)^{-1}` with uncontrolled amplification. This is
the algebraic origin of Experiment E2.4's rank-N Lambert-P/G
catastrophe (33–737 % k\ :sub:`eff` error on the 6-point reference
grid) documented in the research log.

**The precise obstruction: asymmetric µ-multiplication with a
polynomial-truncation leak.** A frame-covariant rewrite of F.4
— conjugating the Lambert closure by :math:`M` on both sides as
:math:`K_{bc}^{F.4} = G_L \cdot M^{\!\top}\,(I - M W_L M^{\!\top})^{-1}\,M\,P_L`
— was tested at rank-2 and rank-3 (symbolic + mpmath, see
``derivations/diagnostics/diag_frame_4_connection_form.py``) and
fails: 22 % relative error at the rank-2 anchor, 48 % at rank-1,
across 5 conjugation variants and 7 values of :math:`\sigma_t R`.
:math:`W` is NOT a (1,1) tensor under :math:`M`; at rank-1 already
:math:`M W_L M^{\!\top} = 0.05` versus :math:`W_M = 0.005` at
:math:`\sigma_t R = 10`, a tenfold discrepancy. The correct
algebraic relationship is the **asymmetric identity**

.. math::
   :label: peierls-WM-WL-asymmetric

   W_M \;=\; B^{\mu}\,W_L,
   \qquad
   B^{\mu} \;=\; M^{\!\top} M,

verified bit-exactly at infinite rank. At rank :math:`N`, this
identity holds on rows :math:`0, 1, \ldots, N-2` of
:math:`W_M - B^{\mu} W_L` (vanishing symbolically) but **row**
:math:`N-1` **carries a non-vanishing τ-dependent polynomial-
truncation residual** — because :math:`\mu \cdot \tilde P_{N-1}`
has a :math:`\tilde P_N` component that the rank-:math:`N` basis
cannot represent. Equation :math:numref:`peierls-WM-WL-asymmetric`
is the exact statement of Lambert ↔ Marshak at the transmission
operator; the rank-1 scalar-gauge picture of
:math:numref:`peierls-M-rank-1` is the finite-truncation limit
where the leaking row has empty support. This obstruction is
structural and cannot be cured by any basis rotation on the rank-
:math:`N` space.

**Connection to the gauge-theoretic literature.** Sanchez (2014)
NSE 177(1), DOI ``10.13182/NSE12-95``, establishes that the
first-order :math:`P_N` equations are **degenerate** — multiple IC/BC
closures are admissible, with uniqueness imposed via second-order-
parity equivalence and solid-harmonic expansions that reproduce
results originally due to Davison (1957) and Rumyantsev (1950s). The
Sanchez theorem applies to the differential :math:`P_N` equations,
not directly to the integral CP operator F.4 lives on, but the
gauge-freedom framing is the right language: **at rank-1 the integral
CP admits a scalar gauge (equation** :math:numref:`peierls-M-rank-1`
**) that F.4 exploits**; at rank-N the "gauge" becomes a full basis
rotation (equation
:math:numref:`peierls-change-of-basis`) that the Marshak closure
must account for explicitly. Canonical open-access channels to the
underlying solid-harmonic material are Davison (1957) *Neutron
Transport Theory* (Oxford; AEC NAA-SR-3509) and Case & Zweifel (1967)
*Linear Transport Theory*.

**Statistical-mechanical picture (partial).** The scalar gauge
:math:`\alpha(\tau, \rho) \approx 0.38` has a partial statistical-
mechanical interpretation via the surface Markov chain on the outer
hemisphere: state space is outgoing :math:`\mu \in [0, 1]`; the
transition kernel is ballistic chord through the hollow sphere
followed by isotropic re-emission with in-shell scattering
:math:`c = 1/3`. Direct 500 k-history Monte Carlo of this chain
(see ``derivations/diagnostics/diag_frame_3_surface_markov_mc.py``)
gives a Perron eigenfunction :math:`p_{\infty}(\mu)` whose moments
are **ρ-independent to :math:`\le 1.3\,\%`** across
:math:`\rho \in [0.3, 0.5]` at fixed :math:`\tau` — this is the
mechanism behind the observed ρ-flatness of :math:`\alpha`. But
:math:`p_{\infty}(\mu)` is **not Laplace-type**: an exponential
fit :math:`A e^{-\lambda\mu} + B` leaves 7–11 % residual; the
histogram is monotone *increasing* in :math:`\mu`, with
:math:`\mathbb E[\mu] \approx 0.56\text{–}0.61` and
:math:`\mathbb E[\mu^2]/\mathbb E[\mu] \approx 0.70\text{–}0.72`.
No natural moment of :math:`p_{\infty}` identifies :math:`\alpha`
to better than 5 % uniformly across the 6-point grid. The rank-N
polynomial expansion of :math:`p_{\infty}` is therefore
**basis-resistant** because :math:`p_{\infty}` is neither polynomial
nor single-exponential — supplementing the algebraic Schur-reduction
story (equation :math:numref:`peierls-WM-WL-asymmetric`) with an
independent statistical obstruction. An analytical computation of
:math:`p_{\infty}` as the left Peierls-kernel eigenvector on the
outer surface would settle the quantitative :math:`\alpha`
identification without MC bias; this is unresolved and listed as a
follow-up.

**Production decision.** The guard on
``boundary="white", n_bc_modes > 1`` stays in place. F.4 remains
production. The rank-N Marshak primitives below are retained because
they are tested, reciprocity-verified, and equal-to-F.4 at :math:`N=1`
— but no principled rank-N generalisation of the Lambert-side trick
exists. Any future rank-N white-BC closure candidate must compete
against F.4 under the two-quadrature stability protocol (Direction N,
Issue #123) — see L19 in the research log.

.. _peierls-rank-n-stability:

Rank-N stability protocol (L19)
--------------------------------

A rank-N white-BC closure candidate :math:`C` claims to beat F.4 at a
reference point :math:`(\tau, \rho)` if and only if the claim survives
the **two-quadrature signed-error stability protocol** defined here.
This protocol is the operational response to lessons L17–L19 in
``.claude/plans/rank-n-closure-research-log.md``: RICH = (4, 8, 64) is
below F.4's own structural floor at :math:`\sigma_t R \ge 10`, so a
naive single-quadrature comparison rewards quadrature-noise cancellation
rather than truncation-residual reduction.

**Protocol.** Let :math:`\mathcal Q = (q_1, q_2, \ldots, q_K)`,
:math:`K \ge 2`, be a sequence of quadrature triples
:math:`(n_{\rm panels}, p_{\rm order}, n_{\rm ang})` of monotonically
increasing refinement (lexicographic in any component). Let
:math:`e_C(q_k)` and :math:`e_{F.4}(q_k)` denote the **signed** relative
errors :math:`(k_{\rm eff}^{C/F.4}(q_k) - k_\infty)/k_\infty` at each
quadrature.

The candidate :math:`C` is a **structural win** over F.4 at
:math:`(\tau, \rho)` iff all five of the following hold:

.. math::
   :label: peierls-rank-n-stability

   \begin{aligned}
   & \textbf{(S1)} \quad K \ge 2, \\
   & \textbf{(S2)} \quad |e_C(q_k)| \;<\; |e_{F.4}(q_k)|
       \quad \text{for every } k, \\
   & \textbf{(S3)} \quad \operatorname{sign}\,e_C(q_k) \;=\; \operatorname{sign}\,e_C(q_{k+1})
       \quad \text{for every } k, \\
   & \textbf{(S4)} \quad |e_C(q_k)| \;\ge\; |e_C(q_{k+1})|
       \quad \text{for every } k, \\
   & \textbf{(S5)} \quad \operatorname{sign}\,e_{F.4}(q_k)
       \;=\; \operatorname{sign}\,e_{F.4}(q_{k+1})
     \; \wedge \; |e_{F.4}(q_k)| \;\ge\; |e_{F.4}(q_{k+1})|
       \quad \text{for every } k.
   \end{aligned}

Assertion **(S5)** is the L17/L19 reference-verifiability gate: if
F.4 itself sign-flips or grows in magnitude under refinement at
:math:`(\tau, \rho)`, the *reference* is unverifiable at
:math:`\mathcal Q` and no rank-N comparison there is admissible.
This is strictly stronger than L16's "match quadrature across
compared closures" — it additionally demands that the match resolves
the smaller structural floor.

**Implementation.** The helper
``tests.cp.test_peierls_rank_n_protocol.assert_rank_n_structural_win``
raises :exc:`AssertionError` on any of S1–S5 failing and returns a
``StabilityReport`` dataclass with the full signed-error trajectory
otherwise. The helper ships with pinning tests for the RICH vs
RICH+panels pair at the six reference points
:math:`(\sigma_t R, \rho) \in \{5, 10, 20\} \times \{0.3, 0.5\}`;
two of those six (:math:`\sigma_t R = 10, \rho = 0.3` and
:math:`\sigma_t R = 20, \rho = 0.5`) reproduce the canonical L17
sign-flip of F.4 itself and are tagged ``@pytest.mark.slow``.

A baseline of F.4 at ULTRA = (5, 10, 96) and richer at all six points
— the reference that would make :math:`\mathcal Q` trivially
L19-compliant without needing S5 — is currently **unresolved** on the
devcontainer hardware used during Issue #123 development: ULTRA and
RICH+pp exceeded the 120-s-per-point budget at every point tested.
Resolving the full ULTRA baseline requires either richer hardware or
a relaxed wall budget (target: :math:`\ge` 300 s per point at
:math:`\sigma_t R = 20`). See L20 in the research log.

**Randomized QMC alternative (validated 2026-04-22).** The product-
Gauss bias that produces L17 sign flips is driven by the tangent-
angle kink in the exp(−τd) integrand, whose Hardy-Krause variation
is *bounded in* :math:`\tau`. Owen-scrambled Sobol' on the angular
dimension (32 scrambles × 4096 points) gives 95 % bootstrap CI
widths of :math:`6 \times 10^{-5}` to :math:`6 \times 10^{-4}` per
cent on F.4 at all six reference points — 20–100× tighter than the
PG RICH vs RICH+panels spread the L19 protocol uses to *detect*
instability. Both L17 sign-flip points
(:math:`\sigma_t R = 10, \rho = 0.3` and
:math:`\sigma_t R = 20, \rho = 0.5`) resolve to crisp negative
QMC means whose CIs do not cross zero. See
``derivations/diagnostics/diag_f4_qmc_quadrature.py`` and the
Frame 5 memo. A future rank-N closure candidate can therefore
replace the S3–S4 gates of :math:numref:`peierls-rank-n-stability`
by a **single CI-separation assertion**:

   closure mean, F.4 mean with disjoint 95 % CIs, AND closure CI
   strictly tighter than :math:`|{\text{F.4 mean}}|`.

The thin wrapper ``assert_rank_n_qmc_structural_win(closure, f4,
point, N=4096, n_scrambles=32)`` implementing this is sketched in
the Frame 5 memo; not shipped because no current closure passes
Frame 5. Issue #128 tracks the optional migration of F.4 production
quadrature from product-Gauss to randomized QMC; LOW priority, not
on the critical path.

Infrastructure retained (dead-code-guarded for future research)
----------------------------------------------------------------

The following rank-:math:`N` primitives landed in Phase F.5 are kept
in the source tree because they are tested, correct at
:math:`\sigma_t \to 0`, and useful for the two open research paths
documented below. They are unreachable through the public API
while the :exc:`NotImplementedError` guard remains in place.

- :func:`~orpheus.derivations.peierls_geometry.compute_hollow_sph_transmission_rank_n`
  — :math:`(2N \times 2N)` surface-to-surface transmission matrix
  for hollow sphere; bit-exact reduction to the scalar transmission
  at :math:`N = 1`; Sanchez-McCormick reciprocity
  :math:`A_k\,W_{jk}^{(m,n)} = A_j\,W_{kj}^{(n,m)}` (transposed
  mode indices) verified to :math:`10^{-14}`; Monte-Carlo
  cross-checked to (m, n) = (2, 2) at 4 M samples.

- Lambert-basis per-face mode primitives
  (``compute_{P_esc, G_bc}_{outer, inner}_mode``): integrand
  :math:`\sin\theta\,\tilde P_n(\mu_{\rm exit})\,K_{\rm esc}(\tau)`
  — no :math:`\mu` weight. At :math:`n = 0` these reduce bit-exactly
  to the scalar primitives used by the Phase F.4 rank-2 closure.

- Marshak-basis per-face mode primitives
  (``compute_{P_esc, G_bc}_{outer, inner}_mode_marshak``): integrand
  :math:`\sin\theta\,\mu\,\tilde P_n(\mu)\,K_{\rm esc}(\tau)` — with
  :math:`\mu` weight for every mode including :math:`n = 0`.
  Verified at :math:`\sigma_t \to 0` against independent
  ``mpmath.quad`` references. Places :math:`P`, :math:`G`, and
  :math:`W` in the same partial-current-moment half-range inner
  product (half-range Gram matrix
  :math:`B^{\mu}_{mn} = \int_0^1 \mu\,\tilde P_m\,\tilde P_n\,\mathrm d\mu`).

- ``_build_closure_operator_rank_n_white`` — rank-:math:`N` assembly
  helper using Marshak primitives; reachable only if the
  :exc:`NotImplementedError` guard is lifted.

These primitives would be the starting point for a Sanchez 2002
piecewise-constant angular sector closure or a geometry-adapted
basis — they are retained so that future work does not need to
reproduce ~1500 lines of peer-reviewed infrastructure.

Open research (not production-blocking)
---------------------------------------

Two paths might break the 1.42 % plateau. Both require novel
mathematics outside the five-reference corpus. A ``research`` tag
GitHub issue is filed against each.

1. **Geometry-adapted Legendre basis.** Use
   :math:`\{\tilde P_n(\mu_{\rm emit})\}` at the outer surface and
   :math:`\{\tilde P_n(c_{\rm in}(\mu_{\rm emit}))\}` at the inner
   surface, with the Jacobian

   .. math::
      :label: c-in-jacobian

      \frac{\mathrm d c_{\rm in}}{\mathrm d \mu_{\rm emit}}
      \;=\; \left(\frac{R}{r_0}\right)^{\!2}
         \frac{\mu_{\rm emit}}{c_{\rm in}(\mu_{\rm emit})},

   folded into the transmission integrand. This *diagonalises* the
   outer→inner geometric map at the basis level, which the plain
   Legendre ladder fails to do. The closure would need to reduce to
   F.4 at :math:`N = 1` and then converge to k\ :sub:`inf` as
   :math:`N \to \infty`. Genuinely novel — not in Ligou, Sanchez
   2002, Stamm'ler IV, Stacey 9, or Hébert 2009.

2. **Piecewise-constant angular (PCA) sectors, Sanchez 2002
   style.** Partition each hemisphere into :math:`N_\theta \times
   N_\phi` angular sectors and use characteristic functions as the
   basis (Sanchez 2002 Eq. 37). Conservation is exact by
   construction because the basis elements are indicator functions
   and the measure is handled per-sector. Closure reduces to F.4 at
   :math:`N_\theta \times N_\phi = 1 \times 1`. This is the angular
   representation APOLLO2's TDT module actually uses in production
   pin-cell solvers. Major infrastructure lift
   (~1–2 engineering weeks) because it requires new sector data
   structures, sector-averaged P/G/W primitives, and new
   trajectory-tracking.

Session trail
-------------

The investigation spanned ~150 k tokens across three sessions on
``investigate/peierls-solver-bugs``:

- ``b9bc3df`` — measure-mismatch diagnostics (earlier hypothesis).
- ``d890a1e`` — feat: rank-:math:`N` per-face infrastructure
  (Lambert primitives + :math:`(2N \times 2N)` W).
- ``ca9d68f`` — feat: Marshak partial-current per-face primitives
  (dead code behind guard).
- ``53fae60`` — fix: ``solve_peierls_1g`` forwards ``inner_radius``
  to ``composite_gl_r`` (one-line pre-existing bug) + 33 MC
  cross-check tests of :math:`W` (all pass — :math:`W` is correct).
- ``cf6ab48`` — docs: earlier Marshak rank-:math:`N` plan
  (superseded by four-reference synthesis).
- ``0b0533b`` — diag: Sanchez-McCormick §III.F.1 recipe
  investigation (60+ variants, plateaus at 1.43 %).
- ``a2e2205`` — diag: closure characterisation + cross-:math:`\sigma_t
  \cdot R` parameter scan.
- ``a640a83`` — docs: four-reference synthesis (Ligou, Sanchez
  2002, Stamm'ler IV, Stacey 9). **Headline commit.**
- ``4a169ea`` — docs: F.4 quadrature-floor data added to close-out.
- ``ed69a09`` — docs: next-session plan for Hébert extraction.
- **This commit** — docs: five-reference synthesis + V-S
  falsification + Issue #119 close-out.

Diagnostic scripts in :file:`derivations/diagnostics/`:

- ``diag_rank_n_W_mc_crosscheck.py`` — :math:`W_N` Monte-Carlo
  cross-check (33 tests, all pass).
- ``diag_rank_n_sph_marshak_primitives_sigt_zero.py`` — Marshak
  primitive :math:`\sigma_t = 0` verification.
- ``diag_rank_n_closure_characterization.py`` — F.4 vs Sanchez
  :math:`\sigma_t \cdot R` scan (source of the production residual
  table above).
- ``diag_rank_n_sanchez_conservation_probe.py`` — structural
  diagnosis (per-mode conservation table above).
- ``diag_sanchez_N_convergence.py`` — Sanchez :math:`N = 1,\ldots,4`
  plateau proof.
- ``diag_rank_n_villarino_stammler_per_mode.py`` — V-S per-mode
  falsification (source of the V-S table above).

Memory files consulted:

- :file:`.claude/agent-memory/literature-researcher/rank_n_closure_four_references_synthesis.md`
- :file:`.claude/agent-memory/literature-researcher/hebert_2009_ch3_interface_currents.md`
- :file:`.claude/agent-memory/literature-researcher/sanchez_mccormick_rank_n_per_face.md`
- :file:`.claude/agent-memory/numerics-investigator/peierls_rank_n_sanchez_closure_failed.md`
- :file:`.claude/agent-memory/numerics-investigator/peierls_villarino_stammler_per_mode.md`

.. _peierls-rank-n-class-b-mr-mg-falsification:

Phase F.6 — Rank-N on Class B (solid cyl/sph) MR×MG: empirical falsification + Issue #132 (open 2026-04-25)
============================================================================================================

Status
------

**Issue #132 OPEN as of 2026-04-25.** This section is the empirical
falsification of "rank-:math:`N` Marshak closure converges on Class B
(solid cylinder, solid sphere) cells" for any cell with non-trivial
:math:`\Sigma_t` breakpoints. The structural failure is that
:func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
mixes two incompatible partial-current normalisations into the same
rank-:math:`N` outer-product expansion (mode 0 uses one Jacobian
convention, modes :math:`n \ge 1` use a different one). In single-region
the calibration of mode 0 hides the mismatch — see the
:ref:`peierls-rank-n-jacobian-section` retraction note. In
multi-region the two normalisations decouple from the calibration
fixed-point and the closure produces sign-flip catastrophes on the
order of :math:`+57\,\%` in :math:`k_{\rm eff}`. Pure-canonical mode-0
is **not a fix** — it makes the 1R results uniformly worse without
fixing MR. Re-derivation of the rank-:math:`N` partial-current
moment basis end-to-end is required and tracked in Issue #132.

This section is the Class B sibling of the
:ref:`peierls-rank-n-per-face-closeout` (Class A hollow rank-:math:`N`
falsification). **Rank-:math:`N` has now been falsified on both
topological classes for distinct reasons:**

- **Class A (hollow cyl/sph)**: F.4-style per-face Marshak rank-:math:`N`
  fails because the Legendre basis does not diagonalise the nonlinear
  outer→inner :math:`c_{\rm in}` arrival-cosine remap
  (:eq:`c-in-remapping`); plateau at 1.42 % under µ-ortho rank-2,
  no further improvement under V-S correction. **Closed L21**, F.4
  rank-2 scalar is production.
- **Class B (solid cyl/sph)**: rank-:math:`N` *single-surface* Marshak
  closure mixes two incompatible partial-current normalisations
  (mode 0 vs mode :math:`n \ge 1` in
  :func:`build_closure_operator`); +57 % sphere 1G/2R catastrophe
  exposed only in multi-region. **OPEN under Issue #132** —
  conditional on a corrective re-derivation, this falsification
  may yet flip.

The Class B falsification is therefore *less final* than the Class A
falsification: a principled re-derivation of the mode-0 normalisation
that lives in the same expansion space as mode :math:`n \ge 1`
remains a viable path to a working rank-:math:`N` Class B closure.
Three candidate paths are listed below.

Hypothesis under test (Plan §3 H_B, Issue #131 echo)
------------------------------------------------------

The plan at
:file:`.claude/plans/issue-100-103-rank-n-class-b-multi-region.md` §3
hypothesised three outcomes for a multi-region × multi-group rank-:math:`N`
sweep on Class B:

- **H_A (clean extension)** — rank-:math:`N` plateaus on Class B in
  MR×MG at the same 1G/1R floor; falsification simply extends from
  Class A to Class B.
- **H_B (hidden bug, Issue #131 template)** — rank-:math:`N` exhibits
  MR×MG behaviour the 1G/1R sweep missed; probe-cascade to localise.
- **H_C (rank-N actually beats rank-1 Mark in MR×MG)** — would
  invalidate the 1R falsification.

The investigation landed unambiguously on **H_B**. Outcome A would
have been the strong prior (Class A had already falsified F.4-style
rank-:math:`N` per-face; the 1G/1R Class B numbers in the docstring
table looked superficially convincing). Outcome B was the user's
explicit echo of the Issue #131 lesson — *single-region single-group
passing rates are degenerate evidence*. The Issue #131 precedent on
slab (2-region 2-group parity gap of 1.5 % invisible at 1G/1R or
2G/1R, surfaced only at 2G/2R because the multi-region branch was a
silently underconvergent quadrature where a closed-form integral
existed, see :ref:`theory-peierls-slab-polar-g5-diagnosis`) primed
the search for an analogous bug on the curvilinear side.

The investigation is recorded in
:file:`.claude/agent-memory/numerics-investigator/issue_100_class_b_mr_mg.md`
and pinned by the L1 test suite at
:file:`tests/derivations/test_peierls_rank_n_class_b_mr_mg.py`
(14 passing + 2 ``xfail strict=True`` regression-pinning the
catastrophe). The L0 catalog entry is **ERR-030** in
:file:`tests/l0_error_catalog.md`.

The MR×MG empirical evidence
----------------------------

Test cell: solid cyl / sph, ``radii = cp_module._RADII[n_regions]``
from :mod:`orpheus.derivations.cp_cylinder` /
:mod:`orpheus.derivations.cp_sphere` (``[1.0]`` for 1R,
``[0.5, 1.0]`` for 2R), with the
:mod:`orpheus.derivations._xs_library` ``LAYOUTS`` mapping
(``LAYOUTS = {1: ["A"], 2: ["A", "B"], 4: ["A", "D", "C", "B"]}``).
Material A is fuel (:math:`\nu\Sigma_f > 0`), material B is
moderator (:math:`\nu\Sigma_f = 0`, :math:`\sigma_s = 1.9`). The
load-bearing 2R configuration is fuel-A inner
(:math:`r \in [0, 0.5]`, :math:`\sigma_t = 1`) + moderator-B outer
(:math:`r \in [0.5, 1.0]`, :math:`\sigma_t = 2`); the analytical
reference :math:`k_\infty = 0.6479728` comes from
:meth:`~orpheus.derivations.cp_sphere._build_case`
(``"1g", 2``).k_inf via the analytical CP-matrix path, which is
independent of any Peierls assembly.

Quadrature: BASE preset
``(n_panels_per_region, p_order, n_angular, n_rho, n_surf_quad, dps)
= (2, 3, 24, 24, 24, 15)``. The same sweep at RICH preset
``(4, 5, 64, 48, 64, 20)`` is recorded in
:file:`derivations/diagnostics/diag_class_b_rank_n_rich_check.py`
and reproduces the BASE values to within :math:`0.022\,\%` —
**the catastrophe is structural, not quadrature noise.**

.. list-table:: Sphere rank-\ :math:`N`: signed :math:`(k_{\rm eff} - k_\infty)/k_\infty` at BASE preset
   :header-rows: 2
   :widths: 12 18 18 18 18 18

   * -
     - 1G/1R
     - 1G/2R
     - 2G/1R
     - 2G/2R
     -
   * - :math:`N`
     - :math:`k_\infty = 1.5`
     - :math:`k_\infty = 0.648`
     - (control)
     - :math:`k_\infty = 0.414`
     - notes
   * - 1 (Mark)
     - -27.0 %
     - -15.0 %
     - moderate
     - **-79 %**
     - rank-1 Mark closure floor
   * - 2 (Marshak)
     - **-1.10 %**
     - **+56.7 %**
     - moderate
     - large
     - 1R deceptive convergence; 2R sign flip
   * - 3
     - +2.22 %
     - +66.5 %
     - —
     - —
     - 1R modest; 2R plateau begins
   * - 5
     - ~+2.4 %
     - +66.8 %
     - —
     - —
     - plateau
   * - 8
     - ~+2.5 %
     - +66.8 %
     - —
     - —
     - plateau to wrong value

.. list-table:: Cylinder rank-\ :math:`N`: signed :math:`(k_{\rm eff} - k_\infty)/k_\infty` at BASE preset
   :header-rows: 2
   :widths: 12 18 18 18 18 18

   * -
     - 1G/1R
     - 1G/2R
     - 2G/1R
     - 2G/2R
     -
   * - :math:`N`
     - :math:`k_\infty = 1.5`
     - :math:`k_\infty = 0.99`
     - (control)
     - :math:`k_\infty = 0.740`
     - notes
   * - 1 (Mark)
     - -20.9 %
     - moderate
     - moderate
     - **-77 %**
     - rank-1 Mark closure floor
   * - 2 (Marshak)
     - +8.3 %
     - **+18.3 %**
     - moderate
     - large
     - 2R magnitude smaller than sphere
   * - 3
     - +26.7 %
     - large
     - —
     - —
     - cylinder rank-3 already divergent (Issue #112 Phase C)

**Three observations, in order of importance:**

1. **The 1R rank-N table is a calibration artifact, not a closure
   convergence.** The sphere 1R rank-2 = -1.10 % was the headline
   value supporting the published "rank-N Marshak converges on
   Class B" claim (peierls_geometry.py:3934-3961 docstring; original
   :ref:`peierls-rank-n-jacobian-section` section). The same
   closure produces +56.7 % on 1G/2R and plateaus to +67 % at high
   :math:`N` — *increasing* :math:`N` makes the 2R answer
   monotonically worse rather than better. A genuinely converging
   closure would have rank-2 reduce the rank-1 error in both
   topologies; this one does not.

2. **The 2G/2R rank-1 Mark floor is catastrophically large
   (-77 % cyl, -79 % sph).** The 2G amplification of the thin
   fast-group Mark closure error (the fast group is only
   :math:`\sigma_t \cdot R = 0.5` MFP thick at 1R, where Mark's
   isotropic re-entry assumption is poorly justified) folds into
   the multi-group fission/scatter coupling and produces wildly
   subcritical k\ :sub:`eff`. This is independent of the rank-:math:`N`
   bug (rank-1 has no closure-mixing problem) and reflects the
   Class B "no continuous Peierls reference" gap noted in the
   capability matrix at :ref:`theory-peierls-capabilities`.

3. **Cylinder 2R magnitude (+18.3 %) is smaller than sphere
   (+56.7 %) only because Issue #112 Phase C is already broken on
   cylinder rank-:math:`N` for separate reasons.** The cylinder
   rank-:math:`N` primitive
   :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_mode`
   uses the 2-D projected cosine :math:`\mu_{s,2D}` in the
   :math:`\mathrm{Ki}_1/d` integrand rather than the 3-D
   :math:`\mu_{s,3D} = \sin\theta_p \cdot \mu_{s,2D}` (Knyazev 1993
   :math:`\mathrm{Ki}_{2+k}` polynomial expansion). That separate
   floor masks the underlying mode-0 mismatch at low :math:`N`;
   cylinder rank-3 at 1R already diverges to +26.7 % from the
   :math:`\mathrm{Ki}_{k+2}` issue alone. The two bugs may share a
   root once a canonical re-derivation lands.

**Auxiliary numerical control: BASE↔RICH stability.**
:file:`derivations/diagnostics/diag_class_b_rank_n_rich_check.py`
re-runs the sphere 1G/2R rank-2 catastrophe at the RICH preset
``(4, 5, 64, 48, 64, 20)``. Result: BASE k\ :sub:`eff` = 1.0152, RICH
k\ :sub:`eff` = 1.0150, signed-error stability of :math:`0.022\,\%`.
The catastrophe is **not** Issue #114 ρ-quadrature noise (Issue #114's
floor at BASE is :math:`\sim 1\,\%` cylinder, :math:`\sim 10^{-3}`
sphere; the +57 % gap is :math:`50\times` larger than even the
cylinder floor and :math:`5 \times 10^4` times larger than the sphere
floor). It is structural.

Root cause — Probe G normalisation mismatch
-------------------------------------------

The full probe-cascade (Probes B–H, see
:file:`derivations/diagnostics/diag_class_b_rank_n_probe_{b,c,d,e,f,g}_*.py`)
ruled out:

- **Volume kernel multi-region path** (Probe B, ``vacuum_2r``) — the
  vacuum-BC :math:`K_{\rm vol}` MR routing is tight against the 1R
  baseline within :math:`\sim 2 \times 10^{-4}`. The volume kernel is
  not the bug.
- **Routing path under uniform :math:`\Sigma_t`** (Probe C,
  ``homogeneous_2r``, promoted to permanent passing test
  ``test_class_b_mr_routing_invariance_uniform_sigma``) — sphere/cyl
  with ``radii=[0.5, 1.0]`` and uniform :math:`\Sigma_t = 1` matches
  ``radii=[1.0]`` within :math:`5 \times 10^{-3}` (Issue #114 noise
  floor). The 2R routing path is consistent with 1R at the
  :math:`\sim 10^{-3}` level when the :math:`\Sigma_t` profile is
  flat — *the divergence requires a real :math:`\Sigma_t` step*.
- **Primitive convergence under quadrature refinement** (Probe D,
  ``primitive_quadrature``) — :func:`compute_P_esc_mode` /
  :func:`compute_G_bc_mode` plateau at :math:`\sim 10^{-5}` under
  ``n_angular`` refinement to 192. The primitives are essentially
  correct; no closed-form-avoidance anti-pattern à la Issue #131.
- **Conservation defect localisation** (Probe E,
  ``conservation``) — per-node :math:`(K \cdot 1 - \Sigma_t)/\Sigma_t`
  defects are 5–7 % rms in 2R-Z, but the 1R control has 9 % rms
  defect with k\ :sub:`eff` *still right*. **The conservation defect
  is not a strong predictor of k\ :sub:`eff` error in MR.** This is
  important methodologically: the conservation row-sum test at
  :file:`tests/derivations/test_peierls_rank_n_conservation.py` uses
  uniform :math:`\Sigma_t = 1` where :math:`K \cdot \mathbf 1 = \mathbf 1`
  holds by construction; **the test is structurally blind to MR
  mismatches** (the identity becomes an integrated identity, not
  pointwise, when :math:`\Sigma_t` is piecewise). The numerics-
  investigator's initial conservation diagnostic was structurally
  wrong for this reason; the lesson is captured in ERR-030 and
  pinned by the new MR routing-invariance test.
- **Per-mode K\ :sub:`bc` isolation** (Probe F,
  ``mode_isolation``) — adding mode-1 alone to mode-0 jumps
  k\ :sub:`eff` by **+84 %** on sphere 1G/2R (vs +35 % on the 1R
  control). Mode-1 contribution does not scale linearly between 1R
  and 2R — the mode-1 → mode-0 ratio depends sensitively on the
  :math:`\Sigma_t` profile in a way that no per-mode primitive
  could be wrong about (it would need to go wrong at *first contact*
  with mode 1).

Probe G (``normalization_mismatch``) is the localisation. It runs
the same 1R / 2R sweep with two variants of the mode-0 routing:

- **LEGACY** (production, the bug):
  :func:`compute_P_esc` + :func:`compute_G_bc` at mode 0; no
  :math:`(\rho_{\max}/R)^2` Jacobian. Modes :math:`n \ge 1` use
  :func:`compute_P_esc_mode` + :func:`compute_G_bc_mode` (with
  Jacobian).
- **CANONICAL**: :func:`compute_P_esc_mode` (n=0) +
  :func:`compute_G_bc_mode` (n=0) at mode 0, identical Jacobian-
  weighted form as modes :math:`n \ge 1`.

.. list-table:: Probe G — sphere :math:`(k_{\rm eff} - k_\infty)/k_\infty` at rank-2, BASE preset
   :header-rows: 1
   :widths: 35 32 33

   * - Configuration
     - LEGACY (production, mismatched)
     - CANONICAL (consistent, all-Jacobian)
   * - 1R control (:math:`\Sigma_t = 1`, :math:`k_\infty = 1.5`)
     - **-1.10 %**
     - -29.3 %
   * - 2R-Z (:math:`\Sigma_t = [1, 2]`, :math:`k_\infty = 0.648`)
     - **+56.7 %**
     - -28.0 %

The CANONICAL variant produces *consistent* errors across 1R and
2R (both plateau near -25 % to -29 % at high :math:`N`) — the
mismatch is gone, the routing is internally consistent. **But the
CANONICAL closure is uniformly worse**. The legacy mode-0 form is
not a *bug* in the sense of "wrong code"; it is a *calibration*:
the legacy :func:`compute_P_esc` was historically tuned to make the
rank-1 Mark closure (no rank-:math:`N` involved) approximately
right on solid cells. When summed with mode-:math:`n \ge 1` to form
a rank-:math:`N` outer-product expansion, that calibration breaks
the algebraic structure of the rank-:math:`N` partial-current basis
because the two terms do not live in the same expansion space.

**Why the legacy/canonical hybrid is structurally inconsistent.**
The Marshak partial-current moment of order :math:`n` from a uniform
unit volumetric source at radial node :math:`r_i` is, by
:eq:`peierls-rank-n-jacobian-derivation`,

.. math::
   :label: peierls-class-b-Jn-canonical

   J^{+}_n(r_i) \;=\; \frac{1}{A_d}\,
                      \int_\Omega \tilde P_n(\mu_{\rm exit})\,
                      \rho_{\max}^{\,2}(r_i,\Omega)\,e^{-\tau}\,
                      \mathrm d\Omega,

where the surface-to-observer Jacobian
:math:`\mathrm d A_s\,|\mu_s|\,\mathrm d\Omega_{\rm out} = d^{\,2}\,
\mathrm d\Omega_{r_i}` (with :math:`d = \rho_{\max}`) makes
:math:`|\mu_{\rm out}|` cancel against :math:`|\mu_s|` so that
*every* mode :math:`n` (including :math:`n = 0`) carries the
:math:`\rho_{\max}^{2}` weight in the observer-centred integrand.
The legacy mode-0 :func:`compute_P_esc` *omits* this Jacobian: it
returns the unweighted half-sphere outgoing-hemisphere integral
:math:`\int_{2\pi^+} e^{-\tau}\,\mathrm d\Omega` divided by an
isotropic-source escape-probability normalisation. The two integrals
span **different sub-spaces** of the half-range partial-current
basis — :math:`\mu`-weighted (Marshak inner product
:math:`\langle f, g\rangle_M = \int_0^1 f g\,\mu\,\mathrm d\mu`) vs
unweighted (Lambert inner product
:math:`\langle f, g\rangle_L = \int_0^1 f g\,\mathrm d\mu`). The
mismatch is exactly the Lambert / Marshak basis change documented
for the Class A F.4 closure at
:ref:`peierls-f4-rank-1-gauge-why` — the algebraic bridge between
the two is a non-trivial upper-bidiagonal change-of-basis matrix
:math:`M^{(N)}` (:eq:`peierls-M-rank-2`) which becomes a genuine
*basis rotation* (not a scalar gauge) at :math:`N \ge 2`. F.4
gets away with the Lambert/Marshak hybrid at rank-1 because the
mismatch is the scalar :math:`M^{(1)} = \sqrt{2}/2` and factors out;
the Class B legacy mode-0 / canonical mode-:math:`n \ge 1`
hybrid does the same trick at rank-1, but at rank :math:`\ge 2`
the basis rotation is genuine and the closure is structurally
inconsistent.

**Why pure-canonical is not the fix.** Switching mode-0 to
:func:`compute_P_esc_mode` (n=0) makes the rank-:math:`N` expansion
internally consistent, but the resulting closure is not the right
closure: it converges to a wrong limit (~-25 % across all 1R and
2R configurations at high :math:`N`). The reason is that the
canonical Marshak partial-current basis with mode-0 weighted by
:math:`\rho_{\max}^{2}` does not reduce to the production rank-1
Mark closure at :math:`N = 1` — it gives a different (worse)
single-mode closure. Pure-canonical breaks the rank-1 regression
gate and converges to the wrong answer. **The production rank-1
Mark closure and the canonical rank-:math:`N` Marshak closure
disagree at :math:`N = 1`**; reconciling them requires either
re-deriving rank-1 Mark in the canonical basis (which is not the
shipped Mark) or re-deriving mode-:math:`n \ge 1` so that the
:math:`N = 1` truncation is the shipped Mark, with mode-:math:`n
\ge 1` corrections living in a basis where they compose
consistently with that mode-0.

Production decision and forward path
------------------------------------

**Production today**: Class B ships only the rank-1 Mark closure
(``solve_peierls_*_1g`` with default ``n_bc_modes=1``), preserving
the historical calibration. Calls with ``n_bc_modes ≥ 2`` on
``boundary="white_rank1_mark"`` are still callable through
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_mg` —
they do not raise, but they are **not safe**: the closure is
structurally inconsistent in MR and quietly produces sign-flip
errors with no diagnostic. The new XFAIL-strict tests at
:file:`tests/derivations/test_peierls_rank_n_class_b_mr_mg.py`
(``test_class_b_mr_catastrophe_sphere_1g_2r_rank2``,
``test_class_b_mr_catastrophe_cylinder_1g_2r_rank2``) pin the
catastrophe magnitude so that any future improvement (or further
regression) is detectable; the tests flip from xfail-pass to
unexpected-pass when Issue #132 lands a corrective re-derivation.

**Issue #132 candidate fix paths** (from the issue body):

1. **Canonical mode-0 normalisation that reduces to Mark at
   :math:`N = 1`.** Re-derive :func:`compute_P_esc_mode` (n=0) and
   :func:`compute_G_bc_mode` (n=0) so that the single-mode closure
   matches the production rank-1 Mark byte-exactly while preserving
   the canonical Jacobian-weighted observer-centred form. This
   would require deriving rank-1 Mark in the Marshak basis from
   first principles and finding the per-geometry compensation
   factor; it may not exist (Mark and Marshak rank-1 may be
   genuinely different closures).

2. **Full Sanchez & McCormick 1982 §III.F.1 partial-current-moment
   basis.** Replace the entire rank-:math:`N` assembly
   (mode 0 + mode :math:`n \ge 1`) with the canonical Eqs. 165–169
   end-to-end. This drops the rank-1 Mark regression gate but
   places all modes in a single, internally consistent expansion
   space. The Class A rank-:math:`N` per-face primitives in
   :file:`peierls_geometry.py` (``compute_{P_esc, G_bc}_{outer,
   inner}_mode_marshak``) are already in this basis — Phase F.5
   retained them as tested dead code precisely for this kind of
   future research (see "Infrastructure retained" in
   :ref:`peierls-rank-n-per-face-closeout`). The Class B
   single-surface analogue would be a substantial lift but the
   primitives exist for the analogous Class A path.

3. **Knyazev 1993 :math:`\mathrm{Ki}_{2+k}` polynomial expansion
   for cylinder.** Phase C of Issue #112; carries a 3-D
   :math:`\mu_{s,3D} = \sin\theta_p \cdot \mu_{s,2D}` projection
   inside the :math:`\tilde P_n`-weighted integrand and produces
   higher-order Bickley functions. May share root with the
   mode-0 / mode-:math:`n \ge 1` mismatch documented here if the
   3-D projection is what brings mode-0 into the same expansion
   space as mode :math:`n \ge 1` for cylinder. Independent path
   for sphere.

References:

- Sanchez, R., & McCormick, N. J. (1982). "A review of neutron
  transport approximations." *Nuclear Science and Engineering*
  vol. 80, no. 4, pp. 481–535. §III.F.1, Eqs. 165–169.
- Knyazev, V. A. (1993). "Method of expanding angular distribution
  in spherical functions for cylindrical geometry." *Atomic Energy*
  vol. 74, no. 5, DOI 10.1007/BF00844623.
- :ref:`peierls-rank-n-per-face-closeout` — F.4 Schur-reduction
  precedent for hollow Class A; the basis-rotation argument
  (:eq:`peierls-M-rank-2`) applies analogously to the Class B
  legacy / canonical hybrid.

Infrastructure retained (not removed by this falsification)
------------------------------------------------------------

The following remain in the source tree and are not retired by this
falsification — they are the working primitives at :math:`N = 1`
(Mark) and the working :math:`(\rho_{\max}/R)^2` Jacobian at
mode :math:`n \ge 1`, so all of them are needed for Issue #132
re-derivation work:

- :func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction`
  — rank-1 Mark closure assembly. Production-shipped, not affected
  by the rank-:math:`N` mismatch (no mode mixing).
- :func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction_rank_n`
  — rank-:math:`N` Marshak closure assembly. Reachable but unsafe
  in MR; the docstring 1G/1R table at lines 3934-3961 stays for
  historical reference, with the retraction note at the parent
  section above.
- :func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
  — the routing function with the legacy/canonical mode-0 split at
  lines 3618-3642. Fix lives here once Issue #132 lands.
- :func:`~orpheus.derivations.peierls_geometry.compute_P_esc_mode`
  / :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_mode`
  — canonical Jacobian-weighted primitives for modes :math:`n \ge 1`.
  Correct as far as the Probe D quadrature refinement can verify
  (algebraic ~1/N convergence to ~1e-5 at ``n_angular=192``).
- :func:`~orpheus.derivations.peierls_geometry.compute_P_esc` /
  :func:`~orpheus.derivations.peierls_geometry.compute_G_bc` —
  legacy Mark-tuned primitives at mode 0. The historical
  calibration is the *correct* choice for rank-1 Mark; it is the
  *wrong* choice for mode 0 of a rank-:math:`N` Marshak expansion.
  Both clients must be supported until Issue #132 lands.

Diagnostic scripts in :file:`derivations/diagnostics/`:

- ``diag_class_b_rank_n_probe.py`` — full BASE-preset rank-:math:`N`
  sweep on sphere/cyl × {1G, 2G} × {1R, 2R} × N ∈ {1, 2, 3, 5, 8}.
  Source of the headline tables above.
- ``diag_class_b_rank_n_probe_b_vacuum_2r.py`` — Probe B (volume-
  kernel routing).
- ``diag_class_b_rank_n_probe_c_homogeneous_2r.py`` — Probe C
  (uniform-:math:`\Sigma_t` routing invariance, promoted to passing
  test).
- ``diag_class_b_rank_n_probe_d_primitive_quadrature.py`` — Probe D
  (primitive convergence under ``n_angular`` refinement).
- ``diag_class_b_rank_n_probe_e_conservation.py`` — Probe E (per-node
  :math:`(K \cdot 1 - \Sigma_t)/\Sigma_t` defect localisation).
- ``diag_class_b_rank_n_probe_f_mode_isolation.py`` — Probe F
  (per-mode K\ :sub:`bc` isolation: mode-1 alone vs mode-0+mode-1).
- ``diag_class_b_rank_n_probe_g_normalization_mismatch.py`` — Probe G
  (LEGACY vs CANONICAL mode-0 routing — source of the bug
  localisation table above).
- ``diag_class_b_rank_n_rich_check.py`` — BASE↔RICH stability of
  the sphere 1G/2R rank-2 catastrophe (0.022 % shift,
  confirms structural).

Test files (L1):

- :file:`tests/derivations/test_peierls_rank_n_class_b_mr_mg.py`
  (14 pass + 2 ``xfail strict=True``):

  - ``test_class_b_1g_1r_reproduces_published_table`` — sanity
    baseline against the docstring table at peierls_geometry.py
    lines 3934-3961, ±2 percentage points.
  - ``test_class_b_mr_routing_invariance_uniform_sigma`` — Probe C
    promoted; 2R routing path matches 1R within :math:`5 \times
    10^{-3}` (sphere) / :math:`2 \times 10^{-2}` (cylinder).
  - ``test_class_b_mr_catastrophe_sphere_1g_2r_rank2`` — XFAIL
    strict=True on Issue #132; pins the +57 % sphere catastrophe.
  - ``test_class_b_mr_catastrophe_cylinder_1g_2r_rank2`` — XFAIL
    strict=True on Issue #132; pins the +18 % cylinder analog.
  - ``test_class_b_2g_2r_rank1_mark_floor_pinned`` — pins the
    -77 %/-79 % 2G/2R rank-1 Mark floor as a regression gate.

Memory and plan files:

- :file:`.claude/agent-memory/numerics-investigator/issue_100_class_b_mr_mg.md`
  — full probe-cascade walkthrough.
- :file:`.claude/plans/issue-100-103-rank-n-class-b-multi-region.md`
  — the closing plan (§3 hypothesis under test, §6 decision tree,
  §7 acceptance criteria).
- :file:`tests/l0_error_catalog.md` ERR-030 — the bug catalog
  entry (failure-mode classification, how it hid, lesson).

Lesson (recorded in ERR-030)
----------------------------

A "rank-:math:`N` converges" claim must be verified at MR×MG —
single-region single-group passing rates are *degenerate evidence*
because two structurally-different normalisations can produce the
same k\ :sub:`eff` by historical calibration. For any partial-
current-moment closure, the rank-1 → rank-2 step must hold across
``radii=[1.0]`` AND ``radii=[0.5, 1.0]`` with non-trivial
:math:`\Sigma_t` breakpoints. The Issue #131 anti-pattern audit
("does the multi-region branch silently differ from the single-
region branch?") must be performed for every closure primitive, not
just the volume kernel. The conservation row-sum identity is *not*
a sufficient gate when :math:`\Sigma_t` is uniform (the identity
collapses to a tautology); it must be tested with piecewise
:math:`\Sigma_t` to discriminate real conservation from algebraic
self-consistency. This lesson generalises L19 (signed-error
stability under quadrature refinement) and L21 (basis refinement
cannot beat F.4 on Class A): on Class B, *configuration refinement*
(MR/MG) is an additional necessary axis the L19 quadrature axis
does not cover.

Session trail
-------------

This investigation spanned one session on
``feature/rank-n-class-b-mr-mg`` (2026-04-25), following the plan
landed at
:file:`.claude/plans/issue-100-103-rank-n-class-b-multi-region.md`:

- Probe-cascade dispatch by numerics-investigator agent through
  Probes B–H, localising the bug to the
  :func:`build_closure_operator` mode-0 routing.
- Test-file landing at
  :file:`tests/derivations/test_peierls_rank_n_class_b_mr_mg.py`
  with two XFAIL-strict pinning tests + three passing regression
  gates.
- ERR-030 catalog entry.
- Issue #132 filed (open) with the candidate fix paths.
- This Sphinx subsection (the falsification archive).


.. _peierls-rank-n-bc-closure-section:

Section 9 — Test-bed evidence from Phase 4.2 (cylinder)
========================================================

The cylindrical Peierls reference is the most fully-exercised
instantiation of the unified architecture. The test evidence
recorded in :ref:`peierls-cylinder-row-sum` of
:doc:`collision_probability` provides the numerical weight behind
the theoretical unification on this page. Three independent
checks are on the books for the cylinder:

**Row-sum identity (homogeneous).** For a bare homogeneous
cylinder at :math:`R = 10` mean free paths and
:math:`(n_\beta, n_\rho) = (20, 20)` to :math:`(32, 32)`
quadrature,
:math:`\max_i |\Sigma_t - \sum_j K_{ij}\,\Sigma_t(r_j)| < 10^{-3}`
at :math:`r_i \le R/2`. Tested in
``TestRowSumIdentity.test_interior_row_sum_equals_sigma_t`` in
``tests/derivations/test_peierls_cylinder_prefactor.py``.

**Sanchez–McCormick 1982 tie-point.** For a bare 1-G homogeneous
cylinder with :math:`\Sigma_t = 1` cm⁻¹,
:math:`(\Sigma_s, \nu\Sigma_f) = (0.5, 0.75)` giving :math:`k_\infty
= 1.5`, [Sanchez1982]_ Table IV reports :math:`R_{\rm crit} = 1.9798`
cm. Present solver gives
:math:`k_{\rm eff}(R = 1.9798) = 1.00421 \pm 10^{-5}` under
polar-quadrature refinement. 0.42 % offset from unity reflects the
ambiguous scatter/fission split in the Sanchez problem (see
:ref:`peierls-cylinder-row-sum`). Tested in
``TestSanchezTiePoint.test_k_eff_at_R_equals_1_dot_9798`` in
``tests/derivations/test_peierls_cylinder_eigenvalue.py``.

**Row-sum identity (multi-region).** On a two-annulus cylinder
with :math:`(r_1, R) = (3, 10)` MFP,
:math:`(\Sigma_{t,\mathrm{inner}}, \Sigma_{t,\mathrm{outer}})
= (0.8, 1.4)`, the multi-region identity
:math:`\sum_j K_{ij}\,\Sigma_t(r_j) = \Sigma_t(r_i)` holds to
0.5 % at every interior observer. Tested in
``TestMultiRegionKernel.test_K_applied_to_sig_t_gives_local_sig_t``
in ``tests/derivations/test_peierls_cylinder_multi_region.py``.

**Vacuum-BC thick-cylinder limit.** As :math:`R \to \infty`,
leakage vanishes and :math:`k_{\rm eff} \to k_\infty = 1.5`.
At :math:`R = 30` MFP the solver reaches :math:`k_{\rm eff} \approx
1.49` (0.5 % gap from :math:`k_\infty`), with monotone-increasing
:math:`k_{\rm eff}(R)` for :math:`R \in \{1.5, 3, 6, 12, 24\}` MFP.
Tested in ``TestVacuumBCThickLimit``.

**Peierls vs CP eigenvalue.** On a 1-G 1-region homogeneous bare
cylinder at :math:`R = 10` MFP, the Peierls :math:`k_{\rm eff}` and
the flat-source CP :math:`k_{\rm eff}` agree to 1 % — a cross-method
verification that relies on nothing shared between the two codes
except the integral transport equation they both discretise.

The slab Peierls reference has a parallel test-bed:
:ref:`peierls-scattering-convention` and the surrounding
Phase-4.1 tests exercise the :math:`E_1` kernel, the
product-integration weights, and the rank-2 white-BC closure. The
test depth is comparable; the evidence table is deliberately
not reproduced here since the geometry is specific.

The sphere Peierls reference **is shipped** (Phase-4.3; commits
``435c0b3``, ``9d03948``, ``cad2f0b``). The three-checks pattern —
row-sum, vacuum-BC leakage limit, CP cross-check — transfers
verbatim:

- **Row-sum identity (homogeneous).** At :math:`R = 10` MFP with
  :math:`(n_\theta, n_\rho) = (24, 24)`,
  :math:`\max_i |\Sigma_t - \sum_j K_{ij}| < 10^{-3}` at
  :math:`r_i \le R/2`. Tested in
  ``TestSphereRowSumIdentity.test_interior_row_sum_equals_sigma_t``
  in ``tests/derivations/test_peierls_sphere_prefactor.py``.
- **Vacuum-BC thick-sphere limit.** At :math:`R = 30` MFP,
  :math:`|k_{\rm eff} - k_\infty|/k_\infty < 10^{-2}`. Monotone
  growth in :math:`R` on :math:`R \in \{1.5, 3, 6, 12, 24\}` MFP.
  Tested in ``TestVacuumBCThickLimit``.
- **CP-vs-Peierls eigenvalue + flux shape.** At :math:`R = 10` MFP
  CP :math:`k_{\rm eff}` agrees with Peierls to < 2 %, and the
  volume-weighted normalised flux profiles agree to L2 < 5 %.
  Tested in ``TestCPvsPeierlsSphereAtThickR`` in
  ``tests/cp/test_peierls_sphere_flux.py``.

The rank-1 white-BC deficit is bounded by Issue #103 (N1); see
:ref:`issue-100-retraction` in :doc:`collision_probability` for
the numerical evidence.


Section 10 — Extending to a new geometry: a checklist
=====================================================

.. note::

   **Status as of 2026-04-18.** The three standard 1-D geometries are
   complete:

   - **Slab (1-D Cartesian)**: Phase-4.1 ✓ (``peierls_slab.py``,
     :math:`E_1` kernel, rank-2 white-BC closure).
   - **Cylinder (1-D radial)**: Phase-4.2 ✓
     (``peierls_cylinder.py`` + ``peierls_geometry.py``,
     :math:`\mathrm{Ki}_1` kernel, rank-1 white-BC closure).
   - **Sphere (1-D radial)**: Phase-4.3 ✓
     (``peierls_sphere.py`` + ``peierls_geometry.py``,
     :math:`e^{-\tau}` kernel, rank-1 white-BC closure with
     geometry-aware :math:`R^{2}` surface divisor).

   The unified :class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
   covers both curvilinear cases; any further 1-D extension (e.g. a
   1-D Cartesian Nyström sharing the cylinder's sweep machinery)
   follows the steps below.

Based on the architecture documented above, adding a Peierls Nyström
reference for a new 1-D geometry requires:

1. **Derive** :math:`\kappa_d(\tau)` and :math:`C_d` by writing the
   native point kernel, integrating out symmetry directions, and
   performing the Jacobian cancellation of Section 3. Verify the
   unit-area identity :math:`\int_0^\infty \kappa_d(u)\,\mathrm du
   \cdot S_d = 1`.

2. **Write closed forms** for :math:`r'(r,\rho,\Omega)` and
   :math:`\rho_{\max}(r,\Omega)`. For any geometry where the
   boundary is a quadratic surface (lines, cylinders, spheres,
   ellipsoids) these are elementary.

3. **Reuse** the Lagrange-basis, composite-GL, optical-depth walker,
   and power-iteration primitives from
   :mod:`orpheus.derivations.peierls_cylinder` verbatim. The walker
   must be taught the new boundary-crossing algebra, but its
   structure (sort crossings, accumulate :math:`\Sigma_{t,k}\Delta s`)
   is unchanged.

4. **Implement the four primitives** as Python functions or methods
   on a ``PeierlsGeometry`` concrete class, and assemble the kernel
   via the common scaffolding.

5. **Test-bed**: homogeneous row-sum identity
   (:math:`\sum_j K_{ij}\Sigma_t(r_j) = \Sigma_t(r_i)`), a literature
   tie-point (Sanchez Table for the sphere; slab equivalents for
   the :math:`E_1` kernel), vacuum-BC thick-cell limit, and
   Peierls-vs-CP eigenvalue agreement.

6. **White-BC closure**: start with the rank-1 approximation
   (identical structure across curvilinear geometries) and document
   the pointwise-Nyström deficit size as a function of cell radius.
   Issue #100's resolution path — augmented Nyström system or
   higher-rank angular decomposition — applies to all curvilinear
   geometries.

The effort to add a new geometry is therefore bounded: three
analytical derivations (:math:`\kappa_d`, :math:`r'`,
:math:`\rho_{\max}`), one small implementation module, and four
standard tests.


.. seealso::

   :doc:`collision_probability` — geometry-specific Peierls
   sections:

   - :ref:`peierls-conservation` and the slab :eq:`peierls-equation`
     — :math:`E_1` kernel and :ref:`peierls-scattering-convention`.
   - :eq:`peierls-cylinder-equation` and :eq:`peierls-cylinder-nystrom`
     — cylinder :math:`\mathrm{Ki}_1` kernel, polar form,
     Lagrange interpolation, and the chord-form pivot.

   :doc:`../verification/reference_solutions` — :math:`E_n` and
   :math:`\mathrm{Ki}_n` kernel primitives
   (:eq:`en-definition`, :eq:`kin-definition`,
   :eq:`en-kernel-integral`, :eq:`kin-kernel-derivative`).

   :mod:`orpheus.derivations.peierls_slab` — Phase-4.1 slab Peierls
   reference (:math:`E_1`).

   :mod:`orpheus.derivations.peierls_cylinder` — Phase-4.2 cylinder
   Peierls reference (:math:`\mathrm{Ki}_1`), including the
   vacuum-BC driver
   (:func:`~orpheus.derivations.peierls_cylinder.solve_peierls_cylinder_1g`
   with ``boundary="vacuum"``) and the rank-1 white-BC correction
   (:func:`~orpheus.derivations.peierls_cylinder.build_white_bc_correction`).

   :mod:`orpheus.derivations.peierls_sphere` — Phase-4.3 sphere
   Peierls reference (:math:`e^{-\tau}`), a thin facade over the
   unified :class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
   (``kind = "sphere-1d"``). Eigenvalue driver
   :func:`~orpheus.derivations.peierls_sphere.solve_peierls_sphere_1g`;
   white-BC correction
   :func:`~orpheus.derivations.peierls_sphere.build_white_bc_correction`.

   :mod:`orpheus.derivations.peierls_geometry` — unified
   polar-form Nyström infrastructure; ``CurvilinearGeometry``
   singletons ``CYLINDER_1D`` and ``SPHERE_1D``.

   GitHub Issue `#100
   <https://github.com/deOliveira-R/ORPHEUS/issues/100>`_ —
   Sphere Peierls white-BC rank-1 closure (**retracted** /
   closed-by-fix; the original :math:`k_{\rm eff} \approx 6.7`
   failure was a missing :math:`R^{2}` surface divisor, not a
   structural rank-1 defect). See :ref:`issue-100-retraction`.

   GitHub Issue `#103
   <https://github.com/deOliveira-R/ORPHEUS/issues/103>`_ — N1:
   Higher-rank white-BC closure for pointwise Peierls (cyl +
   sphere); open.


Section 11 — The three-tier integration hierarchy
=================================================

Sections 1–10 unified the **pointwise** (Nyström) Peierls equation
across slab, cylinder, and sphere, treating :math:`\varphi(r)` as a
continuous function sampled at radial collocation nodes. The
flat-source CP method used by :mod:`orpheus.cp.solver` operates at a
different rung of the integration ladder: it averages over entire
regions rather than sampling at points. The two methods consume the
**same** 3-D point kernel :eq:`peierls-point-kernel-3d`; they differ
only in how many successive spatial integrations of that kernel have
been carried out in closed form before numerics takes over.

This section and the three that follow extend the unification up to
the flat-source level. The key organising principle is the
**integration hierarchy**: each successive integration of the 3-D
isotropic point kernel defines a new kernel level, and the slab,
cylinder, and sphere each occupy the same three levels with different
special-function names.

.. _cp-three-tier-hierarchy-note:

The three kernel levels
-----------------------

Let :math:`R = |\mathbf r - \mathbf r'|` be the centre-to-centre
distance, :math:`\tau = \Sigma_t R` the line-integrated optical
depth, and :math:`d` the native geometric dimensionality
(1 for slab, 2 for cylinder, 3 for sphere). Define three kernel
levels:

- **Level 0 — the 3-D point kernel.** The "un-integrated" Green's
  function for the isotropic point emitter,
  :math:`G_{3\mathrm D}(R) = e^{-\tau}/(4\pi R^{2})`, identified as
  :eq:`peierls-point-kernel-3d`. It is native 3-D for *every*
  geometry — the geometry only enters through the optical-path
  integral :math:`\tau(r,r')` and through which symmetry directions
  one elects to integrate out.
- **Level 1 — the pointwise (Peierls) kernel.** Integrating the 3-D
  point kernel over the unbroken symmetry directions of each geometry
  yields the one-argument kernel
  :math:`\kappa_d(\tau)` used by :eq:`peierls-unified`:

  .. math::

     \kappa_1^{\rm slab}(\tau) \;=\; \tfrac{1}{2} E_1(\tau),\qquad
     \kappa_1^{\rm cyl}(\tau)  \;=\; \frac{\mathrm{Ki}_1(\tau)}{2\pi},\qquad
     \kappa_1^{\rm sph}(\tau)  \;=\; \frac{e^{-\tau}}{4\pi}.

  These are the Level-1 kernels *already derived* in Section 2. They
  are consumed by the pointwise Peierls Nyström drivers
  (:mod:`~orpheus.derivations.peierls_slab`,
  :mod:`~orpheus.derivations.peierls_cylinder`,
  :mod:`~orpheus.derivations.peierls_sphere`).
- **Level 2 — the partial-current / escape kernel.** Integrating the
  Level-1 kernel *once more* along a line of flight (the neutron's
  path through a single region) gives the Level-2 kernel that
  underwrites *escape* and *partial-current* probabilities:

  .. math::

     \kappa_2^{\rm slab}(\tau) \;=\; E_2(\tau),\qquad
     \kappa_2^{\rm cyl}(\tau)  \;=\; \mathrm{Ki}_2(\tau),\qquad
     \kappa_2^{\rm sph}(\tau)  \;=\; e^{-\tau}.

- **Level 3 — the flat-source / CP kernel.** Integrating Level-2 a
  *second* time — specifically over the spatial extent of the
  *target* region, with a flat source assumed in the *emitting*
  region — gives the Level-3 kernel that the flat-source CP
  second-difference formula evaluates at four arguments per
  :math:`P_{ij}` element:

  .. math::

     \kappa_3^{\rm slab}(\tau) \;=\; E_3(\tau),\qquad
     \kappa_3^{\rm cyl}(\tau)  \;=\; \mathrm{Ki}_3(\tau),\qquad
     \kappa_3^{\rm sph}(\tau)  \;=\; e^{-\tau}.

The ladder can be stated compactly as the differential identities

.. math::
   :label: cp-kernel-differential-identities

   E_n'(\tau) \;=\; -E_{n-1}(\tau),\qquad
   \mathrm{Ki}_n'(\tau) \;=\; -\mathrm{Ki}_{n-1}(\tau),

.. vv-status: cp-kernel-differential-identities tested

valid for all :math:`n \ge 1` (A&S 5.1.26 and 11.2.11). These
identities are already implemented as
:func:`~orpheus.derivations._kernels.e_n_derivative` and
:func:`~orpheus.derivations._kernels.ki_n_derivative`, and they
are tested term-by-term at L0 by
``tests/derivations/test_kernels.py`` via finite-difference
agreement with the direct mpmath evaluators
:func:`~orpheus.derivations._kernels.e_n_mp` and
:func:`~orpheus.derivations._kernels.ki_n_mp`. Passing up the ladder
from :math:`E_{n-1} \to E_n` or :math:`\mathrm{Ki}_{n-1} \to
\mathrm{Ki}_n` is therefore a pure antiderivation — one indefinite
integral against the same variable.

Hierarchy in one picture
------------------------

.. list-table:: Three-tier integration hierarchy
   :header-rows: 1
   :widths: 16 28 28 28

   * - Level
     - Slab
     - Cylinder
     - Sphere
   * - **L0** — 3-D point kernel
     - :math:`e^{-\tau}/(4\pi R^{2})`
     - :math:`e^{-\tau}/(4\pi R^{2})`
     - :math:`e^{-\tau}/(4\pi R^{2})`
   * - **L1** — Peierls kernel
       (pointwise)
     - :math:`\tfrac{1}{2}E_1(\tau)`
     - :math:`\mathrm{Ki}_1(\tau)/(2\pi)`
     - :math:`e^{-\tau}/(4\pi)`
   * - **L2** — escape / partial-current
     - :math:`E_2(\tau)`
     - :math:`\mathrm{Ki}_2(\tau)`
     - :math:`e^{-\tau}`
   * - **L3** — flat-source CP (second-difference antiderivative)
     - :math:`E_3(\tau)`
     - :math:`\mathrm{Ki}_3(\tau)`
     - :math:`e^{-\tau}`

.. note::

   **The sphere does not "promote" through the ladder.** Because
   :math:`e^{-\tau}` is its own antiderivative up to sign, all three
   sphere levels use the *same* special function. This is a
   coincidence of the 3-D point kernel being the identity
   :math:`\tfrac{\mathrm d}{\mathrm d\tau} e^{-\tau} = -e^{-\tau}` —
   the same coincidence that lets the sphere skip the Bickley /
   exponential-integral tabulation enterprise entirely. It is **not**
   a sign that the sphere is "skipping levels"; the integration is
   still being performed, it just happens to close on itself.

   In the CP literature ([BellGlasstone1970]_ §2.7, [Stamm1983]_
   §6.4) this is sometimes reported as "the sphere kernel needs no
   special functions". That statement is correct only after the three
   levels have been identified — before that, it sounds like an
   asymmetry between the geometries. The three-tier hierarchy makes
   the symmetry manifest: what differs between geometries is the
   *dimensionality* of the outer integral at each level (§13), not
   the kernel ladder itself.

Scope of the unification
------------------------

Sections 12–14 extend the Phase-4.2 unified architecture to **Level
3** — i.e., to the flat-source CP matrix. The current flat-source CP
modules (:mod:`~orpheus.derivations.cp_slab`,
:mod:`~orpheus.derivations.cp_cylinder`,
:mod:`~orpheus.derivations.cp_sphere`) each implement the same
geometry at the same kernel level but in three separate files, with
the same :math:`\Delta^{2}` operator rewritten once per geometry and
the same outer y-quadrature duplicated between the two curvilinear
cases. Phase B of the CP refactor (see GitHub Issue
`#107 <https://github.com/deOliveira-R/ORPHEUS/issues/107>`_) will
collapse them into a single ``cp_geometry.py`` module, exactly
mirroring the Phase-4.2 collapse of the pointwise modules into
:mod:`~orpheus.derivations.peierls_geometry`.

Sections 12–14 present the derivational target for that refactor;
Section 15 revisits the escape probability as the explicit Level-2
"bridge" between Level 1 and Level 3; and Section 16 documents the
retirement of the legacy ``BickleyTables`` tabulation, which the
Phase B.4 commit obsoleted in favour of a Chebyshev interpolant of
the canonical mpmath-backed :math:`\mathrm{Ki}_3`.


Section 12 — The flat-source integral: going from Level 1 to Level 3
====================================================================

Starting point. Consider a target region :math:`V_i` and a source
region :math:`V_j` with volumetric emission density
:math:`q(\mathbf r') = q_j` constant on :math:`V_j` (the flat-source
assumption). The region-averaged collision rate in :math:`V_i`
produced by :math:`V_j` is

.. math::
   :label: cp-flat-source-double-integral

   \Sigma_{t,i}\,\bar\varphi_i\,V_i
     \;=\; \int_{V_i}\!\mathrm dV \int_{V_j}\!\mathrm dV'\,
           G_d\bigl(|\mathbf r - \mathbf r'|\bigr)\,q_j,

.. vv-status: cp-flat-source-double-integral tested

where :math:`G_d` is the Level-1 kernel already pre-integrated against
the symmetry directions of the geometry
(:eq:`peierls-ki1-derivation`, :eq:`peierls-e1-derivation`, or the
3-D point kernel itself for the sphere). The flat-source
:math:`P_{ij}` matrix is obtained by factoring out
:math:`q_j V_j / (4\pi)` or the geometry-appropriate normalisation;
the quantity we track here is the reduced collision probability

.. math::

   \text{rcp}_{ij} \;\equiv\;
     \int_{V_i}\!\mathrm dV \int_{V_j}\!\mathrm dV'\,
     G_d\bigl(|\mathbf r - \mathbf r'|\bigr),

which differs from :math:`P_{ij}\,\Sigma_{t,i}\,V_i` only by the
kernel's normalisation convention and by the white-BC closure added
on top (see :eq:`p-inf` in :doc:`collision_probability`).

The derivation in :doc:`collision_probability`, §
:ref:`second-diff-derivation`, performs the double integral in
**chord coordinates** :math:`(y, s, t)` where :math:`y` is the impact
parameter of the chord through the pair of annuli, :math:`s` is the
birth position along the chord within :math:`V_j`, and :math:`t` is
the collision position along the chord within :math:`V_i`. In those
coordinates, the two spatial integrations over :math:`s` and
:math:`t` are integrations of :math:`G_d` along a **one-dimensional
optical-path variable**, so each brings the kernel up one level in
the hierarchy.

That derivation — the integration-by-parts chain from
:eq:`peierls-point-kernel-3d` to the four-term
:eq:`rcp-from-double-antideriv` — is already presented at full length
in :doc:`collision_probability`. We restate its result here in the
language of the three-tier hierarchy and use it to identify the
**geometry-invariant operator** that underwrites the Phase B unified
architecture.

Inner integration: :math:`V_j \to` Level 2
------------------------------------------

Fix a chord of impact parameter :math:`y` and a collision position
:math:`t` along the chord in :math:`V_i`. Parametrise the source
point along the same chord by :math:`s`; the optical distance between
source and collision point along the chord is :math:`(\tau_j - s) +
g + t`, where :math:`\tau_j` is the chord's optical traversal of the
source region and :math:`g` is the optical gap between the two
regions' chord intersections. With :math:`u = (\tau_j - s) + g + t`:

.. math::
   :label: cp-inner-integral-antiderivative

   I(t) \;=\; \int_{0}^{\tau_j}\! F_1\bigl((\tau_j - s) + g + t\bigr)\,\mathrm ds
        \;=\; \int_{g+t}^{\tau_j + g + t}\! F_1(u)\,\mathrm du
        \;=\; \hat F_1\bigl(\tau_j + g + t\bigr) - \hat F_1\bigl(g + t\bigr),

.. vv-status: cp-inner-integral-antiderivative tested

where :math:`F_1` is the **Level-1 chord kernel** — the one-argument
function of optical path that results from the symmetry integrations
of Section 2 (:math:`E_1` for the slab, :math:`\mathrm{Ki}_1`
*weighted by the azimuthal Jacobian that the chord form collapses*
for the cylinder, :math:`e^{-\tau}` for the sphere along each chord)
— and :math:`\hat F_1(x) = \int_0^x F_1(u)\,\mathrm du` is its
antiderivative.

The antiderivative :math:`\hat F_1` is **exactly the Level-2 kernel
of §11**:

.. math::

   \widehat{E_1}     &\;=\; -E_2 \quad (\text{modulo the boundary term}), \\
   \widehat{\mathrm{Ki}_1} &\;=\; -\mathrm{Ki}_2, \\
   \widehat{e^{-\tau}} &\;=\; -e^{-\tau}.

The minus signs are absorbed into the convention
:math:`E_n'(\tau) = -E_{n-1}(\tau)` (A&S 5.1.26) — i.e., raising
:math:`n` by 1 **is** antiderivation, and the overall sign is
chosen such that :math:`E_n(0)` is finite (positive) for
:math:`n \ge 2`. The takeaway is physical rather than
conventional: *the inner integration over the source region
promotes the kernel from Level 1 to Level 2*. Level 2 is the
escape-probability level (§15) — an :math:`F_2` evaluation at the
right argument is the uncollided probability that a neutron emitted
from a point passes through a specific optical thickness before its
first collision.

Outer integration: :math:`V_i \to` Level 3
------------------------------------------

With :math:`I(t)` in hand, integrate :math:`I(t)` over the collision
position :math:`t \in [0, \tau_i]`:

.. math::
   :label: cp-outer-integral-antiderivative

   \text{rcp}_{ij}^{(y)}
     \;=\; \int_{0}^{\tau_i}\! I(t)\,\mathrm dt
     \;=\; \int_{0}^{\tau_i}\!\Bigl[\hat F_1(\tau_j + g + t)
                               - \hat F_1(g + t)\Bigr]\mathrm dt.

.. vv-status: cp-outer-integral-antiderivative tested

Substituting :math:`\hat F_1 = -F_2` (with :math:`F_2` the Level-2
kernel) and integrating once more gives :math:`\hat F_2 = F_3`,
the Level-3 kernel:

.. math::
   :label: cp-flat-source-derivation

   \text{rcp}_{ij}^{(y)}
     \;=\; F_3(g) - F_3(g + \tau_i) - F_3(g + \tau_j)
             + F_3(g + \tau_i + \tau_j),

.. vv-status: cp-flat-source-derivation tested

where the superscript :math:`(y)` reminds us that this is the
contribution to :math:`\text{rcp}_{ij}` from one chord at impact
parameter :math:`y`. The four terms come from the four corners of
the integration box :math:`\{(s, t) : s \in [0, \tau_j],
t \in [0, \tau_i]\}` under the antiderivation chain — i.e., from
evaluating the double antiderivative :math:`\hat{\hat F_1} = F_3`
at the four corners :math:`(0, 0), (\tau_i, 0), (0, \tau_j),
(\tau_i, \tau_j)` after the change of variable to optical path. The
full step-by-step derivation is the content of
:eq:`rcp-from-double-antideriv` in :doc:`collision_probability`.

The four-argument structure is the geometry-invariant core of
flat-source CP. Factor it out as an **operator**:

.. math::
   :label: cp-second-difference-operator

   \Delta^{2}\!\bigl[\mathcal F\bigr]\!\bigl(\tau_i, \tau_j;\,\mathrm{gap}\bigr)
     \;\equiv\;
     \mathcal F(\mathrm{gap})
     \;-\; \mathcal F(\mathrm{gap}+\tau_i)
     \;-\; \mathcal F(\mathrm{gap}+\tau_j)
     \;+\; \mathcal F(\mathrm{gap}+\tau_i+\tau_j).

.. vv-status: cp-second-difference-operator tested

This is nothing more than the second finite difference of
:math:`\mathcal F` on the rectangular grid
:math:`\{(\mathrm{gap},\mathrm{gap}+\tau_j)\}\times
\{(\mathrm{gap},\mathrm{gap}+\tau_i)\}`. The two coordinate
"differences" pick up the two optical traversals :math:`\tau_i` and
:math:`\tau_j`, and the mixed term
:math:`\mathcal F(\mathrm{gap}+\tau_i+\tau_j)` closes the rectangle.

**The operator :math:`\Delta^{2}` is geometry-invariant.** It knows
nothing about slab vs cylinder vs sphere; it knows only that
:math:`\mathcal F` is a scalar function of one scalar argument. What
makes the geometry enter the *reduced collision probability* is the
**choice of** :math:`\mathcal F_d`:

.. list-table:: Level-3 kernels :math:`\mathcal F_d` per geometry
   :header-rows: 1
   :widths: 16 24 28 32

   * - Geometry
     - :math:`\mathcal F_d`
     - Small-:math:`\tau` value
     - Large-:math:`\tau` tail
   * - Slab
     - :math:`E_3(\tau)`
     - :math:`E_3(0) = 1/2`
     - :math:`E_3(\tau) \to e^{-\tau}/\tau^{3}` (A&S 5.1.51)
   * - Cylinder
     - :math:`\mathrm{Ki}_3(\tau)`
     - :math:`\mathrm{Ki}_3(0) = \pi/4`
     - :math:`\mathrm{Ki}_3(\tau) \to \sqrt{\pi/(2\tau)}\,e^{-\tau}`
   * - Sphere
     - :math:`e^{-\tau}`
     - :math:`e^{0} = 1`
     - :math:`e^{-\tau}` (self-similar)

The operator is **shared**; the table is the only per-geometry
data. This is the architectural lever for Phase B — one
``_second_difference`` function in ``cp_geometry.py`` serves all
three geometries; three one-line kernel methods
(``kernel_F3_slab = e_n(3, ·)`` etc.) distinguish them.

.. tip::

   **Why ":math:`\Delta^{2}`" and not ":math:`\Delta_2`"?** The
   existing flat-source derivation in :doc:`collision_probability`
   (:eq:`second-diff-general`, :eq:`rcp-from-double-antideriv`)
   writes the operator as :math:`\Delta_2[F]`. The two notations
   name the same object. We adopt :math:`\Delta^{2}` on this page
   because the *unified* presentation emphasises that the four-term
   formula is the **second** (finite) difference in the discrete
   sense — one difference per region's chord traversal — which is a
   cleaner abstraction than reading the subscripted 2 as "two
   variables". The :math:`\Delta_2` notation is still correct in
   its source document; no equation labels collide.

A Level-2 sanity check: the one-region limit
--------------------------------------------

As a check on :eq:`cp-flat-source-derivation`, consider :math:`i = j`
with :math:`\mathrm{gap} = 0` and :math:`\tau_i = \tau_j = \tau`
(self-collision within one region). The operator evaluates to

.. math::

   \Delta^{2}[\mathcal F_d](\tau, \tau;\,0)
     \;=\; \mathcal F_d(0) - 2\,\mathcal F_d(\tau) + \mathcal F_d(2\tau).

For the slab, :math:`\mathcal F_d = E_3`, :math:`E_3(0) = 1/2`, and
the small-:math:`\tau` expansion
:math:`E_3(\tau) = 1/2 - \tau + \tfrac{\tau^{2}}{2}(3/2 - \ln\tau) +
O(\tau^{3})` gives
:math:`\Delta^{2}[E_3](\tau,\tau;0) = \tau\cdot[2 - (\tau/2)(\text{stuff})]`,
recovering the small-:math:`\tau` limit :math:`\text{rcp}_{ii}
\sim \tau\,V_i` that the self-collision probability must satisfy.
This limit is checked inside the diagonal self-collision formula
documented at :eq:`self-slab`, :eq:`self-cyl`, and :eq:`self-sph` of
:doc:`collision_probability`.

The derivation source of record
-------------------------------

The IBP chain above is verified programmatically by the SymPy script
embedded in :ref:`second-diff-derivation` (:doc:`collision_probability`,
lines under "SymPy verification of the four-term structure"). That
script builds the double integral :math:`\int_0^{\tau_i}\int_0^{\tau_j}
F_1((\tau_j - s) + g + t)\,\mathrm ds\,\mathrm dt` for a *generic*
:math:`F_1` and asserts symbolically that the result equals the
four-term :eq:`cp-second-difference-operator`. The assertion holds
without specialising :math:`F_1` to :math:`E_1`, :math:`\mathrm{Ki}_1`,
or :math:`e^{-\tau}`, which is the strongest possible form of the
claim that the operator is geometry-invariant.

For Phase B.3, the same SymPy script will be lifted into a proper
``derivations/cp_geometry.py`` derivation module (with a
``derive_second_difference()`` function returning the SymPy
expression tree) and a test case
``test_second_difference_operator_is_geometry_invariant`` will
exercise the symbolic claim at L1. Today the verification is
documented but not automated.


Section 13 — Geometry-specific outer integration
================================================

The operator :math:`\Delta^{2}` and its kernel table collapse the
*inner* algebraic structure of flat-source CP to a single shared
routine. What still varies between geometries is the **outer**
integration — how :math:`\text{rcp}_{ij}^{(y)}` is aggregated over
the chord family. The outer integration captures the *dimensionality
of the physical source region*: a slab region is 1-D (no outer
integral needed), a cylindrical region has one transverse dimension
(impact parameter :math:`y`), a spherical region has the same
:math:`y` dimension plus a measure factor.

Concretely:

.. math::
   :label: cp-unified-outer-integration

   \text{rcp}_{ij} \;=\;
   \begin{cases}
     \tfrac{1}{2\Sigma_{t,i}}\,\Delta^{2}[E_3](\tau_i,\tau_j;\,\mathrm{gap})
        & d = 1 \text{ (slab)}, \\[6pt]
     \displaystyle
     \int_{0}^{R}\! 2\,\bigl[\Delta^{2}[\mathrm{Ki}_3]_{\rm SS}
                        + \Delta^{2}[\mathrm{Ki}_3]_{\rm TC}\bigr]\,\mathrm dy
        & d = 2 \text{ (cylinder)}, \\[6pt]
     \displaystyle
     \int_{0}^{R}\! 2\,\bigl[\Delta^{2}[e^{-\tau}]_{\rm SS}
                        + \Delta^{2}[e^{-\tau}]_{\rm TC}\bigr]\,y\,\mathrm dy
        & d = 3 \text{ (sphere)},
   \end{cases}

.. vv-status: cp-unified-outer-integration tested

where "SS" and "TC" are the **same-side** and **through-centre**
chord branches documented at :eq:`tau-m`, :eq:`tau-p`,
:eq:`dd-slab`, :eq:`dc-slab`, :eq:`second-diff-cyl`, and
:eq:`second-diff-sph` of :doc:`collision_probability`.

.. list-table:: Outer-integration rule per geometry
   :header-rows: 1
   :widths: 16 22 28 34

   * - Geometry
     - Outer variable
     - Measure
     - Origin of the measure
   * - Slab
     - None (:math:`y \equiv 0`)
     - —
     - Region is a 1-D interval; the chord *is* the region.
   * - Cylinder
     - :math:`y \in [0, R]`
     - :math:`2\,\mathrm dy`
     - Per-chord length :math:`2\sqrt{R^{2}-y^{2}}`-equivalent; the
       factor of 2 accounts for the two sides of the symmetry axis.
   * - Sphere
     - :math:`y \in [0, R]`
     - :math:`2y\,\mathrm dy`
     - Spherical ring area :math:`2\pi y\,\mathrm dy` divided by the
       :math:`\pi` the kernel already absorbs; see
       :mod:`~orpheus.derivations.cp_sphere` line
       ``y_wts = y_wts * y_pts``.

Same-side and through-centre branches
-------------------------------------

For any chord with :math:`y < \min(r_{i-1}, r_{j-1})` (both regions
intersect on both sides of the symmetry axis), the chord family
splits into two branches:

- **Same-side (SS).** The chord intersects both regions on the same
  side of the axis. The optical gap is
  :math:`\mathrm{gap}_{\rm SS} = |\text{optical position of
  region-}j - \text{region-}i|`, computed via the chord-walker in
  :func:`~orpheus.derivations._kernels.chord_half_lengths` followed
  by optical-depth accumulation along the sorted annular crossings.
- **Through-centre (TC).** The chord intersects region :math:`i` on
  one side of the axis and region :math:`j` on the other. The optical
  gap is the full chord path across the central regions plus both
  regions' inner half-chords:
  :math:`\mathrm{gap}_{\rm TC} = (\text{optical position of }i) +
  (\text{optical position of }j)`.

Both branches use **the same** :math:`\Delta^{2}` operator; only the
``gap`` argument differs. The SS branch becomes zero when
:math:`y > \min(r_{i-1}, r_{j-1})` (one of the regions doesn't have a
hollow core at that :math:`y`), in which case only the TC branch
survives. This is the reason the :math:`\Delta^{2}[\mathcal F]_{\rm SS}`
term in :eq:`cp-unified-outer-integration` is conditional.

This branch structure is **identical** between cylinder and sphere.
In code it is one chord-walker, one ``bnd_pos`` array, one
``gap_d = max(bnd_pos[j] - bnd_pos[i+1], 0)`` expression. Both
:mod:`~orpheus.derivations.cp_cylinder` and
:mod:`~orpheus.derivations.cp_sphere` already share this structure
verbatim (compare the respective ``for j in range(N_reg):`` inner
loops); the only differences between the two modules are the kernel
function and the final :math:`y`-weighting. This is the raw material
for the Phase B unification.

The slab as a degenerate outer integral
---------------------------------------

The slab deserves a brief derivational comment because it looks like
a different case (no :math:`y`-quadrature) but is actually the
degenerate limit of the curvilinear formula. A slab region is a 1-D
interval, so the "chord family" parametrised by :math:`y` collapses
to a single chord; the outer integral reduces to a Dirac delta at
:math:`y = 0`, giving the direct algebraic
:math:`\text{rcp}_{ij} = \tfrac{1}{2\Sigma_{t,i}}
\Delta^{2}[E_3](\tau_i,\tau_j;\,\mathrm{gap})` formula. The factor
of :math:`1/(2\Sigma_{t,i})` is the slab's residual angular
normalisation (the :math:`1/2` of :math:`E_3` after its 1-D
angular integration, divided by the :math:`\Sigma_t` that turns
"optical rcp" into "linear-distance rcp"). See :eq:`self-slab` for
the self-region form.

The SS/TC distinction is vacuous for the slab: regions have no
"other side of the axis" to route through, so the TC term is zero
and only SS survives. This is another reason Phase B's
``FlatSourceCPGeometry`` will carry a ``has_through_centre`` flag
(True for curvilinear, False for slab) — the same kernel-evaluation
pipeline then handles all three cases.


Section 14 — The unified ``FlatSourceCPGeometry`` abstraction
=============================================================

Sections 12 and 13 establish that the entire flat-source CP matrix
construction factors as

.. math::

   \text{rcp}_{ij}
     \;=\;
     \underbrace{\int_{0}^{R}\!\mathrm{(outer\ measure)}}_{\text{geometry-specific}}
     \;\cdot\;
     \underbrace{\Delta^{2}\!\bigl[\mathcal F_d\bigr]}_{\text{geometry-invariant operator}}
     \;\bigl(\tau_i,\tau_j;\,\mathrm{gap}(y)\bigr),

with the :math:`(\mathrm{gap}, \tau_i, \tau_j)` arguments supplied
by the chord-walker shared between cylinder and sphere. The rest
of this section describes the class structure that implements this
factorisation in Phase B.2 and motivates the design choices.

Design intent
-------------

Phase 4.2 delivered the pointwise Peierls unification as
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`,
a single class whose concrete instances (``CYLINDER_1D``,
``SPHERE_1D``) dispatch on the geometry-specific primitives
(angular measure, Level-1 kernel, ray-boundary distance, source
position). Phase B.2 will deliver the analogous abstraction at
Level 3:

.. code-block:: python

   # orpheus/derivations/cp_geometry.py  (Phase B.2, not yet shipped)

   @dataclass(frozen=True)
   class FlatSourceCPGeometry:
       """Level-3 flat-source CP abstraction.

       Mirrors CurvilinearGeometry at a different rung of the
       three-tier hierarchy. See docs/theory/peierls_unified.rst
       §14 for the design rationale.
       """

       kind: str   # "slab" | "cylinder-1d" | "sphere-1d"

       # --- kernel methods --------------------------------------
       def kernel_F3(self, tau: float) -> float:
           """Level-3 kernel F_3: E_3 / Ki_3 / exp, by geometry."""
           ...

       def kernel_F3_at_zero(self) -> float:
           """F_3(0) in closed form: 1/2, π/4, 1 respectively."""
           ...

       # --- outer-integration measure ---------------------------
       def outer_y_weight(self, y: np.ndarray) -> np.ndarray:
           """1 (slab / cyl) vs y (sph) weighting for the y-quadrature."""
           ...

       has_through_centre: bool   # False for slab, True otherwise
       surface_area: float        # 1, 2πR, 4πR² per unit cell

Four primitives (``kernel_F3``, ``kernel_F3_at_zero``,
``outer_y_weight``, ``surface_area``) and two flags
(``has_through_centre``, ``kind``) are the full list of
geometry-specific data. Everything else — the
:math:`\Delta^{2}` operator, the chord-walker, the y-quadrature
rule, the composite Gauss–Legendre panels on :math:`[0, R]`, the
self-collision formula, the white-BC geometric series — is
shared and parametrised by these primitives.

One class vs two: recommended path
----------------------------------

A natural question for the Phase B.2 design: should
:class:`FlatSourceCPGeometry` be folded *into*
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
(so one class covers all three kernel levels), or should it remain
a **sibling** class?

**Option (a) — one class, three kernel levels.** Extend
``CurvilinearGeometry`` with ``level_3_kernel``, ``level_3_outer_weight``
etc., and keep ``level_1_kernel`` aliased to the existing
``volume_kernel_mp``. Pros: one source-of-truth object per geometry,
closer correspondence to the three-tier ladder, any future
Level-4-and-beyond extension lands in the same class. Cons: the
class grows to ~12 methods with two loosely-coupled sets
(pointwise Nyström uses :math:`\kappa_1`, ray-walker,
:math:`\rho_{\max}`, etc.; flat-source CP uses :math:`\mathcal F_3`,
chord-walker, :math:`y`-quadrature), and the
pointwise-vs-flat-source distinction becomes an implicit flag on
the methods rather than an explicit class signature.

**Option (b) — sibling classes, shared primitives.** Keep
:class:`CurvilinearGeometry` as the pointwise abstraction;
introduce :class:`FlatSourceCPGeometry` as the flat-source
abstraction. Both call the **same** ``chord_half_lengths``,
``composite_gl_r``, and ray-walker primitives from
:mod:`~orpheus.derivations._kernels` and
:mod:`~orpheus.derivations.peierls_geometry`. Pros: the two classes
compute fundamentally different quantities (pointwise
:math:`\varphi(r)` vs region-average :math:`P_{ij}`), so the class
signature advertises that distinction and the type system enforces
it. Cons: two classes instead of one; the three-tier ladder is
implicit in which class you instantiate rather than in a method
argument.

**Recommendation: option (b) for Phase B.2.** The decision is
driven by *what is being computed* more than by *how the kernel is
integrated*. A ``CurvilinearGeometry`` instance answers "give me
:math:`\varphi(r)` at a collocation node"; a
``FlatSourceCPGeometry`` instance answers "give me
:math:`P_{ij}` for a pair of regions". These are different
physical quantities — collapsing them under one class would force
the type system to encode a union over quantities, which is less
clear than two classes.

Option (b) does **not** duplicate code. The shared infrastructure
lives in :mod:`~orpheus.derivations._kernels` and
:mod:`~orpheus.derivations.peierls_geometry`; both classes import
and call those primitives:

- ``chord_half_lengths(radii, y_pts)`` — already shipped in
  ``_kernels.py``; common to both classes.
- ``composite_gl_r(radii, n_panels, p_order, dps)`` — already
  shipped in ``peierls_geometry.py``; common to both.
- The chord-walker / ``bnd_pos`` accumulation pattern — factored
  into a new ``peierls_geometry.optical_boundary_positions()``
  helper that both classes call.

Phase B.2 therefore delivers ``cp_geometry.py`` containing
:class:`FlatSourceCPGeometry` and its three singleton instances
(``SLAB``, ``CYLINDER_1D``, ``SPHERE_1D``), plus a
``build_cp_matrix(geom, sig_t, radii, ...)`` entry point that
mirrors the existing ``_cylinder_cp_matrix`` / ``_sphere_cp_matrix``
signature. Then ``cp_slab.py``, ``cp_cylinder.py``, ``cp_sphere.py``
become thin facades re-exporting ``build_cp_matrix`` with the
pre-selected geometry.

.. _cp-unified-class-architecture:

Implementation shape (Phase B.2 target)
---------------------------------------

.. code-block:: python
   :caption: ``cp_geometry.py`` — intended skeleton; not yet shipped.

   def _second_difference(kernel, gap, tau_i, tau_j):
       """The geometry-invariant operator Δ²[F](τ_i, τ_j; gap).

       See docs/theory/peierls_unified.rst §12 for the derivation.
       """
       return (kernel(gap)
               - kernel(gap + tau_i)
               - kernel(gap + tau_j)
               + kernel(gap + tau_i + tau_j))

   def build_cp_matrix(
       geom: FlatSourceCPGeometry,
       sig_t_all: np.ndarray,   # (N_reg, ng)
       radii: np.ndarray,       # (N_reg,), outer radii (slab: thicknesses)
       volumes: np.ndarray,
       n_quad_y: int = 64,
   ) -> np.ndarray:
       """Unified flat-source CP matrix constructor.

       Delegates kernel evaluation and outer-measure choice to `geom`;
       shares chord-walker, second-difference, and white-BC closure
       across all three geometries.
       """
       ...

One unit of work — adding the sphere to the unified code path — is
a five-line change (pass ``SPHERE_1D`` instead of ``CYLINDER_1D``
when invoking ``build_cp_matrix``). Contrast this with the present
three-file implementation in which each geometry carries its own
~100-line kernel loop.

.. tip::

   **Why not collapse the pointwise and flat-source modules into a
   single class after all?** Because the four primitives that
   distinguish pointwise Peierls geometries (``rho_max``, ``r_prime``,
   angular measure, Level-1 kernel) have **no flat-source analogue**
   — flat-source CP integrates over a region's chord family, which is
   a *simpler* geometric structure than a general observer-centred
   polar parametrisation. Conversely, the flat-source primitives
   (``outer_y_weight``, Level-3 kernel, ``has_through_centre``) have
   no pointwise analogue — they presume a region-averaged quantity.
   Forcing both sets under one class obscures which primitives are
   active for a given computation. Two classes make the active
   primitive set explicit.


Section 15 — Escape probabilities as Level 2
============================================

The escape probability :math:`P_{\rm esc}(r_i)` — the uncollided
probability that a neutron emitted at :math:`r_i` escapes the current
cell — is the Level-2 quantity of §11. It is **one integration above
the Level-1 point kernel** (integrate :math:`\kappa_1` along the
outward ray to the boundary) and **one integration below the Level-3
flat-source CP kernel** (region-averaging the escape integrand gives
:math:`\mathcal F_3`). It is therefore the natural "bridge" between
the pointwise Peierls and flat-source CP methods, and it appears
explicitly in both codebases.

Level-2 kernel evaluations
--------------------------

.. list-table:: Level-2 kernels and their escape-probability role
   :header-rows: 1
   :widths: 16 28 28 28

   * - Geometry
     - :math:`\mathcal F_2`
     - Pointwise use
     - Flat-source use
   * - Slab
     - :math:`E_2(\tau)`
     - :math:`P_{\rm esc}(x)` via ray integration
     - White-BC closure :math:`P_{\rm in}` factor
       (:eq:`pin-from-reciprocity`)
   * - Cylinder
     - :math:`\mathrm{Ki}_2(\tau)`
     - Same — :func:`~orpheus.derivations.peierls_geometry.compute_P_esc`
       with ``CYLINDER_1D``
     - White-BC closure of
       :func:`~orpheus.derivations.cp_cylinder._cylinder_cp_matrix`
   * - Sphere
     - :math:`e^{-\tau}`
     - Same — :func:`~orpheus.derivations.peierls_geometry.compute_P_esc`
       with ``SPHERE_1D``
     - White-BC closure of
       :func:`~orpheus.derivations.cp_sphere._sphere_cp_matrix`

For the curvilinear cases the pointwise :math:`P_{\rm esc}` is
already implemented via the unified
:func:`orpheus.derivations.peierls_geometry.compute_P_esc` call,
which integrates the geometry-specific ``escape_kernel_mp`` along
each outgoing ray. The slab equivalent lives in
:mod:`~orpheus.derivations.peierls_slab` as the
``build_white_bc_correction`` helper (rank-2 because of two boundary
faces — see :eq:`peierls-white-bc` and §8 above).

The flat-source white-BC closure reuses the *same* :math:`P_{\rm esc}`
value but extracted from the CP matrix itself:

.. math::
   :label: cp-escape-from-p-cell

   P_{{\rm esc},i}^{\rm CP}
     \;=\; 1 - \sum_{j} P_{ij}^{\rm cell}
     \;=\; 1 - \frac{1}{\Sigma_{t,i}\,V_i}\sum_{j}\text{rcp}_{ij}^{\rm cell},

.. vv-status: cp-escape-from-p-cell tested

which is the code line ``P_out = np.maximum(1.0 - P_cell.sum(axis=1),
0.0)`` in all three CP derivation modules. The two routes agree at
the sum level because the row sum identity
:math:`\sum_j P_{ij}^{\rm cell} + P_{{\rm esc},i} = 1` is exactly the
Level-2 statement that the kernel integrates to unit escape-or-collision
probability.

Cross-check at Level 2: flat-source vs pointwise
------------------------------------------------

An L2 regression test available for Phase B.3 is the following
algebraic identity: evaluate :math:`P_{\rm esc}` two ways —
(a) pointwise via :func:`compute_P_esc`, then volume-average over the
region; (b) flat-source via :eq:`cp-escape-from-p-cell`. Both should
agree on the region-averaged level to the CP-matrix quadrature
tolerance (``tolerance = 1e-5`` in
:func:`~orpheus.derivations.cp_cylinder.all_cases`). This is a
**cross-level** verification: it checks that the Level-2 kernel is
correctly related to the Level-1 kernel by one antiderivation, using
the Level-3 machinery as the consumer. It is the natural L2-bridge
test for the three-tier hierarchy.

The sphere is a useful edge case here because its Level-2 kernel is
the unmodified :math:`e^{-\tau}` — the cross-check therefore reduces
to comparing mpmath's ``mpmath.exp(-tau)`` against the composite-GL
pointwise integral of the same. Any discrepancy would immediately
signal a coordinate / Jacobian error in one of the two routes.


Section 16 — ``BickleyTables`` retirement (completed)
=====================================================

.. note::

   **Status: retired.** The legacy ``BickleyTables`` class and
   ``bickley_tables()`` cache function were deleted from
   :mod:`orpheus.derivations._kernels` in commit ``6badbe5``
   (`Issue #94 <https://github.com/deOliveira-R/ORPHEUS/issues/94>`_,
   Phase B.4). Every former consumer now routes through the
   Chebyshev interpolant :func:`orpheus.derivations.cp_geometry._ki3_mp`,
   which is shared by the flat-source CP derivation and the runtime
   solver :mod:`orpheus.cp.solver`. This section is retained as the
   project's **authoritative postmortem** of a 20 000-point
   tabulation that had been a ceiling on cylindrical CP accuracy
   since the solver was first written.

Until Phase B.4 the legacy ``BickleyTables`` was a 20 000-point
tabulation of :math:`\mathrm{Ki}_n` evaluations built at
~:math:`10^{-3}` accuracy by :func:`scipy.integrate.quad` and cached
via a ``bickley_tables()`` ``lru_cache`` wrapper. It was introduced
with the original flat-source CP cylindrical module and survived
every subsequent refactor because the CP formulas were written
against its (non-A&S) naming and because every replacement candidate
would have required a simultaneous audit of the physics.

Two conditions had to be met before the table could be safely
retired:

- The high-precision :func:`~orpheus.derivations._kernels.e_n_mp` and
  :func:`~orpheus.derivations._kernels.ki_n_mp` evaluators had to
  be shipped and tested to 30+ digit precision. Done in Phase 0 of
  the verification campaign.
- The flat-source CP construction had to be re-expressed as a full
  geometry-dispatching module
  (:mod:`orpheus.derivations.cp_geometry`, Phase B.1 theory and
  Phase B.2 code) so that the kernel swap could happen in one place
  rather than three. Done in commit ``f1b869b``.

With both conditions met, the off-by-one naming discrepancy (GitHub
Issue `#94 <https://github.com/deOliveira-R/ORPHEUS/issues/94>`_ —
now **CLOSED**) was resolved **structurally**: the new code never
uses the legacy names ``Ki3_vec`` or ``ki4_vec``, so there was
nothing to rename — only a kernel to replace.

Replacement table (what each legacy call became)
------------------------------------------------

.. list-table:: Legacy ``BickleyTables`` methods and their canonical replacements
   :header-rows: 1
   :widths: 36 36 28

   * - Legacy call (pre-Phase B.4)
     - Canonical replacement (post-Phase B.4)
     - A&S identity
   * - ``tables.ki3(x)`` / ``ki3_vec(x)``
     - ``ki_n_mp(2, x, dps=30)``
     - canonical :math:`\mathrm{Ki}_2(x)`
   * - ``tables.ki4(x)`` / ``ki4_vec(x)``
     - :func:`~orpheus.derivations.cp_geometry._ki3_mp` (fast) or
       ``ki_n_mp(3, x, dps=30)`` (arbitrary precision)
     - canonical :math:`\mathrm{Ki}_3(x)`
   * - ``tables.Ki2_vec(x)`` (canonical alias, added Phase 4.2)
     - ``ki_n_mp(2, x, dps=30)``
     - canonical :math:`\mathrm{Ki}_2(x)`
   * - ``tables.Ki3_vec(x)`` (canonical alias, added Phase 4.2)
     - :func:`~orpheus.derivations.cp_geometry._ki3_mp` (fast) or
       ``ki_n_mp(3, x, dps=30)`` (arbitrary precision)
     - canonical :math:`\mathrm{Ki}_3(x)`
   * - ``e3(x)`` / ``e3_vec(x)`` (slab)
     - :func:`~orpheus.derivations._kernels.e3_vec` (retained;
       already double-precision via :func:`scipy.special.expn`)
     - canonical :math:`E_3(x)`

Retirement sequence (what actually happened)
--------------------------------------------

1. **Phase B.1** (commit ``ea6b05e``, theory-first):
   :doc:`peierls_unified` §§12–17 landed as a theory page before
   any code changed, naming the forthcoming modules and the unified
   :math:`\Delta^{2}` operator.
2. **Phase B.2** (commits ``f1b869b`` +  ``bf128d3``): the new
   :mod:`orpheus.derivations.cp_geometry` module was implemented
   with ``FlatSourceCPGeometry`` and the three singletons
   :data:`SLAB`, :data:`CYLINDER_1D`, :data:`SPHERE_1D`; the
   pre-existing :mod:`~orpheus.derivations.cp_slab`,
   :mod:`~orpheus.derivations.cp_cylinder`, and
   :mod:`~orpheus.derivations.cp_sphere` modules became thin facades
   over the geometry-dispatching core. ``BickleyTables`` was
   **no longer imported** by any ``cp_*`` derivation module, but
   the class itself was kept in :mod:`~orpheus.derivations._kernels`
   so the Phase B.2 commit was a drop-in refactor with bit-identity
   to Phase A (safety milestone).
3. **Phase B.4** (commit ``6badbe5``, this postmortem's subject):
   ``BickleyTables`` and ``bickley_tables()`` were deleted from
   :mod:`~orpheus.derivations._kernels`. The cylinder kernel was
   replaced by :func:`~orpheus.derivations.cp_geometry._ki3_mp`, a
   Chebyshev polynomial of degree 63 fit to the scaled kernel
   :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)` on :math:`[0, 50]` at
   Chebyshev-Gauss-Lobatto nodes (~:math:`5\times 10^{-6}`
   absolute accuracy; build cost ~0.3 s, lazy via
   :func:`functools.lru_cache`). The runtime solver
   :mod:`orpheus.cp.solver` was rewired in the *same commit* to
   import ``_ki3_mp`` from :mod:`~orpheus.derivations.cp_geometry`
   and consume it via ``_setup_cylindrical``; the solver's own
   private ``_build_ki_tables`` + ``_ki4_lookup`` pair (~30 lines
   of cumsum-based :math:`O(h)` quadrature) were deleted. Solver
   ``keff`` and derivation ``k_inf`` now evaluate :math:`\mathrm{Ki}_3`
   through the **same code path** — the solver/derivation
   kernel-split bias that had been hiding behind the
   ``CPParams.n_ki_table`` knob is gone (the knob is retained as
   an unused no-op for construction-site backwards compatibility).

Phase B.4 postmortem — measured impact
--------------------------------------

The kernel swap was an **improvement**, not a regression. The
measurable shifts are:

- **Cylinder** :math:`k_\infty` **reference values** shifted by up to
  ~:math:`4\times 10^{-4}` for multi-region 1-group cases. The
  Bickley tabulation's trapezoidal :math:`O(\Delta x^2)` error had
  been the dominant bias in the reference; each new value is
  closer to the exact mpmath result than the pre-refactor one.
- **Solver/reference agreement**. The ``solve_cp`` cylinder
  ``keff`` now agrees with the shifted :math:`k_\infty` reference
  to machine precision (same kernel on both sides). All nine
  ``cp_cyl1D_*`` L1 eigenvalue tests pass at their declared
  ``tolerance = 1e-5`` with actual error ~:math:`10^{-7}` — about
  100× headroom, where previously the ``1e-5`` tolerance had been
  the *actual* floor set by kernel bias.
- **Tabulation-size sensitivity test retired**. The old
  ``test_cylindrical_ki4_convergence_with_table_size`` (which
  documented that 5 000 → 20 000 → 40 000 points gave diminishing
  returns) was replaced by
  ``test_ki3_kernel_is_insensitive_to_n_ki_table``: ``n_ki_table``
  is a no-op, and ``keff`` is bit-identical across
  ``{5000, 20000, 40000}``.
- **Solver startup latency**. The 20 000-point
  :func:`scipy.integrate.quad` loop at ``CPMesh`` construction is
  gone; the Chebyshev polynomial is built lazily on first call to
  ``_ki3_mp`` (~0.3 s once per process) and cached via
  :func:`~functools.lru_cache`. Repeated solves pay zero kernel
  setup cost.

Why a Chebyshev interpolant and not direct mpmath
--------------------------------------------------

Each mpmath :math:`\mathrm{Ki}_n` call is ~100× slower than a
double-precision table lookup. The flat-source CP matrix requires
:math:`O(N_{\rm reg}^{2}\cdot n_y)` kernel evaluations per group;
for a 4-region 64-quadrature-point cylindrical case at 4 groups,
this is ~4 × 16 × 64 = 4 096 kernel calls per :math:`P_{ij}` matrix
(~16 384 per full 4-group case). At ~100 μs per ``ki_n_mp`` call,
matrix construction at full 30 dps would cost ~1.5 s per group —
a showstopper for any iterative solve.

The chosen compromise is a **single-scale Chebyshev polynomial on
the scaled kernel** :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)`. Scaling
by :math:`e^{\tau}` converts the exponentially-decaying tail into
a slowly-varying function that a degree-63 polynomial fits to
~:math:`10^{-6}` relative accuracy over :math:`[0, 50]`; the
evaluation cost is one :class:`numpy.polynomial.Chebyshev` call
plus one :func:`numpy.exp`, comparable to the legacy
:func:`numpy.interp` on the 20 000-point grid. Beyond
:math:`\tau = 50`, :math:`\mathrm{Ki}_3(\tau) \approx
3\times 10^{-23}` and the interpolant clamps to zero; this is
already below double precision.

The tabulation is an implementation detail of
:mod:`orpheus.derivations.cp_geometry`, invisible to callers. The
key structural difference from the legacy ``BickleyTables`` is that
the new interpolant is built **from the canonical mpmath primitive**
(:func:`~orpheus.derivations._kernels.ki_n_mp` at 30 dps) rather
than from a quad-based 20k-point linear interpolant — the accuracy
ceiling is raised by ~3 orders of magnitude in one step.


Section 17 — Relationship to the existing literature
====================================================

The second-difference formulas documented above are not original to
this codebase. They originate in the classical flat-source CP
literature, which pre-dates computer-based transport by a decade or
more. We credit the standard sources here and flag the
specialisations that each contributed:

- **Slab** :math:`\Delta^{2}[E_3]`. The four-term structure was in
  wide use by the 1960s; it appears explicitly in
  [Carlvik1966]_ §III for the infinite-cylinder case (with a brief
  side-remark on the slab analogue obtained by the :math:`\sin\theta
  \to 1` limit) and is presented in full in [Stamm1983]_ §6.3 with
  the :math:`E_n` derivative identity :math:`E_n' = -E_{n-1}` made
  explicit. The slab second-difference formula has no single
  canonical citation because it emerged as a straightforward
  specialisation of the cylinder derivation.
- **Cylinder** :math:`\Delta^{2}[\mathrm{Ki}_3]`. [Carlvik1966]_ is
  the canonical modern reference: it introduces the chord-form
  :math:`y`-quadrature, derives the four-term
  :math:`\mathrm{Ki}_3`-based formula, and applies it to Dancoff
  factors. [Stamm1983]_ §6.4 presents a cleaner exposition with
  careful attention to the annular geometry's SS / TC branch
  distinction (§13 above) and to the :math:`\mathrm{Ki}_n`
  derivative identity. The derivation in our
  :mod:`~orpheus.derivations.cp_cylinder` follows [Stamm1983]_
  more closely than [Carlvik1966]_.
- **Sphere** :math:`\Delta^{2}[e^{-\tau}]`. [BellGlasstone1970]_
  §2.7 derives the spherical CP matrix with the bare
  :math:`e^{-\tau}` kernel and the :math:`y\,\mathrm dy` outer
  measure. The derivation is brief because the spherical case
  inherits the chord machinery verbatim from the cylinder — only
  the outer weight changes. [BellGlasstone1970]_ also presents
  the limiting cases (small :math:`R`, large :math:`R`) as
  sanity checks on the formula, which
  :mod:`~orpheus.derivations.cp_sphere` replicates at
  :math:`R = 10` MFP via the :math:`k_\infty \to 1.5` agreement
  with the homogeneous analytic solution.

Historical context — the pre-computer origin
--------------------------------------------

The reason these formulas are presented in chord coordinates in
every historical reference is that chord coordinates give **closed-form
flat-source annular integrals**. A pre-1970 user computing a CP
matrix by hand needed to read :math:`\mathrm{Ki}_3(\tau)` or
:math:`E_3(\tau)` off a tabulated function, multiply by a handful
of geometric factors, and sum four terms per :math:`P_{ij}` element.
The alternative — the polar-form pointwise Peierls — required
:math:`O(n_\beta \cdot n_\rho)` kernel evaluations per observer
and was simply not computable by hand.

By the time computers made the polar form feasible, the chord-form
derivations had become standard pedagogy and the flat-source CP
method had its own decade of established engineering use. The
pointwise Peierls Nyström formulation — the basis for the modern
verification pipeline — therefore was *not* developed in parallel
with flat-source CP. It emerged later, from the integral-transport
verification literature (e.g. [Sanchez1982]_, then the integral
benchmark programmes of the 1990s-2000s), and used the polar form
because by then :math:`\mathrm{Ki}_1` was cheaper to evaluate than
:math:`\mathrm{Ki}_3` (one fewer antiderivation, and
machine-precision special-function libraries had matured).

**The unification on this page is the first time the polar-form
Level-1 and the chord-form Level-3 have been put into a single
conceptual framework in ORPHEUS.** This framework makes it obvious
that:

- The three flat-source CP modules can (and should) share one
  :math:`\Delta^{2}` operator and one chord-walker. Phase B does
  this.
- The pointwise Peierls and flat-source CP are two instances of the
  same integral transport equation at two different rungs of the
  integration ladder. Cross-level verification (§15) is therefore
  *meaningful* — it checks one geometry at two kernel levels — not
  a coincidence.
- The sphere's "no special functions" result is not an asymmetry but
  a consequence of :math:`e^{-\tau}` being its own antiderivative.
  The ladder structure explains why this is the case, removing what
  looked like a geometry-specific surprise.

No equation on this page originates here — all four derivations
(:eq:`cp-flat-source-double-integral`,
:eq:`cp-inner-integral-antiderivative`,
:eq:`cp-outer-integral-antiderivative`,
:eq:`cp-second-difference-operator`) reproduce steps already in the
classical references or in :doc:`collision_probability`. What is new
is the *organisation*: naming the levels, naming the operator, and
factoring the per-geometry data into four primitives + two flags.


.. _peierls-part-iii:

======================================================================
Part III — Beyond the basic unification
======================================================================

Sections 1–17 described the *basic* unification: three 1-D geometries
as one polar-form Peierls equation, and the three-tier kernel hierarchy
that links pointwise Nyström to flat-source CP. Part III steps beyond
that baseline to record the three extensions identified in session-N
planning that have **not** yet been implemented, but whose mathematics
are load-bearing enough that they must be preserved here before any
session forgets them.

The three extensions are independent of each other but share a common
theme: every one of them moves numerical difficulty around a carefully
chosen coordinate transform until the remaining integrand is in a form
that a standard quadrature rule handles well. The transforms are *not*
black magic — each one has a one-line physical or geometric
justification — but they are easy to confuse with the original
integrand if the motivation is not recorded.

The reserved section numbers are:

- **Section 20 — Topology of 1-D radial geometries.** 2-boundary vs
  1-boundary+singularity classes; diffeomorphism between Class-A
  members; hollow-core as the Class-A extension. *Deferred to Phase F
  (Issue #110).*
- **Section 21 — The planar limit.** Slab as hollow-cylinder with
  :math:`r_0 \to \infty`; the :math:`\mathrm{Ki}_1 \to E_1/2` kernel
  reduction; numerical verification strategy. *Deferred to Phase F
  (Issue #110).*
- **Section 22 — Coordinate transformations in Nyström quadrature.**
  Below. (Phase H, Issue #109 — this page.)
- **Section 23 — Monte Carlo connections.** Below. (Phase H, Issue
  #109 — this page.)

Section 20 and 21 are reserved by number so that the toc of this page
stays stable across future commits. Do NOT renumber §§22–23 if §§20–21
land in a later phase; the reservation is deliberate.


Section 22 — Coordinate transformations in Nyström quadrature
=============================================================

.. _section-22-coordinate-transforms:

This section catalogues the coordinate transforms available to the
Nyström ray integrator. Some are already used
(§22.2 — the :math:`s^2 = r'^2-y^2` Jacobian-absorbing substitution);
others are merely available as fallbacks (§22.5 — Gauss-Jacobi); one
(§22.3 — the optical-depth coordinate :math:`\tau`) is the dominant
under-exploited opportunity and is the subject of the bulk of this
section.

22.1 Principle: transforms relocate singularities, they don't eliminate them
----------------------------------------------------------------------------

Every coordinate transform in this catalogue is an exercise in
**relocation**: the singular or stiff behaviour of the original
integrand is pushed into one of three places where a standard
quadrature rule handles it gracefully:

1. *Into the quadrature weight*, where Gauss-Jacobi (§22.5) or
   Gauss-Laguerre (§22.3) absorbs the singular factor exactly and
   polynomial accuracy is recovered for the remaining smooth part.
2. *Into the Jacobian*, where the polar-form volume element
   :math:`\rho^{d-1}\,\mathrm d\Omega\,\mathrm d\rho` of §3 cancels
   the kernel's :math:`1/|r-r'|^{d-1}` inverse-distance singularity
   (:eq:`peierls-polar-jacobian-cancellation`).
3. *Into the angular measure*, where the slab polar form pushes the
   :math:`-\ln\tau` E\ :sub:`1` singularity at :math:`\tau = 0` into
   the grazing-ray limit :math:`\mu \to 0` (the source of stiffness
   handled by §22.4 — the exp-stretched :math:`\mu` substitution).

No transform eliminates a singularity that is intrinsic to the
physics. The log singularity of :math:`E_1(\tau)` at :math:`\tau = 0`
is physically real — it reflects the mild divergence of the 1-D
pointwise scalar flux near a delta-source in a slab. The polar form
relocates it into the angular integrand at :math:`\mu = 0`, where a
tanh-sinh or exp-stretched quadrature handles it cleanly, but the
singularity is still there. Likewise the curvilinear chord form's
:math:`1/\sqrt{r'^2 - y^2}` Jacobian singularity at the tangent radius
is a geometric singularity of the chord parametrisation; the polar
form cancels it via Jacobian cancellation (§3), but a chord-form
Nyström integrator has to resolve it either by Gauss-Jacobi weights
or by an explicit :math:`s^2 = r'^2 - y^2` substitution (§22.2).

A corollary is that **the choice of coordinates is a choice of which
quadrature rule becomes the natural one**. Section 3 picks polar
coordinates because polynomial quadrature is natural after Jacobian
cancellation. Section 22.3 picks the :math:`\tau`-coordinate because
Gauss-Laguerre with :math:`e^{-\tau}` weight is exactly the right rule
for the resulting integrand. The transforms in this catalogue are
not optimisations — they are *choices of basis* in which different
quadrature rules become natural.


22.2 Jacobian-absorbing transforms (background — already in use)
-----------------------------------------------------------------

The chord-form Peierls integrand for a curvilinear geometry
(cylinder or sphere) has a Jacobian singularity
:math:`1/\sqrt{r'^2 - y^2}` at the tangent radius :math:`r' = y`.
Textbook treatments ([Sanchez1982]_ §IV, [Carlvik1966]_,
[Stamm1983]_ §6.4) absorb this singularity via the
**half-chord substitution**

.. math::

   s^2 \;=\; r'^2 - y^2, \qquad r' = \sqrt{s^2 + y^2},
   \qquad r'\,\mathrm dr' = s\,\mathrm ds,

which turns the :math:`1/\sqrt{r'^2-y^2} \cdot r'\,\mathrm dr'` chord
measure into the smooth Cartesian measure :math:`\mathrm ds` on
:math:`[0, s_{\max}]`. The same transform underlies the classical
:math:`\mathrm{Ki}_n` arc-length parametrisation.

In the unified polar form of §§3–4 this transform is **not needed**:
the polar volume element :math:`\rho^{d-1}\,\mathrm d\rho` of the
observer-centred parametrisation cancels the
:math:`1/|r-r'|^{d-1}` inverse-distance factor of the Green's function
*before* any chord variable is introduced. The Jacobian cancellation
of :eq:`peierls-polar-jacobian-cancellation` is this page's route to
a smooth curvilinear integrand.

**Why the two routes agree.** The chord form and the polar form are
the same integral written in two coordinate systems. Given
:math:`\rho = |r - r'|` and :math:`y = r\sin\beta`, the chord-form
Jacobian :math:`r'/\sqrt{r'^2-y^2}` times the chord-form area element
:math:`y\,\mathrm dy` equals the polar-form volume element
:math:`\rho\,\mathrm d\beta\,\mathrm d\rho` for the cylinder, and
likewise :math:`y^2/\sqrt{r'^2-y^2}` times :math:`y^2\sin\gamma\,
\mathrm dy\,\mathrm d\gamma` equals :math:`\rho^2\sin\theta\,\mathrm
d\theta\,\mathrm d\rho` for the sphere. Both routes produce the same
numerical answer; the polar form is preferred on this page because it
extends to pointwise sources (Level 1 in §11) without any
flat-source-specific bookkeeping.

This subsection establishes the **baseline**: §§22.3–22.6 are
additional transforms that further reduce the difficulty of the
polar-form integrand along different axes (kernel-uniformity,
grazing-ray stiffness, endpoint singularities, interior
singularities).


22.3 The optical-depth coordinate :math:`\tau`
-----------------------------------------------

This is the heart of §22 — the one transform of the four listed here
that has not been implemented anywhere in ORPHEUS yet, and the one
with the highest expected efficiency gain. The derivation is preserved
in full mathematical detail here, so that later sessions implementing
Phase H.2–H.5 have an unambiguous target.

Motivation
~~~~~~~~~~

In the current ρ-coordinate ray integrator, every node inside the
per-ray radial quadrature (§8 step 2) forces the optical-depth walker
(step 4) to evaluate :math:`\tau(\rho) = \int_0^\rho \Sigma_t(r(s))\,
\mathrm ds`, a piecewise-linear function of :math:`\rho` with slope
changes at every annular crossing. The kernel :math:`e^{-\tau(\rho)}`
then has to track this piecewise structure through the walker.

The question this section answers is: *what is the "natural"
integration variable*? The answer is :math:`\tau` itself. The exponent
in the integrating factor of :eq:`peierls-point-kernel-3d` is
:math:`\tau`, not :math:`\rho`; if we integrate in :math:`\tau` the
kernel becomes the bare :math:`e^{-\tau}`, which has no piecewise
structure and is the canonical weight for Gauss-Laguerre quadrature.

Derivation
~~~~~~~~~~

Start from the ρ-parametrisation of the Peierls ray integrand at
observer :math:`r_i` in direction :math:`\Omega`:

.. math::

   I(r_i, \Omega) \;=\;
     \int_0^{\rho_{\max}(r_i,\Omega)}
       e^{-\tau(\rho)}\,q(r'(\rho,\Omega,r_i))\,\mathrm d\rho,

where :math:`\tau(\rho) = \int_0^\rho \Sigma_t(r(s))\,\mathrm ds` is
the cumulative optical depth along the ray. Since :math:`\Sigma_t
\ge 0`, :math:`\tau(\rho)` is monotonically non-decreasing in
:math:`\rho`; wherever :math:`\Sigma_t > 0` it is strictly
increasing and therefore admits an inverse :math:`\rho = \rho(\tau)`.
Regions where :math:`\Sigma_t = 0` (vacuum cavities, Phase F) are
handled separately — see the "cavity special case" paragraph below.

**Step 1 — differentials.** From the definition
:math:`\tau(\rho) = \int_0^\rho \Sigma_t(r(s))\,\mathrm ds`:

.. math::

   \frac{\mathrm d\tau}{\mathrm d\rho} \;=\; \Sigma_t(r(\rho)),
   \qquad \mathrm d\rho \;=\; \frac{\mathrm d\tau}{\Sigma_t(r'(\tau))},

where we abbreviate :math:`r'(\tau) \equiv r(\rho(\tau))`.

**Step 2 — substitute.** Change variable :math:`\tau = \tau(\rho)` in
the ρ-integral. The upper limit maps to
:math:`\tau_{\max} \equiv \tau(\rho_{\max})`:

.. math::
   :label: peierls-tau-coordinate-transform

   I(r_i, \Omega) \;=\;
     \int_0^{\tau_{\max}(r_i,\Omega)}\!
       e^{-\tau}\,
       \frac{q(r'(\tau))}{\Sigma_t(r'(\tau))}\,
       \mathrm d\tau.

.. vv-status: peierls-tau-coordinate-transform documented

This is the **τ-coordinate Peierls ray integrand**. Compared with the
ρ-form, four properties change simultaneously:

1. **The kernel is geometry- and medium-invariant.** The integrating
   factor is now the bare :math:`e^{-\tau}`, independent of
   :math:`r'(\tau)` and of :math:`\Sigma_t(r'(\tau))`. Multi-region
   structure, vacuum cavities, anisotropic scattering, and the ray's
   specific direction have all been absorbed into the two objects
   :math:`r'(\tau)` and :math:`\Sigma_t(r'(\tau))` — the "ray-walker
   outputs" — which enter only through the remaining integrand
   :math:`q \cdot (1/\Sigma_t)`.

2. **Gauss-Laguerre is the ideal quadrature.** The weight of
   :math:`n`-point Gauss-Laguerre on :math:`[0, \infty)` is
   :math:`e^{-\tau}`; the rule is exact for any polynomial
   :math:`q(r'(\tau))/\Sigma_t(r'(\tau))` of degree :math:`\le 2n-1`
   (and spectrally accurate for general smooth integrands). By
   contrast, Gauss-Legendre on :math:`[0, \rho_{\max}]` treats the
   exponentially-decaying :math:`e^{-\tau(\rho)}` as "just another
   factor" of the integrand and over-samples the tail where the
   kernel is already exponentially small. For optically thick cells
   (:math:`\Sigma_t \cdot R \gtrsim 5`) the expected node-count
   reduction is 2–4× for equal precision (plan §5.5).

3. **Hollow cavities become τ-jumps of zero measure** — see the
   dedicated paragraph below. This is the single most important
   downstream benefit: Phase F hollow-core geometries become
   *trivial* to integrate in the τ-coordinate.

4. **Multi-region uniformity.** The kernel no longer tracks annular
   boundaries; they live entirely in the :math:`\rho(\tau)` map.

Multi-region :math:`\rho(\tau)` is piecewise linear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a multi-region ray with :math:`\Sigma_t` piecewise constant on
:math:`K` segments — with values :math:`\Sigma_{t,k}` on segment
:math:`[\rho_{k-1}, \rho_k]` — the cumulative optical depth is
piecewise linear in :math:`\rho`:

.. math::

   \tau(\rho) \;=\; \sum_{j < k} \Sigma_{t,j}\,(\rho_j - \rho_{j-1})
     + \Sigma_{t,k}\,(\rho - \rho_{k-1})
   \qquad \text{for } \rho \in [\rho_{k-1}, \rho_k].

The inverse map :math:`\rho(\tau)` is therefore also piecewise
linear, with slope :math:`1/\Sigma_{t,k}` on each :math:`\tau`-segment.
The ORPHEUS code pattern is to return a list of breakpoints
:math:`\{(\tau_k, \rho_k)\}_{k=0}^{K}` from the ray walker — one
breakpoint per annular crossing — after which :math:`\rho(\tau)` can
be evaluated at any :math:`\tau` by linear interpolation between the
bracketing breakpoints. The forthcoming
``optical_depth_along_ray_with_map`` helper (Phase H.2; see
`Issue #109 <https://github.com/deOliveira-R/ORPHEUS/issues/109>`_)
returns exactly this list.

Crucially, **the kernel** :math:`e^{-\tau}` **never sees the
breakpoints.** The Gauss-Laguerre nodes :math:`\{\tau_m\}` are placed
on :math:`[0, \tau_{\max}]` purely by the kernel weight; for each
:math:`\tau_m` the ray walker returns :math:`\rho(\tau_m)` — and
thence :math:`r'(\tau_m)` and :math:`\Sigma_t(r'(\tau_m))` — by
interpolation against the breakpoint list. The piecewise structure of
the medium lives in the ray walker, not in the quadrature.

Cavity special case
~~~~~~~~~~~~~~~~~~~

When a ray traverses a vacuum cavity — a segment with
:math:`\Sigma_t = 0` — the τ-coordinate behaviour is remarkable:

.. math::

   \frac{\mathrm d\tau}{\mathrm d\rho} \;=\; \Sigma_t \;=\; 0
   \qquad \Longrightarrow \qquad
   \text{τ is constant across the cavity}.

The cavity maps to a **single point** :math:`\tau = \tau(\rho_a) =
\tau(\rho_b)` in τ-space, regardless of the cavity's geometric extent
:math:`\rho_b - \rho_a`. A Gauss-Laguerre quadrature with nodes on
:math:`[0, \tau_{\max}]` sees no contribution from the cavity
interval: it is a point of measure zero. The cavity is **invisible**
in τ-space.

This is the single largest structural benefit of the τ-coordinate for
Phase F (hollow-core support, Issue #110): in ρ-coordinates, traversing
a cavity requires the ray walker to skip :math:`[\rho_a, \rho_b]` with
special-case logic while carrying the pre-cavity :math:`e^{-\tau}`
factor forward; in τ-coordinates, the cavity *is* skipped without any
special-case logic, because the cavity segment has zero τ-measure.

The equivalent statement in Monte Carlo language (§23 below): a
delta-tracking history that samples a flight of length :math:`\tau_i`
from :math:`\text{Exp}(1)` and walks the ρ-coordinate to find the
corresponding :math:`\rho_i` will walk *straight through* a cavity
without advancing its sampled :math:`\tau`. The deterministic and
stochastic formulations agree that the cavity is a no-op.

Monte Carlo connection — delta / Woodcock tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The τ-coordinate is not a numerical curiosity — it is the **canonical
coordinate of Monte Carlo neutron transport**, introduced in the GEM
code by Woodcock, Murphy, Hemmings and Longworth in 1965
([Woodcock1965]_). In Monte Carlo particle transport, the distance to
the next collision is sampled as :math:`\tau_i \sim \text{Exp}(1)`,
:math:`\tau_i = -\ln(1 - \xi)` for :math:`\xi \in [0,1)` uniform, and
the :math:`\rho`-coordinate is then walked on-the-fly until the
accumulated optical depth reaches :math:`\tau_i`. In heterogeneous
geometries Woodcock tracking samples against a majorant cross-section
:math:`\Sigma_{\max}` and accepts / rejects at each virtual collision;
both variants share the same underlying statement — the natural
coordinate is :math:`\tau`, and :math:`\rho(\tau)` is computed by the
ray walker on demand.

**The deterministic Nyström and Monte Carlo are the same integral
evaluated two ways.** Both produce an estimate of

.. math::

   \int_0^{\tau_{\max}}
     e^{-\tau}\,\bigl[q(r'(\tau))/\Sigma_t(r'(\tau))\bigr]\,\mathrm d\tau,

differing only in the estimator:

- **Monte Carlo** samples :math:`\{\tau_i\}_{i=1}^{N_{\rm hist}}` from
  :math:`\text{Exp}(1)`, evaluates the integrand at each, and averages.
  Error decays as :math:`O(1/\sqrt{N_{\rm hist}})`.
- **Nyström (τ-coordinate)** places :math:`\{\tau_m\}_{m=1}^{n}` at
  Gauss-Laguerre nodes and averages with Gauss weights. Error decays
  super-algebraically in :math:`n` for smooth integrands.

For further reading on delta-tracking in modern MC codes see the
review in [MartinBrown2003]_ (LANL technical memorandum on the
algorithm's numerical properties) and the Serpent implementation
documented in [Leppanen2010]_. The related discussion in
:doc:`monte_carlo` §"Woodcock delta-tracking" describes the
algorithm in its original stochastic context; this page treats the
same mathematics from the deterministic Nyström side.

Quadrature choice
~~~~~~~~~~~~~~~~~

Two quadrature rules are natural for the finite interval
:math:`[0, \tau_{\max}]` with the :math:`e^{-\tau}` weight:

**Option A — Gauss-Laguerre on** :math:`[0, \infty)` **with
truncation.** The :math:`n`-point Gauss-Laguerre rule is tuned for
:math:`\int_0^\infty e^{-\tau} f(\tau)\,\mathrm d\tau`. For our finite
upper limit :math:`\tau_{\max}`, the approximation

.. math::

   \int_0^{\tau_{\max}}\!e^{-\tau} f(\tau)\,\mathrm d\tau
     \;=\; \int_0^\infty e^{-\tau} f(\tau)\,\mathrm d\tau
           \;-\; \int_{\tau_{\max}}^\infty\!e^{-\tau} f(\tau)\,\mathrm d\tau

is exact. The first term is handled by the full Laguerre rule; the
second (the tail) is bounded in magnitude by
:math:`e^{-\tau_{\max}} \cdot \max_{\tau \ge \tau_{\max}} |f(\tau)|`,
which for :math:`\tau_{\max} \gtrsim 20` is already below double
precision. In practice the tail is simply dropped (equivalent to
treating :math:`f(\tau) = 0` on :math:`[\tau_{\max}, \infty)`), and
only Laguerre nodes :math:`\tau_m < \tau_{\max}` are retained.

**Option B — Tanh-sinh (double-exponential) on**
:math:`[0, \tau_{\max}]`. For very small :math:`\tau_{\max}` (thin
cells, :math:`\tau_{\max} \lesssim 1`) Laguerre may place only a
handful of nodes in the integration interval, and tanh-sinh on the
finite interval is more robust. This is the fallback for thin cells.

The benchmark planned in Phase H.4 (Issue #109 commit `H.4`) measures
node-count-vs-precision for both rules across
:math:`\Sigma_t R \in \{1, 2, 5, 10, 20\}` and sets the default via the
threshold :math:`\Sigma_t R > 5` (Laguerre) vs
:math:`\Sigma_t R \le 5` (tanh-sinh or Gauss-Legendre).

Equivalence with the standard ρ-form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`\tau`-coordinate Peierls equation is *exactly equivalent*
to the ρ-coordinate form of §§3–4 — not an approximation. The only
differences are which variable the Nyström nodes are placed on and
which quadrature weight is absorbed into the rule. Cross-verification
(Phase H.4) therefore consists of comparing the two quadrature paths
at matching node counts on matching problems — any discrepancy is a
bug in either the ray walker or one of the two quadrature rules, not
a physics-level disagreement.


22.4 Exp-stretched :math:`\mu` for slab grazing rays
-----------------------------------------------------

The slab's observer-centred polar form (§4) has
:math:`\rho_{\max}(x,\mu) = L/|\mu|` for :math:`\mu > 0` and
:math:`x/|\mu|` for :math:`\mu < 0`. As :math:`\mu \to 0` the grazing
ray length :math:`\rho_{\max}` diverges, and the polar-form integrand
becomes stiff: most of the ray's contribution comes from a
vanishingly small neighbourhood of :math:`\mu = 0` and the kernel
:math:`e^{-\Sigma_t L/|\mu|}` has an essential singularity at the
endpoint.

The exp-stretched substitution absorbs this stiffness by mapping
:math:`\mu \in (0,1]` onto the half-line :math:`v \in [0, \infty)`:

.. math::
   :label: peierls-exp-stretched-mu

   v \;=\; -\ln|\mu|,
   \qquad \mu \;=\; e^{-v},
   \qquad \mathrm d\mu \;=\; -e^{-v}\,\mathrm dv.

.. vv-status: peierls-exp-stretched-mu documented

Full derivation of the slab-integrand equivalence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To connect the exp-stretched polar form to the textbook
:math:`E_1`-Nyström, integrate the source-free slab characteristic
integrand (i.e., take :math:`q \equiv 0` except for a delta at the
boundary):

.. math::

   J(x) \;=\; \frac{1}{2}\int_0^1 e^{-\Sigma_t L/\mu}\,\mathrm d\mu
     \;=\; \frac{1}{2}\,E_1^{\rm eff}(\Sigma_t L),

where the final equality is the slab vacuum-boundary escape
probability, as given for example in [BellGlasstone1970]_ §2.6.
Substitute :math:`v = -\ln\mu`:

.. math::

   \int_0^1 e^{-\Sigma_t L/\mu}\,\mathrm d\mu
     \;=\; \int_0^\infty e^{-\Sigma_t L\,e^{v}}\,e^{-v}\,\mathrm dv.

The integrand factorises as :math:`e^{-v}` (the standard
Gauss-Laguerre weight) times a **super-exponentially decaying**
bounded function :math:`e^{-\Sigma_t L\,e^v}`. One additional
substitution :math:`u = \Sigma_t L\,e^v` gives
:math:`\mathrm du = u\,\mathrm dv` hence :math:`\mathrm dv = \mathrm
du / u`, and :math:`e^{-v} = \Sigma_t L / u`:

.. math::

   \int_0^\infty e^{-\Sigma_t L\,e^v}\,e^{-v}\,\mathrm dv
     \;=\; \int_{\Sigma_t L}^\infty
             e^{-u}\,\frac{\Sigma_t L}{u^2}\,\mathrm du
     \;=\; \Sigma_t L \int_{\Sigma_t L}^\infty\!\frac{e^{-u}}{u^2}\,\mathrm du.

The integral on the right is the standard definition of the
second exponential integral scaled by :math:`1/u^2` — by integration
by parts, :math:`\int_a^\infty e^{-u}/u^2\,\mathrm du = e^{-a}/a -
E_1(a)`, so:

.. math::

   \int_0^1 e^{-\Sigma_t L/\mu}\,\mathrm d\mu
     \;=\; e^{-\Sigma_t L} - \Sigma_t L\,E_1(\Sigma_t L)
     \;\equiv\; E_2(\Sigma_t L)

(the last equality is the standard recursion
:math:`E_2(x) = e^{-x} - x E_1(x)`, [AbramowitzStegun1964]_ §5.1.14).
The exp-stretched polar-form slab is **numerically equivalent** to the
:math:`E_n`-Nyström form: it is the same integral evaluated through a
different quadrature rule.

Why this matters for the unified framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding slab to the unified ``CurvilinearGeometry`` framework
(Phase G, Issue #111) without sacrificing precision-per-node
requires a quadrature rule whose weight function matches the stiff
polar-form integrand. The derivation above shows that such a rule
**exists and is standard**: Gauss-Laguerre on :math:`v \in [0,\infty)`
with the substitution :math:`\mu = e^{-v}` recovers the
:math:`E_1`-Nyström accuracy without introducing an
:math:`E_1`-specific special-function evaluator. The slab entry in the
Phase-G unified framework then uses:

- ``kind = "slab"``;
- ``angular_range = (-1, 1)`` with ``angular_quadrature = "exp-stretched"``;
- ``rho_max(x, mu) = (L - x)/mu`` for :math:`\mu > 0`,
  :math:`x/|\mu|` for :math:`\mu < 0`;
- ``source_position(x, rho, mu) = x + rho * mu``;
- ``volume_kernel = exp(-tau)`` — identical to the sphere.

(The observation that *slab and sphere share the same Level-1 polar
kernel* :math:`e^{-\tau}` — modulo the prefactor :math:`1/2` vs
:math:`1/(4\pi)` — is the surprising consequence of the polar-form
unification; see §4 for discussion.)


22.5 Gauss-Jacobi endpoint weights (fallback)
---------------------------------------------

**General method.** For integrals of the form

.. math::

   \int_a^b (x - a)^\alpha\,(b - x)^\beta\,f(x)\,\mathrm dx,
   \qquad \alpha,\beta > -1,

with :math:`f` smooth, the :math:`n`-point Gauss-Jacobi rule places
nodes and weights such that the rule is exact for polynomial :math:`f`
of degree :math:`\le 2n-1` (the same polynomial-exactness guarantee as
Gauss-Legendre). The singular weight :math:`(x-a)^\alpha (b-x)^\beta`
is handled analytically by the quadrature, so power-law endpoint
singularities of the original integrand are treated without stiffness.

**Application to chord-form** :math:`1/\sqrt{r'^2 - y^2}`. The
curvilinear chord-form integrand has
:math:`1/\sqrt{r'^2 - y^2} = 1/\sqrt{(r'-y)(r'+y)}`. Over
:math:`r' \in [y, R]` the left endpoint carries the singular weight
:math:`(r'-y)^{-1/2}` — the factor :math:`(r'+y)^{-1/2}` is bounded
and smooth. This is exactly a Gauss-Jacobi weight with
:math:`\alpha = -1/2`, :math:`\beta = 0`. The rule would absorb the
singular factor and deliver polynomial accuracy on the remaining
smooth :math:`f(r') = (r'+y)^{-1/2}\,f_0(r')`.

**Why this page does not use Gauss-Jacobi by default.** The unified
polar form of §§3–4 **sidesteps the chord-form singularity entirely**
via the :math:`\rho^{d-1}` Jacobian cancellation of
:eq:`peierls-polar-jacobian-cancellation`. Once that cancellation has
been performed there is no power-law endpoint singularity left to
absorb; Gauss-Legendre on the bounded :math:`\rho`-interval is already
spectrally accurate for the resulting smooth integrand.

Gauss-Jacobi is nevertheless kept as a documented fallback:

- For any future chord-form Nyström (e.g. if a hypothetical 2-D
  ray-tracing CP code — Issue #55 — found it more natural to keep
  the chord parametrisation), Gauss-Jacobi with
  :math:`\alpha = -1/2, \beta = 0` is the correct rule.
- For any endpoint power-law singularity encountered in a future
  variant of Nyström quadrature where the polar form isn't available
  (e.g. 3-D general-geometry ray tracing with oblique chord paths
  through faces), Gauss-Jacobi is the standard tool.

**No Gauss-Jacobi for the slab** :math:`E_1` **log singularity.**
The slab pointwise kernel :math:`E_1(\tau) \sim -\ln\tau - \gamma`
has a logarithmic singularity at :math:`\tau = 0`, not a power-law
one. Gauss-Jacobi weights are power-law :math:`(x-a)^\alpha` — they
cannot absorb a log singularity. The correct method for the slab
:math:`E_1` Nyström is the singularity-subtraction approach already
implemented in :mod:`orpheus.derivations.peierls_slab` (§5 of that
page), which decomposes :math:`E_1(\tau) = -\ln\tau \cdot
g_1(\tau) + g_2(\tau)` with :math:`g_1, g_2` smooth, integrates the
smooth part by Gauss-Legendre, and handles the log part by product
integration against a log-weighted quadrature.


22.6 Davison's :math:`u = r\cdot\varphi` substitution (historical)
-------------------------------------------------------------------

A classical trick for the spherical Peierls equation is the
substitution

.. math::
   :label: peierls-davison-urho

   u(r) \;\equiv\; r\,\varphi(r),

.. vv-status: peierls-davison-urho documented

with the natural boundary condition :math:`u(0) = 0` (since
:math:`\varphi(0)` is finite and :math:`r = 0` is a coordinate
singularity, not a physical one). The substitution transforms the
spherical Peierls equation

.. math::

   \Sigma_t(r)\,\varphi(r) \;=\; \int_0^R K_{\rm sph}(r, r')\,
                                \varphi(r')\,\mathrm dr' + S_{\rm bc}(r)

into an equivalent 1-D integral equation on :math:`u(r)`, with a
modified kernel :math:`\tilde K(r,r') = (r'/r)\,K_{\rm sph}(r,r')`
and a natural :math:`u(0) = 0` boundary condition that **regularises
the coordinate singularity at** :math:`r = 0` **at the level of the
unknown**.

Davison's substitution is attributed to B. Davison in the classical
spherical transport literature; it appears in
[BellGlasstone1970]_ §2.7 and is a standard technique in the bare-sphere
analytic solutions of [CaseZweifel1967]_ Chapter 2 (where it is the
canonical change of variable for the Case-de Hoffmann-Placzek bare-sphere
critical-radius derivations).

Why this page does not implement Davison's substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The observer-centred polar form of §§3–4 **does not encounter the
coordinate singularity at** :math:`r = 0`. Rays emanating from an
interior observer :math:`r_i > 0` pass through the centre without
incident — :math:`r = 0` is geometrically a regular interior point of
the ray, not a special point. The composite-GL radial grid used by
:mod:`orpheus.derivations.peierls_geometry` places no collocation node
at :math:`r = 0` (the first panel is :math:`[0, r_1]` with GL nodes
strictly interior to it), so the pointwise unknown :math:`\varphi(r)`
is never evaluated at :math:`r = 0` and there is nothing to
regularise.

The polar-form cancellation of §3 is therefore a *structural*
replacement for Davison's substitution at the Nyström level: both
methods remove the :math:`r = 0` singular behaviour, but the polar
form does so at the level of the coordinate system (the Jacobian
cancels the coordinate singularity) whereas Davison does so at the
level of the unknown (rescaling :math:`\varphi` by :math:`r` absorbs
the singularity).

Where Davison's substitution remains useful
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Differential-form transport.** If ORPHEUS ever switches to a
  differential formulation of the sphere Peierls equation, the
  coordinate singularity at :math:`r = 0` returns (it is native to
  the operator :math:`\partial_r + 2/r`, not to the integral form),
  and Davison's substitution is then the standard regulariser.
- **Analytic bare-sphere critical radii.** The classical Case-de
  Hoffmann-Placzek-style closed-form critical-radius expressions all
  use :math:`u = r\varphi` as the natural variable. Any cross-check
  against these analytic results (e.g. [CaseZweifel1967]_ Chapter 2)
  would flow more cleanly through Davison's form than through the
  polar-form Nyström.
- **Milne problem asymptotics.** The asymptotic expansion of the
  angular flux near a curved vacuum boundary is conventionally
  expressed in the :math:`u(r) = r\varphi(r)` variable, again because
  the curved-boundary analogue of the Milne problem is naturally set
  up that way.

**Recommendation.** Document, do not implement. If a specific need
emerges (a Case-Zweifel critical-radius cross-check, or a differential
spherical module) the substitution can be added as a specialist
transform at that time. Until then, the polar form in §§3–4 is the
canonical ORPHEUS treatment.


Section 23 — Monte Carlo connections
====================================

.. _section-23-mc-connections:

The τ-coordinate unification of §22.3 is the deterministic analogue of
a family of Monte Carlo algorithms that have been standard since the
GEM code introduced delta-tracking in 1965 ([Woodcock1965]_). This
section makes the correspondence explicit, and identifies a new V&V
band — cross-stochastic-deterministic verification — that becomes
possible once the τ-Nyström is implemented (Phase H.2+).

23.1 Delta-tracking :math:`\equiv` τ-coordinate sampling
---------------------------------------------------------

Woodcock delta-tracking samples a particle's flight distance as

.. math::

   \tau_i \;\sim\; \text{Exp}(1),
   \qquad \tau_i \;=\; -\ln(1 - \xi_i),
   \quad \xi_i \sim \text{Uniform}[0, 1),

and walks the ρ-coordinate until :math:`\tau` (accumulated along the
ray) reaches :math:`\tau_i`. In heterogeneous media the walk uses a
majorant :math:`\Sigma_{\max} \ge \max_r \Sigma_t(r)` and accepts /
rejects virtual collisions at each trial point; this is a variance
reduction on top of the underlying :math:`\tau \sim \text{Exp}(1)`
sampling. The full algorithm is described in :doc:`monte_carlo`
§"Woodcock delta-tracking".

The deterministic τ-coordinate Peierls Nyström
:eq:`peierls-tau-coordinate-transform` evaluates **the same integral
that delta-tracking estimates**:

.. math::
   :label: peierls-delta-tracking-equivalence

   I(r_i,\Omega) \;=\;
     \int_0^{\tau_{\max}} e^{-\tau}\,g_\Omega(\tau)\,\mathrm d\tau
   \quad\text{where}\quad
   g_\Omega(\tau) \;\equiv\; \frac{q(r'(\tau))}{\Sigma_t(r'(\tau))}.

.. vv-status: peierls-delta-tracking-equivalence documented

The two evaluators differ only in how the integral is estimated:

- **Delta-tracking MC**: estimate :math:`I` as
  :math:`\hat I_{\rm MC} = \frac{1}{N}\sum_{i=1}^{N} g_\Omega(\tau_i)`
  with :math:`\tau_i \sim \text{Exp}(1)`. Error decays as
  :math:`O(1/\sqrt{N})` (Monte Carlo).
- **τ-coordinate Nyström**: estimate :math:`I` as
  :math:`\hat I_{\rm Nys} = \sum_{m=1}^{n} w_m^{\rm Lag}\,g_\Omega(\tau_m)`
  with :math:`(\tau_m, w_m^{\rm Lag})` the Gauss-Laguerre nodes and
  weights. Error decays super-algebraically in :math:`n` for smooth
  :math:`g_\Omega`.

Both estimators share the ray-walker primitive that maps
:math:`\tau \mapsto r'(\tau)` and :math:`\tau \mapsto
\Sigma_t(r'(\tau))`. The shared primitive is the Nexus-graph "bridge"
connecting the two methods: any bug in the ray walker produces a
correlated error in both, whereas bugs unique to one estimator produce
disagreement.

23.2 Next-event estimator :math:`\equiv` fixed-ray Nyström
----------------------------------------------------------

The **next-event estimator** in Monte Carlo (also called the
"surface-crossing" or "track-length" estimator, depending on context)
accumulates contributions at every virtual collision along a
sampled ray direction, weighted by the collision probability at that
point. For a fixed incoming direction :math:`\Omega`, the
next-event estimator of the scalar flux at :math:`r_i` is

.. math::

   \hat\varphi_{\rm NEE}(r_i; \Omega) \;=\;
     \frac{1}{N}\sum_{i=1}^{N}
       e^{-\tau_i}\,g_\Omega(\tau_i),

which is MC quadrature on the same integrand as
:eq:`peierls-delta-tracking-equivalence` but with :math:`\tau_i` now
sampled uniformly along the geometric ray rather than from
:math:`\text{Exp}(1)` — i.e., an importance-sampling variant. This
corresponds exactly to a single direction :math:`\Omega` in the
Nyström angular sum, with the :math:`\tau`-nodes chosen by a different
rule (uniform or track-length) in place of Gauss-Laguerre.

The correspondence is: **fixing** :math:`\Omega` **in the Nyström sum
and evaluating at one angle with** :math:`n` **τ-nodes is the
deterministic analogue of a single-direction next-event estimator
history.**

23.3 Cross-code V&V opportunity
-------------------------------

The existence of the τ-Nyström (Phase H.2+) and the Woodcock tracker
(already shipped in :mod:`orpheus.mc.solver` per the
:doc:`monte_carlo` documentation) creates a new **deterministic ↔
stochastic verification band** that has no precedent in ORPHEUS's
current L0–L3 ladder.

**Proposed verification experiment** (future commit; see the session
plan `.claude/plans/post-cp-topology-and-coordinate-transforms.md`
Appendix B.6):

1. Choose a canonical test problem with known :math:`k_{\rm eff}` —
   e.g. the bare homogeneous cylinder at :math:`R = 2` MFP
   (:math:`k_\infty = 1.5`, rank-1 white BC k_eff tabulated in §3 and
   §10 of this page).
2. Run τ-coordinate Nyström with :math:`n = 16, 32, 64` Laguerre
   nodes; record :math:`k_{\rm eff}` for each.
3. Run Woodcock delta-tracking MC with
   :math:`N_{\rm hist} = 10^4, 10^5, 10^6` histories; record
   :math:`k_{\rm eff}` and its :math:`1\sigma` error bar for each.
4. Verify: :math:`|k_{\rm eff}^{\rm MC} - k_{\rm eff}^{\rm Nys}|
   \le 3\sigma_{\rm MC}` at all :math:`N_{\rm hist}`, for a
   sufficiently converged :math:`n` (say :math:`n = 64`).

**Interpretation.** If both methods agree within Monte Carlo error,
the shared ray-walker primitive is correct — a simultaneous L1
verification for both codes. If they disagree systematically (i.e.,
:math:`|k^{\rm MC} - k^{\rm Nys}|` does not shrink as
:math:`\sigma_{\rm MC}` shrinks), the bug is in the **shared
primitive** (ray walker geometry, optical-depth accumulation,
cross-section lookup), localised to the one module both paths go
through.

This is **new V&V coverage**. The existing L0–L3 ladder validates
within-code correctness against analytics or cross-code reference
solutions; the τ-Nyström ↔ Woodcock agreement validates the shared
geometry-walking primitive across two fundamentally different
algorithms (stochastic vs deterministic) that *must* produce the same
answer. A disagreement localises faults to code shared by both paths.

It also enables a novel use of the MC code: as a reference solution
for a Nyström quadrature convergence study. By running MC to
:math:`\sigma < 10^{-5}` on a suitably small problem, the MC result
becomes a reference value against which the Nyström-in-τ convergence
curve (precision vs :math:`n`) can be measured directly — faster than
generating mpmath-at-30-dps references for the same problem, and
automatically medium-invariant by construction.

**Relationship to Issue #55** (2D ray-tracing CP). The τ-coordinate
is the natural primitive for any future 2-D or 3-D ray-tracing code
because the ray walker's output is already the right integrand for
Gauss-Laguerre. Issue #55 (planned 2-D CP via ray tracing) and any
future general-geometry Monte Carlo module in ORPHEUS would share the
same τ-walker by construction — the MC ↔ Nyström cross-verification
of this subsection extends to any such future module.


Mathematical appendix (Part III)
================================

.. _part-iii-appendix:

This appendix collects numerical techniques that are load-bearing
for Part III's transforms but are general enough to be of use
elsewhere. The lessons were learned by session-N+1 during Phase B.4
(BickleyTables retirement) and the Phase H.1 session (this commit).

App D — Scaled-kernel Chebyshev interpolation
---------------------------------------------

.. _app-d-scaled-chebyshev:

**Problem.** Interpolate :math:`\mathrm{Ki}_3(\tau)` on
:math:`\tau \in [0, 50]` at double precision (error
:math:`\lesssim 10^{-15}`) using a single polynomial. Plain Chebyshev
interpolation of :math:`\mathrm{Ki}_3` fails: the kernel spans 22
orders of magnitude on the interval (:math:`\mathrm{Ki}_3(0) =
\pi/4 \approx 0.785` vs :math:`\mathrm{Ki}_3(50) \approx
2 \times 10^{-23}`), and no polynomial of reasonable degree can
resolve both the plateau near :math:`\tau = 0` and the
exponentially-decaying tail simultaneously — at degree 128, best
relative accuracy caps at :math:`\sim 3\times 10^{-8}`.

**Technique.** Interpolate the **scaled kernel**

.. math::
   :label: peierls-scaled-chebyshev

   f(\tau) \;\equiv\; e^{\tau}\,\mathrm{Ki}_3(\tau),

which varies from :math:`f(0) = \pi/4 \approx 0.785` to
:math:`f(60) \approx 0.16` — a **slowly-varying function of order
unity**, with no exponential dynamic range. A degree-63 Chebyshev
interpolant of :math:`f` on :math:`[0, 50]` achieves :math:`\sim
2\times 10^{-6}` relative accuracy on :math:`f`; evaluation of
:math:`\mathrm{Ki}_3(\tau)` is then

.. math::

   \mathrm{Ki}_3(\tau) \;=\; f(\tau)\,e^{-\tau}
     \;=\; \bigl[\text{Cheb63 interpolant of } f\bigr](\tau)
             \cdot e^{-\tau},

i.e. one polynomial evaluation plus one :func:`numpy.exp` call. The
:math:`e^{-\tau}` factor is exact at double precision; the relative
error of the interpolant on :math:`f` transfers directly to the
relative error on :math:`\mathrm{Ki}_3`, bounded by :math:`2 \times
10^{-6}` uniformly over :math:`[0, 50]`.

**Why this works.** The scaling converts the
dynamic range of :math:`\mathrm{Ki}_3` from 22 orders of magnitude to
a single order of magnitude, which is exactly the regime where
polynomial interpolation is efficient. The :math:`e^{-\tau}` factor
that is "removed" is restored exactly at evaluation time — the
double-precision exponential is accurate to one ulp over the entire
range, so the exponential acts as a noiseless post-processor.

.. vv-status: peierls-scaled-chebyshev documented

**Generalisation.** The same idea applies to any kernel of the form
:math:`e^{-\tau} \cdot g(\tau)` with :math:`g` slowly varying —
precisely the structure that all three Level-1 Peierls kernels share
(§2):

- :math:`\tfrac{1}{2}E_1(\tau) = \tfrac{1}{2}\int_1^\infty
  e^{-\tau t}/t\,\mathrm dt` — for
  :math:`\tau \gtrsim 1`, :math:`E_1(\tau) \sim e^{-\tau}/\tau`.
- :math:`\mathrm{Ki}_1(\tau) = \int_0^{\pi/2} e^{-\tau/\sin\theta}\,
  \mathrm d\theta` — similarly exponentially decaying.
- :math:`e^{-\tau}/(4\pi)` — trivially of this form.

For any of these, scaling by :math:`e^{\tau}` converts the kernel into
a slowly-varying target that Chebyshev interpolation handles with
modest degree. The trick is **not** obvious — session-N+1 burned
20 minutes trying plain Chebyshev at degree 128 before pivoting — so
this appendix records it explicitly.

**Connection to the Nyström quadrature choice.** The analytic
analogue of the scaled-kernel Chebyshev technique is the
Gauss-Laguerre quadrature of §22.3: the τ-coordinate Nyström
absorbs the :math:`e^{-\tau}` factor into the quadrature weight
natively, which is the integration-theoretic parallel of moving
the exponential factor outside of the interpolant in
:eq:`peierls-scaled-chebyshev`. The same idea applied to the exp-stretched
slab quadrature (§22.4) recovers the :math:`E_1`-Nyström accuracy
without an :math:`E_1`-specific table: the
:math:`e^{-\Sigma_t L e^v}` factor in the integrand is exactly the
slowly-varying part after the :math:`e^{-v}` Laguerre weight has been
factored out.

**Reference implementation.** See
:func:`orpheus.derivations.cp_geometry._ki3_scaled_cheb` for the
implementation used in the shipped :mod:`orpheus.derivations.cp_geometry`
module. The key lines are:

.. code-block:: python

   def _ki3_scaled_cheb():
       """Chebyshev interpolant of f(tau) = exp(tau) * Ki_3(tau)."""
       def func_scaled(tau):
           return np.array(
               [float(ki_n_mp(3, t, 30)) * float(np.exp(t)) for t in tau]
           )
       return np.polynomial.Chebyshev.interpolate(
           func_scaled, deg=_KI3_DEG, domain=[0.0, _KI3_TAU_MAX]
       )

   def _ki3_mp(tau):
       """Double-precision Ki_3(tau) via scaled Chebyshev."""
       poly = _ki3_scaled_cheb()
       return poly(tau) * np.exp(-tau)

The build cost is one-time and cached via :func:`functools.lru_cache`;
subsequent evaluations cost one Chebyshev polynomial evaluation plus
one exponential.


App E — ``mpmath.quad`` composition pitfall
-------------------------------------------

.. _app-e-mpmath-composition:

**Problem.** Verification tests for Part III's identities
(e.g. :eq:`peierls-tau-coordinate-transform`,
:eq:`peierls-exp-stretched-mu`) will need to compare the left-hand
side (an integral) against the right-hand side (another integral, or
a closed-form expression involving antiderivatives). The temptation
is to write both sides as :func:`mpmath.quad` calls and compare.

**The pitfall.** Some of the integrals in these derivations are
themselves defined via inner integrals — e.g.
:math:`\mathrm{Ki}_1(\tau) = \int_0^{\pi/2}\!e^{-\tau/\sin\theta}\,
\mathrm d\theta`. Composing the outer :func:`mpmath.quad` with an
inner :func:`mpmath.quad` produces an :math:`O(N^2)` blow-up in
evaluation count that can render even simple identity checks
intractable.

The concrete pattern (observed 2026-04-18 during Phase B.3 test
writing):

.. code-block:: python

   # DO NOT DO THIS
   integral = mpmath.quad(
       lambda t: mpmath.quad(
           lambda u: mpmath.exp(-t * mpmath.cosh(u)) / mpmath.cosh(u),
           [0, mpmath.inf],
       ),  # inner Ki_1 integral
       [a, b],
   )

The outer :func:`mpmath.quad` adaptive algorithm samples ~50 points
on :math:`[a, b]`, and **each outer sample triggers ~50 inner
samples**. Total evaluation count :math:`\sim O(2{,}500)` per call,
each already slow at 30 dps. The original test hung for **45+
minutes** before being killed.

**The fix.** Use the existing single-quad evaluator
:func:`~orpheus.derivations._kernels.ki_n_mp` as the **integrand** of
a single outer :func:`mpmath.quad`. The inner integral is then
evaluated in closed form (or by ``ki_n_mp``'s own internal quad, but
only **once per outer sample**):

.. code-block:: python

   # DO THIS
   left = float(mpmath.quad(
       lambda x: ki_n_mp(1, float(x), dps=20),
       [a, b],
   ))
   right = float(ki_n_mp(2, a, dps=30)) - float(ki_n_mp(2, b, dps=30))
   assert abs(left - right) < 1e-10

Runtime: :math:`\sim 0.5` s per test, same verification content.

**General rules for numerical identity tests.**

1. **Never compose** :func:`mpmath.quad` **with** :func:`mpmath.quad`
   **inside a test.** Even one level of nesting produces
   :math:`O(N^2)` blow-up and often exceeds the test suite timeout.
2. **Verify numerical identities via endpoint evaluation of
   antiderivatives** on one side, compared to **one-level
   integration** on the other side. The identity
   :math:`\int_a^b \mathrm{Ki}_1(\tau)\,\mathrm d\tau = \mathrm{Ki}_2(a)
   - \mathrm{Ki}_2(b)` is verified with one :func:`mpmath.quad` for
   the LHS and two :func:`~orpheus.derivations._kernels.ki_n_mp`
   evaluations for the RHS — not by evaluating :math:`\mathrm{Ki}_2`
   itself via another :func:`mpmath.quad`.
3. **Prefer closed-form equivalents where available**: for
   :math:`E_n`, use :func:`mpmath.expint` (reduced-noise closed form);
   for :math:`\mathrm{Ki}_n`, use
   :func:`~orpheus.derivations._kernels.ki_n_mp` (wraps a single
   adaptive quad at the point of call, not composed).
4. **For τ-coordinate verification** (Phase H.4): compare the
   τ-Nyström and ρ-Nyström evaluations of the **same ray integral**
   at matched Gauss-Laguerre and Gauss-Legendre node counts. Both are
   single-level quadratures; the comparison is direct.

This pitfall is recorded here because every future session verifying
a τ-coordinate identity will encounter the same temptation. The rule
"never compose mpmath.quad inside a test" prevents a 45-minute hang
that otherwise requires an external timeout and an after-the-fact
re-derivation.


.. seealso::

   **Phase B target modules** (all shipped; see commits
   ``f1b869b`` → ``bf128d3`` → ``6badbe5``):

   :mod:`orpheus.derivations.cp_geometry` — the unified
   :class:`~orpheus.derivations.cp_geometry.FlatSourceCPGeometry`
   class and :func:`~orpheus.derivations.cp_geometry.build_cp_matrix`
   entry point. Hosts the double-precision kernel
   :func:`~orpheus.derivations.cp_geometry._ki3_mp` (Chebyshev
   interpolant of :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)` built from
   :func:`~orpheus.derivations._kernels.ki_n_mp` at 30 dps), now
   shared by derivation and runtime solver.

   :mod:`orpheus.derivations.cp_slab`,
   :mod:`orpheus.derivations.cp_cylinder`,
   :mod:`orpheus.derivations.cp_sphere` — thin facades over
   ``cp_geometry`` with the respective ``SLAB`` / ``CYLINDER_1D`` /
   ``SPHERE_1D`` singletons preselected.

   **Shared building blocks**:

   :func:`orpheus.derivations._kernels.chord_half_lengths` — the
   chord-walker consumed by both curvilinear CP modules and by the
   cylinder Peierls reference.

   :func:`orpheus.derivations._kernels.e_n_mp`,
   :func:`orpheus.derivations._kernels.ki_n_mp` — canonical
   arbitrary-precision mpmath evaluators for :math:`E_n` and
   :math:`\mathrm{Ki}_n`. The former :class:`BickleyTables`
   tabulation is retired (Issue #94) — double-precision
   :math:`\mathrm{Ki}_3` goes through
   :func:`~orpheus.derivations.cp_geometry._ki3_mp`.

   :func:`orpheus.derivations._kernels.e_n_derivative`,
   :func:`orpheus.derivations._kernels.ki_n_derivative` — the
   differential identities :eq:`cp-kernel-differential-identities`.

   :func:`orpheus.derivations.peierls_geometry.compute_P_esc` —
   pointwise Level-2 escape probability used by §15's cross-level
   test.

   **Theory cross-references:**

   :doc:`collision_probability` §:ref:`second-diff-derivation` —
   the full IBP chain underlying :eq:`cp-flat-source-derivation`,
   already programmatically verified by an embedded SymPy script.

   :doc:`collision_probability` §:ref:`ki-table-construction` —
   historical documentation of the retired ``BickleyTables``
   tabulation.

   GitHub Issue `#107
   <https://github.com/deOliveira-R/ORPHEUS/issues/107>`_ — N6:
   Phase B tracking issue (CP flat-source unification).

   GitHub Issue `#94
   <https://github.com/deOliveira-R/ORPHEUS/issues/94>`_ —
   ``BickleyTables`` naming discrepancy, **CLOSED** by commit
   ``6badbe5`` (Phase B.4).


Part IV — Tensor structure of the boundary closure
==================================================

The rank-:math:`N` Marshak closure introduced in Part III is not a
collection of ad-hoc integrals — it is the **finite-tensor factorisation
of a Hilbert-Schmidt integral operator** through a small intermediate
mode space. Once the factorisation is made explicit, the physics of
each boundary condition, the algorithmic cost of applying the kernel,
and the programme of work remaining in Issue #112 all become trivially
readable from a single tensor network diagram. This part develops the
operator-level picture, connects it to classical reduced-order-modelling
theory, and motivates the
:class:`~orpheus.derivations.peierls_geometry.BoundaryClosureOperator`
dataclass that now carries the structure in code.


Section 24 — From continuous integral equation to finite tensor
===============================================================

The pointwise Peierls equation, written as in :eq:`peierls-unified`,
reads

.. math::
   :label: peierls-operator-form

   \Sigma_t\,\varphi \;=\; T_{\rm vol}\,q \;+\; S_{\rm bc},

where :math:`T_{\rm vol}` is the volumetric integral operator

.. math::

   (T_{\rm vol}\,q)(r) \;=\; \int_V K_{\rm vol}(r, r')\,q(r')\,\mathrm d V',

with the polar-form volume kernel

.. math::

   K_{\rm vol}(r, r')
     \;=\; \frac{\Sigma_t(r)}{S_d}\,
           \int_{\Omega_d}\!\mathrm d\Omega\!
           \int_0^{\rho_{\max}(r,\Omega)}\!\!
             \kappa_d(\Sigma_t\rho)\,\delta\!\bigl(r'-r'(\rho,\Omega,r)\bigr)\,
           \mathrm d\rho,

and :math:`S_{\rm bc}(r)` the re-entering boundary source. Under a
white / albedo / lattice BC, :math:`S_{\rm bc}` is proportional to the
outgoing flux, which is itself the result of uncollided transport of
:math:`q` to the boundary. Closing this loop expresses :math:`S_{\rm bc}`
as a linear functional of :math:`q`:

.. math::
   :label: peierls-bc-operator

   S_{\rm bc}(r) \;=\; (T_{\rm bc}\,q)(r)
     \;=\; \int_V K_{\rm bc}(r, r')\,q(r')\,\mathrm d V'.

The Peierls equation becomes the **second-kind Fredholm equation** on
:math:`V = L^{2}([0,R],\,r^{d-1}\,\mathrm d r)`

.. math::

   (\Sigma_t - T)\,\varphi \;=\; 0,
   \qquad T \;\equiv\; T_{\rm vol} + T_{\rm bc},

with an integral kernel :math:`K(r, r') = K_{\rm vol}(r, r') + K_{\rm
bc}(r, r')` on :math:`V \times V`.

**Hilbert-Schmidt regularity.** The total kernel
:math:`K \in L^{2}(V \times V)` (it decays exponentially at optical
distance :math:`|r - r'|` and is bounded at short range because of the
polar-form Jacobian cancellation, Section 3), so :math:`T` is a
**compact Hilbert-Schmidt operator** on :math:`V`. In particular:

- :math:`T` has a discrete spectrum accumulating only at :math:`0`.
- The singular-value expansion

  .. math::
     :label: peierls-svd

     K(r, r') \;=\; \sum_{k=1}^{\infty} \sigma_k\,u_k(r)\,v_k(r'),
     \qquad \sigma_k \to 0,

  converges in :math:`L^{2}`.
- Finite-rank approximations are well-defined and have a best
  :math:`L^{2}`-error given by
  :math:`\bigl(\sum_{k > N} \sigma_k^{2}\bigr)^{1/2}`.

**Nyström discretisation.** Replace :math:`V` by the finite-dimensional
radial Nyström space :math:`V_h \cong \mathbb R^{N_r}` spanned by
piecewise-Lagrange basis functions on composite-GL panels
(Section 6). The integral operator :math:`T` restricts to a matrix
:math:`\mathbf K \in \mathbb R^{N_r \times N_r}`. As an element of
:math:`V_h \otimes V_h^{*}`, the discrete kernel is — in the strict
mathematical sense — a :math:`(1, 1)` tensor.

Our object of study is therefore

.. math::

   \mathbf K_{\rm bc} \;\in\; V_h \otimes V_h^{*}
   \;\cong\; \mathrm{Hom}(V_h, V_h)
   \;\cong\; \mathbb R^{N_r \times N_r}.

Part III gave us a formula for computing :math:`\mathbf K_{\rm bc}`
mode-by-mode as the sum :math:`\sum_n u_n \otimes v_n`. We now show
that this sum is not accidental: it is the **canonical factorisation
of a surface-coupled operator through a mode space**.


Section 25 — The factored form :math:`K_{\rm bc} = G\,R\,P`
===========================================================

At the continuous level, the boundary operator :math:`T_{\rm bc}`
**factors through the inward-hemisphere surface angular flux**
:math:`\psi^{-}`. Let :math:`A_\infty := L^{2}([0,1])` be the space of
square-integrable functions on the inward-hemisphere cosine
:math:`\mu_s \in [0, 1]`. Define three continuous operators:

.. math::

   P_\infty\;:\; V \to A_\infty, \quad
     (P_\infty q)(\mu_s)
     \;=\; \int_V \mathcal P(r', \mu_s)\,q(r')\,\mathrm d V',

.. math::

   R_\infty\;:\; A_\infty \to A_\infty, \quad
     (R_\infty \psi^{+})(\mu_s)
     \;=\; \int_0^1 \mathcal R(\mu_s, \mu'_s)\,\psi^{+}(\mu'_s)\,
          \mathrm d\mu'_s,

.. math::

   G_\infty\;:\; A_\infty \to V, \quad
     (G_\infty \psi^{-})(r)
     \;=\; \int_0^1 \mathcal G(r, \mu_s)\,\psi^{-}(\mu_s)\,\mathrm d\mu_s,

with kernels:

- :math:`\mathcal P(r', \mu_s)` = rate at which a unit source at
  :math:`r'` contributes to the outgoing angular flux at the surface
  in direction :math:`\mu_s` (the **escape kernel**, cosine-weighted);
- :math:`\mathcal R(\mu_s, \mu'_s)` = fraction of outgoing flux in
  direction :math:`\mu'_s` that re-enters in direction :math:`\mu_s`
  (the **reflection kernel**, entirely determined by the BC physics);
- :math:`\mathcal G(r, \mu_s)` = contribution to interior flux at
  :math:`r` from unit inward angular flux at the surface in
  direction :math:`\mu_s` (the **response kernel**).

Under rotational symmetry (radial 1-D cells) the surface angular flux
depends only on :math:`\mu_s`; :math:`\psi^{-}` and :math:`\psi^{+}`
are functions on the one-dimensional hemisphere :math:`[0, 1]`. With
these three operators, the boundary operator is

.. math::
   :label: peierls-operator-factorisation

   T_{\rm bc} \;=\; G_\infty \;\circ\; R_\infty \;\circ\; P_\infty.

Let :math:`\{\phi_n\}_{n=0}^{\infty}` be an orthonormal basis of
:math:`A_\infty` (e.g. shifted Legendre :math:`\tilde P_n` — see
Section 26). Truncating to the first :math:`N` basis vectors gives a
finite-dimensional mode space :math:`A_N := \mathrm{span}\{\phi_0,
\ldots, \phi_{N-1}\} \cong \mathbb R^{N}`. Projecting onto :math:`A_N`
yields matrix representations

.. math::

   P \;\in\; \mathbb R^{N \times N_r},
   \quad
   R \;\in\; \mathbb R^{N \times N},
   \quad
   G \;\in\; \mathbb R^{N_r \times N},

whose entries are the projections of the continuous kernels onto the
Nyström radial basis on the :math:`V`-side and onto
:math:`\{\phi_0,\ldots,\phi_{N-1}\}` on the :math:`A`-side. The
discrete boundary kernel is

.. math::
   :label: peierls-factored-kernel

   \boxed{\;
     \mathbf K_{\rm bc} \;=\; G \cdot R \cdot P
   \;}
   \qquad
   \Longleftrightarrow
   \qquad
   (\mathbf K_{\rm bc})^{i}{}_{j}
     \;=\; G^{i}{}_{n}\,R^{n}{}_{m}\,P^{m}{}_{j},

with Einstein summation on the shared mode indices :math:`n, m`.

Tensor network. In the graphical language of tensor networks
(Penrose / Bridgeman-Chubb diagrams), :eq:`peierls-factored-kernel`
is the composition::

     V_h* ──[ P ]── A ──[ R ]── A ──[ G ]── V_h
      j          m        n        i

with one free index on each side (:math:`i \in V_h`, :math:`j \in V_h^{*}`)
and the mode indices :math:`m, n \in A` contracted. :math:`K_{\rm bc}`
is the (1,1) tensor obtained by summing over the mode space.

**Algorithmic payoff.** The factored form has

- **Storage** :math:`\mathcal O(N_r N + N^{2})` floats
  (the three tensors :math:`P`, :math:`G`, :math:`R`) versus
  :math:`\mathcal O(N_r^{2})` for the dense :math:`\mathbf K_{\rm bc}`.
  For typical :math:`N_r \sim 50`, :math:`N \sim 4`: 420 floats vs
  2500 — a 6× compression.

- **Matrix-free application**
  :math:`\mathbf K_{\rm bc}\,q = G\bigl(R\,(P\,q)\bigr)` in
  :math:`\mathcal O(N_r N + N^{2})` flops, versus
  :math:`\mathcal O(N_r^{2})` for the dense multiply.

- **Rank** :math:`\mathrm{rank}(\mathbf K_{\rm bc}) =
  \mathrm{rank}(R)` (generically, when :math:`P` and :math:`G` have
  full mode rank). The rank of the boundary kernel is literally the
  rank of the BC's reflection matrix.

The final point is the structural payoff of the whole exercise.


Section 26 — The reflection operator :math:`R`: where the BCs live
==================================================================

The factorisation :math:`K_{\rm bc} = G R P` separates what depends on
the cell geometry (:math:`P` and :math:`G`) from what depends on the
boundary condition physics (:math:`R`). Every boundary condition
supported in 1-D radial transport corresponds to a specific choice of
:math:`R`:

.. list-table:: Reflection operator :math:`R` for standard 1-D BCs
   :header-rows: 1
   :widths: 20 20 30 30

   * - BC
     - :math:`R`
     - Rank
     - Physical meaning
   * - Vacuum
     - :math:`0`
     - 0
     - No re-entering flux
   * - Mark (isotropic white)
     - :math:`e_0 e_0^{\top}`
     - 1
     - Only scalar mode; :math:`J^{-}_0 = J^{+}_0`
   * - Marshak DP\ :sub:`N-1`
     - :math:`\mathrm{diag}(1, 3, \ldots, 2N{-}1)`
     - :math:`N`
     - Each Legendre moment preserved; :math:`(2n{+}1)` normalisation
   * - Reflective (specular)
     - :math:`\mathrm{diag}((-1)^{n})` (parity basis)
     - :math:`N`
     - :math:`\psi^{-}(\mu) = \psi^{+}(-\mu)`
   * - Albedo :math:`\alpha`
     - :math:`\alpha\,R_{\rm white}`
     - :math:`N` (or < N)
     - Fractional reflection
   * - Interface current (lattice)
     - Non-square :math:`R: A^{\rm out}_{\rm cell-i} \to A^{\rm in}_{\rm cell-j}`
     - coupled
     - Cell-to-cell mode coupling

The rank-progression Mark → Marshak is now just the dimension of the
mode subspace :math:`R` acts on non-trivially. Albedo, partial
reflection, lattice coupling — all are different :math:`R`'s. The
geometry-specific tensors :math:`P` and :math:`G` are **shared** across
every BC of a given cell.

This is the **separation of concerns** that the rank-N effort was
reaching towards without naming. Once :math:`R` is recognised as the
entire locus of the BC physics, the implementation reduces to:

1. Compute :math:`P` and :math:`G` once per cell (expensive, depends
   on the multi-region :math:`\Sigma_t` profile and the quadrature
   settings).
2. Choose :math:`R` (trivial, tabular, :math:`N \times N`).
3. Compose.

The Gelbard :math:`(2n + 1)` normalisation factor that previously
lived on the :math:`v_n` side of the outer-product assembly is now
carried cleanly by :math:`R = \mathrm{diag}(2n + 1)`, as part of the
reflection-operator constructor :func:`reflection_marshak`. This is
the right place for it because it is a **statement about the basis
of :math:`A`**, not about the escape integral :math:`P`.


Section 27 — The escape and response tensors: where geometry lives
==================================================================

The two remaining tensors :math:`P \in \mathbb R^{N \times N_r}` and
:math:`G \in \mathbb R^{N_r \times N}` carry all the geometry-specific
work. Mode-by-mode:

.. math::
   :label: peierls-tensor-P-definition

   P^{n}{}_{j} \;\propto\;
     r_j^{d-1}\,w_j\,\mathcal P^{(n)}(r_j),

.. math::
   :label: peierls-tensor-G-definition

   G^{i}{}_{n} \;\propto\;
     \frac{\Sigma_t(r_i)\,\mathcal G^{(n)}(r_i)}{A_d^{\rm divisor}},

where :math:`\mathcal P^{(n)}` and :math:`\mathcal G^{(n)}` are the
mode-:math:`n` escape integral and mode-:math:`n` response integral
respectively (implemented in
:func:`~orpheus.derivations.peierls_geometry.compute_P_esc_mode` and
:func:`~orpheus.derivations.peierls_geometry.compute_G_bc_mode`, with
mode 0 routed through the legacy
:func:`~orpheus.derivations.peierls_geometry.compute_P_esc` /
:func:`~orpheus.derivations.peierls_geometry.compute_G_bc` for rank-1
bit-exact recovery).

**Both tensors are BC-independent.** Once :math:`P` and :math:`G` are
built, switching between vacuum, Mark, Marshak, albedo, or
interface-current BCs is a replacement of :math:`R` with no
recomputation of geometry integrals. This is a genuine and measurable
improvement over the pre-factored code: previously, each BC had its
own bespoke assembly function with overlapping integrals.

**Mode-independent contributions**. The source-side weights
:math:`r_j^{d-1}\,w_j` (volume-element Jacobian from the composite-GL
quadrature) and :math:`\Sigma_t(r_i)` (the macroscopic total
cross-section at node :math:`i`), and the surface-area divisor
:math:`A_d^{\rm divisor}`, are all mode-independent. They appear
as scaling factors on :math:`P` and :math:`G` but do not enter
:math:`R`. The factorisation naturally isolates them.


Section 28 — Connection to Hilbert-Schmidt and SVD theory
=========================================================

The finite factorisation :math:`K_{\rm bc} = G R P` has a canonical
infinite-dimensional counterpart via :eq:`peierls-svd`, the SVD of the
compact Hilbert-Schmidt operator :math:`T_{\rm bc}`:

.. math::

   T_{\rm bc} \;=\; \sum_{k=1}^{\infty} \sigma_k\,u_k \otimes v_k.

The **rank-:math:`N` SVD truncation**

.. math::

   T_{\rm bc}^{\rm SVD\text{-}N}
     \;:=\; \sum_{k=1}^{N} \sigma_k\,u_k \otimes v_k

is the **best** rank-:math:`N` approximation to :math:`T_{\rm bc}` in
the Frobenius / Hilbert-Schmidt norm. By the Eckart-Young theorem,

.. math::

   \|T_{\rm bc} - T_{\rm bc}^{\rm SVD\text{-}N}\|_{\rm HS}
     \;=\; \Bigl(\sum_{k > N} \sigma_k^{2}\Bigr)^{\!1/2}
     \;\le\; \|T_{\rm bc} - T_{\rm bc}^{(N)}\|_{\rm HS}

for any rank-:math:`N` operator :math:`T_{\rm bc}^{(N)}`, including
our Gelbard rank-:math:`N` closure. The singular vectors
:math:`\{u_k, v_k\}` are the **Karhunen-Loève basis** of
:math:`T_{\rm bc}` — the optimal modal basis for that specific
operator.

Why not use SVD directly?
-------------------------

Two reasons.

**(a) SVD is data-adaptive, Gelbard is physics-adaptive.** The
singular vectors :math:`(u_k, v_k)` depend on the cell geometry, the
optical profile :math:`\Sigma_t(r)`, and the BC (through :math:`R`
itself). Using them would require one SVD per problem:

- Per cell radius :math:`R`
- Per multi-region :math:`\Sigma_t` profile
- Per BC flavour (vacuum, white, albedo, lattice)
- Per multi-group energy dependence (with group-dependent
  :math:`\Sigma_t`, each group has a different SVD)

For a multi-group multi-cell lattice, the SVD bookkeeping becomes
prohibitive. Gelbard's :math:`\tilde P_n(\mu_s)` basis is **fixed once
and for all** — same basis for every cell, every material, every
group, every BC. The tensors :math:`P` and :math:`G` change; the mode
space :math:`A` does not.

**(b) The Gelbard basis has a physical interpretation.**
:math:`\tilde P_0` is the scalar (total) mode, :math:`\tilde P_1` is
the linear anisotropy, :math:`\tilde P_2` the quadratic, etc. Truncating
at :math:`N` says "I care about the first :math:`N` angular moments of
the re-entering flux." This interpretability is inherited by every
derived quantity: flux shapes, k\ :sub:`eff` sensitivities, surface
currents. SVD modes have no such interpretation — they are whatever
numerical artefacts the specific problem produces.

**Sub-optimality, quantified.** Let :math:`\epsilon_N^{\rm Gelbard}`
and :math:`\epsilon_N^{\rm SVD}` be the Frobenius errors at rank :math:`N`.
Then :math:`\epsilon_N^{\rm Gelbard} \ge \epsilon_N^{\rm SVD}` with
equality iff the Gelbard basis happens to coincide with the
Karhunen-Loève basis. For smooth angular-flux distributions with
exponential decay of high-Legendre-moment content, the gap is small:
the first few Gelbard modes closely match the dominant singular
vectors. For sharp boundary layers — thin cells, grazing rays — the
gap widens, which is precisely the regime where higher-rank-:math:`N`
is needed.

**The Marshak / Gelbard basis is the ansatz choice**: fix the basis,
accept sub-optimality, gain interpretability and geometry independence.
This is the same trade-off that motivates POD vs physics-based modal
bases throughout reduced-order modelling; see for example
[Atkinson1997]_ §6 on projection methods for integral equations.


Section 29 — Phases A and C: canonical Marshak investigated, deferred
=====================================================================

The factored :math:`K_{\rm bc} = G\,R\,P` picture suggests that the
remaining work in Issue #112 should localise cleanly: Phase A (sphere
plateau) as a change of basis in :math:`A`, Phase C (cylinder
divergence) as rebuilding :math:`P` and :math:`G` in a shared 3-D
frame. A 2026-04-18 empirical investigation (diagnostics
``diag_rank_n_{09,10,11,12}_*.py``, summarised in
``diag_rank_n_13_phaseAC_summary.md``) tested four canonical variants
against the current V1 implementation and found that none of the
natural "elegant" rewrites improve the convergence uniformly. The
investigation's results matter more than its failure, so they are
preserved here.

The continuous derivation
-------------------------

At the continuous level, the Marshak closure
:math:`J^{-}_n = J^{+}_n` with cosine-weighted moments

.. math::

   J_n \;=\; \int_0^1 \mu\,\tilde P_n(\mu)\,\psi(\mu)\,\mathrm d\mu

expands as

.. math::

   J_n \;=\; \sum_m \alpha_m\,B_{nm}, \qquad
   B_{nm} \;=\; \int_0^1 \mu\,\tilde P_n(\mu)\,\tilde P_m(\mu)\,
                \mathrm d\mu,

where :math:`\alpha_m` are the coefficients of :math:`\psi =
\sum_m \alpha_m \tilde P_m`. Marshak is therefore equivalent (within
rank :math:`N`, for invertible :math:`B^{(N)}`) to the coefficient-space
closure :math:`\alpha^-_m = \alpha^+_m`. Translating to our factored
form

.. math::

   K_{\mathrm{bc}} \;=\; G\,B^{-1}\,P_{\rm moment},

where :math:`P_{\rm moment}` represents the cosine-weighted moment
:math:`J^{+}_m(r_j)`. This identifies **three candidate rewrites**:

1. **Mode-0 convention**: use the canonical cosine-weighted moment for
   mode 0 as well as for :math:`n \ge 1` (instead of the legacy
   isotropic-source escape probability).
2. **Reflection operator**: replace :math:`R = \mathrm{diag}(2n+1)`
   with :math:`R = B^{-1}`.
3. **Integrand form**: use :math:`\mu_{\rm exit}` as the explicit
   cosine weight in the integrand, rather than the :math:`(\rho_{\max}
   /R)^2` surface-to-observer Jacobian.

All three adjustments look "more canonical" from the derivation and
each suggests itself as the Phase A / C fix. They are tested
empirically as variants V2, V4, V6 below.

Empirical variant scan
----------------------

Bare homogeneous 1G 1-region white-BC eigenvalue
(:math:`\Sigma_t = 1`, :math:`\Sigma_s = 0.5`,
:math:`\nu\Sigma_f = 0.75`, :math:`k_\infty = 1.5`):

.. list-table:: Sphere :math:`k_{\rm eff}` error vs rank :math:`N`
   :header-rows: 1
   :widths: 10 14 14 14 14 14 14 14

   * - Variant
     - Description
     - R=1 N=1
     - R=1 N=2
     - R=1 N=3
     - R=1 N=8
     - R=10 N=1
     - R=10 N=8
   * - V1
     - Current (shipped)
     - 27 %
     - **1.2 %**
     - 2.3 %
     - 2.5 %
     - 0.28 %
     - 0.17 %
   * - V2
     - Jacobian mode 0, :math:`R=\mathrm{diag}(2n+1)`
     - 50 %
     - 29 %
     - 25 %
     - 24 %
     - 5.3 %
     - 5.2 %
   * - V4
     - Jacobian mode 0, :math:`R=B^{-1}`
     - 29 %
     - 16 %
     - 15 %
     - 15 %
     - 5.2 %
     - 5.2 %
   * - V6
     - Cosine-:math:`\mu` integrand, :math:`R=B^{-1}`
     - **1.1 %**
     - 19 %
     - 19 %
     - 19 %
     - 6.3 %
     - 6.9 %

.. list-table:: Cylinder :math:`k_{\rm eff}` error vs rank :math:`N`
   :header-rows: 1
   :widths: 10 14 14 14 14 14 14 14

   * - Variant
     - Description
     - R=1 N=1
     - R=1 N=2
     - R=1 N=3
     - R=1 N=8
     - R=10 N=1
     - R=10 N=8
   * - V1
     - Current (shipped)
     - 21 %
     - 8.3 %
     - 27 %
     - 107 %
     - 1.1 %
     - 0.9 %
   * - V2
     - Jacobian mode 0, :math:`R=\mathrm{diag}(2n+1)`
     - 41 %
     - 17 %
     - **0.45 %**
     - 65 %
     - —
     - —
   * - V4
     - Jacobian mode 0, :math:`R=B^{-1}`
     - 23 %
     - 2.6 %
     - 13 %
     - 78 %
     - —
     - —

Observations
------------

1. **V1 is empirically the best overall**. Sphere converges to a
   :math:`\sim 2.5\,\%` plateau at thin :math:`R`; thick cells are
   sub-percent. Cylinder rank-1 matches legacy Mark accurately.

2. **V2 cylinder at rank-3 R=1 MFP hits the canonical DP_2
   prediction** (:math:`0.45\,\%`). This *proves* that the factored
   structure :math:`G R P` is capable of canonical rank-:math:`N`
   convergence — but only at the cost of a **degraded rank-1**
   convention (41 % error at rank 1).

3. **V6's cosine-integrand rank-1 is accidentally good for sphere at
   R=1** (1.1 %) but its rank-:math:`N` never improves. The rank-1
   match is a geometric coincidence at thin :math:`R` where
   :math:`\mu_{\rm exit} \approx 1` for most rays, not a canonical
   result.

4. **Changing** :math:`R` **from** :math:`\mathrm{diag}(2n+1)` **to**
   :math:`B^{-1}` **does not rescue** a degraded mode-0 convention.
   V4 improves sphere from 24 % plateau (V2) to 15 % plateau — a
   marginal cosmetic change on a badly-conditioned rank-1.

What this teaches
-----------------

The investigation reveals a **conceptual entanglement**:

- The legacy mode-0 convention (isotropic-source escape probability,
  no Jacobian, paired with :math:`R_{00} = 1`) is *not* the canonical
  partial-current moment. It happens to give the Mark rank-1 answer
  that ORPHEUS has treated as the baseline.
- The canonical mode-0 convention (cosine-weighted moment with
  :math:`(\rho_{\max}/R)^2` Jacobian or explicit :math:`\mu_{\rm
  exit}` weight) gives a *different* rank-1 result — one that,
  without calibration, is further from :math:`k_\infty` than the Mark
  answer for this particular test problem.
- The mode-:math:`n \ge 1` corrections that worked in V1 (with the
  Jacobian factor) *were empirically tuned to the legacy mode 0*,
  not to a canonical basis.
- No combination of reflection operator :math:`R` and integrand
  redefinition has produced uniform convergence across
  :math:`(R, N)`.

The canonical DP_N convergence is **available** (V2's cylinder
rank-3 at R=1: 0.45 %) but only with a mode-0 convention that the
downstream ORPHEUS tests assume. Reconciling the two is the real
Phase A / C fix, and it is **not a one-line integrand swap**.

A principled path forward
-------------------------

The canonical resolution requires **calibration against a published
reference**. Stepanek 1981 ([Stepanek1981]_) provides the slab
DP\ :sub:`N` :math:`k_{\rm eff}` tables at Marshak rank 0, 1, 2 for a
range of :math:`R/\mathrm{MFP}`, with the canonical expansion and
closure conventions fixed. Calibrating a new ORPHEUS
``compute_P_esc_mode_canonical`` / ``compute_G_bc_mode_canonical``
pair against the slab DP\ :sub:`N` tables determines the correct
mode-0 normalisation and reveals whether the subsequent extension to
sphere / cylinder needs extra geometry factors. This is a
multi-session effort: (i) implement Stepanek slab DP\ :sub:`N`, (ii)
port the normalisation to sphere, (iii) port to cylinder with
Knyazev's :math:`\mathrm{Ki}_{2+k}` polynomial expansion.

Deferred work
-------------

**Phase A** (sphere plateau at :math:`\sim 2.5\,\%` for thin R):
**deferred**. No simple fix identified in the 2026-04-18 investigation.
Phase-A-true needs Stepanek slab calibration as the anchor.

**Phase C** (cylinder divergence at :math:`N \ge 3`): **deferred**
for the same reason. The :math:`\mathrm{Ki}_{2+k}` polynomial
expansion infrastructure from [Knyazev1993]_ is theoretically ready
(the polar-angle integration of
:math:`\tilde P_n(\sin\theta_p \cdot \mu_{s,2D})` against
:math:`\exp(-\tau/\sin\theta_p)` produces exactly this series) but
requires the calibrated mode-0 convention to attach to.

**V1 remains the shipped implementation.** Its sphere plateau is
documented as a theoretical ceiling of the mixed-convention rank-N
closure; its cylinder divergence at high N is documented as a known
limitation. The factored :class:`~orpheus.derivations.peierls_geometry.BoundaryClosureOperator`
architecture makes both fixes *localised* to two integrand functions
once the correct conventions are found — a structural payoff that
survives the empirical setback.

The diagnostic scripts ``diag_rank_n_{09-12}_*.py`` and the summary
``diag_rank_n_13_phaseAC_summary.md`` preserve the variant data for
the next investigator who resumes this work.


Section 30 — The ``BoundaryClosureOperator`` dataclass
======================================================

The :class:`~orpheus.derivations.peierls_geometry.BoundaryClosureOperator`
dataclass carries the three tensors :math:`P`, :math:`G`, :math:`R` and
exposes the factored-form operations directly:

.. code-block:: python

   @dataclass(frozen=True)
   class BoundaryClosureOperator:
       P: np.ndarray  # shape (N_modes, N_nodes)  — V → A
       G: np.ndarray  # shape (N_nodes, N_modes)  — A → V
       R: np.ndarray  # shape (N_modes, N_modes)  — A → A

       def apply(self, q):
           return self.G @ (self.R @ (self.P @ q))

       def as_matrix(self):
           return self.G @ self.R @ self.P

       @property
       def closure_rank(self):
           return np.linalg.matrix_rank(self.R)

The three methods embody the three algorithmic regimes:

- :meth:`~.BoundaryClosureOperator.apply` — matrix-free, for large
  :math:`N_r` with iterative solvers (GMRES, BiCGStab). Cost
  :math:`\mathcal O(N_r N + N^{2})` per apply.
- :meth:`~.BoundaryClosureOperator.as_matrix` — on-demand dense
  assembly for direct-LU eigenvalue iteration (what
  :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
  currently uses). Cost :math:`\mathcal O(N_r^{2} N)` once, per
  cell, per group, per BC.
- :attr:`~.BoundaryClosureOperator.closure_rank` — structural
  diagnostic, checks that :math:`R` has the expected rank signature
  (rank 0 for vacuum, 1 for Mark, :math:`N` for Marshak, non-trivial
  for lattice couplings).

The assembly function
:func:`~orpheus.derivations.peierls_geometry.build_closure_operator`
takes a geometry, a radial grid, the optical profile, and a
reflection choice (``"vacuum"``, ``"mark"``, ``"marshak"``, or any
user-supplied :math:`N \times N` matrix) and returns the factored
operator. It is the **canonical** entry point for constructing the
boundary closure at any rank and for any BC — no separate
per-BC assembly functions are required.

The pre-existing
:func:`~orpheus.derivations.peierls_geometry.build_white_bc_correction_rank_n`
remains as a thin convenience wrapper that calls
:func:`build_closure_operator` with ``reflection="marshak"`` and
returns :meth:`~.BoundaryClosureOperator.as_matrix`. This is purely
for API continuity with the pre-factored-form callers (chiefly
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`).

Reflection-operator constructors
--------------------------------

Three canonical reflection operators are provided as free functions in
:mod:`~orpheus.derivations.peierls_geometry`:

- :func:`~orpheus.derivations.peierls_geometry.reflection_vacuum` —
  :math:`R = 0` (vacuum BC)
- :func:`~orpheus.derivations.peierls_geometry.reflection_mark` —
  :math:`R = e_0 e_0^{\top}` (rank-1 isotropic white)
- :func:`~orpheus.derivations.peierls_geometry.reflection_marshak` —
  :math:`R = \mathrm{diag}(1, 3, 5, \ldots, 2N{-}1)` (Marshak
  DP\ :sub:`N-1` with Gelbard normalisation)

A user wishing to experiment with, say, an albedo-0.7 BC need only
pass ``reflection = 0.7 * reflection_marshak(N)`` to
:func:`build_closure_operator`. The geometry integrals are not
recomputed.

Regression gates
----------------

The factored form is gated by three classes of tests in
``tests/derivations/test_peierls_closure_operator.py`` (foundation
tests, software-invariant contracts):

1. **Algebraic consistency**:
   :meth:`~.BoundaryClosureOperator.apply` and
   :meth:`~.BoundaryClosureOperator.as_matrix` must agree on random
   input vectors to machine precision (rtol 1e-13).
2. **Rank structure**:
   :attr:`~.BoundaryClosureOperator.closure_rank` must equal
   :math:`\mathrm{rank}(R)` for each canonical reflection constructor.
3. **BC-as-R**: Marshak via ``reflection="marshak"`` matches
   :func:`build_white_bc_correction_rank_n` exactly; rank-1 Mark /
   Marshak matches legacy :func:`build_white_bc_correction` exactly;
   vacuum gives the zero matrix.

Together these gates ensure that any future refactor — including
the Phase A basis change and the Phase C 3-D frame — cannot break
the tensor-network structure without one of these contracts firing.


References
==========

.. [Sanchez1982] R. Sanchez and N.J. McCormick, "A Review of Neutron
   Transport Approximations," *Nucl. Sci. Eng.* **80**, 481–535
   (1982). DOI: 10.13182/nse80-04-481.

.. [Hebert2020] A. Hébert, *Applied Reactor Physics*, 3rd ed.,
   Presses Internationales Polytechnique, 2020.
   DOI: 10.1515/9782553017445.

.. [Stamm1983] R. Stamm'ler and M.J. Abbate, *Methods of Steady-State
   Reactor Physics in Nuclear Design*, Academic Press, 1983.

.. [BellGlasstone1970] G.I. Bell and S. Glasstone, *Nuclear Reactor
   Theory*, Van Nostrand Reinhold, 1970.

.. [Carlvik1966] I. Carlvik, "A method for calculating collision
   probabilities in general cylindrical geometry and applications to
   flux distributions and Dancoff factors," *Proc. Third United Nations
   Int. Conf. Peaceful Uses of Atomic Energy*, Vol. 2, 1966.

.. [CaseZweifel1967] K.M. Case and P.F. Zweifel,
   *Linear Transport Theory*, Addison-Wesley, 1967.

.. [Davison1957] B. Davison, *Neutron Transport Theory*,
   Clarendon Press, 1957. Ch. 5 covers the slab albedo / partial-current
   balance relations that underpin the Mark (isotropic rank-1)
   white-BC closed form :eq:`peierls-white-bc-slab`.

.. [Atkinson1997] K.E. Atkinson, *The Numerical Solution of Integral
   Equations of the Second Kind*, Cambridge University Press, 1997.

.. [AbramowitzStegun1964] M. Abramowitz and I.A. Stegun (eds.),
   *Handbook of Mathematical Functions with Formulas, Graphs, and
   Mathematical Tables*, National Bureau of Standards Applied
   Mathematics Series 55 (1964); §5.1 (exponential integral
   :math:`E_n`), §11.2 (Bickley–Naylor functions
   :math:`\mathrm{Ki}_n`).

.. [Bickley] W.G. Bickley and J. Naylor, "A short table of the
   functions :math:`\mathrm{Ki}_n(x)`, from :math:`n=1` to
   :math:`n=16`," *Philosophical Magazine Series 7*,
   **20**, 343–347 (1935).

.. [MartinBrown2003] W.R. Martin and F.B. Brown, "Status of MCNP5,"
   Los Alamos National Laboratory technical memorandum LA-UR-03-7127,
   2003. Discusses delta-tracking performance and accuracy in the
   MCNP5 implementation, and reviews the Woodcock algorithm's
   correspondence to deterministic ray integration in optically thick
   media.

.. [Leppanen2010] J. Leppänen, "Performance of Woodcock delta-tracking
   in lattice physics applications using the Serpent Monte Carlo
   reactor physics burnup calculation code," *Annals of Nuclear
   Energy* **37** (5), 715–722 (2010).
   DOI: 10.1016/j.anucene.2010.01.011. Serpent code implementation of
   delta-tracking with optimisations for continuous-energy
   cross-section lookup; the description of the τ-space ray walker
   (§3 of the paper) is the closest stochastic analogue of the
   deterministic ``optical_depth_along_ray_with_map`` helper proposed
   for Phase H.2 (`Issue #109 <https://github.com/deOliveira-R/ORPHEUS/issues/109>`_).

.. [Stepanek1981] J. Stepanek, "The DP\ :sub:`N` Surface Flux Integral
   Neutron Transport Method for Slab Geometry," *Nuclear Science and
   Engineering* **78**, 171–179 (1981).
   DOI: 10.13182/NSE81-A19606. Canonical derivation of the rank-N DP
   closure for the surface-flux integral equation in slab geometry;
   the slab DP\ :sub:`N` k\ :sub:`eff` tables in Tables I–III are the
   calibration anchor for the Phase-A sphere normalisation audit
   (see :ref:`theory-peierls-unified` §29 and Issue #112).

.. [Knyazev1993] A.P. Knyazev, "Solution of the transport equation in
   integral form in a one-dimensional cylindrical geometry with
   linearly anisotropic scattering," *Atomic Energy* **74** (5),
   385–389 (1993). DOI: 10.1007/BF00844623. Derivation of the
   higher-order Bickley-Naylor functions :math:`\mathrm{Ki}_{2+k}`
   arising from the :math:`P_n`-weighted polar-angle integration in
   cylindrical geometry — the analytic identity behind the Phase-C
   3-D angular-quadrature reformulation of the cylinder
   :func:`~orpheus.derivations.peierls_geometry.compute_P_esc_mode` /
   :func:`~orpheus.derivations.peierls_geometry.compute_G_bc_mode`
   primitives (Issue #112).
