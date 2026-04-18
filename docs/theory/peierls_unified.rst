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
   (:func:`~orpheus.derivations.peierls_cylinder.solve_peierls_cylinder_1g_vacuum`)
   and the rank-1 white-BC correction
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
