.. _theory-diffusion-1d:

==================
1D Diffusion (P1)
==================

The diffusion equation is the lowest-order angular approximation of
the neutron transport equation: the P1 (first spherical-harmonic)
expansion truncated to the scalar flux. It is not a transport
solver in the strict sense — it discards angular information
beyond the current — but it is the workhorse of reactor design and
its verification is a mathematical problem in its own right.

This page carries the continuous reference solutions for the
ORPHEUS diffusion module and the equation labels the verification
tests point at. It is deliberately scoped to **1D plane geometry**:
multi-dimensional diffusion and cylindrical/spherical variants are
tracked as follow-ups.

.. contents::
   :local:
   :depth: 2


Key Facts
=========

- The 1D multigroup diffusion equation in plane geometry is a
  second-order elliptic boundary-value problem in :math:`\phi_g(x)`
  with vacuum Dirichlet conditions at the slab edges:

  .. math::
     :label: diffusion-operator

     -\frac{d}{dx}\!\left(D_g(x)\,\frac{d\phi_g}{dx}\right)
       + \Sigma_{r,g}(x)\,\phi_g
       = \sum_{g' \ne g} \Sigma_{s,g' \to g}\,\phi_{g'}
       + \frac{1}{k}\,\chi_g\sum_{g'}\nu\Sigma_{f,g'}\,\phi_{g'}.

- The diffusion coefficient in each region is computed from the
  transport cross section:

  .. math::
     :label: diffusion-coefficient

     D_g(x) \;=\; \frac{1}{3\,\Sigma_{\text{tr},g}(x)}.

- The removal cross section is the total minus in-group
  scattering:

  .. math::

     \Sigma_{r,g} \;=\; \Sigma_{t,g} - \Sigma_{s,g\to g}.

- Standard discretisation is central finite difference on a
  uniform mesh; the design spatial order is
  :math:`\mathcal O(h^{2})` for smooth cross sections and
  :math:`\mathcal O(h)` at material interfaces that do not lie on
  cell faces.

- Boundary conditions: ORPHEUS uses hard Dirichlet
  :math:`\phi_g(0) = \phi_g(L) = 0` for all groups — the zero-flux
  variant of the vacuum condition (contrast with the
  extrapolation-distance Marshak form). This choice is
  intentional: it lets the analytical reference solutions below be
  pure sinusoids without an extrapolation-length fudge, so the
  spatial convergence test isolates finite-difference error.


.. _diffusion-1g-bare-slab:

1-group bare slab
=================

The simplest verification configuration: a homogeneous slab of
thickness :math:`L` with zero-flux vacuum boundaries and a
single energy group. The diffusion equation collapses to

.. math::

   -D\,\phi''(x) + \Sigma_r\,\phi(x)
      = \frac{1}{k}\,\nu\Sigma_f\,\phi(x)

with :math:`\phi(0) = \phi(L) = 0`. Separation of variables
gives the eigenfunction

.. math::
   :label: bare-slab-eigenfunction

   \phi(x) \;=\; \sin\!\left(\frac{\pi x}{L}\right)

and the **geometric buckling**

.. math::
   :label: bare-slab-buckling

   B^{2} \;=\; \left(\frac{\pi}{L}\right)^{2}.

Substituting :eq:`bare-slab-eigenfunction` into the diffusion
equation yields the eigenvalue condition

.. math::
   :label: bare-slab-critical-equation

   D\,B^{2} + \Sigma_r \;=\; \frac{1}{k}\,\nu\Sigma_f,

which solves to

.. math::

   k \;=\; \frac{\nu\Sigma_f}{D\,B^{2} + \Sigma_r}.

Because the eigenfunction is independent of group in the
multigroup generalisation (all groups share the same spatial
:math:`\sin(\pi x/L)` shape), multigroup reduces to a
:math:`ng \times ng` matrix eigenvalue problem in the spectrum
vector — exactly what
:func:`orpheus.derivations.homogeneous.kinf_and_spectrum_homogeneous`
solves, plus an extra ``D B²`` removal term on the diagonal of
:math:`\mathbf{A}`.

This is a **T1 analytical reference**: no integration, no
quadrature, no iteration. See
:func:`orpheus.derivations.diffusion.derive_1rg_continuous` for the
Phase-0 :class:`~orpheus.derivations.ContinuousReferenceSolution`
that carries :math:`k_{\text{eff}}` and the continuous
multigroup eigenfunction callable.


.. _diffusion-2rg-fuel-reflector:

2-group fuel + reflector slab
=============================

A more demanding verification problem: fuel surrounded by a
reflector, both treated with 2-group diffusion, with vacuum
boundaries on the outer faces. The eigenfunction is no longer
a single sine — it is a linear combination of region-local
exponential and/or trigonometric modes, matched across the
fuel/reflector interface.

The rest of this section describes the implementation that
actually works. :ref:`diffusion-2rg-investigation-history` at
the end of the section records the two earlier approaches that
were tried and abandoned, with the numerical evidence of their
failure modes, so no future session reinvents them.

Region ODE and spatial modes
----------------------------

In each homogeneous region the multigroup diffusion equation
:eq:`diffusion-operator` reduces to

.. math::
   :label: diffusion-region-ode

   -\mathbf D\,\boldsymbol\phi''(x) + \mathbf M(k)\,\boldsymbol\phi(x)
     \;=\; \mathbf 0,

where the net removal matrix

.. math::
   :label: diffusion-M-matrix

   \mathbf M(k) \;=\; \text{diag}(\Sigma_{a,g} + \Sigma_{s,g,\text{out}})
                    \;-\; (\text{downscatter coupling})
                    \;-\; \frac{1}{k}\,\chi \otimes (\nu\Sigma_f)

absorbs the in-scatter source, the fission source, and the
removal losses into a single :math:`ng \times ng` operator.

Look for solutions of the form
:math:`\boldsymbol\phi(x) = e^{\lambda(x-x_0)}\,\mathbf u`:
substituting into :eq:`diffusion-region-ode` gives the
generalised eigenvalue problem

.. math::
   :label: diffusion-mode-decomposition

   \mathbf D^{-1}\mathbf M(k)\,\mathbf u_i \;=\; \mu_i\,\mathbf u_i,
   \qquad \mu_i \;=\; \lambda_i^{2}.

For a 2-group problem, :math:`\mathbf D^{-1}\mathbf M` is
:math:`2 \times 2`, so there are two eigenvalues
:math:`\mu_1, \mu_2` and two eigenvectors
:math:`\mathbf u_1, \mathbf u_2`. Each eigenvalue gives a
**pair** of spatial modes — the sign of :math:`\mu_i`
determines whether they are exponentials or trigonometrics:

.. math::
   :label: diffusion-exponential-branch

   \mu_i > 0 \;\Rightarrow\;
     \phi(x) \;=\; c_{i}^{+}\,e^{-\sqrt{\mu_i}\,(L_{\text{reg}} - (x - x_0))}\,\mathbf u_i
     \;+\; c_{i}^{-}\,e^{-\sqrt{\mu_i}\,(x - x_0)}\,\mathbf u_i

(subcritical region — pure decay modes, anchored to opposite
edges so both exponentials are bounded by :math:`1` within the
region), and

.. math::
   :label: diffusion-trigonometric-branch

   \mu_i < 0 \;\Rightarrow\;
     \phi(x) \;=\; c_{i}^{c}\,\cos\!\bigl(\sqrt{-\mu_i}(x - x_{\text{mid}})\bigr)\,\mathbf u_i
     \;+\; c_{i}^{s}\,\sin\!\bigl(\sqrt{-\mu_i}(x - x_{\text{mid}})\bigr)\,\mathbf u_i

(supercritical region — bounded oscillations centred at the
region midpoint :math:`x_{\text{mid}} = x_0 + L_{\text{reg}}/2`).

**Why this sign-branched basis matters.** The fuel region at
the fundamental mode :math:`k \approx 0.87` has one
:math:`\mu_i > 0` and one :math:`\mu_i < 0` — i.e. the fast
thermal couple includes *both* an exponential and a
trigonometric mode. The reflector at the same :math:`k` has
two exponential modes (both subcritical, as expected).
**Every basis mode in every region is bounded by 1** on its
domain. This bound is load-bearing: it is what makes the
assembled matching matrix :math:`\mathbf C(k)` below have
entries of :math:`\mathcal{O}(1)` and hence a determinant that
does not suffer catastrophic cancellation. See
:ref:`diffusion-2rg-investigation-history` for the earlier
approach that got this wrong.

Interface matching and vacuum boundary conditions
-------------------------------------------------

With the mode basis above, the solution in each region is a
linear combination of 4 basis functions (2 eigenvalues × 2
modes per eigenvalue). For the fuel + reflector slab we have
8 unknown mode coefficients total — 4 in fuel, 4 in reflector.

The 8 constraints that close the system:

.. math::

   \boldsymbol\phi_{\text{fuel}}(0) \;=\; \mathbf 0
   \quad (\text{vacuum left, 2 equations}),

.. math::
   :label: diffusion-interface-matching

   \boldsymbol\phi_{\text{fuel}}(H_f) \;=\;
     \boldsymbol\phi_{\text{refl}}(H_f),
   \qquad
   \mathbf J_{\text{fuel}}(H_f) \;=\; \mathbf J_{\text{refl}}(H_f)
   \quad (\text{interface, 4 equations}),

.. math::

   \boldsymbol\phi_{\text{refl}}(H_f + H_r) \;=\; \mathbf 0
   \quad (\text{vacuum right, 2 equations}),

where the group current
:math:`\mathbf J_g(x) = -D_g\,\phi_g'(x)` is derived from the
mode basis analytically (the derivative of each basis mode has
a known closed form).

Collecting the 8 constraints into a matrix equation on the
8 mode coefficients
:math:`\mathbf c = [\mathbf c_{\text{fuel}};\;
\mathbf c_{\text{refl}}]` gives

.. math::
   :label: diffusion-matching-matrix

   \mathbf C(k)\,\mathbf c \;=\; \mathbf 0,
   \qquad \mathbf C(k) \in \mathbb R^{8 \times 8}.

Because every basis mode is bounded by 1,
:math:`\mathbf C(k)` has :math:`\mathcal{O}(1)` entries
(typical condition number :math:`\sim 30` at non-root values
of :math:`k`). A non-trivial mode coefficient vector exists
iff :math:`\mathbf C(k)` is singular, and the transcendental
eigenvalue condition is

.. math::
   :label: diffusion-transcendental

   \det\!\bigl(\mathbf C(k)\bigr) \;=\; 0.

This is bracketed by a coarse scan over :math:`k` and refined
via :func:`scipy.optimize.brentq` to xtol :math:`= 10^{-14}`.

Physical validation of candidate roots
--------------------------------------

The bracketing-and-refine pipeline above finds **more** sign
changes in :math:`\det(\mathbf C(k))` than there are actual
eigenvalues. These extra sign changes are an artefact of how
:func:`numpy.linalg.eig` orders the eigenvalues of
:math:`\mathbf D^{-1}\mathbf M(k)`: the order is not
continuous in :math:`k` across critical values where two
eigenvalues cross. When the order permutes, the columns of
:math:`\mathbf C(k)` permute discontinuously, and
:math:`\det(\mathbf C)` flips sign **by permutation** rather
than by passing through zero in a physically meaningful way.
``brentq`` then "converges" to a :math:`k` where
:math:`\mathbf C` is numerically singular by accident of the
eigenvalue labelling, not because the boundary-value problem
has a genuine solution there.

Each candidate root is therefore **physically validated**:

.. math::
   :label: diffusion-spurious-root-validation

   \boldsymbol\phi_{\text{fuel}}(0) \;\overset{?}{\approx}\; \mathbf 0,
   \quad
   \boldsymbol\phi_{\text{fuel}}(H_f) - \boldsymbol\phi_{\text{refl}}(H_f)
     \;\overset{?}{\approx}\; \mathbf 0,
   \quad
   \boldsymbol\phi_{\text{refl}}(H_f + H_r) \;\overset{?}{\approx}\; \mathbf 0.

The null vector of :math:`\mathbf C(k_{\text{cand}})` is
extracted via SVD, substituted back into the mode basis, and
the three residuals above are evaluated. A candidate passes
validation only when all three are below :math:`10^{-7}`
relative to the peak flux; otherwise it is rejected as a
spurious sign change. On the default 50 + 30 cm geometry the
scan between :math:`k \in [0.1, 3.0]` returns **six**
candidates, of which **three** are physical eigenvalues (0.370,
0.590, 0.870 — the fundamental plus two harmonics) and the
other three are eigenvalue-crossing artefacts. The fundamental
mode is the largest validated root.

Back-substitution for continuous :math:`\phi(x)`
-------------------------------------------------

Once the fundamental :math:`k_{\text{fund}}` is known and
validated, the null vector of
:math:`\mathbf C(k_{\text{fund}})` — extracted one last time
by SVD — gives the 8 mode coefficients
:math:`\mathbf c = [\mathbf c_{\text{fuel}};\;
\mathbf c_{\text{refl}}]`. The continuous flux at any
:math:`x` in the slab is then evaluated **pointwise** from
the region-local mode basis:

.. math::
   :label: diffusion-back-substitution

   \boldsymbol\phi(x) \;=\;
   \begin{cases}
     \displaystyle\sum_{j=1}^{4} c_{\text{fuel},j}\,m_j^{\text{fuel}}(x)\,\mathbf u_j^{\text{fuel}}
       & 0 \le x \le H_f, \\[0.3em]
     \displaystyle\sum_{j=1}^{4} c_{\text{refl},j}\,m_j^{\text{refl}}(x - H_f)\,\mathbf u_j^{\text{refl}}
       & H_f \le x \le H_f + H_r,
   \end{cases}

where :math:`m_j` is the :math:`j`-th basis mode
(:eq:`diffusion-exponential-branch` or
:eq:`diffusion-trigonometric-branch`) of the enclosing
region. **No matrix exponentials, no composition, no
condition-number explosion.** Every evaluation is a handful
of real multiplications plus at most two calls to
:func:`numpy.exp` / :func:`numpy.cos` / :func:`numpy.sin`.

The back-substituted :math:`\boldsymbol\phi(x)` is
**mesh-independent**: the test chooses its own cell centres,
calls
:meth:`~orpheus.derivations.ContinuousReferenceSolution.phi_on_mesh`,
and compares the diffusion solver's output to the continuous
reference at exactly those points. See
:func:`orpheus.derivations.diffusion.derive_2rg_continuous`.

This is a **T2 semi-analytical reference**: the eigenvalue
:math:`k` is found to ``xtol=1e-14`` via brentq on a
well-conditioned determinant, the null vector is SVD-accurate
to machine precision on a matrix with condition number
:math:`\sim 10^{15}` *only at the eigenvalue itself*, and
:math:`\phi(x)` evaluation is pure O(1) algebra.

Numerical evidence at the default 50 + 30 cm geometry
-----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Expected
     - Observed
   * - :math:`k_{\text{eff}}` (transcendental)
     - —
     - :math:`0.8705068089`
   * - :math:`k_{\text{eff}}` vs legacy Richardson cache
     - :math:`\lesssim 10^{-5}` (Richardson :math:`\mathcal O(h^{4})`)
     - :math:`7 \times 10^{-8}`
   * - :math:`\phi_g(0)` residual (both groups)
     - machine :math:`\epsilon`
     - :math:`\sim 10^{-16}`
   * - :math:`\phi_g(L)` residual (both groups)
     - machine :math:`\epsilon`
     - :math:`\sim 10^{-16}`
   * - Interface continuity of :math:`\phi_g`
     - machine :math:`\epsilon`
     - :math:`\sim 10^{-11}`
   * - :math:`\mathbf C(k_{\text{fund}})` condition number
     - :math:`\sim 10^{15}` (singular)
     - :math:`\sim 10^{15}`
   * - :math:`\mathbf C(k_{\text{off-root}})` condition number
     - :math:`\mathcal{O}(1\text{–}100)`
     - :math:`\sim 30`

The finite-difference diffusion solver is then verified
against this reference by running a mesh refinement study;
see the :ref:`diffusion-2rg-verification` section below.


.. _diffusion-2rg-investigation-history:

Investigation history — two abandoned approaches
-------------------------------------------------

This section exists **on purpose** (Cardinal Rule 2: Sphinx
is the LLM's brain). The implementation above went through
two serious dead ends before converging. Both failure modes
would otherwise be repeated by any future session that reads
the textbooks and writes the "obvious" code.

**Dead end #1 — First-order ODE state-vector composition
with** :func:`scipy.linalg.expm`.

The textbook approach
(Duderstadt & Hamilton 1976 §7-6; Stammler-Abbate §4.2) is
to carry the state vector
:math:`\mathbf y = [\boldsymbol\phi;\,\mathbf J]` and
propagate it through each region by the matrix exponential
:math:`\mathbf T(t) = \exp(\mathbf S\,t)` where

.. math::

   \mathbf S(k) \;=\;
   \begin{pmatrix} \mathbf 0 & -\mathbf D^{-1} \\
                    -\mathbf M(k) & \mathbf 0 \end{pmatrix}.

Continuity of :math:`\phi` and :math:`J` at the fuel/reflector
interface is then automatic because the state vector is the
continuous quantity. Vacuum BCs
:math:`\phi(0) = \phi(L) = 0` pick out the upper-right
:math:`ng \times ng` block of the composed transfer matrix,
and the eigenvalue condition is
:math:`\det\bigl(\mathbf T_{\text{total}}(k)_{[0{:}ng,\,ng{:}2ng]}\bigr) = 0`.

This was implemented, the determinant brentq converged, and
the resulting "null vector" gave
:math:`|\phi_g(L)| \approx 5 \times 10^{-4}` relative to peak
flux — far from the machine-precision zero the vacuum BC
demands. Investigation of the intermediate quantities
revealed:

- Condition number of the composed transfer matrix:
  :math:`\sim 10^{17}` over the 80 cm slab.
- Norm of the upper-right block: :math:`\sim 10^{15}`.
- Entries of the block: magnitudes up to :math:`\sim 10^{15}`.
- Pairwise products forming the determinant: :math:`\sim 10^{26}`.
- The returned determinant: :math:`\sim 10^{9}`.

That is a 17-decade cancellation of two 26-decade numbers,
leaving a "determinant" whose last 17 digits are numerical
noise. brentq dutifully finds a :math:`k` where this noisy
residual changes sign, but the resulting matrix is *not*
actually singular at that :math:`k` — its smallest singular
value is :math:`\sim 10^{-3}`, not the :math:`\sim 10^{-16}`
it would be at a genuine eigenvalue.

The root cause is that
:math:`\mathbf T_{\text{total}}(k) = \exp(\mathbf S\,t)` for
:math:`t = 80` cm contains exponentials
:math:`e^{\pm\lambda t}` with
:math:`\lambda \sim 0.5 \text{ cm}^{-1}` (the fast diffusion
length reciprocal), so
:math:`e^{\lambda t} \sim e^{40} \sim 10^{17}`. The growing
mode dominates the matrix; the physically meaningful decaying
mode lives in the last 17 bits of double precision. **No
condition-number improvement to the block, no tighter brentq
tolerance, no alternative root-finding criterion rescues
this formulation.** The error is baked into :math:`\mathbf T`
before any downstream step sees it.

**Dead end #2 — Complex eigenvalues and null-vector phase
ambiguity.**

The fix for dead end #1 is to diagonalise
:math:`\mathbf S` analytically per region and propagate in
the eigenbasis, so the growing and decaying modes are
separable and each mode is a bounded exponential. The first
attempt at this used complex eigenvalues throughout (because
:math:`\mathbf D^{-1}\mathbf M` can have negative eigenvalues
for supercritical regions, and "one size fits all" complex
arithmetic seemed simpler than sign-branching).

This nearly worked. The eigenvalue condition converged,
:math:`\phi(0)` and :math:`\phi(L)` were both machine precision
at each candidate root — but the SVD null vector was complex,
and the reconstructed :math:`\phi(x)` at interior points had
imaginary components comparable to its real part. The
imaginary components were artefacts of the absolute phase of
the null vector (which is arbitrary for SVD on a complex
matrix), not genuine complex physics. A phase-fix that made
one component of :math:`\mathbf c` real-positive did not
eliminate the imaginary parts everywhere, because different
:math:`k`-branches of :math:`\sqrt{\mu_i}` (the square roots
of complex :math:`\mu`) give different complex-conjugate
pairs, and the null vector cannot lie simultaneously on all
real-conjugate pairs without further projection.

The fix — used in the implementation above — is to treat the
two sign branches of :math:`\mu_i` separately from the start:
**real** exponentials when :math:`\mu_i > 0` and **real**
:math:`\cos\,/\,\sin` pairs when :math:`\mu_i < 0`. The
matching matrix :math:`\mathbf C(k)` is then real by
construction, the null vector is real, and no phase
corrections are needed anywhere.

**Dead end #3 — Spurious sign changes from eigenvalue
reordering.**

After the real-basis rewrite the null vector gave
:math:`\phi(0), \phi(L) \sim 10^{-16}` — machine precision,
as intended. But the first full smoke test returned
:math:`k = 1.0275` as the fundamental mode, contradicting
both the Richardson cache (:math:`\sim 0.87051`) and the
earlier complex-basis prototype. Diagnostic output showed six
sign-change candidates in :math:`\det(\mathbf C(k))`:

.. code-block:: text

   k=0.1466  s[-1]/s[0]=1.19e-16  ← validated by SVD
   k=0.2302  s[-1]/s[0]=9.13e-17  ← validated by SVD
   k=0.3700  s[-1]/s[0]=1.46e-17  ← validated by SVD
   k=0.5901  s[-1]/s[0]=1.79e-17  ← validated by SVD
   k=0.8705  s[-1]/s[0]=1.25e-16  ← validated by SVD
   k=1.0275  s[-1]/s[0]=5.75e-18  ← validated by SVD

All six had smallest singular value at machine precision.
Naïvely taking the maximum gave :math:`k = 1.0275`. But
when the null vector was substituted back into the mode
basis, :math:`\phi(0)`, interface continuity, and
:math:`\phi(L)` *all* held at :math:`\sim 10^{-7}` for the
three "extra" candidates — nothing like the :math:`\sim 10^{-16}`
seen at the three genuine roots. The extras were spurious.

Root cause: :func:`numpy.linalg.eig` returns eigenvalues in
an order set by the underlying LAPACK routine
(``DGEEV``/``ZGEEV``), and the order is **not a continuous
function of** :math:`k`. When two eigenvalues cross, their
labels can swap — and because the columns of
:math:`\mathbf C(k)` are built from the labelled eigenvectors,
the column ordering permutes discontinuously at the crossing
point. Permuting columns of a square matrix flips the sign
of its determinant, so :math:`\det(\mathbf C(k))` flips
sign at every crossing — and brentq happily "finds a root"
there even though the matrix is continuously non-singular on
both sides of the crossing.

The smallest-singular-value check does not catch these:
**at** the crossing, the swapped-column matrix is
instantaneously rank-deficient in the label-permutation
sense, so :math:`s_{\min}/s_{\max}` does drop to machine
precision. But when the null vector is substituted back
into the mode basis, it does not solve the boundary-value
problem, because the labels do not correspond to a physical
mode decomposition at the crossing.

The only reliable discriminator is the **physical
validation** in :eq:`diffusion-spurious-root-validation`:
reconstruct :math:`\phi(0)`, the interface continuity, and
:math:`\phi(L)` from the null vector and check whether they
actually vanish. Genuine eigenvalues pass to machine
precision; crossings fail to :math:`\sim 10^{-7}`. This is
the load-bearing filter in :func:`_solve_2region_vacuum_eigenvalue`.

**Dead end #4 (solver-side) — Hardcoded outer-iteration
tolerance masked quadratic convergence.**

Once the reference was correct, the finite-difference solver
:func:`orpheus.diffusion.solver.solve_diffusion_1d` was run
at four mesh refinements to measure convergence order. The
expected :math:`\mathcal{O}(h^{2})` order of central finite
differences produced this embarrassing error sequence on the
bare slab:

.. code-block:: text

   dz=5.0    shape_err = 1.21e-04    order = -
   dz=2.5    shape_err = 3.19e-05    order = 1.93  ← quadratic
   dz=1.25   shape_err = 1.31e-05    order = 1.28  ← pre-plateau
   dz=0.625  shape_err = 1.01e-05    order = 0.37  ← plateau

The finest-mesh error plateaus at :math:`\sim 10^{-5}`, not
the :math:`\sim 10^{-6}` that would extend the quadratic
trend. Initial misdiagnosis: BiCGSTAB inner solver tolerance
(``errtol=1e-6``). Raising ``errtol=1e-12`` did not move the
plateau. Correct diagnosis: the outer power iteration in
:class:`DiffusionSolver` had a **hardcoded** convergence
criterion ``rel_change < 1e-5`` on the flux relative change
between outer iterations. That threshold is the floor — the
outer solve stops as soon as the flux is within :math:`10^{-5}`
of its own previous iterate, which is exactly where the
convergence tests were plateauing. The finite-difference
discretisation error was *below* the outer-iteration noise at
the finest meshes.

The fix is a two-line change to
:class:`orpheus.diffusion.solver.DiffusionSolver`: add an
``outer_tol`` keyword (default ``1e-5``, preserving legacy
behaviour) and replace the hardcoded ``< 1e-5`` with
``< self.outer_tol``. The Phase-1.2 convergence-order tests
then pass ``outer_tol=1e-11`` and see the quadratic
convergence they were looking for. General-purpose callers
see no change.

The lesson is **not** specific to this problem. Any time a
convergence-order verification test plateaus, check both the
inner and outer solver tolerances before blaming the
reference solution. Finite-difference diffusion is well
understood; if the measured order is pathological, the
solver's own convergence machinery is the first suspect, not
the reference.


.. _diffusion-2rg-verification:

Verification
============

- **Bare slab L1 eigenvalue** — the Phase-0 continuous reference
  ``dif_slab_2eg_1rg`` pulls :math:`k` from the analytical matrix
  eigenvalue and :math:`\phi_g(x)` from :eq:`bare-slab-eigenfunction`;
  the diffusion solver must reproduce both to better than
  :math:`10^{-10}` for the eigenvalue and :math:`\mathcal O(h^{2})`
  for the flux shape.
- **Fuel + reflector L1** — the continuous reference
  ``dif_slab_2eg_2rg`` produces :math:`k` from
  :eq:`diffusion-transcendental` (validated by
  :eq:`diffusion-spurious-root-validation`) and
  :math:`\phi_g(x)` from :eq:`diffusion-back-substitution` in
  the real-basis mode decomposition of
  :eq:`diffusion-mode-decomposition`,
  :eq:`diffusion-exponential-branch`, and
  :eq:`diffusion-trigonometric-branch`. Comparison against the
  solver at successive mesh refinements gives the measured order
  of the finite-difference discretisation. This replaces the
  Richardson-extrapolated reference that previously served this
  role (see the verification-campaign audit in
  :doc:`/verification/reference_solutions`).

  Note the solver's outer-iteration tolerance matters: the
  Phase-1.2 tests pass ``outer_tol=1e-11`` to push the outer
  power-iteration residual well below the finite-difference
  discretisation error at every refinement. See dead end #4 in
  :ref:`diffusion-2rg-investigation-history` for why the default
  ``outer_tol=1e-5`` masked the quadratic convergence.

Both cases live under ``operator_form="diffusion"`` in the
Phase-0 registry and are tested in
:mod:`tests.diffusion.test_continuous_reference`. The
measured numerical evidence at convergence is:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Mesh :math:`\Delta z` (cm)
     - Bare slab shape error
     - 2-region shape error
     - Bare slab :math:`\Delta k`
   * - 5.0
     - :math:`1.2 \times 10^{-4}`
     - pre-asymptotic
     - :math:`1.4 \times 10^{-3}`
   * - 2.5
     - :math:`3.2 \times 10^{-5}`
     - :math:`1.3 \times 10^{-1}`
     - :math:`3.4 \times 10^{-4}`
   * - 1.25
     - :math:`1.0 \times 10^{-5}`
     - :math:`5.6 \times 10^{-2}`
     - :math:`8.5 \times 10^{-5}`
   * - 0.625
     - :math:`2.6 \times 10^{-6}`
     - :math:`1.7 \times 10^{-2}`
     - :math:`2.1 \times 10^{-5}`
   * - 0.3125
     - :math:`6.4 \times 10^{-7}`
     - :math:`4.5 \times 10^{-3}`
     - :math:`4.6 \times 10^{-6}`

Observed convergence orders (successive ratios):
bare slab :math:`\approx 1.92, 1.99, 2.00, 2.01`;
2-region :math:`\approx 1.23, 1.73, 1.92` (the first ratio is
pre-asymptotic on the 80 cm slab, so the test uses refinement
range ``[2.5, 1.25, 0.625, 0.3125]`` and asserts the final
two ratios exceed :math:`1.8`).


References
==========

- Bell, G. I. and Glasstone, S., *Nuclear Reactor Theory*,
  Van Nostrand Reinhold, 1970. Chapter 7 covers multigroup
  diffusion theory; §7.4 specifically treats the slab
  eigenvalue problem and the two-region interface matching.
- Stacey, W. M., *Nuclear Reactor Physics*, 3rd ed., Wiley,
  2018. Ch. 3 on diffusion theory and Ch. 8 on multigroup
  formulation.
- Duderstadt, J. J. and Hamilton, L. J., *Nuclear Reactor
  Analysis*, Wiley, 1976. Ch. 5 (one-group) and Ch. 7
  (multigroup) — the transfer matrix formulation is
  spelled out explicitly.
