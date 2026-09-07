.. _sn-slab-one-group:

The slab, one group: the whole machine at its simplest
======================================================

This chapter poses and solves the complete S\ :sub:`N` problem on the
smallest phase space that exercises every part of the machine: a 1-D
Cartesian slab, one energy group, isotropic scattering. Everything the
later chapters *broaden* — energy (:math:`\to` multigroup), space
(:math:`\to` 2-D/3-D), curvature (:math:`\to` spherical/cylindrical) —
appears here once, in its simplest honest form, following the chain the
whole book repeats:

1. **the invariant** (sinks = sources on the cell) → *pose* the balance;
2. **the within-group operator** :math:`A = L + C - S - B`, and *why*
   its streaming-collision part :math:`L+C` is invertible — the
   lower-triangular structure in :term:`sweep` order;
3. **the matrix representation** — the rawest form of the invertible
   operator;
4. **the strategy-encoding operators** — the sweep (the triangular
   structure *is* forward substitution) and Krylov (avoid the matrix
   entirely), realized on the same one discretization.

The generic discretization machinery — the cell balance, the
Step/DD/LD closure schemes, the blend-weight spectrum, the
space-vs-angle unification — lives once in
:doc:`/theory/foundations/discretization` and is **cross-linked, never
re-derived**. This chapter instantiates it on the slab.

.. admonition:: Key Facts
   :class: tip

   * The slab transport equation is :eq:`transport-cartesian`; its cell
     balance is :eq:`dd-cartesian-1d` — no :math:`\alpha`
     redistribution, no :math:`\Delta A` factor, because slab geometry
     has no curvature.
   * The within-group operator equation is
     :math:`(L + C - S - B)\,\psi = q`
     (:eq:`si-within-group-operator-eq`). The sweep is
     :math:`(L+C)^{-1}` — the inner kernel of the full inverse, never
     the full inverse itself.
   * :math:`L+C` is invertible in one pass because it is
     **lower-triangular in sweep order**; the DD recurrence
     :eq:`dd-recurrence` unrolls to a vectorised cumulative product.
   * Source iteration converges geometrically at the :term:`scattering ratio`
     :math:`\rho_J = c = \Sigma_s/\Sigma_t` (:eq:`si-spectral-rate`);
     it slows arbitrarily as :math:`c \to 1` — the canonical motivation
     for Krylov and DSA.
   * The scattering :math:`c`-mode **cannot** be folded into the sweep:
     :math:`\sigma_r\,\mathbb{I} \ne \Sigma_{s,0}\,P_{\rm iso}`
     (:eq:`si-sigma-r-fold-mismatch`) — a measured 46–56 % silent error
     outside the isotropic-flux regime (issue #215).


The posing: invariant first
===========================

A math problem is posed by identifying what it holds invariant
(:doc:`/theory/foundations/discretization`, §1). For steady-state
transport the invariant is **neutron balance — sinks = sources** — on
every region of phase space. (What *all* transport methods hold
invariant, and where S\ :sub:`N` lands among them, is the corpus
root: :doc:`/theory/foundations/path_integral`.) On the slab's phase space
:math:`(x, \mu)`, with :math:`\mu = \cos\theta` the direction cosine,
the pointwise statement of that invariant is the steady-state
transport equation:

.. math::
   :label: transport-cartesian

   \mu \frac{\partial \psi(x, \mu)}{\partial x}
   + \Sigt{} \, \psi(x, \mu)
   = \frac{Q}{W}

where :math:`Q` is the total isotropic source (fission + scattering)
and :math:`W = \sum_n w_n` is the :term:`quadrature` weight sum. The two
left-hand terms are the sinks seen by the beam at :math:`(x,\mu)` —
streaming out of the volume, and removal by collision — and the
right-hand side is the source into it. There is no angular derivative:
in Cartesian geometry a neutron's direction cosine never changes along
its flight, so each :term:`ordinate` couples to the others only through the
source term. This is the structural fact the whole method exploits.

Discretizing angle on the ordinate set :math:`\{\mu_n, w_n\}` (the
Gauss–Legendre quadrature of :doc:`angular_quadrature`) makes
:eq:`transport-cartesian` a family of :math:`N` one-dimensional
advection–reaction equations, coupled only through :math:`Q`. The
semi-discrete → fully-discrete pipeline is the generic one of
:doc:`/theory/foundations/discretization` §2; the slab instantiation
follows.


The discrete balance and its closure
====================================

.. _balance-cartesian-1d:

Integrating :eq:`transport-cartesian` over a spatial cell
:math:`[x_{i-1/2}, x_{i+1/2}]` of width :math:`\Delta x_i` and applying
the divergence theorem to the streaming term:

.. math::
   :label: balance-cartesian-1d-eq

   \mu_n \bigl[\psi_{i+\frac12} - \psi_{i-\frac12}\bigr]
   + \Sigt{} \Delta x_i\, \psi_{n,i} = S_i \Delta x_i

.. (vv-status rationale) Definitional identity: the EXACT cell invariant
   (streaming + collision = source), stated before any closure — "exact, no
   approximation yet".  Not a solver claim; it has no standalone implementing
   function.  Its verifiable content is the DD-closed form it becomes,
   :eq:`dd-cartesian-1d`, which carries the slab-MMS ``verifies`` markers.
.. vv-status: balance-cartesian-1d-eq documented

where :math:`S_i = Q_i / W` and face areas are unity in slab geometry.
This is the **cell invariant** — exact, no approximation yet — and it
has the one-equation-two-unknowns shape that every discretization must
close (:doc:`/theory/foundations/discretization` §3): the cell average
:math:`\psi_{n,i}` and the downstream face are both unknown.

Applying the :term:`diamond-difference <diamond difference>` closure
:math:`\psi_{n,i} = \frac{1}{2}(\psi_{\rm in} + \psi_{\rm out})` and
:math:`\psi_{\rm out} = 2\psi_{n,i} - \psi_{\rm in}` (the
:math:`w = \tfrac12` point of the closure spectrum,
:doc:`/theory/foundations/discretization` §4), we solve for the
cell-average :term:`angular flux`:

.. math::
   :label: dd-cartesian-1d

   \psi_{n,i}
   = \frac{S_i + \dfrac{2|\mu_n|}{\Delta x_i}\, \psi_{\rm in}}
          {\Sigt{} + \dfrac{2|\mu_n|}{\Delta x_i}}

This is the simplest balance equation: no :math:`\alpha` redistribution
and no :math:`\Delta A` factor, because slab geometry has no curvature.
The streaming coefficient :math:`2|\mu|/\Delta x` is precomputed by
:class:`SNMesh` as ``streaming(0)[n, i]``.

The closure choice is a **dial, not a commitment**: Step
(:math:`w \to 1`, positivity-preserving, :math:`O(h)`), Diamond
Difference (:math:`w = \tfrac12`, :math:`O(h^2)`, can dip negative),
and Linear Discontinuous (adaptive :math:`w = 1/(1+k)`,
:math:`O(h^2)`, diffusion-limit-consistent) are one parameterized
family — derivations, truncation orders, and the Péclet-type blend
analysis live in :doc:`/theory/foundations/discretization` §4. The
production realization is the geometry-polymorphic
:class:`~orpheus.transport.spatial.diamond.DiamondDifference` strategy,
whose slab branch (:eq:`dd-slab-scalar`) is the per-cell scalar form of
the recurrence derived next.


.. _sn-streaming-operator:

The within-group operator
=========================

The one-group transport problem, posed as operators on the angular
flux, is the four-operator equation derived below
(:eq:`si-within-group-operator-eq`):

.. math::

   (L + C - S - B)\,\psi \;=\; q ,

with :math:`L` the streaming operator, :math:`C` the collision
multiplier, :math:`S` the within-group scattering gain, and :math:`B`
the boundary-reflection gain. This is the slab instance of the honest
operator algebra :math:`A = L + C - S - B` of
:doc:`/theory/foundations/operator_algebra`; with energy and fission
(next chapter) the same operator poses the eigenvalue problem
:math:`A\,\psi = \tfrac{1}{k} F\,\psi`. In ORPHEUS the equation is a
single Python expression composed from
:class:`~orpheus.numerics.operator.LinearOperator` objects; the
composers (:class:`~orpheus.numerics.operator.OperatorSum`,
:class:`~orpheus.numerics.operator.OperatorProduct`,
:class:`~orpheus.numerics.operator.ScaledOperator`) compute
:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible` /
:attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
recursively from the constituents per the closure laws (see
:ref:`operator-algebra`).

The load-bearing structural fact sits in the sub-composite
:math:`L + C`: **it is invertible in a single pass**. Order the
unknowns in the direction of neutron travel and every cell's equation
reads only its upstream face — the matrix representation of
:math:`L+C` (its rawest form; see the shape catalog in
:doc:`loss_representation` and the bridge in
:doc:`/theory/foundations/discretization` §7) is **lower-triangular
with a nonzero diagonal**. A triangular system needs no factorization:
its inverse action *is* forward substitution, and forward substitution
*is* the transport sweep of the next section. That is the entire
reason S\ :sub:`N` production codes never materialize :math:`(L+C)`:
the structure already encodes the solve strategy. The gains
:math:`S` and :math:`B` deliberately stay *outside* the inverted
composite — they are what the iteration lags (see
:ref:`si-within-group-splitting`).

The full operator surface — apply, solve, apply_transpose
---------------------------------------------------------

:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` realizes
the loss composite :math:`L + C` with all three verbs — ``apply``,
``solve``, ``apply_transpose`` — and reports both ``is_invertible`` and
``is_adjointable`` ``True`` (:ref:`capability-set-semantics`):

* :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply` —
  matrix-free forward action :math:`(L+C)\,\psi` via the operator's own
  :attr:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.loss_representation`
  through the shared apply-direction walk
  (:meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action`,
  the ``(L+C)ψ`` single emission — the apply-direction twin of
  :meth:`~orpheus.sn.loss_representation.LossRepresentation.sweep`,
  L21 "matvec ≡ sweep"; #206 Phase C).

* :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` —
  inverse action :math:`(L+C)^{-1}\,q` via the operator's own
  :attr:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.loss_representation`
  sweep (the :term:`weighted-diamond-difference <weighted diamond difference>` — WDD — forward
  substitution; the 1-D scan or multi-D wavefront selected by
  :func:`~orpheus.sn.loss_representation.default_for`).

* :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply_transpose` —
  adjoint action :math:`(L+C)^{\mathsf T}\,\varphi` via the
  loss-representation's named
  :meth:`~orpheus.sn.loss_representation.LossRepresentation.loss_action_transpose`
  (the reverse-direction walk; the multi-D Cartesian adjoint is an
  honest deferral raise, never a silent wrong answer). It gates the
  reciprocity invariant below.

``apply`` and ``solve`` run the **one** loss-representation walk —
"one walk", a code fact (L21): there is no separate matvec
discretization and no by-design bit-difference between the two
directions. (The Wave-D-era design in which ``apply`` was a distinct
finite-difference operator is retained as a historical record, with the
ERR-026 closure-bias reasoning it carries, in the streaming-collision
history section of :doc:`index`.)

Why ship :meth:`solve` at all, if Krylov can invert with ``apply``
alone? Two reasons. First, the sweep's
:math:`O(N\cdot N_{\rm cells})` forward substitution is the canonical
S\ :sub:`N` preconditioner (:cite:`AdamsLarsen2002` review; Lewis & Miller
§4.5) — exposing ``solve`` keeps that path discoverable through the
operator surface. Second, the composers need a uniform contract: when
a downstream consumer composes the full operator, its
``is_invertible`` is derived recursively from each operand, and the
Krylov path requests a sweep-preconditioned matvec *through* the
algebra rather than around it.

Reciprocity invariant
---------------------

The reciprocity identity is the defining property of the
operator-transpose pairing under the discrete L\ :sup:`2` inner
product:

.. math::
    :label: sn-streaming-reciprocity

    \langle A\,\psi,\,\varphi\rangle \;=\;
    \langle\psi,\,A^*\,\varphi\rangle

.. (vv-status rationale) Mathematical identity: reciprocity is the DEFINING
   property of the operator-transpose pairing under the discrete L² inner
   product, not a solver claim.  Its verifiable content is the foundation
   reciprocity gate ⟨(L+C)ψ,φ⟩=⟨ψ,(L+C)ᵀφ⟩
   (``tests/sn/sweep/core/test_phase_c_gates.py`` Gate 1.3, ``@pytest.mark.foundation``)
   plus the linearity gate it rides on
   (``test_streaming_operator.py::TestLinearity.test_apply_is_linear``);
   the gates are unwired, so the label stays ``documented`` with the
   gates named here.
.. vv-status: sn-streaming-reciprocity documented

for any pair :math:`(\psi, \varphi)` in the discrete unknown space.
Per Lewis & Miller §10 (adjoint transport), this identity links
forward and adjoint sources / fluxes; it is the foundation on which
detector sensitivity, perturbation theory, and adjoint-weighted
kinetics all rest.

:meth:`apply_transpose` is inherited from
:class:`~orpheus.numerics.operator.OperatorSum`'s adjoint-propagation
closure law: each leaf's ``.H`` adjoint composes via the
sum/difference algebra, so the composite transpose is built
analytically — no dense-matrix probing. (The pre-Depth-B
implementation assembled a dense matrix by probing :meth:`apply` with
unit basis vectors and returned the explicit transpose; that path
retired with the bundled ``SNStreamingOperator`` class in D-K.)

Reciprocity gating today: the foundation linearity gate
:func:`tests.sn.operators.test_streaming_operator.TestLinearity.test_apply_is_linear`
catches non-linearity in :meth:`apply`, and the Resolution A bit-exact
decomposition gate
:file:`tests/sn/test_streaming_operator_decomposition.py` catches
:math:`(L+C).{\rm apply} \neq M(\psi;\sigma_t)` drift.

The typed carrier
-----------------

:meth:`apply`, :meth:`apply_transpose`, and :meth:`solve` all operate
on the **same** typed composite carrier
:class:`~orpheus.transport.full_field.FullField` (bulk
:class:`~orpheus.transport.fields.angular_flux.AngularFlux` +
boundary
:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`),
in the principled ``(N, ng, nx, ny)`` layout
(see :ref:`theory-sn-index-convention`):

* **Source** — carried as the composite bulk
  (``rhs.interior.values``, per-ordinate shape ``(N, ng, nx, ny)``);
  the P\ :sub:`ℓ` (:math:`\ell\ge 1`) anisotropic contribution is
  folded into this one per-ordinate source.
* **Boundary state** — carried as the typed
  :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
  face views on ``rhs.boundary`` (keyed by face name); the sweep seeds
  its mutable boundary buffer from these trace slots and persists
  reflective/pole state through them between outer iterations.

.. note:: **Superseded packed-vector layout (Wave D / early Wave E).**
   The previous design gave :meth:`apply` / :meth:`apply_transpose` a
   **packed 1-D solution vector** (an ``EquationMap`` selecting the
   unknown ``(ordinate, cell)`` combinations) that differed from
   :meth:`solve`'s structured arrays, and ``solve`` consumed a separate
   ``Q`` / ``psi_bc`` dict / optional ``Q_aniso`` triple. The
   typed-field campaign (#197, then Depth-B D-H/D-I/D-J) retired the
   packed-vector convention, the ``EquationMap`` codec family, and the
   ``psi_bc`` dict in favour of the single
   :class:`~orpheus.transport.full_field.FullField` composite; the
   ``Q_aniso`` kwarg folded into the one per-ordinate source. There is
   no longer a layout difference between :meth:`apply` and
   :meth:`solve`.


The sweep: forward substitution as a strategy object
====================================================

Because each cell's outgoing flux becomes the next cell's incoming
flux, the equations must be solved in the direction of neutron travel
— this is the **transport sweep**, and it is nothing more than
forward substitution on the lower-triangular :math:`L+C` of the
previous section, executed without ever materializing the matrix. The
strategy has a name in the operator algebra —
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator`, the
schedule-triangular member of the direct inverse family
(:doc:`/theory/foundations/operator_inverse_family`) — and a
vectorised realization on the slab, derived next. Boundary traces
(what seeds the first cell, what the last cell emits) are the
:doc:`boundary_conditions` chapter's story.

.. _sweep-cumprod:

Cartesian 1D: cumprod recurrence
--------------------------------

For the 1D slab with Gauss–Legendre quadrature, the DD equation
:eq:`dd-cartesian-1d` defines a recurrence for the outgoing face flux:

.. math::
   :label: dd-recurrence

   \psi_{\rm out} = a_i\, \psi_{\rm in} + b_i


.. implements:: dd-recurrence
   :by: orpheus.sn.loss_representation.CumprodScan

   **Implemented by** 5 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: dd-recurrence
   :by: orpheus.sn.sweep.scan.ordinate_scan

.. implements:: dd-recurrence
   :by: orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients

.. implements:: dd-recurrence
   :by: orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients

.. implements:: dd-recurrence
   :by: orpheus.derivations.discrete.sn.balance.derive_cumprod_recurrence

where the coefficients for cell :math:`i` are:

.. math::
   :label: dd-recurrence-coefficients

   a_i = \frac{2|\mu_n|/\Delta x_i - \Sigt{}}
              {2|\mu_n|/\Delta x_i + \Sigt{}},
   \qquad
   b_i = \frac{S_i}
              {2|\mu_n|/\Delta x_i + \Sigt{}}

This arises from substituting the DD closure
:math:`\psi_{\rm out} = 2\psi_{\rm avg} - \psi_{\rm in}` into
:eq:`dd-cartesian-1d`.  The coefficient :math:`a_i` is the
**stream-to-collision ratio**: it controls how much incoming flux
propagates through cell :math:`i`.

Unrolling the recurrence :math:`\psi_{\rm out}^{(i)} = a_i\, \psi_{\rm out}^{(i-1)} + b_i`
gives a linear first-order relation that can be solved analytically
using **cumulative products**.  Define:

.. math::
   :label: sweep-cumprod-factors

   C_i = \prod_{k=0}^{i} a_k, \qquad
   R_i = \sum_{k=0}^{i} \frac{b_k}{C_k}

.. (vv-status rationale) Derivation step: the cumulative-product / cumulative-
   sum factors are an intermediate in the analytic closed-form solution of the
   DD recurrence :eq:`dd-recurrence`.  Not a standalone solver claim; the
   terminal result — the recurrence itself — is pinned against the symbolic
   derivation by
   ``tests/sn/sweep/slab/test_dd_recurrence.py::test_dd_per_cell_recurrence_matches_symbolic_derivation``.
.. vv-status: sweep-cumprod-factors documented

Then the incoming face flux at cell :math:`i+1` is:

.. math::
   :label: sweep-cumprod-solution

   \psi_{\rm in}^{(i+1)} = C_i \bigl(\psi_{\rm in}^{(0)} + R_i\bigr)

.. (vv-status rationale) Derivation step: the closed-form (cumprod) solution of
   the DD recurrence :eq:`dd-recurrence`, built from the factors above.  Not a
   standalone solver claim; the terminal recurrence it solves is pinned by
   ``tests/sn/sweep/slab/test_dd_recurrence.py::test_dd_per_cell_recurrence_matches_symbolic_derivation``,
   and the sweep it realises is exercised by the slab MMS / regression suites.
.. vv-status: sweep-cumprod-solution documented

and the cell-average flux is :math:`\psi_{\rm avg}^{(i)} = \frac{1}{2}(\psi_{\rm in}^{(i)} + \psi_{\rm out}^{(i)})`.

The implementation in
:meth:`~orpheus.sn.loss_representation.CumprodScan.sweep` (the
free-function ``_sweep_1d_cumprod`` of the dissolved ``sweep.py``)
computes :math:`C` and
:math:`R` via ``np.cumprod`` and ``np.cumsum``, giving an
:math:`O(N \cdot n_x)` **vectorised** sweep --- all spatial cells for a
given ordinate are resolved simultaneously in numpy array operations,
with no Python-level cell loop.  This typically runs in sub-millisecond
time for practical meshes.

Exploiting GL symmetry, only positive-:math:`\mu` ordinates are swept
forward; negative-:math:`\mu` ordinates are obtained by reversing the
cell array and sweeping with the same coefficients.

.. _sn-affine-outgoing-face-reconstruction:

Generic affine outflow reconstruction
--------------------------------------

.. math::
   :label: sn-affine-outgoing-face-reconstruction-eq

   \psi_{\rm out} = \frac{\bar\psi - (1-w)\,\psi_{\rm in}}{w}

.. (vv-status rationale) Algebraic reduction invariant: the single-source
   downstream-face reconstruction, the exact algebraic inverse of the convex
   cell-average blend.  Its verifiable content — the exact-inverse round-trip,
   the DD w=½ byte-identity, the LD w=1/(1+k) algebraic equality — is the
   ``@pytest.mark.foundation`` unit gate
   :mod:`tests.transport.spatial.test_affine_closure`; foundation software-
   invariant tests carry no ``verifies(...)`` by design.
.. vv-status: sn-affine-outgoing-face-reconstruction-eq documented

The single-source inverse of the convex cell-average blend
:math:`\bar\psi = (1-w)\psi_{\rm in} + w\,\psi_{\rm out}`
(:eq:`dd-recurrence` closure :math:`\psi_{\rm out} = 2\bar\psi - \psi_{\rm in}`
is the :math:`w=\tfrac12` case).  Every consistent affine spatial scheme
reconstructs its downstream face from this one parameterized formula:
Diamond Difference at :math:`w=\tfrac12` (diamond mean), Linear
Discontinuous at its optical-thickness-adaptive
:math:`w = 1/(1+k)` — the blend-weight spectrum and its Péclet-type
adaptivity are derived once in
:doc:`/theory/foundations/discretization` §4 (eq
``discretization-ld-blend``).  At :math:`w=\tfrac12`
the reconstruction is byte-identical to the inlined :math:`2\bar\psi - \psi_{\rm in}`
(division by :math:`\tfrac12` is the exact power-of-two doubling, which commutes
with round-to-nearest); LD's :math:`w=1/(1+k)` is algebraically equal to its
inlined Schur form :math:`\bar\psi + (|\mu|/\theta)(\bar\psi - \psi_{\rm in})/D_2`
but takes a different floating-point reduction tree (a principled
:math:`\sim`\ 1-ULP re-baseline).

The one parameterized formula above is the **single source** of the
downstream-face reconstruction for *every* affine 1-D spatial scheme.  It is
homed (with its forward partner, the cell-average blend
:math:`\bar\psi = (1-w)\psi_{\rm in} + w\,\psi_{\rm out}`, and the source
emission) as a ``@staticmethod`` on the
:class:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase`, so the per-scheme
classes (:class:`~orpheus.transport.spatial.diamond.DiamondDifference`,
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`, Step)
carry NO inlined reconstruction of their own — they differ only by the value
they pass for the blend weight :math:`w`.  The #240 Phase 2 Step D1 carve
collapsed the previously-duplicated inline forms (Diamond Difference's
:math:`2\bar\psi - \psi_{\rm in}`, Linear Discontinuous's
:math:`(1+k)\bar\psi - k\,\psi_{\rm in}`) onto this one op
(:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.outgoing_face_from_average`),
the algebraic inverse of
:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cell_average`; Step D2
made the trio (``source_emission`` / ``cell_average`` /
``outgoing_face_from_average``) generic advection–reaction reconstructions
(diffusion-consumable, retiring the dangling ``affine_closure`` module).  The
unit gate is :mod:`tests.transport.spatial.test_affine_closure`: the exact-inverse
round-trip :math:`\bar\psi(\,\psi_{\rm in}, \psi_{\rm out}(\psi_{\rm in}, \bar\psi)\,) = \bar\psi`,
the DD :math:`w = \tfrac12` byte-identity, and the LD :math:`w = 1/(1+k)`
algebraic equality.

.. note::

   **The spatial closure factors out of the angular index, and the multi-D
   extension factors out of the dimension.**  The reconstruction op above is
   stated per ordinate per axis, so it is a *spectator* to the angular moment
   axis (the angular reduction :eq:`two-moment-angular` rides over it, see
   :ref:`two-moment-axes`) — the same op serves a P0 and a P3 calculation
   unchanged.  In the same spirit, the multi-dimensional LD closure
   (:ref:`ld-ubld-multidim`) is the **tensor product** of this 1-D
   per-axis reconstruction across :math:`d` axes: the per-cell
   :math:`2^d \times 2^d` operator is assembled as a Kronecker product of the
   verified 1-D factor operators, so the affine 1-D closure documented here is
   the literal :math:`d=1` building block of the :math:`d`-generic UBLD system
   :eq:`ld-ubld-cell-system`.  Spatial scheme :math:`\otimes` angular order
   :math:`\otimes` dimension: three orthogonal axes of choice, each a tensor
   factor, none special-casing the others.


P\ :sub:`0` scattering: the isotropic projection
================================================

The default mode (``scattering_order=0``). A direction-independent
source is added to all ordinates equally; in the one-group problem the
within-group scattering source is

.. math::
   :label: p0-scatter-source

   Q_{\rm scatter}(\hat{\Omega}_n) = \Sigma_{s,0}\, \phi / W ,
   \qquad \phi = \sum_n w_n\, \psi_n .

(In the multigroup extension the coefficient becomes the in-scatter
sum :math:`\sum_{g'} \Sigs{g'\to g}^{(0)}\, \phi_{g'}` — next chapter;
the production hook is the array verb
:meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`,
which performs ``phi @ SigS[0]`` per material — reached through the
:math:`\ell = 0` energy binding the collision gain lifts.  A thin
``SNSolver._add_scattering_source`` delegator stood in front of it until
#448.)

As an operator this is :math:`S = \Sigma_{s,0}\, P_{\rm iso}` with
:math:`P_{\rm iso}\,\psi = \phi / \sum_n w_n` — a **rank-1-in-angle
projection**, not a diagonal: it couples every ordinate to every other
through the :term:`scalar flux`. In the intrinsic-type partition of
:doc:`/theory/foundations/operator_algebra`, :math:`S` is a *kernel*
(nonlocal in angle) where :math:`C` is a *local multiplication
operator* — and that type difference is precisely what makes the
:math:`\sigma_r`-fold below illegal. Anisotropy (the P\ :sub:`N`
moment expansion, :ref:`pn-scattering`) broadens this operator in the
multigroup chapter.


.. _si-within-group-splitting:

Source iteration and its alternatives
=====================================

With the operators posed, the solve strategies follow from the
structure. :class:`~orpheus.numerics.iteration.SourceIteration` (in
:mod:`orpheus.numerics.iteration`, a stand-alone operator-algebra
consumer with no transport-solver knowledge beyond the
:class:`~orpheus.numerics.operator.LinearOperator` contract) realizes
the classical fixed-point iteration; Krylov realizes the
rate-optimal alternative on the same one discretization. This section
derives the source-iteration spectral radius :math:`\rho_J = c` from
the within-group operator splitting and gives the
iterations-to-tolerance estimate the rate implies. (The multi-D
boundary Gauss-Seidel schedule, its honest scope, and the
diagonal-cubature shared-face rule ERR-056 are documented with the
multi-D machinery in :doc:`cartesian_multid`; the reified splitting matrix is
:ref:`si-gauss-seidel-reification`.)

.. admonition:: Key Facts (SI rate)
   :class: tip

   * The within-group SI iteration matrix is
     :math:`(L+C)^{-1}(S+B)`; its spectral radius is the scattering
     ratio :math:`\rho_J = c = \max_g \Sigma_{s,g}/\Sigma_{t,g}`
     (:eq:`si-spectral-rate`).  Iterations to relative tolerance
     :math:`\varepsilon`: :math:`n_{\rm Jacobi} \approx
     \log\varepsilon / \log c`.
   * **Boundary Gauss-Seidel** (Phase 3, ``inner_schedule=
     "gauss_seidel"``, default) folds **only** the boundary
     reflection :math:`B` into the resolvent
     (:math:`(L+C-B_{\rm lower})^{-1}` forward substitution).  It
     touches the *boundary-layer transient* only, NOT the dominant flat
     scattering :math:`c`-mode.  **This is NOT the textbook
     scattering-G-S** :math:`c^2`-halving.  ⛔ this bullet claimed a
     *"measured, regime-independent* **~0.86–0.92×**\ *"* until
     2026-08-08; that is REFUTED — with any leakage it is a wash (0.97)
     at every dimension, and at zero leakage it ranges from a 2.5× win
     to a 5.3× loss.  ⛔ a follow-on wording *"a win at d=2, a loss at
     d=3"* is ALSO refuted (2026-08-09, #341): ``ndim`` is not the
     discriminating variable, and **the splitting is not a** *regular*
     **splitting**, so no comparison theorem bounds the two rates in
     either direction.  Multi-D is not this page's subject; the measured
     table, the counterexamples and the structural obstruction live at
     :ref:`sn-boundary-gs-rate-regime` and
     :ref:`sn-boundary-gs-not-regular` in
     :doc:`/theory/methods/sn/cartesian_multid`.
   * The dominant within-group scattering rate is recovered ONLY by
     **Krylov** (already production — rate-optimal,
     splitting-invariant) or by **consistent DSA** (future, GitHub
     issue #2).  The scattering :math:`c`-mode **cannot** be folded
     into the directional sweep (the σ\ :sub:`r`-fold trap, issue
     #215).
   * The converged fixed point is **invariant** under the schedule
     (any consistent splitting of :math:`(L+C-S-B)\psi=q` shares
     :math:`\psi^\ast`); only the SI spectral rate changes.  Krylov
     ignores the schedule entirely.  ⚠ That holds here because a slab
     is **kernel-free** — ``[M]`` :math:`\dim\ker A = 0` at
     :math:`d=1`, where no zero-mean face mode exists.  At
     :math:`d \ge 2` with :math:`\ge 2` reflective axis pairs
     :math:`A` is *singular* and the splittings share a solution
     **set**, not a point (:ref:`sn-loss-kernel-gauge`).
   * 1-D meshes always fall back to Jacobi (the 1-D scan is not a
     wavefront; the regime is scattering-dominated — boundary G-S is
     a no-op).

The four-operator within-group equation
---------------------------------------

Within a single energy group, the steady transport equation factors
into four operators acting on the angular flux :math:`\psi`:

.. math::
   :label: si-within-group-operator-eq

   (L + C - S - B)\,\psi \;=\; q,

.. (vv-status rationale) Governing equation: the honest four-operator posing of
   the within-group problem, A = L+C−S−B.  Definitional — it names the operator
   algebra, with no single implementing function distinct from the solver
   itself.  Its constituents are individually verified (streaming decomposition
   gate, the ``ScatteringOperator`` / ``SNBoundaryOperator`` tests) and the SI
   splitting it factors into is exercised by the slab convergence suite.
.. vv-status: si-within-group-operator-eq documented

where

* :math:`L = \hat\Omega\cdot\nabla` is **streaming** (the sweep's
  spatial derivative — see :ref:`sn-streaming-operator`);
* :math:`C = \Sigma_t\,\mathbb{I}` is the **collision** (total
  removal) operator, diagonal in angle;
* :math:`S = \Sigma_{s,0}\,P_{\rm iso}` is the **within-group
  scattering** gain — it couples back through the scalar flux
  :math:`\phi = \int\psi\,d\Omega`, so :math:`P_{\rm iso}\psi =
  \phi/\!\sum_n\mathcal{W}_n` is the isotropic-projection (rank-1
  in angle) operator (the convention used by
  :class:`~orpheus.transport.operators.scattering.ScatteringOperator`; higher
  Legendre orders add the :math:`P_\ell` blocks);
* :math:`B` is the **boundary reflection** gain — trace-only,
  realised by :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`,
  delivering :math:`\psi.\text{inflow} = B\,\psi.\text{outflow}` on
  specular faces (see :ref:`bc-extraction` in
  :doc:`/theory/foundations/boundary_conditions`);
* :math:`q` is the external/fission within-group source
  (:eq:`phase-f-q-1d-decomposition`).

Source iteration is a **splitting** of :eq:`si-within-group-operator-eq`:
the **streaming-collision** part :math:`(L+C)` is kept on the LHS
(inverted exactly by **one WDD sweep** — a triangular
forward-substitution, since the sweep visits cells in causal order),
while the **scattering** :math:`S` and the **boundary reflection**
:math:`B` are *lagged* on the RHS, evaluated from the previous
iterate :math:`\psi_n`:

.. math::
   :label: si-jacobi-fixed-point

   \psi_{n+1} \;=\; (L+C)^{-1}\bigl(S\,\psi_n
                    \;+\; B\,\psi_n \;+\; q\bigr).

.. (vv-status rationale) Governing iteration: the source-iteration splitting of
   the within-group operator (lag S and B, invert L+C exactly).  A definitional
   iteration, not a per-term solver claim.  Its convergence to the correct
   fixed point and rate ρ_J=c are pinned by the L1 closed-form anchor
   ``tests/sn/verification/analytical/test_si_convergence_rate.py``.
.. vv-status: si-jacobi-fixed-point documented

The iteration matrix is therefore :math:`M = (L+C)^{-1}(S+B)`, and
the asymptotic convergence rate is :math:`\rho(M)`. The convergence
test is the relative L2 residual on the iterate — the same metric
:meth:`SNSolver._solve_source_iteration` uses, since that
within-group inner consumes this primitive directly (via the
:func:`~orpheus.sn.coupled_system.build_within_group_system`
single-source-of-truth builder):

.. math::
    :label: si-convergence-residual

    {\rm res}_n \;=\; \frac{\|\psi_n - \psi_{n-1}\|_2}
                            {\max(\|\psi_n\|_2,\,10^{-30})}

.. (vv-status rationale) Notation definition: the relative-L² convergence
   metric of the source iteration (the 10⁻³⁰ floor guards the first-iterate
   divide).  It defines the stopping test, not a physics claim; its downstream
   effect (SI drives the residual below tol at the rate ρ_J=c) is pinned by
   ``tests/sn/verification/analytical/test_si_convergence_rate.py``.
.. vv-status: si-convergence-residual documented

with the iteration breaking when :math:`{\rm res}_n < {\rm tol}`.

Spectral radius :math:`\rho_J = c`
----------------------------------

For an infinite homogeneous medium with isotropic scattering, the
boundary term vanishes (:math:`B=0`) and the streaming derivative
:math:`L` drops in the flat-flux Fourier mode :math:`k\to0`.  The
spatial operator :math:`(L+C)^{-1}` then reduces to multiplication by
:math:`1/\Sigma_t`, and the isotropic-scattering gain :math:`S`
contributes :math:`\Sigma_{s,0}` per collision.  The dominant
eigenvalue of :math:`(L+C)^{-1}S` is thus the **scattering ratio**:

.. math::
   :label: si-spectral-rate

   \rho_J \;=\; \rho\!\bigl((L+C)^{-1}(S+B)\bigr) \;=\;
   c \;\equiv\; \max_g \frac{\Sigma_{s,g}}{\Sigma_{t,g}},
   \qquad
   n_{\rm Jacobi} \;\approx\; \frac{\log\varepsilon}{\log c}


.. implements:: si-spectral-rate
   :by: orpheus.data.macro_xs.mixture.Mixture.scattering_ratio

   **Implemented by** 5 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: si-spectral-rate
   :by: orpheus.numerics.convergence.StoppingCriterion.projected_iterations

.. implements:: si-spectral-rate
   :by: orpheus.numerics.convergence.StoppingCriterion.rate

.. implements:: si-spectral-rate
   :by: orpheus.numerics.convergence._budget_from_law

.. implements:: si-spectral-rate
   :by: orpheus.numerics.convergence.default_iteration_budget

(the Fourier / mode analysis of Lewis & Miller §4.4, Adams & Larsen
2002 §II).  The right-hand identity gives the iterations
:math:`n_{\rm Jacobi}` needed to drive the relative residual to a
tolerance :math:`\varepsilon`: each iteration multiplies the error
by :math:`c`, so :math:`c^{\,n} = \varepsilon` solves to
:math:`n = \log\varepsilon/\log c`.  Because :math:`c\to1` for a
nearly-pure scatterer, source iteration becomes arbitrarily slow as
:math:`c\to1` — the canonical motivation for acceleration (DSA,
Krylov).

.. note::

   The :math:`c` in :eq:`si-spectral-rate` is the **within-group
   scattering ratio** :math:`\Sigma_{s,0}^{g\to g}/\Sigma_{t,g}` that
   governs the *within-group* fixed point.  The
   :meth:`Mixture.scattering_ratio <orpheus.data.macro_xs.mixture.Mixture.scattering_ratio>`
   property exposes the slightly larger **Case–Zweifel** secondaries-
   per-collision parameter :math:`c_g = (\Sigma_{s,g} +
   \nu\Sigma_{f,g})/\Sigma_{t,g}` (it folds in fission emission for a
   multiplying medium).  The L1 rate anchor
   :func:`tests.sn.verification.analytical.test_si_convergence_rate.test_si_jacobi_rate_matches_scattering_ratio`
   pins :math:`n_{\rm Jacobi}` against ``log(tol)/log(c_max)`` using
   the Case–Zweifel form and accepts a 0.6–1.2 band: the measured
   B-2g slab count was **655** against a predicted
   :math:`\log(10^{-8})/\log(0.975) = 728` (ratio **0.90** — the
   gap is the finite-slab leakage that lowers the effective rate
   below the infinite-medium :math:`c`, plus the multigroup
   coupling).  This is the structurally-independent target the
   recovery improves upon, NOT another ORPHEUS solver
   (``vv-principles`` structural-independence; MMS is **not** paired
   here because MMS does not prove rates against an eigenvalue —
   the rate is a closed-form property of the cross sections).

Why the scattering :math:`c`-mode cannot be folded into the sweep
-----------------------------------------------------------------

It is tempting to try to fold the within-group self-scatter
:math:`\Sigma_{s,0}^{g\to g}` *into* the sweep — to accelerate the
:math:`c`-mode by absorbing the self-scatter into the cell-balance
denominator as a **removal cross-section** :math:`\sigma_r =
\Sigma_t - \Sigma_{s,0}^{g\to g}`, then iterating only on the
residual scattering :math:`\psi_{n+1} = A_{\rm wg}^{-1}(S_{\rm
residual}\psi_n + q)` with a :math:`\sigma_r`-sweep as :math:`A_{\rm
wg}^{-1}`.  **This is a latent correctness trap** (GitHub issue
#215, measured 2026-06-05), and documenting *why* it fails prevents
a future session from re-attempting it.

The σ\ :sub:`r`-sweep inverts :math:`(\hat\Omega\cdot\nabla +
\sigma_r\,\mathbb{I})` — a removal that is **diagonal in angle**.
But the within-group self-scatter is :math:`S_{\rm foldable} =
\Sigma_{s,0}\,P_{\rm iso}` — the **isotropic-projection** operator
(rank-1 in angle, :math:`\phi/\!\sum_n\mathcal{W}_n`).  The two
operators **coincide only for isotropic flux**:

.. math::
   :label: si-sigma-r-fold-mismatch

   \underbrace{\sigma_r\,\mathbb{I}}_{\text{diagonal in angle}}
   \;\ne\;
   \underbrace{\Sigma_{s,0}\,P_{\rm iso}}_{\text{isotropic projection}}
   \qquad\text{unless }\psi\text{ is isotropic, i.e. }
   \psi_n = \tfrac{\phi}{\sum_n\mathcal{W}_n}\ \forall n.

.. (vv-status rationale) Structural identity: the operator-type inequality
   (diagonal-in-angle σ_r𝕀 ≠ rank-1 isotropic projection Σ_s0·P_iso) that
   documents WHY the σ_r-fold is illegal (issue #215).  There is no code to
   test — the fold is deliberately NOT implemented; this label preserves the
   falsification so a future session does not re-attempt it.  The correct
   handling (a consistent DSA operator) is the tracked alternative (issue #2).
.. vv-status: si-sigma-r-fold-mismatch documented

The consequence is a verification-mode-2 (variable-swap / operator
mismatch) bug that the *standard* test regime cannot see:

.. list-table:: σ\ :sub:`r`-fold failure across the BC regime (issue #215)
   :header-rows: 1
   :widths: 30 22 48

   * - Variant
     - Result
     - Why
   * - σ\ :sub:`r`-sweep approximation, **fully-reflective uniform**
       box
     - **exact** (round-off)
     - Flux is isotropic ⟹
       :math:`\sigma_r\mathbb{I}\equiv\Sigma_{s,0}P_{\rm iso}`.  The
       isotropic unit tests pass.
   * - σ\ :sub:`r`-sweep approximation, **anisotropic**
       (vacuum / heterogeneous)
     - **46–56 % flux error**
     - Flux is anisotropic ⟹ the diagonal removal is the wrong
       operator; the error is silent (no crash) and corrupts real
       cases.
   * - "exact" variant (keep the :math:`-\Sigma_{s,0}\!\odot\!\psi`
       remnant on the RHS)
     - **DIVERGES**
     - The remnant gain has spectral radius
       :math:`\Sigma_{s,0}/\sigma_r \approx 39` — the splitting is
       unstable.

This is the textbook reason **DSA needs a *consistent* diffusion
operator**: the correct synthetic acceleration of the
isotropic-projection self-scatter is a diffusion solve whose removal
matches the transport operator's low-order limit, not a directional
sweep with a doctored denominator.  The
:meth:`ScatteringOperator.foldable_part <orpheus.transport.operators.scattering.ScatteringOperator.foldable_part>`
/ :meth:`residual_part <orpheus.transport.operators.scattering.ScatteringOperator.residual_part>`
split (the data API landed under Issue #197 PR-TYPED-1) produces
:math:`\Sigma_{s,0}^{g\to g}` precisely as the input a DSA
preconditioner consumes (the diffusion removal coefficient) — it is
the right input **for DSA**, NOT for a sweep fold.  Any future
within-group accelerator MUST be gated on an **anisotropic** config;
the isotropic box cannot see this error (``vv-principles``
anti-pattern #4: homogeneous/isotropic verification is blind to the
angular structure).

That "future within-group accelerator" has since landed: the consistent
DSA low-order build is the **first production consumer** of the foldable
accessors, and it is legitimate there precisely because a low-order
operator is correction\ :math:`\to 0`-safe by construction (a wrong
low-order degrades the rate, never the fixed point) — so folding
:math:`\sigma_r` onto the *low-order* removal diagonal is safe, while
the identical fold onto the *sweep* is the ERR-070 fixed-point bug this
table exhibits.  The full story — the 43 % measured shift, the D10
routing sentinel that fences the accessors to their three legitimate
consumers, and the correction\ :math:`\to 0` partition that makes the
distinction rigorous — is in
:ref:`sn-dsa-three-discoveries` of :doc:`acceleration`.

The Krylov alternative: same walk, different strategy
-----------------------------------------------------

Instead of lagging the gains, the within-group problem can be handed
to a Krylov method: GMRES on
:meth:`StreamingCollisionOperator.apply <orpheus.sn.operators.streaming.StreamingCollisionOperator.apply>`
with the sweep wrapped as a left preconditioner — the SAILOR /
Larsen–Adams preconditioned-Krylov framework (:cite:`AdamsLarsen2002`
§III). The preconditioned iteration converges at the rate of the
*preconditioned* spectrum rather than the raw scattering ratio, so it
recovers the :math:`c`-mode that source iteration cannot; it is
rate-optimal and splitting-invariant (the Key Facts above).

Because ``apply`` and ``solve`` run the **one** loss-representation
walk (L21, :ref:`sn-streaming-operator`), the operator Krylov inverts
and the sweep that preconditions it are the *same discretization by
construction* — there is no second operator to drift, and the
converged Krylov solution is the same fixed point source iteration
converges to. (In the Wave-D/E era these were two distinct
discretizations — an upwind-FD matvec against a WDD sweep — whose
coarse-mesh discrepancy and curvilinear closure bias are recorded in
the streaming-collision history section of :doc:`index`; the #206
Phase C matvec ≡ sweep unification dissolved the split.)


Verification hooks
==================

The slab one-group machine is pinned by the slab slice of
:doc:`/theory/verification/sn`:

* **Balance / closure**: the DD balance :eq:`dd-cartesian-1d` and
  recurrence :eq:`dd-recurrence` carry ``verifies`` markers from the
  slab MMS ladder (``tests/sn/verification/mms/test_mms.py``,
  ``test_mms_ld_slab.py``) and the recurrence unit gate
  (``tests/sn/sweep/slab/test_dd_recurrence.py``).
* **Rate**: the SI spectral rate :eq:`si-spectral-rate` is pinned by
  the L1 closed-form anchor
  ``tests/sn/verification/analytical/test_si_convergence_rate.py``
  (structurally independent: the rate is a property of the cross
  sections, not of another solver).
* **Order vs implementation**: a scheme's convergence order is a
  *theoretical* property proven analytically in
  :doc:`/theory/foundations/discretization`; the MMS tables *recover*
  that order numerically to verify the implementation. The two claims
  are kept distinct (order-proof there, order-recovery in
  :doc:`/theory/verification/sn`).

What broadens next
==================

Each of the following chapters relaxes exactly one restriction of
this one, reusing everything else:

* **Energy** (multigroup): the scattering operator gains the
  group-to-group matrix and the P\ :sub:`N` anisotropy expansion;
  fission :math:`F` enters and poses the :math:`k`-eigenvalue.
* **Space** (2-D/3-D Cartesian): streaming becomes a true gradient,
  the sweep becomes a wavefront over a dependency DAG, and the LD
  closure tensor-products across axes
  (:ref:`ld-ubld-multidim`).
* **Curvature** (spherical/cylindrical): the angular cell balance
  activates — the same closure machinery of
  :doc:`/theory/foundations/discretization` §5, applied on the angular
  axis — bringing redistribution, the :math:`\alpha` dome, and the
  starting-direction state.
