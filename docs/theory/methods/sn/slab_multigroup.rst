.. _sn-slab-multigroup:

The slab, multigroup: energy enters the algebra
===============================================

This chapter broadens exactly one axis of :doc:`slab_one_group`:
**energy**. The phase space gains a group index, and with it the three
things one group could not express: group-to-group scattering transfer
(including the P\ :sub:`N` anisotropy expansion), fission with its
emission spectrum :math:`\chi`, and the :math:`k`-eigenvalue problem
that fission poses. Everything else — the mesh, the :term:`quadrature`, the
cell balance and its closure, the :term:`sweep`, source iteration and its
Krylov alternative — is **reused unchanged**: the group axis couples
the transport equations *only through their sources*, so each group's
within-group problem is precisely the machine of the previous chapter.

The chain of the book (:doc:`slab_one_group`, intro) repeats on the
new axis:

1. **the invariant** — sinks = sources, now per (cell × :term:`ordinate` ×
   group) → *pose* the multigroup balance;
2. **the operators** — scattering becomes a group-coupling kernel, the
   :math:`(n,2n)` emission the *same* kernel with yield :math:`y = 2`
   over its own channel (:ref:`n2n-reactions`), fission a
   rank-1-in-energy emitter; the within-group operator
   :math:`A = L + C - S - N_{2n} - B`
   (:eq:`sn-within-group-with-n2n`) keeps its shape — the
   :math:`(n,2n)` term contributes nothing to this chapter's *derivation*
   fixtures (``[M]`` 2026-09-07, all 12 ``xs_library`` mixtures carry a
   :math:`\Sigma_{2n}` with zero non-zeros), so writing it out moves no
   number here; the numbers that *do* need a live channel are injected
   deliberately and live at :ref:`n2n-reactions` — and its
   streaming-collision part :math:`L+C` stays group-diagonal —
   invertible per group by the same one-pass sweep;
3. **the eigenvalue posing** — criticality as the generalized
   eigenproblem :math:`A\,\psi = \tfrac{1}{k}\,F\,\psi`, i.e. the
   dominant eigenpair of the multiplication operator
   :math:`K = A^{-1} F`;
4. **the strategy-encoding operators** — power iteration as the outer
   loop over the inner resolvent realisation the previous chapter
   built.

.. admonition:: Key Facts
   :class: tip

   * The multigroup transport equation is :eq:`multigroup`. The group
     axis couples **only through the sources** — streaming and
     collision are group-diagonal — so the sweep, closure, and inner
     iteration of :doc:`slab_one_group` are reused per group
     unchanged.
   * Cross-section storage is
     :math:`\text{SigS}[g_{\rm from}, g_{\rm to}]`; the in-scatter
     source therefore uses the **transpose**,
     :math:`Q = \text{SigS}^T \phi`
     (:ref:`scattering-matrix-convention`).
   * P\ :sub:`N` anisotropic scattering :eq:`pn-scatter` reaches the
     :term:`angular flux` **only through its moments** :eq:`flux-moments` —
     the factored composition :math:`S_{\rm aniso} = R\,\Lambda\,M`
     with the :math:`(2\ell+1)` addition-theorem normalisation
     :eq:`addition-theorem`.
   * Fission is **rank-1 in energy**:
     :math:`(F\phi)_g = \chi_g \sum_{g'} \nSigf{g'}\,\phi_{g'}` — an
     outer product of spectrum and rate with no useful inverse; the
     operator is ``apply``-only by construction.
   * The eigenvalue posing is
     :math:`(L + C - S - N_{2n} - B)\,\psi = \tfrac{1}{k}\,F\,\psi`
     (:eq:`sn-within-group-with-n2n`); the
     power method iterates the resolvent action
     :math:`K = A^{-1}F` and converges at the dominance ratio
     :math:`|k_1/k_0|` (:ref:`eigenvalue-posing`).
   * The :math:`k`-update is the hardwired Rayleigh ratio *fission
     production over net removal*; at the converged fixed point
     **every consistent functional returns the same number** — the
     estimator-injection seam this retired was freedom that never
     existed.


The posing: the multigroup invariant
====================================

Energy is discretized by partitioning the continuous energy axis into
:math:`G` groups and integrating the transport equation over each
group (the group cross sections are the flux-weighted averages the
data pipeline supplies; see :ref:`mg-eigenvalue-problem` for the
infinite-medium statement of the same discretization). The invariant
is unchanged — **sinks = sources**, now on every (cell × ordinate ×
group) region of phase space. For :math:`G` energy groups, each
transport equation becomes a coupled system with scattering transfer
:math:`\Sigs{g' \to g}` between groups:

.. math::
   :label: multigroup

   \text{streaming} + \Sigt{g} \psi_g
   = \frac{1}{W} \left[
       \sum_{g'} \Sigs{g' \to g} \phi_{g'}
       + \frac{\chi_g}{k} \sum_{g'} \nSigf{g'} \phi_{g'}
   \right]


.. implements:: multigroup
   :by: orpheus.sn.coupled_system.WithinGroupSystem

   **Implemented by** 13 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

   The list grew by one and moved two on 2026-09-04 (#426 step 2): the
   :math:`\Lambda` factor and the producer-side :math:`1/W` combine are
   now the transfer family's shared
   :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
   and
   :class:`~orpheus.transport.operators.transfer.TransferOperator`, and
   :class:`~orpheus.transport.operators.n2n.N2NOperator` joins its
   sibling :math:`S` as a declared implementer — the :math:`(n,2n)`
   gain now executes the same in-scatter arithmetic this display
   states, at the same Legendre order, where until that date it
   executed only the :math:`\ell = 0` block of it.

   The :math:`1/W` combine moved once more the same day (CS4c step 5),
   from the transfer core down onto the shared lift base
   ``AngularLift._combine`` — so the fission gain divides by the same
   :math:`W`, on the same line, as the two transfer gains. The site
   COUNT is unchanged; one member of the list is now the base's method
   rather than the core's (:ref:`cs4c-ends-select-the-body`).

.. implements:: multigroup
   :by: orpheus.sn.coupled_system.build_within_group_system

.. implements:: multigroup
   :by: orpheus.sn.operators.streaming.StreamingCollisionOperator

.. implements:: multigroup
   :by: orpheus.sn.operators.streaming.StreamingOperator

.. implements:: multigroup
   :by: orpheus.transport.operators.fission.FissionOperator

.. implements:: multigroup
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicFission

.. implements:: multigroup
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicN2N

.. implements:: multigroup
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicScattering

.. implements:: multigroup
   :by: orpheus.transport.operators.multiplication_operator.MultiplicationOperator

.. implements:: multigroup
   :by: orpheus.transport.operators.transfer.LegendreMomentTransfer

.. implements:: multigroup
   :by: orpheus.transport.operators.scattering.ScatteringOperator

.. implements:: multigroup
   :by: orpheus.transport.operators.n2n.N2NOperator

.. implements:: multigroup
   :by: orpheus.transport.operators.angular_lift.AngularLift._combine

where the streaming operator depends on the coordinate system (for
the slab, :math:`\mu\,\partial_x` as in :eq:`transport-cartesian`)
and :math:`\phi_g = \sum_n w_n \psi_{g,n}` is the :term:`scalar flux`.

The structural fact that organizes everything downstream: the
left-hand side — streaming and collision — is **diagonal in the group
index**. A neutron's group changes only at a scattering or fission
event, never in flight; all group coupling therefore sits on the
right-hand side, inside the sources. Consequently:

* the streaming-collision composite :math:`L + C` of
  :doc:`slab_one_group` (:ref:`sn-streaming-operator`) is a **direct
  sum over groups** of the one-group operator — lower-triangular in
  sweep order per group, invertible by the same one-pass sweep, group
  by group (in the implementation: one vectorised sweep with a
  trailing group axis);
* the scattering operator :math:`S` and the fission operator
  :math:`F` gain the group-coupling structure this chapter derives;
* the within-group operator equation keeps its honest shape,

  .. math::

     (L + C - S - N_{2n} - B)\,\psi \;=\; q ,

  with :math:`S` now carrying the full group-to-group (and, below,
  angular-moment) transfer, :math:`N_{2n}` the :math:`(n,2n)` gain
  carrying that same transfer structure with yield :math:`y = 2`
  (:ref:`n2n-reactions`), and :math:`B` the boundary-reflection
  gain exactly as before — the energy axis does not touch the
  boundary algebra.

With fission present the "source" :math:`q` stops being external
data: fission regenerates the flux that drives it, and the
self-consistent statement is the eigenvalue posing of
:ref:`sn-mg-eigenvalue-posing` below.


The scattering kernel: group-to-group transfer
==============================================

In :doc:`slab_one_group`, isotropic within-group scattering was the
rank-1-in-angle projection :math:`S = \Sigma_{s,0}\,P_{\rm iso}` with
the scalar coefficient :math:`\Sigma_{s,0}`. The energy axis promotes
that coefficient to a **matrix acting on the group index** — the
in-scatter sum the previous chapter promised.

Matrix convention
-----------------

The ``Mixture.SigS[l]`` matrices use the convention
:math:`\text{SigS}[g_{\rm from}, g_{\rm to}]`:

.. math::

   \text{SigS}[0] = \begin{pmatrix}
       \Sigs{0\to0} & \Sigs{0\to1} \\
       \Sigs{1\to0} & \Sigs{1\to1}
   \end{pmatrix}

For the in-scatter source (total scattering into group :math:`g` from
all groups :math:`g'`):

.. math::
   :label: mg-inscatter-source

   Q_{\rm scatter}[g]
   = \sum_{g'} \Sigs{g'\to g}\, \phi_{g'}
   = (\text{SigS}^T \cdot \phi)[g]

The vectorised form for batched cells is ``phi @ SigS`` (equivalent to
:math:`(\text{SigS}^T \phi^T)^T` for row-vector :math:`\phi`); the
production hook is the array verb
:meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`,
which performs exactly this contraction per material and carries the
channel's yield.  It is reached through the :math:`\ell = 0` half of the
collision gain — the energy binding
:attr:`~orpheus.transport.operators.transfer.TransferOperator.isotropic_energy`
that :meth:`TransferOperator.apply
<orpheus.transport.operators.transfer.TransferOperator.apply>` lifts.
(A thin ``SNSolver._add_scattering_source`` delegator stood in front of it
until #448; it retired with the hand-built finalize source that was its
only production caller.)

Note the transpose: :math:`\text{SigS}^T[g, g'] = \Sigs{g'\to g}`
gives the in-scatter contribution, so
:math:`\text{diag}(\Sigt{}) - \text{SigS}^T` is the net removal
matrix. The full derivation of this convention — and the
infinite-medium eigenvalue problem
:math:`\kinf = \lambda_{\max}(A^{-1}F)` built from it, the analytical
anchor the verification suite pins against — lives in
:ref:`scattering-matrix-convention` and
:ref:`mg-eigenvalue-problem`; this chapter consumes the convention,
it does not re-derive it.

.. _pn-scattering:

P\ :sub:`N` anisotropic scattering
----------------------------------

When ``scattering_order >= 1``, per-ordinate anisotropic sources are
computed from the Legendre moments of the angular flux.  The full
anisotropic scattering source for ordinate :math:`n` and group :math:`g`
is:

.. math::
   :label: pn-scatter

   Q_{\rm scatter}(\hat{\Omega}_n, g)
   = \sum_{\ell=0}^{L} (2\ell+1)
     \sum_{m=-\ell}^{\ell}
     \sum_{g'} \Sigs{g'\to g}^{(\ell)}\,
     f_{\ell,g'}^m \; Y_\ell^m(\hat{\Omega}_n)

where :math:`Y_\ell^m` are real spherical harmonics and the angular flux
moments are computed by quadrature:

.. math::
   :label: flux-moments

   f_{\ell,g}^m = \sum_{n=1}^{N} w_n \, \psi_{n,g} \, Y_\ell^m(\hat{\Omega}_n)

The :math:`(2\ell+1)` factor is the addition theorem normalisation for
real spherical harmonics: it ensures that the P\ :sub:`L` expansion
reproduces the angular flux moments exactly when the angular flux is a
polynomial of degree :math:`\leq L`.

**Implementation**, in the collision gain's own :math:`\ell \ge 1` body
(:meth:`TransferOperator.apply
<orpheus.transport.operators.transfer.TransferOperator.apply>` selects it at
construction from the binding's two ends; on the per-ordinate end that body
is ``TransferOperator._redistribute_ordinates``):

1. **Evaluate the angular basis** once, at frame-construction time —
   :meth:`SphericalHarmonicBasis.evaluate
   <orpheus.numerics.basis.SphericalHarmonicBasis.evaluate>` at the
   ordinates, cached on the interned
   :class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame` as its
   analysis and reconstruction faces.  ⛔ This step read *"stored as* ``self._Y`` *with
   shape* ``(N, L+1, 2L+1)``\ *"* until 2026-09-06: no such attribute has
   existed on :class:`SNSolver` since the frame carve, and the head's shape
   is the BASIS's rather than a fixed table — see the ⚠ note below.

   **Convention.** The polar axis is :math:`\mu_x`, so
   :math:`\cos\theta = \mu_x` and
   :math:`\sin\theta = \sqrt{1 - \mu_x^2}`.  Azimuth is measured in the
   :math:`(\mu_y, \mu_z)` plane:
   :math:`\cos\phi = \mu_y / \sin\theta`,
   :math:`\sin\phi = \mu_z / \sin\theta`.  This matches the MATLAB
   ``discreteOrdinatesPWR.m`` reference for :math:`\ell \le 1`:

   .. math::
      :label: real-spherical-harmonics-l1

      Y_0^0 &= 1 = P_0(\cos\theta)\\
      Y_1^{-1} &= \sin\theta\,\sin\phi = \mu_z\\
      Y_1^0    &= \cos\theta              = \mu_x\\
      Y_1^{1}  &= \sin\theta\,\cos\phi   = \mu_y

   For :math:`\ell \ge 2` the formula extends as standard real
   spherical harmonics in this frame:

   .. math::
      :label: real-spherical-harmonics

      Y_\ell^0(\hat{\Omega}) &= P_\ell(\mu_x)\\
      Y_\ell^{m}(\hat{\Omega}) &= \sqrt{\frac{2(\ell-m)!}{(\ell+m)!}}\,
                                 P_\ell^{m}(\mu_x)\,\cos(m\phi),
                                 \quad m > 0\\
      Y_\ell^{-m}(\hat{\Omega}) &= \sqrt{\frac{2(\ell-m)!}{(\ell+m)!}}\,
                                  P_\ell^{m}(\mu_x)\,\sin(m\phi),
                                  \quad m > 0

   where :math:`P_\ell^m` is the unnormalised associated Legendre
   function (the :math:`(-1)^m` Condon–Shortley phase included by
   ``scipy.special.lpmv`` is removed at the call site).  The
   normalisation is the **"no** :math:`4\pi/(2\ell+1)` **prefactor"**
   convention under which the addition theorem reads

   .. math::
      :label: addition-theorem

      \sum_{m=-\ell}^{\ell} Y_\ell^m(\hat{\Omega})\,Y_\ell^m(\hat{\Omega}')
      = P_\ell(\hat{\Omega} \cdot \hat{\Omega}')


   .. implements:: addition-theorem
      :by: orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis

      **Implemented by** 6 sites. Every symbol that executes this
      equation's arithmetic is declared, not only the canonical one: a
      test is adjudicated against the transcription it actually ran, so
      declaring a single site would refute the tests that exercise the
      others.

   .. implements:: addition-theorem
      :by: orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.addition_theorem_factor

   .. implements:: addition-theorem
      :by: orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.evaluate

   .. implements:: addition-theorem
      :by: orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.reconstruct

   .. implements:: addition-theorem
      :by: orpheus.numerics.basis.spherical_harmonic_basis._evaluate_real_sh

   which is the identity used by Eq. :eq:`pn-scatter` to expand the
   :math:`P_\ell` scattering kernel as a finite tensor product over
   :math:`m`.  Equivalently the discrete orthogonality on a quadrature
   exact for polynomials of degree :math:`\ge 2\ell` reads

   .. math::
      :label: harmonic-discrete-orthogonality

      \sum_{n=1}^{N} w_n \, Y_\ell^m(\hat{\Omega}_n)\,
                            Y_{\ell'}^{m'}(\hat{\Omega}_n)
      = \frac{4\pi}{2\ell+1}\,\delta_{\ell\ell'}\,\delta_{mm'}.

   Both identities are verified at :math:`\ell \le 3` by
   ``test_spherical_harmonics_addition_theorem_L3`` and
   ``test_spherical_harmonics_orthogonality_L3`` in
   ``tests/sn/test_solver_components.py``.  The :math:`\ell \le 1`
   branch is kept as bit-identical hardcoded values so existing
   :math:`P_0/P_1` test outputs are preserved at any tolerance
   (``test_spherical_harmonics_l1_unchanged_after_extension``).

2. **Compute flux moments** via an ``einsum`` contraction over the
   ordinate index:

   .. code-block:: python

      fiL[:, :, :, l, l+m] = np.einsum(
          'n,nxyg->xyg', w * Y[:, l, l+m], angular_flux,
      )

   This contracts :math:`\sum_n w_n Y_\ell^m(\hat{\Omega}_n) \psi_n(x,y,g)`
   into a spatial-energy moment field carrying the principled
   ``(ng, nx, ny)`` layout (see :ref:`theory-sn-index-convention`;
   the codepath presents the full moment field as
   ``(<angular head>, ng, nx, ny)`` with energy leading the spatial
   axes). ⚠ On a **slab** — this page's subject — that head has been
   FLAT since 2026-09-02: a 1-D rule binds
   :class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis`, so the
   field is ``(L+1, ng, nx, ny)`` and there is no :math:`m` axis. It was
   ``(L+1, 2L+1, …)`` before, and the :math:`m \ne 0` columns it carried
   were :doc:`ERR-080 </theory/verification/error_catalog>`'s
   fabrication (:ref:`sh-legendre-is-the-1d-family`).

3. **Reconstruct per-ordinate source**: for each Legendre order
   :math:`\ell \geq 1` (the :math:`\ell = 0` term is the energy
   binding's, added by the producer-side combine — step 4) and each
   :math:`m`, the
   scattered moment ``moment @ sig_s_l[l]`` is multiplied by
   :math:`(2\ell+1) Y_\ell^m(\hat{\Omega}_n)` and accumulated into
   ``Q_aniso[n, :, :, :]``.

4. The resulting ``Q_aniso`` array of shape ``(N, ng, nx, ny)`` is
   added to the isotropic source on a per-ordinate basis and consumed by
   the within-group sweep (the resolvent ``solve``).

**Equivalence of the code to the mathematical form.**
Equation :eq:`pn-scatter` writes the sum as
:math:`\sum_\ell \sum_m \sum_{g'} \Sigs{}^{(\ell)} f_\ell^m Y_\ell^m`.
The code separates the :math:`\ell = 0` term (isotropic — the energy
binding's reaction-rate fast path, no moment tensor allocated) from the
:math:`\ell \geq 1` terms (anisotropic — the cached :math:`R\Lambda M`
redistribution), and adds them in the ONE producer-side combine
(``AngularLift._combine``, which is also where the :math:`1/W` lives).
For :math:`\ell = 0`,
:math:`Y_0^0 = 1` and :math:`(2 \cdot 0 + 1) = 1`, so the sum reduces to
:math:`\sum_{g'} \Sigs{g' \to g}^{(0)} f_{0,g'}^0 = \sum_{g'} \Sigs{g' \to g}^{(0)} \phi_{g'}`,
which is exactly the P\ :sub:`0` source.  The split is therefore exact
with no double-counting.

The 421-group cross-section library provides both P0 and P1 matrices.

.. note::

   Because the scattering source :eq:`pn-scatter` depends on the
   angular flux **only** through its moments :math:`f_{\ell,g}^m`
   :eq:`flux-moments`, the within-group source iteration's *fixed point
   lives in moment space*: the persistent iterate need not carry all
   :math:`N` ordinates. The 2-D Cartesian SI iterate is therefore held
   as the moment tensor (:math:`N \to (L{+}1)(2L{+}1)`, "angular
   windowing"), with the :math:`\ell\ge 1` reconstruction
   :math:`R\,\Lambda` shared between the windowed and full-angular
   paths. The moments are accumulated **in-sweep** per anti-diagonal
   (:math:`\phi_\ell^m \mathrel{+}= \sum_n w_n Y_\ell^m \psi_n`), so the
   full per-ordinate field is never materialized in the windowed iterate
   (a 3.06× peak-memory win). See :ref:`sn-angular-windowing` for the
   derivation, the geometry restriction, and the bit-identity /
   principled-equivalence story, and
   :ref:`sn-angular-windowing-in-sweep-accumulation` for the in-sweep
   accumulation.

.. _n2n-reactions:

(n,2n): secondary emission, and the operator it grew into
----------------------------------------------------------

The :math:`(n,2n)` reaction is a threshold reaction in which a neutron
is absorbed by a nucleus, which then emits **two** neutrons.  The net
effect is a gain of one neutron per reaction (the incident neutron is
consumed, two are produced).

The :math:`(n,2n)` cross section is stored as a group-to-group transfer
matrix ``Mixture.Sig2`` with the same ``[g_from, g_to]`` convention as
the scattering matrix — and, like ``Mixture.SigS``, as a **list over
Legendre order** (#426 step 1, 2026-09-03).  Since #426 step 2
(2026-09-04) the algebra below consumes **all** of it, at the solve's
Legendre order, exactly as it consumes ``SigS``:

.. important::

   **The emission carries the reaction's angular distribution, and has
   done since 2026-09-04.**  Between 2026-04 and that date ORPHEUS
   *modelled* it isotropic; the model was a defect, catalogued as
   ERR-082 (:ref:`the L0 error catalogue
   <theory-verification-error-catalog>`).  The evaluated data ORPHEUS
   ships carries seven Legendre moments for the :math:`(n,2n)` channel
   — the same order as elastic scattering — the GENDF reader keeps all
   of them (:ref:`the ingest stack note
   <n2n-legendre-stack-at-ingest>`), and the binding now brings them to
   the solve.

   ⛔ **This block has been corrected twice, and both superseded
   versions are recorded because a reader may still meet either.**
   Until 2026-09-03 it said the channel was stored "as **one** matrix
   rather than a list over Legendre order" and that "the GENDF reader
   keeps the :math:`\ell = 0` one"; step 1 repealed both.  From
   2026-09-03 to 2026-09-04 it said the :math:`P_0` model "now lives at
   exactly two sites one layer up" — ``N2NKernel.from_mixture``, which
   densified ``Sig2[0]`` alone, and ``N2NOperator``, which minted its
   frame at order 0 — and that *"nothing about the model changed"*.
   Step 2 retired both sites: the kernel is now a
   :class:`~orpheus.transport.kernels.TransferKernel` (a Legendre stack
   plus a yield) and the tier-2 mint is the shared core's
   (:meth:`~orpheus.transport.operators.transfer.TransferOperator.from_solver_data`),
   which takes ``scattering_order`` and mints the SAME interned frame
   for :math:`S` and :math:`N_{2n}` alike — the role supplying only
   which ``Mixture`` channel to read.

   ``[M]`` what the model was worth: :math:`-413.55` in
   :math:`\Delta k\cdot10^{5}` (:math:`-346.01` in
   :math:`\Delta\rho\cdot10^{5}`) on a Be-reflected fast slab, the
   dipole carrying essentially all of it.  The ladder, its three
   conventions, its controls and what it is blind to are at
   :ref:`the shipped ladder <sn-n2n-anisotropy-shipped-ladder>`; the
   history and the surviving :math:`\ell = 0`-by-physics reads are at
   :ref:`the (n,2n) P0-truncation record <sn-n2n-p0-truncation>`
   (`#426 <https://github.com/deOliveira-R/ORPHEUS/issues/426>`_).

The source contribution is the per-:math:`\ell` transfer the tape
stores, scaled by the yield:

.. math::
   :label: n2n-source-per-ell

   Q_{(n,2n),\ell}(g) = \nu_{2n} \sum_{g'}
   \Sigma_{2n,\ell}\,(g'\!\to\! g)\, \phi_{\ell}(g') ,
   \qquad \nu_{2n} = 2 ,
   \qquad \ell = 0 \ldots L ,

.. (vv-status rationale) Definitional identity: the (n,2n) emission
   source per Legendre order — the generalisation #426 step 2 shipped,
   which is :eq:`n2n-source` at ℓ = 0 and the frame-conjugated
   redistribution above it.  Its verifiable content is that the ℓ ≥ 1
   moments REACH the action (a P0 twin leaves the difference at exactly
   0.0) and that the two channels differ by the yield alone, both
   ``@pytest.mark.foundation`` in
   ``tests/sn/operators/test_n2n_operator.py::TestTheBindingAtTheSolveOrder``
   (``test_the_first_moment_reaches_the_action``,
   ``test_the_two_terms_differ_by_the_yield_alone``); the EIGENVALUE
   consequence is ``@pytest.mark.l2`` in
   ``tests/sn/verification/analytical/test_be_reflected_n2n_anisotropy.py``.
   The label stays ``documented`` because neither gate is wired to it
   (wiring backlog: #309).
.. vv-status: n2n-source-per-ell documented

with :math:`\phi_\ell` the flux's :math:`\ell`-th angular moment and
:math:`L` the solve's ``scattering_order``.  Its :math:`\ell = 0` row is
the scalar-flux-driven source this section stated as the whole of it
until 2026-09-04,

.. math::
   :label: n2n-source

   Q_{(n,2n)}(g) = \nu_{2n} \sum_{g'} \Sigma_{2,g'\to g}\, \phi_{g'} ,
   \qquad \nu_{2n} = 2

and that row keeps its own label because it is what several distinct
things still are: the row the reaction-rate **fast path** evaluates
(through the P0 energy binding, with no moment tensor allocated), the
row the P0 energy binding evaluates as the driver's lagged gain, and the
**whole** source for every scalar-flux consumer — CP, MoC, Monte Carlo
and the 1-D diffusion solver — whose emission is isotropic *by
construction*, a property of the method and not of this channel.
:eq:`n2n-source` is therefore neither retired nor a truncation; it is a
row of :eq:`n2n-source-per-ell` that a great deal of the tree is
entitled to read on its own.

The multiplicity :math:`\nu_{2n} = 2` accounts for the two neutrons
produced per reaction.  It has exactly **one home** in the tree —
:data:`~orpheus.transport.kernels.N2N_MULTIPLICITY`, a module constant
beside the kernel type, which every channel kernel carries as
:attr:`TransferKernel.multiplicity
<orpheus.transport.kernels.TransferKernel.multiplicity>` — and every
production site that needs it reads it there.  (It was a ``ClassVar``
on a dedicated ``N2NKernel`` until #426 step 2; when the two channels
became one type the yield had to become a **field**, since scattering's
value is 1 and :math:`(n,2n)`'s is 2 on the same class.)  That was not always so: until 2026-08-30 the number
was an inline ``2.0`` (or, in one place, an integer ``2``) at
**fourteen** production sites across S\ :sub:`N`, CP, MoC and Monte
Carlo, and a census gate
(``tests/transport/test_n2n_multiplicity_census.py``) now asserts that
no production literal survives outside the kernel module.  The gate's
AST predicate is validated against all four historical spellings the
sweep had to catch — ``2.0 *``, a bare integer ``2 *``, an augmented
``w *= 2.0``, and the module-constant form MC hoisted to — because a
predicate that only recognises the spelling you happened to look at is
a census of your own filter, not of the tree.

This source is added to the isotropic source before the transport
sweep, on the same footing as the P\ :sub:`0` scattering source.  The
:math:`(n,2n)` contribution also enters the :math:`\keff` production
term in :meth:`SNSolver.compute_keff`, where row sums of ``Sig2[0]``
(total :math:`(n,2n)` removal rate) are used — the :math:`P_0` block
specifically, and correctly so: a reaction rate is a :math:`P_0`
quantity, and every higher Legendre moment integrates to zero over
angle.  That read is unaffected by the restored stack, and by step 2.

**Where the arithmetic lives, and why that is not where the grouping
is decided.**  The per-material dispatch — the loop over materials, the
gathered ``einsum``, the yield — is the array verb
:meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`
(with its transpose sibling and the per-:math:`\ell`
:meth:`~orpheus.transport.material_field.TransferMaterialField.moment_source`
pair), on the kernel-field pairing described at
:ref:`scattering-binding-cs4c`.  (A solver-facing ``SNSolver._add_n2n_source``
delegator routed to it until #448, when the eigenvalue finalize — its only
production caller — stopped building a source of its own.)  What that verb does
*not* decide is which operator the channel belongs to — and that
question turns out to be the interesting one.

⭐ **Those verbs were channel-NAMED until 2026-09-04, and that naming
was the truncation wearing a method name.**  ``N2NMaterialField`` had
``add_emission`` / ``moment_emission`` where ``ScatteringMaterialField``
had ``add_p0_source`` / ``moment_source``: member-for-member twins whose
only differences were a factor of 2 and the fact that the
:math:`(n,2n)` pair existed for :math:`\ell = 0` **alone**.  #426 step 2
read the factor for what it is — the channel's yield, a **datum**, which
the tape stores folded into every Legendre order — and collapsed the two
fields onto
:class:`~orpheus.transport.material_field.TransferMaterialField`, whose
every verb carries ``scale = self.multiplicity``.  There is now one verb
set, and the scattering path is bit-identical because at :math:`y = 1`
the scale branch is skipped.

**Why it was folded into scattering, and why it is not any more.**  The
original ruling (Wave D, recorded on this page until 2026-08-30) folded
:math:`(n,2n)` into the **scattering** side of the algebra rather than
giving it its own operator, for three stated reasons:

1. The bookkeeping is identical to in-scatter (vectorise-by-material,
   add-into-:math:`Q`).
2. The legacy code placement (then ``SNSolver._add_n2n_source``, retired
   at #448) was inside the same source-
   construction block as scattering. Wave D Issue 13's bit-identical
   extraction needed to preserve that placement to keep the regression
   snapshots bit-identical.
3. Architecturally, both are *secondary-emission scalar-flux-driven*
   sources --- they belong to the same algebra slot.

All three remain true as statements.  What CS4c step 3 (design record
§14.1, 2026-08-30) established is that **none of them is an argument
about the right place to decide a grouping.**  Reason 1 is about
implementation shape, reason 2 about a migration constraint that has
since been discharged, and reason 3 — the only structural one — is
half a classification:

   :math:`(n,2n)` is **scattering-like** (a group-to-group transfer,
   in principle carrying its own anisotropy) **and production-like**
   (it carries a multiplicity).  Which one it should be bundled with
   therefore depends on the question: with :math:`S` when scattering
   anisotropy is the axis of interest, with :math:`F` when production
   accounting is.  A bundling that is context-dependent **must not be
   decided at the operator level**, because an operator that hard-codes
   one grouping makes the other unspellable.

So the channel became the first-class
:class:`~orpheus.transport.operators.n2n.N2NOperator`, the within-group
algebra spells it explicitly, :math:`A = L + C - S - N_{2n} - B`
(:eq:`sn-within-group-with-n2n`), and any bundling is a solver-side
:class:`~orpheus.numerics.operator.OperatorSum` grouping.  The two
shipped solvers now make **different** choices, legibly: the
S\ :sub:`N` within-group builder keeps the two terms apart, while the
1-D diffusion solver sums
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
with
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
into the single :math:`S` its :math:`A = L + C - S - B` expects — that
:math:`S` **is** :math:`S + N_{2n}`, so the four-term spelling is exact
there and is a statement about the *composition*, not about the member
list.  Under
the old design that disagreement was unrepresentable; under the new one
it is two lines at two composition sites.

The forward action of :math:`N_{2n}` on the angular composite, its
transpose, and the fixture blindness that transpose hides are derived
at :ref:`sn-n2n-adjoint`.

**What the CS4c extraction did not touch, and what #426 step 2 did.**
The extraction (2026-08-30) left the emission *modelled* isotropic, so
the operator kept the reaction-rate fast path with no moment tensor,
exactly as the fused version had.  Step 2 (2026-09-04) split that
sentence in two, and the split is the whole shape of the change:

* the :math:`\ell = 0` half **still** rides the reaction-rate fast path
  — one ``einsum`` over groups through the P0 energy binding
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`,
  no moment tensor allocated;
* the :math:`\ell \ge 1` half rides the **frame**, reconstructed from
  the flux moments exactly as :math:`S`'s does;
* and the two are combined by the same producer-side
  :math:`(\text{iso}/W) + \text{aniso}` verb, single-sourced in one
  place for both channels.

That is precisely the split :math:`S` has always had — which is the
point: the :math:`(n,2n)` gain did not acquire a *new* evaluation
strategy, it stopped being denied the one its sibling used.  The
algebra is still stated as a frame conjugation and gated against it
while the :math:`\ell = 0` evaluation stays cheap.

⛔ This paragraph read, until 2026-09-04, *"the operator keeps the
reaction-rate fast path (no moment tensor) exactly as the fused version
did"* — true then, and now true of the :math:`\ell = 0` half only.  The
neighbouring paragraph read that #426 step 1 was **bit-identical**
through this page's arithmetic "because every consumer named above
reads ``Sig2[0]``, which is the matrix that used to be the whole of
it", and that *"spending them is step 2"*.  Step 1's bit-identity claim
stands and is now history; step 2 spent the moments, and this page's
:math:`(n,2n)` arithmetic is no longer bit-identical to the pre-step-2
tree — the shipped ladder measures the difference
(:ref:`the shipped ladder <sn-n2n-anisotropy-shipped-ladder>`).  What is unchanged, and was
predicted to be, is everything that reads ``Sig2[0]``: the
:math:`\keff` term, the removal side, and the three isotropic solver
families.

The normalization chain
-----------------------

The normalization chain in the code ensures consistent scaling:

1. **Fission source** (:meth:`SNSolver.compute_fission_source`):
   :math:`Q_f = \chi \cdot (\nSigf{} \cdot \phi) / k` --- raw,
   un-normalised.  Since CS4c step 4 this is a thin delegator to the
   fission **energy** binding's ``apply`` (the dyad bound at the mesh's
   scalar bulk space); the :math:`1/k` stays here.

2. **Scattering source** — the collision gain applied to the iterate,
   :meth:`TransferOperator.apply
   <orpheus.transport.operators.transfer.TransferOperator.apply>`.  Its
   :math:`\ell = 0` row is :math:`Q_s = \text{SigS}^T \cdot \phi`, in
   scalar-flux units and un-normalised; its :math:`\ell \ge 1` rows are
   already per-ordinate, which is why the :math:`1/W` of step 3 is applied
   to the isotropic part *inside* the operator's own combine
   (:math:`(\text{iso}/W) + \text{aniso}`) rather than again by the sweep.
   ⚠ Since #448 this is the ONLY assembly of the scattering source on the
   eigenvalue path — the finalize evaluates these same gains instead of
   rebuilding a :math:`P_0` copy of them
   (:doc:`ERR-083 </theory/verification/error_catalog>`).

3. **Sweep** (the within-group resolvent ``solve``,
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`): applies
   :math:`Q_{\rm scaled} = Q \cdot w_{\rm norm}` where
   :math:`w_{\rm norm} = 1/\sum w_n`.  This is the :math:`1/W` division
   in the S\ :sub:`N` equation.

4. **Scalar flux** (inside sweep):
   :math:`\phi = \sum_n w_n \psi_n` --- standard quadrature integration.

5. **keff** (:meth:`SNSolver.compute_keff`):
   :math:`k = (\nSigf{} \cdot \phi \cdot V) / (\Sigma_a \cdot \phi \cdot V)`
   --- volume-weighted ratio (the method-layer estimator; see
   :ref:`sn-keff-estimator`).

The :math:`1/W` in step 3 and the :math:`W` implicit in step 4 cancel:
:math:`\phi = \sum w_n \cdot Q/(W \Sigt{}) = Q/\Sigt{}` for uniform
isotropic source.

**Convention rule:** Sources passed to the sweep must NOT include
:math:`1/W` --- the sweep applies it.  A direct-operator path that
solves :math:`T\psi = b` without the sweep must divide sources by
:math:`W` itself.


.. _sn-scattering-fission-operators:

Scattering and fission as operators
===================================

Wave A Issue 1 of the SN reshape campaign installed the
:class:`~orpheus.numerics.operator.LinearOperator` Protocol --- a
predicate-typed matrix-free operator algebra (see :ref:`operator-algebra`).
In that algebra the multigroup problem of this chapter is posed on the
honest within-group operator :math:`A = L + C - S - N_{2n} - B`
(:eq:`sn-within-group-with-n2n`,
:doc:`/theory/foundations/operator_algebra`):

.. math::

    (L + C - S - N_{2n} - B)\,\psi = q
    \qquad\text{(fixed source)}

.. math::

    (L + C - S - N_{2n} - B)\,\psi = \tfrac{1}{k}\,F\,\psi
    \qquad\text{(eigenvalue)}

where :math:`L + C` is the group-diagonal streaming-collision
composite the sweep inverts (:ref:`sn-streaming-operator`), :math:`S`
is the scattering gain of this chapter, :math:`N_{2n}` the first-class
:math:`(n,2n)` source extracted from it at CS4c step 3 (the subsection
above), :math:`B` the boundary-reflection gain, and :math:`F` the
fission source operator.  (Both displays read :math:`A = L + C - S - B`
until 2026-08-30, when the extraction gave :math:`(n,2n)` its own term;
the 1-D diffusion solver's :math:`A` still reads that way, because it
sums the two isotropic energy leaves into one :math:`S` at its own
composition site — the disagreement is deliberate and legible.)
Wave D Issue 13 lifted :math:`S` and :math:`F` out of
:class:`~orpheus.sn.solver.SNSolver` and into
:class:`~orpheus.transport.operators.scattering.ScatteringOperator` and
:class:`~orpheus.transport.operators.fission.FissionOperator` respectively. The math is
**moved verbatim** --- the regression contract on the 11 frozen
snapshots at ``tests/sn/regression/snapshots/`` gates the extraction.

Why ``apply``-only
------------------

Both operators report ``is_invertible = False`` and carry no ``solve``
— they are *structural* non-invertibles, declaring no ``inverse()`` /
``solve`` at all (:ref:`design-c-structural-value-split`). (They *are*
adjointable — ``is_adjointable = True`` — since each gained a working
``apply_transpose`` for the outer-layer adjoint / DSA posing; the
"``apply``-only" name refers to the **inverse** axis.)

* **Scattering**, :math:`S`, is rank-:math:`O(N_{\text{cells}}\cdot
  N_{\text{groups}})`. There is no efficient inverse --- the operator
  is *applied*, never *inverted*. The upper-triangular structure that
  would make a sweep-based ``solve`` tractable does not survive the
  Pℓ Galerkin reconstruction. An algebraic consumer that asks for
  :math:`S^{-1}` cannot even spell ``S.inverse()`` (the method is not
  declared --- a *static* error), never silently wrong results at call
  time --- this is the load-bearing payoff of the three-layer operator
  surface (see :ref:`operator-algebra`).

* **Fission**, :math:`F`, has rank-1-in-energy structure: the action
  factorises as :math:`(F\phi)_g = \chi_g\,\sum_{g'}\nu\Sigma_{f,g'}
  \phi_{g'}`, an outer product of the emission spectrum with a scalar
  per-cell rate. This rank-1 structure forbids a useful inverse on
  the energy axis (the rate has lost direction information). The
  :math:`1/k` eigenvalue division stays at the **solver** level ---
  the operator returns :math:`F\,\phi` and the EigenvalueSolver
  Protocol's ``compute_fission_source`` divides by :math:`k`. This
  separation preserves linearity of the operator (Wave A Protocol
  contract: an operator's ``apply`` is independent of solver state).

  ⚠ Since CS4c step 4 (2026-08-30) the equation just written is the
  **energy** binding's contract, not the angular one's:
  :meth:`IsotropicFission.apply
  <orpheus.transport.operators.isotropic_transfer.IsotropicFission.apply>`
  is what maps a scalar flux :math:`(n_g, *\text{spatial})` to
  :math:`F\phi`, and it is what
  :meth:`~orpheus.sn.solver.SNSolver.compute_fission_source` delegates
  to.  :meth:`FissionOperator.apply
  <orpheus.transport.operators.fission.FissionOperator.apply>` is the
  *angular* lift of the same dyad, whose operand is the composite
  :class:`~orpheus.transport.full_field.FullField` its ends declare: a
  scalar carrier is **refused** by the admission, naming the operator
  and saying that a typed bulk field rides inside a composite while a
  bare array is the plain binding's carrier
  (:ref:`cs4c-ends-select-the-body`). The message that points a scalar
  consumer explicitly at the energy binding is the tier-2 mint's
  (``FissionOperator.from_solver_data`` refuses a scalar composite by
  name — :ref:`sn-fission-binding-adjoint`).  Both bindings are non-invertible
  and adjointable, so the "``apply``-only" reading above holds for each.

Pℓ Galerkin projection on :math:`Y_\ell^m`
-------------------------------------------

The :math:`\ell\ge 1` contribution to :math:`S` is the
:eq:`pn-scatter` Galerkin reconstruction in real spherical
harmonics :math:`Y_\ell^m`, expanded with the discrete-orthogonality
identity :eq:`addition-theorem` (the Lebedev-quadrature L0
verification of the addition theorem lives at
``tests/sn/test_solver_components.py::TestAnisotropicScattering
::test_spherical_harmonics_addition_theorem_L3``).

.. note::

   The Pℓ Galerkin reconstruction is realised by the
   spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`
   (Frame/Basis carve), built on a quadrature via
   :meth:`Quadrature.angular_frame(L)
   <orpheus.numerics.quadrature.Quadrature.angular_frame>`. Its
   ``analysis`` face is the :math:`\Pi = Y^* W` projection on the
   angular axis; its ``reconstruction`` face is the addition-theorem
   reconstruction with the :math:`(2\ell+1)` factor. The full-space
   projector is the tensor product
   :math:`\Pi \otimes I_x \otimes I_y \otimes I_g`, built via the
   ``&`` dunder of
   :class:`~orpheus.numerics.operator.TensorProductOperator`. See
   :ref:`galerkin-projection` for the discrete-frame narrative and
   the cross-method consumer table
   (PN solver, energy condensation, MC adjoint moments) and
   :ref:`spherical-harmonics` for the convention and addition
   theorem. The :math:`Y_\ell^m` evaluator
   :meth:`SphericalHarmonicBasis.evaluate
   <orpheus.numerics.basis.SphericalHarmonicBasis.evaluate>`
   is the canonical generic infrastructure consumed here.

   **Wave 1 (commit ff454f2)**: the anisotropic source verb — then
   ``ScatteringOperator.build_aniso_source``, since #448 the
   construction-selected body ``TransferOperator._redistribute_ordinates``
   behind :meth:`TransferOperator.apply
   <orpheus.transport.operators.transfer.TransferOperator.apply>` — became
   the literal §9 line 1230 operator-algebra composition

   .. math::
      :label: pn-scatter-rlm

      Q^{\rm aniso}_n(\vec r) \;=\; R\,\Lambda\,M\,\psi

   .. (vv-status rationale) Representational identity: the operator-algebra
      spelling (analysis M, moment-space transfer Λ, reconstruction R) of the
      anisotropic source :eq:`pn-scatter`, which is itself wired
      (``verifies("pn-scatter")`` on ``TestAnisotropicScattering``).  The RΛM
      composition is gated by the ``slab_2g_p1_aniso_dd_n20`` regression
      snapshot, the ``tests/sn/verification/mms/test_mms_aniso.py`` Pℓ MMS
      convergence suite, and the forward-reproduction cross-check
      ``test_scattering_adjoint.py::TestFullScatterKernel::test_reproduces_forward_scattering_source``.
   .. vv-status: pn-scatter-rlm documented

   where :math:`\Lambda` is
   :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
   --- the per-ℓ block-diagonal transfer on moment space (the §15.2
   sum-of-tensor-products form
   :math:`\Lambda = y \sum_\ell P_\ell \otimes \Sigma_{c,\ell}`,
   with :math:`y = 1` for scattering).  It was named
   ``LegendreMomentScattering`` and carried no yield until #426 step 2
   (2026-09-04), when the :math:`(n,2n)` channel's own :math:`\Lambda`
   collapsed into it; the scattering path is bit-identical, because at
   :math:`y = 1` the scale is never applied.
   The previous ``for n in range(N)`` Python loop over ordinates is
   gone *by construction*: each constituent's :meth:`apply` carries
   the ordinate iteration internally via :func:`numpy.einsum`, not
   via a Python loop. Total flop count is unchanged; the iteration
   is structural rather than buried in a triple-nested loop. The
   refactor is gated by the
   ``slab_2g_p1_aniso_dd_n20`` regression snapshot (rtol=1e-12,
   atol=1e-13) and the full
   :file:`tests/sn/verification/mms/test_mms_aniso.py` Pℓ MMS convergence suite.

Per-cell flux moments :eq:`flux-moments` are computed by the
discrete projection

.. math::

    \phi^{\ell m}_g(\vec r)
    \;\approx\; \sum_n w_n\,\psi_{n,g}(\vec r)\,Y_\ell^m(\hat\Omega_n).

The reconstruction back to per-ordinate scattering source uses the
addition theorem:

.. math::

    Q^{(\ell\ge 1)}_{n,g}(\vec r)
    = \sum_{\ell=1}^{L}\,(2\ell+1)\,\sum_m Y_\ell^m(\hat\Omega_n)\,
      \sum_{g'}\Sigma_{s,\ell}^m(g'\to g)\,\phi^{\ell m}_{g'}(\vec r).

The :math:`(2\ell+1)` factor is the discrete-orthogonality
normalisation
:math:`\langle Y_\ell^m | Y_{\ell'}^{m'}\rangle =
(4\pi/(2\ell+1))\,\delta_{\ell\ell'}\delta_{mm'}` working out across
both projection and reconstruction. The Galerkin frame is **real**
spherical harmonics (the
:meth:`~orpheus.numerics.quadrature.Quadrature.angular_frame` (its ``table``; the ``spherical_harmonics`` pass-through accessor was retired 2026-09-02 with #429's fix)
implementation), not complex --- this is the convention native to
the Lebedev tabulation and avoids carrying complex arithmetic
through the source-iteration inner loop.

Development history — the Wave D/E wiring
-----------------------------------------

Wave E Round 1 (Issue #163) lifted the iteration primitives out of
:class:`~orpheus.sn.solver.SNSolver` into stand-alone operator-algebra
consumers in :mod:`orpheus.numerics.iteration`, consuming the Wave A
:class:`~orpheus.numerics.operator.LinearOperator` Protocol triple
:math:`(A, S, F)` directly — no transport-solver knowledge beyond the
operator contract.

Wave E Round 2 (Issue #164) wired the operator algebra
:math:`(A, S, F)` into :class:`SNSolver` and replaced the legacy
BiCGSTAB inner-solver path with Krylov-on-``A.apply`` (GMRES
with the sweep as preconditioner).  The
``build_transport_linear_operator*`` and ``build_rhs*`` helpers
were retired, and four per-method delegators on :class:`SNSolver` —
``_add_scattering_source``, ``_build_aniso_scattering``,
``_add_n2n_source`` and
:meth:`~orpheus.sn.solver.SNSolver.compute_fission_source` — were kept as
thin wrappers over the new operators.

⛔ **The reason this paragraph gave for keeping them was wrong, and it was
wrong when written.**  It read *"for the EigenvalueSolver Protocol
surface"*; ``[M]`` :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver`
declares exactly **five** members — ``initial_flux_distribution``,
``compute_fission_source``, ``solve_fixed_source``, ``compute_keff`` and
``measure_stopping_criteria`` — and only ``compute_fission_source`` is
among them.  The other three survived because the eigenvalue finalize
called them to build its own reconstruction source, which is
:doc:`ERR-083 </theory/verification/error_catalog>`; all three retired with
that source at #448 (2026-09-06).  ``compute_fission_source`` is genuinely
on the Protocol and stays.

Wave E Round 3 (Issue #98 follow-up) extended the FD operator's
boundary handling to consume the
:class:`~orpheus.geometry.boundary.BoundaryTraceLaw` infrastructure
(Wave B Issue 7), so the then-``solution_to_angular_flux*`` codec and the
matvec helpers dispatched boundary fills via the realiser-routed
1-arg :meth:`apply` on the resolved
:class:`~orpheus.numerics.operator.LinearOperator` — vacuum,
reflective, white, albedo, periodic, and mixed BCs are honoured
uniformly. (Post Issue #186 / B3 + β2, the law itself is a pure
descriptor; the SN realiser produces the callable. See
:ref:`bc-trace-law-descriptor-model`.)


.. _sn-mg-eigenvalue-posing:

The eigenvalue posing: criticality as a resolvent eigenproblem
==============================================================

With fission first-class, the fixed-source posing loses its footing:
the fission source is proportional to the very flux it drives.  The
self-consistent question — *at what scaling of production does a
steady flux exist?* — is the criticality eigenproblem.  Scale the
production by :math:`1/k` and demand balance:

.. math::
   :label: sn-mg-eigenvalue-posing-eq

   (L + C - S - N_{2n} - B)\,\psi \;=\; \frac{1}{k}\,F\,\psi .

.. (vv-status rationale) Governing equation: the criticality posing (scale
   production by 1/k and demand balance).  Definitional — it states the
   eigenproblem, not a per-term solver claim.  Its solved eigenvalue is pinned
   independently of transport by the analytical infinite-medium anchor
   k∞=λ_max(A⁻¹F) (``tests/sn/verification/analytical/test_kinf_homogeneous.py``,
   :ref:`mg-eigenvalue-problem`).
.. vv-status: sn-mg-eigenvalue-posing-eq documented

This is the slab-multigroup instance of the generalized eigenproblem
:eq:`eigen-standard-form` — :math:`A_{\rm loss}\,\psi = \lambda M \psi`
with :math:`A_{\rm loss} = L+C-S-N_{2n}-B`, :math:`M = F`,
:math:`\lambda = 1/k` — whose full four-layer architecture (posing, resolvent,
algorithm, and the :math:`\alpha`/adjoint seams) lives at
:ref:`eigenvalue-posing`.  Inverting the loss operator turns it into a
standard eigenproblem for the **multiplication operator**

.. math::
   :label: mg-multiplication-operator

   K \;=\; A^{-1} F ,
   \qquad
   K\,\psi \;=\; k\,\psi :

.. (vv-status rationale) Definitional identity: the multiplication operator
   that turns the generalized eigenproblem into a standard one; k_eff is its
   dominant eigenvalue.  It names the resolvent-eigenproblem, not a computation
   distinct from the eigenvalue solve; the k it produces is pinned by the
   analytical k∞ anchor (:ref:`mg-eigenvalue-problem`).  The intrinsic
   object-level teeth for a K=A⁻¹F carve (Mode-12 spectral-invisibility) live
   with the operator-algebra taxonomy (#226), not here.
.. vv-status: mg-multiplication-operator documented

:math:`k_{\rm eff}` is the dominant eigenvalue of :math:`K`, and the
fundamental mode is its (unique, non-negative — Krein–Rutman)
eigenvector.  Note what the posing reuses: :math:`A^{-1}` is exactly
the within-group resolvent :doc:`slab_one_group` §7 realises — source
iteration lagging the :math:`S` and :math:`B` gains through the
:math:`(L+C)^{-1}` sweep kernel, or Krylov on ``apply`` with the sweep
as preconditioner.  The eigenvalue posing adds an **outer loop around
that resolvent**; it changes nothing inside it.

The inverse-operator family taxonomy (#226) is the reason this
sentence can be spelled as code: ``K = A.inverse() @ F`` composes the
resolvent with the fission apply, and *which* family member realises
``A.inverse()`` is a type choice
(:doc:`/theory/foundations/operator_inverse_family`), invisible to the
outer loop.

Power iteration: the outer loop
===============================

:class:`~orpheus.numerics.iteration.KEigenvalue` poses the
k-eigenvalue problem from its operator triple and **delegates** the
outer loop to the canonical
:func:`~orpheus.numerics.eigenvalue.power_iteration` (one loop engine;
see :ref:`eigenvalue-posing`).  The triple it consumes is
:math:`(L{+}C,\; S,\; F)` — its first operand is the *invertible
loss composite* the sweep inverts (the constructor parameter is named
``A`` after this operand), and the lagged gains are subtracted
explicitly by the iteration, exactly the within-group splitting of
:doc:`slab_one_group`.  Each outer step (run by ``power_iteration``)
is classical power iteration on the :math:`k`-update, with
:class:`SourceIteration` driving the inner fixed-source solve:

.. math::
    :label: power-iteration-flux-update

    \psi_{n+1} \;=\; \bigl((L{+}C) - S\bigr)^{-1}\,F\,\psi_n / k_n

.. (vv-status rationale) Governing iteration: the power-method flux update
   (inner resolvent applied to the scaled fission source).  Definitional — it
   states the outer-loop step, not a per-term claim.  The end-to-end power
   iteration is exercised by the synthetic KEigenvalue-vs-``numpy.linalg.eig``
   ground truth and the KEigenvalue-vs-``solve_sn`` L1 gate in
   ``tests/numerics/test_iteration.py``.
.. vv-status: power-iteration-flux-update documented

.. math::
    :label: power-iteration-keff-update

    k_{n+1} \;=\; \frac{\sum (F\,\psi_{n+1})}
                       {\sum \bigl((L{+}C)\,\psi_{n+1}\bigr)
                        - \sum (S\,\psi_{n+1})}

.. (vv-status rationale) Governing iteration: the hardwired operator-form
   Rayleigh k-update, fission production over net removal.  Definitional (the
   numerics-layer spelling of the unified k discipline); by the consistency
   theorem below every consistent functional returns k* at the fixed point, so
   "k matches" carries limited object-level mutation coverage (vv Mode 12).
   The method-layer functional it mirrors, :eq:`sn-keff-update`, is wired with
   leakage-drop teeth in ``tests/sn/eigenvalue/test_keff_estimator_gate.py``.
.. vv-status: power-iteration-keff-update documented

The dominance ratio :math:`|k_1/k_0|` governs outer-loop
convergence (:cite:`TrefethenBau1997` §27).  The inner solve uses
:class:`SourceIteration` with operator triple :math:`(L{+}C, S, 0)` —
the fission contribution at the inner level is the **external
source** :math:`F\psi_n/k_n`, NOT a within-group fixed-point term.
Every outer iteration warms up its inner :class:`SourceIteration`
from :math:`\psi_n` (the previous outer iterate); this is the same
amortisation pattern :class:`SNSolver` uses today.

The numerics-level triple carries no boundary-gain operand: the
:math:`B` of the honest algebra is a *method-layer* concern, carried
by the SN solver's own within-group system (which is why SN
implements the :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver`
Protocol directly rather than routing through
:class:`KEigenvalue` — see the cross-solver map below).

The :math:`k`-update is the **hardwired** operator-form Rayleigh
quotient (:meth:`KEigenvalue.compute_keff
<orpheus.numerics.iteration.KEigenvalue.compute_keff>`) — the
operator-level spelling of the unified :math:`k` discipline,
fission production over net removal (see :ref:`sn-keff-estimator`).
Because the first operand carries streaming + collision,
:math:`\sum((L{+}C)\psi) - \sum(S\psi)` is absorption + leakage − the
neutron-multiplying :math:`(n,2n)` emission (the in- and out-group
scatter cancel into :math:`\Sigma_a` via :math:`\Sigma_t - \Sigma_s`),
term-for-term the method-layer functional :eq:`sn-keff-update` with
the volume measure absorbed into the operators' action.  Because the
leakage rides **inside** the loss composite, this spelling never had
the #291 omission.

.. note:: **Estimator injection retired (R8, #259 P1, 2026-07-03).**

   Pre-R8, :class:`KEigenvalue` accepted ``keff_estimator`` /
   ``production_estimator`` *callables* (defaulting to module-level
   ``_default_*`` functions) so a caller could substitute its own
   field-to-scalar functional.  Those kwargs, the ``KeffEstimator`` /
   ``ProductionEstimator`` aliases, and the ``_default_*`` module
   functions are **gone**: the estimators are now hardwired methods
   (:meth:`~orpheus.numerics.iteration.KEigenvalue.compute_keff` /
   :meth:`~orpheus.numerics.iteration.KEigenvalue.compute_production_rate`),
   arithmetic bit-identical to the retired defaults.

   The seam was **dead by design, not dead by being unwired.**  The
   five method-layer solver families (SN / CP / diffusion / MoC /
   homogeneous) implement the
   :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` Protocol
   *directly* and never routed through ``KEigenvalue`` at all; and by
   the consistency theorem below, injection could only ever have
   introduced an *inconsistent* functional.  Removing the hook makes
   that illegal estimator/problem pairing unrepresentable
   (``coding-elegance`` Pattern 4 — illegal states unrepresentable).

   **Consistency theorem.**  The loop poses
   :math:`\bigl((L{+}C) - S\bigr)\,\psi = F\psi/k` and converges to
   the fixed point :math:`\psi^\star` with
   :math:`\bigl((L{+}C)-S\bigr)\,\psi^\star = F\psi^\star/k^\star`.
   Applying the all-ones covector :math:`\mathbf 1^\top` (the
   ``\sum``) to both sides gives
   :math:`\sum((L{+}C)\psi^\star) - \sum(S\psi^\star)
   = \sum(F\psi^\star)/k^\star`, so the hardwired ratio returns
   **exactly** :math:`k^\star`.  Every functional that agrees with the
   posed balance at the fixed point returns the same number; the
   "freedom" the injection seam advertised was illusory — all
   *consistent* choices collapse to one value, and any *different*
   injected estimator is by construction inconsistent with the problem
   the loop solves.

   **Honest-**\ ``A.apply``\ **contract.**  The theorem needs
   :math:`\sum((L{+}C)\psi)` to be the true net-removal rate — i.e.
   the loss operand's ``apply`` must compute the real streaming +
   collision action, not a stub.  The injection seam had historically
   existed to paper over a *scalar-level test adapter* whose ``apply``
   was dishonest; R8 removes the paper and requires the honest action.
   An adapter with a stubbed ``apply`` now yields a visible
   non-eigenvalue here — the correct failure, surfaced by design
   rather than masked by a substituted functional.

Relocating the :math:`k`-update into the algorithm itself — a Rayleigh
step on the resolvent :math:`A_{\rm loss}^{-1}M` so the loop is
*literally* K/α-agnostic — is the α-eigenvalue wave's first step; see
the :ref:`eigenvalue-posing` honest-scope note.

Operand requirements
--------------------

The primitive constructors enforce their operands' surface at
construction time, NEVER mid-iteration (the same Wave A philosophy that
gates :class:`~orpheus.numerics.operator.OperatorSum` etc.).  Since the
#226 taxonomy carve the requirement is stated on the operator surface
directly (predicate + verb), not a capability tag:

* :class:`SourceIteration` consumes a **pre-inverted** step operator
  ``A_inv``, which MUST provide a callable ``apply`` — the step operator
  arrives already inverted, so an apply-only object is legitimate by
  design (#226 taxonomy step 3).  A missing ``apply`` raises
  ``TypeError`` at construction.
* :class:`KEigenvalue` poses the ``(A, S, F)`` triple: ``A`` MUST report
  :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
  ``= True`` — the posing layer builds :math:`A^{-1}` via
  :func:`~orpheus.numerics.iteration.seeded_inverse`, and a
  non-invertible ``A`` raises
  :class:`~orpheus.numerics.operator.NotInvertible`.  ``S`` / ``F`` MUST
  provide ``apply`` (pass
  :class:`~orpheus.numerics.operator.ZeroOperator` for the
  scattering- / fission-free case).

Constructor failure raises ``TypeError`` (a missing ``apply``) or
:class:`~orpheus.numerics.operator.NotInvertible` (a non-invertible
``A`` where the posing layer needs its inverse), naming the operand at
fault.

Forward hook: FEAST and beyond
------------------------------

:class:`KEigenvalue` accepts ``eigenvalue_method``, currently only
``"power"``.  The hook reserves a path for FEAST-style contour-
integral methods (:cite:`Polizzi2009`) and Krylov-Schur deflation methods
(Stewart 2001) when accuracy on closely-spaced eigenvalues becomes
load-bearing.  Other values raise :class:`NotImplementedError` at
construction time.

.. _cross-solver-eigenvalue-consumers:

Cross-solver consumers of ``power_iteration``
---------------------------------------------

:func:`orpheus.numerics.eigenvalue.power_iteration` is the
**canonical** Layer-4 power-method algorithm — NOT a legacy
primitive.  It iterates over the method-agnostic
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` Protocol
boundary (the *late-bound* resolvent layer), which is **strictly more
general** than the operator-triple form: it admits both the
triple-based resolvent (SN, MoC) *and* the
**monolithic-matrix resolvent** (CP, diffusion, homogeneous) that has
no separable :math:`\bigl((L{+}C)-S\bigr)^{-1}` factor.  All five
solver families are
therefore **co-consumers of the same canonical boundary**, each
supplying its own ``EigenvalueSolver``-Protocol realization of the
Layer-3 resolvent.  There is no migration to a single ``KEigenvalue``
engine — and no retirement of ``power_iteration`` — because no narrower
layer can express the no-triple families without manufacturing
fictitious :math:`L`, :math:`S` operators for methods that have no
sweep.  See :ref:`eigenvalue-posing` for the full four-layer
architecture and why the Protocol layer is canonical.

* **SN** (discrete ordinates) — drives ``power_iteration`` directly
  via :func:`~orpheus.sn.solver.solve_sn`; its Layer-3 resolvent is
  the within-group :class:`SourceIteration` /
  :class:`KrylovAcceleration` inner solve built from the
  :func:`~orpheus.sn.coupled_system.build_within_group_system` SSoT (see
  :ref:`sn-solver-operator-algebra-coordinator`).
* **CP** (collision-probability) — drives ``power_iteration`` through
  its own ``EigenvalueSolver``-Protocol implementation; its resolvent
  is **one BiCGSTAB on a monolithic collision-probability matrix**,
  which has no :math:`(A, S, F)` split.  This is exactly the family
  the late-bound Protocol layer exists to admit.
* **Diffusion** — drives ``power_iteration`` with a finite-difference
  resolvent; the BiCGSTAB inner loop *is* the
  :math:`A_{\rm loss}^{-1}` action.
* **MoC** (method of characteristics) — drives ``power_iteration``
  with a track-based inner sweep as its resolvent, via the same
  late-bound boundary :class:`StreamingCollisionOperator.solve` exposes.
* **Homogeneous** — drives ``power_iteration`` over a direct linear
  solve; the analytical
  :func:`~orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous`
  is the closed-form algebra-of-record this family also realizes.

The :class:`~orpheus.numerics.iteration.KEigenvalue` adapter is **one
Layer-2b implementer** of this boundary, for callers who *have* a
natural :math:`(A, S, F)` triple and want to skip writing a full
solver class; its :meth:`~orpheus.numerics.iteration.KEigenvalue.solve`
delegates its loop to ``power_iteration`` (one engine — see the
same-morphism analysis in :ref:`eigenvalue-posing`).  Making the
Layer-4 loop *literally* K/α-agnostic — relocating the eigenvalue
scaling out of the K-specific
:meth:`~orpheus.numerics.eigenvalue.EigenvalueSolver.compute_keff` and
renaming the K-flavoured Protocol methods to ``eigen_operator`` /
``mu_to_eigenvalue`` — touches all five families' Protocol surface and
is the **first step of the α-wave**, deferred under *unify-after-two*
because only the k-row exists today (the α-generic agnostic relocation
future seam, :ref:`eigenvalue-posing`).

.. _choosing-inverse-realisation:

Choosing the :math:`A^{-1}` realisation
=======================================

The outer loop applies :math:`A^{-1}` without ever asking how it is
realised: since the #226 taxonomy **step 3**, neither iteration
primitive takes an ``inverter`` callable.  The *solver* layer builds
the inverse **once** — as an operator, ``A.inverse()`` (or an explicit
family member) — and the drivers **apply** it through the ONE
seeded-apply contract
(:class:`~orpheus.numerics.iteration.SupportsSeededApply`).  The
family of inverse KINDS — the direct sweep, the preconditioned
Green splitting, the dense LU materialisation, the generic
solve-backed wrapper, and the structure-keyed ``.inverse()`` factory
that picks among them — is documented once, in
:doc:`/theory/foundations/operator_inverse_family`; this chapter
records only the SN-specific consumption:

* the sweep-invertible :math:`(L+C)`
  (:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`)
  **shadows the factory by MRO**: ``(L+C).inverse()`` returns the
  direct triangular
  :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` — the
  WDD sweep of :doc:`slab_one_group`, reached as an operator;
* :class:`~orpheus.numerics.iteration.KrylovAcceleration`'s
  ``preconditioner`` parameter (renamed from ``inverter`` — a GMRES
  *left preconditioner* :math:`M \approx \bigl(A - \sum_i g_i\bigr)^{-1}`
  approximates the FULL within-group system inverse over the variadic
  gains — for the SN within-group system,
  :math:`(L+C-S-N_{2n}-B)^{-1}` — a
  different object from the iteration's step inverse; the old name was
  a category mistake) defaults to ``A.inverse().apply`` — the sweep —
  when ``A`` is invertible.

.. note:: **Re-framed (2026-06-12, Issue #195).**

   This choice was historically believed to be a *correctness* fork on
   curvilinear meshes — the two-distinct-closures picture, under which
   the WDD sweep's fixed point differed from the symmetric ``apply``
   closure and routing to Krylov-on-``apply`` was what "closed
   ERR-026".  ERR-058 (#195) showed the curvilinear wrong fixed point
   was the *closure-seed* family; once the seeds are fixed the sweep
   and the matvec are ONE discrete system and SI ``A.solve``
   :math:`\equiv` Krylov-on-``apply`` **bit-identical**
   (:ref:`sn-err-058-closure-seed-closeout`).  The choice of inverse
   realisation survives — as a **type choice, not a callable
   injection** — but it is a choice of **rate**, not of fixed point:
   the standard transport-Krylov win as :math:`c \to 1`
   (:cite:`AdamsLarsen2002`).

Because every family member conforms to the one seeded-apply
contract, the driver is **inversion-strategy-agnostic**: the same
:class:`SourceIteration` runs the synthetic L0 case (a dense matrix
inverse), the L1 SN case (``(L+C).inverse()`` → the WDD sweep), and
the Krylov-on-``apply`` case — with no re-implementation, because the
inversion strategy rides the *type* of the step operator, not a
branch inside the driver.


Verification hooks
==================

* **Never one group.** A 1-group eigenvalue is flux-shape independent
  (:math:`k = \nSigf{}/\Sigma_a` pointwise) — it cannot see group
  coupling at all.  Every multigroup verification case uses
  :math:`\ge 2` groups with an **asymmetric** scattering matrix, so a
  transpose error (Mode 2/6: ``SigS`` vs ``SigS^T``) moves the
  answer (the degeneracy-traps gotcha, :doc:`index`).
* **The infinite-medium anchor.** The analytical
  :math:`\kinf = \lambda_{\max}(A^{-1}F)` built from the same
  ``SigS``/:math:`\chi`/:math:`\nSigf{}` data
  (:ref:`mg-eigenvalue-problem`) pins the group algebra
  independently of transport: a homogeneous reflective slab must
  reproduce it to solver tolerance.
* **The addition-theorem L0 gates.** The spherical-harmonic
  identities behind :eq:`pn-scatter` are pinned at :math:`\ell \le 3`
  by ``test_spherical_harmonics_addition_theorem_L3`` /
  ``test_spherical_harmonics_orthogonality_L3``; the P\ :sub:`ℓ` MMS
  convergence suite (:file:`tests/sn/verification/mms/test_mms_aniso.py`)
  exercises the full moment chain.
* **The frozen snapshots.** The 11 regression snapshots at
  ``tests/sn/regression/snapshots/`` gate every operator extraction
  (Wave D) and refactor (Wave 1 :math:`R\Lambda M`) bit-identically —
  including the 2-group P\ :sub:`1` anisotropic case
  ``slab_2g_p1_aniso_dd_n20``.


What broadens next
==================

* **Space** (2-D/3-D Cartesian, :doc:`cartesian_multid`): streaming
  becomes a true gradient, the sweep
  becomes a wavefront over a dependency DAG, and the within-group
  iterate can live in moment space (angular windowing,
  :ref:`sn-angular-windowing`) — the energy machinery of this chapter
  rides along unchanged.
* **Curvature** (spherical/cylindrical, :doc:`curvilinear_one_group`):
  the angular cell balance
  activates redistribution and the starting-direction state; the
  group axis again rides along unchanged
  (:doc:`curvilinear_multigroup`).
