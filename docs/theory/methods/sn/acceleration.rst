.. _sn-acceleration:

Consistent Diffusion Synthetic Acceleration
===========================================

This chapter is the S\ :sub:`N` book's acceleration rung.  Source
iteration converges at a rate that degrades to uselessness in exactly
the regime reactor physics cares about most — optically thick,
strongly scattering media — and Diffusion Synthetic Acceleration (DSA)
is the classical cure.  The load-bearing word is **consistent**: not
"pair the sweep with a diffusion solve" but "pair it with *the*
diffusion solve that is the discrete moment-reduction of the sweep's
own operator".  Consistency is a **theorem** here, proven symbolically
in the algebra of record
(:mod:`orpheus.derivations.discrete.sn.dsa`) and realized
entry-for-entry in production (:mod:`orpheus.sn.acceleration.dsa`),
never a hand-imposed stencil.

The chapter builds the story in the order the mathematics demands: the
Fourier analysis that says *why* an accelerator is needed and what the
best one can achieve; the derivation that makes "consistent" precise;
the restriction/prolongation pair that turns out to have already
existed as the angular frame's :math:`\ell=0` faces; the f-form that
makes the correction vanish at convergence (and the deep verification
consequence of that vanishing); the P1 extension for anisotropic
scattering; the two acceleration postures; the three consistency
discoveries the build surfaced; and the rate/stability evidence that
pins every claim to a measured number.

.. admonition:: Key Facts
   :class: tip

   * **Source iteration collapses as** :math:`c \to 1`.  The SI
     spectral radius is :math:`\rho_{\rm SI} = c` (the scattering
     ratio), so a pure scatterer needs :math:`O(10^{m})` iterations for
     :math:`m` digits.  The slow modes are flat in **space and angle**
     (:math:`\lambda \approx 0`) — precisely the modes a diffusion
     equation resolves in one solve (:ref:`sn-dsa-the-fourier-story`;
     :eq:`sn-dsa-si-fourier`).
   * **Consistent DSA caps the rate at** :math:`\rho_{\rm DSA} \le
     0.2247\,c` **for every mesh**, independent of cell optical
     thickness :math:`\sigma_t h` (:cite:`AdamsLarsen2002` eq. (3.65);
     :eq:`sn-dsa-consistent-fourier`).  Measured production: 2110 SI
     iterations :math:`\to` 16 with DSA at :math:`c = 0.99`
     (:ref:`sn-dsa-rate-and-stability`).
   * **"Consistent" = reduce-the-discrete, NOT
     discretize-the-reduced.**  The low-order operator is the two-moment
     (:math:`\ell \le 1`) Galerkin reduction of the **assembled** DD
     transport operator followed by a Schur elimination of the current
     block — Larsen's four-step (:cite:`Larsen1982a`), proven to
     reproduce his row (27)/(23a–f)
     (:eq:`sn-dsa-consistent-low-order`).  Fixed-point compatibility
     **alone is insufficient** (Reed's scheme had it and still
     diverges) — the operative property is derived-by-moment-reduction
     (:ref:`sn-dsa-consistency-is-derived`).
   * **The correction operator mints NOTHING.**  Restriction and
     prolongation are the :math:`\ell = 0` faces of the existing
     spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`
     (``Quadrature.angular_frame(0)``), pinned 0-ULP against
     :meth:`~orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular`;
     the P1 arm's :math:`\ell = 1` row is ``angular_frame(1)``'s slab
     component, :math:`\mu` bit-exactly (:ref:`sn-dsa-restriction-prolongation`;
     :eq:`sn-dsa-restriction`).
   * **The correction** :math:`\to 0` **at convergence**, so DSA is
     correctness-safe *by construction* — a wrong accelerator degrades
     the **rate**, never the answer (:eq:`sn-dsa-correction-vanishes`).
     The deep consequence: fixed-point-invariance gates are
     **structurally blind** to 7 of the 8 canonical implementation
     errors; the verification weight rides **object gates** and **rate
     gates** (:ref:`sn-dsa-the-f-form`).
   * **Two postures, one operator.**  SI+DSA (the ``corrector`` hook on
     :class:`~orpheus.numerics.iteration.SourceIteration`) and the
     Krylov left-preconditioner :math:`M = (I + \mathcal{C}) \circ
     (L+C)^{-1}` (:eq:`sn-dsa-krylov-preconditioner`; the first
     re-enabled preconditioner, folding #200) consume the **same**
     :class:`~orpheus.sn.acceleration.dsa.DSACorrection`
     (:ref:`sn-dsa-both-postures`).
   * **Scope: 1-D slab, DD, within-group, P0 + P1** (arm 1).  The build
     refuses everything else loudly.  Deferred with reasons:
     the LD-consistent arm (structurally unspellable without the M4S
     reduction — R5a), 2-D Cartesian, curvilinear (no stability theory
     exists — :cite:`AdamsLarsen2002` p. 79), and the k-outer
     (:ref:`sn-dsa-honest-scope`).
   * **Three consistency discoveries** the build surfaced, each a
     teaching artifact: the :math:`\sigma_r`-fold (ERR-070; #215's
     class), the weighted-diamond partial-consistency negative control,
     and the singular composite sweep-inverse (ERR-071)
     (:ref:`sn-dsa-three-discoveries`).

.. contents:: On this page
   :local:
   :depth: 2


.. _sn-dsa-the-fourier-story:

The Fourier story — why acceleration, and how good it can get
=============================================================

Everything about DSA follows from one spectral fact about source
iteration.  Consider the within-group, isotropic-scattering,
fixed-source problem in an infinite homogeneous medium, in mean-free
path units (:math:`\sigma_t = 1`, :math:`\sigma_{s0} = c`).  Source
iteration sweeps the transport equation with the previous scalar flux
in the scattering source, then re-integrates:

.. math::

   \mu\frac{\partial\psi^{l+1/2}}{\partial x} + \psi^{l+1/2}
     = c\,\phi_0^{l} + S,
   \qquad
   \phi_0^{l+1} = \tfrac12\!\int_{-1}^{1}\psi^{l+1/2}\,d\mu .

A Fourier mode :math:`\phi_0 \propto e^{i\lambda x}` is attenuated per
iteration by the eigenvalue

.. math::
   :label: sn-dsa-si-fourier

   \omega_{\rm SI}(\lambda) = \frac{c}{2}\int_{-1}^{1}
     \frac{d\mu}{1 + \lambda^2\mu^2}
   = c\,\frac{\arctan\lambda}{\lambda},
   \qquad
   \rho_{\rm SI} = \sup_{\lambda}\,|\omega_{\rm SI}(\lambda)| = c .

.. vv-status: sn-dsa-si-fourier documented
.. (literature transcription — the classic SI dispersion relation,
..  :cite:`Larsen1982a` eq. (4)/(7), :cite:`AdamsLarsen2002` eq. (2.17);
..  the measured attainment ρ_est(SI) ≈ c is the plain-SI honesty
..  control in test_dsa_rate.py::TestD11SpectralRadius.)

The supremum sits at :math:`\lambda \to 0`: the **slowest** mode is the
one that is flat in space, and (because :math:`\arctan\lambda/\lambda
\to 1`) flat in angle too — a near-constant angular flux.  As
:math:`c \to 1` the rate :math:`\rho_{\rm SI} = c \to 1` and SI stalls:
a pure scatterer with a target of :math:`10^{-m}` needs :math:`\approx
2.3\,m/(1-c)` iterations (:cite:`AdamsLarsen2002` eq. (2.53)).  At
:math:`c = 0.999` that is over two thousand sweeps per digit.

The discrete DD :math:`S_N` iteration has the *same* structure, with
the continuum mode axis merely reparametrized
(:cite:`AdamsLarsen2002` eqs. (3.28)–(3.30)):

.. math::

   \omega_{\rm SI}(\Lambda) = \frac{c}{2}\sum_n
     \frac{w_n}{1 + \mu_n^2\Lambda^2},
   \qquad
   \Lambda = \frac{2}{\sigma_t h}\tan\!\Big(\frac{\sigma_t h\,\lambda}{2}\Big) ,

so :math:`\rho_{\rm SI} = c` for **any** quadrature order and cell size
(:math:`\sum_n w_n = 2` in the ORPHEUS raw slab convention).  Adding
leakage — a finite vacuum-bounded system — pulls :math:`\rho` slightly
below :math:`c`, and that help *diminishes* as the system grows
optically thick (:cite:`AdamsLarsen2002` p. 51).  A reflective or
periodic problem removes the leakage entirely and **realizes the
infinite-medium worst case** — which is exactly why acceleration
trouble surfaces first on reflected problems (this is the physical
root of the reflective-stability tier, :ref:`sn-dsa-rate-and-stability`).

The DSA fix
-----------

Because the slow modes are flat in space *and* angle, they are exactly
the modes a diffusion equation resolves in a single solve.  DSA
interposes a low-order (diffusion) solve between sweeps that
annihilates those modes.  For the **continuous** P1 low-order operator
the analysis gives the celebrated bound (:cite:`Larsen1982a` eq. (7),
:cite:`AdamsLarsen2002` eqs. (2.50)–(2.51)):

.. math::
   :label: sn-dsa-continuum-bound

   \omega_{\rm DSA}(\lambda) = \frac{3c}{\lambda^2 + 3(1-c)}
     \left[\Big(\tfrac{\lambda^2}{3} + 1\Big)\frac{\arctan\lambda}{\lambda}
       - 1\right],
   \qquad
   \rho_{\rm DSA} = \sup_\lambda |\omega_{\rm DSA}(\lambda)|
     \le 0.2247\,c .

.. vv-status: sn-dsa-continuum-bound documented
.. (literature transcription — the Gelbard–Hageman/Larsen continuum
..  bound; the c-scaling is sharp only near c = 1, and at c = 1 the sup
..  sits at an intermediate λ ≈ 2.5, NOT at λ = 0. This is the target the
..  discrete consistent operator must respect from below.)

The number ``0.2247`` is the whole game: DSA converts an
:math:`O(1/(1-c))` iteration count into a **bounded** one (≈ 1.54
iterations per digit, :cite:`AdamsLarsen2002` eq. (2.54)) for *every*
scattering ratio.  Note the sup no longer sits at :math:`\lambda = 0`
(DSA killed the flat mode) but at an intermediate frequency
:math:`\lambda \approx 2.5` where the diffusion approximation is
weakest.

The catch is that the *discrete* accelerator must reproduce this.  The
sharpest statement in the corpus is the discrete consistent-DD rate,
:cite:`AdamsLarsen2002` eq. (3.65):

.. math::
   :label: sn-dsa-consistent-fourier

   \omega(\Lambda) = \omega_{\rm SI}(\Lambda)
     - \frac{c\,\bigl(1 - \omega_{\rm SI}(\Lambda)\bigr)}
            {1 - c + \tfrac13\Lambda^2},
   \qquad
   \rho < 0.2247 \quad\text{for all } \sigma_t h,\ N,\ 0 < c \le 1 .

The decisive property is **Σ_th-independence**: the spectral radius of
the *consistent* discrete operator does not degrade as cells thicken.
This is the quantitative gate the production accelerator is measured
against (:ref:`sn-dsa-rate-and-stability`, D11); it is a **rate**
claim (flux-shape independent, so a 1-group model problem is
legitimate — declared per the V&V principles), never an eigenvalue or
flux-shape claim.

What was tried and failed — Reed's scheme
-----------------------------------------

The trap the whole theory exists to avoid is pairing the sweep with an
**independently discretized** diffusion operator.  Reed's original
synthetic method (:cite:`AdamsLarsen2002` [51]) used DD transport with
a conventional **cell-centered** diffusion discretization.  It is
*fixed-point compatible* — its converged solution equals the discrete
SN solution — yet its spectral radius is
(:cite:`Larsen1982a` p. 49–50):

.. math::

   \rho_{\rm Reed} \ge \frac{c}{\tfrac{4}{3h^2} + 1 - c}
   \;\xrightarrow{\ h > h^*\ }\; > 1,
   \qquad h^* = \frac{2}{\sqrt{3(2c-1)}}\ \approx\ 1.15\ \text{mfp at } c\approx 1 .

It **diverges** for cells thicker than about one mean free path.
Alcouffe diagnosed the mechanism precisely (:cite:`Alcouffe1977`
p. 348, restated by :cite:`AdamsLarsen2002` as the (3.41) monotone
degradation): the DD P1-moment equation couples the **cell-center
current** to **cell-edge scalar-flux differences** — a staggered
Fick's law — while the cell-centered diffusion operator evaluates the
current at cell edges.  The two discrete diffusion limits disagree by
:math:`O(1)` at finite :math:`h`, so the correction source stays finite
exactly where it should vanish, and the accelerator overshoots into
instability.

Two lessons hard-earned by the field, both load-bearing here:

* **Inconsistency has no safe dose.**  A cell-*edge* (rather than
  cell-center) inconsistent scheme degrades gracefully toward
  :math:`\rho \to \rho_{\rm SI}` — "convergent but ineffective, it
  performs like source iteration" (:cite:`AdamsLarsen2002` eq. (3.43));
  a partially consistent scheme (right operator shape, wrong closure
  constant) diverges at a threshold that merely *moves* to thicker
  cells as the inconsistency shrinks (:cite:`McCoyLarsen1982` Table II;
  the negative control, :ref:`sn-dsa-three-discoveries`).
* **Both the operator AND its boundary conditions must be consistent.**
  McCoy & Larsen's morals (:cite:`McCoyLarsen1982` §IV): "it is
  essential that both the diffusion equation and its boundary
  conditions be compatible with the transport calculation."  This is
  why the Marshak boundary rows are *derived*, not borrowed from a
  standalone diffusion solver (:ref:`sn-dsa-boundary-rows`).


.. _sn-dsa-consistency-is-derived:

Consistency is derived, not chosen
==================================

"Consistent" is the single most abused word in the DSA literature, so
this section fixes its precise meaning and shows it is a **computed**
property of the sweep's own operator.  Three levels of "consistent with
the discrete SN equations" appear in the sources, strongest last
(:cite:`AdamsLarsen2002` §III.B; the literature memo §6.1):

#. **Fixed-point compatibility** — the low-order equation's converged
   solution equals the discrete SN solution.  *Necessary only.*  Reed's
   unstable scheme had exactly this and still diverges
   (:cite:`Morel1982` p. 37: "consistency is a necessary condition for
   convergence, but it is not a sufficient one").
#. **Diffusion-limit compatibility** — Alcouffe's diagnostic: the
   correction source (the diffusion defect of the transport iterate)
   must vanish *identically* on the discrete diffusion limit at any
   :math:`h`.  Alcouffe's pivotal identity (:cite:`Alcouffe1977`
   eq. (27)) makes it concrete: for the consistent scheme the
   correction source is *exactly* a second difference of the transport
   iterate's **second** Legendre moment,
   :math:`R_i = -(2D/h^2)(\tilde\phi_{2,i+3/2} - 2\tilde\phi_{2,i+1/2}
   + \tilde\phi_{2,i-1/2})`, so :math:`R \equiv 0` wherever
   :math:`\tilde\phi_2 \equiv 0` (the P1 limit) — at *any* mesh size.
#. **Moment reduction / Schur complement** — the operative definition,
   and the one ORPHEUS computes.  The low-order operator is *derived*
   from the assembled discrete SN operator by the discrete-P1
   projection: :cite:`AdamsLarsen2002` (1.37)–(1.40) write it as
   :math:`M = E\,D^{-1}K`, with :math:`K` the discrete-P1 analysis
   (:math:`\ell \le 1` moments of **both** the balance and the closure),
   :math:`E` the linear-in-angle prolongation, and :math:`D = K(L-S)E`
   the **projected assembled operator**.

The phrase that captures the trap is **reduce-the-discrete, never
discretize-the-reduced**.  These do not commute: taking the P1 moments
of the *assembled* DD stencil produces an edge-centered operator with
:math:`h`-baked closure coefficients; writing down "the" P1 diffusion
equation and *then* discretizing it produces a cell-centered operator —
Reed's, which diverges.  The order of operations *is* the consistency.

Larsen's four-step, executed symbolically
-----------------------------------------

The algebra of record
(:mod:`orpheus.derivations.discrete.sn.dsa`) executes Larsen's
four-step procedure (:cite:`Larsen1982a` §II) **symbolically over a
general symmetric quadrature** — proving, not transcribing, the
low-order operator.  The four steps, with the weighted-diamond (WD,
:math:`\alpha_{mi} = a_i\,\mathrm{sgn}\,\mu_m`; DD is :math:`a_i = 0`)
equations:

* **Step 1 — the discretized transport iteration.** The per-ordinate
  balance and WD closure (:cite:`Larsen1982a` (10a/10b)):

  .. math::

     \frac{\mu_m}{h_i}\bigl(\psi^{l+1/2}_{m,i+1/2} - \psi^{l+1/2}_{m,i-1/2}\bigr)
       + \sigma_{Ti}\psi^{l+1/2}_{mi}
       = \sigma_{S0i}\phi^{l}_{0i} + 3\sigma_{S1i}\mu_m\phi^{l}_{1i} + S_{mi},
     \\
     \psi^{l+1/2}_{mi} = \tfrac{1+\alpha_{mi}}{2}\psi^{l+1/2}_{m,i+1/2}
       + \tfrac{1-\alpha_{mi}}{2}\psi^{l+1/2}_{m,i-1/2}.

* **Step 2 — take the** :math:`L_0` **and** :math:`L_1` **moments of
  BOTH equations.** Reducing the balance *and* the closure (not just the
  balance — this is what carries the scheme's :math:`\alpha`-weights
  into the low-order operator, via :math:`\rho_i = L_1[\alpha_i] =
  \sum_m \mu_m\alpha_{mi}\omega_m`) gives four moment equations
  (:cite:`Larsen1982a` (16a–d)).  The classic constants emerge as
  *quadrature properties*, each at its true mechanism, never as
  transcribed literals:

  * the :math:`1/3` and :math:`2/3` in the P1 balance moment are the
    **Legendre-recursion** coefficients of :math:`\mu^2 = (2P_2 +
    P_0)/3` (``prove_p_recursion_lemma`` — quadrature-independent);
  * the :math:`1/3` in :math:`D` and the :math:`3` in the closure
    coupling are the **quadrature moment** :math:`W_2 = \sum_m
    \mu_m^2\omega_m` (exactly :math:`1/3` for any rule integrating
    :math:`\mu^2` exactly under :math:`W_0 = 1`);
  * the annihilation identities :math:`L_0\gamma = L_1\gamma = L_0\beta
    = L_1\beta = 0` (``derive_closure_weight_identities``) hold for
    every symmetric quadrature — Larsen's printed "3" in (14b) is
    :math:`1/W_2` in disguise.

* **Step 3 — promote the iteration index.** Replace :math:`l+1/2 \to
  l+1` on every :math:`\phi_0, \phi_1` slot (edge *and* average) and the
  scattering source; **lag** at :math:`l+1/2` the :math:`\phi_2` terms
  and the opaque closure functionals :math:`L_0[\gamma\psi]`,
  :math:`L_0[\beta\psi]`.  The derivation carries the lagged
  functionals as *opaque symbols* precisely so the promotion cannot
  touch them — "the slot assignment IS the method": expanding a lagged
  term in moment coordinates would silently promote part of it, a
  derivation bug the module's own machinery catches.

* **Step 4 — subtract and reduce to one tridiagonal.**  Promoted minus
  original, with edge corrections :math:`f_{n,i+1/2} = \phi^{l+1}_{n} -
  \phi^{l+1/2}_{n}`.  The :math:`\phi_2` and closure-residual terms
  **cancel** in the subtraction, so the correction system is closed in
  :math:`f_0, f_1` alone; Schur-eliminate the current corrections
  :math:`f_1` and the cell-average corrections onto the edge scalar-flux
  corrections :math:`f_0`.

The coefficients that result are (:cite:`Larsen1982a` (23a–f)):

.. math::
   :label: sn-dsa-coefficients

   \hat\sigma_{Ri} &= \frac{\sigma_{Ti} - \sigma_{S0i}}
       {1 + \tfrac32\rho_i(\sigma_{Ti} - \sigma_{S0i})h_i},
   \qquad
   \hat\sigma_{Si} = \frac{\sigma_{S0i}}
       {1 + \tfrac32\rho_i(\sigma_{Ti} - \sigma_{S0i})h_i}, \\
   D_i &= \frac{1}{3(\sigma_{Ti} - \sigma_{S1i})} + \tfrac12\rho_i h_i,
   \qquad
   a_i = \frac{\sigma_{S1i}}{\sigma_{Ti} - \sigma_{S1i}}, \\
   g_{0i} &= \hat\sigma_{Si}\,h_i\,d_{0i},
   \qquad
   g_{1i} = a_i\,d_{1i},

.. (structural — the transcribed comparison target (23a–f); the derived
..  row is PROVEN equal to the Larsen row built from these by
..  tests/derivations/test_dsa_rules.py::test_dd_instance_coefficients
..  (the α=0 DD member) — a foundation-level SymPy identity, not a
..  solver claim.)

where the DSA residual sources :math:`d_{ni} = \phi^{l+1/2}_{ni} -
\phi^{l}_{ni}` are the **raw** sweep displacement moments (the
:math:`\sigma`-weighting lives in the source map, applied once).  The
interior consistency condition — obtained by writing :math:`f_1` at a
shared edge from **both** adjacent cells and demanding continuity — is
the tridiagonal diffusion row (:cite:`Larsen1982a` (27)):

.. math::
   :label: sn-dsa-consistent-low-order

   &-\frac{D_{i+1}}{h_{i+1}}\bigl(f_{0,i+3/2} - f_{0,i+1/2}\bigr)
     + \frac{D_i}{h_i}\bigl(f_{0,i+1/2} - f_{0,i-1/2}\bigr) \\
   &\quad + \tfrac14\bigl[\hat\sigma_{R,i+1}h_{i+1}(f_{0,i+3/2}+f_{0,i+1/2})
     + \hat\sigma_{Ri}h_i(f_{0,i+1/2}+f_{0,i-1/2})\bigr] \\
   &= \tfrac12(g_{0,i+1} + g_{0i}) - (g_{1,i+1} - g_{1i}),
   \qquad i = 1,\dots,I-1 .

.. (the main-theorem target — the derived shared-edge f₁-continuity row
..  is PROVEN a scalar multiple of this transcribed Larsen (27) with
..  coefficients (23a–f) by
..  tests/derivations/test_dsa_rules.py::test_main_theorem_interior_row_is_larsen_27,
..  and the production build is pinned entry-for-entry against the
..  reference builder by
..  test_dsa_low_order.py::test_low_order_matches_reference_builder.
..  A derivation identity + a build tie, not an eigenvalue/flux claim.)

with the accelerated cell-average updates (:cite:`Larsen1982a`
(28a/28b)):

.. math::
   :label: sn-dsa-cell-update

   \phi^{l+1}_{0i} &= \phi^{l+1/2}_{0i}
     + \bigl(\tfrac12 - \tfrac34\rho_i\hat\sigma_{Ri}h_i\bigr)
       (f_{0,i+1/2} + f_{0,i-1/2}) + \tfrac32\rho_i g_{0i}, \\
   \phi^{l+1}_{1i} &= \phi^{l+1/2}_{1i}
     + \bigl(\tfrac12\rho_i - D_i/h_i\bigr)(f_{0,i+1/2} - f_{0,i-1/2})
     + g_{1i}.

.. (the derived cell-average updates, PROVEN equal to the printed forms
..  by tests/derivations/test_dsa_rules.py::test_update_relations_are_larsen_28;
..  for the DD member ρ=0 they collapse to the edge average
..  ½(f₀_{i-1/2}+f₀_{i+1/2}) and the (28b) moment-1 update the P1 arm
..  consumes. A foundation-level identity.)

**The DD member** (:math:`\alpha = 0 \Rightarrow \rho = 0`) is the
object arm 1 wires: :math:`\hat\sigma_R = \sigma_t - \sigma_{s0}`,
:math:`\hat\sigma_S = \sigma_{s0}`, :math:`D = 1/[3(\sigma_t -
\sigma_{s1})]`, :math:`a = \sigma_{s1}/(\sigma_t - \sigma_{s1})` — the
edge-centered scheme with the :math:`\tfrac14(1,2,1)` three-point
removal mass, which is exactly Alcouffe's DD result recovered as the
:math:`a \to 0` member of the *proven* WD family (nothing DD-specific
is re-derived, and none of Alcouffe's errata-bearing printed forms are
consumed — see the note below).  For :math:`\rho \ne 0` the four-step
generates *h-dependent* coefficients (:math:`D` grows with
:math:`\rho h`, :math:`\hat\sigma_R` is :math:`h`-damped) — proof that
"consistent" is not "plug in the analytic :math:`D` and
:math:`\sigma_R`", it *bakes the closure weights in*.

.. note:: **Larsen is the transcription reference; Alcouffe is a
   cross-check only.**  :cite:`Alcouffe1977`'s printed discrete pairs
   (17) and (23) carry sign errata (the leading signs of the correction
   source and the low-order source term; literature memo §1.5,
   resolved unambiguously by the continuous scheme and each pair's
   stated convergence identity).  The algebra of record transcribes its
   target forms only from :cite:`Larsen1982a` and recovers Alcouffe's
   DD scheme as the :math:`a \to 0` member — so no errata-bearing form
   is ever consumed.  Never copy Alcouffe's printed (17)/(23) signs.

.. _sn-dsa-boundary-rows:

The boundary rows — Marshak is the boundary Fick
------------------------------------------------

The interior tridiagonal is only consistent if its boundary rows are
too (:cite:`McCoyLarsen1982` §IV: both the operator and its BCs).  The
correction-equation boundary rows come from the **same** moment
reduction applied to the **half-range** current, under the shared
trace measure :math:`|\Omega\cdot\hat n|\,w`.  The correction edge
angular flux is the :math:`\ell \le 1` synthesis (:cite:`Larsen1982a`
(33)) — Larsen's honest "3" is again :math:`1/W_2`:

.. math::

   \Psi_m = f_0 + \frac{\mu_m}{W_2}\,f_1 .

At a left vacuum boundary the zero-incident condition is imposed on
:math:`\Psi` in the **Marshak** (half-range current) sense:

.. math::
   :label: sn-dsa-marshak

   0 = \sum_{\mu_m > 0}\mu_m\omega_m\,\Psi_m
     = \gamma_N\,f_{0,1/2} + \frac{W_2^+}{W_2}\,f_{1,1/2},
   \qquad
   \gamma_N = \sum_{\mu_m > 0}\mu_m\omega_m,

.. (the Marshak (38a) + reflecting (39a) boundary rows; the coefficients
..  γ_N and W₂⁺/W₂ are DERIVED from the half-range reduction and checked
..  against the printed (γ_N, ½) by
..  tests/derivations/test_dsa_rules.py::test_boundary_rows.
..  A derivation identity — the discrete Marshak, not a solver claim.)

so Marshak is the boundary analog of Fick: a closure that eliminates
the incoming current, not a reconstruction.  The reflecting row is
:math:`f_{1,\text{edge}} = 0` (:cite:`Larsen1982a` (39a)).  Both rows
are **closed one-sidedly** by eliminating the boundary-edge :math:`f_1`
via the boundary cell's own relation (:cite:`Larsen1982a` (25)/(26),
``derive_one_sided_f1_forms``), yielding a row in :math:`f_0` alone.

Two discretisation subtleties that are quiet consistency breaks if
ignored (:cite:`AdamsLarsen2002` §III.B item M2-4):

* :math:`\gamma_N = \sum_{\mu_m>0}\mu_m\omega_m` is the **discrete**
  half-range current — measurably *not* the continuum :math:`1/4`
  (:math:`\gamma_N \approx 0.2606` for :math:`S_4` Gauss–Legendre under
  :math:`\sum\omega = 1`).  Substituting the continuum value at small
  :math:`N` breaks consistency.
* the :math:`f_1` coefficient :math:`W_2^+/W_2 = 3\sum_{\mu>0}
  \mu_m^2\omega_m` equals the printed :math:`1/2` **only** because
  :math:`W_2 = 1/3` for a rule integrating :math:`\mu^2` exactly — it is
  a quadrature property, not a constant.

.. _sn-dsa-r4-derived-vs-landed:

Why the edge-centered system, and not the diffusion module's operator
---------------------------------------------------------------------

ORPHEUS already ships an in-algebra diffusion loss operator
(:mod:`orpheus.diffusion`, #290).  Is *that* the consistent partner?
The 3a derivation answered by computation, and ruling **R4**
(2026-07-26) recorded the verdict: **no** — the two systems coexist by
*defining law*, not as a twin path.

The landed diffusion loss is the right **standalone** discretization: a
cell-centered RT0/harmonic-mean, M-matrix operator chosen for
*accuracy* of a diffusion answer.  Measured as an *accelerator*, it is
the :cite:`AdamsLarsen2002` (3.43)/(3.44) inconsistent class — spectral
radius up to :math:`\rho \approx 54.7` (divergent) for :math:`\sigma_t
h \ge 2`, the historical spike's regime (D2 characterization,
``.claude/plans/archive/dsa_d2_characterization.md``).  The **derived**
edge-centered system is unconditionally stable (measured :math:`\rho
\le 0.181` over :math:`\sigma_t h \in [0.1, 30]`, :math:`c \le 0.99`,
vacuum and reflective).  The structural diffs the D2 report confirmed:

.. list-table:: Derived (consistent) vs landed (standalone) low-order — the R4 diff
   :header-rows: 1
   :widths: 22 39 39

   * - Feature
     - Derived edge-centered (DSA)
     - Landed cell-centered (standalone)
   * - Unknown home
     - scalar flux :math:`f_0` on the :math:`K+1` **edges**
     - scalar flux :math:`\phi` on the :math:`K` **cells** (+ trace)
   * - Removal mass
     - consistent :math:`\tfrac14(1,2,1)\hat\sigma_R h` — off-diagonal
       flips sign at thick cells (the consistency fingerprint)
     - lumped diagonal :math:`\sigma_R V` (M-matrix)
   * - Harmonic mean
     - **none** — an edge unknown straddles a material-homogeneous cell
     - required — a cell unknown straddles material faces
   * - :math:`D` definition
     - within-group P1 **self**-scatter :math:`1/[3(\sigma_t -
       \sigma_{s1}^{g\to g})]`
     - total P1 out-scatter :math:`\Sigma_{\rm tr}` (they coincide only
       for isotropic data)
   * - Boundary
     - discrete Marshak :math:`\gamma_N \approx 0.2606` (S\ :sub:`4`)
     - continuum :math:`\tfrac14`
   * - As an accelerator
     - :math:`\rho \le 0.181` (unconditional)
     - :math:`\rho` up to 54.7 — **divergent** for :math:`\sigma_t h
       \ge 2`

Because the coefficients are properties of the **SN** discretization
(the :math:`\gamma_N` half-range moment, the :math:`W_2` quadrature
moment, the WD :math:`\rho`), the production system lives on the SN
side (:class:`~orpheus.sn.acceleration.dsa.DSALowOrderSystem`), *not* in
:mod:`orpheus.diffusion` — ruling **R4b**.  The reference builder
:func:`~orpheus.derivations.discrete.sn.dsa.build_consistent_dd_system`
realizes the proven :math:`(A, G)` matrices numerically, and the
production build is pinned against it entry-for-entry, so a drift in
either is a red gate, not a silent fork.  Two convention boundaries are
guarded numerically: the raw ORPHEUS slab quadrature
(:math:`\sum w = 2`) maps once to Larsen's :math:`\omega = w/2` (with
:math:`W_2 = 1/3` asserted), and the within-group data rows
(:math:`\sigma_{s0}^{g\to g}`, :math:`\sigma_{s1}^{g\to g}`) come from
the foldable cross-section accessors — the source of the first
consistency discovery, ERR-070 (:ref:`sn-dsa-three-discoveries`).


.. _sn-dsa-restriction-prolongation:

Restriction and prolongation minted nothing
===========================================

Every DSA implementation needs a **restriction** :math:`R` (angular
flux :math:`\to` scalar source for the low-order solve) and a
**prolongation** :math:`P` (scalar correction :math:`\to` angular
correction).  The elegance-detector reflex is to hand-build them as two
independent operators — and that is exactly the classic DSA
inconsistency where :math:`R` and :math:`P` disagree by a
:math:`4\pi/2` normalization factor and silently break conservation.
The structural verdict of the 3-P0 frame analysis
(``.claude/plans/archive/dsa_rp_frame_analysis.md``) is: **mint nothing** — the
pair already exists as the :math:`\ell = 0` faces of the
spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`.

Read straight off ``Quadrature.angular_frame(0)`` (the
:math:`\ell = 0` branch: :math:`Y_0 = 1`, addition-theorem factor
:math:`2\ell + 1 = 1`):

.. math::
   :label: sn-dsa-restriction

   (R\psi) = \sum_n w_n\,Y_0(\Omega_n)\,\psi_n = \sum_n w_n\psi_n = \phi_0,
   \qquad
   (P\phi_0)_n = (2\cdot 0 + 1)\,Y_0\,\frac{\phi_0}{\sum_n w_n}
     = \frac{\phi_0}{\sum_n w_n},

.. (structural — R is the ℓ=0 analysis face of the existing SH
..  GalerkinFrame; pinned 0-ULP against
..  AngularFlux.integrate_angular by
..  test_dsa_low_order.py::test_d8_restriction_is_the_frame_moment_row,
..  and its particle-conservation identity ⟨1,Rr⟩=⟨1,r⟩ by
..  test_d7_restriction_conserves_particles. A frame-law identity, not a
..  solver claim.)

so :math:`R` is the un-normalized moment-0 (the scalar flux) and
:math:`P` the normalized isotropic broadcast.  The projector :math:`\Pi
= P\circ R` is **W-self-adjoint under the plain angular quadrature
measure** — there is no *solution* weighting, which is exactly what
would make it Petrov–Galerkin — so the pair is **Galerkin** on the
angular axis, and the frame makes :math:`P = R^{+}` (and :math:`\Pi`
self-adjoint) a *theorem* rather than a hand-tuned normalization.

.. important:: **Galerkin here is the ANGULAR frame; do not conflate it
   with the Petrov–Galerkin homogenisation frame.**  The DSA
   restriction/prolongation are the Galerkin :math:`\ell = 0` faces
   because the scattering kernel :math:`\Sigma_s(\Omega\cdot\Omega')` is
   SO(3)-zonal, so the constant :math:`Y_0` *is* its trivial-irrep
   eigenbasis (:ref:`operator-adjoint` and the frame theory in
   :doc:`/theory/foundations/frame`).  This is a different axis from the
   *spatial/energy* flux-weighting of homogenisation and condensation,
   which is genuinely **Petrov–Galerkin** (test-weight
   :math:`\ne` trial-weight).  Same word, different frame.

In production, :math:`R` is
:meth:`~orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular`
(pinned 0-ULP: a separate einsum would differ at ULP from the frame's
fused reduction because floating-point addition is not associative).
The P1 arm's :math:`\ell = 1` analysis coefficient is
:meth:`quadrature.axis_cosines(0)
<orpheus.numerics.quadrature.Quadrature.axis_cosines>` — the polar
cosine :math:`\mu` per ordinate, read from the **coordinate** accessor.
A hand-rolled :math:`R = \sum_n w_n r_n` would be a fourth spelling of a
projector that already has three (the frame face, the in-sweep moment
accumulation, and the scattering :math:`\ell = 0` in-scatter) — the
Smell-16 twin path the frame exists to prevent.

.. note::

   ⛔ **This read used to be** ``angular_frame(1).table[:, 1, 1]``\ **,
   and #429's fused commit had to move it** (2026-09-02). That slot is
   the rectangular spherical-harmonic layout's :math:`(\ell, m) = (1,0)`
   Cartesian component — and a 1-D rule no longer binds that basis, so
   the index would be reading the FLAT Legendre table's :math:`\ell = 1`
   column *by accident of shape* rather than by contract
   (:ref:`sh-legendre-is-the-1d-family`). ⭐ The replacement is not a
   weaker source, it is the **right** one: "the :math:`\ell = 1`
   coefficient" is a COORDINATE question, and
   :meth:`~orpheus.numerics.quadrature.Quadrature.axis_cosines`
   is the coordinate accessor — which since 2026-09-01 *refuses* a
   suppressed axis, so a rule with no :math:`x` cosine fails loudly here
   instead of silently taking a zero. `[M]` bit-identical to the old
   slot on **5 of 5** Gauss–Legendre rules, which is what makes it a
   re-pointing rather than a change of value.

The consistency derivation (:ref:`sn-dsa-consistency-is-derived`) reads
cleanly in this language: the low-order operator is the Schur
complement of a **two-moment** (:math:`\ell \le 1`) Galerkin triple
product :math:`R_1\,A_{\rm high}\,P_1` on the *assembled* DD operator,
with the Fick/P1 closure being the Schur elimination of the
:math:`\ell = 1` (current) block.  The boundary arm is the *same*
Galerkin frame family over a *different* measure — the half-range trace
measure :math:`|\Omega\cdot\hat n|\,w` — where the partial-current pair
:math:`(J^+, J^-)` is the two-sign :math:`\ell = 0` half-range moment
and Marshak is the boundary analog of Fick.


.. _sn-dsa-the-f-form:

The f-form, and why fixed-point gates cannot see the accelerator
================================================================

Larsen derives DSA in two equivalent forms and recommends the
**correction (f) form** over the full-flux form (:cite:`Larsen1982a`
§II.3).  Instead of solving the low-order system for the full new flux
with a defect-corrected source (Alcouffe's original), it solves for an
additive **correction** :math:`f_0` whose source is the scattering
residual.  In the continuous prototype (:cite:`Larsen1982a` (6a/6b)):

.. math::

   -\tfrac13\frac{d^2 f_0^{l+1}}{dx^2} + (1-c)f_0^{l+1}
     = c\,\bigl(\phi_0^{l+1/2} - \phi_0^{l}\bigr),
   \qquad
   \phi_0^{l+1} = \phi_0^{l+1/2} + f_0^{l+1} .

Three advantages, all load-bearing (:cite:`Larsen1982a` p. 53–54):

* **Storage** — 5 arrays (:math:`f_0` edge; :math:`g_0, g_1,
  \phi_0^{l+1/2}, \phi_1^{l+1/2}`) vs 9 for the full-flux form.
* **Homogeneous boundary conditions** — because :math:`f_0 \to 0` at
  convergence, "any homogeneous boundary conditions (ones that admit
  the zero solution) are permissible", and the error problem's vacuum
  face is *exactly* Marshak with zero incident current.
* **Fixup robustness** — at convergence the correction row collapses to
  the identity "``0 = 0``", whereas the full-flux form reduces to the
  transport equations only if the closure holds in every cell.  A
  negative-flux fixup breaks the closure in fixed-up cells; the
  correction form is *insensitive to where the transport iterate came
  from*, which structurally resolves Alcouffe's fixup-incompatibility
  worry.

The correction :math:`\to 0` property, and its verification consequence
-----------------------------------------------------------------------

The source of the correction is the scattering residual :math:`c(\phi^{l+1/2}
- \phi^{l})`, which **vanishes at the fixed point**.  Writing the whole
correction operator as :math:`\Delta\psi \mapsto P\,[(28)]\,A_{\rm
low}^{-1}\,G\,R\,\Delta\psi` (restriction, low-order solve, cell
updates, prolongation), the input displacement :math:`\Delta\psi =
\psi^{l+1/2} - \psi^{l}` is zero at convergence, so:

.. math::
   :label: sn-dsa-correction-vanishes

   \psi^{l+1/2} = \psi^{l}
   \;\;\Longrightarrow\;\;
   \mathcal{C}\,\Delta\psi = 0
   \quad\text{(exactly, regardless of how wrong } A_{\rm low}\text{ is).}

.. (the correctness-safety property — a zero displacement maps to an
..  EXACT zero correction, pinned by
..  test_dsa_acceleration.py::TestD6CorrectionVanishes.
..  Closed-form identity (residual→0), not a solver claim; the D6 gate's
..  own blind spot — a dead R that returns 0 also passes trivially — is
..  closed by the non-trivial-first-iterate pairing.)

This is why DSA is **correctness-safe by construction**: a wrong
accelerator — a mis-scaled :math:`A_{\rm low}`, a sign-flipped
correction, a broken boundary row — degrades the *rate*, never the
converged answer.  But the same property has a sharp verification
consequence, and it is the organizing principle of the whole battery
(verification spec §0).  The **correction :math:`\to 0` partition**
splits the failure surface into two disjoint classes with disjoint
catchers:

.. list-table:: The correction→0 partition — where a bug lives decides who catches it
   :header-rows: 1
   :widths: 34 22 22 22

   * - Bug lives in…
     - Converged fixed point
     - Convergence rate
     - Caught by
   * - the within-group **transport** operator
       :math:`A = L+C-S-N_{2n}-B` (:eq:`sn-within-group-with-n2n`)
       (the :math:`\sigma_r`-fold; a sweep sign flip; a wrong closure
       fed to *both* sweep and low-order)
     - **CHANGES** (value-wrong)
     - —
     - **FP-invariance** gates + the routing sentinel
   * - the **accelerator** machinery (:math:`R`, :math:`P`,
       :math:`A_{\rm low}`, correction sign/scale, boundary rows)
     - **unchanged** — value-safe *by construction*
     - degrades / diverges
     - **object gates** + **rate/stability** gates

The consequence the plan-of-record had to internalize: a verification
plan that gates DSA correctness **only** on "converged flux
:math:`\equiv` SI fixed point" is blind to *every* accelerator-quality
error — those leave the fixed point **identically** unchanged (not
sub-floor; identically, because the correction is zero at the FP no
matter how wrong :math:`A_{\rm low}` is).  Of the **eight** canonical
implementation errors, exactly **one** (the :math:`\sigma_r`-fold) reds
the fixed-point gates; the other **seven** are caught only by the
object gates (:math:`R`-conservation, :math:`R/P` adjoint-consistency,
the entry-for-entry build tie) and the rate gates
(:ref:`sn-dsa-rate-and-stability`).  This is why the rate/stability
tier is **load-bearing, not supplementary** — it is the *only* catcher
for the majority of the plausible-error surface.  The correction
:math:`\to 0` gate itself (D6) proves the safety property, and its own
blind spot (a **dead** :math:`R` that returns 0 also passes
"correction :math:`\to 0`" trivially) is closed by pairing it with a
non-trivial first-iterate check.


.. _sn-dsa-p1-arm:

The P1 extension — accelerating the current under anisotropy
============================================================

P0-DSA accelerates only the :math:`\ell = 0` (scalar-flux) moment.
That is sufficient while the within-group :math:`\ell = 1` scattering
is weak, but it degrades predictably as scattering becomes forward-
peaked.  The mode-by-mode criterion is :cite:`AdamsLarsen2002`
eq. (7.14): at :math:`\lambda = 0` the per-moment SI eigenvalues are
:math:`\rho_n = \Sigma_{sn}/\Sigma_t`, and DSA removes the accelerated
moments, leaving :math:`\rho \approx \max_{n\,>\,\text{accelerated}}
\Sigma_{sn}/\Sigma_t`.  So P0-only DSA leaves the :math:`\ell = 1` mode
:math:`\Sigma_{s1}/\Sigma_t` iterating plainly — measured here as the
anisotropy ladder climbing from 14 to 86 iterations as
:math:`\sigma_{s1}/\sigma_{s0} \to 0.9`.  Morel showed the cure is to
accelerate the current :math:`\phi_1` as well (:cite:`Morel1982` §III;
:cite:`McCoyLarsen1982` Table V: P1 acceleration is
:math:`\beta`-independent at 5 iterations where P0-only needs 64).

Larsen's four-step already carries this: the moment-1 update
(:eq:`sn-dsa-cell-update`, the (28b) row) is derived alongside the
scalar update.  Ruling **R5b** (2026-07-26) wired the **d₁
moment-pair arm**.  Three pieces, each a *called single source*, no new
angular reduction minted:

* **the** :math:`\ell = 1` **restriction** :math:`d_1 = \sum_n w_n\mu_n
  \Delta\psi_n` — the frame's :math:`\ell = 1` analysis row
  (``angular_frame(1)`` slab component :math:`= \mu` bit-exact).  It is
  a *raw* moment (degree-1 homogeneous, so the :math:`\sum w = 2`
  normalization flows through untouched, exactly as :math:`d_0`);
* **the** (28b) **update** :math:`\Delta\phi_{1,i} = -(D_i/h_i)(f_{0,i+1/2}
  - f_{0,i-1/2}) + a_i d_{1,i}` (the DD member :math:`\rho = 0` of
  :eq:`sn-dsa-cell-update`), fed the :math:`g_1 = a\,d_1` sources the
  edge solve already carries;
* **the injection** — Larsen's (33) :math:`\ell \le 1` synthesis:

  .. math::
     :label: sn-dsa-synthesis

     \Psi_m = f_0 + \frac{\mu_m}{W_2}\,f_1
       \qquad(\text{the ``3'' is } 1/W_2,\ \text{computed from the quadrature}),

.. (the ℓ≤1 synthesis / P1-arm injection — the whole d₁ convention chain
..  (w·μ restriction → g₁ columns → (28b) → this synthesis) is verified
..  in one number by the S2-so=1 machine-zero anchor,
..  test_dsa_rate.py::TestP1DSAArm::test_s2_exactness_with_l1_scattering,
..  because S₂'s angular space IS span{1, μ}. 1/W₂ = 3 is a quadrature
..  property, never a transcribed constant.)

with :math:`R\circ P = I` on the moment pair by the :math:`W_2`
quadrature exactness (moment-0 of the :math:`\mu`-arm vanishes and
moment-1 recovers :math:`d_1` exactly).

The arm is **gated on** ``scattering_order >= 1`` — the *same*
consistency-with-the-iterated-operator rule that gates the
:math:`\sigma_{s1}` data row: consistency is with the discrete system
being iterated, so the :math:`\ell = 1` gain enters the low-order
correction only when the sweep itself retains :math:`\ell \ge 1`.  At
``scattering_order = 0`` the d₁ arm is byte-identical to the P0 path.
The **trace arm stays** :math:`\ell = 0` **by theorem** even when the
interior :math:`\ell = 1` arm is live: the reflecting row (39) forces
the wall-edge :math:`f_1 = 0`, and a vacuum wall's trace is read by
nothing — so an :math:`\ell = 1` trace component is identically zero
where it would matter.

**Measured payoff.**  The moment-pair arm restores the flat
Adams–Larsen rate — the anisotropy ladder's worst rung returns from 86
iterations to 15, at :math:`\rho_{\rm est} = 0.175` (the same Fourier
band as the isotropic row):

.. list-table:: The P1 arm flattens the anisotropy ladder (1g slab, c = 0.9, S8, σ\ :sub:`t`\ h = 1)
   :header-rows: 1
   :widths: 24 18 30 28

   * - :math:`\sigma_{s1}/\sigma_{s0}`
     - :math:`n` (plain SI)
     - :math:`n` (SI+DSA, **P0-only**)
     - :math:`n` (SI+DSA, **d₁ pair arm**)
   * - 0.0
     - 236
     - 14
     - 14
   * - 0.3
     - 234
     - 24
     - 14
   * - 0.6
     - 231
     - 39
     - 15
   * - 0.9
     - 217
     - 86
     - 15

And the :math:`\ell = 1` :math:`S_2` system lands at machine zero
(:math:`5.4\times10^{-15}` in one correction) — because :math:`S_2`'s
angular space **is** :math:`\mathrm{span}\{1, \mu\}`, so this one number
verifies the entire d₁ convention chain (:math:`w\mu` restriction
:math:`\to` :math:`g_1` columns :math:`\to` (28b) :math:`\to` the (33)
synthesis) at once.


.. _sn-dsa-both-postures:

Both postures — SI+DSA and the Krylov preconditioner
====================================================

The **same** :class:`~orpheus.sn.acceleration.dsa.DSACorrection`
operator drives two acceleration postures; the choice is a driver
decision, not a re-derivation.

**SI + DSA** is a driver construct: the ``corrector`` parameter on
:class:`~orpheus.numerics.iteration.SourceIteration`.  Each iteration,
after the sweep produces the increment :math:`\Delta\psi =
\psi^{l+1/2} - \psi^{l}`, the corrector applies
:math:`\mathcal{C}\,\Delta\psi` and the update is the plain vector add
:math:`\psi + \mathcal{C}\Delta\psi`.  Both operands are elements of the
flux vector space :math:`V` — flux lives in :math:`V`, and its positive
cone :math:`K` is a predicate on elements rather than a type invariant
(:ref:`cone-typed-field-algebra`).  ``apply`` admits **one** interior type
(:class:`~orpheus.transport.fields.angular_flux.AngularFlux`) and returns
flux-typed correction blocks, so the SI sweep increment and the Krylov
swept vector are the same type.  ⛔ Until 2026-08-19 this read *"the
update is the* **torsor action** :math:`\psi \oplus
\mathcal{C}\Delta\psi` *— the correction is a* **displacement** *(tangent
vector), never a state, so* ``flux + flux`` *stays unspellable"*; the
affine ontology was overturned at campaign-1 CS3
(:ref:`cone-the-overturned-affine-design`).  The extension is a no-op
through the single generic body: byte-inert when ``corrector`` is
``None``, with the SI stop-identity's corrected-arm exemption
documented.

**Krylov + DSA** replaces the identity preconditioner (the #200 seam)
with the Adams–Larsen transport-corrected **left** preconditioner
(:cite:`AdamsLarsen2002` §VI):

.. math::
   :label: sn-dsa-krylov-preconditioner

   M = (I + \mathcal{C}) \circ (L+C)^{-1}
     = \text{sweep} + \text{correction-of-sweep} .

.. vv-status: sn-dsa-krylov-preconditioner documented
.. (structural — the DSA-preconditioned GMRES posture, the first
..  re-enabled preconditioner (folds #200); exercised end-to-end by
..  test_dsa_acceleration.py::TestD4FixedPointInvarianceKrylov and the
..  count gate test_dsa_rate.py::TestD13IterationCounts. A composition
..  identity, not a solver claim.)

Here :math:`(L+C)^{-1}` is the sweep and the swept vector *is* the
displacement from zero, so the same corrector applies.  This is the
identification :math:`\mathrm{SI} \equiv` Richardson on the moment-space
operator and :math:`\mathrm{DSA} \equiv` preconditioned Richardson
(:cite:`AdamsLarsen2002` (1.27)–(1.34)), with the unifying estimate
:math:`\mathrm{cond}(T)\,\kappa \approx 1/(1 - \sigma_{\rm SI})` —
"finding a good preconditioner is the same problem as finding a good
low-order operator".  A good DSA scheme already suppresses every mode,
so the Krylov wrapper "generally reduces iteration counts by only one
or two" (:cite:`AdamsLarsen2002` p. 110–111) — but it is invaluable
where a scheme is slightly inconsistent, on unstructured grids, or (the
ORPHEUS motivation) as the #200 preconditioner slot finally filled.

**Production iteration counts** (1g homogeneous, :math:`K = 40`,
:math:`S_8`, tol :math:`10^{-11}`; the D13 evidence pack):

.. list-table:: DSA iteration counts across the c → 1 corner
   :header-rows: 1
   :widths: 16 10 14 14 16 20

   * - BC
     - :math:`c`
     - :math:`\sigma_t h`
     - SI
     - SI+DSA
     - Krylov / Krylov+DSA
   * - vac / vac
     - 0.9
     - 0.5
     - 225
     - **15**
     - 195 / **11**
   * - vac / vac
     - 0.99
     - 1
     - 2110
     - **16**
     - 174 / **12**
   * - refl / refl
     - 0.9
     - 0.5
     - 249
     - **20**
     - 218 / **12**
   * - refl / refl
     - 0.99
     - 1
     - 2554
     - **21**
     - 197 / **13**

The 2110 :math:`\to` 16 and 2554 :math:`\to` 21 rows are the
c-independence gate (D13): the accelerated count barely moves as
:math:`c \to 1` while SI blows up like :math:`1/(1-c)`.  With the
:math:`\ell \ge 1` P1 arm on a 2g heterogeneous anisotropic problem the
Krylov posture converges 287 :math:`\to` 12 (vacuum) / 305 :math:`\to`
16 (reflective) — the ERR-071 fix (:ref:`sn-dsa-three-discoveries`) was
what made the :math:`\ell \ge 1` Krylov posture converge at all.


.. _sn-dsa-three-discoveries:

Three consistency discoveries
=============================

Building a *consistent* accelerator forced three latent inconsistencies
into the open — each a Mode-9 splitting (exact on a degenerate subspace,
wrong on its complement) that no isotropic fixture could see.  They are
kept here as teaching artifacts: the mechanism, the diagnosis chain,
and the committed catcher.

The σ\ :sub:`r`-fold (ERR-070) — #215's class, first production consumer
------------------------------------------------------------------------

The DSA low-order build is the **first production consumer** of the
foldable cross-section accessors
(:meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.foldable_sigma`
/ ``residual_sig_s``), which produce the within-group
:math:`\sigma_{s0}^{g\to g}` and :math:`\sigma_{s1}^{g\to g}` rows.  The
tempting optimization those accessors invite is the **σ_r-fold**:
realizing the within-group scattering gain as a diagonal removal sweep
with :math:`\sigma_r = \sigma_t - \sigma_{s0}^{g\to g}`.  That treats the
rank-1 isotropic gain :math:`\Sigma_{s0}\,P_{\rm iso}` as
:math:`\Sigma_{s0}\,\mathbb{I}` — the two coincide **iff** the flux is
isotropic (:math:`P_{\rm iso}\psi = \psi`), and the difference operator
:math:`\Sigma_{s0}(\mathbb{I} - P_{\rm iso})` annihilates exactly the
isotropic subspace (the full mismatch is derived at
:eq:`si-sigma-r-fold-mismatch` of :doc:`slab_one_group`).

Measured (the seeded reproduction, D9/D10): on a heterogeneous 2-zone
2g slab, :math:`S_4`, vacuum walls — with flux anisotropy coming from
the vacuum boundary layers *alone*, not from scattering anisotropy —
the fold moves the fixed point by a max **43.2 %** (group 0: 11.4 %,
group 1: 43.2 %; #215's own configs measured 46–56 %).  On the
fully-reflective isotropic box the shift is **identically zero** — the
designed-green degeneracy that let the fold look safe.

Why it is legitimate *here* and only here: the fold would change the
fixed point if wired into the **sweep** (the #215 trap), but the DSA
low-order operator is correction :math:`\to 0`-safe by construction, so
a low-order that carries :math:`\sigma_{s0}` on its removal diagonal
degrades only the rate.  The accessors are fenced for this one consumer.
Two catchers, one numeric and one structural:

* :class:`TestSigmaRFoldCaught` (``catches("ERR-070")``) — the folded
  operator's fixed point measurably departs the true one on an
  anisotropic-flux config, reddening the FP-invariance band on any
  wiring of the fold into the accelerated path;
* the **D10 routing sentinel** (``catches("ERR-070")``) — an AST sweep
  of ``orpheus/`` asserting the foldable accessors' production consumers
  are exactly ``{definition site, split layer, the DSA build}``.  A new
  consumer reds at *design time*, before numerics can be wrong; the
  tooth is a planted ``foldable_sigma`` module the sentinel flags.

The lesson is the Mode-9 canon: an FP-invariance gate for a splitting
must run on a config that **exits** the degenerate subspace
(vacuum / heterogeneous :math:`\Rightarrow` anisotropic flux), and the
convenient-but-wrong data path deserves a structural tripwire alongside
the numerical gate.

The weighted-diamond partial-consistency negative control
---------------------------------------------------------

The battery must prove it can *see* inconsistency, not merely that the
consistent scheme converges.  The negative control pairs a
**weighted-diamond** sweep (:math:`\bar\psi = a\,\psi_{\rm out} +
(1-a)\,\psi_{\rm in}`; :math:`a = \tfrac12` **is** diamond) with the
**DD-consistent** low-order — the Adams–Larsen partially-consistent
class (right operator shape, wrong closure constant, since the WD
:math:`\rho \ne 0` is not modelled).  It reproduces
:cite:`McCoyLarsen1982` Table II's shape exactly (matrix :math:`\rho`,
S4, :math:`K = 40`, vac/vac, :math:`c = 0.99`):

.. list-table:: Partial-consistency reproduces Table-II divergence (matrix ρ)
   :header-rows: 1
   :widths: 20 13 13 13 13 18

   * - :math:`a`
     - :math:`\sigma_t h = 0.1`
     - 1
     - 5
     - 10
     - 30
   * - **0.5** (consistent)
     - 0.165
     - 0.181
     - 0.177
     - 0.168
     - 0.128
   * - 0.6 (partial)
     - 0.165
     - 0.200
     - 0.504
     - 0.820
     - **1.535**
   * - 0.75 (partial)
     - 0.166
     - 0.351
     - 0.964
     - **1.440**
     - **1.779**

The consistent :math:`a = \tfrac12` row is flat and inside the
:math:`0.2247c = 0.222` band at every thickness; any :math:`a \ne
\tfrac12` climbs into **divergence** (:math:`\rho \ge 1`, bold) as cells
thicken.  This is McCoy & Larsen's "partial consistency has no safe
dose" made a live gate — the degradation threshold merely moves to
thicker cells as the inconsistency shrinks.  The in-file independence
proof is the :math:`a = \tfrac12` composite :math:`S_2` machine-zero
anchor: the diamond member with the *production* low-order must close
the :math:`K_2 = 0` system at machine zero, one number tying the
instrument's sweep, the shipped build, and the update composition.

The singular composite sweep-inverse (ERR-071)
----------------------------------------------

The third discovery is the deepest, and the reason the ERR-071 root fix
extended the campaign.  The composite forward :math:`(L+C)` carries the
boundary as a sibling block: inflow rows are identities on the given
inflow; **outflow** rows are the self-consistency defect
:math:`\psi_{\rm out} = \text{streamed} - \psi_{\rm out}`.  The exact
inverse must therefore emit :math:`\psi_{\rm out} = \text{streamed} -
\text{rhs}_{\rm out}`.  But the sweep seeded its mutable boundary buffer
from ``rhs.boundary`` (consuming the *inflow* rows correctly), then let
the march **overwrite** the outflow slots — silently dropping
:math:`\text{rhs}_{\rm out}`.  So :math:`(L+C)^{-1}` mapped every
pure-outflow-row rhs to (essentially) zero: an inverse **exact on every
physical rhs** (whose outflow rows are :math:`0 = 0` defect identities)
and **singular** on the full composite space.

The identity that must hold, and now does:

.. math::
   :label: sn-dsa-sweep-inverse-identity

   (L+C) \circ (L+C)^{-1} = I
   \qquad\text{on the WHOLE composite space (not merely the } \psi_{\rm out}=0
   \text{ subspace physical data spans).}

.. (the composite round-trip identity — pinned by
..  tests/sn/operators/test_sweep_inverse_identity.py (catches ERR-071)
..  on a RANDOM composite with every block populated, parametrized over
..  {vacuum slab, reflective slab, product-quadrature cylinder}, plus
..  the pure-outflow-row leg and the selector-emptying mutation tooth.
..  A structural inverse-contract identity, not a solver claim.)

**How it surfaced.**  Nothing excited the kernel until the P1-DSA arm.
The DSA-preconditioned GMRES posture builds :math:`M = (I +
\mathcal{C})\circ(L+C)^{-1}`; with the :math:`\ell \ge 1` gain active
the Krylov residual acquired a pure outflow-row component (measured
:math:`\|Mq\|/\|q\| = 1.07\times10^{-15}` on that vector — :math:`M`
singular), so full-restart GMRES stalled at an :math:`O(1)` **true**
residual while its *preconditioned* residual sat at
:math:`10^{-31}`.  The **end-of-solve certificate**
(``_certify_within_group_exit``, the #290-era machinery) refused the
claimed convergence: "the honest equation residual is 1.49" — a
loud refusal of a silent wrong-answer stall.

**The diagnosis chain** (recorded because the instrument order
mattered): (1) probe the corrector directly — linear, healthy; (2)
materialize :math:`(I + \mathcal{C})` — smallest :math:`|{\rm eig}| =
1.0`, no kernel; (3) a per-call spy on the production preconditioner
found :math:`\min\|Mq\|/\|q\| = 10^{-15}`; (4) capturing *that*
:math:`q` showed a pure outflow-trace vector with :math:`\text{sweep}(q)
\approx 0` — the sweep, not the corrector.  A
healthy-component-by-component elimination converging on the one
composition seam nobody had gated.

**The root fix** (ruling **R6**, four parts):

#. *Solve half* — one post-march restore in
   :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator._solve_timed_full_field`:
   the outflow rows get ``-=`` the seed's outflow rows (the sign pinned
   by the round-trip gate; the forward's row is :math:`\text{streamed} -
   \psi_{\rm out}`).  Bit-inert on every physical path (:math:`-= 0`),
   nonsingular :math:`M` for Krylov.
#. *Transpose half* — the restore's matrix :math:`E_{\rm out}` is a
   diagonal partial identity, hence **symmetric**, so
   :math:`(A^{\mathsf T})^{-1} = (S_{\rm old})^{\mathsf T} - E_{\rm out}`
   — the same one-site restore in ``solve_transpose``, whose absence the
   G3 full-composite reciprocity gates red.
#. *Call-site role conversions* — every caller that passed an
   **iterate** trace (stale outflow rows) as ``rhs.boundary`` (the
   ``solve_sn`` eigenvalue-finalize and a test helper) was routed
   through the existing
   :meth:`AngularBoundarySourceSink.prescribed_inflow` factory (inflow
   slots only; outflow rows unrepresentable by construction — the
   Pattern-4 projection already in the tree, bypassed by raw
   ``from_mesh`` casts).  The finalize conflation was invisible to every
   :math:`k_{\rm eff}` gate (interior marches read inflow slots only)
   and caught **only** by the 2-D reflective trace-balance gate (defect
   :math:`8.8\times10^{-2}` at exact :math:`k_{\rm eff}` — a Mode-12
   lesson: the balance functional sees what the eigenvalue cannot).
   ⛔ The finalize's own call site is **moot since #448**
   (:ref:`sn-finalize-one-step`): it passes no trace at all — its
   external boundary source is zero and :math:`B` arrives as a gain — so
   the role confusion is unspellable there rather than projected away.
   The test helper's use, the factory, and the solve/transpose halves of
   this fix are untouched.
#. *Honest scope, two frontiers.*  On a product-quadrature cylinder the
   degenerate pure-azimuthal rows (:math:`\mu_r = 0`, excluded from both
   selectors) are **free DOFs** of the composite — the forward is a
   structural zero row there and the inverse completes with the identity
   (#284's free-DOF slots, not a partial-inverse regression); the gate
   asserts both halves of that pair.  And the **scheduled** sibling
   :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`
   (the Gauss–Seidel :math:`M = (L+C) - B_{\rm lower}`) interleaves
   mid-march reflects that consume the buffer's streamed outflow *before*
   the end-of-march restore fires, so its inverse is exact only on
   :math:`\{y : y_{\rm out} = 0\}` (every production rhs).  The
   full-space completion is deferred until a G-S-preconditioned Krylov
   consumer exists; the honest-scope witness is the W2 off-domain
   characterization pin (the tripwire that reds when the completion
   lands).

The committed catchers are the round-trip identity gate
(:eq:`sn-dsa-sweep-inverse-identity`) and the G3 reciprocity gates.  The
lesson is a clean rule: an inverse operator's contract is the identity
on the **whole** space, not on the subspace physical data happens to
span — and the round-trip gate :math:`A\circ A^{-1} \equiv I` on a
random full-space vector is cheap, total, and catches every
partial-inverse class at once.  Wire it the day the inverse is born, not
the day a Krylov method wanders into the kernel.  The root cause was a
**role conflation** — an iterate trace (state) cast raw into a source
slot (given data) — and the correct projection already existed: a
hand-rolled ``from_mesh(trace.values.copy())`` beside a Pattern-4
factory is the same smell as a hand-rolled loop beside an operator.


.. _sn-dsa-rate-and-stability:

Rate and stability evidence
===========================

Because the correction :math:`\to 0` partition makes the rate tier the
*only* catcher for seven of eight error classes, every numeric bar
below is pinned to a measured value (the 3c design scan; the durable
copies live in the evidence pack
``.claude/plans/archive/dsa_rate_characterization.md`` and are re-measured by
``tests/sn/acceleration/test_dsa_rate.py``).

The spectral-radius bound (D11) — one-sided, by design
------------------------------------------------------

The primary quantitative gate measures the estimated spectral radius
:math:`\hat\rho \approx \|\phi^{l+1} - \phi^{l}\| / \|\phi^{l} -
\phi^{l-1}\|` against the consistent-DD Fourier bound
(:eq:`sn-dsa-consistent-fourier`).  Production :math:`S_8`, 1g
homogeneous vacuum slab, :math:`K = 40`: :math:`\hat\rho = 0.176`–:math:`0.180`
at :math:`c = 0.9` (bound :math:`0.2247c = 0.2022`), and
:math:`0.074` at :math:`c = 0.5` (bound :math:`0.1124`).

The gate is deliberately **one-sided** — :math:`\hat\rho \le 0.2247c` —
not a two-sided band :math:`|\hat\rho - 0.2247c| < \varepsilon`.  The
one-sided form is the load-bearing theory claim (the discrete
:math:`S_N` spectral radius *must respect* the continuum Fourier
supremum), and the discrete :math:`\rho` legitimately sits **below** the
continuum sup, so a two-sided band would be **reference contamination**:
it would assert the measured value equals a bound the theory only says
it must not exceed.  The gate is instead the honest split of
(i) the one-sided bound, (ii) a measured attainment **floor** (a
collapsed estimator or a dead accelerator cannot fake a healthy rate),
and (iii) the **plain-SI honesty control** — :math:`\hat\rho(\text{SI})
\approx c` (measured 0.894 / 0.903), proving the estimator measures the
operator, not an artifact.  This is a **rate** claim, flux-shape
independent, so the 1-group model problem is legitimate (declared).

Reflective stability (D12) — the Jacobi wall lag
------------------------------------------------

Reflective problems remove leakage and realize the infinite-medium
worst case (:ref:`sn-dsa-the-fourier-story`) — historically the
divergence regime.  The consistent scheme converges **flat**:
:math:`n = 21` iterations over :math:`\sigma_t h \in \{1, 5, 20\}` at
:math:`c = 0.99`, thickness-independent.

A subtlety worth its own paragraph: the *production* reflective rate
(:math:`\hat\rho \approx 0.28`–:math:`0.31`) is **higher** than the
matrix operator's spectral radius (:math:`\rho \approx 0.15`–:math:`0.19`),
and the gap is **not** a consistency failure — it is the **Jacobi wall
lag**.  The production splitting lags each wall reflection one iteration
(the :math:`B` gain reading the *previous* iterate's outgoing trace),
whereas Larsen's reflecting low-order row models a *within-iteration*
inflow error.  The fully-coupled both-walls-resolved sweep certifies the
operator healthy at the gated BC (:math:`\rho(\text{DSA}) = 0.191 /
0.024 / 0.050` at :math:`c = 0.9`–:math:`0.99`, :math:`\sigma_t h = 1`–:math:`100`);
the elevated production rate is the lag, not the low-order.  At
:math:`\sigma_t h = 100` a **double-lag** mode appears — both
reflections lag, :math:`\hat\rho \approx 0.745`, *c-independent* (75
iterations at :math:`c = 0.9` and 0.99 alike) while the matrix
:math:`\rho` stays :math:`0.065`–:math:`0.108`.  Convergent, stable,
bounded — an improvement seam (wall ordering / trace relaxation), not a
stability failure (filed as a follow-up at close).

The S\ :sub:`2` exactness anchor (K\ :sub:`2` = 0)
--------------------------------------------------

The sharpest single unit test.  With two ordinates every :math:`\psi_m`
**is** linear in :math:`\mu`, so :math:`S_2`-:math:`S_N` **is**
diffusion, discretely: the two-moment reduction closes it in one
correction (:cite:`Larsen1982a` p. 56, :math:`K_2 = 0`).

.. math::
   :label: sn-dsa-s2-exactness

   K_2 = 0
   \quad\Longleftrightarrow\quad
   \text{consistent DSA converges in ONE accelerated iteration (to roundoff)},

.. (the K₂=0 one-iteration exactness — verified by
..  test_dsa_rate.py::TestS2Exactness::test_one_correction_exactness (the
..  vacuum n=2 machine-zero landing and +1 per lagged reflective wall)
..  and, with ℓ=1 scattering, TestP1DSAArm. One number self-verifies the
..  whole convention chain; NOT an eigenvalue claim — S₂-SN is diffusion
..  by construction, so this is an object/rate exactness property.)

Measured: :math:`n = 2` on vacuum (post-correction residual
:math:`3.2\times10^{-15}`), and each **lagged** reflective wall costs
exactly :math:`+1` iteration before the same machine-zero landing
(refl/vac: :math:`n = 3`, refl/refl: :math:`n = 3`) — the second iterate
reads the corrected outflow and closes exactly.  This one number
self-verifies the restriction :math:`T`, the source map :math:`G`, the
edge operator :math:`A_{\rm edge}`, the discrete Marshak rows, the trace
arm, and the (28a) update in a single shot; it is the de-facto catcher
for the boundary-row family, and qa mutation-verified it breaks by 13+
decades under four independent corruption classes (Marshak / sign /
:math:`G` / update).

The c → 1 corner and the scale-free metric
------------------------------------------

Two corner findings a caller must know.  First, the **fixed-point
floor**: at :math:`c \to 1` the flux scale grows like
:math:`O(1/(1-c))`, so the achievable *absolute* residual floor
(:math:`\approx \text{scale} \times 10^{-14}`) can meet an absolute stop
tolerance before the rate does — the run contracts (930 :math:`\to` 0.55
in one correction, ratios :math:`\approx 0.18`) and then stalls at
:math:`2`–:math:`6\times10^{-11}` on double precision, *not* on the
rate (matrix :math:`\rho = 0.205` — healthy).  **Callers at**
:math:`c \to 1` **must scale** ``inner_tol`` **with the flux scale.**
This is why the characterization grid reports a **scale-free** metric:
iterations to a 10-decade *relative* residual reduction, which stays
flat (11–16) across the whole :math:`c \times \sigma_t h` grid.

Second, **lumping the** :math:`\tfrac14(1,2,1)` **removal mass** (a
plausible "simplification") degrades but does not diverge in the scanned
regimes (matrix :math:`\rho` up to 0.82 at :math:`\sigma_t h = 30` vs
consistent 0.10–0.16), so the runtime inconsistent-low-order tooth pins
a **count blow-up** (:math:`> 3\times`); the outright-divergent class
stays with the WD negative-control rows and the landed-cell D2 scan
(:math:`\rho` up to 54.7).


.. _sn-dsa-honest-scope:

Honest scope, deferrals, and rulings
====================================

Arm 1 is **1-D slab, Cartesian, DD, within-group fixed source, P0 +
P1, f-form**.  The build refuses everything outside it *loudly* (a
:class:`NotImplementedError` at
:meth:`~orpheus.sn.acceleration.dsa.DSALowOrderSystem.from_sn_mesh`),
because a silent approximation of the low-order operator is exactly the
partial-consistency divergence the negative control demonstrates.  What
is deferred, and why:

.. list-table:: Deferred arms — reason and follow-up
   :header-rows: 1
   :widths: 24 50 26

   * - Arm
     - Why deferred
     - Ruling / follow-up
   * - **LD-consistent** low-order
     - The 1-D LD iterate is an ``AngularFlux`` of shape
       :math:`(n_g, K, 2)` (a trailing spatial-moment axis); its
       moment-0 restriction is :math:`(n_g, K, 2)`, which the arm-1
       ``solve_correction`` refuses.  The partial pairing "LD sweep +
       DD-derived low-order" is not merely inadvisable — it is
       **structurally unspellable** without the LD moment reduction
       (the M4S build, :cite:`AdamsMartin1992`), and the WD control
       shows what spellable partial pairings do (Table-II divergence).
     - **R5a: defer.**  References: :cite:`Larsen1982a` §V (fully
       consistent LD, :math:`K_N \le 0.300`) and :cite:`AdamsMartin1992`
       (M4S).  Follow-up issue filed.
   * - **2-D Cartesian**
     - DD eliminates to a 9-point vertex diffusion stencil
       (:cite:`Alcouffe1977` §II.C corner moments); the reduction
       interacts with the windowed-sweep angular schedule.
     - Follow-up issue filed.
   * - **Curvilinear**
     - **No discrete curvilinear Fourier stability theory exists**
       (:cite:`AdamsLarsen2002` p. 79); the practice is Morel's
       cell-centered-area approximation with *no* unconditional proof
       (:cite:`Morel1982` pp. 39–40).  Blocked on the #282-family pole
       structure.
     - Empirically-gated only, when it lands.
   * - **k-outer / eigenvalue**
     - Part I/II are fixed-source; Alcouffe's k-variants are nonlinear
       (D̂/removal, DANTSYS lineage) with fixup interactions.  DSA stays
       at the *within-group* level in either posture.
     - Separate design decision (the linear-DSA-k route,
       :cite:`AdamsLarsen2002` §VIII.E, is a designed-for follow-up).
   * - **Gauss–Seidel full-space** sweep-inverse completion
     - The scheduled walk's inverse is exact only on
       :math:`\{y_{\rm out} = 0\}` (ERR-071 part 5); the full-space
       completion needs the restore interleaved per-group with the
       schedule.
     - Deferred until a G-S-preconditioned Krylov consumer exists; the
       W2 off-domain characterization pin is the tripwire.

**Recorded rulings** (from the campaign roadmap):

* **R4** (2026-07-26) — (a) the DSA correction operator is the
  **derived edge-centered** consistent system, pinned entry-for-entry
  vs the reference builder; the standalone diffusion solver keeps its
  cell-centered RT0/harmonic stencil (two defining laws — consistency
  theorem vs standalone accuracy — not a twin path).  (b) The production
  build home is **SN-side** (coefficients are properties of the SN
  discretisation).
* **R5a** (2026-07-26) — the LD arm **defers** (structurally
  unspellable without the M4S reduction).
* **R5b** (2026-07-26) — **P1-DSA wires now** (the d₁ moment-pair arm;
  the ladder flattens 24/39/86 :math:`\to` 14/15/15).
* **R6** (2026-07-27) — the ERR-071 resolution is the **root fix** (the
  honest full-space sweep inverse); the bounded preconditioner-local
  alternative was rejected.


Development history
===================

The consistent-DSA arm 1 landed on branch ``feature/sn-dsa`` (issue #2,
Phase 3 of the stencil-assembly campaign), consuming the assembly mode
(Phase 2) and the unified orientation×kernel walk (Phase 2.5).

* **3a — the four-step derivation of record** (commits ``dbdbb2b9`` /
  ``614eee19``).  :mod:`orpheus.derivations.discrete.sn.dsa` executes
  Larsen's four-step symbolically over a general symmetric quadrature:
  the main theorem (shared-edge :math:`f_1` continuity :math:`\equiv`
  Larsen (27) with (23a–f)), the annihilation identities, the two
  distinct :math:`1/3` mechanisms kept separate, the (28) updates, the
  Marshak (38)/reflecting (39) rows, and the numeric reference builder
  :func:`~orpheus.derivations.discrete.sn.dsa.build_consistent_dd_system`.
  The D2 characterization measured the landed cell-centered diffusion
  loss **divergent** as an accelerator (:math:`\rho` up to 54.7),
  producing ruling **R4** (the derived edge-centered system, SN-side).
* **3b — the accelerator, both postures** (commit ``ab78a15f``).
  :class:`~orpheus.sn.acceleration.dsa.DSALowOrderSystem` (the SN-side
  build, the foldable accessors' first production consumer) and
  :class:`~orpheus.sn.acceleration.dsa.DSACorrection` (one operator both
  postures consume; :math:`R`/:math:`P` = the frame's :math:`\ell = 0`
  faces, nothing minted).  Two consistency discoveries: the trace arm is
  load-bearing under lagged reflection, and :math:`\sigma_{s1}` enters
  the low-order only when the sweep retains :math:`\ell \ge 1`.
* **R5b — the P1-DSA d₁ arm** (commit ``5c350f18``).  The moment-pair
  restriction (:math:`w\mu`, the frame's :math:`\ell = 1` row), the
  (28b) moment-1 update, and the (33) synthesis injection — the
  anisotropy ladder flattens 24/39/86 :math:`\to` 14/15/15; the
  :math:`\ell = 1` :math:`S_2` system lands at machine zero
  (:math:`5.4\times10^{-15}`).
* **R6 — the honest full-space sweep inverse** (commit ``46485eed``,
  ERR-071).  The composite :math:`(L+C)^{-1}` was singular on the
  outflow-trace subspace; the root fix restores the outflow rows in
  ``solve`` and ``solve_transpose``, converts the iterate-trace callers
  to the ``prescribed_inflow`` projection, and documents the
  Gauss–Seidel free-DOF / source-subspace honest scope.  Measured: 2g
  het :math:`\ell \ge 1` Krylov+DSA converges 287 :math:`\to` 12 /
  305 :math:`\to` 16.
* **3c — the rate/stability tier** (commit ``cacabcd0``).  69 gates
  (D11–D13, S2 exactness, the WD partial-consistency negative control,
  the c :math:`\to` 1 corner findings, ERR-070) + the evidence pack.


References
==========

The founding consistent-DSA paper for the diamond-differenced
:math:`S_N` equations is :cite:`Alcouffe1977` (used here as a
cross-check only — its printed (17)/(23) carry sign errata).  The
four-step recipe, the f-form, the boundary rows, and the Fourier
spectral-radius bounds are :cite:`Larsen1982a` (Part I — the
transcription reference for every target form) with the measured
stability envelopes and the partial-consistency Table II in
:cite:`McCoyLarsen1982` (Part II).  P1 (current) acceleration for
anisotropic scattering follows :cite:`Morel1982`.  The consistency
taxonomy, the :math:`\rho = 0.2247c` continuum bound, the discrete
consistent-DD rate (3.65), and DSA-as-Krylov-preconditioner are the
review :cite:`AdamsLarsen2002`; the M4S route for the deferred LD arm is
:cite:`AdamsMartin1992`.  The restriction/prolongation frame theory is
in :doc:`/theory/foundations/frame`; the :math:`\sigma_r`-fold mismatch
that ERR-070 documents is derived in :doc:`slab_one_group`.
