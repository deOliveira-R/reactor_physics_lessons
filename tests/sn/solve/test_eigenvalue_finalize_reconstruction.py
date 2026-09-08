r"""#448 — the eigenvalue finalize must RETURN a flux that solves the equation
it reports.

**Until #448 (fixed 2026-09-06)** ``solve_sn`` built ``Solution.angular_flux``
from ONE final sweep whose source it assembled BY HAND — ``F\phi/k +
\Sigma_{s,0}^T\phi + 2\Sigma_{2n,0}^T\phi``, **P0 only** — through
``SNSolver._add_scattering_source`` / ``_build_aniso_scattering`` /
``_add_n2n_source`` and ``TransferOperator.add_iso_source`` /
``build_aniso_source``, all five now RETIRED.  At every
``scattering_order >= 1`` the :math:`\ell \ge 1` emission (:eq:`pn-scatter`,
and since #426 the (n,2n) stack's own :math:`\ell \ge 1` moments) was
**absent from the reconstruction's right-hand side** while the loss arm the
iterate converged against carried it, so the returned :math:`\psi` solved a
DIFFERENT equation from the one the solve converged and its own angular
moment did not reproduce the ``scalar_flux`` shipped beside it.

⛔ **Do NOT re-point that description at a line number.**  It named
``orpheus/sn/solver.py:2577-2579`` until 2026-09-06; those lines now hold the
#340 N6b exit-balance block — an unrelated, LIVE twin that a stale cite
falsely accuses.  The defect is named by the five RETIRED symbols above,
which grep to nothing, and that is the point.

**Since the carve** the finalize is ONE
:func:`~orpheus.numerics.iteration.fixed_point_step` of the same map the
inner solves drive, taken from ``SNSolver._inner`` — the same ``M``, the same
gains ``(S, N₂ₙ, B)``, the same ``q_ext`` builder
(``_eigenvalue_driver_source``) — so the source the reconstruction sees IS
the source the iteration converged against, by construction.  **Every row in
this module is GREEN, and the module is ERR-083's catcher**; the §6c table
below is the measured RED-BEFORE, kept as the evidence that these rows can
fail at all.

``keff`` and ``Solution.scalar_flux`` come from the converged power iteration
and are NOT affected — that is a claim this module PINS (``TestKAndPhiAreNot
Affected``), not an assumption it makes.

────────────────────────────────────────────────────────────────────────────
CLAIM LEDGER
────────────────────────────────────────────────────────────────────────────
========================================  ==============  =============  ===========================
row                                       claim layer     kind           truth source
========================================  ==============  =============  ===========================
``…_integrates_to_the_reported_scalar…``  flux-shape      IDENTITY       :math:`\phi=\int\psi\,d\Omega` — the definition
``…_the_l_ge_1_moments_are_live``         (precondition)  ACTIVATION     the operator's own ``is_isotropic``
``…_keff_is_frozen…`` / ``…_phi_is_…``    eigenvalue      RECORD         pre-carve capture, this config
``…_equals_an_independent_fixed_source…`` flux-shape      CROSS-ROUTE    the fixed-source entry's OWN reconstruction
``…_balance_defect_responds_to_the_bud…`` (diagnostic)    RATE           the shipped ``_exit_balance_defect``
``…_be_reflected_…``                      flux-shape      IDENTITY       same identity, production 421-group data
========================================  ==============  =============  ===========================

The load-bearing row is the FIRST, and it needs no external reference: it is
the *definition* of the scalar flux.  ``Solution`` ships both members; if the
angular one does not reduce to the scalar one, at least one of the two is
wrong, whatever any reference says.  That is why this file can be an L1 gate
on an eigenvalue problem for which ORPHEUS has no structurally-independent
421-group reference (``vv-principles``: MMS does not prove eigenvalues, and
there is no closed form for a heterogeneous reflected slab) — the claim layer
here is **flux-shape**, not eigenvalue, and the reference is an identity.

────────────────────────────────────────────────────────────────────────────
§6c — the RED-BEFORE, `[M]` 2026-09-05 on the PRE-CARVE tree ``f75a9e59``
────────────────────────────────────────────────────────────────────────────
⛔ **HISTORY, not a current reading.**  Every number in the ``L >= 1`` column
below was measured on the tree that still hand-built the source; on the
carved tree they read ``1.8e-11 … 6.5e-11`` (the post-carve table under it).
The table stays because it IS the §6c evidence — these rows demonstrably can
fail, and a green suite that never showed its red is a coverage claim nobody
checked.

``max|∫ψ dΩ − φ| / max|φ|`` at ``inner_tol=1e-11, flux_tol=1e-9,
keff_tol=1e-10``; the band is ``1e-8`` (derived below).

============================  ==============  ==============  ==========  ==========
arm                           ``L = 0``       ``L >= 1``      band        cost/pair
============================  ==============  ==============  ==========  ==========
``slab_vac``                  1.806e-11 ✅    **2.815e-02**   1e-8        1.3 s
``slab_refl``                 5.207e-12 ✅    **1.606e-02**   1e-8        1.7 s
``slab_vac_n2n``              1.238e-11 ✅    **3.628e-02**   1e-8        1.1 s
``sphere_refl`` (coupled)     3.164e-11 ✅    **2.277e-02**   1e-8        8.2 s
``cylinder_vac`` (coupled)    2.409e-11 ✅    **2.748e-02**   1e-8        7.7 s
``cart2d`` (windowed)         2.408e-11 ✅    **2.612e-02**   1e-8        5.1 s
``slab_krylov``               1.544e-11 ✅    **2.815e-02**   1e-8        3.8 s
``be_reflected`` (421 g, L2)  1.558e-10 ✅    **3.405e-02**   1e-7        13.3 s
============================  ==============  ==============  ==========  ==========

and the two cross-route instruments, same fixture (``slab_vac``):

============================  ==============  ==============  ==========
instrument                    ``L = 0``       ``L = 1``       gate
============================  ==============  ==============  ==========
fixed-source re-solve, ψ      2.59e-12 ✅     **1.4728e-01**  ≤ 1e-8
fixed-source re-solve, φ      1.7e-12 ✅      1.67e-11 ✅     (control)
balance defect, 3 → 12        1.45e6 × ✅     **1.0002 ×**    > 100 ×
============================  ==============  ==============  ==========

⟹ pre-carve, every ``L >= 1`` row was red by **1.6e6 … 3.6e6 ×** its band and
every ``L = 0`` row green with **≥ 316 ×** headroom.  The ``L = 0`` rows were
the CONTROL: they proved the band, the fixture and the identity sound, so the
``L >= 1`` reds attributed to the ℓ ≥ 1 term and nothing else.

**POST-carve** (`[M]` 2026-09-06, same configuration, all 8 arms — the
``cart2d_gs`` row is new at R2 and has no pre-carve twin):

======================  ==========  ==========  ======================  ==========  ==========
arm                     ``L = 0``   ``L >= 1``  arm                     ``L = 0``   ``L >= 1``
======================  ==========  ==========  ======================  ==========  ==========
``slab_vac``            6.242e-11   2.604e-11   ``cylinder_vac``        5.546e-11   6.456e-11
``slab_refl``           1.427e-11   4.492e-11   ``cart2d``              6.553e-11   1.982e-11
``slab_vac_n2n``        2.552e-11   1.811e-11   ``cart2d_gs``           6.489e-11   1.934e-11
``sphere_refl``         6.107e-11   5.221e-11   ``slab_krylov``         5.480e-11   1.959e-11
======================  ==========  ==========  ======================  ==========  ==========

— the two orders now read the SAME scale, which is the whole claim: the
reconstruction no longer knows what ``scattering_order`` is.  Worst reading
**6.6e-11**, 152 × inside the band.

────────────────────────────────────────────────────────────────────────────
THE BAND — and ⛔ its stated MECHANISM was wrong until 2026-09-06
────────────────────────────────────────────────────────────────────────────
The reconstruction is ONE application of :math:`M^{-1}` to the converged
source, so the returned :math:`\psi` differs from the converged iterate by
that solve's own residual (bounded by ``inner_tol``), and the reported
:math:`\phi` differs from the iterate's moment by one outer increment
(bounded by ``flux_tol``).  Hence

    ``band = SAFETY(10) × max(inner_tol, flux_tol)`` = 1e-8

— read off the run config, never hardcoded (``feedback_regression_tolerance
_design.md``).  The BAND is unchanged and still bounds every reading by ≥ 150 ×.

⛔ **REFUTED 2026-09-06 (qa) — this section previously read "the empirical
driver is ``inner_tol`` alone", citing a ``flux_tol`` 1e-6 → 1e-9 sweep that
"leaves the L = 0 reading FLAT at 1.806e-11".  The four rows of that sweep
were ONE measurement.**  A tolerance acts on an iterative solve only through
the ITERATION COUNT it induces, so four decades of it can land in a single
equivalence class — `[M]` ``n_outer = 10`` on every one of those rows.  The
honest reading of "flat" was *"none of my values changed the outer count"*,
not *"this tolerance does not matter"*.  (vv #13's fourth disguise, which
this refutation is the founding example of: **report the iteration count
beside every row of a tolerance sweep**.)

`[M]` 2026-09-06, ``slab_vac``, polish := ``max|ψ_ret − ψ_conv| / max|ψ_conv|``
(``scratch/_448/probe_r2.py``; qa's independent ``scratch/_448_qa/
probe2_polish.py`` reproduces every row):

=========  ==========  ==========  ==========  =========  ==========
``L``      ``k_tol``   ``f_tol``   ``i_tol``   n_outer    polish
=========  ==========  ==========  ==========  =========  ==========
0          1e-10       1e-9        **1e-8**    10         6.800e-10
0          1e-10       1e-9        **1e-10**   10         3.485e-11
0          1e-10       1e-9        **1e-11**   10         3.426e-11
0          1e-10       1e-9        **1e-12**   10         3.426e-11
0          1e-10       **1e-6**    1e-11       10         3.426e-11
0          1e-10       **1e-9**    1e-11       10         3.426e-11
0          1e-10       **1e-11**   1e-11       **12**     6.964e-13
0          **1e-6**    1e-9        1e-11       10         3.426e-11
0          **1e-12**   1e-9        1e-11       **12**     6.964e-13
=========  ==========  ==========  ==========  =========  ==========

Read it by the ``n_outer`` column.  Tightening ``inner_tol`` past 1e-10
buys **0.02 %** — the polish SATURATES at 3.43e-11 — while either knob that
moves ``n_outer`` 10 → 12 drops it **49 ×** to 6.96e-13.  So at this file's
``inner_tol = 1e-11`` the term that dominates is the OUTER one:

    polish ≈ max( ONE outer fission-source increment through ``M⁻¹``,
                  the inverse's own residual )

and the first is binding.  ``L = 1`` behaves the same way (saturates at
1.03e-11 in ``inner_tol``; ``keff_tol`` 1e-12 gives ``n_outer = 13`` and
6.40e-13).  ⟹ a later session tightening ``inner_tol`` alone would find this
floor immovable and go looking for a bug that is not there — which is why the
mechanism is corrected here rather than only in the memo.

⛔ **NOT ``np.array_equal``, and that was decided BEFORE the carve.**  The fix
replaced a hand-assembled
``from_isotropic(F\phi/k + S_0^T\phi + 2\Sigma_{2n,0}^T\phi)`` with
``q_F + \sum_i N_i\psi``: the same quantity through a different reduction
tree AND off a different operand (:math:`\psi_{\rm conv}` rather than the
power iteration's :math:`\phi`).  `[M]` pre-carve those two sources already
differed by 6.5e-11 at ``L = 0`` — iteration-residual scale, not ULP — so a
bit-identity contract would have been a false red on a correct carve
(``vv-principles`` §bit-identity-vs-principled-equivalence).  It is also why
``TestKAndPhiAreNotAffected`` uses ``assert_regression(kind="iterative")``
rather than ``array_equal``: `[M]` all 28 pre-carve anchors reproduced inside
``SAFETY × conv_tol``.

────────────────────────────────────────────────────────────────────────────
CONFIG-BLINDNESS DECLARATION (``AGENT.md`` §0.6, run row by row)
────────────────────────────────────────────────────────────────────────────
* **NOT flat flux** — every arm is heterogeneous (fuel ``A`` | moderator
  ``B``), so the redistribution terms are live.
* **NOT 1-group** — every arm is 2-group.  (The claim layer is flux-shape, so
  the 1-group degeneracy would not bite it directly; the ≥2G choice is what
  makes the ``L >= 1`` group-coupling of :math:`\Sigma_{s,1}` non-trivial.)
* **NOT homogeneous** — three-region slabs / two-region curvilinear.
* **NOT slab-only** — the coupled (``sphere_refl``, ``cylinder_vac``) and the
  2-D windowed (``cart2d``) arms cover the two finalize branches a slab
  cannot reach (``CoupledOperator.solve``; the ``HarmonicMomentFlux``
  iterate).
* **NOT isotropic-source-blind** — this is the exact blindness #448 hides in,
  so ``TestTheLGe1TermIsLive`` is a committed ACTIVATION leg on every arm:
  it asserts the operator's own ``is_isotropic`` predicate reads ``False``
  at ``L = 1``, i.e. the ℓ ≥ 1 body genuinely runs.
* **NOT (n,2n)-blind** — every ``orpheus.derivations.common.xs_library``
  mixture ships ``Sig2 = 0``, so the library arms exercise the ℓ ≥ 1
  SCATTERING leg only.  ``slab_vac_n2n`` manufactures a two-moment
  :math:`\Sigma_{2n}` stack (and the ``be_reflected`` row reads the real
  tape) so the ℓ ≥ 1 (n,2n) leg — the one #426 step 2 landed and ERR-082
  catalogues — has a witness at both the fast and the production tier.

⛔ **The manufactured (n,2n) stack MUST be balanced into ``SigT``.**  `[M]`
adding ``Sig2`` without ``SigT += rowsum(Sig2)`` makes the reported
:math:`\phi` differ from :math:`\int\psi_{\rm conv}\,d\Omega` by an EXACT
global factor 1.100212 — the power iteration's normalisation reading an
unbalanced medium — so the ``L = 0`` control reds too (3.100e-02) and the
gate attributes nothing.  Balanced (the ``tests/cp``/``tests/mc`` house
convention ``sig_t = sig_c + sig_f + rowsum(sig_s) + rowsum(sig2)``) the
``L = 0`` control returns to 1.238e-11.

────────────────────────────────────────────────────────────────────────────
WHAT EACH ARM IS THE *ONLY* WITNESS FOR — `[M]` by mutation, not by argument
────────────────────────────────────────────────────────────────────────────
An arm that reddens whenever its neighbours do is row inflation (vv #20).
Battery ``scratch/_448_battery.py`` measures the partition; the driver is
``scratch/_448_battery.sh`` and the verdicts are in ``_448_battery.status``.

⛔ **The table below is `[M]` 2026-09-05 on the PRE-carve tree, over the 45
rows this module then had, and four of its arms are NO LONGER INSTALLABLE**:
``M2``/``M3``/``M4`` mutate ``SNSolver._add_scattering_source`` /
``_add_n2n_source`` / the finalize's own ``compute_fission_source`` call, and
``M5``/``M5b`` mutate a reflect the finalize no longer performs — all
retired by the carve.  It is kept because the PARTITION it measured is the
argument for each arm's existence, and that argument did not move.  The
post-carve battery, re-keyed onto ``fixed_point_step`` /
``_eigenvalue_driver_source`` / the ``B`` gain and run over all 86 rows, is
``scratch/_448_verification_plan.md`` §12.

=========================  ==============================================
mutation (finalize-scoped) reds — and ONLY these
=========================  ==============================================
``M2`` P0 scatter dropped  ALL 8 arms at BOTH orders (20 of 45)
``M4`` fission rhs x1.01   ALL 8 arms at BOTH orders (20 of 45)
``M6`` cell-average x1.001 ALL 8, both orders, minus ``G3c[L0]`` (19)
``M3`` (n,2n) P0 dropped   **``slab_vac_n2n`` + ``be_reflected`` ONLY** (12)
``M5b`` a WRONG ``B`` (x2) **``slab_refl`` + ``sphere_refl`` + ``cart2d``
                           ONLY** — the three with a reflective face (13)
``M5`` ``B`` SKIPPED       **0** — a DECLARED NULL, see below
``M9`` l>=1 removed EVERY- **0 G1 reds**; 7 ``G2[L1]`` + all 9 activation
       where (converged     rows.  The Mode-12 disclosure this file owes:
       solve included)      **G1 measures CONSISTENCY, never PRESENCE.**
``N1`` returned trace x1.5  **0** — a DECLARED NULL, see below
=========================  ==============================================

⟹ ``M2``/``M4`` are what prove the ``L = 0`` CONTROL rows have TEETH: without
them a green L0 is compatible with "the identity is trivially satisfied".
⟹ ``M3`` is what earns ``slab_vac_n2n`` its 1.1 s, and it MEASURES the
(n,2n) blindness of the six library arms rather than asserting it.
⟹ ``M5b`` is what earns ``slab_refl`` / ``sphere_refl`` theirs: `[M]` G1 goes
5.207e-12 -> **2.758e-01** and 3.164e-11 -> **2.569e-01**, while ``slab_vac``
is BIT-IDENTICAL (``B = 0`` — structurally inert).

⛔ **``M5`` (skip the reflect entirely) WAS a MEASURED NULL, and the reason
was a property of the eigenvalue exit, not of these gates.**  `[M]` the bite
check (``scratch/_448_bite.py``) confirmed the mutation installed (1 skip per
solve) and moved the returned flux by **2.03e-13** (``slab_refl``) /
**2.31e-15** (``sphere_refl``) / **exactly 0.0** (``slab_vac``), against
traces of 4.2e-01 / 2.4e-03 / 1.7e-01.  That reflect's own docstring (the
pre-#448 finalize's ``_reflect_outflow_into_inflow``, now the sweep-tier
gates' helper ``tests/sn/_test_helpers.py::reflect_outflow_into_inflow``)
already said why — *"idempotent here (the converged inflow already equals
B·psi.outflow)"* — and that was the sentence, measured.  ⟹ on a CONVERGED
eigenvalue exit that reflect WAS inert to the solve's own noise, so **no value
gate anywhere could witness its removal** — which is precisely the licence the
carve took: option B deleted it, and `[M]` all 45 rows stayed green through
the deletion.  What the reflective arms gate is a WRONG ``B``, never a MISSING
one, and post-carve ``B`` reaches the finalize as a GAIN, so the successor
arms are ``P5a``/``P5b`` on ``SNBoundaryOperator`` (§12), not ``M5``/``M5b``
on a call site that no longer exists.

────────────────────────────────────────────────────────────────────────────
DECLARED BLIND — what this file CANNOT see
────────────────────────────────────────────────────────────────────────────
* ``Solution.radial_characteristic`` — System B's converged ψ½ member.  The
  identity ``φ = ∫ψ dΩ`` is a System-A BULK statement and says nothing about
  it; the carrying arms reach it only through the coupled solve's own
  consistency.  Still uncovered here.
* ``Solution.boundary_flux`` — **NO LONGER BLIND, as of R2.**  This bullet
  used to close *"if the carve replaces the whole finalize block (its option
  B), the trace's provenance changes and this file stays green — that claim
  needs its own gate."*  The carve DID take option B, so
  ``TestTheReturnedTrace`` is that gate.  ``G1`` itself remains bulk-blind —
  `[M]` the ``N1`` arm scaled the returned trace by 72 % of its own magnitude
  and reddened 0 of 45 — which is why the trace needed a class of its own
  rather than a leg on an existing row.
* ``history.balance_defect`` on a CONVERGED solve.  `[M]` it is ``None`` by
  construction (``_exit_balance_defect`` returns early on
  ``record.fully_converged``), which is exactly why the shipped diagnostic
  never caught #448.  ``TestTheShippedDiagnostic`` pins that fact and then
  measures the defect where it IS live.
* the fixed-source entries.  Their windowed reconstruction already builds
  ``q + Σ gains·ψ`` — since the carve through the very same
  ``fixed_point_step``; ``tests/sn/solve/
  test_2d_anisotropic_windowing.py`` is the standing gate there.

────────────────────────────────────────────────────────────────────────────
COST
────────────────────────────────────────────────────────────────────────────
`[M]` 2026-09-05, ``python -O -m pytest -p no:randomly -q`` on the whole
module: **10 failed, 35 passed in 44.55 s** (the 10 are the §6c red-before
above).  The per-arm pair costs are in that table; every arm is solved ONCE
per order and shared across the classes by ``_CACHE``, so ``TestKAndPhiAre
NotAffected`` and ``TestTheLGe1TermIsLive`` are free.

⛔ **NOT ``slow``, and it must not become one.**  #428 F-4 is the measured
precedent (an ERR's only catcher marked ``slow`` is a catcher the canonical
``-m "not slow"`` gate cannot see).
"""

from __future__ import annotations

import contextlib
import copy
import io
import warnings
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.data.macro_xs.mixture import Mixture, compute_macro_xs
from orpheus.data.micro_xs import load_isotope
from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import (
    BC,
    CoordSystem,
    Mesh1D,
    Region,
    RegionMesh,
    StructuredGeometry,
)
from orpheus.geometry.mesh import Mesh2D
from orpheus.numerics.convergence import ConvergenceWarning
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn import solve_sn_fixed_source
from orpheus.sn.solution import Solution
from orpheus.sn.solver import (
    InnerSolve,
    SNSolver,
    _system_a_member,
    _system_b_member,
    solve_sn,
)
from orpheus.transport.fields.angular_flux import AngularFlux
from tests.sn._test_helpers import (
    SN_TESTS_ROOT,
    curvilinear_two_region_mesh,
    reflect_outflow_into_inflow,
)
from tests.sn.regression._regression_assert import assert_regression

# ── The ONE run configuration (the band reads off it; nothing hardcodes) ──
_KEFF_TOL = 1e-10
_FLUX_TOL = 1e-9
_INNER_TOL = 1e-11
_MAX_OUTER = 800

#: ``SAFETY(10) × max(inner_tol, flux_tol)`` — see the module docstring's band
#: derivation.  The ONE name both the identity gate and its messages read.
_SAFETY = 10.0
_BAND = _SAFETY * max(_INNER_TOL, _FLUX_TOL)

#: The ℓ ≥ 1 order every "subject" row runs at.  ``0`` is its control.
_L_ANISO = 1

_BASELINE_DIR = SN_TESTS_ROOT / "_data" / "finalize_reconstruction_448"


# ══════════════════════════════════════════════════════════════════════
# Materials
# ══════════════════════════════════════════════════════════════════════

def _balanced_n2n(mix: Mixture, p0: np.ndarray, l1_over_l0: float) -> Mixture:
    r"""``mix`` with a TWO-moment :math:`\Sigma_{2n}` stack, balanced into
    :math:`\Sigma_t`.

    The library mixtures all ship ``Sig2 = 0``, so without this the fast arms
    are blind to the ℓ ≥ 1 (n,2n) leg entirely.  ``SigT`` MUST absorb
    ``rowsum(Sig2)``: the (n,2n) reaction removes its incident neutron, and a
    medium that emits without removing is unbalanced — see the ⛔ in the
    module docstring for the exact 1.100212 artefact that produces.  The
    balance spelling is the one ``tests/cp/test_verification.py`` and
    ``tests/mc/test_gaps.py`` already use.
    """
    return replace(
        mix,
        Sig2=[csr_matrix(p0), csr_matrix(p0 * l1_over_l0)],
        SigT=np.asarray(mix.SigT, dtype=float) + p0.sum(axis=1),
    )


#: A fast-group (n,2n) block on the moderator, Be-like in shape (fast in,
#: both groups out, no up-scatter) with a strongly forward-peaked ℓ = 1
#: moment.  `[M]` zeroing its ℓ = 1 moment moves k by 2.16e-02 on
#: ``slab_vac_n2n``, so the term is not decorative.
_N2N_P0 = np.array([[0.02, 0.06], [0.0, 0.0]])
_N2N_L1_OVER_L0 = 0.6

_FUEL = get_mixture("A", "2g")
_MOD = get_mixture("B", "2g")          # μ̄ = 0.6 — strongly anisotropic P1
_MOD_N2N = _balanced_n2n(_MOD, _N2N_P0, _N2N_L1_OVER_L0)

_LIBRARY = {2: _FUEL, 0: _MOD}
_LIBRARY_N2N = {2: _FUEL, 0: _MOD_N2N}


# ══════════════════════════════════════════════════════════════════════
# Meshes
# ══════════════════════════════════════════════════════════════════════

def _slab(bc: BC) -> Mesh1D:
    """moderator | fuel | moderator, 20 cells."""
    geom = StructuredGeometry(
        geometry="SLB",
        regions=(
            Region(mat_id=0, outer_thickness_cm=2.0),
            Region(mat_id=2, outer_thickness_cm=6.0),
            Region(mat_id=0, outer_thickness_cm=2.0),
        ),
        bcs=(bc, bc),
    )
    return Mesh1D.from_geometry(geom, region_meshes=(
        RegionMesh(n_cells=6), RegionMesh(n_cells=8), RegionMesh(n_cells=6),
    ))


def _curvilinear(coord: CoordSystem, bc: BC) -> Mesh1D:
    """fuel core | moderator reflector, 14 cells."""
    return curvilinear_two_region_mesh(
        outers=(6.0, 8.0), mat_ids=(2, 0), n_cells=(8, 6), coord=coord, bc=bc,
    )


def _cart2d() -> Mesh2D:
    """6 x 4 Cartesian, fuel slab in x, vacuum-x / reflective-y."""
    mat = np.zeros((6, 4), dtype=int)
    mat[1:5, :] = 2
    return Mesh2D(
        edges_x=np.linspace(0.0, 10.0, 7),
        edges_y=np.linspace(0.0, 10.0, 5),
        mat_map=mat,
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("reflective"), bc_ymax=BC("reflective"),
    )


# ══════════════════════════════════════════════════════════════════════
# The arm table — one row per FINALIZE BRANCH, not one per geometry
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _Arm:
    """One finalize branch, with the structural fact that makes it distinct."""

    materials: dict[int, Mixture]
    mesh: Callable[[], Mesh1D | Mesh2D]
    quadrature: Callable[[], Quadrature]
    inner_solver: str
    #: what this arm covers that no other arm does — read by the reader, and
    #: by the reviewer asking whether the row is anti-#20 inflation.
    branch: str
    #: ``solve_sn``'s default is ``"jacobi"``, NOT boundary Gauss-Seidel — so
    #: without an arm that says otherwise, no row in this file ever poses the
    #: G-S splitting.  `[M]` 2026-09-06 (qa F-1): a census over the module's
    #: 161 inner solves found **0** ``ScheduledInvertibleOperator``.
    inner_schedule: str = "jacobi"


_ARMS: dict[str, _Arm] = {
    "slab_vac": _Arm(
        _LIBRARY, lambda: _slab(BC.vacuum),
        lambda: Quadrature.gauss_legendre(n_ordinates=8),
        "source_iteration",
        "1-D un-windowed SI; B = 0 (the plainest reconstruction)",
    ),
    "slab_refl": _Arm(
        _LIBRARY, lambda: _slab(BC.reflective),
        lambda: Quadrature.gauss_legendre(n_ordinates=8),
        "source_iteration",
        "1-D un-windowed SI with B != 0 — the only arms (with sphere_refl and\n         cart2d) that see a WRONG B: [M] M5b 5.207e-12 -> 2.758e-01",
    ),
    "slab_vac_n2n": _Arm(
        _LIBRARY_N2N, lambda: _slab(BC.vacuum),
        lambda: Quadrature.gauss_legendre(n_ordinates=8),
        "source_iteration",
        "the only FAST arm with a live l>=1 (n,2n) moment",
    ),
    "sphere_refl": _Arm(
        _LIBRARY, lambda: _curvilinear(CoordSystem.SPHERICAL, BC.reflective),
        lambda: Quadrature.gauss_legendre(n_ordinates=8),
        "source_iteration",
        "CARRYING mesh: CoupledOperator finalize + B != 0 on the psi-half\n         corner ([M] M5b 3.164e-11 -> 2.569e-01)",
    ),
    "cylinder_vac": _Arm(
        _LIBRARY, lambda: _curvilinear(CoordSystem.CYLINDRICAL, BC.vacuum),
        lambda: Quadrature.folded_product(n_mu=4, n_phi=8),
        "source_iteration",
        "CARRYING mesh with MULTI-LEVEL psi-half (folded_product)",
    ),
    "cart2d": _Arm(
        _LIBRARY, _cart2d,
        lambda: Quadrature.level_symmetric(sn_order=4),
        "source_iteration",
        "2-D Cartesian WINDOWED SI: the iterate is a HarmonicMomentFlux",
    ),
    "cart2d_gs": _Arm(
        _LIBRARY, _cart2d,
        lambda: Quadrature.level_symmetric(sn_order=4),
        "source_iteration",
        "2-D Cartesian windowed SI under boundary GAUSS-SEIDEL — the only arm\n"
        "         whose finalize reconstructs through a ScheduledInvertible\n"
        "         Operator M and an SNMaskedBoundaryOperator gain",
        inner_schedule="gauss_seidel",
    ),
    "slab_krylov": _Arm(
        _LIBRARY, lambda: _slab(BC.vacuum),
        lambda: Quadrature.gauss_legendre(n_ordinates=8),
        "krylov",
        "the Krylov inner (full-angular iterate, no windowing)",
    ),
}

_ARM_IDS = tuple(_ARMS)


# ══════════════════════════════════════════════════════════════════════
# The shared solve cache — every arm solved ONCE per order
# ══════════════════════════════════════════════════════════════════════

_CACHE: dict[tuple[str, int], Solution] = {}
#: The ``SNSolver`` each cached solve built — see :func:`_capturing_solvers`.
_SOLVERS: dict[tuple[str, int], SNSolver] = {}


@contextlib.contextmanager
def _capturing_solvers(sink: list[SNSolver]):
    """Record every ``SNSolver`` ``solve_sn`` builds, for the duration.

    ``solve_sn`` constructs its solver internally and returns only the
    ``Solution``, so ``SNSolver._inner`` — the ``InnerSolve`` record the
    finalize reconstructs FROM — has no public route.  The trace gates need
    it (the converged trace is ``_inner.iterate``'s System-A boundary), and
    inferring it from the returned object would make the comparison
    self-referential.  Restored in ``finally``: a leaked patch would corrupt
    every module collected after this one.
    """
    original = SNSolver.__init__

    def recording(self, *args, **kwargs):
        original(self, *args, **kwargs)
        sink.append(self)

    SNSolver.__init__ = recording
    try:
        yield
    finally:
        SNSolver.__init__ = original


def _solve(arm_id: str, order: int) -> Solution:
    key = (arm_id, order)
    if key in _CACHE:
        return _CACHE[key]
    arm = _ARMS[arm_id]
    built: list[SNSolver] = []
    with contextlib.redirect_stdout(io.StringIO()), _capturing_solvers(built):
        sol = solve_sn(
            arm.materials, arm.mesh(), arm.quadrature(),
            scattering_order=order,
            inner_solver=arm.inner_solver,
            inner_schedule=arm.inner_schedule,
            keff_tol=_KEFF_TOL, flux_tol=_FLUX_TOL,
            inner_tol=_INNER_TOL, max_outer=_MAX_OUTER,
        )
    history = sol.history
    if history is None or not history.fully_converged:
        pytest.fail(
            f"{arm_id}[L={order}] did not fully converge — a starved solve "
            f"degrades the RATE, not the limit, so no budget certifies this "
            f"tolerance (#340 N5) and every reading below is unattributable."
        )
    if not built:
        pytest.fail(
            f"{arm_id}[L={order}]: the capture recorded no SNSolver — "
            f"solve_sn stopped building one, so every gate reading "
            f"_inner below is measuring nothing."
        )
    _CACHE[key] = sol
    _SOLVERS[key] = built[-1]
    return sol


def _inner_of(arm_id: str, order: int) -> InnerSolve:
    """The ``InnerSolve`` record the finalize reconstructed from."""
    _solve(arm_id, order)
    inner = _SOLVERS[(arm_id, order)]._inner
    if inner is None:
        pytest.fail(
            f"{arm_id}[L={order}]: SNSolver._inner is None after a converged "
            f"solve — the record the finalize reads was never written."
        )
    return inner


def _trace_values(field) -> np.ndarray:
    return np.asarray(field.values, dtype=np.float64)


def _self_consistency(sol: Solution) -> float:
    r"""``max|∫ψ dΩ − φ| / max|φ|`` — the claim, as one number.

    The ``isinstance`` is a PARSE, not ceremony: the identity is only
    meaningful for a per-ordinate carrier.  A finalize that returned the
    moment iterate whole would make ``integrate_angular`` unavailable, and
    the honest failure is that sentence rather than an ``AttributeError``
    three frames down.
    """
    bulk = sol.angular_flux.interior
    if not isinstance(bulk, AngularFlux):
        pytest.fail(
            f"Solution.angular_flux carries a {type(bulk).__name__} bulk, not "
            f"an AngularFlux — the returned flux is not per-ordinate, so the "
            f"identity φ = ∫ψ dΩ is not even expressible on it."
        )
    moment = np.asarray(
        bulk.integrate_angular().values, dtype=np.float64,
    )
    phi = np.asarray(sol.scalar_flux.values, dtype=np.float64)
    denom = float(np.max(np.abs(phi)))
    if denom == 0.0:
        pytest.fail("the reported scalar flux is identically zero — the "
                    "fixture is degenerate and the ratio is undefined.")
    return float(np.max(np.abs(moment - phi)) / denom)


# ══════════════════════════════════════════════════════════════════════
# G0 — the ACTIVATION legs (vv #20 / lessons L40c: a row that cannot see
#      the term it is credited with is inflation, not coverage)
# ══════════════════════════════════════════════════════════════════════

class TestTheLGe1TermIsLive:
    """Preconditions. Without these the whole file is a designed-green suite."""

    @pytest.mark.l1
    @pytest.mark.parametrize("arm_id", _ARM_IDS)
    def test_the_scattering_operator_is_anisotropic_at_l1(self, arm_id: str):
        """At ``L = 1`` the scattering binding's OWN predicate says ℓ ≥ 1 runs.

        ``TransferOperator.is_isotropic`` reads the PADDED moment VALUES, not
        the declared order (the CS4c step-5b census), so this is the honest
        activation predicate: ``False`` means an ℓ ≥ 1 body genuinely
        executes on this arm's data.  It is the difference between "the
        reconstruction drops a term" and "the reconstruction drops a term
        that happens to be zero here".
        """
        sn_mesh = _solve(arm_id, _L_ANISO).mesh
        solver = SNSolver(sn_mesh, scattering_order=_L_ANISO)
        assert not solver.scattering_op.is_isotropic, (
            f"{arm_id}: the scattering binding reads ISOTROPIC at "
            f"scattering_order={_L_ANISO}, so no ℓ ≥ 1 emission exists to "
            f"drop and every subject row on this arm is designed-green."
        )
        # and the control's own precondition: at L = 0 it MUST read isotropic
        control = SNSolver(sn_mesh, scattering_order=0)
        assert control.scattering_op.is_isotropic, (
            f"{arm_id}: the L = 0 control is NOT isotropic — the control and "
            f"the subject differ by something other than the ℓ ≥ 1 term."
        )

    @pytest.mark.l1
    def test_the_n2n_binding_is_anisotropic_on_the_n2n_arm(self):
        """The ℓ ≥ 1 **(n,2n)** leg has a fast witness — and only here.

        `[M]` every ``xs_library`` mixture ships ``Sig2 = 0``, so the six
        library arms are structurally blind to the (n,2n) half of #448.  This
        row is what makes ``slab_vac_n2n`` worth its 1.1 s, and its failure
        means the manufactured stack stopped reaching the binding.
        """
        sn_mesh = _solve("slab_vac_n2n", _L_ANISO).mesh
        solver = SNSolver(sn_mesh, scattering_order=_L_ANISO)
        assert not solver.n2n_op.is_isotropic, (
            "the manufactured two-moment Sig2 stack reads ISOTROPIC at the "
            "binding — the ℓ ≥ 1 (n,2n) leg is not activated and this arm "
            "has become a duplicate of slab_vac."
        )
        # the library arms are the negative leg: they MUST read isotropic,
        # which is the measurement that justifies this arm's existence.
        lib_mesh = _solve("slab_vac", _L_ANISO).mesh
        lib_solver = SNSolver(lib_mesh, scattering_order=_L_ANISO)
        assert lib_solver.n2n_op.is_isotropic, (
            "an xs_library arm now carries an anisotropic (n,2n) binding — "
            "the library gained Sig2 data, so this file's claim that only "
            "slab_vac_n2n witnesses the (n,2n) leg is stale."
        )

    @pytest.mark.l1
    def test_the_n2n_l1_moment_moves_the_eigenvalue(self):
        """vv #19: the manufactured ℓ = 1 (n,2n) moment is LOADED, not inert.

        A binding can report anisotropic and still contribute nothing the
        solve can feel.  `[M]` zeroing the ℓ = 1 moment (keeping the stack
        LENGTH — the ERR-082 ruling: vary the VALUES, never the length) moves
        k by 2.16e-02 = 2165 Δk·1e-5.
        """
        subject = _solve("slab_vac_n2n", _L_ANISO)
        flat = _balanced_n2n(_MOD, _N2N_P0, 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            control = solve_sn(
                {2: _FUEL, 0: flat}, _slab(BC.vacuum),
                Quadrature.gauss_legendre(n_ordinates=8),
                scattering_order=_L_ANISO, inner_solver="source_iteration",
                keff_tol=_KEFF_TOL, flux_tol=_FLUX_TOL,
                inner_tol=_INNER_TOL, max_outer=_MAX_OUTER,
            )
        k_sub, k_ctl = subject.keff, control.keff
        assert k_sub is not None and k_ctl is not None
        delta = abs(k_sub - k_ctl)
        assert delta > 1e-4, (
            f"zeroing the (n,2n) ℓ = 1 moment moves k by only {delta:.3e} — "
            f"the moment is inert on this fixture, so slab_vac_n2n cannot "
            f"witness an (n,2n) ℓ ≥ 1 reconstruction defect."
        )


# ══════════════════════════════════════════════════════════════════════
# G1 — THE CLAIM
# ══════════════════════════════════════════════════════════════════════

class TestTheReturnedFluxIsSelfConsistent:
    r""":math:`\int \psi_{\rm returned}\,d\Omega = \phi_{\rm returned}`."""

    @pytest.mark.l1
    @pytest.mark.verifies("pn-scatter", "flux-moments", "n2n-source")
    @pytest.mark.catches("ERR-083")
    @pytest.mark.parametrize("arm_id", _ARM_IDS)
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0-control", "L1"])
    def test_returned_angular_flux_integrates_to_the_reported_scalar_flux(
        self, arm_id: str, order: int,
    ):
        r"""``Solution.angular_flux`` must reduce to ``Solution.scalar_flux``.

        The two members are shipped side by side and the scalar one is
        DEFINED as the angular one's zeroth moment.  ``solve_sn`` builds them
        from two different objects — the scalar from the converged power
        iteration, the angular from a hand-assembled final sweep — so the
        identity is a real constraint on the finalize block, not a tautology.

        **The ``L0-control`` rows are the instrument's own calibration.**
        They run the identical code path with the ℓ ≥ 1 term absent by
        construction, so a green L0 beside a red L1 attributes the red to
        that term and to nothing about the band, the fixture or the identity.
        """
        residual = _self_consistency(_solve(arm_id, order))
        assert residual <= _BAND, (
            f"{arm_id}[scattering_order={order}]: "
            f"max|∫ψ dΩ − φ| / max|φ| = {residual:.4e} > band {_BAND:.1e} "
            f"(SAFETY {_SAFETY:g} × max(inner_tol {_INNER_TOL:.0e}, "
            f"flux_tol {_FLUX_TOL:.0e})).  The returned angular flux does not "
            f"reduce to the scalar flux shipped beside it: the finalize's "
            f"reconstruction solved a different equation from the one the "
            f"power iteration converged.  At scattering_order ≥ 1 the "
            f"expected cause is #448 — the ℓ ≥ 1 scattering / (n,2n) "
            f"emission missing from the reconstruction's rhs "
            f"(orpheus/sn/solver.py:2577-2579)."
        )


# ══════════════════════════════════════════════════════════════════════
# G2 — what the carve must NOT move
# ══════════════════════════════════════════════════════════════════════

class TestKAndPhiAreNotAffected:
    """``keff`` and ``scalar_flux`` are the POWER ITERATION's, not the
    finalize's — frozen here pre-carve so the fix can prove it.

    ⚠ This is a RECORD (``numerical-bug-signatures`` Sig 10): a red says
    *something moved*, not which side is right.  Its value is entirely in
    being captured on the DEFECTIVE tree — the carve had to pass through it
    to prove "#448 fixed" was not "#448 fixed and the eigenvalue quietly
    moved".  `[M]` it did: all 28 pre-carve anchors reproduced.

    ⛔ **The 32 anchor files (8 arms × 2 orders × ``{keff, scalar_flux}``)
    do NOT share a provenance, and the asymmetry is load-bearing.**

    * **28 are PRE-carve** (`[M]` 2026-09-05, tree ``f75a9e59``) — the seven
      original arms × two orders × two quantities.  They are the carve's
      acceptance evidence: a value that survived a change to the code
      that produced it.
    * **4 are POST-carve** — ``cart2d_gs``, added at R2
      because ``solve_sn``'s ``inner_schedule`` default is ``"jacobi"`` and no
      row had ever posed the Gauss-Seidel splitting (qa F-1).  They pin
      nothing across the carve; they are an ORDINARY forward regression floor.

    ⟹ do not read a ``cart2d_gs`` red as "the #448 carve moved k".  What
    licenses those two arms instead is
    ``TestTheGaussSeidelArmPosesItsOwnSplitting``, whose FP-invariance row
    checks them against the ``cart2d`` Jacobi arm — an independent splitting
    of the same ``A``, which is a stronger statement than a frozen number
    (`[M]` ``|Δk| = 6.28e-14``).
    """

    @pytest.mark.l1
    @pytest.mark.parametrize("arm_id", _ARM_IDS)
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_keff_and_scalar_flux_are_frozen_pre_carve(
        self, arm_id: str, order: int, request: pytest.FixtureRequest,
    ):
        """Capture with ``--capture-baseline``; assert without it.

        Reuses the root conftest's flag and ``assert_regression`` (which
        reads ``conv_tol`` off the run config and carries the ``DriftWarning``
        tripwire) rather than minting a second mechanism.
        """
        sol = _solve(arm_id, order)
        keff = sol.keff
        assert keff is not None, f"{arm_id}[L={order}] returned no eigenvalue"
        phi = np.ascontiguousarray(
            np.asarray(sol.scalar_flux.values, dtype=np.float64),
        )
        case = f"{arm_id}_L{order}"
        k_path = _BASELINE_DIR / f"{case}_keff.npy"
        phi_path = _BASELINE_DIR / f"{case}_scalar_flux.npy"

        if request.config.getoption("--capture-baseline", default=False):
            _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(k_path, np.asarray(keff, dtype=np.float64))
            np.save(phi_path, phi)
            pytest.skip(f"captured {case} baselines")

        for path in (k_path, phi_path):
            if not path.exists():
                pytest.fail(
                    f"missing baseline {path}; run this module once with "
                    f"--capture-baseline (on the tree whose values you mean "
                    f"to freeze) to write it.  A gate's inputs are part of "
                    f"the commit — these live under tests/, never scratch/."
                )
        assert_regression(
            np.asarray(keff, dtype=np.float64), np.load(k_path),
            conv_tol=_KEFF_TOL, case_name=case, kind="iterative",
            quantity="k_eff",
        )
        assert_regression(
            phi, np.load(phi_path),
            conv_tol=_FLUX_TOL, case_name=case, kind="iterative",
            quantity="scalar_flux",
        )


# ══════════════════════════════════════════════════════════════════════
# G6 — the RETURNED TRACE (R1's owed gate; the carve took option B)
# ══════════════════════════════════════════════════════════════════════

#: The arms whose boundary operator is the ZERO morphism — every face
#: vacuum — so ``inflow == B·outflow`` reads ``0 == 0``.  DECLARED here and
#: ASSERTED by the gate below against a live measurement of ``|B·ψ|``, so a
#: BC change in the arm table cannot silently move an arm between the two
#: populations and quietly retire a row's teeth (vv #20).
_VACUUM_ARM_IDS = ("slab_vac", "slab_vac_n2n", "cylinder_vac", "slab_krylov")


class TestTheReturnedTrace:
    r"""What ``G1`` is structurally blind to: the trace's PROVENANCE.

    ``φ = ∫ψ dΩ`` is a BULK statement, so the boundary trace lies in its
    kernel exactly — `[M]` the pre-carve battery's ``N1`` arm scaled the
    returned trace by ``1.5× + 0.1`` (**72 %** of its own magnitude, bulk
    bit-identical) and reddened **0 of 45** rows.  Until #448 nothing needed
    a trace gate here, because the finalize SET the inflow by hand
    (``_reflect_outflow_into_inflow`` + ``prescribed_inflow``) and the law
    held by construction.  The carve took the memo's **option B**: the
    finalize is now one ``fixed_point_step`` and the reflective coupling
    arrives as the ``B`` gain, so ``inflow == B·outflow`` on the returned
    trace became a real fixed-point CLAIM about the solve rather than a
    tautology about an assignment.  This class is that claim.

    ⚠ **DECLARED BLIND to a wrong reflective LAW — and MEASURED, not
    argued (vv #22, the shared-object axis).**  ``SNBoundaryOperator.apply``
    (what the gain applies) and the sweep-tier helper
    ``tests/sn/_test_helpers.py::reflect_outflow_into_inflow`` (what this
    reference applies — since CS4c step 6 item 6.5 the Jacobi split's
    ``upper`` mask's ``reflect_rows_inplace`` after zeroing the inflow rows;
    until then the retired ``reflect_inflow_inplace``) are two routes into
    ONE body: ``_apply_faces`` merely LIFTS the trace-only ``_reflect_trace``
    onto the full field, the mask's verb calls it directly, and
    ``_reflect_trace`` is where both meet.  So a wrong law moves the gain and
    this reference together and both legs stay green.

    `[M]` 2026-09-06, battery arm ``P5d`` — ``_reflect_trace × 1.001``, small
    enough that the solve still converges:

    ==============================  ==================================
    row                             verdict
    ==============================  ==================================
    ``T-law`` (6 rows, B ≠ 0)       **GREEN** — the declared blindness
    ``T-conv`` (6 rows)             **GREEN** — blind too: both traces move
    ``G1`` (6 rows)                 **RED**, 6.84e-04 / 6.16e-04
    ``G2`` frozen k / φ (6 rows)    **RED**
    ==============================  ==================================

    ⟹ the partition is clean and worth stating: **``G1``/``G2`` catch a wrong
    LAW; this class catches wrong WIRING** — the gain applied to the wrong
    operand (``P2``: 6 rows), a splitting whose ``M`` and gains disagree
    (``P6``: 2 rows), the moment/angular end mismatch (``P3``: 4 rows), the
    coupled seed (``P4b``: 3 rows).  Neither is redundant.

    ⛔ Two earlier attempts are recorded because each was wrong in an
    instructive way.  ``P5a`` mutated ``_apply_faces`` believing it shared —
    it does not, it is the gain's outer lift, so that arm is a second
    gain-route mutation.  ``P5c`` mutated ``_reflect_trace`` by ``×2``, which
    is not a perturbation of a reflective eigenvalue problem but a different
    and DIVERGENT one: `[M]` all 9 of its reds were *"did not fully
    converge"*, attributing nothing.  A law mutation must stay inside the
    problem's convergent regime to be readable at all.
    """

    @pytest.mark.l1
    @pytest.mark.verifies("reflective-bc")
    @pytest.mark.parametrize("arm_id", _ARM_IDS)
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_reflective_law_holds_on_the_returned_trace(
        self, arm_id: str, order: int,
    ):
        r"""``ψ_ret.inflow == (B·ψ_ret)|_inflow`` — to the band, where ``B ≠ 0``.

        And where ``B = 0`` the row asserts the OTHER thing: that the reading
        is EXACTLY ``0.0`` and the arm is one of the declared vacuum arms.
        That is not a weaker assertion dressed up — it is what makes the
        blindness a measurement instead of a comment, and it makes a BC
        change in the arm table loud (an arm that leaves ``_VACUUM_ARM_IDS``
        by acquiring a reflective face reddens here, not silently).

        `[M]` 2026-09-06, this configuration: the four ``B ≠ 0`` rows read
        9.595e-13 … 2.051e-11 against a 1e-8 band; the four vacuum arms read
        **0.000e+00 exactly**, both orders.
        """
        sol = _solve(arm_id, order)
        returned = sol.boundary_flux
        reflected = copy.deepcopy(returned)
        reflect_outflow_into_inflow(reflected, sol.mesh)
        got = _trace_values(returned)
        want = _trace_values(reflected)
        scale = float(np.max(np.abs(got)))
        if scale == 0.0:
            pytest.fail(
                f"{arm_id}[L={order}]: the returned trace is identically "
                f"zero, so neither leg of this row can discriminate."
            )
        # ``|B·ψ|`` itself — the ACTIVATION datum the partition is asserted
        # from: the reflected buffer's INFLOW rows carry ``B·outflow`` and
        # nothing else (the helper zeroes them before the additive reflect),
        # so their magnitude is the boundary operator's action alone; reading
        # ``max|reflected|`` over the whole trace would pick up the untouched
        # outflow rows and read non-zero even where ``B`` is the zero
        # morphism.  (Until CS4c step 6 item 6.5 this datum came from the
        # retired boundary-only ``reflect_into_inflow`` source, whose outflow
        # rows were zero by construction; the exclusion is by INDEX now.)
        _trace = sol.mesh.angular_trace
        b_magnitude = max(
            float(np.max(np.abs(
                reflected.face_view(face)[_trace.inflow_indices_for_face(face)]
            )))
            for face in reflected.layout.faces
        )
        residual = float(np.max(np.abs(got - want))) / scale

        declared_vacuum = arm_id in _VACUUM_ARM_IDS
        if declared_vacuum:
            assert residual == 0.0, (
                f"{arm_id}[L={order}] is declared VACUUM (B = 0) but the "
                f"reflective-law residual is {residual:.4e}, not exactly "
                f"0.0 — the arm's boundary is no longer the zero morphism, "
                f"so it has silently joined the population this row can "
                f"actually discriminate on.  Move it out of "
                f"_VACUUM_ARM_IDS."
            )
            return
        assert b_magnitude > 0.0, (
            f"{arm_id}[L={order}] is NOT declared vacuum, yet |B·ψ_ret| = "
            f"{b_magnitude:.4e} — the boundary operator is the zero morphism "
            f"here, so this row reads 0 == 0 and gates nothing (vv #20). "
            f"Either the arm's BCs changed or it belongs in _VACUUM_ARM_IDS."
        )
        assert residual <= _BAND, (
            f"{arm_id}[L={order}]: the RETURNED trace does not satisfy its "
            f"own reflective law — max|ψ.inflow − (B·ψ)|_inflow| / max|ψ| = "
            f"{residual:.4e} > band {_BAND:.1e}.  Since #448 the finalize is "
            f"one fixed_point_step and B arrives as a GAIN, so this identity "
            f"is a property of the returned object, not of a hand-written "
            f"assignment: a red here means the reconstruction delivered a "
            f"trace the boundary condition does not close on."
        )

    @pytest.mark.l1
    @pytest.mark.verifies("reflective-bc")
    @pytest.mark.parametrize("arm_id", _ARM_IDS)
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_returned_trace_is_the_converged_iterates(
        self, arm_id: str, order: int,
    ):
        r"""The returned trace reproduces ``_inner.iterate``'s to the band.

        The complement of the law leg, and NOT redundant with it: on the
        vacuum arms the law leg is a structural ``0.0`` while this one still
        carries information — `[M]` ``slab_vac`` L = 0 reads law ``0.000e+00``
        and ``1.779e-11`` here.  Its reference is the CONVERGED iterate the
        finalize reconstructed from, captured in-process
        (:func:`_capturing_solvers`), so it measures the polish step's effect
        on the trace rather than re-deriving what the trace should be.

        `[M]` 2026-09-06: 9.595e-13 … 2.051e-11 across all 16 rows.
        """
        sol = _solve(arm_id, order)
        inner = _inner_of(arm_id, order)
        got = _trace_values(sol.boundary_flux)
        want = _trace_values(_system_a_member(inner.iterate).boundary)
        scale = float(np.max(np.abs(got)))
        residual = float(np.max(np.abs(got - want))) / scale
        assert residual <= _BAND, (
            f"{arm_id}[L={order}]: the returned boundary trace differs from "
            f"the converged iterate's by {residual:.4e} > band {_BAND:.1e}. "
            f"The finalize is ONE fixed-point step from that iterate, so the "
            f"trace should move by at most the step's own residual."
        )


class TestTheReturnedRayMember:
    r"""System B's ψ½ — the last member of the returned ``Solution`` that
    nothing read.

    ``G1`` is a System-A BULK identity and ``TestTheReturnedTrace`` a
    System-A TRACE one; ``Solution.radial_characteristic`` — the marched
    half-angle flux a carrying mesh's coupled solve produces — was in
    neither's reach.  Since #448 the finalize is one ``fixed_point_step``
    through the COUPLED ``M``, so the ray member comes out of
    ``_system_b_member(final_state)`` rather than being carried over from
    the iterate: it is reconstructed like everything else, and it owes the
    same claim.

    ⚠ **This class is the ψ½ analogue of the trace class, not a duplicate
    of it.**  The carrying arms' ``G1`` rows already reddened under the
    coupled-seed arms (`[M]` ``P4b``: the fission q½ seed × 1.5 reddens
    ``G1`` on both), which shows System A FEELS a wrong ψ½ — it does not
    show the returned ψ½ is the one the solve converged, and a finalize
    that reconstructed System A correctly while handing back a
    differently-sourced System B would satisfy every other row in this file.

    ⛔ **These rows carry NO ``catches("ERR-083")``, and the omission is
    MEASURED, not cautious.**  They are provenance/wiring gates — the ψ½
    analogue of the trace class, marked the same way (``verifies`` only).
    ERR-083 is an ANGULAR-MOMENT defect on System A's bulk source (the
    ℓ ≥ 1 half of both collision channels absent from the reconstruction's
    rhs), while System B's half of the driver source is the ℓ = 0 fold of
    the FISSION source alone — the gains carry everything else, so a
    dropped ℓ ≥ 1 emission never reaches it.  `[M]` 2026-09-06, battery arm
    ``P1b`` (``_redistribute_moments → 0`` inside the finalize window, the
    closest installable model of the documented bug): **9 of 9 rows GREEN**,
    while the same arm reddens 6 elsewhere in this file.  That green is the
    PARTITION between the two systems, not a defect — and a ``catches``
    naming a bug this class cannot see would have read as coverage in the
    audit and been none (same-area is not coverage).

    **What DOES redden them** — `[M]` 2026-09-06, all windowed at the
    finalize so the converged iterate stays pristine (vv #18/#22):

    ===========================================  ==========  ===============
    arm                                          value rows  positivity
    ===========================================  ==========  ===============
    ``P0``   M⁻¹ skipped entirely                  4 of 4     2 of 4 (sphere)
    ``P7``   ray seed × 1.05                       4 of 4     0 of 4
    ``P8``   ray seed × −1                         4 of 4     0 of 4
    ``P8b``  ray seed × −50                        4 of 4     4 of 4
    ``P1b``  ℓ ≥ 1 emission dropped                0 of 4     0 of 4
    ===========================================  ==========  ===============

    ``P8`` is the informative one, and it is why the positivity leg carries
    its own pricing below: a SIGN flip on the fission seed does not leave
    the cone, because that seed is a minority of the ψ½ march's drive
    (`[M]` min ψ½ on the sphere falls only 7.540e-04 → 6.004e-04, and the
    cylinder's does not move).  The partition row is mutation-INVARIANT by
    design — it asserts a structure, and no arm here moves one.
    """

    #: The arms whose mesh CARRIES a radial-characteristic space, so the
    #: returned ``Solution`` has a System-B member at all.  DECLARED here
    #: and ASSERTED below against the live partition — the same discipline
    #: as ``_VACUUM_ARM_IDS``, and for the same reason: a mesh change must
    #: not silently move an arm out of the population that can discriminate.
    CARRYING = ("sphere_refl", "cylinder_vac")

    @pytest.mark.l1
    @pytest.mark.verifies("sn-direct-seed-augmented-composite")
    def test_the_carrying_partition_is_what_the_table_declares(self):
        """``radial_characteristic`` is present on exactly the declared arms.

        One row, not eight: the partition is ONE claim, and parametrising it
        would inflate the count without adding a case (vv #20).  It reddens
        if an arm gains or loses its ray space — which is exactly when the
        two rows below silently stop covering what they name.
        """
        for arm_id in _ARM_IDS:
            for order in (0, _L_ANISO):
                sol = _solve(arm_id, order)
                inner = _inner_of(arm_id, order)
                returned = sol.radial_characteristic
                converged = _system_b_member(inner.iterate)
                declared = arm_id in self.CARRYING
                assert (returned is not None) is declared, (
                    f"{arm_id}[L={order}]: Solution.radial_characteristic is "
                    f"{'present' if returned is not None else 'None'} but the "
                    f"arm is {'' if declared else 'NOT '}declared carrying. "
                    f"Update TestTheReturnedRayMember.CARRYING — the two rows "
                    f"below cover only the declared arms."
                )
                assert (converged is not None) is declared, (
                    f"{arm_id}[L={order}]: the CONVERGED iterate's System-B "
                    f"member disagrees with the declared partition — the "
                    f"reference the row below compares against does not exist."
                )

    @pytest.mark.l1
    @pytest.mark.verifies("sn-direct-seed-augmented-composite")
    @pytest.mark.parametrize("arm_id", CARRYING)
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_returned_ray_member_is_the_converged_iterates(
        self, arm_id: str, order: int,
    ):
        r"""``Solution.radial_characteristic`` reproduces ``_inner.iterate``'s
        System-B member — INTERIOR and BOUNDARY — to the band.

        Both members, because they are produced differently: the interior is
        the inward march's marched ψ½, the boundary its ray corner (the
        ``B_b`` datum) — `[M]` they read orders of magnitude apart, so a
        defect confined to one of them survives a single-member check.

        ⚠ **What this row canNOT distinguish, stated so nobody credits it
        with more:** a finalize handing back the iterate's ψ½ *unchanged*
        passes, and must — at a fixed point the stepped and the carried-over
        states agree to the step's own residual BY DEFINITION of
        convergence, so no test at this tier separates them.  What it
        catches is a ψ½ built from a DIFFERENT source or through a different
        operator — the #448 shape on System B, and what ``P0``/``P7``/``P8``
        model.

        `[M]` 2026-09-06: interior 1.524e-11 … 2.477e-11, boundary
        1.084e-11 … 1.627e-12 across the four rows.
        """
        returned = _solve(arm_id, order).radial_characteristic
        converged = _system_b_member(_inner_of(arm_id, order).iterate)
        assert returned is not None and converged is not None

        for member in ("interior", "boundary"):
            got = np.asarray(
                getattr(returned, member).values, dtype=np.float64,
            )
            want = np.asarray(
                getattr(converged, member).values, dtype=np.float64,
            )
            scale = float(np.max(np.abs(want)))
            if scale == 0.0:
                pytest.fail(
                    f"{arm_id}[L={order}]: the converged ψ½ {member} is "
                    f"identically zero, so this leg cannot discriminate."
                )
            residual = float(np.max(np.abs(got - want))) / scale
            assert residual <= _BAND, (
                f"{arm_id}[L={order}]: the returned ψ½ {member} differs from "
                f"the converged iterate's by {residual:.4e} > band "
                f"{_BAND:.1e}.  The finalize is ONE fixed-point step through "
                f"the coupled M, so System B should move by at most that "
                f"step's own residual — a red means the ray member was "
                f"re-marched from a different source, or carried over stale."
            )

    @pytest.mark.l1
    @pytest.mark.parametrize("arm_id", CARRYING)
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_returned_ray_member_stays_in_the_positive_cone(
        self, arm_id: str, order: int,
    ):
        r"""ψ½ ≥ 0, and the exact zeros are exactly the vacuum corners.

        The inward march carries a positive emission into a positive medium,
        so the marched half-angle flux is strictly positive in the INTERIOR —
        `[M]` ``min ψ½`` 7.54e-04 / 7.84e-04 (sphere), 2.03e-05 / 2.78e-05
        (cylinder).  A sign flip in the seed, a dropped ``1/W``, or a
        march run in the wrong direction breaks it O(1), and none of those
        is visible to a relative-difference row that compares two equally
        wrong arrays.

        ⚠ **Priced, because it did not redden where I first expected.**
        `[M]` 2026-09-06 the leg needs a LARGE excursion: the seed
        sign-flipped (``P8``) leaves ψ½ positive on both arms, and it takes
        ×−50 (``P8b``) to reach −5.05e-02 (sphere) / −7.62e-02 (cylinder).
        ``P0`` — the finalize's M⁻¹ skipped outright — reddens it on the
        sphere at both orders.  So the leg is not inert, but it is a COARSE
        instrument: it guards the O(1) failures its message names and
        nothing finer.  The value row above is the precise one.

        ⚠ The BOUNDARY is a different claim and the split is load-bearing:
        `[M]` the vacuum arm's corner is **exactly 0.0** (nothing enters at
        ``r = R``) while the reflective arm's is **1.17e-03 … 1.19e-03**.
        So a blanket ``> 0`` would be false on ``cylinder_vac`` and a blanket
        ``>= 0`` would be vacuous on ``sphere_refl``; the row asserts the
        cone on both and then DISCRIMINATES on the BC, which is what makes
        each arm's leg carry information.
        """
        returned = _solve(arm_id, order).radial_characteristic
        assert returned is not None
        interior = np.asarray(returned.interior.values, dtype=np.float64)
        boundary = np.asarray(returned.boundary.values, dtype=np.float64)

        assert float(np.min(interior)) > 0.0, (
            f"{arm_id}[L={order}]: the returned ψ½ interior reaches "
            f"{float(np.min(interior)):.4e} — the inward march of a positive "
            f"emission through a positive medium cannot leave the cone, so a "
            f"non-positive value is a seed sign, a lost 1/W, or a reversed "
            f"march."
        )
        assert float(np.min(boundary)) >= 0.0, (
            f"{arm_id}[L={order}]: the returned ψ½ corner reaches "
            f"{float(np.min(boundary)):.4e} < 0."
        )
        corner_min = float(np.min(boundary))
        if arm_id in _VACUUM_ARM_IDS:
            assert corner_min == 0.0, (
                f"{arm_id}[L={order}] is a VACUUM arm, so its r = R ψ½ "
                f"corner must be exactly 0.0 (nothing enters); it reads "
                f"{corner_min:.4e}.  Either the BC changed or the corner is "
                f"being seeded from somewhere."
            )
        else:
            assert corner_min > 0.0, (
                f"{arm_id}[L={order}] is a REFLECTIVE arm, so its r = R ψ½ "
                f"corner is fed by B_b and must be positive; it reads "
                f"{corner_min:.4e}.  A zero here means the corner reflect "
                f"never reached System B."
            )


class TestTheGaussSeidelArmPosesItsOwnSplitting:
    """``cart2d_gs``'s precondition — without it the arm is a duplicate.

    `[M]` 2026-09-06 (qa F-1): ``solve_sn``'s ``inner_schedule`` default is
    ``"jacobi"``, so before this arm existed **0 of 161** inner solves in
    this module posed the boundary-Gauss-Seidel splitting, and the finalize's
    ``ScheduledInvertibleOperator`` reconstruction arm had no
    self-consistency witness anywhere.
    """

    @pytest.mark.l1
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_gs_arm_reconstructs_through_a_scheduled_operator(
        self, order: int,
    ):
        """G-S poses a DIFFERENT ``M`` and a DIFFERENT ``B`` gain than Jacobi.

        The finalize applies ``inner.implicit.inverse()`` with
        ``inner.gains`` — so the *object* the reconstruction runs through is
        what this row pins.  `[M]` Jacobi gives
        ``StreamingCollisionOperator`` + ``SNBoundaryOperator``; G-S gives
        ``ScheduledInvertibleOperator`` + ``SNMaskedBoundaryOperator``.
        """
        jacobi = _inner_of("cart2d", order)
        gs = _inner_of("cart2d_gs", order)
        assert type(gs.implicit).__name__ == "ScheduledInvertibleOperator", (
            f"cart2d_gs[L={order}] posed a "
            f"{type(gs.implicit).__name__} — the boundary-Gauss-Seidel "
            f"splitting did not reach the finalize, so this arm is a "
            f"duplicate of cart2d and the scheduled reconstruction arm is "
            f"un-witnessed (qa F-1)."
        )
        assert type(jacobi.implicit).__name__ != type(gs.implicit).__name__, (
            "cart2d and cart2d_gs pose the SAME implicit operator — the "
            "schedule knob stopped selecting, so the pair is inflation."
        )
        gs_gains = [type(g).__name__ for g in gs.gains]
        jac_gains = [type(g).__name__ for g in jacobi.gains]
        assert gs_gains != jac_gains, (
            f"cart2d and cart2d_gs drive the SAME gains ({gs_gains}) — the "
            f"G-S split of B did not happen."
        )

    @pytest.mark.l1
    @pytest.mark.verifies("dd-solve", "transport-cartesian")
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_schedule_does_not_move_the_converged_answer(
        self, order: int,
    ):
        r"""vv Mode 9: a SPLITTING changes the rate, never the fixed point.

        Two coherent splittings of one ``A = M − N`` share ``ψ*``, so k and φ
        must agree to the solve's own tolerance while the reconstruction runs
        through two structurally different ``M``\ s.  `[M]` 2026-09-06:
        ``|Δk| = 6.28e-14`` (L0) / ``6.21e-14`` (L1), ``|Δφ|/max = 2.35e-12``
        / ``2.00e-12``.

        ⚠ Mode 9's own caveat applies and is why this row is here rather than
        assumed: the 2-D Cartesian box is VACUUM in x, so ``A`` is
        non-singular (#344's ``ker A`` needs ≥ 2 reflective axis pairs) and
        the two splittings really do share a POINT, not a manifold.
        """
        jac, gs = _solve("cart2d", order), _solve("cart2d_gs", order)
        k_j, k_g = jac.keff, gs.keff
        assert k_j is not None and k_g is not None
        np.testing.assert_allclose(
            k_g, k_j, rtol=_BAND, atol=0.0,
            err_msg=(
                f"boundary Gauss-Seidel moved keff (L={order}): {k_g!r} vs "
                f"Jacobi {k_j!r}.  A splitting must change the RATE, never "
                f"the converged fixed point (vv Mode 9)."
            ),
        )
        phi_j = np.asarray(jac.scalar_flux.values, dtype=np.float64)
        phi_g = np.asarray(gs.scalar_flux.values, dtype=np.float64)
        rel = float(np.max(np.abs(phi_g - phi_j)) / np.max(np.abs(phi_j)))
        assert rel <= _BAND, (
            f"boundary Gauss-Seidel moved the converged scalar flux "
            f"(L={order}) by {rel:.4e} > {_BAND:.1e}."
        )


# ══════════════════════════════════════════════════════════════════════
# G3 — independent instruments (NOT the fix's own body)
# ══════════════════════════════════════════════════════════════════════

class TestAgainstAnIndependentRoute:
    """Two instruments that share no code with the finalize's rhs assembly."""

    @pytest.mark.l1
    @pytest.mark.verifies("pn-scatter", "n2n-source")
    @pytest.mark.catches("ERR-083")
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0-control", "L1"])
    def test_returned_flux_equals_an_independent_fixed_source_resolve(
        self, order: int,
    ):
        r"""Re-solve :math:`(L+C-S-N_{2n}-B)\psi = F\phi_{\rm conv}/k` through
        ``solve_sn_fixed_source`` and compare the two angular fluxes.

        **Why this is an independent route and not #22's shared-object
        tautology.**  The two sides share the OPERATORS (they must — the
        claim is about which source the finalize assembles, not about the
        operator algebra) and share NOTHING of the reconstruction: the
        eigenvalue entry hand-builds its rhs at ``solver.py:2577-2579`` and
        applies ``final_implicit.solve`` once; the fixed-source SI entry
        drives ``S``/``N₂ₙ``/``B`` as lagged gains to convergence and, on
        this un-windowed arm, returns the converged ITERATE with no
        reconstruction at all (``solver.py:4041``).  Neither calls the other.

        **The φ leg is this row's own positive control.**  `[M]` the two
        SCALAR fluxes agree to 1.7e-12 at BOTH orders — which proves the two
        entries really are posing the same problem, so the ψ leg's 1.47e-01
        at L = 1 is a statement about the reconstruction and not about a
        mis-posed oracle.
        """
        eig = _solve("slab_vac", order)
        keff, sn_mesh = eig.keff, eig.mesh
        assert keff is not None
        phi_conv = np.asarray(eig.scalar_flux.values, dtype=np.float64)

        # The converged fission source, as the per-ordinate q_ext the
        # fixed-source entry takes (producer-side /W, R-1 Step 4 A1).
        probe = SNSolver(sn_mesh, scattering_order=order)
        fission = probe.compute_fission_source(phi_conv, keff)
        quad = Quadrature.gauss_legendre(n_ordinates=8)
        q_ext = np.broadcast_to(
            fission / float(quad.weights.sum()),
            (quad.N,) + fission.shape,
        ).copy()

        with contextlib.redirect_stdout(io.StringIO()):
            ref = solve_sn_fixed_source(
                _LIBRARY, _slab(BC.vacuum), quad, q_ext,
                scattering_order=order, inner_solver="source_iteration",
                max_inner=200_000, inner_tol=_INNER_TOL / 10.0,
            )

        phi_ref = np.asarray(ref.scalar_flux.values, dtype=np.float64)
        rel_phi = float(
            np.max(np.abs(phi_conv - phi_ref)) / np.max(np.abs(phi_ref))
        )
        assert rel_phi <= _BAND, (
            f"POSITIVE CONTROL FAILED (L={order}): the fixed-source re-solve's "
            f"scalar flux differs from the eigenvalue's by {rel_phi:.3e} > "
            f"{_BAND:.1e}.  The oracle is not posing the same problem, so the "
            f"ψ comparison below carries no information — fix the oracle "
            f"before reading its verdict."
        )

        psi_eig = np.asarray(
            eig.angular_flux.interior.values, dtype=np.float64,
        )
        psi_ref = np.asarray(
            ref.angular_flux.interior.values, dtype=np.float64,
        )
        rel_psi = float(
            np.max(np.abs(psi_eig - psi_ref)) / np.max(np.abs(psi_ref))
        )
        assert rel_psi <= _BAND, (
            f"L={order}: the eigenvalue entry's returned angular flux differs "
            f"from an independent fixed-source solve of its OWN converged "
            f"fission source by {rel_psi:.4e} > band {_BAND:.1e}, while the "
            f"two scalar fluxes agree to {rel_phi:.2e}.  Same equation, same "
            f"operators, two reconstructions — the finalize's is the one that "
            f"differs (#448)."
        )


class TestTheShippedDiagnostic:
    """``history.balance_defect`` — the instrument that SHOULD have caught
    this, and the guard that keeps it asleep."""

    @pytest.mark.l1
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0", "L1"])
    def test_the_balance_defect_is_silent_on_a_converged_solve(self, order: int):
        """`[M]` ``None`` on every converged solve — by construction.

        ``_exit_balance_defect`` returns early on ``record.fully_converged``
        (it is the complement of ``_certify_within_group_exit``, which
        asserts on the converged side).  But the eigenvalue entry never runs
        the certificate on its RETURNED ψ — the certificate fires inside the
        inner solves, on the ITERATE.  So between the two of them nothing
        ever evaluates the object the caller receives, which is precisely the
        hole #448 lived in for the whole of its life.

        This row is a CHARACTERIZATION: it pins the fact so the next reader
        does not spend a cycle wondering why an existing diagnostic did not
        fire, and it is the premise the budget row below depends on.
        """
        history = _solve("slab_vac", order).history
        assert history is not None
        assert history.balance_defect is None, (
            f"balance_defect = {history.balance_defect} on a fully-converged "
            f"solve.  The early return on record.fully_converged has moved; "
            f"the budget row below assumes the defect is only live on a "
            f"TRUNCATED exit and must be re-derived."
        )

    @pytest.mark.l1
    @pytest.mark.verifies("mg-balance", "pn-scatter")
    @pytest.mark.catches("ERR-083")
    @pytest.mark.parametrize("order", [0, _L_ANISO], ids=["L0-control", "L1"])
    def test_the_balance_defect_responds_to_the_outer_budget(self, order: int):
        r"""On a TRUNCATED exit the reported defect must FALL with the budget.

        ``_exit_balance_defect`` evaluates the shipped loss arm
        :math:`L+C-S-B` on the RETURNED :math:`\psi` against a fission-only
        rhs — an instrument that shares nothing with the finalize's source
        assembly.  Its docstring forbids THRESHOLDING it ("a diagnostic,
        never a gate; do not branch on it, do not assert on its magnitude").
        This row does not: it asks only whether the number RESPONDS to the
        thing it claims to measure.  A truncation defect must shrink as the
        truncation is relaxed; a RECONSTRUCTION defect cannot.

        `[M]` on the defective tree, ``max_outer`` 3 → 12:

        * ``L = 0``: 1.250e-05 → 8.593e-12 — falls by **1.45e6 ×**;
        * ``L = 1``: 7.4865e-02 → 7.4847e-02 — falls by **1.0002 ×**, i.e.
          the number the user is shown as "how truncated was I" is a FLOOR
          set by #448 and is reading the reconstruction instead.

        ⚠ OPEN RULING (see ``scratch/_448_verification_plan.md`` §R3): if the
        user reads ``_exit_balance_defect``'s prohibition as absolute — no
        assertion of any kind on this quantity — this row retires and the
        finding survives as prose in the error-catalogue entry.
        """
        defects: dict[int, float] = {}
        for budget in (3, 12):
            with contextlib.redirect_stdout(io.StringIO()):
                with pytest.warns(ConvergenceWarning):
                    sol = solve_sn(
                        _LIBRARY, _slab(BC.vacuum),
                        Quadrature.gauss_legendre(n_ordinates=8),
                        scattering_order=order,
                        inner_solver="source_iteration",
                        keff_tol=1e-12, flux_tol=1e-12,
                        inner_tol=_INNER_TOL, max_outer=budget,
                    )
            history = sol.history
            assert history is not None
            if history.fully_converged:
                pytest.fail(
                    f"max_outer={budget} converged — the truncation this row "
                    f"needs did not happen, so balance_defect is None and the "
                    f"row measures nothing.  Lower the budget."
                )
            defect = history.balance_defect
            if defect is None:
                pytest.fail(
                    f"balance_defect is None on a truncated exit "
                    f"(max_outer={budget}); the reporting arm has moved."
                )
            defects[budget] = float(defect)

        ratio = defects[3] / defects[12] if defects[12] > 0.0 else np.inf
        assert ratio > 100.0, (
            f"L={order}: the exit balance defect barely moved when the outer "
            f"budget went 3 → 12 ({defects[3]:.4e} → {defects[12]:.4e}, "
            f"ratio {ratio:.4g}).  A TRUNCATION defect shrinks with the "
            f"budget; this one does not, so the number the diagnostic reports "
            f"is dominated by something the budget cannot fix — the returned "
            f"ψ not solving its own equation (#448)."
        )


# ══════════════════════════════════════════════════════════════════════
# G5 — the same identity on PRODUCTION 421-group data (l2)
# ══════════════════════════════════════════════════════════════════════

_N_U_METAL = 0.04894  # /b·cm, 19.1 g/cc
_N_BE = 0.1236        # /b·cm, 1.85 g/cc
_TEMP_K = 294
#: `[M]` the Be band is one decade wider than the fast one, and the reason is
#: stated rather than assumed: this fixture runs at ``inner_tol = 1e-10``
#: (the #426 §0 configuration, kept verbatim so the two files' k values stay
#: comparable), so ``SAFETY × max(inner_tol, flux_tol) = 10 × 1e-8 = 1e-7``.
_BE_KEFF_TOL = 1e-9
_BE_FLUX_TOL = 1e-8
_BE_INNER_TOL = 1e-10
_BE_BAND = _SAFETY * max(_BE_INNER_TOL, _BE_FLUX_TOL)
_BE_L_ANISO = 2


def _be_mixture(names: list[str], densities: list[float]) -> Mixture:
    """Pure-isotope macroscopic mixture from the tracked ``.GXS`` library.

    ⚠ The σ₀ solve divides by zero on a pure isotope and clips to σ₀ = 1 b,
    identically in every arm.  The filter is NARROW; anything else is surfaced
    (the ``qa`` D4 finding on the #426 sibling).
    """
    isotopes = [load_isotope(name, _TEMP_K) for name in names]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            mix = compute_macro_xs(
                isotopes, np.asarray(densities), escape_xs=0.0,
            )
    unexpected = [
        w for w in caught
        if not issubclass(w.category, RuntimeWarning)
        or ("divide by zero" not in str(w.message)
            and "invalid value" not in str(w.message))
    ]
    if unexpected:
        pytest.fail(
            "unexpected warnings from the σ₀ path — the pure-isotope clip is "
            f"the only one this fixture licenses: "
            f"{[str(w.message) for w in unexpected]}"
        )
    return mix


@pytest.fixture(scope="module")
def be_library() -> dict[int, Mixture]:
    """`[M]` 0.29 s, once for the module."""
    return {
        0: _be_mixture(["U_235"], [_N_U_METAL]),
        1: _be_mixture(["BE009"], [_N_BE]),
    }


@pytest.fixture(scope="module")
def be_mesh() -> Mesh1D:
    """The #426 §0 fixture verbatim: Be | U-235 metal | Be, vacuum."""
    geom = StructuredGeometry(
        geometry="SLB",
        regions=(
            Region(mat_id=1, outer_thickness_cm=3.0),
            Region(mat_id=0, outer_thickness_cm=4.0),
            Region(mat_id=1, outer_thickness_cm=3.0),
        ),
        bcs=(BC.vacuum, BC.vacuum),
    )
    return Mesh1D.from_geometry(geom, region_meshes=(
        RegionMesh(n_cells=12), RegionMesh(n_cells=16), RegionMesh(n_cells=12),
    ))


class TestOnProductionData:
    """The same identity, on the 421-group Be-reflected fixture whose (n,2n)
    anisotropy is REAL tape data rather than a manufactured stack."""

    @pytest.mark.l2
    @pytest.mark.verifies("pn-scatter", "n2n-source", "flux-moments")
    @pytest.mark.catches("ERR-083")
    @pytest.mark.parametrize(
        "order", [0, _BE_L_ANISO], ids=["L0-control", "L2"],
    )
    def test_be_reflected_returned_flux_integrates_to_its_scalar_flux(
        self, be_library: dict[int, Mixture], be_mesh: Mesh1D, order: int,
    ):
        """`[M]` L2 reads **3.405e-02** today against a 1e-7 band; the L0
        control reads 1.6e-10.

        Why this row exists beside the fast ones: it is the only place both
        ℓ ≥ 1 legs are simultaneously live on data nobody authored — the
        elastic Legendre stack AND the (n,2n) stack #426 step 2 landed
        (ERR-082).  `[M]` on the fast arms the (n,2n) leg is present only
        through ``slab_vac_n2n``'s manufactured moment.

        `[M]` 8.9 s at ``scattering_order=2``, 4.4 s at 0.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            sol = solve_sn(
                be_library, be_mesh, Quadrature.gauss_legendre(n_ordinates=8),
                scattering_order=order, inner_solver="source_iteration",
                keff_tol=_BE_KEFF_TOL, flux_tol=_BE_FLUX_TOL,
                inner_tol=_BE_INNER_TOL, max_outer=3000,
            )
        history = sol.history
        if history is None or not history.fully_converged:
            pytest.fail(
                f"the Be arm did not fully converge (scattering_order={order})"
            )
        residual = _self_consistency(sol)
        assert residual <= _BE_BAND, (
            f"be_reflected[scattering_order={order}]: "
            f"max|∫ψ dΩ − φ| / max|φ| = {residual:.4e} > band "
            f"{_BE_BAND:.1e}.  On production 421-group data the returned "
            f"angular flux does not reduce to the reported scalar flux — "
            f"the finalize dropped the ℓ ≥ 1 emission from its rhs (#448); "
            f"at this order that is BOTH the elastic Legendre stack and the "
            f"(n,2n) stack (ERR-082)."
        )
