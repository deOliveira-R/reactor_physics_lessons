r"""Octant-group sweep SCHEDULE — the polymorphic Jacobi / Gauss-Seidel strategy.

Phase 3 (SI Gauss-Seidel rate recovery). The Wave O BC extraction made the
transport sweep BARE and applies the reflective coupling ``B`` externally, which
turned the intra-sweep Gauss-Seidel reflective coupling into inter-sweep Jacobi
(``B`` fully lagged) — same converged fixed point, slower SI spectral rate. The
recovery interleaves the external ``−B`` reflect at octant-group granularity
inside the SI resolvent, realizing the forward substitution ``(L+C−B_lower)⁻¹``.

Jacobi and Gauss-Seidel are the SAME uniform sweep-and-reflect loop differing
ONLY in this schedule (there is NO ``if jacobi/gs`` branch in the iteration — the
splitting is selected ONCE, by choosing the schedule):

* **Jacobi** — ONE group containing every octant, with NO inter-group reflect.
  All octants read the same frozen inflow seed (``B·ψₙ``); identical to the
  pre-recovery bare all-octants sweep.
* **Gauss-Seidel** — one group per in-plane octant (:class:`OctantLabel`), in
  quadrature sweep order; after each group its reflective OUTGOING faces are
  re-reflected (the face-restricted
  :meth:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace`,
  bound by the scheduled resolvent),
  so a LATER group reads the fresh current-iterate reflection. Octants swept
  before their specular partner keep the lagged seed (the cyclic ``B_upper``
  back-edges — e.g. a both-faces-reflective axis is a 2-cycle ⟹ only PARTIAL
  one-pass G-S); octants swept after read the fresh value (the order-respecting
  ``B_lower`` edges).

The schedule is a **mesh-time derived object** — it depends only on the
quadrature's octant partition + the mesh's reflective-face set, not on fluxes /
sources / iteration state — so it is built once and reused across every SI
iterate (the same lifetime contract as
:class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`).

The fixed point is INVARIANT under the schedule: any consistent splitting of
``(L+C−S−B)ψ=q`` shares the dominant solution ψ\* (at convergence the seed and
all re-reflects equal ``B·ψ\*``). The schedule changes only the SI spectral rate.
Krylov is splitting-invariant and ignores the schedule entirely.

.. warning::

   ⛔ **REFUTED 2026-08-09 (#341).** This paragraph used to close with
   *"(``ρ_J = c`` Jacobi vs ``ρ_GS ≈ c²`` for the symmetric reflective model
   problem)"*.  That is the textbook result for **scattering** Gauss-Seidel,
   and this schedule folds **B**, not **S** — the sweep never re-scatters
   mid-sweep (see :func:`~orpheus.sn.solver._select_si_splitting`).  Quoting a
   scattering-splitting rate for a boundary splitting imported a law from the
   wrong operator.

   `[M]` it fails in **both** directions.  Zero-leakage pure absorber,
   Σ_t = (0.8, 1.6), level-symmetric S4, 2 groups
   (``scratch/probe_inner_budget_law.py`` for ρ_GS,
   ``scratch/probe_341_iteration_spectrum.py`` for ρ_J):

   ===========  ==========  ==============  ============
   geometry     ρ_GS meas.  ``ρ_J²`` claim  ρ_J measured
   ===========  ==========  ==============  ============
   d=2 (3,4)    0.90641     0.9286          0.9636
   d=3 (3,4,5)  0.98538     0.9514          0.97541
   ===========  ==========  ==============  ============

   So G-S beats the claim at d=2 and **loses to it** at d=3.  The mechanism is
   in :ref:`sn-boundary-gs-rate-regime`: at zero leakage ``B``'s octant action
   is the hypercube ``Q_d``, whose intra-octant DD transmission
   ``Σ = (2/D)·1wᵀ − I`` is rank-one-minus-identity — one absorption-damped
   eigenvalue ``1 − 2Σ_tV/D`` plus **d−1 eigenvalues exactly −1** (the
   undamped zero-cell-average face sawtooth, which ``Σ_t V ψ_c`` cannot see).
   No single closed-form rate law survives that, which is why the honest
   statement here is now the *invariance* of the fixed point and nothing about
   the rate.

See also
========

* :mod:`orpheus.sn.loss_representation.sweep_graph` — :class:`OctantLabel` + the per-octant
  :class:`SweepDependencyGraph` (the cell DAG the schedule's groups are swept
  through; a distinct concept — cell causality vs octant ordering).
* ``.claude/agent-memory/explorer/si_gs_substep3_carve_substrate.md`` — the
  carve substrate map (octant↔face geometry, the resolvent solve path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from orpheus.geometry.boundary import law_permutes_ordinates
from orpheus.numerics.face_layout import face_name
from .sweep_graph import OctantLabel

if TYPE_CHECKING:
    from orpheus.numerics.measure import DiscreteMeasurePartition
    from orpheus.sn.mesh.augmented_mesh import SNMesh


__all__ = ["OctantSweep", "OctantSweepGroup", "SweepSchedule"]


@dataclass(frozen=True, slots=True)
class OctantSweep:
    """One octant's sweep unit: its in-plane :class:`OctantLabel` (selects the
    per-octant :class:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph`) + the
    ordinate indices into the ``(N, …)`` angular axis it sweeps."""

    label: OctantLabel
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OctantSweepGroup:
    """Octants swept together over ONE frozen inflow; after the group, its
    ``reflect_faces`` are re-reflected so later groups read the fresh inflow.

    ``reflect_faces`` is empty for the Jacobi group, for a grazing / pure-z
    octant (no in-plane outflow), and for non-reflective (vacuum / white)
    boundaries — in all of which the inter-group reflect is a no-op.
    """

    sweeps: tuple[OctantSweep, ...]
    reflect_faces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SweepSchedule:
    """Polymorphic octant-group schedule for the SI resolvent's uniform
    sweep-and-reflect loop. See the module docstring for the Jacobi/G-S split.
    """

    groups: tuple[OctantSweepGroup, ...]
    kind: str  # "jacobi" | "gauss_seidel" — diagnostic / introspection only

    @classmethod
    def jacobi(
        cls, ndim: int, octants: "tuple[DiscreteMeasurePartition, ...]",
    ) -> "SweepSchedule":
        """One group, all octants, no inter-group reflect — the bare all-octants
        sweep with the whole ``B·ψₙ`` seed frozen for the entire sweep.
        Mesh-free (un-weld arc O-1): callers pass ``(sn_mesh.ndim,
        sn_mesh.quad.octants)``."""
        sweeps = tuple(
            _octant_sweep(entry, ndim) for entry in octants
        )
        return cls(
            groups=(OctantSweepGroup(sweeps=sweeps, reflect_faces=()),),
            kind="jacobi",
        )

    @classmethod
    def gauss_seidel(
        cls,
        ndim: int,
        octants: "tuple[DiscreteMeasurePartition, ...]",
        reflective: frozenset[str],
    ) -> "SweepSchedule":
        """One group per in-plane octant, in quadrature sweep order; each group
        re-reflects the reflective faces its octants outflow through.
        Mesh-free (un-weld arc O-1): callers pass ``(sn_mesh.ndim,
        sn_mesh.quad.octants, reflective_faces(sn_mesh))``.

        Octant partition entries that share an in-plane :class:`OctantLabel`
        (they differ only in out-of-plane signs the in-plane sweep ignores —
        e.g. ``sign_z`` over a 2-D mesh) are MERGED into one group so a
        face's outflow is complete before it is reflected.
        """
        ordered: list[OctantLabel] = []
        by_label: dict[OctantLabel, list[OctantSweep]] = {}
        for entry in octants:
            sweep = _octant_sweep(entry, ndim)
            if sweep.label not in by_label:
                by_label[sweep.label] = []
                ordered.append(sweep.label)
            by_label[sweep.label].append(sweep)

        # Assign each reflective face to the LAST in-plane octant group (in
        # sweep order) that OUTFLOWS through it — reflecting only after that
        # group guarantees the face's outflow is COMPLETE (every octant that
        # streams out through it has been swept this pass), so the reflected
        # inflow is consistent.
        #
        # ⚠ Correctness (NOT just rate): a face is shared by EVERY octant whose
        # sign on that axis matches (e.g. ``xmax`` ← all +x octants).  For an
        # axis-aligned quadrature (``product`` — single-face octants) each face
        # has exactly ONE outflowing group, so "last" = that group.  But for a
        # diagonal / spherical cubature (``lebedev`` / ``level_symmetric`` —
        # each octant outflows TWO faces) a face is shared by ≥2 groups;
        # reflecting after the FIRST would absorb the not-yet-swept octants'
        # SEED value (the wavefront is rebuilt + seeded each solve, so their
        # outflow slots still hold the inflow seed, NOT real outflow) and
        # reflect garbage — converging to a WRONG fixed point.  Deferring to
        # the LAST outflowing group fixes this; octants reading the face that
        # are swept BEFORE its reflect keep the lagged seed (the cyclic
        # back-edge → partial one-pass G-S), which is always valid.
        last_group_for_face: dict[str, int] = {}
        for gi, label in enumerate(ordered):
            for f in _outgoing_faces(label):
                if f in reflective:
                    last_group_for_face[f] = gi   # later gi wins → the last
        reflect_by_group: dict[int, list[str]] = {
            gi: [] for gi in range(len(ordered))
        }
        for face, gi in last_group_for_face.items():
            reflect_by_group[gi].append(face)

        groups = tuple(
            OctantSweepGroup(
                sweeps=tuple(by_label[label]),
                reflect_faces=tuple(sorted(reflect_by_group[gi])),
            )
            for gi, label in enumerate(ordered)
        )
        return cls(groups=groups, kind="gauss_seidel")

    def lower_inflow_rows(self, sn_mesh: "SNMesh") -> dict[str, np.ndarray]:
        r"""Per-face inflow ordinate rows that read the CURRENT iterate under
        this schedule — the row support of the strictly-lower boundary part
        :math:`B_{\rm lower}` in the splitting
        :math:`(L+C-B) = \underbrace{(L+C-B_{\rm lower})}_{M} - B_{\rm upper}`.

        The split law (#226 §17 W2): face ``f`` is reflected exactly once,
        after its LAST outflowing octant group (``reflect_faces`` above), at
        which point every outflow feeding ``f`` is complete.  An inflow
        ordinate row ``(f, m')`` therefore reads the FRESH current-iterate
        reflection iff its octant group is swept strictly AFTER ``f``'s
        reflect group — those rows are ``B_lower`` (realized in-sweep by the
        forward substitution); rows swept at-or-before the reflect keep the
        lagged seed — ``B_upper`` (the cyclic back-edges, lagged by the
        driver).  The specular map flips one direction-cosine sign, so a row
        and its source always sit in DIFFERENT octants — ``B`` has no
        octant-diagonal and ``B = B_lower + B_upper`` is an exact partition.

        Returns the mapping only for faces this schedule reflects; a face
        never reflected in-sweep (vacuum, white, albedo, periodic — and every
        face under the Jacobi schedule, which returns ``{}``) has ALL its
        rows in ``B_upper``.  Consumed by
        :meth:`~orpheus.sn.operators.boundary.SNBoundaryOperator.split`.
        """
        reflect_gi = {
            face: gi
            for gi, group in enumerate(self.groups)
            for face in group.reflect_faces
        }
        trace = sn_mesh.angular_trace
        rows: dict[str, np.ndarray] = {}
        for face, gi_f in reflect_gi.items():
            fresh_ordinates = np.array(
                [
                    i
                    for group in self.groups[gi_f + 1 :]
                    for sweep in group.sweeps
                    for i in sweep.indices
                ],
                dtype=np.intp,
            )
            rows[face] = np.intersect1d(
                trace.inflow_indices_for_face(face), fresh_ordinates,
            )
        return rows


def _octant_sweep(entry, ndim: int) -> OctantSweep:
    """Project a quadrature octant partition entry to its in-plane sweep unit.

    The partition ``.label`` carries one sign per direction-space axis
    (``(sign_x[, sign_y[, sign_z]])``, each ∈ {−1, 0, +1}).  The in-plane
    :class:`OctantLabel` keeps the mesh's first ``ndim`` signs and projects
    the rest out (the in-plane sweep is invariant under the out-of-plane
    signs — e.g. a 2-D Cartesian mesh under an ``S²`` cubature drops
    ``sign_z``); a quadrature with FEWER signs than the mesh has axes
    (a slab quadrature over a multi-D mesh) zero-pads — sign ``0`` means
    "no streaming on this axis".  This is the SOLE in-plane projection
    site: every :class:`OctantLabel` downstream (the walk, the per-octant
    DAG keys, the G-S face assignment) has exactly ``ndim`` signs.
    """
    label = entry.label
    signs = tuple(
        int(label[a]) if a < len(label) else 0 for a in range(ndim)
    )
    return OctantSweep(
        label=OctantLabel(signs),
        indices=tuple(int(i) for i in entry.indices),
    )


def _outgoing_faces(label: OctantLabel) -> tuple[str, ...]:
    """The boundary faces an octant OUTFLOWS through (strict sign — a grazing
    ``sign == 0`` axis contributes no net outflow on that axis).

    Derived per axis from the label's own signs (an octant streaming in the
    ``+a`` direction outflows through the ``a``-max face), so the face set is
    correct at any ``ndim`` — there is no hand-listed axis table to fall out
    of date when a third axis arrives.
    """
    return tuple(
        face_name(a, +1 if s > 0 else -1)
        for a, s in enumerate(label.signs)
        if s != 0
    )


def reflective_faces(sn_mesh: "SNMesh") -> frozenset[str]:
    """The mesh's SPECULAR-reflective boundary faces.

    The question is *does this face's law RELABEL ordinates?* — a specular
    mirror sends outgoing ordinate :math:`n` to exactly one incoming ordinate,
    so the octant order survives and forward substitution is legal. Vacuum (no
    coupling at all) and white (couples ALL ordinates on the face ⟹ the
    octant-order Gauss-Seidel degenerates to Jacobi) are EXCLUDED.

    That is :attr:`BoundaryGeometryMap.permutes_ordinates`, and it is what the
    law is asked directly. Until campaign phase B2 this read
    ``sn_mesh.bc[face] == "reflective"`` — the same question spelled as a
    string comparison, which was all the pre-B2.0 shim could answer because it
    discarded the law at realization. The two agree on every registered law
    (``tests/geometry/test_boundary_factor_consumers.py`` compares the old
    tag expression against this one, law by law), so the repoint is
    behaviour-preserving; what it buys is that a NEW ordinate-
    permuting law joins this set by construction instead of by remembering to
    add its tag here.
    """
    return frozenset(
        face
        for face in sn_mesh.angular_trace.layout.faces
        if law_permutes_ordinates(sn_mesh.bc[face].law)
    )
