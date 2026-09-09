r"""Macroscopic cross-section field on a discretized (group × spatial) phase space.

The typed coefficient leaf for a macroscopic cross section

.. math::

    \Sigma_x(\vec r, g) \quad [\mathrm{cm^{-1}}]

— the total :math:`\Sigma_t`, absorption :math:`\Sigma_a`, fission production
:math:`\nu\Sigma_f`, or scattering-diagonal :math:`\Sigma_{s0}` — stored as the
broadcast per-cell ``(ng, *spatial)`` array every operator's per-cell math
consumes (the same layout as :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux`).

A cross section is a **coefficient**, not a state: it is the *symbol* of a
zeroth-order multiplication operator. Under the grand-report promotion
(§5.5–5.7) ``C = M[Σ_t]`` is this field *promoted* to a
:class:`~orpheus.numerics.operator.LinearOperator` (the multiplier-algebra
embedding ``M: L^∞ → B(L²)``; built in #257 S3). This leaf is the field side
of that promotion.

The coefficient algebra
=======================

:class:`CrossSectionField` carries
:class:`~orpheus.transport.fields._coefficient_role.CoefficientRole`
(historically the complement of the retired ``FluxRole`` affine gate — since
campaign 1 CS3, 2026-08-19, every field family shares this algebra). It keeps
the plain :class:`~orpheus.numerics.field.Field` vector-space algebra:

* ``Σ + Σ′`` is legitimate (homogenisation ``Σ_mix = Σ_m N_m Σ_m`` is a
  number-density-weighted sum).
* ``Σ = 0`` is a genuine origin (it promotes to ``M_0 = ZeroOperator``) — the
  coefficient space is a vector space *with* an origin (as, since campaign 1
  CS3, is every field family — this module's doctrine generalized).
* scalar ``λ·Σ`` and unary ``−Σ`` are inherited unchanged.

**Nonnegativity is the cone, a property — not a type invariant.** Physical
cross sections satisfy ``Σ ≥ 0`` (the cone), and the intrinsic-property tests
verify the cone is *closed* under the cone operations (``+``, ``λ≥0·``) and has
an origin. But the type does NOT reject a signed value at construction: a
cross-section *difference* ``Σ − Σ′`` (a perturbation) is itself a coefficient
and may be signed, and the multiplier-algebra ``M`` is a linear map on the full
(signed) coefficient vector space. Nonnegativity is enforced at the data
boundary (when reading a material library), not on every intermediate.

Units (View-G — the ``UNITS`` class constant)
==============================================

``1/cm`` (:data:`~orpheus.numerics.units.CROSS_SECTION_UNITS`). Multiplied into
a flux it yields the matching rate density — the #208 operator unit-gain
``ANGULAR_FLUX_UNITS × CROSS_SECTION_UNITS = ANGULAR_RATE_UNITS``.

References
----------

* Lewis, E.E. & Miller, W.F. (1993). *Computational Methods of Neutron
  Transport*. ANS. §1.1 — macroscopic cross sections.
* ``.claude/plans/issue_257_coefficient_field_promotion.md`` — S1.
* ``.claude/agent-memory/cross-domain-attacker/coefficient_field_promotion_frames.md``
  — Frame 1 (the multiplier-algebra embedding) + Frame 2 (the cone algebra).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from orpheus.numerics.units import CROSS_SECTION_UNITS, Unit
from orpheus.transport.fields._bases import ScalarField
from orpheus.transport.fields._coefficient_role import CoefficientRole

__all__ = ["CrossSectionField"]


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class CrossSectionField(CoefficientRole, ScalarField):
    r"""Macroscopic cross-section field :math:`\Sigma_x(\vec r, g)` ``[1/cm]``.

    Parameters
    ----------
    values : NDArray
        Cross-section values of shape ``(ng, *spatial)`` — ``(ng, nx)`` on a
        1-D mesh, ``(ng, nx, ny)`` on a 2-D mesh.
    space : FunctionSpace
        The function space — the carrier's cached ``mesh.bulk_space``
        (CS4b S5). Any
        :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` mint
        is a legitimate source, and so is a space minted with no carrier
        at all (the infinite-medium problem's fields are born on
        ``HomogeneousProblem.space`` since the CS4c coda); this leaf does
        NOT narrow to ``SNMesh``.

    Notes
    -----
    Algebra is the plain :class:`~orpheus.numerics.field.Field` vector space
    (``+``, unary ``−``, scalar ``·`` / ``/``), inherited unchanged via
    :class:`~orpheus.transport.fields._coefficient_role.CoefficientRole` — which
    adds NO gate (``Σ + Σ′`` is legitimate, ``Σ = 0`` is the origin —
    historically the complement of the flux torsor, and since campaign 1 CS3
    the shared shape of every field family). Construction does NOT enforce
    ``Σ ≥ 0``:
    nonnegativity is the physical cone (a tested property — the doctrine the
    CS3 flux ruling generalized; :meth:`~orpheus.numerics.field.Field.cone_violations`
    is its element predicate), and a signed
    difference ``Σ − Σ′`` is a valid coefficient. The mesh-identity check is
    inherited from :class:`~orpheus.transport.fields._bases.BulkField`.
    """

    #: Dimensional identity (View-G): macroscopic cross section ``1/cm``
    #: (:data:`~orpheus.numerics.units.CROSS_SECTION_UNITS`). Metadata, not the
    #: arithmetic gate. See :mod:`orpheus.numerics.units`.
    UNITS: ClassVar[Unit] = CROSS_SECTION_UNITS
