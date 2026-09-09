r"""Scalar flux field on a discretized (group × spatial) phase space.

The L2 typed wrapper for

.. math::

    \phi(\vec r, g) = \int_{4\pi} \psi(\vec r, \hat\Omega, g) \, d\hat\Omega

(continuous form) or, in the discrete-ordinates form consumed by SN
solvers,

.. math::

    \phi_g(\vec r) = \sum_n w_n \, \psi_{n,g}(\vec r)

where :math:`w_n` is the angular quadrature weight on ordinate
:math:`\hat\Omega_n`.

Migration status (Depth B, step D-D)
====================================

This class moved from ``orpheus.sn.scalar_flux`` to
``orpheus.transport.fields.scalar_flux`` and now inherits from
:class:`~orpheus.numerics.field.Field` rather than carrying a
hand-coded dunder skeleton. The migration:

* Drops the six per-class hand-coded dunders (Cardinal Rule 2 —
  single source of truth; the algebra is now inherited from
  :class:`Field` via :func:`dataclasses.replace`).
* Adds the ``space: FunctionSpace`` field (mandatory, per the
  Field ABC contract).
* Keeps ``mesh`` as an additive field — since #267 typed as the
  method-agnostic
  :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` (a scalar
  field reads only material-mesh data — no quadrature, no trace — so
  any carrier in the hierarchy serves it). Runtime-wise the
  ``mesh`` field is duck-typed; TYPE_CHECKING-only imports keep the
  layer contract clean.
* Preserves the strict semantics: arithmetic across two
  :class:`ScalarFlux` instances with DIFFERENT mesh identities is
  forbidden, even when the meshes have matching shapes — enforced by
  the inherited
  :meth:`~orpheus.transport.fields._bases.BulkField._check_partner`
  (the mesh-identity guard lives on the storage base, not this leaf).
* Introduced ``from_mesh`` / ``from_ndarray`` classmethods for
  2-arg construction (retired at CS4b S5 — construction is space-primary
  on the carrier's cached ``mesh.bulk_space``; the dataclass stays
  ``kw_only=True`` per Depth B plan §8 risk #1).

Method-agnostic — and consumed as such since #290
=================================================

A scalar flux distribution on a discretized phase space is a concept
shared by SN, CP, MoC, and diffusion. The SN solver chain was the
first consumer; the diffusion integration (#290) is the anticipated
SECOND consumer — the scalar composite
``FullField(interior=ScalarFlux, boundary=ScalarBoundaryFlux)`` is the
diffusion operator family's carrier. With method #2 live, the
deferred protocol trigger FIRED: the
:class:`~orpheus.transport.method.TransportMethod` Protocol is minted
over both method-meshes (#290 P7b).

Units (B.4 — declared as the ``UNITS`` class constant)
======================================================

:math:`[1/(\mathrm{cm^2 \cdot s})]` — areal angle-integrated flux,
:data:`~orpheus.numerics.units.SCALAR_FLUX_UNITS`. **eV-free**: a stored
flux is always integrated over an energy *bin* (a multigroup group, or a
Monte-Carlo tally bin), so :math:`\phi_g = \int_{E_g}\phi(E)\,dE` is
group-integrated by construction — the ``eV`` cancels. Continuous energy
lives in the cross-section data / collision kernel, not in this field
(so a CE-MC tally and an MG-deterministic solve share this signature).
Under View-G (issues #205 / #207) units are NOT a space property; they
are the role-leaf's ``UNITS`` constant, and the operator-side unit-gain
check gates composition at operator-construction time (#208). See
:mod:`orpheus.numerics.units` for the full convention.

References
----------

* Lewis, E.E. & Miller, W.F. (1993). *Computational Methods of
  Neutron Transport*. American Nuclear Society. §1.2 — scalar /
  angular flux definitions.
* Depth B plan §3.3 (L2 field type spec), §6 step D-D
  (migration plan), §8 risk #1 (kw_only mitigation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from numpy.typing import NDArray

from orpheus.numerics.units import SCALAR_FLUX_UNITS, Unit
from orpheus.transport.fields._bases import ScalarField


__all__ = ["ScalarFlux"]


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class ScalarFlux(ScalarField):
    r"""Scalar flux field :math:`\phi(\vec r, g)`.

    Parameters
    ----------
    values : NDArray
        Field values of shape ``(ng, *spatial)`` — rank-adaptive
        (``(ng, nx)`` on a 1-D mesh, ``(ng, nx, ny)`` on 2-D; the
        principled group-leading layout, Issue #196 PR-INDEX-7).
    space : FunctionSpace
        The function space this flux lives on — the carrier's cached
        ``mesh.bulk_space`` (CS4b S5: construction is space-primary; SN
        callers read it off their :class:`SNMesh`, diffusion / CP off
        the plain :class:`MaterialMesh`).
    Notes
    -----
    Algebra is inherited from :class:`~orpheus.numerics.field.Field`
    (dunders ``+``, ``-``, unary ``-``, scalar ``*``, scalar ``/``,
    plus diagnostics ``linf``, ``l2``, ``inner_product``, ``copy``).
    The inherited
    :meth:`~orpheus.transport.fields._bases.BulkField._check_partner`
    adds the mesh-identity check on top of Field's class-and-space gate.

    Per-group selectors (:meth:`at_group`) return ``np.ndarray``
    VIEWS into ``values`` — downstream callers must not mutate them.
    """

    #: Dimensional identity (View-G, B.4): areal angle-integrated flux
    #: ``1/(cm²·s)`` (eV-free — see module docstring). Metadata, not the
    #: arithmetic gate. See :mod:`orpheus.numerics.units`.
    UNITS: ClassVar[Unit] = SCALAR_FLUX_UNITS

    # ── Selectors ────────────────────────────────────────────────────

    def at_group(self, g: int) -> NDArray:
        r"""Return the per-group slice ``values[g]``, shape ``(nx, ny)``."""
        return self.values[g]
