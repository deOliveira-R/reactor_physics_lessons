r"""Member contracts of :class:`HarmonicMomentFlux` (#399 / CS4b S6.1 — G6.7).

The derived-view members — :meth:`truncate`, :meth:`isotropic_part`,
:meth:`anisotropic_part`, :meth:`l_block` — plus :meth:`scalar_flux`'s
self-derive contract, gated at BOTH widths (``spatial_moments`` 1 and 2).

#399's finding (census 2026-08-21) was that the members were
spatial-moment-blind. Re-measured at HEAD 2026-08-24 before designing
(the plan-authoring shelf-life check): the parts were ALREADY
tail-correct — the S4 space-passing rework repaired them, ungated — and
``truncate`` carried a loud widened ``NotImplementedError`` defer, not
the census's latent broadcast crash. S6.1 lifted the defer with a
space-derived rebuild (swap the spherical-harmonic head factor, keep
every remaining factor verbatim) and this module is the first gate
coverage the members have at ANY width — `[M]` 2026-08-24, zero callers
of all four members in orpheus/ or tests/ before it.

Mutation record (G6.7's row, `[M]` in-process 2026-08-24): restoring the
un-tailed ``new_shape = (L+1, 2L+1, *spatial-only)`` in ``truncate``
reddens the widened truncation row; dropping the tail factor from the
rebuilt space (the pre-S6.1 ``factors[1]``-only rebuild) reddens the
space-content row; zeroing the partition's ℓ=0 copy reddens the
partition law.

vv Mode-8 discipline: assertions are function calls (``np.testing.*`` /
``pytest.raises`` / ``pytest.fail``) — canonical invocation is
``python -O``.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.spatial import LinearDiscontinuous
from tests.sn._test_helpers import placeholder_materials
from orpheus.numerics.spaces.moment_head import MomentHead

pytestmark = [pytest.mark.foundation]

_L, _NG, _NX = 2, 2, 5

#: ⛔ RE-KEYED 2026-09-02 (#429). The whole module rode ONE 1-D fixture and
#: built its values as ``(L+1, 2L+1, …)`` — the RECTANGULAR spherical-harmonic
#: head — on a rule that now binds the FLAT Legendre head ``(L+1,)``. The
#: claims here (truncate, the ℓ-partition, the views) are layout-GENERIC, so
#: the repair is to read the layout off the frame and to run every row on
#: BOTH families: a gate keyed on one certifies the other by accident.
_FAMILIES: tuple[str, ...] = ("flat", "rectangular")


def _sn(family: str = "flat") -> SNMesh:
    """The 1-D mesh, with the quadrature that induces the requested angular head.

    ``"flat"``    — ``gauss_legendre(4)``: its measure lives on
    :math:`S^2/O(2)_x`, so its frame binds the Legendre basis and the head is
    ``(L+1,)``.
    ``"rectangular"`` — ``level_symmetric(4)``: a full-sphere rule, so the head
    is the harmonics' ``(L+1, 2L+1)``.
    """
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, _NX + 1),
        mat_ids=np.zeros(_NX, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    quadrature = (
        Quadrature.gauss_legendre(4) if family == "flat"
        else Quadrature.level_symmetric(4)
    )
    # Linear-Discontinuous: the widened (spatial_moments = 2) rows need a
    # scheme that MINTS a moment axis — since CS4c step 6 item 6.2c-iii the
    # moment product's tail is the scheme's own mass-weighted axis (the
    # angular side's rule since CS4b S4), so a widened request on a DD
    # carrier is refused rather than given a Euclidean tail. Width-1 rows
    # are scheme-blind (no tail).
    return SNMesh(mesh, quadrature, placeholder_materials(ng=_NG), scheme=LinearDiscontinuous())


def _head(sn: SNMesh, L: int = _L) -> MomentHead:
    """The angular head this mesh's frame induces — the single source of the layout (the frame's Parseval-dressed head, item 6.2c-ii)."""
    head = sn.quad.angular_frame(L).basis_space
    assert isinstance(head, MomentHead)
    return head


def _field(sn: SNMesh, spatial_moments: int, seed: int) -> HarmonicMomentFlux:
    tail = (spatial_moments,) if spatial_moments > 1 else ()
    values = np.random.default_rng(seed).standard_normal(
        (*_head(sn).shape, _NG, _NX, *tail)
    )
    return HarmonicMomentFlux.from_mesh_and_L(
        values, sn, _L, spatial_moments=spatial_moments,
    )


# ── truncate — the space-derived rebuild (both widths) ───────────────


class TestTruncate:
    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sm", [1, 2])
    def test_truncate_matches_the_factory_mint_at_both_widths(self, sm, family):
        """The truncated field's space content-equals the carrier's OWN
        mint at (mesh, L_new, spatial_moments) — ``SNMesh.moment_space``
        since CS4c step 6 item 6.2b, the cached object the factory reads
        — the single-source done-when of the space-derived rebuild. The sm=2 row is #399's
        FLIPPED witness (pre-S6.1 it raised the widened defer; pre-S4 it
        was the census's broadcast ValueError). Reddened by dropping the
        tail factor from the rebuild (the pre-S6.1 factors[1]-only
        spelling) or by re-introducing the un-tailed new_shape."""
        sn = _sn(family)
        f = _field(sn, sm, seed=20 + sm)
        L_new = 1
        g = f.truncate(L_new)
        tail = (sm,) if sm > 1 else ()
        expected_shape = (*_head(sn, L_new).shape, _NG, _NX, *tail)
        if g.values.shape != expected_shape:
            pytest.fail(
                f"truncate(sm={sm}) shape {g.values.shape} != "
                f"{expected_shape}"
            )
        factory_space = HarmonicMomentFlux.from_mesh_and_L(
            np.zeros(expected_shape), sn, L_new, spatial_moments=sm,
        ).space
        if g.space != factory_space:
            pytest.fail(
                f"truncated space must content-equal the factory mint at "
                f"(mesh, L_new={L_new}, sm={sm}); got {g.space!r}"
            )
        if g.L != L_new or g.spatial_moments != sm:
            pytest.fail(
                f"truncate must thread L and spatial_moments: got "
                f"L={g.L}, spatial_moments={g.spatial_moments}"
            )
        head_new = _head(sn, L_new)
        for l in range(L_new + 1):
            npt.assert_array_equal(
                g.values[head_new.degree_block(l)],
                f.values[_head(sn).degree_block(l)],
            )

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_truncate_widened_reads_the_width_off_its_space(self, family):
        """The truncated widened field's SPACE carries the moment factor
        (the single source): spatial_moments_per_axis reads 2 off it."""
        g = _field(_sn(family), 2, seed=23).truncate(0)
        if g.spatial_moments_per_axis != 2:
            pytest.fail(
                f"the truncated space must carry the moment factor; "
                f"per-axis width read {g.spatial_moments_per_axis}"
            )

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sm", [1, 2])
    def test_truncate_at_full_order_is_the_identity(self, sm, family):
        """truncate(self.L) reproduces the field bit-exactly, same-space
        content — the no-op leg of the truncation contract."""
        f = _field(_sn(family), sm, seed=25 + sm)
        g = f.truncate(f.L)
        npt.assert_array_equal(g.values, f.values)
        if g.space != f.space:
            pytest.fail("full-order truncation must keep the space content")

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_truncate_range_refusals(self, family):
        f = _field(_sn(family), 1, seed=27)
        with pytest.raises(ValueError, match="L_new=3 >"):
            f.truncate(_L + 1)
        with pytest.raises(ValueError, match="< 0"):
            f.truncate(-1)


# ── the ℓ-partition (both widths) ────────────────────────────────────


class TestPartition:
    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sm", [1, 2])
    def test_parts_partition_bit_exactly(self, sm, family):
        """iso + aniso == self, values bit-exact (the partition law the
        docstring names), at both widths — the sm=2 legs are #399's
        other two flipped witnesses (already repaired by S4, first
        GATED here). Reddened by zeroing the partition's ℓ=0 copy."""
        sn = _sn(family)
        f = _field(sn, sm, seed=30 + sm)
        head = _head(sn)
        iso, aniso = f.isotropic_part(), f.anisotropic_part()
        npt.assert_array_equal(iso.values + aniso.values, f.values)
        # the slots are enumerated through the HEAD, never through a layout
        # literal: ``values[0, 1:]`` on a flat head is group 1's spatial slice
        # (a real array of the wrong quantity, no exception).
        keep = np.zeros(head.shape, dtype=bool)
        keep[head.isotropic_slot] = True
        if np.any(iso.values[~keep]):
            pytest.fail("isotropic_part must zero every non-isotropic slot")
        if np.any(aniso.values[head.isotropic_slot]):
            pytest.fail("anisotropic_part must zero the isotropic slot")

    @pytest.mark.parametrize("family", _FAMILIES)
    @pytest.mark.parametrize("sm", [1, 2])
    def test_parts_share_the_space_instance(self, sm, family):
        """The parts are replace-style derivations: SAME space instance
        (not a re-mint), same L, same width — the space-derived
        contract's cheapest observable."""
        f = _field(_sn(family), sm, seed=33 + sm)
        for part in (f.isotropic_part(), f.anisotropic_part()):
            if part.space is not f.space:
                pytest.fail("a part must carry the parent's space instance")
            if part.L != f.L or part.spatial_moments != sm:
                pytest.fail("a part must thread L and spatial_moments")


# ── l_block (view) + scalar_flux (self-derive contract) ──────────────


class TestViews:
    @pytest.mark.parametrize("family", _FAMILIES)
    def test_l_block_shapes_carry_the_tail(self, family):
        """l_block(l) is a raw view of the head's degree block, tail riding.

        The block's own leading extent is the HEAD's: ``2l+1`` on the
        rectangular harmonics, no leading axis at all on the flat Legendre
        head (one coefficient per degree).
        """
        sn = _sn(family)
        f = _field(sn, 2, seed=36)
        head = _head(sn)
        for l in range(_L + 1):
            blk = f.l_block(l)
            lead = np.zeros(head.shape)[head.degree_block(l)].shape
            if blk.shape != (*lead, _NG, _NX, 2):
                pytest.fail(
                    f"[{family}] l_block({l}) shape {blk.shape} != "
                    f"{(*lead, _NG, _NX, 2)}"
                )
        with pytest.raises(ValueError, match="out of range"):
            f.l_block(_L + 1)

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_scalar_flux_width1_self_derives_the_bulk_factor(self, family):
        """Width-1 scalar_flux derives its target from the product's
        cell-group factor — the carrier's shared bulk mint."""
        sn = _sn(family)
        f = _field(sn, 1, seed=37)
        s = f.scalar_flux()
        npt.assert_array_equal(s.values, f.values[_head(sn).isotropic_slot])
        if s.space != f.space.factors[1]:  # type: ignore[union-attr]
            pytest.fail("scalar target must be the product's bulk factor")

    @pytest.mark.parametrize("family", _FAMILIES)
    def test_scalar_flux_widened_self_derives_the_widened_bulk(self, family):
        """The widened self-derive is HONEST since CS4c step 6 item
        6.2c-iii: the moment product's cell-group factor is the carrier's
        widened bulk carrying the scheme's mass-weighted moment axis (the
        S4 refusal's reason — a Euclidean ``SpatialMomentSpace`` tail
        without that axis — is gone), so the target is the product's own
        factor, equal to the widened angular space's scalar marginal a
        caller used to have to pass."""
        sn = _sn(family)
        f = _field(sn, 2, seed=38)
        s = f.scalar_flux()
        npt.assert_array_equal(s.values, f.values[_head(sn).isotropic_slot])
        if s.space != f.space.factors[1]:  # type: ignore[union-attr]
            pytest.fail("scalar target must be the product's widened bulk factor")
        from orpheus.numerics.moment_layout import SPATIAL_MOMENT_AXIS_LABEL
        assert s.space.axes is not None
        tail = [ax for ax in s.space.axes if ax.label == SPATIAL_MOMENT_AXIS_LABEL]
        if len(tail) != 1 or tail[0] != sn.scheme.moment_axis(sn.axes):
            pytest.fail("the widened scalar target must carry the scheme's own moment axis")
        # and it equals the explicit-target path
        explicit = f.scalar_flux(space=s.space)
        if explicit.space != s.space:
            pytest.fail("explicit and self-derived targets must be one space")
