"""Foundation tests for the Issue #197 PR-TYPED-4 HarmonicMomentFlux.

Pins the structural contract of
:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux` — shape
validation, named slicing / decomposition primitives, dunder algebra,
truncation, ``scalar_flux()`` agreement with the bare-ordinate
reduction, and the SN-side typed ``R \\cdot \\Lambda \\cdot M``
round-trip.

These are foundation tests — they verify software invariants
(constructor shape check, frozen-ness, dunder algebra) and the
algebraic identities the type carries; they do NOT make physics
claims, so they carry ``@pytest.mark.foundation`` per the V&V harness
convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.geometry import BC, CoordSystem, Mesh1D, Mesh2D
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.numerics.quadrature import Quadrature
from orpheus.transport.fields.scalar_flux import ScalarFlux

from tests.sn._test_helpers import placeholder_materials
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.numerics.spaces.moment_head import MomentHead
from orpheus.transport.material_field import TransferMaterialField

pytestmark = pytest.mark.foundation


# ── Fixtures ─────────────────────────────────────────────────────────


def _slab_mesh(nx: int = 4, ng: int = 2) -> SNMesh:
    """Build a small slab :class:`SNMesh` for unit testing."""
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, nx + 1),
        mat_ids=np.zeros(nx, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"),
        bc_right=BC("vacuum"),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=4)
    return SNMesh(mesh, quad, placeholder_materials(ng=ng))


def _stretched_mesh(nx: int = 4, ng: int = 2) -> SNMesh:
    """Same shape as ``_slab_mesh``, doubled width — the cell VOLUMES differ,
    so the carrier mints an UNEQUAL space (the F2 content discriminator)."""
    mesh = Mesh1D(
        edges=np.linspace(0.0, 2.0, nx + 1),
        mat_ids=np.zeros(nx, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"),
        bc_right=BC("vacuum"),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=4)
    return SNMesh(mesh, quad, placeholder_materials(ng=ng))


def _head_shape(mesh: SNMesh, L: int) -> tuple[int, ...]:
    r"""The angular HEAD's own axes for this mesh at order :math:`L`.

    ⭐ **Read off the FRAME, never spelled** (#429 tracker 2.5). Two families
    ship and they have different ranks: the real spherical harmonics'
    rectangular ``(L+1, 2L+1)`` table on a sphere rule, and the Legendre
    family's FLAT ``(L+1,)`` on a 1-D rule's :math:`S^2/SO(2)_a` — one
    coefficient per degree, because the trivial isotypic component of
    :math:`SO(2)` is one-dimensional in every degree.

    ⛔ Until the ERR-080 repair every fixture in this file hard-coded
    ``(L+1, 2L+1)``, which read the FIRST family's layout as if it were the
    contract. On the slab that is not merely a shape mismatch: on a flat head
    ``values[0, 0]`` is **group 0's spatial slice** and raises nothing, so
    ``scalar_flux`` / ``isotropic_part`` / ``anisotropic_part`` / ``l_block``
    would have returned a wrong array SILENTLY.

    ⚠ Both layouts stay exercised in this module: ``_slab_mesh`` now carries
    the flat head, ``_2d_mesh`` (level-symmetric) the rectangular one.
    """
    return _head_of(mesh, L).shape


def _head_of(mesh: SNMesh, L: int) -> MomentHead:
    """The angular head OBJECT — the surface that says where the isotropic slot and each degree block live."""
    head = mesh.quad.angular_frame(L).basis.space
    assert isinstance(head, MomentHead)
    return head


def _2d_mesh(nx: int = 3, ny: int = 3, ng: int = 1) -> SNMesh:
    """Build a small 2-D Cartesian :class:`SNMesh`."""
    mesh = Mesh2D(
        edges_x=np.linspace(0, 1, nx + 1),
        edges_y=np.linspace(0, 1, ny + 1),
        mat_map=np.zeros((nx, ny), dtype=int),
    )
    quad = Quadrature.level_symmetric(sn_order=4)
    return SNMesh(mesh, quad, placeholder_materials(ng=ng))


# ════════════════════════════════════════════════════════════════════
# Construction + shape validation
# ════════════════════════════════════════════════════════════════════


class TestHarmonicMomentFluxConstruction:
    def test_construct_from_factory(self) -> None:
        m = _slab_mesh()
        L = 2
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L)
        # (L+1, 2L+1, ng, *spatial) = (3, 5, 2, 4) on the 1-D slab
        assert phi.values.shape == (*_head_shape(m, L), m.ng, *m.spatial_shape)
        assert np.all(phi.values == 0.0)
        assert isinstance(phi, HarmonicMomentFlux)
        assert phi.L == L

    def test_construct_explicit(self) -> None:
        m = _2d_mesh(ng=1)
        L = 1
        # shape (2, 3, 1, 3, 3)
        vals = np.arange(2 * 3 * 1 * 3 * 3, dtype=float).reshape(
            (2, 3, 1, 3, 3),
        )
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        assert phi.L == 1
        np.testing.assert_array_equal(phi.values, vals)

    def test_shape_validation_wrong_L_block_size(self) -> None:
        m = _slab_mesh()
        L = 2
        # Wrong shape: (L+1, 2L, ...) instead of (L+1, 2L+1, ...).
        with pytest.raises(ValueError, match="HarmonicMomentFlux.*does not match"):
            HarmonicMomentFlux.from_mesh_and_L(
                np.zeros((L + 1, 2 * L, m.ng, *m.spatial_shape)), m, L,
            )

    def test_shape_validation_wrong_mesh_dims(self) -> None:
        m = _slab_mesh()
        L = 1
        with pytest.raises(ValueError, match="HarmonicMomentFlux.*does not match"):
            HarmonicMomentFlux.from_mesh_and_L(
                np.zeros((L + 1, 2 * L + 1, m.ng + 1, *m.spatial_shape)), m, L,
            )

    def test_metadata_read_throughs(self) -> None:
        m = _slab_mesh()
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=0)
        assert phi.ng == m.ng
        # C5.2 (#225): nx/ny field read-throughs retired; spatial shape
        # reads are rank-generic (the space's shape contract).
        # ⛔ RE-KEYED 2026-09-02 (#429): ``shape[3:]`` hard-coded a TWO-axis
        # head plus the group axis. The head's rank is the head's to say
        # (``len(phi.head.shape)``) — 2 for the rectangular harmonics, 1 for
        # the flat Legendre family a 1-D rule now binds.
        assert phi.values.shape[len(phi.head.shape) + 1:] == m.spatial_shape
        assert phi.head.shape == _head_shape(m, 0)

    def test_space_is_tensor_product_of_the_frames_head_and_the_cell_group(
        self,
    ) -> None:
        r"""D-E invariant: the field's ``space`` is ``<angular head> ⊗ cells``.

        ⛔ **RE-KEYED 2026-09-02 (#429).** The head used to be asserted as a
        :class:`SphericalHarmonicSpace` unconditionally. **Two** families now
        ship, and which one a field carries is READ off the frame the
        quadrature bound: the rectangular harmonics on a sphere rule, the
        FLAT :class:`LegendreSpace` on a 1-D rule's :math:`S^2/O(2)_x`. Both
        rows are asserted below, because a gate keyed on one family would
        certify the other by accident.

        The moment-axis structure stays type-visible through the composition
        tree (Issue #207): ``phi.space.factors[0].L == phi.L`` on both.
        """
        from orpheus.numerics.space import FunctionSpace, TensorProductSpace
        from orpheus.numerics.spaces.legendre_space import LegendreSpace
        from orpheus.numerics.spaces.spherical_harmonic_space import (
            SphericalHarmonicSpace,
        )

        L = 2
        for mesh, family, head_shape in (
            (_slab_mesh(), LegendreSpace, (L + 1,)),
            (_2d_mesh(ng=1), SphericalHarmonicSpace, (L + 1, 2 * L + 1)),
        ):
            phi = HarmonicMomentFlux.zeros_for_mesh_and_L(mesh, L=L)
            assert isinstance(phi.space, TensorProductSpace)
            assert len(phi.space.factors) == 2

            head = phi.space.factors[0]
            assert isinstance(head, family), (
                f"{type(mesh.quad._harmonic_basis(L)).__name__} should induce "
                f"a {family.__name__} head; got {type(head).__name__}"
            )
            # …and whichever family it is, it satisfies the ONE surface every
            # consumer reads (the structural Protocol, not the class).
            assert isinstance(head, MomentHead)
            assert head.L == L
            assert head.shape == head_shape
            assert head is phi.head

            cells = phi.space.factors[1]
            assert isinstance(cells, FunctionSpace)
            assert cells.shape == (mesh.ng, *mesh.spatial_shape)
            assert phi.space.shape == phi.values.shape


# ════════════════════════════════════════════════════════════════════
# Slicing / decomposition primitives (Pattern 3 — named intermediates)
# ════════════════════════════════════════════════════════════════════


class TestHarmonicMomentFluxSlicing:
    def test_l_block_returns_view_with_right_shape(self) -> None:
        m = _slab_mesh()
        L = 2
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L)
        head = phi.head
        for l in range(L + 1):
            block = phi.l_block(l)
            # the block's own leading extent is the HEAD's degree block —
            # ``2l+1`` on the rectangular harmonics, a single coefficient
            # (no leading axis at all) on the flat Legendre head.
            expected_lead = np.zeros(head.shape)[head.degree_block(l)].shape
            assert block.shape == (*expected_lead, m.ng, *m.spatial_shape)

    def test_l_block_values_match_underlying_slice(self) -> None:
        m = _slab_mesh()
        L = 2
        rng = np.random.default_rng(seed=0)
        vals = rng.standard_normal(
            (*_head_shape(m, L), m.ng, *m.spatial_shape),
        )
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        head = phi.head
        for l in range(L + 1):
            np.testing.assert_array_equal(
                phi.l_block(l), vals[head.degree_block(l)],
            )

    def test_l_block_out_of_range_raises(self) -> None:
        m = _slab_mesh()
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        with pytest.raises(ValueError):
            phi.l_block(2)
        with pytest.raises(ValueError):
            phi.l_block(-1)

    def test_isotropic_part_zeros_l_ge_1(self) -> None:
        m = _slab_mesh()
        L = 2
        rng = np.random.default_rng(seed=1)
        vals = rng.standard_normal(
            (*_head_shape(m, L), m.ng, *m.spatial_shape),
        )
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        iso = phi.isotropic_part()
        head = phi.head
        assert isinstance(iso, HarmonicMomentFlux)
        assert iso.L == L
        # the isotropic slot is preserved — its INDEX is the head's to say
        # (``(0, 0)`` rectangular, ``(0,)`` flat); ``values[0, 0]`` on a flat
        # head is group 0's spatial slice and raises nothing, which is the
        # silent-wrongness this row now cannot have.
        np.testing.assert_array_equal(
            iso.values[head.isotropic_slot], vals[head.isotropic_slot]
        )
        # every OTHER slot of the head is zeroed — enumerated through the
        # head's own degree blocks rather than by a layout literal.
        for l in range(1, L + 1):
            np.testing.assert_array_equal(iso.values[head.degree_block(l)], 0.0)
        # …and within degree 0, anything that is not the isotropic slot
        # (the rectangular head's m >= 1 padding; empty on a flat head).
        degree_zero = np.zeros(head.shape)
        degree_zero[head.degree_block(0)] = 1.0
        degree_zero[head.isotropic_slot] = 0.0
        np.testing.assert_array_equal(iso.values[degree_zero == 1.0], 0.0)

    def test_anisotropic_part_zeros_l_eq_0(self) -> None:
        m = _slab_mesh()
        L = 2
        rng = np.random.default_rng(seed=2)
        vals = rng.standard_normal(
            (*_head_shape(m, L), m.ng, *m.spatial_shape),
        )
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        aniso = phi.anisotropic_part()
        head = phi.head
        assert isinstance(aniso, HarmonicMomentFlux)
        # the isotropic slot is zeroed…
        np.testing.assert_array_equal(aniso.values[head.isotropic_slot], 0.0)
        # …and every higher degree is preserved.
        for l in range(1, L + 1):
            np.testing.assert_array_equal(
                aniso.values[head.degree_block(l)], vals[head.degree_block(l)]
            )

    def test_iso_plus_aniso_recovers_self(self) -> None:
        r"""The ℓ=0 / ℓ≥1 decomposition is complete:
        ``isotropic_part().values + anisotropic_part().values == self.values``
        bit-exactly (the contract stated in ``anisotropic_part``'s
        docstring).

        Both parts are :class:`HarmonicMomentFlux` states in V, so since
        the CS3 cone carve (2026-08-19) the recombination ``iso + aniso``
        is the DIRECT spelling — the ℓ-disjoint reconstruction is exactly
        the vector sum the retired affine gate used to forbid, and this
        row now asserts it typed as well as at the array level.
        """
        m = _slab_mesh()
        L = 2
        rng = np.random.default_rng(seed=3)
        vals = rng.standard_normal(
            (*_head_shape(m, L), m.ng, *m.spatial_shape),
        )
        # Zero everything in degree 0 that is not the isotropic slot (the
        # rectangular head's m >= 1 padding; on a FLAT head the degree-0
        # block IS the isotropic slot, so this zeroes nothing) so that
        # iso + aniso reproduces ``vals`` exactly — the padding is not part
        # of the legitimate moment-space content.
        _head = _head_of(m, L)
        _degree_zero = np.zeros(_head.shape)
        _degree_zero[_head.degree_block(0)] = 1.0
        _degree_zero[_head.isotropic_slot] = 0.0
        vals[_degree_zero == 1.0] = 0.0
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        # The typed recombination — the capability the CS3 carve unlocked.
        recombined = phi.isotropic_part() + phi.anisotropic_part()
        if type(recombined) is not HarmonicMomentFlux:
            raise AssertionError("iso + aniso left the leaf type")
        np.testing.assert_array_equal(recombined.values, phi.values)
        # Complete decomposition, verified at the array level (the iso /
        # aniso parts are disjoint slices, so their array sum is exactly
        # the original moment content).
        recombined_values = (
            phi.isotropic_part().values + phi.anisotropic_part().values
        )
        np.testing.assert_array_equal(recombined_values, vals)


# ════════════════════════════════════════════════════════════════════
# scalar_flux() — extracts ℓ=0, m=0 moment as a ScalarFlux
# ════════════════════════════════════════════════════════════════════


class TestHarmonicMomentFluxScalarFlux:
    def test_scalar_flux_returns_ScalarFlux_with_isotropic_slot(self) -> None:
        m = _slab_mesh()
        L = 1
        rng = np.random.default_rng(seed=4)
        vals = rng.standard_normal(
            (*_head_shape(m, L), m.ng, *m.spatial_shape),
        )
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        sf = phi.scalar_flux()
        assert isinstance(sf, ScalarFlux)
        assert sf.values.shape == (m.ng, *m.spatial_shape)
        np.testing.assert_array_equal(sf.values, vals[phi.head.isotropic_slot])

    def test_scalar_flux_agrees_with_integrate_angular(self) -> None:
        """``M(\\psi).scalar_flux() == \\psi.integrate_angular()``.

        Under the no-prefactor SH convention (``Y_0^0 = 1``), the
        isotropic moment :math:`\\sum_n w_n Y_0^0 \\psi_n` reduces to
        :math:`\\sum_n w_n \\psi_n` — the bare angular reduction.  This
        identity makes the moment-space and ordinate-space scalar
        fluxes algebraically equivalent.
        """
        m = _slab_mesh()
        L = 1
        rng = np.random.default_rng(seed=5)
        N = m.quad.N
        psi_values = rng.standard_normal((N, m.ng, *m.spatial_shape))
        psi = AngularFlux(values=psi_values, space=m.angular_bulk_space)

        # Direct angular reduction.
        sf_direct = psi.integrate_angular()

        # Via moment projection (frame analysis face) + scalar_flux extraction.
        moments_values = m.quad.angular_frame(L).analysis.apply(psi.values)
        moments = HarmonicMomentFlux.from_mesh_and_L(moments_values, m, L)
        sf_via_moments = moments.scalar_flux()

        np.testing.assert_allclose(
            sf_via_moments.values, sf_direct.values, rtol=1e-13,
        )


# ════════════════════════════════════════════════════════════════════
# truncate
# ════════════════════════════════════════════════════════════════════


class TestHarmonicMomentFluxTruncate:
    def test_truncate_preserves_lower_blocks(self) -> None:
        m = _slab_mesh()
        L = 3
        rng = np.random.default_rng(seed=6)
        vals = rng.standard_normal(
            (*_head_shape(m, L), m.ng, *m.spatial_shape),
        )
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, m, L)
        for L_new in range(L + 1):
            trunc = phi.truncate(L_new)
            assert trunc.L == L_new
            assert trunc.values.shape == (
                *_head_shape(m, L_new), m.ng, *m.spatial_shape,
            )
            # ⭐ and the truncated head stays in the SAME family — a flat
            # head truncates to a flat head, a rectangular one to a
            # rectangular one (``MomentHead.truncated`` is the family's own
            # verb, not a shape literal).
            assert trunc.head == phi.head.truncated(L_new)
            # For each ℓ ≤ L_new, the (ℓ, m) entries inside |m|≤ℓ
            # match the source.
            for l in range(L_new + 1):
                np.testing.assert_array_equal(
                    trunc.l_block(l), phi.l_block(l),
                )

    def test_truncate_rejects_L_new_greater_than_L(self) -> None:
        m = _slab_mesh()
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        with pytest.raises(ValueError, match="truncate"):
            phi.truncate(2)

    def test_truncate_rejects_negative(self) -> None:
        m = _slab_mesh()
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        with pytest.raises(ValueError, match="truncate"):
            phi.truncate(-1)


# ════════════════════════════════════════════════════════════════════
# Dunder algebra (Pattern 1 — match the math)
# ════════════════════════════════════════════════════════════════════


class TestHarmonicMomentFluxAlgebra:
    def test_add_within_type_and_update_round_trip(self) -> None:
        r"""Flux lives in V (campaign 1 CS3): ``moment + moment`` is the
        plain vector sum in the same leaf type, and the update step
        ``ψ + (ψ' − ψ) ≈ ψ'`` is ordinary arithmetic — mirrors
        ``test_angular_flux.py::TestAlgebra::
        test_flux_add_flux_legal_and_update_round_trip``. (Until
        2026-08-19 the #208 affine gate raised here.)
        """
        m = _slab_mesh()
        L = 1
        shape = (*_head_shape(m, L), m.ng, *m.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m, L)
        b = HarmonicMomentFlux.from_mesh_and_L(2.0 * np.ones(shape), m, L)
        s = a + b
        if type(s) is not HarmonicMomentFlux:
            raise AssertionError("moment + moment left the leaf type")
        np.testing.assert_array_equal(s.values, 3.0 * np.ones(shape))
        out = a + (b - a)
        if type(out) is not HarmonicMomentFlux:
            raise AssertionError("the update step left the leaf type")
        np.testing.assert_array_equal(out.values, b.values)
        # Frozen — originals unchanged.
        np.testing.assert_array_equal(a.values, np.ones(shape))

    def test_sub_within_type(self) -> None:
        r"""``moment − moment`` returns the SAME leaf type carrying the
        signed difference (flux lives in V — campaign 1 CS3; until
        2026-08-19 this minted a ``MomentDisplacement``).
        """
        m = _slab_mesh()
        L = 1
        shape = (*_head_shape(m, L), m.ng, *m.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(3.0 * np.ones(shape), m, L)
        b = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m, L)
        c = a - b
        if type(c) is not HarmonicMomentFlux:
            raise AssertionError("moment − moment left the leaf type")
        np.testing.assert_array_equal(c.values, a.values - b.values)

    def test_scalar_mul_left_and_right(self) -> None:
        m = _slab_mesh()
        L = 1
        shape = (*_head_shape(m, L), m.ng, *m.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m, L)
        np.testing.assert_array_equal(
            (3.0 * a).values, (a * 3.0).values,
        )
        np.testing.assert_array_equal(
            (3.0 * a).values, 3.0 * np.ones(shape),
        )

    def test_div(self) -> None:
        m = _slab_mesh()
        L = 0
        shape = (*_head_shape(m, L), m.ng, *m.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(2.0 * np.ones(shape), m, L)
        np.testing.assert_array_equal(
            (a / 2.0).values, np.ones(shape),
        )

    def test_neg(self) -> None:
        m = _slab_mesh()
        L = 1
        shape = (*_head_shape(m, L), m.ng, *m.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m, L)
        np.testing.assert_array_equal((-a).values, -np.ones(shape))

    def test_partner_must_be_same_type(self) -> None:
        m = _slab_mesh()
        L = 1
        shape = (*_head_shape(m, L), m.ng, *m.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m, L)
        with pytest.raises(TypeError):
            a + 5  # type: ignore[operator]  # not a HarmonicMomentFlux

    def test_partner_must_share_space_content(self) -> None:
        # CS4b S3 (F2): twin carriers mint EQUAL moment spaces and mix; a
        # volumes-differing carrier mints an UNEQUAL cell-group factor and
        # both binary ops refuse on space content.
        m1 = _slab_mesh()
        m2 = _stretched_mesh()
        L = 1
        shape = (*_head_shape(m1, L), m1.ng, *m1.spatial_shape)
        a = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m1, L)
        twin = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), _slab_mesh(), L)
        _ = a + twin  # twin content — legal since the F2 re-key
        b = HarmonicMomentFlux.from_mesh_and_L(np.ones(shape), m2, L)
        with pytest.raises(ValueError, match="equal space"):
            a - b
        with pytest.raises(ValueError, match="equal space"):
            a + b

    def test_partner_must_share_L(self) -> None:
        m = _slab_mesh()
        a = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        b = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=2)
        # Post-D-E: L mismatch surfaces as a space-equality error,
        # because different L values produce different
        # SphericalHarmonicSpace shapes, which propagates to different
        # TensorProductSpace identities. Field._check_partner's space
        # check fires before the explicit L check in
        # HarmonicMomentFlux._check_partner. Both gate the same
        # invariant; the space-level message is more general.
        #
        # Exercised through ``__sub__`` (either binary op reaches
        # _check_partner since the CS3 cone carve).
        with pytest.raises(ValueError, match="equal space"):
            a - b


# ════════════════════════════════════════════════════════════════════
# R · Λ · M · ψ round-trip through the typed pipeline
# ════════════════════════════════════════════════════════════════════


class TestRLambdaMRoundTrip:
    def test_aniso_part_zero_under_isotropic_psi(self) -> None:
        r"""For isotropic :math:`\psi_n = c \forall n`, the
        Pℓ≥1 reconstruction :math:`R \cdot \Lambda \cdot M \cdot \psi`
        is identically zero (the anisotropic moments vanish by
        construction).  This test pins the typed-pipeline output as
        a :class:`AngularSourceSink` matching that algebraic claim.
        """
        from orpheus.derivations.common.xs_library import get_mixture
        from orpheus.transport.operators.scattering import ScatteringOperator
        from orpheus.sn.solver import SNSolver
        from orpheus.transport.source_sinks import AngularSourceSink
        mix = get_mixture("A", "2g")
        if len(mix.SigS) < 2:
            pytest.skip("No P1 data in test mixture")

        nx, ny = 2, 2
        mesh = Mesh2D(
            edges_x=np.linspace(0, 1, nx + 1),
            edges_y=np.linspace(0, 1, ny + 1),
            mat_map=np.zeros((nx, ny), dtype=int),
        )
        quad = Quadrature.level_symmetric(sn_order=4)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, scattering_order=1)
        op = solver.scattering_op

        # Build a typed AngularFlux with isotropic content.
        N = quad.N
        psi_values = np.ones((N, mix.ng, nx, ny))
        psi = AngularFlux(values=psi_values, space=sn_mesh.angular_bulk_space)

        # Typed pipeline output.
        out = op._redistribute_ordinates(psi)
        # Must be AngularSourceSink (not bare ndarray) under typed-in.
        assert isinstance(out, AngularSourceSink)
        assert out.values.shape == (N, mix.ng, nx, ny)
        np.testing.assert_allclose(out.values, 0.0, atol=1e-12)

    def test_lambda_apply_flux_in_source_out(self) -> None:
        """``LegendreMomentTransfer.apply(HarmonicMomentFlux)`` is the
        **role-changing** edge of the carrier grid: a flux moment scatters
        into the in-scatter SOURCE moment it emits, so the typed return is a
        :class:`HarmonicMomentSourceSink` (NOT a flux) with matching mesh + L.
        """
        from orpheus.derivations.common.xs_library import get_mixture
        from orpheus.transport.operators.transfer import LegendreMomentTransfer
        from orpheus.sn.solver import SNSolver
        from orpheus.transport.source_sinks import HarmonicMomentSourceSink
        mix = get_mixture("A", "2g")
        if len(mix.SigS) < 2:
            pytest.skip("No P1 data in test mixture")

        nx, ny = 2, 2
        mesh = Mesh2D(
            edges_x=np.linspace(0, 1, nx + 1),
            edges_y=np.linspace(0, 1, ny + 1),
            mat_map=np.zeros((nx, ny), dtype=int),
        )
        quad = Quadrature.level_symmetric(sn_order=4)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, scattering_order=1)
        op = solver.scattering_op

        L = 1
        rng = np.random.default_rng(seed=7)
        moments_values = rng.standard_normal(
            (*_head_shape(sn_mesh, L), mix.ng, nx, ny),
        )
        moments = HarmonicMomentFlux.from_mesh_and_L(moments_values, sn_mesh, L)

        Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver.mat_xs), SphericalHarmonicBasis(L=L), skip_l0=True,
        )
        out = Lam.apply(moments)
        # flux moment IN → source moment OUT (the explicit role change).
        assert isinstance(out, HarmonicMomentSourceSink)
        assert not isinstance(out, HarmonicMomentFlux)
        assert out.space == moments.space  # same space, new role (S4)
        assert out.L == L
        assert out.values.shape == moments.values.shape
        # the numbers are the bare-ndarray Λ kernel (typed arm only re-wraps).
        np.testing.assert_array_equal(
            out.values, Lam.apply(moments_values),
        )

    def test_lambda_bare_in_bare_out_legacy_path(self) -> None:
        """Bare-ndarray path is preserved for legacy probe tests."""
        from orpheus.derivations.common.xs_library import get_mixture
        from orpheus.transport.operators.transfer import LegendreMomentTransfer
        from orpheus.sn.solver import SNSolver
        mix = get_mixture("A", "2g")
        if len(mix.SigS) < 2:
            pytest.skip("No P1 data in test mixture")

        nx, ny = 2, 2
        mesh = Mesh2D(
            edges_x=np.linspace(0, 1, nx + 1),
            edges_y=np.linspace(0, 1, ny + 1),
            mat_map=np.zeros((nx, ny), dtype=int),
        )
        quad = Quadrature.level_symmetric(sn_order=4)
        sn_mesh = SNMesh(mesh, quad, {0: mix})
        solver = SNSolver(sn_mesh, scattering_order=1)
        op = solver.scattering_op

        L = 1
        moments_values = np.zeros(
            (*_head_shape(sn_mesh, L), mix.ng, nx, ny),
        )
        Lam = LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(solver.mat_xs), SphericalHarmonicBasis(L=L), skip_l0=True,
        )
        out = Lam.apply(moments_values)
        assert isinstance(out, np.ndarray)
        assert out.shape == moments_values.shape


# ════════════════════════════════════════════════════════════════════
# Factory: HarmonicMomentFlux.zeros_for_mesh_and_L
# ════════════════════════════════════════════════════════════════════


class TestZerosForMeshAndL:
    def test_factory_for_L_zero(self) -> None:
        m = _slab_mesh()
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=0)
        assert phi.L == 0
        # ⛔ RE-KEYED 2026-09-02 (#429): ``(1, 1, ...)`` is the RECTANGULAR
        # head at L = 0. The slab's Legendre head is FLAT, ``(1,)``.
        # ⭐ And L = 0 is not a corner case here: ``fission.py`` mints
        # ``for_space(interior, 0)`` on EVERY solve, and both transfer
        # bindings do on every P0 solve — this is the shape the whole tree
        # sees most.
        assert phi.values.shape == (*_head_shape(m, 0), m.ng, *m.spatial_shape)
        assert phi.values.shape == (1, m.ng, *m.spatial_shape)

        # the rectangular twin, so both layouts stay pinned at L = 0
        sphere = _2d_mesh(ng=1)
        rect = HarmonicMomentFlux.zeros_for_mesh_and_L(sphere, L=0)
        assert rect.values.shape == (1, 1, sphere.ng, *sphere.spatial_shape)

    def test_factory_returns_owned_ndarray(self) -> None:
        m = _slab_mesh()
        phi1 = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        phi2 = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        # Independent allocations.
        assert phi1.values is not phi2.values
        phi1.values.flags.writeable
        # One space identity (the carrier's mint, per (name, shape)).
        assert phi1.space == phi2.space

    def test_copy_creates_independent(self) -> None:
        m = _slab_mesh()
        phi = HarmonicMomentFlux.zeros_for_mesh_and_L(m, L=1)
        phi_copy = phi.copy()
        assert phi.values is not phi_copy.values
        np.testing.assert_array_equal(phi.values, phi_copy.values)


# ════════════════════════════════════════════════════════════════════
# B13 (#429) — a RANK-1 angular head, dispatched rather than assumed
# ════════════════════════════════════════════════════════════════════


class TestRankOneAngularHead:
    r"""Every view verb reads the head's LAYOUT; none assumes two axes.

    ⛔ **The failure this class exists to prevent is SILENT.** Before #429 the
    views indexed ``values[0, 0]`` (``scalar_flux``, ``isotropic_part``),
    sliced ``values[l, :2l+1]`` (``l_block``, ``anisotropic_part``) and rebuilt
    ``(L_new+1, 2L_new+1, *shape[2:])`` (``truncate``). On a FLAT head —
    ``(L+1, ng, nx)`` — ``values[0, 0]`` is **group 0's spatial slice**: a
    real array of the wrong quantity, no exception, no shape error. So the
    rows below assert the correct answers AND demonstrate that the wrong
    layout reads would have returned something rather than raising.

    ⭐ **The** :math:`L = 0` **chain is where this lands hardest.**
    ``fission.py`` mints ``for_space(interior, 0)`` on EVERY solve, and both
    transfer bindings (S and N₂ₙ, since #426 step 2 at the same order) do on
    every P0 solve — so a flat head at :math:`L = 0` is the single
    most-travelled shape in the tree, not a corner case.
    """

    def test_the_slab_field_carries_a_rank_one_head_and_every_view_is_correct(
        self,
    ) -> None:
        mesh = _slab_mesh()
        L = 1
        head = _head_of(mesh, L)
        assert len(head.shape) == 1, "the slab's Legendre head is FLAT"
        assert head.isotropic_slot == (0,)
        assert head.degree_block(1) == (1,)

        rng = np.random.default_rng(20260902)
        vals = rng.standard_normal((*head.shape, mesh.ng, *mesh.spatial_shape))
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, mesh, L)

        # equality, not identity: the head is re-derived per space mint and
        # LegendreSpace compares by (name, shape) like every FunctionSpace.
        assert phi.head == head
        assert phi.ng == mesh.ng
        np.testing.assert_array_equal(phi.scalar_flux().values, vals[0])
        np.testing.assert_array_equal(phi.isotropic_part().values[0], vals[0])
        np.testing.assert_array_equal(phi.isotropic_part().values[1], 0.0)
        np.testing.assert_array_equal(phi.anisotropic_part().values[0], 0.0)
        np.testing.assert_array_equal(phi.anisotropic_part().values[1], vals[1])
        np.testing.assert_array_equal(phi.l_block(0), vals[0])
        np.testing.assert_array_equal(phi.l_block(1), vals[1])

        truncated = phi.truncate(0)
        assert truncated.values.shape == (1, mesh.ng, *mesh.spatial_shape)
        np.testing.assert_array_equal(truncated.values[0], vals[0])

    def test_the_two_axis_read_would_have_been_SILENT_on_a_flat_head(self) -> None:
        r"""The demonstration, not an assertion about production: ``values[0, 0]`` RETURNS on a flat head.

        This is why the repair could not have been found by running the
        suite: a wrong-layout read raises nothing, it just answers the wrong
        question. ``[M]`` on the slab at :math:`L = 1`, ``values[0, 0]`` is
        ``values[0][0]`` — the ℓ=0 moment of GROUP 0 — with shape
        ``(nx,)`` where the scalar flux has ``(ng, nx)``.
        """
        mesh = _slab_mesh()
        L = 1
        head = _head_of(mesh, L)
        rng = np.random.default_rng(20260902)
        vals = rng.standard_normal((*head.shape, mesh.ng, *mesh.spatial_shape))

        two_axis_read = vals[0, 0]                 # what the old code did
        honest_read = vals[head.isotropic_slot]    # what the head says

        assert two_axis_read.shape == mesh.spatial_shape
        assert honest_read.shape == (mesh.ng, *mesh.spatial_shape)
        assert two_axis_read.shape != honest_read.shape
        np.testing.assert_array_equal(two_axis_read, honest_read[0])

    def test_the_rank_two_head_is_unchanged(self) -> None:
        """The CONTROL: the rectangular family still answers exactly as before."""
        mesh = _2d_mesh(ng=1)
        L = 1
        head = _head_of(mesh, L)
        assert len(head.shape) == 2
        assert head.isotropic_slot == (0, 0)
        assert head.degree_block(1) == (1, slice(0, 3))

        rng = np.random.default_rng(20260902)
        vals = rng.standard_normal((*head.shape, mesh.ng, *mesh.spatial_shape))
        phi = HarmonicMomentFlux.from_mesh_and_L(vals, mesh, L)

        np.testing.assert_array_equal(phi.scalar_flux().values, vals[0, 0])
        np.testing.assert_array_equal(phi.l_block(1), vals[1, :3])
        assert phi.truncate(0).values.shape == (
            1, 1, mesh.ng, *mesh.spatial_shape
        )
