r"""Foundation suite for the optional spatial-moment field-space factor (#240 D5b-S3-A0).

Since CS4b S5 the widening is a SPACE selection, not a factory parameter:
the un-windowed carriers construct on the carrier's cached mints — the
width-1 ``angular_bulk_space`` / ``bulk_space``, or the scheme-widened
``angular_trial_space`` (the ``spatial_moments`` int retired with the sugar
tier; only the S6-pending ``HarmonicMomentFlux`` keyed factory still
threads it). The "append iff > 1" gate lives in the trial mint: slopeless
schemes collapse it to the width-1 instance, byte-identical.

These are ``foundation`` (software-invariant) tests. The load-bearing gate
is **scheme-blindness of the width-1 mints** (DD, Step, AND LD read the
same bulk spaces — construct-general / select-narrow: widening is the
CALLER's property choice, never a scheme auto-read). The widened path is
the CAPABILITY, exercised on the trial mint / the composed scalar space.

Mode-8 / L26: every assertion is a FUNCTION CALL (``np.testing.*`` /
``pytest.fail`` / ``pytest.raises``) — bare ``assert`` is a NO-OP under the
canonical ``-O`` invocation. Structural independence (L11): expected shapes
are hand-built from the mesh's own ``(N/ng, *spatial)`` + an
independently-computed ``per_axis ** ndim`` tail, NOT read off the field.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.geometry import BC, CoordSystem, Mesh1D, Mesh2D
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.axis import BasisKind
from orpheus.numerics.moment_layout import SPATIAL_MOMENT_AXIS_LABEL
from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.spaces import SphericalHarmonicSpace
from orpheus.transport.fields._bases import BulkField
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.spatial import DiamondDifference, LinearDiscontinuous
from orpheus.transport.fields import HarmonicMomentFlux
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.scalar_flux import ScalarFlux

from tests.sn._test_helpers import placeholder_materials


def _check(cond: bool, msg: str) -> None:
    """Mode-8-safe boolean assertion (a function call, fires under ``-O``)."""
    if not cond:
        pytest.fail(msg)


# ─────────────────────────────────────────────────────────────────────
# Mesh fixtures: a 2-D Cartesian mesh under each scheme + a 1-D LD mesh.
# ─────────────────────────────────────────────────────────────────────


def _mesh_2d(scheme):
    mesh = Mesh2D(
        edges_x=np.linspace(0.0, 2.0, 4),   # nx = 3
        edges_y=np.linspace(0.0, 3.0, 5),   # ny = 4
        mat_map=np.zeros((3, 4), dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_xmin=BC("reflective"), bc_xmax=BC("reflective"),
        bc_ymin=BC("reflective"), bc_ymax=BC("reflective"),
    )
    return SNMesh(
        mesh, Quadrature.level_symmetric(4), placeholder_materials(ng=2),
        scheme=scheme,
    )


def _mesh_1d(scheme):
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, 6),     # nx = 5
        mat_ids=np.zeros(5, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    return SNMesh(
        mesh, Quadrature.gauss_legendre(4), placeholder_materials(ng=2),
        scheme=scheme,
    )


@pytest.fixture
def dd_2d():
    return _mesh_2d(DiamondDifference())


@pytest.fixture
def ld_2d():
    return _mesh_2d(LinearDiscontinuous())


@pytest.fixture
def ld_1d():
    return _mesh_1d(LinearDiscontinuous())


# ─────────────────────────────────────────────────────────────────────
# (c) BYTE-IDENTITY AT DEFAULT — the negative control (load-bearing).
#
# No production field carries the axis yet, so the default factory output
# must be the EXACT pre-S3 space + shape for EVERY scheme (DD/Step AND LD).
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.parametrize("scheme_name", ["dd", "ld"])
def test_angular_flux_default_byte_identical_all_schemes(scheme_name, dd_2d, ld_2d):
    r"""The width-1 angular mint == the pre-S3 ``(N, ng, *spatial)``.

    ``angular_bulk_space`` appends NO factor regardless of the mesh's
    scheme — DD and LD produce the IDENTICAL space (widening is the
    caller's ``angular_trial_space`` selection, CS4b S5). Pinned against
    the independently-built expected shape from the mesh's own dims.
    """
    mesh = {"dd": dd_2d, "ld": ld_2d}[scheme_name]
    field = AngularFlux.zeros(mesh.angular_bulk_space)
    expected = (mesh.quad.N, mesh.ng, *mesh.spatial_shape)
    np.testing.assert_equal(field.space.shape, expected)
    np.testing.assert_equal(field.values.shape, expected)
    # the space is a bare FunctionSpace (no tensor-product factor at default)
    _check(not hasattr(field.space, "factors"), "default space must be bare")


@pytest.mark.foundation
@pytest.mark.parametrize("scheme_name", ["dd", "ld"])
def test_scalar_flux_default_byte_identical_all_schemes(scheme_name, dd_2d, ld_2d):
    r"""The width-1 scalar mint == the pre-S3 ``(ng, *spatial)``.

    The :class:`ScalarSourceSink` scattering accumulator constructs on
    this same ``bulk_space`` mint; it must stay byte-identical for both
    schemes (the negative control for the scattering-source widening).
    """
    mesh = {"dd": dd_2d, "ld": ld_2d}[scheme_name]
    field = ScalarFlux.zeros(mesh.bulk_space)
    expected = (mesh.ng, *mesh.spatial_shape)
    np.testing.assert_equal(field.space.shape, expected)
    np.testing.assert_equal(field.values.shape, expected)
    _check(not hasattr(field.space, "factors"), "default space must be bare")


@pytest.mark.foundation
@pytest.mark.parametrize("scheme_name", ["dd", "ld"])
def test_harmonic_moment_flux_default_byte_identical(scheme_name, dd_2d, ld_2d):
    r"""``HarmonicMomentFlux.zeros_for_mesh_and_L`` default == pre-S3 shape.

    The windowed iterate carrier. Default ``spatial_moments=1`` →
    ``(L+1, 2L+1, ng, *spatial)`` with NO trailing spatial-moment axis, AND
    the composition tree carries only the angular ``SphericalHarmonicSpace``
    factor (no axis labelled ``spatial_moment`` rides the product — since
    CS4c step 6 item 6.2c-iii the tail is the scheme's own axis, never a
    separate class).
    """
    mesh = {"dd": dd_2d, "ld": ld_2d}[scheme_name]
    L = 1
    field = HarmonicMomentFlux.zeros_for_mesh_and_L(mesh, L)
    expected = (L + 1, 2 * L + 1, mesh.ng, *mesh.spatial_shape)
    np.testing.assert_equal(field.space.shape, expected)
    np.testing.assert_equal(field.values.shape, expected)
    np.testing.assert_equal(field.spatial_moments, 1)
    # the angular factor is present; the spatial-moment axis is NOT (byte-id tree)
    np.testing.assert_equal(field.space.find_factor(SphericalHarmonicSpace).L, L)
    assert field.space.axes is not None
    np.testing.assert_equal(
        [ax.label for ax in field.space.axes if ax.label == SPATIAL_MOMENT_AXIS_LABEL], [],
    )
    np.testing.assert_equal(BulkField.spatial_moments_per_axis_of(field.space), 1)


# ─────────────────────────────────────────────────────────────────────
# (d) WIDENED PATH — explicit spatial_moments > 1 composes the factor.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.parametrize(
    "field_factory",
    [
        # angular: the carrier's named widened mint (CS4b S5)
        pytest.param(
            lambda m: AngularFlux.zeros(m.angular_trial_space),
            id="angular_flux",
        ),
        # scalar: no production consumer selects a widened scalar space
        # yet, so there is no named carrier mint — the endgame spelling
        # is the composed space (base axes + the scheme's moment axis).
        pytest.param(
            lambda m: ScalarFlux.zeros(
                FunctionSpace.of_axes(
                    *m.bulk_space.axes, m.scheme.moment_axis(m.axes),
                )
            ),
            id="scalar_flux",
        ),
    ],
)
def test_bulk_field_widened_2d_shape(field_factory, ld_2d):
    r"""A widened bulk field gets a trailing ``per_axis ** ndim`` axis (d=2).

    For the LD scheme (``per_axis = 2``) on a 2-D mesh the trailing axis
    is ``2 ** 2 = 4``. The expected trailing length is recomputed inline
    (not read off the field).
    """
    mesh = ld_2d
    per_axis = mesh.scheme.spatial_basis_per_axis
    field = field_factory(mesh)
    independent_tail = (per_axis ** mesh.ndim,)
    np.testing.assert_equal(field.space.shape[-1:], independent_tail)
    np.testing.assert_equal(field.values.shape, field.space.shape)
    # CS4b S2: on an axis-built bulk space the tail is the scheme-owned
    # MODAL moment AXIS (mass-carrying) — the scheme's own, the ONE spelling of the tail (6.2c-iii).
    tail_axis = field.space.axes[-1]
    np.testing.assert_equal(tail_axis.label, SPATIAL_MOMENT_AXIS_LABEL)
    assert tail_axis.kind is BasisKind.MODAL
    np.testing.assert_array_equal(
        tail_axis.weights, mesh.scheme.moment_mass_diagonal(mesh.axes),
    )
    np.testing.assert_equal(field.spatial_moments_per_axis, per_axis)


@pytest.mark.foundation
def test_angular_flux_widened_1d_shape(ld_1d):
    r"""A widened 1-D LD bulk field gets a trailing ``per_axis ** 1`` axis.

    The 1-D scan carrier: the LD trial mint on a 1-D mesh → trailing
    axis of ``2`` (the ``[bar, slope]`` per-cell moment pair).
    """
    mesh = ld_1d
    field = AngularFlux.zeros(mesh.angular_trial_space)
    np.testing.assert_equal(mesh.ndim, 1)
    np.testing.assert_equal(field.space.shape[-1], 2)
    np.testing.assert_equal(field.values.shape, field.space.shape)
    # CS4b S2: the tail is the MODAL moment axis; the width reader is the
    # production accessor (mechanism-blind).
    np.testing.assert_equal(field.space.axes[-1].label, SPATIAL_MOMENT_AXIS_LABEL)
    np.testing.assert_equal(field.spatial_moments_per_axis, 2)


@pytest.mark.foundation
def test_harmonic_moment_flux_widened_2d_shape(ld_2d):
    r"""A widened windowed iterate gets a trailing ``per_axis ** ndim`` axis.

    ``spatial_moments=2`` on a 2-D mesh → ``(L+1, 2L+1, ng, *spatial, 4)``;
    BOTH moment factors coexist — the angular head queryable by type, the
    spatial tail the scheme's own MODAL axis (mass-weighted, labelled
    ``spatial_moment``; CS4c step 6 item 6.2c-iii) read by label — the
    orthogonal-axes invariant on the live carrier.
    """
    mesh = ld_2d
    L = 1
    field = HarmonicMomentFlux.zeros_for_mesh_and_L(mesh, L, spatial_moments=2)
    expected = (L + 1, 2 * L + 1, mesh.ng, *mesh.spatial_shape, 2 ** mesh.ndim)
    np.testing.assert_equal(field.space.shape, expected)
    np.testing.assert_equal(field.values.shape, expected)
    np.testing.assert_equal(field.spatial_moments, 2)
    # both moment factors coexist (orthogonal axes)
    np.testing.assert_equal(field.space.find_factor(SphericalHarmonicSpace).L, L)
    np.testing.assert_equal(BulkField.spatial_moments_per_axis_of(field.space), 2)
    assert field.space.axes is not None
    tail = [ax for ax in field.space.axes if ax.label == SPATIAL_MOMENT_AXIS_LABEL]
    np.testing.assert_equal(len(tail), 1)
    np.testing.assert_equal(tail[0], mesh.scheme.moment_axis(mesh.axes))


@pytest.mark.foundation
def test_ctor_widened_roundtrip(ld_2d):
    r"""The ctor accepts a pre-shaped widened buffer on the trial mint.

    The reconstruction path (the iterate carrier's S3-A spelling, now
    space-primary): a buffer already carrying the trailing moment axis
    round-trips through ``AngularFlux(values=…,
    space=mesh.angular_trial_space)`` and passes the Field shape gate
    (``values.shape == space.shape`` — Pattern 4).
    """
    mesh = ld_2d
    shape = (mesh.quad.N, mesh.ng, *mesh.spatial_shape, 2 ** mesh.ndim)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    field = AngularFlux(values=values, space=mesh.angular_trial_space)
    np.testing.assert_equal(field.space.shape, shape)
    np.testing.assert_array_equal(field.values, values)


@pytest.mark.foundation
def test_widened_space_rejects_wrong_shape_buffer(ld_2d):
    r"""A buffer whose trailing axis disagrees with the trial space raises.

    Pattern 4: the Field shape gate cross-checks ``values.shape`` against
    the widened space; a mismatched trailing axis is an illegal state.
    Production invariant → real ``raise`` (fires under ``-O``).
    """
    mesh = ld_2d
    # the trial space carries tail 4 (per_axis² on 2-D); feed a tail-2 buffer
    bad_shape = (mesh.quad.N, mesh.ng, *mesh.spatial_shape, 2)
    bad = np.zeros(bad_shape)
    with pytest.raises(ValueError):
        AngularFlux(values=bad, space=mesh.angular_trial_space)
