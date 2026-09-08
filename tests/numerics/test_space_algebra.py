r"""Foundation tests for :mod:`orpheus.numerics.space`'s algebraic
constructions — :class:`TensorProductSpace`, :class:`DualSpace`, and
the :meth:`FunctionSpace.__mul__` / :meth:`FunctionSpace.dual` dunders.

Depth B step D-B (load-bearing). Pins the grand-report §6.1 / §15
tensor-product algebra at the L1 layer; consumed by Wave T (see
``.claude/plans/wave_t_tensor_network.md``) to type the codomains of
the boundary-realizer, fission, scattering, and streaming operators
that are rewired to ``TensorProductOperator`` / ``SumOfTensorProductsOperator``.

The invariants tested below are the type-system gates Wave T will
exercise — associativity of ``*``, shape composition, inner-product
factorisation, dual idempotency. Any failure here breaks the Wave T
operator-algebra rewires by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.metric import FactoredMetric
from orpheus.numerics.space import (
    DualSpace,
    FunctionSpace,
    TensorProductSpace,
)


# ─────────────────────────────────────────────────────────────────────
# TensorProductSpace construction + identity
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_mul_returns_tensor_product_space():
    a = FunctionSpace(name="X", shape=(4,))
    b = FunctionSpace(name="G", shape=(2,))
    tp = a * b
    assert isinstance(tp, TensorProductSpace)
    assert tp.shape == (4, 2)
    assert tp.factors == (a, b)


@pytest.mark.foundation
def test_mul_three_factors_associativity_left():
    """``(A * B) * C`` flattens to a 3-factor product (no nesting)."""
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="Omega", shape=(8,))
    c = FunctionSpace(name="G", shape=(2,))
    tp = (a * b) * c
    assert isinstance(tp, TensorProductSpace)
    assert tp.factors == (a, b, c)
    assert tp.shape == (3, 8, 2)


@pytest.mark.foundation
def test_mul_three_factors_associativity_right():
    """``A * (B * C)`` also flattens to a 3-factor product."""
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="Omega", shape=(8,))
    c = FunctionSpace(name="G", shape=(2,))
    tp = a * (b * c)
    assert isinstance(tp, TensorProductSpace)
    assert tp.factors == (a, b, c)
    assert tp.shape == (3, 8, 2)


@pytest.mark.foundation
def test_mul_associativity_produces_identical_results():
    """``(A*B)*C`` and ``A*(B*C)`` compare equal at the space level."""
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="Omega", shape=(8,))
    c = FunctionSpace(name="G", shape=(2,))
    left = (a * b) * c
    right = a * (b * c)
    assert left == right


@pytest.mark.foundation
def test_mul_name_format_uses_otimes():
    """The name reads as the math — ``X ⊗ G`` for ``X * G``."""
    a = FunctionSpace(name="X", shape=(4,))
    b = FunctionSpace(name="G", shape=(2,))
    tp = a * b
    assert "X" in tp.name and "G" in tp.name
    assert "⊗" in tp.name  # the actual U+2297 character


@pytest.mark.foundation
def test_mul_rejects_non_function_space():
    a = FunctionSpace(name="X", shape=(4,))
    with pytest.raises(TypeError):
        _ = a * 5  # int is not FunctionSpace


@pytest.mark.foundation
def test_from_factors_requires_at_least_two_factors():
    a = FunctionSpace(name="X", shape=(4,))
    with pytest.raises(ValueError, match="at least 2 factors"):
        TensorProductSpace.from_factors((a,))


# ─────────────────────────────────────────────────────────────────────
# Inner product factorisation
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_inner_product_factorises_euclidean():
    r"""For Euclidean factors,
    :math:`\langle x \otimes y, a \otimes b\rangle = \langle x, a\rangle \cdot \langle y, b\rangle`.

    With no inner-product weights, the TP space is Euclidean too.
    """
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="G", shape=(2,))
    tp = a * b
    # Euclidean: TP weights stay None.
    assert tp.inner_product_weights is None
    # Element-level identity: <x⊗y, a⊗b> = (Σ x·a)(Σ y·b).
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0])
    a_vec = np.array([2.0, 1.0, -1.0])
    b_vec = np.array([1.0, 3.0])
    # Tensor product values: shape (3, 2).
    lhs = tp.inner_product(np.outer(x, y), np.outer(a_vec, b_vec))
    rhs = a.inner_product(x, a_vec) * b.inner_product(y, b_vec)
    assert lhs == pytest.approx(rhs)


@pytest.mark.foundation
def test_inner_product_factorises_weighted():
    r"""For weighted factors, the tensor-product inner product is the
    factor inner products multiplied together.

    Pins the §15 identity that
    :math:`\langle \cdot, \cdot \rangle_{V_1 \otimes V_2}` factorises
    as
    :math:`\langle \cdot, \cdot \rangle_{V_1} \cdot \langle \cdot, \cdot \rangle_{V_2}`.
    """
    w_a = np.array([1.0, 2.0, 3.0])
    w_b = np.array([0.5, 0.5])
    a = FunctionSpace(name="X", shape=(3,), inner_product_weights=w_a)
    b = FunctionSpace(name="G", shape=(2,), inner_product_weights=w_b)
    tp = a * b
    # The product carries its metric FACTORED (CS4c step 6 item 6.2a) —
    # never the outer product as a stored tensor; the VALUES it applies
    # are that outer product.
    assert tp.inner_product_weights is None
    assert isinstance(tp.metric, FactoredMetric)
    np.testing.assert_array_almost_equal(
        tp.apply_metric(np.ones((3, 2))), np.outer(w_a, w_b),
    )
    # Factorisation identity.
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 1.0])
    a_vec = np.array([2.0, 2.0, 2.0])
    b_vec = np.array([1.0, 1.0])
    lhs = tp.inner_product(np.outer(x, y), np.outer(a_vec, b_vec))
    rhs = a.inner_product(x, a_vec) * b.inner_product(y, b_vec)
    assert lhs == pytest.approx(rhs)


@pytest.mark.foundation
def test_inner_product_mixed_euclidean_and_weighted():
    """Mixing one Euclidean factor with one weighted factor produces a
    weighted TP (the Euclidean side contributes ones-of-shape)."""
    a = FunctionSpace(name="X", shape=(3,))  # Euclidean
    w_b = np.array([1.0, 2.0])
    b = FunctionSpace(name="G", shape=(2,), inner_product_weights=w_b)
    tp = a * b
    assert tp.inner_product_weights is None
    assert isinstance(tp.metric, FactoredMetric)
    applied = tp.apply_metric(np.ones((3, 2)))
    # Each row equals w_b (since the Euclidean factor contributes 1).
    for i in range(3):
        np.testing.assert_array_almost_equal(applied[i], w_b)


# ─────────────────────────────────────────────────────────────────────
# Equality / hashing
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_two_tps_with_same_factors_compare_equal():
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="G", shape=(2,))
    tp1 = a * b
    tp2 = TensorProductSpace.from_factors((a, b))
    assert tp1 == tp2


@pytest.mark.foundation
def test_tp_usable_as_dict_key():
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="G", shape=(2,))
    tp = a * b
    d = {tp: "value"}
    assert d[a * b] == "value"


# ─────────────────────────────────────────────────────────────────────
# Dual space
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_dual_returns_dual_space():
    a = FunctionSpace(name="X", shape=(4,))
    a_dual = a.dual()
    assert isinstance(a_dual, DualSpace)
    assert a_dual.shape == a.shape
    assert a_dual.primal is a


@pytest.mark.foundation
def test_dual_name_appends_star():
    a = FunctionSpace(name="X", shape=(4,))
    assert "X" in a.dual().name
    assert "*" in a.dual().name


@pytest.mark.foundation
def test_dual_idempotent():
    r""":math:`V^{**} = V` (Riesz identification, double-dual is the primal)."""
    a = FunctionSpace(name="X", shape=(4,))
    a_doubledual = a.dual().dual()
    assert a_doubledual is a  # exact identity, not just equal


@pytest.mark.foundation
def test_dual_preserves_weights():
    """The L²-Riesz dual carries the same inner-product weights as the primal.

    Units are NOT a space property (View-G, issues #205 / #207) — they
    live on the field role-leaf, so there is nothing unit-like for the
    dual to preserve here.
    """
    w = np.array([1.0, 2.0, 3.0])
    a = FunctionSpace(name="X", shape=(3,), inner_product_weights=w)
    a_dual = a.dual()
    np.testing.assert_array_equal(a_dual.inner_product_weights, w)


# ─────────────────────────────────────────────────────────────────────
# Repr
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_tp_repr_includes_otimes_name():
    a = FunctionSpace(name="X", shape=(3,))
    b = FunctionSpace(name="G", shape=(2,))
    r = repr(a * b)
    assert "TensorProductSpace" in r
    assert "⊗" in r


@pytest.mark.foundation
def test_dual_repr_includes_star():
    a = FunctionSpace(name="X", shape=(4,))
    r = repr(a.dual())
    assert "DualSpace" in r
    assert "X*" in r
