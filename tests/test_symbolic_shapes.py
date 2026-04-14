"""Tests for dynamic/symbolic shape support.

Validates the new StableHLO ops (GetDimensionSizeOp, DynamicIotaOp,
DynamicBroadcastInDimOp, CustomCallOp/shape_assertion) and the modified
dot_general fast-path for symbolic dimensions produced by JAX symbolic export.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export

from stablehlo_coreml.translation_context import DYNAMIC_DIM, TranslationContext
from tests.utils import run_and_compare_symbolic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(spec: str):
    """Shorthand for jax.export.symbolic_shape."""
    return export.symbolic_shape(spec)


# ---------------------------------------------------------------------------
# Elementwise ops with symbolic batch dimension
# Exercises: build_func Symbol replacement, DynamicBroadcastInDimOp,
#            GetDimensionSizeOp, CustomCallOp(shape_assertion)
# ---------------------------------------------------------------------------

def test_symbolic_add_scalar():
    """x + 1.0 with symbolic first dim."""
    def f(x):
        return x + 1.0

    (n,) = _sym("n")
    specs = [jax.ShapeDtypeStruct((n, 4), jnp.float32)]
    test_shapes = [
        (np.random.randn(1, 4).astype(np.float32),),
        (np.random.randn(8, 4).astype(np.float32),),
        (np.random.randn(32, 4).astype(np.float32),),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


def test_symbolic_mul_elementwise():
    """x * y with matching symbolic dim."""
    def f(x, y):
        return x * y

    (n,) = _sym("n")
    specs = [
        jax.ShapeDtypeStruct((n, 3), jnp.float32),
        jax.ShapeDtypeStruct((n, 3), jnp.float32),
    ]
    test_shapes = [
        (np.random.randn(1, 3).astype(np.float32),
         np.random.randn(1, 3).astype(np.float32)),
        (np.random.randn(16, 3).astype(np.float32),
         np.random.randn(16, 3).astype(np.float32)),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


def test_symbolic_broadcast_add():
    """x + y where y is a smaller shape broadcast into x's symbolic dim."""
    def f(x, y):
        return x + y

    (n,) = _sym("n")
    specs = [
        jax.ShapeDtypeStruct((n, 4), jnp.float32),
        jax.ShapeDtypeStruct((1, 4), jnp.float32),
    ]
    test_shapes = [
        (np.random.randn(1, 4).astype(np.float32),
         np.random.randn(1, 4).astype(np.float32)),
        (np.random.randn(10, 4).astype(np.float32),
         np.random.randn(1, 4).astype(np.float32)),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


# ---------------------------------------------------------------------------
# Matrix multiplication with symbolic dims
# Exercises: _dot_general_dynamic fast path
# ---------------------------------------------------------------------------

def test_symbolic_matmul():
    """Matrix multiply (n, k) @ (k, m) with symbolic n."""
    def f(x, y):
        return x @ y

    (n,) = _sym("n")
    specs = [
        jax.ShapeDtypeStruct((n, 4), jnp.float32),
        jax.ShapeDtypeStruct((4, 8), jnp.float32),
    ]
    test_shapes = [
        (np.random.randn(1, 4).astype(np.float32),
         np.random.randn(4, 8).astype(np.float32)),
        (np.random.randn(5, 4).astype(np.float32),
         np.random.randn(4, 8).astype(np.float32)),
        (np.random.randn(32, 4).astype(np.float32),
         np.random.randn(4, 8).astype(np.float32)),
    ]
    run_and_compare_symbolic(f, specs, test_shapes, atol=1e-3)


def test_symbolic_matmul_rhs_dynamic():
    """Matrix multiply with symbolic dim on rhs: (k, n) where n is dynamic."""
    def f(x, y):
        return x @ y

    (n,) = _sym("n")
    specs = [
        jax.ShapeDtypeStruct((3, 4), jnp.float32),
        jax.ShapeDtypeStruct((4, n), jnp.float32),
    ]
    test_shapes = [
        (np.random.randn(3, 4).astype(np.float32),
         np.random.randn(4, 1).astype(np.float32)),
        (np.random.randn(3, 4).astype(np.float32),
         np.random.randn(4, 16).astype(np.float32)),
    ]
    run_and_compare_symbolic(f, specs, test_shapes, atol=1e-3)


def test_symbolic_einsum_batched():
    """Batched matmul via einsum: bnk,bkm->bnm with symbolic n."""
    def f(x, y):
        return jnp.einsum("bnk,bkm->bnm", x, y)

    (n,) = _sym("n")
    specs = [
        jax.ShapeDtypeStruct((2, n, 4), jnp.float32),
        jax.ShapeDtypeStruct((2, 4, 8), jnp.float32),
    ]
    test_shapes = [
        (np.random.randn(2, 1, 4).astype(np.float32),
         np.random.randn(2, 4, 8).astype(np.float32)),
        (np.random.randn(2, 7, 4).astype(np.float32),
         np.random.randn(2, 4, 8).astype(np.float32)),
    ]
    run_and_compare_symbolic(f, specs, test_shapes, atol=1e-3)


# ---------------------------------------------------------------------------
# Dynamic iota (arange with symbolic length)
# Exercises: DynamicIotaOp
# ---------------------------------------------------------------------------

def test_symbolic_arange():
    """jnp.arange(x.shape[0]) with symbolic dim => DynamicIotaOp."""
    def f(x):
        return jnp.arange(x.shape[0])

    (n,) = _sym("n")
    specs = [jax.ShapeDtypeStruct((n,), jnp.float32)]
    test_shapes = [
        (np.ones(1, dtype=np.float32),),
        (np.ones(5, dtype=np.float32),),
        (np.ones(16, dtype=np.float32),),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


# ---------------------------------------------------------------------------
# Reductions over symbolic dims
# ---------------------------------------------------------------------------

def test_symbolic_sum():
    """Sum over a symbolic-length axis."""
    def f(x):
        return jnp.sum(x, axis=0)

    (n,) = _sym("n")
    specs = [jax.ShapeDtypeStruct((n, 4), jnp.float32)]
    test_shapes = [
        (np.random.randn(1, 4).astype(np.float32),),
        (np.random.randn(10, 4).astype(np.float32),),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


def test_symbolic_mean():
    """Mean over a symbolic-length axis."""
    def f(x):
        return jnp.mean(x, axis=0)

    (n,) = _sym("n")
    specs = [jax.ShapeDtypeStruct((n, 4), jnp.float32)]
    test_shapes = [
        (np.random.randn(1, 4).astype(np.float32),),
        (np.random.randn(10, 4).astype(np.float32),),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


# ---------------------------------------------------------------------------
# Multiple symbolic dimensions
# ---------------------------------------------------------------------------

def test_symbolic_multi_dim():
    """Function with two independent symbolic dims."""
    def f(x):
        return x * 2.0

    (n, m) = _sym("n, m")
    specs = [jax.ShapeDtypeStruct((n, m), jnp.float32)]
    test_shapes = [
        (np.random.randn(1, 1).astype(np.float32),),
        (np.random.randn(4, 8).astype(np.float32),),
        (np.random.randn(16, 3).astype(np.float32),),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


# ---------------------------------------------------------------------------
# Attention-like pattern (the motivating use case)
# ---------------------------------------------------------------------------

def test_symbolic_scaled_dot_product_attention():
    """Simplified scaled dot-product attention with symbolic seq_len."""
    def attention(q, k, v):
        d_k = q.shape[-1]
        scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(d_k).astype(q.dtype)
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(weights, v)

    (seq,) = _sym("seq")
    d_model = 8
    specs = [
        jax.ShapeDtypeStruct((seq, d_model), jnp.float32),
        jax.ShapeDtypeStruct((seq, d_model), jnp.float32),
        jax.ShapeDtypeStruct((seq, d_model), jnp.float32),
    ]
    test_shapes = [
        (np.random.randn(1, d_model).astype(np.float32),
         np.random.randn(1, d_model).astype(np.float32),
         np.random.randn(1, d_model).astype(np.float32)),
        (np.random.randn(4, d_model).astype(np.float32),
         np.random.randn(4, d_model).astype(np.float32),
         np.random.randn(4, d_model).astype(np.float32)),
        (np.random.randn(16, d_model).astype(np.float32),
         np.random.randn(16, d_model).astype(np.float32),
         np.random.randn(16, d_model).astype(np.float32)),
    ]
    run_and_compare_symbolic(attention, specs, test_shapes, atol=1e-3)


# ---------------------------------------------------------------------------
# TranslationContext.validate_shapes with dynamic sentinel
# ---------------------------------------------------------------------------

def test_validate_shapes_dynamic_sentinel():
    """validate_shapes accepts dynamic dim sentinel paired with any MIL dim."""
    ctx = TranslationContext()

    class FakeType:
        def __init__(self, shape):
            self.shape = shape

    class FakeResult:
        def __init__(self, shape, name):
            self.type = FakeType(shape)
            self._name = name

        def get_name(self):
            return self._name

    from coremltools.converters.mil._deployment_compatibility import AvailableTarget
    from coremltools.converters.mil.mil import Builder as mb
    from coremltools.converters.mil.mil import Function, types

    with Function(
        {"x": mb.placeholder(shape=(1, 4), dtype=types.fp32)},
        opset_version=AvailableTarget.iOS18,
    ):
        # Create a real MIL var (not a Placeholder) via a const
        import numpy as np
        x = mb.const(val=np.zeros((1, 4), dtype=np.float32))
        scalar = mb.const(val=np.zeros((1,), dtype=np.float32))

        # Exact match
        ctx.add_result(FakeResult((1, 4), "exact"), x)

        # Scalar tolerance
        ctx.add_result(FakeResult((), "scalar"), scalar)

        # Dynamic dim — any MIL dim should be accepted
        ctx.add_result(FakeResult((DYNAMIC_DIM, 4), "dyn1"), x)
        ctx.add_result(FakeResult((DYNAMIC_DIM, DYNAMIC_DIM), "dyn2"), x)

        # Mismatch should raise
        with pytest.raises(ValueError, match="different from the actual MIL result shape"):
            ctx.add_result(FakeResult((2, 4), "bad"), x)


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------

def test_custom_call_unsupported_raises():
    """CustomCallOp with unknown target should raise ValueError."""
    import coremltools as ct
    from jax._src.interpreters import mlir as jax_mlir
    from jax._src.lib.mlir import ir

    from stablehlo_coreml.converter import convert

    # Craft a minimal MLIR module with an unsupported custom_call
    mlir_text = """
    module {
      func.func public @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
        %0 = stablehlo.custom_call @unsupported_target(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
        return %0 : tensor<2x3xf32>
      }
    }
    """
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(mlir_text, context=context)

    with pytest.raises(ValueError, match="Custom call is not supported"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18)


# ---------------------------------------------------------------------------
# DynamicReshapeOp
# ---------------------------------------------------------------------------

def test_symbolic_reshape():
    """Reshape with a runtime-computed output shape via DynamicReshapeOp."""
    def f(x):
        # Merges the symbolic dim with the static dim
        return jnp.reshape(x, (x.shape[0] * 4,))

    (n,) = _sym("n")
    specs = [jax.ShapeDtypeStruct((n, 4), jnp.float32)]
    test_shapes = [
        (np.random.randn(1, 4).astype(np.float32),),
        (np.random.randn(3, 4).astype(np.float32),),
        (np.random.randn(8, 4).astype(np.float32),),
    ]
    run_and_compare_symbolic(f, specs, test_shapes)


def test_symbolic_dynamic_iota_unsupported_raises():
    """DynamicIotaOp with rank > 1 should raise ValueError."""
    import coremltools as ct
    from jax._src.interpreters import mlir as jax_mlir
    from jax._src.lib.mlir import ir

    from stablehlo_coreml.converter import convert

    mlir_text = """
    module {
      func.func public @main(%arg0: tensor<2xi32>) -> tensor<2x3xi64> {
        %0 = stablehlo.dynamic_iota %arg0, dim = 0 : (tensor<2xi32>) -> tensor<2x3xi64>
        return %0 : tensor<2x3xi64>
      }
    }
    """
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(mlir_text, context=context)

    with pytest.raises(ValueError, match=r"DynamicIotaOp with rank=2.*is not yet supported"):
        convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
