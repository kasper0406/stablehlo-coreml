import math

import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipelineManager
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)

from stablehlo_coreml.passes.fuse_attention_to_sdpa import _Atom, _regroup
from tests.utils import get_model_instruction_types, run_and_compare

PASS_NAME = "common::fuse_attention_to_sdpa"
DCE_PASS_NAME = "common::dead_code_elimination"

# [B, H, L, S] scores from [B, H, L, E] queries and [B, H, S, E] keys.
B, H, L, S, E = 2, 3, 5, 7, 4

NEG_INF = np.float32(-np.inf)
FINITE_FILL = np.float32(-1e9)
# What HuggingFace actually uses for "masked out": large, but finite.
TORCH_MIN_FILL = np.float32(np.finfo(np.float32).min)


def _apply(prog, **options):
    # The pass leaves the matched ops behind; DCE is what removes them.
    if not options:
        result = apply_pass_and_basic_check(prog, PASS_NAME, skip_output_shape_check=True)
        apply_pass_and_basic_check(prog, DCE_PASS_NAME, skip_output_shape_check=True)
        return result
    pipeline = ct.PassPipeline([PASS_NAME, DCE_PASS_NAME], "fuse_attention")
    pipeline.set_options(PASS_NAME, options)
    PassPipelineManager.apply_pipeline(prog, pipeline)
    return prog


def _ops(prog):
    return list(prog.functions["main"].operations)


def _sdpa_ops(prog, recurse: bool = True) -> int:
    return get_op_types_in_program(prog, recurse=recurse).count("scaled_dot_product_attention")


def _all_neg_inf_rows(scores, shape=(B, H, L, S)):
    """The row predicate of torch's ``_safe_softmax``, as torchax lowers it.

    ``not any(scores != -inf)``, broadcast back over the key axis.
    """
    *lead, keys = shape
    finite = mb.logical_not(x=mb.equal(x=scores, y=NEG_INF))
    any_finite = mb.reduce_max(x=mb.cast(x=finite, dtype="int32"), axes=[len(shape) - 1], keep_dims=False)
    rows = mb.reshape(x=mb.cast(x=any_finite, dtype="bool"), shape=[*lead, 1])
    return mb.tile(x=mb.logical_not(x=rows), reps=[1] * len(lead) + [keys])


def _qkv_specs():
    return [
        mb.TensorSpec(shape=(B, H, L, E)),
        mb.TensorSpec(shape=(B, H, S, E)),
        mb.TensorSpec(shape=(B, H, S, E)),
    ]


def _predict(prog, **inputs):
    """Convert ``prog`` and run it; returns the single output as an ndarray."""
    model = ct.convert(
        prog,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        # Keep fp32 throughout; the default pipeline would otherwise downcast and
        # leave only ~1e-3 of accuracy to compare against.
        compute_precision=ct.precision.FLOAT32,
    )
    names = [feature.name for feature in model.get_spec().description.input]
    result = model.predict({name: inputs[name] for name in names})
    return np.array(next(iter(result.values())))


def _reference_attention(q, k, v, mask=None, fill=-1e9, bias=None):
    """``softmax(where(mask, q @ k^T [+ bias], fill)) @ v``, over the last two axes."""
    scores = np.matmul(q, np.swapaxes(k, -2, -1))
    if bias is not None:
        scores = scores + bias
    if mask is not None:
        scores = np.where(mask, scores, np.float32(fill))
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v)


class TestFuseAttentionToSdpa:
    """Unit tests on hand-built MIL programs."""

    def test_masked_attention_with_neg_inf_fill(self):
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            masked = mb.select(cond=mask, a=scaled, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_dot_product_attention"]
        sdpa = _ops(prog)[-1]
        # A -inf fill has the same semantics as SDPA's boolean mask.
        assert sdpa.attn_mask.dtype == types.bool
        assert_model_is_valid(
            prog,
            {"q": (B, H, L, E), "k": (B, H, S, E), "v": (B, H, S, E), "mask": (B, 1, 1, S)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_masked_attention_with_finite_fill_defaults_to_additive(self):
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            masked = mb.select(cond=mask, a=scaled, b=FINITE_FILL)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops[-1] == "scaled_dot_product_attention"
        assert ops.count("scaled_dot_product_attention") == 1
        sdpa = _ops(prog)[-1]
        assert sdpa.attn_mask.dtype == types.fp32

    def test_masked_attention_with_torch_finfo_min_fill_is_additive(self):
        """HuggingFace masks with `torch.finfo(dtype).min`, which is finite."""
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            masked = mb.select(cond=mask, a=scaled, b=TORCH_MIN_FILL)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        sdpa = _ops(prog)[-1]
        # A fully masked row stays finite in the original graph, so the mask has
        # to be additive even though the fill is way below the fp32 range.
        assert sdpa.attn_mask.dtype == types.fp32

    def test_masked_attention_with_finite_fill_bool_option(self):
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            masked = mb.select(cond=mask, a=scaled, b=FINITE_FILL)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog, finite_fill_mask="bool")
        assert get_op_types_in_program(prog) == ["scaled_dot_product_attention"]
        assert _ops(prog)[-1].attn_mask.dtype == types.bool

    def test_swapped_select_operands(self):
        """`select(cond, fill, scores)` masks where the condition is true."""
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            masked = mb.select(cond=mask, a=NEG_INF, b=scaled)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["logical_not", "scaled_dot_product_attention"]

    def test_unmasked_attention(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.real_div(x=scores, y=np.float32(math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_dot_product_attention"]
        assert _ops(prog)[-1].attn_mask is None

    def test_additive_float_mask(self):
        @mb.program(input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, L, S))], opset_version=ct.target.iOS18)
        def prog(q, k, v, bias):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            biased = mb.add(x=scaled, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_dot_product_attention"]
        sdpa = _ops(prog)[-1]
        assert sdpa.attn_mask is prog.functions["main"].inputs["bias"]

    def test_additive_float_mask_is_cast_to_the_query_dtype(self):
        """SDPA wants `attn_mask` in the operand dtype; a bias behind a cast is not.

        Keeping the softmax in fp32 while Q/K/V stay fp16 is what torch and JAX
        emit; the backward walk follows the bias across that cast.
        """
        bias = np.linspace(-2.0, 2.0, L * S, dtype=np.float32).reshape(1, 1, L, S)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(B, H, L, E), dtype=types.fp16),
                mb.TensorSpec(shape=(B, H, S, E), dtype=types.fp16),
                mb.TensorSpec(shape=(B, H, S, E), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float16(1.0 / math.sqrt(E)))
            biased = mb.add(x=mb.cast(x=scaled, dtype="fp32"), y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            return mb.matmul(x=mb.cast(x=weights, dtype="fp16"), y=v, transpose_y=False)

        _apply(prog)
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        assert sdpa.query.dtype == types.fp16
        assert sdpa.attn_mask.dtype == sdpa.query.dtype

        rng = np.random.RandomState(0)
        q = rng.randn(B, H, L, E).astype(np.float16)
        k = rng.randn(B, H, S, E).astype(np.float16)
        v = rng.randn(B, H, S, E).astype(np.float16)
        expected = _reference_attention(
            q.astype(np.float32) / math.sqrt(E), k.astype(np.float32), v.astype(np.float32), bias=bias
        )
        np.testing.assert_allclose(_predict(prog, q=q, k=k, v=v), expected, atol=2e-2, rtol=2e-2)

    def test_scale_applied_after_the_mask_scales_the_fill(self):
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            masked = mb.select(cond=mask, a=scores, b=FINITE_FILL)
            scaled = mb.mul(x=masked, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        # The fill is applied before the scale, so it has to be scaled as well.
        fill = next(op for op in _ops(prog) if op.op_type == "mul" and op.x.op is not None)
        np.testing.assert_allclose(fill.y.val, -1e9 / math.sqrt(E), rtol=1e-5)

    def test_not_fused_when_the_mask_broadcasts_the_scores_up(self):
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, H, L + 1, S))],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, bias):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            reduced = mb.reduce_sum(x=scores, axes=[2], keep_dims=True)
            biased = mb.add(x=reduced, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            merged = mb.reduce_sum(x=weights, axes=[2], keep_dims=True)
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_uniform_constant_bias_is_not_a_mask(self):
        """A uniform shift is a no-op for softmax; there is nothing to fuse."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            biased = mb.add(x=scores, y=np.float32(1.0))
            weights = mb.softmax(x=biased, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_non_uniform_constant_bias_is_an_additive_mask(self):
        """T5 folds its relative position bias into a constant added to the scores."""
        bias = np.arange(L * S, dtype=np.float32).reshape(1, 1, L, S)

        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            biased = mb.add(x=scaled, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["scaled_dot_product_attention"]
        sdpa = _ops(prog)[-1]
        np.testing.assert_array_equal(sdpa.attn_mask.val, bias)

    def test_no_scale_prescales_the_query(self):
        """Without an explicit scale, Q must cancel SDPA's built-in 1/sqrt(E)."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["mul", "scaled_dot_product_attention"]
        scale = _ops(prog)[-2]
        np.testing.assert_allclose(scale.y.val, math.sqrt(E), rtol=1e-5)

    @pytest.mark.parametrize("scale", [1.0 / math.sqrt(E), 0.5 / math.sqrt(E)])
    def test_explicit_scale(self, scale):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=np.float32(scale), y=scores)
            weights = mb.softmax(x=scaled, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        factor = scale * math.sqrt(E)
        if factor == 1.0:
            assert ops == ["scaled_dot_product_attention"]
        else:
            assert ops == ["mul", "scaled_dot_product_attention"]
            np.testing.assert_allclose(_ops(prog)[-2].y.val, factor, rtol=1e-5)

    def test_near_builtin_scale_is_preserved(self):
        scale = np.float32(0.4999)

        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=scale)
            weights = mb.softmax(x=scaled, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["mul", "scaled_dot_product_attention"]
        np.testing.assert_allclose(_ops(prog)[-2].y.val, float(scale) * math.sqrt(E), rtol=1e-7)

    def test_transposed_key(self):
        """`transpose_y=False` means the key still has to be transposed."""
        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(B, H, L, E)),
            mb.TensorSpec(shape=(B, H, E, S)),
            mb.TensorSpec(shape=(B, H, S, E)),
        ])
        def prog(q, k_t, v):
            scores = mb.matmul(x=q, y=k_t, transpose_y=False)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert ops.count("transpose") == 1

    def test_existing_key_transpose_is_peeled(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            k_t = mb.transpose(x=k, perm=[0, 1, 3, 2])
            scores = mb.matmul(x=q, y=k_t, transpose_y=False)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "transpose" not in ops

    def test_transposed_value(self):
        """The converter emits `matmul(w, v_t, transpose_y=True)`; peel the transpose."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            v_t = mb.transpose(x=v, perm=[0, 1, 3, 2])
            return mb.matmul(x=weights, y=v_t, transpose_y=True)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "transpose" not in ops

    def test_reshaped_gqa_layout_with_key_mask(self):
        """Gemma-style GQA: [B, K, G*T, E] queries reshaped to [B, K, G, T, E]."""
        batch, heads, groups, tokens, keys, embedding = 1, 2, 3, 5, 7, 4

        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(batch, heads, groups * tokens, embedding)),
            mb.TensorSpec(shape=(batch, heads, keys, embedding)),
            mb.TensorSpec(shape=(batch, heads, keys, embedding)),
            mb.TensorSpec(shape=(batch, 1, 1, 1, keys), dtype=types.bool),
        ])
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            split = mb.reshape(x=scores, shape=[batch, heads, groups, tokens, keys])
            masked = mb.select(cond=mask, a=split, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            merged = mb.reshape(x=weights, shape=[batch, heads, groups * tokens, keys])
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "softmax" not in ops
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        # A pure key mask collapses to [1, 1, 1, S].
        assert tuple(sdpa.attn_mask.shape) == (1, 1, 1, keys)

    def test_reshaped_gqa_layout_with_full_mask(self):
        """A mask that varies per query row has to be materialised in matmul space."""
        batch, heads, groups, tokens, keys, embedding = 1, 2, 3, 5, 7, 4

        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(batch, heads, groups * tokens, embedding)),
            mb.TensorSpec(shape=(batch, heads, keys, embedding)),
            mb.TensorSpec(shape=(batch, heads, keys, embedding)),
            mb.TensorSpec(shape=(batch, 1, 1, tokens, keys), dtype=types.bool),
        ])
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            split = mb.reshape(x=scores, shape=[batch, heads, groups, tokens, keys])
            masked = mb.select(cond=mask, a=split, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            merged = mb.reshape(x=weights, shape=[batch, heads, groups * tokens, keys])
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "tile" in ops
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        assert tuple(sdpa.attn_mask.shape) == (batch, heads, groups * tokens, keys)

    def test_degenerate_single_query_and_key(self):
        """Whisper's decoder self-attention: L = S = 1 around a [1, H, 1, 1] reshape."""
        heads = 4

        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(heads, 1, E)),
            mb.TensorSpec(shape=(heads, E, 1)),
            mb.TensorSpec(shape=(heads, 1, E)),
            mb.TensorSpec(shape=(1, 1, 1, 1)),
        ])
        def prog(q, k_t, v, bias):
            scores = mb.matmul(x=q, y=k_t, transpose_y=False)   # [heads, 1, 1]
            split = mb.reshape(x=scores, shape=[1, heads, 1, 1])
            biased = mb.add(x=split, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            merged = mb.reshape(x=weights, shape=[heads, 1, 1])
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "softmax" not in ops
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        assert tuple(sdpa.attn_mask.shape) == (1, 1, 1)
        assert_model_is_valid(
            prog,
            {"q": (heads, 1, E), "k_t": (heads, E, 1), "v": (heads, 1, E), "bias": (1, 1, 1, 1)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_transposed_score_layout(self):
        """Softmax may run over the *first* matmul result axis (roles swapped)."""
        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(B, H, S, E)),
            mb.TensorSpec(shape=(B, H, L, E)),
            mb.TensorSpec(shape=(B, H, S, E)),
        ])
        def prog(k, q, v):
            scores = mb.matmul(x=k, y=q, transpose_y=True)      # [B, H, S, L]
            swapped = mb.transpose(x=scores, perm=[0, 1, 3, 2])  # [B, H, L, S]
            weights = mb.softmax(x=swapped, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        assert sdpa.key is prog.functions["main"].inputs["k"]

    def test_symbolic_sequence_dim(self):
        seq = get_new_symbol()

        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(B, H, L, E)),
            mb.TensorSpec(shape=(B, H, seq, E)),
            mb.TensorSpec(shape=(B, H, seq, E)),
        ])
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 1

    def test_not_fused_when_weights_are_used_elsewhere(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            attention = mb.matmul(x=weights, y=v, transpose_y=False)
            return mb.add(x=attention, y=mb.reduce_sum(x=weights, axes=[3], keep_dims=True))

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_not_fused_when_scores_are_used_elsewhere(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            attention = mb.matmul(x=weights, y=v, transpose_y=False)
            return mb.add(x=attention, y=mb.reduce_sum(x=scores, axes=[3], keep_dims=True))

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_not_fused_for_rank_2_matmuls(self):
        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(L, E)),
            mb.TensorSpec(shape=(S, E)),
            mb.TensorSpec(shape=(S, E)),
        ])
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_not_fused_when_batch_dims_disagree(self):
        """SDPA does not broadcast batch dimensions."""
        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(B, H, L, E)),
            mb.TensorSpec(shape=(B, H, S, E)),
            mb.TensorSpec(shape=(1, H, S, E)),
        ])
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_not_fused_when_softmax_is_not_over_the_last_axis(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=2)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_fused_inside_nested_block(self):
        @mb.program(input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(1,), dtype=types.bool)], opset_version=ct.target.iOS18)
        def prog(q, k, v, pred):
            def true_fn():
                scores = mb.matmul(x=q, y=k, transpose_y=True)
                weights = mb.softmax(x=scores, axis=-1)
                return mb.matmul(x=weights, y=v, transpose_y=False)

            def false_fn():
                return mb.mul(x=q, y=np.float32(0.0))

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        assert _sdpa_ops(prog) == 1

    def test_is_idempotent(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops_after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == ops_after_first


class TestAdditiveBiasAndSelectMask:
    """``select(cond, scores + bias, fill)``: both masks at once.

    ``jax.nn.dot_product_attention`` -- and therefore ``flax.nnx``, which calls
    it -- adds the ``bias`` to the logits and *then* applies ``mask`` as a
    ``where`` against a large negative fill. SDPA has a single ``attn_mask``, so
    the two have to collapse into ``select(cond, bias, fill)``.
    """

    @staticmethod
    def _prog(fill=NEG_INF, mask_shape=(B, 1, 1, S)):
        @mb.program(
            opset_version=ct.target.iOS18,
            input_specs=[
                *_qkv_specs(),
                mb.TensorSpec(shape=(B, 1, L, S)),
                mb.TensorSpec(shape=mask_shape, dtype=types.bool),
            ],
        )
        def prog(q, k, v, bias, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            biased = mb.add(x=scaled, y=bias)
            masked = mb.select(cond=mask, a=biased, b=fill)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        return prog

    def test_bias_and_neg_inf_mask_collapse_into_one_float_mask(self):
        prog = self._prog()
        _apply(prog)

        assert get_op_types_in_program(prog) == ["select", "scaled_dot_product_attention"]
        select = next(op for op in _ops(prog) if op.op_type == "select")
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        assert select.cond is prog.functions["main"].inputs["mask"]
        assert select.a is prog.functions["main"].inputs["bias"]
        assert math.isinf(float(select.b.val)) and float(select.b.val) < 0
        assert sdpa.attn_mask is select.outputs[0]
        assert sdpa.attn_mask.dtype == types.fp32
        assert_model_is_valid(
            prog,
            {"q": (B, H, L, E), "k": (B, H, S, E), "v": (B, H, S, E),
             "bias": (B, 1, L, S), "mask": (B, 1, 1, S)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_bias_and_finite_fill_keep_the_finite_fill(self):
        """A finite fill leaves a fully masked row finite; do not turn it to -inf."""
        prog = self._prog(fill=TORCH_MIN_FILL)
        _apply(prog)

        assert get_op_types_in_program(prog) == ["select", "scaled_dot_product_attention"]
        select = next(op for op in _ops(prog) if op.op_type == "select")
        np.testing.assert_allclose(select.b.val, TORCH_MIN_FILL, rtol=1e-6)

    def test_bias_and_finite_fill_with_the_bool_option_masks_with_neg_inf(self):
        """``finite_fill_mask='bool'`` asks for -inf semantics, float mask or not."""
        prog = self._prog(fill=TORCH_MIN_FILL)
        _apply(prog, finite_fill_mask="bool")

        assert get_op_types_in_program(prog) == ["select", "scaled_dot_product_attention"]
        select = next(op for op in _ops(prog) if op.op_type == "select")
        assert math.isinf(float(select.b.val))

    def test_negated_mask_with_a_bias(self):
        @mb.program(
            opset_version=ct.target.iOS18,
            input_specs=[
                *_qkv_specs(),
                mb.TensorSpec(shape=(B, 1, L, S)),
                mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool),
            ],
        )
        def prog(q, k, v, bias, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            biased = mb.add(x=scaled, y=bias)
            masked = mb.select(cond=mask, a=NEG_INF, b=biased)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops == ["logical_not", "select", "scaled_dot_product_attention"]

    def test_a_scale_between_the_bias_and_the_mask_scales_only_the_bias(self):
        """``(scores + bias) * c`` then mask: SDPA applies attn_mask after the scale."""
        scale = np.float32(0.25)

        @mb.program(
            opset_version=ct.target.iOS18,
            input_specs=[
                *_qkv_specs(),
                mb.TensorSpec(shape=(B, 1, L, S)),
                mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool),
            ],
        )
        def prog(q, k, v, bias, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            biased = mb.add(x=scores, y=bias)
            scaled = mb.mul(x=biased, y=scale)
            masked = mb.select(cond=mask, a=scaled, b=FINITE_FILL)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        ops = _ops(prog)
        assert get_op_types_in_program(prog).count("scaled_dot_product_attention") == 1
        bias_scale = next(op for op in ops if op.op_type == "mul" and op.x.op is None)
        np.testing.assert_allclose(bias_scale.y.val, scale, rtol=1e-6)
        # The mask runs after the scale, so its fill stays as it is.
        select = next(op for op in ops if op.op_type == "select")
        np.testing.assert_allclose(select.b.val, FINITE_FILL, rtol=1e-6)

    def test_keeps_the_original_numerics(self):
        prog = self._prog(fill=NEG_INF, mask_shape=(B, 1, L, S))
        _apply(prog)
        assert get_op_types_in_program(prog).count("scaled_dot_product_attention") == 1

        rng = np.random.RandomState(0)
        q = rng.randn(B, H, L, E).astype(np.float32)
        k = rng.randn(B, H, S, E).astype(np.float32)
        v = rng.randn(B, H, S, E).astype(np.float32)
        bias = rng.randn(B, 1, L, S).astype(np.float32)
        # Every row keeps at least one key, so -inf never produces a NaN row.
        mask = (np.arange(S) < S - 2).reshape(1, 1, 1, S) & np.ones((B, 1, L, S), dtype=bool)

        # The program scales the scores before adding the bias.
        expected = _reference_attention(
            q / math.sqrt(E), k, v, mask=mask, fill=-np.inf, bias=bias
        )
        # Core ML has no boolean model input; it exposes the mask as fp32.
        got = _predict(prog, q=q, k=k, v=v, bias=bias, mask=mask.astype(np.float32))
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)

    def test_a_bias_applied_after_the_mask_is_not_fused(self):
        """``where(cond, scores, fill) + bias`` also shifts the fill; stay away."""
        @mb.program(
            opset_version=ct.target.iOS18,
            input_specs=[
                *_qkv_specs(),
                mb.TensorSpec(shape=(B, 1, L, S)),
                mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool),
            ],
        )
        def prog(q, k, v, bias, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            masked = mb.select(cond=mask, a=scores, b=FINITE_FILL)
            biased = mb.add(x=masked, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_two_additive_masks_are_not_fused(self):
        @mb.program(
            opset_version=ct.target.iOS18,
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, L, S)),
                         mb.TensorSpec(shape=(B, 1, L, S))],
        )
        def prog(q, k, v, bias_0, bias_1):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            biased = mb.add(x=mb.add(x=scores, y=bias_0), y=bias_1)
            weights = mb.softmax(x=biased, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_two_select_masks_are_not_fused(self):
        @mb.program(
            opset_version=ct.target.iOS18,
            input_specs=[*_qkv_specs(),
                         mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool),
                         mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
        )
        def prog(q, k, v, mask_0, mask_1):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            masked = mb.select(cond=mask_0, a=scores, b=NEG_INF)
            masked = mb.select(cond=mask_1, a=masked, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0


class TestShapeBounds:
    """``max_head_dim`` / ``max_key_len`` opt a block out of the fusion by size.

    A fused SDPA over global attention (large ``E``, long ``S``) is known to
    break the ANE partitioner and the BNNS fp16 kernel; consumers targeting
    those backends leave such sites decomposed.
    """

    @staticmethod
    def _prog(key_len=S):
        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(B, H, L, E)),
            mb.TensorSpec(shape=(B, H, key_len, E)),
            mb.TensorSpec(shape=(B, H, key_len, E)),
        ])
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            weights = mb.softmax(x=scores, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        return prog

    @pytest.mark.parametrize("option, fused_at, unfused_at", [
        ("max_head_dim", E, E - 1),
        ("max_key_len", S, S - 1),
    ])
    def test_bound_is_inclusive(self, option, fused_at, unfused_at):
        prog = self._prog()
        _apply(prog, **{option: fused_at})
        assert _sdpa_ops(prog) == 1

        prog = self._prog()
        _apply(prog, **{option: unfused_at})
        assert _sdpa_ops(prog) == 0
        assert "softmax" in get_op_types_in_program(prog)

    def test_bounds_accept_the_string_form_set_options_documents(self):
        prog = self._prog()
        _apply(prog, max_head_dim=str(E - 1))
        assert _sdpa_ops(prog) == 0

        prog = self._prog()
        _apply(prog, max_head_dim="none", max_key_len="")
        assert _sdpa_ops(prog) == 1

    def test_symbolic_key_length_counts_as_too_long(self):
        prog = self._prog(key_len=get_new_symbol())
        _apply(prog, max_key_len=4096)
        assert _sdpa_ops(prog) == 0

        # ...but only when a bound is set; unbounded fuses as before.
        prog = self._prog(key_len=get_new_symbol())
        _apply(prog)
        assert _sdpa_ops(prog) == 1

    @pytest.mark.parametrize("value", [0, -1, "abc", 1.5])
    def test_invalid_bound_is_rejected(self, value):
        with pytest.raises(ValueError, match="max_key_len"):
            _apply(self._prog(), max_key_len=value)

    def test_options_do_not_leak_between_runs(self):
        """The pass instance is a registry singleton; a bound set once must not stick."""
        prog = self._prog()
        _apply(prog, max_key_len=S - 1)
        assert _sdpa_ops(prog) == 0

        prog = self._prog()
        _apply(prog)
        assert _sdpa_ops(prog) == 1


class TestUnitQueryLength:
    """The autoregressive decode step: a single query row.

    ``dot_general`` on a ``[1, H, 1, E] x [1, H, S, E]`` pair lowers to a rank-3
    ``matmul`` over ``[H, 1, S]``, and the converter puts the leading batch axis
    back afterwards with ``reshape [H, 1, 1, S] -> transpose [1, H, 1, S]``.

    That reshape places a *second* size-1 axis right next to the query axis, so
    which of the two holds the query rows cannot be decided from the sizes -- and
    the transpose then moves the two apart, dropping the query axis into the
    batch position. It makes no difference to the elements (a size-1 axis is
    always indexed 0), and the pass must not be confused by it.
    """

    HEADS, KEYS, EMBED = 3, 7, 4

    @classmethod
    def _specs(cls, *extra):
        return [
            mb.TensorSpec(shape=(cls.HEADS, 1, cls.EMBED)),
            mb.TensorSpec(shape=(cls.HEADS, cls.KEYS, cls.EMBED)),
            mb.TensorSpec(shape=(cls.HEADS, cls.KEYS, cls.EMBED)),
            *extra,
        ]

    def test_unit_query_axis_is_fused(self):
        heads, keys, embedding = self.HEADS, self.KEYS, self.EMBED

        @mb.program(opset_version=ct.target.iOS18, input_specs=self._specs())
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)              # [H, 1, S]
            split = mb.reshape(x=scores, shape=[heads, 1, 1, keys])     # [H, 1, 1, S]
            moved = mb.transpose(x=split, perm=[2, 0, 1, 3])            # [1, H, 1, S]
            weights = mb.softmax(x=moved, axis=-1)
            merged = mb.reshape(x=weights, shape=[heads, 1, keys])      # [H, 1, S]
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "softmax" not in ops
        assert "matmul" not in ops
        assert_model_is_valid(
            prog,
            {"q": (heads, 1, embedding), "k": (heads, keys, embedding), "v": (heads, keys, embedding)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_unit_query_axis_with_key_mask_is_fused(self):
        """The decode mask is one flag per key, shared by every head."""
        heads, keys = self.HEADS, self.KEYS
        mask_spec = mb.TensorSpec(shape=(1, 1, 1, keys), dtype=types.bool)

        @mb.program(opset_version=ct.target.iOS18, input_specs=self._specs(mask_spec))
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            split = mb.reshape(x=scores, shape=[heads, 1, 1, keys])
            moved = mb.transpose(x=split, perm=[2, 0, 1, 3])
            masked = mb.select(cond=mask, a=moved, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            merged = mb.reshape(x=weights, shape=[heads, 1, keys])
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        assert "softmax" not in ops
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        # A pure key mask collapses to [1, 1, S] in the rank-3 matmul space.
        assert tuple(sdpa.attn_mask.shape) == (1, 1, keys)

    def test_unit_query_axis_with_per_head_mask_is_fused(self):
        """A mask that varies per head has to be re-laid out in matmul space."""
        heads, keys = self.HEADS, self.KEYS
        mask_spec = mb.TensorSpec(shape=(1, heads, 1, keys), dtype=types.bool)

        @mb.program(opset_version=ct.target.iOS18, input_specs=self._specs(mask_spec))
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            split = mb.reshape(x=scores, shape=[heads, 1, 1, keys])
            moved = mb.transpose(x=split, perm=[2, 0, 1, 3])
            masked = mb.select(cond=mask, a=moved, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            merged = mb.reshape(x=weights, shape=[heads, 1, keys])
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        sdpa = next(op for op in _ops(prog) if op.op_type == "scaled_dot_product_attention")
        assert tuple(sdpa.attn_mask.shape) == (heads, 1, keys)

    def test_unit_query_axis_keeps_the_original_numerics(self):
        heads, keys, embedding = self.HEADS, self.KEYS, self.EMBED
        mask_spec = mb.TensorSpec(shape=(1, 1, 1, keys), dtype=types.bool)

        @mb.program(opset_version=ct.target.iOS18, input_specs=self._specs(mask_spec))
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            split = mb.reshape(x=scores, shape=[heads, 1, 1, keys])
            moved = mb.transpose(x=split, perm=[2, 0, 1, 3])
            masked = mb.select(cond=mask, a=moved, b=FINITE_FILL)
            weights = mb.softmax(x=masked, axis=-1)
            merged = mb.reshape(x=weights, shape=[heads, 1, keys])
            return mb.matmul(x=merged, y=v, transpose_y=False)

        _apply(prog)
        assert get_op_types_in_program(prog).count("scaled_dot_product_attention") == 1

        rng = np.random.RandomState(0)
        q = rng.randn(heads, 1, embedding).astype(np.float32)
        k = rng.randn(heads, keys, embedding).astype(np.float32)
        v = rng.randn(heads, keys, embedding).astype(np.float32)
        mask = (np.arange(keys) < keys - 2).reshape(1, 1, 1, keys)

        expected = _reference_attention(q, k, v, mask.reshape(1, 1, keys), fill=float(FINITE_FILL))
        # Core ML has no boolean model input; it exposes the mask as fp32.
        got = _predict(prog, q=q, k=k, v=v, mask=mask.astype(np.float32))
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)

    def test_not_fused_when_a_transpose_permutes_real_batch_axes(self):
        """A unit query axis must not make a genuine batch permutation look free."""
        b0, b1, keys, embedding = 2, 3, 7, 4

        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(b0, b1, 1, embedding)),
            mb.TensorSpec(shape=(b0, b1, keys, embedding)),
            mb.TensorSpec(shape=(b1, b0, keys, embedding)),
        ])
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)      # [b0, b1, 1, S]
            swapped = mb.transpose(x=scores, perm=[1, 0, 2, 3])  # [b1, b0, 1, S]
            weights = mb.softmax(x=swapped, axis=-1)
            return mb.matmul(x=weights, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_not_fused_when_the_softmax_axis_merges_a_real_batch_axis(self):
        """``[B, 1, S] -> [1, B * S]`` reduces over more than the key axis."""
        batch, keys, embedding = 2, 7, 4

        @mb.program(opset_version=ct.target.iOS18, input_specs=[
            mb.TensorSpec(shape=(batch, 1, embedding)),
            mb.TensorSpec(shape=(batch, keys, embedding)),
            mb.TensorSpec(shape=(batch, keys, embedding)),
        ])
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)             # [B, 1, S]
            merged = mb.reshape(x=scores, shape=[1, 1, batch * keys])  # [1, 1, B * S]
            weights = mb.softmax(x=merged, axis=-1)
            split = mb.reshape(x=weights, shape=[batch, 1, keys])
            return mb.matmul(x=split, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0


class TestSafeSoftmaxWrapper:
    """`torch._safe_softmax` zeroes rows that are entirely -inf; peel that."""

    @staticmethod
    def _assert_peeled(prog):
        ops = get_op_types_in_program(prog)
        assert ops.count("scaled_dot_product_attention") == 1
        for op_type in ("softmax", "select", "reduce_max", "reduce_min", "equal", "not_equal"):
            assert op_type not in ops, f"{op_type} was left behind: {ops}"

    def test_wrapper_is_peeled_when_there_is_no_mask(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(scaled), a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        self._assert_peeled(prog)
        assert_model_is_valid(
            prog,
            {"q": (B, H, L, E), "k": (B, H, S, E), "v": (B, H, S, E)},
            minimum_deployment_target=ct.target.iOS18,
            backend=("mlprogram", "fp32"),
        )

    def test_wrapper_is_peeled_when_the_weights_are_the_true_branch(self):
        """`select(any(scores != -inf), softmax, 0.0)` -- the mirrored polarity."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            finite = mb.cast(x=mb.logical_not(x=mb.equal(x=scaled, y=NEG_INF)), dtype="int32")
            any_finite = mb.reduce_max(x=finite, axes=[3], keep_dims=True)
            cond = mb.cast(x=any_finite, dtype="bool")
            safe = mb.select(cond=cond, a=weights, b=np.float32(0.0))
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        self._assert_peeled(prog)

    def test_wrapper_is_peeled_when_expressed_as_a_min_reduction(self):
        """`select(all(scores == -inf), 0.0, softmax)`."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            masked_out = mb.cast(x=mb.equal(x=scaled, y=NEG_INF), dtype="int32")
            all_masked = mb.reduce_min(x=masked_out, axes=[3], keep_dims=True)
            cond = mb.cast(x=all_masked, dtype="bool")
            safe = mb.select(cond=cond, a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        self._assert_peeled(prog)

    def test_wrapper_with_inverted_polarity_is_not_fused(self):
        """`select(all(scores == -inf), softmax, 0.0)` keeps only the masked rows."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(scaled), a=weights, b=np.float32(0.0))
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0
        assert "softmax" in get_op_types_in_program(prog)

    def test_wrapper_reducing_with_the_wrong_quantifier_is_not_fused(self):
        """`any(scores == -inf)` is not `all(scores == -inf)`."""
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            masked_out = mb.cast(x=mb.equal(x=scaled, y=NEG_INF), dtype="int32")
            any_masked = mb.reduce_max(x=masked_out, axes=[3], keep_dims=True)
            cond = mb.cast(x=any_masked, dtype="bool")
            safe = mb.select(cond=cond, a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_wrapper_reducing_over_the_wrong_axis_is_not_fused(self):
        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            scaled = mb.mul(x=scores, y=np.float32(1.0 / math.sqrt(E)))
            weights = mb.softmax(x=scaled, axis=-1)
            finite = mb.cast(x=mb.logical_not(x=mb.equal(x=scaled, y=NEG_INF)), dtype="int32")
            any_finite = mb.reduce_max(x=finite, axes=[2], keep_dims=True)
            cond = mb.cast(x=any_finite, dtype="bool")
            safe = mb.select(cond=cond, a=weights, b=np.float32(0.0))
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_wrapper_is_peeled_for_a_finite_select_fill(self):
        """With a finite fill no row can be all -inf, so the wrapper is an identity."""
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            masked = mb.select(cond=mask, a=scores, b=TORCH_MIN_FILL)
            weights = mb.softmax(x=masked, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(masked), a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        self._assert_peeled(prog)

    def test_wrapper_is_kept_for_a_neg_inf_select_fill(self):
        """A -inf fill *can* produce an all -inf row, which the wrapper zeroes."""
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, 1, S), dtype=types.bool)],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, mask):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            masked = mb.select(cond=mask, a=scores, b=NEG_INF)
            weights = mb.softmax(x=masked, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(masked), a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_wrapper_is_peeled_for_a_constant_additive_mask(self):
        bias = np.arange(L * S, dtype=np.float32).reshape(1, 1, L, S)

        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            biased = mb.add(x=scores, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(biased), a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        self._assert_peeled(prog)

    def test_wrapper_is_kept_for_a_constant_additive_mask_holding_neg_inf(self):
        bias = np.zeros((1, 1, L, S), dtype=np.float32)
        bias[0, 0, 0, :] = -np.inf

        @mb.program(input_specs=_qkv_specs(), opset_version=ct.target.iOS18)
        def prog(q, k, v):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            biased = mb.add(x=scores, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(biased), a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0

    def test_wrapper_is_kept_for_a_runtime_additive_mask(self):
        """A mask built at runtime (Whisper) may well contain an all -inf row."""
        @mb.program(
            input_specs=[*_qkv_specs(), mb.TensorSpec(shape=(B, 1, L, S))],
            opset_version=ct.target.iOS18,
        )
        def prog(q, k, v, bias):
            scores = mb.matmul(x=q, y=k, transpose_y=True)
            biased = mb.add(x=scores, y=bias)
            weights = mb.softmax(x=biased, axis=-1)
            safe = mb.select(cond=_all_neg_inf_rows(biased), a=np.float32(0.0), b=weights)
            return mb.matmul(x=safe, y=v, transpose_y=False)

        _apply(prog)
        assert _sdpa_ops(prog) == 0


class TestRegroup:
    """`_regroup` models a reshape as a re-grouping of the score axes."""

    @staticmethod
    def _sizes(grouped):
        return [[atom.size for atom in group] for group in grouped]

    def test_trailing_size_one_dimensions_each_get_an_atom(self):
        """`[4, 1, 1] -> [1, 4, 1, 1]`: the last axis must stay identifiable."""
        atoms = [_Atom(4), _Atom(1), _Atom(1)]
        grouped = _regroup(atoms, [1, 4, 1, 1])
        assert self._sizes(grouped) == [[], [4], [1], [1]]
        assert grouped[-1][0] is atoms[-1]
        assert grouped[-2][0] is atoms[-2]

    def test_size_one_atoms_are_absorbed_when_they_are_spare(self):
        atoms = [_Atom(1), _Atom(4), _Atom(1)]
        assert self._sizes(_regroup(atoms, [4, 1])) == [[1, 4], [1]]
        assert self._sizes(_regroup([_Atom(1), _Atom(4), _Atom(1)], [4])) == [[1, 4, 1]]

    def test_a_unit_atom_lands_in_an_arbitrary_unit_dimension(self):
        """`[H, 1, S] -> [H, 1, 1, S]` cannot tell the two size-1 axes apart.

        The query atom ends up in the *second* of them; a following transpose may
        then carry it anywhere. Matching therefore compares layouts by their
        non-unit atoms only -- a size-1 axis never reorders elements.
        """
        atoms = [_Atom(3), _Atom(1), _Atom(7)]
        grouped = _regroup(atoms, [3, 1, 1, 7])
        assert self._sizes(grouped) == [[3], [], [1], [7]]
        assert grouped[2][0] is atoms[1]

    def test_axes_are_split_and_merged(self):
        atoms = [_Atom(2), _Atom(15)]
        assert self._sizes(_regroup(atoms, [6, 5])) == [[2, 3], [5]]
        assert _regroup([_Atom(2), _Atom(15)], [4, 8]) is None
        assert _regroup([_Atom(6)], [4]) is None


class TestFuseAttentionToSdpaEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    @staticmethod
    def _assert_fused(cml_model, expected: int = 1):
        ops = get_model_instruction_types(cml_model)
        assert ops.count("scaled_dot_product_attention") == expected
        assert "softmax" not in ops
        assert "matmul" not in ops
        return ops

    def test_plain_attention_with_bool_mask(self):
        def attention(q, k, v, mask):
            scores = jnp.einsum("bhld,bhsd->bhls", q, k) / jnp.sqrt(4.0)
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhls,bhsd->bhld", weights, v)

        cml_model = run_and_compare(attention, [
            jax.ShapeDtypeStruct((2, 3, 5, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 7), jnp.bool_),
        ])
        self._assert_fused(cml_model)

    def test_gemma_style_gqa_einsum_attention(self):
        def attention(q, k, v, mask):
            scores = jnp.einsum("btkgh,bskh->bkgts", q, k)
            scores = jnp.where(mask[:, None, None, :, :], scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bkgts,bskh->btkgh", weights, v)

        cml_model = run_and_compare(attention, [
            jax.ShapeDtypeStruct((1, 5, 2, 3, 4), jnp.float32),   # b t k g h
            jax.ShapeDtypeStruct((1, 7, 2, 4), jnp.float32),      # b s k h
            jax.ShapeDtypeStruct((1, 7, 2, 4), jnp.float32),      # b s k h
            jax.ShapeDtypeStruct((1, 5, 7), jnp.bool_),           # b t s
        ])
        self._assert_fused(cml_model)

    def test_decode_step_attention_with_unit_query_length(self):
        """One query row against the whole cache -- the autoregressive hot path."""
        def attention(q, k, v, valid):
            scores = jnp.einsum("bhld,bhsd->bhls", q, k) / jnp.sqrt(4.0)
            scores = jnp.where(valid[:, None, None, :], scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhls,bhsd->bhld", weights, v)

        cml_model = run_and_compare(attention, [
            jax.ShapeDtypeStruct((1, 3, 1, 4), jnp.float32),
            jax.ShapeDtypeStruct((1, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((1, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((1, 7), jnp.bool_),
        ])
        self._assert_fused(cml_model)

    def test_gqa_decode_step_attention(self):
        """A Gemma-style decode step: one query row, grouped KV heads."""
        heads, kv_heads, keys, embedding = 4, 2, 7, 4

        def attention(q, k, v, valid):
            k = jnp.repeat(k, heads // kv_heads, axis=2)
            v = jnp.repeat(v, heads // kv_heads, axis=2)
            scores = jnp.matmul(
                jnp.transpose(q, (0, 2, 1, 3)),
                jnp.swapaxes(jnp.transpose(k, (0, 2, 1, 3)), -2, -1),
            )
            scores = jnp.where(valid[None, None, None, :], scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            out = jnp.matmul(weights, jnp.transpose(v, (0, 2, 1, 3)))
            return jnp.transpose(out, (0, 2, 1, 3)).reshape(1, 1, heads * embedding)

        cml_model = run_and_compare(attention, [
            jax.ShapeDtypeStruct((1, 1, heads, embedding), jnp.float32),
            jax.ShapeDtypeStruct((1, keys, kv_heads, embedding), jnp.float32),
            jax.ShapeDtypeStruct((1, keys, kv_heads, embedding), jnp.float32),
            jax.ShapeDtypeStruct((keys,), jnp.bool_),
        ])
        self._assert_fused(cml_model)

    def test_attention_without_mask(self):
        def attention(q, k, v):
            scores = jnp.einsum("bhld,bhsd->bhls", q, k) / jnp.sqrt(4.0)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhls,bhsd->bhld", weights, v)

        cml_model = run_and_compare(attention, [
            jax.ShapeDtypeStruct((2, 3, 5, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
        ])
        ops = self._assert_fused(cml_model)
        assert ops == ["scaled_dot_product_attention"]

    def test_attention_with_additive_bias(self):
        def attention(q, k, v, bias):
            scores = jnp.einsum("bhld,bhsd->bhls", q, k) / jnp.sqrt(4.0) + bias
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhls,bhsd->bhld", weights, v)

        cml_model = run_and_compare(attention, [
            jax.ShapeDtypeStruct((2, 3, 5, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 1, 5, 7), jnp.float32),
        ])
        self._assert_fused(cml_model)

    def test_two_layer_attention_stack(self):
        def layer(x, k, v, mask):
            scores = jnp.einsum("bhld,bhsd->bhls", x, k) / jnp.sqrt(4.0)
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhls,bhsd->bhld", weights, v)

        def stack(x, k1, v1, k2, v2, mask):
            hidden = layer(x, k1, v1, mask)
            return layer(hidden, k2, v2, mask)

        cml_model = run_and_compare(stack, [
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 7, 4), jnp.float32),
            jax.ShapeDtypeStruct((2, 7), jnp.bool_),
        ])
        self._assert_fused(cml_model, expected=2)

    def test_fully_masked_row_stays_finite(self):
        """The additive mask keeps a fully masked row finite, as -1e9 did."""
        from tests.utils import run_and_compare_specific_input  # noqa: PLC0415

        def attention(q, k, v, mask):
            scores = jnp.einsum("bhld,bhsd->bhls", q, k) / jnp.sqrt(4.0)
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhls,bhsd->bhld", weights, v)

        key = jax.random.PRNGKey(0)
        q = jax.random.normal(key, (2, 3, 5, 4), jnp.float32)
        k = jax.random.normal(key, (2, 3, 7, 4), jnp.float32)
        v = jax.random.normal(key, (2, 3, 7, 4), jnp.float32)
        mask = jnp.array([[False] * 7, [True] * 7])

        cml_model = run_and_compare_specific_input(attention, [q, k, v, mask])
        self._assert_fused(cml_model)


def _f32(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.float32)


def _bool(*shape):
    return jax.ShapeDtypeStruct(shape, jnp.bool_)


class TestLibraryAttentionEndToEnd:
    """The attention the real libraries write, not just the spelling we picked.

    Every case goes through the converter and the full pass pipeline and has to
    come out as a single ``scaled_dot_product_attention``; ``run_and_compare``
    checks the numerics against JAX along the way.
    """

    @staticmethod
    def _assert_fused(cml_model, expected: int = 1):
        ops = get_model_instruction_types(cml_model)
        assert ops.count("scaled_dot_product_attention") == expected
        assert "softmax" not in ops
        return ops

    # -- jax.nn.dot_product_attention ---------------------------------------

    def test_jax_dot_product_attention(self):
        def attention(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, implementation="xla")

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4),
        ]))

    def test_jax_dot_product_attention_with_mask(self):
        def attention(q, k, v, mask):
            return jax.nn.dot_product_attention(q, k, v, mask=mask, implementation="xla")

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4), _bool(2, 1, 5, 7),
        ]))

    def test_jax_dot_product_attention_is_causal(self):
        def attention(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 5, 3, 4), _f32(2, 5, 3, 4),
        ]))

    def test_jax_dot_product_attention_grouped_query(self):
        """Six query heads over three key/value heads.

        ``_dot_product_attention_xla`` vmaps its core over the group axis, so the
        scores gain an extra axis that the layout tracking has to see through.
        """
        def attention(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, implementation="xla")

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 6, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4),
        ]))

    def test_jax_dot_product_attention_decode_step(self):
        def attention(q, k, v):
            return jax.nn.dot_product_attention(q, k, v, implementation="xla")

        self._assert_fused(run_and_compare(attention, [
            _f32(1, 1, 3, 4), _f32(1, 7, 3, 4), _f32(1, 7, 3, 4),
        ]))

    def test_jax_dot_product_attention_with_bias(self):
        def attention(q, k, v, bias):
            return jax.nn.dot_product_attention(q, k, v, bias=bias, implementation="xla")

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4), _f32(2, 3, 5, 7),
        ]))

    def test_jax_dot_product_attention_with_bias_and_mask(self):
        """``logits + bias`` and then ``where(mask, logits, big_neg)``: two masks."""
        def attention(q, k, v, bias, mask):
            return jax.nn.dot_product_attention(
                q, k, v, bias=bias, mask=mask, implementation="xla"
            )

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4),
            _f32(2, 3, 5, 7), _bool(2, 1, 5, 7),
        ]))

    def test_jax_dot_product_attention_with_bias_and_causal_mask(self):
        def attention(q, k, v, bias):
            return jax.nn.dot_product_attention(
                q, k, v, bias=bias, is_causal=True, implementation="xla"
            )

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 5, 3, 4), _f32(2, 5, 3, 4), _f32(2, 3, 5, 5),
        ]))

    def test_jax_dot_product_attention_grouped_query_with_bias_and_mask(self):
        def attention(q, k, v, bias, mask):
            return jax.nn.dot_product_attention(
                q, k, v, bias=bias, mask=mask, implementation="xla"
            )

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 6, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4),
            _f32(2, 6, 5, 7), _bool(2, 1, 5, 7),
        ]))

    # -- flax.nnx ------------------------------------------------------------

    @staticmethod
    def _flax_fn(layer, call):
        """A plain JAX function around ``layer``, with its state at trace level."""
        from flax import nnx  # noqa: PLC0415

        graphdef, state = nnx.split(layer)

        def fn(*args):
            merged = nnx.merge(graphdef, jax.tree.map(lambda leaf: leaf, state))
            return call(merged, *args)

        return fn

    def test_flax_multi_head_attention(self):
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=5, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
        )
        fn = self._flax_fn(layer, lambda m, q, k, v: m(q, k, v))
        self._assert_fused(run_and_compare(fn, [
            _f32(4, 3, 2, 5), _f32(4, 3, 2, 5), _f32(4, 3, 2, 5),
        ]))

    def test_flax_multi_head_attention_with_mask(self):
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=5, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
        )
        fn = self._flax_fn(layer, lambda m, q, k, v, mask: m(q, k, v, mask=mask))
        self._assert_fused(run_and_compare(fn, [
            _f32(2, 6, 5), _f32(2, 7, 5), _f32(2, 7, 5), _bool(2, 1, 6, 7),
        ]))

    def test_flax_multi_head_attention_is_causal(self):
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=5, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
        )
        fn = self._flax_fn(layer, lambda m, x: m(x, x, x, is_causal=True))
        self._assert_fused(run_and_compare(fn, [_f32(2, 6, 5)]))

    def test_flax_multi_head_attention_grouped_query(self):
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, num_kv_heads=2, in_features=8, qkv_features=16,
            decode=False, rngs=nnx.Rngs(0),
        )
        fn = self._flax_fn(layer, lambda m, q, k, v: m(q, k, v))
        self._assert_fused(run_and_compare(fn, [
            _f32(2, 6, 8), _f32(2, 7, 8), _f32(2, 7, 8),
        ]))

    def test_flax_multi_head_attention_single_query(self):
        """A ``(1, 1, D)`` query: the autoregressive decode shape."""
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=5, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
        )
        fn = self._flax_fn(layer, lambda m, q, k, v, mask: m(q, k, v, mask=mask))
        self._assert_fused(run_and_compare(fn, [
            _f32(1, 1, 5), _f32(1, 7, 5), _f32(1, 7, 5), _bool(1, 1, 1, 7),
        ]))

    def test_flax_multi_head_attention_decode_step(self):
        """``decode=True``: one token against the KV cache, masked by the cache index."""
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=5, qkv_features=16, decode=True, rngs=nnx.Rngs(0)
        )
        layer.init_cache((1, 1, 5))
        fn = self._flax_fn(layer, lambda m, x: m(x, x, x))
        self._assert_fused(run_and_compare(fn, [_f32(1, 1, 5)]))

    def test_flax_multi_head_attention_with_dropout_configured(self):
        """A non-zero ``dropout_rate`` takes flax's own attention implementation
        instead of ``jax.nn.dot_product_attention``: it folds the ``1/sqrt(E)``
        into the query and contracts with a different einsum."""
        from flax import nnx  # noqa: PLC0415

        layer = nnx.MultiHeadAttention(
            num_heads=4, in_features=5, qkv_features=16, decode=False,
            dropout_rate=0.1, deterministic=True, rngs=nnx.Rngs(0),
        )
        fn = self._flax_fn(layer, lambda m, q, k, v, mask: m(q, k, v, mask=mask))
        self._assert_fused(run_and_compare(fn, [
            _f32(2, 6, 5), _f32(2, 7, 5), _f32(2, 7, 5), _bool(2, 1, 6, 7),
        ]))

    def test_flax_dot_product_attention_with_bias_and_mask(self):
        from flax import nnx  # noqa: PLC0415

        def attention(q, k, v, bias, mask):
            return nnx.dot_product_attention(q, k, v, bias=bias, mask=mask)

        self._assert_fused(run_and_compare(attention, [
            _f32(2, 5, 3, 4), _f32(2, 7, 3, 4), _f32(2, 7, 3, 4),
            _f32(2, 3, 5, 7), _bool(2, 1, 5, 7),
        ]))

    # -- equinox -------------------------------------------------------------

    @staticmethod
    def _equinox_fn(model):
        import equinox as eqx  # noqa: PLC0415
        import equinox.internal as eqxi  # noqa: PLC0415

        return eqxi.finalise_fn(eqx.nn.inference_mode(model))

    def test_equinox_multihead_attention(self):
        import equinox as eqx  # noqa: PLC0415

        layer = eqx.nn.MultiheadAttention(
            num_heads=4, query_size=24, key=jax.random.PRNGKey(0)
        )
        model = self._equinox_fn(jax.vmap(lambda q, k, v: layer(q, k, v)))
        self._assert_fused(run_and_compare(model, [
            _f32(2, 6, 24), _f32(2, 7, 24), _f32(2, 7, 24),
        ]))

    def test_equinox_multihead_attention_with_mask(self):
        """Equinox masks with ``finfo(dtype).min``: a large but finite fill."""
        import equinox as eqx  # noqa: PLC0415

        layer = eqx.nn.MultiheadAttention(
            num_heads=4, query_size=24, key=jax.random.PRNGKey(0)
        )
        model = self._equinox_fn(
            jax.vmap(lambda q, k, v, mask: layer(q, k, v, mask=mask))
        )
        self._assert_fused(run_and_compare(model, [
            _f32(2, 6, 24), _f32(2, 7, 24), _f32(2, 7, 24), _bool(2, 4, 6, 7),
        ]))

    def test_equinox_multihead_attention_single_query(self):
        import equinox as eqx  # noqa: PLC0415

        layer = eqx.nn.MultiheadAttention(
            num_heads=4, query_size=24, key=jax.random.PRNGKey(0)
        )
        model = self._equinox_fn(jax.vmap(lambda q, k, v: layer(q, k, v)))
        self._assert_fused(run_and_compare(model, [
            _f32(1, 1, 24), _f32(1, 7, 24), _f32(1, 7, 24),
        ]))
