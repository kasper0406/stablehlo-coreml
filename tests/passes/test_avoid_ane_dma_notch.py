import coremltools as ct
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol, types
from coremltools.converters.mil.testing_utils import get_op_types_in_program

from stablehlo_coreml import build_pass_pipeline, convert
from stablehlo_coreml.passes.avoid_ane_dma_notch import (
    MAX_SPLITS,
    NOTCH_BYTES,
    NUM_CORES,
    WINDOW_BYTES,
    choose_chunks,
    chunk_sizes,
    hits_notch,
    per_core_payload_bytes,
)
from tests.passes.helpers import apply_pass, count_ops, ops_of_type, predict
from tests.utils import export_hlo_module, get_model_instruction_types

PASS_NAME = "common::avoid_ane_dma_notch"

KIB = 1 << 10
MIB = 1 << 20

# The shape the erratum was measured on: a 1x1 convolution with 2048 output and
# 4096 input channels streams ceil(2048/16) * 4096 * 2 == exactly 1 MiB per core.
NOTCH_OUT, NOTCH_IN = 2048, 4096
# Same number of cores' worth of work, one line-window below the notch: 126
# units per core instead of 128, i.e. 1 MiB - 16 KiB, right at the edge of the
# window where throughput has recovered.
CONTROL_OUT = 2016

# Tolerance for comparing a converted program's fp16 output against an fp32
# reference. Every numeric test below runs a 4096-term reduction at the notch
# shape, whose outputs have magnitude ~5, where one fp16 ulp is 0.004. Measured
# over 12 seeds of each configuration on macOS 26.6 / coremltools 9.0, Core ML's
# CPU fp16 kernels land at most 0.074 (split program) and 0.097 (unsplit) from
# fp32; the regrouping this pass does is worth ~1 ulp of that, the rest is the
# kernels' own accumulation order. Anything the pass could get structurally
# wrong -- a slice with the wrong bounds, a dropped or repeated partial, a bias
# applied twice -- moves an output by a good fraction of its own magnitude, far
# past this.
FP16_REDUCTION_ATOL = 0.25


def _weight(*shape, dtype=np.float16):
    """A weight constant of the given shape. The values never matter structurally."""
    return np.zeros(shape, dtype=dtype)


def _apply(prog):
    apply_pass(prog, PASS_NAME)


def _conv_prog(weight, x_shape, *, bias=None, **conv_kwargs):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=x_shape, dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        kwargs = dict(conv_kwargs)
        if bias is not None:
            kwargs["bias"] = bias
        return mb.conv(x=x, weight=weight, **kwargs)

    return prog


class TestPayloadModel:
    """The arithmetic the pass decides on, without any MIL around it."""

    def test_the_measured_notch_shape_is_exactly_one_mib_per_core(self):
        assert per_core_payload_bytes(NOTCH_OUT, NOTCH_IN, 1) == MIB

    def test_the_control_shape_sits_at_the_edge_of_the_window(self):
        payload = per_core_payload_bytes(CONTROL_OUT, NOTCH_IN, 1)
        assert payload == MIB - WINDOW_BYTES
        assert not hits_notch(payload)

    def test_output_units_are_rounded_up_to_whole_cores(self):
        """A core that owns one extra unit streams the whole contracting dim for it."""
        assert per_core_payload_bytes(NUM_CORES + 1, 1024, 1) == 2 * 1024 * 2

    def test_kernel_elements_count_towards_the_payload(self):
        assert per_core_payload_bytes(16, 512, 32 * 32) == MIB

    @pytest.mark.parametrize("multiple", [1, 2, 6, 7])
    def test_exact_multiples_of_the_ring_size_hit_the_notch(self, multiple):
        assert hits_notch(multiple * NOTCH_BYTES)

    @pytest.mark.parametrize("offset", [-WINDOW_BYTES + 64, -64, 64, WINDOW_BYTES - 64])
    def test_payloads_inside_the_window_hit_the_notch(self, offset):
        assert hits_notch(2 * NOTCH_BYTES + offset)

    @pytest.mark.parametrize("offset", [-WINDOW_BYTES, WINDOW_BYTES, 3 * WINDOW_BYTES])
    def test_payloads_outside_the_window_do_not(self, offset):
        assert not hits_notch(2 * NOTCH_BYTES + offset)

    @pytest.mark.parametrize("payload", [0, 64 * KIB, MIB - WINDOW_BYTES - 64])
    def test_payloads_below_the_first_notch_never_hit_it(self, payload):
        """The ring has to wrap at least once before the prefetcher stalls."""
        assert not hits_notch(payload)

    def test_chunk_sizes_are_equal_when_they_divide(self):
        assert chunk_sizes(4096, 4) == [1024] * 4

    def test_chunk_sizes_spread_the_remainder_over_the_first_chunks(self):
        assert chunk_sizes(4096, 3) == [1366, 1365, 1365]
        assert sum(chunk_sizes(4099, 8)) == 4099

    @pytest.mark.parametrize(
        "laps, expected_chunks",
        [
            pytest.param(2, 3, id="2MiB"),
            pytest.param(3, 2, id="3MiB"),
            pytest.param(5, 2, id="5MiB"),
            pytest.param(6, 4, id="6MiB"),
            pytest.param(7, 2, id="7MiB"),
        ],
    )
    def test_split_counts_for_the_payloads_seen_in_real_models(self, laps, expected_chunks):
        """The affected projections listed in the blog post, as payload laps.

        The post splits 2 MiB four ways (4 x 0.5 MiB) because it only considered
        splits that divide the contracting dimension evenly; three near-equal
        chunks of ~0.67 MiB clear the notch just as well and cost one partial
        less, so that is what this pass picks.
        """
        # 1024 input channels per lap, so that `laps` laps is `laps` MiB per core.
        contracting, kernel = 1024 * laps, 512
        assert per_core_payload_bytes(NUM_CORES, contracting, kernel) == laps * MIB
        chunks = choose_chunks(NUM_CORES, contracting, kernel)
        assert len(chunks) == expected_chunks
        assert sum(chunks) == contracting
        assert all(
            not hits_notch(per_core_payload_bytes(NUM_CORES, chunk, kernel)) for chunk in chunks
        )

    def test_the_two_mib_chunks_clear_the_notch_without_being_equidistant(self):
        """Why three near-equal chunks are as safe as the blog's four equal ones.

        Not because they are equally far from a multiple of 1 MiB -- they are
        not, 0.33 MiB against 0.5 MiB -- but because both land inside the very
        first lap of the prefetch ring, below the first notch altogether.
        """
        out_units = 4096  # 4096 x 4096, the 2 MiB per core shape
        assert per_core_payload_bytes(out_units, NOTCH_IN, 1) == 2 * MIB

        chunks = choose_chunks(out_units, NOTCH_IN, 1)
        assert chunks == [1366, 1365, 1365]
        payloads = [per_core_payload_bytes(out_units, chunk, 1) for chunk in chunks]
        blog_payload = per_core_payload_bytes(out_units, NOTCH_IN // 4, 1)
        assert blog_payload == MIB // 2

        def distance_to_a_multiple(payload):
            remainder = payload % NOTCH_BYTES
            return min(remainder, NOTCH_BYTES - remainder)

        # Closer to 1 MiB than the blog's 0.5 MiB chunks, and still clear: below
        # the first notch, so `hits_notch` says the ring never wraps.
        assert all(distance_to_a_multiple(p) < distance_to_a_multiple(blog_payload)
                   for p in payloads)
        assert all(p < NOTCH_BYTES - WINDOW_BYTES for p in payloads)
        assert not any(hits_notch(p) for p in payloads)

    def test_no_split_is_reported_when_every_chunk_count_stays_in_the_notch(self):
        """A contracting dimension of 1 cannot be split at all."""
        assert choose_chunks(NUM_CORES * MIB // 2, 1, 1) is None

    def test_splits_never_exceed_the_bound(self):
        chunks = choose_chunks(NUM_CORES, 1024 * 6, 512)
        assert 2 <= len(chunks) <= MAX_SPLITS


class TestAvoidAneDmaNotch:
    """Unit tests on hand-built MIL programs."""

    def test_conv_1x1_in_the_notch_is_split(self):
        prog = _conv_prog(_weight(NOTCH_OUT, NOTCH_IN, 1, 1), (1, NOTCH_IN, 1, 1))
        _apply(prog)
        assert get_op_types_in_program(prog) == [
            "slice_by_index", "conv", "slice_by_index", "conv", "add",
        ]

        convs = ops_of_type(prog, "conv")
        assert [conv.weight.shape for conv in convs] == [
            (NOTCH_OUT, NOTCH_IN // 2, 1, 1),
            (NOTCH_OUT, NOTCH_IN // 2, 1, 1),
        ]
        assert [conv.x.shape for conv in convs] == [
            (1, NOTCH_IN // 2, 1, 1),
            (1, NOTCH_IN // 2, 1, 1),
        ]

    def test_the_partials_slice_consecutive_chunks_of_the_weight(self):
        weight = np.arange(NOTCH_OUT * NOTCH_IN, dtype=np.float16).reshape(
            NOTCH_OUT, NOTCH_IN, 1, 1
        )
        prog = _conv_prog(weight, (1, NOTCH_IN, 1, 1))
        _apply(prog)

        first, second = ops_of_type(prog, "conv")
        np.testing.assert_array_equal(first.weight.val, weight[:, : NOTCH_IN // 2])
        np.testing.assert_array_equal(second.weight.val, weight[:, NOTCH_IN // 2:])

    def test_the_slices_cover_the_contracting_axis_and_nothing_else(self):
        prog = _conv_prog(_weight(NOTCH_OUT, NOTCH_IN, 1, 1), (1, NOTCH_IN, 1, 1))
        _apply(prog)

        first, second = ops_of_type(prog, "slice_by_index")
        assert list(first.begin.val) == [0, 0, 0, 0]
        assert list(first.end.val) == [0, NOTCH_IN // 2, 0, 0]
        assert list(second.begin.val) == [0, NOTCH_IN // 2, 0, 0]
        assert list(second.end.val) == [0, NOTCH_IN, 0, 0]
        # Every axis but the contracting one is left to the masks, so that a
        # symbolic size never has to be spelled out.
        for op in (first, second):
            assert list(op.begin_mask.val) == [True, False, True, True]
            assert list(op.end_mask.val) == [True, False, True, True]

    def test_control_shape_is_left_alone(self):
        prog = _conv_prog(_weight(CONTROL_OUT, NOTCH_IN, 1, 1), (1, NOTCH_IN, 1, 1))
        _apply(prog)
        assert get_op_types_in_program(prog) == ["conv"]

    def test_payload_just_inside_the_window_is_split(self):
        """1 MiB - 8 KiB per core: still on the slope of the V, so still split."""
        contracting = NOTCH_IN - 32
        assert per_core_payload_bytes(NOTCH_OUT, contracting, 1) == MIB - 8 * KIB

        prog = _conv_prog(_weight(NOTCH_OUT, contracting, 1, 1), (1, contracting, 1, 1))
        _apply(prog)
        assert count_ops(prog, "conv") == 2

    def test_bias_is_applied_once(self):
        bias = np.arange(NOTCH_OUT, dtype=np.float16)
        prog = _conv_prog(_weight(NOTCH_OUT, NOTCH_IN, 1, 1), (1, NOTCH_IN, 1, 1), bias=bias)
        _apply(prog)

        first, second = ops_of_type(prog, "conv")
        np.testing.assert_array_equal(first.bias.val, bias)
        assert second.bias is None

    def test_conv_attributes_are_preserved(self):
        # 3x3 over 512 Ki input channels is 9 MiB per core, a notch multiple.
        # A single output channel keeps the constant itself at 9 MiB, where a
        # [2048, 4096, 3, 3] weight with the same per-core payload would be
        # 144 MiB and the pass would then copy it twice.
        contracting = 512 * KIB
        assert per_core_payload_bytes(1, contracting, 9) == 9 * MIB
        prog = _conv_prog(
            _weight(1, contracting, 3, 3),
            (1, contracting, 8, 8),
            strides=[2, 2],
            pad_type="custom",
            pad=[1, 2, 3, 4],
            dilations=[2, 3],
        )
        _apply(prog)

        for conv in ops_of_type(prog, "conv"):
            assert list(conv.strides.val) == [2, 2]
            assert conv.pad_type.val == "custom"
            assert list(conv.pad.val) == [1, 2, 3, 4]
            assert list(conv.dilations.val) == [2, 3]
            assert conv.groups.val == 1
        assert count_ops(prog, "conv") == 2

    def test_symbolic_batch_is_split(self):
        batch = get_new_symbol()
        prog = _conv_prog(_weight(NOTCH_OUT, NOTCH_IN, 1, 1), (batch, NOTCH_IN, 1, 1))
        _apply(prog)
        assert count_ops(prog, "conv") == 2
        assert prog.functions["main"].outputs[0].shape == (batch, NOTCH_OUT, 1, 1)

    def test_two_mib_payload_needs_more_than_two_chunks(self):
        """Halving a 2 MiB payload lands on 1 MiB, which is the notch again."""
        weight = _weight(1, 1024, 32, 32)  # 2 MiB per core, on a single core
        prog = _conv_prog(weight, (1, 1024, 32, 32))
        assert per_core_payload_bytes(1, 1024, 32 * 32) == 2 * MIB
        _apply(prog)
        assert count_ops(prog, "conv") == 3
        assert count_ops(prog, "add") == 2
        assert [conv.weight.shape[1] for conv in ops_of_type(prog, "conv")] == [342, 341, 341]

    @pytest.mark.parametrize(
        "weight_shape, x_shape, splits",
        [
            pytest.param((NOTCH_OUT, NOTCH_IN, 1, 1), (1, NOTCH_IN, 1, 1), 2, id="2-way"),
            pytest.param((1, 1024, 32, 32), (1, 1024, 32, 32), 3, id="3-way"),
        ],
    )
    def test_a_k_way_split_costs_k_slices_and_k_minus_one_adds(
        self, weight_shape, x_shape, splits
    ):
        """Every partial gets its own slice, the first one included."""
        prog = _conv_prog(_weight(*weight_shape), x_shape)
        _apply(prog)
        assert count_ops(prog, "conv") == splits
        assert count_ops(prog, "slice_by_index") == splits
        assert count_ops(prog, "add") == splits - 1

    def test_grouped_conv_is_skipped(self):
        prog = _conv_prog(
            _weight(NOTCH_OUT, NOTCH_IN // 2, 1, 1), (1, NOTCH_IN, 1, 1), groups=2
        )
        _apply(prog)
        assert get_op_types_in_program(prog) == ["conv"]

    def test_fp32_weight_is_skipped(self):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, NOTCH_IN, 1, 1), dtype=types.fp32)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.conv(x=x, weight=_weight(NOTCH_OUT, NOTCH_IN, 1, 1, dtype=np.float32))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["conv"]

    def test_runtime_weight_is_skipped(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, NOTCH_IN, 1, 1), dtype=types.fp16),
                mb.TensorSpec(shape=(NOTCH_OUT, NOTCH_IN, 1, 1), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, weight):
            return mb.conv(x=x, weight=weight)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["conv"]

    def test_constexpr_weight_is_skipped(self):
        """A palettized weight is decompressed on the fly; its DMA payload is not
        the number of fp16 elements the pass would count."""
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, NOTCH_IN, 1, 1), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            weight = mb.constexpr_blockwise_shift_scale(
                data=np.zeros((NOTCH_OUT, NOTCH_IN, 1, 1), dtype=np.int8),
                scale=np.ones((NOTCH_OUT, 1, 1, 1), dtype=np.float16),
            )
            return mb.conv(x=x, weight=weight)

        _apply(prog)
        assert count_ops(prog, "conv") == 1
        assert count_ops(prog, "slice_by_index") == 0

    def test_weight_with_other_consumers_is_still_split(self):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(1, NOTCH_IN, 1, 1), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            weight = mb.const(val=_weight(NOTCH_OUT, NOTCH_IN, 1, 1))
            return mb.conv(x=x, weight=weight), mb.mul(x=weight, y=np.float16(2.0))

        _apply(prog)
        assert count_ops(prog, "conv") == 2
        # The shared constant stays: the `mul` still reads it.
        assert count_ops(prog, "mul") == 1

    @pytest.mark.parametrize("with_bias", [True, False], ids=["with_bias", "without_bias"])
    def test_linear_is_split(self, with_bias):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(3, NOTCH_IN), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            kwargs = {"bias": np.arange(NOTCH_OUT, dtype=np.float16)} if with_bias else {}
            return mb.linear(x=x, weight=_weight(NOTCH_OUT, NOTCH_IN), **kwargs)

        _apply(prog)
        assert get_op_types_in_program(prog) == [
            "slice_by_index", "linear", "slice_by_index", "linear", "add",
        ]
        first, second = ops_of_type(prog, "linear")
        assert first.weight.shape == (NOTCH_OUT, NOTCH_IN // 2)
        # `linear` always materialises a bias (zeros when the caller gave none),
        # so "applied once" means the real one sits on the first partial only.
        zeros = np.zeros(NOTCH_OUT, dtype=np.float16)
        expected = np.arange(NOTCH_OUT, dtype=np.float16) if with_bias else zeros
        np.testing.assert_array_equal(first.bias.val, expected)
        np.testing.assert_array_equal(second.bias.val, zeros)

    @pytest.mark.parametrize("transpose_y", [True, False])
    def test_matmul_is_split(self, transpose_y):
        weight_shape = (NOTCH_OUT, NOTCH_IN) if transpose_y else (NOTCH_IN, NOTCH_OUT)
        weight = np.arange(NOTCH_OUT * NOTCH_IN, dtype=np.float16).reshape(weight_shape)

        @mb.program(
            input_specs=[mb.TensorSpec(shape=(3, NOTCH_IN), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.matmul(x=x, y=weight, transpose_y=transpose_y)

        _apply(prog)
        assert count_ops(prog, "matmul") == 2

        half = NOTCH_IN // 2
        first, second = ops_of_type(prog, "matmul")
        assert bool(first.transpose_y.val) == transpose_y
        if transpose_y:
            np.testing.assert_array_equal(first.y.val, weight[:, :half])
            np.testing.assert_array_equal(second.y.val, weight[:, half:])
        else:
            np.testing.assert_array_equal(first.y.val, weight[:half])
            np.testing.assert_array_equal(second.y.val, weight[half:])

    def test_matmul_with_transposed_x_is_skipped(self):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(NOTCH_IN, 3), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.matmul(
                x=x, y=_weight(NOTCH_OUT, NOTCH_IN), transpose_x=True, transpose_y=True
            )

        _apply(prog)
        assert get_op_types_in_program(prog) == ["matmul"]

    def test_matmul_with_a_batched_weight_is_skipped(self):
        """A rank-3 constant `y` is a batch of products, not one weight matrix."""
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(2, 3, NOTCH_IN), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.matmul(x=x, y=_weight(2, NOTCH_IN, NOTCH_OUT))

        _apply(prog)
        assert get_op_types_in_program(prog) == ["matmul"]

    def test_matmul_with_a_constant_x_is_skipped(self):
        """Only the `y` operand is treated as the streamed weight."""
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(NOTCH_IN, 3), dtype=types.fp16)],
            opset_version=ct.target.iOS18,
        )
        def prog(x):
            return mb.matmul(x=_weight(NOTCH_OUT, NOTCH_IN), y=x)

        _apply(prog)
        assert get_op_types_in_program(prog) == ["matmul"]

    def test_split_inside_a_nested_block(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(1, NOTCH_IN, 1, 1), dtype=types.fp16),
                mb.TensorSpec(shape=(1,), dtype=types.bool),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, pred):
            def true_fn():
                return mb.conv(x=x, weight=_weight(NOTCH_OUT, NOTCH_IN, 1, 1))

            def false_fn():
                return mb.conv(x=x, weight=_weight(NOTCH_OUT, NOTCH_IN, 1, 1) + 1)

            return mb.cond(pred=mb.squeeze(x=pred), _true_fn=true_fn, _false_fn=false_fn)

        _apply(prog)
        assert count_ops(prog, "conv", recurse=True) == 4
        assert count_ops(prog, "add", recurse=True) == 2

    def test_is_idempotent(self):
        prog = _conv_prog(_weight(NOTCH_OUT, NOTCH_IN, 1, 1), (1, NOTCH_IN, 1, 1))
        _apply(prog)
        after_first = get_op_types_in_program(prog)
        _apply(prog)
        assert get_op_types_in_program(prog) == after_first


class TestAvoidAneDmaNotchNumerics:
    """The split has to compute the same thing, to within fp16 accuracy.

    The comparison is against an fp32 numpy reference rather than against the
    unsplit program. Both programs run Core ML's CPU fp16 kernels, and the
    order those kernels accumulate a 4096-term reduction in costs far more
    accuracy than the regrouping this pass does: measured on macOS 26.6 /
    coremltools 9.0, regrouping the reduction into two partials that are each
    accumulated in fp32, rounded to fp16 and added moves the result by ~1 ulp,
    while the two programs land 0.04-0.10 apart -- and with
    ``transpose_y=False`` it is the *unsplit* result that is the further of the
    two from fp32. So a split-vs-unsplit bound would be a bound on the CPU
    kernels, which a Core ML update can change without any defect here.
    Bounding each program against fp32 is a real accuracy bound instead.
    """

    def _check(self, build, x, reference, op_type, partials=2):
        """Both the unsplit and the split program have to match ``reference``."""
        np.testing.assert_allclose(predict(build(), x=x), reference, atol=FP16_REDUCTION_ATOL, rtol=0)

        prog = build()
        _apply(prog)
        assert count_ops(prog, op_type) == partials
        np.testing.assert_allclose(predict(prog, x=x), reference, atol=FP16_REDUCTION_ATOL, rtol=0)

    def test_conv_partials_sum_to_the_fp32_reference(self):
        rng = np.random.default_rng(0)
        weight = (rng.standard_normal((NOTCH_OUT, NOTCH_IN, 1, 1)) * 0.02).astype(np.float16)
        bias = (rng.standard_normal(NOTCH_OUT) * 0.1).astype(np.float16)
        x = rng.standard_normal((1, NOTCH_IN, 1, 1)).astype(np.float16)

        reference = (
            weight[:, :, 0, 0].astype(np.float32) @ x[0, :, 0, 0].astype(np.float32)
            + bias.astype(np.float32)
        ).reshape(1, NOTCH_OUT, 1, 1)

        self._check(lambda: _conv_prog(weight, x.shape, bias=bias), x, reference, "conv")

    @pytest.mark.parametrize("transpose_y", [True, False])
    def test_matmul_partials_sum_to_the_fp32_reference(self, transpose_y):
        """``transpose_y=False`` is the layout whose unsplit CPU kernel is the looser one."""
        rng = np.random.default_rng(1)
        shape = (NOTCH_OUT, NOTCH_IN) if transpose_y else (NOTCH_IN, NOTCH_OUT)
        weight = (rng.standard_normal(shape) * 0.02).astype(np.float16)
        x = rng.standard_normal((3, NOTCH_IN)).astype(np.float16)

        def build():
            @mb.program(
                input_specs=[mb.TensorSpec(shape=x.shape, dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )
            def prog(x):
                return mb.matmul(x=x, y=weight, transpose_y=transpose_y)

            return prog

        weight32 = weight.astype(np.float32)
        reference = x.astype(np.float32) @ (weight32.T if transpose_y else weight32)
        self._check(build, x, reference, "matmul")

    def test_linear_partials_sum_to_the_fp32_reference(self):
        """``linear`` carries a bias that exactly one partial may keep."""
        rng = np.random.default_rng(2)
        weight = (rng.standard_normal((NOTCH_OUT, NOTCH_IN)) * 0.02).astype(np.float16)
        bias = (rng.standard_normal(NOTCH_OUT) * 0.1).astype(np.float16)
        x = rng.standard_normal((3, NOTCH_IN)).astype(np.float16)

        def build():
            @mb.program(
                input_specs=[mb.TensorSpec(shape=x.shape, dtype=types.fp16)],
                opset_version=ct.target.iOS18,
            )
            def prog(x):
                return mb.linear(x=x, weight=weight, bias=bias)

            return prog

        reference = x.astype(np.float32) @ weight.astype(np.float32).T + bias.astype(np.float32)
        self._check(build, x, reference, "linear")

    def test_uneven_three_way_split_sums_to_the_fp32_reference(self):
        """A 2 MiB payload is split into chunks that do not divide the contracting dim.

        ``4096 -> 1366 + 1365 + 1365``: the partials cover different numbers of
        input channels, so an off-by-one in the running offset would show up
        here and nowhere else.
        """
        out_units = 2 * NOTCH_OUT
        assert per_core_payload_bytes(out_units, NOTCH_IN, 1) == 2 * MIB
        sizes = choose_chunks(out_units, NOTCH_IN, 1)
        assert sizes == [1366, 1365, 1365]

        rng = np.random.default_rng(3)
        weight = (rng.standard_normal((out_units, NOTCH_IN, 1, 1)) * 0.02).astype(np.float16)
        x = rng.standard_normal((1, NOTCH_IN, 1, 1)).astype(np.float16)

        reference = (
            weight[:, :, 0, 0].astype(np.float32) @ x[0, :, 0, 0].astype(np.float32)
        ).reshape(1, out_units, 1, 1)

        self._check(
            lambda: _conv_prog(weight, x.shape), x, reference, "conv", partials=len(sizes)
        )


def _convert_jax(jax_func, input_specs, *, avoid_ane_dma_notch):
    """Convert ``jax_func`` the way a user would, with the ANE group opt-in.

    Unlike ``tests.utils.run_and_compare`` this keeps ``common::add_fp16_cast``:
    the pass only matches fp16 constants, which is what the weights become once
    that pass and the ``const_elimination`` behind it have run.
    """
    hlo_module = export_hlo_module(jax_func, input_specs)
    mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
    return ct.convert(
        mil_program,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        pass_pipeline=build_pass_pipeline(avoid_ane_dma_notch=avoid_ane_dma_notch),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


def _predict(cml_model, value):
    name = cml_model.get_spec().description.input[0].name
    result = cml_model.predict({name: np.asarray(value)})
    return np.asarray(next(iter(result.values())))


class TestAvoidAneDmaNotchEndToEnd:
    """End-to-end tests going through the real converter + pipeline."""

    @staticmethod
    def _conv_fn(weight):
        def f(x):
            return jax.lax.conv_general_dilated(
                x, jnp.asarray(weight), (1, 1), "VALID",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
        return f

    def test_jax_1x1_conv_in_the_notch_is_split(self):
        rng = np.random.default_rng(0)
        weight = (rng.standard_normal((NOTCH_OUT, NOTCH_IN, 1, 1)) * 0.02).astype(np.float32)
        x = rng.standard_normal((1, NOTCH_IN, 1, 1)).astype(np.float32)
        f = self._conv_fn(weight)
        specs = [jax.ShapeDtypeStruct(x.shape, jnp.float32)]

        cml_model = _convert_jax(f, specs, avoid_ane_dma_notch=True)
        ops = get_model_instruction_types(cml_model)
        assert ops.count("conv") == 2
        assert ops.count("slice_by_index") == 2
        assert ops.count("add") == 1

        # fp16 weights and a 4096-long reduction, so this is an fp16 comparison.
        np.testing.assert_allclose(
            _predict(cml_model, x),
            np.asarray(f(jnp.asarray(x))),
            atol=FP16_REDUCTION_ATOL,
            rtol=0,
        )

    def test_jax_1x1_conv_is_untouched_without_the_flag(self):
        rng = np.random.default_rng(0)
        weight = (rng.standard_normal((NOTCH_OUT, NOTCH_IN, 1, 1)) * 0.02).astype(np.float32)
        specs = [jax.ShapeDtypeStruct((1, NOTCH_IN, 1, 1), jnp.float32)]

        cml_model = _convert_jax(
            self._conv_fn(weight), specs, avoid_ane_dma_notch=False
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("conv") == 1
        assert ops.count("slice_by_index") == 0

    def test_jax_matmul_in_the_notch_is_split(self):
        rng = np.random.default_rng(2)
        weight = (rng.standard_normal((NOTCH_IN, NOTCH_OUT)) * 0.02).astype(np.float32)
        x = rng.standard_normal((3, NOTCH_IN)).astype(np.float32)

        def f(value):
            return value @ jnp.asarray(weight)

        cml_model = _convert_jax(
            f, [jax.ShapeDtypeStruct(x.shape, jnp.float32)], avoid_ane_dma_notch=True
        )
        ops = get_model_instruction_types(cml_model)
        assert ops.count("matmul") == 2
        assert ops.count("slice_by_index") == 2
        assert ops.count("add") == 1

        np.testing.assert_allclose(
            _predict(cml_model, x),
            np.asarray(f(jnp.asarray(x))),
            atol=FP16_REDUCTION_ATOL,
            rtol=0,
        )
