"""MIL pass: split weight-streaming ops whose per-core weight payload is a multiple of 1 MiB.

The Apple Neural Engine has an RTL erratum in the kernel-DMA prefetch ring
(described in https://eiln.github.io/posts/ane-dma.html): the prefetcher walks a
ring of ``0x4000`` 64-byte lines, i.e. exactly 1 MiB, and whenever the number of
weight bytes a core has to stream is an integer multiple of that ring size, the
DMA engine stalls at the wrap-around. Measured DRAM throughput collapses from
~40-60 GB/s to ~17-25 GB/s, and the effect is 1 MiB-periodic: it recurs at every
multiple of 1 MiB, and recovers linearly over a window of 256 DMA lines (16 KiB)
on either side of the notch.

The payload model
-----------------
The ANE distributes the output units (the output channels of a ``conv``, the
rows of a ``linear``/``matmul`` weight) over its 16 cores, and each core streams
the *whole* contracting dimension for the units it owns::

    payload_bytes = ceil(N / NUM_CORES) * D * prod(kernel) * itemsize

with ``N`` the number of output units, ``D`` the contracting (input-channel)
dimension and ``itemsize == 2`` (the ANE only runs fp16). That is the quantity
the erratum is periodic in, which is why the notch shows up at shapes that look
harmless: a 2048x4096 fp16 1x1 convolution is ``ceil(2048/16) * 4096 * 2``
= exactly 1 MiB per core.

The workaround
--------------
A convolution (and a matrix product) is linear in its contracting dimension, so

    y = W @ x  ==  W[:, :d] @ x[:d] + W[:, d:] @ x[d:]

Splitting an affected op into ``k`` partial products whose per-core payloads are
*not* near a multiple of 1 MiB, and summing the partials, restores full
bandwidth. Measured on a Mac mini M4 (fp16 1x1 ``conv``, single token, medians of
3 rounds x 200 predicts):

===================  ==========  ==========  =====================
Weight (Cout x Cin)  Per core    Unsplit     Split
===================  ==========  ==========  =====================
2048 x 4096          1 MiB       840 us      430 us (2-way)
4096 x 4096          2 MiB       1142 us     697 us (4-way)
2048 x 8192          2 MiB       1350 us     698 us (4-way)
4096 x 14336         7 MiB       3522 us     1989 us (2-way)
2016 x 4096          0.98 MiB    426 us      426 us (control)
===================  ==========  ==========  =====================

The split column records the chunk count each row was measured with. This pass
picks the *fewest* chunks that clear the notch, so for a 2 MiB payload it emits
three near-equal chunks of ~0.67 MiB where the rows above used four equal ones
of 0.5 MiB. The two chunk sizes are not equally far from a multiple of 1 MiB
(0.33 MiB against 0.5 MiB), but both are far outside the 16 KiB window and both
stay inside the very first lap of the ring, where ``hits_notch`` says the stall
cannot happen -- and they measure the same: over three separate runs the 3-way
split took 684-730 us on 4096x4096 and 685-730 us on 2048x8192, against
685-698 us and 695-753 us for the hand-written 4-way one.

Why this pass is opt-in
-----------------------
The erratum is an ANE one. On the GPU there is no notch to avoid in the first
place (2048x4096 runs in 158 us against 156 us for the control shape), so the
extra slices and adds are pure overhead there, and the CPU shows no notch
either. The rewrite also adds ops (one slice
per partial and one add per extra partial, i.e. k slices and k-1 adds for a
k-way split) and changes fp16 rounding a little, because the accumulation is
now split into partial sums. So it only runs when the caller
asks for it with ``build_pass_pipeline(avoid_ane_dma_notch=True)``.

Scope
-----
Only leaf ops whose weight is a plain fp16 ``const`` with a static shape are
touched -- a ``constexpr_*`` (palettized/quantized) weight has a payload this
model does not describe, and a runtime weight has no compile-time size at all.
Grouped convolutions are skipped: their contracting dimension is per-group, so
the payload model above does not apply.
"""

import logging

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .pattern_utils import RewritePass

logger = logging.getLogger(__name__)

# Number of ANE cores the output units are spread over. The erratum is about
# what *one* core streams, so this is what turns a weight size into a payload.
NUM_CORES = 16

# Size of the kernel-DMA prefetch ring: 0x4000 lines of 64 bytes. Payloads that
# are an integer multiple of this wrap the ring exactly and hit the stall.
NOTCH_BYTES = 1 << 20  # 1 MiB

# Half-width of the measured V-notch: 256 DMA lines (one VM page worth of
# lines) on either side of a multiple of `NOTCH_BYTES`. Throughput recovers
# linearly across it, so a payload this far away is already out of trouble.
WINDOW_BYTES = 256 * 64  # 16 KiB

# Largest number of partial products the pass is willing to emit. Every extra
# partial costs a slice and an add; the blog's affected models all need <= 4.
MAX_SPLITS = 8

# The ANE runs fp16 only, so that is the only weight element size that has a
# payload in the sense above.
ITEMSIZE_BYTES = 2

# Ops the pass knows how to split, mapped to the input name holding the weight.
_WEIGHT_INPUT = {"conv": "weight", "linear": "weight", "matmul": "y"}


def per_core_payload_bytes(out_units: int, contracting: int, kernel_elems: int) -> int:
    """Bytes of fp16 weight a single ANE core streams for such an op.

    ``out_units`` (``N``) is spread over :data:`NUM_CORES`; each core streams the
    full contracting dimension (``contracting * kernel_elems`` values) for every
    unit it owns.
    """
    units_per_core = -(-out_units // NUM_CORES)
    return units_per_core * contracting * kernel_elems * ITEMSIZE_BYTES


def hits_notch(payload_bytes: int) -> bool:
    """Whether a per-core payload of ``payload_bytes`` falls in a DMA notch.

    True when the payload is within :data:`WINDOW_BYTES` of a multiple of
    :data:`NOTCH_BYTES`. Payloads below the first notch are never affected: the
    ring has to wrap at least once for the stall to happen, so a transfer that
    stays inside the very first lap is fine however close to 0 it is.
    """
    if payload_bytes < NOTCH_BYTES - WINDOW_BYTES:
        return False
    remainder = payload_bytes % NOTCH_BYTES
    distance = min(remainder, NOTCH_BYTES - remainder)
    return distance < WINDOW_BYTES


def chunk_sizes(contracting: int, splits: int) -> list[int]:
    """Split ``contracting`` into ``splits`` near-equal parts, biggest first."""
    base, remainder = divmod(contracting, splits)
    return [base + 1] * remainder + [base] * (splits - remainder)


def choose_chunks(out_units: int, contracting: int, kernel_elems: int) -> list[int] | None:
    """Chunk sizes along the contracting axis that take every partial out of the notch.

    Returns the split with the fewest chunks that works (fewer chunks means
    fewer slices and adds), or ``None`` when no split up to :data:`MAX_SPLITS`
    does -- in which case the caller leaves the op alone.

    The chunks need not divide ``contracting`` evenly, which is where this
    differs from the hand-written splits in the blog post: for a 2 MiB payload
    it picks 3 near-equal chunks of ~0.67 MiB, where the post (splitting only
    into equal powers of two) used 4 chunks of 0.5 MiB. The 0.67 MiB chunk is
    closer to a multiple of 1 MiB than the 0.5 MiB one is (0.33 MiB against
    0.5 MiB), but both stay inside the first lap of the ring, where
    :func:`hits_notch` says the stall cannot occur, and both measure the same on
    the ANE; three chunks cost one slice and one add less.
    """
    for splits in range(2, min(MAX_SPLITS, contracting) + 1):
        sizes = chunk_sizes(contracting, splits)
        if all(
            not hits_notch(per_core_payload_bytes(out_units, size, kernel_elems))
            for size in sizes
        ):
            return sizes
    return None


def _const_fp16_weight(var) -> np.ndarray | None:
    """``var``'s value if it is a plain fp16 ``const`` with a static shape.

    ``constexpr_*`` weights (palettized, quantized, sparse) are deliberately
    rejected: they are decompressed by the hardware, so the bytes that reach the
    DMA engine are not the ones this pass can count.
    """
    if var is None or var.dtype != types.fp16:
        return None
    op = getattr(var, "op", None)
    if op is None or op.op_type != "const":
        return None
    if var.shape is None or any(is_symbolic(dim) for dim in var.shape):
        return None
    val = var.val
    if val is None:
        return None
    return np.asarray(val)


def _static_dim(shape, axis: int) -> int | None:
    """``shape[axis]`` as an int, or ``None`` if unknown or symbolic."""
    if shape is None or axis >= len(shape):
        return None
    dim = shape[axis]
    if is_symbolic(dim):
        return None
    return int(dim)


def _match(op):
    """Describe how to split ``op``, or return ``None``.

    The result is ``(weight, x_axis, weight_axis, out_units, kernel_elems)``:
    the constant weight array, the contracting axis of ``x``, the axis of the
    weight holding the same dimension, the number of output units ``N``, and the
    number of weight elements per (output unit, input channel) pair.
    """
    weight_input = _WEIGHT_INPUT.get(op.op_type)
    if weight_input is None:
        return None

    weight = _const_fp16_weight(op.inputs.get(weight_input))
    if weight is None:
        return None

    x = op.inputs["x"]
    if x.shape is None:
        return None

    if op.op_type == "conv":
        # [C_out, C_in / groups, *kernel]; x is [N, C_in, *spatial].
        groups = op.inputs["groups"].val
        if groups is None or int(groups) != 1:
            # With groups > 1 a core only streams its own group's slice of the
            # weight, and the contracting dimension is per-group: a different
            # payload model, and splitting C_in would cross group boundaries.
            return None
        if weight.ndim < 3 or x.rank != weight.ndim:
            return None
        out_units = int(weight.shape[0])
        kernel_elems = int(np.prod(weight.shape[2:]))
        x_axis, weight_axis = 1, 1
    elif op.op_type == "linear":
        # [N, D]; x is [..., D].
        if weight.ndim != 2:
            return None
        out_units = int(weight.shape[0])
        kernel_elems = 1
        x_axis, weight_axis = x.rank - 1, 1
    else:  # matmul
        if op.inputs["transpose_x"].val:
            # Then the contracting axis of x is not its last one; rare enough
            # that v1 skips it rather than guessing.
            return None
        if weight.ndim != 2:
            return None
        transpose_y = bool(op.inputs["transpose_y"].val)
        # y is [N, D] when transposed, [D, N] otherwise.
        out_units = int(weight.shape[0] if transpose_y else weight.shape[1])
        kernel_elems = 1
        x_axis, weight_axis = x.rank - 1, 1 if transpose_y else 0

    if x_axis < 0:
        return None
    if _static_dim(x.shape, x_axis) != int(weight.shape[weight_axis]):
        # Either the contracting dimension of x is symbolic/unknown, or it does
        # not line up with the weight (a broadcasting matmul, say).
        return None

    return weight, x_axis, weight_axis, out_units, kernel_elems


def _slice_along(x, axis: int, begin: int, end: int, before_op):
    """``x[..., begin:end, ...]`` along ``axis``, leaving every other axis whole.

    Spelled with ``slice_by_index`` and begin/end masks rather than
    ``slice_by_size``: this is the form that stays on the ANE, and the masks keep
    the other axes symbolic-safe (their sizes are never mentioned).
    """
    rank = x.rank
    begin_indices, end_indices = [0] * rank, [0] * rank
    begin_indices[axis], end_indices[axis] = begin, end
    # Every other axis is covered by its mask, so its size is never spelled out
    # and may stay symbolic.
    begin_mask, end_mask = [True] * rank, [True] * rank
    begin_mask[axis] = end_mask[axis] = False
    return mb.slice_by_index(
        x=x,
        begin=begin_indices,
        end=end_indices,
        begin_mask=begin_mask,
        end_mask=end_mask,
        before_op=before_op,
    )


def _partial(op, x_chunk, weight_chunk, weight_input: str, keep_bias: bool):
    """Rebuild ``op`` on one chunk, keeping every other attribute as it was."""
    kwargs = dict(op.inputs)
    kwargs["x"] = x_chunk
    kwargs[weight_input] = weight_chunk
    if not keep_bias:
        # The bias is a constant added to the result, not part of the product,
        # so exactly one partial may carry it.
        kwargs.pop("bias", None)
    return getattr(mb, op.op_type)(**kwargs, before_op=op)


@register_pass(namespace="common")
class avoid_ane_dma_notch(RewritePass):
    """
    Split ``conv``/``linear``/``matmul`` ops whose per-core fp16 weight payload is
    a multiple of 1 MiB into partial products that are not, and sum them.

    Works around the Apple Neural Engine kernel-DMA prefetch erratum
    (https://eiln.github.io/posts/ane-dma.html), which throttles DRAM weight
    streaming from ~40-60 GB/s to ~17-25 GB/s whenever a core has to stream an
    exact multiple of the 1 MiB prefetch ring. See the module docstring for the
    payload model and the measured numbers.

    Only plain fp16 ``const`` weights with static shapes are split, and only
    along the contracting (input-channel) dimension, where the op is linear.
    The op is left alone when no split into at most ``MAX_SPLITS`` chunks takes
    every chunk out of the notch.

    Given (``x`` is ``[1, 4096, 1, 1]``, so each of the 16 cores streams exactly
    1 MiB of the ``[2048, 4096, 1, 1]`` weight):
        %y = conv(x=%x, weight=%w)

    Result:
        %x0 = slice_by_index(x=%x, begin=[0, 0, 0, 0], end=[0, 2048, 0, 0], ...)
        %x1 = slice_by_index(x=%x, begin=[0, 2048, 0, 0], end=[0, 4096, 0, 0], ...)
        %y0 = conv(x=%x0, weight=%w[:, :2048])
        %y1 = conv(x=%x1, weight=%w[:, 2048:])
        %y  = add(x=%y0, y=%y1)

    This pass is opt-in: it helps on the ANE only, and costs a slice per partial
    and an add per extra partial everywhere else. Enable it with
    ``build_pass_pipeline(avoid_ane_dma_notch=True)``.
    """

    _REWRITES = "weight-streaming op(s)"

    def visit(self, op, block) -> bool:
        match = _match(op)
        if match is None:
            return False
        weight, x_axis, weight_axis, out_units, kernel_elems = match

        contracting = int(weight.shape[weight_axis])
        payload = per_core_payload_bytes(out_units, contracting, kernel_elems)
        if not hits_notch(payload):
            return False

        sizes = choose_chunks(out_units, contracting, kernel_elems)
        if sizes is None:
            logger.debug(
                "%s: %s has a %d byte per-core payload in the DMA notch, but no split "
                "into at most %d chunks avoids it; leaving it alone",
                type(self).__name__, op.name, payload, MAX_SPLITS,
            )
            return False

        weight_input = _WEIGHT_INPUT[op.op_type]
        partials = []
        offset = 0
        for index, size in enumerate(sizes):
            chunk = [slice(None)] * weight.ndim
            chunk[weight_axis] = slice(offset, offset + size)
            partials.append(
                _partial(
                    op,
                    _slice_along(op.inputs["x"], x_axis, offset, offset + size, before_op=op),
                    np.ascontiguousarray(weight[tuple(chunk)]),
                    weight_input,
                    keep_bias=index == 0,
                )
            )
            offset += size

        # Sequentially, so that the partials are consumed in the order they are
        # produced; only the last add takes over the name of the op it replaces.
        total = partials[0]
        last = len(partials) - 1
        for index, partial in enumerate(partials[1:], start=1):
            named = {"name": op.outputs[0].name} if index == last else {}
            total = mb.add(x=total, y=partial, before_op=op, **named)

        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=total)
        return True
