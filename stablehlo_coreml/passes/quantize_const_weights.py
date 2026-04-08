"""
MIL pass: quantize large float weight constants to int8 using symmetric
per-channel quantization, replacing them with constexpr_blockwise_shift_scale
(iOS18) ops that are immune to constant folding.

This pass runs early in the pipeline so that coremltools' ~95 optimization
passes work on a compressed ~4.5GB model instead of a full ~17GB fp32 model,
which prevents OOM crashes on memory-constrained machines during ct.convert.
"""

from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil import Builder as mb

import numpy as np

# Minimum number of elements for a weight tensor to be quantized.
# Bias vectors (1D) and small positional buffers are left uncompressed.
_WEIGHT_THRESHOLD = 2048

# Module-level counter shared between _quantize_consts_in_block and apply().
# [count, total_bytes].  Reset by apply() before each run.
_counter: list = [0, 0]


def _should_quantize(op):
    """Return True if this const op should be compressed to int8."""
    if op.op_type != "const":
        return False
    val = op.outputs[0].val
    if not isinstance(val, np.ndarray):
        return False
    # Only compress multi-dimensional float tensors large enough to matter
    if val.dtype not in (np.float16, np.float32):
        return False
    if val.ndim < 2 or val.size < _WEIGHT_THRESHOLD:
        return False
    # Don't re-compress what is already feeding a constexpr_* op
    for child_op in op.outputs[0].child_ops:
        if child_op.op_type.startswith("constexpr_"):
            return False
    return True


def _quantize_symmetric_per_channel(val: np.ndarray, axis: int = 0):
    """
    Symmetric per-channel int8 quantization (round-to-nearest).

    Returns:
        quantized_data: int8 array with same shape as val
        scale: float array with shape [..., 1, ...] (same rank, 1 in all dims
               except `axis`), dtype matching val

    The implied block_size for constexpr_blockwise_shift_scale is:
        block_size[i] = val.shape[i] / scale.shape[i]
    which equals val.shape[i] for all dims except axis (full-tensor blocks)
    and 1 for axis (per-channel).

    Note: we process along axis=0 in chunks to avoid a 2× fp32 peak.  For
    an embedding table that might be 1.2 GB fp16, a naive val.astype(fp32)
    produces a 2.4 GB temporary that, combined with all the other weights
    still loaded in RAM, tips a 16 GB machine into OOM.
    """
    reduce_axes = tuple(i for i in range(val.ndim) if i != axis)

    # Compute per-channel max without a full fp32 copy (abs/max are safe in fp16)
    axis_max_f32 = np.max(np.abs(val), axis=reduce_axes, keepdims=True).astype(np.float32)
    axis_max_f32 = np.where(axis_max_f32 == 0.0, 1.0, axis_max_f32)
    scale_f32 = axis_max_f32 / 127.0
    scale = scale_f32.astype(val.dtype)
    del axis_max_f32

    # Quantize channel-by-channel in chunks so the fp32 temporary per chunk is
    # O(chunk_size × inner_dims) rather than O(full tensor size).
    n_channels = val.shape[axis]
    _CHUNK = 2048  # channels per chunk — keeps fp32 peak ≤ a few hundred MB
    quantized = np.empty(val.shape, dtype=np.int8)
    for start in range(0, n_channels, _CHUNK):
        end = min(start + _CHUNK, n_channels)
        slc = tuple(
            slice(start, end) if i == axis else slice(None)
            for i in range(val.ndim)
        )
        chunk_f32 = val[slc].astype(np.float32)
        chunk_scale = scale_f32[slc]
        quantized[slc] = np.clip(
            np.round(chunk_f32 / chunk_scale), -127, 127
        ).astype(np.int8)
        del chunk_f32, chunk_scale
    del scale_f32
    return quantized, scale


@block_context_manager
def _quantize_consts_in_block(block):
    import gc as _gc

    # Phase 1: scan block.operations once to find const ops and recurse into
    # any sub-blocks.  We do NOT modify the block here, so it is safe to
    # iterate block.operations directly (no snapshot needed).
    #
    # Collecting only const ops avoids iterating tens of thousands of non-const
    # ops in the main quantize loop (Phase 2).  The previous all-ops snapshot
    # loop continued iterating non-const ops *after* the last const op, which
    # triggered Python's automatic GC at an unsafe callstack position.
    const_ops = []
    for op in block.operations:
        for b in op.blocks:
            _quantize_consts_in_block(b)
        if _should_quantize(op):
            const_ops.append(op)

    if not const_ops:
        return False

    # Phase 2: process only the const ops collected above, using slot-nulling
    # so that gc.collect() can break the op↔var cycles and release the C-level
    # fp16 numpy refs before the int8 data accumulates to OOM levels.
    #
    # gc.disable() prevents Python's automatic GC from firing at unpredictable
    # positions inside C extensions (e.g. replace_uses_of_var_after_op iterates
    # ~50,000 ops, creating hundreds of thousands of Python objects per tensor,
    # easily pushing gen0 past threshold 700 which would trigger auto-GC while
    # MLIR/jaxlib tp_finalize objects are on the C stack).
    #
    # Every 20 tensors we: (1) re-enable GC, (2) call gc.collect() at a safe
    # Python boundary, (3) call malloc_zone_pressure_relief() to evict the just-
    # freed fp16 pages from macOS's malloc large-allocation free list.  Without
    # madvise, freed 42 MB pages stay in the process jetsam footprint as
    # compressed pages — 20 tensors × 42 MB = ~840 MB freed per batch; over the
    # full 317-tensor run that's ~13 GB accumulating even though actual RSS is
    # only ~5 GB.  macOS jetsam kills at ~51 GB compressed footprint.
    import gc as _gc
    import ctypes as _ctypes_q, ctypes.util as _ctu_q
    try:
        _libc_q = _ctypes_q.CDLL(_ctu_q.find_library('c'))
        def _madvise_free():
            _libc_q.malloc_zone_pressure_relief(
                _ctypes_q.c_void_p(0), _ctypes_q.c_size_t(0))
    except Exception:
        def _madvise_free():
            pass

    _gc.disable()
    try:
        n = len(const_ops)
        for i in range(n):
            op = const_ops[i]
            const_ops[i] = None  # drop the list's ref so gc.collect() can collect

            val = op.outputs[0].val
            nbytes = val.nbytes
            quantized_data, scale = _quantize_symmetric_per_channel(val, axis=0)

            # Clear the Python-visible sym_val ref (belt-and-suspenders).
            # The C-level ref inside the op object is released only when the
            # op is garbage-collected after gc.collect() breaks the op↔var cycle.
            op.outputs[0]._sym_val = None
            del val

            new_var = mb.constexpr_blockwise_shift_scale(
                data=quantized_data,
                scale=scale,
                before_op=op,
                name=op.name + "_int8",
            )

            block.replace_uses_of_var_after_op(
                anchor_op=op,
                old_var=op.outputs[0],
                new_var=new_var,
                no_check_var_types=True,
            )
            block.remove_ops([op])

            _counter[0] += 1
            _counter[1] += nbytes

            del op  # drop local ref; only the op↔var cycle remains

            # Per-tensor flush: the write(2) syscall releases the GIL briefly,
            # giving background threads a chance to drop their MLIR/jaxlib cycle
            # refs via normal C++ refcounting before our explicit gc.collect().
            print(f"    t{_counter[0]}", flush=True)

            if _counter[0] % 20 == 0 or i == n - 1:
                print(
                    f"    quantized {_counter[0]} tensors  "
                    f"({_counter[1] / 1e9:.2f} GB fp16 → int8)",
                    flush=True,
                )
                # Collect at a safe Python boundary, then immediately re-disable.
                _gc.enable()
                _gc.collect()
                _gc.disable()
                # Evict the just-freed fp16 pages from macOS's large-allocation
                # free list.  Each batch frees ~840 MB of 42 MB numpy arrays;
                # this call marks those pages as MADV_FREE_REUSABLE so macOS
                # stops counting them in the process's jetsam phys_footprint.
                _madvise_free()
    finally:
        _gc.enable()

    return True


@register_pass(namespace="common")
class quantize_const_weights(AbstractGraphPass):
    """
    Replace large float weight constants with constexpr_blockwise_shift_scale
    ops using symmetric per-channel int8 quantization (iOS18).

    This is inserted at position 0 in the pass pipeline so that all of
    coremltools' subsequent optimization passes work on the compressed ~4.5GB
    int8 model rather than a ~17GB fp32 model. This prevents OOM crashes and
    write_fp16_data failures during ct.convert on memory-constrained machines.

    constexpr_blockwise_shift_scale (iOS18) is exempt from constant folding,
    so the int8 representation is preserved through to the final .mlpackage.
    """

    def apply(self, prog):
        _counter[0] = 0
        _counter[1] = 0
        for f in prog.functions.values():
            # Single pass is sufficient — all const ops in Gemma4 are at the
            # top level of the main function block (no nesting that would
            # require a second pass).  A while loop here would call
            # list(block.operations) a second time over thousands of ops
            # (including the newly-created constexpr ops), which can crash or
            # OOM before the pass summary is printed.
            _quantize_consts_in_block(f)
        if _counter[0]:
            print(
                f"    quantized {_counter[0]} tensors total  "
                f"({_counter[1] / 1e9:.2f} GB fp16 → int8)",
                flush=True,
            )
