"""Pipeline integration for the stablehlo-coreml MIL graph passes.

Importing this module registers every pass in ``stablehlo_coreml.passes`` with
coremltools' ``PASS_REGISTRY``. Use :func:`build_pass_pipeline` to obtain a
pipeline with the passes inserted at the right places, or the pre-built
:data:`DEFAULT_HLO_PIPELINE` convenience object.
"""

import copy

import coremltools as ct

# Importing the pass modules registers them in coremltools' PASS_REGISTRY.
from . import fuse_attention_to_sdpa as _fuse_attention_to_sdpa  # noqa: F401
from . import fuse_gelu_erfc as _fuse_gelu_erfc  # noqa: F401
from . import fuse_gelu_tanh as _fuse_gelu_tanh  # noqa: F401
from . import fuse_logit_softcap as _fuse_logit_softcap  # noqa: F401
from . import fuse_reduce_keep_dims as _fuse_reduce_keep_dims  # noqa: F401
from . import fuse_rmsnorm as _fuse_rmsnorm  # noqa: F401
from . import remove_broadcast_tiles as _remove_broadcast_tiles  # noqa: F401
from . import remove_noop_slice_update as _remove_noop_slice_update  # noqa: F401
from . import replace_decomposed_softmax as _replace_decomposed_softmax  # noqa: F401

# The passes do not clean up after themselves; the ops they leave behind are
# removed by coremltools' dead code elimination, interleaved between them (one
# pass's leftovers would otherwise look like extra consumers to the next one).
_DCE = "common::dead_code_elimination"

# Cleanup passes. They run before the first `const_elimination` so that the
# broadcast `tile`s are gone before constants get folded (otherwise a tiled
# scalar constant is materialised at full size).
CLEANUP_PASSES: list[str] = [
    "common::remove_broadcast_tiles",
    "common::fuse_reduce_keep_dims",
    # `fuse_reduce_keep_dims` leaves the old reduction behind when the reshape
    # was its sole consumer.
    _DCE,
    "common::remove_noop_slice_update",
]

# Fusion passes. They run just before `common::fuse_matmul_weight_bias`: by then
# constants are folded and small, the broadcast tiles are gone, and everything is
# still fp32 (before `add_fp16_cast`), while coremltools' `reduce_transposes` /
# `fuse_transpose_matmul` / DCE still run afterwards to clean up what we emit.
#
# Adding a fusion pass = adding a module in this package + one entry in this list
# followed by a `_DCE` entry (and the corresponding import above).
FUSION_PASSES: list[str] = [
    "common::replace_decomposed_softmax",
    _DCE,
    "common::fuse_attention_to_sdpa",
    _DCE,
    "common::fuse_logit_softcap",
    _DCE,
    "common::fuse_gelu_erfc",
    _DCE,
    "common::fuse_gelu_tanh",
    _DCE,
]

# Late fusion passes. They run right *after* `common::fuse_reduce_mean`, because
# that is the pass that turns the converter's `reduce_sum -> mul(1/N)` into the
# `reduce_mean` they match (StableHLO has no mean instruction). That is still
# before `add_fp16_cast`, so the graph is fp32 there as well.
#
# `fuse_reduce_keep_dims` runs a *second* time here (it is also in
# `CLEANUP_PASSES`). A `jnp.mean(x, axis, keepdims=True)` lowers to
# `reduce_sum -> mul(1/N) -> reshape`, so in the cleanup slot the reduction is not
# yet adjacent to its keep-dims reshape and the pass cannot see the pair. Only
# after `common::fuse_reduce_mean` has folded `reduce_sum -> mul(1/N)` into a
# single `reduce_mean` does the reshape sit directly on the reduction -- which is
# exactly the position this group runs in.
LATE_FUSION_PASSES: list[str] = [
    "common::fuse_reduce_keep_dims",
    _DCE,
    "common::fuse_rmsnorm",
    _DCE,
]

# The pass the CLEANUP group is inserted before (fallback: the front of the pipeline).
_CLEANUP_ANCHOR = "common::const_elimination"
# The pass the FUSION group is inserted before (fallback: the end of the pipeline).
_FUSION_ANCHOR = "common::fuse_matmul_weight_bias"
# The pass the LATE_FUSION group is inserted after (fallback: the end of the pipeline).
_LATE_FUSION_ANCHOR = "common::fuse_reduce_mean"


def _insert_passes(
    pipeline: ct.PassPipeline,
    pass_names: list[str],
    anchor: str,
    fallback_index: int | None,
    after: bool = False,
) -> None:
    """Insert ``pass_names`` (in order) at the first ``anchor`` pass.

    The group goes immediately before ``anchor``, or immediately after it when
    ``after`` is set. The group is inserted as a whole, ``dead_code_elimination``
    entries included: coremltools' default pipeline runs those elsewhere too, but
    we need them right between our own passes. A pass may legitimately appear in
    more than one group (``fuse_reduce_keep_dims`` runs both in ``CLEANUP_PASSES``
    and in ``LATE_FUSION_PASSES``), so "already inserted" is decided per group --
    the group is skipped only when it already occupies the slot next to its
    anchor, which is what makes re-inserting into a pipeline that already has the
    groups a no-op. If ``anchor`` is not part of the pipeline, ``fallback_index``
    is used instead (``None`` meaning "append at the end").
    """
    if len(pass_names) == 0:
        return

    if anchor in pipeline.passes:
        index = pipeline.passes.index(anchor) + (1 if after else 0)
    elif fallback_index is None:
        index = len(pipeline.passes)
    else:
        index = fallback_index

    if after:
        occupied = pipeline.passes[index:index + len(pass_names)]
    else:
        occupied = pipeline.passes[max(index - len(pass_names), 0):index]
    if occupied == pass_names:
        return

    for offset, pass_name in enumerate(pass_names):
        pipeline.insert_pass(index=index + offset, pass_name=pass_name)


def build_pass_pipeline(base: ct.PassPipeline | None = None) -> ct.PassPipeline:
    """Return a new pipeline: ``base`` with the stablehlo-coreml passes inserted.

    ``base`` defaults to ``ct.PassPipeline.DEFAULT``. The input pipeline is never
    mutated, and inserting into a pipeline that already contains our passes is a
    no-op for those passes.
    """
    # `ct.PassPipeline.DEFAULT` already hands out a fresh object on every access.
    pipeline = ct.PassPipeline.DEFAULT if base is None else copy.deepcopy(base)

    _insert_passes(pipeline, CLEANUP_PASSES, _CLEANUP_ANCHOR, fallback_index=0)
    _insert_passes(pipeline, FUSION_PASSES, _FUSION_ANCHOR, fallback_index=None)
    _insert_passes(pipeline, LATE_FUSION_PASSES, _LATE_FUSION_ANCHOR, fallback_index=None, after=True)

    return pipeline


DEFAULT_HLO_PIPELINE: ct.PassPipeline = build_pass_pipeline()
