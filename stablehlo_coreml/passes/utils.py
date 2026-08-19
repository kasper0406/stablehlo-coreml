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
from . import fuse_logit_softcap as _fuse_logit_softcap  # noqa: F401
from . import fuse_reduce_keep_dims as _fuse_reduce_keep_dims  # noqa: F401
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
]

# The pass the CLEANUP group is inserted before (fallback: the front of the pipeline).
_CLEANUP_ANCHOR = "common::const_elimination"
# The pass the FUSION group is inserted before (fallback: the end of the pipeline).
_FUSION_ANCHOR = "common::fuse_matmul_weight_bias"


def _insert_passes(pipeline: ct.PassPipeline, pass_names: list[str], anchor: str, fallback_index: int | None) -> None:
    """Insert ``pass_names`` (in order) immediately before the first ``anchor`` pass.

    Our own passes are only inserted when they are not in the pipeline yet, so
    re-inserting into a pipeline that already has them is a no-op. The
    ``dead_code_elimination`` entries are always inserted along with them:
    coremltools' default pipeline runs those elsewhere too, but we need them
    right between our own passes. If ``anchor`` is not part of the pipeline,
    ``fallback_index`` is used instead (``None`` meaning "append at the end").
    """
    to_insert = [name for name in pass_names if name == _DCE or name not in pipeline.passes]
    if all(name == _DCE for name in to_insert):
        return

    if anchor in pipeline.passes:
        index = pipeline.passes.index(anchor)
    elif fallback_index is None:
        index = len(pipeline.passes)
    else:
        index = fallback_index

    for offset, pass_name in enumerate(to_insert):
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

    return pipeline


DEFAULT_HLO_PIPELINE: ct.PassPipeline = build_pass_pipeline()


def register_optimizations() -> None:
    """Ensure all stablehlo-coreml passes are registered with coremltools.

    Kept for backwards compatibility; importing this module already registers
    them, so this function is a no-op that is safe to call repeatedly.
    """
