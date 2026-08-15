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

# Cleanup passes. They run before the first `const_elimination` so that the
# broadcast `tile`s are gone before constants get folded (otherwise a tiled
# scalar constant is materialised at full size).
CLEANUP_PASSES: list[str] = [
    "common::remove_broadcast_tiles",
    "common::fuse_reduce_keep_dims",
    "common::remove_noop_slice_update",
]

# Fusion passes. They run just before `common::fuse_matmul_weight_bias`: by then
# constants are folded and small, the broadcast tiles are gone, and everything is
# still fp32 (before `add_fp16_cast`), while coremltools' `reduce_transposes` /
# `fuse_transpose_matmul` / DCE still run afterwards to clean up what we emit.
#
# Adding a fusion pass = adding a module in this package + one entry in this list
# (and the corresponding import above).
FUSION_PASSES: list[str] = [
    "common::replace_decomposed_softmax",
    "common::fuse_attention_to_sdpa",
    "common::fuse_logit_softcap",
    "common::fuse_gelu_erfc",
]

# The pass the CLEANUP group is inserted before (fallback: the front of the pipeline).
_CLEANUP_ANCHOR = "common::const_elimination"
# The pass the FUSION group is inserted before (fallback: the end of the pipeline).
_FUSION_ANCHOR = "common::fuse_matmul_weight_bias"


def _insert_passes(pipeline: ct.PassPipeline, pass_names: list[str], anchor: str, fallback_index: int | None) -> None:
    """Insert ``pass_names`` (in order) immediately before the first ``anchor`` pass.

    Passes already present in the pipeline are left where they are. If ``anchor``
    is not part of the pipeline, ``fallback_index`` is used instead (``None``
    meaning "append at the end").
    """
    missing = [name for name in pass_names if name not in pipeline.passes]
    if not missing:
        return

    if anchor in pipeline.passes:
        index = pipeline.passes.index(anchor)
    elif fallback_index is None:
        index = len(pipeline.passes)
    else:
        index = fallback_index

    for offset, pass_name in enumerate(missing):
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
