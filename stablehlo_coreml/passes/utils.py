import coremltools as ct

DEFAULT_HLO_PIPELINE: ct.PassPipeline = ct.PassPipeline.DEFAULT

_registered = False


def register_optimizations():
    global _registered
    if _registered:
        return
    _registered = True

    from .remove_noop_slice_update import remove_noop_slice_update
    from .quantize_const_weights import quantize_const_weights
    from .debug_rss import debug_rss

    # quantize_const_weights runs first so that coremltools' ~95 passes work
    # on a compressed ~4.5GB int8 model instead of a ~17GB fp32 model.
    DEFAULT_HLO_PIPELINE.insert_pass(0, "common::quantize_const_weights")

    # Insert debug_rss at every 5th pass so we can track memory per-pass.
    # Insert from the end backwards to keep indices stable.
    n = len(DEFAULT_HLO_PIPELINE.passes)
    for i in range(n, 0, -5):
        DEFAULT_HLO_PIPELINE.insert_pass(i, "common::debug_rss")

    DEFAULT_HLO_PIPELINE.append_pass(f"common::{remove_noop_slice_update.__name__}")
