import coremltools as ct

DEFAULT_HLO_PIPELINE: ct.PassPipeline = ct.PassPipeline.DEFAULT


def register_optimizations():
    from .remove_noop_slice_update import remove_noop_slice_update
    custom_passes = [remove_noop_slice_update]

    for custom_pass in custom_passes:
        DEFAULT_HLO_PIPELINE.append_pass(f"common::{custom_pass.__name__}")
