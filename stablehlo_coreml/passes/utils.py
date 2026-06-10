import coremltools as ct

DEFAULT_HLO_PIPELINE: ct.PassPipeline = ct.PassPipeline.DEFAULT


def register_optimizations():
    from .remove_noop_slice_update import remove_noop_slice_update  # noqa: PLC0415
    custom_passes = [remove_noop_slice_update]

    for custom_pass in custom_passes:
        pass_name = f"common::{custom_pass.__name__}"
        if pass_name not in DEFAULT_HLO_PIPELINE.passes:
            DEFAULT_HLO_PIPELINE.append_pass(pass_name)
