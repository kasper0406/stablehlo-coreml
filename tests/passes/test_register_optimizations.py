from stablehlo_coreml import register_optimizations
from stablehlo_coreml.passes.utils import DEFAULT_HLO_PIPELINE


def test_register_optimizations_is_idempotent():
    register_optimizations()
    register_optimizations()

    pass_name = "common::remove_noop_slice_update"
    assert DEFAULT_HLO_PIPELINE.passes.count(pass_name) == 1
