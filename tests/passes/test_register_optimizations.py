from stablehlo_coreml import register_optimizations
from stablehlo_coreml.passes.utils import DEFAULT_HLO_PIPELINE


def test_register_optimizations_is_idempotent():
    register_optimizations()
    register_optimizations()

    assert DEFAULT_HLO_PIPELINE.passes.count("common::remove_noop_slice_update") == 1
    assert DEFAULT_HLO_PIPELINE.passes.count("common::remove_noop_state_update") == 1
