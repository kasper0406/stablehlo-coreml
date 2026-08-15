import coremltools as ct
import pytest

from stablehlo_coreml import build_pass_pipeline, register_optimizations
from stablehlo_coreml.passes.utils import CLEANUP_PASSES, DEFAULT_HLO_PIPELINE, FUSION_PASSES

ALL_PASSES = CLEANUP_PASSES + FUSION_PASSES


@pytest.mark.parametrize("pass_name", ALL_PASSES)
def test_pass_is_registered(pass_name):
    from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY  # noqa: PLC0415
    assert pass_name in PASS_REGISTRY


@pytest.mark.parametrize("pass_name", ALL_PASSES)
def test_default_pipeline_contains_each_pass_exactly_once(pass_name):
    assert DEFAULT_HLO_PIPELINE.passes.count(pass_name) == 1


def test_cleanup_passes_run_before_first_const_elimination():
    passes = build_pass_pipeline().passes
    first_const_elimination = passes.index("common::const_elimination")
    for offset, pass_name in enumerate(CLEANUP_PASSES):
        assert passes.index(pass_name) == first_const_elimination - len(CLEANUP_PASSES) + offset


def test_fusion_passes_run_before_fuse_matmul_weight_bias():
    passes = build_pass_pipeline().passes
    anchor = passes.index("common::fuse_matmul_weight_bias")
    for offset, pass_name in enumerate(FUSION_PASSES):
        assert passes.index(pass_name) == anchor - len(FUSION_PASSES) + offset


def test_build_pass_pipeline_returns_distinct_objects():
    first = build_pass_pipeline()
    second = build_pass_pipeline()
    assert first is not second
    assert first.passes is not second.passes
    assert first.passes == second.passes

    first.append_pass("common::dead_code_elimination")
    assert first.passes != second.passes
    assert DEFAULT_HLO_PIPELINE.passes == second.passes


def test_build_pass_pipeline_does_not_mutate_the_base():
    base = ct.PassPipeline.DEFAULT
    original = list(base.passes)
    build_pass_pipeline(base)
    assert base.passes == original


def test_build_pass_pipeline_is_idempotent():
    once = build_pass_pipeline()
    twice = build_pass_pipeline(once)
    assert twice.passes == once.passes


def test_build_pass_pipeline_with_base_missing_the_anchors():
    """Without the anchors, cleanup passes go first and fusion passes last."""
    base = ct.PassPipeline.EMPTY
    base.passes = ["common::noop_elimination", "common::dead_code_elimination"]

    pipeline = build_pass_pipeline(base)
    assert pipeline.passes == (
        CLEANUP_PASSES + ["common::noop_elimination", "common::dead_code_elimination"] + FUSION_PASSES
    )


def test_build_pass_pipeline_preserves_pass_options():
    base = ct.PassPipeline.DEFAULT
    base.set_options(pass_name="common::const_elimination", options={"skip_const_by_size": "100000"})

    pipeline = build_pass_pipeline(base)
    assert "common::const_elimination" in pipeline._pass_options


def test_register_optimizations_is_idempotent():
    register_optimizations()
    register_optimizations()

    for pass_name in ALL_PASSES:
        assert DEFAULT_HLO_PIPELINE.passes.count(pass_name) == 1
