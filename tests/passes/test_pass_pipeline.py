from collections import Counter

import coremltools as ct
import pytest

from stablehlo_coreml import build_pass_pipeline
from stablehlo_coreml.passes.utils import (
    CLEANUP_PASSES,
    DEFAULT_HLO_PIPELINE,
    FUSION_PASSES,
    LATE_FUSION_PASSES,
)

DCE_PASS_NAME = "common::dead_code_elimination"
# The groups interleave coremltools' DCE with our own passes, so the same name
# appears several times; only our own passes are expected in the pipeline exactly
# once.
ALL_PASSES = list(dict.fromkeys(CLEANUP_PASSES + FUSION_PASSES + LATE_FUSION_PASSES))
OUR_PASSES = [name for name in ALL_PASSES if name != DCE_PASS_NAME]
# Most of our passes run once, but a pass may belong to more than one group:
# `fuse_reduce_keep_dims` runs in the cleanup slot and again in the late-fusion
# slot, where `fuse_reduce_mean` has just made new reduce/reshape pairs visible.
EXPECTED_PASS_COUNTS = Counter(
    name for name in CLEANUP_PASSES + FUSION_PASSES + LATE_FUSION_PASSES if name != DCE_PASS_NAME
)


@pytest.mark.parametrize("pass_name", ALL_PASSES)
def test_pass_is_registered(pass_name):
    from coremltools.converters.mil.mil.passes.pass_pipeline import PASS_REGISTRY  # noqa: PLC0415
    assert pass_name in PASS_REGISTRY


@pytest.mark.parametrize("pass_name", OUR_PASSES)
def test_default_pipeline_contains_each_pass_once_per_group(pass_name):
    assert DEFAULT_HLO_PIPELINE.passes.count(pass_name) == EXPECTED_PASS_COUNTS[pass_name]


def test_fuse_reduce_keep_dims_runs_again_in_the_late_fusion_group():
    """The keep-dims reshape only becomes adjacent to its reduction once
    `fuse_reduce_mean` has folded `reduce_sum -> mul(1/N)` into one op."""
    passes = build_pass_pipeline().passes
    occurrences = [i for i, name in enumerate(passes) if name == "common::fuse_reduce_keep_dims"]
    assert len(occurrences) == 2
    assert occurrences[0] < passes.index("common::fuse_reduce_mean") < occurrences[1]
    assert occurrences[1] < passes.index("common::fuse_rmsnorm")


def test_cleanup_passes_run_before_first_const_elimination():
    passes = build_pass_pipeline().passes
    first_const_elimination = passes.index("common::const_elimination")
    assert passes[first_const_elimination - len(CLEANUP_PASSES):first_const_elimination] == CLEANUP_PASSES


def test_fusion_passes_run_before_fuse_matmul_weight_bias():
    passes = build_pass_pipeline().passes
    anchor = passes.index("common::fuse_matmul_weight_bias")
    assert passes[anchor - len(FUSION_PASSES):anchor] == FUSION_PASSES


def test_late_fusion_passes_run_after_fuse_reduce_mean():
    """`fuse_rmsnorm` needs the `reduce_mean` that coremltools' pass creates."""
    passes = build_pass_pipeline().passes
    anchor = passes.index("common::fuse_reduce_mean") + 1
    assert passes[anchor:anchor + len(LATE_FUSION_PASSES)] == LATE_FUSION_PASSES


def test_build_pass_pipeline_returns_distinct_objects():
    first = build_pass_pipeline()
    second = build_pass_pipeline()
    assert first is not second
    assert first.passes is not second.passes
    assert first.passes == second.passes

    first.append_pass(DCE_PASS_NAME)
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
    base.passes = ["common::noop_elimination", DCE_PASS_NAME]

    pipeline = build_pass_pipeline(base)
    assert pipeline.passes == (
        CLEANUP_PASSES
        + ["common::noop_elimination", DCE_PASS_NAME]
        + FUSION_PASSES
        + LATE_FUSION_PASSES
    )


def test_build_pass_pipeline_preserves_pass_options():
    base = ct.PassPipeline.DEFAULT
    base.set_options(pass_name="common::const_elimination", options={"skip_const_by_size": "100000"})

    pipeline = build_pass_pipeline(base)
    assert "common::const_elimination" in pipeline._pass_options


def test_group_is_inserted_even_when_the_base_already_has_one_of_its_passes():
    """Insertion is decided per group, not per pass.

    A pass may belong to two groups (`fuse_reduce_keep_dims`), so "already in the
    pipeline" cannot mean "skip it". The group is skipped only when it already
    occupies the slot next to its anchor -- which is what keeps
    `build_pass_pipeline` idempotent. A stray copy elsewhere in the base is
    therefore left where it is and the group is inserted anyway; our passes are
    idempotent, so the duplicate is harmless.
    """
    base = ct.PassPipeline.EMPTY
    base.passes = ["common::fuse_reduce_keep_dims", "common::const_elimination"]

    passes = build_pass_pipeline(base).passes
    assert passes[: len(CLEANUP_PASSES) + 1] == ["common::fuse_reduce_keep_dims"] + CLEANUP_PASSES
    assert passes.count("common::fuse_reduce_keep_dims") == 3
