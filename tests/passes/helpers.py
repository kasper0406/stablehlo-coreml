"""Helpers shared by the MIL pass tests.

Every test in this package has the same shape: build a small MIL program by
hand, run one pass over it, and look at what is left. These are the pieces all
of them need.
"""

import coremltools as ct
import numpy as np
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipelineManager
from coremltools.converters.mil.testing_utils import apply_pass_and_basic_check

DCE_PASS_NAME = "common::dead_code_elimination"


def apply_pass(prog, pass_name, *, dce=True, skip_output_shape_check=False, **options):
    """Apply ``pass_name`` to ``prog`` in place, returning the program as it was before.

    Our passes leave the ops they matched behind, so dead code elimination runs
    afterwards unless ``dce=False``.

    ``options`` are pass options. Setting those requires running the pass from a
    pipeline, which is the only way coremltools lets them through; that path
    skips the basic checks and returns ``prog`` itself rather than a copy of the
    program as it was before.
    """
    pass_names = [pass_name, DCE_PASS_NAME] if dce else [pass_name]

    if options:
        pipeline = ct.PassPipeline(pass_names, "pass_under_test")
        pipeline.set_options(pass_name, options)
        PassPipelineManager.apply_pipeline(prog, pipeline)
        return prog

    prev_prog, _, _ = apply_pass_and_basic_check(
        prog, pass_name, skip_output_shape_check=skip_output_shape_check
    )
    if dce:
        apply_pass_and_basic_check(
            prog, DCE_PASS_NAME, skip_output_shape_check=skip_output_shape_check
        )
    return prev_prog


def ops_of_type(prog, op_type, fname="main", *, recurse=False):
    """The ``op_type`` ops of ``prog``'s ``fname`` function, in program order.

    ``recurse`` also descends into the blocks nested inside ops.
    """
    def collect(block):
        found = []
        for op in block.operations:
            if op.op_type == op_type:
                found.append(op)
            if recurse:
                for nested in op.blocks:
                    found += collect(nested)
        return found

    return collect(prog.functions[fname])


def count_ops(prog, op_type, fname="main", *, recurse=False):
    """How many ``op_type`` ops ``prog``'s ``fname`` function holds."""
    return len(ops_of_type(prog, op_type, fname, recurse=recurse))


def predict(prog, *, ct_inputs=None, **inputs):
    """Convert ``prog`` and run it on ``inputs``; returns the single output as an ndarray.

    Keeps fp32 throughout, so a comparison is not limited by fp16 accuracy, and
    runs coremltools' stock pipeline: the pass under test is applied by hand
    before this is called. ``ct_inputs`` are ``ct.TensorType``s, needed only when
    the program has symbolic dimensions to pin down.
    """
    model = ct.convert(
        prog,
        source="milinternal",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT32,
        pass_pipeline=ct.PassPipeline.DEFAULT,
        inputs=ct_inputs,
    )
    names = [feature.name for feature in model.get_spec().description.input]
    result = model.predict({name: inputs[name] for name in names})
    return np.array(next(iter(result.values())))
