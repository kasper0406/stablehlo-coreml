import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    get_op_types_in_program,
)

from stablehlo_coreml import register_optimizations

register_optimizations()


class TestRemoveNoopStateUpdate:

    def test_is_removed_direct(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(2, 3)),
                mb.StateTensorSpec(shape=(2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, s):
            r = mb.read_state(input=s)
            mb.coreml_update_state(state=s, value=r)
            return mb.add(x=x, y=1.0)

        assert "coreml_update_state" in get_op_types_in_program(prog)
        assert "read_state" in get_op_types_in_program(prog)

        apply_pass_and_basic_check(prog, "common::remove_noop_state_update")

        op_types = get_op_types_in_program(prog)
        assert "coreml_update_state" not in op_types
        assert "read_state" not in op_types
        assert "add" in op_types

    def test_is_removed_with_casts(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(2, 3)),
                mb.StateTensorSpec(shape=(2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, s):
            r = mb.read_state(input=s)
            c0 = mb.cast(x=r, dtype="fp32")
            c1 = mb.cast(x=c0, dtype="fp16")
            mb.coreml_update_state(state=s, value=c1)
            return mb.add(x=x, y=1.0)

        assert "coreml_update_state" in get_op_types_in_program(prog)
        assert "read_state" in get_op_types_in_program(prog)
        assert "cast" in get_op_types_in_program(prog)

        apply_pass_and_basic_check(prog, "common::remove_noop_state_update")

        op_types = get_op_types_in_program(prog)
        assert "coreml_update_state" not in op_types
        assert "read_state" not in op_types
        assert "cast" not in op_types
        assert "add" in op_types

    def test_not_removed_when_state_is_modified(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(2, 3)),
                mb.StateTensorSpec(shape=(2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, s):
            r = mb.read_state(input=s)
            c0 = mb.cast(x=r, dtype="fp32")
            new_val = mb.add(x=c0, y=1.0)
            c1 = mb.cast(x=new_val, dtype="fp16")
            mb.coreml_update_state(state=s, value=c1)
            return mb.add(x=x, y=1.0)

        assert "coreml_update_state" in get_op_types_in_program(prog)

        apply_pass_and_basic_check(prog, "common::remove_noop_state_update")

        op_types = get_op_types_in_program(prog)
        assert "coreml_update_state" in op_types
        assert "read_state" in op_types

    def test_not_removed_when_written_to_different_state(self):
        @mb.program(
            input_specs=[
                mb.StateTensorSpec(shape=(2, 3), dtype=types.fp16),
                mb.StateTensorSpec(shape=(2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(s0, s1):
            r0 = mb.read_state(input=s0)
            mb.coreml_update_state(state=s1, value=r0)
            return r0

        assert "coreml_update_state" in get_op_types_in_program(prog)

        apply_pass_and_basic_check(prog, "common::remove_noop_state_update")

        op_types = get_op_types_in_program(prog)
        assert "coreml_update_state" in op_types
        assert "read_state" in op_types

    def test_read_state_preserved_when_used_downstream(self):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(2, 3)),
                mb.StateTensorSpec(shape=(2, 3), dtype=types.fp16),
            ],
            opset_version=ct.target.iOS18,
        )
        def prog(x, s):
            r = mb.read_state(input=s)
            c0 = mb.cast(x=r, dtype="fp32")
            c1 = mb.cast(x=c0, dtype="fp16")
            mb.coreml_update_state(state=s, value=c1)
            # c0 is used in the returned calculation
            return mb.add(x=c0, y=1.0)

        assert "coreml_update_state" in get_op_types_in_program(prog)

        apply_pass_and_basic_check(prog, "common::remove_noop_state_update")

        op_types = get_op_types_in_program(prog)
        # coreml_update_state and the c1 cast back to fp16 are removed
        assert "coreml_update_state" not in op_types
        # read_state and c0 (fp32 cast) are preserved for the add calculation
        assert "read_state" in op_types
        assert "cast" in op_types
        assert "add" in op_types
