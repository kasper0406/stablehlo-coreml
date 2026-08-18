from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


def _unwrap_identity(var):
    visited = set()
    curr = var
    while curr.op is not None and curr not in visited:
        visited.add(curr)
        if curr.op.op_type in ("cast", "identity"):
            curr = curr.op.x
        else:
            break
    return curr


def _match_pattern(op):
    if op.op_type == "coreml_update_state":
        source_var = _unwrap_identity(op.value)
        if source_var.op is not None and source_var.op.op_type == "read_state":
            return source_var.op.input == op.state
    return False


def _try_to_transform(update_op):
    block = update_op.enclosing_block
    # Replace occurrences of the `coreml_update_state` output with `update_op.value`
    block.replace_uses_of_var_after_op(
        anchor_op=update_op,
        old_var=update_op.outputs[0],
        new_var=update_op.value,
    )
    val = update_op.value
    update_op.remove_from_block()

    # Clean up upstream casts/identities/read_state if they are no longer consumed
    curr = val
    while curr.op is not None and curr.op.op_type in ("cast", "identity", "read_state"):
        op_to_check = curr.op
        if len(curr.child_ops) == 0 and curr not in block.outputs:
            prev = op_to_check.x if hasattr(op_to_check, "x") else None
            op_to_check.remove_from_block()
            if prev is not None:
                curr = prev
            else:
                break
        else:
            break

    return True


@block_context_manager
def _remove_noop_state_update(block):
    did_optimize = False
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = _remove_noop_state_update(b)
        if len(op.blocks) > 0:
            continue

        if _match_pattern(op):
            if _try_to_transform(op):
                did_optimize = True
    return did_optimize


@register_pass(namespace="common")
class remove_noop_state_update(AbstractGraphPass):
    """
    If a coreml_update_state writes back the exact value read from the same state
    (possibly through intermediate casts or identity operations), remove the redundant
    state write and any unused upstream read/cast ops.

    Given:
        %1 = read_state(input=%state)
        %2 = cast(x=%1, dtype="fp32")
        ...
        %3 = cast(x=%2, dtype="fp16")
        %4 = coreml_update_state(state=%state, value=%3)

    Result:
        (removes %4, and eliminates %3, %2, %1 if they are not used downstream)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            block_changed = True
            while block_changed:
                block_changed = _remove_noop_state_update(f)
