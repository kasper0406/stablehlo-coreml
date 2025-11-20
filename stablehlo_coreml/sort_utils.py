import numpy as np
from jaxlib.mlir import ir
from jaxlib.mlir.dialects.stablehlo import (
    CompareOp, SelectOp, OrOp, AndOp
)
from jax._src.lib.mlir.dialects import hlo


def match_sort(comparator_root, args, inputs):
    """
    Analyzes the comparator region of a SortOp to determine if it implements a multi-key (lexicographical) sort.

    Multi-key sort compares multiple keys in sequence. If the primary keys are equal,
    it moves to the secondary keys, and so on.

    The comparator logic is expected to look like a chain of SelectOps:
    if (k1 == k2) then compare(next_keys) else compare(k1 < k2)

    Args:
        comparator_root: The current operation in the comparator region being analyzed (starts at the return value).
        args: The arguments of the comparator block (representing the two elements being compared).
        inputs: The list of input tensors to the SortOp.

    Returns:
        A list of (tensor, ascending) tuples representing the sort keys, or None if the pattern doesn't match.
    """
    remaining, expecting, sort_keys = None, None, []

    def get_op(val):
        if isinstance(val.owner, ir.Block):
            return None
        return val.owner.opview

    def get_arg_index(value, args):
        if value in args:
            return args.index(value)
        if isinstance(value.owner, ir.Block):
            return None
        return match_total_order_arg(value.owner.opview, args)

    # Walk backwards through the operations graph to understand the sorting logic
    current_op = comparator_root
    while current_op:
        match current_op:
            case SelectOp():
                # Pattern: select(pred, on_true, on_false)
                # This represents the chain: if (k1 == k2) then compare(next) else compare(k1 < k2)
                pred = get_op(current_op.pred)
                on_true = get_op(current_op.on_true)
                on_false = get_op(current_op.on_false)

                if not isinstance(pred, CompareOp):
                    return None
                if hlo.ComparisonDirectionAttr(pred.comparison_direction).value != "EQ":
                    return None

                if not isinstance(on_false, CompareOp):
                    return None
                dir_false = hlo.ComparisonDirectionAttr(on_false.comparison_direction).value
                if dir_false not in ("LT", "GT"):
                    return None

                if {pred.lhs, pred.rhs} != {on_false.lhs, on_false.rhs}:
                    return None

                # Identify which input is being compared in the 'else' branch (on_false)
                lhs = get_arg_index(pred.lhs, args)
                rhs = get_arg_index(pred.rhs, args)
                if lhs is None or rhs is None:
                    return None

                # Arguments are interleaved: (lhs_0, rhs_0, lhs_1, rhs_1, ...)
                k1 = lhs // 2
                k2 = rhs // 2

                if k1 != k2:
                    return None
                k = k1

                is_lhs_first = (lhs % 2 == 0)
                ascending = (dir_false == "LT") == is_lhs_first
                sort_keys.append((inputs[k], ascending))

                # Continue analyzing the 'then' branch (on_true) for the next key
                current_op = on_true

            case CompareOp():
                # Base case: A simple comparison, usually the last one in the chain or the only one.
                direction = hlo.ComparisonDirectionAttr(current_op.comparison_direction).value

                lhs = get_arg_index(current_op.lhs, args)
                rhs = get_arg_index(current_op.rhs, args)
                if lhs is None or rhs is None:
                    return None

                k1 = lhs // 2
                k2 = rhs // 2

                if k1 != k2:
                    return None
                k = k1

                if expecting:
                    # If we were inside an AndOp/OrOp structure
                    exp_lhs, exp_rhs = expecting
                    if {lhs, rhs} != {exp_lhs, exp_rhs}:
                        return None
                    if direction != "EQ":
                        return None
                else:
                    if direction not in ("LT", "GT"):
                        return None
                    is_lhs_first = (lhs % 2 == 0)
                    ascending = (direction == "LT") == is_lhs_first
                    sort_keys.append((inputs[k], ascending))

                expecting = None if expecting else (lhs, rhs)
                current_op, remaining = remaining, None
            case OrOp() if not expecting:
                current_op, remaining = get_op(current_op.lhs), get_op(current_op.rhs)
            case AndOp() if expecting:
                current_op, remaining = get_op(current_op.lhs), get_op(current_op.rhs)
            case _:
                return None
    return sort_keys


def match_total_order_arg(tracing, args):
    """
    Matches the operation chain that handles NaN and zero comparisons for total sort.

    It looks for logic that canonicalizes NaNs and zeros before comparison and returns
    the index of the original argument being compared.
    """
    remaining, idx = None, None
    selecting, comparing = [lambda x: np.array(x) == 0, np.isnan], ["EQ", "NE"]

    def get_op(val):
        if isinstance(val.owner, ir.Block):
            return None
        return val.owner.opview

    while tracing:
        match tracing:
            case SelectOp():
                const_op = get_op(tracing.operands[1])
                if not const_op:
                    return None
                const = const_op.value

                if not len(selecting) or not selecting.pop()(const):
                    return None
                if len(selecting):
                    remaining = get_op(tracing.operands[2])
                    tracing = get_op(tracing.operands[0])
                    if not remaining or not tracing:
                        return None
                else:
                    if idx is None or tracing.operands[2] != args[idx]:
                        return None
                    tracing = get_op(tracing.operands[0])
                    if not tracing:
                        return None
                    remaining = None
            case CompareOp():
                direction = hlo.ComparisonDirectionAttr(tracing.comparison_direction).value
                if not len(comparing) or direction != comparing.pop():
                    return None
                if len(comparing):
                    if tracing.lhs != tracing.rhs or tracing.lhs not in args or len(selecting) != 1:
                        return None
                    idx = args.index(tracing.lhs)
                    tracing, remaining = remaining, None
                else:
                    rhs_op = get_op(tracing.rhs)
                    if not rhs_op:
                        return None
                    const = np.array(rhs_op.value)
                    if const != 0 or tracing.lhs != args[idx]:
                        return None
                    tracing = None
            case _:
                return None
    return idx
