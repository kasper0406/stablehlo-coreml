import numpy as np
from jaxlib.mlir import ir
from jaxlib.mlir.dialects.stablehlo import (
    CompareOp, SelectOp, ConstantOp
)
from jax._src.lib.mlir.dialects import hlo

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types


# TODO: nan, inf, subnormals
def bitcast_fp(x):
    width = x.dtype.width
    ieee754 = { 32: (8, 23), 16: (5, 10) }
    assert types.is_float(x.dtype) and width in ieee754
    exponent, fraction = ieee754[width]
    e_bias = 2 ** (exponent - 1) - 1
    e_offset = 2 ** fraction

    positive = mb.greater_equal(x=x, y=0.)
    zero = mb.equal(x=x, y=0.)
    x = mb.abs(x=x)
    e_raw = mb.floor_div(x=mb.log(x=x), y=mb.log(x=2.))
    e_shifted = mb.cast(x=mb.add(x=e_raw, y=float(e_bias)), dtype="int32")
    fractional = mb.sub(x=mb.real_div(x=x, y=mb.pow(x=2., y=e_raw)), y=1.)
    mantissa = mb.cast(x=mb.floor(x=mb.mul(x=fractional, y=float(e_offset))), dtype="int32")
    bits = mb.add(x=mb.mul(x=e_shifted, y=e_offset), y=mantissa)
    bits = mb.select(cond=zero, a=0, b=bits)
    return positive, bits


def bitcast_int(x):
    width = x.dtype.width
    assert types.is_int(x.dtype) and width in { 8, 16, 32 }
    packed, signed = width == 32, x.dtype.is_unsigned()
    assert not split or not signed, "CoreML has no uint32 type"

    positive = mb.greater_equal(x=x, y=0) if signed else None
    x = mb.abs(x=x) if signed else x
    x = x if packed else mb.cast(x=x, dtype="int32")
    return positive, x


def bitcast_split(x, mask=16, ascending=True):
    width = x.dtype.width
    if types.is_int(x):
        sign, x = bitcast_int(x)
    elif types.is_float(x):
        sign, x = bitcast_fp(x)
    else:
        raise TypeError("only int and float are supported for sorting")

    splits = -(-width // mask) # ceil
    results = []
    for i in range(splits):
        if i < splits - 1:
            split = mb.mod(x=x, y=2 ** mask)
            flipped = mb.sub(x=2 ** mask - 1, y=split)
        else:
            overflow_bit = 2 ** (mask - (0 if sign is None else 1))
            flipped = mb.sub(x=overflow_bit - 1, y=x)
            split = mb.add(x=x, y=overflow_bit)

        if sign is None:
            out = split if ascending else flipped
        elif ascending:
            out = mb.select(cond=sign, a=split, b=flipped)
        else:
            out = mb.select(cond=sign, a=flipped, b=split)
        results.append(out)
        x = mb.floor_div(x=x, y=2 ** mask)
    return tuple(mb.mul(x=x, y=2 ** (31 - mask)) for x in reversed(results))


def bitcast_window(x, n):
    if n < 2 ** 15:
        # 16 bit operand mask, 15 bit index mask
        return 16
    elif x.dtype.width == 32 and n < 2 ** 20:
        # 11 bit operand mask, 20 bit index mask
        return 11
    elif n < 2 ** 23:
        # 8 bit operand mask, 23 bit index mask
        return 8
    else:
        raise ValueError("refusing to split stable argsort into more than 4 unstable argsorts")


def match_sort(comparator_root, args, inputs):
    """
    Analyzes the comparator region of a SortOp to determine if it implements a supported sorting pattern.
    We try to analyze the comparator logic for multi-key sorts.

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
    def get_op(val):
        if isinstance(val.owner, ir.Block):
            return None
        return val.owner.opview

    def get_arg_index(value):
        if value in args:
            return args.index(value)
        op = get_op(value)
        if op is None:
            return None
        return match_nan_and_zero_handling(op, args)

    def identify_comparison_args(compare_op: CompareOp) -> tuple[int | None, bool | None]:
        lhs = get_arg_index(compare_op.lhs)
        rhs = get_arg_index(compare_op.rhs)
        if lhs is None or rhs is None:
            return None, None

        # According to StableHLO sort spec, arguments are guaranteed to be interleaved:
        #   (lhs_0, rhs_0, lhs_1, rhs_1, ...)
        # Therefore the input pair being compared should be adjacent, and the corresponding
        # key index can be derived by integer division by 2.
        # Reference: https://openxla.org/stablehlo/spec#sort
        if (lhs // 2) != (rhs // 2):
            return None, None

        direction = hlo.ComparisonDirectionAttr(compare_op.comparison_direction).value
        is_ascending = lhs < rhs and direction == "LT" or lhs > rhs and direction == "GT"

        return lhs // 2, is_ascending

    def match_comparison(op, expected_direction=None):
        if not isinstance(op, CompareOp):
            return None

        direction = hlo.ComparisonDirectionAttr(op.comparison_direction).value
        if expected_direction and direction != expected_direction:
            return None

        if expected_direction is None and direction not in ("LT", "GT"):
            return None

        return op

    def match_select_chain(op):
        # Matches: select(pred, on_true, on_false)
        # where pred is (k1 == k2) and on_false is (k1 < k2)
        if not isinstance(op, SelectOp):
            return None

        pred = get_op(op.pred)
        on_false = get_op(op.on_false)

        # 1. Check pred: k1 == k2
        pred_cmp = match_comparison(pred, "EQ")
        if not pred_cmp:
            return None

        # 2. Check on_false: k1 < k2 (or >)
        false_cmp = match_comparison(on_false)
        if not false_cmp:
            return None

        # 3. Verify operands match between pred and on_false
        if {pred_cmp.lhs, pred_cmp.rhs} != {false_cmp.lhs, false_cmp.rhs}:
            return None

        # 4. Identify key
        key_info = identify_comparison_args(pred_cmp)
        if key_info[0] is None:
            return None

        return key_info, get_op(op.on_true)

    def match_leaf(op):
        # Matches: k1 < k2
        cmp = match_comparison(op)
        if not cmp:
            return None
        return identify_comparison_args(cmp)

    # Walk through the operations graph to match the sort pattern
    sort_keys = []
    current_op = comparator_root
    while current_op:
        # Try to match a chain node (SelectOp)
        chain_result = match_select_chain(current_op)
        if chain_result:
            (key_idx, is_asc), next_op = chain_result
            sort_keys.append((inputs[key_idx], is_asc))
            current_op = next_op
            continue

        # Try to match a leaf node (CompareOp)
        leaf_result = match_leaf(current_op)
        if leaf_result:
            key_idx, is_asc = leaf_result
            if key_idx is None:
                return None
            sort_keys.append((inputs[key_idx], is_asc))
            return sort_keys

        # If neither matched, it's not a valid sort pattern
        return None

    return sort_keys


def match_nan_and_zero_handling(op, args):
    """
    Jax generates nan-checks and +0/-0 merges in the comparator. This function matches this pattern:
        select(<var> != <var>, NaN, select(<var> == 0, 0, <var>))

    It will extract the argument index of <var> if the pattern matches.

    Notice that this is technically non correct, as MIL does not handle NaN's in the same way as StableHLO.
    +0/-0 looks to be correctly handled.
    """
    def get_op(val):
        if isinstance(val.owner, ir.Block):
            return None
        return val.owner.opview

    def match_constant(val, check_fn):
        const_op = get_op(val)
        if not isinstance(const_op, ConstantOp):
            return False
        return check_fn(const_op.value)

    def match_isnan(val):
        # Matches: <var> != <var>
        compare_op = get_op(val)
        if not isinstance(compare_op, CompareOp):
            return None
        if hlo.ComparisonDirectionAttr(compare_op.comparison_direction).value != "NE":
            return None
        if compare_op.lhs != compare_op.rhs:
            return None
        return compare_op.lhs

    def match_is_zero(val):
        # Matches: <var> == 0
        compare_op = get_op(val)
        if not isinstance(compare_op, CompareOp):
            return None
        if hlo.ComparisonDirectionAttr(compare_op.comparison_direction).value != "EQ":
            return None
        if not match_constant(compare_op.rhs, lambda x: np.array(x) == 0):
            return None
        return compare_op.lhs

    # 1. Match outer select: select(pred, NaN, on_false)
    if not isinstance(op, SelectOp):
        return None

    if not match_constant(op.on_true, np.isnan):
        return None

    # 2. Match NaN check: pred is (x != x)
    matched_arg_idx = match_isnan(op.pred)
    if matched_arg_idx is None:
        return None

    # 3. Match inner select: select(pred, 0, x)
    inner_op = get_op(op.on_false)
    if not isinstance(inner_op, SelectOp):
        return None

    if not match_constant(inner_op.on_true, lambda x: np.array(x) == 0):
        return None

    if inner_op.on_false != matched_arg_idx:
        return None

    # 4. Match Zero check: pred is (x == 0)
    x_val_zero = match_is_zero(inner_op.pred)
    if x_val_zero != matched_arg_idx:
        return None

    # 5. Return index if found in args
    if matched_arg_idx in args:
        return args.index(matched_arg_idx)

    return None
