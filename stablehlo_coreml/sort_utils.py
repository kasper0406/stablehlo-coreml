import numpy as np
from jaxlib.mlir import ir
from jaxlib.mlir.dialects.stablehlo import (
    CompareOp, SelectOp, ConstantOp, OrOp, AndOp
)
from jax._src.lib.mlir.dialects import hlo

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types


# TODO: nan, inf, subnormals
def bitcast_fp(x):
    width = types.nptype_from_builtin(x.dtype)
    ieee754 = { np.float32: (8, 23), np.float16: (5, 10) }
    assert types.is_float(x.dtype) and width in ieee754
    exponent, fraction = ieee754[width]
    e_bias = 2 ** (exponent - 1) - 1
    e_offset = 2 ** fraction

    positive = mb.greater_equal(x=x, y=width(0.))
    zero = mb.equal(x=x, y=width(0.))
    x = mb.abs(x=x)
    e_raw = mb.floor_div(x=mb.log(x=x), y=np.log(width(2.)))
    e_shifted = mb.cast(x=mb.add(x=e_raw, y=width(e_bias)), dtype="int32")
    fractional = mb.sub(x=mb.real_div(x=x, y=mb.pow(x=width(2.), y=e_raw)), y=width(1.))
    mantissa = mb.cast(x=mb.floor(x=mb.mul(x=fractional, y=width(e_offset))), dtype="int32")
    bits = mb.add(x=mb.mul(x=e_shifted, y=e_offset), y=mantissa)
    bits = mb.select(cond=zero, a=0, b=bits)
    return positive, bits, exponent + fraction + 1


def bitcast_int(x):
    width = x.dtype.width
    assert types.is_int(x.dtype) and width in { 8, 16, 32 }
    packed, signed = width == 32, not x.dtype.is_unsigned()
    assert not packed or signed, "CoreML has no uint32 type"

    x = x if packed else mb.cast(x=x, dtype="int32")
    positive = mb.greater_equal(x=x, y=0) if signed else None
    x = mb.abs(x=x) if signed else x
    return positive, x, width


def bitcast_split(x, mask=16, ascending=True):
    if types.is_int(x.dtype):
        sign, x, width = bitcast_int(x)
    elif types.is_float(x.dtype):
        sign, x, width = bitcast_fp(x)
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
    return results


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


def stable_argsort(x, axis=-1, ascending=True):
    arange = np.indices(x.shape)[axis]
    mask = bitcast_window(x, x.shape[axis])
    splits = bitcast_split(x, mask, ascending)
    # return splits[0]
    # return splits[1]
    shifted = lambda x: mb.add(x=mb.mul(x=x, y=2 ** (x.dtype.width - 1 - mask)), y=arange)
    indices = mb.argsort(x=shifted(splits[0]), axis=axis, ascending=True)
    for window in splits[1:]:
        gathered_key = mb.gather_along_axis(x=window, indices=indices, axis=axis)
        # gather bit casts to int16 then value casts back, so fix the sign bit
        gathered_key = mb.select(cond=mb.less(x=gathered_key, y=0), a=mb.add(x=gathered_key, y=0x1_0000), b=gathered_key)
        relative_indices = mb.argsort(x=shifted(x=gathered_key), axis=axis, ascending=True)
        indices = mb.gather_along_axis(x=indices, indices=relative_indices, axis=axis)
    return indices


def match_sort(tracing, args, inputs):
    def get_arg_index(value):
        if value in args:
            return args.index(value)
        return verify_zero_nan(value.owner.opview, args)

    remaining, expecting, priorities = None, None, []
    while tracing:
        match tracing:
            case CompareOp():
                direction = hlo.ComparisonDirectionAttr(tracing.comparison_direction).value
                lhs, rhs = get_arg_index(tracing.lhs), get_arg_index(tracing.rhs)
                if lhs is None or rhs is None:
                    return None
                if (direction != "EQ" or expecting not in ((lhs, rhs), (rhs, lhs))) if expecting else (
                        direction not in ("LT", "GT") or lhs // 2 != rhs // 2 or lhs + 1 != rhs):
                    return None
                if not expecting:
                    priorities.append((inputs[lhs // 2], direction == "LT"))
                expecting = None if expecting else (lhs, rhs)
                tracing, remaining = remaining, None
            case OrOp() if not expecting:
                tracing, remaining = tracing.lhs.owner.opview, tracing.rhs.owner.opview
            case AndOp() if expecting:
                tracing, remaining = tracing.lhs.owner.opview, tracing.rhs.owner.opview
            case _:
                return None
    return priorities


def verify_zero_nan(tracing, args):
    remaining, idx = None, None
    selecting, comparing = [lambda x: np.array(x) == 0, np.isnan], ["EQ", "NE"]
    while tracing:
        match tracing:
            case SelectOp():
                const = tracing.operands[1].owner.opview
                if not isinstance(const, ConstantOp):
                    return None
                if not len(selecting) or not selecting.pop()(const.value):
                    return None
                if len(selecting): # select CompareOp %cst_# (=NaN) SelectOp
                    remaining = tracing.operands[2].owner.opview
                    tracing = tracing.operands[0].owner.opview
                else: # select CompareOp %cst_# (=0) %arg#
                    # we should know the arg index from the popped CompareOp
                    if idx is None or tracing.operands[2] != args[idx]:
                        return None
                    tracing, remaining = tracing.operands[0].owner.opview, None
            case CompareOp():
                direction = hlo.ComparisonDirectionAttr(tracing.comparison_direction).value
                if not len(comparing) or direction != comparing.pop():
                    return None
                if len(comparing): # `np.isnan` via `NE %arg# %arg#`
                    if tracing.lhs != tracing.rhs or tracing.lhs not in args:
                        return None
                    # ensure op types are interleaved as expected
                    # by checking the size of the selection queue
                    if len(selecting) != 1:
                        return None
                    idx = args.index(tracing.lhs)
                    tracing, remaining = remaining, None
                else: # merge `-0` and `+0` via `EQ %arg# %cst_# (=0.)`
                    const = tracing.rhs.owner.opview
                    if not isinstance(const, ConstantOp):
                        return None
                    if np.array(const.value) != 0 or tracing.lhs != args[idx]:
                        return None
                    tracing = None
            case _:
                return None
    return idx
