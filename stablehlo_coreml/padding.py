from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types

from .utils import dtype_str, get_mil_type


def pad_with_cast(x, pad, mode="constant", constant_val=None):
    """
    Helper function to handle padding for integer tensors.
    mb.pad only supports fp16 and fp32 inputs, so we cast to float, pad, and cast back.
    """
    mil_type = get_mil_type(x)
    is_int_input = types.is_int(mil_type)
    if is_int_input:
        x = mb.cast(x=x, dtype="fp32")
        if constant_val is not None:
            constant_val = mb.cast(x=constant_val, dtype="fp32")

    if constant_val is not None and len(constant_val.shape) > 0:
        constant_val = mb.squeeze(x=constant_val)

    padded = mb.pad(x=x, pad=pad, mode=mode, constant_val=constant_val)

    if is_int_input:
        padded = mb.cast(x=padded, dtype=dtype_str(mil_type))

    return padded
