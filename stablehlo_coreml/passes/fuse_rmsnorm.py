"""MIL pass: collapse the RMSNorm elementwise chain onto the ``l2_norm`` op.

RMSNorm is usually written with fp32 statistics over an fp16 activation, which
the converter (helped by ``fuse_reduce_keep_dims`` and coremltools'
``fuse_reduce_mean``) lowers to eight ops::

    %x32  = cast(x=%x, dtype=fp32)
    %sq   = mul(x=%x32, y=%x32)
    %var  = reduce_mean(x=%sq, axes=[-1], keep_dims=True)
    %vare = add(x=%var, y=eps)
    %inv  = rsqrt(x=%vare)
    %n    = mul(x=%x32, y=%inv)
    %s    = mul(x=%n, y=const scale)          # absent for a norm without scale
    %out  = cast(x=%s, dtype=fp16)

Six of those eight ops compute ``x / sqrt(mean(x^2) + eps)``, which is
``l2_norm`` up to a constant:

.. math::
   \\frac{x}{\\sqrt{\\frac{1}{d}\\sum x^2 + \\epsilon}}
   = \\sqrt{d}\\;\\frac{x}{\\sqrt{\\sum x^2 + d\\epsilon}}
   = \\sqrt{d}\\;\\mathrm{l2\\_norm}(x,\\ \\epsilon' = d\\epsilon)

so the whole chain becomes ``l2_norm`` plus one ``mul`` by the precomputed
constant ``sqrt(d) * scale`` -- 8 ops down to 4, with the surrounding casts (and
therefore the fp32/fp16 placement) untouched. The rewrite is exact algebra on
exact constants; the only numerical difference is a division where the original
had an ``rsqrt`` multiply, both in fp32.

Why not ``layer_norm``? Every pattern in coremltools'
``fuse_layernorm_or_instancenorm`` begins ``x -> reduce_mean -> sub``, and the
op itself is defined as ``gamma * (x - E[x]) / sqrt(Var[x] + eps) + beta``.
RMSNorm has no mean subtraction, so that op cannot express it without changing
the numerics, and that pass never matches this chain.

Shape handling: ``l2_norm`` normalizes over the **last three** dimensions and
treats everything before them as batch, so it computes the right reduction only
when ``rank >= 3`` and ``x.shape[-2] == x.shape[-3] == 1``. Chains whose input
is shaped otherwise are left alone.

Off-canonical shapes could be fused too, by reshaping to ``(-1, 1, 1, d)`` and
back (8 ops down to 6), and that variant was tried. It was measurably worse: in
a graph where every site is off-canonical it adds two reshapes of the full
activation per norm, which on a 128-token transformer chunk cost ~0.5 ms
against both the unfused baseline and this shape-restricted version. Fusing
only the shapes that need no reshape adds no op anywhere.

What this pass is and is not worth: it shrinks the graph (~15% of the non-const
ops of a small transformer decoder), which shortens compile time and makes the
program readable, but it is not a latency win on its own -- Core ML's compute
plan puts the whole elementwise population of such a graph at a fraction of a
percent of the estimated cost, against ~21% for the ``matmul``s. Op-count work
of this kind should be judged on graph size, not on latency.
"""

import logging
import math

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

from .pattern_utils import sole_consumer, uniform_const_operand, uniform_scalar_value

logger = logging.getLogger(__name__)


def _matches_scale_shape(val, x_shape) -> bool:
    """True if the constant ``val`` only broadcasts over the normalized axis.

    Anything else (a scale varying along a batch axis) would not survive being
    folded into a single per-channel factor.
    """
    if val.ndim == 0:
        return True
    if val.shape[-1] not in (1, int(x_shape[-1])):
        return False
    return all(dim == 1 for dim in val.shape[:-1])


def _match(rsqrt_op, block):
    """Match the RMSNorm chain ending at ``rsqrt_op``.

    Returns ``(x32, eps, tail_op, scale, dead)``: the normalized input, the
    total epsilon, the last op of the chain (the scale ``mul``, or the
    normalize ``mul`` when the norm has no learnable scale), that scale's
    constant value or ``None``, and the ops the rewrite replaces, in removal
    order.
    """
    # rsqrt(x) is 1/sqrt(x + epsilon); fold that epsilon in with the add's.
    eps = uniform_scalar_value(rsqrt_op.inputs.get("epsilon")) or 0.0

    add_op = rsqrt_op.x.op
    if add_op is None or add_op.op_type != "add" or add_op.enclosing_block is not block:
        return None
    if sole_consumer(rsqrt_op.x) is not rsqrt_op:
        return None
    shifted = uniform_const_operand(add_op)
    if shifted is None:
        return None
    eps_add, var_var = shifted
    eps += eps_add

    mean_op = var_var.op
    if mean_op is None or mean_op.op_type != "reduce_mean" or mean_op.enclosing_block is not block:
        return None
    if sole_consumer(var_var) is not add_op:
        return None
    if mean_op.keep_dims is None or mean_op.keep_dims.val is not True:
        return None
    if mean_op.axes is None or mean_op.axes.val is None:
        return None
    axes = list(np.asarray(mean_op.axes.val).reshape(-1))
    rank = mean_op.x.rank
    if len(axes) != 1 or (axes[0] % rank) != rank - 1:
        return None

    square_op = mean_op.x.op
    if square_op is None or square_op.op_type != "mul" or square_op.enclosing_block is not block:
        return None
    if square_op.x is not square_op.y:
        return None
    if sole_consumer(mean_op.x) is not mean_op:
        return None

    x32 = square_op.x
    if x32.dtype != rsqrt_op.outputs[0].dtype:
        return None
    if x32.rank < 3 or any_symbolic(x32.shape):
        return None
    # `l2_norm` reduces over the last three dims; only these shapes make that
    # the last dim alone.
    if x32.shape[-2] != 1 or x32.shape[-3] != 1:
        return None

    # The rsqrt result must feed exactly one mul, against x32 itself.
    norm_mul = sole_consumer(rsqrt_op.outputs[0])
    if norm_mul is None or norm_mul.op_type != "mul" or norm_mul.enclosing_block is not block:
        return None
    if {id(norm_mul.x), id(norm_mul.y)} != {id(x32), id(rsqrt_op.outputs[0])}:
        return None

    # x32 is read by the square (twice) and by the normalize mul, nothing else.
    if {id(op) for op in x32.child_ops} != {id(square_op), id(norm_mul)}:
        return None

    # Optionally absorb a following multiply by a constant scale.
    tail_op, scale = norm_mul, None
    scale_mul = sole_consumer(norm_mul.outputs[0])
    if scale_mul is not None and scale_mul.op_type == "mul" and scale_mul.enclosing_block is block:
        other = scale_mul.y if scale_mul.x is norm_mul.outputs[0] else scale_mul.x
        if other is not norm_mul.outputs[0] and other.val is not None:
            val = np.asarray(other.val)
            if _matches_scale_shape(val, x32.shape):
                tail_op, scale = scale_mul, val

    # Reverse topological order, so `remove_ops` never sees a live consumer.
    dead = [norm_mul, rsqrt_op, add_op, mean_op, square_op]
    if tail_op is not norm_mul:
        dead.insert(0, tail_op)

    return x32, eps, tail_op, scale, dead


@block_context_manager
def _fuse_rmsnorm(block) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            fused += _fuse_rmsnorm(nested_block)
        if len(op.blocks) > 0 or op.op_type != "rsqrt":
            continue

        match = _match(op, block)
        if match is None:
            continue
        x32, eps, tail_op, scale, dead = match

        d = int(x32.shape[-1])
        np_dtype = types.nptype_from_builtin(x32.dtype)

        normalized = mb.l2_norm(
            x=x32, epsilon=np_dtype(d * eps), before_op=tail_op, name=tail_op.name + "_l2"
        )
        # The `sqrt(d)` the identity above pulls out, folded into the scale.
        factor = math.sqrt(d) if scale is None else np.sqrt(d) * scale.astype(np.float64)
        rescaled = mb.mul(x=normalized, y=np_dtype(factor), before_op=tail_op, name=tail_op.name)

        block.replace_uses_of_var_after_op(
            anchor_op=tail_op,
            old_var=tail_op.outputs[0],
            new_var=rescaled,
        )
        block.remove_ops(dead)
        fused += 1

    return fused


@register_pass(namespace="common")
class fuse_rmsnorm(AbstractGraphPass):
    """
    Fuse the eight-op RMSNorm chain into ``l2_norm`` + a constant ``mul``.

    ``x / sqrt(mean(x^2) + eps)`` equals ``sqrt(d) * l2_norm(x, d * eps)``,
    where ``d`` is the size of the normalized (last) axis, so the reduction and
    the elementwise arithmetic collapse into a single op and the learnable
    scale is folded into the ``sqrt(d)`` factor.

    The rewrite applies when

    1. the reduction is a ``reduce_mean`` with ``keep_dims=True`` over the last
       axis of a ``mul(x, x)`` square,
    2. every intermediate value of the chain has exactly one consumer, and
    3. the input has a static shape with ``rank >= 3`` and
       ``shape[-2] == shape[-3] == 1`` -- the shapes for which ``l2_norm``'s
       last-three-dimensions reduction is the last dimension alone. Other
       shapes would need a reshape around the op, which costs more than the
       ops it saves.

    Given:
        %1 = mul(x=%0, y=%0)
        %2 = reduce_mean(x=%1, axes=[-1], keep_dims=True)
        %3 = add(x=%2, y=1e-6)
        %4 = rsqrt(x=%3)
        %5 = mul(x=%0, y=%4)
        %6 = mul(x=%5, y=<const scale>)          # optional

    Result:
        %5 = l2_norm(x=%0, epsilon=d * 1e-6)
        %6 = mul(x=%5, y=<const sqrt(d) * scale>)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            fused = _fuse_rmsnorm(f)
            if fused:
                logger.debug("fuse_rmsnorm: fused %d RMSNorm chain(s)", fused)
