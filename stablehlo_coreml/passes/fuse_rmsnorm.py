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

That is only one of the spellings the libraries emit. Everything up to and
including the ``rsqrt`` is common to all of them; the tail is reassociated
differently, so the pass walks forward from the ``rsqrt`` and tolerates, in any
order, at most one broadcast ``reshape`` and at most one multiply by a constant
scale before the ``mul`` that normalizes ``x``. The three shapes that occur in
practice are:

* hand-written (the chain above): ``mul(x32, rsqrt)`` and then, optionally,
  ``mul(., scale)``;
* ``flax.nnx.RMSNorm``, whose ``_normalize`` computes ``mul = rsqrt(var + eps)``,
  then ``mul *= scale``, then ``y = x * mul`` -- the scale lands on the *rsqrt*
  result, before the normalize ``mul``;
* ``equinox.nn.RMSNorm``, whose ``jnp.mean(x**2)`` has no axis argument, so it
  reduces to a scalar and JAX broadcasts *after* the ``rsqrt``: a ``reshape``
  sits between the ``rsqrt`` and the normalize ``mul``, and the trailing weight
  multiply has the constant on the left.

A scale on either side of the normalize ``mul`` (or on both) folds into the same
single constant. The reduction itself may keep its dimensions or not: what the
pass validates is the shape the statistic actually has where it broadcasts
against ``x`` (see ``_is_broadcast_partner``), which covers both.

Six of the eight ops above compute ``x / sqrt(mean(x^2) + eps)``, which is
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

import math

import numpy as np
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import any_symbolic

from .pattern_utils import RewritePass, sole_consumer, uniform_const_operand, uniform_scalar_value


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


def _is_broadcast_partner(stat_shape, x_shape) -> bool:
    """True if a statistic of ``stat_shape`` normalizes ``x_shape`` row-wise.

    NumPy broadcasting right-aligns the operands, so ``stat_shape`` is padded
    with leading 1s to ``x``'s rank. The padded shape must then agree with ``x``
    on every axis but the last (otherwise the elementwise ``mul`` would either
    fail or replicate ``x`` along a batch axis instead of scaling it), and its
    last entry must be 1 or the full normalized size -- i.e. one statistic per
    row, or the reduced axis already broadcast back to full width.

    This is what makes ``keep_dims=False`` acceptable: a squeezed ``(1, 1)``
    statistic of a ``(1, 1, d)`` input pads to ``(1, 1, 1)``, which is exactly
    the keep-dims shape. A reshape that permutes non-unit axes always changes
    the shape tuple in a way this rejects, so the walk below never has to reason
    about reshape orderings.
    """
    if stat_shape is None or any_symbolic(stat_shape):
        return False
    if len(stat_shape) > len(x_shape):
        return False
    padded = (1,) * (len(x_shape) - len(stat_shape)) + tuple(stat_shape)
    if [int(dim) for dim in padded[:-1]] != [int(dim) for dim in x_shape[:-1]]:
        return False
    return int(padded[-1]) in (1, int(x_shape[-1]))


def _walk_to_normalize_mul(rsqrt_op, x32, block):
    """Walk from the ``rsqrt`` result to the ``mul`` that normalizes ``x32``.

    Different RMSNorm implementations reassociate the tail of the chain: flax
    folds the learnable scale onto the ``rsqrt`` result *before* multiplying by
    ``x``, and equinox reduces to a scalar and broadcasts the statistic back with
    a ``reshape`` after the ``rsqrt``. The walk therefore tolerates, in any
    order, at most one ``reshape`` and at most one multiply by a constant scale
    on the way to the normalize ``mul``.

    Returns ``(norm_mul, pre_scale, walked)`` -- the normalize ``mul``, the
    constant the walk absorbed (or ``None``), and the ops it consumed in walk
    order -- or ``None`` when the chain is anything else.
    """
    stat = rsqrt_op.outputs[0]
    pre_scale = None
    reshaped = False
    walked = []

    while True:
        consumer = sole_consumer(stat)
        if consumer is None or consumer.enclosing_block is not block:
            return None

        if consumer.op_type == "mul" and {id(consumer.x), id(consumer.y)} == {id(x32), id(stat)}:
            if not _is_broadcast_partner(stat.shape, x32.shape):
                return None
            return consumer, pre_scale, walked

        if consumer.op_type == "reshape":
            if reshaped or not _is_broadcast_partner(consumer.outputs[0].shape, x32.shape):
                return None
            reshaped = True
        elif consumer.op_type == "mul":
            other = consumer.y if consumer.x is stat else consumer.x
            if pre_scale is not None or other is stat or other.val is None:
                return None
            val = np.asarray(other.val)
            if not _matches_scale_shape(val, x32.shape):
                return None
            pre_scale = val
        else:
            return None

        walked.append(consumer)
        stat = consumer.outputs[0]


def _match(rsqrt_op, block):
    """Match the RMSNorm chain ending at ``rsqrt_op``.

    Returns ``(x32, eps, tail_op, scale, dead)``: the normalized input, the
    total epsilon, the last op of the chain (the trailing scale ``mul``, or the
    normalize ``mul`` when there is none), the single constant factor to fold in
    (or ``None``), and the ops the rewrite replaces, in removal order.
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
    # `keep_dims` may be either way -- what matters is the shape the statistic
    # has when it reaches the normalize `mul`, which `_is_broadcast_partner`
    # checks there. Its value does have to be known at compile time.
    if mean_op.keep_dims is None or mean_op.keep_dims.val is None:
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

    # The rsqrt result must reach a single mul against x32 itself, possibly via
    # a broadcast reshape and/or a scale that was reassociated onto it.
    walk = _walk_to_normalize_mul(rsqrt_op, x32, block)
    if walk is None:
        return None
    norm_mul, scale, walked = walk

    # Note there is deliberately no check that x32 is read *only* by the chain.
    # The rewrite never removes the op producing it, and none of the ops in
    # `dead` other than the square and the normalize mul read it, so any other
    # reader keeps working against an unchanged value. Requiring exclusivity
    # would decline every RMSNorm on a residual path (`x + attn(rmsnorm(x))`) in
    # an fp32 graph, where no cast insulates the input from the residual add.

    # Optionally absorb a following multiply by a constant scale. Both scales
    # fold into the same factor when the chain has one on either side.
    tail_op = norm_mul
    scale_mul = sole_consumer(norm_mul.outputs[0])
    if scale_mul is not None and scale_mul.op_type == "mul" and scale_mul.enclosing_block is block:
        other = scale_mul.y if scale_mul.x is norm_mul.outputs[0] else scale_mul.x
        if other is not norm_mul.outputs[0] and other.val is not None:
            val = np.asarray(other.val)
            if _matches_scale_shape(val, x32.shape):
                tail_op = scale_mul
                scale = val if scale is None else scale.astype(np.float64) * val

    # Reverse topological order, so `remove_ops` never sees a live consumer.
    # The walked ops sit between the rsqrt and the normalize mul.
    dead = [norm_mul, *reversed(walked), rsqrt_op, add_op, mean_op, square_op]
    if tail_op is not norm_mul:
        dead.insert(0, tail_op)

    return x32, eps, tail_op, scale, dead


@register_pass(namespace="common")
class fuse_rmsnorm(RewritePass):
    """
    Fuse the RMSNorm elementwise chain into ``l2_norm`` + a constant ``mul``.

    ``x / sqrt(mean(x^2) + eps)`` equals ``sqrt(d) * l2_norm(x, d * eps)``,
    where ``d`` is the size of the normalized (last) axis, so the reduction and
    the elementwise arithmetic collapse into a single op and the learnable
    scale is folded into the ``sqrt(d)`` factor.

    The rewrite applies when

    1. the reduction is a ``reduce_mean`` over the last axis of a ``mul(x, x)``
       square (``keep_dims`` either way, as long as it is known at compile time),
    2. the ``rsqrt`` result reaches the ``mul`` that normalizes ``x`` through at
       most one ``reshape`` and at most one multiply by a constant scale, and
       the statistic broadcasts against ``x`` as one value per row,
    3. every intermediate value of the chain has exactly one consumer -- the
       normalized input itself is exempt, so a residual path reading ``x``
       alongside the norm does not block the rewrite, and
    4. the input has a static shape with ``rank >= 3`` and
       ``shape[-2] == shape[-3] == 1`` -- the shapes for which ``l2_norm``'s
       last-three-dimensions reduction is the last dimension alone. Other
       shapes would need a reshape around the op, which costs more than the
       ops it saves.

    Given (hand-written spelling; flax puts the scale on ``%4`` instead, and
    equinox reshapes ``%4`` before ``%5``):
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

    _REWRITES = "RMSNorm chain(s)"

    def visit(self, op, block) -> bool:
        if op.op_type != "rsqrt":
            return False

        match = _match(op, block)
        if match is None:
            return False
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
        return True
