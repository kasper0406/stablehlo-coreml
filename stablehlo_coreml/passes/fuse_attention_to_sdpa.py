"""MIL pass: fuse a decomposed attention block into ``scaled_dot_product_attention``.

Attention arrives from StableHLO as two ``dot_general``\\ s with a softmax in
between. ``dot_general`` lowers to ``[transpose] -> [reshape] -> matmul(...,
transpose_y=True) -> [reshape] -> [transpose]``, so an attention block looks
like::

    matmul_0(Q, K, transpose_y=True) -> [reshape/transpose]* -> [scale] ->
      [mask] -> [cast] -> softmax(axis=-1) -> [reshape/transpose]* ->
      matmul_1(W, V, transpose_y=True)

The pass matches that and emits a single ``scaled_dot_product_attention``.

Everything happens in "matmul space": the SDPA operands are exactly the matmul
operands, so nothing is assumed about how the model laid out heads, groups or
batches. The reshapes and transposes around the softmax only re-group the two
result axes of ``matmul_0``, and the pass tracks that re-grouping symbolically
(see :class:`_Atom`) to work out

* which of the two matmul result axes is the key axis ``S`` (the one softmax
  reduces over) and which one holds the query rows ``L``,
* how the query rows of ``matmul_0`` are permuted before they reach
  ``matmul_1`` -- Gemma-style GQA einsums merge the group and token axes in a
  different order on each side,
* how a mask expressed in softmax space maps onto SDPA's ``[..., L, S]``
  attention scores.

Whenever that cannot be established (symbolic dimensions where concrete ones
are needed, a result axis split across the softmax axis, an intermediate value
that is consumed elsewhere, ...) the pass leaves the graph alone.

PyTorch's ``_safe_softmax`` wrapper -- ``select(row_is_all_neg_inf, 0.0,
softmax(x))``, which several HuggingFace models lower literally -- is peeled off
as well, but only when the matched mask proves that no row can be entirely
-inf; otherwise the wrapper is not an identity and the block is left alone.
"""

import logging
import math

import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from .pattern_utils import (
    broadcast_shapes,
    const_int_list,
    dims_equal,
    is_broadcast_tile,
    normalize_axis,
    shapes_equal,
    sole_consumer,
    uniform_const_operand,
    uniform_scalar_value,
)

logger = logging.getLogger(__name__)

# Ops that only re-group axes; they never reorder elements within an axis.
_LAYOUT_OPS = frozenset({"reshape", "expand_dims", "squeeze", "transpose"})

# A `select` fill at or below this value means "mask this key out".
_MASK_FILL_THRESHOLD = -1e4

# Ops that a boolean predicate may be routed through without changing which
# rows it selects (a `tile` only when it merely broadcasts).
_PREDICATE_LAYOUT_OPS = frozenset({"reshape", "expand_dims", "squeeze", "cast"})


class _Atom:
    """A (possibly split) axis of the ``matmul_0`` result.

    Reshapes and transposes never reorder elements within an axis: they only
    split axes into factors, merge them again and permute them. Representing the
    axes as trees of atoms therefore describes such a chain exactly.
    """

    __slots__ = ("children", "size")

    def __init__(self, size: int):
        self.size = int(size)
        self.children = None

    def split(self, first: int):
        """Split into a leading factor of ``first`` and the remainder."""
        head, tail = _Atom(first), _Atom(self.size // first)
        self.children = (head, tail)
        return head, tail

    def leaves(self) -> list["_Atom"]:
        """The leaves of the split tree, in row-major (flattened) order."""
        if self.children is None:
            return [self]
        return self.children[0].leaves() + self.children[1].leaves()


def _spare_unit_atoms(flat: list[_Atom], index: int, needed_later: int) -> bool:
    """Can the size-1 atom at ``index`` be absorbed by the current dimension?

    Only if the atoms left over still cover the ``needed_later`` size-1 target
    dimensions that follow: a size-1 target dimension can only ever be filled by
    a size-1 atom, and groups have to stay contiguous.
    """
    available = sum(1 for atom in flat[index + 1:] if atom.size == 1)
    return available >= needed_later


def _regroup(flat: list[_Atom], target_dims) -> list[list[_Atom]] | None:
    """Re-group a flat atom sequence into ``target_dims`` (that is: a reshape).

    Atoms are split when a target dimension ends in the middle of one. Returns
    the new per-dimension atom lists, or ``None`` when the target cannot be
    formed (symbolic dimension, or an unexpected element count).
    """
    flat = list(flat)
    dims = list(target_dims)
    for dim in dims:
        if is_symbolic(dim) or int(dim) <= 0:
            return None
    dims = [int(dim) for dim in dims]
    # Number of size-1 target dimensions strictly after each position; a size-1
    # atom may only be absorbed early when enough of them are left for those.
    unit_dims_after = [0] * (len(dims) + 1)
    for i in range(len(dims) - 1, -1, -1):
        unit_dims_after[i] = unit_dims_after[i + 1] + (1 if dims[i] == 1 else 0)

    grouped: list[list[_Atom]] = []
    index = 0
    for position, dim in enumerate(dims):
        group: list[_Atom] = []
        product = 1
        while index < len(flat):
            atom = flat[index]
            if product == dim:
                # Size-1 atoms carry no information, so they are absorbed
                # greedily -- but not past a later size-1 target dimension that
                # would then be left without an atom of its own.
                if atom.size != 1 or not _spare_unit_atoms(flat, index, unit_dims_after[position + 1]):
                    break
                group.append(atom)
                index += 1
            elif atom.size == 1:
                # Groups are contiguous, so a size-1 atom in the middle of a
                # dimension has to go into it.
                group.append(atom)
                index += 1
            elif product * atom.size <= dim:
                product *= atom.size
                group.append(atom)
                index += 1
            else:
                needed = dim // product
                if needed <= 1 or atom.size % needed != 0:
                    return None
                head, tail = atom.split(needed)
                flat[index:index + 1] = [head, tail]
        if product != dim:
            return None
        grouped.append(group)
    if index != len(flat):
        return None
    return grouped


def _track_layout(layout: list[list[_Atom]], ops) -> list[list[_Atom]] | None:
    """Apply ``ops`` (in execution order) to ``layout``."""
    for op in ops:
        in_shape = tuple(op.inputs["x"].shape)
        out_shape = tuple(op.outputs[0].shape)
        if len(layout) != len(in_shape):
            return None
        if op.op_type == "transpose":
            perm = const_int_list(op.inputs.get("perm"))
            if perm is None or len(perm) != len(in_shape):
                return None
            perm = [normalize_axis(p, len(in_shape)) for p in perm]
            if any(p is None for p in perm):
                return None
            layout = [layout[p] for p in perm]
        else:
            flat = [atom for dim in layout for atom in dim]
            layout = _regroup(flat, out_shape)
            if layout is None:
                return None
    return layout


def _expand(dim) -> list[_Atom]:
    """Expand a dimension's atoms to their (current) leaves."""
    return [leaf for atom in dim for leaf in atom.leaves()]


def _leads_to_matmul(var, depth: int = 12) -> bool:
    """Cheap check whether ``var`` is (transitively) produced by a matmul."""
    for _ in range(depth):
        op = getattr(var, "op", None)
        if op is None:
            return False
        if op.op_type == "matmul":
            return True
        x = op.inputs.get("x")
        if x is None:
            return False
        var = x
    return False


def _peel_broadcast_tile(var):
    """Strip a leading broadcast ``tile`` off ``var``."""
    op = getattr(var, "op", None)
    if op is not None and is_broadcast_tile(op):
        return op.inputs["x"]
    return var


def _is_uniform_neg_inf(var) -> bool:
    """True when ``var`` is a constant whose every element is exactly -inf."""
    value = uniform_scalar_value(var)
    return value is not None and math.isinf(value) and value < 0


def _peel_predicate(var, readers):
    """Peel the layout/cast/``logical_not`` ops in front of a boolean predicate.

    Returns ``(var, negations)``; ``negations`` counts the ``logical_not`` ops
    that were peeled, so the caller can track the polarity. The peeled ops are
    appended to ``readers``.
    """
    negations = 0
    while True:
        op = getattr(var, "op", None)
        if op is None:
            break
        if op.op_type == "logical_not":
            negations += 1
        elif op.op_type == "tile":
            if not is_broadcast_tile(op):
                break
        elif op.op_type not in _PREDICATE_LAYOUT_OPS:
            break
        readers.append(op)
        var = op.inputs["x"]
    return var, negations


class _SafeSoftmax:
    """Torch's ``_safe_softmax`` wrapper around a plain softmax.

    ``torch.nn.functional.scaled_dot_product_attention`` zeroes the rows whose
    scores are all -inf instead of letting them become NaN. Lowered literally,
    that is ``select(row_is_all_neg_inf, 0.0, softmax(x))``.
    """

    def __init__(self, select_op, readers):
        self.select_op = select_op
        # The ops of the wrapper that read the scores; `_match_backward` has to
        # know about them so they do not look like consumers escaping the match.
        self.readers = readers


def _match_safe_softmax(softmax_op) -> _SafeSoftmax | None:
    """Recognise the ``select`` that zeroes fully masked rows after ``softmax_op``."""
    weights = softmax_op.outputs[0]
    select_op = sole_consumer(weights)
    if select_op is None or select_op.op_type != "select":
        return None
    cond = select_op.inputs["cond"]
    a, b = select_op.inputs["a"], select_op.inputs["b"]
    if not shapes_equal(select_op.outputs[0].shape, weights.shape):
        return None

    a_zeros, b_zeros = uniform_scalar_value(a) == 0.0, uniform_scalar_value(b) == 0.0
    if a is weights and b_zeros:
        # The weights survive where the condition holds.
        weights_when_true = True
    elif b is weights and a_zeros:
        weights_when_true = False
    else:
        return None

    readers = [select_op]
    base, cond_negations = _peel_predicate(cond, readers)

    reduce_op = getattr(base, "op", None)
    if reduce_op is None or reduce_op.op_type not in ("reduce_max", "reduce_min"):
        return None
    reduce_in = reduce_op.inputs["x"]
    if reduce_in.shape is None or not shapes_equal(reduce_in.shape, softmax_op.x.shape):
        return None
    axes = const_int_list(reduce_op.inputs.get("axes"))
    rank = len(reduce_in.shape)
    if axes is None or len(axes) != 1 or normalize_axis(axes[0], rank) != rank - 1:
        # The reduction has to be over the softmax axis to describe a "row".
        return None
    readers.append(reduce_op)

    inner, inner_negations = _peel_predicate(reduce_in, readers)
    compare_op = getattr(inner, "op", None)
    if compare_op is None or compare_op.op_type not in ("equal", "not_equal"):
        return None
    x, y = compare_op.inputs["x"], compare_op.inputs["y"]
    if _is_uniform_neg_inf(y):
        scores = x
    elif _is_uniform_neg_inf(x):
        scores = y
    else:
        return None
    if scores is not softmax_op.x:
        # The comparison may run on the pre-/post-cast version of the scores.
        scores_op, softmax_in_op = getattr(scores, "op", None), getattr(softmax_op.x, "op", None)
        if scores_op is not None and scores_op.op_type == "cast" and scores_op.inputs["x"] is softmax_op.x:
            readers.append(scores_op)
        elif softmax_in_op is not None and softmax_in_op.op_type == "cast" and softmax_in_op.inputs["x"] is scores:
            pass
        else:
            return None
    readers.append(compare_op)

    # -- polarity ------------------------------------------------------------
    # `element_is_neg_inf` after the comparison and the `logical_not`s below the
    # reduction; `flipped` means the predicate is "this element is NOT -inf".
    flipped = ((0 if compare_op.op_type == "equal" else 1) + inner_negations) % 2
    if flipped:
        # any(not -inf) == "the row is not all -inf"
        if reduce_op.op_type != "reduce_max":
            return None
        row_not_all_neg_inf = True
    else:
        # all(is -inf) == "the row is all -inf"
        if reduce_op.op_type != "reduce_min":
            return None
        row_not_all_neg_inf = False
    if cond_negations % 2:
        row_not_all_neg_inf = not row_not_all_neg_inf

    # The wrapper must keep the softmax weights exactly for the rows that are
    # not entirely -inf; any other polarity means we misread the graph.
    if weights_when_true != row_not_all_neg_inf:
        return None

    return _SafeSoftmax(select_op, readers)


def _safe_softmax_is_redundant(pattern) -> bool:
    """True when no score row can be entirely -inf, so the wrapper is an identity."""
    if pattern.mask_kind is None:
        return True
    if pattern.mask_kind == "select":
        # A finite fill (`torch.finfo(dtype).min`) leaves a fully masked row
        # finite, so softmax produces a uniform row rather than NaN.
        return not math.isinf(float(pattern.mask_fill))
    mask_value = getattr(pattern.mask_var, "val", None)
    if mask_value is None:
        # A mask computed at runtime (Whisper builds one out of -inf and 0.0)
        # may well have an all -inf row, and then the wrapper is *not* an
        # identity: SDPA would return NaN where the original graph returns 0.
        return False
    return not bool(np.isinf(np.asarray(mask_value)).any())


def _swap_last_two(rank: int) -> list[int]:
    return list(range(rank - 2)) + [rank - 1, rank - 2]


def _np_scalar(value: float, dtype):
    return np.float16(value) if dtype == types.fp16 else np.float32(value)


def _as_sequence_major(var, needs_transpose: bool, before_op, name: str):
    """Return ``var``, or its last-two-axes transpose, as a ``[batch..., seq, E]`` tensor.

    When ``var`` is itself such a transpose it is peeled instead of stacking a
    second one.
    """
    if not needs_transpose:
        return var
    rank = len(var.shape)
    op = getattr(var, "op", None)
    if op is not None and op.op_type == "transpose":
        perm = const_int_list(op.inputs.get("perm"))
        if perm is not None and len(perm) == rank:
            perm = [normalize_axis(p, rank) for p in perm]
            if perm == _swap_last_two(rank):
                return op.inputs["x"]
    return mb.transpose(x=var, perm=_swap_last_two(rank), before_op=before_op, name=name)


class _Pattern:
    """Everything the matcher found for one attention block."""

    def __init__(self, softmax_op):
        self.softmax_op = softmax_op
        self.matmul_0 = None
        self.matmul_1 = None
        self.weights_var = None
        self.back_layout_ops = []       # matmul_0 -> softmax, in execution order
        self.fwd_layout_ops = []        # softmax -> matmul_1, in execution order
        self.mask_kind = None           # None | "select" | "add"
        self.mask_var = None            # the compact condition / additive mask
        self.mask_fill = None           # the `select` fill value
        self.mask_negated = False       # matched select(cond, fill, scores)
        self.scale = None               # multiplicative factor applied to the scores
        self.mask_before_scale = False  # forward order was mask -> scale


class _Space:
    """How softmax space and ``matmul_1``'s weights relate to matmul space."""

    def __init__(self, n_batch, batch_shape, *, s_is_first=False, s_atom=None,
                 l_atom=None, batch_atoms=None, softmax_layout=None, l_prime=None):
        self.n_batch = n_batch
        self.batch_shape = list(batch_shape)
        # True when the softmax axis is `matmul_0`'s first result axis, i.e. the
        # scores are transposed with respect to SDPA's [..., L, S] convention.
        self.s_is_first = s_is_first
        self.s_atom = s_atom
        self.l_atom = l_atom
        self.batch_atoms = batch_atoms
        self.softmax_layout = softmax_layout
        self.l_leaves = None if l_atom is None else l_atom.leaves()
        self.l_prime = l_prime

    @property
    def is_identity(self) -> bool:
        """True when there is no layout op at all between the matmuls and the softmax."""
        return self.softmax_layout is None

    def sdpa_atom_order(self) -> list[_Atom]:
        """The atom order of SDPA's ``[batch..., L, S]`` score space."""
        order = []
        for atom in self.batch_atoms:
            order.extend(atom.leaves())
        return order + list(self.l_leaves) + [self.s_atom]

    @property
    def is_score_space(self) -> bool:
        """True when softmax space already has SDPA's ``[batch..., L, S]`` layout."""
        if self.is_identity:
            return True
        if len(self.softmax_layout) != self.n_batch + 2:
            return False
        for atom, dim in zip(self.batch_atoms, self.softmax_layout):
            if [id(a) for a in dim] != [id(a) for a in atom.leaves()]:
                return False
        if self.softmax_layout[-1] != [self.s_atom]:
            return False
        return [id(a) for a in self.softmax_layout[-2]] == [id(a) for a in self.l_leaves]


def _match_backward(softmax_op, pattern, ignored=()) -> bool:
    """Walk from ``softmax.x`` back to ``matmul_0``, filling in ``pattern``.

    ``ignored`` holds the ``id()`` of ops that were already matched as part of
    the same pattern (the safe-softmax condition chain reads the scores too) and
    that therefore do not count as consumers escaping the pattern.
    """
    var = softmax_op.x
    consumer = softmax_op
    n_casts = 0
    order = []
    back_ops = []

    while True:
        if sole_consumer(var, ignored) is not consumer:
            return False
        op = getattr(var, "op", None)
        if op is None:
            return False
        op_type = op.op_type

        if op_type == "matmul":
            pattern.matmul_0 = op
            break

        if op_type == "cast":
            if n_casts >= 2:
                return False
            n_casts += 1
            next_var = op.inputs["x"]
        elif op_type in _LAYOUT_OPS:
            back_ops.append(op)
            next_var = op.inputs["x"]
        elif op_type == "select" and pattern.mask_kind is None:
            cond, a, b = op.inputs["cond"], op.inputs["a"], op.inputs["b"]
            fill_a, fill_b = uniform_scalar_value(a), uniform_scalar_value(b)
            if fill_b is not None and fill_b <= _MASK_FILL_THRESHOLD:
                pattern.mask_fill, pattern.mask_negated, next_var = fill_b, False, a
            elif fill_a is not None and fill_a <= _MASK_FILL_THRESHOLD:
                pattern.mask_fill, pattern.mask_negated, next_var = fill_a, True, b
            else:
                return False
            pattern.mask_kind = "select"
            pattern.mask_var = cond
            order.append("mask")
        elif op_type == "add" and pattern.mask_kind is None:
            x, y = op.inputs["x"], op.inputs["y"]
            # A uniform constant shifts every score by the same amount, which
            # softmax cancels out; it carries no masking information, so there
            # is nothing to translate and we leave the graph alone. A
            # non-uniform constant (T5's relative position bias, folded to a
            # constant) is a perfectly good additive float mask.
            if uniform_scalar_value(x) is not None or uniform_scalar_value(y) is not None:
                return False
            if _leads_to_matmul(x):
                next_var, pattern.mask_var = x, y
            elif _leads_to_matmul(y):
                next_var, pattern.mask_var = y, x
            else:
                return False
            pattern.mask_kind = "add"
            order.append("mask")
        elif op_type in ("mul", "real_div") and pattern.scale is None:
            if op_type == "real_div":
                divisor = uniform_scalar_value(op.inputs["y"])
                if divisor is None or divisor <= 0:
                    return False
                pattern.scale = 1.0 / divisor
                next_var = op.inputs["x"]
            else:
                operand = uniform_const_operand(op)
                if operand is None or operand[0] <= 0:
                    return False
                pattern.scale, next_var = operand
            order.append("scale")
        else:
            return False

        if op_type not in ("cast", *_LAYOUT_OPS) and not shapes_equal(op.outputs[0].shape, next_var.shape):
            # A mask or scale that broadcasts the scores up would change the
            # score layout; only element-wise application is supported.
            return False

        consumer = op
        var = next_var

    pattern.back_layout_ops = list(reversed(back_ops))
    # The op encountered first walking backwards is the one applied last.
    pattern.mask_before_scale = order[:2] == ["scale", "mask"]
    return True


def _match_forward(softmax_op, pattern, safe_softmax=None) -> bool:
    """Walk from the softmax output forward to ``matmul_1``.

    When ``safe_softmax`` is given the walk starts after its ``select`` instead:
    the wrapper has already been checked to be an identity here.
    """
    var = softmax_op.outputs[0] if safe_softmax is None else safe_softmax.select_op.outputs[0]
    fwd_ops = []
    n_casts = 0
    while True:
        consumer = sole_consumer(var)
        if consumer is None:
            return False
        if consumer.op_type == "matmul":
            pattern.matmul_1 = consumer
            pattern.weights_var = var
            break
        if consumer.op_type == "cast":
            if n_casts >= 2:
                return False
            n_casts += 1
        elif consumer.op_type in _LAYOUT_OPS:
            fwd_ops.append(consumer)
        else:
            return False
        var = consumer.outputs[0]

    pattern.fwd_layout_ops = fwd_ops
    return True


def _analyse_space(pattern) -> _Space | None:
    """Work out the S/L roles and the row permutation between the two matmuls."""
    scores_shape = tuple(pattern.matmul_0.outputs[0].shape)
    n_batch = len(scores_shape) - 2
    if n_batch < 1:
        return None

    if not pattern.back_layout_ops and not pattern.fwd_layout_ops:
        # Softmax runs directly on the matmul result, so S is its last axis.
        return _Space(n_batch, scores_shape[:n_batch])

    if any(is_symbolic(dim) for dim in scores_shape):
        return None

    atoms = [_Atom(dim) for dim in scores_shape]
    batch_atoms, m_atom, n_atom = atoms[:n_batch], atoms[-2], atoms[-1]

    layout = _track_layout([[atom] for atom in atoms], pattern.back_layout_ops)
    if not layout or len(layout[-1]) != 1:
        return None

    s_atom = layout[-1][0]
    if s_atom is m_atom:
        l_atom = n_atom
    elif s_atom is n_atom:
        l_atom = m_atom
    else:
        # Softmax reduces over a batch axis, or over only part of a result axis.
        return None

    weights_layout = _track_layout(layout, pattern.fwd_layout_ops)
    if weights_layout is None or len(weights_layout) != n_batch + 2:
        return None

    # The forward chain may split axes further, so re-expand every layout to the
    # (now finer) leaves before comparing them.
    if s_atom.children is not None:
        # The key axis itself was split; it is no longer a single SDPA axis.
        return None
    if _expand(weights_layout[-1]) != [s_atom]:
        return None
    for atom, dim in zip(batch_atoms, weights_layout):
        # The batch layout of both matmuls has to agree: SDPA keeps it as is.
        if _expand(dim) != atom.leaves():
            return None

    l_leaves = l_atom.leaves()
    l_prime = _expand(weights_layout[n_batch])
    if len(l_prime) != len(l_leaves) or {id(a) for a in l_prime} != {id(a) for a in l_leaves}:
        return None

    return _Space(
        n_batch,
        scores_shape[:n_batch],
        s_is_first=s_atom is m_atom,
        s_atom=s_atom,
        l_atom=l_atom,
        batch_atoms=batch_atoms,
        softmax_layout=[_expand(dim) for dim in layout],
        l_prime=l_prime,
    )


def _reorder_rows(var, space, source_atoms, target_atoms, trailing_dim, before_op, name):
    """Permute the merged row axis of ``var`` from ``source_atoms`` to ``target_atoms`` order.

    ``var`` has shape ``[batch..., prod(source_atoms), trailing_dim]``.
    """
    position = {id(atom): i for i, atom in enumerate(source_atoms)}
    perm_tail = []
    for atom in target_atoms:
        index = position.get(id(atom))
        if index is None:
            return None
        perm_tail.append(index)
    if perm_tail == list(range(len(source_atoms))):
        return var
    if any(is_symbolic(dim) for dim in space.batch_shape) or trailing_dim is None:
        return None

    batch = [int(dim) for dim in space.batch_shape]
    n_batch = len(batch)
    total = 1
    for atom in source_atoms:
        total *= atom.size

    split = mb.reshape(
        x=var,
        shape=batch + [atom.size for atom in source_atoms] + [int(trailing_dim)],
        before_op=before_op,
        name=name + "_split",
    )
    perm = list(range(n_batch)) + [n_batch + p for p in perm_tail] + [n_batch + len(source_atoms)]
    permuted = mb.transpose(x=split, perm=perm, before_op=before_op, name=name + "_perm")
    return mb.reshape(
        x=permuted,
        shape=batch + [total, int(trailing_dim)],
        before_op=before_op,
        name=name + "_merge",
    )


def _mask_to_sdpa_space(mask_var, space, softmax_shape, seq_len, before_op, name):
    """Bring a mask expressed in softmax space into SDPA's ``[..., L, S]`` space."""
    mask_shape = tuple(mask_var.shape)
    softmax_shape = tuple(softmax_shape)
    sdpa_rank = space.n_batch + 2
    if len(mask_shape) == 0 or not dims_equal(mask_shape[-1], seq_len):
        return None
    if len(mask_shape) > len(softmax_shape):
        return None
    if not shapes_equal(broadcast_shapes(mask_shape, softmax_shape), softmax_shape):
        return None

    if space.is_score_space:
        return mask_var if len(mask_shape) <= sdpa_rank else None

    padded = (1,) * (len(softmax_shape) - len(mask_shape)) + mask_shape

    if all(dims_equal(dim, 1) for dim in padded[:-1]):
        # A pure key mask: identical for every query row.
        return mb.reshape(
            x=mask_var,
            shape=[1] * (sdpa_rank - 1) + [int(seq_len)],
            before_op=before_op,
            name=name + "_key",
        )

    if any(is_symbolic(dim) for dim in softmax_shape) or any(is_symbolic(dim) for dim in padded):
        return None
    if any(is_symbolic(dim) for dim in space.batch_shape):
        return None

    var = mask_var
    if len(mask_shape) != len(padded):
        var = mb.reshape(x=var, shape=list(padded), before_op=before_op, name=name + "_expand")
    reps = []
    for target, current in zip(softmax_shape, padded):
        if int(current) == int(target):
            reps.append(1)
        elif int(current) == 1:
            reps.append(int(target))
        else:
            return None
    if any(rep != 1 for rep in reps):
        var = mb.tile(x=var, reps=reps, before_op=before_op, name=name + "_tile")

    # Split softmax space into one axis per atom, then reorder to [batch..., L, S].
    flat_atoms = [atom for dim in space.softmax_layout for atom in dim]
    var = mb.reshape(
        x=var,
        shape=[atom.size for atom in flat_atoms],
        before_op=before_op,
        name=name + "_split",
    )
    position = {id(atom): i for i, atom in enumerate(flat_atoms)}
    perm = []
    for atom in space.sdpa_atom_order():
        index = position.get(id(atom))
        if index is None:
            return None
        perm.append(index)
    if len(perm) != len(flat_atoms):
        return None
    if perm != list(range(len(flat_atoms))):
        var = mb.transpose(x=var, perm=perm, before_op=before_op, name=name + "_perm")
    total = 1
    for atom in space.l_leaves:
        total *= atom.size
    return mb.reshape(
        x=var,
        shape=[int(dim) for dim in space.batch_shape] + [total, int(seq_len)],
        before_op=before_op,
        name=name + "_merge",
    )


def _build_mask(pattern, space, before_op, seq_len, query, finite_fill_mask):
    """Build SDPA's ``attn_mask`` from the matched ``select`` / additive mask."""
    softmax_op = pattern.softmax_op
    mask_var = pattern.mask_var
    # Peel the broadcast tile the converter emits for `select`; the compact
    # condition is what we want to re-layout. A tile that replicates a dimension
    # that is not 1 is a real tile, not a broadcast, and must be kept.
    mask_var = _peel_broadcast_tile(mask_var)

    mask = _mask_to_sdpa_space(
        mask_var, space, softmax_op.x.shape, seq_len, before_op, softmax_op.name + "_mask"
    )
    if mask is None:
        return None

    if pattern.mask_kind == "add":
        if pattern.mask_before_scale and pattern.scale is not None:
            mask = mb.mul(
                x=mask,
                y=_np_scalar(float(pattern.scale), query.dtype),
                before_op=before_op,
                name=softmax_op.name + "_mask_scaled",
            )
        return mask

    if pattern.mask_negated:
        mask = mb.logical_not(x=mask, before_op=before_op, name=softmax_op.name + "_mask_not")

    fill = float(pattern.mask_fill)
    if pattern.mask_before_scale and pattern.scale is not None:
        fill *= float(pattern.scale)

    if math.isinf(fill) or finite_fill_mask == "bool":
        # A -inf fill and SDPA's boolean mask have identical semantics.
        return mask

    # A finite fill (-1e9, torch's `finfo(dtype).min`, ...) keeps a fully masked
    # row finite -- softmax turns it into a uniform row -- while a boolean mask
    # would turn it into NaN. Reproduce it as an additive mask.
    float_mask = mb.cast(
        x=mask,
        dtype=types.builtin_to_string(query.dtype),
        before_op=before_op,
        name=softmax_op.name + "_mask_f",
    )
    inverted = mb.sub(
        x=_np_scalar(1.0, query.dtype),
        y=float_mask,
        before_op=before_op,
        name=softmax_op.name + "_mask_inv",
    )
    return mb.mul(
        x=inverted,
        y=_np_scalar(fill, query.dtype),
        before_op=before_op,
        name=softmax_op.name + "_mask_add",
    )


def _validate_operands(query, key, value, n_batch: int) -> bool:
    for operand in (query, key, value):
        if operand.shape is None or len(operand.shape) != n_batch + 2:
            return False
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return False
    # SDPA does not broadcast the batch dimensions.
    if not shapes_equal(query.shape[:-2], key.shape[:-2]):
        return False
    if not shapes_equal(query.shape[:-2], value.shape[:-2]):
        return False
    if not dims_equal(query.shape[-1], key.shape[-1]):
        return False
    return dims_equal(key.shape[-2], value.shape[-2])


def _try_fuse(softmax_op, block, finite_fill_mask: str) -> bool:
    out_rank = len(softmax_op.outputs[0].shape)
    if normalize_axis(softmax_op.axis.val, out_rank) != out_rank - 1:
        return False

    # Torch lowers `_safe_softmax` literally; peel the wrapper when it cannot
    # make a difference. Its condition reads the scores, so the backward walk
    # has to know about those ops before it checks for escaping consumers.
    safe_softmax = _match_safe_softmax(softmax_op)
    ignored = frozenset(id(op) for op in safe_softmax.readers) if safe_softmax is not None else frozenset()

    pattern = _Pattern(softmax_op)
    if not _match_backward(softmax_op, pattern, ignored):
        return False
    if safe_softmax is not None and not _safe_softmax_is_redundant(pattern):
        return False
    if not _match_forward(softmax_op, pattern, safe_softmax):
        return False

    matmul_0, matmul_1 = pattern.matmul_0, pattern.matmul_1
    if bool(matmul_0.transpose_x.val) or bool(matmul_1.transpose_x.val):
        return False

    space = _analyse_space(pattern)
    if space is None:
        return False

    # -- the two operands of matmul_0 ---------------------------------------
    lhs, rhs = matmul_0.inputs["x"], matmul_0.inputs["y"]
    rhs_transposed = not bool(matmul_0.transpose_y.val)
    if space.s_is_first:
        key_src, key_t = lhs, False
        query_src, query_t = rhs, rhs_transposed
    else:
        query_src, query_t = lhs, False
        key_src, key_t = rhs, rhs_transposed

    # -- the value operand of matmul_1 --------------------------------------
    if matmul_1.inputs["x"] is pattern.weights_var:
        value_src = matmul_1.inputs["y"]
        value_t = bool(matmul_1.transpose_y.val)
        output_transposed = False
    elif matmul_1.inputs["y"] is pattern.weights_var:
        if not bool(matmul_1.transpose_y.val):
            # The contracting axis would not be the last one of the weights.
            return False
        value_src = matmul_1.inputs["x"]
        value_t = True
        output_transposed = True
    else:
        return False

    query = _as_sequence_major(query_src, query_t, matmul_1, softmax_op.name + "_query")
    key = _as_sequence_major(key_src, key_t, matmul_1, softmax_op.name + "_key")
    value = _as_sequence_major(value_src, value_t, matmul_1, softmax_op.name + "_value")

    if not _validate_operands(query, key, value, space.n_batch):
        return False

    embedding = query.shape[-1]
    if is_symbolic(embedding):
        return False
    embedding = int(embedding)
    seq_len = key.shape[-2]

    attn_mask = None
    if pattern.mask_kind is not None:
        attn_mask = _build_mask(pattern, space, matmul_1, seq_len, query, finite_fill_mask)
        if attn_mask is None:
            return False

    # -- scale: SDPA always divides by sqrt(E) ------------------------------
    scale = 1.0 if pattern.scale is None else float(pattern.scale)
    factor = scale * math.sqrt(embedding)
    if factor != 1.0:
        query = mb.mul(
            x=query,
            y=_np_scalar(factor, query.dtype),
            before_op=matmul_1,
            name=softmax_op.name + "_query_scaled",
        )

    attention = mb.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        before_op=matmul_1,
        name=softmax_op.name + "_sdpa",
    )

    # -- restore the row order matmul_1 produced ----------------------------
    if not space.is_identity:
        value_dim = value.shape[-1]
        reordered = _reorder_rows(
            attention,
            space,
            space.l_leaves,
            list(space.l_prime),
            None if is_symbolic(value_dim) else int(value_dim),
            matmul_1,
            softmax_op.name + "_sdpa_rows",
        )
        if reordered is None:
            return False
        attention = reordered

    if output_transposed:
        attention = mb.transpose(
            x=attention,
            perm=_swap_last_two(len(attention.shape)),
            before_op=matmul_1,
            name=softmax_op.name + "_sdpa_out",
        )

    old_var = matmul_1.outputs[0]
    if attention.dtype != old_var.dtype:
        attention = mb.cast(
            x=attention,
            dtype=types.builtin_to_string(old_var.dtype),
            before_op=matmul_1,
            name=softmax_op.name + "_sdpa_cast",
        )
    if not shapes_equal(attention.shape, old_var.shape):
        return False

    block.replace_uses_of_var_after_op(anchor_op=matmul_1, old_var=old_var, new_var=attention)
    return True


@block_context_manager
def _fuse_attention_to_sdpa(block, finite_fill_mask: str) -> int:
    fused = 0
    for op in list(block.operations):
        if op.enclosing_block is None:
            continue

        for nested_block in op.blocks:
            fused += _fuse_attention_to_sdpa(nested_block, finite_fill_mask)
        if len(op.blocks) > 0:
            continue

        if op.op_type != "softmax":
            continue
        # `_try_fuse` may build some replacement ops before a late check makes it
        # give up; those are left dead in the block for `dead_code_elimination`.
        if _try_fuse(op, block, finite_fill_mask):
            fused += 1

    return fused


@register_pass(namespace="common")
class fuse_attention_to_sdpa(AbstractGraphPass):
    """
    Fuse ``matmul -> [scale] -> [mask] -> softmax -> matmul`` into
    ``scaled_dot_product_attention``.

    Support options:

    - ``finite_fill_mask``: how to translate a ``select`` whose fill value is a
      large but finite negative number (``-1e9``, ``torch.finfo(dtype).min``).
      ``"additive"`` (the default) emits a float mask so that a fully masked row
      stays finite, exactly like the original graph; ``"bool"`` emits the
      boolean condition instead, which is cheaper but turns such rows into NaN.
      Only a fill of exactly ``-inf`` maps to a boolean mask unconditionally.

    Given:
        %scores = matmul(x=%q, y=%k, transpose_y=True)
        %scaled = mul(x=%scores, y=0.125)
        %masked = select(cond=%m, a=%scaled, b=-1e9)
        %w = softmax(x=%masked, axis=-1)
        %out = matmul(x=%w, y=%v_t, transpose_y=True)

    Result:
        %out = scaled_dot_product_attention(query=%q, key=%k, value=%v, attn_mask=...)
    """

    _finite_fill_mask = "additive"

    @property
    def finite_fill_mask(self) -> str:
        return self._finite_fill_mask

    @finite_fill_mask.setter
    def finite_fill_mask(self, mode: str):
        if mode not in ("additive", "bool"):
            raise ValueError(f"finite_fill_mask must be 'additive' or 'bool', got `{mode}`")
        self._finite_fill_mask = mode

    def apply(self, prog):
        for f in prog.functions.values():
            fused = _fuse_attention_to_sdpa(f, self._finite_fill_mask)
            if fused:
                logger.debug("fuse_attention_to_sdpa: fused %d attention block(s)", fused)
