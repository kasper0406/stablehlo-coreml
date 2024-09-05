from flax import nnx
import jax
import jax.numpy as jnp
from jax._src.typing import Array
from jax._src import core
from jax._src import dtypes

from functools import partial
from typing import Any, Optional, List
import einops

KeyArray = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float


def uniform(minval: RealNumeric = 0,
            maxval: RealNumeric = 1,
            dtype: DTypeLikeInexact = jnp.float_) -> nnx.Initializer:

    def init(key: KeyArray,
             shape: core.Shape,
             dtype: DTypeLikeInexact = dtype) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)

    return init


class sLSTMCell(nnx.Module):
    cell_input_proj: nnx.Linear
    cell_state_proj: nnx.Linear

    input_proj: nnx.Linear
    input_state_proj: nnx.Linear

    forget_gate_proj: nnx.Linear
    forget_state_proj: nnx.Linear

    output_gate_proj: nnx.Linear
    output_state_proj: nnx.Linear

    if_conv: nnx.Conv

    def __init__(self, num_cells: int, rngs: nnx.Rngs, input_size: Optional[int] = None, apply_if_conv: bool = True):
        self.num_cells = num_cells
        construct_x_proj = partial(
            nnx.Linear,
            in_features=input_size if input_size else num_cells,
            out_features=num_cells,
            use_bias=True,
            rngs=rngs,
        )
        construct_hidden_proj = partial(
            nnx.Linear,
            in_features=num_cells,
            out_features=num_cells,
            use_bias=False,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )

        self.cell_input_proj = construct_x_proj()
        self.cell_state_proj = construct_hidden_proj()

        self.input_proj = construct_x_proj(
            bias_init=nnx.initializers.normal(1e-2)
        )
        self.input_state_proj = construct_hidden_proj()

        self.forget_gate_proj = construct_x_proj(
            bias_init=uniform(3.0, 6.0)
        )
        self.forget_state_proj = construct_hidden_proj()

        self.output_gate_proj = construct_x_proj()
        self.output_state_proj = construct_hidden_proj()

        self.if_conv = None
        if apply_if_conv:
            self.if_conv = nnx.Conv(
                in_features=1,
                out_features=1,
                kernel_size=4,
                rngs=rngs
            )

    def __call__(self, carry, x):
        cell_state, hidden_state, normalizer_state, stabilizer_state = carry

        # HACK: See init_carry for an explanation
        cell_state = einops.rearrange(cell_state, "(b h) -> b h", h=self.num_cells)
        hidden_state = einops.rearrange(hidden_state, "(b h) -> b h", h=self.num_cells)
        normalizer_state = einops.rearrange(normalizer_state, "(b h) -> b h", h=self.num_cells)
        stabilizer_state = einops.rearrange(stabilizer_state, "(b h) -> b h", h=self.num_cells)

        # print(f"Hidden state shape: {hidden_state.shape}")
        out = nnx.sigmoid(self.output_gate_proj(x) + self.output_state_proj(hidden_state))

        if_input = x
        if self.if_conv:
            if_input = self.if_conv(if_input[..., jnp.newaxis])
            if_input = jnp.squeeze(nnx.silu(if_input), axis=2)

        i_tilde = self.input_proj(if_input) + self.input_state_proj(hidden_state)
        # TODO: Consider trying a sigmoid forget activation as well!
        f_tilde = self.forget_gate_proj(if_input) + self.forget_state_proj(hidden_state)

        m = jnp.maximum(f_tilde + stabilizer_state, i_tilde)  # Stabilizer state
        i = jnp.exp(i_tilde - m)
        f = jnp.exp(f_tilde + stabilizer_state - m)

        z = nnx.tanh(self.cell_input_proj(x) + self.cell_state_proj(hidden_state))
        c = f * cell_state + i * z  # Cell state

        n = f * normalizer_state + i  # Normalizer state
        h = out * c / n  # Hidden state

        c = einops.rearrange(c, "b h -> (b h)")
        h_not_compacted = h
        h = einops.rearrange(h, "b h -> (b h)")
        n = einops.rearrange(n, "b h -> (b h)")
        m = einops.rearrange(m, "b h -> (b h)")

        return (c, h, n, m), h_not_compacted

    @classmethod
    def init_carry(cls, batch_size: int, num_cells: int, rngs: nnx.Rngs):
        initializer = partial(nnx.initializers.zeros, shape=(batch_size, num_cells), dtype=jnp.float16)
        c = initializer(key=rngs())
        # HACK: CoreML does not supprot tensors of rank > 5, so we squeeze the c carry to make it fit
        c = einops.rearrange(c, "b h -> (b h)")

        h = initializer(key=rngs())
        h = einops.rearrange(h, "b h -> (b h)")

        n = initializer(key=rngs())
        n = einops.rearrange(n, "b h -> (b h)")

        m = initializer(key=rngs())
        m = einops.rearrange(m, "b h -> (b h)")

        return c, h, n, m


class sLSTMBlock(nnx.Module):
    heads: List[sLSTMCell]
    input_norm: nnx.BatchNorm
    output_norm: nnx.BatchNorm

    up_proj_1: nnx.Linear
    up_proj_2: nnx.Linear
    down_proj: nnx.Linear

    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_heads)
        @nnx.vmap(axis_size=num_heads)
        def create_heads(rngs: nnx.Rngs):
            return sLSTMCell(hidden_size, rngs)
        self.heads = create_heads(rngs)

        self.input_norm = nnx.BatchNorm(hidden_size, rngs=rngs)
        self.output_norm = nnx.BatchNorm(hidden_size, rngs=rngs)

        intermediate_size = int(hidden_size * num_heads * 4.0 / 3.0)
        self.up_proj_1 = nnx.Linear(
            in_features=hidden_size * num_heads,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.up_proj_2 = nnx.Linear(
            in_features=hidden_size * num_heads,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            rngs=rngs,
        )

    def __call__(self, carry, x):
        out = self.input_norm(x)

        head_def, head_states = nnx.split(self.heads)

        def eval_heads(head_state, carry, x):
            head = nnx.merge(head_def, head_state)
            return head(carry, x)
        carry, out = jax.vmap(eval_heads, in_axes=(0, 0, None), out_axes=(0, 0))(head_states, carry, out)

        out = self.output_norm(out)
        out = einops.rearrange(out, "heads batch hidden -> batch (heads hidden)")

        out = self.up_proj_1(out) * nnx.gelu(self.up_proj_2(out))
        out = self.down_proj(out)

        out += x  # Residual

        return carry, out

    @classmethod
    def init_carry(cls, batch_size: int, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_heads)
        @nnx.vmap(axis_size=num_heads)
        def create_carry(rngs: nnx.Rngs):
            return sLSTMCell.init_carry(batch_size, hidden_size, rngs)
        return create_carry(rngs)


class mLSTMCell(nnx.Module):
    query_proj: nnx.Linear
    key_proj: nnx.Linear
    value_proj: nnx.Linear
    qk_conv: nnx.Conv

    input_proj: nnx.Linear
    forget_proj: nnx.Linear
    output_proj: nnx.Linear

    learnable_skip: nnx.Linear

    hidden_size: int

    def __init__(self, hidden_size: int, rngs: nnx.Rngs, input_size: Optional[int] = None):
        self.hidden_size = hidden_size

        construct_qkv_proj = partial(
            nnx.Linear,
            in_features=input_size if input_size else hidden_size,
            out_features=hidden_size,
            use_bias=True,
            rngs=rngs,
        )
        construct_x_proj = partial(
            nnx.Linear,
            in_features=input_size if input_size else hidden_size,
            out_features=1,
            use_bias=True,
            rngs=rngs,
        )

        self.query_proj = construct_qkv_proj()
        self.key_proj = construct_qkv_proj()
        self.value_proj = construct_qkv_proj()
        self.qk_conv = nnx.Conv(
            in_features=1,
            out_features=1,
            kernel_size=4,
            rngs=rngs
        )

        self.input_proj = construct_x_proj(
            bias_init=nnx.initializers.normal(1e-2)
        )

        self.forget_proj = construct_x_proj(
            bias_init=uniform(3.0, 6.0)
        )

        self.output_proj = construct_x_proj(out_features=self.hidden_size)

        self.learnable_skip = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
        )

    def __call__(self, carry, x):
        cell_state, normalizer_state, stabilizer_state = carry
        # HACK: See init_carry for an explanation
        cell_state = einops.rearrange(cell_state, "(b h1 h2) -> b h1 h2", h1=self.hidden_size, h2=self.hidden_size)
        normalizer_state = einops.rearrange(normalizer_state, "(b h) -> b h", h=self.hidden_size)

        out = nnx.sigmoid(self.output_proj(x))

        i_tilde = jnp.squeeze(self.input_proj(x), axis=1)
        f_tilde = jnp.squeeze(self.forget_proj(x), axis=1)

        # print(f"i_tilde shape: {i_tilde.shape}")
        # print(f"f_tilde shape: {f_tilde.shape}")
        # print(f"stabilizer_state shape: {stabilizer_state.shape}")

        m = jnp.maximum(f_tilde + stabilizer_state, i_tilde)  # Stabilizer state
        i = jnp.exp(i_tilde - m)
        f = jnp.exp(f_tilde + stabilizer_state - m)

        qk_input = self.qk_conv(x[..., jnp.newaxis])
        qk_input = jnp.squeeze(nnx.silu(qk_input), axis=2)

        key = self.key_proj(qk_input) / jnp.sqrt(self.hidden_size)
        query = self.query_proj(qk_input)
        value = self.value_proj(x)

        # print(f"f shape: {f.shape}")
        c = jnp.einsum("b,bij->bij", f, cell_state) + jnp.einsum("b,bi,bj->bij", i, value, key)
        n = jnp.einsum("b,bi->bi", f, normalizer_state) + jnp.einsum("b,bi->bi", i, key)

        scaler = jnp.abs(jnp.einsum("bi,bi->b", n, query))
        scaler = 1 / jnp.maximum(scaler, 1.0)
        h = out * jnp.einsum("bik,bk,b->bi", c, query, scaler)

        h += self.learnable_skip(qk_input)

        # HACK: See init_carry for an explanation
        c = einops.rearrange(c, "b h1 h2 -> (b h1 h2)")
        n = einops.rearrange(n, "b h -> (b h)")
        return (c, n, m), h

    @classmethod
    def init_carry(cls, batch_size: int, hidden_size: int, rngs: nnx.Rngs):
        c = nnx.initializers.zeros(shape=(batch_size, hidden_size, hidden_size), dtype=jnp.float16, key=rngs())
        # HACK: CoreML does not supprot tensors of rank > 5, so we squeeze the c carry to make it fit
        c = einops.rearrange(c, "b h1 h2 -> (b h1 h2)")

        n = nnx.initializers.zeros(shape=(batch_size, hidden_size), dtype=jnp.float16, key=rngs())
        n = einops.rearrange(n, "b h -> (b h)")

        m = nnx.initializers.zeros(shape=(batch_size,), dtype=jnp.float16, key=rngs())

        return c, n, m


class mLSTMBlock(nnx.Module):
    heads: List[mLSTMCell]
    input_norm: nnx.BatchNorm
    output_norm: nnx.BatchNorm

    up_proj_1: nnx.Linear
    up_proj_2: nnx.Linear
    down_proj: nnx.Linear
    head_polling: nnx.Linear

    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_heads)
        @nnx.vmap(axis_size=num_heads)
        def create_heads(rngs: nnx.Rngs):
            return mLSTMCell(2 * hidden_size, rngs)
        self.heads = create_heads(rngs)

        self.input_norm = nnx.BatchNorm(hidden_size, rngs=rngs)
        self.output_norm = nnx.BatchNorm(2 * hidden_size, rngs=rngs)

        intermediate_size = hidden_size * 2
        self.up_proj_1 = nnx.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.up_proj_2 = nnx.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            rngs=rngs,
        )
        self.head_polling = nnx.Linear(
            in_features=num_heads * intermediate_size,
            out_features=intermediate_size,
            rngs=rngs,
        )

    def __call__(self, carry, x):
        out = self.input_norm(x)

        triggers = nnx.silu(self.up_proj_2(out))
        mlstm_input = self.up_proj_1(out)

        head_def, head_states = nnx.split(self.heads)

        def eval_heads(head_state, carry, x):
            head = nnx.merge(head_def, head_state)
            return head(carry, x)
        carry, mlstm_out = jax.vmap(eval_heads, in_axes=(0, 0, None), out_axes=(0, 0))(head_states, carry, mlstm_input)

        mlstm_out = self.output_norm(mlstm_out)
        mlstm_out = einops.rearrange(mlstm_out, "heads batch hidden -> batch (heads hidden)")
        mlstm_out = self.head_polling(mlstm_out)

        out = mlstm_out * triggers
        out = self.down_proj(out)

        out += x  # residual

        return carry, out

    @classmethod
    def init_carry(cls, batch_size: int, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=num_heads)
        @nnx.vmap(axis_size=num_heads)
        def create_carry(rngs: nnx.Rngs):
            return mLSTMCell.init_carry(batch_size, 2 * hidden_size, rngs)
        return create_carry(rngs)


class xLSTMModule(nnx.Module):
    num_mlstm: int
    num_slstm: int
    hidden_size: int
    num_heads: int

    mlstms: List[mLSTMBlock]
    slstms: List[sLSTMBlock]

    def __init__(self, hidden_size: int, num_heads: int, num_mlstm: int, num_slstm: int, rngs: nnx.Rngs):
        self.num_mlstm = num_mlstm
        self.num_slstm = num_slstm
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mlstms = []
        for _ in range(num_mlstm):
            self.mlstms.append(mLSTMBlock(hidden_size, num_heads, rngs=rngs))

        self.slstms = []
        for _ in range(num_slstm):
            self.slstms.append(sLSTMBlock(hidden_size, num_heads, rngs=rngs))

    def __call__(self, carry, x):
        mlstm_carry, slstm_carry = carry

        out = x

        # Call the mLSTMs
        mlstm_carries = []
        for mlstm, c in zip(self.mlstms, zip(*mlstm_carry)):
            c, out = mlstm(c, out)
            mlstm_carries.append(c)
        mlstm_carry = [jnp.stack(c, axis=0) for c in zip(*mlstm_carries)]

        # Call the sLSTMs
        slstm_carries = []
        for slstm, c in zip(self.slstms, zip(*slstm_carry)):
            c, out = slstm(c, out)
            slstm_carries.append(c)
        slstm_carry = [jnp.stack(c, axis=0) for c in zip(*slstm_carries)]

        return (mlstm_carry, slstm_carry), out

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        @nnx.split_rngs(splits=self.num_mlstm)
        @nnx.vmap(axis_size=self.num_mlstm)
        def create_mlstm_carry(rngs: nnx.Rngs):
            return mLSTMBlock.init_carry(batch_size, self.hidden_size, self.num_heads, rngs)

        @nnx.split_rngs(splits=self.num_slstm)
        @nnx.vmap(axis_size=self.num_slstm)
        def create_slstm_carry(rngs: nnx.Rngs):
            return sLSTMBlock.init_carry(batch_size, self.hidden_size, self.num_heads, rngs)

        return (create_mlstm_carry(rngs), create_slstm_carry(rngs))


class xLSTM(nnx.Module):
    layers: List[xLSTMModule]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        rngs: nnx.Rngs,
        mlstm_per_layer: int = 1,
        slstm_per_layer: int = 1
    ):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(xLSTMModule(hidden_size, num_heads, mlstm_per_layer, slstm_per_layer, rngs))

    def __call__(self, carry, x):
        out = x
        carries = []
        for layer, c in zip(self.layers, carry):
            c, out = layer(c, out)
            carries.append(c)

        return carries, out

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        carries = []
        for layer in self.layers:
            carries.append(layer.init_carry(batch_size, rngs))
        return carries
