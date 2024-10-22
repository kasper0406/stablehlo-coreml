import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi

from tests.test_jax import run_and_compare


def test_conv_1d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv1d

        def __init__(self, key):
            self.conv = eqx.nn.Conv1d(in_channels=2, out_channels=5, kernel_size=3, key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_conv_transpose_1d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.ConvTranspose1d

        def __init__(self, key):
            self.conv = eqx.nn.ConvTranspose1d(in_channels=5, out_channels=2, kernel_size=3, key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((5, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_conv_2d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv2d

        def __init__(self, key):
            self.conv = eqx.nn.Conv2d(in_channels=2, out_channels=5, kernel_size=(3, 4), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 40, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_conv_transpose_2d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.ConvTranspose2d

        def __init__(self, key):
            self.conv = eqx.nn.ConvTranspose2d(in_channels=5, out_channels=2, kernel_size=(3, 4), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((5, 40, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_conv_3d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv3d

        def __init__(self, key):
            self.conv = eqx.nn.Conv3d(in_channels=2, out_channels=5, kernel_size=(3, 4, 2), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 40, 30, 15)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_conv_transpose_3d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.ConvTranspose3d

        def __init__(self, key):
            self.conv = eqx.nn.ConvTranspose3d(in_channels=5, out_channels=2, kernel_size=(3, 4, 2), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((5, 40, 30, 15)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_odd_batch_dimension():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv1d

        def __init__(self, key):
            self.conv = eqx.nn.Conv1d(in_channels=2, out_channels=5, kernel_size=3, key=key)

        def __call__(self, x):
            return self.conv(x)

    model = SimpleConv(jax.random.PRNGKey(0))
    batched_model = jax.vmap(model, axis_name="batch", in_axes=2, out_axes=2)

    input_spec = (jnp.zeros((2, 30, 5)), )
    run_and_compare(eqxi.finalise_fn(batched_model), input_spec)


def test_linear():
    model = jax.vmap(eqx.nn.Linear(in_features=10, out_features=20, key=jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((20, 10)), )
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_identity():
    model = jax.vmap(eqx.nn.Identity(key=jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((20, 10)), )
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_gru_cell():
    model = jax.vmap(eqx.nn.GRUCell(input_size=10, hidden_size=24, key=jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((20, 10)), jnp.zeros((20, 24)))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_lstm_cell():
    class Model(eqx.Module):
        cell: eqx.nn.LSTMCell

        def __init__(self):
            self.cell = eqx.nn.LSTMCell(input_size=10, hidden_size=24, key=jax.random.PRNGKey(0))

        def __call__(self, xs):
            def scan_fn(state, input):
                return (self.cell(input, state), None)
            init_state = (jnp.zeros(self.cell.hidden_size),
                          jnp.zeros(self.cell.hidden_size))
            (h, c), _ = jax.lax.scan(scan_fn, init_state, xs)
            return h, c

    model = jax.vmap(Model())
    input_spec = (jnp.zeros((20, 30, 10)), )
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_rotary_attention():
    class Model(eqx.Module):
        mha_attention: eqx.nn.MultiheadAttention
        rope_embeddings: eqx.nn.RotaryPositionalEmbedding

        def __init__(self, key: jax.random.PRNGKey):
            attention_key, rope_key = jax.random.split(key, 2)
            self.mha_attention = eqx.nn.MultiheadAttention(
                num_heads=4,
                query_size=24,
                key=attention_key,
            )
            self.rope_embeddings = eqx.nn.RotaryPositionalEmbedding(embedding_size=6)

        def __call__(self, q, k, v):
            def process_heads(key_heads, query_heads, value_heads):
                query_heads = jax.vmap(self.rope_embeddings,
                                       in_axes=1,
                                       out_axes=1)(query_heads)
                key_heads = jax.vmap(self.rope_embeddings,
                                     in_axes=1,
                                     out_axes=1)(key_heads)

                return query_heads, key_heads, value_heads

            x = self.mha_attention(q, k, v, process_heads=process_heads)
            return x

    model = jax.vmap(Model(jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((5, 15, 24)), jnp.zeros((5, 15, 24)), jnp.zeros((5, 15, 24)))
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_prelu():
    model = jax.vmap(eqx.nn.PReLU())
    input_spec = (jnp.zeros((5, 20)), )
    run_and_compare(eqxi.finalise_fn(model), input_spec)


def test_1d_polling():
    channels = 3
    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool1d(kernel_size=3)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool1d(kernel_size=3, stride=2)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool1d(kernel_size=3, stride=3)), (jnp.zeros((channels, 41, )), ))

    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool1d(kernel_size=3, padding=2)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool1d(kernel_size=3, stride=2, padding=3)), (jnp.zeros((channels, 41, )), ))

    run_and_compare(eqxi.finalise_fn(eqx.nn.MaxPool1d(kernel_size=3)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.MaxPool1d(kernel_size=3, stride=2)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.MaxPool1d(kernel_size=3, stride=3)), (jnp.zeros((channels, 41, )), ))

    run_and_compare(eqxi.finalise_fn(eqx.nn.MaxPool1d(kernel_size=3, padding=3)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.MaxPool1d(kernel_size=3, stride=3, padding=2)), (jnp.zeros((channels, 41, )), ))

    run_and_compare(eqxi.finalise_fn(eqx.nn.AdaptiveAvgPool1d(target_shape=4)), (jnp.zeros((channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(eqx.nn.AdaptiveMaxPool1d(target_shape=5)), (jnp.zeros((channels, 41, )), ))

    batch_size = 10
    run_and_compare(eqxi.finalise_fn(jax.vmap(eqx.nn.AvgPool1d(kernel_size=3))), (jnp.zeros((batch_size, channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(jax.vmap(eqx.nn.AvgPool1d(kernel_size=3, stride=2))), (jnp.zeros((batch_size, channels, 41, )), ))
    run_and_compare(eqxi.finalise_fn(jax.vmap(eqx.nn.AvgPool1d(kernel_size=3, stride=3))), (jnp.zeros((batch_size, channels, 41, )), ))


def test_2d_polling():
    channels = 3
    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool2d(kernel_size=(3, 4))), (jnp.zeros((channels, 41, 21)), ))

    batch_size = 10
    run_and_compare(eqxi.finalise_fn(jax.vmap(eqx.nn.AvgPool2d(kernel_size=(3, 4)))), (jnp.zeros((batch_size, channels, 41, 21)), ))


def test_3d_polling():
    channels = 3
    run_and_compare(eqxi.finalise_fn(eqx.nn.AvgPool3d(kernel_size=(5, 4, 3))), (jnp.zeros((channels, 41, 21, 10)), ))

    # Due to the CoreML rank <= 5 condition, the result can unfortunately not fit in a tensor
    # run_and_compare(eqxi.finalise_fn(jax.vmap(eqx.nn.AvgPool3d(kernel_size=(5, 4, 3)))), (jnp.zeros((10, channels, 41, 21, 10)), ))
