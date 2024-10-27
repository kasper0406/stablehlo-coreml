import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi

from tests.utils import run_and_compare, run_and_compare_specific_input

from functools import partial


def run_and_compare_eqx_specific_input(model, inputs):
    return run_and_compare_specific_input(
        eqxi.finalise_fn(eqx.nn.inference_mode(model)),
        inputs
    )


def run_and_compare_eqx(model, input_spec):
    return run_and_compare(
        eqxi.finalise_fn(eqx.nn.inference_mode(model)),
        input_spec
    )


def test_conv_1d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv1d

        def __init__(self, key):
            self.conv = eqx.nn.Conv1d(in_channels=2, out_channels=5, kernel_size=3, key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare_eqx(model, input_spec)


def test_conv_transpose_1d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.ConvTranspose1d

        def __init__(self, key):
            self.conv = eqx.nn.ConvTranspose1d(in_channels=5, out_channels=2, kernel_size=3, key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((5, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare_eqx(model, input_spec)


def test_conv_2d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv2d

        def __init__(self, key):
            self.conv = eqx.nn.Conv2d(in_channels=2, out_channels=5, kernel_size=(3, 4), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 40, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare_eqx(model, input_spec)


def test_conv_transpose_2d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.ConvTranspose2d

        def __init__(self, key):
            self.conv = eqx.nn.ConvTranspose2d(in_channels=5, out_channels=2, kernel_size=(3, 4), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((5, 40, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare_eqx(model, input_spec)


def test_conv_3d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv3d

        def __init__(self, key):
            self.conv = eqx.nn.Conv3d(in_channels=2, out_channels=5, kernel_size=(3, 4, 2), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 40, 30, 15)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare_eqx(model, input_spec)


def test_conv_transpose_3d():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.ConvTranspose3d

        def __init__(self, key):
            self.conv = eqx.nn.ConvTranspose3d(in_channels=5, out_channels=2, kernel_size=(3, 4, 2), key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((5, 40, 30, 15)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare_eqx(model, input_spec)


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
    run_and_compare_eqx(batched_model, input_spec)


def test_linear():
    model = jax.vmap(eqx.nn.Linear(in_features=10, out_features=20, key=jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((20, 10)), )
    run_and_compare_eqx(model, input_spec)


def test_identity():
    model = jax.vmap(eqx.nn.Identity(key=jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((20, 10)), )
    run_and_compare_eqx(model, input_spec)


def test_gru_cell():
    model = jax.vmap(eqx.nn.GRUCell(input_size=10, hidden_size=24, key=jax.random.PRNGKey(0)))
    input_spec = (jnp.zeros((20, 10)), jnp.zeros((20, 24)))
    run_and_compare_eqx(model, input_spec)


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
    run_and_compare_eqx(model, input_spec)


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
    run_and_compare_eqx(model, input_spec)


def test_prelu():
    model = jax.vmap(eqx.nn.PReLU())
    input_spec = (jnp.zeros((5, 20)), )
    run_and_compare_eqx(model, input_spec)


def test_1d_polling():
    channels = 3
    run_and_compare_eqx(eqx.nn.AvgPool1d(kernel_size=3), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.AvgPool1d(kernel_size=3, stride=2), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.AvgPool1d(kernel_size=3, stride=3), (jnp.zeros((channels, 41, )), ))

    run_and_compare_eqx(eqx.nn.AvgPool1d(kernel_size=3, padding=2), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.AvgPool1d(kernel_size=3, stride=2, padding=3), (jnp.zeros((channels, 41, )), ))

    run_and_compare_eqx(eqx.nn.MaxPool1d(kernel_size=3), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.MaxPool1d(kernel_size=3, stride=2), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.MaxPool1d(kernel_size=3, stride=3), (jnp.zeros((channels, 41, )), ))

    run_and_compare_eqx(eqx.nn.MaxPool1d(kernel_size=3, padding=3), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.MaxPool1d(kernel_size=3, stride=3, padding=2), (jnp.zeros((channels, 41, )), ))

    run_and_compare_eqx(eqx.nn.AdaptiveAvgPool1d(target_shape=4), (jnp.zeros((channels, 41, )), ))
    run_and_compare_eqx(eqx.nn.AdaptiveMaxPool1d(target_shape=5), (jnp.zeros((channels, 41, )), ))

    batch_size = 10
    run_and_compare_eqx(jax.vmap(eqx.nn.AvgPool1d(kernel_size=3)), (jnp.zeros((batch_size, channels, 41, )), ))
    run_and_compare_eqx(jax.vmap(eqx.nn.AvgPool1d(kernel_size=3, stride=2)), (jnp.zeros((batch_size, channels, 41, )), ))
    run_and_compare_eqx(jax.vmap(eqx.nn.AvgPool1d(kernel_size=3, stride=3)), (jnp.zeros((batch_size, channels, 41, )), ))


def test_2d_polling():
    channels = 3
    run_and_compare_eqx(eqx.nn.AvgPool2d(kernel_size=(3, 4)), (jnp.zeros((channels, 41, 21)), ))
    run_and_compare_eqx(eqx.nn.AvgPool2d(kernel_size=(3, 4), stride=(2, 3)), (jnp.zeros((channels, 41, 21)), ))
    run_and_compare_eqx(eqx.nn.AvgPool2d(kernel_size=(3, 4), stride=(3, 2)), (jnp.zeros((channels, 41, 21)), ))

    run_and_compare_eqx(eqx.nn.AvgPool2d(kernel_size=(3, 4), padding=(2, 4)), (jnp.zeros((channels, 41, 21)), ))
    run_and_compare_eqx(eqx.nn.AvgPool2d(kernel_size=(3, 4), padding=(3, 1), stride=(1, 2)), (jnp.zeros((channels, 41, 21)), ))
    run_and_compare_eqx(eqx.nn.AvgPool2d(kernel_size=(3, 4), padding=(1, 2), stride=(3, 2)), (jnp.zeros((channels, 41, 21)), ))

    run_and_compare_eqx(eqx.nn.AdaptiveAvgPool2d(target_shape=(4, 2)), (jnp.zeros((channels, 41, 21)), ))
    run_and_compare_eqx(eqx.nn.AdaptiveMaxPool2d(target_shape=(3, 5)), (jnp.zeros((channels, 41, 21)), ))

    batch_size = 10
    run_and_compare_eqx(jax.vmap(eqx.nn.AvgPool2d(kernel_size=(3, 4))), (jnp.zeros((batch_size, channels, 41, 21)), ))


def test_3d_polling():
    channels = 3
    run_and_compare_eqx(eqx.nn.AvgPool3d(kernel_size=(5, 4, 3)), (jnp.zeros((channels, 41, 21, 10)), ))
    run_and_compare_eqx(eqx.nn.MaxPool3d(kernel_size=(5, 4, 3)), (jnp.zeros((channels, 41, 21, 10)), ))

    # Due to the CoreML rank <= 5 condition, the result can unfortunately not fit in a tensor
    # run_and_compare_eqx(jax.vmap(eqx.nn.AvgPool3d(kernel_size=(5, 4, 3))), (jnp.zeros((10, channels, 41, 21, 10)), ))


def test_layernorm():
    batch_size = 3
    input_shape = (10, 3)
    run_and_compare_eqx(jax.vmap(eqx.nn.LayerNorm(shape=input_shape)), (jnp.zeros((batch_size, *input_shape)), ))


def test_rmsnorm():
    batch_size = 3
    input_shape = (10, 3)
    run_and_compare_eqx(jax.vmap(eqx.nn.RMSNorm(shape=input_shape)), (jnp.zeros((batch_size, *input_shape)), ))


def test_groupnorm():
    batch_size = 3
    input_shape = (4, 12)
    run_and_compare_eqx(
        jax.vmap(eqx.nn.GroupNorm(groups=4, channelwise_affine=False)),
        (jnp.zeros((batch_size, *input_shape)), )
    )
    run_and_compare_eqx(
        jax.vmap(eqx.nn.GroupNorm(groups=2, channels=4)),
        (jnp.zeros((batch_size, *input_shape)), )
    )


# Unfortunately this test currently fails due to https://github.com/llvm/llvm-project/pull/113064
# def test_batchnorm():
#     batch_size = 3
#     input_shape = (4, 12)

#     class Model(eqx.Module):
#         batch_norm: eqx.nn.BatchNorm

#         def __init__(self, wrapping_layer: eqx.Module, key: jax.random.PRNGKey):
#             self.v = eqx.nn.BatchNorm(input_size=4, axis_name="batch")

#         def __call__(self, x, state):
#             out, _state = self.batch_norm(x, state)
#             return out

#     model, state = eqx.nn.make_with_state(eqx.nn.BatchNorm)(input_size=4, axis_name="batch")
#     batched_model = jax.vmap(partial(model, state=state), axis_name="batch")
#     run_and_compare_eqx(batched_model, (jnp.zeros((batch_size, *input_shape)), ))


def test_spectralnorm():
    batch_size = 5

    class Model(eqx.Module):
        spectral_norm: eqx.nn.SpectralNorm[eqx.Module]

        def __init__(self, wrapping_layer: eqx.Module, key: jax.random.PRNGKey):
            self.spectral_norm = eqx.nn.SpectralNorm(
                layer=wrapping_layer,
                weight_name="weight",
                key=key,
            )

        def __call__(self, x, state):
            out, _state = self.spectral_norm(x, state)
            return out

    wrapping_key, model_key = jax.random.split(jax.random.PRNGKey(0), 2)

    # Linear wrapping layer
    model, state = eqx.nn.make_with_state(Model)(
        wrapping_layer=eqx.nn.Linear(in_features=12, out_features=24, key=wrapping_key),
        key=model_key,
    )
    batched_model = jax.vmap(partial(model, state=state))
    run_and_compare_eqx(batched_model, (jnp.zeros((batch_size, 12)), ))

    # Convolutional 1d wrapping layer
    model, state = eqx.nn.make_with_state(Model)(
        wrapping_layer=eqx.nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, key=wrapping_key),
        key=model_key,
    )
    batched_model = jax.vmap(partial(model, state=state))
    run_and_compare_eqx(batched_model, (jnp.zeros((batch_size, 12, 31)), ))

    # Convolutional 2d wrapping layer
    model, state = eqx.nn.make_with_state(Model)(
        wrapping_layer=eqx.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, key=wrapping_key),
        key=model_key,
    )
    batched_model = jax.vmap(partial(model, state=state))
    run_and_compare_eqx(batched_model, (jnp.zeros((batch_size, 12, 31, 15)), ))


def test_weightnorm():
    batch_size = 5
    key = jax.random.PRNGKey(0)

    class Model(eqx.Module):
        weight_norm: eqx.nn.WeightNorm[eqx.Module]

        def __init__(self, wrapping_layer: eqx.Module):
            self.weight_norm = eqx.nn.WeightNorm(
                layer=wrapping_layer,
                weight_name="weight",
            )

        def __call__(self, x):
            return self.weight_norm(x)

    # Linear wrapping layer
    model = jax.vmap(Model(
        wrapping_layer=eqx.nn.Linear(in_features=12, out_features=24, key=key),
    ))
    run_and_compare_eqx(model, (jnp.zeros((batch_size, 12)), ))

    # Convolutional 1d wrapping layer
    model = jax.vmap(Model(
        wrapping_layer=eqx.nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, key=key),
    ))
    run_and_compare_eqx(model, (jnp.zeros((batch_size, 12, 31)), ))

    # Convolutional 2d wrapping layer
    model = jax.vmap(Model(
        wrapping_layer=eqx.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, key=key),
    ))
    run_and_compare_eqx(model, (jnp.zeros((batch_size, 12, 31, 15)), ))


def test_embedding():
    key = jax.random.PRNGKey(0)
    run_and_compare_eqx_specific_input(
        jax.vmap(eqx.nn.Embedding(num_embeddings=5, embedding_size=10, key=key)),
        (jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32), )
    )


def test_mlp():
    model = jax.vmap(eqx.nn.MLP(
        in_size=10,
        out_size=20,
        width_size=30,
        depth=3,
        key=jax.random.PRNGKey(0))
    )
    input_spec = (jnp.zeros((20, 10)), )
    run_and_compare_eqx(model, input_spec)


def test_sequential():
    model = jax.vmap(eqx.nn.Sequential(
        [
            eqx.nn.Linear(in_features=10, out_features=20, key=jax.random.PRNGKey(0)),
            eqx.nn.Lambda(jax.nn.relu),
        ]
    ))
    input_spec = (jnp.zeros((20, 10)), )
    run_and_compare_eqx(model, input_spec)
