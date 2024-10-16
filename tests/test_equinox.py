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
