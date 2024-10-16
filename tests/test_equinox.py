import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi

from tests.test_jax import run_and_compare


def test_conv():
    class SimpleConv(eqx.Module):
        conv: eqx.nn.Conv1d

        def __init__(self, key):
            self.conv = eqx.nn.Conv1d(in_channels=2, out_channels=5, kernel_size=3, key=key)

        def __call__(self, x):
            return self.conv(x)

    input_spec = (jnp.zeros((2, 30)), )
    model = SimpleConv(jax.random.PRNGKey(0))
    run_and_compare(eqxi.finalise_fn(model), input_spec)
