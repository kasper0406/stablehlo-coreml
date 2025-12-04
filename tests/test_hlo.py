import jax
import jax.numpy as jnp
from jax.export import export as _jax_export
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir

from tests.utils import run_and_compare_hlo_module


def test_stable_sort_descending():
    input_shape = (8,)
    obj = _jax_export(jnp.argsort)(jnp.zeros(input_shape, dtype=jnp.float32))
    readable = obj.mlir_module()
    assert readable.count("LT") == 1
    modified = readable.replace("LT", "GT")
    context = jax_mlir.make_ir_context()
    descending = ir.Module.parse(modified, context=context)

    test = jax.random.uniform(jax.random.key(0), shape=input_shape)
    expected = jnp.argsort(test, descending=True)

    run_and_compare_hlo_module(descending, (test,), expected)
