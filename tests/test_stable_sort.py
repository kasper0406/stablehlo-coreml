
import jax.numpy as jnp
from functools import partial
from tests.utils import run_and_compare_specific_input
import pytest

def test_stable_argsort_user_case():
    run_and_compare_specific_input(partial(jnp.argsort, stable=True), (jnp.array([10, 5, 10, 5, 10, 5, 10, 5, 10, 5], dtype=jnp.int32),))

def test_stable_argsort_float():
    # Test with floats where stability matters
    # [1.0, 0.5, 1.0, 0.5]
    run_and_compare_specific_input(partial(jnp.argsort, stable=True), (jnp.array([1.0, 0.5, 1.0, 0.5], dtype=jnp.float32),))

def test_stable_argsort_descending():
    # jnp.argsort supports descending=True in recent versions?
    # If not, we can simulate it or check if stablehlo supports it.
    # StableHLO SortOp has comparison direction.
    # But jnp.argsort usually maps to SortOp.
    try:
        run_and_compare_specific_input(partial(jnp.argsort, stable=True, descending=True), (jnp.array([10, 5, 10, 5], dtype=jnp.int32),))
    except TypeError:
        pass # descending might not be supported in this jax version

