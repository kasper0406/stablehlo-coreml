# Convert StableHLO models into Apple Core ML format

**This repo is currently experimental!**

Only a subset of the StableHLO operations have been implemented, and some of them may have restrictions.

Look in the `tests` directory, to see what has currently been tested.

## Converting a model

To convert a StableHLO module, do the following:

```python
import coremltools as ct
from stablehlo_coreml.converter import convert

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
cml_model = ct.convert(mil_program, source="milinternal", minimum_deployment_target=ct.target.iOS18)
```

For a Jax project, the `hlo_module` can be obtained the following way:

```python
import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.experimental import export

context = jax_mlir.make_ir_context()
input_shapes = jnp.zeros((1, 2, 3), dtype=jnp.float16)
jax_exported = export.export(jax.jit(model))(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)
```

For additional examples see the `tests` directory.

## Notes
* `coremltools` supports up to python 3.11. Do not run hatch with a newer version.
  Can be controlled using fx `export HATCH_PYTHON=python3.11`
* Run tests using `hatch run test:pytest tests`
