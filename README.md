# Convert StableHLO models into Apple Core ML format

**This repo is currently experimental!**

Only a subset of the StableHLO operations have been implemented, and some of them may have restrictions.

Due to the current _dot_general_ op implementation, it is only possible to target iOS >= 18.

Look in the `tests` directory, to see what has currently been tested.

The package is published to PyPi as `stablehlo-coreml-experimental`.

## Converting a model

To convert a StableHLO module, do the following:

```python
import coremltools as ct
from stablehlo_coreml.converter import convert
from stablehlo_coreml import DEFAULT_HLO_PIPELINE

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=DEFAULT_HLO_PIPELINE,
)
```

For a Jax project, the `hlo_module` can be obtained the following way:

```python
import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export

import jax.numpy as jnp

def jax_function(a, b):
    return jnp.einsum("ij,jk -> ik", a, b)

context = jax_mlir.make_ir_context()
input_shapes = (jnp.zeros((2, 4)), jnp.zeros((4, 3)))
jax_exported = export(jax.jit(jax_function))(*input_shapes)
hlo_module = ir.Module.parse(jax_exported.mlir_module(), context=context)
```

For the Jax example to work, you will additionally need to install `absl-py` and `flatbuffers` as dependencies.

For additional examples see the `tests` directory.

## Notes
* `coremltools` supports up to python 3.12. Do not run hatch with a newer version.
  Can be controlled using fx `export HATCH_PYTHON=python3.12`
* Run tests using `hatch run test:pytest tests`
