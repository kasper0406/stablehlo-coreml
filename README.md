# stablehlo-coreml

Convert [StableHLO](https://github.com/openxla/stablehlo) models into Apple Core ML format.

StableHLO is the portability layer used by ML frameworks like [JAX](https://github.com/jax-ml/jax) and [PyTorch](https://pytorch.org/). This library converts StableHLO programs into Apple's [Core ML](https://developer.apple.com/documentation/coreml) format via [coremltools](https://github.com/apple/coremltools), enabling deployment on Apple hardware (iOS, macOS, etc.).

## Installation

```bash
pip install stablehlo-coreml
```

Requires Python 3.10–3.13 and targets iOS/macOS 18+.

## Supported Frameworks

Models can be exported from any framework that produces StableHLO:

- **JAX / Flax / Equinox** — via `jax.export`
- **PyTorch** — via [torchax](https://github.com/google/torchax) to trace the model into JAX, then `jax.export` to StableHLO

The test suite validates against a broad set of models, including full HuggingFace Transformers such as TinyLlama, T5, DistilBERT, GPT-2, BERT, and Whisper, as well as vision models like ResNet, EfficientNet, ViT, ConvNeXt, and more.

For a real-world example, see [gemma-coreml-chat](https://github.com/kasper0406/gemma-coreml-chat), which exports Google's Gemma 4 model to Core ML using this library.

## Converting a Model

To convert a StableHLO module:

```python
import coremltools as ct
from stablehlo_coreml import StateSpec, build_pass_pipeline, convert

mil_program = convert(hlo_module, minimum_deployment_target=ct.target.iOS18)
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=build_pass_pipeline(),
)
```

`build_pass_pipeline()` returns a fresh `ct.PassPipeline` built from
`ct.PassPipeline.DEFAULT` with the stablehlo-coreml graph passes inserted at the right
places. Pass your own `base` pipeline to customise it, e.g.
`build_pass_pipeline(my_pipeline)`. Set options on the returned pipeline, e.g.
`pipeline.set_options("common::const_elimination", {"skip_const_by_size": "1e2"})`.

### Obtaining a StableHLO Module from JAX

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

For the JAX example to work, you will additionally need to install `absl-py` and `flatbuffers` as dependencies.

## Stateful models

Core ML can keep tensors across model invocations as *state* instead of passing
them in and out every time. Mark those tensors when converting by mapping each
state input to the output that holds its updated value:

```python
def step(cache, x):
    new_cache = cache + x
    return new_cache * x, new_cache

mil_program = convert(
    hlo_module,
    minimum_deployment_target=ct.target.iOS18,
    states={
        "main": {
            "cache": StateSpec(output=1),
        },
    },
)
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=build_pass_pipeline(),
)

state = cml_model.make_state()
y = cml_model.predict({"x": x}, state=state)
# `cache` is updated in place; inspect or reset it with
# state.read_state(...) / state.write_state(...)
```

Inner keys may be argument indices or names, and `StateSpec.output` an output
index or JAX result name. Use `output=None` for read-only state and `name=...`
to override the Core ML state name. A flat `{input: output}` mapping works for
single-function modules.

State tensors must have a static shape and a floating-point dtype (stored as
fp16). They are removed from the model's inputs, and the outputs that update
them are dropped.

See [`tests/test_stateful.py`](tests/test_stateful.py) for multi-step examples.

## Dynamic / symbolic shapes

JAX models exported with symbolic dimensions are supported. Symbolic dims flow
through `GetDimensionSizeOp`, `DynamicBroadcastInDimOp`, `DynamicIotaOp`, and
shape-assertion `CustomCallOp`s automatically, producing CoreML models with
flexible inputs.

```python
import jax
import jax.numpy as jnp
from jax.export import export, symbolic_shape

jax_exported = export(jax.jit(jax_function))(
    jax.ShapeDtypeStruct(symbolic_shape("batch, 4"), jnp.float32),
    jax.ShapeDtypeStruct((4, 3), jnp.float32),
)
```

When converting to a CoreML model, specify `RangeDim` for each symbolic
dimension so the model accepts a range of sizes at inference time:

```python
cml_model = ct.convert(
    mil_program,
    source="milinternal",
    minimum_deployment_target=ct.target.iOS18,
    pass_pipeline=build_pass_pipeline(),
    inputs=[
        ct.TensorType(name="arg0", shape=(ct.RangeDim(1, 2048, 1), 4)),
        ct.TensorType(name="arg1", shape=(4, 3)),
    ],
)
```

See [`tests/test_symbolic_shapes.py`](tests/test_symbolic_shapes.py) for
symbolic matmul, batched einsum, and multi-axis patterns (for example
transformer-style projections).


### Examples in the test suite

The [`tests/`](tests/) directory has end-to-end export and conversion examples:

- **PyTorch (torchax)** — [`tests/pytorch/test_pytorch.py`](tests/pytorch/test_pytorch.py): `export_to_stablehlo_module`, HuggingFace Transformers, and torchvision models.
- **JAX** — [`tests/test_jax.py`](tests/test_jax.py)
- **Flax / Equinox** — [`tests/test_flax.py`](tests/test_flax.py), [`tests/test_equinox.py`](tests/test_equinox.py)

## Development

* `coremltools` supports up to Python 3.13. Do not run hatch with a newer version.
  Can be controlled using e.g. `export HATCH_PYTHON=python3.13`
* Run tests using `hatch run test:pytest tests`

### Formal verification of optimization passes

The repository includes a separate SMT proof suite for the structural
`remove_noop_slice_update` and `remove_broadcast_tiles` passes, plus a
conditional modeled check for `fuse_reduce_keep_dims` and the canonical
`fuse_logit_softcap` multiply subset. Its Z3 lemmas prove index and shape
identities and selected scalar operation contracts, while finite graph fixtures
exercise production MIL implementations and bounded reduction routes. Run it
with:

```bash
hatch run proofs:check
```

The proof environment runs on Linux with Python 3.12 and does not require the
Apple Core ML runtime. It is a proof of modeled contracts and bounded fixtures;
it does not verify every graph mutation, backend behavior, or the
remaining numerical fusion cases. See [`docs/formal-verification.md`](docs/formal-verification.md)
for the pass inventory, trust boundary, and instructions for adding coverage.

The `formal/` pilot adds a Lean kernel check for the abstract full-coverage
`remove_noop_slice_update` rule. Run `(cd formal && lake build)` and
`python scripts/generate_formal_rules.py --check` locally; this verifies the
generated rule and its modeled theorem, while the MIL adapter, code generation,
graph mutation, and backend correspondence remain explicit trust-boundary
assumptions.
