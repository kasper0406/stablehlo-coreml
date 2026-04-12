# stablehlo-coreml

Convert [StableHLO](https://github.com/openxla/stablehlo) models into Apple Core ML format.

StableHLO is the portability layer used by ML frameworks like [JAX](https://github.com/jax-ml/jax) and [PyTorch](https://pytorch.org/). This library converts StableHLO programs into Apple's [Core ML](https://developer.apple.com/documentation/coreml) format via [coremltools](https://github.com/apple/coremltools), enabling deployment on Apple hardware (iOS, macOS, etc.).

## Installation

```bash
pip install stablehlo-coreml
```

Requires Python 3.9+ and targets iOS/macOS 18+.

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

### Examples in the test suite

The [`tests/`](tests/) directory has end-to-end export and conversion examples:

- **PyTorch (torchax)** — [`tests/pytorch/test_pytorch.py`](tests/pytorch/test_pytorch.py): `export_to_stablehlo_module`, HuggingFace Transformers, and torchvision models.
- **JAX** — [`tests/test_jax.py`](tests/test_jax.py)
- **Flax / Equinox** — [`tests/test_flax.py`](tests/test_flax.py), [`tests/test_equinox.py`](tests/test_equinox.py)

## Development

* `coremltools` supports up to Python 3.13. Do not run hatch with a newer version.
  Can be controlled using e.g. `export HATCH_PYTHON=python3.13`
* Run tests using `hatch run test:pytest tests`
