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

## Apple Neural Engine: avoiding the 1 MiB DMA notch

The Apple Neural Engine has an erratum in its kernel-DMA prefetch ring
([writeup](https://eiln.github.io/posts/ane-dma.html)): whenever the fp16 weight
payload a single core has to stream is an integer multiple of 1 MiB, DRAM
throughput collapses from ~40-60 GB/s to ~17-25 GB/s. The payload of a
weight-streaming op is

```
ceil(N / 16) * D * prod(kernel) * 2 bytes
```

with `N` the output units (output channels / weight rows), `D` the contracting
(input-channel) dimension and 16 the number of ANE cores — so an innocuous
2048x4096 fp16 projection lands on exactly 1 MiB per core.

`common::avoid_ane_dma_notch` rewrites the affected `conv`, `linear` and `matmul`
ops into partial products over chunks of the contracting dimension whose payloads
are *not* near a multiple of 1 MiB, and sums the partials. Enable it with:

```python
pass_pipeline=build_pass_pipeline(avoid_ane_dma_notch=True)
```

Measured on a Mac mini M4 (fp16 1x1 `conv` forced onto the ANE, single token,
medians of 3 rounds x 200 predicts):

| Weight (Cout x Cin) | Per core  | Unsplit | Split          |
| ------------------- | --------- | ------- | -------------- |
| 2048 x 4096         | 1 MiB     | 840 us  | 430 us (2-way) |
| 4096 x 4096         | 2 MiB     | 1142 us | 697 us (4-way) |
| 2048 x 8192         | 2 MiB     | 1350 us | 698 us (4-way) |
| 4096 x 14336        | 7 MiB     | 3522 us | 1989 us (2-way)|
| 2016 x 4096         | 0.98 MiB  | 426 us  | 426 us (control) |

The split column reports the chunk count each measurement used. The pass picks
the *fewest* chunks that take every partial out of the notch, which for a 2 MiB
payload is three near-equal chunks of ~0.67 MiB rather than the four equal
0.5 MiB chunks measured above. The two are not equally far from a multiple of
1 MiB (0.33 MiB against 0.5 MiB), but both are far outside the 16 KiB window
and both stay inside the very first lap of the prefetch ring, where the stall
cannot happen at all — and they measure the same: across three separate runs
the 3-way split the pass emits took 684-730 us on 4096 x 4096 and 685-730 us on
2048 x 8192, against 685-698 us and 695-753 us for the hand-written 4-way one,
with every slice, partial and add staying on the ANE.

On a whole model the effect compounds. A synthetic Llama-3.2-1B-shaped decode
step (16 layers, dim 2048, MLP 8192, 1.95 GB of fp16 weights, every projection
a 1x1 NCHW convolution, single token, everything on the ANE) goes from 76.1 ms
to 31.6 ms per step — 13.1 to 31.6 tokens/s, a 2.4x speedup, at an effective
weight-streaming rate of 25.6 against 61.5 GB/s. The pass split the 48 MLP
projections (2 MiB per core) and correctly left the attention projections
(0.5 and 0.125 MiB per core) alone.

It is opt-in because it is an ANE-only win: on the GPU there is no notch to
avoid in the first place (2048 x 4096 runs in 158 us against 156 us for the
control shape), so the extra slices and adds are pure overhead there, and the
CPU shows no notch either. It also adds ops — a slice per partial and an add
per extra partial, so a k-way split costs k slices and k-1 adds — and changes
fp16 rounding a little, because the accumulation is regrouped into partial
sums.

Only plain fp16 `const` weights with static shapes are split — `constexpr_*`
(palettized/quantized) weights stream different bytes than this model counts,
and grouped convolutions have a per-group contracting dimension the model does
not describe.

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
