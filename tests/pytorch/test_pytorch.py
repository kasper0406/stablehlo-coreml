import pytest
from contextlib import contextmanager

import jax
import numpy as np
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir

from tests.utils import run_and_compare_hlo_module, flatten

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
tx = pytest.importorskip("torchax")
tx_export = pytest.importorskip("torchax.export")


def export_to_stablehlo_module(pytorch_model, inputs):
    pytorch_model.eval()
    weights, jax_func = tx.extract_jax(pytorch_model)

    def wrapped_weights_func(inputs):
        out = jax_func(weights, inputs)

        # This is slightly hacky, but sometimes the output is a dict-like object
        # which is not registered with jax for jitting.
        # We will try to convert it to a dict first.
        try:
            out_dict = dict(out)
            return {k: v for k, v in out_dict.items() if isinstance(v, jax.Array)}
        except (TypeError, ValueError):
            return out

    numpy_inputs = tuple([input.detach().numpy() for input in inputs])

    # Export the JIT-ed function
    jax_exported = jax.export.export(jax.jit(wrapped_weights_func))(numpy_inputs)
    stablehlo = jax_exported.mlir_module()

    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(stablehlo, context=context)

    # Use jaxpr to find which inputs are actually used
    # We analyze the un-jitted function to see actual usage inside the body
    jaxpr = jax.make_jaxpr(wrapped_weights_func)(numpy_inputs)
    filtered_inputs = _filter_unused_inputs(jaxpr, inputs)

    return hlo_module, filtered_inputs


def _filter_unused_inputs(jaxpr, inputs):
    """
    Filters inputs based on their usage in the jaxpr.
    JAX export drops unused arguments from the MLIR module, so we need to align our inputs.
    """
    used_input_indices = []
    for i, invar in enumerate(jaxpr.jaxpr.invars):
        # Check if invar is used in any equation
        is_used = False
        for eqn in jaxpr.jaxpr.eqns:
            if invar in eqn.invars:
                is_used = True
                break

        # Check if invar is used as an output
        if not is_used:
            for outvar in jaxpr.jaxpr.outvars:
                if invar == outvar:
                    is_used = True
                    break

        if is_used:
            used_input_indices.append(i)

    return [inputs[i].detach().numpy() for i in used_input_indices]


def evaluate_pytorch_model(model, inputs):
    hlo_module, module_inputs = export_to_stablehlo_module(model, inputs)

    model_outputs = model(*inputs)
    if isinstance(model_outputs, torch.Tensor):
        expected_outputs = [model_outputs.detach().numpy()]
    else:
        # Sort keys to match JAX's behavior (JAX sorts dict keys)
        keys = sorted(model_outputs.keys())
        expected_outputs = [model_outputs[k] for k in keys]

        expected_outputs = [
            output_tensor.detach().numpy() for output_tensor
            in flatten(expected_outputs)
            if isinstance(output_tensor, torch.Tensor)
        ]

    # Sanity check expected outputs to catch uninitialized weights issues
    for i, out in enumerate(expected_outputs):
        abs_max = np.abs(out).max()
        if abs_max > 1e9:
            raise ValueError(f"Output {i} has insanely large values (max: {abs_max:.2e}). "
                             "This likely means the model has uninitialized weights (batch norm explosion)")

        output_range = out.max() - out.min()
        if output_range < 1e-5 and out.size > 1:
            raise ValueError(f"Output {i} has effectively zero range (range: {output_range:.2e}). "
                             "This likely means the model has uninitialized weights")

    # These models are quite big, so tolerances are relaxed
    run_and_compare_hlo_module(hlo_module, module_inputs, expected_outputs, max_complexity=50_000, atol=5e-01, rtol=5e-02)


@contextmanager
def patch_transformers_compiling():
    # Currently the transformers package is not aware of torchax static compilation / tracing.
    # This causes the jax-export to fail: https://github.com/google/torchax/issues/56
    # For now, we patch the transformers package to indicate that we are compiling.
    from unittest.mock import patch
    patches = []
    targets = [
        "transformers.modeling_attn_mask_utils.is_torchdynamo_compiling",
        "transformers.utils.is_torchdynamo_compiling",
        "transformers.modeling_utils.is_torchdynamo_compiling",
    ]

    for target in targets:
        try:
            # Check if module exists before patching
            module_name = target.rsplit(".", 1)[0]
            __import__(module_name)
            p = patch(target, return_value=True)
            p.start()
            patches.append(p)
        except (ImportError, AttributeError):
            pass

    try:
        yield
    finally:
        for p in patches:
            p.stop()


# ==============================================================================
# LLM / NLP Models
# ==============================================================================

def test_tinyllama():
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Use a much smaller config to avoid OOM in CI
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.use_cache = False
    config.torch_dtype = "float16"

    model = AutoModelForCausalLM.from_config(config)

    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt")

    evaluate_pytorch_model(model, (inputs.input_ids, ))


def test_t5_small():
    from transformers import AutoTokenizer, T5Model, AutoConfig

    # Use AutoTokenizer which might fallback to fast tokenizer (no sentencepiece needed if available)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    config = AutoConfig.from_pretrained("t5-small")
    config.num_layers = 2
    config.num_decoder_layers = 2
    config.d_model = 128
    config.d_kv = 32
    config.d_ff = 512
    config.num_heads = 4
    config.use_cache = False
    model = T5Model(config)

    input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids
    attention_mask = torch.ones_like(input_ids)

    # T5Model forward: (input_ids, attention_mask, decoder_input_ids, ...)
    with patch_transformers_compiling():
        evaluate_pytorch_model(model, (input_ids, attention_mask, decoder_input_ids))


def test_distilbert():
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.n_layers = 2
    config.dim = 128
    config.hidden_dim = 512
    config.n_heads = 4
    model = AutoModel.from_config(config)

    inputs = tokenizer("this is a test of distilbert", return_tensors="pt")
    with patch_transformers_compiling():
        evaluate_pytorch_model(model, (inputs.input_ids, inputs.attention_mask))


def test_gpt2():
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.n_layer = 2
    config.n_embd = 128
    config.n_head = 4
    config.use_cache = False
    model = AutoModel.from_config(config)

    input_ids = tokenizer("this is a test of gpt2", return_tensors="pt").input_ids
    evaluate_pytorch_model(model, (input_ids, ))


def test_bert():
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 512
    config.num_attention_heads = 4
    model = AutoModel.from_config(config)

    inputs = tokenizer("this is a test of bert", return_tensors="pt")
    with patch_transformers_compiling():
        evaluate_pytorch_model(model, (inputs.input_ids, inputs.attention_mask))


# ==============================================================================
# Audio Models
# ==============================================================================

def test_whisper_tiny():
    from transformers import AutoModelForSpeechSeq2Seq, AutoConfig
    import torch

    model_name = "openai/whisper-tiny"
    config = AutoConfig.from_pretrained(model_name)
    config.encoder_layers = 2
    config.decoder_layers = 2
    config.d_model = 128
    config.encoder_attention_heads = 4
    config.decoder_attention_heads = 4
    config.use_cache = False
    model = AutoModelForSpeechSeq2Seq.from_config(config)

    # Workaround for torchax issue with tied weights
    for module in model.modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            module.weight = torch.nn.Parameter(module.weight.clone())

    # Generate dummy audio input
    # Whisper expects input_features of shape (batch, feature_size, sequence_length)
    # feature_size=80, sequence_length=3000 (for 30s audio at 100Hz frame rate roughly)
    input_features = torch.randn(1, 80, 3000)
    decoder_input_ids = torch.tensor([[50258]])
    attention_mask = torch.ones((1, 3000))

    # Whisper forward: (input_features, attention_mask, decoder_input_ids, ...)
    with patch_transformers_compiling():
        evaluate_pytorch_model(model, (input_features, attention_mask, decoder_input_ids))


# ==============================================================================
# Vision Models
# ==============================================================================

def test_convnext_tiny():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.convnext_tiny(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_vit_b_16():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.vit_b_16(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_efficientnet_b0():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.efficientnet_b0(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_mobilenet_v3_small():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_densenet121():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.densenet121(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_resnet50():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.resnet50()
    evaluate_pytorch_model(model, inputs)


def test_resnet18():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.resnet18(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_inception_v3():
    inputs = (torch.randn(2, 3, 299, 299), )
    model = torchvision.models.inception_v3(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)


def test_vgg11():
    inputs = (torch.randn(1, 3, 224, 224), )
    model = torchvision.models.vgg11(weights="DEFAULT")
    evaluate_pytorch_model(model, inputs)
