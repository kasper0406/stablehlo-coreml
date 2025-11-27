import pytest

import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir

from tests.utils import run_and_compare_hlo_module, flatten

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
tx = pytest.importorskip("torchax")
tx_export = pytest.importorskip("torchax.export")


from contextlib import contextmanager

@contextmanager
def patch_transformers_compiling():
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


def export_to_stablehlo_module(pytorch_model, inputs):
    pytorch_model.eval()

    weights, jax_func = tx.extract_jax(pytorch_model)

    @jax.jit
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

    with patch_transformers_compiling():
        jax_exported = jax.export.export(wrapped_weights_func)(tuple([input.detach().numpy() for input in inputs]))
    stablehlo = jax_exported.mlir_module()

    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(stablehlo, context=context)

    return hlo_module


def evaluate_pytorch_model(model, inputs):
    hlo_module = export_to_stablehlo_module(model, inputs)

    module_inputs = [input.numpy() for input in inputs]

    # Check if we need to filter inputs (e.g. if JAX optimized away unused arguments)
    if len(hlo_module.body.operations) > 0:
        main_func = None
        # Find the main function
        for op in hlo_module.body.operations:
            op_name = str(op.name)
            # Check for standard func.func or the weird "main" op produced by jax.export
            if "func.func" in op_name or "main" in op_name:
                # Double check it has arguments
                if len(op.arguments) > 0 or len(op.regions) > 0:
                    main_func = op
                    break
        
        if main_func and len(main_func.arguments) < len(module_inputs):
            # Try to align inputs by shape
            new_inputs = []
            input_idx = 0
            for arg in main_func.arguments:
                arg_shape = tuple(arg.type.shape)
                # Find the next input that matches this shape
                found = False
                for i in range(input_idx, len(module_inputs)):
                    input_shape = module_inputs[i].shape
                    if input_shape == arg_shape:
                        new_inputs.append(module_inputs[i])
                        input_idx = i + 1
                        found = True
                        break
                if not found:
                    # Fallback: if we can't match by shape, maybe we shouldn't filter?
                    # But if counts mismatch, we must do something.
                    raise RuntimeError(f"Could not find input matching shape {arg_shape} in remaining inputs")
            module_inputs = new_inputs
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

    # These models are quite big, so tolerances are relaxed
    run_and_compare_hlo_module(hlo_module, module_inputs, expected_outputs, max_complexity=50_000, atol=5e-01, rtol=5e-02)


def test_resnet18():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_inception_v3():
    inputs = (torch.randn(4, 3, 299, 299), )
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_vgg16():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_efficientnet_b0():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_gpt2():
    from transformers import AutoModel, AutoTokenizer

    model_name = "gpt2"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer("this is a test of gpt2", return_tensors="pt").input_ids
    evaluate_pytorch_model(model, (input_ids, ))


def test_mobilenet_v3_small():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_vit_b_16():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_bert():
    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer("this is a test of bert", return_tensors="pt")
    evaluate_pytorch_model(model, (inputs.input_ids, inputs.attention_mask))


def test_distilbert():
    from transformers import AutoModel, AutoTokenizer

    model_name = "distilbert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer("this is a test of distilbert", return_tensors="pt")
    evaluate_pytorch_model(model, (inputs.input_ids, inputs.attention_mask))


def test_densenet121():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_resnet50():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_convnext_tiny():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


def test_deeplabv3_resnet50():
    inputs = (torch.randn(4, 3, 224, 224), )
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    evaluate_pytorch_model(model, inputs)


@pytest.mark.xfail(reason="Conversion error (ScatterOp not implemented)")
def test_t5_small():
    from transformers import AutoTokenizer, T5Model

    # Use AutoTokenizer which might fallback to fast tokenizer (no sentencepiece needed if available)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small", use_cache=False)

    input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids
    attention_mask = torch.ones_like(input_ids)
    
    # T5Model forward: (input_ids, attention_mask, decoder_input_ids, ...)
    with patch_transformers_compiling():
        evaluate_pytorch_model(model, (input_ids, attention_mask, decoder_input_ids))


def test_whisper_tiny():
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    import torch

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny", use_cache=False)

    # Workaround for torchax issue with tied weights
    for module in model.modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            module.weight = torch.nn.Parameter(module.weight.clone())

    # Generate dummy audio input
    # Whisper expects input_features of shape (batch, feature_size, sequence_length)
    # feature_size=80, sequence_length=3000 (for 30s audio at 100Hz frame rate roughly)
    input_features = torch.randn(1, 80, 3000)
    decoder_input_ids = torch.tensor([[50258]], dtype=torch.int32) # Start token, use int32 to avoid int16 truncation
    attention_mask = torch.ones((1, 3000), dtype=torch.int32)

    # Whisper forward: (input_features, attention_mask, decoder_input_ids, ...)
    with patch_transformers_compiling():
        evaluate_pytorch_model(model, (input_features, attention_mask, decoder_input_ids))


def test_tinyllama():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    evaluate_pytorch_model(model, (inputs.input_ids, ))
