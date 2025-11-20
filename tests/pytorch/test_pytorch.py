import pytest

import jax
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

    jax_exported = jax.export.export(wrapped_weights_func)(tuple([input.detach().numpy() for input in inputs]))
    stablehlo = jax_exported.mlir_module()

    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(stablehlo, context=context)

    return hlo_module


def evaluate_pytorch_model(model, inputs):
    hlo_module = export_to_stablehlo_module(model, inputs)

    module_inputs = [input.numpy() for input in inputs]
    model_outputs = model(*inputs)
    if isinstance(model_outputs, torch.Tensor):
        expected_outputs = [model_outputs.detach().numpy()]
    else:
        expected_outputs = [model_outputs[return_name] for return_name in model_outputs]
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
