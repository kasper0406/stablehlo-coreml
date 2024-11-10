import pytest

import jax
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir

from tests.utils import run_and_compare_hlo_module, flatten

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
torch_xla2_export = pytest.importorskip("torch_xla2.export")


def export_to_stablehlo_module(pytorch_model, inputs):
    pytorch_model.eval()

    exported = torch.export.export(pytorch_model, inputs)
    weights, func = torch_xla2_export.exported_program_to_jax(exported)
    jax_avals = torch_xla2_export.extract_avals(exported)

    @jax.jit
    def wrapped_weights_func(*inputs):
        return func(weights, inputs)

    jax_exported = jax.export.export(wrapped_weights_func)((jax_avals,))
    stablehlo = jax_exported.mlir_module()

    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(stablehlo, context=context)

    return hlo_module


def evaluate_pytorch_model(model, inputs):
    model.eval()
    hlo_module = export_to_stablehlo_module(model, inputs)

    module_inputs = [input.numpy() for input in inputs]
    model_outputs = model(*inputs)
    if isinstance(model_outputs, torch.Tensor):
        expected_outputs = [model_outputs.detach().numpy()]
    else:
        expected_outputs = [model_outputs[return_name] for return_name in model_outputs]
        expected_outputs = [output_tensor.detach().numpy() for output_tensor in flatten(expected_outputs)]

    run_and_compare_hlo_module(hlo_module, module_inputs, expected_outputs)


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


# Currently not testable due to https://github.com/llvm/llvm-project/pull/113064
# def test_bert():
#     from transformers import AutoModel, AutoTokenizer
#
#     model_name = "bert-base-uncased"
#     model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     inputs = tokenizer("this is a test of bert", return_tensors="pt")
#     inputs = tuple([inputs[name] for name in inputs])
#     evaluate_pytorch_model(model, inputs)

# Currently not testable due to https://github.com/llvm/llvm-project/pull/113064
# def test_gpt2():
#     from transformers import AutoModel, AutoTokenizer
#
#     model_name = "gpt2"
#     model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     input_ids = tokenizer("this is a test of gpt2", return_tensors="pt").input_ids
#     evaluate_pytorch_model(model, (input_ids, ))
