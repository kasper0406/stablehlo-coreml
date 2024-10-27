import jax
from flax import nnx
import jax.numpy as jnp

from tests.utils import run_and_compare, run_and_compare_specific_input

from tests.flax_blocks import ResidualConv, Encoder, UNet, UNetWithXlstm
from tests.flax_xlstm import sLSTMCell, sLSTMBlock, mLSTMCell, mLSTMBlock, xLSTMModule, xLSTM

from functools import partial


def test_flax_nnx_linear():
    class TestLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.layer = nnx.Linear(in_features=2, out_features=4, rngs=rngs)

        def __call__(self, x):
            return self.layer(x)

    model = TestLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 2)), ))


def test_flax_stacked_linear():
    class TestStackedLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.upscale_layer = nnx.Linear(in_features=2, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs)

            self.hidden_layers = []
            for _ in range(3):  # 3 hidden layers
                self.hidden_layers.append(nnx.Linear(
                    in_features=4,
                    out_features=4,
                    bias_init=nnx.initializers.ones,
                    rngs=rngs
                ))
            self.downscale_layer = nnx.Linear(
                in_features=4,
                out_features=2,
                bias_init=nnx.initializers.ones,
                rngs=rngs
            )

        def __call__(self, x):
            out = self.upscale_layer(x)
            for layer in self.hidden_layers:
                out = layer(out)
            out = self.downscale_layer(out)
            return out

    model = TestStackedLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 2)), ))


def test_flax_stacked_lax_scan():
    class TestStackedLaxScanLinear(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            @nnx.split_rngs(splits=3)  # 3 hidden layers
            @nnx.vmap(axis_size=3)
            def create_hidden_layers(rngs: nnx.Rngs):
                return nnx.Linear(in_features=4, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs)
            self.hidden_layers = create_hidden_layers(rngs)

            self.upscale_layer = nnx.Linear(in_features=2, out_features=4, bias_init=nnx.initializers.ones, rngs=rngs)
            self.downscale_layer = nnx.Linear(in_features=4, out_features=2, bias_init=nnx.initializers.ones, rngs=rngs)

        def __call__(self, x):
            out = self.upscale_layer(x)

            layer_def, layer_states = nnx.split(self.hidden_layers)

            def forward(x, layer_state):
                layer = nnx.merge(layer_def, layer_state)
                x = layer(x)
                return x, None
            out, _ = jax.lax.scan(forward, out, layer_states)

            out = self.downscale_layer(out)
            return out

    model = TestStackedLaxScanLinear(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 2)), ))


def test_flax_convolution():
    class TestConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=2, out_features=1, kernel_size=3, rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = TestConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 8, 2)), ))


def test_flax_grouped_convolution():
    class TestConvolution(nnx.Module):
        def __init__(self, in_features: int, feature_groups: int, rngs=nnx.Rngs):
            self.conv = nnx.Conv(
                in_features=in_features,
                out_features=2 * feature_groups,
                kernel_size=3,
                feature_group_count=feature_groups,
                rngs=rngs
            )

        def __call__(self, x):
            return self.conv(x)

    run_and_compare(nnx.jit(TestConvolution(4, 2, nnx.Rngs(0))), (jnp.zeros((2, 8, 4)), ))
    run_and_compare(nnx.jit(TestConvolution(9, 3, nnx.Rngs(0))), (jnp.zeros((2, 8, 9)), ))


def test_flax_2d_convolution():
    class TestConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=3, out_features=1, kernel_size=(3, 3), rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = TestConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 8, 8, 3)), ))


def test_flax_3d_convolution():
    class TestConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=3, out_features=1, kernel_size=(3, 3, 3), rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = TestConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((2, 8, 8, 8, 3)), ))


def test_flax_stacked_convolution():
    class TestStackedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            @nnx.split_rngs(splits=3)  # 3 hidden layers
            @nnx.vmap(axis_size=3)
            def create_convs(rngs: nnx.Rngs):
                return nnx.Conv(in_features=2, out_features=2, kernel_size=3, rngs=rngs)
            self.conv_layers = create_convs(rngs)

        def __call__(self, x):
            layer_def, layer_states = nnx.split(self.conv_layers)

            def forward(x, layer_state):
                layer = nnx.merge(layer_def, layer_state)
                x = layer(x)
                x = nnx.relu(x)
                return x, None
            out, _ = jax.lax.scan(forward, x, layer_states)
            return out

    model = TestStackedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((3, 8, 2)), ))


def test_flax_transposed_convolution():
    class TestTransposedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=2, out_features=3, kernel_size=4, rngs=rngs)
            self.conv_transpose = nnx.ConvTranspose(in_features=3, out_features=2, kernel_size=3, rngs=rngs)

        def __call__(self, x):
            x = self.conv(x)
            x = self.conv_transpose(x)
            return x

    model = TestTransposedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 8, 2)), ))


def test_flax_transposed_2d_convolution():
    class TestTransposedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=2, out_features=3, kernel_size=(4, 4), rngs=rngs)
            self.conv_transpose = nnx.ConvTranspose(in_features=3, out_features=2, kernel_size=(3, 4), rngs=rngs)

        def __call__(self, x):
            x = self.conv(x)
            x = self.conv_transpose(x)
            return x

    model = TestTransposedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 10, 8, 2)), ))


def test_flax_transposed_3d_convolution():
    class TestTransposedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=2, out_features=3, kernel_size=(4, 4, 4), rngs=rngs)
            self.conv_transpose = nnx.ConvTranspose(in_features=3, out_features=2, kernel_size=(3, 4, 2), rngs=rngs)

        def __call__(self, x):
            x = self.conv(x)
            x = self.conv_transpose(x)
            return x

    model = TestTransposedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 10, 8, 7, 2)), ))


def test_kernel_dilated_conv():
    class DilatedConvolution(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.Conv(in_features=4, out_features=2, kernel_size=4, kernel_dilation=2, rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = DilatedConvolution(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 4, 4)), ))


def test_strided_conv_transpose():
    class StridedConvTranspose(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.conv = nnx.ConvTranspose(in_features=4, out_features=2, kernel_size=3, strides=2, rngs=rngs)

        def __call__(self, x):
            return self.conv(x)

    model = StridedConvTranspose(nnx.Rngs(0))
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 4, 4)), ))


def test_convolution_ranges():
    class ConvModel(nnx.Module):
        def __init__(
            self,
            conv_type,
            in_features: int,
            out_features: int,
            kernel_size: int,
            strides: int,
            dilation: int,
            rngs=nnx.Rngs
        ):
            self.conv = conv_type(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                kernel_dilation=dilation,
                rngs=rngs
            )

        def __call__(self, x):
            return self.conv(x)

    for conv_type in [nnx.Conv, nnx.ConvTranspose]:
        for in_features in [1, 3]:
            for out_features in [1, 3]:
                for kernel_size in [2, 3]:
                    for strides in [2, 3]:
                        for dilation in [2, 3]:
                            model = ConvModel(
                                conv_type=conv_type,
                                in_features=in_features,
                                out_features=out_features,
                                kernel_size=kernel_size,
                                strides=strides,
                                dilation=dilation,
                                rngs=nnx.Rngs(0)
                            )
                            run_and_compare(nnx.jit(model), (jnp.zeros((2, 8, in_features)), ))


def test_flax_residual_conv_module():
    model_upscale = ResidualConv(in_channels=2, out_channels=4, rngs=nnx.Rngs(0))
    model_upscale.eval()
    run_and_compare(nnx.jit(model_upscale), (jnp.zeros((4, 8, 2)), ))

    model_downscale = ResidualConv(in_channels=4, out_channels=2, rngs=nnx.Rngs(0))
    model_downscale.eval()
    run_and_compare(nnx.jit(model_downscale), (jnp.zeros((4, 4, 4)), ))


def test_encoder():
    model = Encoder(num_layers=3, rngs=nnx.Rngs(0))
    model.eval()
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 8, 1)), ))


def test_unet():
    model = UNet(num_layers=3, rngs=nnx.Rngs(0))
    model.eval()
    run_and_compare(nnx.jit(model), (jnp.zeros((4, 8, 1)), ))


def test_slstm_cell():
    batch_size = 2
    hidden_size = 4
    rngs = nnx.Rngs(0)

    model = sLSTMCell(num_cells=hidden_size, rngs=rngs)
    model.eval()
    x = jnp.zeros((batch_size, hidden_size))
    carry = sLSTMCell.init_carry(batch_size, hidden_size, rngs=rngs)
    run_and_compare(nnx.jit(model), (carry, x))


def test_slstm_block():
    batch_size = 2
    hidden_size = 4
    num_heads = 3
    rngs = nnx.Rngs(0)

    model = sLSTMBlock(hidden_size=hidden_size, num_heads=num_heads, rngs=rngs)
    model.eval()
    x = jnp.zeros((batch_size, hidden_size))
    carry = sLSTMBlock.init_carry(batch_size, hidden_size, num_heads, rngs=rngs)
    run_and_compare(nnx.jit(model), (carry, x))


def test_mlstm_cell():
    batch_size = 2
    hidden_size = 4
    rngs = nnx.Rngs(0)

    model = mLSTMCell(hidden_size=hidden_size, rngs=rngs)
    model.eval()
    x = jnp.zeros((batch_size, hidden_size))
    carry = mLSTMCell.init_carry(batch_size, hidden_size, rngs=rngs)
    run_and_compare(nnx.jit(model), (carry, x))


def test_mlstm_block():
    batch_size = 2
    hidden_size = 4
    num_heads = 3
    rngs = nnx.Rngs(0)

    model = mLSTMBlock(hidden_size=hidden_size, num_heads=num_heads, rngs=rngs)
    model.eval()
    x = jnp.zeros((batch_size, hidden_size))
    carry = mLSTMBlock.init_carry(batch_size, hidden_size, num_heads, rngs=rngs)
    run_and_compare(nnx.jit(model), (carry, x))


def test_xlstm_module():
    batch_size = 2
    hidden_size = 4
    num_heads = 2
    num_mlstm = 1
    num_slstm = 1
    rngs = nnx.Rngs(0)

    model = xLSTMModule(hidden_size=hidden_size, num_heads=num_heads, num_mlstm=num_mlstm, num_slstm=num_slstm, rngs=rngs)
    model.eval()
    x = jnp.zeros((batch_size, hidden_size))
    carry = model.init_carry(batch_size, rngs=rngs)
    # The xLSTM module model is quite deep, so we allow more slack in the outputs
    run_and_compare(nnx.jit(model), (carry, x))


def test_xlstm():
    batch_size = 4
    hidden_size = 3
    rngs = nnx.Rngs(0)

    model = xLSTM(hidden_size=hidden_size, num_heads=1, num_layers=1, rngs=rngs)
    carry = model.init_carry(batch_size, rngs=rngs)
    x = jnp.zeros((batch_size, hidden_size))
    model.eval()

    # The xLSTM model is quite deep, so we allow more slack in the outputs
    run_and_compare(nnx.jit(model), (carry, x))


def test_unet_with_xlstm():
    batch_size = 2
    num_conv_layers = 2
    input_size = 2 ** num_conv_layers
    rngs = nnx.Rngs(0)

    model = UNetWithXlstm(num_conv_layers=num_conv_layers, rngs=rngs)
    carry = model.init_carry(batch_size, rngs=rngs)
    x = jnp.zeros((batch_size, input_size, 1))
    model.eval()

    run_and_compare(nnx.jit(model), (carry, x, ))


def test_activations():
    example_input = (jnp.zeros((20,)),)
    run_and_compare(nnx.celu, example_input)
    run_and_compare(nnx.elu, example_input)
    run_and_compare(nnx.gelu, example_input)
    run_and_compare(nnx.glu, example_input)
    run_and_compare(nnx.hard_sigmoid, example_input)
    run_and_compare(nnx.hard_silu, example_input)
    run_and_compare(nnx.hard_swish, example_input)
    run_and_compare(nnx.hard_tanh, example_input)
    run_and_compare(nnx.leaky_relu, example_input)
    run_and_compare(nnx.log_sigmoid, example_input)
    run_and_compare(nnx.log_softmax, example_input)
    run_and_compare(nnx.logsumexp, example_input)
    run_and_compare(nnx.relu, example_input)
    run_and_compare(nnx.selu, example_input)
    run_and_compare(nnx.sigmoid, example_input)
    run_and_compare(nnx.silu, example_input)
    run_and_compare(nnx.soft_sign, example_input)
    run_and_compare(nnx.softmax, example_input)
    run_and_compare(nnx.softplus, example_input)
    run_and_compare(nnx.standardize, example_input)
    run_and_compare(nnx.swish, example_input)
    run_and_compare(nnx.tanh, example_input)

    run_and_compare_specific_input(partial(nnx.one_hot, num_classes=3), (jnp.array([0, 1, 2]), ))
    run_and_compare_specific_input(partial(nnx.one_hot, num_classes=5), (jnp.array([4, 0, 1, 0]), ))


def test_attantion():
    class TestAttention(nnx.Module):
        def __init__(self, rngs=nnx.Rngs):
            self.layer = nnx.MultiHeadAttention(
                num_heads=4,
                in_features=5,
                qkv_features=16,
                decode=False,
                rngs=rngs,
            )

        def __call__(self, q, k, v):
            return self.layer(q, k, v)

    shape = (4, 3, 2, 5)
    input_spec = (jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape))
    run_and_compare(nnx.jit(TestAttention(nnx.Rngs(0))), input_spec)

    @nnx.jit
    def create_masks(length):
        attention_mask = nnx.make_attention_mask(length, length)
        causal_mask = nnx.make_causal_mask(length)
        return nnx.combine_masks(attention_mask, causal_mask)

    run_and_compare(create_masks, (jnp.zeros((5, 20)), ))


# This test currently makes Python crash due to https://github.com/llvm/llvm-project/pull/113064
# def test_embed():
#     model = nnx.Embed(num_embeddings=10, features=5, rngs=nnx.Rngs(0))
#     example_input = (jnp.array([[1, 5, 3], [9, 3, 0]], dtype=jnp.int32), )
#     run_and_compare_specific_input(nnx.jit(model), example_input)


def test_nnx_einsum():
    layer = nnx.Einsum('nta,hab->nthb', (8, 2, 4), (8, 4), rngs=nnx.Rngs(0))
    example_input = (jnp.zeros((16, 11, 2)), )
    run_and_compare(nnx.jit(layer), example_input)


def test_batch_norm_infer():
    layer = nnx.BatchNorm(num_features=10, momentum=0.9, epsilon=1e-5, rngs=nnx.Rngs(0))
    layer.eval()
    example_input = (jnp.zeros((20, 10)), )
    run_and_compare(nnx.jit(layer), example_input)


def test_layer_norm_infer():
    layer = nnx.LayerNorm(num_features=10, rngs=nnx.Rngs(0))
    layer.eval()
    example_input = (jnp.zeros((20, 10)), )
    run_and_compare(nnx.jit(layer), example_input)


def test_rms_norm_infer():
    layer = nnx.RMSNorm(num_features=10, rngs=nnx.Rngs(0))
    layer.eval()
    example_input = (jnp.zeros((20, 10)), )
    run_and_compare(nnx.jit(layer), example_input)


def test_group_norm_infer():
    layer = nnx.GroupNorm(num_features=10, num_groups=2, rngs=nnx.Rngs(0))
    layer.eval()
    example_input = (jnp.zeros((20, 10)), )
    run_and_compare(nnx.jit(layer), example_input)
