from flax import nnx
import jax.numpy as jnp

from typing import List

from .flax_xlstm import xLSTM


class ResidualConv(nnx.Module):
    scale_conv: nnx.Conv
    conv: nnx.Conv
    normalization_1: nnx.Module
    normalization_2: nnx.Module
    normalization_3: nnx.Module
    shortcut: nnx.Conv

    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs, stride: int = 2):
        conv_type = nnx.Conv if in_channels <= out_channels else nnx.ConvTranspose

        kernel_size = 4
        self.scale_conv = conv_type(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(stride,),
            rngs=rngs
        )
        self.conv = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            rngs=rngs,
        )

        self.normalization_1 = nnx.BatchNorm(num_features=out_channels, rngs=rngs)
        self.normalization_2 = nnx.BatchNorm(num_features=out_channels, rngs=rngs)

        self.shortcut = conv_type(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(stride,),
            strides=(stride,),
            rngs=rngs
        )

    def __call__(self, x):
        out = self.scale_conv(x)
        out = self.normalization_1(out)
        out = nnx.silu(out)

        out = self.conv(out)
        out = nnx.silu(out)

        # Residual
        out = out + self.shortcut(x)
        out = self.normalization_2(out)

        return out


class Encoder(nnx.Module):
    cnn_layers: List[ResidualConv]
    normalization: nnx.Module

    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.cnn_layers = []

        for i in range(num_layers):
            in_channels = (2 ** i)
            out_channels = 2 ** (i + 1)

            self.cnn_layers.append(ResidualConv(
                in_channels=in_channels,
                out_channels=out_channels,
                rngs=rngs,
            ))

        last_layer_features = 2 ** num_layers
        self.normalization = nnx.BatchNorm(num_features=last_layer_features, rngs=rngs)

    def __call__(self, x):
        out = x
        skip_connections = []
        for layer in self.cnn_layers:
            out = layer(out)
            skip_connections.append(out)

        out = self.normalization(out)
        out = nnx.tanh(out)

        return out, skip_connections


class Decoder(nnx.Module):
    cnn_layers: List[ResidualConv]
    residual_norm_layers: List[nnx.Module]
    output_polling: nnx.Conv

    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.cnn_layers = []
        self.residual_norm_layers = []

        input_features = 2 ** num_layers
        for i in range(num_layers):
            # Times two to handle residual connections
            in_channels = 2 * (input_features // (2 ** i))
            out_channels = input_features // (2 ** (i + 1))

            self.residual_norm_layers.append(nnx.BatchNorm(in_channels, rngs=rngs))
            self.cnn_layers.append(ResidualConv(
                in_channels=in_channels,
                out_channels=out_channels,
                rngs=rngs,
            ))

        last_layer_features = input_features // (2 ** num_layers)
        self.output_polling = nnx.Conv(
            in_features=last_layer_features,
            out_features=1,
            kernel_size=3,
            rngs=rngs,
        )

    def __call__(self, x, skip_connections):
        skip_connections = list(reversed(skip_connections))

        out = x
        for i, (cnn_layer, residual_norm) in enumerate(zip(self.cnn_layers, self.residual_norm_layers)):
            residual = skip_connections[i]
            out = residual_norm(jnp.concatenate([out, residual], axis=-1))
            out = cnn_layer(out)

        out = self.output_polling(out)
        return out


class UNet(nnx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, num_layers: int, rngs: nnx.Rngs):
        self.audio_encoding = Encoder(num_layers=num_layers, rngs=rngs)
        self.audio_decoding = Decoder(num_layers=num_layers, rngs=rngs)

    def __call__(self, x):
        def compress_dynamic_range(samples):
            mu = jnp.array(255.0, dtype=jnp.float16)
            return jnp.sign(samples) * jnp.log1p(mu * jnp.abs(samples)) / jnp.log1p(mu)
        x = compress_dynamic_range(x)

        hidden, skip_connections = self.audio_encoding(x)
        out = self.audio_decoding(hidden, skip_connections)

        return out


class UNetWithXlstm(nnx.Module):
    encoder: Encoder
    decoder: Decoder
    xlstm: xLSTM

    def __init__(self, num_conv_layers: int, rngs: nnx.Rngs):
        self.hidden_size = 2 ** num_conv_layers

        self.audio_encoding = Encoder(num_layers=num_conv_layers, rngs=rngs)
        self.audio_decoding = Decoder(num_layers=num_conv_layers, rngs=rngs)
        self.xlstm = xLSTM(hidden_size=self.hidden_size, num_heads=1, num_layers=1, rngs=rngs)

    def __call__(self, carry, x):
        if x.shape[1] != self.hidden_size or x.shape[2] != 1:
            raise ValueError("The input must be exactly squeezed to length 1 with x.shape[1] output features")

        def compress_dynamic_range(samples):
            mu = jnp.array(255.0, dtype=jnp.float16)
            return jnp.sign(samples) * jnp.log1p(mu * jnp.abs(samples)) / jnp.log1p(mu)
        x = compress_dynamic_range(x)

        hidden, skip_connections = self.audio_encoding(x)
        hidden = jnp.squeeze(hidden, axis=1)
        carry, hidden = self.xlstm(carry, hidden)
        hidden = jnp.expand_dims(hidden, axis=1)
        out = self.audio_decoding(hidden, skip_connections)

        return carry, out

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        return self.xlstm.init_carry(batch_size, rngs)
