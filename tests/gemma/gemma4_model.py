"""Standalone Gemma4-like transformer model for testing StableHLO → CoreML conversion.

This implements the key architectural features of Google DeepMind's Gemma4 models
(https://github.com/google-deepmind/gemma) using pure Flax Linen, without the heavy
gemma package dependency chain.

Supports both dense (E2B) and Mixture-of-Experts (26B-A4B) architectures.

Key Gemma4 features exercised:
- RMSNorm (pre-norm transformer)
- Multi-head attention with Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE) with configurable frequency
- SwiGLU gated feedforward (dense) / GELU-gated feedforward (MoE experts)
- Mixture of Experts with top-k routing, per-expert scaling
- Logit soft-capping
- Per-layer input embeddings (PLE)
- Sliding window + global attention pattern

Using pretrained weights:
  Our tests use random initialization, which is sufficient for testing the
  StableHLO → CoreML conversion pipeline. To use real pretrained weights:

  1. Download via kagglehub:
       import kagglehub
       weights_dir = kagglehub.model_download('google/gemma/Flax/gemma4-e2b')

  2. Load checkpoint with orbax:
       import orbax.checkpoint
       ckptr = orbax.checkpoint.PyTreeCheckpointer()
       params = ckptr.restore(weights_dir)

  3. Apply to model:
       model = Gemma4Transformer(config=e2b_config)
       logits = model.apply({'params': params}, tokens=tokens)

  Note: The param tree structure of this standalone model may differ from
  the official checkpoint. A mapping layer would be needed, or you can use
  the official `gemma` package directly (when its dependency issues are
  resolved).
"""

import dataclasses
from typing import Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp


class AttentionType:
    LOCAL_SLIDING = "local_sliding"
    GLOBAL = "global"


@dataclasses.dataclass(frozen=True)
class Gemma4Config:
    num_embed: int = 100
    embed_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 2
    head_dim: int = 32
    num_kv_heads: int = 1
    final_logit_softcap: float = 30.0
    attention_types: Sequence[str] = (
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
    )
    sliding_window_size: int = 64
    rope_base_frequency: float = 10_000.0
    global_rope_base_frequency: float = 1_000_000.0
    per_layer_input_dim: int = 16

    # MoE configuration (only used when enable_moe=True)
    enable_moe: bool = False
    num_experts: int = 0
    expert_dim: int = 0
    top_k_experts: int = 2
    moe_dense_hidden_dim: int = 0

    @property
    def num_layers(self) -> int:
        return len(self.attention_types)


class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        scale = self.param('scale', nn.initializers.ones, (dim,))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + 1e-6)
        return x * scale


def _apply_rope(x, positions, base_frequency, rope_fraction=1.0):
    """Apply rotary position embeddings to input tensor."""
    head_dim = x.shape[-1]
    rope_dim = int(head_dim * rope_fraction)
    if rope_dim == 0:
        return x

    half_dim = rope_dim // 2
    freq_exponent = 2.0 * jnp.arange(half_dim, dtype=jnp.float32) / rope_dim
    timescale = base_frequency ** freq_exponent

    # positions: (B, L) -> (B, L, 1)
    positions = positions[..., jnp.newaxis].astype(jnp.float32)
    # timescale: (half_dim,)
    sinusoid_inp = positions / timescale[jnp.newaxis, jnp.newaxis, :]

    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]

    x1 = x_rope[..., :half_dim]
    x2 = x_rope[..., half_dim:2 * half_dim]

    # Expand sin/cos for broadcasting with heads: (B, L, 1, half_dim)
    sin = sin[:, :, jnp.newaxis, :]
    cos = cos[:, :, jnp.newaxis, :]

    rotated = jnp.concatenate([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], axis=-1)

    return jnp.concatenate([rotated, x_pass], axis=-1)


class GemmaAttention(nn.Module):
    """Multi-head attention with GQA and RoPE, matching Gemma4 patterns."""
    config: Gemma4Config
    attn_type: str = AttentionType.LOCAL_SLIDING

    @nn.compact
    def __call__(self, x, positions):
        B, L, D = x.shape
        cfg = self.config
        num_heads = cfg.num_heads
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim

        q = nn.Dense(num_heads * head_dim, use_bias=False, name='q_proj')(x)
        k = nn.Dense(num_kv_heads * head_dim, use_bias=False, name='k_proj')(x)
        v = nn.Dense(num_kv_heads * head_dim, use_bias=False, name='v_proj')(x)

        q = q.reshape(B, L, num_heads, head_dim)
        k = k.reshape(B, L, num_kv_heads, head_dim)
        v = v.reshape(B, L, num_kv_heads, head_dim)

        # QK normalization with learnable scale (Gemma4 feature)
        q_norm = RMSNorm(name='q_norm')(q)
        k_norm = RMSNorm(name='k_norm')(k)

        base_freq = (cfg.global_rope_base_frequency
                     if self.attn_type == AttentionType.GLOBAL
                     else cfg.rope_base_frequency)
        q_norm = _apply_rope(q_norm, positions, base_freq)
        k_norm = _apply_rope(k_norm, positions, base_freq)

        # GQA: repeat KV heads to match Q heads
        kv_repeat = num_heads // num_kv_heads
        if kv_repeat > 1:
            k_norm = jnp.repeat(k_norm, kv_repeat, axis=2)
            v = jnp.repeat(v, kv_repeat, axis=2)

        # Scaled dot-product attention: (B, L, H, D) -> (B, H, L, D)
        q_t = jnp.transpose(q_norm, (0, 2, 1, 3))
        k_t = jnp.transpose(k_norm, (0, 2, 1, 3))
        v_t = jnp.transpose(v, (0, 2, 1, 3))

        scale = head_dim ** -0.5
        attn_weights = jnp.matmul(q_t, jnp.swapaxes(k_t, -2, -1)) * scale

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))

        # Sliding window mask for local attention
        if self.attn_type == AttentionType.LOCAL_SLIDING:
            window_mask = jnp.triu(
                jnp.ones((L, L), dtype=jnp.bool_),
                k=-cfg.sliding_window_size + 1
            )
            causal_mask = causal_mask & window_mask

        attn_weights = jnp.where(
            causal_mask[jnp.newaxis, jnp.newaxis, :, :],
            attn_weights,
            jnp.finfo(attn_weights.dtype).min,
        )

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_out = jnp.matmul(attn_weights, v_t)

        # (B, H, L, D) -> (B, L, H*D)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        attn_out = attn_out.reshape(B, L, num_heads * head_dim)

        return nn.Dense(D, use_bias=False, name='o_proj')(attn_out)


class GemmaFeedForward(nn.Module):
    """SwiGLU gated feedforward, matching Gemma4 dense architecture."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.hidden_dim, use_bias=False, name='gate_proj')(x)
        up = nn.Dense(self.hidden_dim, use_bias=False, name='up_proj')(x)
        hidden = nn.silu(gate) * up
        return nn.Dense(x.shape[-1], use_bias=False, name='down_proj')(hidden)


class RMSNormNoScale(nn.Module):
    """RMS normalization without a learnable scale parameter."""
    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + 1e-6)


class MoE(nn.Module):
    """Mixture of Experts with top-k routing, matching Gemma4 26B-A4B.

    Each token is routed to top-k experts via learned router logits.
    Experts use GELU-gated FFW (not SwiGLU). Outputs are weighted-summed.
    """
    features: int
    hidden_dim: int
    num_experts: int
    num_experts_per_tok: int

    @nn.compact
    def __call__(self, x):
        B, L, D = x.shape
        features = self.features
        num_experts = self.num_experts
        k = self.num_experts_per_tok
        hidden = self.hidden_dim

        # Router: RMS-norm, scale, compute logits
        router_norm = RMSNormNoScale(name='router_norm')
        router_input = router_norm(x)
        router_scale = self.param(
            'router_scale', nn.initializers.ones, (features,))
        root_size = jax.lax.rsqrt(
            jnp.array(features, dtype=router_input.dtype))
        router_input = (
            router_input * root_size
            * router_scale.astype(router_input.dtype))
        router_w = self.param(
            'router_logits',
            nn.initializers.normal(stddev=0.02),
            (features, num_experts))
        logits = jnp.dot(router_input, router_w)  # (B, L, E)

        # Top-k routing with softmax probabilities
        logits_f32 = logits.astype(jnp.float32)
        probs = jax.nn.softmax(logits_f32, axis=-1)  # (B, L, E)
        top_k_logits, top_k_indices = jax.lax.top_k(logits_f32, k)  # (B,L,k)

        # Renormalize weights among selected experts
        indicator = jax.nn.one_hot(
            top_k_indices, num_experts, dtype=probs.dtype)  # (B,L,k,E)
        gate_weights = indicator.sum(axis=-2) * probs  # (B, L, E)
        renorm = jnp.sum(gate_weights, axis=-1, keepdims=True)  # (B, L, 1)
        renorm = jnp.where(renorm > 0.0, renorm, 1.0)
        weights = probs / renorm  # (B, L, E)

        # Gather weights for selected experts
        top_k_weights = jnp.take_along_axis(
            weights, top_k_indices, axis=-1)  # (B, L, k)

        # Expert parameters: GELU-gated FFW
        gating_w = self.param(
            'gating_einsum',
            nn.initializers.normal(stddev=0.02),
            (num_experts, 2, hidden, features))
        linear_w = self.param(
            'linear',
            nn.initializers.normal(stddev=0.02),
            (num_experts, hidden, features))
        per_expert_scale = self.param(
            'per_expert_scale', nn.initializers.ones, (num_experts,))

        # Process each expert's tokens via dense matmuls
        output = jnp.zeros_like(x)  # (B, L, D)

        for ki in range(k):
            expert_idx = top_k_indices[:, :, ki]  # (B, L)
            expert_w = top_k_weights[:, :, ki]    # (B, L)

            # Gather params for selected experts
            gate_params = gating_w[expert_idx]  # (B, L, 2, H, D)
            lin_params = linear_w[expert_idx]    # (B, L, H, D)
            escale = per_expert_scale[expert_idx]  # (B, L)

            # GELU-gated FFW: two projections from gate_params
            gate_0 = jnp.einsum(
                'bld,blhd->blh', x, gate_params[:, :, 0, :, :])
            gate_1 = jnp.einsum(
                'bld,blhd->blh', x, gate_params[:, :, 1, :, :])
            activated = nn.gelu(gate_0) * gate_1  # (B, L, H)

            # Project back to embed_dim
            expert_out = jnp.einsum(
                'blh,blhd->bld', activated, lin_params)  # (B, L, D)

            # Scale by per-expert scale and routing weight
            expert_out = expert_out * escale[..., jnp.newaxis]
            expert_out = expert_out * expert_w[..., jnp.newaxis]

            output = output + expert_out

        return output


class GemmaBlock(nn.Module):
    """Pre-norm transformer block with optional per-layer input and MoE."""
    config: Gemma4Config
    attn_type: str = AttentionType.LOCAL_SLIDING

    @nn.compact
    def __call__(self, x, positions, per_layer_input=None):
        cfg = self.config

        # Per-layer input embeddings (PLE) - Gemma4 feature
        if per_layer_input is not None:
            ple_proj = nn.Dense(cfg.embed_dim, use_bias=False, name='ple_proj')
            x = x + ple_proj(per_layer_input)

        # Pre-norm attention
        residual = x
        x = RMSNorm(name='pre_attn_norm')(x)
        x = GemmaAttention(config=cfg, attn_type=self.attn_type,
                           name='attention')(x, positions)
        # Post-attention norm (Gemma4 feature)
        x = RMSNorm(name='post_attn_norm')(x)
        attn_output = residual + x

        if cfg.enable_moe:
            # MoE block: two parallel FFW branches (dense + MoE), summed
            # Dense shared branch (mlp2)
            dense_out = RMSNorm(name='pre_ffw2_norm')(attn_output)
            dense_hidden = cfg.moe_dense_hidden_dim or cfg.hidden_dim
            dense_out = GemmaFeedForward(
                hidden_dim=dense_hidden, name='ffw2')(dense_out)
            dense_out = RMSNorm(name='post_ffw2_norm')(dense_out)

            # MoE branch (mlp)
            moe_in = RMSNorm(name='pre_ffw_norm')(attn_output)
            moe_out = MoE(
                features=cfg.embed_dim,
                hidden_dim=cfg.expert_dim,
                num_experts=cfg.num_experts,
                num_experts_per_tok=cfg.top_k_experts,
                name='moe',
            )(moe_in)
            moe_out = RMSNorm(name='post_ffw1_norm')(moe_out)

            # Combine and post-norm
            ffw_out = dense_out + moe_out
            ffw_out = RMSNorm(name='post_ffw_norm')(ffw_out)
            x = attn_output + ffw_out
        else:
            # Standard dense FFW
            residual = attn_output
            x = RMSNorm(name='pre_ffw_norm')(attn_output)
            x = GemmaFeedForward(hidden_dim=cfg.hidden_dim, name='ffw')(x)
            x = RMSNorm(name='post_ffw_norm')(x)
            x = residual + x

        return x


class Gemma4Transformer(nn.Module):
    """Full Gemma4-like transformer with all distinctive features."""
    config: Gemma4Config

    @nn.compact
    def __call__(self, tokens):
        cfg = self.config
        B, L = tokens.shape

        # Token embedding (shared with output projection via tying)
        embedding_table = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (cfg.num_embed, cfg.embed_dim),
        )
        x = jnp.take(embedding_table, tokens, axis=0)

        # Scale embeddings (Gemma convention)
        x = x * jnp.sqrt(cfg.embed_dim).astype(x.dtype)

        # Positions for RoPE
        positions = jnp.arange(L, dtype=jnp.int32)[jnp.newaxis, :]
        positions = jnp.broadcast_to(positions, (B, L))

        # Per-layer input embeddings (PLE)
        if cfg.per_layer_input_dim > 0:
            ple_embed = self.param(
                'ple_embedding',
                nn.initializers.normal(stddev=0.02),
                (cfg.num_embed, cfg.per_layer_input_dim),
            )
            per_layer_input = jnp.take(ple_embed, tokens, axis=0)
        else:
            per_layer_input = None

        # Transformer blocks
        for i, attn_type in enumerate(cfg.attention_types):
            x = GemmaBlock(
                config=cfg,
                attn_type=attn_type,
                name=f'layer_{i}',
            )(x, positions, per_layer_input)

        # Final norm
        x = RMSNorm(name='final_norm')(x)

        # Output projection (logits) with soft-capping
        logits = jnp.dot(x, embedding_table.T)

        if cfg.final_logit_softcap is not None:
            cap = cfg.final_logit_softcap
            logits = jnp.tanh(logits / cap) * cap

        return logits
