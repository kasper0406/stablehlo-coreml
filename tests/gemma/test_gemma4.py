import pytest
import jax
import jax.numpy as jnp

from tests.gemma.gemma4_model import Gemma4Config, Gemma4Transformer, AttentionType
from tests.utils import run_and_compare_specific_input


def test_gemma4_tiny():
    """Test a tiny Gemma4 config (E2B attention pattern, minimal dims)."""
    config = Gemma4Config(
        num_embed=100,
        embed_dim=64,
        hidden_dim=256,
        num_heads=2,
        head_dim=32,
        num_kv_heads=1,
        final_logit_softcap=30.0,
        attention_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        sliding_window_size=64,
        per_layer_input_dim=16,
    )

    model = Gemma4Transformer(config=config)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    variables = model.init(jax.random.key(0), tokens=tokens)

    @jax.jit
    def forward(tokens):
        return model.apply(variables, tokens=tokens)

    run_and_compare_specific_input(
        forward, (tokens,),
        max_complexity=50_000,
        atol=5e-03,
        rtol=1e-03,
    )


@pytest.mark.slow
def test_gemma4_e2b():
    """Test a Gemma4 E2B-scale config (35 layers, text-only, random weights)."""
    attention_pattern = (
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
    )
    # Full E2B attention pattern: 7 repeats of the 5-layer pattern = 35 layers
    full_attention_types = attention_pattern * 7

    config = Gemma4Config(
        num_embed=262144,
        embed_dim=1536,
        hidden_dim=6144,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=30.0,
        attention_types=full_attention_types,
        sliding_window_size=512,
        rope_base_frequency=10_000.0,
        global_rope_base_frequency=1_000_000.0,
        per_layer_input_dim=256,
    )

    model = Gemma4Transformer(config=config)
    tokens = jnp.ones((1, 16), dtype=jnp.int32)
    variables = model.init(jax.random.key(0), tokens=tokens)

    @jax.jit
    def forward(tokens):
        return model.apply(variables, tokens=tokens)

    run_and_compare_specific_input(
        forward, (tokens,),
        max_complexity=500_000,
        atol=5e-01,
        rtol=5e-02,
    )


def test_gemma4_moe_tiny():
    """Test a tiny MoE Gemma4 config (26B-A4B pattern, minimal dims)."""
    config = Gemma4Config(
        num_embed=100,
        embed_dim=64,
        hidden_dim=64,
        num_heads=2,
        head_dim=32,
        num_kv_heads=1,
        final_logit_softcap=30.0,
        attention_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        sliding_window_size=64,
        per_layer_input_dim=0,
        enable_moe=True,
        num_experts=4,
        expert_dim=32,
        top_k_experts=2,
        moe_dense_hidden_dim=64,
    )

    model = Gemma4Transformer(config=config)
    tokens = jnp.ones((1, 8), dtype=jnp.int32)
    variables = model.init(jax.random.key(0), tokens=tokens)

    @jax.jit
    def forward(tokens):
        return model.apply(variables, tokens=tokens)

    run_and_compare_specific_input(
        forward, (tokens,),
        max_complexity=50_000,
        atol=5e-03,
        rtol=1e-03,
    )


@pytest.mark.slow
def test_gemma4_moe_26b():
    """Test Gemma4 26B-A4B MoE-scale config (30 layers, random weights)."""
    attention_pattern = (
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.LOCAL_SLIDING,
        AttentionType.GLOBAL,
    )
    # 26B-A4B: 5 repeats of 6-layer pattern = 30 layers
    full_attention_types = attention_pattern * 5

    config = Gemma4Config(
        num_embed=262144,
        embed_dim=2816,
        hidden_dim=2112,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=full_attention_types,
        sliding_window_size=1024,
        rope_base_frequency=10_000.0,
        global_rope_base_frequency=1_000_000.0,
        per_layer_input_dim=0,
        enable_moe=True,
        num_experts=128,
        expert_dim=704,
        top_k_experts=8,
        moe_dense_hidden_dim=2112,
    )

    model = Gemma4Transformer(config=config)
    tokens = jnp.ones((1, 4), dtype=jnp.int32)
    variables = model.init(jax.random.key(0), tokens=tokens)

    @jax.jit
    def forward(tokens):
        return model.apply(variables, tokens=tokens)

    run_and_compare_specific_input(
        forward, (tokens,),
        max_complexity=500_000,
        atol=5e-01,
        rtol=5e-02,
    )
