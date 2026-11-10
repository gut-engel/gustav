from __future__ import annotations

import json

import pytest

from gustav.hw.ladder import ModelSpec, load_ladder


@pytest.fixture(scope="module")
def ladder():
    return load_ladder()


def test_the_shipped_ladder_loads(ladder):
    assert ladder.models and ladder.quantizations
    assert [m.id for m in ladder.rungs()] == sorted(
        (m.id for m in ladder.models), key=lambda i: ladder.model(i).rung
    )


@pytest.mark.parametrize(
    "model_id,expected_b",
    [
        ("qwen3-0.6b", 0.6), ("qwen3-1.7b", 1.7), ("qwen3-4b", 4.0),
        ("qwen3-8b", 8.2), ("qwen3-14b", 14.8), ("qwen3-30b-a3b", 30.5),
    ],
)
def test_computed_parameter_counts_match_the_published_sizes(ladder, model_id, expected_b):
    """The arithmetic is the check on the architecture numbers in ladder.yaml."""
    assert ladder.model(model_id).params_b == pytest.approx(expected_b, abs=0.35)


def test_the_moe_model_has_the_active_size_its_name_claims(ladder):
    moe = ladder.model("qwen3-30b-a3b")
    assert moe.is_moe
    assert moe.active_params_b == pytest.approx(3.0, abs=0.5)  # "A3B"
    assert moe.active_linear_params < moe.linear_params / 5


def test_quantisations_have_a_preference_order(ladder):
    assert ladder.quant("4bit-nf4").preference > ladder.quant("8bit").preference
    assert ladder.quant("8bit").preference > ladder.quant("bf16").preference


def test_bf16_declares_that_it_needs_bf16(ladder):
    assert ladder.quant("bf16").requires_bf16
    assert not ladder.quant("fp16").requires_bf16
    assert not ladder.quant("4bit-nf4").requires_bf16


def test_lora_params_grow_linearly_with_rank(ladder):
    model = ladder.model("qwen3-4b")
    assert model.lora_params(32) == pytest.approx(model.lora_params(16) * 2, rel=0.001)


def test_moe_models_only_adapt_attention(ladder):
    moe = ladder.model("qwen3-30b-a3b")
    assert "gate_proj" not in moe.lora_targets
    # Adapting 128 experts per layer would dwarf the model.
    assert moe.lora_params(16) < ladder.model("qwen3-4b").lora_params(16)


def test_draft_models_are_marked(ladder):
    drafts = {m.id for m in ladder.draft_models()}
    assert "qwen3-0.6b" in drafts
    assert "qwen3-8b" not in drafts


def test_unknown_model_lists_the_options(ladder):
    with pytest.raises(KeyError, match="qwen3-4b"):
        ladder.model("llama-500b")


def test_unknown_quantisation_lists_the_options(ladder):
    with pytest.raises(KeyError, match="4bit-nf4"):
        ladder.quant("3bit")


def test_a_model_referencing_an_undefined_quantisation_is_an_error(tmp_path):
    path = tmp_path / "ladder.yaml"
    path.write_text(
        "quantizations:\n  - {id: bf16, label: x, bytes_per_param: 2.0}\n"
        "models:\n  - id: m\n    repo: r\n    label: m\n    rung: 1\n    layers: 2\n"
        "    hidden: 8\n    attn_heads: 2\n    kv_heads: 1\n    head_dim: 4\n"
        "    intermediate: 16\n    vocab: 100\n    quantizations: [imaginary]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="imaginary"):
        load_ladder(path)


def test_a_typo_in_a_model_entry_is_an_error(tmp_path):
    path = tmp_path / "ladder.yaml"
    path.write_text(
        "quantizations:\n  - {id: bf16, label: x, bytes_per_param: 2.0}\n"
        "models:\n  - id: m\n    repo: r\n    label: m\n    rung: 1\n    layers: 2\n"
        "    hidden: 8\n    attn_heads: 2\n    kv_heads: 1\n    head_dim: 4\n"
        "    intermediate: 16\n    vocab: 100\n    layerz: 3\n    quantizations: [bf16]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="layerz"):
        load_ladder(path)


# --- reading a real config.json -------------------------------------------
def test_from_hf_config_reads_the_architecture(tmp_path, ladder):
    """Once weights are on disk the numbers stop being ladder.yaml's guess."""
    directory = tmp_path / "qwen3-4b"
    directory.mkdir()
    reference = ladder.model("qwen3-4b")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": reference.layers,
                "hidden_size": reference.hidden,
                "num_attention_heads": reference.attn_heads,
                "num_key_value_heads": reference.kv_heads,
                "head_dim": reference.head_dim,
                "intermediate_size": reference.intermediate,
                "vocab_size": reference.vocab,
                "tie_word_embeddings": True,
                "max_position_embeddings": 32768,
            }
        ),
        encoding="utf-8",
    )
    spec = ModelSpec.from_hf_config(directory / "config.json")
    assert spec.verified
    assert spec.id == "qwen3-4b"
    assert spec.params_b == reference.params_b
    assert "architecture read from" in spec.notes[0]


def test_from_hf_config_handles_a_moe_config(tmp_path):
    directory = tmp_path / "moe"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
                "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
                "intermediate_size": 6144, "vocab_size": 151936, "num_experts": 128,
                "num_experts_per_tok": 8, "moe_intermediate_size": 768,
            }
        ),
        encoding="utf-8",
    )
    spec = ModelSpec.from_hf_config(directory / "config.json")
    assert spec.is_moe and spec.moe_active_experts == 8
    assert spec.params_b == pytest.approx(30.5, abs=0.5)


def test_overrides_keep_the_ladder_position(ladder, tmp_path):
    original = ladder.model("qwen3-4b")
    override = ModelSpec(
        id="qwen3-4b", repo="local", label="local 4B", rung=99, layers=36, hidden=2560,
        attn_heads=32, kv_heads=8, head_dim=128, intermediate=9728, vocab=151936,
        verified=True,
    )
    merged = ladder.with_overrides([override])
    assert merged.model("qwen3-4b").verified
    assert merged.model("qwen3-4b").rung == original.rung       # position preserved
    assert merged.model("qwen3-4b").quantizations == original.quantizations
