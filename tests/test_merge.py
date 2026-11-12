"""Folding an adapter into the base checkpoint.

The operation itself is heavy, but what has to be *true* about it is cheap to test: on a
synthetic two-layer model, merging must change the weights, produce the same outputs the
base-plus-adapter produced, leave the architecture untouched, and write a checkpoint that
loads on its own.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="needs the ML extras")
peft = pytest.importorskip("peft", reason="needs the ML extras")
transformers = pytest.importorskip("transformers", reason="needs the ML extras")

from gustav.expansion.merge import MergeError, merge_adapter_into_base  # noqa: E402

VOCAB = 64


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """A real transformers checkpoint, small enough to be free."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    directory = tmp_path_factory.mktemp("base")
    model.save_pretrained(str(directory), safe_serialization=True)
    # A sidecar, so the copy-across is exercised.
    (directory / "generation_config.json").write_text('{"max_new_tokens": 8}', encoding="utf-8")
    return directory


@pytest.fixture
def tiny_adapter(tiny_base, tmp_path):
    """A LoRA with non-zero B, so merging it actually changes the model."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(tiny_base), local_files_only=True)
    peft_model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                   task_type="CAUSAL_LM"),
    )
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.add_(torch.randn_like(parameter) * 0.05)
    directory = tmp_path / "adapter"
    peft_model.save_pretrained(str(directory))
    return directory / "default" if (directory / "default").exists() else directory


def logits_of(directory, ids):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(directory), local_files_only=True)
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_merging_produces_the_adapted_models_outputs(tiny_base, tiny_adapter, tmp_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    ids = torch.tensor([[1, 2, 3, 4]])
    base_logits = logits_of(tiny_base, ids)

    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(str(tiny_base), local_files_only=True),
        str(tiny_adapter),
    )
    adapted.eval()
    with torch.no_grad():
        adapted_logits = adapted(input_ids=ids).logits

    output = tmp_path / "merged"
    report = merge_adapter_into_base(
        base_dir=tiny_base, adapter_dir=tiny_adapter, output_dir=output, dtype="float32"
    )
    merged_logits = logits_of(output, ids)

    assert report["weight_files"], report
    assert not torch.allclose(base_logits, merged_logits), "the merge changed nothing"
    assert torch.allclose(adapted_logits, merged_logits, atol=1e-4), (
        "the merged weights must behave like the base with the adapter attached"
    )


def test_the_architecture_and_sidecars_survive(tiny_base, tiny_adapter, tmp_path):
    output = tmp_path / "merged"
    report = merge_adapter_into_base(
        base_dir=tiny_base, adapter_dir=tiny_adapter, output_dir=output, dtype="float32"
    )
    before = json.loads((tiny_base / "config.json").read_text(encoding="utf-8"))
    after = json.loads((output / "config.json").read_text(encoding="utf-8"))
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"):
        assert before[key] == after[key], f"{key} changed, so the ladder's numbers would lie"
    assert "generation_config.json" in report["sidecars"]
    assert (output / "generation_config.json").exists()
    assert not list(output.glob("adapter_*")), "the merged checkpoint is not an adapter"


def test_a_directory_that_is_not_a_checkpoint_is_refused(tmp_path, tiny_adapter):
    with pytest.raises(MergeError, match="config.json"):
        merge_adapter_into_base(
            base_dir=tmp_path / "nothing", adapter_dir=tiny_adapter,
            output_dir=tmp_path / "out",
        )


def test_a_directory_that_is_not_an_adapter_is_refused(tiny_base, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MergeError, match="adapter"):
        merge_adapter_into_base(
            base_dir=tiny_base, adapter_dir=empty, output_dir=tmp_path / "out"
        )
