"""The model ladder: architecture facts, parameter counts and LoRA footprints.

Everything the planner needs to size a model is computed from its architecture rather
than from a published "4B" label, so the memory numbers are the model's own and a new
model is one YAML entry away. When the weights are on disk, ``from_hf_config`` reads
``config.json`` and the computed numbers become exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

LADDER_FILE = Path(__file__).with_name("ladder.yaml")

DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class QuantSpec:
    id: str
    label: str
    bytes_per_param: float
    linear_only: bool = True
    trainable: bool = True
    compute_dtype: str = "auto"
    requires_bf16: bool = False
    bandwidth_efficiency: float = 0.3
    #: Tie-breaker when several quantisations of the same model fit. Higher wins.
    preference: int = 0

    @property
    def quantized(self) -> bool:
        return self.bytes_per_param < 2.0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    label: str
    rung: int
    layers: int
    hidden: int
    attn_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tied_embeddings: bool = True
    max_position: int = 32768
    family: str = "unknown"
    quantizations: tuple[str, ...] = ()
    lora_targets: tuple[str, ...] = DEFAULT_LORA_TARGETS
    moe_experts: int = 0
    moe_active_experts: int = 0
    moe_intermediate: int = 0
    expert_offload: bool = False
    draft: bool = False
    verified: bool = False
    notes: tuple[str, ...] = ()

    # --- shapes ---------------------------------------------------------
    @property
    def is_moe(self) -> bool:
        return self.moe_experts > 0

    @property
    def q_dim(self) -> int:
        return self.attn_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def attn_params_per_layer(self) -> int:
        return (
            self.hidden * self.q_dim        # q_proj
            + 2 * self.hidden * self.kv_dim  # k_proj, v_proj
            + self.q_dim * self.hidden       # o_proj
        )

    def _mlp_params(self, experts: int) -> int:
        inter = self.moe_intermediate if self.is_moe else self.intermediate
        per_expert = 3 * self.hidden * inter  # gate, up, down
        if not self.is_moe:
            return per_expert
        return experts * per_expert + self.hidden * self.moe_experts  # + router

    @property
    def mlp_params_per_layer(self) -> int:
        return self._mlp_params(self.moe_experts if self.is_moe else 1)

    @property
    def active_mlp_params_per_layer(self) -> int:
        """What a single token actually reads — the figure that governs decode speed."""
        return self._mlp_params(self.moe_active_experts if self.is_moe else 1)

    # --- parameter counts -----------------------------------------------
    @property
    def linear_params(self) -> int:
        """Parameters in ``nn.Linear`` layers: the only ones bitsandbytes quantises."""
        return self.layers * (self.attn_params_per_layer + self.mlp_params_per_layer)

    @property
    def active_linear_params(self) -> int:
        return self.layers * (self.attn_params_per_layer + self.active_mlp_params_per_layer)

    @property
    def embedding_params(self) -> int:
        copies = 1 if self.tied_embeddings else 2  # embed_tokens (+ lm_head)
        return copies * self.vocab * self.hidden

    @property
    def norm_params(self) -> int:
        return self.layers * 2 * self.hidden + self.hidden

    @property
    def unquantized_params(self) -> int:
        """Embeddings, lm_head and norms stay in 16-bit even in a 4-bit load."""
        return self.embedding_params + self.norm_params

    @property
    def total_params(self) -> int:
        return self.linear_params + self.unquantized_params

    @property
    def params_b(self) -> float:
        return round(self.total_params / 1e9, 2)

    @property
    def active_params_b(self) -> float:
        return round((self.active_linear_params + self.unquantized_params) / 1e9, 2)

    # --- derived sizes --------------------------------------------------
    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        """Grouped-query attention: only the KV heads are cached."""
        return 2 * self.layers * self.kv_dim * dtype_bytes

    def target_dims(self) -> dict[str, tuple[int, int]]:
        inter = self.moe_intermediate if self.is_moe else self.intermediate
        return {
            "q_proj": (self.hidden, self.q_dim),
            "k_proj": (self.hidden, self.kv_dim),
            "v_proj": (self.hidden, self.kv_dim),
            "o_proj": (self.q_dim, self.hidden),
            "gate_proj": (self.hidden, inter),
            "up_proj": (self.hidden, inter),
            "down_proj": (inter, self.hidden),
        }

    def lora_params(self, rank: int, targets: tuple[str, ...] | None = None) -> int:
        """Trainable parameters for one adapter of this rank."""
        dims = self.target_dims()
        chosen = targets or self.lora_targets
        per_layer = sum(rank * (dims[t][0] + dims[t][1]) for t in chosen if t in dims)
        return self.layers * per_layer

    def describe(self) -> str:
        if self.is_moe:
            size = (
                f"{self.params_b}B total / {self.active_params_b}B active, "
                f"{self.moe_active_experts}/{self.moe_experts} experts"
            )
        else:
            size = f"{self.params_b}B"
        return f"{self.label} ({size}, {self.layers} layers, {self.kv_heads} KV heads)"

    # --- construction ---------------------------------------------------
    @classmethod
    def from_hf_config(
        cls, config_path: Path | str, *, spec_id: str | None = None, rung: int = 0
    ) -> "ModelSpec":
        """Build a spec from a real ``config.json``, so the numbers are exact."""
        path = Path(config_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        hidden = int(raw["hidden_size"])
        attn_heads = int(raw["num_attention_heads"])
        head_dim = int(raw.get("head_dim") or hidden // attn_heads)
        experts = int(raw.get("num_experts") or raw.get("num_local_experts") or 0)
        name = spec_id or path.parent.name
        return cls(
            id=name,
            repo=raw.get("_name_or_path", name),
            label=name,
            rung=rung,
            layers=int(raw["num_hidden_layers"]),
            hidden=hidden,
            attn_heads=attn_heads,
            kv_heads=int(raw.get("num_key_value_heads", attn_heads)),
            head_dim=head_dim,
            intermediate=int(raw.get("intermediate_size", 4 * hidden)),
            vocab=int(raw["vocab_size"]),
            tied_embeddings=bool(raw.get("tie_word_embeddings", False)),
            max_position=int(raw.get("max_position_embeddings", 32768)),
            family=str(raw.get("model_type", "unknown")),
            moe_experts=experts,
            moe_active_experts=int(
                raw.get("num_experts_per_tok") or raw.get("num_experts_per_token") or 0
            ),
            moe_intermediate=int(raw.get("moe_intermediate_size") or 0),
            verified=True,
            notes=(f"architecture read from {path}",),
        )


@dataclass
class Ladder:
    models: tuple[ModelSpec, ...]
    quantizations: dict[str, QuantSpec]
    seq_len_options: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
    source: Path | None = None

    def __post_init__(self) -> None:
        self._by_id = {m.id: m for m in self.models}

    def model(self, model_id: str) -> ModelSpec:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise KeyError(
                f"unknown model {model_id!r}; the ladder has: {', '.join(sorted(self._by_id))}"
            ) from None

    def quant(self, quant_id: str) -> QuantSpec:
        try:
            return self.quantizations[quant_id]
        except KeyError:
            raise KeyError(
                f"unknown quantisation {quant_id!r}; known: "
                f"{', '.join(sorted(self.quantizations))}"
            ) from None

    def rungs(self) -> list[ModelSpec]:
        """Models ordered by capability, least capable first."""
        return sorted(self.models, key=lambda m: (m.rung, m.total_params))

    def candidates(self, *, include_drafts: bool = False) -> list[tuple[ModelSpec, QuantSpec]]:
        out = []
        for model in self.rungs():
            if model.draft and not include_drafts:
                # A draft model is still a legitimate main model; it is just tiny.
                pass
            for quant_id in model.quantizations:
                out.append((model, self.quant(quant_id)))
        return out

    def draft_models(self) -> list[ModelSpec]:
        return [m for m in self.rungs() if m.draft]

    def with_overrides(self, specs: list[ModelSpec]) -> "Ladder":
        merged = {m.id: m for m in self.models}
        for spec in specs:
            existing = merged.get(spec.id)
            merged[spec.id] = (
                replace(spec, rung=existing.rung, quantizations=existing.quantizations)
                if existing
                else spec
            )
        return Ladder(tuple(merged.values()), self.quantizations, self.seq_len_options, self.source)


def load_ladder(path: Path | str | None = None) -> Ladder:
    source = Path(path) if path else LADDER_FILE
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}

    quants = {}
    for entry in raw.get("quantizations", []):
        quants[entry["id"]] = QuantSpec(**entry)

    defaults = raw.get("defaults") or {}
    default_targets = tuple(defaults.get("lora_targets") or DEFAULT_LORA_TARGETS)

    models = []
    for entry in raw.get("models", []):
        data = dict(entry)
        data["quantizations"] = tuple(data.get("quantizations") or ())
        data["lora_targets"] = tuple(data.get("lora_targets") or default_targets)
        data["notes"] = tuple(data.get("notes") or ())
        unknown = set(data) - {f.name for f in ModelSpec.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"{source}: model {data.get('id')} has unknown keys: {sorted(unknown)}")
        models.append(ModelSpec(**data))

    missing = {q for m in models for q in m.quantizations} - set(quants)
    if missing:
        raise ValueError(f"{source}: models reference undefined quantisations {sorted(missing)}")

    return Ladder(
        models=tuple(models),
        quantizations=quants,
        seq_len_options=tuple(defaults.get("seq_len_options") or (512, 1024, 2048, 4096, 8192)),
        source=source,
    )
