"""Local model weights: what is on disk, and fetching what is not.

Fetching is the one deliberate exception to "everything is local": it is an explicit
command (``ai model pull``), it runs in a supervised child process so the daemon never
holds a network connection itself, and nothing else ever reaches the network to get
weights. Loading always uses ``local_files_only=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Only the files a transformers checkpoint actually needs.
ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json",
    "tokenizer*", "vocab.json", "merges.txt", "*.model", "special_tokens_map.json",
    "chat_template.*", "preprocessor_config.json",
]
#: Formats we never want: duplicated weights in other serialisations.
IGNORE_PATTERNS = ["*.pth", "*.bin", "*.h5", "*.msgpack", "*.gguf", "original/*", "onnx/*"]


@dataclass
class LocalModel:
    id: str
    path: Path
    size_mb: float
    complete: bool
    missing: list[str] = field(default_factory=list)
    repo: str | None = None
    params_b: float | None = None

    def describe(self) -> str:
        state = "ready" if self.complete else f"INCOMPLETE ({len(self.missing)} file(s) missing)"
        size = f"{self.size_mb / 1024:.2f} GiB" if self.size_mb else "empty"
        return f"{self.id:<16} {size:>10}  {state}"


def directory_size_mb(path: Path) -> float:
    total = 0
    for file in path.rglob("*"):
        if file.is_file() and not file.is_symlink():
            total += file.stat().st_size
    return round(total / (1024 * 1024), 1)


def inspect(path: Path, *, model_id: str | None = None) -> LocalModel:
    from gustav.inference.transformers_backend import missing_weight_files

    missing = missing_weight_files(path) if (path / "config.json").exists() else ["config.json"]
    return LocalModel(
        id=model_id or path.name,
        path=path,
        size_mb=directory_size_mb(path),
        complete=not missing,
        missing=missing,
    )


def list_models(models_dir: Path, ladder=None) -> list[LocalModel]:
    if not models_dir.exists():
        return []
    found: list[LocalModel] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        model = inspect(child)
        if ladder is not None:
            try:
                spec = ladder.model(model.id)
                model.repo = spec.repo
                model.params_b = spec.params_b
            except KeyError:
                pass
        found.append(model)
    return found


def pull_child(ctx, *, repo: str, destination: str, revision: str | None = None) -> dict[str, Any]:
    """Download a checkpoint. Runs in a supervised child process."""
    from huggingface_hub import snapshot_download

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    ctx.heartbeat(f"fetching {repo}")

    class _Progress:
        """huggingface_hub calls tqdm; we turn that into heartbeats."""

        def __init__(self, *args: Any, **kwargs: Any):
            self.total = kwargs.get("total") or 0
            self.n = 0
            self.desc = kwargs.get("desc") or repo

        def update(self, amount: int = 1) -> None:
            self.n += amount
            if self.total:
                ctx.heartbeat(f"{self.desc}: {100 * self.n / self.total:.0f}%")
            else:
                ctx.heartbeat(f"{self.desc}: {self.n / 1e6:.0f} MB")

        def close(self) -> None:
            ctx.heartbeat(f"{self.desc}: done")

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

        def set_description(self, *args: Any, **kwargs: Any) -> None:
            return None

        def refresh(self, *args: Any, **kwargs: Any) -> None:
            return None

        def __getattr__(self, name: str):
            return lambda *a, **k: None

    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(target),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        tqdm_class=_Progress,
        max_workers=4,
    )
    model = inspect(target)
    ctx.checkpoint({"size_mb": model.size_mb, "complete": model.complete})
    return {
        "repo": repo,
        "path": str(target),
        "size_mb": model.size_mb,
        "complete": model.complete,
        "missing": model.missing,
    }
