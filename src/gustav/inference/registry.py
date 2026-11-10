"""The module registry: ``modules/registry.yaml``.

A module is a named LoRA adapter with a domain description. Adding one is an entry here
plus a dataset; nothing in the code knows the names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_THRESHOLD = 0.45
DEFAULT_MAX_ACTIVE = 3


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    description: str = ""
    keywords: tuple[str, ...] = ()
    #: Always applied, regardless of what the router thinks (identity, personal memory).
    always_on: bool = False
    #: Higher wins when the active set has to be trimmed.
    priority: int = 0
    #: Measured once the module exists.
    size_mb: float = 0.0
    load_ms: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """What the router matches a request against."""
        return " ".join([self.name, self.description, *self.keywords])


@dataclass
class Registry:
    modules: tuple[ModuleSpec, ...] = ()
    base_model: str = "auto"
    threshold: float = DEFAULT_THRESHOLD
    max_active: int = DEFAULT_MAX_ACTIVE
    source: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_name = {m.name: m for m in self.modules}

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self.modules)

    def __len__(self) -> int:
        return len(self.modules)

    def get(self, name: str) -> ModuleSpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"unknown module {name!r}; the registry has: {', '.join(sorted(self._by_name))}"
            ) from None

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.modules]

    @property
    def always_on(self) -> list[ModuleSpec]:
        return [m for m in self.modules if m.always_on]

    def add(self, spec: ModuleSpec) -> "Registry":
        """Register a new module (self-expansion, section 8, action 1)."""
        if spec.name in self._by_name:
            raise ValueError(f"module {spec.name!r} already exists")
        return Registry(
            modules=(*self.modules, spec),
            base_model=self.base_model,
            threshold=self.threshold,
            max_active=self.max_active,
            source=self.source,
            raw=self.raw,
        )

    def to_yaml(self) -> str:
        payload = {
            "base": {"model": self.base_model},
            "defaults": {"threshold": self.threshold, "max_active": self.max_active},
            "modules": [
                {
                    "name": m.name,
                    "description": m.description,
                    "keywords": list(m.keywords),
                    **({"always_on": True} if m.always_on else {}),
                    "priority": m.priority,
                    **({"size_mb": m.size_mb} if m.size_mb else {}),
                    **({"load_ms": m.load_ms} if m.load_ms else {}),
                }
                for m in self.modules
            ],
        }
        return yaml.safe_dump(payload, sort_keys=False, width=88)

    def write(self, path: Path | None = None) -> Path:
        target = path or self.source
        if target is None:
            raise ValueError("no path to write the registry to")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_yaml(), encoding="utf-8")
        return target


def load_registry(path: Path | str) -> Registry:
    source = Path(path)
    if not source.exists():
        return Registry(source=source)
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    modules = []
    known = {f.name for f in ModuleSpec.__dataclass_fields__.values()}
    for entry in raw.get("modules") or []:
        data = dict(entry)
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{source}: module {data.get('name')!r} has unknown keys: {sorted(unknown)}"
            )
        data["keywords"] = tuple(data.get("keywords") or ())
        data["notes"] = tuple(data.get("notes") or ())
        data["description"] = (data.get("description") or "").strip()
        modules.append(ModuleSpec(**data))
    names = [m.name for m in modules]
    if len(names) != len(set(names)):
        raise ValueError(f"{source}: duplicate module names")
    return Registry(
        modules=tuple(modules),
        base_model=((raw.get("base") or {}).get("model") or "auto"),
        threshold=float(defaults.get("threshold", DEFAULT_THRESHOLD)),
        max_active=int(defaults.get("max_active", DEFAULT_MAX_ACTIVE)),
        source=source,
        raw=raw,
    )
