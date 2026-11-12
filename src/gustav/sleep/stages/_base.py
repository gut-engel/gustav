"""What a sleep step gets, and the boilerplate it does not have to write.

Every step runs in a spawned child process, so it starts with nothing: no config, no
database handle, no paths. This module turns the four scalars the supervisor passes
(``run_id``, ``step``, ``run_dir``, ``context``) into a working environment, and makes
three habits automatic rather than optional:

* **heartbeat while working** — a step that goes quiet is killed by the watchdog;
* **checkpoint after each unit of work** — not before, so a crash re-does at most one;
* **stop when asked** — a user message during the night wins, every time.

A step body is an ordinary synchronous function of one argument. It returns a dict,
which the journal stores as the step's result and the report reads back.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from gustav.config.load import load_config
from gustav.config.schema import Config
from gustav.core.events import Events, Telemetry
from gustav.core.supervisor import ChildContext
from gustav.learn.examples import ExampleStore
from gustav.learn.facts import FactStore
from gustav.learn.skills import SkillStore
from gustav.paths import Paths
from gustav.store.db import Database, open_db

T = TypeVar("T")


def step_context(paths: Paths) -> dict[str, str]:
    """What the parent must tell a child so it can find the project. Passed as kwargs."""
    return {"root": str(paths.root), "data_dir": str(paths.data_dir)}


@dataclass
class StepEnv:
    ctx: ChildContext
    config: Config
    paths: Paths
    db: Database
    run_id: int
    step: str
    run_dir: Path
    checkpoint: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    dry_run: bool = False

    # --- lazily built helpers -------------------------------------------
    def __post_init__(self) -> None:
        self.events = Events(self.db)
        self.telemetry = Telemetry(self.db)
        self.examples = ExampleStore(self.db, heldout_fraction=self.config.sleep.heldout_fraction)
        self.facts = FactStore(self.db)
        self.skills = SkillStore(self.db)
        self._model: Any = None
        #: Replaced by tests, and by anything that needs a model it controls.
        self.model_factory: Callable[[StepEnv], Any] | None = None

    @property
    def work_dir(self) -> Path:
        d = self.run_dir / self.step
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- talking to the parent ------------------------------------------
    def progress(self, message: str, **data: Any) -> None:
        self.ctx.heartbeat(f"{self.step}: {message}", **data)

    def log(self, message: str, level: str = "info") -> None:
        self.ctx.log(message, level=level)

    def save(self, data: dict[str, Any]) -> None:
        """Record resumable progress. Written to the journal before this returns."""
        self.checkpoint.update(data)
        self.ctx.checkpoint(dict(self.checkpoint))

    def stopping(self) -> bool:
        return self.ctx.should_stop()

    def raise_if_stopping(self) -> None:
        self.ctx.raise_if_stopping()

    def each(
        self,
        items: Iterable[T],
        *,
        label: str = "",
        checkpoint_key: str | None = None,
    ) -> Iterator[tuple[int, T]]:
        """Iterate with heartbeats, checkpoints and a stop check between items.

        With ``checkpoint_key`` the loop resumes where it left off: items already
        accounted for in the checkpoint are skipped, and the index is written *after*
        each item so a kill re-does one item rather than none or all.
        """
        materialised = list(items)
        start = int(self.checkpoint.get(checkpoint_key, 0)) if checkpoint_key else 0
        total = len(materialised)
        for index in range(start, total):
            self.raise_if_stopping()
            self.progress(f"{label or 'item'} {index + 1}/{total}", index=index + 1, total=total)
            yield index, materialised[index]
            if checkpoint_key:
                self.save({checkpoint_key: index + 1})

    # --- files ----------------------------------------------------------
    def write_json(self, name: str, payload: Any) -> Path:
        path = self.work_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.run_dir / name if "/" in name else self.work_dir / name
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def read_step_json(self, step: str, name: str, default: Any = None) -> Any:
        """Read a file another step of this run wrote."""
        try:
            return json.loads((self.run_dir / step / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    # --- the model ------------------------------------------------------
    def new_model(self):
        """A fresh model handle. The step owns it and must close it.

        Separate from ``model()`` because self-optimisation needs to load the model
        again under a changed setting, and because it is the seam tests replace.
        """
        if self.model_factory is not None:
            return self.model_factory(self)
        from gustav.sleep.model import StepModel

        return StepModel(self)

    def model(self):
        """The step's shared model, loaded on first use.

        Never raises: a machine with no GPU, no weights or a broken checkpoint gives a
        handle whose ``available`` is False, and every caller has a path for that.
        """
        if self._model is None:
            self._model = self.new_model()
        return self._model

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
            self._model = None
        self.db.close()


@contextmanager
def open_env(
    ctx: ChildContext,
    *,
    run_id: int,
    step: str,
    run_dir: str | None,
    context: dict[str, str] | None,
    checkpoint: dict[str, Any] | None,
    attempt: int,
    dry_run: bool,
) -> Iterator[StepEnv]:
    context = context or {}
    root = context.get("root")
    data_dir = context.get("data_dir")
    overrides = {"instance": {"data_dir": data_dir}} if data_dir else None
    config, paths = load_config(root=root, overrides=overrides)
    paths.ensure()
    db = open_db(paths.db)
    directory = Path(run_dir) if run_dir else paths.run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    env = StepEnv(
        ctx=ctx,
        config=config,
        paths=paths,
        db=db,
        run_id=run_id,
        step=step,
        run_dir=directory,
        checkpoint=dict(checkpoint or {}),
        attempt=attempt,
        dry_run=dry_run,
    )
    try:
        yield env
    finally:
        env.close()


def sleep_step(fn: Callable[[StepEnv], dict[str, Any] | None]):
    """Adapt a step body to the signature the supervisor calls children with."""

    @functools.wraps(fn)
    def wrapper(
        ctx: ChildContext,
        *,
        run_id: int,
        step: str,
        run_dir: str | None = None,
        context: dict[str, str] | None = None,
        checkpoint: dict[str, Any] | None = None,
        attempt: int = 1,
        dry_run: bool = False,
        **_ignored: Any,
    ) -> dict[str, Any]:
        with open_env(
            ctx,
            run_id=run_id,
            step=step,
            run_dir=run_dir,
            context=context,
            checkpoint=checkpoint,
            attempt=attempt,
            dry_run=dry_run,
        ) as env:
            result = fn(env) or {}
            result.setdefault("step", step)
            if env.dry_run:
                result.setdefault("dry_run", True)
            return result

    return wrapper
