"""Running a sleep step without a daemon, a child process or a GPU.

A step body is a plain function of a ``StepEnv``, which is what makes this possible:
build an env against a throwaway project, hand it a scripted model, call the body, and
assert on what ended up in the store.

The scripted model is the important part. Every step that uses the model has to behave
sensibly when there is none (``available = False``), when it answers something useless,
and when it answers well — and all three are one-line changes here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from gustav.config.schema import Config
from gustav.paths import Paths
from gustav.sleep.stages._base import StepEnv
from gustav.store.db import Database


class RecordingContext:
    """A ``ChildContext`` that keeps what the step said instead of sending it."""

    def __init__(self, *, stop_after: int | None = None):
        self.name = "test"
        self.heartbeats: list[tuple[str | None, dict[str, Any]]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self.logs: list[tuple[str, str]] = []
        #: Report "stop" once this many heartbeats have been sent, to exercise
        #: preemption at a realistic point rather than at the start.
        self.stop_after = stop_after

    def heartbeat(self, progress: str | None = None, **data: Any) -> None:
        self.heartbeats.append((progress, data))

    def checkpoint(self, data: dict[str, Any]) -> None:
        self.checkpoints.append(dict(data))

    def log(self, message: str, level: str = "info") -> None:
        self.logs.append((level, message))

    def should_stop(self) -> bool:
        return self.stop_after is not None and len(self.heartbeats) >= self.stop_after

    def raise_if_stopping(self) -> None:
        if self.should_stop():
            from gustav.core.supervisor import Preempted

            raise Preempted("asked to stop")


class ScriptedModel:
    """A stand-in for ``StepModel``.

    ``answers`` is either a list consumed in order, a dict matched by substring against
    the prompt, or a callable. ``available=False`` makes every step take its no-model
    path, which is the one that has to keep working on a machine with no GPU.
    """

    def __init__(
        self,
        answers: Sequence[str] | dict[str, str] | Callable[[str], str | None] | None = None,
        *,
        available: bool = True,
        error: str | None = None,
    ):
        self.answers = answers
        self.available = available
        self.error = error or (None if available else "no model in this test")
        self.prompts: list[str] = []
        self.candidates: list[tuple[str, str] | None] = []
        self.calls = 0
        self._index = 0

    def _answer_for(self, prompt: str) -> str | None:
        self.calls += 1
        if callable(self.answers):
            return self.answers(prompt)
        if isinstance(self.answers, dict):
            for needle, reply in self.answers.items():
                if needle.lower() in prompt.lower():
                    return reply
            return None
        if isinstance(self.answers, (list, tuple)):
            if not self.answers:
                return None
            reply = self.answers[min(self._index, len(self.answers) - 1)]
            self._index += 1
            return reply
        return None

    def ask(self, prompt: Any, *, system: str | None = None, **_overrides: Any) -> str | None:
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        return self._answer_for(text)

    def evaluate(
        self,
        prompts: Sequence[Any],
        *,
        candidate: tuple[str, str] | None = None,
        system: str | None = None,
        **_overrides: Any,
    ) -> list[str | None]:
        self.candidates.append(candidate)
        out: list[str | None] = []
        for prompt in prompts:
            text = prompt if isinstance(prompt, str) else str(prompt)
            self.prompts.append(text)
            out.append(self._answer_for(text))
        return out

    def ask_batch(self, prompts: Sequence[Any], **kwargs: Any) -> list[str | None]:
        return [self.ask(p, **kwargs) for p in prompts]

    def close(self) -> None:
        return None


def make_env(
    *,
    config: Config,
    paths: Paths,
    db: Database,
    run_id: int = 1,
    step: str = "consolidation",
    model: ScriptedModel | None = None,
    checkpoint: dict[str, Any] | None = None,
    dry_run: bool = False,
    ctx: RecordingContext | None = None,
) -> StepEnv:
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = StepEnv(
        ctx=ctx or RecordingContext(),  # type: ignore[arg-type]
        config=config,
        paths=paths,
        db=db,
        run_id=run_id,
        step=step,
        run_dir=run_dir,
        checkpoint=dict(checkpoint or {}),
        dry_run=dry_run,
    )
    # Steps reach the model through env.model(); pre-setting it is how a test decides
    # what the model says — or that there is not one.
    env._model = model if model is not None else ScriptedModel(available=False)
    return env


def seed_exchange(
    db: Database,
    *,
    question: str,
    answer: str,
    session_id: int | None = None,
    trainable: bool = True,
) -> tuple[int, int]:
    """One user/assistant pair, as the wake phase would have written it."""
    if session_id is None:
        session_id = db.insert("sessions", {"source": "test"})
    user_id = db.insert(
        "turns",
        {
            "session_id": session_id,
            "role": "user",
            "content": question,
            "trainable": int(trainable),
        },
    )
    assistant_id = db.insert(
        "turns",
        {
            "session_id": session_id,
            "role": "assistant",
            "content": answer,
            "trainable": int(trainable),
            "adapter_versions": "{}",
        },
    )
    return user_id, assistant_id


def seed_feedback(
    db: Database, *, turn_id: int, kind: str, payload: str | None = None, weight: int = 1
) -> int:
    return db.insert(
        "feedback",
        {"turn_id": turn_id, "kind": kind, "payload": payload, "source": "user", "reviewed": 1},
    )


def write_adapter(path: Path, *, base_model: str = "fake-model", version: str = "v001") -> Path:
    """A directory that looks enough like a PEFT adapter to be found and loaded."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"\x00")
    (path / "meta.json").write_text(
        f'{{"module": "{path.parent.name}", "version": "{version}",'
        f' "base_model": "{base_model}"}}',
        encoding="utf-8",
    )
    return path
