"""The control plane.

One FastAPI app, served on a Unix socket (always) and optionally on TCP. The CLI is a
thin client over this, so there is exactly one way to talk to the daemon and no way to
act on the system behind its back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import json
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gustav.daemon import Daemon


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session: str | None = None
    stream: bool = False


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIRequest(BaseModel):
    """The subset of the OpenAI chat API that makes sense here."""

    model: str | None = None
    messages: list[OpenAIMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None
    user: str | None = None


class AdapterRequest(BaseModel):
    module: str
    version: str | None = None


class ModelRequest(BaseModel):
    model_id: str


class SleepRequest(BaseModel):
    dry_run: bool = False
    force: bool = False


class LearningRequest(BaseModel):
    paused: bool


class PlanRequest(BaseModel):
    #: Recompute with a different speed/capability balance without changing config.
    balance: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Store the result as the active plan (a base-model change still needs approval).
    apply: bool = False
    #: Any HardwareConfig key, e.g. {"require_training": false}.
    overrides: dict[str, Any] = {}


class ApprovalDecision(BaseModel):
    approve: bool
    note: str = ""


def create_app(daemon: "Daemon") -> FastAPI:
    app = FastAPI(
        title="gustav",
        version=daemon.status()["version"],
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.daemon = daemon

    def authorize(request: Request) -> None:
        """Socket callers are trusted by filesystem permissions; TCP callers are not."""
        token = daemon.config.instance.http_token
        if not token or request.client is None:
            return
        header = request.headers.get("authorization", "")
        presented = header.removeprefix("Bearer ").strip()
        if presented != token:
            raise HTTPException(status_code=401, detail="invalid or missing token")

    guard = [Depends(authorize)]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        ok, message = daemon.responder.health()
        return {
            "ok": ok,
            "message": message,
            "state": str(daemon.state.state),
            "pid": daemon.status()["pid"],
        }

    @app.get("/status", dependencies=guard)
    async def status() -> dict[str, Any]:
        return daemon.status()

    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    @app.post("/chat", dependencies=guard)
    async def chat(req: ChatRequest):
        if not req.stream:
            try:
                return await daemon.chat(req.message, session=req.session)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        async def events():
            try:
                async for event in daemon.chat_stream(req.message, session=req.session):
                    yield sse(event)
            except Exception as exc:  # noqa: BLE001 - the client gets the error, not a hang
                daemon.log.exception("streaming chat failed")
                yield sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/completions", dependencies=guard)
    async def openai_chat(req: OpenAIRequest):
        """OpenAI-compatible, so existing clients work against the local daemon."""
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="no user message in the request")
        prompt = user_messages[-1].content
        session = req.user or "openai"
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = req.model or (daemon.status()["model"].get("base_model") or "gustav")

        if not req.stream:
            result = await daemon.chat(prompt, session=session)
            meta = result.get("meta") or {}
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["reply"]},
                        "finish_reason": meta.get("finish_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "total_tokens": meta.get("prompt_tokens", 0)
                    + meta.get("completion_tokens", 0),
                },
            }

        async def events():
            def frame(delta: dict[str, Any], finish: str | None = None) -> str:
                return sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                )

            yield frame({"role": "assistant", "content": ""})
            try:
                async for event in daemon.chat_stream(prompt, session=session):
                    if event["type"] == "chunk":
                        yield frame({"content": event["text"]})
                    elif event["type"] == "command":
                        yield frame({"content": event["reply"]})
            except Exception as exc:  # noqa: BLE001
                daemon.log.exception("openai-compatible streaming failed")
                yield sse({"error": {"message": str(exc), "type": type(exc).__name__}})
            yield frame({}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/adapter/status", dependencies=guard)
    async def adapter_status() -> dict[str, Any]:
        return daemon.adapter_status()

    @app.post("/adapter/rollback", dependencies=guard)
    async def adapter_rollback(req: AdapterRequest) -> dict[str, Any]:
        try:
            return await daemon.rollback_adapter(req.module)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/adapter/activate", dependencies=guard)
    async def adapter_activate(req: AdapterRequest) -> dict[str, Any]:
        if not req.version:
            raise HTTPException(status_code=400, detail="a version is required")
        try:
            return await daemon.activate_adapter(req.module, req.version)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/control/sleep", dependencies=guard)
    async def sleep_now(req: SleepRequest) -> dict[str, Any]:
        return await daemon.request_sleep(dry_run=req.dry_run, force=req.force)

    @app.post("/control/wake", dependencies=guard)
    async def wake() -> dict[str, Any]:
        return await daemon.request_wake()

    @app.post("/control/learning", dependencies=guard)
    async def learning(req: LearningRequest) -> dict[str, Any]:
        return daemon.set_learning_paused(req.paused)

    @app.post("/control/shutdown", dependencies=guard)
    async def shutdown() -> dict[str, Any]:
        daemon.request_shutdown("shutdown requested over the control plane")
        return {"stopping": True}

    @app.get("/events", dependencies=guard)
    async def events(limit: int = 50, kind: str | None = None) -> dict[str, Any]:
        return {"events": daemon.events.recent(limit=limit, kind=kind)}

    @app.get("/runs", dependencies=guard)
    async def runs(limit: int = 20) -> dict[str, Any]:
        rows = daemon.db.query(
            "SELECT id, kind, trigger, status, started_at, finished_at, reason"
            " FROM sleep_runs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return {"runs": [dict(r) for r in rows]}

    @app.get("/runs/{run_id}", dependencies=guard)
    async def run_detail(run_id: int) -> dict[str, Any]:
        try:
            record = daemon.journal.load_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no run {run_id}") from exc
        return {
            "id": record.id,
            "trigger": record.trigger,
            "status": str(record.status),
            "dry_run": record.dry_run,
            "run_dir": str(record.run_dir) if record.run_dir else None,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "reason": record.reason,
            "steps": [
                {
                    "name": s.name,
                    "position": s.position,
                    "status": str(s.status),
                    "attempts": s.attempts,
                    "on_crash": str(s.on_crash),
                    "checkpoint": s.checkpoint,
                    "result": s.result,
                    "error": s.error,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                }
                for s in record.steps
            ],
        }

    @app.post("/bench", dependencies=guard)
    async def bench(max_tokens: int = 96, repeats: int = 1) -> dict[str, Any]:
        try:
            return await daemon.benchmark(max_tokens=max_tokens, repeats=repeats)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/models", dependencies=guard)
    async def models() -> dict[str, Any]:
        return daemon.list_models()

    @app.post("/models/pull", dependencies=guard)
    async def pull_model(req: ModelRequest) -> dict[str, Any]:
        try:
            return await daemon.pull_model(req.model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/hardware", dependencies=guard)
    async def hardware() -> dict[str, Any]:
        return daemon.hw.describe()

    @app.post("/hardware/plan", dependencies=guard)
    async def plan_hardware(req: PlanRequest) -> dict[str, Any]:
        from gustav.hw.planner import PlannerError

        try:
            if req.apply and req.balance is None and not req.overrides:
                outcome = daemon._plan_hardware(reason="requested", force=True)
                if outcome.plan is None:
                    raise HTTPException(status_code=409, detail=outcome.error or "no usable plan")
                return {
                    "applied": True,
                    "source": outcome.source,
                    "approval_id": outcome.approval_id,
                    "messages": outcome.messages,
                    "plan": outcome.plan.to_dict(),
                    "explanation": outcome.plan.explanation,
                    "rejected": outcome.plan.rejected,
                }
            plan = daemon.hw.compute_plan(balance=req.balance, overrides=req.overrides or None)
        except PlannerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        applied = False
        approval_id = None
        if req.apply:
            daemon.db.set_meta("hardware_plan", plan.to_dict())
            applied = True
        return {
            "applied": applied,
            "source": "planner",
            "approval_id": approval_id,
            "messages": [],
            "plan": plan.to_dict(),
            "explanation": plan.explanation,
            "rejected": plan.rejected,
        }

    @app.get("/approvals", dependencies=guard)
    async def approvals(status: str = "pending", limit: int = 50) -> dict[str, Any]:
        rows = daemon.db.query(
            "SELECT id, kind, subject, rationale, status, created_at, decided_at, note"
            " FROM approvals WHERE (? = 'all' OR status = ?) ORDER BY id DESC LIMIT ?",
            (status, status, limit),
        )
        return {"approvals": [dict(r) for r in rows]}

    @app.post("/approvals/{approval_id}", dependencies=guard)
    async def decide(approval_id: int, req: ApprovalDecision) -> dict[str, Any]:
        try:
            if not req.approve:
                return daemon.hw.reject_approval(approval_id, req.note)
            result = daemon.hw.approve_base_change(approval_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @app.get("/report", dependencies=guard)
    async def report() -> dict[str, Any]:
        raise HTTPException(
            status_code=501, detail="HTML reports arrive with the sleep phase (stage 5)"
        )

    return app
