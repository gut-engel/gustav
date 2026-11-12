"""The control plane.

One FastAPI app, served on a Unix socket (always) and optionally on TCP. The CLI is a
thin client over this, so there is exactly one way to talk to the daemon and no way to
act on the system behind its back.

**Who is allowed to ask what** is decided in one place, ``identify``:

* a caller on the **Unix socket** is the owner. The filesystem already decided that —
  anyone who can open the socket can read the database beside it — so asking for a
  password there would be ceremony rather than security;
* a caller over **TCP** is whoever their token or session cookie says, and nobody
  without one. The TCP listener is off by default (`api.enabled`), so this only matters
  once you have deliberately opened a port;
* `instance.http_token` still works as a single shared service token for the
  simple one-user case, and it grants owner rights because that is what it always did.

Owner-only endpoints are the ones that change what the system *is*: reviewing a tool
(which is letting generated code run on this machine), approving an expansion, taking
backups, restoring, sleeping, stopping. A plain account can chat and read its own chats,
and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import json
import time
import uuid

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gustav.server.auth import COOKIE, Accounts, Identity, RateLimiter

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


class BackupRequest(BaseModel):
    operation: str = "manual"
    dry_run: bool = False
    include_base: bool = False


class RestoreRequest(BaseModel):
    only: list[str] = []
    module: str | None = None
    dry_run: bool = False


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


class ToolReview(BaseModel):
    approve: bool = True
    version: str | None = None
    by: str = "user"
    note: str = ""


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = {}


class ToolRequestBody(BaseModel):
    text: str


def create_app(daemon: "Daemon") -> FastAPI:
    app = FastAPI(
        title="gustav",
        version=daemon.status()["version"],
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.daemon = daemon

    accounts = Accounts(daemon.db, session_days=daemon.config.api.session_days)
    app.state.accounts = accounts
    limiter = RateLimiter(daemon.config.api.rate_limit_per_minute)

    def over_socket(request: Request) -> bool:
        """Unix-socket requests have no client address; TCP ones do."""
        return request.client is None

    def identify(request: Request) -> Identity:
        """Who is asking. Raises 401 rather than guessing."""
        if over_socket(request):
            return Identity(kind="socket")

        limiter.check(request)
        service = daemon.config.instance.http_token
        header = request.headers.get("authorization", "")
        presented = header.removeprefix("Bearer ").strip()
        if service and presented and secrets.compare_digest(presented, service):
            return Identity(kind="service")

        if daemon.config.api.accounts:
            if presented:
                identity = accounts.identify_token(presented)
                if identity is not None:
                    return identity
            cookie = request.cookies.get(COOKIE)
            if cookie:
                identity = accounts.identify_session(cookie)
                if identity is not None:
                    return identity

        raise HTTPException(
            status_code=401,
            detail=(
                "this endpoint needs an account: sign in, or pass a token with "
                "'Authorization: Bearer ...'"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    def require_owner(request: Request) -> Identity:
        identity = identify(request)
        if not identity.is_owner:
            raise HTTPException(
                status_code=403,
                detail=(
                    "that needs the owner account: it changes what the system is, not "
                    "just what it says"
                ),
            )
        return identity

    guard = [Depends(require_owner)]
    #: Anything an ordinary account may use.
    account = [Depends(identify)]

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

    @app.post("/control/release-gpu", dependencies=guard)
    async def release_gpu(reason: str = "requested") -> dict[str, Any]:
        return await daemon.release_gpu(reason)

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

    @app.get("/backups", dependencies=guard)
    async def backups() -> dict[str, Any]:
        return daemon.list_backups()

    @app.post("/backups", dependencies=guard)
    async def create_backup(req: BackupRequest) -> dict[str, Any]:
        from gustav.backup.snapshot import BackupError

        try:
            return daemon.create_backup(
                req.operation, dry_run=req.dry_run, include_base=req.include_base
            )
        except BackupError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc

    @app.post("/backups/{snapshot_id}/verify", dependencies=guard)
    async def verify_backup(snapshot_id: str) -> dict[str, Any]:
        from gustav.backup.snapshot import BackupError

        try:
            return daemon.verify_backup(snapshot_id)
        except BackupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/backups/{snapshot_id}/restore", dependencies=guard)
    async def restore_backup(snapshot_id: str, req: RestoreRequest) -> dict[str, Any]:
        from gustav.backup.snapshot import BackupError

        try:
            return await daemon.restore_backup(
                snapshot_id, only=tuple(req.only), module=req.module, dry_run=req.dry_run
            )
        except BackupError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/backups/retention", dependencies=guard)
    async def retention(dry_run: bool = False) -> dict[str, Any]:
        return daemon.apply_retention(dry_run=dry_run)

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
        row = daemon.db.query_one("SELECT kind FROM approvals WHERE id = ?", (approval_id,))
        if row is None:
            raise HTTPException(status_code=404, detail=f"no approval #{approval_id}")
        try:
            if row["kind"] == "expansion":
                if not req.approve:
                    return daemon.reject_expansion(approval_id, req.note or "")
                # Guarded like everything else that changes weights (section 9).
                return await daemon.approve_expansion(approval_id)
            if not req.approve:
                return daemon.hw.reject_approval(approval_id, req.note)
            result = daemon.approve_base_change(approval_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @app.get("/tools", dependencies=guard)
    async def tools(limit: int = 20) -> dict[str, Any]:
        return daemon.tools_status(limit=limit)

    @app.get("/tools/{name}", dependencies=guard)
    async def tool_detail(name: str, version: str | None = None) -> dict[str, Any]:
        try:
            return daemon.tool_detail(name, version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - ToolError and friends
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/tools/{name}/review", dependencies=guard)
    async def review_tool(name: str, req: ToolReview) -> dict[str, Any]:
        try:
            return daemon.review_tool(
                name, approve=req.approve, version=req.version,
                by=req.by or "user", note=req.note or "",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/tools/{name}/run", dependencies=guard)
    async def run_tool(name: str, req: ToolCallRequest) -> dict[str, Any]:
        try:
            return await daemon.run_tool(name, req.arguments or {})
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/tools/{name}/retire", dependencies=guard)
    async def retire_tool(name: str, req: ToolReview | None = None) -> dict[str, Any]:
        try:
            return daemon.retire_tool(name, note=(req.note if req else "") or "")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/tools/{name}/rollback", dependencies=guard)
    async def rollback_tool(name: str) -> dict[str, Any]:
        try:
            return daemon.rollback_tool(name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/tools/{name}", dependencies=guard)
    async def delete_tool(name: str) -> dict[str, Any]:
        try:
            return daemon.delete_tool(name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/tools/requests", dependencies=guard)
    async def request_tool(req: ToolRequestBody) -> dict[str, Any]:
        try:
            return daemon.request_tool(req.text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/expansion", dependencies=guard)
    async def expansion(limit: int = 20) -> dict[str, Any]:
        return daemon.expansion_status(limit)

    @app.post("/expansion/revert", dependencies=guard)
    async def revert_expansion() -> dict[str, Any]:
        try:
            return await daemon.revert_expansion()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/optimisation", dependencies=guard)
    async def optimisation(limit: int = 20) -> dict[str, Any]:
        return daemon.optimisation_status(limit)

    @app.post("/optimisation/revert", dependencies=guard)
    async def revert_optimisation() -> dict[str, Any]:
        try:
            return await daemon.revert_optimisation()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/facts", dependencies=guard)
    async def facts(
        limit: int = 100, module: str | None = None, subject: str | None = None
    ) -> dict[str, Any]:
        return daemon.facts(limit=limit, module=module, subject=subject)

    @app.get("/skills", dependencies=guard)
    async def skills() -> dict[str, Any]:
        return daemon.skills()

    @app.get("/report", dependencies=guard)
    async def report(run_id: int | None = None) -> dict[str, Any]:
        try:
            return daemon.write_report(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
