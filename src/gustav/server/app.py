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
from pathlib import Path

import secrets

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gustav.server.auth import COOKIE, Accounts, AuthError, Identity, RateLimiter
from gustav.server.chats import Chats

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


class UserRequest(BaseModel):
    name: str = Field(min_length=1)
    password: str = Field(min_length=8)
    role: str = "user"
    note: str = ""


class UserUpdate(BaseModel):
    password: str | None = None
    role: str | None = None
    disabled: bool | None = None
    #: Issue a token for this account and return it, once.
    token: str | None = None


class ChatMessage(BaseModel):
    message: str = Field(min_length=1)
    stream: bool = True


class ChatFeedback(BaseModel):
    kind: str = Field(pattern="^(good|bad|correct|forget)$")
    text: str = ""


class NewChat(BaseModel):
    title: str = ""


class Credentials(BaseModel):
    name: str = Field(min_length=1)
    password: str = Field(min_length=1)


class TokenRequest(Credentials):
    #: What the token is for, so `ai user tokens` is readable later.
    label: str = "cli"
    days: int | None = None


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

    # ------------------------------------------------------------------
    # accounts
    # ------------------------------------------------------------------
    @app.get("/auth/me")
    async def whoami(request: Request) -> dict[str, Any]:
        """Who the daemon thinks you are. The one endpoint a client calls first."""
        identity = identify(request)
        return {
            **identity.to_dict(),
            "accounts": daemon.config.api.accounts,
            "instance": daemon.config.instance.name,
            "version": daemon.status()["version"],
        }

    @app.post("/auth/token")
    async def issue_token(req: TokenRequest, request: Request) -> dict[str, Any]:
        """Exchange a password for a token. What `ai login` calls.

        Deliberately available without being signed in — proving you know the password
        *is* the authentication. It is rate limited like everything else on the TCP
        listener, and it is the only place a password is accepted.
        """
        if not daemon.config.api.accounts:
            raise HTTPException(
                status_code=404,
                detail="this daemon has accounts switched off (api.accounts)",
            )
        try:
            user = accounts.authenticate(req.name, req.password)
            token = accounts.issue_token(user.name, label=req.label, days=req.days)
        except AuthError as exc:
            # One message for a wrong name and a wrong password: which of the two it
            # was is not the caller's business.
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        daemon.events.emit(
            "auth.token_issued",
            f"a {req.label} token was issued for {user.name}",
            user=user.name,
            label=req.label,
        )
        return {"token": token, "user": user.to_dict()}

    @app.post("/auth/login")
    async def login(req: Credentials, request: Request, response: Response) -> dict[str, Any]:
        """Sign a browser in. Sets an httpOnly cookie; the page never sees the token."""
        if not daemon.config.api.accounts:
            raise HTTPException(status_code=404, detail="accounts are switched off")
        try:
            user = accounts.authenticate(req.name, req.password)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        token = accounts.start_session(
            user, user_agent=request.headers.get("user-agent", "")
        )
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            samesite="lax",
            # Only over HTTPS when the request itself came over HTTPS, so this works on
            # a plain-HTTP machine on your own network without silently dropping the
            # cookie — and is secure wherever it matters.
            secure=request.url.scheme == "https",
            max_age=daemon.config.api.session_days * 86400,
            path="/",
        )
        return {"user": user.to_dict()}

    @app.post("/auth/logout")
    async def logout(request: Request, response: Response) -> dict[str, Any]:
        cookie = request.cookies.get(COOKIE)
        if cookie:
            accounts.end_session(cookie)
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/users", dependencies=guard)
    async def list_users() -> dict[str, Any]:
        return {
            "accounts": daemon.config.api.accounts,
            "users": [user.to_dict() for user in accounts.all()],
            "tokens": accounts.tokens(),
        }

    @app.post("/users", dependencies=guard)
    async def create_user(req: UserRequest) -> dict[str, Any]:
        try:
            user = accounts.create(
                req.name, req.password, role=req.role, note=req.note or ""
            )
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        daemon.events.emit(
            "auth.user_created",
            f"the account {user.name!r} was created ({user.role})",
            level="warning",
            user=user.name,
            role=user.role,
        )
        return user.to_dict()

    @app.post("/users/{name}", dependencies=guard)
    async def update_user(name: str, req: UserUpdate) -> dict[str, Any]:
        try:
            if req.password is not None:
                accounts.set_password(name, req.password)
            if req.role is not None:
                accounts.set_role(name, req.role)
            if req.disabled is not None:
                accounts.disable(name, disabled=req.disabled)
            if req.token:
                return {
                    "user": accounts.find(name).to_dict(),
                    "token": accounts.issue_token(name, label=req.token),
                }
        except AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        user = accounts.find(name)
        if user is None:
            raise HTTPException(status_code=404, detail=f"no account called {name!r}")
        return {"user": user.to_dict()}

    @app.delete("/users/{name}", dependencies=guard)
    async def delete_user(name: str) -> dict[str, Any]:
        try:
            accounts.delete(name)
        except AuthError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        daemon.events.emit(
            "auth.user_deleted", f"the account {name!r} was deleted", level="warning"
        )
        return {"deleted": name}

    # ------------------------------------------------------------------
    # chats — what the web interface and a signed-in CLI use
    # ------------------------------------------------------------------
    def chats_for(request: Request) -> tuple[Chats, Identity]:
        identity = identify(request)
        return Chats(daemon.db), identity

    @app.get("/chats", dependencies=account)
    async def list_chats(request: Request, limit: int = 50) -> dict[str, Any]:
        chats, identity = chats_for(request)
        return {
            "chats": [chat.to_dict() for chat in chats.all(user_id=identity.user_id, limit=limit)],
            "you": identity.to_dict(),
        }

    @app.post("/chats", dependencies=account)
    async def new_chat(req: NewChat, request: Request) -> dict[str, Any]:
        chats, identity = chats_for(request)
        return chats.create(user_id=identity.user_id, title=req.title).to_dict()

    @app.get("/chats/{chat_id}/messages", dependencies=account)
    async def chat_messages(chat_id: int, request: Request) -> dict[str, Any]:
        chats, identity = chats_for(request)
        try:
            return {
                "chat": chats.get(chat_id, user_id=identity.user_id).to_dict(),
                "messages": chats.messages(chat_id, user_id=identity.user_id),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/chats/{chat_id}/messages", dependencies=account)
    async def send_message(chat_id: int, req: ChatMessage, request: Request):
        chats, identity = chats_for(request)
        try:
            key = chats.key(chat_id, user_id=identity.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        chats.title_if_unset(chat_id, req.message)

        if not req.stream:
            try:
                result = await daemon.chat(req.message, session=key)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            chats.touch(chat_id)
            return result

        async def events():
            # Once the response has begun there is no status code left to send, so a
            # failure in here has to travel as an event. Without this the connection
            # simply dies and the browser invents its own reason — "Error in input
            # stream" — while the real one (no weights, no GPU, safe mode) stays in
            # the log where nobody in a browser can see it.
            try:
                async for event in daemon.chat_stream(req.message, session=key):
                    yield sse(event)
            except Exception as exc:  # noqa: BLE001 - the page gets the reason, not a dead socket
                daemon.log.exception("streaming chat failed")
                yield sse({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            chats.touch(chat_id)
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/chats/{chat_id}/feedback", dependencies=account)
    async def chat_feedback(chat_id: int, req: ChatFeedback, request: Request) -> dict[str, Any]:
        """`/good`, `/bad`, `/correct …` — the same commands the terminal has."""
        chats, identity = chats_for(request)
        try:
            chats.get(chat_id, user_id=identity.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        command = f"/{req.kind}" + (f" {req.text}" if req.text else "")
        result = daemon.commands.handle(chat_id, command)
        if result is None:
            raise HTTPException(status_code=400, detail=f"{command} is not a command")
        return {"reply": result.reply, "action": result.action}

    @app.delete("/chats/{chat_id}", dependencies=account)
    async def archive_chat(chat_id: int, request: Request) -> dict[str, Any]:
        chats, identity = chats_for(request)
        try:
            chats.archive(chat_id, user_id=identity.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"archived": chat_id}

    @app.get("/status", dependencies=guard)
    async def status() -> dict[str, Any]:
        return daemon.status()

    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # Chatting is what an ordinary account is for. Everything else on this app changes
    # what the system *is* rather than what it says, and stays with the owner.
    @app.post("/chat", dependencies=account)
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

    @app.post("/v1/chat/completions", dependencies=account)
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

    # ------------------------------------------------------------------
    # the web interface
    # ------------------------------------------------------------------
    @app.get("/web/config")
    async def web_config() -> dict[str, Any]:
        """What the page needs before anyone has signed in.

        Deliberately unauthenticated and deliberately boring: a title, whether accounts
        exist, and whether the owner's panels are on. Nothing here tells you anything
        you could not learn by looking at the login form.
        """
        return {
            "title": daemon.config.web.title,
            "enabled": daemon.config.web.enabled,
            "accounts": daemon.config.api.accounts,
            "owner_panels": daemon.config.web.owner_panels,
            "instance": daemon.config.instance.name,
        }

    if daemon.config.web.enabled:
        # Registered last, so every API route above wins over the static files. The
        # three files are served as they are; everything they show comes from the same
        # endpoints the CLI calls, under the same identity rules.
        from fastapi.staticfiles import StaticFiles

        root = Path(__file__).resolve().parent.parent / "web"
        if root.is_dir():
            app.mount("/", StaticFiles(directory=str(root), html=True), name="web")
        else:  # pragma: no cover - only if the package was installed without its data
            daemon.events.emit(
                "web.missing",
                f"web.enabled is set but {root} is not in this install",
                level="warning",
            )

    return app
