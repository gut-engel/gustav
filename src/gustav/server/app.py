"""The control plane.

One FastAPI app, served on a Unix socket (always) and optionally on TCP. The CLI is a
thin client over this, so there is exactly one way to talk to the daemon and no way to
act on the system behind its back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from gustav.daemon import Daemon


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session: str | None = None


class SleepRequest(BaseModel):
    dry_run: bool = False
    force: bool = False


class LearningRequest(BaseModel):
    paused: bool


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

    @app.post("/chat", dependencies=guard)
    async def chat(req: ChatRequest) -> dict[str, Any]:
        try:
            return await daemon.chat(req.message, session=req.session)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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

    @app.get("/report", dependencies=guard)
    async def report() -> dict[str, Any]:
        raise HTTPException(
            status_code=501, detail="HTML reports arrive with the sleep phase (stage 5)"
        )

    return app
