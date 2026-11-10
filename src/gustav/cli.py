"""The ``ai`` command: a thin client for the running daemon.

Nothing here touches the model, the database, or the weights directly — every command
is a request to the one daemon, which is the only process allowed to act.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx

from gustav.config.load import load_config
from gustav.core.lockfile import PidLock
from gustav.paths import Paths

TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)

#: Commands whose implementation lands in a later stage. Listed so `ai --help` shows the
#: full surface and the message is an explanation rather than "unknown command".
NOT_YET: dict[str, str] = {
    "plan-hardware": "stage 2 (hardware detection + planner)",
    "backup": "stage 4 (backups)",
    "report": "stage 5 (sleep phase)",
    "facts": "stage 5 (sleep phase)",
    "approve-base-change": "stage 2 (planner approvals)",
    "approve-expansion": "stage 6 (self-expansion)",
    "expansion": "stage 6 (self-expansion)",
    "optimize": "stage 6 (self-optimisation)",
}


class DaemonUnreachable(RuntimeError):
    pass


class Client:
    def __init__(self, socket: Path, *, base_url: str | None = None, token: str | None = None):
        self.socket = socket
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if base_url:
            self._client = httpx.Client(base_url=base_url, timeout=TIMEOUT, headers=headers)
        else:
            transport = httpx.HTTPTransport(uds=str(socket))
            self._client = httpx.Client(
                transport=transport, base_url="http://gustav", timeout=TIMEOUT, headers=headers
            )

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise DaemonUnreachable(
                f"no daemon is listening on {self.socket}.\n"
                f"Start it with 'ai start' (or 'python daemon.py')."
            ) from exc
        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"daemon returned {response.status_code}: {detail}")
        return response.json()

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _print_status(status: dict[str, Any]) -> None:
    sleep = status["sleep"]
    model = status.get("model") or {}
    lines = [
        f"state            {status['state']}   (pid {status['pid']}, up "
        f"{_duration(status['uptime_seconds'])})",
        f"learning         {'PAUSED' if status['learning_paused'] else 'active'}",
        f"model            {model.get('base_model') or model.get('responder', 'none')}",
        f"adapters         {_adapters(model.get('adapters'))}",
        f"gpu              {status['gpu']['holder'] or 'free'}"
        + ("  (return requested)" if status["gpu"]["return_requested"] else ""),
        f"pending items    {status['pending_items']} "
        f"(threshold {status['pending_threshold']})",
        f"sleep windows    {', '.join(sleep['windows']) or 'none'}",
        f"next sleep       {sleep['next']}",
        f"last sleep       {_last_sleep(sleep)}",
    ]
    if sleep.get("current_run"):
        lines.append(f"current run      {sleep['current_run']} (asleep now)")
    if sleep.get("resumable_runs"):
        lines.append(f"resumable runs   {sleep['resumable_runs']}")
    if status.get("hardware_plan"):
        lines.append(f"hardware plan    {status['hardware_plan']}")
    lines.append(f"last backup      {status.get('last_backup') or 'none yet'}")
    if status["failed_steps"]:
        lines.append("failed steps")
        for step in status["failed_steps"]:
            lines.append(f"  - run {step['run']} {step['step']}: {step['error']}")
    if status["pending_approvals"]:
        lines.append("pending approvals")
        for item in status["pending_approvals"]:
            lines.append(f"  - [{item['id']}] {item['kind']}: {item['subject']}")
    recovery = status.get("recovery")
    if recovery and recovery.get("summary") != "clean startup":
        lines.append(f"recovery         {recovery['summary']}")
        for message in recovery.get("messages") or []:
            lines.append(f"  ! {message}")
    if sleep.get("decision") and sleep["decision"].get("reason"):
        lines.append(f"scheduler        {sleep['decision']['reason']}")
    print("\n".join(lines))


def _adapters(adapters: Any) -> str:
    if not adapters:
        return "none"
    if isinstance(adapters, dict):
        return ", ".join(f"{k}@{v}" for k, v in adapters.items())
    return str(adapters)


def _last_sleep(sleep: dict[str, Any]) -> str:
    last = sleep.get("last_run")
    if not last:
        return "never"
    finished = last.get("finished_at") or "running"
    done = sum(1 for s in last["steps"] if s["status"] == "done")
    return f"run {last['id']} {last['status']} ({done}/{len(last['steps'])} steps) {finished}"


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _print_run(run: dict[str, Any]) -> None:
    print(
        f"run {run['id']}  {run['status']}  trigger={run['trigger']}"
        f"{'  [dry-run]' if run.get('dry_run') else ''}"
    )
    if run.get("reason"):
        print(f"  reason: {run['reason']}")
    for step in run["steps"]:
        mark = {
            "done": "✓",
            "failed": "✗",
            "skipped": "–",
            "running": "…",
            "preempted": "⏸",
            "pending": " ",
        }.get(step["status"], "?")
        extra = ""
        if step["attempts"] > 1:
            extra += f" attempts={step['attempts']}"
        if step.get("checkpoint"):
            extra += f" checkpoint={json.dumps(step['checkpoint'])}"
        if step.get("error"):
            extra += f" error={step['error'].splitlines()[0]}"
        print(f"  {mark} {step['name']:<14}{step['status']:<10}{extra}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> int:
    from gustav.daemon import main as daemon_main

    forwarded: list[str] = []
    if args.config:
        forwarded += ["--config", args.config]
    if args.data_dir:
        forwarded += ["--data-dir", args.data_dir]
    return daemon_main(forwarded)


def cmd_chat(args: argparse.Namespace, client: Client) -> int:
    if args.message:
        result = client.post(
            "/chat", json={"message": " ".join(args.message), "session": args.session}
        )
        if result.get("sleep_paused"):
            print("(sleep paused to answer you)", file=sys.stderr)
        print(result["reply"])
        return 0

    print("Chatting with gustav. Ctrl-D or /quit to leave.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return 0
        result = client.post("/chat", json={"message": line, "session": args.session})
        if result.get("sleep_paused"):
            print("(sleep paused to answer you)", file=sys.stderr)
        print(f"ai > {result['reply']}")


def cmd_status(args: argparse.Namespace, client: Client) -> int:
    status = client.get("/status")
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        _print_status(status)
    return 0


def cmd_sleep(args: argparse.Namespace, client: Client) -> int:
    result = client.post(
        "/control/sleep", json={"dry_run": args.dry_run, "force": args.force}
    )
    if result.get("accepted"):
        print(f"sleeping now (run {result['run_id']})")
        return 0
    print(f"not sleeping: {result['reason']}", file=sys.stderr)
    return 1


def cmd_wake(args: argparse.Namespace, client: Client) -> int:
    result = client.post("/control/wake")
    print(result["reason"])
    return 0 if result.get("accepted") else 1


def cmd_learning(args: argparse.Namespace, client: Client, paused: bool) -> int:
    result = client.post("/control/learning", json={"paused": paused})
    print("learning paused" if result["learning_paused"] else "learning resumed")
    return 0


def cmd_stop(args: argparse.Namespace, client: Client) -> int:
    client.post("/control/shutdown")
    print("daemon is shutting down")
    return 0


def cmd_events(args: argparse.Namespace, client: Client) -> int:
    payload = client.get("/events", params={"limit": args.limit, **({"kind": args.kind} if args.kind else {})})
    for event in reversed(payload["events"]):
        data = f"  {json.dumps(event['data'])}" if event.get("data") else ""
        print(f"{event['ts']}  {event['level']:<7} {event['kind']:<22} {event['message']}{data}")
    return 0


def cmd_runs(args: argparse.Namespace, client: Client) -> int:
    if args.run_id is None:
        payload = client.get("/runs", params={"limit": args.limit})
        if not payload["runs"]:
            print("no sleep runs yet")
            return 0
        for run in payload["runs"]:
            print(
                f"{run['id']:>4}  {run['status']:<12} {run['trigger']:<8} "
                f"{run['started_at']} -> {run['finished_at'] or '...'}  {run['reason'] or ''}"
            )
        return 0
    _print_run(client.get(f"/runs/{args.run_id}"))
    return 0


def cmd_not_yet(name: str) -> int:
    print(f"'ai {name}' arrives in {NOT_YET[name]}.", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai", description="Talk to the gustav daemon.", allow_abbrev=False
    )
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("--data-dir", help="override instance.data_dir")
    parser.add_argument("--socket", help="control socket path")
    parser.add_argument("--url", help="talk to a remote daemon over HTTP instead")
    parser.add_argument("--token", help="bearer token for a remote daemon")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="run the daemon in the foreground")
    p.set_defaults(local=True)

    p = sub.add_parser("chat", help="chat with the model (interactive if no message)")
    p.add_argument("message", nargs="*")
    p.add_argument("--session", default="default")

    p = sub.add_parser("status", help="show state, schedule, adapters, approvals")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("sleep", help="sleep now")
    p.add_argument("when", nargs="?", choices=["now"], default="now")
    p.add_argument("--dry-run", action="store_true", help="plan the night without doing it")
    p.add_argument("--force", action="store_true", help="ignore the scheduler's objections")

    sub.add_parser("wake", help="pause sleep and come back online")
    sub.add_parser("pause-learning", help="no sleeping or training until resumed")
    sub.add_parser("resume-learning", help="allow sleeping again")
    sub.add_parser("stop", help="shut the daemon down gracefully")

    p = sub.add_parser("events", help="recent daemon events")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--kind")

    p = sub.add_parser("runs", help="sleep runs and their steps")
    p.add_argument("run_id", nargs="?", type=int)
    p.add_argument("--limit", type=int, default=20)

    for name, stage in NOT_YET.items():
        stub = sub.add_parser(name, help=f"(arrives in {stage})")
        stub.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
        stub.set_defaults(not_yet=name)

    return parser


def _resolve_socket(args: argparse.Namespace) -> Path:
    if args.socket:
        return Path(args.socket)
    try:
        config, paths = load_config(args.config, overrides=_overrides(args))
    except Exception:  # noqa: BLE001 - the CLI must work even with a broken config
        return Paths.create(Path.cwd()).socket
    if config.instance.socket:
        return Path(config.instance.socket)
    info = PidLock(paths.lock_file).read() or {}
    if info.get("socket"):
        return Path(info["socket"])
    return paths.socket


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {"instance": {"data_dir": args.data_dir}} if args.data_dir else {}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "not_yet", None):
        return cmd_not_yet(args.not_yet)
    if getattr(args, "local", False):
        return cmd_start(args)

    client = Client(_resolve_socket(args), base_url=args.url, token=args.token)
    try:
        if args.command == "chat":
            return cmd_chat(args, client)
        if args.command == "status":
            return cmd_status(args, client)
        if args.command == "sleep":
            return cmd_sleep(args, client)
        if args.command == "wake":
            return cmd_wake(args, client)
        if args.command == "pause-learning":
            return cmd_learning(args, client, True)
        if args.command == "resume-learning":
            return cmd_learning(args, client, False)
        if args.command == "stop":
            return cmd_stop(args, client)
        if args.command == "events":
            return cmd_events(args, client)
        if args.command == "runs":
            return cmd_runs(args, client)
        parser.error(f"unhandled command {args.command}")
        return 2
    except DaemonUnreachable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
