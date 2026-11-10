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
    "backup": "stage 4 (backups)",
    "report": "stage 5 (sleep phase)",
    "facts": "stage 5 (sleep phase)",
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

    def stream(self, path: str, body: dict[str, Any]):
        """Iterate server-sent events from the daemon."""
        try:
            with self._client.stream("POST", path, json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    raise RuntimeError(f"daemon returned {response.status_code}: {response.text}")
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as exc:
            raise DaemonUnreachable(
                f"no daemon is listening on {self.socket}.\n"
                f"Start it with 'ai start' (or 'python daemon.py')."
            ) from exc

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
    hardware = status.get("hardware") or {}
    if hardware.get("plan"):
        lines.append(f"hardware plan    {hardware['plan']}")
        budget = hardware.get("expansion_budget") or {}
        if budget:
            lines.append(
                f"vram             "
                f"{(hardware.get('vram_inference_mb') or 0) / 1024:.1f} GiB inference / "
                f"{(hardware.get('vram_training_mb') or 0) / 1024:.1f} GiB training, "
                f"{budget['max_adapters']} adapters <= r{budget['max_lora_rank']}"
            )
    for gpu in hardware.get("gpus") or []:
        lines.append(f"gpu detected     {gpu}")
    temps = hardware.get("temperatures_c") or {}
    if temps:
        lines.append(
            "temperature      "
            + ", ".join(f"GPU{i} {t:.0f}C" for i, t in sorted(temps.items()))
        )
    for warning in hardware.get("warnings") or []:
        lines.append(f"  ! {warning}")
    if hardware.get("problem"):
        lines.append(f"  ! {hardware['problem']}")
    lines.append(f"last backup      {status.get('last_backup') or 'none yet'}")
    if status.get("load_error"):
        lines.append(f"  ! the model is not loaded: {status['load_error'].splitlines()[0]}")
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


def _say(client: Client, message: str, session: str, *, stream: bool, prefix: str = "") -> int:
    """One turn, streamed token by token unless asked otherwise."""
    if not stream:
        result = client.post("/chat", json={"message": message, "session": session})
        if result.get("sleep_paused"):
            print("(sleep paused to answer you)", file=sys.stderr)
        for kind in result.get("redacted") or []:
            print(f"(a {kind} was detected: stored redacted, never trained on)", file=sys.stderr)
        print(prefix + result["reply"])
        return 0

    printed_prefix = False
    for event in client.stream("/chat", {"message": message, "session": session, "stream": True}):
        kind = event.get("type")
        if kind == "start":
            if event.get("sleep_paused"):
                print("(sleep paused to answer you)", file=sys.stderr)
            for redaction in event.get("redacted") or []:
                print(
                    f"(a {redaction} was detected: stored redacted, never trained on)",
                    file=sys.stderr,
                )
        elif kind in ("chunk", "command"):
            text = event.get("text") or event.get("reply") or ""
            if prefix and not printed_prefix:
                print(prefix, end="", flush=True)
                printed_prefix = True
            print(text, end="", flush=True)
        elif kind == "error":
            print(f"\n[error: {event['error']}]", file=sys.stderr)
            return 1
        elif kind == "done":
            speed = event.get("tokens_per_second")
            print()
            if speed:
                meta = event.get("meta") or {}
                ttft = meta.get("time_to_first_token_ms")
                detail = f"{speed:.1f} tok/s"
                if ttft:
                    detail += f", {ttft:.0f} ms to first token"
                modules = meta.get("modules")
                if modules:
                    detail += f", modules: {', '.join(modules)}"
                print(f"[{detail}]", file=sys.stderr)
    return 0


def cmd_chat(args: argparse.Namespace, client: Client) -> int:
    stream = not args.no_stream
    if args.message:
        return _say(client, " ".join(args.message), args.session, stream=stream)

    print("Chatting with gustav. /help for commands, Ctrl-D to leave.")
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
        _say(client, line, args.session, stream=stream, prefix="ai > ")


def cmd_model(args: argparse.Namespace, client: Client) -> int:
    if args.action == "list":
        payload = client.get("/models")
        print(f"models in {payload['models_dir']}:")
        if not payload["models"]:
            print("  (none yet — 'ai model pull <id>' fetches one)")
        for model in payload["models"]:
            mark = "*" if model["active"] else " "
            print(f" {mark} {model['id']:<16} {model['size_mb'] / 1024:>7.2f} GiB  "
                  + ("ready" if model["complete"] else
                     f"INCOMPLETE ({len(model['missing'])} file(s) missing)"))
        print("\non the ladder (fetch with 'ai model pull <id>'):")
        have = {m["id"] for m in payload["models"] if m["complete"]}
        for spec in payload["ladder"]:
            state = "local" if spec["id"] in have else "not fetched"
            print(f"   {spec['id']:<16} {spec['params_b']:>6.1f}B  {spec['repo']:<28} {state}")
        return 0

    print(f"fetching {args.model_id} (this goes to the network; it is the only part that does)")
    result = client.post("/models/pull", json={"model_id": args.model_id})
    if not result.get("ok"):
        print(f"failed: {result.get('error')}", file=sys.stderr)
        return 1
    print(f"done: {result['path']} ({result['size_mb'] / 1024:.2f} GiB)")
    return 0


def cmd_bench(args: argparse.Namespace, client: Client) -> int:
    print("benchmarking (this loads the model and generates a few hundred tokens)...")
    result = client.post(
        "/bench", params={"max_tokens": args.max_tokens, "repeats": args.repeats}
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(result["summary"])
    for note in result.get("notes") or []:
        print(f"  note: {note}")
    print("\nrecorded: the planner now uses this measurement instead of its estimate")
    return 0


def cmd_adapters(args: argparse.Namespace, client: Client) -> int:
    status = client.get("/adapter/status")
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    modules = status.get("modules") or {}
    if not modules:
        print("no adapters yet — they are trained during sleep (stage 5)")
    for name, info in modules.items():
        versions = ", ".join(info["versions"]) or "none"
        print(
            f"{name:<18} active={info['active'] or '-':<6} loaded={info['loaded']!s:<5} "
            f"tier={info['tier'] or '-':<5} versions=[{versions}]"
        )
    active = status.get("active_in_model") or {}
    print(f"\nactive in the model: {active or 'none'}")
    cache = status.get("cache")
    if cache:
        print(f"adapter cache: VRAM {cache['vram']['used_mb']:.0f}/"
              f"{cache['vram']['budget_mb']:.0f} MiB, hit rate {cache['hit_rate']}")
    return 0


def cmd_rollback(args: argparse.Namespace, client: Client) -> int:
    result = client.post("/adapter/rollback", json={"module": args.module})
    print(f"{result['module']} rolled back to {result['version']}")
    return 0


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


def cmd_plan_hardware_offline(args: argparse.Namespace) -> int:
    """Plan without a running daemon: 'can this machine run it at all?'"""
    from gustav.hw.detect import default_detector, simulated
    from gustav.hw.planner import Planner, PlannerError

    config, paths = load_config(args.config, overrides=_overrides(args))
    detector = simulated(args.simulate) if args.simulate else default_detector(paths.data_dir)
    hardware = detector.detect()
    overrides: dict[str, Any] = {}
    if args.no_training:
        overrides.update(allow_training=False, require_training=False)
    if args.model:
        overrides["prefer_model"] = args.model
    if args.max_seq_len:
        overrides["max_seq_len"] = args.max_seq_len
    hw_config = config.hardware.model_copy(
        update={**({"balance": args.balance} if args.balance is not None else {}), **overrides}
    )

    for line in hardware.describe():
        print(line)
    print()
    try:
        plan = Planner(hardware, hw_config).plan()
    except PlannerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, default=str))
        return 0
    print(f"plan: {_plan_line(plan.to_dict())}")
    print()
    for line in plan.explanation:
        print(f"  {line}")
    for warning in plan.warnings:
        print(f"  ! {warning}")
    if args.verbose and plan.rejected:
        print("\nnot chosen:")
        for line in plan.rejected:
            print(f"  {line}")
    print("\n(computed locally; start the daemon and re-run with --apply to use it)")
    return 0


def cmd_plan_hardware(args: argparse.Namespace, client: Client) -> int:
    payload: dict[str, Any] = {"apply": args.apply}
    if args.balance is not None:
        payload["balance"] = args.balance
    overrides: dict[str, Any] = {}
    if args.no_training:
        overrides["require_training"] = False
        overrides["allow_training"] = False
    if args.model:
        overrides["prefer_model"] = args.model
    if args.max_seq_len:
        overrides["max_seq_len"] = args.max_seq_len
    if overrides:
        payload["overrides"] = overrides

    result = client.post("/hardware/plan", json=payload)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    plan = result["plan"]
    print(f"plan: {_plan_line(plan)}")
    print()
    for line in result["explanation"]:
        print(f"  {line}")
    for warning in plan.get("warnings") or []:
        print(f"  ! {warning}")
    if args.verbose and result.get("rejected"):
        print("\nnot chosen:")
        for line in result["rejected"]:
            print(f"  {line}")
    if result.get("messages"):
        print()
        for message in result["messages"]:
            print(f"  ! {message}")
    if result.get("applied"):
        print("\nstored as the active plan")
    elif not args.apply:
        print("\n(nothing changed; pass --apply to store this plan)")
    return 0


def _plan_line(plan: dict[str, Any]) -> str:
    speed = plan.get("estimated_tokens_per_second") or 0
    kind = "measured" if plan.get("speed_measured") else "estimated"
    gpus = plan.get("training_gpus") or []
    if plan.get("training_allowed") and gpus:
        label = "GPU" if len(gpus) == 1 else "GPUs"
        training = f"training on {label} {', '.join(str(g) for g in gpus)}"
    else:
        training = "inference only"
    return (
        f"{plan['base_model']} @ {plan['quantization']}, seq {plan['max_seq_len']}, "
        f"LoRA r{plan['lora_rank']}, ~{speed:.0f} tok/s ({kind}), {training}"
    )


def cmd_hardware(args: argparse.Namespace, client: Client) -> int:
    info = client.get("/hardware")
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    print(f"fingerprint      {info['fingerprint']}  (source: {info['source']})")
    for gpu in info["gpus"]:
        print(f"                 {gpu}")
    for budget in info["budgets"]:
        print(f"                 {budget}")
    host = info["host"]
    print(
        f"host             {host['cpu_count']} cores, "
        f"{host['ram_total_mb'] / 1024:.0f} GiB RAM, "
        f"{host['disk_free_mb'] / 1024:.0f} GiB disk free"
    )
    print(f"active base      {info['active_base_model'] or 'none yet'}")
    if info.get("plan"):
        print(f"plan             {_plan_line(info['plan'])}")
    if info.get("pending_plan"):
        print(f"proposed         {_plan_line(info['pending_plan'])}  (needs approval)")
    return 0


def cmd_approvals(args: argparse.Namespace, client: Client) -> int:
    rows = client.get("/approvals", params={"status": args.status})["approvals"]
    if not rows:
        print(f"no {args.status} approvals")
        return 0
    for row in rows:
        print(f"[{row['id']}] {row['kind']}: {row['subject']}   ({row['status']})")
        for line in (row.get("rationale") or "").splitlines():
            print(f"      {line}")
    return 0


def cmd_approve_base_change(args: argparse.Namespace, client: Client) -> int:
    result = client.post(f"/approvals/{args.id}", json={"approve": True})
    if result.get("approved"):
        plan = result["plan"]
        print(f"approved: the base model is now {plan['base_model']}")
        print("adapters trained on the previous base no longer apply")
        return 0
    print(result.get("reason", "not approved"), file=sys.stderr)
    return 1


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
    p.add_argument("--no-stream", action="store_true", help="wait for the whole answer")

    p = sub.add_parser("model", help="list or fetch model weights")
    model_sub = p.add_subparsers(dest="action", required=True)
    model_sub.add_parser("list", help="what is on disk, and what the ladder offers")
    pull = model_sub.add_parser("pull", help="fetch a checkpoint (uses the network)")
    pull.add_argument("model_id")

    p = sub.add_parser("bench", help="measure tokens/s, latency and peak VRAM")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("adapters", help="adapter modules, versions and what is loaded")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("rollback", help="roll an adapter back to its previous version")
    p.add_argument("module")

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

    p = sub.add_parser("plan-hardware", help="size the system for this machine")
    p.add_argument("--balance", type=float, help="0.0 = fastest ... 1.0 = most capable")
    p.add_argument("--apply", action="store_true", help="store the plan as active")
    p.add_argument("--no-training", action="store_true", help="plan for inference only")
    p.add_argument("--model", help="pin a base model from the ladder")
    p.add_argument("--max-seq-len", type=int)
    p.add_argument("-v", "--verbose", action="store_true", help="also list what was rejected")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--offline", action="store_true",
        help="plan without a running daemon (answers 'will this run here?')",
    )
    p.add_argument(
        "--simulate", metavar="PRESET",
        help="plan for a simulated machine instead: tiny-4gb, small-12gb, "
             "server-multigpu, cpu-only (implies --offline)",
    )

    p = sub.add_parser("hardware", help="what was detected, and the active plan")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("approvals", help="things waiting for your decision")
    p.add_argument("--status", default="pending", choices=["pending", "approved", "rejected", "all"])

    p = sub.add_parser("approve-base-change", help="accept a proposed base-model change")
    p.add_argument("id", type=int)

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
    if args.command == "plan-hardware" and (args.offline or args.simulate):
        return cmd_plan_hardware_offline(args)

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
        if args.command == "model":
            return cmd_model(args, client)
        if args.command == "bench":
            return cmd_bench(args, client)
        if args.command == "adapters":
            return cmd_adapters(args, client)
        if args.command == "rollback":
            return cmd_rollback(args, client)
        if args.command == "plan-hardware":
            return cmd_plan_hardware(args, client)
        if args.command == "hardware":
            return cmd_hardware(args, client)
        if args.command == "approvals":
            return cmd_approvals(args, client)
        if args.command == "approve-base-change":
            return cmd_approve_base_change(args, client)
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
