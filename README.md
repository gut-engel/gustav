# Gustav

A local, self-improving chat AI with a wake/sleep cycle, run by **one always-on process**.

While awake it chats and records experience. While asleep it consolidates what it learned,
studies documentation on topics where it measurably struggled, practises in a verifiable
sandbox, trains LoRA adapters on the data that survived verification, optionally grows its
own capacity, and promotes new weights **only** if a regression gate passes. Everything is
local. The only component that touches the network is the documentation fetcher.

There is no cron job, no systemd timer and no external scheduler. If `daemon.py` is
running, the system is complete.

```
python daemon.py      # or: ai start
ai status             # talk to it from another terminal
ai chat "hello"
ai sleep now
```

## Status

Built in reviewable stages. **Stage 1 is complete**; later stages are scaffolded behind
the interfaces stage 1 defines, and `ai <command>` tells you when a command arrives.

| Stage | What it is | State |
|---|---|---|
| 1 | Daemon: state machine, scheduler, child supervision, journal, crash recovery, control plane | **done** |
| 2 | Hardware detection + auto planner | next |
| 3 | Inference harness (transformers + PEFT) and the real wake phase | planned |
| 4 | Backups, verification and atomic promotion | planned |
| 5 | Sleep phase: consolidation, study, practice, training, gate | planned |
| 6 | Self-expansion, self-optimisation, packaging | planned |

Today the responder is an echo and the sleep steps do simulated, checkpointed work. That
is deliberate: the hard parts of an always-on daemon — resuming an interrupted night,
killing a hung child, handing the GPU back mid-training, never leaving weights
half-written — are testable now, and the real work hangs off the same interfaces later.

## Install

Needs Python 3.12+ (3.14 is fine; torch 2.15 ships `cp314` wheels).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"      # add ".[ml]" from stage 3 onward
.venv/bin/python daemon.py
```

`pip install -e .` puts `ai` on your PATH inside the venv. Without installing, `python
daemon.py` works straight from a checkout.

## How the daemon schedules itself

One asyncio process holds the state machine:

```
STARTING ──▶ WAKE ──▶ SLEEP_PENDING ──▶ SLEEP ──▶ WAKE
                │                                   │
                └──────── SAFE_MODE / ERROR ◀───────┘
```

Four triggers can start a sleep, in any combination (`config.yaml`, section `schedule`):

1. **Window** — a local-time window, `00:00-06:00` by default.
2. **Idle** — no user activity for `idle_minutes` (45), inside a window or when pending
   work is above the threshold.
3. **Pending work** — enough unprocessed experience has accumulated (`pending_threshold`).
4. **Manual** — `ai sleep now`, `ai wake`.

Guards that stop it being annoying: `min_idle_seconds` (never interrupt a live
conversation — and the same guard governs how soon a paused night resumes),
`cooldown_minutes` between nights, `max_duration_minutes` per night, and
`max_gpu_temp_c` to pause when the card is too hot.

**A message during sleep wins.** The daemon asks the running step to stop, the child
checkpoints at a step boundary and exits, the GPU lease comes back to inference, you get
your answer, and the night resumes later from the journal. With separate inference and
training GPUs this never has to happen at all (`inference_gpus` / `training_gpus`, stage 2).

## What makes it survive being killed

- **One lock.** An exclusive `flock` on `var/gustav.pid`. A second daemon refuses to
  start; a daemon killed with `-9` leaves no lock to clean up, because the kernel drops it.
- **A journalled pipeline.** Every step of the night is written to SQLite before it runs,
  with a status, an attempt count, a checkpoint and an `on_crash` policy:
  `resume` (continue from the checkpoint), `restart` (cheap, redo it cleanly),
  `rollback` (undo and abandon the night — used for anything that changes weights),
  `skip`. Startup reads the journal and applies those policies before serving.
- **Children, not threads.** Training, sandboxing and fetching run in `spawn`ed
  processes with a heartbeat deadline, a wall-clock timeout, an `RLIMIT_AS` cap and
  SIGTERM→SIGKILL escalation. A CUDA OOM or a segfault in a child is an event in the log,
  not an outage.
- **A watchdog.** A child that stops heart-beating is killed, logged, and retried with
  backoff. After `max_attempts` the step is skipped for the night and shown in the report.
- **One choke point for weight changes.** From stage 4, nothing writes adapters or base
  checkpoints except through a guard that takes and verifies a backup first and swaps an
  `active` symlink atomically. A crash at any instant leaves the previous working version live.
- **Safe mode.** If the active adapter will not load or fails its health prompt at
  startup, the daemon serves the last known-good weights (or the plain base model) and
  says so in `ai status` instead of flapping.

## Commands

| | |
|---|---|
| `ai start` | run the daemon in the foreground |
| `ai status [--json]` | state, schedule, adapters, GPU holder, pending work, failures, approvals |
| `ai chat [message]` | one message, or an interactive session |
| `ai sleep now [--dry-run]` | sleep immediately; `--dry-run` plans the night without doing it |
| `ai wake` | pause a running sleep and come back online |
| `ai pause-learning` / `ai resume-learning` | stop/allow sleeping and training |
| `ai runs [id]` | sleep runs and their steps, with checkpoints and errors |
| `ai events [--kind K]` | the daemon's event log |
| `ai stop` | graceful shutdown |

Later stages add `ai plan-hardware`, `ai backup list|verify|restore`,
`ai approve-expansion`, `ai approve-base-change`, `ai expansion --revert`,
`ai optimize --revert`, `ai facts`, and `ai report`.

Everything goes through the running daemon over a Unix socket at `var/gustav.sock`.
Set `instance.http_enabled` (and `http_token`) to also listen on TCP for remote clients.

## Configuration

`config.yaml` in the project root; every key has a default, so an empty file is valid and
an unknown key is an error rather than a silent default. The `tunable` section is the only
thing the daemon's own self-optimisation may ever change (stage 6) — never its source code.

## Layout

```
daemon.py              the entry point
config.yaml            your settings
src/gustav/
  core/                state machine, scheduler, supervisor, journal, GPU lease, recovery
  sleep/               the pipeline and its steps
  wake/                the chat side
  server/ cli.py       control plane and its client
  store/               SQLite schema and migrations
var/                   databases, logs, run checkpoints, the socket and the lock
```

## Tests

```bash
.venv/bin/python -m pytest -q              # everything
.venv/bin/python -m pytest -q -m "not slow"  # skip the tests that start real processes
```

The slow set starts the real daemon as a subprocess and `kill -9`s it while **each** step
of the night is running, then restarts it and checks that finished steps were not re-run,
that checkpoints survived, and that the night finishes.

## Known limitations

- **A dense model has no "coding part" to split out.** Modularity here means LoRA
  adapters and, for a Mixture-of-Experts base, expert offloading. Nothing else.
- **No layer or depth growth.** Self-expansion adds adapters, raises LoRA ranks, adds
  frozen-base experts, and (with your approval) merges into the base. Growing the network
  itself is documented as a future direction, not implemented.
- **One GPU means no chatting during training.** Sleep pauses to answer you and resumes
  afterwards. Two GPUs remove the trade-off.
- **Training on your own output has a ceiling.** Compiler and test verification plus
  documentation grounding raise it; they do not remove it. Expect small, measurable gains
  on narrow verifiable skills, not open-ended self-improvement.
- **Documentation licences are your call.** The fetcher records the licence per source and
  warns when it does not obviously permit training.
