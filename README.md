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

Built in reviewable stages. **Stages 1-3 are complete** — it loads real weights and
answers — and later stages are scaffolded behind the interfaces they define.
`ai <command>` tells you when a command arrives.

| Stage | What it is | State |
|---|---|---|
| 1 | Daemon: state machine, scheduler, child supervision, journal, crash recovery, control plane | **done** |
| 2 | Hardware detection, memory pre-flight, auto planner, base-model approvals | **done** |
| 3 | Inference harness (transformers + PEFT), streaming chat, adapters, the wake phase | **done** |
| 4 | Backups, verification and atomic promotion | next |
| 5 | Sleep phase: consolidation, study, practice, training, gate | planned |
| 6 | Self-expansion, self-optimisation, packaging | planned |

Measured on the development machine (RTX 4070 SUPER, 12 GB) with Qwen3-4B in 4-bit NF4:
**54 tok/s**, 25 ms to first token, 4.3 GiB peak VRAM. The sleep pipeline still does
simulated, checkpointed work — but it already hands the GPU over for real: VRAM drops
from 6.8 GiB to 1.8 GiB when a sleep step takes the card, and a chat request takes it
back and answers within two seconds, reload included.

The sleep steps still do simulated, checkpointed work. That is deliberate: the hard
parts of an always-on daemon — resuming an interrupted night, killing a hung child,
handing the GPU back mid-training, never leaving weights half-written — are tested now,
and the real work hangs off the same interfaces in stage 5.

## Install

Needs Python 3.12+ (3.14 is fine; torch 2.15 ships `cp314` wheels) and an NVIDIA GPU.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ml,dev]"   # ~4 GB of CUDA wheels
.venv/bin/python daemon.py             # plans, then loads the model it chose
```

In another terminal:

```bash
ai model pull qwen3-4b     # ~7.5 GiB, the only part that uses the network
ai chat "what can you do?"
ai bench                   # measures this machine; the planner then stops guessing
```

`pip install -e .` puts `ai` on your PATH inside the venv. Without installing, `python
daemon.py` works straight from a checkout. Without the `ml` extra the daemon still runs
end to end with a placeholder responder (`inference.responder: echo`), which is how most
of the test suite stays fast.

**FlashAttention-2 is optional.** It is not in the extras because it builds from source;
install `flash-attn` yourself and the planner picks it up. Without it, PyTorch SDPA is
used, and the plan says so rather than claiming otherwise.

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

## Chatting

```
$ ai chat
Chatting with gustav. /help for commands, Ctrl-D to leave.
you> name three rust crates for async IO, just the names
ai > tokio, async-std, tokio-stream
[50.0 tok/s, 91 ms to first token, modules: personal-memory, rust, general]
```

Tokens stream as they are produced. The harness is ours end to end — transformers and
PEFT for the weights, but the generation loop, the sampling (temperature, top-p, top-k,
min-p, repetition penalty), the stop sequences and the KV handling are this project's:

- **The model's own chat template**, never a format we invented. Hybrid reasoning models
  such as Qwen3 have their `<think>` block switched off for chat and available for the
  practice tasks in stage 5, detected from the template rather than hardcoded.
- **Stop sequences match across token boundaries.** A model emitting `</tool` then `>`
  has produced `</tool>`, and a partial match at the tail is held back rather than
  streamed, so a client never sees half a stop sequence.
- **Prefix caching.** A multi-turn chat re-sends the whole conversation every turn, so
  the shared prefix — nearly the whole prompt — is reused instead of recomputed. In the
  example above, turn 3 reused 64 of 117 prompt tokens. It is a genuine trade-off:
  a reused cache computes the prefill differently, so a *greedy* answer can differ by a
  token from the same question asked cold. Set `tunable.prefix_cache: false` if you need
  bit-exactness.
- **The context budget keeps room for the answer.** Old turns are dropped (or, from
  stage 5, summarised) to fit the window; the system prompt and the newest message always
  survive, and a single over-long message is trimmed in the middle rather than cut off.

Commands are handled before the model sees anything, and they are your control over what
the system learns:

| | |
|---|---|
| `/good`, `/bad` | becomes a preference pair |
| `/correct <answer>` | highest-priority training data (weighted 10x against a thumb) |
| `/forget [all]` | the last exchange, or the session, is never trained on — and the queued work for it is withdrawn |
| `/clear`, `/help` | fresh conversation, the command list |

**Secrets never reach the training data, or the disk.** Every message is scanned for API
keys, tokens, private keys, credentials in URLs and `password: …` assignments. A hit
marks the turn never-train *and* stores a redacted copy, because the store is backed up,
copied between machines and summarised into reports:

```
$ ai chat "my key is sk-proj-abc... please remember it"
(a openai-key was detected: stored redacted, never trained on)
I cannot store or remember your key. ...

$ sqlite3 var/gustav.db "select content, trainable from turns limit 1"
my key is [REDACTED:openai-key] please remember it|0
```

The live request still sees the original text, so the answer is unaffected — only what is
persisted changes. `wake.redact_secrets: false` keeps the raw text on disk; it is still
never trained on. Detection is deliberately eager: a false positive costs one turn of
training data, a false negative puts a credential in a file that gets copied around.

**Implicit feedback is a candidate, not a verdict.** If you rephrase a question or say
"no, that's wrong", that is recorded with a confidence and a reason and waits for review.
It is never trained on by itself.

## Adapters and routing

Adapters live in `adapters/<module>/vNNN` with an `active` symlink. Promotion is a symlink
swap via `os.replace`, so a crash at any instant leaves the previous version live and
loadable, and `ai rollback <module>` goes back a version without restarting the model.
An adapter trained on a different base model is detected and skipped rather than loaded
into weights it does not belong to.

A router picks which modules a request needs *before* generation, so the right adapters
are resident when the first token is produced. It scores the request against each
module's description with IDF-weighted keyword overlap — cheap and explainable, and
replaceable by an embedding model behind the same interface. Every decision is logged
with its scores, which is what lets stage 5 train it on outcomes instead of thresholds.
When nothing scores above the threshold, that is recorded as an *uncovered topic* — one
of the measured triggers for self-expansion in stage 6.

Modules are held in a tiered cache (VRAM → pinned RAM → disk, LRU) inside the adapter
budget the hardware plan computed, so the VRAM ceiling is never exceeded by accident.

## Hardware and the auto planner

`ai plan-hardware` sizes the system for the machine it is on. It reads the GPUs through
NVML (no torch needed), computes what each model on the ladder would cost in VRAM, and
picks **the largest model that fits and is fast enough**:

```
$ ai plan-hardware -v
plan: qwen3-4b @ 4bit-nf4, seq 4096, LoRA r16, ~36 tok/s (estimated), training on GPU 0

  Chose Qwen3 4B (4.02B, 36 layers, 8 KV heads) in 4-bit NF4 (bitsandbytes, double quant).
  It is the largest model on the ladder that fits and is fast enough: balance 0.5 turns
  the 15 tok/s floor into an effective 30 tok/s requirement.
  GPU0 NVIDIA GeForce RTX 4070 SUPER: 12.0 GiB, Ada Lovelace (sm_89), ~504 GB/s, bfloat16,
  flash_attention_2
  GPU0: 10.20 GiB usable of 11.99 GiB (15% reserved for other software, 690 MiB already in
  use by others)
  Inference at seq 4096 needs 4.43 GiB, leaving 5.77 GiB spare.
  QLoRA training at seq 1024, rank 16 needs 5.35 GiB, so this machine can train itself.
  ...
not chosen:
  qwen3-8b@4bit-nf4   seq 4096  8.11 GiB  17 tok/s  rung 4  17 tok/s is below the 30 tok/s
                                                            floor that balance 0.5 demands
  qwen3-14b@4bit-nf4  seq 4096 12.48 GiB  10 tok/s  rung 5  needs 12.48 GiB but only 10.20
                                                            GiB is usable (2.28 GiB short)
```

**One knob.** `hardware.balance`, from 0.0 to 1.0, does its work by raising the speed
floor rather than by trading capability away point by point — which is what keeps it from
quietly picking a tiny model on a big card. On the 12 GB card above: 0.0 gives Qwen3-1.7B
at ~73 tok/s with speculative decoding on, 0.5 gives Qwen3-4B, 1.0 gives Qwen3-8B at
~17 tok/s.

**The memory model is explicit**, because its job is to refuse a configuration before it
OOMs rather than to be clever. Each estimate is broken down (`ai plan-hardware -v`), and
the two terms usually left out of back-of-the-envelope sizing are both in it:

- **Unquantised weights.** bitsandbytes only touches `nn.Linear`, so Qwen3's
  151 936-token embedding and lm_head stay in 16-bit — 2.5 GiB on the 8B model, more than
  a third of its 4-bit footprint.
- **Training logits.** Cross-entropy over the whole sequence materialises
  `batch x seq x vocab` in fp32: ~1.2 GiB at seq 1024, typically the single largest
  training allocation.

A refusal says what to change: *"needs 12.48 GiB but only 10.20 GiB is usable on GPU0
(2.28 GiB short); try max_seq_len 2048"*.

**The base model never changes by itself.** Adapters are tied to the base they were
trained on, so when the planner would prefer a different one — because you changed
`balance`, or moved the disk into a bigger machine — it files a pending approval, keeps
serving the current base, and tells you:

```
$ ai approvals
[1] base_change: qwen3-1.7b -> qwen3-30b-a3b   (pending)
$ ai approve-base-change 1
```

**It notices the machine changing.** A fingerprint over GPU names, VRAM, compute
capability, RAM and core count is stored and compared at startup and on a timer. It
deliberately ignores anything that moves on its own — free VRAM, clocks, temperature,
what else is running — so it fires when you add a GPU, not when the desktop opens a tab.

**Other software on the card is respected.** Foreign VRAM is measured per process, and
the 15 % reserve and the measured foreign usage are not double-counted: the larger wins.

**Profiles.** `hardware.profile: auto` plans and writes `profiles/auto-generated.yaml`,
a self-documenting file you can edit — anything pinned there wins and the planner will not
overwrite it. Three presets ship: `small-12gb` (12 GB, 4B + QLoRA), `tiny-4gb` (Turing,
fp16 and SDPA only, serve here and train elsewhere) and `server-multigpu` (one card
serves while the others train, so chat stays up during sleep).

**Adding a model** is one entry in `src/gustav/hw/ladder.yaml`. Parameter counts, KV-cache
size and LoRA footprints are computed from the architecture, so nothing is hardcoded
elsewhere — and once the weights are on disk, `config.json` replaces the YAML's numbers
and the estimates become exact.

## Commands

| | |
|---|---|
| `ai start` | run the daemon in the foreground |
| `ai status [--json]` | state, schedule, adapters, GPU holder, pending work, failures, approvals |
| `ai chat [message]` | one message, or an interactive session |
| `ai sleep now [--dry-run]` | sleep immediately; `--dry-run` plans the night without doing it |
| `ai wake` | pause a running sleep and come back online |
| `ai pause-learning` / `ai resume-learning` | stop/allow sleeping and training |
| `ai plan-hardware [--balance X] [--apply] [-v]` | size the system for this machine, and say why |
| `ai hardware` | what was detected, the VRAM budget, the active plan |
| `ai approvals` / `ai approve-base-change <id>` | decisions waiting for you |
| `ai model list` / `ai model pull <id>` | what weights are on disk; fetch one (uses the network) |
| `ai bench` | measure tokens/s, latency and peak VRAM, and record it for the planner |
| `ai adapters` / `ai rollback <module>` | adapter versions and what is loaded; go back one version |
| `ai runs [id]` | sleep runs and their steps, with checkpoints and errors |
| `ai events [--kind K]` | the daemon's event log |
| `ai stop` | graceful shutdown |

Later stages add `ai backup list|verify|restore`, `ai approve-expansion`,
`ai expansion --revert`, `ai optimize --revert`, `ai facts`, and `ai report`.

Everything goes through the running daemon over a Unix socket at `var/gustav.sock`.
Set `instance.http_enabled` (and `http_token`) to also listen on TCP for remote clients.

The HTTP API has `/chat` (JSON or SSE), `/v1/chat/completions` (OpenAI-compatible, so
existing clients work against the local daemon), `/health`, `/status`, `/adapter/status`,
`/adapter/rollback`, `/hardware`, `/models`, `/bench` and `/runs`.

## Configuration

`config.yaml` in the project root; every key has a default, so an empty file is valid and
an unknown key is an error rather than a silent default. The `tunable` section is the only
thing the daemon's own self-optimisation may ever change (stage 6) — never its source code.

## Moving to a bigger machine

The plan is derived, not baked in, so the move is mostly a copy:

1. Copy the project and `var/` (the SQLite store and the adapters) to the new machine.
2. Start the daemon. It detects the new hardware, notices the fingerprint changed, and
   re-plans — but keeps serving the base model your adapters belong to.
3. `ai approvals` shows the base-model change it would prefer. Approving it is a
   deliberate act: adapters trained on the old base no longer apply.
4. With two or more GPUs the planner separates them by itself — `inference_gpus: [0]`,
   `training_gpus: [1, 2, 3]` — so chat stays up while it sleeps. Override either list in
   `config.yaml` if you want a different split.

To keep serving on a small machine and train on a big one, set `allow_training: false` on
the small one (see `profiles/tiny-4gb.yaml`) and copy adapters over; the serving machine
runs the same regression gate before activating anything (stage 5).

## Layout

```
daemon.py              the entry point
config.yaml            your settings
profiles/              hardware profiles, including the generated one
src/gustav/
  core/                state machine, scheduler, supervisor, journal, GPU lease, recovery
  hw/                  detection, the model ladder, memory model, pre-flight, planner
  inference/           backend, generation loop, sampling, adapters, router, caches
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

The planner is tested against simulated machines — a 4 GB Turing laptop, a 12 GB desktop,
a four-card server, and a machine with no GPU at all — so none of it needs the hardware it
plans for. `GUSTAV_SIMULATE_HARDWARE=server-multigpu python daemon.py` runs the daemon
against one of them too.

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
- **Speeds are estimates until you run `ai bench`.** The planner derives tokens/s from
  memory bandwidth and a per-quantisation efficiency factor, labelled "estimated"
  everywhere it appears. One benchmark replaces it with a measurement per model and
  quantisation. (On the development machine the estimate was 36 tok/s and the measurement
  54 — conservative, which is the safe direction.)
- **Requests are served one at a time.** The GPU is single-tenant, so generation is
  serialised behind a lock. The queue interface is in place for continuous batching
  later; it is not implemented yet, and concurrent callers wait their turn.
- **Architecture numbers come from `ladder.yaml`** until the weights are downloaded;
  the planner says so in its warnings, and reads the real `config.json` when it exists.
