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

Built in reviewable stages. **Stages 1-5 are complete** — it loads real weights, answers,
learns from the day, studies, practises under a verifier, trains LoRA adapters and
promotes them only if a regression gate says they are better — and stage 6 is scaffolded
behind the interfaces it defines. `ai <command>` tells you when a command arrives.

| Stage | What it is | State |
|---|---|---|
| 1 | Daemon: state machine, scheduler, child supervision, journal, crash recovery, control plane | **done** |
| 2 | Hardware detection, memory pre-flight, auto planner, base-model approvals | **done** |
| 3 | Inference harness (transformers + PEFT), streaming chat, adapters, the wake phase | **done** |
| 4 | Backups, verification, atomic promotion, the weight-change guard | **done** |
| 5 | Sleep phase: consolidation, study, practice, training, the gate, the report | **done** |
| 6 | Self-expansion, self-optimisation, packaging | next |

Measured on the development machine (RTX 4070 SUPER, 12 GB) with Qwen3-4B in 4-bit NF4:
**54 tok/s**, 25 ms to first token, 4.3 GiB peak VRAM. It hands the GPU over for real:
VRAM drops from 6.8 GiB to 1.8 GiB when a sleep step takes the card, and a chat request
takes it back and answers within two seconds, reload included.

A complete night on the small model (Qwen3-0.6B, bf16, so it fits beside everything else
on the card) takes **44 seconds** end to end in `tests/test_gpu_night.py`: snapshot,
consolidation, reflect, fetch, study, practice, dreaming, a two-step LoRA, the gate, the
retest and the report — each in its own child process, with the model loaded and dropped
per step. Two steps are still simulated and say so in the report: `expansion` and `speed`
belong to stage 6.

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

**It stays out of the way when the GPU is needed elsewhere.** If another process holds
more than `schedule.max_foreign_vram_mb` (2 GiB) or the card is busier than
`max_foreign_gpu_percent` (35%), it will not start a sleep run and says why — playing a
game should not mean fighting the daemon for the card. `wake.idle_unload_minutes` (15)
also releases the model after that long without a message, so several gigabytes of VRAM
are not held for nothing; the next message reloads it. `ai release-gpu` does it now.

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
  such as Qwen3 have their `<think>` block switched off, detected from the template
  rather than hardcoded. The backend takes a flag for it; nothing in the night turns it
  on yet, so practice answers are measured without a reasoning block.
- **Stop sequences match across token boundaries.** A model emitting `</tool` then `>`
  has produced `</tool>`, and a partial match at the tail is held back rather than
  streamed, so a client never sees half a stop sequence.
- **Prefix caching.** A multi-turn chat re-sends the whole conversation every turn, so
  the shared prefix — nearly the whole prompt — is reused instead of recomputed. In the
  example above, turn 3 reused 64 of 117 prompt tokens. It is a genuine trade-off:
  a reused cache computes the prefill differently, so a *greedy* answer can differ by a
  token from the same question asked cold. Set `tunable.prefix_cache: false` if you need
  bit-exactness.
- **The context budget keeps room for the answer.** Old turns are dropped to fit the
  window — summarising them instead is a stage-6 idea, and the interface for it is in
  `ContextBudget`. The system prompt and the newest message always survive, and a single
  over-long message is trimmed in the middle rather than cut off.

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

## The night

`ai sleep now`, or the scheduler, runs thirteen journalled steps. Every one of them is a
spawned child process that is handed two paths, rebuilds the project from them, works
through the SQLite store, and returns a result the journal keeps — so `ai runs <id>`
explains any night, finished or not, and the morning report is regenerated from the
store rather than from whatever the steps happened to print.

| step | what it does |
|---|---|
| `snapshot` | a verified backup. Critical: if it fails, the night stops here |
| `consolidation` | the day's turns and feedback become facts and training examples |
| `reflect` | name tonight's topics from the measured skill table |
| `fetch_docs` | fetch allowlisted documentation for those topics (the only network use) |
| `study` | turn documentation into cited answers and verified exercises |
| `practice` | attempt the exercises greedily; the verifier decides |
| `dreaming` | sample until something passes, and keep only that |
| `training` | one new LoRA version per module, written but not switched on |
| `gate` | measure the candidate against what is being served, and promote or refuse |
| `expansion` | *simulated — stage 6* |
| `retest` | ask the things it got wrong again, with the new weights |
| `speed` | *simulated — stage 6* |
| `report` | the HTML report for the run |

### Nothing is learned on the model's word alone

Every example carries a `verified` flag, and it is only set when something other than the
model agreed:

| evidence | sets `verified` | why |
|---|---|---|
| a `/correct` from you | yes | you are the authority on what you wanted |
| a `/good` from you | yes | you endorsed the answer that was given |
| a test, run in the sandbox | yes | the code either passes or it does not |
| an excerpt from documentation | no | grounded is not the same as correct |
| several samples agreeing with each other | no | self-consistency is evidence, not proof |
| the model saying so | never | this is the failure mode the whole design is about |

`training.verified_only` is off by default, which means unverified material is trained on
too — with lower weight, and with the gate as the backstop. Turning it on is the strict
setting.

### Consolidation: the day, in order of authority

```
consolidation  done  {'exchanges': 2, 'feedback': 2, 'corrections': 1, 'endorsements': 1,
                      'facts_new': 1, 'examples': 3, 'model': True,
                      'by_kind': {'correction': 1, 'fact': 1, 'preference': 1}}
```

- **A correction is the answer.** `/correct` produces an example weighted **ten times** a
  thumbs-up, marked verified, with the rejected answer kept beside it.
- **A thumbs-up is an endorsement**, so the answer that was given becomes an example.
- **A thumbs-down is a question, not an answer.** We know the reply was wrong; we do not
  know what the right one is. It becomes a *rewrite task* for `dreaming` to attempt under
  verification — never a training example on its own.
- **Implicit feedback is a candidate.** A rephrased question is counted and left for
  review. Training on "the user asked again" teaches noise.
- **A turn marked never-train cannot reach a dataset.** The join that pairs questions with
  answers drops the whole exchange if either side is untrainable, so a detected secret or
  a `/forget` is excluded by construction rather than by a later filter that might be
  forgotten.

### Facts are stated in the third person, and superseded rather than overwritten

Extraction is deliberately narrow — a false fact about you is worse than a missed one,
because it gets trained on and then asserted. Only statements, never questions; secrets
never; and every fact is rewritten into the third person, because training on *"my editor
is helix"* teaches the model that **its** editor is helix.

```
$ ai facts
4 facts held
  0.90  The user's name is Simon.  [name]
  0.75  The user's editor is helix.  [editor]
  0.75  The user's GPU is an RTX 4070 SUPER.  [gpu]
  0.70  The user prefers short answers without preamble.

$ ai facts --subject editor
everything it has believed about 'editor':
  was  2026-11-11 16:24:38  The user's editor is vim.
  now  2026-11-11 16:24:38  The user's editor is helix.
```

A newer statement about the same subject supersedes the old one, which leaves the history
readable and makes a wrong extraction retractable rather than invisible. When a model is
available it is also asked to read the day's messages for facts the patterns missed — and
every statement it proposes is checked for being *grounded* in the message it came from.
An invention is thrown away; the overlap test is cheap and catches the common case.

### Reflect: topics are measured, not guessed

```
$ ai skills
4 of 4 topics measured, mean score 0.57
  module           topic             score  trials  demand
  rust             lifetime           0.29       5    0.57
  rust             borrow             0.43       5    0.39
  coding           regex              0.71       5    0.29
  coding           async              0.86       5    0.20
```

Every verified attempt — a practice task, a gate item, a retested failure, a `/correct` —
is a trial against a `(module, topic)` pair, where topics are the keywords the modules
declare in `modules/registry.yaml`. The score is Laplace-smoothed, so one lucky pass is
not a skill, and the ranking is `(1 - score) x (0.25 + demand)`: being bad at something
nobody asks about does not earn a night of GPU time.

Topics with **no** measurement are not treated as weaknesses — "we have never tested this"
is a different problem — but a topic in demand and never measured is the fallback choice
when nothing is measurably weak. A request no module covered at all is recorded as a
**gap**, which is a measured trigger for self-expansion in stage 6 and goes in the report
so the decision stays yours.

### Study: cited answers, and exercises that are checked twice

Documentation is chunked on headings, so a citation means something, and each chunk
becomes question/answer pairs with the URL and heading attached. Every model-written
answer must be grounded in its excerpt or it is discarded.

Exercises are the interesting part. The model is asked for a task **with a reference
solution and tests**, and the task is admitted only if the sandbox agrees twice:

1. the reference solution passes the tests — otherwise the task is incoherent; and
2. the tests **fail** when every function is replaced by a stub — otherwise they assert
   nothing.

That second check is what makes self-generated practice worth anything. A model that
writes `assert True` has produced a task everything passes, and a night of practice
against it measures nothing while looking productive. On the 0.6B model most proposals are
rejected outright, which is the honest outcome and visible in the report.

### Practice measures; dreaming manufactures

Both run the same tasks, and the difference is the point:

- **`practice` is a measurement.** One attempt per task, greedy, and the result goes into
  the skill table whether it passed or not. If it sampled until something worked, the
  number would describe the sampler, and `reflect` would then pick tonight's topics from
  it.
- **`dreaming` is data generation.** Up to four samples at temperature 0.8, keeping only
  what passes the tests — rejection sampling, where the rejection is the valuable part.
  It also answers the questions you marked bad: several candidates are sampled and
  compared with *each other*, and a replacement is kept only if they agree and the winner
  does not simply restate what you rejected.

### The sandbox

A practice result is a measurement because the code is actually run:

- a wall-clock timeout, then `SIGKILL` to the whole session, so an infinite loop costs
  seconds and a child that spawned children does not outlive it;
- `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NOFILE`, `RLIMIT_FSIZE` and `RLIMIT_NPROC`, set by
  the program's own prelude with soft and hard limits equal so the candidate cannot raise
  them again (a `preexec_fn` would run Python between fork and exec in a process that has
  torch's threads in it — a documented way to deadlock);
- a throwaway working directory, an almost empty environment and `python -I`;
- a prelude that blocks sockets, `subprocess` and `os.system`.

That last one is a guardrail against generated code casually reaching for the network, not
a security boundary, and the docstring says so. The code being run is written by a local
model from local documentation; if you intend to run *untrusted* code, put a container
under `Sandbox._spawn`, which is the one place a process is started.

### Training: replay, weights, and the guard

```
training       done  {'modules': ['coding'], 'trained': 1, 'skipped': 0}
```

- **Replay against forgetting.** A LoRA trained on nothing but tonight's forty examples
  will learn them and forget the rest. `sleep.replay_fraction` (0.30) of every mix is
  drawn from examples earlier nights already learned, spread across the module's history
  rather than taken newest-first.
- **Weights are a loss scale, normalised to mean 1.0.** Without the normalisation, raising
  a weight would raise the total gradient and act as a hidden learning-rate change — the
  night would train harder simply because you made more corrections.
- **Only the completion is trained on.** The prompt is rendered with the model's own chat
  template, exactly as the server renders it, and masked out with `-100`.
- **It can be interrupted.** A message mid training saves the adapter, records the step,
  and gives the card back; the next night resumes from there instead of restarting.
- **It writes a version; it does not switch one on.** The adapter lands in
  `adapters/<module>/vNNN` with `active` untouched, inside `guard.change(...)` — so a
  verified snapshot exists first, and a failure restores from it.

### The gate

```
$ ai runs 7
run 7  completed  trigger=window
  ✓ snapshot      done      snapshot 20261111T030112-sleep-run-7, size mb 4.2
  ✓ consolidation done      exchanges 31, examples 24, facts new 3, corrections 2
  ...
  ✓ training      done      trained 1
  ✓ gate          done      promoted coding
```

The step the whole design exists to make possible. A system that trains on its own output
every night will drift, and the only defence is a measurement it cannot talk its way past:

- **Both suites, both versions, the same prompts, greedy, in one process.** The candidate
  and the weights currently being served answer identical questions. Comparing tonight's
  number with one from last week would measure the weather.
- **Better on held-out, *and* not worse on what it already knew.** The held-out split is
  decided by each example's fingerprint before training begins, so it cannot drift into
  the training set on a later night. The retention suite is material from *earlier* nights
  — explicitly excluding what tonight trained on, because measuring retention with the
  answers you were just given flatters the candidate.
- **Too few items is a `skip`, not a promotion.** Below `gate.min_items` there is no
  measurement, and "we did not measure it" must never be a reason to switch weights on.
- **Promotion is one guarded `os.replace` of a symlink.** A crash at any instant leaves
  the previous version live.
- **One module per night by default.** If something breaks tomorrow, the cause should be a
  single, named change.

A rejected candidate stays on disk, unpromoted, with its scores in `gate_results` — so
"why did it not promote" is answerable, and a disabled gate refuses to promote rather than
promoting blind.

### The report

`ai report [run]` writes `var/reports/run-NNNNNN.html` (and `latest.html`): one
self-contained file, no JavaScript, no external fonts, light and dark. It leads with what
changed and why — promotions, gate deltas, the step table with durations and failures —
then the facts it now holds, what it read with the licence you recorded for each source,
where it still struggles, and the night's log.

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
with its scores, which is what will let the `speed` step train it on outcomes instead of
thresholds (stage 6; today the threshold is `tunable.router_threshold`).
When nothing scores above the threshold, that is recorded as an *uncovered topic* — one
of the measured triggers for self-expansion in stage 6.

Modules are held in a tiered cache (VRAM → pinned RAM → disk, LRU) inside the adapter
budget the hardware plan computed, so the VRAM ceiling is never exceeded by accident.

## Backups

Section 9 of the specification is a rule, not a feature: *no operation that adds,
changes, merges or removes weights may start without a verified backup; if the backup
fails, the operation is refused.* That rule is a single context manager, and every weight
change goes through it:

```python
with guard.change("promote coding v004", sources=("adapters",)) as change:
    ...                                  # write the new weights
    change.activate(link, "v004")        # atomic symlink swap
```

On entry it takes a snapshot and **verifies it by re-hashing**; if that fails for any
reason — disk full, permissions, a bad hash — it raises and the body never runs. If the
body raises, the touched sources are restored from that snapshot before the error
propagates. A test asserts the body does not execute when the backup fails, and
[a structural test](tests/test_backup_invariants.py) reads the source to prove the guard
is the only path: it parses `daemon.py` and fails if an `activate_adapter` or
`rollback_adapter` call is not lexically inside a `with ... guard.change(...)` block.

**Snapshots** go to `backups/<timestamp>-<operation>/` with a manifest of SHA-256 hashes
covering the adapters and their metadata, the module registry, the SQLite store, the
config, the profiles and hardware plan, the router and draft model, and the held-out
benchmarks. The base checkpoint is included only for operations that could change it,
since it is several gigabytes.

Three details that matter more than they look:

- **The database is copied through SQLite's own backup API**, not as a file. In WAL mode
  the newest commits live in the `-wal` sidecar and a plain copy can catch a torn state.
- **Symlinks are recorded as symlinks, not followed.** `adapters/<module>/active` *is*
  the promotion mechanism; resolving it would quietly turn a pointer into a copy.
- **The manifest is written last and atomically.** A snapshot directory without one is by
  definition incomplete, which is how an interrupted backup is told apart from a finished
  one — and incomplete ones are never restored from.

**A restore is itself a change**, so it snapshots the current state first and tells you
the id, which makes "I restored the wrong one" recoverable. Restores are staged beside the
target and swapped in, so a crash mid-restore leaves the old tree or the new one, never a
mixture — and a failed swap puts the original back.

**An interrupted promotion is undone at startup.** Every change is logged in
`weight_changes` with its snapshot id; an entry still `in_progress` when the daemon starts
is by definition an interrupted operation, and recovery restores from its snapshot before
anything else runs.

```
$ ai backup list
snapshots in /home/simon/devel/gustav/backups  (1649.2 GiB free)
retention: keep the last 5, one daily for 7 days, one weekly for 4 weeks
  20261110T164051-manual             manual               0.1 MiB   3 files
  20261110T164102-promote-coding     promote coding v004  4.2 MiB  18 files  protected

recent weight changes:
  2026-11-10 16:41:02  done  promote coding v004  snapshot=20261110T164102-promote-coding
```

`ai backup create [--dry-run]` (the dry run shows what would be copied and how much space
it needs), `ai backup verify <id>`, `ai backup restore <id> [--module M] [--only adapters]`,
`ai backup retention [--dry-run]`.

**Retention** keeps the last N, one a day for M days and one a week for K weeks — and
never the newest complete snapshot or one a pending approval refers to. When the location
runs low it *warns*; it does not delete beyond the policy, because losing backups to free
space is precisely the failure backups exist to prevent.

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
| `ai backup list\|create\|verify\|restore\|retention` | snapshots; every weight change takes one first |
| `ai release-gpu` | unload now and free the VRAM (reloads on the next message) |
| `ai runs [id]` | sleep runs and their steps, with what each one did, checkpoints and errors |
| `ai facts [--subject S]` | durable facts it holds; `--subject` shows everything it has believed about one |
| `ai skills [--all]` | the measured skill table: score, trials and demand per topic |
| `ai report [run]` | write the HTML report for a run (the newest by default) |
| `ai events [--kind K]` | the daemon's event log |
| `ai stop` | graceful shutdown |

Stage 6 adds `ai approve-expansion`, `ai expansion --revert` and `ai optimize --revert`.

Everything goes through the running daemon over a Unix socket at `var/gustav.sock`.
Set `instance.http_enabled` (and `http_token`) to also listen on TCP for remote clients.

The HTTP API has `/chat` (JSON or SSE), `/v1/chat/completions` (OpenAI-compatible, so
existing clients work against the local daemon), `/health`, `/status`, `/adapter/status`,
`/adapter/rollback`, `/adapter/activate`, `/hardware`, `/models`, `/bench`, `/runs`,
`/facts`, `/skills` and `/report`.

## Configuration

`config.yaml` in the project root; every key has a default, so an empty file is valid and
an unknown key is an error rather than a silent default. The `tunable` section is the only
thing the daemon's own self-optimisation may ever change (stage 6) — never its source code.

The sections the night reads, with the defaults worth knowing:

| section | what it controls |
|---|---|
| `sleep` | `replay_fraction` (0.30), `heldout_fraction` (0.15), `max_topics_per_run` (5), `weak_score_threshold` (0.6), `use_model`, and `simulated_steps` — the steps that still run the placeholder |
| `fetcher` | `enabled`, `allowed_hosts`, `local_sources`, `max_documents_per_run` (12), `cache_ttl_hours` (168), `delay_seconds` (1.0) |
| `sandbox` | `timeout_seconds` (10), `memory_limit_mb` (1024), `isolate_env` |
| `training` | `rank` (from the plan), `learning_rate` (1e-4), `grad_accum` (8), `max_steps` (200), `max_seconds` (1800), `min_examples` (16), `verified_only` (off) |
| `gate` | `min_improvement` (0.0), `max_regression` (0.02), `min_items` (8), `max_promotions_per_run` (1) |

`docs/sources.yaml` maps topics to documentation URLs and records the licence of each
source. It is the complete list of what the fetcher can ever request, so an entry you did
not add is a page it will never visit.

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
the small one (see `profiles/tiny-4gb.yaml`) and copy the adapter versions over. The gate
runs where the training happened, so the receiving machine activates a version
deliberately — `POST /adapter/activate`, or `ai rollback` to step back — rather than
re-measuring it.

## Layout

```
daemon.py              the entry point
config.yaml            your settings
profiles/              hardware profiles, including the generated one
docs/sources.yaml      topic -> documentation map, with licences; the fetcher's whole world
modules/registry.yaml  the modules, their descriptions and their topic keywords
src/gustav/
  core/                state machine, scheduler, supervisor, journal, GPU lease, recovery
  hw/                  detection, the model ladder, memory model, pre-flight, planner
  inference/           backend, generation loop, sampling, adapters, router, caches
  backup/              snapshots, manifests, atomic swaps, restore, retention, the guard
  sleep/               the pipeline, its steps, and the model as a step sees it
  learn/               facts, examples, tasks, skills, benchmarks, chunking, grounding
  fetcher/             the allowlist, the cache, HTML to text — the only network use
  sandbox/             where practice answers are run and judged
  train/               the dataset mix and the LoRA trainer
  report/              the HTML report
  wake/                the chat side
  server/ cli.py       control plane and its client
  store/               SQLite schema and migrations
var/                   databases, logs, run checkpoints, doc cache, reports, socket, lock
```

## Tests

```bash
.venv/bin/python -m pytest -q                 # everything except the GPU set
.venv/bin/python -m pytest -q -m "not slow"   # skip the tests that start real processes
.venv/bin/python -m pytest -q -m gpu          # real weights on a real card
```

Three sets, with different jobs.

**The fast set** (around 700 tests) runs on any machine with no GPU and no weights. Most
of the sleep tests call a step's body directly with a *scripted* model, which is what
makes the interesting cases cheap to pin down: an ungrounded fact, an exercise whose tests
pass without a solution, a candidate adapter that is better on new material and worse on
old, a replacement answer that just restates the one you rejected.

**The slow set** starts the real daemon as a subprocess and `kill -9`s it while **each**
step of the night is running, then restarts it and checks that finished steps were not
re-run, that checkpoints survived, and that the night finishes. `tests/test_sleep_night.py`
runs the whole pipeline in real child processes with no model at all — which is how a
laptop without CUDA behaves, and it has to end with a completed run and a report.

**The GPU set** needs `models/qwen3-0.6b` (`ai model pull qwen3-0.6b`) and takes about a
minute: `test_gpu_training.py` trains a LoRA, checks the loss falls when it is shown the
same thing repeatedly, interrupts training and resumes it, and loads the result through
the *serving* path — and `test_gpu_night.py` runs an entire night with the model loaded
and asserts only on structure, never on the quality of a 0.6B model's answers.

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
- **Preference pairs are trained as plain SFT on the chosen side.** `/good`, `/bad` and
  the rewrites `dreaming` produces are stored as pairs, with the rejected answer kept, but
  the trainer fits the preferred answer rather than running DPO. The data is in the shape a
  preference objective needs; the objective is not implemented.
- **One GPU means no chatting during training.** Sleep pauses to answer you and resumes
  afterwards. Two GPUs remove the trade-off.
- **Training on your own output has a ceiling.** Test verification, documentation
  grounding and the gate raise it; they do not remove it. Expect small, measurable gains
  on narrow verifiable skills, not open-ended self-improvement. A night with no
  corrections, no practice passes and nothing new to read is a night that should and
  usually does change nothing.
- **The weakest link is scoring free text.** Code is run; a `/correct` is authoritative.
  But a held-out prose answer is scored by content-word overlap with the reference, which
  is a blunt instrument: it cannot tell a differently-worded right answer from a wrong
  one. That is why the gate needs a minimum number of items, refuses to promote on a tie,
  and promotes one module a night. An embedding or a judge model behind the same
  `score_answer` interface is the obvious improvement.
- **Self-generated exercises are only as good as their tests.** Both checks (the reference
  solution passes, the stubbed version fails) catch incoherent and vacuous tasks, but not
  a task that tests the wrong thing correctly. Small models mostly fail to produce usable
  JSON for this at all, which the report counts rather than hides.
- **Documentation licences are your call.** `docs/sources.yaml` records the licence you
  declared for each source; the fetcher stores it with every document, the report shows
  it, and a source with no licence is fetched *and* warned about. Nothing verifies the
  declaration, and nothing refuses to study without one.
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
