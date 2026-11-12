/* The web interface.
 *
 * No framework and no build step: the daemon serves three files from the package and
 * the browser talks to the same HTTP API the CLI uses. Everything here is a view over
 * endpoints that already existed — which is why the owner's panels and an ordinary
 * account's chat list behave identically whether you reach them from here or from `ai`.
 *
 * Nothing is ever inserted as HTML. Text from the model, from the store, or from
 * another account goes through textContent, so a conversation cannot script the page.
 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const state = {
  me: null,          // identity from /auth/me
  config: {},        // /web/config
  chats: [],
  current: null,     // chat id
  messages: [],
  streaming: false,
  view: "chat",
};

/* --- transport -------------------------------------------------------- */
async function api(path, options = {}) {
  const opts = { credentials: "same-origin", headers: {}, ...options };
  if (opts.body !== undefined && typeof opts.body !== "string") {
    opts.headers["content-type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
    opts.method = opts.method || "POST";
  }
  const response = await fetch(path, opts);
  if (response.status === 204) return null;
  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { detail: text }; }
  if (!response.ok) {
    const error = new Error((payload && payload.detail) || response.statusText);
    error.status = response.status;
    throw error;
  }
  return payload;
}

/* --- boot ------------------------------------------------------------- */
async function boot() {
  try {
    state.config = (await api("/web/config")) || {};
  } catch { state.config = {}; }
  const title = state.config.title || "gustav";
  document.title = title;
  $("#brand-name").textContent = title;
  $("#login-title").textContent = title;

  try {
    state.me = await api("/auth/me");
  } catch (error) {
    if (error.status === 401) return showLogin();
    throw error;
  }
  showApp();
}

function showLogin() {
  $("#login").classList.remove("hidden");
  $("#app").classList.add("hidden");
}

async function showApp() {
  $("#login").classList.add("hidden");
  $("#app").classList.remove("hidden");
  const owner = state.me.role === "owner" && state.config.owner_panels !== false;
  document.querySelectorAll(".owner-only").forEach((node) => {
    node.classList.toggle("hidden", !owner);
  });
  $("#who").textContent = `${state.me.name} · ${state.me.via}`;
  $("#logout").classList.toggle("hidden", state.me.via !== "cookie");
  await loadChats();
  refreshState();
  setInterval(refreshState, 20000);
}

async function refreshState() {
  try {
    const health = await api("/health");
    $("#brand-state").textContent = health.state;
    $("#brand-state").title = health.message || "";
  } catch { $("#brand-state").textContent = "offline"; }
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  $("#login-error").textContent = "";
  try {
    await api("/auth/login", { body: { name: form.get("name"), password: form.get("password") } });
    state.me = await api("/auth/me");
    await showApp();
  } catch (error) {
    $("#login-error").textContent =
      error.status === 404 ? "this daemon has accounts switched off" : error.message;
  }
});

$("#logout").addEventListener("click", async () => {
  await api("/auth/logout", { method: "POST" });
  location.reload();
});

/* --- views ------------------------------------------------------------ */
function show(view) {
  state.view = view;
  document.querySelectorAll(".view").forEach((node) => {
    node.classList.toggle("hidden", node.id !== `view-${view}`);
    if (node.id === "view-chat") node.style.display = view === "chat" ? "flex" : "none";
  });
  document.querySelectorAll("#nav button").forEach((button) => {
    button.setAttribute("aria-current", String(button.dataset.view === view));
  });
  if (view === "facts") loadFacts();
  if (view === "skills") loadSkills();
  if (view === "tools") loadTools();
  if (view === "system") loadSystem();
  if (window.innerWidth <= 700) $("#sidebar").classList.remove("open");
}

document.querySelectorAll("#nav button").forEach((button) => {
  button.addEventListener("click", () => show(button.dataset.view));
});

/* --- chats ------------------------------------------------------------ */
async function loadChats(select) {
  const data = await api("/chats");
  state.chats = data.chats;
  const list = $("#chat-list");
  list.replaceChildren();
  if (!state.chats.length) {
    list.append(el("p", "empty", "No chats yet."));
  }
  for (const chat of state.chats) {
    const row = el("button", "chat-item");
    row.append(el("span", "title", chat.title));
    const drop = el("span", "drop", "×");
    drop.title = "Archive this chat";
    drop.addEventListener("click", async (event) => {
      event.stopPropagation();
      await api(`/chats/${chat.id}`, { method: "DELETE" });
      if (state.current === chat.id) { state.current = null; $("#thread").replaceChildren(); }
      loadChats();
    });
    row.append(drop);
    row.addEventListener("click", () => openChat(chat.id));
    row.setAttribute("aria-current", String(state.current === chat.id));
    list.append(row);
  }
  const wanted = select || state.current || (state.chats[0] && state.chats[0].id);
  if (wanted && wanted !== state.current) openChat(wanted);
  else if (!wanted) $("#thread").replaceChildren(emptyThread());
}

function emptyThread() {
  return el("p", "empty", "Start a chat. What you say here is what it studies tonight.");
}

async function openChat(id) {
  state.current = id;
  show("chat");
  document.querySelectorAll(".chat-item").forEach((row, index) => {
    row.setAttribute("aria-current", String(state.chats[index] && state.chats[index].id === id));
  });
  const data = await api(`/chats/${id}/messages`);
  state.messages = data.messages;
  renderThread();
  $("#input").focus();
}

function renderThread() {
  const thread = $("#thread");
  thread.replaceChildren();
  if (!state.messages.length) thread.append(emptyThread());
  // The feedback commands act on the most recent exchange — the same as typing /good
  // at a prompt — so only the last answer gets the buttons. Older ones show what you
  // already said about them.
  let last = -1;
  state.messages.forEach((message, index) => {
    if (message.role === "assistant") last = index;
  });
  state.messages.forEach((message, index) => {
    thread.append(messageNode(message, index === last));
  });
  scrollDown();
}

function messageNode(message, live = false) {
  const node = el("div", `msg ${message.role}`);
  node.append(el("div", "who", message.role === "user" ? "you" : "gustav"));
  node.append(bodyNode(message.content || ""));

  const meta = el("div", "meta");
  if (message.redacted) meta.append(pill("redacted", "warn"));
  if (message.trainable === 0) meta.append(pill("not trained on", ""));
  if (message.tokens_per_second) {
    meta.append(el("span", "", `${Number(message.tokens_per_second).toFixed(1)} tok/s`));
  }
  for (const item of message.feedback || []) meta.append(pill(item.kind, item.kind === "good" ? "good" : "bad"));

  if (message.role === "assistant" && live) {
    meta.append(el("span", "spacer"));
    for (const node of feedbackButtons()) meta.append(node);
  }
  if (meta.childNodes.length) node.append(meta);
  return node;
}

function pill(text, kind) {
  return el("span", `pill ${kind || ""}`, text);
}

function feedbackButtons() {
  return [
    feedbackButton("good", "👍", "Train on this answer tonight"),
    feedbackButton("bad", "👎", "Mark this answer wrong"),
    feedbackButton("correct", "correct…", "Give the answer it should have given"),
  ];
}

function feedbackButton(kind, label, title) {
  const button = el("button", "btn small", label);
  button.title = title;
  button.addEventListener("click", async () => {
    let text = "";
    if (kind === "correct") {
      text = prompt("What should it have said?") || "";
      if (!text.trim()) return;
    }
    button.disabled = true;
    try {
      const result = await api(`/chats/${state.current}/feedback`, { body: { kind, text } });
      button.replaceWith(pill(result.action || kind, kind === "bad" ? "bad" : "good"));
    } catch (error) {
      button.disabled = false;
      alert(error.message);
    }
  });
  return button;
}

/* Plain text, with fenced code blocks kept as code. Never innerHTML. */
function bodyNode(text) {
  const body = el("div", "body");
  const parts = String(text).split(/```/);
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const pre = el("pre");
      pre.append(el("code", "", part.replace(/^[\w+-]*\n/, "")));
      body.append(pre);
    } else if (part) {
      body.append(document.createTextNode(part));
    }
  });
  return body;
}

function scrollDown() {
  const scroll = $("#thread-scroll");
  scroll.scrollTop = scroll.scrollHeight;
}

/* --- sending ---------------------------------------------------------- */
async function send() {
  const input = $("#input");
  const message = input.value.trim();
  if (!message || state.streaming) return;
  if (!state.current) {
    const chat = await api("/chats", { body: { title: "" } });
    state.current = chat.id;
  }
  input.value = "";
  input.style.height = "auto";
  state.streaming = true;
  $("#send").disabled = true;

  const thread = $("#thread");
  if (thread.querySelector(".empty")) thread.replaceChildren();
  thread.append(messageNode({ role: "user", content: message }));

  const answer = el("div", "msg assistant");
  answer.append(el("div", "who", "gustav"));
  const body = el("div", "body cursor");
  answer.append(body);
  thread.append(answer);
  scrollDown();

  try {
    const response = await fetch(`/chats/${state.current}/messages`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, stream: true }),
    });
    if (!response.ok) throw new Error((await response.text()) || response.statusText);
    await readStream(response, body, answer);
  } catch (error) {
    body.classList.remove("cursor");
    body.append(el("span", "error", ` ${error.message}`));
  } finally {
    state.streaming = false;
    $("#send").disabled = false;
    loadChats(state.current);
    $("#input").focus();
  }
}

/* The server speaks SSE over a POST, so EventSource is no use: read the body. */
async function readStream(response, body, answer) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") continue;
      let event;
      try { event = JSON.parse(data); } catch { continue; }
      if (event.type === "chunk") {
        text += event.text;
        body.replaceChildren(bodyNode(text));
        body.classList.add("cursor");
        scrollDown();
      } else if (event.type === "command") {
        text = event.reply || "";
        body.replaceChildren(bodyNode(text));
      } else if (event.type === "start" && (event.redacted || []).length) {
        answer.append(metaLine(`redacted before storing: ${event.redacted.join(", ")}`));
      } else if (event.type === "done") {
        body.classList.remove("cursor");
        finishAnswer(answer, event);
      }
    }
  }
  body.classList.remove("cursor");
}

function metaLine(text) {
  const meta = el("div", "meta");
  meta.append(pill(text, "warn"));
  return meta;
}

function finishAnswer(answer, event) {
  const meta = el("div", "meta");
  if (event.tokens_per_second) {
    meta.append(el("span", "", `${Number(event.tokens_per_second).toFixed(1)} tok/s`));
  }
  if (event.latency_ms) meta.append(el("span", "", `${event.latency_ms} ms`));
  const adapters = Object.values(event.adapters || {}).filter(Boolean);
  if (adapters.length) meta.append(pill(`adapter ${adapters.join(", ")}`, ""));
  meta.append(el("span", "spacer"));
  for (const node of feedbackButtons()) meta.append(node);
  answer.append(meta);
  scrollDown();
}

$("#menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
$("#send").addEventListener("click", send);
$("#new-chat").addEventListener("click", async () => {
  const chat = await api("/chats", { body: { title: "" } });
  await loadChats(chat.id);
  $("#input").focus();
});
$("#input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(); }
});
$("#input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
});

/* --- owner panels ----------------------------------------------------- */
function table(columns, rows) {
  const node = el("table");
  const head = el("tr");
  for (const column of columns) head.append(el("th", "", column));
  const thead = el("thead");
  thead.append(head);
  node.append(thead);
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr");
    for (const cell of row) {
      const td = el("td");
      if (cell instanceof Node) td.append(cell); else td.textContent = cell === null || cell === undefined ? "" : String(cell);
      tr.append(td);
    }
    body.append(tr);
  }
  node.append(body);
  return node;
}

async function panel(id, build) {
  const node = $(id);
  node.replaceChildren(el("p", "empty", "Loading…"));
  try {
    const children = await build();
    node.replaceChildren(...children);
  } catch (error) {
    node.replaceChildren(el("p", "error", error.message));
  }
}

function loadFacts() {
  return panel("#facts-panel", async () => {
    const data = await api("/facts?limit=200");
    const head = el("h2", "", `What it believes about you (${data.total}, ${data.retracted} retracted)`);
    const rows = data.facts.map((fact) => [
      fact.subject || "",
      fact.text,
      Number(fact.confidence).toFixed(2),
      fact.module || "",
      fact.source || "",
      fact.updated_at || fact.created_at || "",
    ]);
    const body = rows.length
      ? table(["subject", "it believes", "conf", "module", "from", "when"], rows)
      : el("p", "empty", "Nothing yet — facts are extracted during the night.");
    return [head, body];
  });
}

function loadSkills() {
  return panel("#skills-panel", async () => {
    const data = await api("/skills");
    const summary = data.summary || {};
    const head = el("h2", "", "Measured skills");
    const sub = el("p", "sub muted",
      `${summary.topics || data.skills.length} topics · mean score ${
        summary.mean_score !== undefined ? Number(summary.mean_score).toFixed(2) : "–"}`);
    const rows = data.skills.map((skill) => [
      skill.module, skill.topic, `${skill.passes}/${skill.trials}`,
      skill.score.toFixed(2), skill.demand.toFixed(2), skill.weakness.toFixed(2), skill.studied_at || "",
    ]);
    const body = rows.length
      ? table(["module", "topic", "passed", "score", "demand", "weakness", "studied"], rows)
      : el("p", "empty", "No practice runs yet.");
    return [head, sub, body];
  });
}

function loadTools() {
  return panel("#tools-panel", async () => {
    const data = await api("/tools");
    const nodes = [el("h2", "", "Tools it wrote for itself")];

    const ask = el("div", "card");
    ask.append(el("h3", "", "Ask for a tool"));
    ask.append(el("p", "sub", "It is written tonight, sandboxed, and waits here for you to read it before it may run."));
    const row = el("div", "row");
    const field = el("input");
    field.placeholder = "e.g. convert a timestamp to my timezone";
    field.style.cssText = "flex:1;padding:.4rem .5rem;border:1px solid var(--line);border-radius:var(--radius);background:var(--bg)";
    const button = el("button", "btn primary", "Request");
    button.addEventListener("click", async () => {
      if (!field.value.trim()) return;
      button.disabled = true;
      try { await api("/tools/requests", { body: { text: field.value } }); field.value = ""; loadTools(); }
      catch (error) { alert(error.message); } finally { button.disabled = false; }
    });
    row.append(field, button);
    ask.append(row);
    nodes.push(ask);

    if (data.requests && data.requests.length) {
      nodes.push(el("h2", "", "Requested, not written yet"));
      nodes.push(table(["what", "status", "asked"],
        data.requests.map((request) => [request.text, request.status, request.created_at])));
    }

    nodes.push(el("h2", "", "Installed"));
    if (!data.tools.length) nodes.push(el("p", "empty", "No tools yet."));
    for (const tool of data.tools) nodes.push(toolCard(tool));
    return nodes;
  });
}

function toolCard(tool) {
  const card = el("div", "card");
  const head = el("div", "row");
  head.append(el("h3", "", tool.name));
  head.append(pill(tool.status, tool.status === "approved" ? "good"
    : tool.status === "rejected" || tool.status === "retired" ? "bad" : "warn"));
  head.append(el("span", "spacer"));
  head.append(el("span", "faint", `v${tool.active_version || "–"} · used ${
    (tool.usage && tool.usage.calls) || 0}×`));
  card.append(head);
  card.append(el("p", "sub", tool.description || ""));
  if (tool.capabilities && tool.capabilities.length) {
    const caps = el("div", "row");
    caps.append(el("span", "faint", "may:"));
    for (const capability of tool.capabilities) caps.append(pill(capability, "warn"));
    card.append(caps);
  }
  const actions = el("div", "row");
  const read = el("button", "btn small", "Read the source");
  read.addEventListener("click", async () => {
    if (card.querySelector(".source")) { card.querySelector(".source").remove(); return; }
    const detail = await api(`/tools/${encodeURIComponent(tool.name)}`);
    const pre = el("pre", "source");
    pre.append(el("code", "", detail.source || ""));
    card.insertBefore(pre, actions);
    if (detail.checks && detail.checks.length) {
      card.insertBefore(el("p", "sub faint", `checks: ${detail.checks.map(
        (check) => `${check.name} ${check.ok ? "ok" : "failed"}`).join(" · ")}`), actions);
    }
  });
  actions.append(read);
  if (tool.status !== "approved") actions.append(reviewButton(tool, true, "Approve"));
  if (tool.status !== "rejected" && tool.status !== "retired") {
    actions.append(reviewButton(tool, false, "Reject"));
  }
  card.append(actions);
  return card;
}

function reviewButton(tool, approve, label) {
  const button = el("button", `btn small${approve ? " primary" : ""}`, label);
  button.addEventListener("click", async () => {
    const note = approve ? "" : (prompt("Why not? (optional)") || "");
    button.disabled = true;
    try {
      await api(`/tools/${encodeURIComponent(tool.name)}/review`, {
        body: { approve, note, by: state.me.name },
      });
      loadTools();
    } catch (error) { button.disabled = false; alert(error.message); }
  });
  return button;
}

function loadSystem() {
  return panel("#system-panel", async () => {
    const [status, approvals, expansion, optimisation] = await Promise.all([
      api("/status"), api("/approvals"), api("/expansion"), api("/optimisation"),
    ]);
    const nodes = [el("h2", "", "State")];

    const card = el("div", "card");
    card.append(table(["", ""], [
      ["state", status.state],
      ["model", (status.model && (status.model.base_model || status.model.responder)) || ""],
      ["adapters", Object.entries((status.model && status.model.adapters) || {})
        .map(([module, version]) => `${module} ${version}`).join(", ") || "none"],
      ["uptime", `${Math.round(status.uptime_seconds / 60)} min`],
      ["learning", status.learning_paused ? "paused" : "on"],
      ["version", status.version],
    ]));
    const controls = el("div", "row");
    controls.append(control("Sleep now", "/control/sleep"));
    controls.append(control("Wake", "/control/wake"));
    controls.append(control(status.learning_paused ? "Resume learning" : "Pause learning",
      "/control/learning", { paused: !status.learning_paused }));
    controls.append(control("Release the GPU", "/control/release-gpu"));
    card.append(controls);
    nodes.push(card);

    nodes.push(el("h2", "", `Waiting for you (${approvals.approvals.length})`));
    if (!approvals.approvals.length) nodes.push(el("p", "empty", "Nothing is waiting."));
    for (const approval of approvals.approvals) nodes.push(approvalCard(approval));

    nodes.push(el("h2", "", "Self-expansion"));
    if (expansion.triggers && expansion.triggers.length) {
      nodes.push(el("p", "sub muted", "What is pushing it to grow right now:"));
      nodes.push(table(["kind", "subject", "strength", "why"],
        expansion.triggers.map((trigger) => [
          trigger.kind, trigger.subject, trigger.strength, trigger.rationale])));
    }
    const history = expansion.history || [];
    nodes.push(history.length
      ? table(["what", "to", "why", "state", "when"], history.slice(0, 10).map((change) => [
          change.kind, change.subject, change.rationale || change.detail || "",
          pill(change.status, change.status === "applied" ? "good"
            : ["rejected", "reverted", "failed"].includes(change.status) ? "bad" : "warn"),
          change.created_at]))
      : el("p", "empty", "It has not changed its own shape yet."));

    nodes.push(el("h2", "", "Self-optimisation"));
    const tried = optimisation.history || [];
    nodes.push(tried.length
      ? table(["setting", "was", "tried", "measured", "state"], tried.slice(0, 10).map((step) => [
          step.setting, step.previous ?? "", step.candidate ?? "",
          step.delta === null || step.delta === undefined
            ? "" : `${Number(step.delta) > 0 ? "+" : ""}${Number(step.delta).toFixed(2)}`,
          pill(step.status, step.status === "kept" ? "good"
            : ["reverted", "refused", "failed"].includes(step.status) ? "bad" : "warn")]))
      : el("p", "empty", "Nothing has been tuned yet."));
    if (optimisation.next_trials && optimisation.next_trials.length) {
      nodes.push(el("p", "sub faint", `next: ${optimisation.next_trials
        .map((trial) => `${trial.setting}=${trial.candidate}`).join(" · ")}`));
    }
    return nodes;
  });
}

function control(label, path, body) {
  const button = el("button", "btn small", label);
  button.addEventListener("click", async () => {
    button.disabled = true;
    try { await api(path, { method: "POST", body: body || {} }); loadSystem(); refreshState(); }
    catch (error) { button.disabled = false; alert(error.message); }
  });
  return button;
}

function approvalCard(approval) {
  const card = el("div", "card");
  const head = el("div", "row");
  head.append(el("h3", "", `#${approval.id} ${approval.kind}`));
  head.append(pill(approval.status, "warn"));
  card.append(head);
  card.append(el("p", "sub", approval.subject || ""));
  if (approval.rationale) card.append(el("p", "sub faint", approval.rationale));
  const actions = el("div", "row");
  for (const [label, approve] of [["Approve", true], ["Reject", false]]) {
    const button = el("button", `btn small${approve ? " primary" : ""}`, label);
    button.addEventListener("click", async () => {
      const note = approve ? "" : (prompt("Why not? (optional)") || "");
      button.disabled = true;
      try { await api(`/approvals/${approval.id}`, { body: { approve, note } }); loadSystem(); }
      catch (error) { button.disabled = false; alert(error.message); }
    });
    actions.append(button);
  }
  card.append(actions);
  return card;
}

boot().catch((error) => {
  document.body.append(el("p", "error", `could not reach the daemon: ${error.message}`));
});
