/* The web interface.
 *
 * No framework and no build step: the daemon serves three files out of the package and
 * the browser talks to the same HTTP API the `ai` command uses. Everything here is a
 * view over endpoints that already existed — which is why the owner's panels and an
 * ordinary account's chat list behave identically whether you reach them from here or
 * from the terminal.
 *
 * Two rules hold the whole file together:
 *
 *  * **Nothing is ever inserted as HTML.** Text from the model, from the store, or from
 *    another account goes through textContent, and every element is built by hand. A
 *    conversation cannot script this page, and the markdown renderer below is a DOM
 *    builder for exactly that reason — not a string templater.
 *  * **Nothing is fetched from anywhere else.** No CDN, no webfont, no icon pack: the
 *    icons are path data in this file. A local assistant has to work with the network
 *    unplugged, and a page that phones out is not local.
 */
"use strict";

/* --- small things ----------------------------------------------------- */
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};

const SVG = "http://www.w3.org/2000/svg";

/* Line icons, as path data. Stroked, 24-unit box, the same weight throughout. */
const ICONS = {
  menu: "M3 6h18|M3 12h18|M3 18h18",
  collapse: "M4 4h16v16H4z|M9.5 4v16",
  plus: "M12 5v14|M5 12h14",
  search: "M11 17a6 6 0 100-12 6 6 0 000 12z|M20 20l-4.6-4.6",
  trash: "M4 7h16|M9.5 7V4.5h5V7|M6 7l1 13h10l1-13",
  chevron: "M6 14l6-6 6 6",
  sun: "M12 7.5a4.5 4.5 0 100 9 4.5 4.5 0 000-9z|M12 2v2|M12 20v2|M2 12h2|M20 12h2|M4.9 4.9l1.4 1.4|M17.7 17.7l1.4 1.4|M19.1 4.9l-1.4 1.4|M6.3 17.7l-1.4 1.4",
  moon: "M20.5 14.5A8.5 8.5 0 019.5 3.5a7.5 7.5 0 1011 11z",
  auto: "M3 5h18v11H3z|M8.5 20h7|M12 16v4",
  send: "M12 19.5V5|M6 11l6-6 6 6",
  stop: "M7.5 7.5h9v9h-9z",
  copy: "M9.5 9.5h10v10h-10z|M14.5 9.5V4.5h-10v10h5",
  check: "M5 13l4.5 4.5L19 7",
  up: "M7.5 11.5v8.5H4.2v-8.5z|M7.5 11.5l3.8-6.8a2 2 0 013 2.1l-.8 3.9h5a1.9 1.9 0 011.9 2.3l-1.4 5.6a2 2 0 01-1.9 1.4H7.5",
  down: "M7.5 12.5V4H4.2v8.5z|M7.5 12.5l3.8 6.8a2 2 0 003-2.1l-.8-3.9h5a1.9 1.9 0 001.9-2.3l-1.4-5.6A2 2 0 0017.1 2H7.5",
  pencil: "M4 20h4.5l11-11-4.5-4.5-11 11z|M14.5 5.5l4 4",
  logout: "M15 4h4.5v16H15|M10.5 12H3|M6.5 8l-4 4 4 4",
  facts: "M12 3.2c4.2 0 7.8 1.2 7.8 2.9S16.2 9 12 9 4.2 7.8 4.2 6.1 7.8 3.2 12 3.2z|M4.2 6.1v11.8c0 1.7 3.6 2.9 7.8 2.9s7.8-1.2 7.8-2.9V6.1",
  skills: "M4 20V9.5|M10 20V4|M16 20v-7|M21.5 20H2.5",
  tools: "M14.8 3.2a5 5 0 00-4 8.1L4 18.1 5.9 20l6.8-6.8a5 5 0 002.1-10z",
  system: "M4 7h9|M17.5 7h2.5|M4 17h3.5|M12 17h8|M13 4v6|M7.5 14v6",
  close: "M6.5 6.5l11 11|M17.5 6.5l-11 11",
  alert: "M12 4l9 16H3z|M12 10v4.5|M12 17.4h.01",
};

function icon(name, size = 16) {
  // The size is set on the element itself, not left to CSS: an <svg> with no width is
  // as big as whatever contains it, and one missed rule turns a 16-pixel icon into a
  // full-page warning triangle.
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.7");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.style.flex = "none";
  for (const piece of (ICONS[name] || "").split("|")) {
    const path = document.createElementNS(SVG, "path");
    path.setAttribute("d", piece);
    svg.append(path);
  }
  return svg;
}

/* localStorage is per-viewer and may be refused outright (a private window, blocked
   site data). Nothing here is worth an exception, so nothing here throws one. */
const remember = {
  get(key, fallback) {
    try {
      const value = localStorage.getItem(`gustav.${key}`);
      return value === null ? fallback : value;
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`gustav.${key}`, value); } catch { /* fine */ }
  },
};

/* SQLite writes UTC without a zone; Date would read it as local and be an hour out. */
function when(text) {
  if (!text) return null;
  const parsed = new Date(`${String(text).replace(" ", "T").replace(/Z?$/, "")}Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function dayGroup(date) {
  if (!date) return "Earlier";
  const midnight = new Date();
  midnight.setHours(0, 0, 0, 0);
  const days = Math.floor((midnight - date) / 86400000);
  if (days < 0) return "Today";
  if (days === 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "This week";
  if (days < 30) return "This month";
  return "Earlier";
}

function initials(name) {
  const words = String(name || "?").trim().split(/\s+/);
  return ((words[0] || "?")[0] + (words.length > 1 ? words[words.length - 1][0] : "")).toUpperCase();
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // clipboard needs a secure context, and a daemon on a LAN address is not one.
    try {
      const area = el("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      const done = document.execCommand("copy");
      area.remove();
      return done;
    } catch { return false; }
  }
}

/* --- state ------------------------------------------------------------ */
const state = {
  me: null,          // identity from /auth/me
  config: {},        // /web/config
  status: null,      // /status, owners only
  chats: [],
  current: null,     // chat id
  messages: [],
  view: "chat",
  filter: "",
  streaming: null,   // the AbortController of the answer being streamed
  owner: false,
  systemAt: "",      // what the System panel was last built from
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

/* --- toasts and sheets ------------------------------------------------ */
function toast(message, kind) {
  const node = el("div", `toast ${kind || ""}`, message);
  $("#toasts").append(node);
  setTimeout(() => node.remove(), kind === "bad" ? 7000 : 3500);
}

/* A sheet instead of prompt()/alert(): those block the page, look like 1998, and on a
   phone they cover the thing you are being asked about. Resolves to null on a cancel. */
function sheet({ title, note, label, value = "", confirm = "Save", multiline = true }) {
  return new Promise((resolve) => {
    const host = $("#modal");
    const box = el("div", "sheet");
    box.append(el("h3", "", title));
    if (note) box.append(el("p", "fine", note));
    const field = el(multiline ? "textarea" : "input");
    field.value = value;
    if (label) field.setAttribute("aria-label", label);
    box.append(field);
    const row = el("div", "row");
    const cancel = el("button", "btn", "Cancel");
    const ok = el("button", "btn primary", confirm);
    row.append(cancel, ok);
    box.append(row);
    host.replaceChildren(box);
    host.hidden = false;
    field.focus();

    const finish = (answer) => {
      host.hidden = true;
      host.replaceChildren();
      document.removeEventListener("keydown", onKey);
      resolve(answer);
    };
    const onKey = (event) => {
      if (event.key === "Escape") finish(null);
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey || !multiline)) {
        event.preventDefault();
        finish(field.value);
      }
    };
    cancel.addEventListener("click", () => finish(null));
    ok.addEventListener("click", () => finish(field.value));
    host.addEventListener("click", (event) => { if (event.target === host) finish(null); });
    document.addEventListener("keydown", onKey);
  });
}

async function confirmSheet(title, note, confirm = "Yes") {
  const answer = await sheet({ title, note, value: "", confirm, multiline: false, label: title });
  return answer !== null;
}

/* --- theme ------------------------------------------------------------ */
const THEMES = ["auto", "dark", "light"];

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  remember.set("theme", theme);
  const button = $("#theme");
  button.replaceChildren(icon(theme === "auto" ? "auto" : theme === "dark" ? "moon" : "sun"));
  button.title = `Theme: ${theme} — click for ${THEMES[(THEMES.indexOf(theme) + 1) % 3]}`;
}

$("#theme").addEventListener("click", () => {
  const next = THEMES[(THEMES.indexOf(document.documentElement.dataset.theme) + 1) % THEMES.length];
  applyTheme(next);
});

/* --- boot ------------------------------------------------------------- */
async function boot() {
  applyTheme(remember.get("theme", "auto"));
  for (const [id, name] of [["#collapse", "collapse"], ["#new-chat", "plus"], ["#menu", "menu"]]) {
    $(id).replaceChildren(icon(name));
  }
  $("#search-wrap").firstElementChild.replaceChildren(icon("search", 14));
  $("#send").replaceChildren(icon("send"));
  for (const button of document.querySelectorAll("#workspace button")) {
    const names = { facts: "Facts", skills: "Skills", tools: "Tools", system: "System" };
    button.replaceChildren(icon(button.dataset.view), el("span", "", names[button.dataset.view]));
    button.addEventListener("click", () => show(button.dataset.view));
  }

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
  await showApp();
}

function showLogin() {
  $("#login").hidden = false;
  $("#app").hidden = true;
}

async function showApp() {
  $("#login").hidden = true;
  $("#app").hidden = false;
  state.owner = state.me.role === "owner" && state.config.owner_panels !== false;
  for (const node of document.querySelectorAll(".owner-only")) node.hidden = !state.owner;

  $("#me-name").textContent = state.me.name;
  $("#me-via").textContent = `${state.me.role} · over ${state.me.via}`;
  $("#me-avatar").textContent = initials(state.me.name);
  if (remember.get("sidebar", "open") === "closed") $("#app").classList.add("collapsed");

  // Read before loadChats: opening a chat is a view change, and a view change rewrites
  // the address bar, so by then the fragment someone arrived with is gone.
  const wanted = location.hash.slice(1);
  await loadChats();
  if (state.owner && VIEWS.includes(wanted) && wanted !== "chat") show(wanted);
  await refreshState();
  setInterval(refreshState, 20000);
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

/* --- the state chip, and the banner that interrupts ------------------- */
async function refreshState() {
  let health = null;
  try {
    health = await api("/health");
  } catch {
    setChip("offline", "the daemon is not answering");
    return banner({
      text: "The daemon is not answering. It may be restarting; this page retries every "
            + "20 seconds.",
    });
  }
  setChip(health.state, health.message || "");

  // Only an owner can be told why: /status is owner-only, and the reason can name a
  // path on the server. Everyone else gets the state, which is all they can act on.
  if (!state.owner) {
    return banner(health.ok ? null : { text: health.message || "the model is not loaded" });
  }
  try {
    state.status = await api("/status");
  } catch { state.status = null; }
  const status = state.status || {};
  if (status.load_error) {
    banner({
      text: `The model is not loaded, so questions cannot be answered: ${String(status.load_error).split("\n")[0]}`,
      help: status.load_help || "",
      hint: status.load_help ? "ai doctor" : "",
    });
  } else if (!health.ok) {
    banner({ text: health.message || "the model is not loaded" });
  } else {
    banner(null);
  }
  // The System panel is rebuilt on a poll only when something it shows has actually
  // moved. Rebuilding it every twenty seconds throws away the scroll position of
  // whoever is reading a table on it.
  const fingerprint = `${health.state}|${status.load_error || ""}|${status.learning_paused}`;
  if (state.view === "system" && fingerprint !== state.systemAt) loadSystem();
  state.systemAt = fingerprint;
}

function setChip(name, title) {
  const chip = $("#state-chip");
  const text = String(name || "…").toLowerCase();
  chip.dataset.state = text;
  $("#state-text").textContent = text.replace("_", " ");
  chip.title = title || "";
}

/* `hint` is a command to run on the server, shown as a command and not as prose: the
   whole point of the banner is that you can read it and then type it. */
function banner(what) {
  const node = $("#banner");
  // It appears above the thread and takes two lines with it, so a conversation that
  // was sitting at the bottom would quietly end up a paragraph short of it.
  const pinned = nearBottom();
  const settle = () => { if (pinned) scrollDown(true); };
  if (!what) {
    node.hidden = true;
    node.replaceChildren();
    node.dataset.text = "";
    settle();
    return;
  }
  if (node.dataset.text === what.text) return;   // do not flicker on every poll
  node.dataset.text = what.text;
  node.replaceChildren();
  const mark = icon("alert");
  mark.style.color = "var(--warn)";
  mark.style.flex = "none";
  mark.style.marginTop = "2px";
  const body = el("div", "body");
  body.append(el("div", "", what.text));
  if (what.help) body.append(el("div", "fine", what.help));
  if (what.hint) {
    const line = el("div", "fine");
    line.append(document.createTextNode("on the server: "));
    line.append(el("code", "", what.hint));
    body.append(line);
  }
  const close = el("button", "close", "");
  close.append(icon("close", 16));
  close.title = "Dismiss";
  close.addEventListener("click", () => { node.hidden = true; });
  node.append(mark, body, close);
  node.hidden = false;
  settle();
}

/* --- views ------------------------------------------------------------ */
const VIEWS = ["chat", "facts", "skills", "tools", "system"];

function show(view) {
  state.view = view;
  if (location.hash.slice(1) !== view) {
    // replaceState, not a push: the back button belongs to the page you came from,
    // not to four taps around this one.
    history.replaceState(null, "", view === "chat" ? location.pathname : `#${view}`);
  }
  for (const node of document.querySelectorAll(".view")) {
    node.hidden = node.id !== `view-${view}`;
  }
  for (const button of document.querySelectorAll("#workspace button")) {
    button.setAttribute("aria-current", String(button.dataset.view === view));
  }
  const names = { chat: "", facts: "Facts", skills: "Skills", tools: "Tools", system: "System" };
  if (view !== "chat") {
    $("#chat-title").textContent = names[view];
    $("#chat-sub").textContent = "";
  } else {
    showChatTitle();
  }
  if (view === "facts") loadFacts();
  if (view === "skills") loadSkills();
  if (view === "tools") loadTools();
  if (view === "system") loadSystem();
  closeSidebarOnNarrow();
}

function closeSidebarOnNarrow() {
  if (window.innerWidth <= 820) $("#app").classList.remove("open");
  $("#scrim").hidden = !$("#app").classList.contains("open") || window.innerWidth > 820;
}

$("#menu").addEventListener("click", () => {
  const app = $("#app");
  if (window.innerWidth <= 820) {
    app.classList.toggle("open");
    $("#scrim").hidden = !app.classList.contains("open");
  } else {
    app.classList.remove("collapsed");
    remember.set("sidebar", "open");
  }
});
$("#collapse").addEventListener("click", () => {
  $("#app").classList.add("collapsed");
  remember.set("sidebar", "closed");
});
$("#scrim").addEventListener("click", closeSidebarOnNarrow);

/* --- the account menu ------------------------------------------------- */
$("#me").addEventListener("click", () => {
  const menu = $("#me-menu");
  if (!menu.hidden) {
    menu.hidden = true;
    return;
  }
  menu.replaceChildren();
  const theme = el("button", "", "");
  theme.append(icon("auto", 15), el("span", "", "Theme"));
  theme.addEventListener("click", () => { $("#theme").click(); menu.hidden = true; });
  menu.append(theme);
  if (state.me.via === "cookie") {
    const out = el("button", "", "");
    out.append(icon("logout", 15), el("span", "", "Sign out"));
    out.addEventListener("click", async () => {
      await api("/auth/logout", { method: "POST" });
      location.reload();
    });
    menu.append(el("div", "sep"), out);
  }
  menu.append(
    el("div", "sep"),
    el("span", "fine", `${state.config.instance || "gustav"} · ${state.me.version || ""}`),
  );
  menu.hidden = false;
  $("#me").setAttribute("aria-expanded", "true");
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".side-foot")) {
    $("#me-menu").hidden = true;
    $("#me").setAttribute("aria-expanded", "false");
  }
});

/* --- the chat list ---------------------------------------------------- */
async function loadChats(select) {
  const data = await api("/chats");
  state.chats = data.chats || [];
  renderChats();
  const wanted = select || state.current || (state.chats[0] && state.chats[0].id);
  if (wanted && wanted !== state.current) await openChat(wanted);
  else if (!wanted) { state.messages = []; renderThread(); }
  else renderChats();
}

function renderChats() {
  const list = $("#chat-list");
  list.replaceChildren();
  const needle = state.filter.trim().toLowerCase();
  const shown = needle
    ? state.chats.filter((chat) => (chat.title || "").toLowerCase().includes(needle))
    : state.chats;

  if (!shown.length) {
    list.append(el("p", "empty", needle ? "Nothing matches." : "No chats yet."));
    return;
  }
  let group = null;
  for (const chat of shown) {
    const label = dayGroup(when(chat.last_activity_at));
    if (label !== group) {
      group = label;
      list.append(el("p", "side-label", label));
    }
    list.append(chatRow(chat));
  }
}

function chatRow(chat) {
  const row = el("div", "chat-item");
  row.setAttribute("role", "listitem");
  const open = el("button", "title", chat.title || "new chat");
  open.style.cssText = "background:none;border:0;padding:0;text-align:left;cursor:pointer;color:inherit;font:inherit;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
  open.addEventListener("click", () => openChat(chat.id));
  row.append(open);

  const rename = el("button", "drop", "");
  rename.append(icon("pencil", 14));
  rename.title = "Rename";
  rename.addEventListener("click", async (event) => {
    event.stopPropagation();
    const title = await sheet({
      title: "Rename this chat",
      value: chat.title || "",
      confirm: "Rename",
      multiline: false,
      label: "Title",
    });
    if (title === null || !title.trim()) return;
    try {
      await api(`/chats/${chat.id}/rename`, { body: { title } });
      await loadChats(state.current);
      showChatTitle();
    } catch (error) { toast(error.message, "bad"); }
  });

  const drop = el("button", "drop", "");
  drop.append(icon("trash", 14));
  drop.title = "Archive this chat";
  drop.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!await confirmSheet(
      "Archive this chat?",
      "It leaves the list. Nothing it taught the model is undone — what was learned stays learned.",
      "Archive",
    )) return;
    await api(`/chats/${chat.id}`, { method: "DELETE" });
    if (state.current === chat.id) {
      state.current = null;
      state.messages = [];
    }
    await loadChats();
    renderThread();
  });

  row.append(rename, drop);
  row.setAttribute("aria-current", String(state.current === chat.id));
  return row;
}

$("#search").addEventListener("input", (event) => {
  state.filter = event.target.value;
  renderChats();
});

async function openChat(id) {
  state.current = id;
  show("chat");
  renderChats();
  const data = await api(`/chats/${id}/messages`);
  state.messages = data.messages || [];
  renderThread();
  showChatTitle();
  $("#input").focus();
}

function showChatTitle() {
  const chat = state.chats.find((item) => item.id === state.current);
  $("#chat-title").textContent = (chat && chat.title) || "New chat";
  const date = chat && when(chat.last_activity_at);
  $("#chat-sub").textContent = chat
    ? `${chat.messages || state.messages.length} messages${date ? ` · ${date.toLocaleString()}` : ""}`
    : "";
}

$("#new-chat").addEventListener("click", async () => {
  const chat = await api("/chats", { body: { title: "" } });
  await loadChats(chat.id);
  $("#input").focus();
});

/* --- the thread ------------------------------------------------------- */
function renderThread() {
  const thread = $("#thread");
  thread.replaceChildren();
  if (!state.messages.length) {
    thread.append(greeting());
    return;
  }
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
  scrollDown(true);
}

const SUGGESTIONS = [
  ["What do you know about me?", "the facts it kept, in its own words"],
  ["What did you learn last night?", "the last sleep run, from its side"],
  ["/tools", "what it can call right now, and what is waiting to be read"],
  ["/tool ", "have one written tonight — say what it should do"],
];

function greeting() {
  const hour = new Date().getHours();
  const box = el("div", "greeting");
  box.append(el("div", "mark"));
  box.append(el("h2", "", `Good ${hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening"}.`));
  box.append(el("p", "", "What you say here is what it studies tonight."));
  const grid = el("div", "suggest");
  for (const [text, why] of SUGGESTIONS) {
    const button = el("button");
    button.append(el("b", "", text), el("span", "", why));
    button.addEventListener("click", () => {
      const input = $("#input");
      input.value = text;
      // "/tool " is half a sentence: put the cursor after it instead of sending it.
      if (text.endsWith(" ")) {
        input.focus();
        input.setSelectionRange(text.length, text.length);
        return;
      }
      send();
    });
    grid.append(button);
  }
  box.append(grid);
  return box;
}

function messageNode(message, live = false) {
  const node = el("div", `msg ${message.role}`);
  node.append(el("div", "who", message.role === "user" ? initials(state.me.name) : "G"));
  const column = el("div", "col");
  column.append(bodyNode(message.content || ""));
  column.append(actionRow(message, live));
  node.append(column);
  return node;
}

function actionRow(message, live) {
  const acts = el("div", "acts");
  if (message.redacted) acts.append(pill("redacted", "warn"));
  if (message.trainable === 0) acts.append(pill("not trained on", ""));
  if (message.tokens_per_second) {
    acts.append(el("span", "meta", `${Number(message.tokens_per_second).toFixed(1)} tok/s`));
  }
  for (const item of message.feedback || []) {
    acts.append(pill(item.kind, item.kind === "good" ? "good" : "bad"));
  }
  acts.append(copyButton(message.content || ""));
  if (message.role === "assistant" && live) {
    for (const node of feedbackButtons()) acts.append(node);
  }
  return acts;
}

function pill(text, kind) {
  return el("span", `pill ${kind || ""}`, text);
}

function actButton(name, label, title, ghost = true) {
  const button = el("button", `act${ghost ? " ghost" : ""}`);
  if (name) button.append(icon(name, 15));
  if (label) button.append(el("span", "", label));
  button.title = title;
  button.setAttribute("aria-label", title);
  return button;
}

function copyButton(text) {
  const button = actButton("copy", "", "Copy");
  button.addEventListener("click", async () => {
    if (await copyText(text)) {
      button.replaceChildren(icon("check", 15));
      button.title = "Copied";
      setTimeout(() => button.replaceChildren(icon("copy", 15)), 1400);
    } else {
      toast("the browser would not let the page use the clipboard", "bad");
    }
  });
  return button;
}

function feedbackButtons() {
  return [
    feedbackButton("good", "up", "", "Train on this answer tonight"),
    feedbackButton("bad", "down", "", "Mark this answer wrong"),
    feedbackButton("correct", "pencil", "correct", "Say what it should have answered"),
  ];
}

function feedbackButton(kind, name, label, title) {
  const button = actButton(name, label, title);
  button.addEventListener("click", async () => {
    let text = "";
    if (kind === "correct") {
      text = await sheet({
        title: "What should it have said?",
        note: "This is kept as the right answer and trained on tonight.",
        confirm: "Teach it",
        label: "The answer it should have given",
      });
      if (text === null || !text.trim()) return;
    }
    button.disabled = true;
    try {
      const result = await api(`/chats/${state.current}/feedback`, { body: { kind, text } });
      button.replaceWith(pill(result.action || kind, kind === "bad" ? "bad" : "good"));
      toast(kind === "correct" ? "kept as the right answer" : "noted for tonight", "good");
    } catch (error) {
      button.disabled = false;
      toast(error.message, "bad");
    }
  });
  return button;
}

/* --- markdown, as DOM ------------------------------------------------- */
/* Enough of it to read an answer by: fenced code with a copy button, headings, lists,
 * quotes, rules, and inline code/bold/italic/links. Built element by element, because
 * the alternative is innerHTML and the text comes from a language model. */
function bodyNode(text) {
  const body = el("div", "body");
  const blocks = String(text).split(/```/);
  blocks.forEach((block, index) => {
    if (index % 2 === 1) {
      body.append(codeBlock(block));
    } else if (block.trim()) {
      for (const node of markdownBlocks(block)) body.append(node);
    }
  });
  return body;
}

function codeBlock(block) {
  const firstBreak = block.indexOf("\n");
  const head = firstBreak === -1 ? "" : block.slice(0, firstBreak);
  const language = /^[\w+#.-]*$/.test(head.trim()) ? head.trim() : "";
  const source = language || firstBreak === -1 ? block.slice(firstBreak + 1) : block;

  const wrap = el("div", "code");
  const bar = el("div", "code-head");
  bar.append(el("span", "", language || "code"), el("span", "spacer"));
  const copy = copyButton(source.replace(/\n$/, ""));
  copy.classList.remove("ghost");
  bar.append(copy);
  const pre = el("pre");
  pre.append(el("code", "", source.replace(/\n$/, "")));
  wrap.append(bar, pre);
  return wrap;
}

function markdownBlocks(text) {
  const nodes = [];
  const lines = text.replace(/^\n+|\n+$/g, "").split("\n");
  let paragraph = [];
  let blank = false;        // the line before this one was empty
  const stack = [];         // one entry per depth of list we are inside

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const node = el("p");
    inline(node, paragraph.join("\n"));
    nodes.push(node);
    paragraph = [];
  };
  const flushList = () => { stack.length = 0; };

  /* A list inside a list: "1. **Importing**" with "   - the script starts…" under it.
     The stack holds one entry per level and the indentation of the line decides which
     level it belongs to — deeper opens a list inside the last item, shallower closes
     back to the level it matches. */
  const addItem = (indent, kind, content) => {
    while (stack.length > 1 && indent < stack[stack.length - 1].indent) stack.pop();
    let top = stack[stack.length - 1];
    const deeper = top && indent >= top.indent + 2;
    const different = top && !deeper && top.node.tagName.toLowerCase() !== kind;
    if (!top || deeper || different) {
      if (different) stack.pop();
      const list = el(kind);
      const parent = stack[stack.length - 1];
      if (parent && deeper) (parent.node.lastElementChild || parent.node).append(list);
      else if (parent) parent.node.append(list);
      else nodes.push(list);
      stack.push({ indent, node: list });
      top = stack[stack.length - 1];
    }
    const item = el("li");
    inline(item, content);
    top.node.append(item);
  };

  /* The line after a list item, wrapped by whatever produced it, belongs to that item
     and not to a new paragraph. Getting this wrong is what turns one numbered list
     into four lists that all start at 1. */
  const continueItem = (content) => {
    const top = stack[stack.length - 1];
    const item = top && top.node.lastElementChild;
    if (!item) return false;
    item.append(document.createTextNode(" "));
    inline(item, content);
    return true;
  };

  for (const line of lines) {
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const numbered = /^\s*(\d+)[.)]\s+(.*)$/.exec(line);
    const quote = /^>\s?(.*)$/.exec(line);

    if (!line.trim()) {
      // A blank line ends a paragraph. It does not end a list: markdown's own "loose"
      // lists are written with blank lines between the items.
      flushParagraph();
      blank = true;
      continue;
    }
    if (/^(\s*[-*_]){3,}\s*$/.test(line)) {
      flushParagraph(); flushList();
      nodes.push(el("hr"));
      blank = false;
      continue;
    }
    if (heading) {
      flushParagraph(); flushList();
      const level = Math.min(heading[1].length + 2, 5);   // h1 in a message is shouting
      const node = el(`h${level}`);
      inline(node, heading[2]);
      nodes.push(node);
      blank = false;
      continue;
    }
    if (bullet || numbered) {
      flushParagraph();
      const indent = (/^[ \t]*/.exec(line)[0] || "").replace(/\t/g, "  ").length;
      addItem(indent, bullet ? "ul" : "ol", bullet ? bullet[1] : numbered[2]);
      blank = false;
      continue;
    }
    if (quote) {
      flushParagraph(); flushList();
      const previous = nodes[nodes.length - 1];
      if (previous && previous.tagName === "BLOCKQUOTE") {
        previous.append(document.createTextNode(" "));
        inline(previous, quote[1]);
      } else {
        const node = el("blockquote");
        inline(node, quote[1]);
        nodes.push(node);
      }
      blank = false;
      continue;
    }
    const indented = /^\s{1,}/.test(line);
    if (!paragraph.length && stack.length && (indented || !blank) && continueItem(line.trim())) {
      blank = false;
      continue;
    }
    flushList();
    paragraph.push(line);
    blank = false;
  }
  flushParagraph();
  return nodes;
}

/* `code`, **bold**, *italic*, and bare or bracketed links. One pass, left to right. */
const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|(\[[^\]]+\]\((?:https?:\/\/|\/)[^\s)]+\))|(<https?:\/\/[^\s>]+>)|((?:https?:\/\/)[^\s<>()]+)/g;

function inline(parent, text) {
  let at = 0;
  for (const match of String(text).matchAll(INLINE)) {
    if (match.index > at) parent.append(document.createTextNode(text.slice(at, match.index)));
    const [whole, code, bold, italic, linked, angled, bare] = match;
    if (code) {
      parent.append(el("code", "", code.slice(1, -1)));
    } else if (bold) {
      parent.append(el("strong", "", bold.slice(2, -2)));
    } else if (italic) {
      parent.append(el("em", "", italic.slice(1, -1)));
    } else if (linked) {
      const split = linked.indexOf("](");
      parent.append(anchor(linked.slice(split + 2, -1), linked.slice(1, split)));
    } else if (angled) {
      // <https://…> is how a model writes a bare URL. The brackets are punctuation.
      const url = angled.slice(1, -1);
      parent.append(anchor(url, url));
    } else if (bare) {
      parent.append(anchor(bare, bare));
    }
    at = match.index + whole.length;
  }
  if (at < text.length) parent.append(document.createTextNode(text.slice(at)));
}

function anchor(href, label) {
  // Only http(s) and same-origin paths become links: javascript: and data: never do.
  if (!/^(https?:\/\/|\/)/.test(href)) return document.createTextNode(label);
  const node = el("a", "", label);
  node.href = href;
  node.target = "_blank";
  node.rel = "noopener noreferrer";
  return node;
}

/* --- sending ---------------------------------------------------------- */
function nearBottom() {
  const scroll = $("#thread-scroll");
  return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120;
}

function scrollDown(force) {
  const scroll = $("#thread-scroll");
  if (!force && !nearBottom()) return;
  scroll.scrollTop = scroll.scrollHeight;
  // scrollHeight is only final once the browser has laid the thread out, and a long
  // conversation is never laid out by the time the fetch that produced it returns.
  requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
}

$("#composer-form").addEventListener("submit", (event) => {
  event.preventDefault();
  send();
});

async function send(text) {
  const input = $("#input");
  const message = (text !== undefined ? text : input.value).trim();
  if (!message || state.streaming) return;
  if (!state.current) {
    const chat = await api("/chats", { body: { title: "" } });
    state.current = chat.id;
  }
  input.value = "";
  input.style.height = "auto";
  setSending(true);

  const thread = $("#thread");
  if (thread.querySelector(".greeting")) thread.replaceChildren();
  thread.append(messageNode({ role: "user", content: message }));

  const answer = el("div", "msg assistant");
  answer.append(el("div", "who", "G"));
  const column = el("div", "col");
  const body = el("div", "body cursor");
  column.append(body);
  answer.append(column);
  thread.append(answer);
  scrollDown(true);

  const controller = new AbortController();
  state.streaming = controller;
  try {
    const response = await fetch(`/chats/${state.current}/messages`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message, stream: true }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error((await response.text()) || response.statusText);
    await readStream(response, body, column);
  } catch (error) {
    body.classList.remove("cursor");
    if (error.name === "AbortError") {
      column.append(el("div", "acts", "stopped"));
    } else {
      column.append(el("p", "error", error.message));
    }
  } finally {
    state.streaming = null;
    setSending(false);
    await loadChats(state.current);
    showChatTitle();
    input.focus();
  }
}

function setSending(sending) {
  $("#send").disabled = sending;
  $("#stop").hidden = !sending;
  $("#input").disabled = false;
}

$("#stop").addEventListener("click", () => {
  if (state.streaming) state.streaming.abort();
});

/* The server speaks SSE over a POST, so EventSource is no use: read the body. */
async function readStream(response, body, column) {
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
        body.replaceChildren(...bodyNode(text).childNodes);
        body.classList.add("cursor");
        scrollDown();
      } else if (event.type === "command") {
        text = event.reply || "";
        body.replaceChildren(...bodyNode(text).childNodes);
      } else if (event.type === "start" && (event.redacted || []).length) {
        column.append(metaLine(`redacted before storing: ${event.redacted.join(", ")}`));
      } else if (event.type === "error") {
        // The daemon could not answer. It said why; say that, not "stream ended".
        body.classList.remove("cursor");
        if (!text) body.replaceChildren();
        column.append(el("p", "error", event.error));
        refreshState();
      } else if (event.type === "done") {
        body.classList.remove("cursor");
        finishAnswer(column, event, text);
      }
    }
  }
  body.classList.remove("cursor");
}

function metaLine(text) {
  const acts = el("div", "acts");
  acts.append(pill(text, "warn"));
  return acts;
}

function finishAnswer(column, event, text) {
  const acts = el("div", "acts");
  if (event.tokens_per_second) {
    acts.append(el("span", "meta", `${Number(event.tokens_per_second).toFixed(1)} tok/s`));
  }
  if (event.latency_ms) acts.append(el("span", "meta", `${event.latency_ms} ms`));
  const adapters = Object.values(event.adapters || {}).filter(Boolean);
  if (adapters.length) acts.append(pill(`adapter ${adapters.join(", ")}`, ""));
  acts.append(copyButton(text));
  for (const node of feedbackButtons()) acts.append(node);
  column.append(acts);
  scrollDown();
}

$("#input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    send();
  }
});
$("#input").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 224)}px`;
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "o") {
    event.preventDefault();
    $("#new-chat").click();
  } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    $("#app").classList.remove("collapsed");
    $("#search").focus();
  } else if (event.key === "Escape" && $("#modal").hidden) {
    closeSidebarOnNarrow();
  }
});

/* --- the owner's panels ----------------------------------------------- */
function table(columns, rows) {
  const wrap = el("div", "table-wrap");
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
      if (cell instanceof Node) td.append(cell);
      else td.textContent = cell === null || cell === undefined ? "" : String(cell);
      tr.append(td);
    }
    body.append(tr);
  }
  node.append(body);
  wrap.append(node);
  return wrap;
}

function stats(pairs) {
  const grid = el("div", "stats");
  for (const [key, value] of pairs) {
    const box = el("div", "stat");
    box.append(el("span", "k", key));
    if (value instanceof Node) box.append(value);
    else box.append(el("span", "v", value === null || value === undefined ? "–" : String(value)));
    grid.append(box);
  }
  return grid;
}

async function panel(id, build) {
  const node = $(id);
  if (!node.childNodes.length) node.replaceChildren(el("p", "empty", "Loading…"));
  try {
    node.replaceChildren(...(await build()));
  } catch (error) {
    node.replaceChildren(el("p", "error", error.message));
  }
}

function loadFacts() {
  return panel("#facts-panel", async () => {
    const data = await api("/facts?limit=200");
    const rows = data.facts.map((fact) => [
      fact.subject || "",
      fact.text,
      Number(fact.confidence).toFixed(2),
      fact.module || "",
      fact.source || "",
      (fact.updated_at || fact.created_at || "").slice(0, 16),
    ]);
    return [
      el("h2", "", "What it believes about you"),
      el("p", "sub", `${data.total} facts · ${data.retracted} retracted · extracted during the night, never on the model's word alone`),
      rows.length
        ? table(["subject", "it believes", "conf", "module", "from", "when"], rows)
        : el("p", "empty", "Nothing yet — facts are extracted during the night."),
    ];
  });
}

function loadSkills() {
  return panel("#skills-panel", async () => {
    const data = await api("/skills");
    const summary = data.summary || {};
    const rows = data.skills.map((skill) => [
      skill.module, skill.topic, `${skill.passes}/${skill.trials}`,
      skill.score.toFixed(2), skill.demand.toFixed(2), skill.weakness.toFixed(2),
      (skill.studied_at || "").slice(0, 16),
    ]);
    return [
      el("h2", "", "Measured skills"),
      el("p", "sub", "Nothing here is a guess: every score is a pass rate over exercises it was given and marked on."),
      stats([
        ["topics", summary.topics || data.skills.length],
        ["mean score", summary.mean_score !== undefined ? Number(summary.mean_score).toFixed(2) : "–"],
      ]),
      rows.length
        ? table(["module", "topic", "passed", "score", "demand", "weakness", "studied"], rows)
        : el("p", "empty", "No practice runs yet."),
    ];
  });
}

function loadTools() {
  return panel("#tools-panel", async () => {
    const data = await api("/tools");
    const nodes = [el("h2", "", "Tools it wrote for itself")];

    const ask = el("div", "card");
    ask.append(el("h3", "", "Ask for a tool"));
    ask.append(el("p", "sub", "It is written tonight, sandboxed, checked six ways, and waits here for you to read it before it may run."));
    const row = el("div", "row");
    const field = el("input", "field");
    field.placeholder = "e.g. convert a timestamp to my timezone";
    const button = el("button", "btn primary", "Request");
    const request = async () => {
      if (!field.value.trim()) return;
      button.disabled = true;
      try {
        await api("/tools/requests", { body: { text: field.value } });
        field.value = "";
        toast("asked for; it is written tonight", "good");
        loadTools();
      } catch (error) { toast(error.message, "bad"); } finally { button.disabled = false; }
    };
    button.addEventListener("click", request);
    field.addEventListener("keydown", (event) => { if (event.key === "Enter") request(); });
    row.append(field, button);
    ask.append(row);
    nodes.push(ask);

    if (data.requests && data.requests.length) {
      nodes.push(el("h2", "", "Requested, not written yet"));
      nodes.push(table(["what", "status", "asked"],
        data.requests.map((item) => [item.text, pill(item.status, "warn"), (item.created_at || "").slice(0, 16)])));
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
  head.append(el("span", "fine", `v${tool.active_version || "–"} · used ${(tool.usage && tool.usage.calls) || 0}×`));
  card.append(head);
  card.append(el("p", "sub", tool.description || ""));
  if (tool.capabilities && tool.capabilities.length) {
    const caps = el("div", "row");
    caps.append(el("span", "fine", "may:"));
    for (const capability of tool.capabilities) caps.append(pill(capability, "warn"));
    card.append(caps);
  }
  const actions = el("div", "row");
  const read = el("button", "btn small", "Read the source");
  read.addEventListener("click", async () => {
    const already = card.querySelector(".source");
    if (already) {
      already.remove();
      read.textContent = "Read the source";
      return;
    }
    try {
      const detail = await api(`/tools/${encodeURIComponent(tool.name)}`);
      const pre = el("pre", "source");
      pre.append(el("code", "", detail.source || ""));
      card.insertBefore(pre, actions);
      if (detail.checks && detail.checks.length) {
        const line = el("p", "fine");
        for (const check of detail.checks) {
          line.append(pill(`${check.name} ${check.ok ? "ok" : "failed"}`, check.ok ? "good" : "bad"));
          line.append(document.createTextNode(" "));
        }
        card.insertBefore(line, actions);
      }
      read.textContent = "Hide the source";
    } catch (error) { toast(error.message, "bad"); }
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
    let note = "";
    if (!approve) {
      note = await sheet({ title: `Reject ${tool.name}?`, note: "Why not? (optional — it is read when the tool is rewritten)", confirm: "Reject" });
      if (note === null) return;
    }
    button.disabled = true;
    try {
      await api(`/tools/${encodeURIComponent(tool.name)}/review`, {
        body: { approve, note, by: state.me.name },
      });
      loadTools();
    } catch (error) { button.disabled = false; toast(error.message, "bad"); }
  });
  return button;
}

function loadSystem() {
  return panel("#system-panel", async () => {
    const [status, approvals, expansion, optimisation] = await Promise.all([
      api("/status"), api("/approvals"), api("/expansion"), api("/optimisation"),
    ]);
    state.status = status;
    const nodes = [el("h2", "", "State")];

    nodes.push(stats([
      ["state", status.state],
      ["model", (status.model && (status.model.base_model || status.model.responder)) || "–"],
      ["uptime", `${Math.round(status.uptime_seconds / 60)} min`],
      ["learning", status.learning_paused ? "paused" : "on"],
      ["version", status.version],
      ["adapters", Object.entries((status.model && status.model.adapters) || {})
        .map(([module, version]) => `${module} ${version}`).join(", ") || "none"],
    ]));

    if (status.load_error) {
      const card = el("div", "card");
      card.append(el("h3", "", "The model is not loaded"));
      card.append(el("p", "sub", String(status.load_error).split("\n")[0]));
      if (status.load_help) card.append(el("p", "sub", status.load_help));
      const hint = el("p", "fine");
      hint.append(document.createTextNode("On the server: "));
      hint.append(el("code", "", "ai doctor"));
      hint.append(document.createTextNode(" says which of the two fixes this machine needs."));
      card.append(hint);
      nodes.push(card);
    }

    const controls = el("div", "row");
    controls.append(control("Sleep now", "/control/sleep"));
    controls.append(control("Wake", "/control/wake"));
    controls.append(control(status.learning_paused ? "Resume learning" : "Pause learning",
      "/control/learning", { paused: !status.learning_paused }));
    controls.append(control("Release the GPU", "/control/release-gpu"));
    nodes.push(controls);

    nodes.push(el("h2", "", `Waiting for you (${approvals.approvals.length})`));
    if (!approvals.approvals.length) nodes.push(el("p", "empty", "Nothing is waiting."));
    for (const approval of approvals.approvals) nodes.push(approvalCard(approval));

    nodes.push(el("h2", "", "Self-expansion"));
    if (expansion.triggers && expansion.triggers.length) {
      nodes.push(el("p", "sub", "What is pushing it to grow right now:"));
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
          (change.created_at || "").slice(0, 16)]))
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
      nodes.push(el("p", "fine", `next: ${optimisation.next_trials
        .map((trial) => `${trial.setting}=${trial.candidate}`).join(" · ")}`));
    }
    return nodes;
  });
}

function control(label, path, body) {
  const button = el("button", "btn small", label);
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      await api(path, { method: "POST", body: body || {} });
      toast(label.toLowerCase(), "good");
      loadSystem();
      refreshState();
    } catch (error) { button.disabled = false; toast(error.message, "bad"); }
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
  if (approval.rationale) card.append(el("p", "fine", approval.rationale));
  const actions = el("div", "row");
  for (const [label, approve] of [["Approve", true], ["Reject", false]]) {
    const button = el("button", `btn small${approve ? " primary" : ""}`, label);
    button.addEventListener("click", async () => {
      let note = "";
      if (!approve) {
        note = await sheet({ title: `Reject #${approval.id}?`, note: "Why not? (optional)", confirm: "Reject" });
        if (note === null) return;
      }
      button.disabled = true;
      try { await api(`/approvals/${approval.id}`, { body: { approve, note } }); loadSystem(); }
      catch (error) { button.disabled = false; toast(error.message, "bad"); }
    });
    actions.append(button);
  }
  card.append(actions);
  return card;
}

boot().catch((error) => {
  document.body.append(el("p", "error", `could not reach the daemon: ${error.message}`));
});
