const LOGGER_RECONNECT_MS = 1800;
const LOGGER_MAX_ROWS = 500;
const LOGGER_BATCH_MS = 100;

let loggerSocket = null;
let loggerReconnectTimer = null;
let activeLogFilter = "";
let pendingLoggerEntries = [];
let loggerFlushTimer = null;

function setActiveDrawer(targetId, { scroll = false } = {}) {
  const drawer = document.getElementById("tool-drawer");
  if (!drawer || !document.getElementById(targetId)) return;
  drawer.classList.remove("collapsed");
  document.getElementById("drawerCollapse")?.setAttribute("aria-expanded", "true");
  document.querySelectorAll(".drawer-panel").forEach((panel) => {
    const active = panel.id === targetId;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  document.querySelectorAll("[data-drawer-target]").forEach((button) => button.classList.toggle("active", button.dataset.drawerTarget === targetId));
  document.querySelectorAll("[data-open-drawer]").forEach((button) => button.classList.toggle("active", button.dataset.openDrawer === targetId));
  document.dispatchEvent(new CustomEvent("workspace:panel-open", { detail: { targetId } }));
  if (scroll) drawer.scrollIntoView({ behavior: "smooth", block: "start" });
}

function formatLoggerTime(value) {
  if (typeof value === "number") {
    const millis = value > 1e15 ? value / 1e6 : value > 1e12 ? value : value * 1000;
    const date = new Date(millis);
    if (!Number.isNaN(date.getTime())) return `${date.toLocaleTimeString("zh-CN", { hour12: false })}.${String(date.getMilliseconds()).padStart(3, "0")}`;
  }
  const text = String(value ?? "");
  const time = text.match(/\d{2}:\d{2}:\d{2}(?:\.\d+)?/);
  return time?.[0] || text.slice(0, 18) || "—";
}

function buildLoggerRow(entry) {
  const level = String(entry.level || "INFO").toUpperCase();
  const file = String(entry.file_name || "").split(/[\\/]/).pop();
  const module = file || entry.tid || "runtime";
  const location = entry.line ? `${module}:${entry.line}` : module;
  const row = document.createElement("div");
  row.className = `logger-row ${level.toLowerCase()}`;
  row.dataset.level = level;
  row.hidden = Boolean(activeLogFilter) && level !== activeLogFilter;

  const fields = [
    ["log-time", formatLoggerTime(entry.timestamp)],
    ["log-level-cell", level],
    ["log-module", location],
    ["log-message", entry.msg ?? ""],
  ];
  fields.forEach(([className, value]) => {
    const cell = document.createElement("span");
    cell.className = className;
    cell.textContent = String(value);
    if (className === "log-message" && entry.function_name) cell.title = `${entry.function_name} · ${location}`;
    row.appendChild(cell);
  });
  return row;
}

function flushLoggerEntries() {
  loggerFlushTimer = null;
  if (document.hidden || !pendingLoggerEntries.length) return;
  const root = document.getElementById("logger-log");
  if (!root) return;
  root.querySelector(".logger-empty")?.remove();
  const entries = pendingLoggerEntries.splice(0);
  const fragment = document.createDocumentFragment();
  entries.forEach((entry) => fragment.appendChild(buildLoggerRow(entry)));
  root.appendChild(fragment);
  while (root.childElementCount > LOGGER_MAX_ROWS) root.firstElementChild?.remove();
  root.scrollTop = root.scrollHeight;
}

function appendLoggerEntry(entry) {
  pendingLoggerEntries.push(entry);
  if (pendingLoggerEntries.length > LOGGER_MAX_ROWS) pendingLoggerEntries.splice(0, pendingLoggerEntries.length - LOGGER_MAX_ROWS);
  if (!document.hidden && !loggerFlushTimer) loggerFlushTimer = window.setTimeout(flushLoggerEntries, LOGGER_BATCH_MS);
}

function sendLoggerCommand(command) {
  if (!loggerSocket || loggerSocket.readyState !== WebSocket.OPEN) return false;
  loggerSocket.send(JSON.stringify(command));
  return true;
}

function connectLoggerSocket() {
  window.clearTimeout(loggerReconnectTimer);
  const port = Number(document.body.dataset.loggerWsPort || 1234);
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname || "127.0.0.1";
  try {
    loggerSocket = new WebSocket(`${protocol}://${host}:${port}`);
  } catch (error) {
    window.setStatus?.("log-status", "bad", "WS 创建失败");
    loggerReconnectTimer = window.setTimeout(connectLoggerSocket, LOGGER_RECONNECT_MS);
    return;
  }
  loggerSocket.addEventListener("open", () => {
    window.loggerSocketConnected = true;
    window.setStatus?.("log-status", "warn", "中继在线");
    loggerSocket.send(JSON.stringify({ client: "newvision-web", command: "SUBSCRIBE" }));
  });
  loggerSocket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") return;
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "log" && payload.entry) {
        appendLoggerEntry(payload.entry);
        window.setStatus?.("log-status", "ok", "aw_logger 在线");
      } else if (payload.level && payload.msg) {
        appendLoggerEntry(payload);
        window.setStatus?.("log-status", "ok", "aw_logger 在线");
      }
    } catch (_) { /* 非日志反馈忽略 */ }
  });
  loggerSocket.addEventListener("close", () => {
    window.loggerSocketConnected = false;
    window.setStatus?.("log-status", "warn", "等待 logger");
    loggerReconnectTimer = window.setTimeout(connectLoggerSocket, LOGGER_RECONNECT_MS);
  });
  loggerSocket.addEventListener("error", () => window.setStatus?.("log-status", "bad", "WS 连接失败"));
}

function showLogView(view, filter = "") {
  const stream = document.getElementById("logger-log");
  const snapshot = document.getElementById("json-log-container");
  const showingSnapshot = view === "snapshot";
  if (stream) stream.hidden = showingSnapshot;
  if (snapshot) snapshot.hidden = !showingSnapshot;
  activeLogFilter = showingSnapshot ? "" : filter;
  document.querySelectorAll(".logger-row").forEach((row) => { row.hidden = Boolean(activeLogFilter) && row.dataset.level !== activeLogFilter; });
}

function initWorkspace() {
  document.querySelectorAll("[data-drawer-target]").forEach((button) => button.addEventListener("click", () => setActiveDrawer(button.dataset.drawerTarget)));
  document.querySelectorAll("[data-open-drawer]").forEach((button) => button.addEventListener("click", () => setActiveDrawer(button.dataset.openDrawer, { scroll: true })));
  document.getElementById("drawerCollapse")?.addEventListener("click", (event) => {
    const drawer = document.getElementById("tool-drawer");
    const collapsed = drawer?.classList.toggle("collapsed");
    event.currentTarget.textContent = collapsed ? "展开工作抽屉" : "收起工作抽屉";
    event.currentTarget.setAttribute("aria-expanded", String(!collapsed));
  });
  document.querySelectorAll(".log-tab").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".log-tab").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    showLogView(button.dataset.logView || "stream", button.dataset.logFilter || "");
  }));
  document.getElementById("clearLoggerBtn")?.addEventListener("click", () => {
    const root = document.getElementById("logger-log");
    pendingLoggerEntries = [];
    if (root) root.innerHTML = '<div class="logger-empty">日志已清空，等待新消息…</div>';
  });
  document.getElementById("loggerLevel")?.addEventListener("change", (event) => {
    if (sendLoggerCommand({ command: "SET_LEVEL", level: event.target.value })) window.showToast?.(`日志等级已切换为 ${event.target.value}`);
    else window.showToast?.("logger WebSocket 尚未连接", "bad");
  });
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    const shortcuts = { "1": "tri-view-panel", "2": "params-panel", "3": "replay-panel", "4": "individual-chart-panel" };
    if (shortcuts[event.key]) setActiveDrawer(shortcuts[event.key], { scroll: true });
    if (event.key.toLowerCase() === "f") document.getElementById("fullscreenBtn")?.click();
    if (event.key.toLowerCase() === "v") document.getElementById("liveToggleBtn")?.click();
  });
  connectLoggerSocket();
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) flushLoggerEntries();
  });
}

document.addEventListener("DOMContentLoaded", initWorkspace);
