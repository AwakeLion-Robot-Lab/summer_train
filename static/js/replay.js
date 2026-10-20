const REPLAY_MAX_POINTS = 4000;
let replayChart = null;
let replayFrames = 0;
let replayPlaying = false;
let replayTimer = null;
let activeReplay = "";

function replayStatus(text) { const el = document.getElementById("replayPos"); if (el) el.textContent = text; }

function renderReplaySeries(series) {
  const canvas = document.getElementById("replayChart");
  if (!canvas || typeof window.createTelemetryChart !== "function") return;
  const keys = Object.keys(series).filter((key) => key !== "time" && Array.isArray(series[key]));
  const preferred = ["aim/yaw", "gimbal/yaw", "aim/pitch", "gimbal/pitch", "ekf/nis"];
  const selected = preferred.filter((key) => keys.includes(key));
  if (!selected.length && keys.length) selected.push(keys[0]);
  if (!replayChart) replayChart = window.createTelemetryChart(canvas);
  replayChart.setData(series, selected.slice(0, 6));
}

function showReplayFrame(index) {
  const image = document.getElementById("replayImg");
  if (!image || !activeReplay) return;
  if (replayFrames <= 0) { image.removeAttribute("src"); replayStatus("无帧"); return; }
  const clamped = Math.max(0, Math.min(replayFrames - 1, index));
  image.src = `/replays/${encodeURIComponent(activeReplay)}/frames/${clamped}`;
  const scrub = document.getElementById("replayScrub"); if (scrub) scrub.value = String(clamped);
  replayStatus(`${clamped + 1} / ${replayFrames}`);
}

function stopReplayPlayback() {
  replayPlaying = false;
  if (replayTimer) { window.clearInterval(replayTimer); replayTimer = null; }
  const button = document.getElementById("replayPlayBtn"); if (button) button.textContent = "播放";
}

async function loadReplay(name, frames) {
  stopReplayPlayback(); activeReplay = name; replayFrames = frames;
  const view = document.getElementById("replay-view"); if (view) view.hidden = false;
  const scrub = document.getElementById("replayScrub");
  if (scrub) { scrub.max = String(Math.max(0, frames - 1)); scrub.value = "0"; scrub.oninput = () => showReplayFrame(Number(scrub.value)); }
  showReplayFrame(0); replayStatus("正在加载曲线…");
  try {
    const response = await fetch(`/replays/${encodeURIComponent(name)}/series?max_points=${REPLAY_MAX_POINTS}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || response.statusText);
    renderReplaySeries(data.series);
    showReplayFrame(0);
    if (data.sampled) window.showToast?.(`回放曲线已从 ${data.raw_count} 点抽样到 ${data.returned_count} 点`);
  } catch (error) { replayStatus(`曲线失败：${error.message}`); }
}

function renderReplayList(sessions) {
  const root = document.getElementById("replay-list");
  if (!root) return;
  root.replaceChildren();
  if (!sessions.length) { const empty = document.createElement("p"); empty.className = "params-hint"; empty.textContent = "records/ 下暂无可回放会话。"; root.appendChild(empty); return; }
  sessions.forEach((session) => {
    const button = document.createElement("button"); button.type = "button"; button.className = "replay-item";
    const title = document.createElement("strong"); title.textContent = session.name;
    const meta = document.createElement("small"); meta.textContent = `${session.samples} samples · ${session.frames} frames`;
    button.append(title, meta); button.addEventListener("click", () => loadReplay(session.name, session.frames)); root.appendChild(button);
  });
}

async function loadReplays() {
  const root = document.getElementById("replay-list"); if (root) root.textContent = "正在扫描录制目录…";
  try {
    const response = await fetch("/replays", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || response.statusText);
    renderReplayList(data.sessions || []);
  } catch (error) { if (root) root.textContent = `回放列表失败：${error.message}`; }
}

function initReplay() {
  document.getElementById("reloadReplaysBtn")?.addEventListener("click", loadReplays);
  document.getElementById("replayPlayBtn")?.addEventListener("click", () => {
    if (replayPlaying) { stopReplayPlayback(); return; }
    if (replayFrames <= 0) return;
    replayPlaying = true; document.getElementById("replayPlayBtn").textContent = "暂停";
    replayTimer = window.setInterval(() => {
      const scrub = document.getElementById("replayScrub"); const next = Number(scrub?.value || 0) + 1;
      if (next >= replayFrames) { stopReplayPlayback(); return; }
      showReplayFrame(next);
    }, 50);
  });
  loadReplays();
}
