const DATA_POLL_INTERVAL_MS = 500;
const BACKGROUND_POLL_INTERVAL_MS = 2000;
const LOG_POLL_INTERVAL_MS = 2000;
const BACKGROUND_LOG_POLL_INTERVAL_MS = 5000;
const VIDEO_RETRY_MS = 2000;
const VIDEO_VIEWPORT_GAP_PX = 18;

let dataPollTimer = null;
let logPollTimer = null;
let lastDataSuccessAt = 0;
let lastSampleChangedAt = 0;
let lastSampleSignature = "";
let pollingGeneration = 0;
let videoLiveEnabled = true;
let videoLiveSource = "";
let videoRateMode = "30";

function setStatus(id, state, detail) {
  const el = document.getElementById(id);
  if (!el) return;
  el.dataset.state = state;
  const small = el.querySelector("small");
  if (small && detail) small.textContent = detail;
}

function latestNumber(data, key) {
  const values = data?.[key];
  if (!Array.isArray(values)) return null;
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = Number(values[index]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function formatFps(value) {
  if (!Number.isFinite(value) || value <= 0) return "—";
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function formatCounter(value) {
  return Number.isFinite(value) ? Math.max(0, Math.round(value)).toLocaleString() : "—";
}

function updateVideoMetrics(data) {
  const camera = latestNumber(data, "runtime/camera_fps");
  const algorithm = latestNumber(data, "runtime/algorithm_fps");
  const web = latestNumber(data, "runtime/web_fps");
  const cameraCaptured = latestNumber(data, "runtime/camera_captured");
  const cameraDropped = latestNumber(data, "runtime/camera_dropped");
  const algorithmReceived = latestNumber(data, "runtime/algorithm_received");
  const webEncoded = latestNumber(data, "runtime/web_encoded");
  const webDropped = latestNumber(data, "runtime/web_dropped");
  const webEncodeMs = latestNumber(data, "runtime/web_encode_ms");

  const cameraValue = document.getElementById("cameraFps");
  const algorithmValue = document.getElementById("algorithmFps");
  const webValue = document.getElementById("webFps");
  if (cameraValue) cameraValue.textContent = formatFps(camera);
  if (algorithmValue) algorithmValue.textContent = formatFps(algorithm);
  if (webValue) webValue.textContent = formatFps(web);

  const cameraMetric = document.getElementById("cameraMetric");
  const algorithmMetric = document.getElementById("algorithmMetric");
  const webMetric = document.getElementById("webMetric");
  const cameraActive = Number.isFinite(camera) && camera > 0;
  const algorithmActive = Number.isFinite(algorithm) && algorithm > 0;
  const webActive = Number.isFinite(web) && web > 0;
  if (cameraMetric) {
    cameraMetric.dataset.state = cameraActive ? "ok" : "idle";
    cameraMetric.title = `CAM：相机驱动实际采集 · 总计 ${formatCounter(cameraCaptured)} · 缓冲覆盖 ${formatCounter(cameraDropped)}`;
  }
  if (algorithmMetric) {
    const fallingBehind = cameraActive && Number.isFinite(algorithm) && algorithm < camera * 0.8;
    algorithmMetric.dataset.state = algorithmActive ? (fallingBehind ? "warn" : "ok") : "idle";
    algorithmMetric.title = `ALG：主算法循环成功处理 · 总计 ${formatCounter(algorithmReceived)}${fallingBehind ? " · 低于相机输入" : ""}`;
  }
  if (webMetric) {
    webMetric.dataset.state = webActive ? "ok" : "idle";
    webMetric.title = `WEB：后台 JPEG 输出 · 编码 ${formatCounter(webEncoded)} · 覆盖 ${formatCounter(webDropped)} · ${Number.isFinite(webEncodeMs) ? webEncodeMs.toFixed(1) : "—"} ms/帧`;
  }
}

function showToast(message, state = "ok") {
  const root = document.getElementById("toast-region");
  if (!root) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.dataset.state = state;
  toast.textContent = message;
  root.appendChild(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function setForceModeAlert(mode, yamlMode = "") {
  const alert = document.getElementById("force-mode-alert");
  const copy = document.getElementById("force-mode-alert-copy");
  if (!alert) return;
  alert.hidden = !mode && !yamlMode;
  if (copy && mode) {
    copy.textContent = `当前强制为 ${mode}，下位机 WorkMode 正被覆盖，比赛前必须清空。`;
  } else if (copy && yamlMode) {
    copy.textContent = `当前进程已临时清空，但 YAML 仍为 ${yamlMode}，重启后会重新启用。比赛前必须写回空值。`;
  }
}

window.setStatus = setStatus;
window.showToast = showToast;
window.setForceModeAlert = setForceModeAlert;

function updateVideoPanelHeightLimit() {
  const panel = document.getElementById("video-panel");
  if (!panel || document.fullscreenElement) return;
  const top = panel.getBoundingClientRect().top;
  const available = Math.max(260, window.innerHeight - top - VIDEO_VIEWPORT_GAP_PX);
  document.documentElement.style.setProperty("--video-panel-max-height", `${available}px`);
}

async function toggleFullscreen() {
  const container = document.querySelector(".video-container");
  if (!container) return;
  try {
    if (!document.fullscreenElement) await container.requestFullscreen();
    else await document.exitFullscreen();
  } catch (error) {
    showToast(`无法切换全屏：${error.message}`, "bad");
  }
}

function bindVideoStatus() {
  const image = document.getElementById("video-stream");
  const placeholder = document.getElementById("video-placeholder");
  const caption = document.getElementById("video-caption");
  if (!image) return;
  videoLiveSource = image.getAttribute("src") || "/video";

  const markReady = () => {
    if (!videoLiveEnabled) return;
    placeholder?.setAttribute("hidden", "");
    setStatus("video-status", "ok", "视频在线");
    if (caption) caption.textContent = "视频在线";
  };
  const markFailed = () => {
    if (!videoLiveEnabled) return;
    placeholder?.removeAttribute("hidden");
    setStatus("video-status", "bad", "等待重连");
    if (caption) caption.textContent = "连接中断，自动重试";
    window.setTimeout(() => {
      if (videoLiveEnabled) image.src = buildVideoUrl("retry");
    }, VIDEO_RETRY_MS);
  };
  image.addEventListener("load", markReady);
  image.addEventListener("error", markFailed);
  if (image.complete && image.naturalWidth > 0) markReady();
}

function buildVideoUrl(marker = "resume") {
  const url = new URL(videoLiveSource || "/video", window.location.href);
  url.searchParams.set("fps", videoRateMode === "realtime" ? "0" : "30");
  url.searchParams.delete("retry");
  url.searchParams.delete("resume");
  url.searchParams.set(marker, String(Date.now()));
  return `${url.pathname}${url.search}`;
}

function setVideoRateMode(mode, reconnect = true) {
  videoRateMode = mode === "realtime" ? "realtime" : "30";
  const label = document.getElementById("videoFpsLabel");
  document.querySelectorAll("[data-video-rate]").forEach((button) => {
    const active = button.dataset.videoRate === videoRateMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const labelText = videoRateMode === "realtime" ? "REALTIME" : "30 FPS";
  if (label) label.textContent = labelText;
  if (reconnect && videoLiveEnabled) {
    const image = document.getElementById("video-stream");
    if (image) image.src = buildVideoUrl("rate");
    window.showToast?.(`视频刷新率：${videoRateMode === "realtime" ? "实时源帧" : "30 FPS"}`);
  }
}

function setVideoLiveEnabled(enabled) {
  const image = document.getElementById("video-stream");
  const container = document.querySelector(".video-container");
  const placeholder = document.getElementById("video-placeholder");
  const caption = document.getElementById("video-caption");
  const indicator = document.getElementById("liveIndicator");
  const button = document.getElementById("liveToggleBtn");
  if (!image || !container) return;

  videoLiveEnabled = enabled;
  container.classList.toggle("live-disabled", !enabled);
  if (indicator) {
    indicator.dataset.state = enabled ? "on" : "off";
    indicator.textContent = enabled ? "LIVE" : "LIVE OFF";
  }
  if (button) {
    button.textContent = enabled ? "关闭 LIVE" : "开启 LIVE";
    button.setAttribute("aria-pressed", String(!enabled));
  }
  if (!enabled) {
    image.removeAttribute("src");
    placeholder?.removeAttribute("hidden");
    const strong = placeholder?.querySelector("strong");
    const small = placeholder?.querySelector("small");
    if (strong) strong.textContent = "LIVE 显示已关闭";
    if (small) small.textContent = "视频请求已断开，曲线与日志继续更新";
    if (caption) caption.textContent = "视频显示已关闭";
    setStatus("video-status", "warn", "显示已关闭");
    return;
  }

  const strong = placeholder?.querySelector("strong");
  const small = placeholder?.querySelector("small");
  if (strong) strong.textContent = "正在恢复视频流";
  if (small) small.textContent = "等待下一帧";
  if (caption) caption.textContent = "正在重新连接";
  setStatus("video-status", "pending", "重新连接");
  image.src = buildVideoUrl("resume");
}

window.setVideoLiveEnabled = setVideoLiveEnabled;

function bindControls() {
  document.getElementById("fullscreenBtn")?.addEventListener("click", toggleFullscreen);
  document.getElementById("liveToggleBtn")?.addEventListener("click", () => setVideoLiveEnabled(!videoLiveEnabled));
  document.querySelectorAll("[data-video-rate]").forEach((button) => {
    button.addEventListener("click", () => setVideoRateMode(button.dataset.videoRate));
  });
  document.addEventListener("fullscreenchange", () => {
    document.body.classList.toggle("fullscreen-mode", Boolean(document.fullscreenElement));
    if (!document.fullscreenElement) updateVideoPanelHeightLimit();
  });
  window.addEventListener("resize", updateVideoPanelHeightLimit);
  window.addEventListener("orientationchange", updateVideoPanelHeightLimit);
  bindVideoStatus();
  setVideoRateMode("30", false);
}

function updateSampleAge() {
  const el = document.getElementById("sample-age");
  if (!el) return;
  if (!lastDataSuccessAt) {
    el.textContent = "—";
    return;
  }
  const age = Math.max(0, performance.now() - lastDataSuccessAt);
  el.textContent = age < 1000 ? `${Math.round(age)} ms` : `${(age / 1000).toFixed(1)} s`;
}

async function pollDataOnce(generation) {
  const maxPoints = typeof window.getChartMaxPoints === "function" ? window.getChartMaxPoints() : 200;
  try {
    const response = await fetch(`/data?max_points=${maxPoints}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || response.statusText);

    const time = Array.isArray(data.time) ? data.time : [];
    const signature = `${time.length}:${time.at(-1) ?? "none"}`;
    const now = performance.now();
    lastDataSuccessAt = now;
    if (signature !== lastSampleSignature) {
      lastSampleSignature = signature;
      lastSampleChangedAt = now;
    }
    const stale = !time.length || (lastSampleChangedAt && now - lastSampleChangedAt > 2000);
    setStatus("data-status", stale ? "warn" : "ok", stale ? "数据未更新" : `${time.length} 点`);
    updateVideoMetrics(data);
    document.dispatchEvent(new CustomEvent("newvision:data", { detail: data }));
  } catch (error) {
    setStatus("data-status", "bad", "读取失败");
    document.dispatchEvent(new CustomEvent("newvision:data-error", { detail: error }));
  } finally {
    if (generation !== pollingGeneration) return;
    const delay = document.hidden ? BACKGROUND_POLL_INTERVAL_MS : DATA_POLL_INTERVAL_MS;
    dataPollTimer = window.setTimeout(() => pollDataOnce(generation), delay);
  }
}

async function pollLogOnce(generation) {
  try {
    const snapshot = document.getElementById("json-log-container");
    if (!snapshot?.hidden || !window.loggerSocketConnected) {
      await fetchAndDisplayJsonWithTree("json-log", "/log");
    }
  } catch (error) {
    const snapshot = document.getElementById("json-log");
    if (snapshot && !window.loggerSocketConnected) snapshot.textContent = `诊断快照读取失败：${error.message}`;
  } finally {
    if (generation !== pollingGeneration) return;
    const delay = document.hidden ? BACKGROUND_LOG_POLL_INTERVAL_MS : LOG_POLL_INTERVAL_MS;
    logPollTimer = window.setTimeout(() => pollLogOnce(generation), delay);
  }
}

function restartPollers() {
  pollingGeneration += 1;
  window.clearTimeout(dataPollTimer);
  window.clearTimeout(logPollTimer);
  pollDataOnce(pollingGeneration);
  pollLogOnce(pollingGeneration);
}

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  updateVideoPanelHeightLimit();
  if (typeof initCharts === "function") initCharts();
  if (typeof initHealthAndTri === "function") initHealthAndTri();
  if (typeof initParams === "function") initParams();
  if (typeof initReplay === "function") initReplay();
  restartPollers();
  window.setInterval(updateSampleAge, 500);
  document.addEventListener("visibilitychange", restartPollers);
});
