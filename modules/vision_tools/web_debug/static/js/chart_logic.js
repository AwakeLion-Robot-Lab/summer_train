const DEFAULT_KEYS = ["aim/yaw", "gimbal/yaw", "aim/pitch", "gimbal/pitch", "delay/before_fire", "fire/yaw_err", "fire/pitch_err", "ekf/nis"];
const CHART_PRESETS = {
  tracking: ["aim/yaw", "gimbal/yaw", "aim/pitch", "gimbal/pitch", "obs/yaw", "ekf/nis"],
  delay: ["delay/image_to_plan", "delay/plan_to_send", "delay/send_to_control", "delay/control_to_fire", "delay/fire_to_hit", "delay/before_fire"],
  fire: ["fire/yaw_err", "fire/pitch_err", "fire/feasible", "fire/shoot", "fire/reason"],
  buff: ["buff/rot_speed", "buff/rot_angle", "buff/plan_valid", "buff/sine_valid", "buff/degraded"],
};
const CHART_COLORS = ["#d6a34a", "#729eff", "#63c49b", "#df6658", "#b9c0bc", "#8aa6d8", "#c4a979", "#88a895"];

function formatAxis(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000 || (magnitude > 0 && magnitude < .01)) return value.toExponential(1);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

class TelemetryChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.data = {};
    this.keys = [];
    this.hoverIndex = null;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener("pointermove", (event) => this.onPointer(event));
    canvas.addEventListener("pointerleave", () => { this.hoverIndex = null; this.draw(); });
  }

  setData(data, keys) {
    this.data = data || {};
    this.keys = keys.filter((key) => Array.isArray(this.data[key]));
    this.draw();
  }

  onPointer(event) {
    const count = this.data.time?.length || this.data[this.keys[0]]?.length || 0;
    if (!count) return;
    const rect = this.canvas.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left - 48) / Math.max(1, rect.width - 64)));
    this.hoverIndex = Math.round(ratio * (count - 1));
    this.draw();
  }

  draw() {
    const width = Math.max(120, this.canvas.clientWidth || 300);
    const height = Math.max(100, this.canvas.clientHeight || 180);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#141618";
    ctx.fillRect(0, 0, width, height);

    const series = this.keys.map((key, index) => ({ key, color: CHART_COLORS[index % CHART_COLORS.length], values: this.data[key] }));
    const finite = series.flatMap((item) => item.values.filter(Number.isFinite));
    if (!series.length || !finite.length) {
      ctx.fillStyle = "#5f6668";
      ctx.font = "12px Noto Sans CJK SC, sans-serif";
      ctx.fillText("暂无遥测数据，请先选择通道", 18, 28);
      return;
    }

    let min = Math.min(...finite);
    let max = Math.max(...finite);
    if (min === max) { min -= 1; max += 1; }
    const padding = (max - min) * .08;
    min -= padding; max += padding;
    const left = 48, right = 14, top = series.length > 1 ? 31 : 15, bottom = 25;
    const plotW = Math.max(1, width - left - right);
    const plotH = Math.max(1, height - top - bottom);
    const count = Math.max(...series.map((item) => item.values.length));
    const xAt = (index) => left + (count <= 1 ? 0 : index / (count - 1)) * plotW;
    const yAt = (value) => top + (max - value) / (max - min) * plotH;

    ctx.lineWidth = 1;
    ctx.font = "10px Noto Sans Mono, monospace";
    for (let i = 0; i <= 4; i += 1) {
      const y = top + i / 4 * plotH;
      const value = max - i / 4 * (max - min);
      ctx.strokeStyle = "rgba(150,156,156,.13)";
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.fillStyle = "#5f6668"; ctx.textAlign = "right"; ctx.fillText(formatAxis(value), left - 6, y + 3);
    }
    const time = Array.isArray(this.data.time) ? this.data.time : [];
    const t0 = Number.isFinite(time[0]) ? time[0] : 0;
    ctx.textAlign = "left"; ctx.fillStyle = "#5f6668"; ctx.fillText("0", left, height - 7);
    const duration = time.length > 1 && Number.isFinite(time.at(-1)) ? time.at(-1) - t0 : count - 1;
    ctx.textAlign = "right"; ctx.fillText(time.length > 1 ? `${duration.toFixed(2)} s` : `${count} 点`, width - right, height - 7);

    series.forEach((item) => {
      ctx.strokeStyle = item.color; ctx.lineWidth = 1.45; ctx.lineJoin = "round"; ctx.beginPath();
      let drawing = false;
      item.values.forEach((value, index) => {
        if (!Number.isFinite(value)) { drawing = false; return; }
        const x = xAt(index), y = yAt(value);
        if (!drawing) { ctx.moveTo(x, y); drawing = true; } else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    if (series.length > 1) {
      let x = left;
      ctx.textAlign = "left"; ctx.font = "10px Noto Sans Mono, monospace";
      for (const item of series) {
        const label = item.key.length > 22 ? `…${item.key.slice(-21)}` : item.key;
        ctx.fillStyle = item.color; ctx.fillRect(x, 9, 7, 2);
        ctx.fillStyle = "#909697"; ctx.fillText(label, x + 11, 13);
        x += Math.min(170, ctx.measureText(label).width + 24);
        if (x > width - 140) break;
      }
    }

    if (this.hoverIndex !== null) {
      const index = Math.min(count - 1, Math.max(0, this.hoverIndex));
      const x = xAt(index);
      ctx.strokeStyle = "rgba(232,238,241,.45)"; ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotH); ctx.stroke();
      const rows = series.map((item) => `${item.key}: ${formatAxis(item.values[index] ?? 0)}`);
      ctx.font = "10px Noto Sans Mono, monospace";
      const boxW = Math.min(width - 12, Math.max(...rows.map((row) => ctx.measureText(row).width)) + 14);
      const boxH = rows.length * 14 + 8;
      const boxX = x + boxW + 8 < width ? x + 7 : x - boxW - 7;
      ctx.fillStyle = "rgba(20,22,24,.96)"; ctx.fillRect(boxX, top + 5, boxW, boxH);
      ctx.strokeStyle = "#3b4042"; ctx.strokeRect(boxX, top + 5, boxW, boxH);
      rows.forEach((row, i) => { ctx.fillStyle = series[i].color; ctx.textAlign = "left"; ctx.fillText(row, boxX + 7, top + 18 + i * 14); });
    }
  }

  destroy() { this.resizeObserver.disconnect(); }
}

window.createTelemetryChart = (canvas) => new TelemetryChart(canvas);

let mainChart = null;
const individualCharts = new Map();
let selectedKeys = new Set(DEFAULT_KEYS);
let maxPoints = 200;
let latestData = {};
let knownDataKeySignature = "";

window.getChartMaxPoints = () => maxPoints;

function renderMainChart() {
  const canvas = document.getElementById("mainChart");
  const surface = document.getElementById("main-chart-surface");
  if (!canvas || document.hidden || surface?.classList.contains("is-hidden")) return;
  if (!mainChart) mainChart = new TelemetryChart(canvas);
  mainChart.setData(latestData, [...selectedKeys]);
}

function renderIndividualCharts() {
  const root = document.getElementById("individualCharts");
  const panel = document.getElementById("individual-chart-panel");
  const drawer = document.getElementById("tool-drawer");
  if (!root || document.hidden || panel?.hidden || drawer?.classList.contains("collapsed")) return;
  for (const key of selectedKeys) {
    if (!Array.isArray(latestData[key])) continue;
    if (!individualCharts.has(key)) {
      const card = document.createElement("div");
      card.className = "chart-card";
      card.dataset.key = key;
      const title = document.createElement("h3"); title.textContent = key;
      const canvas = document.createElement("canvas");
      card.append(title, canvas); root.appendChild(card);
      individualCharts.set(key, { chart: new TelemetryChart(canvas), card });
    }
    individualCharts.get(key).chart.setData(latestData, [key]);
  }
  for (const [key, entry] of individualCharts) {
    if (!selectedKeys.has(key)) { entry.chart.destroy(); entry.card.remove(); individualCharts.delete(key); }
  }
}

function syncSelector() {
  document.querySelectorAll("#chartSelectControls input").forEach((input) => { input.checked = selectedKeys.has(input.dataset.key); });
  const count = document.getElementById("selected-key-count");
  if (count) count.textContent = `${selectedKeys.size} 项`;
}

function rebuildKeySelector(keys) {
  const root = document.getElementById("chartSelectControls");
  if (!root) return;
  const sortedKeys = [...keys].sort();
  const signature = sortedKeys.join("\u0000");
  if (signature === knownDataKeySignature) return;
  knownDataKeySignature = signature;
  const known = new Set([...root.querySelectorAll("input")].map((input) => input.dataset.key));
  for (const key of sortedKeys) {
    if (known.has(key)) continue;
    const label = document.createElement("label"); label.dataset.search = key.toLowerCase();
    const input = document.createElement("input"); input.type = "checkbox"; input.dataset.key = key; input.checked = selectedKeys.has(key);
    label.append(input, document.createTextNode(key)); root.appendChild(label);
  }
  syncSelector();
}

function setPreset(name) {
  selectedKeys = new Set(CHART_PRESETS[name] || DEFAULT_KEYS);
  document.querySelectorAll("[data-chart-preset]").forEach((button) => button.classList.toggle("is-active", button.dataset.chartPreset === name));
  syncSelector(); renderMainChart(); renderIndividualCharts();
}

function initCharts() {
  document.addEventListener("newvision:data", (event) => {
    latestData = event.detail || {};
    rebuildKeySelector(Object.keys(latestData).filter((key) => key !== "time" && Array.isArray(latestData[key])));
    renderMainChart(); renderIndividualCharts();
  });
  document.getElementById("chartSelectControls")?.addEventListener("change", (event) => {
    const input = event.target;
    if (!input.matches("input[type=checkbox]")) return;
    if (input.checked) selectedKeys.add(input.dataset.key); else selectedKeys.delete(input.dataset.key);
    document.querySelectorAll("[data-chart-preset]").forEach((button) => button.classList.remove("is-active"));
    syncSelector(); renderMainChart(); renderIndividualCharts();
  });
  document.getElementById("chartKeySearch")?.addEventListener("input", (event) => {
    const query = event.target.value.trim().toLowerCase();
    document.querySelectorAll("#chartSelectControls label").forEach((label) => { label.hidden = Boolean(query) && !label.dataset.search.includes(query); });
  });
  document.querySelectorAll("[data-chart-preset]").forEach((button) => button.addEventListener("click", () => setPreset(button.dataset.chartPreset)));
  document.getElementById("multiLineChart")?.addEventListener("change", (event) => {
    document.getElementById("main-chart-surface")?.classList.toggle("is-hidden", !event.target.checked);
    if (event.target.checked) renderMainChart();
  });
  document.addEventListener("workspace:panel-open", (event) => {
    if (event.detail?.targetId === "individual-chart-panel") renderIndividualCharts();
  });
  document.getElementById("applyMainRange")?.addEventListener("click", () => {
    const input = document.getElementById("mainMaxPts");
    maxPoints = Math.min(1000, Math.max(10, Number(input?.value) || 200));
    if (input) input.value = String(maxPoints);
    window.showToast?.(`时序窗口已设为 ${maxPoints} 点`);
  });
  syncSelector();
}
