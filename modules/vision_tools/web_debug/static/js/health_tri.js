const HEALTH_FIELDS = [
  { key: "serial/rx_dropped", label: "串口丢包", kind: "counter", bad: (v) => v > 0 },
  { key: "serial/rx_skipped_bytes", label: "跳字节", kind: "counter", bad: (v) => v > 0 },
  { key: "serial/pose_wait", label: "姿态等待 ms", kind: "value", warn: (v) => v > 8, bad: (v) => v > 20 },
  { key: "serial/pose_ready", label: "姿态就绪", kind: "flag", ok: (v) => v > .5 },
  { key: "track/state", label: "跟踪状态", kind: "enum", enum: ["Lost", "Detecting", "Tracking", "TempLost", "Switching"] },
  { key: "track/resets", label: "EKF 重建", kind: "counter", warn: (v) => v > 0 },
  { key: "track/n_obs", label: "观测数", kind: "value", ok: (v) => v >= 1 },
  { key: "delay/image_to_plan", label: "图像→规划 ms", kind: "value", warn: (v) => v > 15, bad: (v) => v > 30 },
  { key: "delay/plan_to_send", label: "规划→发送 ms", kind: "value", warn: (v) => v > 5, bad: (v) => v > 12 },
  { key: "delay/before_fire", label: "开火前总延迟 ms", kind: "value", warn: (v) => v > 120, bad: (v) => v > 200 },
  { key: "fire/shoot", label: "开火指令", kind: "flag", active: (v) => v > .5 },
  { key: "fire/reason", label: "拒绝原因", kind: "enum", enum: ["None", "NoFeasible", "YawErr", "PitchErr", "Jump", "Other"] },
  { key: "ekf/nis", label: "NIS", kind: "value", warn: (v) => v > 7.8, bad: (v) => v > 13.8 },
  { key: "aim/rejects", label: "规划拒绝帧", kind: "counter", warn: (v) => v > 0 },
  { key: "buff/rot_speed", label: "符转速 rad/s", kind: "value" },
  { key: "buff/plan_valid", label: "打符规划", kind: "flag", ok: (v) => v > .5 },
  { key: "buff/n_icon", label: "关联点", kind: "value", ok: (v) => v >= 1 },
  { key: "buff/n_pred", label: "预测点", kind: "value" },
  { key: "buff/sine_valid", label: "正弦拟合", kind: "flag", ok: (v) => v > .5 },
  { key: "buff/sine_omega", label: "拟合 ω", kind: "value" },
  { key: "buff/degraded", label: "打符降级", kind: "flag", ok: (v) => v < .5 },
  { key: "buff/update_count", label: "EKF 更新数", kind: "value" },
];

function lastOf(values) { return Array.isArray(values) && values.length ? values.at(-1) : undefined; }

function formatHealth(field, value) {
  if (value === undefined) return "—";
  if (field.kind === "enum") return field.enum?.[Math.round(value)] ?? String(Math.round(value));
  if (field.kind === "flag") return value > .5 ? "是" : "否";
  if (field.kind === "counter") return String(Math.round(value));
  return Number(value).toFixed(2);
}

function healthStatus(field, value) {
  if (value === undefined) return "unknown";
  if (field.active) return field.active(value) ? "active" : "ok";
  if (field.bad?.(value)) return "bad";
  if (field.warn?.(value)) return "warn";
  if (field.ok) return field.ok(value) ? "ok" : "warn";
  return "ok";
}

function renderHealth(data) {
  const root = document.getElementById("health-grid");
  if (!root) return;
  if (!root.childElementCount) {
    for (const field of HEALTH_FIELDS) {
      const cell = document.createElement("div"); cell.className = "health-cell"; cell.dataset.key = field.key;
      const label = document.createElement("div"); label.className = "health-label"; label.textContent = field.label;
      const value = document.createElement("div"); value.className = "health-value"; value.textContent = "—";
      cell.append(label, value); root.appendChild(cell);
    }
  }
  const counts = { bad: 0, warn: 0, unknown: 0 };
  for (const field of HEALTH_FIELDS) {
    const cell = root.querySelector(`[data-key="${field.key}"]`);
    if (!cell) continue;
    const value = lastOf(data[field.key]);
    const status = healthStatus(field, value);
    cell.querySelector(".health-value").textContent = formatHealth(field, value);
    cell.dataset.status = status;
    if (Object.hasOwn(counts, status)) counts[status] += 1;
  }
  const summary = document.getElementById("health-summary");
  if (summary) {
    if (counts.bad) summary.textContent = `${counts.bad} 项异常`;
    else if (counts.warn) summary.textContent = `${counts.warn} 项注意`;
    else if (counts.unknown === HEALTH_FIELDS.length) summary.textContent = "等待数据";
    else if (counts.unknown) summary.textContent = `${counts.unknown} 项无数据`;
    else summary.textContent = "链路正常";
  }
}

function drawTriView(canvasId, series, axisA, axisB, labelA, labelB) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#141618"; ctx.fillRect(0, 0, w, h);
  const target = [lastOf(series["ekf/x"]), lastOf(series["ekf/y"]), lastOf(series["ekf/z"])];
  const obs = [lastOf(series["obs/x"]), lastOf(series["obs/y"]), lastOf(series["obs/z"])];
  const points = [];
  if (target.every(Number.isFinite)) points.push({ x: target[axisA], y: target[axisB], color: "#d6a34a", label: "估计" });
  if (obs.every(Number.isFinite)) points.push({ x: obs[axisA], y: obs[axisB], color: "#63c49b", label: "观测" });
  if (!points.length) { ctx.fillStyle = "#5f6668"; ctx.font = "12px sans-serif"; ctx.fillText("等待目标", 12, 23); return; }

  let minX = 0, maxX = 0, minY = 0, maxY = 0;
  points.forEach((point) => { minX = Math.min(minX, point.x); maxX = Math.max(maxX, point.x); minY = Math.min(minY, point.y); maxY = Math.max(maxY, point.y); });
  const spanX = Math.max(maxX - minX, .2), spanY = Math.max(maxY - minY, .2);
  minX -= spanX * .15; maxX += spanX * .15; minY -= spanY * .15; maxY += spanY * .15;
  const toPx = (x, y) => [((x - minX) / (maxX - minX)) * (w - 26) + 13, h - 14 - ((y - minY) / (maxY - minY)) * (h - 30)];
  const [ox, oy] = toPx(0, 0);
  ctx.strokeStyle = "#34383a"; ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, h); ctx.moveTo(0, oy); ctx.lineTo(w, oy); ctx.stroke();
  ctx.fillStyle = "#5f6668"; ctx.beginPath(); ctx.arc(ox, oy, 2.5, 0, Math.PI * 2); ctx.fill();
  points.forEach((point) => { const [x, y] = toPx(point.x, point.y); ctx.fillStyle = point.color; ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.font = "11px sans-serif"; ctx.fillText(point.label, x + 7, y - 6); });
  ctx.fillStyle = "#5f6668"; ctx.font = "10px sans-serif"; ctx.textAlign = "right"; ctx.fillText(`${labelA} →`, w - 8, h - 4);
  ctx.save(); ctx.translate(10, 42); ctx.rotate(-Math.PI / 2); ctx.textAlign = "left"; ctx.fillText(`${labelB} →`, 0, 0); ctx.restore(); ctx.textAlign = "left";
}

function renderTriView(data) {
  const panel = document.getElementById("tri-view-panel");
  const drawer = document.getElementById("tool-drawer");
  if (document.hidden || panel?.hidden || drawer?.classList.contains("collapsed")) return;
  drawTriView("triTop", data, 0, 1, "前后 x", "左右 y"); drawTriView("triSide", data, 0, 2, "前后 x", "上下 z"); drawTriView("triFront", data, 1, 2, "左右 y", "上下 z");
  const target = [lastOf(data["ekf/x"]), lastOf(data["ekf/y"]), lastOf(data["ekf/z"])];
  const obs = [lastOf(data["obs/x"]), lastOf(data["obs/y"]), lastOf(data["obs/z"])];
  const deviation = target.every(Number.isFinite) && obs.every(Number.isFinite)
    ? Math.hypot(target[0] - obs[0], target[1] - obs[1], target[2] - obs[2]) : undefined;
  const values = [["云台 Yaw", lastOf(data["gimbal/yaw"]), "°", 1], ["云台 Pitch", lastOf(data["gimbal/pitch"]), "°", 1], ["观测距离", lastOf(data["obs/distance"]), " m", 2], ["估计/观测偏差", deviation, " m", 2]];
  const caption = document.getElementById("tri-caption");
  if (caption) caption.textContent = values.filter(([, value]) => Number.isFinite(value)).map(([name, value, unit, digits]) => `${name} ${value.toFixed(digits)}${unit}`).join("  ·  ") || "—";
}

function initHealthAndTri() {
  let latestData = {};
  document.addEventListener("newvision:data", (event) => { latestData = event.detail; renderHealth(latestData); renderTriView(latestData); });
  document.addEventListener("workspace:panel-open", (event) => {
    if (event.detail?.targetId === "tri-view-panel") renderTriView(latestData);
  });
}
