let paramCatalog = [];
const loadedValues = new Map();

const PARAM_GROUP_META = {
  plan: {
    title: "规划与补偿",
    description: "延迟、固定零偏与回退弹速。建议先标定链路延迟，再用静止靶修正零偏。",
  },
  ekf: {
    title: "EKF 状态估计",
    description: "过程噪声决定跟随速度，观测方差决定滤波器对 PnP 测量的信任程度。",
  },
  fire: {
    title: "开火容差",
    description: "只调整几何放行条件，不会绕过 shoot_enable、模式与通信状态等总闸门。",
  },
  debug: {
    title: "调试输出",
    description: "只影响本地可视化与网页画面负载，正常情况下不应改变瞄准结果。",
  },
  camera: {
    title: "相机运行时参数",
    description: "曝光与增益需要一起下发，仅作用于当前进程；重启后恢复相机配置文件中的值。",
  },
};

function paramsStatus(text, state = "neutral") {
  const el = document.getElementById("params-status");
  if (!el) return;
  el.textContent = text;
  el.dataset.state = state;
}

function setParamsBusy(busy) {
  ["applyParamsBtn", "saveParamsBtn", "reloadParamsBtn", "clearParamsBtn", "applyForceModeBtn"].forEach((id) => {
    const button = document.getElementById(id);
    if (button) button.disabled = busy;
  });
}

function valuesEqual(a, b) {
  if (typeof a === "number" || typeof b === "number") return Number(a) === Number(b);
  return a === b;
}

function updateDirtyState(row, value) {
  row.classList.toggle("is-dirty", !valuesEqual(value, loadedValues.get(row.dataset.key)));
  const dirty = document.querySelectorAll(".param-row.is-dirty").length;
  const apply = document.getElementById("applyParamsBtn");
  if (apply) apply.textContent = dirty ? `下发修改 · ${dirty}` : "下发修改";
}

function buildParamsUi(catalog, yamlValues, cmdValues) {
  const root = document.getElementById("params-grid");
  if (!root) return;
  root.replaceChildren(); loadedValues.clear(); paramCatalog = catalog;
  const groups = new Map();
  catalog.forEach((item) => { if (!groups.has(item.group)) groups.set(item.group, []); groups.get(item.group).push(item); });

  for (const [group, items] of groups) {
    const section = document.createElement("section"); section.className = "params-group";
    const groupMeta = PARAM_GROUP_META[group] || { title: group, description: "" };
    const groupHead = document.createElement("header"); groupHead.className = "params-group-head";
    const titleLine = document.createElement("div"); titleLine.className = "params-group-title";
    const title = document.createElement("h3"); title.textContent = groupMeta.title;
    const technicalName = document.createElement("span"); technicalName.textContent = group;
    const groupHelp = document.createElement("button"); groupHelp.type = "button"; groupHelp.className = "group-help";
    groupHelp.textContent = "i"; groupHelp.setAttribute("aria-label", `查看${groupMeta.title}说明`);
    titleLine.append(title, technicalName, groupHelp); groupHead.appendChild(titleLine);
    if (groupMeta.description) {
      const description = document.createElement("p"); description.className = "group-description"; description.setAttribute("role", "tooltip"); description.textContent = groupMeta.description;
      groupHead.appendChild(description);
    }
    section.appendChild(groupHead);
    for (const item of items) {
      const row = document.createElement("div"); row.className = `param-row ${item.kind === "bool" ? "boolean" : ""}`; row.dataset.key = item.key;
      const hasCommand = Object.hasOwn(cmdValues || {}, item.key);
      const initial = hasCommand ? cmdValues[item.key] : yamlValues?.[item.key];
      loadedValues.set(item.key, initial);
      row.classList.toggle("is-runtime", hasCommand);

      const copy = document.createElement("div"); copy.className = "param-copy";
      const labelLine = document.createElement("div"); labelLine.className = "param-label-line";
      const label = document.createElement("label"); label.htmlFor = `param-${item.key}`;
      label.textContent = item.unit ? `${item.name || item.label} · ${item.unit}` : (item.name || item.label);
      labelLine.appendChild(label); copy.appendChild(labelLine);
      if (item.description) {
        const tooltipId = `param-tip-${item.key.replaceAll(".", "-")}`;
        const help = document.createElement("button"); help.type = "button"; help.className = "param-help"; help.textContent = "i";
        help.setAttribute("aria-label", `查看${item.name || item.label}说明`); help.setAttribute("aria-describedby", tooltipId);
        const tooltip = document.createElement("div"); tooltip.className = "param-tooltip"; tooltip.id = tooltipId; tooltip.setAttribute("role", "tooltip");
        const key = document.createElement("span"); key.className = "param-key"; key.textContent = item.key;
        const description = document.createElement("p"); description.className = "param-description"; description.textContent = item.description;
        tooltip.append(key, description); labelLine.appendChild(help); copy.appendChild(tooltip);
      }
      let input;
      if (item.kind === "bool") {
        input = document.createElement("input"); input.type = "checkbox"; input.checked = Boolean(initial);
        input.addEventListener("change", () => updateDirtyState(row, input.checked));
        row.append(copy, input);
      } else {
        input = document.createElement("input"); input.type = "number"; input.min = item.min; input.max = item.max; input.step = item.step;
        if (typeof initial === "number") input.value = String(initial);
        const range = document.createElement("input"); range.type = "range"; range.min = item.min; range.max = item.max; range.step = item.step;
        range.value = String(typeof initial === "number" ? initial : item.min);
        input.addEventListener("input", () => { range.value = input.value; updateDirtyState(row, Number(input.value)); });
        range.addEventListener("input", () => { input.value = range.value; updateDirtyState(row, Number(range.value)); });
        row.append(copy, input, range);
      }
      input.id = `param-${item.key}`; input.dataset.paramInput = "";
      const meta = document.createElement("div"); meta.className = "param-meta";
      const rangeText = item.kind === "bool" ? "布尔开关" : `${item.min} … ${item.max}`;
      meta.innerHTML = `<span>${rangeText}</span><span>${item.persist === false ? "仅运行时" : hasCommand ? "命令覆盖" : "YAML"}</span>`;
      row.appendChild(meta); section.appendChild(row);
    }
    root.appendChild(section);
  }
}

function collectParamValues({ dirtyOnly = false, persistentOnly = false } = {}) {
  const values = {};
  for (const item of paramCatalog) {
    if (persistentOnly && item.persist === false) continue;
    const row = document.querySelector(`.param-row[data-key="${item.key}"]`);
    if (!row || (dirtyOnly && !row.classList.contains("is-dirty"))) continue;
    const input = row.querySelector("[data-param-input]");
    if (!input) continue;
    if (item.kind === "bool") values[item.key] = input.checked;
    else if (input.value !== "" && Number.isFinite(Number(input.value))) values[item.key] = Number(input.value);
  }
  return values;
}

async function parseJsonResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || (data.rejected && Object.values(data.rejected).join("；")) || response.statusText);
  return data;
}

async function loadParams() {
  setParamsBusy(true); paramsStatus("正在读取 YAML 与运行时覆盖…");
  try {
    const response = await fetch("/params", { cache: "no-store" });
    const data = await parseJsonResponse(response);
    buildParamsUi(data.catalog || [], data.yaml || {}, data.cmd || {});
    const force = data.force_mode || {};
    const effective = Object.hasOwn(force, "cmd") ? force.cmd : (force.yaml || "");
    const select = document.getElementById("forceWorkMode"); if (select) select.value = effective;
    const source = document.getElementById("force-mode-source");
    if (source) {
      source.textContent = Object.hasOwn(force, "cmd")
        ? `运行时覆盖：${force.cmd || "空"} · YAML：${force.yaml || "空"}`
        : `当前来自 YAML：${force.yaml || "空"} · 非空时忽略下位机模式`;
    }
    window.setForceModeAlert?.(effective, force.yaml || "");
    const overrides = Object.keys(data.cmd || {}).length + (Object.hasOwn(force, "cmd") ? 1 : 0);
    paramsStatus(`已加载 ${data.catalog.length} 项${overrides ? `，${overrides} 项命令覆盖` : "，当前无命令覆盖"}`, "ok");
  } catch (error) {
    paramsStatus(`加载失败：${error.message}`, "bad");
  } finally { setParamsBusy(false); }
}

async function applyParams() {
  const values = collectParamValues({ dirtyOnly: true });
  if (!Object.keys(values).length) { paramsStatus("没有尚未下发的修改"); return; }
  setParamsBusy(true);
  try {
    const data = await parseJsonResponse(await fetch("/params", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) }));
    window.showToast?.(`已下发 ${data.accepted.length} 项运行时参数`);
    await loadParams();
  } catch (error) { paramsStatus(`热更失败：${error.message}`, "bad"); window.showToast?.(error.message, "bad"); }
  finally { setParamsBusy(false); }
}

async function saveParams() {
  const values = collectParamValues({ persistentOnly: true });
  if (!window.confirm("把界面中的可持久化参数写入 config/auto_aim.yaml？原注释将保留。")) return;
  setParamsBusy(true);
  try {
    const data = await parseJsonResponse(await fetch("/params/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values) }));
    window.showToast?.(`已写入 YAML：${data.written.length} 项`);
    await loadParams();
  } catch (error) { paramsStatus(`写入失败：${error.message}`, "bad"); window.showToast?.(error.message, "bad"); }
  finally { setParamsBusy(false); }
}

async function clearParams() {
  if (!window.confirm("清空命令覆盖文件？当前进程已经应用的值不会自动回退，重启后才完全回到 YAML。")) return;
  setParamsBusy(true);
  try {
    await parseJsonResponse(await fetch("/params/clear", { method: "POST" }));
    window.showToast?.("命令覆盖已清空；当前进程值保持到重启", "ok");
    await loadParams();
  } catch (error) { paramsStatus(`清空失败：${error.message}`, "bad"); }
  finally { setParamsBusy(false); }
}

async function applyForceMode() {
  const value = document.getElementById("forceWorkMode")?.value ?? "";
  if (value && !window.confirm(`强制 WorkMode = ${value}？下位机上报的模式会被忽略。`)) return;
  setParamsBusy(true);
  try {
    const data = await parseJsonResponse(await fetch("/params", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ "debug.force_work_mode": value }) }));
    if (!data.accepted.includes("debug.force_work_mode")) throw new Error("后端未接受强制模式字段");
    window.setForceModeAlert?.(value); window.showToast?.(value ? `强制模式已切换为 ${value}` : "强制模式覆盖已清空");
    await loadParams();
  } catch (error) { paramsStatus(`强制模式失败：${error.message}`, "bad"); window.showToast?.(error.message, "bad"); }
  finally { setParamsBusy(false); }
}

function initParams() {
  document.getElementById("applyParamsBtn")?.addEventListener("click", applyParams);
  document.getElementById("saveParamsBtn")?.addEventListener("click", saveParams);
  document.getElementById("reloadParamsBtn")?.addEventListener("click", loadParams);
  document.getElementById("clearParamsBtn")?.addEventListener("click", clearParams);
  document.getElementById("applyForceModeBtn")?.addEventListener("click", applyForceMode);
  loadParams();
}
