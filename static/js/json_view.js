const jsonSignatures = new Map();

function jsonToHtml(data, container) {
  const previousOpen = new Map(
    [...container.querySelectorAll("details[data-path]")].map((node) => [node.dataset.path, node.open])
  );
  const scrollParent = container.closest(".json-container");
  const scrollTop = scrollParent?.scrollTop ?? 0;
  const fragment = document.createDocumentFragment();

  function buildTree(value, parent, path = "root") {
    if (typeof value !== "object" || value === null) {
      parent.textContent = String(value);
      return;
    }
    const ul = document.createElement("ul");
    ul.className = "json-tree";
    const entries = Array.isArray(value) ? value.map((v, i) => [i, v]) : Object.entries(value);
    for (const [key, child] of entries) {
      const li = document.createElement("li");
      const childPath = `${path}/${key}`;
      if (typeof child === "object" && child !== null) {
        const details = document.createElement("details");
        details.dataset.path = childPath;
        details.open = previousOpen.has(childPath) ? previousOpen.get(childPath) : path === "root";
        const summary = document.createElement("summary");
        summary.textContent = Array.isArray(child) ? `${key} [${child.length}]` : key;
        details.appendChild(summary);
        buildTree(child, details, childPath);
        li.appendChild(details);
      } else {
        const keySpan = document.createElement("span");
        keySpan.className = "json-leaf-key";
        keySpan.textContent = `${key}: `;
        const valueSpan = document.createElement("span");
        valueSpan.className = "json-leaf-value";
        valueSpan.textContent = String(child);
        li.append(keySpan, valueSpan);
      }
      ul.appendChild(li);
    }
    parent.appendChild(ul);
  }

  buildTree(data, fragment);
  container.replaceChildren(fragment);
  if (scrollParent) scrollParent.scrollTop = scrollTop;
}

async function fetchAndDisplayJsonWithTree(id, url) {
  const container = document.getElementById(id);
  if (!container) return null;
  const response = await fetch(url, { cache: "no-store" });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  const signature = JSON.stringify(data);
  if (jsonSignatures.get(id) !== signature) {
    jsonToHtml(data, container);
    jsonSignatures.set(id, signature);
  }
  return data;
}
