const dataNode = document.getElementById("report-bundle-data");
const rawBundle = dataNode ? JSON.parse(dataNode.textContent || "{}") : {};
const byId = (id) => document.getElementById(id);
const normalizedBundle = normalizeBundle(rawBundle);
const state = {
  expandedByOutput: {},
  selectedOutputId: defaultOutputId(normalizedBundle),
  activeView: "home",
  colorMode: "entropy",
  entropyMode: "95_5",
  mainColWidth: 50,
};

function normalizeBundle(bundle) {
  const outputs = Array.isArray(bundle.outputs) ? bundle.outputs : [];
  if (outputs.length > 0) return bundle;
  if (bundle.step_views) {
    return {
      outputs: [
        {
          id: "default-output",
          label: "Default Output",
          prompt: String(((bundle.config || {}).prompt || "")),
          run_dir: "",
          report: bundle,
        },
      ],
      algorithm_overview: "",
    };
  }
  return { outputs: [], algorithm_overview: "" };
}

function defaultOutputId(bundle) {
  const outputs = Array.isArray(bundle.outputs) ? bundle.outputs : [];
  return outputs.length > 0 ? String(outputs[0].id || "default-output") : null;
}

function outputs() {
  return Array.isArray(normalizedBundle.outputs) ? normalizedBundle.outputs : [];
}

function selectedOutput() {
  const currentId = String(state.selectedOutputId || "");
  const match = outputs().find((output) => String(output.id) === currentId);
  if (match) return match;
  return outputs()[0] || null;
}

function activeData() {
  const output = selectedOutput();
  return output ? (output.report || {}) : {};
}

function stepViews(data) {
  return Array.isArray(data.step_views) ? data.step_views : [];
}

function expandedState() {
  const output = selectedOutput();
  const outputId = output ? String(output.id) : "__none__";
  if (!state.expandedByOutput[outputId]) state.expandedByOutput[outputId] = {};
  return state.expandedByOutput[outputId];
}

function rolloutProbabilitySamples(data) {
  return sortedRolloutProbabilities(data.rollout_probabilities || []);
}

function rolloutEntropySamples(data) {
  return (data.rollout_entropies || [])
    .map((value) => Math.max(0, Number(value)))
    .filter((value) => Number.isFinite(value))
    .sort((left, right) => left - right);
}
function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
function sortedRolloutProbabilities(values) {
  const samples = [];
  for (const value of (values || [])) {
    const probability = Number(value);
    if (!Number.isFinite(probability)) continue;
    samples.push(Math.max(0, Math.min(1, probability)));
  }
  samples.sort((left, right) => left - right);
  return samples;
}

function upperBound(sortedValues, target) {
  let low = 0;
  let high = sortedValues.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (sortedValues[mid] <= target) low = mid + 1;
    else high = mid;
  }
  return low;
}

function probabilityPercentile(probability) {
  const samples = rolloutProbabilitySamples(activeData());
  const p = Math.max(0, Math.min(1, Number(probability || 0)));
  if (!samples.length) return p;
  const rank = upperBound(samples, p);
  return Math.max(0, Math.min(1, rank / samples.length));
}
function colorScore(probability, entropy) {
  if (state.colorMode === "none") return null;
  if (state.colorMode === "entropy") {
    const entropySamples = rolloutEntropySamples(activeData());
    const entropyValue = Math.max(0, Number(entropy || 0));
    const entropyPercentile = entropySamples.length
      ? Math.max(0, Math.min(1, upperBound(entropySamples, entropyValue) / entropySamples.length))
      : (entropyValue / (1 + entropyValue));
    const entropyBinaryThresholds = { "80_20": 0.8, "90_10": 0.9, "95_5": 0.95 };
    const binaryCutoff = entropyBinaryThresholds[state.entropyMode];
    const yellowCutoffPercentile = 0.8;
    const yellowColorScore = 0.51;
    const entropyContinuousScore = entropyPercentile <= yellowCutoffPercentile ? (1 - ((1 - yellowColorScore) * Math.pow(entropyPercentile / yellowCutoffPercentile, 1.4))) : (yellowColorScore * (1 - Math.pow((entropyPercentile - yellowCutoffPercentile) / (1 - yellowCutoffPercentile), 0.75)));
    if (binaryCutoff !== undefined) return entropyPercentile >= binaryCutoff ? entropyContinuousScore : null;
    return entropyContinuousScore;
  }
  const raw = Math.max(0, Math.min(1, Number(probability || 0)));
  const percentile = probabilityPercentile(raw);
  return Math.pow((raw * 0.85) + (percentile * 0.15), 1.2);
}
function probColor(probability, entropy) {
  const shaped = colorScore(probability, entropy);
  if (shaped === null) return "transparent";
  const hue = Math.round(4 + (shaped * 110));
  const sat = 84;
  const light = Math.round(76 - (shaped * 14));
  return `hsl(${hue}, ${sat}%, ${light}%)`;
}
function formatProbabilityPercent(probability) {
  const p = Math.max(0, Math.min(1, Number(probability || 0)));
  const rounded = Math.round((p * 100) * 10) / 10;
  if (Number.isInteger(rounded)) return `${rounded}%`;
  return `${rounded.toFixed(1)}%`;
}
function tooltipHtml(chip) {
  const selectedToken = String(chip.token || "");
  const selectedProb = Number(chip.probability || 0);
  const alternatives = (chip.alternatives || []);
  const dedup = [];
  const seen = new Set();
  dedup.push({ token: selectedToken, probability: selectedProb, selected: true });
  seen.add(`${selectedToken}@@${selectedProb}`);
  for (const alt of alternatives) {
    const token = String(alt.token || "");
    const prob = Number(alt.probability || 0);
    const key = `${token}@@${prob}`;
    if (token === selectedToken && Math.abs(prob - selectedProb) < 1e-9) continue;
    if (seen.has(key)) continue;
    seen.add(key);
    dedup.push({ token, probability: prob, selected: false });
  }
  const rows = dedup.slice(0, 5).map((alt) => {
    const prob = Number(alt.probability || 0);
    const selectedClass = alt.selected ? " selected" : "";
    const fill = `${Math.round(Math.max(0, Math.min(1, prob)) * 100)}%`;
    return `
      <div class="alt-row${selectedClass}" style="--fill:${fill};">
        <span class="alt-swatch" style="background:${probColor(prob, null)};"></span>
        <span class="alt-token">${esc(alt.token)}</span>
        <span class="alt-prob">${formatProbabilityPercent(prob)}</span>
      </div>
    `;
  }).join("");
  return `
    <div class="alt-grid">${rows || "<div class='alt-row'><span class='alt-swatch'></span><span class='alt-token'>no alternatives</span><span class='alt-prob'>-</span></div>"}</div>
  `;
}

function showTooltip(event, chip) {
  const tooltip = byId("tooltip");
  tooltip.innerHTML = tooltipHtml(chip);
  tooltip.classList.remove("hidden");
  const x = Math.min(window.innerWidth - 205, event.clientX + 14);
  const y = Math.min(window.innerHeight - 120, event.clientY + 14);
  tooltip.style.left = `${Math.max(8, x)}px`;
  tooltip.style.top = `${Math.max(8, y)}px`;
}

function hideTooltip() {
  byId("tooltip").classList.add("hidden");
}

function renderMeta() {
  if (state.activeView !== "report") {
    byId("meta").innerHTML = "";
    return;
  }
  const reportData = activeData();
  const cfg = reportData.config || {};
  document.documentElement.style.setProperty("--main-col-width", `${state.mainColWidth}%`);
  const pills = [
    `model: ${esc(cfg.model || "")}`,
    `mode: ${esc((cfg.api_mode_config || {}).default_mode || "")}`,
    `branch_factor: ${esc(cfg.branch_factor || "")}`,
    `trajectory_tokens: ${esc(reportData.trajectory_token_count || 0)}`,
    `cluster_mode: ${esc(reportData.cluster_mode || "")}`,
  ];
  const modeButton = (value, label) => (`<button class="mode-btn${state.colorMode === value ? " active" : ""}" data-mode="${value}">${label}</button>`);
  const entropyModeButton = (value, label) => (`<button class="mode-btn${state.entropyMode === value ? " active" : ""}" data-entropy-mode="${value}">${label}</button>`);
  const warnings = reportData.cluster_warnings || [];
  const warning = warnings.length ? `<div class="muted">cluster warning: ${esc(warnings.join(" | "))}</div>` : "";
  const entropySubmodeRow = state.colorMode === "entropy"
    ? `<div class="mode-row"><span class="muted">Entropy Style:</span>${entropyModeButton("continuous", "Continuous")}${entropyModeButton("80_20", "80/20")}${entropyModeButton("90_10", "90/10")}${entropyModeButton("95_5", "95/5")}</div>`
    : "";
  byId("meta").innerHTML = `
    <section class="meta-block"><h3>Metadata</h3><div class="meta">${pills.map((pill) => `<div class="meta-pill">${pill}</div>`).join("")}</div>${warning}</section>
    <section class="meta-block"><h3>Controls</h3><div class="mode-row"><span class="muted">Token Coloring:</span>${modeButton("none", "No Color")}${modeButton("hybrid", "Probability")}${modeButton("entropy", "Entropy")}</div>${entropySubmodeRow}<div class="mode-row"><span class="muted">Column Width:</span><input type="range" min="30" max="70" value="${esc(state.mainColWidth)}" data-main-col><span class="muted">${esc(state.mainColWidth)}%</span></div></section>
  `;
  for (const button of byId("meta").querySelectorAll("[data-mode]")) button.onclick = () => {
    state.colorMode = String(button.getAttribute("data-mode") || "hybrid");
    hideTooltip(); renderMeta(); renderTimeline(); renderFinalAnswer();
  };
  for (const button of byId("meta").querySelectorAll("[data-entropy-mode]")) button.onclick = () => {
    state.entropyMode = String(button.getAttribute("data-entropy-mode") || "95_5");
    hideTooltip(); renderMeta(); renderTimeline(); renderFinalAnswer();
  };
  const widthInput = byId("meta").querySelector("[data-main-col]"); if (widthInput) widthInput.oninput = () => { state.mainColWidth = Math.max(30, Math.min(70, Number(widthInput.value || 50))); renderMeta(); };
}

function parseInlineMarkdown(raw) {
  let html = esc(raw);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  let html = "";
  let paragraph = [];
  let inList = false;
  let inCode = false;
  let codeLines = [];
  const flushParagraph = () => {
    if (!paragraph.length) return;
    html += `<p>${parseInlineMarkdown(paragraph.join(" "))}</p>`;
    paragraph = [];
  };
  const closeList = () => {
    if (!inList) return;
    html += "</ul>";
    inList = false;
  };
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      flushParagraph();
      closeList();
      if (!inCode) {
        inCode = true;
        codeLines = [];
      } else {
        html += `<pre><code>${esc(codeLines.join("\n"))}</code></pre>`;
        inCode = false;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (!trimmed) {
      flushParagraph();
      closeList();
      continue;
    }
    if (trimmed.startsWith("- ")) {
      flushParagraph();
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${parseInlineMarkdown(trimmed.slice(2))}</li>`;
      continue;
    }
    if (trimmed.startsWith("### ")) {
      flushParagraph();
      closeList();
      html += `<h3>${parseInlineMarkdown(trimmed.slice(4))}</h3>`;
      continue;
    }
    if (trimmed.startsWith("## ")) {
      flushParagraph();
      closeList();
      html += `<h2>${parseInlineMarkdown(trimmed.slice(3))}</h2>`;
      continue;
    }
    if (trimmed.startsWith("# ")) {
      flushParagraph();
      closeList();
      html += `<h1>${parseInlineMarkdown(trimmed.slice(2))}</h1>`;
      continue;
    }
    paragraph.push(trimmed);
  }
  if (inCode) html += `<pre><code>${esc(codeLines.join("\n"))}</code></pre>`;
  flushParagraph();
  closeList();
  return html || "<p><em>No execution text captured.</em></p>";
}

function buildTokenStrip(tokens) {
  const strip = document.createElement("div");
  strip.className = "token-strip";
  for (const chip of (tokens || [])) {
    const node = document.createElement("span");
    node.className = "token-chip";
    node.textContent = chip.token || "";
    const synthetic = Boolean(chip.synthetic);
    const background = synthetic ? "transparent" : probColor(chip.probability || 0, chip.entropy || 0);
    node.style.background = background;
    node.style.color = state.colorMode === "none" || background === "transparent" ? "#dbeafe" : "#0f172a";
    if (!synthetic && state.colorMode !== "none") { node.onmouseenter = (event) => showTooltip(event, chip); node.onmousemove = (event) => showTooltip(event, chip); node.onmouseleave = hideTooltip; }
    else node.style.cursor = "text";
    strip.appendChild(node);
  }
  return strip;
}

function splitSteerExecTokenRows(tokens) {
  const rows = Array.isArray(tokens) ? tokens : [];
  if (!rows.length) return { steerTokens: [], execTokens: [] };
  let stitched = "";
  const spans = [];
  for (const row of rows) {
    const tokenText = String(row.token || "");
    const start = stitched.length;
    stitched += tokenText;
    spans.push([start, stitched.length]);
  }
  const steerCloseMatch = /<\/\s*steer\s*>/i.exec(stitched);
  if (!steerCloseMatch) return { steerTokens: rows, execTokens: [] };
  const steerCloseEnd = steerCloseMatch.index + steerCloseMatch[0].length;
  const execOpenPattern = /<\s*exec(?:ute)?\s*>/ig;
  execOpenPattern.lastIndex = steerCloseEnd;
  const execOpenMatch = execOpenPattern.exec(stitched);
  if (!execOpenMatch) return { steerTokens: rows, execTokens: [] };
  const execStart = execOpenMatch.index;
  const execClosePattern = /<\/\s*exec(?:ute)?\s*>/ig;
  execClosePattern.lastIndex = execStart;
  const execCloseMatch = execClosePattern.exec(stitched);
  const execEnd = execCloseMatch
    ? (execCloseMatch.index + execCloseMatch[0].length)
    : stitched.length;
  const steerTokens = [];
  const execTokens = [];
  for (let index = 0; index < rows.length; index += 1) {
    const [start, end] = spans[index];
    const midpoint = (start + end) / 2;
    if (midpoint < execStart) steerTokens.push(rows[index]);
    else if (midpoint <= execEnd) execTokens.push(rows[index]);
  }
  return { steerTokens, execTokens };
}

function extractExecTokenRows(tokens) {
  const rows = Array.isArray(tokens) ? tokens : [];
  if (!rows.length) return [];
  let stitched = "";
  const spans = [];
  for (const row of rows) {
    const tokenText = String(row.token || "");
    const start = stitched.length;
    stitched += tokenText;
    spans.push([start, stitched.length]);
  }
  const execOpenMatch = /<\s*exec(?:ute)?\s*>/i.exec(stitched);
  if (!execOpenMatch) return [];
  const execStart = execOpenMatch.index;
  const execClosePattern = /<\/\s*exec(?:ute)?\s*>/ig;
  execClosePattern.lastIndex = execStart;
  const execCloseMatch = execClosePattern.exec(stitched);
  const execEnd = execCloseMatch
    ? (execCloseMatch.index + execCloseMatch[0].length)
    : stitched.length;
  const execTokens = [];
  for (let index = 0; index < rows.length; index += 1) {
    const [start, end] = spans[index];
    const midpoint = (start + end) / 2;
    if (midpoint >= execStart && midpoint <= execEnd) {
      execTokens.push(rows[index]);
    }
  }
  return execTokens;
}

function buildChosenStructure(entry) {
  const structure = document.createElement("div");
  structure.className = "chosen-structure";
  const steerLine = document.createElement("div"); steerLine.style.cssText = "white-space:pre-wrap;word-break:break-word;";
  const steerOpen = document.createElement("span");
  steerOpen.className = "chosen-tag";
  steerOpen.textContent = "<steer>";
  const fullTokens = Array.isArray(entry.full_tokens) ? entry.full_tokens : [];
  const rolloutTokens = Array.isArray(entry.rollout_tokens) ? entry.rollout_tokens : [];
  const steerOnlyTokens = Array.isArray(entry.tokens) ? entry.tokens : [];
  const candidateRows = fullTokens.length ? fullTokens : steerOnlyTokens;
  const { steerTokens, execTokens: candidateExecTokens } = splitSteerExecTokenRows(
    candidateRows
  );
  const steerRows = steerTokens.length ? steerTokens : candidateRows;
  const rolloutExecTokens = extractExecTokenRows(rolloutTokens);
  const execTokens = rolloutExecTokens.length ? rolloutExecTokens : candidateExecTokens;
  const steerTokenStrip = buildTokenStrip(steerRows);
  steerTokenStrip.classList.add("chosen-token-strip");
  steerTokenStrip.style.display = "inline";
  steerLine.appendChild(steerOpen);
  steerLine.appendChild(steerTokenStrip);
  const execBody = execTokens.length
    ? buildTokenStrip(execTokens)
    : document.createElement("div");
  if (execTokens.length) {
    execBody.classList.add("chosen-token-strip", "chosen-exec-strip", "clamped");
  } else {
    const executionText = String(entry.execution_text || "").trim();
    execBody.className = "chosen-token-strip chosen-exec-fallback clamped";
    execBody.textContent = executionText
      ? `<exec>\n${executionText}\n</exec>`
      : "<exec></exec>";
  }
  const seeMore = document.createElement("button");
  seeMore.type = "button";
  seeMore.className = "see-more-btn hidden-block";
  seeMore.textContent = "+ see more";
  seeMore.onclick = () => {
    const expanded = seeMore.textContent === "+ see more";
    execBody.classList.toggle("clamped", !expanded);
    seeMore.textContent = expanded ? "- show less" : "+ see more";
  };
  structure.appendChild(steerLine);
  structure.appendChild(execBody);
  structure.appendChild(seeMore);
  requestAnimationFrame(() => {
    if (execBody.scrollHeight > (execBody.clientHeight + 1)) {
      seeMore.classList.remove("hidden-block");
      return;
    }
    seeMore.remove();
  });
  return structure;
}

function buildCandidatePanel(entry, chosen) {
  const panel = document.createElement("div");
  panel.className = chosen ? "candidate-panel chosen" : "candidate-panel";
  if (entry.selected) panel.classList.add("selected");
  const countTag = `<span class="tag">${esc(entry.count)}x</span>`;
  if (chosen) {
    panel.innerHTML = `<div class="candidate-meta">${countTag}${entry.selected ? "<span class='tag'>selected</span>" : ""}</div>`;
    panel.appendChild(buildChosenStructure(entry));
    return panel;
  }
  const row = document.createElement("div");
  row.className = "candidate-inline";
  row.innerHTML = countTag;
  const strip = buildTokenStrip(entry.tokens || []);
  strip.classList.add("candidate-inline-strip");
  row.appendChild(strip);
  panel.appendChild(row);
  return panel;
}

function ensureStepState(step) {
  const expanded = expandedState();
  const key = String(step.step_index);
  if (!(key in expanded)) expanded[key] = Number(step.step_index) === 0;
}

function uniqueCandidateCount(step) {
  let total = 0;
  for (const cluster of (step.clusters || [])) {
    total += Number(cluster.unique_count || 0);
  }
  return total;
}

function stepRolloutCount(step) {
  const explicitCount = Number(step.candidate_count || 0);
  if (Number.isFinite(explicitCount) && explicitCount > 0) return explicitCount;
  let total = 0;
  for (const cluster of (step.clusters || [])) {
    total += Number(cluster.count || 0);
  }
  return Math.max(1, total);
}

function rolloutPercent(step, cluster) {
  const total = stepRolloutCount(step);
  const ratio = Number(cluster.count || 0) / total;
  return Math.max(0, Math.min(100, ratio * 100));
}

function rolloutPercentLabel(step, cluster) {
  const percentValue = rolloutPercent(step, cluster);
  const rounded = Math.round(percentValue * 10) / 10;
  const text = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${text}%`;
}

function formatClusterName(name) {
  return String(name || "")
    .replace(/_/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function isTopCluster(cluster) {
  return (cluster.items || []).some((entry) => Boolean(entry.selected));
}

function clustersSelectedFirst(step) {
  const clusters = [...(step.clusters || [])];
  clusters.sort((left, right) => {
    const leftRank = isTopCluster(left) ? 0 : 1;
    const rightRank = isTopCluster(right) ? 0 : 1;
    return leftRank - rightRank;
  });
  return clusters;
}

function buildClusterRow(step, cluster) {
  const details = document.createElement("details");
  details.className = "cluster-row";
  const topCluster = isTopCluster(cluster);
  if (topCluster) details.classList.add("top-cluster");
  const fill = rolloutPercent(step, cluster);
  details.style.setProperty("--fill", `${fill}%`);
  const normalizedName = formatClusterName(cluster.name);
  const percentPrefix = rolloutPercentLabel(step, cluster);
  const clusterDisplayName = normalizedName
    ? `${percentPrefix} ${normalizedName}`
    : percentPrefix;
  const summary = document.createElement("summary");
  summary.className = "cluster-head";
  summary.innerHTML = `
    <div class="cluster-name-wrap">
      <span class="cluster-caret">&gt;</span>
      <div class="cluster-name">${esc(clusterDisplayName)}</div>
    </div>
    <div class="cluster-tags">
      <span class="cluster-count">${esc(cluster.count)} total</span>
      <span class="cluster-unique">${esc(cluster.unique_count)} unique</span>
    </div>
  `;
  details.appendChild(summary);
  const body = document.createElement("div");
  body.className = "cluster-body";
  const rows = (cluster.items || []);
  if (!rows.length) {
    const note = document.createElement("div");
    note.className = "hidden-note";
    note.textContent = "No additional unchosen variants in this cluster.";
    body.appendChild(note);
  } else {
    for (const entry of rows) body.appendChild(buildCandidatePanel(entry, false));
  }
  details.appendChild(body);
  return details;
}

function renderClusterRows(step, container) {
  for (const cluster of clustersSelectedFirst(step)) {
    container.appendChild(buildClusterRow(step, cluster));
  }
}

function buildStepCard(step) {
  ensureStepState(step);
  const reportData = activeData();
  const views = stepViews(reportData);
  const clusterBubbleMax = Math.max(
    1,
    ...(views.map((view) => Number(view.cluster_count || 0)))
  );
  const uniqueBubbleMax = Math.max(
    1,
    ...(views.map((view) => (view.clusters || []).reduce((sum, cluster) => sum + Number(cluster.unique_count || 0), 0)))
  );
  const expanded = expandedState();
  const key = String(step.step_index);
  const isExpanded = Boolean(expanded[key]);
  const card = document.createElement("section");
  card.className = "panel step-card";
  const stepNumber = Number(step.step_index) + 1;
  const clusterRatio = Math.max(0, Math.min(1, Number(step.cluster_count || 0) / clusterBubbleMax)); const uniqueRatio = Math.max(0, Math.min(1, uniqueCandidateCount(step) / uniqueBubbleMax));
  const clusterStyle = `border-color: hsla(${Math.round(208 - (clusterRatio * 74))}, 84%, 68%, ${0.42 + (clusterRatio * 0.44)});`; const uniqueStyle = `border-color: hsla(${Math.round(54 - (uniqueRatio * 26))}, 86%, 66%, ${0.42 + (uniqueRatio * 0.44)});`;
  const header = document.createElement("button");
  header.className = "step-header";
  header.innerHTML = `
    <div class="step-head-left">
      <span class="step-index">Step ${esc(stepNumber)}</span>
      <span class="step-steer">${esc(step.selected_text || "No selected steer text")}</span>
    </div>
    <div class="step-right">
      <span class="stat-pill" style="${clusterStyle}">${esc(step.cluster_count)} clusters</span>
      <span class="stat-pill" style="${uniqueStyle}">${esc(uniqueCandidateCount(step))} unique</span>
      <span class="step-chevron">${isExpanded ? "▾" : "▸"}</span>
    </div>
  `;
  header.onclick = () => {
    expanded[key] = !isExpanded;
    renderTimeline();
  };
  card.appendChild(header);
  const body = document.createElement("div");
  body.className = isExpanded ? "step-body" : "step-body collapsed";
  const clusterHeader = document.createElement("h3");
  clusterHeader.textContent = "Candidate Clusters";
  body.appendChild(clusterHeader);
  const clusterGrid = document.createElement("div");
  clusterGrid.className = "cluster-grid";
  renderClusterRows(step, clusterGrid);
  body.appendChild(clusterGrid);
  card.appendChild(body);
  return card;
}

function renderTimeline() {
  const timeline = byId("timeline");
  timeline.innerHTML = "";
  if (state.activeView !== "report") return;
  for (const step of stepViews(activeData())) timeline.appendChild(buildStepCard(step));
}
function renderFinalAnswer() {
  const panel = byId("final-answer");
  panel.innerHTML = "";
  if (state.activeView !== "report") return;
  const reportData = activeData();
  const prompt = String(((reportData.config || {}).prompt || "")).trim();
  const trajectory = String(reportData.final_text || "").trim();
  const trajectoryTokens = Array.isArray(reportData.trajectory_tokens) ? reportData.trajectory_tokens : [];
  const answer = String(reportData.final_answer_text || "").trim();
  panel.className = "side-column";
  panel.innerHTML = `<section class="panel hero"><h2>Prompt</h2><pre class="trajectory-pre clamped" data-prompt-text>${esc(prompt || "No prompt captured.")}</pre><button type="button" class="see-more-btn" data-prompt-toggle>+ see more</button></section><section class="panel hero" data-trajectory-panel><h2>Trajectory</h2></section><section class="panel hero" data-final-answer-panel><h2>Final Answer</h2></section>`;
  const promptText = panel.querySelector("[data-prompt-text]"); const promptToggle = panel.querySelector("[data-prompt-toggle]");
  if (promptText && promptToggle) { promptToggle.onclick = () => { const expanded = promptToggle.textContent === "+ see more"; promptText.classList.toggle("clamped", !expanded); promptToggle.textContent = expanded ? "- show less" : "+ see more"; }; requestAnimationFrame(() => { if (promptText.scrollHeight <= (promptText.clientHeight + 1)) promptToggle.remove(); }); }
  const trajectoryPanel = panel.querySelector("[data-trajectory-panel]"); const finalAnswerPanel = panel.querySelector("[data-final-answer-panel]"); if (!trajectoryPanel || !finalAnswerPanel) return;
  if (trajectoryTokens.length) { const trajectoryStrip = buildTokenStrip(trajectoryTokens); trajectoryStrip.classList.add("trajectory-pre"); trajectoryPanel.appendChild(trajectoryStrip); } else {
    trajectoryPanel.innerHTML += `<pre class="trajectory-pre">${esc(trajectory || "No trajectory text captured.")}</pre>`;
  }
  finalAnswerPanel.innerHTML += answer ? `<div class="exec-md">${markdownToHtml(answer)}</div>` : `<div class="muted">No final answer captured.</div>`;
}
function setSidebarOpen({ openButton, closeButton, sidebar, scrim, open, focusTarget }) { document.body.classList.toggle("sidebar-open", open); openButton.setAttribute("aria-expanded", open ? "true" : "false"); sidebar.setAttribute("aria-hidden", open ? "false" : "true"); scrim.setAttribute("aria-hidden", open ? "false" : "true"); if (focusTarget) focusTarget.focus({ preventScroll: true }); }

function outputLabel(output) {
  const prompt = String(output.prompt || "").trim();
  if (prompt) return prompt;
  const fallback = String(output.label || "").trim();
  return fallback || "(no prompt)";
}

function renderSidebarOutputs() {
  const container = byId("sidebar-output-list");
  if (!container) return;
  const currentId = String((selectedOutput() || {}).id || "");
  const outputButtons = outputs().map((output) => {
    const outputId = String(output.id || "");
    const activeClass = state.activeView === "report" && outputId === currentId ? " active" : "";
    return `<button type="button" class="sidebar-output-btn${activeClass}" data-select-output="${esc(outputId)}">${esc(outputLabel(output))}</button>`;
  }).join("");
  container.innerHTML = `<section class="meta-block"><h3>Outputs</h3><div class="output-list"><button type="button" class="sidebar-output-btn${state.activeView === "home" ? " active" : ""}" data-show-home>Home</button>${outputButtons || "<div class='muted'>No outputs found.</div>"}</div></section>`;
  const homeButton = container.querySelector("[data-show-home]");
  if (homeButton) homeButton.onclick = () => { state.activeView = "home"; renderApp(); };
  for (const button of container.querySelectorAll("[data-select-output]")) {
    button.onclick = () => {
      const nextId = String(button.getAttribute("data-select-output") || "");
      state.selectedOutputId = nextId;
      state.activeView = "report";
      renderApp();
    };
  }
}

function renderHome() {
  const homePanel = byId("home-view");
  const algorithmText = String(normalizedBundle.algorithm_overview || "").trim()
    || "The analysis process branches steers, clusters candidate behavior, selects a trajectory, and tracks token-level uncertainty metrics.";
  const cards = outputs().map((output) => {
    const outputId = String(output.id || "");
    const runDir = String(output.run_dir || "").trim();
    return `<article class="panel home-output-card"><h2>${esc(outputLabel(output))}</h2><div class="muted">${esc(runDir || "run directory not captured")}</div><button type="button" class="mode-btn" data-open-output="${esc(outputId)}">Open output</button></article>`;
  }).join("");
  homePanel.innerHTML = `<h1>Steer Branching Explorer</h1><section class="meta-block"><h3>How generation works</h3><div class="muted">${esc(algorithmText)}</div><div class="muted">Use this app to compare candidate clusters, selected <exec> execution blocks, and uncertainty metrics across outputs.</div></section><section class="home-grid">${cards || "<div class='muted'>No outputs found in this bundle.</div>"}</section>`;
  for (const button of homePanel.querySelectorAll("[data-open-output]")) {
    button.onclick = () => {
      state.selectedOutputId = String(button.getAttribute("data-open-output") || "");
      state.activeView = "report";
      renderApp();
    };
  }
}

function applyActiveView() {
  const showHome = state.activeView === "home";
  const homePanel = byId("home-view");
  const reportPanel = byId("report-view");
  if (homePanel) homePanel.classList.toggle("hidden-block", !showHome);
  if (reportPanel) reportPanel.classList.toggle("hidden-block", showHome);
}

function renderApp() {
  applyActiveView();
  renderSidebarOutputs();
  renderHome();
  renderMeta();
  renderTimeline();
  renderFinalAnswer();
}

function initSidebar() {
  const openButton = document.querySelector("[data-sidebar-open]"); const closeButton = document.querySelector("[data-sidebar-close]"); const sidebar = document.querySelector(".sidebar-panel"); const scrim = byId("sidebar-scrim");
  if (!openButton || !closeButton || !sidebar || !scrim) return;
  const closeSidebar = ({ focus } = { focus: true }) => { setSidebarOpen({ openButton, closeButton, sidebar, scrim, open: false, focusTarget: focus ? openButton : null }); try { window.localStorage.setItem("report_sidebar_open", "0"); } catch (_error) {} };
  const openSidebar = ({ focus } = { focus: true }) => { setSidebarOpen({ openButton, closeButton, sidebar, scrim, open: true, focusTarget: focus ? closeButton : null }); try { window.localStorage.setItem("report_sidebar_open", "1"); } catch (_error) {} };
  let startOpen = true; try { startOpen = window.localStorage.getItem("report_sidebar_open") !== "0"; } catch (_error) {}
  if (startOpen) openSidebar({ focus: false }); else closeSidebar({ focus: false });
  openButton.onclick = () => openSidebar(); closeButton.onclick = () => closeSidebar(); scrim.onclick = () => closeSidebar();
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) closeSidebar(); });
}
function bootstrap() {
  initSidebar();
  if (!outputs().length) state.activeView = "home";
  renderApp();
}
try {
  bootstrap();
} catch (error) {
  const message = (error && error.message) ? error.message : String(error);
  document.body.innerHTML = `<main><section class='panel hero'><h1>Report Render Error</h1><div class='muted'>${esc(message)}</div></section></main>`;
}
