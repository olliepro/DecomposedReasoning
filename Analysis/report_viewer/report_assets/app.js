const dataNode = document.getElementById("report-bundle-data");
let rawBundle = dataNode ? JSON.parse(dataNode.textContent || "{}") : {};
const byId = (id) => document.getElementById(id);
let normalizedBundle = normalizeBundle(rawBundle);
const state = {
  expandedByOutput: {},
  selectedOutputId: defaultOutputId(normalizedBundle),
  activeView: "home",
  colorMode: "entropy",
  entropyMode: "95_5",
  mainColWidth: 50,
  loadingOutputId: null,
  outputLoadErrors: {},
  trajectoryRenderJobsByOutput: {},
};
const reportCacheByOutput = {};
let activeTrajectoryCleanup = null;

function normalizeBundle(bundle) {
  const outputs = Array.isArray(bundle.outputs) ? bundle.outputs : [];
  const promptTitles = (
    bundle
    && typeof bundle === "object"
    && bundle.prompt_titles
    && typeof bundle.prompt_titles === "object"
  )
    ? bundle.prompt_titles
    : {};
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
      prompt_titles: promptTitles,
    };
  }
  return { outputs: [], algorithm_overview: "", prompt_titles: promptTitles };
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

function outputId(output) {
  return String((output || {}).id || "");
}

function hasReportPayload(report) {
  if (!report || typeof report !== "object") return false;
  if (Array.isArray(report.step_views)) return true;
  if (typeof report.final_text === "string") return true;
  if (typeof report.final_answer_text === "string") return true;
  if (Array.isArray(report.token_list)) return true;
  return false;
}

function cachedReport(output) {
  const id = outputId(output);
  if (!id) return null;
  const existing = reportCacheByOutput[id];
  if (existing) return existing;
  const inlineReport = (output || {}).report;
  if (!hasReportPayload(inlineReport)) return null;
  reportCacheByOutput[id] = inlineReport;
  return inlineReport;
}

function isOutputReady(output) {
  return Boolean(cachedReport(output));
}

function activeData() {
  const output = selectedOutput();
  if (!output) return {};
  const report = cachedReport(output);
  return report || {};
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

function disposeActiveTrajectoryLoader() {
  if (typeof activeTrajectoryCleanup === "function") {
    activeTrajectoryCleanup();
  }
  activeTrajectoryCleanup = null;
}

function rolloutProbabilitySamples(data) {
  if (data._c_rollout_prob_samples) return data._c_rollout_prob_samples;
  data._c_rollout_prob_samples = sortedRolloutProbabilities(
    getRolloutProbabilities(data),
  );
  return data._c_rollout_prob_samples;
}

function rolloutEntropySamples(data) {
  if (data._c_rollout_entropy_samples) return data._c_rollout_entropy_samples;
  data._c_rollout_entropy_samples = (getRolloutEntropies(data))
    .map((value) => Math.max(0, Number(value)))
    .filter((value) => Number.isFinite(value))
    .sort((left, right) => left - right);
  return data._c_rollout_entropy_samples;
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

function tokenColorLegend() {
  if (state.colorMode === "none") return null;
  if (state.colorMode === "entropy") {
    const labelByMode = {
      continuous: "Continuous Entropy Tokens",
      "80_20": "80/20 Entropy Tokens",
      "90_10": "90/10 Entropy Tokens",
      "95_5": "95/5 Entropy Tokens",
    };
    const label = labelByMode[state.entropyMode] || "Entropy Tokens";
    const color = probColor(0, 1e6);
    return {
      label,
      color: color === "transparent" ? "hsl(52, 86%, 66%)" : color,
      textColor: "#0f172a",
    };
  }
  const color = probColor(0.98, 0);
  return {
    label: "Probability Tokens",
    color: color === "transparent" ? "hsl(102, 86%, 66%)" : color,
    textColor: "#0f172a",
  };
}

function tokenColorLegendInlineHtml({
  wrapperClass,
  chipClass,
  textClass,
  includeEquals = false,
}) {
  const legend = tokenColorLegend();
  if (!legend) return "";
  return `
    <span class="${esc(wrapperClass)}">
      <span class="${esc(chipClass)}" style="background:${legend.color};color:${legend.textColor};">tok</span>
      <span class="${esc(textClass)}">${includeEquals ? "= " : ""}${esc(legend.label)}</span>
    </span>
  `;
}

function tokenColorLegendHtml() {
  return tokenColorLegendInlineHtml({
    wrapperClass: "alt-legend",
    chipClass: "alt-legend-chip",
    textClass: "alt-legend-text",
    includeEquals: true,
  });
}

function tokenColorHeaderLegendHtml() {
  return tokenColorLegendInlineHtml({
    wrapperClass: "trajectory-token-legend",
    chipClass: "trajectory-token-legend-chip",
    textClass: "trajectory-token-legend-text",
    includeEquals: true,
  });
}

function renderTokenColorHeaderLegend(node) {
  if (!node) return;
  const html = tokenColorHeaderLegendHtml();
  if (!html) {
    node.classList.add("hidden");
    node.innerHTML = "";
    return;
  }
  node.classList.remove("hidden");
  node.innerHTML = html;
}

function formatProbabilityPercent(probability) {
  const p = Math.max(0, Math.min(1, Number(probability || 0)));
  const rounded = Math.round((p * 100) * 100) / 100;
  if (Number.isInteger(rounded)) return `${rounded}%`;
  return `${rounded.toFixed(2)}%`;
}

function tokenAlternatives(chip) {
  if (Array.isArray(chip.alternatives)) return chip.alternatives;
  const store = chip._altStore;
  const count = Number(chip._altCount || 0);
  const start = Number(chip._altStart || 0);
  if (!store || count <= 0) return [];
  const result = [];
  for (let index = 0; index < count; index += 1) {
    const itemIndex = start + index;
    if (itemIndex >= store.topIds.length) break;
    result.push({
      token: store.tokenList[store.topIds[itemIndex]] || "",
      probability: Number(store.topProbs[itemIndex] || 0),
    });
  }
  chip.alternatives = result;
  chip._altStore = null;
  chip._altStart = 0;
  chip._altCount = 0;
  return result;
}

function tooltipHtml(chip) {
  const selectedToken = String(chip.token || "");
  let selectedProb = Number(chip.probability || 0);
  const alternatives = tokenAlternatives(chip);

  // Fallback: If selected token has 0 probability, try to find a non-zero version in alternatives
  if (selectedProb === 0) {
    let match = alternatives.find(a => String(a.token || "") === selectedToken && Number(a.probability || 0) > 0);

    // Trimmed fallback
    if (!match) {
      match = alternatives.find(a => String(a.token || "").trim() === selectedToken.trim() && Number(a.probability || 0) > 0);
    }

    if (match) {
      selectedProb = Number(match.probability);
    }
  }

  const dedup = [];
  const seen = new Set();
  dedup.push({ token: selectedToken, probability: selectedProb, selected: true });
  seen.add(selectedToken);
  for (const alt of alternatives) {
    const token = String(alt.token || "");
    const prob = Number(alt.probability || 0);

    if (token === selectedToken && Math.abs(prob - selectedProb) < 1e-9) continue;
    if (seen.has(token)) continue;
    seen.add(token);
    dedup.push({ token, probability: prob, selected: false });
  }
  const rows = dedup.slice(0, 5).map((alt) => {
    const prob = Number(alt.probability || 0);
    const selectedClass = alt.selected ? " selected" : "";
    const fill = `${Math.round(Math.max(0, Math.min(1, prob)) * 100)}%`;
    const swatchColor = alt.selected ? "var(--accent-soft)" : probColor(prob, null);
    return `
      <div class="alt-row${selectedClass}" style="--fill:${fill};">
        <span class="alt-swatch" style="background:${swatchColor};"></span>
        <span class="alt-token">${esc(alt.token)}</span>
        <span class="alt-prob">${formatProbabilityPercent(prob)}</span>
      </div>
    `;
  }).join("");
  const legendHtml = tokenColorLegendHtml();
  return `
    <div class="alt-grid">${rows || "<div class='alt-row'><span class='alt-swatch'></span><span class='alt-token'>no alternatives</span><span class='alt-prob'>-</span></div>"}</div>
    ${legendHtml}
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
  const sidebarControlsNode = byId("sidebar-controls");
  const topControlsNode = byId("top-controls");
  if (!sidebarControlsNode && !topControlsNode) return;
  const reportData = activeData();
  const clusterMode = String(reportData.cluster_mode || "").trim();
  const modeButton = (value, label) => (`<button class="mode-btn${state.colorMode === value ? " active" : ""}" data-mode="${value}">${label}</button>`);
  const entropyModeButton = (value, label) => (`<button class="mode-btn${state.entropyMode === value ? " active" : ""}" data-entropy-mode="${value}">${label}</button>`);
  const entropySubmodeRow = state.colorMode === "entropy"
    ? `<div class="mode-row"><span class="muted">Entropy Style:</span>${entropyModeButton("continuous", "Continuous")}${entropyModeButton("80_20", "80/20")}${entropyModeButton("90_10", "90/10")}${entropyModeButton("95_5", "95/5")}</div>`
    : "";
  const controlsHtml = `
    <h3>Controls</h3>
    ${clusterMode ? `<div class="muted">cluster mode: ${esc(clusterMode)}</div>` : ""}
    <div class="mode-row">
      <span class="muted">Token Coloring:</span>
      ${modeButton("none", "No Color")}
      ${modeButton("hybrid", "Probability")}
      ${modeButton("entropy", "Entropy")}
    </div>
    ${entropySubmodeRow}
  `;
  if (sidebarControlsNode) sidebarControlsNode.innerHTML = controlsHtml;
  if (topControlsNode) topControlsNode.innerHTML = controlsHtml;
  const bindNodes = [sidebarControlsNode, topControlsNode].filter(Boolean);
  for (const node of bindNodes) {
    for (const button of node.querySelectorAll("[data-mode]")) button.onclick = () => {
      state.colorMode = String(button.getAttribute("data-mode") || "hybrid");
      hideTooltip(); renderMeta(); renderTimeline(); renderFinalAnswer();
    };
    for (const button of node.querySelectorAll("[data-entropy-mode]")) button.onclick = () => {
      state.entropyMode = String(button.getAttribute("data-entropy-mode") || "95_5");
      hideTooltip(); renderMeta(); renderTimeline(); renderFinalAnswer();
    };
  }
}

function trajectoryOverlayElements() {
  const overlay = byId("trajectory-overlay");
  if (!overlay) return null;
  return {
    overlay,
    panel: byId("trajectory-overlay-panel"),
    title: overlay.querySelector("[data-trajectory-overlay-title]"),
    body: overlay.querySelector("[data-trajectory-overlay-body]"),
    closeButton: overlay.querySelector("[data-trajectory-overlay-close]"),
  };
}

function closeTrajectoryOverlay() {
  const refs = trajectoryOverlayElements();
  if (!refs) return;
  refs.overlay.classList.add("hidden");
  refs.overlay.setAttribute("aria-hidden", "true");
  if (refs.body) refs.body.innerHTML = "";
  document.body.classList.remove("trajectory-overlay-open");
}

function showTrajectoryOverlayShell() {
  const refs = trajectoryOverlayElements();
  if (!refs || !refs.title || !refs.body) return null;
  refs.body.innerHTML = "";
  refs.overlay.classList.remove("hidden");
  refs.overlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("trajectory-overlay-open");
  return refs;
}

function openTrajectoryOverlay({
  outputKey,
  trajectorySource,
  decodedTrajectoryTokens,
  trajectoryTokenCount,
  trajectoryText,
}) {
  const refs = showTrajectoryOverlayShell();
  if (!refs) return;
  const tokenCount = Number(trajectoryTokenCount || 0);
  refs.title.textContent = `Trajectory (${tokenCount.toLocaleString()} tokens shown)`;
  if (tokenCount > 0) {
    refs.body.innerHTML = "<div data-trajectory-overlay-host></div>";
    const hostNode = refs.body.querySelector("[data-trajectory-overlay-host]");
    if (decodedTrajectoryTokens > 0 && hostNode) {
      renderTrajectoryTokenStripOnScroll({
        outputKey: `${String(outputKey || "__none__")}-overlay`,
        source: trajectorySource,
        target: hostNode,
        progressNode: null,
      });
    } else {
      refs.body.innerHTML += `<pre class="trajectory-pre">${esc(trajectoryText || "No trajectory text captured.")}</pre>`;
    }
    return;
  }
  refs.body.innerHTML = `<pre class="trajectory-pre">${esc(trajectoryText || "No trajectory text captured.")}</pre>`;
}

function openFinalAnswerOverlay({ answerMarkdown }) {
  const refs = showTrajectoryOverlayShell();
  if (!refs) return;
  refs.title.textContent = "Final Answer";
  const answerContainer = document.createElement("div");
  answerContainer.className = "exec-md final-answer-md";
  refs.body.appendChild(answerContainer);
  renderMarkdownPanelContent({
    container: answerContainer,
    markdown: String(answerMarkdown || ""),
    emptyMessage: "No final answer captured.",
  });
}

function initTrajectoryOverlay() {
  const refs = trajectoryOverlayElements();
  if (!refs || refs.overlay.dataset.boundOverlay === "1") return;
  refs.overlay.dataset.boundOverlay = "1";
  if (refs.closeButton) refs.closeButton.onclick = () => closeTrajectoryOverlay();
  refs.overlay.addEventListener("click", (event) => {
    if (event.target === refs.overlay) closeTrajectoryOverlay();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeTrajectoryOverlay();
  });
}

function setTopControlsOpen(open) {
  const button = byId("top-settings-toggle");
  const popover = byId("top-controls-popover");
  if (!button || !popover) return;
  popover.classList.toggle("hidden", !open);
  popover.setAttribute("aria-hidden", open ? "false" : "true");
  button.classList.toggle("active", open);
  button.setAttribute("aria-expanded", open ? "true" : "false");
}

function initTopControls() {
  const wrap = byId("top-controls-wrap");
  const button = byId("top-settings-toggle");
  const popover = byId("top-controls-popover");
  if (!wrap || !button || !popover || wrap.dataset.boundTopControls === "1") return;
  wrap.dataset.boundTopControls = "1";
  setTopControlsOpen(false);
  button.onclick = (event) => {
    const open = popover.classList.contains("hidden");
    setTopControlsOpen(open);
    event.stopPropagation();
  };
  popover.addEventListener("pointerdown", (event) => {
    event.stopPropagation();
  });
  popover.addEventListener("click", (event) => {
    event.stopPropagation();
  });
  document.addEventListener("click", (event) => {
    const path = typeof event.composedPath === "function" ? event.composedPath() : [];
    if (path.includes(wrap)) return;
    if (event.target instanceof Node && wrap.contains(event.target)) return;
    setTopControlsOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setTopControlsOpen(false);
  });
}

let markdownRenderer = null;

function parseInlineMarkdown(raw) {
  let html = esc(raw);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function fallbackMarkdownToHtml(markdown) {
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
  return html;
}

function splitByCodeFences(markdown) {
  const source = String(markdown || "");
  const parts = source.split(/(```[\s\S]*?```)/g);
  return parts.map((value, index) => ({ value, isCode: index % 2 === 1 }));
}

function normalizeIndentedBracketMathDelimiters(markdown) {
  const parts = splitByCodeFences(markdown);
  const normalized = parts.map((part) => {
    if (part.isCode) return part.value;
    let next = part.value;
    next = next.replace(/^[ \t]{4,}(\\\[)\s*$/gm, "$1");
    next = next.replace(/^[ \t]{4,}(\\\])\s*$/gm, "$1");
    // Treat standalone bracket lines as display-math delimiters too.
    next = next.replace(/^[ \t]*\[\s*$/gm, "\\[");
    next = next.replace(/^[ \t]*\]\s*$/gm, "\\]");
    // markdown-it-texmath can miss \\[...\\] inside list blocks; normalize to $$...$$.
    next = next.replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_match, expression) => (
      `$$\n${String(expression || "").trim()}\n$$`
    ));
    return next;
  });
  return normalized.join("");
}

function createMarkdownRenderer() {
  if (typeof window.markdownit !== "function") return fallbackMarkdownToHtml;
  const renderer = window.markdownit({
    html: false,
    linkify: true,
    breaks: true,
    typographer: true,
  });
  // Keep fenced code blocks, but treat 4-space indents as normal text.
  renderer.disable(["code"]);
  if (typeof window.texmath === "function" && window.katex) {
    renderer.use(window.texmath, {
      engine: window.katex,
      delimiters: ["dollars", "brackets"],
      outerSpace: false,
      katexOptions: {
        throwOnError: false,
        strict: "ignore",
        trust: false,
      },
    });
  }
  return (markdown) => renderer.render(String(markdown || ""));
}

function ensureMarkdownRenderer() {
  if (markdownRenderer) return markdownRenderer;
  markdownRenderer = createMarkdownRenderer();
  return markdownRenderer;
}

function markdownToHtml(markdown) {
  const normalized = normalizeIndentedBracketMathDelimiters(markdown);
  const rendered = ensureMarkdownRenderer()(normalized);
  const html = String(rendered || "").trim();
  return html || "<p><em>No text captured.</em></p>";
}

function renderMarkdownPanelContent({ container, markdown, emptyMessage }) {
  if (!container) return;
  const text = String(markdown || "").trim();
  if (!text) {
    container.innerHTML = `<div class="muted">${esc(emptyMessage)}</div>`;
    return;
  }
  container.innerHTML = markdownToHtml(text);
}

function stripSingleParagraphWrapper(html) {
  const text = String(html || "").trim();
  const match = text.match(/^<p>([\s\S]*)<\/p>$/i);
  if (match) return String(match[1] || "").trim();
  return text;
}

function renderMarkdownInlineContent({ container, markdown, emptyMessage }) {
  if (!container) return;
  const text = String(markdown || "").trim();
  if (!text) {
    container.textContent = String(emptyMessage || "");
    return;
  }
  const rendered = markdownToHtml(text);
  container.innerHTML = stripSingleParagraphWrapper(rendered);
}

function buildTokenChipNode(chip) {
  const node = document.createElement("span");
  node.className = "token-chip";
  node.textContent = chip.token || "";
  const synthetic = Boolean(chip.synthetic);
  const background = synthetic ? "transparent" : probColor(chip.probability || 0, chip.entropy || 0);
  node.style.background = background;
  node.style.color = state.colorMode === "none" || background === "transparent" ? "#dbeafe" : "#0f172a";
  if (!synthetic && state.colorMode !== "none") {
    node.onmouseenter = (event) => showTooltip(event, chip);
    node.onmousemove = (event) => showTooltip(event, chip);
    node.onmouseleave = hideTooltip;
  } else {
    node.style.cursor = "text";
  }
  return node;
}

function buildTokenStrip(tokens) {
  const strip = document.createElement("div");
  strip.className = "token-strip";
  for (const chip of (tokens || [])) strip.appendChild(buildTokenChipNode(chip));
  return strip;
}

function renderTrajectoryTokenStripOnScroll({
  outputKey,
  source,
  target,
  progressNode,
}) {
  const ids = source && source.ids ? source.ids : new Uint32Array();
  const probs = source && source.probs ? source.probs : new Float32Array();
  const entropies = source && source.entropies ? source.entropies : new Float32Array();
  const topCounts = source && source.topCounts ? source.topCounts : null;
  const altStore = (source && source.topIds && source.topProbs && source.tokenList)
    ? {
      topIds: source.topIds,
      topProbs: source.topProbs,
      tokenList: source.tokenList,
    }
    : null;
  const total = Number(ids.length);
  const jobId = `${Date.now()}-${Math.random()}`;
  const key = String(outputKey || "__none__");
  state.trajectoryRenderJobsByOutput[key] = jobId;
  target.innerHTML = "";
  const strip = document.createElement("div");
  strip.className = "trajectory-pre token-strip";
  target.appendChild(strip);
  let topOffset = 0;
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < total; index += 1) {
    if (state.trajectoryRenderJobsByOutput[key] !== jobId) return;
    const chip = {
      token: source.tokenList[ids[index]] || "",
      probability: Number(probs[index] || 0),
      entropy: Number(entropies[index] || 0),
      alternatives: null,
    };
    if (altStore && topCounts) {
      const count = Number(topCounts[index] || 0);
      chip._altStore = altStore;
      chip._altStart = topOffset;
      chip._altCount = count;
      topOffset += count;
    }
    fragment.appendChild(buildTokenChipNode(chip));
  }
  strip.appendChild(fragment);
  if (progressNode && progressNode.parentNode) progressNode.remove();
  activeTrajectoryCleanup = () => {};
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
  const fullTokens = getEntryFullTokens(entry);
  const rolloutTokens = getEntryRolloutTokens(entry);
  const steerOnlyTokens = getEntryTokens(entry);
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
  const strip = buildTokenStrip(getEntryTokens(entry));
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

function buildStepCard(step, scale) {
  ensureStepState(step);
  const expanded = expandedState();
  const key = String(step.step_index);
  const isExpanded = Boolean(expanded[key]);
  const card = document.createElement("section");
  card.className = "panel step-card";
  const stepNumber = Number(step.step_index) + 1;
  const clusterRatio = Math.max(0, Math.min(1, Number(step.cluster_count || 0) / scale.clusterMax));
  const uniqueRatio = Math.max(0, Math.min(1, uniqueCandidateCount(step) / scale.uniqueMax));
  const clusterHue = Math.round(228 - (clusterRatio * 148));
  const clusterBorderAlpha = 0.34 + (clusterRatio * 0.62);
  const uniqueHue = Math.round(72 - (uniqueRatio * 60));
  const uniqueBorderAlpha = 0.34 + (uniqueRatio * 0.62);
  const clusterStyle = `
    border-color: hsla(${clusterHue}, 92%, 68%, ${clusterBorderAlpha});
    color: hsla(${clusterHue}, 88%, 78%, 0.98);
  `;
  const uniqueStyle = `
    border-color: hsla(${uniqueHue}, 92%, 66%, ${uniqueBorderAlpha});
    color: hsla(${uniqueHue}, 90%, 78%, 0.98);
  `;
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
  if (isExpanded) {
    const clusterHeader = document.createElement("h3");
    clusterHeader.textContent = "Candidate Clusters";
    body.appendChild(clusterHeader);
    const clusterGrid = document.createElement("div");
    clusterGrid.className = "cluster-grid";
    renderClusterRows(step, clusterGrid);
    body.appendChild(clusterGrid);
  }
  card.appendChild(body);
  return card;
}

function timelineScale(views) {
  const clusterMax = Math.max(
    1,
    ...(views.map((view) => Number(view.cluster_count || 0)))
  );
  const uniqueMax = Math.max(
    1,
    ...(views.map((view) => (view.clusters || []).reduce((sum, cluster) => sum + Number(cluster.unique_count || 0), 0)))
  );
  return { clusterMax, uniqueMax };
}

function renderTimeline() {
  const timeline = byId("timeline");
  timeline.innerHTML = "";
  if (state.activeView !== "report") return;
  const output = selectedOutput();
  const currentId = outputId(output);
  const loadingCurrent = currentId && state.loadingOutputId === currentId;
  const loadError = currentId ? String(state.outputLoadErrors[currentId] || "") : "";
  if (loadingCurrent) {
    timeline.innerHTML = "<section class='panel hero'><div class='muted'>Loading output data...</div></section>";
    return;
  }
  if (loadError) {
    timeline.innerHTML = `<section class='panel hero'><div class='muted'>${esc(loadError)}</div></section>`;
    return;
  }
  if (!isOutputReady(output)) {
    timeline.innerHTML = "<section class='panel hero'><div class='muted'>Select an output to load report data.</div></section>";
    return;
  }
  const views = stepViews(activeData());
  const scale = timelineScale(views);
  for (const step of views) timeline.appendChild(buildStepCard(step, scale));
}
function renderFinalAnswer() {
  disposeActiveTrajectoryLoader();
  const panel = byId("final-answer");
  panel.innerHTML = "";
  if (state.activeView !== "report") return;
  const output = selectedOutput();
  const currentId = outputId(output);
  const loadingCurrent = currentId && state.loadingOutputId === currentId;
  const loadError = currentId ? String(state.outputLoadErrors[currentId] || "") : "";
  if (loadingCurrent) {
    panel.className = "side-column";
    panel.innerHTML = "<section class='panel hero'><h2>Report</h2><div class='muted'>Loading output data...</div></section>";
    return;
  }
  if (loadError) {
    panel.className = "side-column";
    panel.innerHTML = `<section class='panel hero'><h2>Report</h2><div class='muted'>${esc(loadError)}</div></section>`;
    return;
  }
  if (!isOutputReady(output)) {
    panel.className = "side-column";
    panel.innerHTML = "<section class='panel hero'><h2>Report</h2><div class='muted'>Select an output to load report data.</div></section>";
    return;
  }
  const reportData = activeData();
  const prompt = String(((reportData.config || {}).prompt || "")).trim();
  const trajectory = String(reportData.final_text || "").trim();
  const answer = String(reportData.final_answer_text || "").trim();
  const trajectorySource = getTrajectoryTokenSource(reportData);
  const decodedTrajectoryTokens = Number((trajectorySource.ids || []).length);
  const trajectoryTokenCount = Math.max(
    decodedTrajectoryTokens,
    Number(reportData.trajectory_token_count || 0),
  );
  panel.className = "side-column";
  panel.innerHTML = `
    <section class="panel hero report-section prompt-section">
      <h2 class="report-section-title">Prompt</h2>
      <div class="report-section-scroll">
        <div class="exec-md prompt-md" data-prompt-text></div>
      </div>
    </section>
    <section class="panel hero report-section trajectory-section" data-trajectory-panel>
      <h2 class="report-section-title">
        <span class="trajectory-title-row">
          <span>Trajectory <span class="report-section-meta" data-trajectory-count></span></span>
          <span class="hidden" data-trajectory-token-legend></span>
        </span>
        <button type="button" class="trajectory-fullscreen-btn" data-trajectory-fullscreen aria-label="Open trajectory fullscreen">
          <span class="material-symbols-rounded" aria-hidden="true">open_in_full</span>
        </button>
      </h2>
      <div class="report-section-scroll" data-trajectory-scroll></div>
    </section>
    <section class="panel hero report-section final-section" data-final-answer-panel>
      <h2 class="report-section-title">
        <span>Final Answer</span>
        <button type="button" class="trajectory-fullscreen-btn" data-final-answer-fullscreen aria-label="Open final answer fullscreen">
          <span class="material-symbols-rounded" aria-hidden="true">open_in_full</span>
        </button>
      </h2>
      <div class="report-section-scroll" data-final-answer-scroll></div>
    </section>
  `;
  const promptText = panel.querySelector("[data-prompt-text]");
  if (promptText) renderMarkdownPanelContent({ container: promptText, markdown: prompt, emptyMessage: "No prompt captured." });
  const trajectoryCountNode = panel.querySelector("[data-trajectory-count]");
  if (trajectoryCountNode) trajectoryCountNode.textContent = `${trajectoryTokenCount.toLocaleString()} tokens shown`;
  const trajectoryLegendNode = panel.querySelector("[data-trajectory-token-legend]");
  renderTokenColorHeaderLegend(trajectoryLegendNode);
  const fullscreenButton = panel.querySelector("[data-trajectory-fullscreen]");
  if (fullscreenButton) {
    fullscreenButton.onclick = () => openTrajectoryOverlay({
      outputKey: currentId,
      trajectorySource,
      decodedTrajectoryTokens,
      trajectoryTokenCount,
      trajectoryText: trajectory,
    });
  }
  const finalFullscreenButton = panel.querySelector("[data-final-answer-fullscreen]");
  if (finalFullscreenButton) {
    finalFullscreenButton.onclick = () => openFinalAnswerOverlay({
      answerMarkdown: answer,
    });
  }
  const trajectoryPanel = panel.querySelector("[data-trajectory-panel]"); const finalAnswerPanel = panel.querySelector("[data-final-answer-panel]"); if (!trajectoryPanel || !finalAnswerPanel) return;
  const trajectoryScroll = trajectoryPanel.querySelector("[data-trajectory-scroll]");
  const finalAnswerScroll = finalAnswerPanel.querySelector("[data-final-answer-scroll]");
  if (trajectoryTokenCount > 0) {
    if (trajectoryScroll) trajectoryScroll.innerHTML = "<div data-trajectory-token-host></div>";
    const hostNode = trajectoryScroll ? trajectoryScroll.querySelector("[data-trajectory-token-host]") : null;
    if (decodedTrajectoryTokens > 0 && hostNode) {
      renderTrajectoryTokenStripOnScroll({
        outputKey: currentId,
        source: trajectorySource,
        target: hostNode,
        progressNode: null,
      });
    } else {
      if (trajectoryScroll) trajectoryScroll.innerHTML += `<pre class="trajectory-pre">${esc(trajectory || "No trajectory text captured.")}</pre>`;
    }
  } else {
    if (trajectoryScroll) trajectoryScroll.innerHTML += `<pre class="trajectory-pre">${esc(trajectory || "No trajectory text captured.")}</pre>`;
  }
  const answerContainer = document.createElement("div");
  answerContainer.className = "exec-md final-answer-md";
  if (finalAnswerScroll) finalAnswerScroll.appendChild(answerContainer);
  renderMarkdownPanelContent({ container: answerContainer, markdown: answer, emptyMessage: "No final answer captured." });
  syncColumnDividerHeight();
}
function setSidebarOpen({ openButton, closeButton, sidebar, scrim, open, focusTarget }) { document.body.classList.toggle("sidebar-open", open); openButton.setAttribute("aria-expanded", open ? "true" : "false"); sidebar.setAttribute("aria-hidden", open ? "false" : "true"); scrim.setAttribute("aria-hidden", open ? "false" : "true"); if (focusTarget) focusTarget.focus({ preventScroll: true }); }

async function ensureOutputLoaded(outputIdValue) {
  const targetId = String(outputIdValue || "");
  const output = outputs().find((item) => outputId(item) === targetId);
  if (!output) return {};
  const alreadyLoaded = cachedReport(output);
  if (alreadyLoaded) return alreadyLoaded;
  if (output._reportPromise) return output._reportPromise;
  const reportFile = String(output.report_file || "").trim();
  if (!reportFile) {
    reportCacheByOutput[targetId] = {};
    return reportCacheByOutput[targetId];
  }
  output._reportPromise = (async () => {
    const response = await fetch(reportFile);
    if (!response.ok) {
      throw new Error(`Failed to load output report: ${reportFile}`);
    }
    const report = await response.json();
    output.report = report;
    reportCacheByOutput[targetId] = report;
    return report;
  })();
  try {
    return await output._reportPromise;
  } finally {
    output._reportPromise = null;
  }
}

async function openOutput(outputIdValue) {
  const targetId = String(outputIdValue || "");
  closeTrajectoryOverlay();
  disposeActiveTrajectoryLoader();
  state.trajectoryRenderJobsByOutput = {};
  state.selectedOutputId = targetId;
  state.activeView = "report";
  state.loadingOutputId = targetId;
  state.outputLoadErrors[targetId] = "";
  renderApp();
  try {
    await ensureOutputLoaded(targetId);
  } catch (error) {
    const message = (error && error.message) ? error.message : String(error);
    state.outputLoadErrors[targetId] = message;
  } finally {
    if (state.loadingOutputId === targetId) state.loadingOutputId = null;
    renderApp();
  }
}

function outputLabel(output) {
  const prompt = String(output.prompt || "").trim();
  if (prompt) return prompt;
  const fallback = String(output.label || "").trim();
  return fallback || "(no prompt)";
}

function promptLookupKey(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function promptTitleMap() {
  const raw = normalizedBundle && normalizedBundle.prompt_titles;
  if (!raw || typeof raw !== "object") return {};
  return raw;
}

function decoratedPromptTitle({ title, prompt }) {
  const baseTitle = String(title || "").trim();
  if (!baseTitle) return "";
  if (baseTitle.includes("🧮") || baseTitle.includes("🍓") || baseTitle.includes("🚗")
    || baseTitle.includes("⚙️") || baseTitle.includes("🏏") || baseTitle.includes("📝")
    || baseTitle.includes("🔢") || baseTitle.includes("🌍") || baseTitle.includes("👋")) {
    return baseTitle;
  }
  const haystack = `${baseTitle} ${String(prompt || "")}`.toLowerCase();
  const rules = [
    { pattern: /\baime\b|math|functions|divisors|number of/, emoji: "🧮" },
    { pattern: /strawberry|count the r/, emoji: "🍓" },
    { pattern: /car wash|drive or walk|wash my car/, emoji: "🚗" },
    { pattern: /machines|widgets/, emoji: "⚙️" },
    { pattern: /bat[- ]and[- ]ball|bat and a ball/, emoji: "🏏" },
    { pattern: /word count|self-referential|how many words/, emoji: "📝" },
    { pattern: /9\.9|9\.11|comparison|which number is greater/, emoji: "🔢" },
    { pattern: /earth|physics|tunnel/, emoji: "🌍" },
    { pattern: /\bhello\b/, emoji: "👋" },
  ];
  for (const rule of rules) {
    if (rule.pattern.test(haystack)) return `${rule.emoji} ${baseTitle}`;
  }
  return baseTitle;
}

function outputTitle(output) {
  const prompt = String(output.prompt || "").trim();
  const key = promptLookupKey(prompt);
  if (key.toLowerCase() === "hello") return "👋 Hello World";
  const title = String(promptTitleMap()[key] || "").trim();
  if (title) return decoratedPromptTitle({ title, prompt });
  const fallback = String(output.label || "").trim();
  if (fallback && fallback !== prompt) return decoratedPromptTitle({ title: fallback, prompt });
  const promptFallback = prompt || "(no prompt)";
  return decoratedPromptTitle({ title: promptFallback, prompt });
}

function renderSidebarOutputs() {
  const container = byId("sidebar-output-list");
  if (!container) return;
  const currentId = String((selectedOutput() || {}).id || "");
  const outputButtons = outputs().map((output) => {
    const outputId = String(output.id || "");
    const label = outputTitle(output);
    const activeClass = state.activeView === "report" && outputId === currentId ? " active" : "";
    return `<button type="button" class="sidebar-output-btn${activeClass}" data-select-output="${esc(outputId)}" aria-label="${esc(label)}"><span class="sidebar-output-label" data-sidebar-output-label="${esc(outputId)}"></span></button>`;
  }).join("");
  container.innerHTML = `<section class="meta-block"><h3>Outputs to Inspect</h3><div class="output-list">${outputButtons || "<div class='muted'>No outputs found.</div>"}</div></section>`;
  for (const labelNode of container.querySelectorAll("[data-sidebar-output-label]")) {
    const targetId = String(labelNode.getAttribute("data-sidebar-output-label") || "");
    const output = outputs().find((item) => String(item.id || "") === targetId);
    if (!output) continue;
    renderMarkdownInlineContent({
      container: labelNode,
      markdown: outputTitle(output),
      emptyMessage: "(no prompt)",
    });
  }
  for (const button of container.querySelectorAll("[data-select-output]")) {
    button.onclick = () => {
      const nextId = String(button.getAttribute("data-select-output") || "");
      void openOutput(nextId);
    };
  }
  for (const homeButton of document.querySelectorAll("[data-show-home]")) {
    homeButton.onclick = () => { state.activeView = "home"; renderApp(); };
    homeButton.classList.toggle("active", state.activeView === "home");
  }
}

function renderHome() {
  const homePanel = byId("home-view");
  const algorithmText = String(normalizedBundle.algorithm_overview || "").trim()
    || "The analysis process branches steers, clusters candidate behavior, selects a trajectory, and tracks token-level uncertainty metrics.";
  const cards = outputs().map((output) => {
    const outputId = String(output.id || "");
    return `<button type="button" class="panel home-output-card" data-open-output="${esc(outputId)}" aria-label="Inspect output ${esc(outputTitle(output))}"><h2 class="home-output-title" data-home-output-title="${esc(outputId)}"></h2><div class="home-output-prompt muted" data-home-output-prompt="${esc(outputId)}"></div></button>`;
  }).join("");
  homePanel.innerHTML = `
    <h1>Steer Branching Explorer</h1>
    <section class="meta-block pipeline-block">
      <h3>Generation Pipeline</h3>
      <div class="muted">${esc(algorithmText)}</div>
      <ol class="pipeline-list">
        <li><strong>Prompt intake and branch setup:</strong> each run starts from a single prompt and expands into many candidate steer continuations per step using the configured branch factor.</li>
        <li><strong>Candidate clustering:</strong> per-step candidates are grouped into behavior clusters so semantically similar steers collapse into a small, inspectable set of modes.</li>
        <li><strong>Trajectory selection:</strong> one candidate path is chosen at each step, producing the committed steer chain that determines what execution block is generated next.</li>
        <li><strong>Execution extraction:</strong> selected <code>&lt;steer&gt;</code> and downstream <code>&lt;exec&gt;</code> content are aligned so you can inspect chosen reasoning and final surfaced answer text separately.</li>
        <li><strong>Token-level uncertainty audit:</strong> rollout token probabilities, entropy percentiles, and hover alternatives are preserved to expose where generation was stable versus uncertain.</li>
      </ol>
      <div class="muted">Use the timeline to inspect decision points step-by-step, and use the right column to compare prompt, full trajectory, and final answer rendering with math support.</div>
    </section>
    <section class="home-grid">${cards || "<div class='muted'>No outputs found in this bundle.</div>"}</section>
  `;
  for (const titleNode of homePanel.querySelectorAll("[data-home-output-title]")) {
    const targetId = String(titleNode.getAttribute("data-home-output-title") || "");
    const output = outputs().find((item) => String(item.id || "") === targetId);
    if (!output) continue;
    renderMarkdownInlineContent({
      container: titleNode,
      markdown: outputTitle(output),
      emptyMessage: "(no prompt)",
    });
  }
  for (const promptNode of homePanel.querySelectorAll("[data-home-output-prompt]")) {
    const targetId = String(promptNode.getAttribute("data-home-output-prompt") || "");
    const output = outputs().find((item) => String(item.id || "") === targetId);
    if (!output) continue;
    renderMarkdownInlineContent({
      container: promptNode,
      markdown: String(output.prompt || "").trim(),
      emptyMessage: "(no prompt)",
    });
  }
  for (const button of homePanel.querySelectorAll("[data-open-output]")) {
    button.onclick = () => {
      const nextId = String(button.getAttribute("data-open-output") || "");
      void openOutput(nextId);
    };
  }
}

function applyActiveView() {
  const showHome = state.activeView === "home";
  const homePanel = byId("home-view");
  const reportPanel = byId("report-view");
  if (showHome) closeTrajectoryOverlay();
  if (homePanel) homePanel.classList.toggle("hidden-block", !showHome);
  if (reportPanel) reportPanel.classList.toggle("hidden-block", showHome);
}

function syncColumnDividerHeight() {
  const divider = byId("column-divider");
  const leftColumn = byId("final-answer");
  if (!divider || !leftColumn) return;
  divider.style.height = `${Math.max(0, leftColumn.offsetHeight)}px`;
}

function applyMainColumnWidth() {
  document.documentElement.style.setProperty("--main-col-width", `${state.mainColWidth}%`);
  syncColumnDividerHeight();
}

function renderApp() {
  applyMainColumnWidth();
  applyActiveView();
  renderSidebarOutputs();
  renderHome();
  renderMeta();
  renderTimeline();
  renderFinalAnswer();
  syncColumnDividerHeight();
}

function initSidebar() {
  const openButton = document.querySelector("[data-sidebar-open]"); const closeButton = document.querySelector("[data-sidebar-close]"); const sidebar = document.querySelector(".sidebar-panel"); const scrim = byId("sidebar-scrim");
  if (!openButton || !closeButton || !sidebar || !scrim) return;
  const closeSidebar = ({ focus } = { focus: true }) => { setSidebarOpen({ openButton, closeButton, sidebar, scrim, open: false, focusTarget: focus ? openButton : null }); try { window.localStorage.setItem("report_sidebar_open", "0"); } catch (_error) { } };
  const openSidebar = ({ focus } = { focus: true }) => { setSidebarOpen({ openButton, closeButton, sidebar, scrim, open: true, focusTarget: focus ? closeButton : null }); try { window.localStorage.setItem("report_sidebar_open", "1"); } catch (_error) { } };
  let startOpen = true; try { startOpen = window.localStorage.getItem("report_sidebar_open") !== "0"; } catch (_error) { }
  if (startOpen) openSidebar({ focus: false }); else closeSidebar({ focus: false });
  openButton.onclick = () => openSidebar(); closeButton.onclick = () => closeSidebar(); scrim.onclick = () => closeSidebar();
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) closeSidebar(); });
}

function initColumnDivider() {
  const divider = byId("column-divider");
  const reportView = byId("report-view");
  if (!divider || !reportView || divider.dataset.boundDivider === "1") return;
  divider.dataset.boundDivider = "1";
  let dragging = false;
  const updateFromClientX = (clientX) => {
    const grid = reportView.querySelector(".content-grid");
    if (!grid) return;
    const rect = grid.getBoundingClientRect();
    const dividerRect = divider.getBoundingClientRect();
    const ratio = ((clientX - rect.left - (dividerRect.width / 2)) / rect.width) * 100;
    state.mainColWidth = Math.max(30, Math.min(70, Number(ratio)));
    applyMainColumnWidth();
  };
  const stopDragging = () => {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove("dragging");
    document.body.classList.remove("resizing-columns");
  };
  divider.addEventListener("pointerdown", (event) => {
    dragging = true;
    divider.classList.add("dragging");
    document.body.classList.add("resizing-columns");
    divider.setPointerCapture(event.pointerId);
    updateFromClientX(event.clientX);
    event.preventDefault();
  });
  divider.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    updateFromClientX(event.clientX);
    event.preventDefault();
  });
  divider.addEventListener("pointerup", stopDragging);
  divider.addEventListener("pointercancel", stopDragging);
  window.addEventListener("resize", () => syncColumnDividerHeight(), { passive: true });
}
// Decoders
function decodeFloat32(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function decodeUint32(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return new Uint32Array(bytes.buffer);
}

function decodeUint8(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return bytes;
}

function inflateTokens(compressed, tokenList) {
  const ids = decodeUint32(compressed.ids || "");
  const probs = decodeFloat32(compressed.probs || "");
  const entropies = decodeFloat32(compressed.entropies || "");
  const result = [];

  // Top logprobs decoding
  const topIds = compressed.top_ids ? decodeUint32(compressed.top_ids) : null;
  const topProbs = compressed.top_probs ? decodeFloat32(compressed.top_probs) : null;
  const topCounts = compressed.top_counts ? decodeUint8(compressed.top_counts) : null;
  const altStore = (topCounts && topIds && topProbs)
    ? { topIds, topProbs, tokenList }
    : null;

  let topOffset = 0;

  for (let i = 0; i < ids.length; i++) {
    const token = {
      token: tokenList[ids[i]] || "",
      probability: Number(probs[i] || 0),
      entropy: Number(entropies[i] || 0),
      alternatives: null,
    };

    if (altStore) {
      const count = Number(topCounts[i] || 0);
      token._altStore = altStore;
      token._altStart = topOffset;
      token._altCount = count;
      topOffset += count;
    }

    result.push(token);
  }
  return result;
}

function decodeCompressedTokenSource(compressed, tokenList) {
  const ids = decodeUint32(compressed.ids || "");
  const probs = decodeFloat32(compressed.probs || "");
  const entropies = decodeFloat32(compressed.entropies || "");
  const topIds = compressed.top_ids ? decodeUint32(compressed.top_ids) : null;
  const topProbs = compressed.top_probs ? decodeFloat32(compressed.top_probs) : null;
  const topCounts = compressed.top_counts ? decodeUint8(compressed.top_counts) : null;
  return {
    ids,
    probs,
    entropies,
    topIds,
    topProbs,
    topCounts,
    tokenList: tokenList || [],
  };
}

// Accessors
function getEntryTokens(entry) {
  if (entry._c_tokens) return entry._c_tokens;
  entry._c_tokens = inflateTokens(entry.tokens || {}, activeData().token_list || []);
  return entry._c_tokens;
}

function getEntryFullTokens(entry) {
  if (entry._c_full_tokens) return entry._c_full_tokens;
  entry._c_full_tokens = inflateTokens(entry.full_tokens || {}, activeData().token_list || []);
  return entry._c_full_tokens;
}

function getEntryRolloutTokens(entry) {
  if (entry._c_rollout_tokens) return entry._c_rollout_tokens;
  entry._c_rollout_tokens = inflateTokens(entry.rollout_tokens || {}, activeData().token_list || []);
  return entry._c_rollout_tokens;
}

function getTrajectoryTokens(data) {
  if (data._c_trajectory_tokens) return data._c_trajectory_tokens;
  data._c_trajectory_tokens = inflateTokens(data.trajectory_tokens || {}, data.token_list || []);
  return data._c_trajectory_tokens;
}

function getTrajectoryTokenSource(data) {
  if (data._c_trajectory_source) return data._c_trajectory_source;
  data._c_trajectory_source = decodeCompressedTokenSource(
    data.trajectory_tokens || {},
    data.token_list || [],
  );
  return data._c_trajectory_source;
}

function getRolloutProbabilities(data) {
  if (data._c_rollout_probs) return data._c_rollout_probs;
  const raw = data.rollout_probabilities;
  if (typeof raw === "string") {
    data._c_rollout_probs = decodeFloat32(raw);
    return data._c_rollout_probs;
  }
  return raw || [];
}

function getRolloutEntropies(data) {
  if (data._c_rollout_entropies) return data._c_rollout_entropies;
  const raw = data.rollout_entropies;
  if (typeof raw === "string") {
    data._c_rollout_entropies = decodeFloat32(raw);
    return data._c_rollout_entropies;
  }
  return raw || [];
}

async function bootstrap() {
  initSidebar();
  initTopControls();
  initTrajectoryOverlay();
  initColumnDivider();
  if (!outputs().length) state.activeView = "home";
  renderApp();
  try {
    const response = await fetch("report_data.json");
    if (!response.ok) throw new Error("Failed to load report data");
    rawBundle = await response.json();
    normalizedBundle = normalizeBundle(rawBundle);
    if (!selectedOutput()) {
      state.selectedOutputId = defaultOutputId(normalizedBundle);
    }
    if (!outputs().length) state.activeView = "home";
    renderApp();
  } catch (error) {
    // Keep embedded payload so local-file viewing still works if fetch is blocked.
    console.warn("Report data fetch failed; using embedded payload.", error);
  }
}

bootstrap();
