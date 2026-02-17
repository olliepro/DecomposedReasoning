"""UI style and script blocks for the static steer report."""

STYLE_BLOCK = """
<style>
:root {
  --bg0: #070a15;
  --bg1: #0f1836;
  --bg2: #16294b;
  --panel: rgba(15, 23, 42, 0.76);
  --panel-border: rgba(148, 163, 184, 0.26);
  --text: #e2e8f0;
  --muted: #9fb0ca;
  --accent: #38bdf8;
  --accent2: #22c55e;
  --danger: #fb7185;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; min-height: 100vh; overscroll-behavior: none; background-color: var(--bg0);
  color: var(--text);
  font-family: "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 14% 12%, rgba(56, 189, 248, 0.22), transparent 34%),
    radial-gradient(circle at 80% 18%, rgba(34, 197, 94, 0.14), transparent 34%),
    radial-gradient(circle at 54% 92%, rgba(59, 130, 246, 0.14), transparent 45%),
    linear-gradient(130deg, var(--bg0), var(--bg1) 54%, var(--bg2));
}
main { max-width: 1480px; margin: 0 auto; padding: 24px 16px 16px 16px; display: grid; gap: 16px; }
.workspace-grid { position: relative; }
body.sidebar-open { overflow: hidden; }
.menu-toggle { position: fixed; top: 14px; left: 14px; z-index: 2200; width: 40px; height: 40px; border: 1px solid rgba(148, 163, 184, 0.44); border-radius: 12px; background: rgba(2, 6, 23, 0.88); color: #dbeafe; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; }
.menu-toggle:hover { border-color: rgba(125, 211, 252, 0.72); background: rgba(14, 116, 144, 0.32); }
body.sidebar-open .menu-toggle { opacity: 0; pointer-events: none; }
.sidebar-panel { position: fixed; inset: 0 auto 0 0; width: min(392px, 92vw); height: 100dvh; max-height: 100dvh; z-index: 2300; overflow: hidden; display: flex; flex-direction: column; border-radius: 0 16px 16px 0; transform: translateX(-105%); transition: transform 140ms ease; }
body.sidebar-open .sidebar-panel { transform: translateX(0); }
.sidebar-scrim { position: fixed; inset: 0; z-index: 2250; border: 0; margin: 0; padding: 0; opacity: 0; pointer-events: none; background: rgba(2, 6, 23, 0.62); backdrop-filter: blur(2px); transition: opacity 120ms ease; }
body.sidebar-open .sidebar-scrim { opacity: 1; pointer-events: auto; }
.sidebar-header { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 12px; border-bottom: 1px solid rgba(148, 163, 184, 0.24); }
.sidebar-title { margin: 0; font-size: 13px; font-weight: 700; color: #dbeafe; letter-spacing: 0.02em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sidebar-hamb, .sidebar-close-icon { font-size: 15px; line-height: 1; color: #dbeafe; }
.sidebar-close { width: 32px; height: 32px; border: 1px solid rgba(148, 163, 184, 0.38); border-radius: 10px; background: rgba(2, 6, 23, 0.52); color: #dbeafe; cursor: pointer; padding: 0; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
.sidebar-close:hover { border-color: rgba(251, 113, 133, 0.62); background: rgba(159, 18, 57, 0.22); }
.sidebar-body { padding: 12px; display: grid; gap: 12px; overflow: auto; }
.panel { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 16px; box-shadow: 0 22px 56px rgba(2, 6, 23, 0.45); backdrop-filter: blur(12px); }
.hero { padding: 20px; display: grid; gap: 11px; }
h1 { margin: 0; font-size: 28px; letter-spacing: 0.02em; }
h2 { margin: 0; font-size: 16px; color: #dbeafe; }
h3 { margin: 0; font-size: 13px; color: #bfdbfe; letter-spacing: 0.07em; text-transform: uppercase; }
.muted { color: var(--muted); font-size: 13px; }
#meta { display: grid; gap: 12px; }
.meta-block { border: 1px solid rgba(148, 163, 184, 0.26); border-radius: 12px; background: rgba(2, 6, 23, 0.42); padding: 10px; display: grid; gap: 8px; height: 100%; }
.meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.meta-pill { border: 1px solid rgba(148, 163, 184, 0.33); padding: 5px 9px; border-radius: 999px; font-size: 12px; background: rgba(15, 23, 42, 0.62); }
.mode-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 6px; }
.mode-btn { border: 1px solid rgba(148, 163, 184, 0.38); border-radius: 999px; background: rgba(2, 6, 23, 0.6); color: #bfdbfe; font-size: 11px; padding: 3px 10px; cursor: pointer; }
.mode-btn.active { border-color: rgba(56, 189, 248, 0.72); color: #dbeafe; background: rgba(14, 116, 144, 0.32); }
.content-grid { display: grid; gap: 16px; grid-template-columns: minmax(0, var(--main-col-width, 50%)) minmax(320px, calc(100% - var(--main-col-width, 50%))); align-items: start; }
.timeline {
  position: relative;
  display: grid;
  gap: 14px;
  padding-left: 24px;
}
.timeline::before {
  content: "";
  position: absolute;
  left: 10px;
  top: 0;
  width: 2px;
  height: 100%;
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.6), rgba(34, 197, 94, 0.58));
}
.step-card {
  position: relative;
  padding: 0;
  overflow: hidden;
}
.step-card::before {
  content: "";
  position: absolute;
  left: -16px;
  top: 19px;
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: linear-gradient(135deg, #67e8f9, #22c55e);
  box-shadow: 0 0 0 4px rgba(103, 232, 249, 0.2);
}
.step-header {
  border: 0;
  width: 100%;
  color: inherit;
  text-align: left;
  cursor: pointer;
  background: linear-gradient(90deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.62));
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.step-head-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
.step-index {
  font-size: 12px;
  color: #a5f3fc;
  border: 1px solid rgba(125, 211, 252, 0.42);
  border-radius: 999px;
  padding: 3px 8px;
  white-space: nowrap;
}
.step-steer {
  font-size: 13px;
  color: #dbeafe;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.step-right { display: flex; gap: 8px; align-items: center; }
.stat-pill {
  font-size: 11px;
  color: #bfdbfe;
  border: 1px solid rgba(148, 163, 184, 0.42);
  border-radius: 999px;
  padding: 2px 8px;
  background: rgba(15, 23, 42, 0.62);
}
.step-chevron { color: #93c5fd; font-size: 13px; }
.step-body { padding: 12px 14px 14px 14px; display: grid; gap: 12px; }
.step-body.collapsed { display: none; }
.cluster-grid { display: grid; gap: 7px; }
.cluster-row {
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 11px;
  background: rgba(15, 23, 42, 0.56);
  overflow: hidden;
}
.cluster-row[open] {
  border-color: rgba(52, 211, 153, 0.72);
  box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.36) inset;
}
.cluster-row.top-cluster {
  border-color: rgba(34, 197, 94, 0.9);
  box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.4) inset;
}
.cluster-row.top-cluster[open] {
  border-color: rgba(148, 163, 184, 0.3);
  box-shadow: none;
}
.cluster-row > summary {
  list-style: none;
  cursor: pointer;
  padding: 8px 10px;
  position: relative;
  overflow: hidden;
}
.cluster-row > summary::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--fill, 0%);
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.30), rgba(34, 197, 94, 0.28));
  pointer-events: none;
}
.cluster-row > summary > * {
  position: relative;
  z-index: 1;
}
.cluster-row > summary::-webkit-details-marker { display: none; }
.cluster-row > summary::marker { content: ""; }
.cluster-row:hover { border-color: rgba(125, 211, 252, 0.62); }
.cluster-row.top-cluster:hover { border-color: rgba(34, 197, 94, 0.95); }
.cluster-row.top-cluster[open]:hover { border-color: rgba(125, 211, 252, 0.62); }
.cluster-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
  border: 1px solid rgba(34, 197, 94, 0.54);
  border-radius: 8px;
  padding: 4px 6px;
}
.cluster-name-wrap {
  display: flex;
  align-items: center;
  gap: 7px;
  min-width: 0;
}
.cluster-caret {
  color: #a5f3fc;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 13px;
  font-weight: 800;
  flex: 0 0 auto;
  transition: transform 120ms ease;
}
.cluster-row[open] .cluster-caret { transform: rotate(90deg); }
.cluster-name {
  font-size: 13px;
  color: #dbeafe;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.cluster-body {
  display: grid;
  gap: 8px;
  padding: 8px 10px 10px 10px;
}
.cluster-tags { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.cluster-count,
.cluster-unique {
  font-size: 11px;
  border: 1px solid rgba(96, 165, 250, 0.46);
  border-radius: 999px;
  color: #93c5fd;
  padding: 2px 7px;
}
.cluster-selected {
  font-size: 11px;
  border: 1px solid rgba(34, 197, 94, 0.66);
  border-radius: 999px;
  color: #86efac;
  background: rgba(2, 44, 34, 0.6);
  padding: 2px 7px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.candidate-panel {
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 12px;
  background: rgba(2, 6, 23, 0.55);
  padding: 9px 10px;
  display: grid;
  gap: 8px;
}
.candidate-panel.chosen {
  border-color: rgba(34, 197, 94, 0.62);
  box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.34) inset;
}
.candidate-panel.selected {
  border-color: rgba(34, 197, 94, 0.9);
  box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.44) inset;
}
.chosen-structure {
  border: 1px dashed rgba(148, 163, 184, 0.36);
  border-radius: 10px;
  background: rgba(15, 23, 42, 0.48);
  padding: 8px;
  display: grid;
  gap: 4px;
}
.chosen-tag {
  color: #e2e8f0;
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
.chosen-token-strip {
  margin: 0;
  font-size: 12px;
  line-height: 1.42;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
.chosen-exec-strip.clamped {
  max-height: calc(1.42em * 4);
  overflow: hidden;
}
.chosen-exec-fallback {
  white-space: pre-wrap;
  word-break: break-word;
  color: #dbeafe;
}
.chosen-exec-fallback.clamped {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 4;
  overflow: hidden;
}
.see-more-btn {
  justify-self: start;
  border: 1px solid rgba(148, 163, 184, 0.42);
  border-radius: 8px;
  background: rgba(2, 6, 23, 0.7);
  color: #bfdbfe;
  font-size: 11px;
  padding: 2px 8px;
  cursor: pointer;
}
.see-more-btn:hover {
  border-color: rgba(125, 211, 252, 0.62);
  color: #dbeafe;
}
.candidate-meta { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.candidate-inline { display: flex; align-items: flex-start; gap: 8px; min-width: 0; }
.candidate-inline-strip { flex: 1; min-width: 0; }
.tag {
  border: 1px solid rgba(148, 163, 184, 0.36);
  border-radius: 999px;
  color: #bfdbfe;
  background: rgba(15, 23, 42, 0.73);
  font-size: 11px;
  padding: 2px 7px;
}
.token-strip {
  display: block;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
.token-chip {
  display: inline;
  border: 0;
  padding: 0;
  margin: 0;
  border-radius: 0;
  font-size: 12px;
  font-family: inherit;
  color: #0f172a;
  cursor: pointer;
}
.exec-md {
  margin-top: 7px;
  color: #dbeafe;
  font-size: 12px;
  line-height: 1.45;
}
.exec-md pre {
  margin: 6px 0;
  overflow: auto;
  background: rgba(2, 6, 23, 0.72);
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  padding: 8px;
}
.exec-md code {
  background: rgba(2, 6, 23, 0.62);
  border-radius: 4px;
  padding: 1px 4px;
}
.side-column { position: sticky; top: 16px; height: calc(100vh - 32px); overflow: hidden; display: grid; gap: 12px; grid-template-rows: auto minmax(0, 1fr); }
.side-column [data-trajectory-panel] { min-height: 0; overflow: auto; }
.trajectory-pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: #dbeafe; font-size: 12px; line-height: 1.45; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
.trajectory-pre.clamped { max-height: calc(1.45em * 4); overflow: hidden; }
.tooltip {
  position: fixed;
  z-index: 2100;
  min-width: 130px;
  max-width: 190px;
  pointer-events: none;
  border-radius: 12px;
  padding: 5px;
  border: 1px solid rgba(125, 211, 252, 0.45);
  background: rgba(2, 6, 23, 0.97);
  color: #e2e8f0;
  box-shadow: 0 16px 44px rgba(2, 6, 23, 0.58);
}
.tooltip.hidden { display: none; }
.alt-grid { display: grid; gap: 5px; }
.alt-row {
  position: relative;
  overflow: hidden;
  display: grid;
  grid-template-columns: 7px 1fr auto;
  gap: 7px;
  align-items: center;
  font-size: 11px;
  padding: 3px 6px;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.65);
  border: 1px solid rgba(148, 163, 184, 0.2);
}
.alt-row::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--fill, 0%);
  background: linear-gradient(90deg, rgba(56, 189, 248, 0.28), rgba(34, 197, 94, 0.22));
  pointer-events: none;
}
.alt-row.selected {
  border-color: rgba(34, 197, 94, 0.9);
  box-shadow: 0 0 0 1px rgba(34, 197, 94, 0.5) inset;
}
.alt-row > * {
  position: relative;
  z-index: 1;
}
.alt-swatch { width: 7px; height: 16px; border-radius: 3px; }
.alt-token {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #e2e8f0;
}
.alt-prob { color: #bfdbfe; font-variant-numeric: tabular-nums; }
.hidden-note { font-size: 12px; color: #93c5fd; border: 1px dashed rgba(148, 163, 184, 0.34); border-radius: 9px; padding: 8px; }
.hidden-block { display: none; }
@media (max-width: 960px) {
  main { padding: 16px 10px 16px 10px; }
  .menu-toggle { top: 10px; left: 10px; width: 36px; height: 36px; }
  .sidebar-panel { width: min(94vw, 360px); }
  .content-grid { grid-template-columns: 1fr; }
  .side-column { position: static; height: auto; max-height: none; overflow: visible; }
  .timeline { padding-left: 18px; }
  .timeline::before { left: 7px; }
  .step-card::before { left: -12px; }
  .step-steer { max-width: 44vw; }
}
</style>
""".strip()

SCRIPT_BLOCK = r"""
<script>
const dataNode = document.getElementById("report-data");
const data = dataNode ? JSON.parse(dataNode.textContent || "{}") : {};
const byId = (id) => document.getElementById(id);
const state = { expanded: {}, colorMode: "entropy", entropyMode: "95_5", mainColWidth: 50 };
const clusterBubbleMax = Math.max(1, ...((data.step_views || []).map((step) => Number(step.cluster_count || 0))));
const uniqueBubbleMax = Math.max(1, ...((data.step_views || []).map((step) => (step.clusters || []).reduce((sum, cluster) => sum + Number(cluster.unique_count || 0), 0))));
const rolloutProbabilitySamples = sortedRolloutProbabilities(data.rollout_probabilities || []);
const rolloutEntropySamples = (data.rollout_entropies || []).map((value) => Math.max(0, Number(value))).filter((value) => Number.isFinite(value)).sort((left, right) => left - right);
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
  const p = Math.max(0, Math.min(1, Number(probability || 0)));
  if (!rolloutProbabilitySamples.length) return p;
  const rank = upperBound(rolloutProbabilitySamples, p);
  return Math.max(0, Math.min(1, rank / rolloutProbabilitySamples.length));
}
function colorScore(probability, entropy) {
  if (state.colorMode === "none") return null;
  if (state.colorMode === "entropy") {
    const entropyValue = Math.max(0, Number(entropy || 0));
    const entropyPercentile = rolloutEntropySamples.length ? Math.max(0, Math.min(1, upperBound(rolloutEntropySamples, entropyValue) / rolloutEntropySamples.length)) : (entropyValue / (1 + entropyValue));
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
  const cfg = data.config || {};
  document.documentElement.style.setProperty("--main-col-width", `${state.mainColWidth}%`);
  const pills = [
    `model: ${esc(cfg.model || "")}`,
    `mode: ${esc((cfg.api_mode_config || {}).default_mode || "")}`,
    `branch_factor: ${esc(cfg.branch_factor || "")}`,
    `trajectory_tokens: ${esc(data.trajectory_token_count || 0)}`,
    `cluster_mode: ${esc(data.cluster_mode || "")}`,
  ];
  const modeButton = (value, label) => (`<button class="mode-btn${state.colorMode === value ? " active" : ""}" data-mode="${value}">${label}</button>`);
  const entropyModeButton = (value, label) => (`<button class="mode-btn${state.entropyMode === value ? " active" : ""}" data-entropy-mode="${value}">${label}</button>`);
  const warnings = data.cluster_warnings || [];
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
  const key = String(step.step_index);
  if (!(key in state.expanded)) state.expanded[key] = Number(step.step_index) === 0;
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
  const key = String(step.step_index);
  const expanded = Boolean(state.expanded[key]);
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
      <span class="step-chevron">${expanded ? "▾" : "▸"}</span>
    </div>
  `;
  header.onclick = () => {
    state.expanded[key] = !expanded;
    renderTimeline();
  };
  card.appendChild(header);
  const body = document.createElement("div");
  body.className = expanded ? "step-body" : "step-body collapsed";
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

function renderTimeline() { const timeline = byId("timeline"); timeline.innerHTML = ""; for (const step of (data.step_views || [])) timeline.appendChild(buildStepCard(step)); }
function renderFinalAnswer() {
  const panel = byId("final-answer");
  const prompt = String(((data.config || {}).prompt || "")).trim();
  const trajectory = String(data.final_text || "").trim();
  const trajectoryTokens = Array.isArray(data.trajectory_tokens) ? data.trajectory_tokens : [];
  const answer = String(data.final_answer_text || "").trim();
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
  renderMeta();
  renderTimeline();
  renderFinalAnswer();
}
try {
  bootstrap();
} catch (error) {
  const message = (error && error.message) ? error.message : String(error);
  document.body.innerHTML = `<main><section class='panel hero'><h1>Report Render Error</h1><div class='muted'>${esc(message)}</div></section></main>`;
}
</script>
""".strip()
