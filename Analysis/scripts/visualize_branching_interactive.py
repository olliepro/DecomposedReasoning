"""Interactive tree HTML helpers for branching replay pages."""

from __future__ import annotations

import json
from typing import Any


def workspace_css() -> str:
    """Return page-local CSS for interactive tree workspace widgets."""

    return """
.branch-workspace {
  display: grid;
  gap: 0.9rem;
  width: calc(100vw - 2rem);
  max-width: none;
  margin-left: calc(50% - 50vw + 1rem);
  margin-right: 0;
}
.workspace-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.8rem;
  align-items: center;
  justify-content: space-between;
}
.mode-toggle {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.18rem;
  gap: 0.15rem;
  background: rgba(255, 255, 255, 0.04);
}
.mode-btn {
  border: 0;
  border-radius: 999px;
  background: transparent;
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.73rem;
  padding: 0.28rem 0.7rem;
  cursor: pointer;
}
.mode-btn.active {
  color: #182033;
  background: var(--accent-soft);
  font-weight: 600;
}
.workspace-layout {
  display: grid;
  gap: 0;
  width: 100%;
  align-items: stretch;
  grid-template-columns: minmax(0, 1fr) 12px minmax(260px, var(--inspector-width, 380px));
}
.canvas-stack {
  display: grid;
  gap: 0.85rem;
  min-width: 0;
  width: 100%;
}
.canvas-card {
  border: 1px solid var(--line);
  border-radius: 0.75rem;
  background: rgba(10, 16, 28, 0.62);
  padding: 0.52rem;
  display: grid;
  gap: 0.45rem;
}
.viz-scroll {
  width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  border-radius: 0.65rem;
}
.viz-scroll::-webkit-scrollbar {
  height: 10px;
}
.viz-scroll::-webkit-scrollbar-thumb {
  background: rgba(170, 186, 219, 0.38);
  border-radius: 999px;
}
.canvas-title {
  color: var(--accent-soft);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.viz-canvas {
  display: block;
  width: 100%;
  min-height: 260px;
  border-radius: 0.65rem;
  border: 1px solid var(--line);
  background: rgba(8, 12, 22, 0.84);
}
.workspace-divider {
  width: 12px;
  cursor: col-resize;
  position: relative;
  user-select: none;
  touch-action: none;
}
.workspace-divider::before {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: 50%;
  width: 2px;
  transform: translateX(-50%);
  background: rgba(158, 178, 217, 0.45);
  border-radius: 999px;
}
.workspace-divider:hover::before,
.workspace-layout.resizing .workspace-divider::before {
  background: rgba(244, 180, 193, 0.9);
}
.inspector {
  border: 1px solid var(--line);
  border-radius: 0.75rem;
  background: rgba(10, 16, 28, 0.65);
  padding: 0.75rem;
  display: grid;
  gap: 0.65rem;
  align-content: start;
  max-height: 82dvh;
  overflow: auto;
  min-width: 0;
  width: 100%;
}
.inspector h3 {
  font-family: "IBM Plex Serif", serif;
  font-size: 1.02rem;
}
.detail-kv {
  display: grid;
  gap: 0.32rem;
}
.detail-row {
  display: grid;
  grid-template-columns: 130px minmax(0, 1fr);
  gap: 0.45rem;
  align-items: start;
  font-size: 0.76rem;
}
.detail-label {
  color: var(--accent-soft);
  font-family: "IBM Plex Mono", monospace;
}
.detail-value {
  color: var(--text);
  word-break: break-word;
}
.inspector pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  border: 1px solid var(--line);
  border-radius: 0.55rem;
  background: rgba(2, 8, 20, 0.58);
  padding: 0.5rem;
  font-size: 0.74rem;
}
.event-list {
  display: grid;
  gap: 0.32rem;
}
.event-btn {
  width: 100%;
  text-align: left;
  border: 1px solid var(--line);
  border-radius: 0.48rem;
  background: rgba(255, 255, 255, 0.03);
  color: var(--text);
  padding: 0.42rem 0.45rem;
  font-size: 0.74rem;
  cursor: pointer;
}
.event-btn:hover {
  border-color: rgba(255, 177, 153, 0.7);
}
.event-btn.active {
  border-color: rgba(255, 177, 153, 0.95);
  background: rgba(255, 177, 153, 0.14);
}
.node-pill-title {
  font-family: "IBM Plex Mono", monospace;
  font-size: 8.3px;
  fill: #dce8ff;
}
.node-pill-meta {
  font-family: "IBM Plex Mono", monospace;
  font-size: 7.7px;
  fill: #9eb2d9;
}
.axis-label {
  font-family: "IBM Plex Mono", monospace;
  font-size: 8.8px;
  fill: #9eb2d9;
}
.token-alt-list {
  display: grid;
  gap: 0.5rem;
}
.token-alt-card {
  border: 1px solid var(--line);
  border-radius: 0.55rem;
  background: rgba(2, 8, 20, 0.48);
  padding: 0.42rem;
  display: grid;
  gap: 0.34rem;
}
.token-alt-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.71rem;
}
.token-alt-head code {
  color: var(--text);
  font-size: 0.74rem;
  padding: 0.02rem 0.16rem;
}
.alt-grid {
  display: grid;
  gap: 0.28rem;
  min-width: 10rem;
}
.alt-row {
  position: relative;
  display: grid;
  grid-template-columns: 0.65rem 1fr auto;
  gap: 0.35rem;
  align-items: center;
  padding: 0.14rem 0.28rem;
  border-radius: 0.4rem;
  overflow: hidden;
}
.alt-row::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--fill, 0%);
  background: rgba(188, 196, 211, 0.22);
  pointer-events: none;
}
.alt-row.selected {
  border: 1px solid rgba(244, 180, 193, 0.84);
}
.alt-swatch {
  width: 0.52rem;
  height: 0.52rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  position: relative;
  z-index: 1;
}
.alt-token,
.alt-prob {
  position: relative;
  z-index: 1;
}
.alt-token {
  font-family: "IBM Plex Mono", monospace;
  color: #f2f6ff;
  font-size: 0.72rem;
}
.alt-prob {
  color: #d6dfef;
  font-size: 0.71rem;
}
.token-strip-wrap {
  border: 1px solid var(--line);
  border-radius: 0.55rem;
  background: rgba(2, 8, 20, 0.48);
  padding: 0.44rem 0.48rem;
  display: grid;
  gap: 0.32rem;
}
.token-strip-hint {
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.69rem;
}
.token-strip {
  white-space: normal;
  word-break: break-word;
  line-height: 1.56;
}
.gen-token {
  display: inline;
  border-radius: 0;
  padding: 0;
  margin: 0;
  border: 0;
  cursor: default;
  white-space: pre;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
}
.gen-token:hover {
  outline: 1px solid rgba(244, 180, 193, 0.95);
  outline-offset: 0;
}
.token-hover-tooltip {
  position: fixed;
  z-index: 9999;
  max-width: min(390px, 90vw);
  border: 1px solid var(--line);
  border-radius: 0.55rem;
  background: rgba(6, 10, 20, 0.97);
  box-shadow: 0 18px 38px rgba(0, 0, 0, 0.35);
  padding: 0.42rem 0.45rem;
  display: none;
  pointer-events: none;
}
.token-hover-tooltip.visible {
  display: block;
}
.candidate-list {
  display: grid;
  gap: 0.5rem;
}
.candidate-card {
  border: 1px solid var(--line);
  border-radius: 0.55rem;
  background: rgba(2, 8, 20, 0.48);
  padding: 0.46rem;
  display: grid;
  gap: 0.32rem;
}
.candidate-card.selected {
  border-color: rgba(244, 180, 193, 0.95);
  box-shadow: 0 0 0 1px rgba(244, 180, 193, 0.35) inset;
}
.candidate-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.5rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.73rem;
}
.candidate-text {
  border: 1px solid var(--line);
  border-radius: 0.45rem;
  background: rgba(0, 0, 0, 0.18);
  padding: 0.34rem 0.4rem;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
  font-size: 0.74rem;
}
.cluster-mode-meta {
  color: var(--muted);
  font-size: 0.73rem;
  margin: 0.05rem 0 0.25rem;
}
.cluster-grid {
  display: grid;
  gap: 0.5rem;
}
.cluster-row {
  border: 1px solid var(--line);
  border-radius: 0.6rem;
  background: rgba(255, 255, 255, 0.02);
  overflow: hidden;
}
.cluster-row.top-cluster {
  border-color: rgba(244, 180, 193, 0.78);
}
.cluster-row.top-cluster[open] {
  border-color: var(--line);
}
.cluster-row > summary {
  padding: 0.56rem 0.62rem;
  cursor: pointer;
  list-style: none;
  position: relative;
}
.cluster-row > summary::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: var(--fill, 0%);
  background: rgba(229, 77, 103, 0.08);
  pointer-events: none;
}
.cluster-row > summary::-webkit-details-marker {
  display: none;
}
.cluster-row > summary::marker {
  content: "";
}
.cluster-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.55rem;
}
.cluster-name-wrap {
  display: flex;
  align-items: center;
  gap: 0.36rem;
  min-width: 0;
}
.cluster-caret {
  color: var(--accent-soft);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  transition: transform 140ms ease;
}
.cluster-row[open] .cluster-caret {
  transform: rotate(90deg);
}
.cluster-name {
  color: var(--text);
  font-weight: 600;
  font-size: 0.76rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.cluster-tags {
  display: flex;
  align-items: center;
  gap: 0.32rem;
  flex-wrap: wrap;
}
.cluster-count,
.cluster-selected {
  border: 1px solid rgba(244, 180, 193, 0.52);
  background: rgba(229, 77, 103, 0.09);
  color: var(--accent-soft);
  border-radius: 999px;
  font-size: 0.67rem;
  padding: 0.1rem 0.42rem;
}
.cluster-body {
  display: grid;
  gap: 0.44rem;
  padding: 0.38rem 0.62rem 0.62rem;
}
.candidate-panel {
  padding: 0.46rem;
  border: 1px solid var(--line);
  border-radius: 0.5rem;
  background: rgba(255, 255, 255, 0.01);
  display: grid;
  gap: 0.34rem;
}
.candidate-panel.selected {
  border-color: rgba(244, 180, 193, 0.95);
  background: rgba(229, 77, 103, 0.07);
}
.candidate-panel .candidate-text {
  margin: 0;
}
.hidden-note {
  color: var(--muted);
  font-size: 0.73rem;
}
@media (max-width: 980px) {
  .branch-workspace {
    width: 100%;
    margin-left: 0;
  }
  .workspace-layout {
    grid-template-columns: 1fr;
    gap: 0.9rem;
  }
  .workspace-divider {
    display: none;
  }
  .inspector {
    max-height: none;
  }
}
"""


def render_tree_workspace(*, payload: dict[str, Any]) -> str:
    """Render interactive tree panel with embedded JSON payload.

    Args:
        payload: JSON-ready attempt visualization payload.

    Returns:
        HTML panel string.
    """

    payload_json = json.dumps(payload).replace("</", "<\\/")
    return f"""
<style>{workspace_css()}</style>
<section class="panel branch-workspace">
  <div class="workspace-toolbar">
    <h2>Tree</h2>
    <div class="mode-toggle" role="group" aria-label="x-axis mode">
      <button class="mode-btn" type="button" data-mode="tokens">Tokens</button>
      <button class="mode-btn active" type="button" data-mode="steps">Steps</button>
      <button class="mode-btn" type="button" data-mode="time">Time generated</button>
    </div>
  </div>
  <p class="muted" style="margin:0">
    Click a node or event chip to inspect request prefixes and generated tokens.
  </p>
  <div class="workspace-layout">
    <div class="canvas-stack">
      <div class="canvas-card">
        <div class="canvas-title">Tree Structure</div>
        <div class="viz-scroll" id="tree-scroll"><svg id="tree-svg" class="viz-canvas"></svg></div>
      </div>
    </div>
    <div class="workspace-divider" id="workspace-divider" role="separator" aria-orientation="vertical" aria-label="Resize inspector"></div>
    <aside class="inspector" id="viz-inspector">
      <h3>Inspector</h3>
      <div id="inspector-content" class="detail-kv"></div>
    </aside>
  </div>
  <script type="application/json" id="tree-data">{payload_json}</script>
</section>
"""


def tree_workspace_script() -> str:
    """Return client-side script for interactive tree workspace."""

    return """
(() => {
  const payloadNode = document.getElementById("tree-data");
  if (!payloadNode) return;
  let payload = {};
  try {
    payload = JSON.parse(payloadNode.textContent || "{}");
  } catch {
    return;
  }
  const treeSvg = document.getElementById("tree-svg");
  const inspectorHost = document.getElementById("inspector-content");
  const workspaceLayout = document.querySelector(".workspace-layout");
  const workspaceDivider = document.getElementById("workspace-divider");
  const inspectorPane = document.getElementById("viz-inspector");
  if (!treeSvg || !inspectorHost || !workspaceLayout) return;

  const nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
  const edges = Array.isArray(payload.edges) ? payload.edges : [];
  const nodeEvents = payload.node_events && typeof payload.node_events === "object"
    ? payload.node_events
    : {};

  const eventById = new Map();
  const candidatePoolByBranchPoint = new Map();
  for (const [nodeId, events] of Object.entries(nodeEvents)) {
    if (!Array.isArray(events)) continue;
    for (const event of events) {
      if (!event || typeof event !== "object") continue;
      event.node_id = nodeId;
      if (event.event_id) eventById.set(String(event.event_id), event);
      if (event.event_type === "candidate_pool_resolved" && event.details) {
        const branchPointId = String(event.details.branch_point_id || "");
        if (branchPointId) candidatePoolByBranchPoint.set(branchPointId, event.details);
      }
    }
  }

  const modeKey = {
    tokens: "tokens",
    steps: "steps",
    time: "time_seconds",
  };
  const modeLabel = {
    tokens: "Tokens",
    steps: "Steps",
    time: "Time generated (s)",
  };
  const axisScaleByMode = {
    tokens: {
      pixelsPerUnit: 0.56,
      tickStep: 512,
    },
    steps: {
      pixelsPerUnit: 12.0,
      tickStep: 10,
    },
    time: {
      pixelsPerUnit: 1.5,
      tickStep: 60,
    },
  };
  let currentMode = "steps";
  let selectedNodeId = null;
  let selectedEventId = null;
  let tokenHoverSeq = 0;
  const tokenHoverRowsById = new Map();
  const treeStepMetrics = buildTreeStepMetrics();
  const treeHierarchyLayout = buildTreeHierarchyLayout();
  const leafOutcomeByNode = buildLeafOutcomeByNode();

  const svgNs = "http://www.w3.org/2000/svg";

  function buildTreeStepMetrics() {
    const parentByNode = {};
    const allNodeIds = new Set();
    for (const node of nodes) {
      const nodeId = String(node.node_id || "");
      if (!nodeId) continue;
      allNodeIds.add(nodeId);
      if (node.parent_node_id !== null && node.parent_node_id !== undefined) {
        const parentId = String(node.parent_node_id || "");
        if (parentId) parentByNode[nodeId] = parentId;
      }
    }
    for (const edge of edges) {
      const parentId = String(edge.parent_node_id || "");
      const childId = String(edge.child_node_id || "");
      if (!childId) continue;
      allNodeIds.add(childId);
      if (parentId) {
        allNodeIds.add(parentId);
        if (!parentByNode[childId]) {
          parentByNode[childId] = parentId;
        }
      }
    }
    for (const nodeId of Object.keys(nodeEvents)) {
      allNodeIds.add(String(nodeId));
    }
    const localEventsByNode = {};
    for (const nodeId of allNodeIds) {
      localEventsByNode[nodeId] = sortedNodeEvents(nodeId);
    }
    const rangeByNode = new Map();
    const stepByEventId = new Map();
    const visiting = new Set();

    function visit(nodeId) {
      const existing = rangeByNode.get(nodeId);
      if (existing) return existing;
      if (visiting.has(nodeId)) {
        const fallback = { start: 0, end: 0 };
        rangeByNode.set(nodeId, fallback);
        return fallback;
      }
      visiting.add(nodeId);
      const parentId = parentByNode[nodeId];
      let baseStep = 1;
      if (parentId && parentId !== nodeId) {
        const parentRange = visit(parentId);
        baseStep = asNumber(parentRange.end) + 1;
      }
      const events = Array.isArray(localEventsByNode[nodeId])
        ? localEventsByNode[nodeId]
        : [];
      let endStep = baseStep - 1;
      for (let index = 0; index < events.length; index += 1) {
        const event = events[index];
        const step = baseStep + index;
        stepByEventId.set(String(event.event_id || ""), step);
        endStep = step;
      }
      if (!events.length && parentId && parentId !== nodeId) {
        const parentRange = visit(parentId);
        endStep = asNumber(parentRange.end);
      }
      const result = {
        start: Math.max(0, baseStep),
        end: Math.max(0, endStep),
      };
      rangeByNode.set(nodeId, result);
      visiting.delete(nodeId);
      return result;
    }

    for (const nodeId of allNodeIds) {
      visit(nodeId);
    }
    return { stepByEventId, rangeByNode };
  }

  function buildTreeHierarchyLayout() {
    const parentByNode = {};
    const childrenByNode = {};
    const allNodeIds = new Set();
    for (const node of nodes) {
      const nodeId = String(node.node_id || "");
      if (!nodeId) continue;
      allNodeIds.add(nodeId);
      childrenByNode[nodeId] = childrenByNode[nodeId] || [];
      const parentId = node.parent_node_id === null || node.parent_node_id === undefined
        ? ""
        : String(node.parent_node_id || "");
      if (parentId && parentId !== nodeId) {
        parentByNode[nodeId] = parentId;
        allNodeIds.add(parentId);
        childrenByNode[parentId] = childrenByNode[parentId] || [];
      }
    }
    for (const edge of edges) {
      const parentId = String(edge.parent_node_id || "");
      const childId = String(edge.child_node_id || "");
      if (!childId) continue;
      allNodeIds.add(childId);
      childrenByNode[childId] = childrenByNode[childId] || [];
      if (parentId && parentId !== childId) {
        parentByNode[childId] = parentId;
        allNodeIds.add(parentId);
        childrenByNode[parentId] = childrenByNode[parentId] || [];
      }
    }
    for (const [childId, parentId] of Object.entries(parentByNode)) {
      if (!childrenByNode[parentId]) childrenByNode[parentId] = [];
      if (!childrenByNode[parentId].includes(childId)) {
        childrenByNode[parentId].push(childId);
      }
    }
    for (const nodeId of Object.keys(childrenByNode)) {
      childrenByNode[nodeId] = childrenByNode[nodeId]
        .filter((childId) => childId !== nodeId)
        .sort((left, right) => left.localeCompare(right));
    }
    const roots = Array.from(allNodeIds)
      .filter((nodeId) => {
        const parentId = parentByNode[nodeId];
        return !parentId || !allNodeIds.has(parentId);
      })
      .sort((left, right) => {
        if (left === "node_root") return -1;
        if (right === "node_root") return 1;
        return left.localeCompare(right);
      });
    const visited = new Set();
    const rowByNode = new Map();
    const labelByNode = new Map();
    let nextRow = 0;
    let rootIndex = 1;

    function allocateRow(preferredRow) {
      if (preferredRow !== null && preferredRow !== undefined) {
        const preferred = Math.max(0, Math.floor(asNumber(preferredRow)));
        nextRow = Math.max(nextRow, preferred + 1);
        return preferred;
      }
      const allocated = nextRow;
      nextRow += 1;
      return allocated;
    }

    function dfs(nodeId, numbering, preferredRow) {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      const rowIndex = allocateRow(preferredRow);
      rowByNode.set(nodeId, rowIndex);
      labelByNode.set(nodeId, numbering);
      const children = Array.isArray(childrenByNode[nodeId]) ? childrenByNode[nodeId] : [];
      for (let childIndex = 0; childIndex < children.length; childIndex += 1) {
        const childId = children[childIndex];
        const childPreferredRow = childIndex === 0 ? rowIndex : null;
        dfs(childId, `${numbering}.${childIndex + 1}`, childPreferredRow);
      }
    }

    for (const nodeId of roots) {
      dfs(nodeId, String(rootIndex), null);
      rootIndex += 1;
    }
    const remaining = Array.from(allNodeIds)
      .filter((nodeId) => !visited.has(nodeId))
      .sort((left, right) => left.localeCompare(right));
    for (const nodeId of remaining) {
      dfs(nodeId, String(rootIndex), null);
      rootIndex += 1;
    }
    return { rowByNode, labelByNode, rowCount: Math.max(1, nextRow) };
  }

  function asNumber(value) {
    const cast = Number(value);
    return Number.isFinite(cast) ? cast : 0;
  }

  function metricValue(metrics) {
    if (!metrics || typeof metrics !== "object") return 0;
    return asNumber(metrics[modeKey[currentMode]]);
  }

  function sortedNodeEvents(nodeId) {
    const events = Array.isArray(nodeEvents[nodeId]) ? nodeEvents[nodeId].slice() : [];
    events.sort((left, right) => asNumber(left.event_index) - asNumber(right.event_index));
    return events;
  }

  function treeNodeMetricValue(node) {
    if (currentMode !== "steps") {
      return metricValue(node.metrics);
    }
    const nodeId = String(node.node_id || "");
    const range = treeStepMetrics.rangeByNode.get(nodeId);
    if (range) {
      return asNumber(range.end);
    }
    return 0;
  }

  function treeEventMetricValue(event) {
    if (!event || typeof event !== "object") return 0;
    if (currentMode === "steps") {
      const mapped = treeStepMetrics.stepByEventId.get(String(event.event_id || ""));
      if (mapped !== undefined) return asNumber(mapped);
      return 0;
    }
    return metricValue(event.metrics);
  }

  function buildLeafOutcomeByNode() {
    const outcomeByNode = new Map();
    for (const [nodeId, events] of Object.entries(nodeEvents)) {
      if (!Array.isArray(events)) continue;
      for (const event of events) {
        if (!event || typeof event !== "object") continue;
        if (String(event.event_type || "") !== "leaf_scored") continue;
        const eventIndex = asNumber(event.event_index);
        const details = event.details && typeof event.details === "object"
          ? event.details
          : {};
        const previous = outcomeByNode.get(nodeId);
        if (previous && asNumber(previous.eventIndex) > eventIndex) continue;
        outcomeByNode.set(nodeId, {
          eventIndex,
          verification: details.verification,
          stopReason: String(details.stop_reason || ""),
        });
      }
    }
    return outcomeByNode;
  }

  function stopReasonIcon(stopReason) {
    const normalized = String(stopReason || "").trim().toLowerCase();
    if (!normalized) return "";
    if (normalized.includes("repeated_") && normalized.includes("_block_loop")) {
      return "🔁";
    }
    if (normalized.includes("length") || normalized.includes("max_gen_toks_reached")) {
      return "🛑";
    }
    return "🏁";
  }

  function verificationIcon(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric === 1) return "✅";
    if (Number.isFinite(numeric) && numeric === 0) return "❌";
    if (String(value).trim() === "1") return "✅";
    if (String(value).trim() === "0") return "❌";
    return "";
  }

  function leafStatusSuffix(nodeId) {
    const outcome = leafOutcomeByNode.get(String(nodeId));
    if (!outcome) return "";
    const verify = verificationIcon(outcome.verification);
    const stop = stopReasonIcon(outcome.stopReason);
    const pieces = [];
    if (verify) pieces.push(verify);
    if (stop) pieces.push(stop);
    return pieces.join(" ");
  }

  function esc(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function normalizeTokenText(tokenText) {
    const token = String(tokenText || "");
    if (!token.length) return "∅";
    const slash = String.fromCharCode(92);
    const newline = String.fromCharCode(10);
    const tab = String.fromCharCode(9);
    return token.split(newline).join(`${slash}n`).split(tab).join(`${slash}t`);
  }

  function cleanBranchLabelText(labelText) {
    const raw = String(labelText || "");
    const stripped = raw.replace(/(?:\\s*<\\/steer>\\s*<exec>\\s*)+$/g, "").trim();
    return stripped || raw.trim();
  }

  function formatProbabilityPercent(probability) {
    const p = Math.max(0, Math.min(1, Number(probability || 0)));
    const rounded = Math.round((p * 100) * 100) / 100;
    if (Number.isInteger(rounded)) return `${rounded}%`;
    return `${rounded.toFixed(2)}%`;
  }

  function probabilityFromLogprob(logprob) {
    const value = Number(logprob);
    if (!Number.isFinite(value)) return 0;
    const probability = Math.exp(value);
    return Number.isFinite(probability) ? Math.max(0, Math.min(1, probability)) : 0;
  }

  function probColor(probability) {
    const p = Math.max(0, Math.min(1, Number(probability || 0)));
    const hue = Math.round(8 + (p * 106));
    const sat = 84;
    const light = Math.round(76 - (p * 14));
    return `hsl(${hue}, ${sat}%, ${light}%)`;
  }

  function withoutRawTokenIds(value) {
    if (Array.isArray(value)) {
      return value.map((item) => withoutRawTokenIds(item));
    }
    if (!value || typeof value !== "object") {
      return value;
    }
    const filtered = {};
    for (const [key, innerValue] of Object.entries(value)) {
      const keyLower = String(key).toLowerCase();
      if (keyLower.includes("token_id")) continue;
      filtered[key] = withoutRawTokenIds(innerValue);
    }
    return filtered;
  }

  function clearSvg(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }

  function xScale(value, minValue, maxValue, width, left, right) {
    const span = Math.max(1e-6, maxValue - minValue);
    const normalized = (value - minValue) / span;
    return left + normalized * Math.max(10, width - left - right);
  }

  function axisScaleConfig() {
    return axisScaleByMode[currentMode] || axisScaleByMode.tokens;
  }

  function computeCanvasWidth({ minValue, maxValue }) {
    const metricSpan = Math.max(0, maxValue - minValue);
    const config = axisScaleConfig();
    const leftPadding = 64;
    const rightPadding = 24;
    const drawableWidth = Math.max(
      140,
      Math.round(metricSpan * asNumber(config.pixelsPerUnit)),
    );
    return Math.min(180000, Math.max(336, leftPadding + rightPadding + drawableWidth));
  }

  function fitAxisToContainer({ minValue, maxValue, containerWidth }) {
    const boundedContainerWidth = Math.max(336, Math.round(asNumber(containerWidth)));
    const config = axisScaleConfig();
    const pixelsPerUnit = Math.max(1e-6, asNumber(config.pixelsPerUnit));
    const baseWidth = computeCanvasWidth({ minValue, maxValue });
    let fittedMin = minValue;
    let fittedMax = maxValue;
    let fittedWidth = baseWidth;
    if (baseWidth < boundedContainerWidth) {
      fittedWidth = Math.min(180000, boundedContainerWidth);
      const drawableWidth = Math.max(10, fittedWidth - 64 - 24);
      const fittedSpan = drawableWidth / pixelsPerUnit;
      const currentSpan = Math.max(1e-6, maxValue - minValue);
      if (fittedSpan > currentSpan) {
        fittedMax = fittedMin + fittedSpan;
      }
    }
    return {
      minValue: fittedMin,
      maxValue: fittedMax,
      width: fittedWidth,
    };
  }

  function evenlySpacedTicks(minValue, maxValue, tickCount) {
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || tickCount <= 0) return [0];
    if (maxValue <= minValue) return [minValue];
    const ticks = [];
    for (let i = 0; i <= tickCount; i += 1) {
      const ratio = i / tickCount;
      ticks.push(minValue + (maxValue - minValue) * ratio);
    }
    return ticks;
  }

  function fixedStepTicks(minValue, maxValue, stepSize) {
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || stepSize <= 0) return [0];
    if (maxValue <= minValue) return [minValue];
    const epsilon = stepSize * 1e-6;
    const ticks = [];
    const first = Math.floor(minValue / stepSize) * stepSize;
    for (let value = first; value <= maxValue + epsilon; value += stepSize) {
      if (value < minValue - epsilon) continue;
      ticks.push(value);
    }
    if (ticks.length === 0 || Math.abs(ticks[0] - minValue) > epsilon) ticks.unshift(minValue);
    if (Math.abs(ticks[ticks.length - 1] - maxValue) > epsilon) ticks.push(maxValue);
    const normalized = [];
    for (const value of ticks) {
      const rounded = Number(value.toFixed(6));
      if (normalized.length === 0 || Math.abs(normalized[normalized.length - 1] - rounded) > epsilon) {
        normalized.push(rounded);
      }
    }
    return normalized;
  }

  function axisTicksForMode(minValue, maxValue) {
    const config = axisScaleConfig();
    return fixedStepTicks(minValue, maxValue, asNumber(config.tickStep));
  }

  function formatAxisValue(value) {
    if (currentMode === "time") {
      return Number.isInteger(value) ? String(value) : value.toFixed(1);
    }
    return String(Math.round(value));
  }

  function truncateSvgTextToWidth({ textElement, fullText, maxWidth }) {
    const ellipsis = "...";
    if (maxWidth <= 6) return "";
    textElement.textContent = fullText;
    if (asNumber(textElement.getComputedTextLength()) <= maxWidth) {
      return fullText;
    }
    let low = 0;
    let high = fullText.length;
    while (low < high) {
      const mid = Math.floor((low + high + 1) / 2);
      const trimmed = fullText.slice(0, mid).trimEnd();
      const candidate = trimmed ? `${trimmed}${ellipsis}` : ellipsis;
      textElement.textContent = candidate;
      if (asNumber(textElement.getComputedTextLength()) <= maxWidth) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }
    const best = fullText.slice(0, low).trimEnd();
    return best ? `${best}${ellipsis}` : ellipsis;
  }

  function renderAxis(svg, width, top, bottom, minValue, maxValue) {
    const ticks = axisTicksForMode(minValue, maxValue);
    for (const value of ticks) {
      const x = xScale(value, minValue, maxValue, width, 64, 24);
      const line = document.createElementNS(svgNs, "line");
      line.setAttribute("x1", String(x));
      line.setAttribute("x2", String(x));
      line.setAttribute("y1", String(top));
      line.setAttribute("y2", String(bottom));
      line.setAttribute("stroke", "rgba(150, 171, 213, 0.22)");
      line.setAttribute("stroke-width", "0.8");
      svg.appendChild(line);
      const label = document.createElementNS(svgNs, "text");
      label.setAttribute("x", String(x));
      label.setAttribute("y", String(top - 6));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("class", "axis-label");
      label.textContent = formatAxisValue(value);
      svg.appendChild(label);
    }
    const axisName = document.createElementNS(svgNs, "text");
    axisName.setAttribute("x", "66");
    axisName.setAttribute("y", String(top - 19));
    axisName.setAttribute("class", "axis-label");
    axisName.textContent = modeLabel[currentMode];
    svg.appendChild(axisName);
  }

  function colorForEvent(type) {
    if (type === "vllm_step") return "#63e6be";
    if (type === "trigger_fired") return "#ff875f";
    if (type === "trigger_skipped_max_branch_points") return "#ff9f7f";
    if (type === "candidate_pool_resolved") return "#ffd166";
    if (type === "selector_applied") return "#f6a6ff";
    if (type === "leaf_completed") return "#9fe2ff";
    if (type === "leaf_scored") return "#b8f28d";
    return "#9fb4db";
  }

  function nodeById(nodeId) {
    return nodes.find((row) => String(row.node_id) === String(nodeId)) || null;
  }

  function renderTree() {
    clearSvg(treeSvg);
    const treeScroll = document.getElementById("tree-scroll");
    const containerWidth = Math.max(
      336,
      treeScroll ? asNumber(treeScroll.clientWidth) : asNumber(treeSvg.clientWidth || 920),
    );
    const rowCount = Math.max(1, asNumber(treeHierarchyLayout.rowCount));
    const laneHeight = 39;
    const axisTop = 56;
    const rootHeadroom = 29;
    const nodeTop = axisTop + rootHeadroom;
    const bottom = nodeTop + Math.max(0, rowCount - 1) * laneHeight;
    const height = bottom + 34;

    let minMetric = Number.POSITIVE_INFINITY;
    let maxMetric = Number.NEGATIVE_INFINITY;
    for (const node of nodes) {
      const value = treeNodeMetricValue(node);
      minMetric = Math.min(minMetric, value);
      maxMetric = Math.max(maxMetric, value);
    }
    for (const [nodeId, events] of Object.entries(nodeEvents)) {
      if (!Array.isArray(events)) continue;
      for (const event of events) {
        const value = treeEventMetricValue(event);
        minMetric = Math.min(minMetric, value);
        maxMetric = Math.max(maxMetric, value);
      }
    }
    if (!Number.isFinite(minMetric)) {
      minMetric = 0;
      maxMetric = 1;
    }
    if (Math.abs(maxMetric - minMetric) < 1e-6) {
      maxMetric = minMetric + 1;
    }
    const fittedAxis = fitAxisToContainer({
      minValue: minMetric,
      maxValue: maxMetric,
      containerWidth,
    });
    minMetric = fittedAxis.minValue;
    maxMetric = fittedAxis.maxValue;
    const width = fittedAxis.width;
    treeSvg.style.width = `${width}px`;
    treeSvg.style.minWidth = `${width}px`;
    treeSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    renderAxis(treeSvg, width, axisTop, bottom, minMetric, maxMetric);

    const positions = {};
    const nodeEventRanges = {};
    for (const node of nodes) {
      const nodeId = String(node.node_id);
      const events = sortedNodeEvents(nodeId);
      const startMetric = events.length
        ? treeEventMetricValue(events[0])
        : treeNodeMetricValue(node);
      const endMetric = events.length
        ? treeEventMetricValue(events[events.length - 1])
        : treeNodeMetricValue(node);
      const startX = xScale(startMetric, minMetric, maxMetric, width, 64, 24);
      const endX = xScale(endMetric, minMetric, maxMetric, width, 64, 24);
      const rowIndex = asNumber(treeHierarchyLayout.rowByNode.get(nodeId) || 0);
      positions[nodeId] = {
        x: startX,
        y: nodeTop + rowIndex * laneHeight,
      };
      nodeEventRanges[nodeId] = {
        startMetric,
        endMetric,
        startX,
        endX,
        eventCount: events.length,
      };
    }

    for (const edge of edges) {
      const parentId = String(edge.parent_node_id || "");
      const childId = String(edge.child_node_id || "");
      const parent = positions[parentId];
      const child = positions[childId];
      if (!parent || !child) continue;
      const parentRange = nodeEventRanges[parentId];
      const childRange = nodeEventRanges[childId];
      const edgeStartX = parentRange ? asNumber(parentRange.endX) : asNumber(parent.x);
      const edgeEndX = childRange ? asNumber(childRange.startX) : asNumber(child.x);
      const bend = Math.max(19, Math.abs(edgeEndX - edgeStartX) * 0.35);
      const path = document.createElementNS(svgNs, "path");
      path.setAttribute(
        "d",
        `M ${edgeStartX} ${parent.y} C ${edgeStartX + bend} ${parent.y}, ${edgeEndX - bend} ${child.y}, ${edgeEndX} ${child.y}`,
      );
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "rgba(165, 187, 230, 0.55)");
      path.setAttribute("stroke-width", "1.3");
      treeSvg.appendChild(path);
    }

    const selectedEventNodeId = selectedEventId
      ? String((eventById.get(String(selectedEventId)) || {}).node_id || "")
      : "";
    const pillHeight = 19;
    const pillPaddingX = 4;
    const nodeDrawById = new Map();
    const rowNodePills = new Map();
    for (const node of nodes) {
      const nodeId = String(node.node_id || "");
      const pos = positions[nodeId];
      if (!pos) continue;
      const events = sortedNodeEvents(nodeId);
      const nodeRange = nodeEventRanges[nodeId];
      const rawStartX = nodeRange ? asNumber(nodeRange.startX) : asNumber(pos.x);
      const rawEndX = nodeRange ? asNumber(nodeRange.endX) : asNumber(pos.x);
      const eventSpanWidth = Math.max(0, rawEndX - rawStartX);
      const pillWidth = eventSpanWidth + (pillPaddingX * 2) + 2;
      let pillX = rawStartX - pillPaddingX;
      const minX = 66;
      const maxX = Math.max(minX, width - 22 - pillWidth);
      pillX = Math.max(minX, Math.min(maxX, pillX));
      const pillY = pos.y - (pillHeight / 2);
      const rowIndex = asNumber(treeHierarchyLayout.rowByNode.get(nodeId) || 0);
      nodeDrawById.set(nodeId, {
        node,
        nodeId,
        events,
        pos,
        rawStartX,
        rawEndX,
        pillX,
        pillY,
        pillWidth,
        rowIndex,
      });
      if (!rowNodePills.has(rowIndex)) {
        rowNodePills.set(rowIndex, []);
      }
      rowNodePills.get(rowIndex).push({ nodeId, pillX });
    }
    const nextPillStartByNode = new Map();
    for (const rowEntries of rowNodePills.values()) {
      rowEntries.sort((left, right) => asNumber(left.pillX) - asNumber(right.pillX));
      for (let index = 0; index < rowEntries.length; index += 1) {
        const current = rowEntries[index];
        const next = rowEntries[index + 1];
        if (!next) continue;
        nextPillStartByNode.set(String(current.nodeId), asNumber(next.pillX));
      }
    }
    for (const node of nodes) {
      const nodeId = String(node.node_id || "");
      const draw = nodeDrawById.get(nodeId);
      if (!draw) continue;
      const depth = asNumber(draw.node.depth);
      const root = nodeId === "node_root";
      const hue = root ? 16 : Math.max(192, 224 - depth * 7);
      const sat = root ? 88 : 64;
      const light = root ? 62 : Math.max(38, 64 - depth * 2.6);
      const active = selectedNodeId === nodeId || selectedEventNodeId === nodeId;
      const group = document.createElementNS(svgNs, "g");
      group.style.cursor = "pointer";
      treeSvg.appendChild(group);
      const body = document.createElementNS(svgNs, "rect");
      body.setAttribute("x", String(draw.pillX));
      body.setAttribute("y", String(draw.pillY));
      body.setAttribute("width", String(draw.pillWidth));
      body.setAttribute("height", String(pillHeight));
      body.setAttribute("rx", String(pillHeight / 2));
      body.setAttribute("fill", `hsl(${hue}, ${sat}%, ${light}%)`);
      body.setAttribute("fill-opacity", active ? "0.52" : "0.32");
      body.setAttribute("stroke", active ? "#ffffff" : "rgba(255,255,255,0.22)");
      body.setAttribute("stroke-width", active ? "1.6" : "1");
      group.appendChild(body);

      const title = document.createElementNS(svgNs, "text");
      title.setAttribute("x", String(draw.pillX));
      title.setAttribute("y", String(draw.pillY - 5));
      title.setAttribute("class", "node-pill-title");
      group.appendChild(title);
      const hierarchyLabel = String(treeHierarchyLayout.labelByNode.get(nodeId) || "?");
      const rawLabel = cleanBranchLabelText(draw.node.candidate_preview || "");
      const cleanLabel = rawLabel.length > 56 ? `${rawLabel.slice(0, 53)}...` : rawLabel;
      const statusSuffix = leafStatusSuffix(nodeId);
      const displaySuffix = statusSuffix ? ` ${statusSuffix}` : "";
      const fullTitle = `${hierarchyLabel} ${cleanLabel || "Node"}${displaySuffix}`;
      const fallbackRightBoundary = width - 24;
      const nextPillStart = nextPillStartByNode.get(nodeId);
      const titleRightBoundary = nextPillStart === undefined
        ? fallbackRightBoundary
        : Math.min(fallbackRightBoundary, nextPillStart - 8);
      const titleMaxWidth = Math.max(12, titleRightBoundary - draw.pillX);
      title.textContent = truncateSvgTextToWidth({
        textElement: title,
        fullText: fullTitle,
        maxWidth: titleMaxWidth,
      });

      const innerLeft = draw.pillX + pillPaddingX;
      const innerRight = draw.pillX + draw.pillWidth - pillPaddingX;
      const eventGuideY = draw.pillY + (pillHeight / 2);
      if (draw.events.length > 1) {
        const guide = document.createElementNS(svgNs, "line");
        guide.setAttribute("x1", String(innerLeft));
        guide.setAttribute("x2", String(innerRight));
        guide.setAttribute("y1", String(eventGuideY));
        guide.setAttribute("y2", String(eventGuideY));
        guide.setAttribute("stroke", "rgba(199, 219, 255, 0.4)");
        guide.setAttribute("stroke-width", "0.8");
        group.appendChild(guide);
      }
      draw.events.forEach((event, eventIndex) => {
        let eventX = xScale(treeEventMetricValue(event), minMetric, maxMetric, width, 64, 24);
        if (!Number.isFinite(eventX)) {
          eventX = innerLeft;
        }
        if (draw.events.length > 1 && Math.abs(draw.rawEndX - draw.rawStartX) < 1e-6) {
          const ratio = eventIndex / Math.max(1, draw.events.length - 1);
          eventX = innerLeft + ratio * Math.max(0, innerRight - innerLeft);
        }
        eventX = Math.max(innerLeft, Math.min(innerRight, eventX));
        const selected = selectedEventId === String(event.event_id);
        const chipWidth = selected ? 10 : 8;
        const chipHeight = selected ? 10 : 7;
        const chip = document.createElementNS(svgNs, "rect");
        chip.setAttribute("x", String(eventX - (chipWidth / 2)));
        chip.setAttribute("y", String(eventGuideY - (chipHeight / 2)));
        chip.setAttribute("width", String(chipWidth));
        chip.setAttribute("height", String(chipHeight));
        chip.setAttribute("rx", String(chipHeight / 2));
        chip.setAttribute("fill", colorForEvent(String(event.event_type || "")));
        chip.setAttribute("stroke", selected ? "#ffffff" : "rgba(255,255,255,0.28)");
        chip.setAttribute("stroke-width", selected ? "1.1" : "0.8");
        chip.style.cursor = "pointer";
        chip.addEventListener("click", (clickEvent) => {
          clickEvent.stopPropagation();
          selectedNodeId = nodeId;
          selectedEventId = String(event.event_id || "");
          renderAll();
        });
        group.appendChild(chip);
      });

      group.addEventListener("click", () => {
        selectedNodeId = nodeId;
        selectedEventId = null;
        renderAll();
      });
    }
  }

  function detailRow(label, value) {
    return `<div class=\"detail-row\"><div class=\"detail-label\">${esc(label)}</div><div class=\"detail-value\">${esc(value)}</div></div>`;
  }

  function renderNodeInspector(nodeId) {
    hideTokenTooltip();
    const node = nodeById(nodeId);
    if (!node) {
      inspectorHost.innerHTML = "<p class='muted'>Select a node or event.</p>";
      return;
    }
    const metrics = node.metrics || {};
    const rows = [];
    const stepRange = treeStepMetrics.rangeByNode.get(String(nodeId));
    rows.push(`<h3><code>${esc(nodeId)}</code></h3>`);
    rows.push(detailRow("label", cleanBranchLabelText(node.candidate_preview || "")));
    rows.push(detailRow("depth", String(node.depth ?? "")));
    rows.push(detailRow("events", String(node.event_count ?? "")));
    rows.push(detailRow("tokens", String(asNumber(metrics.tokens).toFixed(0))));
    rows.push(
      detailRow(
        "steps_so_far",
        String(stepRange ? asNumber(stepRange.end).toFixed(0) : 0),
      ),
    );
    rows.push(detailRow("time_s", String(asNumber(metrics.time_seconds).toFixed(2))));
    const events = Array.isArray(nodeEvents[nodeId]) ? nodeEvents[nodeId] : [];
    if (!events.length) {
      rows.push("<p class='muted'>No node-local events.</p>");
      inspectorHost.innerHTML = rows.join("");
      return;
    }
    const buttons = events
      .slice()
      .sort((left, right) => asNumber(left.event_index) - asNumber(right.event_index))
      .map((event) => {
        const active = selectedEventId === String(event.event_id) ? "active" : "";
        const summary = `${event.event_index} · ${event.event_type} · ${event.summary || ""}`;
        return `<button class=\"event-btn ${active}\" data-event-id=\"${esc(event.event_id)}\">${esc(summary)}</button>`;
      });
    rows.push("<div class='event-list'>" + buttons.join("") + "</div>");
    inspectorHost.innerHTML = rows.join("");
    inspectorHost.querySelectorAll(".event-btn").forEach((button) => {
      button.addEventListener("click", () => {
        const eventId = button.getAttribute("data-event-id");
        if (!eventId) return;
        selectedEventId = eventId;
        renderAll();
      });
    });
  }

  function tokenAlternativeRows(tokenRow) {
    const selectedToken = normalizeTokenText(String(tokenRow.token_text || ""));
    const selectedProbability = selectedProbabilityForToken(tokenRow);
    const merged = [{ token: selectedToken, probability: selectedProbability, selected: true }];
    const seen = new Set([selectedToken]);
    const alternatives = Array.isArray(tokenRow.top_logprob_alternatives)
      ? tokenRow.top_logprob_alternatives
      : [];
    for (const alt of alternatives) {
      if (!alt || typeof alt !== "object") continue;
      const altToken = normalizeTokenText(String(alt.token_text || ""));
      if (seen.has(altToken)) continue;
      seen.add(altToken);
      merged.push({
        token: altToken,
        probability: probabilityFromLogprob(alt.logprob),
        selected: false,
      });
    }
    return merged.slice(0, 5);
  }

  function selectedProbabilityForToken(tokenRow) {
    const probability = Number(tokenRow.selected_probability);
    if (Number.isFinite(probability)) {
      return Math.max(0, Math.min(1, probability));
    }
    return probabilityFromLogprob(tokenRow.selected_logprob);
  }

  function tokenTextForStrip(tokenText) {
    const raw = String(tokenText || "");
    if (!raw.length) return String.fromCharCode(8203);
    return raw;
  }

  function renderTokenAlternatives(tokenRow) {
    const rows = tokenAlternativeRows(tokenRow)
      .map((alt) => {
        const probability = Number(alt.probability || 0);
        const selectedClass = alt.selected ? " selected" : "";
        const fill = `${Math.round(Math.max(0, Math.min(1, probability)) * 100)}%`;
        const swatch = alt.selected ? "var(--accent-soft)" : probColor(probability);
        return `
          <div class="alt-row${selectedClass}" style="--fill:${fill};">
            <span class="alt-swatch" style="background:${swatch};"></span>
            <span class="alt-token">${esc(alt.token)}</span>
            <span class="alt-prob">${formatProbabilityPercent(probability)}</span>
          </div>
        `;
      })
      .join("");
    return rows || `
      <div class="alt-row">
        <span class="alt-swatch"></span>
        <span class="alt-token">no alternatives</span>
        <span class="alt-prob">-</span>
      </div>
    `;
  }

  function renderChoiceTokenStrip(choice, options = {}) {
    const showHint = options.showHint !== false;
    const tokenRows = Array.isArray(choice.tokens) ? choice.tokens : [];
    if (!tokenRows.length) {
      return "<p class='muted'>No token probability rows.</p>";
    }
    const chips = tokenRows.map((tokenRow) => {
      const tokenId = `hover-token-${tokenHoverSeq}`;
      tokenHoverSeq += 1;
      tokenHoverRowsById.set(tokenId, tokenRow);
      const probability = selectedProbabilityForToken(tokenRow);
      const tokenText = tokenTextForStrip(tokenRow.token_text);
      return `<span class="gen-token" data-hover-token-id="${esc(tokenId)}" style="background:${probColor(probability)};color:#0f172a;">${esc(tokenText)}</span>`;
    });
    const hint = showHint
      ? `<div class="token-strip-hint">Hover a token for top alternatives.</div>`
      : "";
    return `<div class="token-strip-wrap"><div class="token-strip">${chips.join("")}</div>${hint}</div>`;
  }

  function tooltipNode() {
    let node = document.getElementById("token-hover-tooltip");
    if (node) return node;
    node = document.createElement("div");
    node.id = "token-hover-tooltip";
    node.className = "token-hover-tooltip";
    document.body.appendChild(node);
    return node;
  }

  function positionTokenTooltip(mouseEvent) {
    const tooltip = tooltipNode();
    const pad = 14;
    const rect = tooltip.getBoundingClientRect();
    const left = Math.min(window.innerWidth - rect.width - pad, mouseEvent.clientX + pad);
    const top = Math.min(window.innerHeight - rect.height - pad, mouseEvent.clientY + pad);
    tooltip.style.left = `${Math.max(pad, left)}px`;
    tooltip.style.top = `${Math.max(pad, top)}px`;
  }

  function showTokenTooltip(mouseEvent, tokenRow) {
    const tooltip = tooltipNode();
    const selectedProbability = selectedProbabilityForToken(tokenRow);
    const entropyText = Number.isFinite(Number(tokenRow.selected_entropy))
      ? Number(tokenRow.selected_entropy).toFixed(3)
      : "-";
    const tokenText = normalizeTokenText(String(tokenRow.token_text || ""));
    const altGrid = renderTokenAlternatives(tokenRow);
    tooltip.innerHTML = `
      <div class="token-alt-head">
        <span>Selected</span>
        <code>${esc(tokenText)}</code>
        <span>${formatProbabilityPercent(selectedProbability)} · H=${entropyText}</span>
      </div>
      <div class="alt-grid">${altGrid}</div>
    `;
    tooltip.classList.add("visible");
    positionTokenTooltip(mouseEvent);
  }

  function hideTokenTooltip() {
    const node = document.getElementById("token-hover-tooltip");
    if (!node) return;
    node.classList.remove("visible");
  }

  function bindTokenHoverTooltips() {
    inspectorHost.querySelectorAll("[data-hover-token-id]").forEach((tokenNode) => {
      tokenNode.addEventListener("mouseenter", (event) => {
        const tokenId = tokenNode.getAttribute("data-hover-token-id");
        if (!tokenId) return;
        const tokenRow = tokenHoverRowsById.get(tokenId);
        if (!tokenRow) return;
        showTokenTooltip(event, tokenRow);
      });
      tokenNode.addEventListener("mousemove", (event) => {
        const tooltip = document.getElementById("token-hover-tooltip");
        if (!tooltip || !tooltip.classList.contains("visible")) return;
        positionTokenTooltip(event);
      });
      tokenNode.addEventListener("mouseleave", () => {
        hideTokenTooltip();
      });
    });
  }

  function renderRequestDetails(details) {
    const rows = [];
    rows.push(detailRow("request_id", String(details.request_id || "")));
    rows.push(detailRow("request_kind", String(details.request_kind || "")));
    rows.push(detailRow("stream", String(details.request_stream_id || "")));
    rows.push(detailRow("base_prefix_tokens", String(details.base_prefix_token_count || 0)));
    rows.push(detailRow("delta_token_count", String(details.delta_token_count || 0)));
    rows.push(detailRow("input_token_count", String(details.current_input_token_count || 0)));
    rows.push(`<h4 style=\"margin:0.2rem 0\">assistant_prefix_tail</h4>`);
    rows.push(`<pre>${esc(String(details.assistant_prefix_tail || ""))}</pre>`);
    return rows.join("");
  }

  function renderResponseDetails(details) {
    const rows = [];
    tokenHoverRowsById.clear();
    tokenHoverSeq = 0;
    rows.push(detailRow("request_id", String(details.request_id || "")));
    rows.push(detailRow("status", String(details.status || "")));
    rows.push(detailRow("latency_s", String(details.latency_seconds || "")));
    if (details.error_message) rows.push(detailRow("error", String(details.error_message)));
    const choices = Array.isArray(details.choices) ? details.choices : [];
    if (!choices.length) {
      rows.push("<p class='muted'>No response choices.</p>");
      return rows.join("");
    }
    choices.forEach((choice, index) => {
      rows.push(`<h4 style=\"margin:0.25rem 0\">choice ${index}</h4>`);
      rows.push(detailRow("finish", String(choice.finish_reason || "")));
      rows.push(detailRow("stop", String(choice.stop_reason || "")));
      rows.push(detailRow("output_tokens", String(choice.output_token_count || 0)));
      rows.push(`<h4 style="margin:0.2rem 0">Generated tokens</h4>`);
      rows.push(renderChoiceTokenStrip(choice));
    });
    return rows.join("");
  }

  function renderVllmStepDetails(details) {
    const rows = [];
    rows.push('<h4 style="margin:0.2rem 0">Request</h4>');
    rows.push(renderRequestDetails(details));
    rows.push('<h4 style="margin:0.25rem 0">Response</h4>');
    rows.push(renderResponseDetails(details));
    return rows.join("");
  }

  function normalizeClusterName(name) {
    return String(name || "")
      .replaceAll("_", " ")
      .trim();
  }

  function candidateMapForBranchPoint(branchPointId) {
    const details = candidatePoolByBranchPoint.get(String(branchPointId || ""));
    const rows = details && Array.isArray(details.candidates) ? details.candidates : [];
    const byId = new Map();
    for (const candidate of rows) {
      byId.set(String(candidate.candidate_id ?? ""), candidate);
    }
    return byId;
  }

  function renderClusterCandidateCard({ candidateId, selected, candidate }) {
    const selectedClass = selected ? " selected" : "";
    const selectedTag = selected ? "selected" : "not selected";
    if (!candidate) {
      return `
        <section class="candidate-panel${selectedClass}">
          <div class="candidate-head">
            <span>candidate ${esc(String(candidateId))}</span>
            <span>${esc(selectedTag)}</span>
          </div>
          <div class="hidden-note">Candidate details unavailable in this event window.</div>
        </section>
      `;
    }
    const tokenCount = String(candidate.output_token_count || 0);
    const finish = String(candidate.finish_reason || "");
    const stop = String(candidate.stop_reason || "");
    const candidateText = String(candidate.text || "");
    const strip = renderChoiceTokenStrip(
      { tokens: candidate.tokens || [] },
      { showHint: false },
    );
    return `
      <section class="candidate-panel${selectedClass}">
        <div class="candidate-head">
          <span>candidate ${esc(String(candidateId))}</span>
          <span>${esc(selectedTag)}</span>
        </div>
        <div class="detail-kv">
          ${detailRow("tokens", tokenCount)}
          ${detailRow("finish", finish)}
          ${detailRow("stop", stop)}
        </div>
        <pre class="candidate-text">${esc(candidateText)}</pre>
        ${strip}
      </section>
    `;
  }

  function renderClusterModeDetails({ modeName, groups, selectedIds, candidateById }) {
    if (!groups.length) {
      return `<p class="muted">No clusters logged for ${esc(String(modeName))}.</p>`;
    }
    const selectedSet = new Set(selectedIds.map((value) => String(value)));
    const totalCount = groups.reduce((sum, group) => {
      const groupCount = Number(group.candidate_count);
      if (Number.isFinite(groupCount) && groupCount > 0) return sum + groupCount;
      const ids = Array.isArray(group.candidate_ids) ? group.candidate_ids.length : 0;
      return sum + ids;
    }, 0);
    const rows = groups.map((group, groupIndex) => {
      const candidateIds = Array.isArray(group.candidate_ids) ? group.candidate_ids : [];
      const selectedInGroup = Array.isArray(group.selected_candidate_ids)
        ? group.selected_candidate_ids
        : candidateIds.filter((candidateId) => selectedSet.has(String(candidateId)));
      const selectedCount = selectedInGroup.length;
      const groupCountRaw = Number(group.candidate_count);
      const groupCount = Number.isFinite(groupCountRaw) && groupCountRaw > 0
        ? groupCountRaw
        : candidateIds.length;
      const fill = totalCount > 0 ? Math.max(0, Math.min(100, (groupCount / totalCount) * 100)) : 0;
      const topClass = selectedCount > 0 ? " top-cluster" : "";
      const openAttr = "";
      const clusterNameRaw = normalizeClusterName(group.cluster_name);
      const clusterName = clusterNameRaw || `Cluster ${groupIndex + 1}`;
      const candidateCards = candidateIds.map((candidateId) => renderClusterCandidateCard({
        candidateId,
        selected: selectedSet.has(String(candidateId)),
        candidate: candidateById.get(String(candidateId)),
      })).join("");
      const body = candidateCards || `<div class="hidden-note">No candidate members logged.</div>`;
      return `
        <details class="cluster-row${topClass}"${openAttr} style="--fill:${fill.toFixed(2)}%;">
          <summary class="cluster-head">
            <div class="cluster-name-wrap">
              <span class="cluster-caret">&gt;</span>
              <div class="cluster-name">${esc(clusterName)}</div>
            </div>
            <div class="cluster-tags">
              <span class="cluster-count">${esc(String(groupCount))} total</span>
              <span class="cluster-selected">${esc(String(selectedCount))} selected</span>
            </div>
          </summary>
          <div class="cluster-body">${body}</div>
        </details>
      `;
    });
    const selectedModeCount = selectedSet.size;
    return `
      <div class="cluster-mode-meta">${esc(String(selectedModeCount))} selected in ${esc(String(modeName))}</div>
      <div class="cluster-grid">${rows.join("")}</div>
    `;
  }

  function renderSelectorAppliedDetails(details) {
    const rows = [];
    tokenHoverRowsById.clear();
    tokenHoverSeq = 0;
    rows.push(detailRow("branch_point_id", String(details.branch_point_id || "")));
    rows.push(detailRow("node_id", String(details.node_id || "")));
    rows.push(detailRow("active_selector_mode", String(details.active_selector_mode || "")));
    const selectedIds = Array.isArray(details.selected_candidate_ids)
      ? details.selected_candidate_ids
      : [];
    rows.push(detailRow("selected_candidate_ids", selectedIds.join(", ")));
    const selectedByMode = details.selected_by_mode && typeof details.selected_by_mode === "object"
      ? details.selected_by_mode
      : {};
    const clusterGroupsByMode = details.cluster_groups_by_mode && typeof details.cluster_groups_by_mode === "object"
      ? details.cluster_groups_by_mode
      : {};
    const clusterModes = Object.keys(clusterGroupsByMode).sort();
    if (!clusterModes.length) {
      rows.push("<p class='muted'>No cluster summaries found in this selector event.</p>");
      return rows.join("");
    }
    const candidateById = candidateMapForBranchPoint(details.branch_point_id);
    for (const modeName of clusterModes) {
      const groups = Array.isArray(clusterGroupsByMode[modeName])
        ? clusterGroupsByMode[modeName]
        : [];
      const modeSelected = Array.isArray(selectedByMode[modeName])
        ? selectedByMode[modeName]
        : [];
      rows.push(`<h4 style="margin:0.38rem 0 0.12rem">Clusters · ${esc(String(modeName))}</h4>`);
      rows.push(renderClusterModeDetails({
        modeName,
        groups,
        selectedIds: modeSelected,
        candidateById,
      }));
    }
    return rows.join("");
  }

  function renderCandidatePoolDetails(details) {
    const rows = [];
    tokenHoverRowsById.clear();
    tokenHoverSeq = 0;
    rows.push(detailRow("branch_point_id", String(details.branch_point_id || "")));
    rows.push(detailRow("candidate_pool_id", String(details.candidate_pool_id || "")));
    rows.push(detailRow("trigger_type", String(details.trigger_type || "")));
    rows.push(detailRow("loaded_from_cache", String(Boolean(details.loaded_from_cache))));
    const selectedIds = Array.isArray(details.selected_candidate_ids)
      ? details.selected_candidate_ids
      : [];
    rows.push(detailRow("selected_candidate_ids", selectedIds.join(", ")));
    const candidates = Array.isArray(details.candidates) ? details.candidates : [];
    if (!candidates.length) {
      rows.push("<p class='muted'>No candidate rows logged.</p>");
      return rows.join("");
    }
    const cards = candidates.map((candidate) => {
      const candidateId = String(candidate.candidate_id ?? "");
      const selectedClass = candidate.selected ? " selected" : "";
      const selectedTag = candidate.selected ? "chosen" : "not chosen";
      const candidateText = String(candidate.text || "");
      const tokenCount = String(candidate.output_token_count || 0);
      const finish = String(candidate.finish_reason || "");
      const stop = String(candidate.stop_reason || "");
      const strip = renderChoiceTokenStrip({ tokens: candidate.tokens || [] });
      return `
        <section class="candidate-card${selectedClass}">
          <div class="candidate-head">
            <span>candidate ${esc(candidateId)}</span>
            <span>${esc(selectedTag)}</span>
          </div>
          <div class="detail-kv">
            ${detailRow("tokens", tokenCount)}
            ${detailRow("finish", finish)}
            ${detailRow("stop", stop)}
          </div>
          <pre class="candidate-text">${esc(candidateText)}</pre>
          ${strip}
        </section>
      `;
    });
    rows.push(`<div class=\"candidate-list\">${cards.join("")}</div>`);
    return rows.join("");
  }

  function renderEventInspector(eventId) {
    const event = eventById.get(String(eventId));
    if (!event) {
      if (selectedNodeId) {
        renderNodeInspector(selectedNodeId);
        return;
      }
      inspectorHost.innerHTML = "<p class='muted'>Select a node or event.</p>";
      return;
    }
    const details = event.details && typeof event.details === "object" ? event.details : {};
    const rows = [];
    rows.push(`<h3>${esc(String(event.event_type || "event"))}</h3>`);
    rows.push(detailRow("node", String(event.node_id || "")));
    rows.push(detailRow("event_index", String(event.event_index || "")));
    rows.push(detailRow("summary", String(event.summary || "")));
    rows.push(detailRow("timestamp", String(event.timestamp_utc || "")));
    if (event.event_type === "vllm_step") {
      rows.push(renderVllmStepDetails(details));
    } else if (event.event_type === "candidate_pool_resolved") {
      rows.push(renderCandidatePoolDetails(details));
    } else if (event.event_type === "selector_applied") {
      rows.push(renderSelectorAppliedDetails(details));
    } else {
      rows.push(`<pre>${esc(JSON.stringify(withoutRawTokenIds(details), null, 2))}</pre>`);
    }
    rows.push("<button class='event-btn' id='back-to-node'>Back to node events</button>");
    inspectorHost.innerHTML = rows.join("");
    if (
      event.event_type === "vllm_step"
      || event.event_type === "candidate_pool_resolved"
      || event.event_type === "selector_applied"
    ) {
      bindTokenHoverTooltips();
    } else {
      hideTokenTooltip();
    }
    const backButton = document.getElementById("back-to-node");
    if (backButton) {
      backButton.addEventListener("click", () => {
        selectedEventId = null;
        renderAll();
      });
    }
  }

  function renderInspector() {
    if (selectedEventId) {
      renderEventInspector(selectedEventId);
      return;
    }
    if (selectedNodeId) {
      renderNodeInspector(selectedNodeId);
      return;
    }
    hideTokenTooltip();
    inspectorHost.innerHTML = "<p class='muted'>Click a node or event chip to inspect details.</p>";
  }

  function syncModeButtons() {
    document.querySelectorAll(".mode-btn[data-mode]").forEach((button) => {
      const buttonMode = button.getAttribute("data-mode");
      if (!buttonMode) return;
      button.classList.toggle("active", buttonMode === currentMode);
    });
  }

  function clampInspectorWidth(widthValue) {
    const layoutWidth = Math.max(520, asNumber(workspaceLayout.clientWidth || window.innerWidth));
    const minWidth = 260;
    const maxWidth = Math.max(minWidth, Math.min(760, layoutWidth - 320));
    return Math.max(minWidth, Math.min(maxWidth, widthValue));
  }

  function initWorkspaceSplitter() {
    if (!workspaceDivider || !inspectorPane) return;
    const widthStorageKey = "branching_viz_inspector_width";
    let dragState = null;

    function applyDesktopWidth(widthValue) {
      const clampedWidth = clampInspectorWidth(widthValue);
      workspaceLayout.style.setProperty("--inspector-width", `${clampedWidth}px`);
    }

    function applyResponsiveWidth() {
      if (window.matchMedia("(max-width: 980px)").matches) {
        workspaceLayout.style.removeProperty("--inspector-width");
        return;
      }
      const savedWidth = Number(window.localStorage.getItem(widthStorageKey));
      if (Number.isFinite(savedWidth)) {
        applyDesktopWidth(savedWidth);
        return;
      }
      applyDesktopWidth(380);
    }

    workspaceDivider.addEventListener("pointerdown", (event) => {
      if (window.matchMedia("(max-width: 980px)").matches) return;
      dragState = {
        startX: asNumber(event.clientX),
        startWidth: asNumber(inspectorPane.getBoundingClientRect().width),
      };
      workspaceLayout.classList.add("resizing");
      workspaceDivider.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    workspaceDivider.addEventListener("pointermove", (event) => {
      if (!dragState) return;
      const delta = dragState.startX - asNumber(event.clientX);
      applyDesktopWidth(dragState.startWidth + delta);
      renderTree();
    });

    function stopDrag(pointerId) {
      if (!dragState) return;
      dragState = null;
      workspaceLayout.classList.remove("resizing");
      const widthValue = clampInspectorWidth(
        asNumber(inspectorPane.getBoundingClientRect().width),
      );
      workspaceLayout.style.setProperty("--inspector-width", `${widthValue}px`);
      window.localStorage.setItem(widthStorageKey, String(Math.round(widthValue)));
      if (workspaceDivider.hasPointerCapture(pointerId)) {
        workspaceDivider.releasePointerCapture(pointerId);
      }
      renderAll();
    }

    workspaceDivider.addEventListener("pointerup", (event) => {
      stopDrag(event.pointerId);
    });
    workspaceDivider.addEventListener("pointercancel", (event) => {
      stopDrag(event.pointerId);
    });
    window.addEventListener("resize", () => {
      applyResponsiveWidth();
      renderAll();
    });
    applyResponsiveWidth();
  }

  function syncInspectorHeight() {
    if (!inspectorPane) return;
    if (window.matchMedia("(max-width: 980px)").matches) {
      inspectorPane.style.height = "";
      inspectorPane.style.maxHeight = "";
      return;
    }
    const treeCard = document.querySelector(".canvas-stack .canvas-card");
    if (!treeCard) return;
    const targetHeight = Math.max(260, Math.round(asNumber(treeCard.getBoundingClientRect().height)));
    inspectorPane.style.height = `${targetHeight}px`;
    inspectorPane.style.maxHeight = `${targetHeight}px`;
  }

  function renderAll() {
    syncModeButtons();
    renderTree();
    syncInspectorHeight();
    renderInspector();
  }

  document.querySelectorAll(".mode-btn[data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      const mode = button.getAttribute("data-mode");
      if (!mode || !(mode in modeKey)) return;
      currentMode = mode;
      renderAll();
    });
  });

  initWorkspaceSplitter();
  renderAll();
})();
"""
