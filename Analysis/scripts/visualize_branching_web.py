"""Small HTML/CSS helpers for branching event-log visualization pages."""

from __future__ import annotations

from html import escape


def theme_css() -> str:
    """Return shared CSS theme used by generated visualization pages.

    Args:
        None.

    Returns:
        CSS stylesheet text.
    """

    return """
:root {
  --bg0: #0c111c;
  --bg1: #121a2b;
  --bg2: #182033;
  --panel: rgba(20, 28, 44, 0.62);
  --panel-border: rgba(134, 157, 196, 0.35);
  --text: #eef3ff;
  --muted: #b8c6e5;
  --accent: #ff7a59;
  --accent-soft: #ffb199;
  --line: rgba(147, 171, 214, 0.3);
  --good: #60d394;
  --warn: #ffd166;
  --bad: #ef476f;
  --shadow: 0 22px 44px rgba(2, 8, 22, 0.45);
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  min-height: 100%;
  color: var(--text);
  font-family: "IBM Plex Sans", sans-serif;
  background:
    radial-gradient(circle at 12% 8%, rgba(255, 122, 89, 0.22), transparent 26%),
    radial-gradient(circle at 88% 12%, rgba(74, 111, 209, 0.27), transparent 28%),
    linear-gradient(145deg, var(--bg0), var(--bg1) 48%, var(--bg2));
  background-attachment: fixed;
}
.top-chrome {
  position: sticky;
  top: 0;
  z-index: 50;
  border-bottom: 1px solid var(--line);
  background: rgba(12, 17, 28, 0.86);
  backdrop-filter: blur(10px);
  overflow: hidden;
}
.chrome-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0.8rem 1rem;
  display: flex;
  gap: 0.8rem;
  align-items: center;
  min-width: 0;
}
.chrome-kicker {
  font-family: "IBM Plex Serif", serif;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent-soft);
  font-size: 0.78rem;
  flex: 0 0 auto;
}
.chrome-title {
  font-weight: 600;
  font-size: 1rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
  flex: 1 1 auto;
}
.chrome-sub {
  margin-left: auto;
  color: var(--muted);
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.76rem;
  max-width: min(48vw, 680px);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
  flex: 0 1 auto;
}
main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem;
  display: grid;
  gap: 1rem;
  min-width: 0;
}
.panel {
  border: 1px solid var(--panel-border);
  border-radius: 0.95rem;
  background: linear-gradient(170deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015));
  box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
  padding: 0.95rem;
  min-width: 0;
}
.grid-two {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 1rem;
}
.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.25rem 0.55rem;
  font-size: 0.76rem;
  font-family: "IBM Plex Mono", monospace;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.03);
}
.pill.good { color: #0f2d1f; background: var(--good); border-color: transparent; }
.pill.warn { color: #2c2406; background: var(--warn); border-color: transparent; }
.pill.bad { color: #2e0712; background: var(--bad); border-color: transparent; }
.muted { color: var(--muted); }
h1, h2, h3 {
  margin: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
}
h1 {
  font-family: "IBM Plex Serif", serif;
  font-size: clamp(1.4rem, 3vw, 2rem);
}
h2 {
  font-family: "IBM Plex Serif", serif;
  font-size: 1.12rem;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}
th, td {
  border-bottom: 1px solid var(--line);
  text-align: left;
  padding: 0.42rem 0.35rem;
  vertical-align: top;
}
th {
  color: var(--accent-soft);
  font-size: 0.73rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}
code, pre {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.77rem;
}
.path-row {
  margin: 0.4rem 0 0.85rem 0;
  display: flex;
  align-items: baseline;
  gap: 0.35rem;
  min-width: 0;
}
.path-label {
  flex: 0 0 auto;
}
.path-code {
  display: block;
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
a {
  color: var(--accent-soft);
  text-decoration: none;
}
a:hover { text-decoration: underline; }
svg {
  width: 100%;
  min-height: 320px;
  border-radius: 0.8rem;
  background: rgba(12, 17, 28, 0.55);
  border: 1px solid var(--line);
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
.legend-chip {
  border: 1px solid var(--line);
  border-radius: 0.6rem;
  padding: 0.2rem 0.45rem;
  font-size: 0.74rem;
  font-family: "IBM Plex Mono", monospace;
}
@media (max-width: 960px) {
  .grid-two { grid-template-columns: 1fr; }
  .chrome-inner { flex-wrap: wrap; }
  .chrome-sub {
    margin-left: 0;
    width: 100%;
    max-width: none;
    white-space: normal;
    overflow: visible;
    text-overflow: clip;
    overflow-wrap: anywhere;
  }
  main { padding: 0.7rem; }
}
"""


def wrap_page(
    *,
    title: str,
    subtitle: str,
    body_html: str,
    footer_text: str | None = None,
    script: str = "",
) -> str:
    """Wrap body HTML in a complete themed document shell.

    Args:
        title: Page title displayed in browser and top bar.
        subtitle: Subtitle text shown in top chrome.
        body_html: Main page HTML.
        footer_text: Optional footer text.
        script: Optional inline script tag contents.

    Returns:
        Complete HTML document string.
    """

    footer_html = (
        f"<p class='muted' style='margin:0'>{escape(footer_text)}</p>"
        if footer_text
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap" rel="stylesheet">
  <style>{theme_css()}</style>
</head>
<body>
  <header class="top-chrome">
    <div class="chrome-inner">
      <div class="chrome-kicker">Branching Eval</div>
      <div class="chrome-title">{escape(title)}</div>
      <div class="chrome-sub">{escape(subtitle)}</div>
    </div>
  </header>
  <main>
    {body_html}
    {footer_html}
  </main>
  <script>{script}</script>
</body>
</html>
"""
