from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

BIN_COUNT = 10
HIST_COLOR = "#2b6cb0"
SOURCE_COLORS = {
    "combined": "#111827",
    "AceReason-Math": "#2563eb",
    "DAPO-Math-17k-Processed": "#dc2626",
    "MathSub-30K": "#059669",
    "omega-combined-no-boxed": "#7c3aed",
    "rlvr_orz_math_57k_collected": "#d97706",
}


@dataclass(frozen=True)
class PlotPaths:
    """Output locations for passrate distribution figures.

    Args:
        output_dir: Directory where plots should be written.

    Returns:
        Dataclass with canonical output file paths.

    Example:
        >>> paths = PlotPaths(output_dir=Path("output/plots"))
        >>> paths.histogram_path.name
        'passrate_histograms_by_source.png'
    """

    output_dir: Path

    @property
    def histogram_path(self) -> Path:
        """Return the faceted histogram output path."""

        return self.output_dir / "passrate_histograms_by_source.png"

    @property
    def ecdf_path(self) -> Path:
        """Return the ECDF overlay output path."""

        return self.output_dir / "passrate_ecdf_overlay.png"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for passrate plotting."""

    parser = argparse.ArgumentParser(description="Plot passrate distributions for the RL training parquet.")
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_passrate_frame(*, parquet_path: Path) -> pd.DataFrame:
    """Load the passrate/source columns needed for plotting.

    Args:
        parquet_path: Path to the exported RL parquet.

    Returns:
        DataFrame containing `passrate` and `source_family`.
    """

    frame = pd.read_parquet(parquet_path, columns=["passrate", "source_family"])
    return frame.dropna(subset=["passrate"]).copy()


def ordered_sources(*, frame: pd.DataFrame) -> list[str]:
    """Return the plotting order for source families plus combined."""

    sources = sorted(frame["source_family"].dropna().unique().tolist())
    return ["combined", *sources]


def configure_percent_axis(*, axis: Axes) -> None:
    """Apply consistent percent-style y-axis formatting."""

    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.grid(axis="y", alpha=0.25)


def histogram_series(*, frame: pd.DataFrame, source_name: str) -> pd.Series:
    """Return passrates for one source family or the combined dataset."""

    if source_name == "combined":
        return cast(pd.Series, frame["passrate"]).astype(float)
    return cast(pd.Series, frame.loc[frame["source_family"] == source_name, "passrate"]).astype(float)


def draw_histogram_panel(*, axis: Axes, frame: pd.DataFrame, source_name: str, bins: list[float]) -> None:
    """Draw one histogram panel for a source-family passrate distribution."""

    series = histogram_series(frame=frame, source_name=source_name)
    weights = [1.0 / len(series)] * len(series)
    axis.hist(
        series,
        bins=bins,
        weights=weights,
        color=HIST_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_title(f"{source_name} (n={len(series)})")
    axis.set_xlabel("Passrate")
    axis.set_ylabel("Share of rows")
    axis.set_xlim(0, 1)
    axis.xaxis.set_ticks([index / 10 for index in range(11)])
    axis.set_yscale("log")
    configure_percent_axis(axis=axis)


def build_histogram_figure(*, frame: pd.DataFrame) -> Figure:
    """Build faceted binned histograms for combined and per-source passrates."""

    bins = np.linspace(0.0, 1.0, num=BIN_COUNT + 1, dtype=float).tolist()
    bins[-1] = 1.0000001
    sources = ordered_sources(frame=frame)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axis_array = cast(np.ndarray, axes)
    for axis, source_name in zip(axis_array.flat, sources):
        draw_histogram_panel(axis=axis, frame=frame, source_name=source_name, bins=bins)
    figure.suptitle("Passrate Histograms by Source Family", fontsize=16)
    return figure


def ecdf_xy(*, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted x/y vectors for an empirical CDF."""

    sorted_values = series.sort_values(ignore_index=True)
    x_values = sorted_values.to_numpy(dtype=float, copy=False)
    cumulative = np.arange(1, len(x_values) + 1, dtype=float) / len(x_values)
    return x_values, cumulative


def build_ecdf_figure(*, frame: pd.DataFrame) -> Figure:
    """Build an ECDF overlay comparing combined and per-source passrates."""

    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis = cast(Axes, axis)
    for source_name in ordered_sources(frame=frame):
        series = histogram_series(frame=frame, source_name=source_name)
        x_values, y_values = ecdf_xy(series=series)
        color = SOURCE_COLORS.get(source_name, None)
        axis.step(x_values, y_values, where="post", linewidth=2, label=source_name, color=color)
    axis.set_title("Passrate ECDF Overlay")
    axis.set_xlabel("Passrate")
    axis.set_ylabel("Cumulative share of rows")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.xaxis.set_ticks([index / 10 for index in range(11)])
    configure_percent_axis(axis=axis)
    axis.legend(loc="lower right", fontsize=9)
    return figure


def save_figure(*, figure: Figure, path: Path) -> None:
    """Write one matplotlib figure to disk and close it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Render histogram and ECDF passrate plots for the RL parquet."""

    args = parse_args()
    paths = PlotPaths(output_dir=args.output_dir)
    frame = load_passrate_frame(parquet_path=args.parquet_path)
    save_figure(figure=build_histogram_figure(frame=frame), path=paths.histogram_path)
    save_figure(figure=build_ecdf_figure(frame=frame), path=paths.ecdf_path)
    print(paths.histogram_path)
    print(paths.ecdf_path)


if __name__ == "__main__":
    main()
