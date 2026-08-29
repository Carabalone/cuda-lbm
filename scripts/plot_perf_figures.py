#!/usr/bin/env python3
"""Plot whitepaper §5.2 performance figures across benchmark campaigns."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"

CAMPAIGNS = [
    {
        "id": "thesis",
        "title": "Pre-thesis (whitepaper)",
        "path": ROOT / "benchmark_data" / "thesis_whitepaper_128.csv",
        "color": "#888888",
    },
    {
        "id": "gtx1050ti",
        "title": "Post-thesis (GTX 1050 Ti)",
        "path": ROOT / "benchmark_results_gtx1050ti_128.txt",
        "color": "#4477AA",
    },
    {
        "id": "rtx2080",
        "title": "Current (RTX 2080)",
        "path": ROOT / "benchmark_results_nvidia_geforce_rtx_2080_128.txt",
        "color": "#EE7733",
    },
]

LAYOUT_ORDER = ["AOS", "CSOA", "SOA"]
LAYOUT_LABELS = {"AOS": "AoS", "CSOA": "CSoA", "SOA": "SoA"}
OP_ORDER = ["BGK", "CM", "MRT"]
OP_LABELS = {"BGK": "BGK", "CM": "CM", "MRT": "MRT opt"}

SLOWDOWN_WITH_BASELINE = [
    ("unop_MRT", "Unoptimized MRT"),
    ("MRT_opt_SoA", "MRT opt (SoA)"),
    ("CM_SoA", "CM (SoA)"),
    ("BGK_SoA", "BGK (SoA)"),
]

SLOWDOWN_NO_BASELINE = [
    ("MRT_opt_SoA", "MRT opt (SoA)"),
    ("CM_SoA", "CM (SoA)"),
    ("BGK_SoA", "BGK (SoA)"),
]


def read_campaign(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: dict[str, float] = {}
    with path.open(newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("label,"):
                continue
            parts = next(csv.reader([line]))
            if len(parts) < 2:
                continue
            label = parts[0]
            mlups_raw = parts[-1]
            if mlups_raw in ("", "BUILD_FAIL", "WARNING", "TIMEOUT"):
                continue
            try:
                rows[label] = float(mlups_raw)
            except ValueError:
                continue
    return rows


def label_to_key(label: str) -> tuple[str, str] | None:
    m = re.match(r"^(BGK|CM|MRT_opt)_(AoS|CSoA|SoA)$", label)
    if not m:
        return None
    op_raw, layout_raw = m.group(1), m.group(2)
    op = "MRT" if op_raw == "MRT_opt" else op_raw
    layout = {"AoS": "AOS", "CSoA": "CSOA", "SoA": "SOA"}[layout_raw]
    return op, layout


def slowdown_vs_bgk(data: dict[str, float], labels: list[tuple[str, str]]) -> tuple[list[str], list[float]]:
    ref = data.get("BGK_SoA")
    names: list[str] = []
    values: list[float] = []
    for key, name in labels:
        val = data.get(key)
        if ref and val and val > 0:
            names.append(name)
            values.append(ref / val)
    return names, values


def layout_matrix(data: dict[str, float]) -> np.ndarray:
    mat = np.full((len(OP_ORDER), len(LAYOUT_ORDER)), np.nan)
    for label, mlups in data.items():
        parsed = label_to_key(label)
        if not parsed:
            continue
        op, layout = parsed
        mat[OP_ORDER.index(op), LAYOUT_ORDER.index(layout)] = mlups
    return mat


def plot_fig56(campaigns_with_data: list[tuple[dict, dict]], out_path: Path, include_baseline: bool) -> None:
    labels_spec = SLOWDOWN_WITH_BASELINE if include_baseline else SLOWDOWN_NO_BASELINE
    n = len(campaigns_with_data)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5), sharey=True, squeeze=False)
    title = "Fig 5.6 — slowdown vs optimized BGK (SoA)"
    if not include_baseline:
        title = "Fig 5.7 — slowdown vs optimized BGK (no baseline)"

    ymax = 1.0
    for _, camp in campaigns_with_data:
        names, vals = slowdown_vs_bgk(camp["data"], labels_spec)
        if vals:
            ymax = max(ymax, max(vals) * 1.08)

    for ax, (meta, camp) in zip(axes[0], campaigns_with_data):
        names, vals = slowdown_vs_bgk(camp["data"], labels_spec)
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color=meta["color"], edgecolor="black", linewidth=0.4)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_title(meta["title"], fontsize=10)
        ax.set_ylim(0, ymax)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{val:.2f}×",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if ax is axes[0, 0]:
            ax.set_ylabel("Slowdown (× vs BGK SoA)")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_fig58(campaigns_with_data: list[tuple[dict, dict]], out_path: Path) -> None:
    n = len(campaigns_with_data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), sharey=False, squeeze=False)

    for ax, (meta, camp) in zip(axes[0], campaigns_with_data):
        mat = layout_matrix(camp["data"])
        x = np.arange(len(LAYOUT_ORDER))
        width = 0.24
        for i, op in enumerate(OP_ORDER):
            offset = (i - 1) * width
            vals = mat[i]
            ax.bar(x + offset, vals, width=width, label=OP_LABELS[op], edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([LAYOUT_LABELS[l] for l in LAYOUT_ORDER])
        ax.set_title(meta["title"], fontsize=10)
        ax.set_xlabel("Memory layout")
        ax.set_ylabel("MLUPS")
        ax.legend(fontsize=8)

    fig.suptitle("Fig 5.8 — layout × collision operator (128³ TGV)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_evolution(campaigns_with_data: list[tuple[dict, dict]], out_path: Path) -> None:
    """BGK SoA MLUPS across campaigns (absolute throughput growth)."""
    labels = []
    vals = []
    colors = []
    for meta, camp in campaigns_with_data:
        bgk = camp["data"].get("BGK_SoA")
        if bgk is None:
            continue
        labels.append(meta["title"].replace("\n", " "))
        vals.append(bgk)
        colors.append(meta["color"])

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("MLUPS (BGK SoA, 128³)")
    ax.set_title("Optimized BGK SoA throughput across campaigns")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(campaigns_with_data: list[tuple[dict, dict]], out_path: Path) -> None:
    lines = ["# Benchmark comparison summary (128³ TGV)", ""]
    lines.append("| Campaign | BGK SoA | CM SoA | MRT SoA | CM/BGK | MRT/BGK |")
    lines.append("|----------|--------:|-------:|--------:|-------:|--------:|")
    for meta, camp in campaigns_with_data:
        d = camp["data"]
        bgk = d.get("BGK_SoA")
        cm = d.get("CM_SoA")
        mrt = d.get("MRT_opt_SoA")
        if not bgk:
            continue
        cm_ratio = f"{bgk / cm:.2f}×" if cm else "—"
        mrt_ratio = f"{bgk / mrt:.2f}×" if mrt else "—"
        lines.append(
            f"| {meta['title']} | {bgk:.2f} | {cm:.2f} | {mrt:.2f} | {cm_ratio} | {mrt_ratio} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGDIR.mkdir(exist_ok=True)
    loaded: list[tuple[dict, dict]] = []
    for meta in CAMPAIGNS:
        data = read_campaign(meta["path"])
        if data:
            loaded.append((meta, {"data": data}))
            print(f"Loaded {len(data)} rows from {meta['path'].name}")
        else:
            print(f"Skip (missing/empty): {meta['path']}")

    if not loaded:
        raise SystemExit("No benchmark CSV files found.")

    plot_fig56(loaded, FIGDIR / "fig_5_6_slowdown_with_baseline.png", include_baseline=True)
    plot_fig56(loaded, FIGDIR / "fig_5_7_slowdown_no_baseline.png", include_baseline=False)
    plot_fig58(loaded, FIGDIR / "fig_5_8_layout_collision_128.png")
    plot_speedup_evolution(loaded, FIGDIR / "fig_bgk_soa_evolution.png")
    write_summary_table(loaded, FIGDIR / "benchmark_comparison_summary.md")
    print(f"Wrote figures to {FIGDIR}/")


if __name__ == "__main__":
    main()
