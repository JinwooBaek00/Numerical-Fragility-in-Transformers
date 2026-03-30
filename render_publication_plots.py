from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_figures"

PRECISION_COLORS = {
    "bf16": "#C27A2C",
    "fp16": "#2E6FBB",
}

LENGTH_MARKERS = {
    128: "o",
    512: "s",
    1024: "^",
}

POLICY_COLORS = {
    "none": "#5B6570",
    "static_global": "#8A5FBF",
    "random_same_budget": "#D17C2F",
    "bgss": "#218457",
}

POLICY_LABELS = {
    "none": "None",
    "static_global": "Static global",
    "random_same_budget": "Random (same budget)",
    "bgss": "BGSS",
}

MECHANISM_LABELS = {
    "attention": "Attention",
    "layernorm": "LayerNorm",
    "residual": "Residual",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _save_figure_set(fig: plt.Figure, output_stem: Path, formats: Iterable[str]) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _precision_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6, label=precision.upper())
        for precision, color in PRECISION_COLORS.items()
    ]


def _length_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker=marker, linestyle="none", color="#444444", markerfacecolor="#DDDDDD", markeredgecolor="#444444", markersize=6, label=f"L={length}")
        for length, marker in LENGTH_MARKERS.items()
    ]


def _write_e1_publication_plots(output_root: Path, formats: Iterable[str]) -> list[Path]:
    outputs_dir = REPO_ROOT / "e1_controlled" / "outputs"
    summary_rows = _read_csv_rows(outputs_dir / "e1_controlled_summary_table.csv")
    attention_path = outputs_dir / "e1_controlled_attention_records.csv"
    layernorm_path = outputs_dir / "e1_controlled_layernorm_records.csv"
    residual_path = outputs_dir / "e1_controlled_residual_records.csv"
    written: list[Path] = []

    if attention_path.exists() and layernorm_path.exists() and residual_path.exists():
        attention_rows = _read_csv_rows(attention_path)
        layernorm_rows = _read_csv_rows(layernorm_path)
        residual_rows = _read_csv_rows(residual_path)

        fig, ax = plt.subplots(figsize=(3.25, 2.95))
        value_scales = sorted({_to_float(row, "value_scale") for row in attention_rows})
        scale_colors = {
            value_scale: color
            for value_scale, color in zip(value_scales, ["#2E6FBB", "#3CA370", "#C27A2C", "#C74E39"])
        }
        margin_markers = {
            0.25: "o",
            0.5: "s",
            1.0: "^",
            2.0: "D",
            4.0: "P",
        }
        for row in attention_rows:
            theory = _to_float(row, "theory_proxy")
            measured = _to_float(row, "measured_error")
            value_scale = _to_float(row, "value_scale")
            margin = _to_float(row, "margin")
            ax.scatter(
                theory,
                measured,
                s=42,
                marker=margin_markers.get(margin, "o"),
                color=scale_colors[value_scale],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.95,
            )
        max_xy = max(
            max(_to_float(row, "theory_proxy"), _to_float(row, "measured_error"))
            for row in attention_rows
        )
        min_xy = min(
            min(_to_float(row, "theory_proxy"), _to_float(row, "measured_error"))
            for row in attention_rows
            if _to_float(row, "theory_proxy") > 0.0 and _to_float(row, "measured_error") > 0.0
        )
        ax.plot([min_xy, max_xy], [min_xy, max_xy], color="#999999", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Attention proxy")
        ax.set_ylabel("Measured attention error")
        pearson = next(_to_float(row, "pearson") for row in summary_rows if row["mechanism"] == "attention")
        ax.text(
            0.03,
            0.97,
            f"Pearson = {pearson:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6, label=f"V={value_scale:g}")
            for value_scale, color in scale_colors.items()
        ]
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.5,
            title="Value scale",
        )
        written.extend(_save_figure_set(fig, output_root / "e1" / "e1_attention_proxy_scatter", formats))

        fig, ax = plt.subplots(figsize=(3.25, 2.55))
        eps_values = [_to_float(row, "epsilon") for row in layernorm_rows]
        measured_values = [_to_float(row, "measured_change") for row in layernorm_rows]
        theory_values = [_to_float(row, "theory_magnitude") for row in layernorm_rows]
        measured_max = max(measured_values) if measured_values else 1.0
        theory_max = max(theory_values) if theory_values else 1.0
        ax.plot(eps_values, [value / measured_max for value in measured_values], marker="o", color="#2E6FBB", linewidth=1.8, label="Measured")
        ax.plot(eps_values, [value / theory_max for value in theory_values], marker="s", color="#C27A2C", linewidth=1.8, label="Theory proxy")
        ax.set_xscale("log")
        ax.set_xlabel(r"LayerNorm $\epsilon$")
        ax.set_ylabel("Normalized magnitude")
        spearman = next(_to_float(row, "spearman") for row in summary_rows if row["mechanism"] == "layernorm")
        ax.text(
            0.03,
            0.97,
            f"Spearman = {spearman:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        ax.legend(frameon=False, loc="upper right")
        written.extend(_save_figure_set(fig, output_root / "e1" / "e1_layernorm_epsilon_curve", formats))

        fig, ax = plt.subplots(figsize=(3.25, 2.55))
        rhos = [_to_float(row, "rho") for row in residual_rows]
        measured_transport = [_to_float(row, "measured_transport") for row in residual_rows]
        transport_bound = [_to_float(row, "transport_bound") for row in residual_rows]
        ax.plot(rhos, measured_transport, marker="o", color="#2E6FBB", linewidth=1.8, label="Measured transport")
        ax.plot(rhos, transport_bound, marker="s", color="#C27A2C", linewidth=1.8, label="Theory bound")
        ax.fill_between(rhos, measured_transport, transport_bound, color="#C27A2C", alpha=0.12)
        ax.set_xlabel(r"Residual gain $\rho$")
        ax.set_ylabel("Transport factor")
        ax.legend(frameon=False, loc="upper left")
        written.extend(_save_figure_set(fig, output_root / "e1" / "e1_residual_transport_bound", formats))
        return written

    order = ["attention", "layernorm", "residual"]
    ordered_rows = sorted(summary_rows, key=lambda row: order.index(row["mechanism"]))
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    y_positions = list(range(len(ordered_rows)))
    for y, row in zip(y_positions, ordered_rows):
        pearson = _to_float(row, "pearson")
        spearman = _to_float(row, "spearman")
        ax.plot([pearson, spearman], [y, y], color="#B9C3CC", linewidth=2.0, zorder=1)
        ax.scatter(pearson, y, s=42, color="#2E6FBB", edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(spearman, y, s=42, marker="s", color="#C27A2C", edgecolor="white", linewidth=0.8, zorder=3)

    x_min = min(min(_to_float(row, "pearson"), _to_float(row, "spearman")) for row in ordered_rows)
    ax.set_xlim(max(0.85, x_min - 0.03), 1.01)
    ax.set_xlabel("Correlation with measured mechanism change")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([MECHANISM_LABELS[row["mechanism"]] for row in ordered_rows])
    ax.invert_yaxis()
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color="#2E6FBB", markerfacecolor="#2E6FBB", markersize=6, label="Pearson"),
            Line2D([0], [0], marker="s", linestyle="none", color="#C27A2C", markerfacecolor="#C27A2C", markersize=6, label="Spearman"),
        ],
        loc="lower left",
        frameon=False,
    )
    written.extend(_save_figure_set(fig, output_root / "e1" / "e1_local_mechanism_correlations", formats))
    return written


def _write_e2_publication_plots(output_root: Path, formats: Iterable[str]) -> list[Path]:
    support_rows = _read_csv_rows(REPO_ROOT / "e2_predictor" / "outputs" / "e2_predictor_support_summary.csv")
    trend_rows = _read_csv_rows(REPO_ROOT / "e2_predictor" / "outputs" / "e2_predictor_binned_trend.csv")
    written: list[Path] = []

    max_val = max(
        max(_to_float(row, "combined_pearson"), _to_float(row, "no_transport_pearson"))
        for row in support_rows
    )

    fig, ax = plt.subplots(figsize=(3.25, 3.65))
    for row in support_rows:
        precision = row["precision"]
        length = _to_int(row, "sequence_length")
        x_value = _to_float(row, "no_transport_pearson")
        y_value = _to_float(row, "combined_pearson")
        ax.scatter(
            x_value,
            y_value,
            s=50,
            marker=LENGTH_MARKERS[length],
            color=PRECISION_COLORS[precision],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )
    ax.plot([0.0, max_val + 0.04], [0.0, max_val + 0.04], color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, max_val + 0.04)
    ax.set_ylim(0.0, max_val + 0.04)
    ax.set_xlabel("No-transport Pearson")
    ax.set_ylabel("Combined Pearson")
    fig.text(0.08, 0.985, "Color:", ha="left", va="top", fontsize=7)
    fig.legend(
        handles=_precision_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.18, 0.99),
        frameon=False,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    fig.text(0.08, 0.93, "Marker:", ha="left", va="top", fontsize=7)
    fig.legend(
        handles=_length_legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.18, 0.935),
        frameon=False,
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    fig.subplots_adjust(top=0.82)
    written.extend(_save_figure_set(fig, output_root / "e2" / "e2_transport_vs_no_transport_pearson", formats))

    sorted_rows = sorted(support_rows, key=lambda row: _to_float(row, "delta_pearson_vs_no_transport"))
    fig, ax = plt.subplots(figsize=(3.25, 4.7))
    y_positions = list(range(len(sorted_rows)))
    for y, row in zip(y_positions, sorted_rows):
        delta = _to_float(row, "delta_pearson_vs_no_transport")
        precision = row["precision"]
        length = _to_int(row, "sequence_length")
        ax.scatter(
            delta,
            y,
            s=50,
            marker=LENGTH_MARKERS[length],
            color=PRECISION_COLORS[precision],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.axvline(0.0, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$\Delta$ Pearson vs no transport")
    ax.set_ylabel("Run")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            f"{row['precision']} / L={_to_int(row, 'sequence_length')} / s{_to_int(row, 'seed')}"
            for row in sorted_rows
        ]
    )
    legend1 = ax.legend(handles=_precision_legend_handles(), loc="lower right", frameon=False, title="Precision")
    ax.add_artist(legend1)
    ax.legend(handles=_length_legend_handles(), loc="upper left", frameon=False, title="Sequence")
    written.extend(_save_figure_set(fig, output_root / "e2" / "e2_transport_gain_delta_pearson", formats))

    trend_rows = sorted(trend_rows, key=lambda row: _to_float(row, "bin_index"))
    pred_values = [_to_float(row, "pred_mean") for row in trend_rows]
    mismatch_values = [_to_float(row, "mismatch_mean") for row in trend_rows]
    pred_max = max(pred_values) if pred_values else 1.0
    mismatch_max = max(mismatch_values) if mismatch_values else 1.0
    x_values = [int(_to_float(row, "bin_index")) + 1 for row in trend_rows]

    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    ax.plot(x_values, [value / pred_max for value in pred_values], marker="o", color="#2E6FBB", linewidth=1.8, label="Predicted risk")
    ax.plot(x_values, [value / mismatch_max for value in mismatch_values], marker="s", color="#C27A2C", linewidth=1.8, label="Observed mismatch")
    ax.set_xlabel("Risk bin (low to high)")
    ax.set_ylabel("Normalized bin mean")
    ax.set_xticks(x_values)
    ax.legend(frameon=False, loc="upper left")
    written.extend(_save_figure_set(fig, output_root / "e2" / "e2_binned_risk_trend", formats))

    return written


def _write_e3_publication_plots(output_root: Path, formats: Iterable[str]) -> list[Path]:
    run_rows = _read_csv_rows(REPO_ROOT / "e3_attribution" / "outputs" / "e3_attribution_run_metrics.csv")
    layer_rows = _read_csv_rows(REPO_ROOT / "e3_attribution" / "outputs" / "e3_attribution_layer_points.csv")
    written: list[Path] = []

    rank_count = 12
    heatmap = [[0 for _ in range(rank_count)] for _ in range(rank_count)]
    for row in layer_rows:
        proxy_rank = max(1, min(rank_count, int(round(_to_float(row, "proxy_rank")))))
        exact_rank = max(1, min(rank_count, int(round(_to_float(row, "exact_rank")))))
        heatmap[exact_rank - 1][proxy_rank - 1] += 1

    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    image = ax.imshow(heatmap, origin="lower", cmap="YlGnBu")
    ax.grid(False)
    ax.plot([-0.5, rank_count - 0.5], [-0.5, rank_count - 0.5], color="#222222", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Proxy rank")
    ax.set_ylabel("Exact rank")
    ax.set_xticks(range(rank_count))
    ax.set_xticklabels(range(1, rank_count + 1))
    ax.set_yticks(range(rank_count))
    ax.set_yticklabels(range(1, rank_count + 1))
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Layer-point count")
    written.extend(_save_figure_set(fig, output_root / "e3" / "e3_proxy_vs_exact_rank_heatmap", formats))

    fig, ax = plt.subplots(figsize=(3.25, 3.0))
    for row in run_rows:
        precision = row["precision"]
        length = _to_int(row, "sequence_length")
        ax.scatter(
            _to_float(row, "mean_spearman"),
            _to_float(row, "mean_pairwise_accuracy"),
            s=52,
            marker=LENGTH_MARKERS[length],
            color=PRECISION_COLORS[precision],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
        )
    ax.axvline(0.0, color="#999999", linestyle=":", linewidth=1.0)
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Mean Spearman")
    ax.set_ylabel("Mean pairwise accuracy")
    legend1 = ax.legend(handles=_precision_legend_handles(), loc="lower right", frameon=False, title="Precision")
    ax.add_artist(legend1)
    ax.legend(handles=_length_legend_handles(), loc="upper left", frameon=False, title="Sequence")
    written.extend(_save_figure_set(fig, output_root / "e3" / "e3_run_level_fidelity", formats))

    return written


def _policy_order_key(policy: str) -> int:
    order = ["none", "static_global", "random_same_budget", "bgss"]
    return order.index(policy)


def _write_e5_publication_plots(output_root: Path, formats: Iterable[str]) -> list[Path]:
    run_rows = _read_csv_rows(REPO_ROOT / "e5_bgss" / "outputs" / "e5_bgss_run_metrics.csv")
    summary_rows = _read_csv_rows(REPO_ROOT / "e5_bgss" / "outputs" / "e5_bgss_policy_summary.csv")
    written: list[Path] = []

    summary_rows = sorted(summary_rows, key=lambda row: _policy_order_key(row["policy"]))

    fig, ax = plt.subplots(figsize=(3.3, 2.85))
    label_offsets = {
        "none": (6, 4),
        "static_global": (-34, 6),
        "random_same_budget": (8, 6),
        "bgss": (6, 6),
    }
    for row in summary_rows:
        policy = row["policy"]
        events = _to_float(row, "mean_num_events")
        final_mismatch_milli = 1000.0 * _to_float(row, "mean_final_mismatch")
        budget = _to_float(row, "mean_protected_layer_steps")
        marker_size = 65.0 if budget <= 0 else 65.0 + 0.06 * math.sqrt(budget) * 10.0
        ax.scatter(
            events,
            final_mismatch_milli,
            s=marker_size,
            color=POLICY_COLORS[policy],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            POLICY_LABELS[policy],
            (events, final_mismatch_milli),
            textcoords="offset points",
            xytext=label_offsets[policy],
            ha="left" if label_offsets[policy][0] >= 0 else "right",
            va="bottom",
            fontsize=7,
        )
    ax.set_xlabel("Mean mismatch-onset events")
    ax.set_ylabel(r"Mean final mismatch ($\times 10^{-3}$)")
    ax.text(
        0.97,
        0.05,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    written.extend(_save_figure_set(fig, output_root / "e5" / "e5_policy_tradeoff_events_vs_final_mismatch", formats))

    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in run_rows:
        grouped_rows.setdefault(row["policy"], []).append(row)

    ordered_policies = ["none", "static_global", "random_same_budget", "bgss"]
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    jitter = [-0.14, 0.0, 0.14]
    for y, policy in enumerate(ordered_policies):
        policy_rows = sorted(grouped_rows.get(policy, []), key=lambda row: _to_int(row, "seed"))
        x_values = [1000.0 * _to_float(row, "max_mismatch") for row in policy_rows]
        for offset, x_value in zip(jitter, x_values):
            ax.scatter(
                x_value,
                y + offset,
                s=42,
                color=POLICY_COLORS[policy],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                alpha=0.95,
            )
        if x_values:
            ax.scatter(
                statistics.mean(x_values),
                y,
                s=78,
                marker="D",
                color=POLICY_COLORS[policy],
                edgecolor="#222222",
                linewidth=0.8,
                zorder=4,
            )
    ax.set_xlabel(r"Max mismatch ($\times 10^{-3}$)")
    ax.set_yticks(range(len(ordered_policies)))
    ax.set_yticklabels([POLICY_LABELS[policy] for policy in ordered_policies])
    ax.invert_yaxis()
    written.extend(_save_figure_set(fig, output_root / "e5" / "e5_policy_max_mismatch_seedwise", formats))

    return written


PLOTTERS = {
    "e1": _write_e1_publication_plots,
    "e2": _write_e2_publication_plots,
    "e3": _write_e3_publication_plots,
    "e5": _write_e5_publication_plots,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication-friendly one-column figures from experiment outputs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(PLOTTERS.keys()),
        default=sorted(PLOTTERS.keys()),
        help="Experiments to render.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where publication figures will be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png", "svg"],
        help="Figure formats to write.",
    )
    args = parser.parse_args()

    _configure_style()

    all_written: list[Path] = []
    for experiment in args.experiments:
        written = PLOTTERS[experiment](args.output_root, args.formats)
        all_written.extend(written)
        print(f"[plots] {experiment}: wrote {len(written)} files")
    for path in all_written:
        print(path)


if __name__ == "__main__":
    main()
