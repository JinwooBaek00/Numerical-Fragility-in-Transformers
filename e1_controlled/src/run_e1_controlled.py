from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from common import create_run_context, load_config, save_json_artifact, save_text_artifact, write_rows
else:
    from ...common import create_run_context, load_config, save_json_artifact, save_text_artifact, write_rows


SVG_WIDTH = 1200
SVG_HEIGHT = 360
PANEL_WIDTH = 360
PANEL_HEIGHT = 250
PANEL_TOP = 60
PANEL_LEFTS = (60, 430, 800)
AXIS_COLOR = "#1f2937"
TEXT_COLOR = "#111827"
BLUE = "#1d4ed8"
ORANGE = "#ea580c"
GREEN = "#059669"
RED = "#b91c1c"


def _dot(x: list[float], y: list[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    b_t = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in b_t] for row in a]


def _matadd(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def _matsub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def _matscale(a: list[list[float]], scalar: float) -> list[list[float]]:
    return [[scalar * x for x in row] for row in a]


def _fro_norm(a: list[list[float]]) -> float:
    return math.sqrt(sum(x * x for row in a for x in row))


def _vec_norm(x: list[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def _normalize(vector: list[float]) -> list[float]:
    norm = _vec_norm(vector)
    if norm == 0.0:
        return vector[:]
    return [value / norm for value in vector]


def _candidate_start_vectors(n_cols: int) -> list[list[float]]:
    if n_cols <= 0:
        return []
    candidates = [
        [float(idx + 1) for idx in range(n_cols)],
        [1.0 if idx % 2 == 0 else -1.0 for idx in range(n_cols)],
    ]
    for idx in range(n_cols):
        basis = [0.0] * n_cols
        basis[idx] = 1.0
        candidates.append(basis)
    return candidates


def _spectral_norm(matrix: list[list[float]], *, iters: int = 30) -> float:
    if not matrix or not matrix[0]:
        return 0.0
    n_cols = len(matrix[0])
    transpose = list(map(list, zip(*matrix)))
    vec: list[float] | None = None
    for candidate in _candidate_start_vectors(n_cols):
        unit = _normalize(candidate)
        if _vec_norm(_matvec(matrix, unit)) > 1e-15:
            vec = unit
            break
    if vec is None:
        return 0.0

    for _ in range(iters):
        left = _matvec(matrix, vec)
        left_norm = _vec_norm(left)
        if left_norm <= 1e-15:
            return 0.0
        left = [x / left_norm for x in left]

        right = _matvec(transpose, left)
        right_norm = _vec_norm(right)
        if right_norm <= 1e-15:
            return 0.0
        vec = [x / right_norm for x in right]

    image = _matvec(matrix, vec)
    return _vec_norm(image)


def _softmax_row(row: list[float]) -> list[float]:
    max_val = max(row)
    exps = [math.exp(x - max_val) for x in row]
    total = sum(exps)
    return [x / total for x in exps]


def _softmax_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [_softmax_row(row) for row in matrix]


def _softmax_jacobian(p: list[float]) -> list[list[float]]:
    return [[(p[i] if i == j else 0.0) - p[i] * p[j] for j in range(len(p))] for i in range(len(p))]


def _max_row_jac_norm(probabilities: list[list[float]]) -> float:
    return max(_spectral_norm(_softmax_jacobian(row)) for row in probabilities)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx
        while end + 1 < len(order) and order[end + 1][1] == order[idx][1]:
            end += 1
        avg_rank = 0.5 * (idx + end) + 1.0
        for pos in range(idx, end + 1):
            ranks[order[pos][0]] = avg_rank
        idx = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _is_nonincreasing(values: list[float], *, tolerance: float = 1e-12) -> bool:
    return all(values[idx] <= values[idx - 1] + tolerance for idx in range(1, len(values)))


def _layernorm_path(x: list[float], gamma: list[float], eps: float) -> list[float]:
    mean_x = sum(x) / len(x)
    centered = [value - mean_x for value in x]
    variance = sum(value * value for value in centered) / len(centered)
    scale = math.sqrt(variance + eps)
    return [g * c / scale for g, c in zip(gamma, centered)]


def _layernorm_magnitude(x: list[float], gamma: list[float], eps: float) -> float:
    mean_x = sum(x) / len(x)
    centered = [value - mean_x for value in x]
    variance = sum(value * value for value in centered) / len(centered)
    gamma_norm = max(abs(g) for g in gamma)
    x_norm = _vec_norm(x)
    return gamma_norm * (eps + 2.0 * variance) / ((variance + eps) ** 1.5) * x_norm


def _attention_records(config: dict[str, Any]) -> list[dict[str, float]]:
    delta_scale = float(config["delta_scale"])
    records: list[dict[str, float]] = []
    for margin in config["margins"]:
        for value_scale in config["value_scales"]:
            s = [[0.0, -float(margin)], [-float(margin), 0.0]]
            delta_s = _matscale(s, delta_scale)
            p = _softmax_matrix(s)
            p_pert = _softmax_matrix(_matadd(s, delta_s))
            v = _matscale([[1.0, -1.0], [-1.0, 1.0]], float(value_scale))
            measured = _fro_norm(_matmul(_matsub(p_pert, p), v))
            jac_norm = _max_row_jac_norm(p)
            p_norm = _fro_norm(p)
            s_norm = _fro_norm(s)
            kappa_softmax = 0.0 if p_norm == 0.0 else (s_norm / p_norm) * jac_norm
            theory = kappa_softmax * p_norm * _spectral_norm(v)
            records.append(
                {
                    "margin": float(margin),
                    "value_scale": float(value_scale),
                    "score_norm": s_norm,
                    "softmax_jac_norm": jac_norm,
                    "kappa_softmax": kappa_softmax,
                    "measured_error": measured,
                    "theory_proxy": theory,
                }
            )
    return records


def _layernorm_records(config: dict[str, Any]) -> list[dict[str, float]]:
    x = [float(v) for v in config["x"]]
    dx = [float(v) for v in config["dx"]]
    gamma = [float(v) for v in config["gamma"]]
    x_pert = [a + b for a, b in zip(x, dx)]
    records: list[dict[str, float]] = []
    for eps in config["eps_values"]:
        eps_value = float(eps)
        z = _layernorm_path(x, gamma, eps_value)
        z_pert = _layernorm_path(x_pert, gamma, eps_value)
        measured = _vec_norm([a - b for a, b in zip(z_pert, z)])
        theory = _layernorm_magnitude(x, gamma, eps_value)
        records.append(
            {
                "epsilon": eps_value,
                "measured_change": measured,
                "theory_magnitude": theory,
                "log10_epsilon": math.log10(eps_value),
            }
        )
    return records


def _residual_records(config: dict[str, Any]) -> list[dict[str, float]]:
    depth = int(config["depth"])
    delta = [1.0, 1.0]
    base_norm = _vec_norm(delta)
    records: list[dict[str, float]] = []
    for rho in config["rhos"]:
        rho_value = float(rho)
        t = [[1.0 + rho_value, 0.0], [0.0, 1.0 - 0.25 * rho_value]]
        state = delta[:]
        for _ in range(depth):
            state = _matvec(t, state)
        measured = _vec_norm(state) / base_norm
        theory = (1.0 + rho_value) ** depth
        records.append(
            {
                "rho": rho_value,
                "measured_transport": measured,
                "transport_bound": theory,
                "bound_gap": theory - measured,
            }
        )
    return records


def _attention_summary(records: list[dict[str, float]]) -> dict[str, Any]:
    theory = [row["theory_proxy"] for row in records]
    measured = [row["measured_error"] for row in records]
    return {
        "pearson": _pearson(theory, measured),
        "spearman": _spearman(theory, measured),
        "max_measured_error": max(measured),
        "min_measured_error": min(measured),
    }


def _layernorm_summary(records: list[dict[str, float]]) -> dict[str, Any]:
    theory = [row["theory_magnitude"] for row in records]
    measured = [row["measured_change"] for row in records]
    return {
        "pearson": _pearson(theory, measured),
        "spearman": _spearman(theory, measured),
        "measured_nonincreasing": _is_nonincreasing(measured),
        "theory_nonincreasing": _is_nonincreasing(theory),
    }


def _residual_summary(records: list[dict[str, float]]) -> dict[str, Any]:
    measured = [row["measured_transport"] for row in records]
    theory = [row["transport_bound"] for row in records]
    return {
        "pearson": _pearson(theory, measured),
        "spearman": _spearman(theory, measured),
        "bound_respected": all(row["bound_gap"] >= -1e-12 for row in records),
        "max_bound_gap": max(row["bound_gap"] for row in records),
    }


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", color: str = TEXT_COLOR, weight: str = "normal") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}">{safe}</text>'


def _svg_line(x1: float, y1: float, x2: float, y2: float, *, color: str = AXIS_COLOR, width: float = 1.5, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"{dash_attr}/>'


def _svg_circle(x: float, y: float, *, radius: float = 4.0, color: str = BLUE, opacity: float = 0.9) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" opacity="{opacity:.2f}"/>'


def _svg_polyline(points: list[tuple[float, float]], *, color: str = BLUE, width: float = 2.0) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width:.1f}"/>'


def _linear_scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def _panel_axes(panel_left: float, panel_top: float) -> tuple[float, float, float, float]:
    return panel_left, panel_top, panel_left + PANEL_WIDTH, panel_top + PANEL_HEIGHT


def _panel_frame(panel_left: float, panel_top: float, title: str, x_label: str, y_label: str) -> str:
    x0, y0, x1, y1 = _panel_axes(panel_left, panel_top)
    parts = [
        _svg_text(panel_left, panel_top - 18, title, size=15, weight="bold"),
        _svg_line(x0, y0, x0, y1),
        _svg_line(x0, y1, x1, y1),
        _svg_text((x0 + x1) / 2, y1 + 30, x_label, size=12, anchor="middle"),
        _svg_text(x0 - 35, (y0 + y1) / 2, y_label, size=12, anchor="middle"),
    ]
    return "\n".join(parts)


def _scatter_panel(records: list[dict[str, float]], left: float, summary: dict[str, Any]) -> str:
    x0, y0, x1, y1 = _panel_axes(left, PANEL_TOP)
    theory = [row["theory_proxy"] for row in records]
    measured = [row["measured_error"] for row in records]
    parts = [_panel_frame(left, PANEL_TOP, "Softmax-Side Proxy vs Attention Change", "softmax-side proxy", "measured attention change")]
    xmin, xmax = min(theory), max(theory)
    ymin, ymax = min(measured), max(measured)
    for row in records:
        px = _linear_scale(row["theory_proxy"], xmin, xmax, x0 + 8, x1 - 8)
        py = _linear_scale(row["measured_error"], ymin, ymax, y1 - 8, y0 + 8)
        parts.append(_svg_circle(px, py, color=BLUE))
    parts.append(_svg_text(left, PANEL_TOP + PANEL_HEIGHT + 52, f"Pearson={summary['pearson']:.3f}  Spearman={summary['spearman']:.3f}", size=12))
    return "\n".join(parts)


def _ln_panel(records: list[dict[str, float]], left: float, summary: dict[str, Any]) -> str:
    x0, y0, x1, y1 = _panel_axes(left, PANEL_TOP)
    eps_logs = [row["log10_epsilon"] for row in records]
    measured = [row["measured_change"] for row in records]
    theory = [row["theory_magnitude"] for row in records]
    scale = max(measured) / max(theory) if max(theory) > 0 else 1.0
    theory_scaled = [value * scale for value in theory]
    parts = [_panel_frame(left, PANEL_TOP, "LayerNorm Epsilon Sweep", "log10(epsilon)", "magnitude / change")]
    xmin, xmax = min(eps_logs), max(eps_logs)
    ymin, ymax = 0.0, max(max(measured), max(theory_scaled))
    measured_points = []
    theory_points = []
    for row, measured_val, theory_val in zip(records, measured, theory_scaled):
        px = _linear_scale(row["log10_epsilon"], xmin, xmax, x0 + 8, x1 - 8)
        py_measured = _linear_scale(measured_val, ymin, ymax, y1 - 8, y0 + 8)
        py_theory = _linear_scale(theory_val, ymin, ymax, y1 - 8, y0 + 8)
        measured_points.append((px, py_measured))
        theory_points.append((px, py_theory))
    parts.append(_svg_polyline(measured_points, color=ORANGE))
    parts.append(_svg_polyline(theory_points, color=GREEN))
    parts.append(_svg_text(left, PANEL_TOP + PANEL_HEIGHT + 52, f"Measured monotone={summary['measured_nonincreasing']}  Theory monotone={summary['theory_nonincreasing']}", size=12))
    parts.append(_svg_text(left, PANEL_TOP + PANEL_HEIGHT + 68, "orange: measured change   green: scaled theory", size=11, color="#374151"))
    return "\n".join(parts)


def _residual_panel(records: list[dict[str, float]], left: float, summary: dict[str, Any]) -> str:
    x0, y0, x1, y1 = _panel_axes(left, PANEL_TOP)
    xs = [row["rho"] for row in records]
    measured = [row["measured_transport"] for row in records]
    theory = [row["transport_bound"] for row in records]
    parts = [_panel_frame(left, PANEL_TOP, "Residual Transport Bound", "rho", "amplification")]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, max(max(measured), max(theory))
    measured_points = []
    theory_points = []
    for row in records:
        px = _linear_scale(row["rho"], xmin, xmax, x0 + 8, x1 - 8)
        py_measured = _linear_scale(row["measured_transport"], ymin, ymax, y1 - 8, y0 + 8)
        py_theory = _linear_scale(row["transport_bound"], ymin, ymax, y1 - 8, y0 + 8)
        measured_points.append((px, py_measured))
        theory_points.append((px, py_theory))
    parts.append(_svg_polyline(measured_points, color=BLUE))
    parts.append(_svg_polyline(theory_points, color=RED))
    parts.append(_svg_text(left, PANEL_TOP + PANEL_HEIGHT + 52, f"Bound respected={summary['bound_respected']}  Pearson={summary['pearson']:.3f}", size=12))
    parts.append(_svg_text(left, PANEL_TOP + PANEL_HEIGHT + 68, "blue: measured transport   red: theory bound", size=11, color="#374151"))
    return "\n".join(parts)


def _render_svg(attn_records: list[dict[str, float]], ln_records: list[dict[str, float]], residual_records: list[dict[str, float]], summaries: dict[str, dict[str, Any]]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        _svg_text(40, 28, "E1 Controlled Validation", size=20, weight="bold"),
        _svg_text(40, 48, "Softmax-side attention proxy, LayerNorm epsilon sweep, and residual transport", size=12, color="#374151"),
        _scatter_panel(attn_records, PANEL_LEFTS[0], summaries["attention"]),
        _ln_panel(ln_records, PANEL_LEFTS[1], summaries["layernorm"]),
        _residual_panel(residual_records, PANEL_LEFTS[2], summaries["residual"]),
        "</svg>",
    ]
    return "\n".join(parts)


def _make_table_rows(attn_summary: dict[str, Any], ln_summary: dict[str, Any], residual_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "mechanism": "attention",
            "primary_check": "softmax_proxy_tracks_attention_change",
            "pearson": round(attn_summary["pearson"], 6),
            "spearman": round(attn_summary["spearman"], 6),
            "pass_flag": attn_summary["pearson"] >= 0.9 and attn_summary["spearman"] >= 0.9,
            "note": "Softmax-side attention proxy should increase with measured attention-output change.",
        },
        {
            "mechanism": "layernorm",
            "primary_check": "epsilon_monotonicity",
            "pearson": round(ln_summary["pearson"], 6),
            "spearman": round(ln_summary["spearman"], 6),
            "pass_flag": ln_summary["measured_nonincreasing"] and ln_summary["theory_nonincreasing"],
            "note": "Measured normalization-path change should decrease as epsilon increases.",
        },
        {
            "mechanism": "residual",
            "primary_check": "transport_bound",
            "pearson": round(residual_summary["pearson"], 6),
            "spearman": round(residual_summary["spearman"], 6),
            "pass_flag": residual_summary["bound_respected"],
            "note": "Measured downstream amplification should stay below the residual transport factor.",
        },
    ]


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    script_path = Path(__file__).resolve()
    experiment_dir = script_path.parents[1]
    workspace_root = script_path.parents[2]
    config_path = Path(argv[0]).resolve() if argv else experiment_dir / "configs" / "default.json"
    config = load_config(config_path)

    context = create_run_context(
        experiment_dir,
        short_tag=f"e1_{config.get('run_tag', 'default')}",
        config=config,
        metadata={
            "model_name": "controlled-suite",
            "dataset_name": "synthetic",
            "precision": "fp64-analytic",
            "seed": 0,
            "sequence_length": 0,
        },
        workspace_root=workspace_root,
    )
    context.append_stdout(f"Running E1 controlled suite with config: {config_path}")

    attn_records = _attention_records(config["attention"])
    ln_records = _layernorm_records(config["layernorm"])
    residual_records = _residual_records(config["residual"])

    attn_summary = _attention_summary(attn_records)
    ln_summary = _layernorm_summary(ln_records)
    residual_summary = _residual_summary(residual_records)

    table_rows = _make_table_rows(attn_summary, ln_summary, residual_summary)
    summaries = {
        "attention": attn_summary,
        "layernorm": ln_summary,
        "residual": residual_summary,
    }
    report_body = "\n".join(
        [
            "# E1 Controlled Report",
            "",
            "## Purpose",
            "",
            "Validate the sign, monotonicity, and transport structure of the local theory in controlled low-noise cases.",
            "",
            "## Highlights",
            "",
            f"- Attention Pearson: {attn_summary['pearson']:.3f}",
            f"- Attention Spearman: {attn_summary['spearman']:.3f}",
            f"- LayerNorm measured monotone: {ln_summary['measured_nonincreasing']}",
            f"- LayerNorm theory monotone: {ln_summary['theory_nonincreasing']}",
            f"- Residual bound respected: {residual_summary['bound_respected']}",
            f"- Residual Pearson: {residual_summary['pearson']:.3f}",
            "",
            "## Notes",
            "",
            "- This report is generated from synthetic controlled cases, not from an LM run.",
            "- Execute the runner to confirm the numeric values and SVG output in the local Python environment.",
            "",
        ]
    )

    context.write_metrics(
        {
            "attention_summary": attn_summary,
            "layernorm_summary": ln_summary,
            "residual_summary": residual_summary,
            "all_pass": all(bool(row["pass_flag"]) for row in table_rows),
        }
    )
    context.write_rows(
        "per_layer_metrics.csv",
        [],
        fieldnames=[
            "step",
            "layer",
            "seed",
            "precision",
            "sequence_length",
            "risk_score",
            "ln_magnitude",
            "attn_magnitude",
            "remainder_magnitude",
            "ln_dominance",
            "rho_ln",
        ],
    )
    context.write_rows(
        "per_step_metrics.csv",
        [
            {
                "step": 0,
                "seed": 0,
                "precision": "fp64-analytic",
                "sequence_length": 0,
                "loss": 0.0,
                "final_mismatch": 0.0,
                "predicted_risk_sum": 0.0,
                "event_flag": 0,
            }
        ],
        fieldnames=["step", "seed", "precision", "sequence_length", "loss", "final_mismatch", "predicted_risk_sum", "event_flag"],
    )
    context.write_rows("attention_records.csv", attn_records)
    context.write_rows("layernorm_records.csv", ln_records)
    context.write_rows("residual_records.csv", residual_records)
    context.write_rows("summary_table.csv", table_rows)

    svg = _render_svg(attn_records, ln_records, residual_records, summaries)
    save_text_artifact(context.paths.outputs_dir, "e1_controlled_summary.svg", svg)
    save_text_artifact(context.paths.outputs_dir, "e1_controlled_report.md", report_body)
    save_json_artifact(context.paths.outputs_dir, "e1_controlled_metrics.json", summaries)
    write_rows(context.paths.outputs_dir / "e1_controlled_summary_table.csv", table_rows)
    write_rows(context.paths.outputs_dir / "e1_controlled_attention_records.csv", attn_records)
    write_rows(context.paths.outputs_dir / "e1_controlled_layernorm_records.csv", ln_records)
    write_rows(context.paths.outputs_dir / "e1_controlled_residual_records.csv", residual_records)
    context.write_summary(
        {
            "goal": "Validate the sign, monotonicity, and transport structure of the local theory in controlled low-noise cases.",
            "setup": [
                f"Config: {config_path.name}",
                "Attention: margin and value-scale sweeps under a fixed score perturbation",
                "LayerNorm: epsilon sweep with fixed x, dx, and gamma",
                "Residual: downstream amplification across a rho sweep at fixed depth",
            ],
            "key_metrics": [
                f"Attention Pearson={attn_summary['pearson']:.3f}, Spearman={attn_summary['spearman']:.3f}",
                f"LayerNorm measured monotone={ln_summary['measured_nonincreasing']}, theory monotone={ln_summary['theory_nonincreasing']}",
                f"Residual bound respected={residual_summary['bound_respected']}, Pearson={residual_summary['pearson']:.3f}",
            ],
            "pass_fail_verdict": "Pass" if all(bool(row["pass_flag"]) for row in table_rows) else "Needs review",
            "anomalies": "None recorded at code-generation time. Execute the run to confirm the numeric outputs and SVG render cleanly.",
            "follow_up": "Use the same run contract and output layout for E2 predictor fidelity.",
        }
    )
    context.mark_completed(status="completed_unverified_runtime")
    context.append_stdout("E1 controlled suite generation completed.")


if __name__ == "__main__":
    main()
