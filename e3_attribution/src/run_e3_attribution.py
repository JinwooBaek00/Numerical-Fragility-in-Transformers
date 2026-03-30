from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from common import (
        create_run_context,
        load_config,
        manual_forward_with_prefixes,
        manual_patched_forward,
        save_json_artifact,
        save_text_artifact,
        write_rows,
    )
else:
    from ...common import (
        create_run_context,
        load_config,
        manual_forward_with_prefixes,
        manual_patched_forward,
        save_json_artifact,
        save_text_artifact,
        write_rows,
    )

try:
    import torch
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E3 requires PyTorch to be installed on the execution environment.") from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E3 requires the `datasets` package on the execution environment.") from exc

try:
    from transformers import AutoTokenizer, GPT2LMHeadModel
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E3 requires the `transformers` package on the execution environment.") from exc


TARGET_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

SVG_WIDTH = 1120
SVG_HEIGHT = 430
PANEL_TOP = 82
PANEL_WIDTH = 470
PANEL_HEIGHT = 280
LEFT_PANEL_X = 60
RIGHT_PANEL_X = 600
TEXT_COLOR = "#111827"
AXIS_COLOR = "#1f2937"
PRECISION_COLOR = {
    "fp16": "#1d4ed8",
    "bf16": "#ea580c",
    "fp32": "#6b7280",
}


@dataclass(frozen=True)
class SourceRun:
    run_dir: Path
    run_id: str
    model_name: str
    dataset_cfg: dict[str, Any]
    precision: str
    sequence_length: int
    seed: int
    batch_size: int
    max_sequences: int
    stride: int
    trust_remote_code: bool
    per_step_rows: list[dict[str, Any]]
    per_layer_rows: list[dict[str, Any]]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype_name_to_torch(name: str) -> torch.dtype:
    if name not in TARGET_DTYPE:
        raise ValueError(f"Unsupported precision: {name}")
    return TARGET_DTYPE[name]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(values: list[float], *, descending: bool = False) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1], reverse=descending)
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


def _spearman_desc(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs, descending=True), _ranks(ys, descending=True))


def _pairwise_ordering_accuracy(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    matches = 0
    total = 0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0.0 or dy == 0.0:
                continue
            total += 1
            if (dx > 0.0 and dy > 0.0) or (dx < 0.0 and dy < 0.0):
                matches += 1
    if total == 0:
        return 0.0
    return matches / total


def _topk_overlap(xs: list[float], ys: list[float], k: int) -> float:
    if len(xs) != len(ys) or not xs or k <= 0:
        return 0.0
    take = min(k, len(xs))
    proxy_top = {idx for idx, _ in sorted(enumerate(xs), key=lambda item: item[1], reverse=True)[:take]}
    exact_top = {idx for idx, _ in sorted(enumerate(ys), key=lambda item: item[1], reverse=True)[:take]}
    return len(proxy_top & exact_top) / take


def _top1_hit(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    return 1.0 if max(range(len(xs)), key=lambda idx: xs[idx]) == max(range(len(ys)), key=lambda idx: ys[idx]) else 0.0


def _relative_mismatch(target_hidden: torch.Tensor, reference_hidden: torch.Tensor) -> float:
    numerator = torch.norm(target_hidden.detach().float() - reference_hidden.detach().float()).item()
    denominator = max(torch.norm(reference_hidden.detach().float()).item(), 1e-12)
    return numerator / denominator


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_token_ids(tokenizer: Any, dataset_cfg: dict[str, Any]) -> list[int]:
    dataset = load_dataset(
        dataset_cfg["name"],
        dataset_cfg["config_name"],
        split=dataset_cfg["split"],
    )
    text_field = dataset_cfg.get("text_field", "text")
    texts = [row[text_field] for row in dataset if row.get(text_field) and row[text_field].strip()]
    return tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]


def _resolve_stride(stride: int, sequence_length: int) -> int:
    return sequence_length if stride <= 0 else stride


def _build_token_windows(
    token_ids: list[int],
    sequence_length: int,
    max_sequences: int,
    seed: int,
    stride: int,
) -> list[list[int]]:
    step = _resolve_stride(stride, sequence_length)
    windows: list[list[int]] = []
    for start in range(0, max(0, len(token_ids) - sequence_length + 1), step):
        window = token_ids[start : start + sequence_length]
        if len(window) == sequence_length:
            windows.append(window)
    if max_sequences > 0 and len(windows) > max_sequences:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(len(windows)), max_sequences))
        windows = [windows[idx] for idx in chosen]
    return windows


def _batch_windows(windows: list[list[int]], batch_size: int) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    for start in range(0, len(windows), batch_size):
        batches.append(torch.tensor(windows[start : start + batch_size], dtype=torch.long))
    return batches


def _read_source_run(run_dir: Path) -> SourceRun | None:
    config_path = run_dir / "config.json"
    step_path = run_dir / "per_step_metrics.csv"
    layer_path = run_dir / "per_layer_metrics.csv"
    if not (config_path.exists() and step_path.exists() and layer_path.exists()):
        return None
    config = load_config(config_path)
    per_step_rows = _read_csv_rows(step_path)
    per_layer_rows = _read_csv_rows(layer_path)
    if not per_step_rows or not per_layer_rows:
        return None
    return SourceRun(
        run_dir=run_dir,
        run_id=run_dir.name,
        model_name=str(config["model_name"]),
        dataset_cfg=dict(config["dataset"]),
        precision=str(config["target_precision"]),
        sequence_length=int(config["sequence_length"]),
        seed=int(config["seed"]),
        batch_size=int(config.get("batch_size", 1)),
        max_sequences=int(config.get("max_sequences_per_run", len(per_step_rows))),
        stride=int(config.get("stride", 0)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        per_step_rows=per_step_rows,
        per_layer_rows=per_layer_rows,
    )


def _discover_source_runs(source_dir: Path, config: dict[str, Any]) -> list[SourceRun]:
    runs_dir = source_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Source experiment runs directory does not exist: {runs_dir}")
    wanted_precisions = {str(value) for value in config["target_precisions"]}
    wanted_lengths = {int(value) for value in config["sequence_lengths"]}
    wanted_seeds = {int(value) for value in config["seeds"]}
    discovered: list[SourceRun] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        source_run = _read_source_run(run_dir)
        if source_run is None:
            continue
        if source_run.precision not in wanted_precisions:
            continue
        if source_run.sequence_length not in wanted_lengths:
            continue
        if source_run.seed not in wanted_seeds:
            continue
        discovered.append(source_run)
    if not discovered:
        raise RuntimeError(f"No matching E2 runs were found under {runs_dir}")

    support_summary_path = source_dir / "outputs" / "e2_predictor_support_summary.csv"
    if support_summary_path.exists():
        support_rows = _read_csv_rows(support_summary_path)
        preferred_ids = {
            str(row.get("run_id", ""))
            for row in support_rows
            if str(row.get("run_id", ""))
        }
        preferred_runs = [run for run in discovered if run.run_id in preferred_ids]
        if preferred_runs:
            preferred_runs.sort(key=lambda run: run.run_id)
            return preferred_runs

    latest_by_spec: dict[tuple[str, int, int], SourceRun] = {}
    for source_run in discovered:
        key = (source_run.precision, source_run.sequence_length, source_run.seed)
        current = latest_by_spec.get(key)
        if current is None or source_run.run_id > current.run_id:
            latest_by_spec[key] = source_run
    return sorted(latest_by_spec.values(), key=lambda run: run.run_id)


def _select_steps(per_step_rows: list[dict[str, str]], *, strategy: str, max_steps: int) -> list[dict[str, str]]:
    if strategy == "top_mismatch":
        ordered = sorted(
            per_step_rows,
            key=lambda row: (-_safe_float(row.get("final_mismatch")), _safe_int(row.get("step"))),
        )
        return ordered[:max_steps]
    if strategy == "evenly_spaced":
        ordered = sorted(per_step_rows, key=lambda row: _safe_int(row.get("step")))
        if len(ordered) <= max_steps or max_steps <= 1:
            return ordered[:max_steps]
        indices = [round(i * (len(ordered) - 1) / (max_steps - 1)) for i in range(max_steps)]
        return [ordered[idx] for idx in indices]
    raise ValueError(f"Unsupported selection_strategy: {strategy}")


def _group_layer_rows(per_layer_rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in per_layer_rows:
        step = _safe_int(row.get("step"))
        grouped.setdefault(step, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: _safe_int(item.get("layer")))
    return grouped


def _clone_model_to_precision(
    model_name: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.eval()
    model.to(device=device, dtype=dtype)
    return model


def _prepare_models(
    model_name: str,
    target_precision: str,
    device: torch.device,
    trust_remote_code: bool,
) -> tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
    ref_model = _clone_model_to_precision(
        model_name,
        dtype=torch.float32,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    tgt_model = _clone_model_to_precision(
        model_name,
        dtype=_dtype_name_to_torch(target_precision),
        device=device,
        trust_remote_code=trust_remote_code,
    )
    return ref_model, tgt_model


def _get_batches_for_run(
    source_run: SourceRun,
    *,
    tokenizer_cache: dict[tuple[str, bool], Any],
    token_ids_cache: dict[tuple[str, bool, str, str, str, str], list[int]],
) -> list[torch.Tensor]:
    tokenizer_key = (source_run.model_name, source_run.trust_remote_code)
    if tokenizer_key not in tokenizer_cache:
        tokenizer = AutoTokenizer.from_pretrained(
            source_run.model_name,
            trust_remote_code=source_run.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer_cache[tokenizer_key] = tokenizer
    tokenizer = tokenizer_cache[tokenizer_key]

    dataset_key = (
        source_run.model_name,
        source_run.trust_remote_code,
        str(source_run.dataset_cfg["name"]),
        str(source_run.dataset_cfg["config_name"]),
        str(source_run.dataset_cfg["split"]),
        str(source_run.dataset_cfg.get("text_field", "text")),
    )
    if dataset_key not in token_ids_cache:
        token_ids_cache[dataset_key] = _load_token_ids(tokenizer, source_run.dataset_cfg)
    token_ids = token_ids_cache[dataset_key]

    windows = _build_token_windows(
        token_ids,
        source_run.sequence_length,
        source_run.max_sequences,
        source_run.seed,
        source_run.stride,
    )
    return _batch_windows(windows, source_run.batch_size)


def _step_input_ids(batches: list[torch.Tensor], step_idx: int, device: torch.device) -> torch.Tensor:
    if step_idx < 0 or step_idx >= len(batches):
        raise IndexError(f"Step {step_idx} is outside the reconstructed batch list of length {len(batches)}")
    return batches[step_idx].to(device)


def _evaluate_step_attribution(
    ref_model: GPT2LMHeadModel,
    tgt_model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    source_step_row: dict[str, str],
    layer_rows: list[dict[str, str]],
    topk_values: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with torch.no_grad():
        reference_final, _ = manual_forward_with_prefixes(ref_model, input_ids)
        target_final, target_prefixes = manual_forward_with_prefixes(tgt_model, input_ids)

        recomputed_base_mismatch = _relative_mismatch(target_final, reference_final)
        source_mismatch = _safe_float(source_step_row.get("final_mismatch"))
        per_layer_output_rows: list[dict[str, Any]] = []
        proxy_values: list[float] = []
        exact_values: list[float] = []

        for layer_row in layer_rows:
            layer_idx = _safe_int(layer_row.get("layer"))
            proxy_score = _safe_float(layer_row.get("risk_score"))
            patched_final = manual_patched_forward(
                ref_model,
                tgt_model,
                target_prefixes[layer_idx],
                patch_layer=layer_idx,
            )
            patched_mismatch = _relative_mismatch(patched_final, reference_final)
            exact_reduction = recomputed_base_mismatch - patched_mismatch
            reduction_fraction = 0.0 if recomputed_base_mismatch == 0.0 else exact_reduction / recomputed_base_mismatch

            proxy_values.append(proxy_score)
            exact_values.append(exact_reduction)
            per_layer_output_rows.append(
                {
                    "step": _safe_int(source_step_row.get("step")),
                    "layer": layer_idx,
                    "precision": source_step_row.get("precision", ""),
                    "sequence_length": _safe_int(source_step_row.get("sequence_length")),
                    "seed": _safe_int(source_step_row.get("seed")),
                    "source_mismatch": source_mismatch,
                    "recomputed_base_mismatch": recomputed_base_mismatch,
                    "proxy_score": proxy_score,
                    "scaled_proxy_score": _safe_float(layer_row.get("scaled_risk_score")),
                    "proxy_ln_dominance": _safe_float(layer_row.get("ln_dominance")),
                    "patched_mismatch": patched_mismatch,
                    "exact_reduction": exact_reduction,
                    "reduction_fraction": reduction_fraction,
                }
            )

        proxy_ranks = _ranks(proxy_values, descending=True)
        exact_ranks = _ranks(exact_values, descending=True)
        for idx, row in enumerate(per_layer_output_rows):
            row["proxy_rank"] = proxy_ranks[idx]
            row["exact_rank"] = exact_ranks[idx]
            row["rank_gap"] = abs(proxy_ranks[idx] - exact_ranks[idx])

        step_metrics = {
            "step": _safe_int(source_step_row.get("step")),
            "precision": source_step_row.get("precision", ""),
            "sequence_length": _safe_int(source_step_row.get("sequence_length")),
            "seed": _safe_int(source_step_row.get("seed")),
            "source_mismatch": source_mismatch,
            "recomputed_base_mismatch": recomputed_base_mismatch,
            "recompute_gap": abs(recomputed_base_mismatch - source_mismatch),
            "mean_exact_reduction": _mean(exact_values),
            "max_exact_reduction": max(exact_values) if exact_values else 0.0,
            "mean_rank_gap": _mean([abs(p - e) for p, e in zip(proxy_ranks, exact_ranks)]),
            "spearman": _spearman_desc(proxy_values, exact_values),
            "pairwise_accuracy": _pairwise_ordering_accuracy(proxy_values, exact_values),
            "top1_hit": _top1_hit(proxy_values, exact_values),
            "proxy_best_layer": max(range(len(proxy_values)), key=lambda idx: proxy_values[idx]) if proxy_values else -1,
            "exact_best_layer": max(range(len(exact_values)), key=lambda idx: exact_values[idx]) if exact_values else -1,
        }
        for k in topk_values:
            step_metrics[f"topk_overlap_{k}"] = _topk_overlap(proxy_values, exact_values, k)
        return step_metrics, per_layer_output_rows


def _metric_fieldnames(topk_values: list[int]) -> list[str]:
    return [
        "run_id",
        "source_run_id",
        "step",
        "selection_rank",
        "precision",
        "sequence_length",
        "seed",
        "source_mismatch",
        "recomputed_base_mismatch",
        "recompute_gap",
        "mean_exact_reduction",
        "max_exact_reduction",
        "mean_rank_gap",
        "spearman",
        "pairwise_accuracy",
        "top1_hit",
        *[f"topk_overlap_{k}" for k in topk_values],
        "proxy_best_layer",
        "exact_best_layer",
    ]


def _layer_fieldnames() -> list[str]:
    return [
        "run_id",
        "source_run_id",
        "step",
        "layer",
        "precision",
        "sequence_length",
        "seed",
        "source_mismatch",
        "recomputed_base_mismatch",
        "proxy_score",
        "scaled_proxy_score",
        "proxy_ln_dominance",
        "patched_mismatch",
        "exact_reduction",
        "reduction_fraction",
        "proxy_rank",
        "exact_rank",
        "rank_gap",
    ]


def _summary_row_fieldnames(topk_values: list[int]) -> list[str]:
    return [
        "run_id",
        "source_run_id",
        "precision",
        "sequence_length",
        "seed",
        "num_selected_steps",
        "mean_source_mismatch",
        "mean_recomputed_mismatch",
        "mean_recompute_gap",
        "mean_spearman",
        "mean_pairwise_accuracy",
        "mean_top1_hit",
        *[f"mean_topk_overlap_{k}" for k in topk_values],
        "mean_rank_gap",
    ]


def _run_single_source_run(
    experiment_dir: Path,
    workspace_root: Path,
    e3_config: dict[str, Any],
    source_run: SourceRun,
    *,
    device: torch.device,
    batches: list[torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _seed_everything(source_run.seed)
    ref_model, tgt_model = _prepare_models(
        source_run.model_name,
        source_run.precision,
        device,
        trust_remote_code=source_run.trust_remote_code,
    )

    context = create_run_context(
        experiment_dir,
        short_tag=f"{source_run.precision}_len{source_run.sequence_length}_seed{source_run.seed}",
        config={
            **e3_config,
            "source_run_id": source_run.run_id,
            "model_name": source_run.model_name,
            "dataset": source_run.dataset_cfg,
            "target_precision": source_run.precision,
            "sequence_length": source_run.sequence_length,
            "seed": source_run.seed,
            "batch_size": source_run.batch_size,
        },
        metadata={
            "model_name": source_run.model_name,
            "dataset_name": f"{source_run.dataset_cfg['name']}:{source_run.dataset_cfg['config_name']}:{source_run.dataset_cfg['split']}",
            "precision": source_run.precision,
            "seed": source_run.seed,
            "sequence_length": source_run.sequence_length,
            "source_run_id": source_run.run_id,
        },
        workspace_root=workspace_root,
    )
    context.append_stdout(
        f"Starting E3 attribution run for source={source_run.run_id}, precision={source_run.precision}, sequence_length={source_run.sequence_length}, seed={source_run.seed}"
    )

    grouped_layers = _group_layer_rows(source_run.per_layer_rows)
    selected_steps = _select_steps(
        source_run.per_step_rows,
        strategy=str(e3_config["selection_strategy"]),
        max_steps=int(e3_config["max_steps_per_run"]),
    )
    topk_values = [int(value) for value in e3_config["topk_values"]]

    per_step_output_rows: list[dict[str, Any]] = []
    per_layer_output_rows: list[dict[str, Any]] = []
    for selection_rank, source_step_row in enumerate(selected_steps):
        step_idx = _safe_int(source_step_row.get("step"))
        input_ids = _step_input_ids(batches, step_idx, device)
        step_metrics, layer_output_rows = _evaluate_step_attribution(
            ref_model,
            tgt_model,
            input_ids,
            source_step_row,
            grouped_layers.get(step_idx, []),
            topk_values,
        )
        step_metrics["run_id"] = context.run_id
        step_metrics["source_run_id"] = source_run.run_id
        step_metrics["selection_rank"] = selection_rank
        per_step_output_rows.append(step_metrics)

        for row in layer_output_rows:
            row["run_id"] = context.run_id
            row["source_run_id"] = source_run.run_id
            per_layer_output_rows.append(row)

    summary_row = {
        "run_id": context.run_id,
        "source_run_id": source_run.run_id,
        "precision": source_run.precision,
        "sequence_length": source_run.sequence_length,
        "seed": source_run.seed,
        "num_selected_steps": len(per_step_output_rows),
        "mean_source_mismatch": _mean([float(row["source_mismatch"]) for row in per_step_output_rows]),
        "mean_recomputed_mismatch": _mean([float(row["recomputed_base_mismatch"]) for row in per_step_output_rows]),
        "mean_recompute_gap": _mean([float(row["recompute_gap"]) for row in per_step_output_rows]),
        "mean_spearman": _mean([float(row["spearman"]) for row in per_step_output_rows]),
        "mean_pairwise_accuracy": _mean([float(row["pairwise_accuracy"]) for row in per_step_output_rows]),
        "mean_top1_hit": _mean([float(row["top1_hit"]) for row in per_step_output_rows]),
        "mean_rank_gap": _mean([float(row["mean_rank_gap"]) for row in per_step_output_rows]),
    }
    for k in topk_values:
        summary_row[f"mean_topk_overlap_{k}"] = _mean([float(row[f"topk_overlap_{k}"]) for row in per_step_output_rows])

    context.write_rows("per_step_metrics.csv", per_step_output_rows, fieldnames=_metric_fieldnames(topk_values))
    context.write_rows("per_layer_metrics.csv", per_layer_output_rows, fieldnames=_layer_fieldnames())
    context.write_rows("summary_table.csv", [summary_row], fieldnames=_summary_row_fieldnames(topk_values))
    context.write_metrics(
        {
            "summary_row": summary_row,
            "source_run_id": source_run.run_id,
            "num_selected_steps": len(per_step_output_rows),
            "num_layer_rows": len(per_layer_output_rows),
        }
    )
    context.write_summary(
        {
            "goal": "Measure whether E2 proxy risk scores preserve the layer ordering induced by exact-ish single-layer precision patches.",
            "setup": [
                f"source_run={source_run.run_id}",
                f"model={source_run.model_name}",
                f"dataset={source_run.dataset_cfg['name']}:{source_run.dataset_cfg['config_name']}:{source_run.dataset_cfg['split']}",
                f"precision={source_run.precision}",
                f"sequence_length={source_run.sequence_length}",
                f"seed={source_run.seed}",
                f"selection_strategy={e3_config['selection_strategy']}",
                f"selected_steps={len(per_step_output_rows)}",
            ],
            "key_metrics": [
                f"mean Spearman={summary_row['mean_spearman']:.3f}",
                f"mean pairwise accuracy={summary_row['mean_pairwise_accuracy']:.3f}",
                f"mean top1 hit={summary_row['mean_top1_hit']:.3f}",
                f"mean recompute gap={summary_row['mean_recompute_gap']:.6f}",
            ],
            "pass_fail_verdict": "Pass" if summary_row["mean_spearman"] >= 0.5 else "Needs review",
            "anomalies": "Runtime verification required on the target HPC environment.",
            "follow_up": "Use the run-level ranking metrics and recompute-gap trend to populate the exact-vs-proxy fidelity subsection in the paper.",
        }
    )
    context.mark_completed(status="completed_unverified_runtime", extra_metadata={"source_run_id": source_run.run_id})
    context.append_stdout("Finished E3 attribution run.")

    del ref_model
    del tgt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary_row, per_layer_output_rows


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "start",
    color: str = TEXT_COLOR,
    weight: str = "normal",
) -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}">{safe}</text>'


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = AXIS_COLOR,
    width: float = 1.3,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"{dash_attr}/>'


def _svg_circle(x: float, y: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" opacity="0.80"/>'


def _svg_rect(x: float, y: float, width: float, height: float, *, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}" opacity="0.85"/>'


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def _render_summary_svg(layer_points: list[dict[str, Any]], run_rows: list[dict[str, Any]]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        _svg_text(48, 30, "E3 Exact-vs-Proxy Attribution Fidelity", size=20, weight="bold"),
        _svg_text(48, 50, "Single-layer fp32 patch reduction versus E2 proxy ranking on GPT-2 evaluation steps", size=12, color="#374151"),
    ]

    left_x0 = LEFT_PANEL_X
    left_y0 = PANEL_TOP
    left_x1 = LEFT_PANEL_X + PANEL_WIDTH
    left_y1 = PANEL_TOP + PANEL_HEIGHT
    parts.extend(
        [
            _svg_text(left_x0, left_y0 - 18, "Layer Rank Agreement", size=15, weight="bold"),
            _svg_line(left_x0, left_y0, left_x0, left_y1),
            _svg_line(left_x0, left_y1, left_x1, left_y1),
            _svg_line(left_x0 + 8, left_y1 - 8, left_x1 - 8, left_y0 + 8, color="#9ca3af", width=1.0, dash="6,4"),
            _svg_text((left_x0 + left_x1) / 2, left_y1 + 30, "proxy rank (1 = highest risk)", anchor="middle", size=12),
            _svg_text(left_x0 - 28, (left_y0 + left_y1) / 2, "exact-ish rank", anchor="middle", size=12),
        ]
    )
    if layer_points:
        x_values = [float(row["proxy_rank"]) for row in layer_points]
        y_values = [float(row["exact_rank"]) for row in layer_points]
        xmin, xmax = min(x_values), max(x_values)
        ymin, ymax = min(y_values), max(y_values)
        for row in layer_points:
            px = _scale(float(row["proxy_rank"]), xmin, xmax, left_x0 + 10, left_x1 - 10)
            py = _scale(float(row["exact_rank"]), ymin, ymax, left_y1 - 10, left_y0 + 10)
            parts.append(_svg_circle(px, py, PRECISION_COLOR.get(str(row["precision"]), "#111827")))

    right_x0 = RIGHT_PANEL_X
    right_y0 = PANEL_TOP
    right_x1 = RIGHT_PANEL_X + PANEL_WIDTH
    right_y1 = PANEL_TOP + PANEL_HEIGHT
    parts.extend(
        [
            _svg_text(right_x0, right_y0 - 18, "Run-Level Mean Spearman", size=15, weight="bold"),
            _svg_line(right_x0, right_y0, right_x0, right_y1),
            _svg_line(right_x0, right_y1, right_x1, right_y1),
            _svg_text((right_x0 + right_x1) / 2, right_y1 + 30, "source E2 run", anchor="middle", size=12),
            _svg_text(right_x0 - 25, (right_y0 + right_y1) / 2, "Spearman", anchor="middle", size=12),
        ]
    )
    if run_rows:
        bar_slot = PANEL_WIDTH / max(1, len(run_rows))
        zero_y = _scale(0.0, -1.0, 1.0, right_y1 - 10, right_y0 + 10)
        parts.append(_svg_line(right_x0 + 8, zero_y, right_x1 - 8, zero_y, color="#9ca3af", width=1.0))
        for idx, row in enumerate(run_rows):
            value = max(-1.0, min(1.0, float(row["mean_spearman"])))
            bar_top = _scale(value, -1.0, 1.0, right_y1 - 10, right_y0 + 10)
            bar_x = right_x0 + idx * bar_slot + 12
            bar_w = max(16.0, bar_slot - 24)
            rect_y = min(zero_y, bar_top)
            rect_h = max(2.0, abs(zero_y - bar_top))
            parts.append(_svg_rect(bar_x, rect_y, bar_w, rect_h, color=PRECISION_COLOR.get(str(row["precision"]), "#111827")))
            label = f"{str(row['precision']).upper()}-{int(row['sequence_length'])}-s{int(row['seed'])}"
            parts.append(_svg_text(bar_x + 0.5 * bar_w, right_y1 + 16, label, size=10, anchor="middle"))

    legend_y = SVG_HEIGHT - 42
    legend_x = 80
    for precision_name in ("fp16", "bf16", "fp32"):
        parts.append(_svg_circle(legend_x, legend_y - 5, PRECISION_COLOR[precision_name]))
        parts.append(_svg_text(legend_x + 12, legend_y, precision_name.upper(), size=12))
        legend_x += 92

    if run_rows:
        mean_spearman = _mean([float(row["mean_spearman"]) for row in run_rows])
        mean_pairwise = _mean([float(row["mean_pairwise_accuracy"]) for row in run_rows])
        parts.append(_svg_text(720, SVG_HEIGHT - 28, f"mean Spearman={mean_spearman:.3f}  mean pairwise accuracy={mean_pairwise:.3f}", size=12))

    parts.append("</svg>")
    return "\n".join(parts)


def _aggregate_report(config: dict[str, Any], run_rows: list[dict[str, Any]]) -> str:
    topk_values = [int(value) for value in config["topk_values"]]
    mean_spearman = _mean([float(row["mean_spearman"]) for row in run_rows])
    mean_pairwise = _mean([float(row["mean_pairwise_accuracy"]) for row in run_rows])
    mean_top1 = _mean([float(row["mean_top1_hit"]) for row in run_rows])
    best_run = max(run_rows, key=lambda row: float(row["mean_spearman"])) if run_rows else None

    lines = [
        "# E3 Attribution Report",
        "",
        "## Purpose",
        "",
        "Validate that the E2 layerwise proxy ranking is faithful to exact-ish single-layer precision patch reductions.",
        "",
        "## Protocol",
        "",
        "- Source experiment: E2 predictor runs",
        "- Exact-ish attribution: rerun the same manual GPT-2 forward path used by E2 while replacing one low-precision block with the FP32 reference block",
        "- Main fidelity metrics: Spearman, pairwise ordering accuracy, top-1 hit, top-k overlap",
        "",
        "## Sweep",
        "",
        f"- selection strategy: {config['selection_strategy']}",
        f"- max steps per run: {config['max_steps_per_run']}",
        f"- top-k values: {', '.join(str(value) for value in topk_values)}",
        f"- completed runs: {len(run_rows)}",
        "",
        "## Aggregate Metrics",
        "",
        f"- mean Spearman: {mean_spearman:.3f}",
        f"- mean pairwise accuracy: {mean_pairwise:.3f}",
        f"- mean top-1 hit: {mean_top1:.3f}",
    ]
    for k in topk_values:
        lines.append(f"- mean top-{k} overlap: {_mean([float(row[f'mean_topk_overlap_{k}']) for row in run_rows]):.3f}")

    if best_run is not None:
        lines.extend(
            [
                "",
                "## Best Run",
                "",
                f"- source run: {best_run['source_run_id']}",
                f"- precision: {best_run['precision']}",
                f"- sequence length: {int(best_run['sequence_length'])}",
                f"- seed: {int(best_run['seed'])}",
                f"- mean Spearman: {float(best_run['mean_spearman']):.3f}",
                f"- mean pairwise accuracy: {float(best_run['mean_pairwise_accuracy']):.3f}",
                f"- mean top-1 hit: {float(best_run['mean_top1_hit']):.3f}",
            ]
        )

    lines.extend(
        [
            "",
        "## Notes",
        "",
        "- `recompute_gap` checks whether E3 reproduces the same manual forward-path mismatch definition used in E2.",
        "- Negative exact reductions are kept as-is so that the layer ranking remains honest.",
        "- Execute on the HPC environment to materialize the run directories and outputs.",
        "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    script_path = Path(__file__).resolve()
    experiment_dir = script_path.parents[1]
    workspace_root = script_path.parents[2]
    config_path = Path(argv[0]).resolve() if argv else experiment_dir / "configs" / "default.json"
    config = load_config(config_path)
    source_dir = (experiment_dir / config["source_experiment"]).resolve()
    source_runs = _discover_source_runs(source_dir, config)
    device = _resolve_device(str(config.get("device", "auto")))

    tokenizer_cache: dict[tuple[str, bool], Any] = {}
    token_ids_cache: dict[tuple[str, bool, str, str, str, str], list[int]] = {}
    all_run_rows: list[dict[str, Any]] = []
    all_layer_points: list[dict[str, Any]] = []

    for source_run in source_runs:
        batches = _get_batches_for_run(
            source_run,
            tokenizer_cache=tokenizer_cache,
            token_ids_cache=token_ids_cache,
        )
        run_row, layer_points = _run_single_source_run(
            experiment_dir=experiment_dir,
            workspace_root=workspace_root,
            e3_config=config,
            source_run=source_run,
            device=device,
            batches=batches,
        )
        all_run_rows.append(run_row)
        all_layer_points.extend(layer_points)

    topk_values = [int(value) for value in config["topk_values"]]
    save_text_artifact(experiment_dir / "outputs", "e3_attribution_summary.svg", _render_summary_svg(all_layer_points, all_run_rows))
    save_text_artifact(experiment_dir / "outputs", "e3_attribution_report.md", _aggregate_report(config, all_run_rows))
    save_json_artifact(
        experiment_dir / "outputs",
        "e3_attribution_metrics.json",
        {
            "num_runs": len(all_run_rows),
            "num_layer_points": len(all_layer_points),
            "mean_spearman": _mean([float(row["mean_spearman"]) for row in all_run_rows]),
            "mean_pairwise_accuracy": _mean([float(row["mean_pairwise_accuracy"]) for row in all_run_rows]),
            "mean_top1_hit": _mean([float(row["mean_top1_hit"]) for row in all_run_rows]),
            **{
                f"mean_topk_overlap_{k}": _mean([float(row[f"mean_topk_overlap_{k}"]) for row in all_run_rows])
                for k in topk_values
            },
        },
    )
    write_rows(
        experiment_dir / "outputs" / "e3_attribution_run_metrics.csv",
        all_run_rows,
        fieldnames=_summary_row_fieldnames(topk_values),
    )
    write_rows(
        experiment_dir / "outputs" / "e3_attribution_layer_points.csv",
        all_layer_points,
        fieldnames=_layer_fieldnames(),
    )


if __name__ == "__main__":
    main()
