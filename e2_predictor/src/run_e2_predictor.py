from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from common import (
        create_run_context,
        load_config,
        manual_attention_forward,
        save_json_artifact,
        save_text_artifact,
        write_rows,
    )
else:
    from ...common import (
        create_run_context,
        load_config,
        manual_attention_forward,
        save_json_artifact,
        save_text_artifact,
        write_rows,
    )

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E2 requires PyTorch to be installed on the execution environment.") from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E2 requires the `datasets` package on the execution environment.") from exc

try:
    from transformers import AutoTokenizer, GPT2LMHeadModel
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E2 requires the `transformers` package on the execution environment.") from exc


EPSILON_MACH = {
    "fp16": 2.0 ** -11,
    "bf16": 2.0 ** -8,
    "fp32": 2.0 ** -24,
}

TARGET_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

SVG_WIDTH = 1100
SVG_HEIGHT = 420
PANEL_WIDTH = 470
PANEL_HEIGHT = 280
PANEL_TOP = 80
SCATTER_LEFT = 60
CAL_LEFT = 580
TEXT_COLOR = "#111827"
AXIS_COLOR = "#1f2937"
PRECISION_COLOR = {
    "fp16": "#1d4ed8",
    "bf16": "#ea580c",
    "fp32": "#6b7280",
}
TOPK_FRACTION = 0.125


@dataclass(frozen=True)
class PrecisionRunSpec:
    target_precision: str
    sequence_length: int
    seed: int


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


def _safe_norm(tensor: torch.Tensor) -> float:
    values = tensor.detach().float()
    if values.numel() == 0:
        return 0.0
    finite_mask = torch.isfinite(values)
    if not torch.any(finite_mask):
        return 0.0
    safe_values = torch.where(finite_mask, values, torch.zeros_like(values))
    return torch.norm(safe_values).item()


def _normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector)
    if norm.item() == 0.0:
        return vector
    return vector / norm


def _candidate_start_vectors(length: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    if length <= 0:
        return []
    basis = torch.zeros(length, device=device, dtype=dtype)
    basis[0] = 1.0
    candidates = [
        basis,
        torch.ones(length, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, length, device=device, dtype=dtype),
    ]
    if length > 1:
        alt = torch.zeros(length, device=device, dtype=dtype)
        alt[0::2] = 1.0
        alt[1::2] = -1.0
        candidates.append(alt)
    return [_normalize_vector(candidate) for candidate in candidates if torch.norm(candidate).item() > 0.0]


def _finite_pairs(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0
    xs_f = [x for x, _ in pairs]
    ys_f = [y for _, y in pairs]
    mean_x = sum(xs_f) / len(xs_f)
    mean_y = sum(ys_f) / len(ys_f)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs_f))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys_f))
    if den_x == 0.0 or den_y == 0.0:
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
        average_rank = 0.5 * (idx + end) + 1.0
        for pos in range(idx, end + 1):
            ranks[order[pos][0]] = average_rank
        idx = end + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0
    xs_f = [x for x, _ in pairs]
    ys_f = [y for _, y in pairs]
    return _pearson(_ranks(xs_f), _ranks(ys_f))


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float]:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0, 0.0
    xs_f = [x for x, _ in pairs]
    ys_f = [y for _, y in pairs]
    mean_x = sum(xs_f) / len(xs_f)
    mean_y = sum(ys_f) / len(ys_f)
    var_x = sum((x - mean_x) ** 2 for x in xs_f)
    if var_x == 0.0:
        return mean_y, 0.0
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _r2(xs: list[float], ys: list[float]) -> float:
    pairs = _finite_pairs(xs, ys)
    if len(pairs) < 2:
        return 0.0
    xs_f = [x for x, _ in pairs]
    ys_f = [y for _, y in pairs]
    intercept, slope = _linear_regression(xs_f, ys_f)
    mean_y = sum(ys_f) / len(ys_f)
    ss_tot = sum((y - mean_y) ** 2 for y in ys_f)
    if ss_tot == 0.0:
        return 0.0
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs_f, ys_f))
    return max(0.0, 1.0 - ss_res / ss_tot)


def _loglog_slope(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in _finite_pairs(xs, ys) if x > 0.0 and y > 0.0]
    if len(pairs) < 2:
        return float("nan")
    log_x = [math.log10(x) for x, _ in pairs]
    log_y = [math.log10(y) for _, y in pairs]
    _, slope = _linear_regression(log_x, log_y)
    return slope


def _metric_bundle(xs: list[float], ys: list[float]) -> dict[str, float]:
    num_pairs = float(len(_finite_pairs(xs, ys)))
    return {
        "pearson": _pearson(xs, ys),
        "spearman": _spearman(xs, ys),
        "r2": _r2(xs, ys),
        "loglog_slope": _loglog_slope(xs, ys),
        "num_points": num_pairs,
    }


def _sample_evenly(num_rows: int, max_rows: int) -> list[int]:
    if num_rows <= max_rows:
        return list(range(num_rows))
    if max_rows <= 1:
        return [0]
    return [round(i * (num_rows - 1) / (max_rows - 1)) for i in range(max_rows)]


def _rss(values: Iterable[float]) -> float:
    total = 0.0
    for value in values:
        value_f = float(value)
        if math.isfinite(value_f):
            total += value_f * value_f
    return math.sqrt(total)


def _estimate_softmax_jacobian_norm(
    attn_probs: torch.Tensor,
    *,
    max_rows: int,
    power_iters: int,
) -> float:
    rows = attn_probs.detach().float().reshape(-1, attn_probs.shape[-1])
    if rows.numel() == 0:
        return 0.0
    indices = _sample_evenly(rows.shape[0], max_rows)
    best = 0.0
    for row_idx in indices:
        p = rows[row_idx]
        vec = torch.linspace(-1.0, 1.0, p.numel(), device=p.device, dtype=p.dtype)
        vec = vec - vec.mean()
        vec_norm = torch.norm(vec)
        if vec_norm.item() == 0.0:
            continue
        vec = vec / vec_norm
        for _ in range(power_iters):
            jv = p * vec - p * torch.dot(p, vec)
            norm_jv = torch.norm(jv)
            if norm_jv.item() == 0.0:
                break
            vec = jv / norm_jv
        jv = p * vec - p * torch.dot(p, vec)
        best = max(best, torch.norm(jv).item())
    return best


def _precision_scale(precision: str) -> float:
    if precision not in EPSILON_MACH:
        raise ValueError(f"Unknown precision name: {precision}")
    return EPSILON_MACH[precision]


def _resolve_stride(stride: int, sequence_length: int) -> int:
    return sequence_length if stride <= 0 else stride


def _load_token_ids(tokenizer: Any, dataset_cfg: dict[str, Any]) -> list[int]:
    dataset = load_dataset(
        dataset_cfg["name"],
        dataset_cfg["config_name"],
        split=dataset_cfg["split"],
    )
    text_field = dataset_cfg.get("text_field", "text")
    texts = [row[text_field] for row in dataset if row.get(text_field) and row[text_field].strip()]
    return tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]


def _build_token_windows(
    token_ids: list[int],
    sequence_length: int,
    max_sequences: int,
    seed: int,
    stride: int,
) -> list[list[int]]:
    step = _resolve_stride(stride, sequence_length)
    windows = []
    for start in range(0, max(0, len(token_ids) - sequence_length + 1), step):
        window = token_ids[start : start + sequence_length]
        if len(window) == sequence_length:
            windows.append(window)
    if max_sequences > 0 and len(windows) > max_sequences:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(range(len(windows)), max_sequences))
        windows = [windows[idx] for idx in chosen]
    return windows


def _batch_windows(windows: list[list[int]], batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        yield torch.tensor(chunk, dtype=torch.long)


def _layernorm_site_stats(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    epsilon_mach: float,
) -> dict[str, float]:
    x_f = x.detach().float()
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    variance = centered.pow(2).mean(dim=-1)
    variance_scalar = variance.mean().item()
    coeff = (eps + 2.0 * variance_scalar) / ((variance_scalar + eps) ** 1.5)
    z = centered / torch.sqrt(variance.unsqueeze(-1) + eps)
    z = z * weight.detach().float()
    magnitude = _safe_norm(z) * coeff
    rho_ln = (variance_scalar / eps) * x_f.shape[-1] * epsilon_mach
    return {
        "magnitude": magnitude,
        "rho_ln": rho_ln,
        "z_norm": _safe_norm(z),
        "coeff_mean": coeff,
    }


def _spectral_norm_right(value: torch.Tensor, power_iters: int = 12) -> float:
    value_f = value.detach().float()
    rows = value_f.reshape(-1, value_f.shape[-1])
    if rows.numel() == 0:
        return 0.0
    best = 0.0
    for vec in _candidate_start_vectors(rows.shape[-1], rows.device, rows.dtype):
        current = vec
        for _ in range(power_iters):
            left = rows @ current
            left_norm = torch.norm(left)
            if left_norm.item() == 0.0:
                current = torch.zeros_like(current)
                break
            left = left / left_norm
            current = rows.transpose(0, 1) @ left
            current_norm = torch.norm(current)
            if current_norm.item() == 0.0:
                current = torch.zeros_like(current)
                break
            current = current / current_norm
        if torch.norm(current).item() == 0.0:
            continue
        best = max(best, torch.norm(rows @ current).item())
    return best


def _module_operator_norm(module: Any, power_iters: int = 8) -> float:
    weight = getattr(module, "weight", None)
    if weight is None:
        return 0.0
    return _spectral_norm_right(weight.detach().float(), power_iters=power_iters)


def _diag_operator_norm(weight: torch.Tensor) -> float:
    values = weight.detach().float().abs()
    if values.numel() == 0:
        return 0.0
    return values.max().item()


def _annotate_static_block_surrogates(model: GPT2LMHeadModel) -> None:
    for block in model.transformer.h:
        block._nft_ln1_gamma_norm = _diag_operator_norm(block.ln_1.weight)
        block._nft_ln2_gamma_norm = _diag_operator_norm(block.ln_2.weight)
        block._nft_attn_proj_gain = _module_operator_norm(block.attn.c_proj)
        block._nft_mlp_fc_gain = _module_operator_norm(block.mlp.c_fc)
        block._nft_mlp_proj_gain = _module_operator_norm(block.mlp.c_proj)


def _gain_ratio(output_norm: float, input_norm: float) -> float:
    return output_norm / max(input_norm, 1e-12)


def _residual_transport_surrogate_from_components(
    *,
    hidden_in_norm: float,
    ln1_path_norm: float,
    projected_attn_norm: float,
    residual_after_attn_norm: float,
    ln2_path_norm: float,
    mlp_norm: float,
) -> dict[str, float]:
    ln1_gain = _gain_ratio(ln1_path_norm, hidden_in_norm)
    attn_kernel_gain = _gain_ratio(projected_attn_norm, ln1_path_norm)
    attn_branch_gain = ln1_gain * attn_kernel_gain

    ln2_gain = _gain_ratio(ln2_path_norm, residual_after_attn_norm)
    mlp_kernel_gain = _gain_ratio(mlp_norm, ln2_path_norm)
    mlp_branch_gain = ln2_gain * mlp_kernel_gain

    residual_transport_surrogate = attn_branch_gain + mlp_branch_gain * (1.0 + attn_branch_gain)
    return {
        "ln1_gain": ln1_gain,
        "attn_kernel_gain": attn_kernel_gain,
        "attn_branch_gain": attn_branch_gain,
        "ln2_gain": ln2_gain,
        "mlp_kernel_gain": mlp_kernel_gain,
        "mlp_branch_gain": mlp_branch_gain,
        "residual_transport_surrogate": residual_transport_surrogate,
    }


def _attention_forward_with_stats(
    attn_module: Any,
    hidden_states: torch.Tensor,
    *,
    layer_idx: int,
    softmax_row_samples: int,
    softmax_power_iters: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    attention = manual_attention_forward(attn_module, hidden_states, layer_idx=layer_idx)
    query = attention.query
    key = attention.key
    value = attention.value
    _, num_heads, _, head_dim = value.shape

    valid_scores = attention.valid_scores
    attn_probs = attention.attn_probs
    core_attn_output = attention.core_attn_output
    core_attn_norm = _safe_norm(core_attn_output)
    attn_output = attention.projected_attn_output
    projected_attn_norm = _safe_norm(attn_output)

    score_norms: list[float] = []
    prob_norms: list[float] = []
    q_norms: list[float] = []
    k_norms: list[float] = []
    v_norms: list[float] = []
    v_operator_norms: list[float] = []
    d_smx_values: list[float] = []
    for head_idx in range(num_heads):
        scores_h = valid_scores[:, head_idx, :, :]
        probs_h = attn_probs[:, head_idx, :, :]
        query_h = query[:, head_idx, :, :]
        key_h = key[:, head_idx, :, :]
        value_h = value[:, head_idx, :, :]

        score_norm_h = _safe_norm(scores_h)
        prob_norm_h = _safe_norm(probs_h)
        q_norm_h = _safe_norm(query_h)
        k_norm_h = _safe_norm(key_h)
        v_norm_h = _safe_norm(value_h)
        v_operator_norm_h = _spectral_norm_right(value_h)
        d_smx_h = _estimate_softmax_jacobian_norm(
            probs_h,
            max_rows=softmax_row_samples,
            power_iters=softmax_power_iters,
        )

        score_norms.append(score_norm_h)
        prob_norms.append(prob_norm_h)
        q_norms.append(q_norm_h)
        k_norms.append(k_norm_h)
        v_norms.append(v_norm_h)
        v_operator_norms.append(v_operator_norm_h)
        d_smx_values.append(d_smx_h)

    score_norm = _rss(score_norms)
    prob_norm = _rss(prob_norms)
    q_norm = _rss(q_norms)
    k_norm = _rss(k_norms)
    v_norm = _rss(v_norms)
    v_operator_norm = max(v_operator_norms) if v_operator_norms else 0.0
    d_smx = max(d_smx_values) if d_smx_values else 0.0
    kappa_softmax = 0.0 if prob_norm == 0.0 else (score_norm / prob_norm) * d_smx
    chi_score = 0.0 if prob_norm == 0.0 else (q_norm * k_norm / (prob_norm * math.sqrt(head_dim))) * d_smx
    attn_magnitude = (kappa_softmax + chi_score) * prob_norm * v_operator_norm
    kappa_val = 0.0 if core_attn_norm == 0.0 else (prob_norm * v_operator_norm) / core_attn_norm

    stats = {
        "score_norm": score_norm,
        "prob_norm": prob_norm,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "v_norm": v_norm,
        "v_operator_norm": v_operator_norm,
        "d_smx": d_smx,
        "kappa_softmax": kappa_softmax,
        "chi_score": chi_score,
        "kappa_val": kappa_val,
        "core_attn_norm": core_attn_norm,
        "projected_attn_norm": projected_attn_norm,
        "attn_magnitude": attn_magnitude,
    }
    return attn_output, stats


def _compute_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
    )
    return loss.item()


def _instrumented_target_forward(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    *,
    precision_name: str,
    softmax_row_samples: int,
    softmax_power_iters: int,
) -> dict[str, Any]:
    epsilon_mach = _precision_scale(precision_name)
    transformer = model.transformer
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    hidden_states = transformer.wte(input_ids) + transformer.wpe(position_ids)
    hidden_states = transformer.drop(hidden_states)

    layer_rows: list[dict[str, Any]] = []
    residual_surrogates: list[float] = []
    local_magnitudes: list[float] = []
    attention_terms: list[float] = []
    layernorm_terms: list[float] = []
    remainder_terms: list[float] = []

    for layer_idx, block in enumerate(transformer.h):
        hidden_in = hidden_states
        hidden_in_norm = max(_safe_norm(hidden_in), 1e-12)

        ln1_stats = _layernorm_site_stats(hidden_in, block.ln_1.weight, block.ln_1.eps, epsilon_mach)
        ln1_out = block.ln_1(hidden_in)
        attn_out, attn_stats = _attention_forward_with_stats(
            block.attn,
            ln1_out,
            layer_idx=layer_idx,
            softmax_row_samples=softmax_row_samples,
            softmax_power_iters=softmax_power_iters,
        )
        residual_after_attn = hidden_in + attn_out

        ln2_stats = _layernorm_site_stats(residual_after_attn, block.ln_2.weight, block.ln_2.eps, epsilon_mach)
        ln2_out = block.ln_2(residual_after_attn)
        mlp_fc = block.mlp.c_fc(ln2_out)
        mlp_act = block.mlp.act(mlp_fc)
        mlp_proj = block.mlp.c_proj(mlp_act)
        mlp_out = block.mlp.dropout(mlp_proj)
        hidden_states = residual_after_attn + mlp_out

        attn_proj_norm = attn_stats["projected_attn_norm"]
        mlp_norm = _safe_norm(mlp_out)
        residual_after_attn_norm = max(_safe_norm(residual_after_attn), 1e-12)
        ln1_mag = block._nft_ln1_gamma_norm * ln1_stats["coeff_mean"] * hidden_in_norm
        ln2_mag = block._nft_ln2_gamma_norm * ln2_stats["coeff_mean"] * residual_after_attn_norm
        ln_mag = ln1_mag + ln2_mag
        attn_proj_mag = block._nft_attn_proj_gain * attn_stats["core_attn_norm"]
        mlp_fc_mag = block._nft_mlp_fc_gain * max(_safe_norm(ln2_out), 1e-12)
        mlp_proj_mag = block._nft_mlp_proj_gain * max(_safe_norm(mlp_act), 1e-12)
        remainder_mag = attn_proj_mag + mlp_fc_mag + mlp_proj_mag
        local_mag = attn_stats["attn_magnitude"] + ln_mag + remainder_mag
        transport_stats = _residual_transport_surrogate_from_components(
            hidden_in_norm=hidden_in_norm,
            ln1_path_norm=ln1_stats["z_norm"],
            projected_attn_norm=attn_proj_norm,
            residual_after_attn_norm=residual_after_attn_norm,
            ln2_path_norm=ln2_stats["z_norm"],
            mlp_norm=mlp_norm,
        )
        residual_transport_surrogate = transport_stats["residual_transport_surrogate"]
        rho_ln = max(ln1_stats["rho_ln"], ln2_stats["rho_ln"])

        residual_surrogates.append(residual_transport_surrogate)
        local_magnitudes.append(local_mag)
        attention_terms.append(attn_stats["attn_magnitude"])
        layernorm_terms.append(ln_mag)
        remainder_terms.append(remainder_mag)

        layer_rows.append(
            {
                "layer": layer_idx,
                "attn_magnitude": attn_stats["attn_magnitude"],
                "ln_magnitude": ln_mag,
                "ln1_magnitude": ln1_mag,
                "ln2_magnitude": ln2_mag,
                "remainder_magnitude": remainder_mag,
                "attn_proj_magnitude": attn_proj_mag,
                "mlp_fc_magnitude": mlp_fc_mag,
                "mlp_proj_magnitude": mlp_proj_mag,
                "local_magnitude": local_mag,
                "ln1_gain": transport_stats["ln1_gain"],
                "attn_kernel_gain": transport_stats["attn_kernel_gain"],
                "attn_branch_gain": transport_stats["attn_branch_gain"],
                "ln2_gain": transport_stats["ln2_gain"],
                "mlp_kernel_gain": transport_stats["mlp_kernel_gain"],
                "mlp_branch_gain": transport_stats["mlp_branch_gain"],
                "residual_transport_surrogate": residual_transport_surrogate,
                "rho_ln": rho_ln,
                "kappa_softmax": attn_stats["kappa_softmax"],
                "chi_score": attn_stats["chi_score"],
                "kappa_val": attn_stats["kappa_val"],
                "softmax_jac_norm": attn_stats["d_smx"],
                "score_norm": attn_stats["score_norm"],
                "prob_norm": attn_stats["prob_norm"],
                "v_norm": attn_stats["v_norm"],
                "v_operator_norm": attn_stats["v_operator_norm"],
                "core_attn_norm": attn_stats["core_attn_norm"],
                "projected_attn_norm": attn_stats["projected_attn_norm"],
            }
        )

    final_hidden = transformer.ln_f(hidden_states)
    logits = model.lm_head(final_hidden)
    final_norm = max(_safe_norm(final_hidden), 1e-12)

    downstream = [1.0] * len(layer_rows)
    running = 1.0
    for idx in range(len(layer_rows) - 1, -1, -1):
        downstream[idx] = running
        running *= 1.0 + residual_surrogates[idx]

    for idx, row in enumerate(layer_rows):
        risk_score = (local_magnitudes[idx] / final_norm) * downstream[idx]
        row["risk_score"] = risk_score
        row["scaled_risk_score"] = epsilon_mach * risk_score
        row["ln_dominance"] = 0.0 if local_magnitudes[idx] == 0.0 else layernorm_terms[idx] / local_magnitudes[idx]
        row["downstream_transport"] = downstream[idx]
        row["no_transport_score"] = local_magnitudes[idx] / final_norm
        row["attention_only_score"] = (attention_terms[idx] / final_norm) * downstream[idx]
        row["layernorm_only_score"] = (layernorm_terms[idx] / final_norm) * downstream[idx]
        row["remainder_only_score"] = (remainder_terms[idx] / final_norm) * downstream[idx]

    return {
        "final_hidden": final_hidden,
        "loss": _compute_loss(logits, input_ids),
        "layer_rows": layer_rows,
        "predicted_risk_sum": sum(row["risk_score"] for row in layer_rows),
        "scaled_predicted_risk_sum": epsilon_mach * sum(row["risk_score"] for row in layer_rows),
        "attention_only_sum": epsilon_mach * sum(row["attention_only_score"] for row in layer_rows),
        "layernorm_only_sum": epsilon_mach * sum(row["layernorm_only_score"] for row in layer_rows),
        "remainder_only_sum": epsilon_mach * sum(row["remainder_only_score"] for row in layer_rows),
        "no_transport_sum": epsilon_mach * sum(row["no_transport_score"] for row in layer_rows),
        "target_final_norm": final_norm,
    }


def _reference_forward(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model.transformer(input_ids=input_ids, return_dict=True)
    return outputs.last_hidden_state


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
    _annotate_static_block_surrogates(tgt_model)
    return ref_model, tgt_model


def _make_metric_rows(per_step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatch = [float(row["final_mismatch"]) for row in per_step_rows]
    signal_map = {
        "combined_scaled": [float(row["scaled_predicted_risk_sum"]) for row in per_step_rows],
        "combined_raw": [float(row["predicted_risk_sum"]) for row in per_step_rows],
        "attention_only": [float(row["attention_only_sum"]) for row in per_step_rows],
        "layernorm_only": [float(row["layernorm_only_sum"]) for row in per_step_rows],
        "remainder_only": [float(row["remainder_only_sum"]) for row in per_step_rows],
        "no_transport": [float(row["no_transport_sum"]) for row in per_step_rows],
    }
    metric_rows = []
    for signal_name, signal_values in signal_map.items():
        bundle = _metric_bundle(signal_values, mismatch)
        metric_rows.append(
            {
                "signal_name": signal_name,
                "pearson": bundle["pearson"],
                "spearman": bundle["spearman"],
                "r2": bundle["r2"],
                "loglog_slope": bundle["loglog_slope"],
                "num_points": int(bundle["num_points"]),
            }
        )
    return metric_rows


def _metric_lookup(metric_rows: list[dict[str, Any]], signal_name: str, key: str) -> float:
    for row in metric_rows:
        if row["signal_name"] == signal_name:
            return float(row[key])
    return float("nan")


def _topk_overlap_ratio(
    signal_values: list[float],
    target_values: list[float],
    *,
    fraction: float = TOPK_FRACTION,
) -> float:
    if not signal_values or not target_values:
        return 0.0
    size = min(len(signal_values), len(target_values))
    k = max(1, round(size * fraction))

    def _top_indices(values: list[float]) -> set[int]:
        scored = []
        for idx, value in enumerate(values[:size]):
            value_f = float(value)
            if not math.isfinite(value_f):
                value_f = float("-inf")
            scored.append((idx, value_f))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        return {idx for idx, _ in ranked[:k]}

    signal_top = _top_indices(signal_values)
    target_top = _top_indices(target_values)
    return len(signal_top & target_top) / float(k)


def _run_support_metrics(
    per_step_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatch = [float(row["final_mismatch"]) for row in per_step_rows]
    combined = [float(row["scaled_predicted_risk_sum"]) for row in per_step_rows]
    no_transport = [float(row["no_transport_sum"]) for row in per_step_rows]

    combined_pearson = _metric_lookup(metric_rows, "combined_scaled", "pearson")
    combined_spearman = _metric_lookup(metric_rows, "combined_scaled", "spearman")
    no_transport_pearson = _metric_lookup(metric_rows, "no_transport", "pearson")
    no_transport_spearman = _metric_lookup(metric_rows, "no_transport", "spearman")

    combined_topk = _topk_overlap_ratio(combined, mismatch)
    no_transport_topk = _topk_overlap_ratio(no_transport, mismatch)

    delta_pearson = combined_pearson - no_transport_pearson
    delta_spearman = combined_spearman - no_transport_spearman
    delta_topk = combined_topk - no_transport_topk

    combined_positive = combined_pearson > 0.0 and combined_spearman > 0.0
    transport_improves = delta_pearson > 0.0 and delta_spearman > 0.0
    retrieval_improves = delta_topk > 0.0

    if combined_positive and transport_improves:
        verdict = "Supports transport-aware predictor"
    elif combined_positive and (delta_pearson > 0.0 or delta_spearman > 0.0 or retrieval_improves):
        verdict = "Partial support"
    else:
        verdict = "Needs review"

    return {
        "combined_pearson": combined_pearson,
        "combined_spearman": combined_spearman,
        "no_transport_pearson": no_transport_pearson,
        "no_transport_spearman": no_transport_spearman,
        "delta_pearson_vs_no_transport": delta_pearson,
        "delta_spearman_vs_no_transport": delta_spearman,
        "combined_topk_overlap": combined_topk,
        "no_transport_topk_overlap": no_transport_topk,
        "delta_topk_vs_no_transport": delta_topk,
        "combined_positive_correlation": combined_positive,
        "transport_improves_correlation": transport_improves,
        "transport_improves_retrieval": retrieval_improves,
        "support_verdict": verdict,
    }


def _run_single_combo(
    experiment_dir: Path,
    workspace_root: Path,
    config: dict[str, Any],
    spec: PrecisionRunSpec,
    device: torch.device,
    windows: list[list[int]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _seed_everything(spec.seed)
    ref_model, tgt_model = _prepare_models(
        config["model_name"],
        spec.target_precision,
        device,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )

    context = create_run_context(
        experiment_dir,
        short_tag=f"{spec.target_precision}_len{spec.sequence_length}_seed{spec.seed}",
        config={
            **config,
            "target_precision": spec.target_precision,
            "sequence_length": spec.sequence_length,
            "seed": spec.seed,
        },
        metadata={
            "model_name": config["model_name"],
            "dataset_name": f"{config['dataset']['name']}:{config['dataset']['config_name']}:{config['dataset']['split']}",
            "precision": spec.target_precision,
            "seed": spec.seed,
            "sequence_length": spec.sequence_length,
        },
        workspace_root=workspace_root,
    )
    context.append_stdout(
        f"Starting E2 run: precision={spec.target_precision}, sequence_length={spec.sequence_length}, seed={spec.seed}, windows={len(windows)}"
    )
    print(
        f"[E2] start precision={spec.target_precision} seq={spec.sequence_length} seed={spec.seed} windows={len(windows)}",
        flush=True,
    )

    per_step_rows: list[dict[str, Any]] = []
    per_layer_rows: list[dict[str, Any]] = []
    progress_interval = max(1, len(windows) // 8) if windows else 1

    with torch.no_grad():
        step_idx = 0
        for batch in _batch_windows(windows, int(config["batch_size"])):
            input_ids = batch.to(device)
            target_result = _instrumented_target_forward(
                tgt_model,
                input_ids,
                precision_name=spec.target_precision,
                softmax_row_samples=int(config["softmax_row_samples"]),
                softmax_power_iters=int(config["softmax_power_iters"]),
            )
            reference_hidden = _reference_forward(ref_model, input_ids)

            target_hidden = target_result["final_hidden"].detach().float()
            ref_hidden = reference_hidden.detach().float()
            mismatch = torch.norm(target_hidden - ref_hidden).item() / max(torch.norm(ref_hidden).item(), 1e-12)

            per_step_rows.append(
                {
                    "step": step_idx,
                    "seed": spec.seed,
                    "precision": spec.target_precision,
                    "sequence_length": spec.sequence_length,
                    "loss": target_result["loss"],
                    "final_mismatch": mismatch,
                    "predicted_risk_sum": target_result["predicted_risk_sum"],
                    "scaled_predicted_risk_sum": target_result["scaled_predicted_risk_sum"],
                    "attention_only_sum": target_result["attention_only_sum"],
                    "layernorm_only_sum": target_result["layernorm_only_sum"],
                    "remainder_only_sum": target_result["remainder_only_sum"],
                    "no_transport_sum": target_result["no_transport_sum"],
                    "target_final_norm": target_result["target_final_norm"],
                    "event_flag": 0,
                }
            )

            for row in target_result["layer_rows"]:
                per_layer_rows.append(
                    {
                        "step": step_idx,
                        "layer": row["layer"],
                        "seed": spec.seed,
                        "precision": spec.target_precision,
                        "sequence_length": spec.sequence_length,
                        "risk_score": row["risk_score"],
                        "ln_magnitude": row["ln_magnitude"],
                        "ln1_magnitude": row["ln1_magnitude"],
                        "ln2_magnitude": row["ln2_magnitude"],
                        "attn_magnitude": row["attn_magnitude"],
                        "remainder_magnitude": row["remainder_magnitude"],
                        "attn_proj_magnitude": row["attn_proj_magnitude"],
                        "mlp_fc_magnitude": row["mlp_fc_magnitude"],
                        "mlp_proj_magnitude": row["mlp_proj_magnitude"],
                        "local_magnitude": row["local_magnitude"],
                        "ln_dominance": row["ln_dominance"],
                        "rho_ln": row["rho_ln"],
                        "scaled_risk_score": row["scaled_risk_score"],
                        "downstream_transport": row["downstream_transport"],
                        "no_transport_score": row["no_transport_score"],
                        "attention_only_score": row["attention_only_score"],
                        "layernorm_only_score": row["layernorm_only_score"],
                        "remainder_only_score": row["remainder_only_score"],
                        "kappa_softmax": row["kappa_softmax"],
                        "chi_score": row["chi_score"],
                        "kappa_val": row["kappa_val"],
                        "softmax_jac_norm": row["softmax_jac_norm"],
                        "score_norm": row["score_norm"],
                        "prob_norm": row["prob_norm"],
                        "v_norm": row["v_norm"],
                        "v_operator_norm": row["v_operator_norm"],
                        "core_attn_norm": row["core_attn_norm"],
                        "projected_attn_norm": row["projected_attn_norm"],
                        "ln1_gain": row["ln1_gain"],
                        "attn_kernel_gain": row["attn_kernel_gain"],
                        "attn_branch_gain": row["attn_branch_gain"],
                        "ln2_gain": row["ln2_gain"],
                        "mlp_kernel_gain": row["mlp_kernel_gain"],
                        "mlp_branch_gain": row["mlp_branch_gain"],
                        "residual_transport_surrogate": row["residual_transport_surrogate"],
                    }
                )
            step_idx += 1
            if step_idx % progress_interval == 0 or step_idx == len(windows):
                progress = f"[E2] progress precision={spec.target_precision} seq={spec.sequence_length} seed={spec.seed} step={step_idx}/{len(windows)}"
                print(progress, flush=True)
                context.append_stdout(progress)

    metric_rows = _make_metric_rows(per_step_rows)
    context.write_rows(
        "per_step_metrics.csv",
        per_step_rows,
        fieldnames=[
            "step",
            "seed",
            "precision",
            "sequence_length",
            "loss",
            "final_mismatch",
            "predicted_risk_sum",
            "scaled_predicted_risk_sum",
            "attention_only_sum",
            "layernorm_only_sum",
            "remainder_only_sum",
            "no_transport_sum",
            "target_final_norm",
            "event_flag",
        ],
    )
    context.write_rows(
        "per_layer_metrics.csv",
        per_layer_rows,
        fieldnames=[
            "step",
            "layer",
            "seed",
            "precision",
            "sequence_length",
            "risk_score",
            "ln_magnitude",
            "ln1_magnitude",
            "ln2_magnitude",
            "attn_magnitude",
            "remainder_magnitude",
            "attn_proj_magnitude",
            "mlp_fc_magnitude",
            "mlp_proj_magnitude",
            "local_magnitude",
            "ln_dominance",
            "rho_ln",
            "scaled_risk_score",
            "downstream_transport",
            "no_transport_score",
            "attention_only_score",
            "layernorm_only_score",
            "remainder_only_score",
            "kappa_softmax",
            "chi_score",
            "kappa_val",
            "softmax_jac_norm",
            "score_norm",
            "prob_norm",
            "v_norm",
            "v_operator_norm",
            "core_attn_norm",
            "projected_attn_norm",
            "ln1_gain",
            "attn_kernel_gain",
            "attn_branch_gain",
            "ln2_gain",
            "mlp_kernel_gain",
            "mlp_branch_gain",
            "residual_transport_surrogate",
        ],
    )
    context.write_rows("summary_table.csv", metric_rows)
    support_metrics = _run_support_metrics(per_step_rows, metric_rows)
    context.write_metrics(
        {
            "metric_rows": metric_rows,
            "support_metrics": support_metrics,
            "num_steps": len(per_step_rows),
            "num_layer_rows": len(per_layer_rows),
            "precision": spec.target_precision,
            "sequence_length": spec.sequence_length,
            "seed": spec.seed,
        }
    )
    context.write_summary(
        {
            "goal": "Measure whether the transport-aware combined predictor tracks output-level mismatch on GPT-2 evaluation windows and improves over the no-transport ablation.",
            "setup": [
                f"model={config['model_name']}",
                f"dataset={config['dataset']['name']}:{config['dataset']['config_name']}:{config['dataset']['split']}",
                f"precision={spec.target_precision}",
                f"sequence_length={spec.sequence_length}",
                f"seed={spec.seed}",
                f"num_windows={len(per_step_rows)}",
            ],
            "key_metrics": [
                f"combined Pearson={support_metrics['combined_pearson']:.3f}",
                f"combined Spearman={support_metrics['combined_spearman']:.3f}",
                f"delta Pearson vs no_transport={support_metrics['delta_pearson_vs_no_transport']:.3f}",
                f"top-k overlap delta vs no_transport={support_metrics['delta_topk_vs_no_transport']:.3f}",
            ],
            "pass_fail_verdict": support_metrics["support_verdict"],
            "anomalies": "Runtime verification required on the target HPC environment.",
            "follow_up": "Reuse the per-layer logs from this run for E3 attribution; single-mechanism ablations remain diagnostic references rather than pass-fail targets.",
        }
    )
    context.mark_completed(status="completed_unverified_runtime")
    context.append_stdout("Finished E2 run.")
    print(
        f"[E2] finished precision={spec.target_precision} seq={spec.sequence_length} seed={spec.seed}",
        flush=True,
    )
    del ref_model
    del tgt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": context.run_id,
        "precision": spec.target_precision,
        "sequence_length": spec.sequence_length,
        "seed": spec.seed,
        "num_steps": len(per_step_rows),
        **support_metrics,
    }, per_step_rows, metric_rows


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", color: str = TEXT_COLOR, weight: str = "normal") -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}">{safe}</text>'


def _svg_line(x1: float, y1: float, x2: float, y2: float, *, color: str = AXIS_COLOR, width: float = 1.3) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"/>'


def _svg_circle(x: float, y: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{color}" opacity="0.8"/>'


def _svg_polyline(points: list[tuple[float, float]], *, color: str, width: float = 2.0) -> str:
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="{width:.1f}"/>'


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def _bin_trend(points: list[dict[str, Any]], bins: int) -> list[dict[str, float]]:
    valid = [row for row in points if row["scaled_predicted_risk_sum"] > 0.0 and row["final_mismatch"] > 0.0]
    if not valid:
        return []
    sorted_points = sorted(valid, key=lambda row: row["scaled_predicted_risk_sum"])
    rows = []
    for bin_idx in range(bins):
        start = round(bin_idx * len(sorted_points) / bins)
        end = round((bin_idx + 1) * len(sorted_points) / bins)
        chunk = sorted_points[start:end]
        if not chunk:
            continue
        rows.append(
            {
                "bin_index": float(bin_idx),
                "pred_mean": sum(row["scaled_predicted_risk_sum"] for row in chunk) / len(chunk),
                "mismatch_mean": sum(row["final_mismatch"] for row in chunk) / len(chunk),
            }
        )
    return rows


def _render_summary_svg(points: list[dict[str, Any]], trend_rows: list[dict[str, float]], run_metric_rows: list[dict[str, Any]]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        _svg_text(50, 30, "E2 GPT-2 Predictor Fidelity", size=20, weight="bold"),
        _svg_text(50, 50, "Combined practical predictor versus FP32 reference mismatch on WikiText-103 validation", size=12, color="#374151"),
    ]

    scatter_x0, scatter_y0 = SCATTER_LEFT, PANEL_TOP
    scatter_x1, scatter_y1 = SCATTER_LEFT + PANEL_WIDTH, PANEL_TOP + PANEL_HEIGHT
    cal_x0, cal_y0 = CAL_LEFT, PANEL_TOP
    cal_x1, cal_y1 = CAL_LEFT + PANEL_WIDTH, PANEL_TOP + PANEL_HEIGHT

    parts.extend(
        [
            _svg_text(scatter_x0, PANEL_TOP - 18, "Scaled Predictor vs Mismatch", size=15, weight="bold"),
            _svg_line(scatter_x0, scatter_y0, scatter_x0, scatter_y1),
            _svg_line(scatter_x0, scatter_y1, scatter_x1, scatter_y1),
            _svg_text((scatter_x0 + scatter_x1) / 2, scatter_y1 + 30, "log10(epsilon_mach * predicted risk sum)", anchor="middle", size=12),
            _svg_text(scatter_x0 - 35, (scatter_y0 + scatter_y1) / 2, "log10(mismatch)", anchor="middle", size=12),
        ]
    )
    scatter_points = [row for row in points if row["scaled_predicted_risk_sum"] > 0.0 and row["final_mismatch"] > 0.0]
    if scatter_points:
        xs = [math.log10(row["scaled_predicted_risk_sum"]) for row in scatter_points]
        ys = [math.log10(row["final_mismatch"]) for row in scatter_points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        for row, x_val, y_val in zip(scatter_points, xs, ys):
            px = _scale(x_val, xmin, xmax, scatter_x0 + 8, scatter_x1 - 8)
            py = _scale(y_val, ymin, ymax, scatter_y1 - 8, scatter_y0 + 8)
            parts.append(_svg_circle(px, py, PRECISION_COLOR.get(row["precision"], "#111827")))

    parts.extend(
        [
            _svg_text(cal_x0, PANEL_TOP - 18, "Binned Trend View", size=15, weight="bold"),
            _svg_line(cal_x0, cal_y0, cal_x0, cal_y1),
            _svg_line(cal_x0, cal_y1, cal_x1, cal_y1),
            _svg_text((cal_x0 + cal_x1) / 2, cal_y1 + 30, "log10(mean scaled predictor)", anchor="middle", size=12),
            _svg_text(cal_x0 - 35, (cal_y0 + cal_y1) / 2, "log10(mean mismatch)", anchor="middle", size=12),
        ]
    )
    if trend_rows:
        points_xy = []
        filtered = [row for row in trend_rows if row["pred_mean"] > 0.0 and row["mismatch_mean"] > 0.0]
        if filtered:
            x_logs = [math.log10(row["pred_mean"]) for row in filtered]
            y_logs = [math.log10(row["mismatch_mean"]) for row in filtered]
            xmin, xmax = min(x_logs), max(x_logs)
            ymin, ymax = min(y_logs), max(y_logs)
            for row in filtered:
                px = _scale(math.log10(row["pred_mean"]), xmin, xmax, cal_x0 + 8, cal_x1 - 8)
                py = _scale(math.log10(row["mismatch_mean"]), ymin, ymax, cal_y1 - 8, cal_y0 + 8)
                points_xy.append((px, py))
            parts.append(_svg_polyline(points_xy, color="#059669"))
            for point_x, point_y in points_xy:
                parts.append(_svg_circle(point_x, point_y, "#059669"))

    legend_y = SVG_HEIGHT - 45
    legend_x = 70
    for precision_name in ("fp16", "bf16", "fp32"):
        parts.append(_svg_circle(legend_x, legend_y - 5, PRECISION_COLOR[precision_name]))
        parts.append(_svg_text(legend_x + 12, legend_y, precision_name.upper(), size=12))
        legend_x += 90

    combined_rows = [row for row in run_metric_rows if row["signal_name"] == "combined_scaled"]
    no_transport_rows = [row for row in run_metric_rows if row["signal_name"] == "no_transport"]
    if combined_rows and no_transport_rows:
        mean_pearson = sum(float(row["pearson"]) for row in combined_rows) / len(combined_rows)
        mean_spearman = sum(float(row["spearman"]) for row in combined_rows) / len(combined_rows)
        mean_delta = mean_pearson - (sum(float(row["pearson"]) for row in no_transport_rows) / len(no_transport_rows))
        parts.append(_svg_text(660, SVG_HEIGHT - 28, f"combined mean Pearson={mean_pearson:.3f}  mean Spearman={mean_spearman:.3f}  delta vs no_transport={mean_delta:.3f}", size=12))

    parts.append("</svg>")
    return "\n".join(parts)


def _aggregate_report(
    config: dict[str, Any],
    run_summaries: list[dict[str, Any]],
    run_metric_rows: list[dict[str, Any]],
) -> str:
    combined_rows = [row for row in run_metric_rows if row["signal_name"] == "combined_scaled"]
    no_transport_rows = [row for row in run_metric_rows if row["signal_name"] == "no_transport"]
    best_combined = max(combined_rows, key=lambda row: float(row["pearson"])) if combined_rows else None
    mean_pearson = (
        sum(float(row["pearson"]) for row in combined_rows) / len(combined_rows)
        if combined_rows
        else float("nan")
    )
    mean_spearman = (
        sum(float(row["spearman"]) for row in combined_rows) / len(combined_rows)
        if combined_rows
        else float("nan")
    )
    mean_no_transport_pearson = (
        sum(float(row["pearson"]) for row in no_transport_rows) / len(no_transport_rows)
        if no_transport_rows
        else float("nan")
    )
    mean_no_transport_spearman = (
        sum(float(row["spearman"]) for row in no_transport_rows) / len(no_transport_rows)
        if no_transport_rows
        else float("nan")
    )
    mean_topk_overlap = (
        sum(float(row["combined_topk_overlap"]) for row in run_summaries) / len(run_summaries)
        if run_summaries
        else float("nan")
    )
    mean_no_transport_topk_overlap = (
        sum(float(row["no_transport_topk_overlap"]) for row in run_summaries) / len(run_summaries)
        if run_summaries
        else float("nan")
    )
    positive_runs = sum(1 for row in run_summaries if bool(row["combined_positive_correlation"]))
    transport_corr_runs = sum(1 for row in run_summaries if bool(row["transport_improves_correlation"]))
    transport_retrieval_runs = sum(1 for row in run_summaries if bool(row["transport_improves_retrieval"]))
    lines = [
        "# E2 Predictor Report",
        "",
        "## Purpose",
        "",
        "Validate that the transport-aware combined predictor tracks FP32 reference mismatch on GPT-2 evaluation windows and improves on the no-transport ablation.",
        "",
        "## Sweep",
        "",
        f"- model: {config['model_name']}",
        f"- precisions: {', '.join(config['target_precisions'])}",
        f"- sequence lengths: {', '.join(str(v) for v in config['sequence_lengths'])}",
        f"- seeds: {', '.join(str(v) for v in config['seeds'])}",
        f"- max sequences per run: {config['max_sequences_per_run']}",
        "",
        "## Aggregate Support Metrics",
        "",
        f"- combined_scaled mean Pearson: {mean_pearson:.3f}",
        f"- combined_scaled mean Spearman: {mean_spearman:.3f}",
        f"- no_transport mean Pearson: {mean_no_transport_pearson:.3f}",
        f"- no_transport mean Spearman: {mean_no_transport_spearman:.3f}",
        f"- mean Pearson delta vs no_transport: {mean_pearson - mean_no_transport_pearson:.3f}",
        f"- mean Spearman delta vs no_transport: {mean_spearman - mean_no_transport_spearman:.3f}",
        f"- combined top-k overlap mean: {mean_topk_overlap:.3f}",
        f"- no_transport top-k overlap mean: {mean_no_transport_topk_overlap:.3f}",
        f"- positive-correlation runs: {positive_runs}/{len(run_summaries)}",
        f"- transport-improved correlation runs: {transport_corr_runs}/{len(run_summaries)}",
        f"- transport-improved retrieval runs: {transport_retrieval_runs}/{len(run_summaries)}",
        f"- completed runs: {len(run_summaries)}",
    ]
    precisions = sorted({str(row["precision"]) for row in run_summaries})
    if precisions:
        lines.extend(
            [
                "",
                "## Precision Breakdown",
                "",
            ]
        )
        for precision in precisions:
            precision_runs = [row for row in run_summaries if str(row["precision"]) == precision]
            if not precision_runs:
                continue
            mean_precision_pearson = sum(float(row["combined_pearson"]) for row in precision_runs) / len(precision_runs)
            mean_precision_spearman = sum(float(row["combined_spearman"]) for row in precision_runs) / len(precision_runs)
            mean_precision_delta = sum(float(row["delta_pearson_vs_no_transport"]) for row in precision_runs) / len(precision_runs)
            support_count = sum(1 for row in precision_runs if str(row["support_verdict"]) == "Supports transport-aware predictor")
            partial_count = sum(1 for row in precision_runs if str(row["support_verdict"]) == "Partial support")
            lines.extend(
                [
                    f"- {precision}: mean Pearson={mean_precision_pearson:.3f}, mean Spearman={mean_precision_spearman:.3f}, mean Pearson delta vs no_transport={mean_precision_delta:.3f}, supports={support_count}/{len(precision_runs)}, partial={partial_count}/{len(precision_runs)}",
                ]
            )
    if best_combined is not None:
        lines.extend(
            [
                "",
                "## Best Combined Run",
                "",
                f"- precision: {best_combined['precision']}",
                f"- sequence_length: {best_combined['sequence_length']}",
                f"- seed: {best_combined['seed']}",
                f"- Pearson: {float(best_combined['pearson']):.3f}",
                f"- Spearman: {float(best_combined['spearman']):.3f}",
                f"- R2: {float(best_combined['r2']):.3f}",
                f"- log-log slope: {float(best_combined['loglog_slope']):.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `combined_scaled` is the main theory-guided practical predictor from the paper-facing estimator.",
            "- The primary E2 comparison is against `no_transport`, because the unified theory specifically adds downstream residual transport to local magnitudes.",
            "- Single-term ablations are diagnostic mechanism probes and are not expected to dominate the unified predictor on every run.",
            "- Execute on the HPC environment to materialize the run directories and outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def _augment_metric_rows(
    run_summary: dict[str, Any],
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [{**run_summary, **row} for row in metric_rows]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _coerce_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coerced: list[dict[str, Any]] = []
    for row in rows:
        coerced.append(
            {
                "step": int(row["step"]),
                "seed": int(row["seed"]),
                "precision": row["precision"],
                "sequence_length": int(row["sequence_length"]),
                "loss": float(row["loss"]),
                "final_mismatch": float(row["final_mismatch"]),
                "predicted_risk_sum": float(row["predicted_risk_sum"]),
                "scaled_predicted_risk_sum": float(row["scaled_predicted_risk_sum"]),
                "attention_only_sum": float(row["attention_only_sum"]),
                "layernorm_only_sum": float(row["layernorm_only_sum"]),
                "remainder_only_sum": float(row["remainder_only_sum"]),
                "no_transport_sum": float(row["no_transport_sum"]),
                "target_final_norm": float(row["target_final_norm"]),
                "event_flag": int(row.get("event_flag", 0)),
            }
        )
    return coerced


def _collect_existing_run_data(experiment_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_step_rows: list[dict[str, Any]] = []
    all_run_metric_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run_dir in sorted((experiment_dir / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        per_step_path = run_dir / "per_step_metrics.csv"
        if not per_step_path.exists():
            continue
        raw_step_rows = _read_csv_rows(per_step_path)
        if not raw_step_rows:
            continue
        per_step_rows = _coerce_step_rows(raw_step_rows)
        metric_rows = _make_metric_rows(per_step_rows)
        support_metrics = _run_support_metrics(per_step_rows, metric_rows)
        first = per_step_rows[0]
        run_summary = {
            "run_id": run_dir.name,
            "precision": first["precision"],
            "sequence_length": int(first["sequence_length"]),
            "seed": int(first["seed"]),
            "num_steps": len(per_step_rows),
            **support_metrics,
        }
        run_summaries.append(run_summary)
        all_step_rows.extend(per_step_rows)
        all_run_metric_rows.extend(_augment_metric_rows(run_summary, metric_rows))

    return run_summaries, all_step_rows, all_run_metric_rows


def _write_aggregate_outputs(
    experiment_dir: Path,
    config: dict[str, Any],
    run_summaries: list[dict[str, Any]],
    all_step_rows: list[dict[str, Any]],
    all_run_metric_rows: list[dict[str, Any]],
) -> None:
    trend_bins = int(config.get("trend_bins", config.get("calibration_bins", 12)))
    trend_rows = _bin_trend(all_step_rows, trend_bins)
    summary_svg = _render_summary_svg(all_step_rows, trend_rows, all_run_metric_rows)
    summary_report = _aggregate_report(config, run_summaries, all_run_metric_rows)

    save_text_artifact(experiment_dir / "outputs", "e2_predictor_summary.svg", summary_svg)
    save_text_artifact(experiment_dir / "outputs", "e2_predictor_report.md", summary_report)
    save_json_artifact(
        experiment_dir / "outputs",
        "e2_predictor_metrics.json",
        {
            "run_summaries": run_summaries,
            "num_points": len(all_step_rows),
            "combined_scaled_mean_pearson": (
                sum(float(row["pearson"]) for row in all_run_metric_rows if row["signal_name"] == "combined_scaled")
                / max(1, sum(1 for row in all_run_metric_rows if row["signal_name"] == "combined_scaled"))
            ),
            "no_transport_mean_pearson": (
                sum(float(row["pearson"]) for row in all_run_metric_rows if row["signal_name"] == "no_transport")
                / max(1, sum(1 for row in all_run_metric_rows if row["signal_name"] == "no_transport"))
            ),
            "transport_improved_correlation_runs": sum(
                1 for row in run_summaries if bool(row["transport_improves_correlation"])
            ),
        },
    )
    write_rows(
        experiment_dir / "outputs" / "e2_predictor_run_metrics.csv",
        all_run_metric_rows,
        fieldnames=[
            "run_id",
            "precision",
            "sequence_length",
            "seed",
            "num_steps",
            "signal_name",
            "pearson",
            "spearman",
            "r2",
            "loglog_slope",
            "num_points",
        ],
    )
    write_rows(
        experiment_dir / "outputs" / "e2_predictor_support_summary.csv",
        run_summaries,
        fieldnames=[
            "run_id",
            "precision",
            "sequence_length",
            "seed",
            "num_steps",
            "combined_pearson",
            "combined_spearman",
            "no_transport_pearson",
            "no_transport_spearman",
            "delta_pearson_vs_no_transport",
            "delta_spearman_vs_no_transport",
            "combined_topk_overlap",
            "no_transport_topk_overlap",
            "delta_topk_vs_no_transport",
            "combined_positive_correlation",
            "transport_improves_correlation",
            "transport_improves_retrieval",
            "support_verdict",
        ],
    )
    write_rows(
        experiment_dir / "outputs" / "e2_predictor_binned_trend.csv",
        trend_rows,
        fieldnames=["bin_index", "pred_mean", "mismatch_mean"],
    )


def main(argv: list[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    script_path = Path(__file__).resolve()
    experiment_dir = script_path.parents[1]
    workspace_root = script_path.parents[2]
    postprocess_only = bool(argv and argv[0] == "--postprocess-existing")
    config_arg_index = 1 if postprocess_only else 0
    config_path = Path(argv[config_arg_index]).resolve() if len(argv) > config_arg_index else experiment_dir / "configs" / "default.json"
    config = load_config(config_path)
    if postprocess_only:
        run_summaries, all_step_rows, all_run_metric_rows = _collect_existing_run_data(experiment_dir)
        _write_aggregate_outputs(experiment_dir, config, run_summaries, all_step_rows, all_run_metric_rows)
        return
    device = _resolve_device(config.get("device", "auto"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = _load_token_ids(tokenizer, config["dataset"])

    run_specs = [
        PrecisionRunSpec(target_precision=precision, sequence_length=int(seq_len), seed=int(seed))
        for precision in config["target_precisions"]
        for seq_len in config["sequence_lengths"]
        for seed in config["seeds"]
    ]

    all_step_rows: list[dict[str, Any]] = []
    all_run_metric_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for spec in run_specs:
        windows = _build_token_windows(
            token_ids,
            spec.sequence_length,
            int(config["max_sequences_per_run"]),
            spec.seed,
            int(config.get("stride", 0)),
        )
        run_summary, step_rows, metric_rows = _run_single_combo(
            experiment_dir=experiment_dir,
            workspace_root=workspace_root,
            config=config,
            spec=spec,
            device=device,
            windows=windows,
        )
        run_summaries.append(run_summary)
        all_step_rows.extend(step_rows)
        all_run_metric_rows.extend(_augment_metric_rows(run_summary, metric_rows))

    _write_aggregate_outputs(experiment_dir, config, run_summaries, all_step_rows, all_run_metric_rows)


if __name__ == "__main__":
    main()
