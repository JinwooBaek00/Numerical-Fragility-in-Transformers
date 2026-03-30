from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from common import create_run_context, load_config, manual_attention_forward, save_json_artifact, save_text_artifact, write_rows
else:
    from ...common import create_run_context, load_config, manual_attention_forward, save_json_artifact, save_text_artifact, write_rows

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E5 requires PyTorch to be installed on the execution environment.") from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E5 requires the `datasets` package on the execution environment.") from exc

try:
    from transformers import AutoTokenizer, GPT2LMHeadModel
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("E5 requires the `transformers` package on the execution environment.") from exc


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

POLICY_LABEL = {
    "none": "None",
    "static_global": "Static",
    "random_same_budget": "Random",
    "bgss": "BGSS",
}

POLICY_COLOR = {
    "none": "#6b7280",
    "static_global": "#ea580c",
    "random_same_budget": "#7c3aed",
    "bgss": "#1d4ed8",
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


@dataclass(frozen=True)
class PolicySeedSpec:
    policy: str
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


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    q = min(1.0, max(0.0, q))
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _sample_evenly(num_rows: int, max_rows: int) -> list[int]:
    if num_rows <= max_rows:
        return list(range(num_rows))
    if max_rows <= 1:
        return [0]
    return [round(i * (num_rows - 1) / (max_rows - 1)) for i in range(max_rows)]


def _rss(values: list[float]) -> float:
    total = 0.0
    for value in values:
        value_f = float(value)
        if math.isfinite(value_f):
            total += value_f * value_f
    return math.sqrt(total)


def _estimate_softmax_jacobian_norm(attn_probs: torch.Tensor, *, max_rows: int, power_iters: int) -> float:
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
    *,
    shuffle_windows: bool,
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
    if shuffle_windows:
        rng = random.Random(seed)
        rng.shuffle(windows)
    return windows


def _batch_windows(windows: list[list[int]], batch_size: int) -> list[torch.Tensor]:
    return [
        torch.tensor(windows[start : start + batch_size], dtype=torch.long)
        for start in range(0, len(windows), batch_size)
    ]


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


def _split_qkv(attn_module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = attn_module.c_attn(hidden_states)
    split_size = qkv.shape[-1] // 3
    query, key, value = qkv.split(split_size, dim=2)
    num_heads = attn_module.num_heads
    head_dim = split_size // num_heads

    def reshape(x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    return reshape(query), reshape(key), reshape(value)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(batch, seq, heads * head_dim)


def _spectral_norm_right(value: torch.Tensor, power_iters: int = 10) -> float:
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

        score_norms.append(_safe_norm(scores_h))
        prob_norms.append(_safe_norm(probs_h))
        q_norms.append(_safe_norm(query_h))
        k_norms.append(_safe_norm(key_h))
        v_norms.append(_safe_norm(value_h))
        v_operator_norms.append(_spectral_norm_right(value_h))
        d_smx_values.append(
            _estimate_softmax_jacobian_norm(
                probs_h,
                max_rows=softmax_row_samples,
                power_iters=softmax_power_iters,
            )
        )

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

    return attn_output, {
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


def _compute_loss_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
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
    bgss_ln_bonus_weight: float,
) -> dict[str, Any]:
    epsilon_mach = EPSILON_MACH[precision_name]
    transformer = model.transformer
    _annotate_static_block_surrogates(model)
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

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
        ln_bonus = 1.0 + bgss_ln_bonus_weight * max(row["ln_dominance"], 0.0)
        row["bgss_score"] = row["scaled_risk_score"] * ln_bonus * (1.0 + row["rho_ln"])

    return {
        "final_hidden": final_hidden,
        "loss": _compute_loss_from_logits(logits, input_ids),
        "layer_rows": layer_rows,
        "predicted_risk_sum": sum(row["risk_score"] for row in layer_rows),
        "scaled_predicted_risk_sum": epsilon_mach * sum(row["risk_score"] for row in layer_rows),
        "attention_only_sum": epsilon_mach * sum(attention_terms[idx] * downstream[idx] / final_norm for idx in range(len(layer_rows))),
        "layernorm_only_sum": epsilon_mach * sum(layernorm_terms[idx] * downstream[idx] / final_norm for idx in range(len(layer_rows))),
        "remainder_only_sum": epsilon_mach * sum(remainder_terms[idx] * downstream[idx] / final_norm for idx in range(len(layer_rows))),
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
    model.to(device=device, dtype=dtype)
    return model


def _prepare_models(
    model_name: str,
    train_precision: str,
    monitor_precision: str,
    device: torch.device,
    trust_remote_code: bool,
) -> tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
    master_model = _clone_model_to_precision(
        model_name,
        dtype=_dtype_name_to_torch(train_precision),
        device=device,
        trust_remote_code=trust_remote_code,
    )
    shadow_model = _clone_model_to_precision(
        model_name,
        dtype=_dtype_name_to_torch(monitor_precision),
        device=device,
        trust_remote_code=trust_remote_code,
    )
    _annotate_static_block_surrogates(shadow_model)
    return master_model, shadow_model


def _copy_weights_between_models(source_model: GPT2LMHeadModel, destination_model: GPT2LMHeadModel) -> None:
    destination_model.load_state_dict(source_model.state_dict(), strict=True)


def _grad_norm(model: GPT2LMHeadModel) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def _shadow_event_reason(
    *,
    mismatch_value: float,
    shadow_loss: float,
    mismatch_history: list[float],
    shadow_loss_history: list[float],
    warmup_steps: int,
    rolling_window: int,
    mismatch_spike_ratio: float,
    loss_spike_ratio: float,
) -> str:
    if not math.isfinite(mismatch_value):
        return "mismatch_nonfinite"
    if not math.isfinite(shadow_loss):
        return "shadow_loss_nonfinite"
    if len(mismatch_history) < max(warmup_steps, rolling_window):
        return ""

    mismatch_ref = max(_median(mismatch_history[-rolling_window:]), 1e-12)
    loss_ref = max(_median(shadow_loss_history[-rolling_window:]), 1e-12)
    reasons: list[str] = []
    if mismatch_value > mismatch_spike_ratio * mismatch_ref:
        reasons.append("mismatch_spike")
    if shadow_loss > loss_spike_ratio * loss_ref:
        reasons.append("shadow_loss_spike")
    return "+".join(reasons)


def _reason_tokens(reason: str) -> set[str]:
    if not reason:
        return set()
    return {token for token in reason.split("+") if token}


def _apply_epsilon_state(model: GPT2LMHeadModel, epsilon_by_layer: list[float]) -> None:
    for layer_idx, eps_value in enumerate(epsilon_by_layer):
        model.transformer.h[layer_idx].ln_1.eps = float(eps_value)
        model.transformer.h[layer_idx].ln_2.eps = float(eps_value)


def _initial_epsilon_state(model: GPT2LMHeadModel, config: dict[str, Any], policy: str) -> list[float]:
    num_layers = len(model.transformer.h)
    if policy == "static_global":
        return [float(config["static_epsilon"])] * num_layers
    return [float(config["base_epsilon"])] * num_layers


def _trigger_threshold(signal_history: list[float], config: dict[str, Any]) -> float:
    if not signal_history:
        return 0.0
    return _quantile(signal_history, float(config["controller_trigger_quantile"]))


def _eligible_layers(
    layer_rows: list[dict[str, Any]],
    epsilon_by_layer: list[float],
    cooldown_until: dict[int, int],
    action_count_by_layer: dict[int, int],
    *,
    step: int,
    config: dict[str, Any],
    policy: str,
) -> list[dict[str, Any]]:
    epsilon_max = float(config["epsilon_max"])
    max_actions_per_layer = int(config.get("max_actions_per_layer", 10**9))
    rows: list[dict[str, Any]] = []
    for row in layer_rows:
        layer_idx = int(row["layer"])
        if epsilon_by_layer[layer_idx] >= epsilon_max:
            continue
        if step < cooldown_until.get(layer_idx, -1):
            continue
        if action_count_by_layer.get(layer_idx, 0) >= max_actions_per_layer:
            continue
        rows.append(row)
    return rows


def _choose_action_layers(
    policy: str,
    eligible_rows: list[dict[str, Any]],
    *,
    max_actions: int,
    rng: random.Random,
    random_scope: str,
) -> list[dict[str, Any]]:
    if max_actions <= 0 or not eligible_rows:
        return []
    if policy == "bgss":
        ordered = sorted(
            eligible_rows,
            key=lambda row: (
                float(row.get("controller_priority", row["bgss_score"])),
                float(row["scaled_risk_score"]),
                float(row["downstream_transport"]),
            ),
            reverse=True,
        )
        return ordered[:max_actions]
    if policy == "random_same_budget":
        pool = eligible_rows
        if random_scope == "positive_bgss":
            pool = [row for row in eligible_rows if float(row["bgss_score"]) > 0.0]
            if not pool:
                pool = eligible_rows
        take = min(max_actions, len(pool))
        chosen_idx = sorted(rng.sample(range(len(pool)), take))
        return [pool[idx] for idx in chosen_idx]
    return []


def _controller_actions_for_step(
    *,
    policy: str,
    layer_rows: list[dict[str, Any]],
    combined_signal: float,
    signal_history: list[float],
    epsilon_by_layer: list[float],
    cooldown_until: dict[int, int],
    action_rows: list[dict[str, Any]],
    action_count: int,
    step: int,
    rng: random.Random,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if policy not in {"bgss", "random_same_budget"}:
        return []
    if step < int(config["controller_warmup_steps"]):
        return []
    if action_count >= int(config["max_total_actions"]):
        return []
    threshold = _trigger_threshold(signal_history, config)
    if combined_signal < threshold:
        return []

    action_count_by_layer: dict[int, int] = {}
    for action in action_rows:
        layer_idx = int(action["layer"])
        action_count_by_layer[layer_idx] = action_count_by_layer.get(layer_idx, 0) + 1

    eligible_rows = _eligible_layers(
        layer_rows,
        epsilon_by_layer,
        cooldown_until,
        action_count_by_layer,
        step=step,
        config=config,
        policy=policy,
    )
    epsilon_max = float(config["epsilon_max"])
    base_epsilon = float(config["base_epsilon"])
    epsilon_range = max(epsilon_max - base_epsilon, 1e-12)
    prepared_rows: list[dict[str, Any]] = []
    for row in eligible_rows:
        layer_idx = int(row["layer"])
        current_eps = float(epsilon_by_layer[layer_idx])
        headroom = max((epsilon_max - current_eps) / epsilon_range, 0.0)
        repetition_penalty = 1.0 / math.sqrt(1.0 + float(action_count_by_layer.get(layer_idx, 0)))
        controller_priority = float(row["bgss_score"]) * (0.5 + 0.5 * headroom) * repetition_penalty
        prepared_rows.append({**row, "controller_priority": controller_priority})

    remaining_budget = int(config["max_total_actions"]) - action_count
    max_actions = min(int(config["max_actions_per_step"]), remaining_budget)
    chosen_rows = _choose_action_layers(
        policy,
        prepared_rows,
        max_actions=max_actions,
        rng=rng,
        random_scope=str(config.get("random_candidate_scope", "all_layers")),
    )

    multiplier = float(config["epsilon_multiplier"])
    cooldown_steps = int(config["cooldown_steps"])
    headroom_weight = float(config.get("bgss_update_headroom_weight", 0.75))
    repetition_penalty_strength = float(config.get("bgss_update_repetition_penalty", 0.6))
    actions: list[dict[str, Any]] = []
    for offset, row in enumerate(chosen_rows, start=1):
        layer_idx = int(row["layer"])
        old_eps = float(epsilon_by_layer[layer_idx])
        layer_action_count_prev = int(action_count_by_layer.get(layer_idx, 0))
        headroom = max((epsilon_max - old_eps) / epsilon_range, 0.0)
        headroom_scale = (1.0 - headroom_weight) + headroom_weight * headroom
        repetition_scale = 1.0 / (1.0 + repetition_penalty_strength * layer_action_count_prev)
        effective_multiplier = 1.0 + (multiplier - 1.0) * headroom_scale * repetition_scale
        new_eps = min(epsilon_max, max(old_eps * effective_multiplier, float(config["base_epsilon"])))
        if new_eps <= old_eps:
            continue
        actions.append(
            {
                "step": step,
                "layer": layer_idx,
                "policy": policy,
                "old_epsilon": old_eps,
                "new_epsilon": new_eps,
                "effective_multiplier": effective_multiplier,
                "layer_action_count_prev": layer_action_count_prev,
                "action_kind": "controller_update",
                "trigger_score": float(row.get("controller_priority", row["bgss_score"])) if policy == "bgss" else float(row["scaled_risk_score"]),
                "step_signal": combined_signal,
                "cooldown_after": step + cooldown_steps,
                "action_index": action_count + offset,
            }
        )
    return actions


def _apply_actions(
    model: GPT2LMHeadModel,
    epsilon_by_layer: list[float],
    actions: list[dict[str, Any]],
    cooldown_until: dict[int, int],
) -> None:
    for action in actions:
        layer_idx = int(action["layer"])
        epsilon_by_layer[layer_idx] = float(action["new_epsilon"])
        cooldown_until[layer_idx] = int(action["cooldown_after"])
    _apply_epsilon_state(model, epsilon_by_layer)


def _training_batches(
    token_ids: list[int],
    config: dict[str, Any],
    seed: int,
) -> list[torch.Tensor]:
    windows = _build_token_windows(
        token_ids,
        int(config["sequence_length"]),
        int(config["max_train_sequences"]),
        seed,
        int(config.get("stride", 0)),
        shuffle_windows=bool(config.get("shuffle_windows", True)),
    )
    batches = _batch_windows(windows, int(config["batch_size"]))
    if not batches:
        raise RuntimeError("E5 could not build any training batches. Check sequence length or dataset configuration.")
    return batches


def _step_fieldnames() -> list[str]:
    return [
        "step",
        "policy",
        "seed",
        "precision",
        "sequence_length",
        "loss",
        "grad_norm",
        "final_mismatch",
        "predicted_risk_sum",
        "combined_scaled",
        "attention_only_sum",
        "layernorm_only_sum",
        "remainder_only_sum",
        "target_final_norm",
        "raw_mismatch_spike_flag",
        "shadow_loss_spike_flag",
        "immediate_event_flag",
        "event_reason",
        "num_active_layers",
        "mean_layer_epsilon",
        "max_layer_epsilon",
        "controller_triggered",
        "action_count_cum",
        "protected_layer_steps_cum",
    ]


def _layer_fieldnames() -> list[str]:
    return [
        "step",
        "policy",
        "layer",
        "seed",
        "precision",
        "sequence_length",
        "risk_score",
        "scaled_risk_score",
        "bgss_score",
        "ln_magnitude",
        "attn_magnitude",
        "remainder_magnitude",
        "ln_dominance",
        "rho_ln",
        "downstream_transport",
        "residual_transport_surrogate",
        "layer_epsilon",
    ]


def _action_fieldnames() -> list[str]:
    return [
        "step",
        "policy",
        "layer",
        "old_epsilon",
        "new_epsilon",
        "effective_multiplier",
        "layer_action_count_prev",
        "action_kind",
        "trigger_score",
        "step_signal",
        "cooldown_after",
        "action_index",
    ]


def _policy_summary_fieldnames() -> list[str]:
    return [
        "policy",
        "num_runs",
        "mean_num_events",
        "mean_num_raw_mismatch_spikes",
        "mean_num_shadow_loss_spikes",
        "mean_final_loss",
        "mean_max_mismatch",
        "mean_final_mismatch",
        "mean_action_count",
        "mean_protected_layer_steps",
        "mean_active_layers",
        "mean_event_free_rate",
        "mean_terminated_early",
    ]


def _run_single_policy_seed(
    experiment_dir: Path,
    workspace_root: Path,
    config: dict[str, Any],
    spec: PolicySeedSpec,
    *,
    device: torch.device,
    batches: list[torch.Tensor],
) -> dict[str, Any]:
    _seed_everything(spec.seed)
    rng = random.Random(spec.seed + 1009)
    train_precision = str(config.get("train_precision", "fp32"))
    monitor_precision = str(config["target_precision"])
    master_model, shadow_model = _prepare_models(
        config["model_name"],
        train_precision,
        monitor_precision,
        device,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    master_model.train()
    shadow_model.eval()

    epsilon_by_layer = _initial_epsilon_state(master_model, config, spec.policy)
    _apply_epsilon_state(master_model, epsilon_by_layer)
    _apply_epsilon_state(shadow_model, epsilon_by_layer)
    cooldown_until = {layer_idx: -1 for layer_idx in range(len(epsilon_by_layer))}

    optimizer = torch.optim.AdamW(
        master_model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    base_learning_rate = float(config["learning_rate"])
    lr_warmup_steps = int(config.get("lr_warmup_steps", 0))

    context = create_run_context(
        experiment_dir,
        short_tag=f"{spec.policy}_seed{spec.seed}",
        config={**config, "policy": spec.policy, "seed": spec.seed},
        metadata={
            "model_name": config["model_name"],
            "dataset_name": f"{config['dataset']['name']}:{config['dataset']['config_name']}:{config['dataset']['split']}",
            "precision": monitor_precision,
            "train_precision": train_precision,
            "seed": spec.seed,
            "sequence_length": int(config["sequence_length"]),
            "policy": spec.policy,
        },
        workspace_root=workspace_root,
    )
    start_message = (
        f"[E5] start policy={spec.policy} train_precision={train_precision} monitor_precision={monitor_precision} "
        f"seed={spec.seed} train_steps={config['max_train_steps']}"
    )
    context.append_stdout(start_message)
    context.append_stdout(
        f"Starting E5 BGSS run: policy={spec.policy}, train_precision={train_precision}, "
        f"monitor_precision={monitor_precision}, seed={spec.seed}, train_steps={config['max_train_steps']}"
    )
    print(start_message, flush=True)

    step_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    mismatch_history: list[float] = []
    shadow_loss_history: list[float] = []
    combined_history: list[float] = []
    protected_layer_steps = 0
    terminated_early = False
    progress_interval = max(1, int(config["max_train_steps"]) // 8)
    event_cooldown_steps = int(config.get("event_cooldown_steps", 0))
    last_event_onset_step: int | None = None

    if spec.policy == "static_global":
        for layer_idx, eps_value in enumerate(epsilon_by_layer):
            action_rows.append(
                {
                    "step": -1,
                    "policy": spec.policy,
                    "layer": layer_idx,
                    "old_epsilon": float(config["base_epsilon"]),
                    "new_epsilon": eps_value,
                    "action_kind": "static_init",
                    "trigger_score": 0.0,
                    "step_signal": 0.0,
                    "cooldown_after": -1,
                    "action_index": layer_idx + 1,
                }
            )

    for step in range(int(config["max_train_steps"])):
        batch = batches[step % len(batches)].to(device)
        optimizer.zero_grad(set_to_none=True)
        master_model.train()
        warmup_scale = 1.0 if lr_warmup_steps <= 0 else min(1.0, float(step + 1) / float(lr_warmup_steps))
        current_learning_rate = base_learning_rate * warmup_scale
        _set_optimizer_lr(optimizer, current_learning_rate)

        outputs = master_model(input_ids=batch, labels=batch)
        master_loss = outputs.loss
        master_loss_value = float(master_loss.detach().float().item())
        master_loss.backward()
        grad_norm_value = _grad_norm(master_model)

        clip_norm = float(config.get("gradient_clip_norm", 0.0))
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(master_model.parameters(), clip_norm)

        master_event_reason = ""
        if not math.isfinite(master_loss_value):
            master_event_reason = "master_loss_nonfinite"
        elif not math.isfinite(grad_norm_value):
            master_event_reason = "master_grad_nonfinite"
        if master_event_reason:
            event_rows.append(
                {
                    "step": step,
                    "policy": spec.policy,
                    "seed": spec.seed,
                    "precision": monitor_precision,
                    "sequence_length": int(config["sequence_length"]),
                    "learning_rate": current_learning_rate,
                    "loss": master_loss_value,
                    "grad_norm": grad_norm_value,
                    "final_mismatch": float("nan"),
                    "raw_mismatch_spike_flag": 0,
                    "shadow_loss_spike_flag": 0,
                    "immediate_event_flag": 1,
                    "event_reason": master_event_reason,
                }
            )
            print(
                f"[E5] event policy={spec.policy} train_precision={train_precision} monitor_precision={monitor_precision} "
                f"seed={spec.seed} step={step + 1}/{int(config['max_train_steps'])} reason={master_event_reason}",
                flush=True,
            )
            terminated_early = True
            break

        optimizer.step()

        if step % int(config["monitor_interval"]) == 0:
            _copy_weights_between_models(master_model, shadow_model)
            _apply_epsilon_state(shadow_model, epsilon_by_layer)
            _annotate_static_block_surrogates(shadow_model)
            master_model.eval()
            shadow_model.eval()
            with torch.no_grad():
                monitor_result = _instrumented_target_forward(
                    shadow_model,
                    batch,
                    precision_name=monitor_precision,
                    softmax_row_samples=int(config["softmax_row_samples"]),
                    softmax_power_iters=int(config["softmax_power_iters"]),
                    bgss_ln_bonus_weight=float(config.get("bgss_ln_bonus_weight", 1.0)),
                )
                reference_hidden = _reference_forward(master_model, batch)
                target_hidden = monitor_result["final_hidden"].detach().float()
                ref_hidden = reference_hidden.detach().float()
                mismatch = torch.norm(target_hidden - ref_hidden).item() / max(torch.norm(ref_hidden).item(), 1e-12)
                shadow_loss = float(monitor_result["loss"])

            event_reason = _shadow_event_reason(
                mismatch_value=mismatch,
                shadow_loss=shadow_loss,
                mismatch_history=mismatch_history,
                shadow_loss_history=shadow_loss_history,
                warmup_steps=int(config["event_warmup_steps"]),
                rolling_window=int(config["rolling_window"]),
                mismatch_spike_ratio=float(config["mismatch_spike_ratio"]),
                loss_spike_ratio=float(config["loss_spike_ratio"]),
            )
            tokens = _reason_tokens(event_reason)
            raw_mismatch_spike_flag = 1 if ("mismatch_spike" in tokens or "mismatch_nonfinite" in tokens) else 0
            shadow_loss_spike_flag = 1 if ("shadow_loss_spike" in tokens or "shadow_loss_nonfinite" in tokens) else 0
            immediate_event_flag = 0
            if raw_mismatch_spike_flag == 1:
                if last_event_onset_step is None or (step - last_event_onset_step) >= event_cooldown_steps:
                    immediate_event_flag = 1
                    last_event_onset_step = step
            event_rows.append(
                {
                    "step": step,
                    "policy": spec.policy,
                    "seed": spec.seed,
                    "precision": monitor_precision,
                    "sequence_length": int(config["sequence_length"]),
                    "learning_rate": current_learning_rate,
                    "loss": shadow_loss,
                    "grad_norm": grad_norm_value,
                    "final_mismatch": mismatch,
                    "raw_mismatch_spike_flag": raw_mismatch_spike_flag,
                    "shadow_loss_spike_flag": shadow_loss_spike_flag,
                    "immediate_event_flag": immediate_event_flag,
                    "event_reason": event_reason,
                }
            )
            if immediate_event_flag == 1:
                event_message = (
                    f"[E5] event policy={spec.policy} train_precision={train_precision} monitor_precision={monitor_precision} "
                    f"seed={spec.seed} step={step + 1}/{int(config['max_train_steps'])} reason={event_reason}"
                )
                print(event_message, flush=True)
                context.append_stdout(event_message)

            combined_signal = float(monitor_result["scaled_predicted_risk_sum"])
            combined_history.append(combined_signal)
            num_active_layers = sum(1 for eps_value in epsilon_by_layer if eps_value > float(config["base_epsilon"]) + 1e-18)
            protected_layer_steps += num_active_layers
            current_mean_epsilon = _mean(epsilon_by_layer)
            current_max_epsilon = max(epsilon_by_layer)
            current_epsilon_snapshot = list(epsilon_by_layer)

            actions = _controller_actions_for_step(
                policy=spec.policy,
                layer_rows=monitor_result["layer_rows"],
                combined_signal=combined_signal,
                signal_history=combined_history[:-1],
                epsilon_by_layer=epsilon_by_layer,
                cooldown_until=cooldown_until,
                action_rows=action_rows,
                action_count=len(action_rows),
                step=step,
                rng=rng,
                config=config,
            )
            if actions:
                _apply_actions(master_model, epsilon_by_layer, actions, cooldown_until)
                _apply_epsilon_state(shadow_model, epsilon_by_layer)
                action_rows.extend(actions)

            step_rows.append(
                {
                    "step": step,
                    "policy": spec.policy,
                    "seed": spec.seed,
                    "precision": monitor_precision,
                    "sequence_length": int(config["sequence_length"]),
                    "loss": shadow_loss,
                    "grad_norm": grad_norm_value,
                    "final_mismatch": mismatch,
                    "predicted_risk_sum": monitor_result["predicted_risk_sum"],
                    "combined_scaled": combined_signal,
                    "attention_only_sum": monitor_result["attention_only_sum"],
                    "layernorm_only_sum": monitor_result["layernorm_only_sum"],
                    "remainder_only_sum": monitor_result["remainder_only_sum"],
                    "target_final_norm": monitor_result["target_final_norm"],
                    "raw_mismatch_spike_flag": raw_mismatch_spike_flag,
                    "shadow_loss_spike_flag": shadow_loss_spike_flag,
                    "immediate_event_flag": immediate_event_flag,
                    "event_reason": event_reason,
                    "num_active_layers": num_active_layers,
                    "mean_layer_epsilon": current_mean_epsilon,
                    "max_layer_epsilon": current_max_epsilon,
                    "controller_triggered": 1 if actions else 0,
                    "action_count_cum": len(action_rows),
                    "protected_layer_steps_cum": protected_layer_steps,
                }
            )
            for row in monitor_result["layer_rows"]:
                layer_idx = int(row["layer"])
                layer_rows.append(
                    {
                        "step": step,
                        "policy": spec.policy,
                        "layer": layer_idx,
                        "seed": spec.seed,
                        "precision": monitor_precision,
                        "sequence_length": int(config["sequence_length"]),
                        "risk_score": row["risk_score"],
                        "scaled_risk_score": row["scaled_risk_score"],
                        "bgss_score": row["bgss_score"],
                        "ln_magnitude": row["ln_magnitude"],
                        "attn_magnitude": row["attn_magnitude"],
                        "remainder_magnitude": row["remainder_magnitude"],
                        "ln_dominance": row["ln_dominance"],
                        "rho_ln": row["rho_ln"],
                        "downstream_transport": row["downstream_transport"],
                        "residual_transport_surrogate": row["residual_transport_surrogate"],
                        "layer_epsilon": current_epsilon_snapshot[layer_idx],
                    }
                )
            if math.isfinite(mismatch) and math.isfinite(shadow_loss):
                mismatch_history.append(mismatch)
                shadow_loss_history.append(shadow_loss)
            master_model.train()
            if "nonfinite" in event_reason:
                terminated_early = True
                break

        if (step + 1) % progress_interval == 0 or (step + 1) == int(config["max_train_steps"]):
            progress_message = (
                f"[E5] progress policy={spec.policy} train_precision={train_precision} monitor_precision={monitor_precision} "
                f"seed={spec.seed} step={step + 1}/{int(config['max_train_steps'])} monitor_points={len(step_rows)} "
                f"events={sum(int(row['immediate_event_flag']) for row in event_rows)} actions={len(action_rows)}"
            )
            print(progress_message, flush=True)
            context.append_stdout(progress_message)

    num_events = sum(int(row["immediate_event_flag"]) for row in event_rows)
    num_raw_mismatch_spikes = sum(int(row.get("raw_mismatch_spike_flag", 0)) for row in event_rows)
    num_shadow_loss_spikes = sum(int(row.get("shadow_loss_spike_flag", 0)) for row in event_rows)
    first_event_step = next((int(row["step"]) for row in event_rows if int(row["immediate_event_flag"]) == 1), -1)
    final_step_row = step_rows[-1] if step_rows else None
    max_mismatch = max((float(row["final_mismatch"]) for row in step_rows), default=0.0)
    mean_active_layers = _mean([float(row["num_active_layers"]) for row in step_rows])

    context.write_rows("per_step_metrics.csv", step_rows, fieldnames=_step_fieldnames())
    context.write_rows("per_layer_metrics.csv", layer_rows, fieldnames=_layer_fieldnames())
    context.write_rows(
        "events.csv",
        event_rows,
        fieldnames=[
            "step",
            "policy",
            "seed",
            "precision",
            "sequence_length",
            "learning_rate",
            "loss",
            "grad_norm",
            "final_mismatch",
            "raw_mismatch_spike_flag",
            "shadow_loss_spike_flag",
            "immediate_event_flag",
            "event_reason",
        ],
    )
    context.write_rows("controller_actions.csv", action_rows, fieldnames=_action_fieldnames())
    context.write_rows(
        "summary_table.csv",
        [
            {
                "policy": spec.policy,
                "num_events": num_events,
                "num_raw_mismatch_spikes": num_raw_mismatch_spikes,
                "num_shadow_loss_spikes": num_shadow_loss_spikes,
                "first_event_step": first_event_step,
                "final_loss": 0.0 if final_step_row is None else float(final_step_row["loss"]),
                "final_mismatch": 0.0 if final_step_row is None else float(final_step_row["final_mismatch"]),
                "max_mismatch": max_mismatch,
                "action_count": len(action_rows),
                "protected_layer_steps": protected_layer_steps,
                "mean_active_layers": mean_active_layers,
                "terminated_early": 1 if terminated_early else 0,
            }
        ],
    )
    context.write_metrics(
        {
            "policy": spec.policy,
            "seed": spec.seed,
            "num_events": num_events,
            "num_raw_mismatch_spikes": num_raw_mismatch_spikes,
            "num_shadow_loss_spikes": num_shadow_loss_spikes,
            "first_event_step": first_event_step,
            "final_loss": 0.0 if final_step_row is None else float(final_step_row["loss"]),
            "final_mismatch": 0.0 if final_step_row is None else float(final_step_row["final_mismatch"]),
            "max_mismatch": max_mismatch,
            "action_count": len(action_rows),
            "protected_layer_steps": protected_layer_steps,
            "mean_active_layers": mean_active_layers,
            "terminated_early": terminated_early,
        }
    )
    context.write_summary(
        {
            "goal": "Compare BGSS against no intervention, static larger epsilon, and random same-budget intervention under stable master training with low-precision mismatch-onset control.",
            "setup": [
                f"policy={spec.policy}",
                f"model={config['model_name']}",
                f"dataset={config['dataset']['name']}:{config['dataset']['config_name']}:{config['dataset']['split']}",
                f"train_precision={train_precision}",
                f"monitor_precision={monitor_precision}",
                f"sequence_length={config['sequence_length']}",
                f"seed={spec.seed}",
                f"max_total_actions={config['max_total_actions']}",
                f"epsilon_max={config['epsilon_max']}",
            ],
            "key_metrics": [
                f"num_events={num_events}",
                f"num_raw_mismatch_spikes={num_raw_mismatch_spikes}",
                f"num_shadow_loss_spikes={num_shadow_loss_spikes}",
                f"max_mismatch={max_mismatch:.6f}",
                f"action_count={len(action_rows)}",
                f"protected_layer_steps={protected_layer_steps}",
            ],
            "pass_fail_verdict": "Pass" if spec.policy != "bgss" else "BGSS run completed",
            "anomalies": "Runtime verification required on the target HPC environment.",
            "follow_up": "Use the aggregate policy summary to build the budget-efficiency comparison under mismatch-onset control in the paper.",
        }
    )
    context.mark_completed(status="completed_unverified_runtime", extra_metadata={"policy": spec.policy, "terminated_early": terminated_early})
    context.append_stdout("Finished E5 BGSS run.")
    print(
        f"[E5] finished policy={spec.policy} train_precision={train_precision} monitor_precision={monitor_precision} "
        f"seed={spec.seed} monitor_points={len(step_rows)} events={num_events} actions={len(action_rows)} "
        f"terminated_early={int(terminated_early)}",
        flush=True,
    )

    del master_model
    del shadow_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": context.run_id,
        "policy": spec.policy,
        "precision": monitor_precision,
        "train_precision": train_precision,
        "monitor_precision": monitor_precision,
        "sequence_length": int(config["sequence_length"]),
        "seed": spec.seed,
        "num_monitor_points": len(step_rows),
        "num_events": num_events,
        "num_raw_mismatch_spikes": num_raw_mismatch_spikes,
        "num_shadow_loss_spikes": num_shadow_loss_spikes,
        "first_event_step": first_event_step,
        "final_loss": 0.0 if final_step_row is None else float(final_step_row["loss"]),
        "final_mismatch": 0.0 if final_step_row is None else float(final_step_row["final_mismatch"]),
        "max_mismatch": max_mismatch,
        "action_count": len(action_rows),
        "protected_layer_steps": protected_layer_steps,
        "mean_active_layers": mean_active_layers,
        "event_free": 1 if num_events == 0 else 0,
        "terminated_early": 1 if terminated_early else 0,
    }


def _aggregate_policy_rows(run_rows: list[dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    policy_rows: list[dict[str, Any]] = []
    for policy in policies:
        rows = [row for row in run_rows if row["policy"] == policy]
        if not rows:
            continue
        policy_rows.append(
            {
                "policy": policy,
                "num_runs": len(rows),
                "mean_num_events": _mean([float(row["num_events"]) for row in rows]),
                "mean_num_raw_mismatch_spikes": _mean([float(row["num_raw_mismatch_spikes"]) for row in rows]),
                "mean_num_shadow_loss_spikes": _mean([float(row["num_shadow_loss_spikes"]) for row in rows]),
                "mean_final_loss": _mean([float(row["final_loss"]) for row in rows]),
                "mean_max_mismatch": _mean([float(row["max_mismatch"]) for row in rows]),
                "mean_final_mismatch": _mean([float(row["final_mismatch"]) for row in rows]),
                "mean_action_count": _mean([float(row["action_count"]) for row in rows]),
                "mean_protected_layer_steps": _mean([float(row["protected_layer_steps"]) for row in rows]),
                "mean_active_layers": _mean([float(row["mean_active_layers"]) for row in rows]),
                "mean_event_free_rate": _mean([float(row["event_free"]) for row in rows]),
                "mean_terminated_early": _mean([float(row["terminated_early"]) for row in rows]),
            }
        )
    return policy_rows


def _svg_text(x: float, y: float, text: str, *, size: int = 14, anchor: str = "start", color: str = TEXT_COLOR, weight: str = "normal") -> str:
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}">{safe}</text>'


def _svg_line(x1: float, y1: float, x2: float, y2: float, *, color: str = AXIS_COLOR, width: float = 1.3, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width:.1f}"{dash_attr}/>'


def _svg_circle(x: float, y: float, color: str) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.0" fill="{color}" opacity="0.90"/>'


def _svg_rect(x: float, y: float, width: float, height: float, *, color: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{color}" opacity="0.85"/>'


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def _render_summary_svg(policy_rows: list[dict[str, Any]]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        _svg_text(48, 30, "E5 BGSS Under Budgeted Intervention", size=20, weight="bold"),
        _svg_text(48, 50, "Policy-level mismatch-onset control under fp32 master / fp16 shadow monitoring", size=12, color="#374151"),
    ]

    left_x0, left_y0 = LEFT_PANEL_X, PANEL_TOP
    left_x1, left_y1 = LEFT_PANEL_X + PANEL_WIDTH, PANEL_TOP + PANEL_HEIGHT
    parts.extend(
        [
            _svg_text(left_x0, left_y0 - 18, "Budget vs Mismatch Onsets", size=15, weight="bold"),
            _svg_line(left_x0, left_y0, left_x0, left_y1),
            _svg_line(left_x0, left_y1, left_x1, left_y1),
            _svg_text((left_x0 + left_x1) / 2, left_y1 + 30, "mean protected layer-steps", anchor="middle", size=12),
            _svg_text(left_x0 - 20, (left_y0 + left_y1) / 2, "mean onset events", anchor="middle", size=12),
        ]
    )
    if policy_rows:
        x_values = [float(row["mean_protected_layer_steps"]) for row in policy_rows]
        y_values = [float(row["mean_num_events"]) for row in policy_rows]
        xmin, xmax = min(x_values), max(x_values)
        ymin, ymax = min(y_values), max(y_values)
        for row in policy_rows:
            px = _scale(float(row["mean_protected_layer_steps"]), xmin, xmax, left_x0 + 10, left_x1 - 10)
            py = _scale(float(row["mean_num_events"]), ymin, ymax, left_y1 - 10, left_y0 + 10)
            color = POLICY_COLOR[str(row["policy"])]
            parts.append(_svg_circle(px, py, color))
            parts.append(_svg_text(px + 8, py - 6, POLICY_LABEL[str(row["policy"])], size=11))

    right_x0, right_y0 = RIGHT_PANEL_X, PANEL_TOP
    right_x1, right_y1 = RIGHT_PANEL_X + PANEL_WIDTH, PANEL_TOP + PANEL_HEIGHT
    parts.extend(
        [
            _svg_text(right_x0, right_y0 - 18, "Mean Final Mismatch", size=15, weight="bold"),
            _svg_line(right_x0, right_y0, right_x0, right_y1),
            _svg_line(right_x0, right_y1, right_x1, right_y1),
            _svg_text((right_x0 + right_x1) / 2, right_y1 + 30, "policy", anchor="middle", size=12),
            _svg_text(right_x0 - 20, (right_y0 + right_y1) / 2, "final mismatch", anchor="middle", size=12),
        ]
    )
    if policy_rows:
        slot = PANEL_WIDTH / max(1, len(policy_rows))
        max_mismatch = max(float(row["mean_final_mismatch"]) for row in policy_rows)
        for idx, row in enumerate(policy_rows):
            value = float(row["mean_final_mismatch"])
            bar_x = right_x0 + idx * slot + 18
            bar_w = max(24.0, slot - 36)
            bar_top = _scale(value, 0.0, max(max_mismatch, 1e-8), right_y1 - 8, right_y0 + 8)
            color = POLICY_COLOR[str(row["policy"])]
            parts.append(_svg_rect(bar_x, bar_top, bar_w, right_y1 - 8 - bar_top, color=color))
            parts.append(_svg_text(bar_x + 0.5 * bar_w, right_y1 + 16, POLICY_LABEL[str(row["policy"])], size=10, anchor="middle"))

    legend_y = SVG_HEIGHT - 42
    legend_x = 70
    for policy in ("none", "static_global", "random_same_budget", "bgss"):
        parts.append(_svg_circle(legend_x, legend_y - 5, POLICY_COLOR[policy]))
        parts.append(_svg_text(legend_x + 12, legend_y, POLICY_LABEL[policy], size=12))
        legend_x += 110

    parts.append("</svg>")
    return "\n".join(parts)


def _aggregate_report(config: dict[str, Any], policy_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# E5 BGSS Report",
        "",
        "## Purpose",
        "",
        "Compare BGSS against no intervention, static larger epsilon, and random same-budget intervention under stable fp32 master training with fp16 shadow mismatch-onset control.",
        "",
        "## Controller Protocol",
        "",
        f"- train precision: {config.get('train_precision', 'fp32')}",
        f"- monitor precision: {config['target_precision']}",
        f"- learning rate: {config['learning_rate']}",
        f"- lr warmup steps: {config.get('lr_warmup_steps', 0)}",
        f"- gradient clip norm: {config.get('gradient_clip_norm', 0.0)}",
        f"- event warmup steps: {config['event_warmup_steps']}",
        f"- rolling window: {config['rolling_window']}",
        f"- mismatch spike ratio: {config['mismatch_spike_ratio']}",
        f"- loss spike ratio: {config['loss_spike_ratio']}",
        f"- event cooldown steps: {config.get('event_cooldown_steps', 0)}",
        f"- base epsilon: {config['base_epsilon']}",
        f"- static epsilon: {config['static_epsilon']}",
        f"- epsilon multiplier: {config['epsilon_multiplier']}",
        f"- epsilon max: {config['epsilon_max']}",
        f"- max total actions: {config['max_total_actions']}",
        f"- max actions per step: {config['max_actions_per_step']}",
        f"- max actions per layer: {config.get('max_actions_per_layer', 'unbounded')}",
        f"- cooldown steps: {config['cooldown_steps']}",
        f"- BGSS LN bonus weight: {config.get('bgss_ln_bonus_weight', 1.0)}",
        f"- BGSS update headroom weight: {config.get('bgss_update_headroom_weight', 0.75)}",
        f"- BGSS repetition penalty: {config.get('bgss_update_repetition_penalty', 0.6)}",
        "",
        "## Aggregate Policy Summary",
        "",
    ]
    for row in policy_rows:
        lines.extend(
            [
                f"- {POLICY_LABEL[str(row['policy'])]} mean mismatch-onset events: {float(row['mean_num_events']):.3f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean raw mismatch spikes: {float(row['mean_num_raw_mismatch_spikes']):.3f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean shadow loss spikes: {float(row['mean_num_shadow_loss_spikes']):.3f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean max mismatch: {float(row['mean_max_mismatch']):.6f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean final mismatch: {float(row['mean_final_mismatch']):.6f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean final loss: {float(row['mean_final_loss']):.6f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean protected layer-steps: {float(row['mean_protected_layer_steps']):.3f}",
                f"- {POLICY_LABEL[str(row['policy'])]} mean action count: {float(row['mean_action_count']):.3f}",
            ]
        )

    best_budget_policy = min(policy_rows, key=lambda row: float(row["mean_num_events"]) + 0.001 * float(row["mean_protected_layer_steps"])) if policy_rows else None
    if best_budget_policy is not None:
        lines.extend(
            [
                "",
                "## Note",
                "",
                f"- best budget-efficiency proxy: {POLICY_LABEL[str(best_budget_policy['policy'])]}",
                "- Primary comparison should focus on mismatch-onset events versus protected layer-steps under the shared action budget.",
                "- Shadow-loss spikes are diagnostic only; the control target is low-precision mismatch onset.",
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
    device = _resolve_device(str(config.get("device", "auto")))

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = _load_token_ids(tokenizer, config["dataset"])

    run_rows: list[dict[str, Any]] = []
    for policy in config["policies"]:
        for seed in config["seeds"]:
            batches = _training_batches(token_ids, config, int(seed))
            run_rows.append(
                _run_single_policy_seed(
                    experiment_dir=experiment_dir,
                    workspace_root=workspace_root,
                    config=config,
                    spec=PolicySeedSpec(policy=str(policy), seed=int(seed)),
                    device=device,
                    batches=batches,
                )
            )

    policy_rows = _aggregate_policy_rows(run_rows, [str(policy) for policy in config["policies"]])

    save_text_artifact(experiment_dir / "outputs", "e5_bgss_summary.svg", _render_summary_svg(policy_rows))
    save_text_artifact(experiment_dir / "outputs", "e5_bgss_report.md", _aggregate_report(config, policy_rows))
    save_json_artifact(
        experiment_dir / "outputs",
        "e5_bgss_metrics.json",
        {
            "num_runs": len(run_rows),
            "policies": [str(policy) for policy in config["policies"]],
            "mean_events_by_policy": {str(row["policy"]): float(row["mean_num_events"]) for row in policy_rows},
            "mean_raw_mismatch_spikes_by_policy": {
                str(row["policy"]): float(row["mean_num_raw_mismatch_spikes"]) for row in policy_rows
            },
            "mean_shadow_loss_spikes_by_policy": {
                str(row["policy"]): float(row["mean_num_shadow_loss_spikes"]) for row in policy_rows
            },
            "mean_budget_by_policy": {str(row["policy"]): float(row["mean_protected_layer_steps"]) for row in policy_rows},
        },
    )
    write_rows(
        experiment_dir / "outputs" / "e5_bgss_run_metrics.csv",
        run_rows,
        fieldnames=[
            "run_id",
            "policy",
            "precision",
            "train_precision",
            "monitor_precision",
            "sequence_length",
            "seed",
            "num_monitor_points",
            "num_events",
            "num_raw_mismatch_spikes",
            "num_shadow_loss_spikes",
            "first_event_step",
            "final_loss",
            "final_mismatch",
            "max_mismatch",
            "action_count",
            "protected_layer_steps",
            "mean_active_layers",
            "event_free",
            "terminated_early",
        ],
    )
    write_rows(
        experiment_dir / "outputs" / "e5_bgss_policy_summary.csv",
        policy_rows,
        fieldnames=_policy_summary_fieldnames(),
    )


if __name__ == "__main__":
    main()
