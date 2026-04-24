from __future__ import annotations

import os
import math
import random

import torch
import torch.distributed as dist

from stability_metric_utils import activation_quantile_metric_names, summarize_logit_extremes


def validate_wandb_setup() -> None:
    rank = os.environ.get("RANK")
    if rank not in (None, "0"):
        return
    require_wandb_api_key()
    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb must be installed because this trainer always logs to W&B in online mode.") from exc


def require_wandb_api_key() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be set because this trainer always logs to W&B in online mode.")


def setup_wandb(args, print0, *, rank: int):
    if rank != 0:
        return None
    require_wandb_api_key()

    import wandb

    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        dir=args.wandb_dir,
        config=vars(args),
        mode="online",
    )
    run.define_metric("step")
    run.define_metric("tokens_seen")
    run.define_metric("main/*", step_metric="step")
    run.define_metric("logits/*", step_metric="step")
    for metric_group in [
        "layer_embed",
        "layer_attn",
        "layer_mlp",
        "layer_final",
        "matrix_attn_q",
        "matrix_attn_k",
        "matrix_attn_v",
        "matrix_attn_proj",
        "matrix_mlp_fc",
        "matrix_mlp_proj",
        "matrix_embed",
        "matrix_lm_head",
        "matrix_other",
    ]:
        run.define_metric(f"{metric_group}/*", step_metric="step")
    print0(f"W&B run: {run.url}", console=True)
    return run


def should_log(step: int, interval: int) -> bool:
    return step == 1 or step % interval == 0


def tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def tensor_metrics_to_floats(metric_tensors: dict[str, torch.Tensor]) -> dict[str, float]:
    if not metric_tensors:
        return {}
    keys = sorted(metric_tensors)
    values = torch.stack([metric_tensors[key].detach().to(dtype=torch.float32) for key in keys]).cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def _rms_from_vector_norm(norm: torch.Tensor, count: int) -> torch.Tensor:
    return norm.float() / math.sqrt(max(count, 1))


def parameter_metric_name(name: str, kind: str) -> str:
    if name == "embed.weight":
        return f"matrix_embed/{kind}"
    if name == "proj.weight":
        return f"matrix_lm_head/{kind}"
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks":
        block_label = f"block_{int(parts[1]):02d}"
        if parts[2] == "attn":
            mapping = {"q": "matrix_attn_q", "k": "matrix_attn_k", "v": "matrix_attn_v", "proj": "matrix_attn_proj"}
            metric_group = mapping.get(parts[3], "matrix_other")
            metric_name = f"{block_label}_{kind}" if metric_group != "matrix_other" else f"{name.replace('.', '_')}_{kind}"
            return f"{metric_group}/{metric_name}"
        if parts[2] == "mlp":
            mapping = {"fc": "matrix_mlp_fc", "proj": "matrix_mlp_proj"}
            metric_group = mapping.get(parts[3], "matrix_other")
            metric_name = f"{block_label}_{kind}" if metric_group != "matrix_other" else f"{name.replace('.', '_')}_{kind}"
            return f"{metric_group}/{metric_name}"
    return f"matrix_other/{name.replace('.', '_')}_{kind}"


def collect_norm_metrics(model, *, include_matrix: bool) -> tuple[dict[str, float], dict[str, float]]:
    device = next(model.parameters()).device
    zero = lambda: torch.zeros((), device=device, dtype=torch.float32)
    global_grad_sumsq = zero()
    global_param_sumsq = zero()
    global_grad_count = zero()
    global_param_count = zero()
    grad_nonfinite = zero()
    param_nonfinite = zero()
    grad_max_abs = zero()
    param_max_abs = zero()
    matrix_metrics: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        detached_param = param.detach()
        param_norm = torch.linalg.vector_norm(detached_param).float()
        param_sq = param_norm.square()
        param_count = detached_param.new_tensor(detached_param.numel(), dtype=torch.float32)
        param_rms = _rms_from_vector_norm(param_norm, detached_param.numel())
        global_param_sumsq += param_sq
        global_param_count += param_count
        param_nonfinite += (~torch.isfinite(detached_param)).sum()
        param_max_abs = torch.maximum(param_max_abs, detached_param.abs().amax().float())

        if include_matrix:
            matrix_metrics[parameter_metric_name(name, "param_rms")] = param_rms

        if param.grad is None:
            continue

        detached_grad = param.grad.detach()
        grad_norm = torch.linalg.vector_norm(detached_grad).float()
        grad_sq = grad_norm.square()
        grad_count = detached_grad.new_tensor(detached_grad.numel(), dtype=torch.float32)
        grad_rms = _rms_from_vector_norm(grad_norm, detached_grad.numel())
        global_grad_sumsq += grad_sq
        global_grad_count += grad_count
        grad_nonfinite += (~torch.isfinite(detached_grad)).sum()
        grad_max_abs = torch.maximum(grad_max_abs, detached_grad.abs().amax().float())

        if include_matrix:
            matrix_metrics[parameter_metric_name(name, "grad_rms")] = grad_rms

    main_metric_tensors = {
        "main/global_grad_rms": (global_grad_sumsq / global_grad_count.clamp_min(1.0)).sqrt(),
        "main/global_param_rms": (global_param_sumsq / global_param_count.clamp_min(1.0)).sqrt(),
        "main/grad_nonfinite_count": grad_nonfinite,
        "main/param_nonfinite_count": param_nonfinite,
        "main/grad_max_abs": grad_max_abs,
        "main/param_max_abs": param_max_abs,
    }
    main_metrics = tensor_metrics_to_floats(main_metric_tensors)
    return main_metrics, tensor_metrics_to_floats(matrix_metrics) if include_matrix else {}


def log_run_metrics(run, metrics: dict[str, float]) -> None:
    if run is not None and metrics:
        run.log(metrics)


def log_static_model_metrics(run, print0, *, model) -> None:
    model_num_params = sum(param.numel() for param in model.parameters())
    model_param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    static_metrics = {
        "step": 0,
        "tokens_seen": 0,
        "main/model_num_params": float(model_num_params),
        "main/model_param_bytes": float(model_param_bytes),
    }
    print0(
        f"Model params: {model_num_params:,} ({model_param_bytes / (1024 ** 2):.2f} MiB parameter storage)",
        console=True,
    )
    if run is not None:
        run.summary["main/model_num_params"] = model_num_params
        run.summary["main/model_param_bytes"] = model_param_bytes
    log_run_metrics(run, static_metrics)


def _add_mean_rms(metric_tensors: dict[str, torch.Tensor], name: str, tensor: torch.Tensor) -> None:
    detached = tensor.detach()
    norm = torch.linalg.vector_norm(detached).float()
    zero = torch.zeros((), device=norm.device, dtype=torch.float32)
    metric_tensors[f"{name}__sumsq"] = metric_tensors.get(f"{name}__sumsq", zero) + norm.square()
    metric_tensors[f"{name}__count"] = metric_tensors.get(f"{name}__count", zero) + norm.new_tensor(detached.numel(), dtype=torch.float32)


def _sample_abs_values(tensor: torch.Tensor, *, max_values: int = 4096) -> torch.Tensor:
    values = tensor.detach().reshape(-1)
    if values.numel() > max_values:
        sampled_indices = random.sample(range(values.numel()), max_values)
        indices = torch.tensor(sampled_indices, device=values.device, dtype=torch.int64)
        values = values.index_select(0, indices)
    return values.abs().float().cpu()


def _merge_topk(existing: torch.Tensor | None, candidate: torch.Tensor, *, k: int, largest: bool) -> torch.Tensor:
    candidate = candidate.detach().float().cpu()
    if existing is None:
        combined = candidate
    else:
        combined = torch.cat((existing, candidate))
    if combined.numel() == 0:
        return combined
    return torch.topk(combined, k=min(k, combined.numel()), largest=largest).values


def collect_stability_metrics(
    model,
    inputs: torch.Tensor,
    *,
    micro_batch_size: int,
    max_sequences: int,
    rank: int,
) -> dict[str, float]:
    metric_tensors: dict[str, torch.Tensor] = {}
    abs_samples: dict[str, list[torch.Tensor]] = {}
    softcap = model.config.logit_softcap
    sample_count = min(len(inputs), max_sequences)
    seq_len = inputs.size(1)
    saturation_threshold = softcap * 0.95 / math.sqrt(1 - 0.95**2)
    local_top_logits: torch.Tensor | None = None
    local_bottom_logits: torch.Tensor | None = None

    def observe(name: str, tensor: torch.Tensor) -> None:
        if name.startswith("matrix_"):
            abs_samples.setdefault(name, []).append(_sample_abs_values(tensor))
            return
        _add_mean_rms(metric_tensors, name, tensor)

    with torch.inference_mode():
        for offset in range(0, sample_count, micro_batch_size):
            input_chunk = inputs[offset:min(offset + micro_batch_size, sample_count)]
            logits = model.compute_raw_logits(input_chunk, observer=observe)
            flat_logits = logits.detach().reshape(-1).float()
            local_top_logits = _merge_topk(local_top_logits, flat_logits, k=5, largest=True)
            local_bottom_logits = _merge_topk(local_bottom_logits, flat_logits, k=5, largest=False)
            zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            metric_tensors["logits/sum"] = metric_tensors.get("logits/sum", zero) + logits.sum()
            metric_tensors["logits/sumsq"] = metric_tensors.get("logits/sumsq", zero) + torch.linalg.vector_norm(logits).square()
            metric_tensors["logits/count"] = metric_tensors.get("logits/count", zero) + logits.new_tensor(logits.numel(), dtype=torch.float32)
            metric_tensors["logits/max_abs"] = torch.maximum(metric_tensors.get("logits/max_abs", zero), logits.abs().amax())
            pos_sat_count = torch.count_nonzero(logits >= saturation_threshold).float()
            neg_sat_count = torch.count_nonzero(logits <= -saturation_threshold).float()
            metric_tensors["logits/softcap_pos_sat_count"] = metric_tensors.get("logits/softcap_pos_sat_count", zero) + pos_sat_count
            metric_tensors["logits/softcap_neg_sat_count"] = metric_tensors.get("logits/softcap_neg_sat_count", zero) + neg_sat_count

    if not metric_tensors:
        return {}

    sum_keys = sorted(key for key in metric_tensors if key != "logits/max_abs")
    reduced_sum_values = torch.stack([metric_tensors[key].detach().to(dtype=torch.float32) for key in sum_keys])
    dist.all_reduce(reduced_sum_values, op=dist.ReduceOp.SUM)
    reduced_sums = {key: value for key, value in zip(sum_keys, reduced_sum_values)}

    reduced_max = metric_tensors["logits/max_abs"].detach().to(dtype=torch.float32)
    dist.all_reduce(reduced_max, op=dist.ReduceOp.MAX)

    local_abs_samples = {
        name: torch.cat(values) if len(values) > 1 else values[0]
        for name, values in abs_samples.items()
    }
    gathered_abs_samples = [None] * dist.get_world_size() if rank == 0 else None
    gathered_top_logits = [None] * dist.get_world_size() if rank == 0 else None
    gathered_bottom_logits = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_abs_samples, gathered_abs_samples, dst=0)
    dist.gather_object(local_top_logits, gathered_top_logits, dst=0)
    dist.gather_object(local_bottom_logits, gathered_bottom_logits, dst=0)

    if rank != 0:
        return {}

    finalized: dict[str, float] = {}
    finalized["main/stability_sample_sequences_per_rank"] = float(sample_count)
    finalized["main/stability_sample_tokens_per_rank"] = float(sample_count * seq_len)
    finalized["main/stability_sample_fraction_per_rank"] = float(sample_count / max(len(inputs), 1))
    mean_metric_names = sorted(key.removesuffix("__sumsq") for key in reduced_sums if key.endswith("__sumsq"))
    for key in mean_metric_names:
        total_sum = reduced_sums[f"{key}__sumsq"]
        total_count = reduced_sums[f"{key}__count"].clamp_min(1.0)
        finalized[key] = float((total_sum / total_count).sqrt().cpu().item())

    logits_count = reduced_sums["logits/count"].clamp_min(1.0)
    logits_mean = reduced_sums["logits/sum"] / logits_count
    logits_var = reduced_sums["logits/sumsq"] / logits_count - logits_mean.square()
    finalized["logits/mean"] = float(logits_mean.cpu().item())
    finalized["logits/std"] = float(logits_var.clamp_min(0.0).sqrt().cpu().item())
    finalized["logits/max_abs"] = float(reduced_max.cpu().item())
    top_logits = _merge_topk(None, torch.cat([values for values in gathered_top_logits if values is not None]), k=5, largest=True)
    bottom_logits = _merge_topk(None, torch.cat([values for values in gathered_bottom_logits if values is not None]), k=5, largest=False)
    finalized.update(
        summarize_logit_extremes(
            top_values=top_logits.tolist(),
            bottom_values=bottom_logits.tolist(),
        )
    )
    pos_sat_frac = reduced_sums["logits/softcap_pos_sat_count"] / logits_count
    neg_sat_frac = reduced_sums["logits/softcap_neg_sat_count"] / logits_count
    finalized["logits/softcap_positive_saturation_frac"] = float(pos_sat_frac.cpu().item())
    finalized["logits/softcap_negative_saturation_frac"] = float(neg_sat_frac.cpu().item())
    for name in sorted({key for payload in gathered_abs_samples for key in (payload or {})}):
        sample_tensor = torch.cat([payload[name] for payload in gathered_abs_samples if payload is not None and name in payload])
        quantiles = torch.quantile(sample_tensor, torch.tensor([0.5, 0.9, 0.99]))
        metric_names = activation_quantile_metric_names(name)
        finalized[metric_names["p50"]] = float(quantiles[0].item())
        finalized[metric_names["p90"]] = float(quantiles[1].item())
        finalized[metric_names["p99"]] = float(quantiles[2].item())
    return finalized
