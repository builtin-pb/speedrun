from __future__ import annotations

import os
import math

import torch
import torch.distributed as dist


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
    global_grad_sq = zero()
    global_param_sq = zero()
    grad_nonfinite = zero()
    param_nonfinite = zero()
    grad_max_abs = zero()
    param_max_abs = zero()
    matrix_metrics: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        detached_param = param.detach()
        param_norm = torch.linalg.vector_norm(detached_param).float()
        param_sq = param_norm.square()
        global_param_sq += param_sq
        param_nonfinite += (~torch.isfinite(detached_param)).sum()
        param_max_abs = torch.maximum(param_max_abs, detached_param.abs().amax().float())

        if include_matrix:
            matrix_metrics[parameter_metric_name(name, "param_l2")] = param_norm

        if param.grad is None:
            continue

        detached_grad = param.grad.detach()
        grad_norm = torch.linalg.vector_norm(detached_grad).float()
        grad_sq = grad_norm.square()
        global_grad_sq += grad_sq
        grad_nonfinite += (~torch.isfinite(detached_grad)).sum()
        grad_max_abs = torch.maximum(grad_max_abs, detached_grad.abs().amax().float())

        if include_matrix:
            matrix_metrics[parameter_metric_name(name, "grad_l2")] = grad_norm

    main_metric_tensors = {
        "main/global_grad_l2": global_grad_sq.sqrt(),
        "main/global_param_l2": global_param_sq.sqrt(),
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


def _add_mean_l2(metric_tensors: dict[str, torch.Tensor], name: str, tensor: torch.Tensor) -> None:
    norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
    zero = torch.zeros((), device=norms.device, dtype=torch.float32)
    metric_tensors[f"{name}__sum"] = metric_tensors.get(f"{name}__sum", zero) + norms.sum()
    metric_tensors[f"{name}__count"] = metric_tensors.get(f"{name}__count", zero) + norms.new_tensor(norms.numel(), dtype=torch.float32)


def collect_stability_metrics(
    model,
    inputs: torch.Tensor,
    *,
    micro_batch_size: int,
    max_sequences: int,
    rank: int,
) -> dict[str, float]:
    metric_tensors: dict[str, torch.Tensor] = {}
    softcap = model.config.logit_softcap
    sample_count = min(len(inputs), max_sequences)
    seq_len = inputs.size(1)
    saturation_threshold = softcap * 0.95 / math.sqrt(1 - 0.95**2)

    def observe(name: str, tensor: torch.Tensor) -> None:
        _add_mean_l2(metric_tensors, name, tensor)

    with torch.inference_mode():
        for offset in range(0, sample_count, micro_batch_size):
            input_chunk = inputs[offset:min(offset + micro_batch_size, sample_count)]
            logits = model.compute_raw_logits(input_chunk, observer=observe)
            zero = torch.zeros((), device=logits.device, dtype=torch.float32)
            metric_tensors["logits/sum"] = metric_tensors.get("logits/sum", zero) + logits.sum()
            metric_tensors["logits/sumsq"] = metric_tensors.get("logits/sumsq", zero) + torch.linalg.vector_norm(logits).square()
            metric_tensors["logits/count"] = metric_tensors.get("logits/count", zero) + logits.new_tensor(logits.numel(), dtype=torch.float32)
            metric_tensors["logits/max_abs"] = torch.maximum(metric_tensors.get("logits/max_abs", zero), logits.abs().amax())
            sat_count = torch.count_nonzero(logits.abs() >= saturation_threshold).float()
            metric_tensors["logits/softcap_sat_count"] = metric_tensors.get("logits/softcap_sat_count", zero) + sat_count

    if not metric_tensors:
        return {}

    sum_keys = sorted(key for key in metric_tensors if key != "logits/max_abs")
    reduced_sum_values = torch.stack([metric_tensors[key].detach().to(dtype=torch.float32) for key in sum_keys])
    dist.all_reduce(reduced_sum_values, op=dist.ReduceOp.SUM)
    reduced_sums = {key: value for key, value in zip(sum_keys, reduced_sum_values)}

    reduced_max = metric_tensors["logits/max_abs"].detach().to(dtype=torch.float32)
    dist.all_reduce(reduced_max, op=dist.ReduceOp.MAX)

    if rank != 0:
        return {}

    finalized: dict[str, float] = {}
    finalized["main/stability_sample_sequences_per_rank"] = float(sample_count)
    finalized["main/stability_sample_tokens_per_rank"] = float(sample_count * seq_len)
    finalized["main/stability_sample_fraction_per_rank"] = float(sample_count / max(len(inputs), 1))
    mean_metric_names = sorted(key.removesuffix("__sum") for key in reduced_sums if key.endswith("__sum"))
    for key in mean_metric_names:
        total_sum = reduced_sums[f"{key}__sum"]
        total_count = reduced_sums[f"{key}__count"].clamp_min(1.0)
        finalized[key] = float((total_sum / total_count).cpu().item())

    logits_count = reduced_sums["logits/count"].clamp_min(1.0)
    logits_mean = reduced_sums["logits/sum"] / logits_count
    logits_var = reduced_sums["logits/sumsq"] / logits_count - logits_mean.square()
    finalized["logits/mean"] = float(logits_mean.cpu().item())
    finalized["logits/std"] = float(logits_var.clamp_min(0.0).sqrt().cpu().item())
    finalized["logits/max_abs"] = float(reduced_max.cpu().item())
    finalized["logits/softcap_saturation_frac"] = float((reduced_sums["logits/softcap_sat_count"] / logits_count).cpu().item())
    return finalized
