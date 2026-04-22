from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


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
    for metric_group in ["attn_q", "attn_k", "attn_v", "attn_proj", "mlp_fc", "mlp_proj", "embed", "lm_head", "other"]:
        run.define_metric(f"{metric_group}/*", step_metric="step")
    for block_idx in range(args.num_layers):
        run.define_metric(f"block_{block_idx:02d}/*", step_metric="step")
    run.define_metric("final/*", step_metric="step")
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
        return f"embed/{kind}"
    if name == "proj.weight":
        return f"lm_head/{kind}"
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks":
        block_label = f"block_{int(parts[1]):02d}"
        if parts[2] == "attn":
            mapping = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "proj": "attn_proj"}
            return f"{mapping.get(parts[3], parts[3])}/{block_label}_{kind}"
        if parts[2] == "mlp":
            mapping = {"fc": "mlp_fc", "proj": "mlp_proj"}
            return f"{mapping.get(parts[3], parts[3])}/{block_label}_{kind}"
    return f"other/{name.replace('.', '_')}_{kind}"


def collect_norm_metrics(model, *, include_layer: bool, include_matrix: bool) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
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
    return main_metrics, {}, tensor_metrics_to_floats(matrix_metrics) if include_matrix else {}


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


class StepMetricCollector:
    def __init__(self, model, *, log_logits: bool):
        self.log_logits = log_logits
        self.softcap = model.config.logit_softcap
        self.handles = []
        self.scalar_sums: dict[str, torch.Tensor] = {}
        self.scalar_counts: dict[str, torch.Tensor] = {}
        self.logit_sums: dict[str, torch.Tensor] = {}
        self._register(model)

    def _add_mean_l2(self, name: str, tensor: torch.Tensor) -> None:
        norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
        self.scalar_sums[name] = self.scalar_sums.get(name, torch.zeros((), device=norms.device, dtype=torch.float32)) + norms.sum()
        self.scalar_counts[name] = self.scalar_counts.get(name, torch.zeros((), device=norms.device, dtype=torch.float32)) + norms.new_tensor(
            norms.numel(), dtype=torch.float32
        )

    def _record_embed_output(self, _module, _inputs, output) -> None:
        self._add_mean_l2("embed/activation_l2", F.rms_norm(output.detach().float(), (output.size(-1),)))

    def _make_block_pre_hook(self, block_idx: int):
        name = f"block_{block_idx:02d}/residual_l2"

        def hook(_module, inputs) -> None:
            self._add_mean_l2(name, inputs[0])

        return hook

    def _make_block_post_hook(self, block_idx: int, *, is_last: bool):
        name = f"block_{block_idx:02d}/activation_l2"

        def hook(_module, _inputs, output) -> None:
            self._add_mean_l2(name, output)
            if is_last:
                self._add_mean_l2("final/residual_l2", output)

        return hook

    def _record_logits(self, _module, _inputs, output) -> None:
        logits = output.detach().float()
        zero = torch.zeros((), device=logits.device, dtype=torch.float32)
        self.logit_sums["logits/sum"] = self.logit_sums.get("logits/sum", zero) + logits.sum()
        self.logit_sums["logits/sumsq"] = self.logit_sums.get("logits/sumsq", zero) + logits.square().sum()
        self.logit_sums["logits/count"] = self.logit_sums.get("logits/count", zero) + logits.new_tensor(logits.numel(), dtype=torch.float32)
        self.logit_sums["logits/max_abs"] = torch.maximum(self.logit_sums.get("logits/max_abs", zero), logits.abs().amax())
        softcapped = self.softcap * logits * torch.rsqrt(logits.square() + self.softcap**2)
        sat_count = (softcapped.abs() >= 0.95 * self.softcap).sum().float()
        self.logit_sums["logits/softcap_sat_count"] = self.logit_sums.get("logits/softcap_sat_count", zero) + sat_count

    def _register(self, model) -> None:
        self.handles.append(model.embed.register_forward_hook(self._record_embed_output))
        last_block_idx = len(model.blocks) - 1
        for block_idx, block in enumerate(model.blocks):
            self.handles.append(block.register_forward_pre_hook(self._make_block_pre_hook(block_idx)))
            self.handles.append(block.register_forward_hook(self._make_block_post_hook(block_idx, is_last=block_idx == last_block_idx)))
        if self.log_logits:
            self.handles.append(model.proj.register_forward_hook(self._record_logits))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def finalize(self, *, rank: int) -> dict[str, float]:
        metric_tensors: dict[str, torch.Tensor] = {}
        for key, sum_value in self.scalar_sums.items():
            metric_tensors[f"{key}__sum"] = sum_value
            metric_tensors[f"{key}__count"] = self.scalar_counts[key]
        if self.log_logits:
            for key, value in self.logit_sums.items():
                metric_tensors[key] = value
        if not metric_tensors:
            return {}

        sum_keys = sorted(key for key in metric_tensors if key != "logits/max_abs")
        reduced_sum_values = torch.stack([metric_tensors[key].detach().to(dtype=torch.float32) for key in sum_keys])
        dist.all_reduce(reduced_sum_values, op=dist.ReduceOp.SUM)
        reduced_sums = {key: value for key, value in zip(sum_keys, reduced_sum_values)}

        reduced_max = None
        if "logits/max_abs" in metric_tensors:
            reduced_max = metric_tensors["logits/max_abs"].detach().to(dtype=torch.float32)
            dist.all_reduce(reduced_max, op=dist.ReduceOp.MAX)

        if rank != 0:
            return {}

        finalized: dict[str, float] = {}
        for key in self.scalar_sums:
            total_sum = reduced_sums[f"{key}__sum"]
            total_count = reduced_sums[f"{key}__count"].clamp_min(1.0)
            finalized[key] = float((total_sum / total_count).cpu().item())
        if self.log_logits:
            logits_count = reduced_sums["logits/count"].clamp_min(1.0)
            logits_mean = reduced_sums["logits/sum"] / logits_count
            logits_var = reduced_sums["logits/sumsq"] / logits_count - logits_mean.square()
            finalized["logits/mean"] = float(logits_mean.cpu().item())
            finalized["logits/std"] = float(logits_var.clamp_min(0.0).sqrt().cpu().item())
            finalized["logits/max_abs"] = float(reduced_max.cpu().item())
            finalized["logits/softcap_saturation_frac"] = float(
                (reduced_sums["logits/softcap_sat_count"] / logits_count).cpu().item()
            )
        return finalized
