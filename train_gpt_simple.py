"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
In particular, we remove non-standard parameters (e.g., value embeddings) in order to simplfy experiments.
See the speedrun repo for a full list of contributors.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the simplified GPT baseline with explicit, composable flags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--train-pattern", default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--val-pattern", default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--batch-size", type=int, default=8 * 64 * 1024, help="Global batch size in tokens across all ranks.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=64, help="Per-rank microbatch size in sequences.")
    parser.add_argument("--train-steps", type=int, default=3800)
    parser.add_argument("--val-interval", type=int, default=125)
    parser.add_argument("--val-tokens", type=int, default=10_485_760)
    parser.add_argument("--cooldown-frac", type=float, default=0.7)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--wandb-project", default="modded-nanogpt")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-dir", default="wandb")
    parser.add_argument("--wandb-log-interval", type=int, default=1)
    parser.add_argument("--stability-log-interval", type=int, default=25)
    parser.add_argument("--matrix-log-interval", type=int, default=125)

    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--rope-base", type=float, default=1024.0)
    parser.add_argument("--attention-scale", type=float, default=0.12)
    parser.add_argument("--logit-softcap", type=float, default=15.0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-proj", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--adam-head-lr", type=float, default=1 / 320)
    parser.add_argument("--adam-embed-lr", type=float, default=0.3)
    parser.add_argument("--adam-beta1", type=float, default=0.8)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    return parser


def read_code_snapshot() -> str:
    return Path(__file__).read_text()


def validate_args(args: argparse.Namespace) -> None:
    if args.model_dim % args.head_dim != 0:
        raise ValueError("--head-dim must divide --model-dim")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be positive")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be positive")
    if args.val_interval <= 0:
        raise ValueError("--val-interval must be positive")
    if args.val_tokens <= 0:
        raise ValueError("--val-tokens must be positive")
    if not 0 < args.cooldown_frac <= 1:
        raise ValueError("--cooldown-frac must be in (0, 1]")
    if args.wandb_log_interval <= 0:
        raise ValueError("--wandb-log-interval must be positive")
    if args.stability_log_interval <= 0:
        raise ValueError("--stability-log-interval must be positive")
    if args.matrix_log_interval <= 0:
        raise ValueError("--matrix-log-interval must be positive")


def setup_distributed() -> torch.device:
    import torch
    import torch.distributed as dist

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        raise RuntimeError("LOCAL_RANK is not set. Launch training with torchrun.")

    device = torch.device("cuda", int(local_rank))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    if 8 % dist.get_world_size() != 0:
        raise RuntimeError("This trainer expects a world size that divides 8.")
    return device


def make_logger(log_dir: str):
    import torch.distributed as dist

    log_path = None
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/{uuid.uuid4()}.txt"
        print(log_path)

    def print0(message: str, console: bool = False) -> None:
        if dist.get_rank() != 0:
            return
        with open(log_path, "a") as handle:
            if console:
                print(message)
            print(message, file=handle)

    return print0


def log_resolved_config(args: argparse.Namespace, print0) -> None:
    print0("Resolved config:", console=True)
    for key, value in sorted(vars(args).items()):
        print0(f"  {key}={value}", console=True)


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


def setup_wandb(args: argparse.Namespace, print0, *, rank: int):
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
    run.define_metric("layer/*", step_metric="step")
    run.define_metric("matrix/*", step_metric="step")
    run.define_metric("stability/*", step_metric="step")
    print0(f"W&B run: {run.url}", console=True)
    return run


def should_log(step: int, interval: int) -> bool:
    return step == 1 or step % interval == 0


def tensor_scalar(value) -> float:
    return float(value.detach().cpu().item())


def tensor_metrics_to_floats(metric_tensors: dict[str, "torch.Tensor"]) -> dict[str, float]:
    import torch

    if not metric_tensors:
        return {}
    keys = sorted(metric_tensors)
    values = torch.stack([metric_tensors[key].detach().to(dtype=torch.float32) for key in keys]).cpu().tolist()
    return {key: float(value) for key, value in zip(keys, values)}


def matrix_metric_prefix(name: str) -> str:
    if name == "embed.weight":
        return "matrix/embed"
    if name == "proj.weight":
        return "matrix/lm_head"
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks":
        block_idx = int(parts[1])
        block_prefix = f"matrix/block_{block_idx:02d}"
        if parts[2] == "attn":
            mapping = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "proj": "attn_proj"}
            return f"{block_prefix}/{mapping.get(parts[3], parts[3])}"
        if parts[2] == "mlp":
            mapping = {"fc": "mlp_fc", "proj": "mlp_proj"}
            return f"{block_prefix}/{mapping.get(parts[3], parts[3])}"
    return f"matrix/{name.replace('.', '_')}"


def layer_metric_prefix(name: str) -> str:
    if name == "embed.weight":
        return "layer/embed"
    if name == "proj.weight":
        return "layer/lm_head"
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        return f"layer/block_{int(parts[1]):02d}"
    return "layer/other"


def collect_norm_metrics(model, *, include_layer: bool, include_matrix: bool) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    import torch

    device = next(model.parameters()).device
    zero = lambda: torch.zeros((), device=device, dtype=torch.float32)
    global_grad_sq = zero()
    global_param_sq = zero()
    grad_nonfinite = zero()
    param_nonfinite = zero()
    grad_max_abs = zero()
    param_max_abs = zero()
    layer_grad_sq = defaultdict(zero)
    layer_param_sq = defaultdict(zero)
    matrix_metrics: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        detached_param = param.detach()
        param_norm = torch.linalg.vector_norm(detached_param).float()
        param_sq = param_norm.square()
        global_param_sq += param_sq
        param_nonfinite += (~torch.isfinite(detached_param)).sum()
        param_max_abs = torch.maximum(param_max_abs, detached_param.abs().amax().float())

        layer_prefix = layer_metric_prefix(name)
        if include_layer:
            layer_param_sq[layer_prefix] += param_sq
        if include_matrix:
            matrix_prefix = matrix_metric_prefix(name)
            matrix_metrics[f"{matrix_prefix}/param_l2"] = param_norm

        if param.grad is None:
            continue

        detached_grad = param.grad.detach()
        grad_norm = torch.linalg.vector_norm(detached_grad).float()
        grad_sq = grad_norm.square()
        global_grad_sq += grad_sq
        grad_nonfinite += (~torch.isfinite(detached_grad)).sum()
        grad_max_abs = torch.maximum(grad_max_abs, detached_grad.abs().amax().float())

        if include_layer:
            layer_grad_sq[layer_prefix] += grad_sq
        if include_matrix:
            matrix_prefix = matrix_metric_prefix(name)
            matrix_metrics[f"{matrix_prefix}/grad_l2"] = grad_norm

    main_metric_tensors = {
        "main/global_grad_l2": global_grad_sq.sqrt(),
        "main/global_param_l2": global_param_sq.sqrt(),
        "main/grad_nonfinite_count": grad_nonfinite,
        "main/param_nonfinite_count": param_nonfinite,
        "main/grad_max_abs": grad_max_abs,
        "main/param_max_abs": param_max_abs,
    }
    main_metrics = tensor_metrics_to_floats(main_metric_tensors)

    layer_metric_tensors: dict[str, torch.Tensor] = {}
    if include_layer:
        for layer_prefix, param_sq in layer_param_sq.items():
            layer_metric_tensors[f"{layer_prefix}/param_l2"] = param_sq.sqrt()
        for layer_prefix, grad_sq in layer_grad_sq.items():
            layer_metric_tensors[f"{layer_prefix}/grad_l2"] = grad_sq.sqrt()
    return main_metrics, tensor_metrics_to_floats(layer_metric_tensors), tensor_metrics_to_floats(matrix_metrics)

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
        self.scalar_sums: dict[str, "torch.Tensor"] = {}
        self.scalar_counts: dict[str, "torch.Tensor"] = {}
        self.logit_sums: dict[str, "torch.Tensor"] = {}
        self._register(model)

    def _add_mean_l2(self, name: str, tensor) -> None:
        import torch

        norms = torch.linalg.vector_norm(tensor.detach().float(), dim=-1)
        self.scalar_sums[name] = self.scalar_sums.get(name, torch.zeros((), device=norms.device, dtype=torch.float32)) + norms.sum()
        self.scalar_counts[name] = self.scalar_counts.get(name, torch.zeros((), device=norms.device, dtype=torch.float32)) + norms.new_tensor(
            norms.numel(), dtype=torch.float32
        )

    def _record_embed_output(self, _module, _inputs, output) -> None:
        import torch.nn.functional as F

        self._add_mean_l2("layer/embed/activation_l2", F.rms_norm(output.detach().float(), (output.size(-1),)))

    def _make_block_pre_hook(self, block_idx: int):
        name = f"layer/block_{block_idx:02d}/residual_l2"

        def hook(_module, inputs) -> None:
            self._add_mean_l2(name, inputs[0])

        return hook

    def _make_block_post_hook(self, block_idx: int, *, is_last: bool):
        name = f"layer/block_{block_idx:02d}/activation_l2"

        def hook(_module, _inputs, output) -> None:
            self._add_mean_l2(name, output)
            if is_last:
                self._add_mean_l2("layer/final/residual_l2", output)

        return hook

    def _record_logits(self, _module, _inputs, output) -> None:
        import torch

        logits = output.detach().float()
        zero = torch.zeros((), device=logits.device, dtype=torch.float32)
        self.logit_sums["stability/logits_sum"] = self.logit_sums.get("stability/logits_sum", zero) + logits.sum()
        self.logit_sums["stability/logits_sumsq"] = self.logit_sums.get("stability/logits_sumsq", zero) + logits.square().sum()
        self.logit_sums["stability/logits_count"] = self.logit_sums.get("stability/logits_count", zero) + logits.new_tensor(
            logits.numel(), dtype=torch.float32
        )
        absmax = logits.abs().amax()
        self.logit_sums["stability/logits_max_abs"] = torch.maximum(
            self.logit_sums.get("stability/logits_max_abs", zero),
            absmax,
        )
        softcapped = self.softcap * logits * torch.rsqrt(logits.square() + self.softcap**2)
        sat_count = (softcapped.abs() >= 0.95 * self.softcap).sum().float()
        self.logit_sums["stability/logit_softcap_sat_count"] = self.logit_sums.get("stability/logit_softcap_sat_count", zero) + sat_count

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
        import torch
        import torch.distributed as dist

        metric_tensors: dict[str, torch.Tensor] = {}
        for key, sum_value in self.scalar_sums.items():
            metric_tensors[f"{key}__sum"] = sum_value
            metric_tensors[f"{key}__count"] = self.scalar_counts[key]
        if self.log_logits:
            for key, value in self.logit_sums.items():
                metric_tensors[key] = value
        if not metric_tensors:
            return {}

        sum_keys = sorted(key for key in metric_tensors if key != "stability/logits_max_abs")
        reduced_sum_values = torch.stack([metric_tensors[key].detach().to(dtype=torch.float32) for key in sum_keys])
        dist.all_reduce(reduced_sum_values, op=dist.ReduceOp.SUM)
        reduced_sums = {key: value for key, value in zip(sum_keys, reduced_sum_values)}

        reduced_max = None
        if "stability/logits_max_abs" in metric_tensors:
            reduced_max = metric_tensors["stability/logits_max_abs"].detach().to(dtype=torch.float32)
            dist.all_reduce(reduced_max, op=dist.ReduceOp.MAX)

        if rank != 0:
            return {}

        finalized: dict[str, float] = {}
        for key in self.scalar_sums:
            total_sum = reduced_sums[f"{key}__sum"]
            total_count = reduced_sums[f"{key}__count"].clamp_min(1.0)
            finalized[key] = float((total_sum / total_count).cpu().item())
        if self.log_logits:
            logits_count = reduced_sums["stability/logits_count"].clamp_min(1.0)
            logits_mean = reduced_sums["stability/logits_sum"] / logits_count
            logits_var = reduced_sums["stability/logits_sumsq"] / logits_count - logits_mean.square()
            finalized["stability/logits_mean"] = float(logits_mean.cpu().item())
            finalized["stability/logits_std"] = float(logits_var.clamp_min(0.0).sqrt().cpu().item())
            finalized["stability/logits_max_abs"] = float(reduced_max.cpu().item())
            finalized["stability/logit_softcap_saturation_frac"] = float(
                (reduced_sums["stability/logit_softcap_sat_count"] / logits_count).cpu().item()
            )
        return finalized


def _load_data_shard(file: Path) -> torch.Tensor:
    import torch

    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as handle:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        handle.seek(256 * 4)
        nbytes = handle.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    *,
    seq_len: int,
    rank: int,
    world_size: int,
    device: torch.device,
):
    import torch

    files = sorted(Path.cwd().glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No data shards matched {filename_pattern!r}")
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][: local_batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


def get_lr(step: int, train_steps: int, cooldown_frac: float) -> float:
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        return 1.0
    return (1 - progress) / cooldown_frac


def build_model(args: argparse.Namespace) -> GPT:
    from simple_model import GPT, GPTConfig

    config = GPTConfig(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        head_dim=args.head_dim,
        mlp_expansion=args.mlp_expansion,
        rope_base=args.rope_base,
        attention_scale=args.attention_scale,
        logit_softcap=args.logit_softcap,
    )
    model = GPT(config).cuda()
    if args.compile:
        model.compile(dynamic=False)
    return model


def initialize_model(model: GPT, *, zero_proj: bool) -> None:
    import torch.distributed as dist

    for name, param in model.named_parameters():
        if zero_proj and name.endswith("weight") and "proj" in name:
            param.data.zero_()
        dist.broadcast(param.detach(), 0)


def run_validation(
    model: GPT,
    val_loader,
    *,
    val_tokens: int,
    micro_batch_size: int,
) -> torch.Tensor:
    import torch
    import torch.distributed as dist

    val_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    with torch.no_grad():
        inputs, targets = next(val_loader)
        for offset in range(0, len(inputs), micro_batch_size):
            val_loss += model(inputs[offset:offset + micro_batch_size], targets[offset:offset + micro_batch_size])
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    return val_loss / val_tokens


def run_training(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    from simple_optim import build_optimizers

    device = setup_distributed()
    code_snapshot = read_code_snapshot()
    print0 = make_logger(args.log_dir)
    print0(code_snapshot)
    print0("=" * 100)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    log_resolved_config(args, print0)
    print0("=" * 100)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    run = setup_wandb(args, print0, rank=rank)
    model = build_model(args)
    initialize_model(model, zero_proj=args.zero_proj)
    log_static_model_metrics(run, print0, model=model)
    optimizers = build_optimizers(
        model,
        adam_head_lr=args.adam_head_lr,
        adam_embed_lr=args.adam_embed_lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        adam_weight_decay=args.adam_weight_decay,
        muon_lr=args.muon_lr,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum=args.muon_momentum,
        fused_adamw=args.fused_adamw,
    )

    train_loader = distributed_data_generator(
        args.train_pattern,
        args.batch_size,
        seq_len=args.seq_len,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    training_time = 0.0

    dist.barrier()
    t0 = time.perf_counter()
    try:
        for step in range(args.train_steps + 1):
            is_last_step = step == args.train_steps

            if is_last_step or step % args.val_interval == 0:
                dist.barrier()
                training_time += time.perf_counter() - t0
                model.eval()
                val_loader = distributed_data_generator(
                    args.val_pattern,
                    args.val_tokens,
                    seq_len=args.seq_len,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                val_loss = run_validation(
                    model,
                    val_loader,
                    val_tokens=args.val_tokens,
                    micro_batch_size=args.micro_batch_size,
                )
                val_metrics = {
                    "step": step,
                    "tokens_seen": step * args.batch_size,
                    "main/val_loss": tensor_scalar(val_loss),
                    "main/val_ppl": float(val_loss.exp().cpu().item()),
                    "main/train_time_s": training_time,
                    "main/step_avg_ms": 1000 * training_time / max(step, 1),
                }
                print0(
                    f"step:{step}/{args.train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                    f" step_avg:{1000 * training_time / max(step, 1):.2f}ms",
                    console=True,
                )
                log_run_metrics(run, val_metrics)
                model.train()
                dist.barrier()
                t0 = time.perf_counter()
                if is_last_step:
                    break

            current_step = step + 1
            collect_main_metrics = should_log(current_step, args.wandb_log_interval)
            collect_layer_metrics = should_log(current_step, args.stability_log_interval)
            collect_matrix_metrics = should_log(current_step, args.matrix_log_interval)
            collect_norm_diagnostics = collect_layer_metrics or collect_matrix_metrics
            collect_stability_metrics = collect_layer_metrics
            collect_any_metrics = collect_main_metrics or collect_norm_diagnostics or collect_stability_metrics

            step_start = time.perf_counter()
            inputs, targets = next(train_loader)
            train_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_metric_collector = None
            if collect_stability_metrics:
                step_metric_collector = StepMetricCollector(model, log_logits=True)

            try:
                for offset in range(0, len(inputs), args.micro_batch_size):
                    input_chunk = inputs[offset:offset + args.micro_batch_size]
                    target_chunk = targets[offset:offset + args.micro_batch_size]
                    micro_loss = model(input_chunk, target_chunk)
                    train_loss += micro_loss.detach()
                    micro_loss.backward()
            finally:
                if step_metric_collector is not None:
                    step_metric_collector.close()

            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            train_loss_value = tensor_scalar(train_loss / args.batch_size)
            metrics_payload: dict[str, float] = {}
            lr_scale = get_lr(step, args.train_steps, args.cooldown_frac)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * lr_scale

            if collect_any_metrics:
                metrics_payload.update(
                    {
                        "step": current_step,
                        "tokens_seen": current_step * args.batch_size,
                        "main/train_loss": train_loss_value,
                        "main/lr_adam_head": optimizers[0].param_groups[0]["lr"],
                        "main/lr_adam_embed": optimizers[0].param_groups[1]["lr"],
                        "main/lr_muon": optimizers[1].param_groups[0]["lr"],
                        "main/instrumentation_active": float(collect_norm_diagnostics or collect_stability_metrics),
                    }
                )
                if collect_norm_diagnostics:
                    if rank == 0:
                        main_norm_metrics, layer_norm_metrics, matrix_metrics = collect_norm_metrics(
                            model,
                            include_layer=collect_layer_metrics,
                            include_matrix=collect_matrix_metrics,
                        )
                        metrics_payload.update(main_norm_metrics)
                        metrics_payload.update(layer_norm_metrics)
                        metrics_payload.update(matrix_metrics)
                if step_metric_collector is not None:
                    metrics_payload.update(step_metric_collector.finalize(rank=rank))

            for optimizer in optimizers:
                optimizer.step()
            model.zero_grad(set_to_none=True)
            step_time_s = time.perf_counter() - step_start
            if collect_any_metrics:
                step_time_ms = 1000 * step_time_s
                tokens_per_sec = args.batch_size / max(step_time_s, 1e-9)
                if collect_norm_diagnostics or collect_stability_metrics:
                    metrics_payload["main/instrumented_step_time_ms"] = step_time_ms
                    metrics_payload["main/instrumented_tokens_per_sec"] = tokens_per_sec
                else:
                    metrics_payload["main/step_time_ms"] = step_time_ms
                    metrics_payload["main/tokens_per_sec"] = tokens_per_sec
            approx_training_time = training_time + (time.perf_counter() - t0)
            print0(
                f"step:{current_step}/{args.train_steps} train_loss:{train_loss_value:.5f}"
                f" train_time:{approx_training_time:.3f}s step_avg:{1000 * approx_training_time / current_step:.2f}ms",
                console=True,
            )
            log_run_metrics(run, metrics_payload)
    finally:
        if run is not None:
            run.finish()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    validate_wandb_setup()
    try:
        run_training(args)
    finally:
        try:
            import torch.distributed as dist
        except ModuleNotFoundError:
            pass
        else:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
