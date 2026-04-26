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
    parser.add_argument("--warmup-frac", type=float, default=0.0, help="Fraction of optimizer steps used for linear LR warmup.")
    parser.add_argument("--cooldown-frac", type=float, default=0.5)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--wandb-project", default="modded-nanogpt")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-dir", default="wandb")
    parser.add_argument("--wandb-log-interval", type=int, default=1)
    parser.add_argument("--stability-log-interval", type=int, default=25)
    parser.add_argument("--stability-sample-sequences", type=int, default=1, help="Per-rank sequences used for sampled activation/logit diagnostics.")
    parser.add_argument("--matrix-log-interval", type=int, default=125)

    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--rope-base", type=float, default=1024.0)
    parser.add_argument("--attention-scale", type=float, default=0.12)
    parser.add_argument("--logit-softcap", type=float, default=15.0)
    parser.add_argument(
        "--attention-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Kimi-style source-history attention residuals; --no-attention-residual restores standard additive residuals.",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--adam-head-lr", type=float, default=1 / 320)
    parser.add_argument("--adam-embed-lr", type=float, default=0.3)
    parser.add_argument(
        "--adam-attnres-lr",
        type=float,
        default=0.02,
        help="AdamW LR for attention-residual depth-query vectors; active only when --attention-residual is enabled.",
    )
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


def validate_schedule(*, train_steps: int, warmup_frac: float, cooldown_frac: float) -> None:
    if train_steps <= 0:
        raise ValueError("--train-steps must be positive")
    if not 0 <= warmup_frac < 1:
        raise ValueError("--warmup-frac must be in [0, 1)")
    if not 0 < cooldown_frac <= 1:
        raise ValueError("--cooldown-frac must be in (0, 1]")
    if warmup_frac + cooldown_frac > 1:
        raise ValueError("--warmup-frac plus --cooldown-frac must not exceed 1")


def resolve_schedule(*, train_steps: int, warmup_frac: float, cooldown_frac: float) -> tuple[int, int]:
    validate_schedule(
        train_steps=train_steps,
        warmup_frac=warmup_frac,
        cooldown_frac=cooldown_frac,
    )
    warmup_steps = int(train_steps * warmup_frac)
    cooldown_steps = int(train_steps * cooldown_frac)
    return warmup_steps, cooldown_steps


def get_lr_scale(step: int, *, train_steps: int, warmup_frac: float, cooldown_frac: float) -> float:
    warmup_steps, cooldown_steps = resolve_schedule(
        train_steps=train_steps,
        warmup_frac=warmup_frac,
        cooldown_frac=cooldown_frac,
    )
    if not 0 <= step < train_steps:
        raise ValueError("step must be in [0, train_steps)")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if cooldown_steps and step >= train_steps - cooldown_steps:
        return (train_steps - step) / cooldown_steps
    return 1.0


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
    validate_schedule(
        train_steps=args.train_steps,
        warmup_frac=args.warmup_frac,
        cooldown_frac=args.cooldown_frac,
    )
    if args.wandb_log_interval <= 0:
        raise ValueError("--wandb-log-interval must be positive")
    if args.stability_log_interval <= 0:
        raise ValueError("--stability-log-interval must be positive")
    if args.stability_sample_sequences <= 0:
        raise ValueError("--stability-sample-sequences must be positive")
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
        attention_residual=args.attention_residual,
    )
    model = GPT(config).cuda()
    if args.compile:
        model.compile(dynamic=False)
    return model


def broadcast_model_parameters(model: GPT) -> None:
    import torch.distributed as dist

    for _, param in model.named_parameters():
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


def all_reduce_model_gradients(model: GPT) -> None:
    import torch
    import torch.distributed as dist

    small_grads: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        if param.grad.ndim <= 1:
            small_grads.append(param.grad)
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

    if not small_grads:
        return

    flat_grad = torch.cat([grad.reshape(-1) for grad in small_grads])
    dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
    offset = 0
    for grad in small_grads:
        grad_size = grad.numel()
        grad.copy_(flat_grad[offset:offset + grad_size].view_as(grad))
        offset += grad_size


def run_training(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    from simple_optim import build_optimizers
    from train_logging import (
        collect_norm_metrics,
        collect_stability_metrics,
        log_run_metrics,
        log_static_model_metrics,
        setup_wandb,
        should_log,
        tensor_scalar,
    )

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
    broadcast_model_parameters(model)
    log_static_model_metrics(run, print0, model=model)
    optimizers = build_optimizers(
        model,
        adam_head_lr=args.adam_head_lr,
        adam_embed_lr=args.adam_embed_lr,
        adam_attnres_lr=args.adam_attnres_lr,
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
            collect_norm_diagnostics = collect_matrix_metrics
            collect_stability_diagnostics = collect_layer_metrics
            collect_any_metrics = collect_main_metrics or collect_norm_diagnostics or collect_stability_diagnostics

            step_start = time.perf_counter()
            inputs, targets = next(train_loader)
            train_loss = torch.zeros((), device=device, dtype=torch.float32)

            for offset in range(0, len(inputs), args.micro_batch_size):
                input_chunk = inputs[offset:offset + args.micro_batch_size]
                target_chunk = targets[offset:offset + args.micro_batch_size]
                micro_loss = model(input_chunk, target_chunk)
                train_loss += micro_loss.detach()
                micro_loss.backward()

            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            all_reduce_model_gradients(model)

            train_loss_value = tensor_scalar(train_loss / args.batch_size)
            metrics_payload: dict[str, float] = {}
            lr_scale = get_lr_scale(
                step,
                train_steps=args.train_steps,
                warmup_frac=args.warmup_frac,
                cooldown_frac=args.cooldown_frac,
            )
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
                        "main/lr_adam_attnres": optimizers[0].param_groups[2]["lr"] if len(optimizers[0].param_groups) > 2 else 0.0,
                        "main/lr_muon": optimizers[1].param_groups[0]["lr"],
                        "main/norm_instrumentation_active": float(collect_norm_diagnostics),
                        "main/stability_replay_active": float(collect_stability_diagnostics),
                    }
                )
                if collect_norm_diagnostics:
                    if rank == 0:
                        norm_metrics_start = time.perf_counter()
                        main_norm_metrics, matrix_metrics = collect_norm_metrics(
                            model,
                            include_matrix=collect_matrix_metrics,
                        )
                        metrics_payload.update(main_norm_metrics)
                        metrics_payload.update(matrix_metrics)
                        metrics_payload["main/norm_metrics_ms"] = 1000 * (time.perf_counter() - norm_metrics_start)
                if collect_stability_diagnostics:
                    stability_metrics_start = time.perf_counter()
                    metrics_payload.update(
                        collect_stability_metrics(
                            model,
                            inputs,
                            micro_batch_size=args.micro_batch_size,
                            max_sequences=args.stability_sample_sequences,
                            rank=rank,
                        )
                    )
                    if rank == 0:
                        metrics_payload["main/stability_metrics_ms"] = 1000 * (time.perf_counter() - stability_metrics_start)

            for optimizer in optimizers:
                optimizer.step()
            model.zero_grad(set_to_none=True)
            step_time_s = time.perf_counter() - step_start
            if collect_any_metrics:
                step_time_ms = 1000 * step_time_s
                tokens_per_sec = args.batch_size / max(step_time_s, 1e-9)
                if collect_norm_diagnostics or collect_stability_diagnostics:
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
    from train_logging import validate_wandb_setup

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
