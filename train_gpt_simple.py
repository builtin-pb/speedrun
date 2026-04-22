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
    parser.add_argument("--cooldown-frac", type=float, default=0.7)
    parser.add_argument("--log-dir", default="logs")

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
    model = build_model(args)
    initialize_model(model, zero_proj=args.zero_proj)
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
            print0(
                f"step:{step}/{args.train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                f" step_avg:{1000 * training_time / max(step, 1):.2f}ms",
                console=True,
            )
            model.train()
            dist.barrier()
            t0 = time.perf_counter()
            if is_last_step:
                break

        inputs, targets = next(train_loader)
        for offset in range(0, len(inputs), args.micro_batch_size):
            model(inputs[offset:offset + args.micro_batch_size], targets[offset:offset + args.micro_batch_size]).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step, args.train_steps, args.cooldown_frac)
            optimizer.step()
        model.zero_grad(set_to_none=True)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(
            f"step:{step + 1}/{args.train_steps} train_time:{approx_training_time:.3f}s"
            f" step_avg:{1000 * approx_training_time / (step + 1):.2f}ms",
            console=True,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
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
