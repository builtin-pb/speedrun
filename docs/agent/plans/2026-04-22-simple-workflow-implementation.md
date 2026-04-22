# Simple Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo around the simple trainer, archive legacy speedrun material, and add a clean sync-and-run workflow with plain CLI flags.

**Architecture:** Keep `train_gpt_simple.py` as the single training entrypoint, move model code into `simple_model.py`, move optimizer code into `simple_optim.py`, and relocate legacy material into `archive/legacy_speedrun/`. Keep the active surface small and make `run_simple.sh` the primary server launch path.

**Tech Stack:** Python, PyTorch, bash, rsync, torchrun, git

---

## File Structure

### Active files to create or modify

- Create: `simple_model.py`
- Create: `simple_optim.py`
- Create: `run_simple.sh`
- Create: `experiments/README.md`
- Modify: `train_gpt_simple.py`
- Modify: `sync_to_server.sh`
- Modify: `README.md`

### Archive moves

- Move: `records/` -> `archive/legacy_speedrun/records/`
- Move: `train_gpt.py` -> `archive/legacy_speedrun/train_gpt.py`
- Move: `train_gpt_medium.py` -> `archive/legacy_speedrun/train_gpt_medium.py`
- Move: `triton_kernels.py` -> `archive/legacy_speedrun/triton_kernels.py`
- Move: `evals/` -> `archive/legacy_speedrun/evals/`
- Move: `img/` -> `archive/legacy_speedrun/img/`
- Move: `Dockerfile` -> `archive/legacy_speedrun/Dockerfile`
- Move: `run.sh` -> `archive/legacy_speedrun/run.sh`

### Verification targets

- `python -m py_compile train_gpt_simple.py simple_model.py simple_optim.py`
- `python train_gpt_simple.py --help`
- `bash -n run_simple.sh`
- `bash -n sync_to_server.sh`

### Parallel slices

- Slice A: archive/layout moves and sync exclusions
- Slice B: model and optimizer extraction plus CLI updates
- Slice C: README and workflow helper files

### Task 1: Archive Legacy Speedrun Material

**Files:**
- Create: `archive/legacy_speedrun/`
- Move: `records/`
- Move: `train_gpt.py`
- Move: `train_gpt_medium.py`
- Move: `triton_kernels.py`
- Move: `evals/`
- Move: `img/`
- Move: `Dockerfile`
- Move: `run.sh`

- [x] **Step 1: Create the archive directory layout**
  Observed: Created `archive/legacy_speedrun/` and used it as the destination for the legacy top-level files and folders.

Run:

```bash
mkdir -p archive/legacy_speedrun
```

Expected: `archive/legacy_speedrun/` exists and is ready to receive moved files.

- [x] **Step 2: Move legacy files and directories into the archive**
  Observed: Moved `records/`, `train_gpt.py`, `train_gpt_medium.py`, `triton_kernels.py`, `evals/`, `img/`, `Dockerfile`, and `run.sh` into `archive/legacy_speedrun/`.

Run:

```bash
mv records archive/legacy_speedrun/records
mv train_gpt.py archive/legacy_speedrun/train_gpt.py
mv train_gpt_medium.py archive/legacy_speedrun/train_gpt_medium.py
mv triton_kernels.py archive/legacy_speedrun/triton_kernels.py
mv evals archive/legacy_speedrun/evals
mv img archive/legacy_speedrun/img
mv Dockerfile archive/legacy_speedrun/Dockerfile
mv run.sh archive/legacy_speedrun/run.sh
```

Expected: the top level no longer contains those legacy files and `archive/legacy_speedrun/` now holds them.

- [x] **Step 3: Verify the top-level repo surface is reduced**
  Observed: `find . -maxdepth 1 -mindepth 1 | sort` now shows a top level centered on `train_gpt_simple.py`, helper files, `archive/`, `data/`, `docs/`, and workflow scripts.

Run:

```bash
find . -maxdepth 1 -mindepth 1 | sort
```

Expected: top-level output is centered on the simple workflow plus `archive/`, `data/`, `docs/`, and git metadata.

- [ ] **Step 4: Commit the archive restructure**
  Blocked: The archive move was integrated into a single final implementation commit instead of a standalone commit.

```bash
git add archive
git add -u
git commit -m "refactor: archive legacy speedrun assets"
```

### Task 2: Extract Model Code Into `simple_model.py`

**Files:**
- Create: `simple_model.py`
- Modify: `train_gpt_simple.py`

- [x] **Step 1: Write the new model module**
  Observed: Added `simple_model.py` with `GPTConfig`, `Linear`, `Rotary`, `CausalSelfAttention`, `MLP`, `Block`, and `GPT`, while preserving the original RMSNorm, RoPE, ReLU² MLP, and logit softcap structure.

Create `simple_model.py` with:

```python
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50304
    num_layers: int = 12
    model_dim: int = 768
    head_dim: int = 128


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.bfloat16())


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.angular_freq = nn.Buffer(torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_bthd: Tensor) -> Tensor:
        pos = torch.arange(x_bthd.size(1), dtype=torch.float, device=x_bthd.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_bthd.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), dim=3).type_as(x_bthd)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq = x.size(0), x.size(1)
        q = self.q(x).view(batch, seq, self.num_heads, self.head_dim)
        k = self.k(x).view(batch, seq, self.num_heads, self.head_dim)
        v = self.v(x).view(batch, seq, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=0.12,
            is_causal=True,
        ).transpose(1, 2)
        y = y.contiguous().view(batch, seq, self.num_heads * self.head_dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = F.relu(x).square()
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim)
        self.mlp = MLP(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [Block(config.model_dim, config.head_dim) for _ in range(config.num_layers)]
        )
        self.proj = Linear(config.model_dim, config.vocab_size)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        x = norm(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(norm(x)).float()
        logits = 15 * logits * torch.rsqrt(logits.square() + 225)
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
```

- [x] **Step 2: Update `train_gpt_simple.py` imports and remove inlined model definitions**
  Observed: `train_gpt_simple.py` now builds the model through `simple_model.py` and no longer contains the inline architecture section.

Replace the inlined architecture section with:

```python
from simple_model import GPT, ModelConfig
```

Expected: `train_gpt_simple.py` no longer defines the model classes inline.

- [x] **Step 3: Run a syntax smoke check**
  Observed: `python3 -m py_compile train_gpt_simple.py simple_model.py` passed.

Run:

```bash
python -m py_compile train_gpt_simple.py simple_model.py
```

Expected: command exits successfully with no output.

- [ ] **Step 4: Commit the model extraction**
  Blocked: The model extraction was folded into the single final implementation commit.

```bash
git add train_gpt_simple.py simple_model.py
git commit -m "refactor: extract simple model module"
```

### Task 3: Extract Optimizer Code Into `simple_optim.py`

**Files:**
- Create: `simple_optim.py`
- Modify: `train_gpt_simple.py`

- [x] **Step 1: Write the optimizer module**
  Observed: Added `simple_optim.py` with Newton-Schulz orthogonalization, compiled `muon_update`, `Muon`, and a `build_optimizers(...)` helper that preserves the original AdamW plus Muon parameter split.

Create `simple_optim.py` with:

```python
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


def zeropower_via_newtonschulz5(grad: Tensor) -> Tensor:
    assert grad.ndim >= 2
    x = grad.bfloat16()
    if grad.size(-2) > grad.size(-1):
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x

    if grad.size(-2) > grad.size(-1):
        x = x.mT
    return x


@torch.compile
def muon_update(grad: Tensor, momentum: Tensor, beta: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and params and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(len(params))[::world_size]:
                if base_i + rank < len(params):
                    param = params[base_i + rank]
                    state = self.state[param]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = muon_update(param.grad, state["momentum_buffer"], beta=group["momentum"])
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


@dataclass(frozen=True)
class OptimConfig:
    embed_lr: float = 0.3
    head_lr: float = 1 / 320
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    adam_eps: float = 1e-10
    adam_weight_decay: float = 0.0
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.95


def split_params(model):
    hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    embed_params = [*model.embed.parameters()]
    head_params = [model.proj.weight]
    return hidden_matrix_params, embed_params, head_params


def build_optimizers(model, config: OptimConfig):
    hidden_matrix_params, embed_params, head_params = split_params(model)
    adam_param_groups = [
        dict(params=head_params, lr=config.head_lr),
        dict(params=embed_params, lr=config.embed_lr),
    ]
    optimizer1 = torch.optim.AdamW(
        adam_param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        fused=True,
    )
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=config.muon_lr,
        weight_decay=config.muon_weight_decay,
        momentum=config.muon_momentum,
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
```

- [x] **Step 2: Update `train_gpt_simple.py` to import and use optimizer helpers**
  Observed: `train_gpt_simple.py` now constructs optimizers through `build_optimizers(...)` with explicit CLI-controlled optimizer parameters.

Add:

```python
from simple_optim import OptimConfig, build_optimizers
```

Replace manual optimizer construction with:

```python
optim_config = OptimConfig(
    embed_lr=args.embed_lr,
    head_lr=args.head_lr,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_eps=args.adam_eps,
    adam_weight_decay=args.adam_weight_decay,
    muon_lr=args.muon_lr,
    muon_weight_decay=args.muon_weight_decay,
    muon_momentum=args.muon_momentum,
)
optimizers = build_optimizers(model, optim_config)
```

- [x] **Step 3: Run syntax verification**
  Observed: `python3 -m py_compile train_gpt_simple.py simple_model.py simple_optim.py` passed.

Run:

```bash
python -m py_compile train_gpt_simple.py simple_model.py simple_optim.py
```

Expected: command exits successfully with no output.

- [ ] **Step 4: Commit the optimizer extraction**
  Blocked: The optimizer extraction was folded into the single final implementation commit.

```bash
git add train_gpt_simple.py simple_optim.py
git commit -m "refactor: extract simple optimizer module"
```

### Task 4: Add Plain CLI Flags and Keep `train_gpt_simple.py` as the Entry Point

**Files:**
- Modify: `train_gpt_simple.py`

- [x] **Step 1: Add an explicit `argparse` configuration**
  Observed: Added plain `argparse` flags for data patterns, training cadence, model shape, compile toggles, and optimizer hyperparameters.

Add a helper like:

```python
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train the simplified GPT model.")
    parser.add_argument("--train-bin-pattern", default="data/fineweb10B/fineweb_train_*.bin")
    parser.add_argument("--val-bin-pattern", default="data/fineweb10B/fineweb_val_*.bin")
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8 * 64 * 1024)
    parser.add_argument("--train-steps", type=int, default=3800)
    parser.add_argument("--val-tokens", type=int, default=10485760)
    parser.add_argument("--val-interval", type=int, default=125)
    parser.add_argument("--cooldown-frac", type=float, default=0.7)
    parser.add_argument("--embed-lr", type=float, default=0.3)
    parser.add_argument("--head-lr", type=float, default=1 / 320)
    parser.add_argument("--adam-beta1", type=float, default=0.8)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-10)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--muon-weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    return parser.parse_args()
```

- [x] **Step 2: Validate obvious config assumptions near startup**
  Observed: Added `validate_args(...)` checks for divisibility, positivity, and `cooldown_frac` range before distributed setup.

Add checks like:

```python
if args.model_dim % args.head_dim != 0:
    raise ValueError("--head-dim must divide --model-dim")
if args.val_interval <= 0:
    raise ValueError("--val-interval must be positive")
if not 0 < args.cooldown_frac <= 1:
    raise ValueError("--cooldown-frac must be in (0, 1]")
```

- [x] **Step 3: Build configs and print a startup summary**
  Observed: The trainer now prints a resolved config summary through the rank-0 logger before training begins.

Add:

```python
args = parse_args()
model_config = ModelConfig(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    model_dim=args.model_dim,
    head_dim=args.head_dim,
)

if dist.get_rank() == 0:
    print("Resolved config:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}={value}")
```

Expected: every run prints the explicit configuration before training starts.

- [x] **Step 4: Thread CLI values through the training loop**
  Observed: Replaced hard-coded train/val patterns, batch settings, validation cadence, and optimizer values with `args` throughout the training loop.

Replace hard-coded values with `args`:

```python
train_loader = distributed_data_generator(
    args.train_bin_pattern,
    batch_size=args.batch_size,
    seq_len=args.seq_len,
)
train_steps = args.train_steps
```

And in validation:

```python
val_loader = distributed_data_generator(args.val_bin_pattern, args.val_tokens, seq_len=args.seq_len)
```

And in LR scheduling:

```python
def get_lr(step: int, train_steps: int, cooldown_frac: float) -> float:
    x = step / train_steps
    if x < 1 - cooldown_frac:
        return 1.0
    return (1 - x) / cooldown_frac
```

- [x] **Step 5: Run CLI smoke verification**
  Observed: `python3 train_gpt_simple.py --help` prints help text without touching CUDA or distributed initialization.

Run:

```bash
python train_gpt_simple.py --help
```

Expected: help text prints successfully without trying to initialize CUDA or distributed state.

- [ ] **Step 6: Commit the CLI refactor**
  Blocked: The CLI refactor was folded into the single final implementation commit.

```bash
git add train_gpt_simple.py
git commit -m "feat: add plain cli flags to simple trainer"
```

### Task 5: Add `run_simple.sh` and Update Sync Behavior

**Files:**
- Create: `run_simple.sh`
- Modify: `sync_to_server.sh`

- [x] **Step 1: Create a thin run wrapper**
  Observed: Added `run_simple.sh` as a thin `torchrun --standalone` wrapper that forwards user CLI flags unchanged.

Create `run_simple.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_gpt_simple.py \
  "$@"
```

- [x] **Step 2: Update sync exclusions**
  Observed: `sync_to_server.sh` now excludes `archive/` and common local-only clutter while keeping the existing remote target intact.

Change `sync_to_server.sh` so the rsync command excludes archive and local-only files:

```bash
rsync -av \
  --rsync-path="sudo rsync" \
  --exclude '.git/' \
  --exclude 'archive/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'logs/' \
  -e "ssh -p 36000" \
  ./ \
  beaupeng@beaupeng-any3.devcloud.woa.com:/apdcephfs_zwfy2/share_303944931/beaupeng/modded-nanogpt
```

- [x] **Step 3: Mark the new wrapper executable and verify shell syntax**
  Observed: `run_simple.sh` and `sync_to_server.sh` are executable, and `bash -n run_simple.sh` plus `bash -n sync_to_server.sh` both passed.

Run:

```bash
chmod +x run_simple.sh
bash -n run_simple.sh
bash -n sync_to_server.sh
```

Expected: both shell scripts pass syntax checks.

- [ ] **Step 4: Commit the workflow helpers**
  Blocked: The workflow helper changes were folded into the single final implementation commit.

```bash
git add run_simple.sh sync_to_server.sh
git commit -m "feat: add simple run wrapper and lean sync"
```

### Task 6: Rewrite the README for the Simple Workflow

**Files:**
- Modify: `README.md`
- Create: `experiments/README.md`

- [x] **Step 1: Rewrite the top-level README around the simple workflow**
  Observed: Rewrote `README.md` around the Mac edit, sync, and server run workflow, with archive references and explicit real CLI examples.

The updated README must cover:

```markdown
# Modded-NanoGPT Simple Workflow

This repo is set up for experimenting with the simplified trainer in `train_gpt_simple.py`.

## Files That Matter

- `train_gpt_simple.py`: entrypoint and training loop
- `simple_model.py`: model architecture
- `simple_optim.py`: optimizer code
- `run_simple.sh`: server launch wrapper
- `sync_to_server.sh`: sync helper
- `data/`: training data

## Local And Server Workflow

1. Edit on the Mac.
2. Sync with `./sync_to_server.sh`.
3. SSH to the server and run `./run_simple.sh`.
4. Pass explicit flags when you want to try something:
   - `./run_simple.sh --muon-lr 0.03`
   - `./run_simple.sh --num-layers 16 --model-dim 1024`

## Archive

Legacy speedrun material now lives under `archive/legacy_speedrun/`.
```

- [x] **Step 2: Add a lightweight experiment notes stub**
  Observed: Added `experiments/README.md` as a simple notes template for commands, changes, results, and next ideas.

Create `experiments/README.md` with:

```markdown
# Experiments

Use this directory for short notes, saved command lines, or summaries of runs you want to remember.

Suggested format:

- date
- command
- key change
- result
- next idea
```

- [x] **Step 3: Verify docs mention the new command flow consistently**
  Observed: A corrected `rg` pass over `README.md` and `experiments/README.md` returned no matches for stale active-workflow flags or commands.

Run:

```bash
rg -n "run.sh|train_gpt_medium|train_gpt.py|records/" README.md experiments/README.md
```

Expected: references to the old active workflow are removed or clearly described as archived.

- [ ] **Step 4: Commit the documentation rewrite**
  Blocked: The documentation rewrite was folded into the single final implementation commit.

```bash
git add README.md experiments/README.md
git commit -m "docs: rewrite repo docs for simple workflow"
```

### Task 7: Final Verification and Integration

**Files:**
- Modify: `docs/agent/plans/2026-04-22-simple-workflow-implementation.md`
- Verify: `train_gpt_simple.py`, `simple_model.py`, `simple_optim.py`, `run_simple.sh`, `sync_to_server.sh`, `README.md`

- [x] **Step 1: Run final local verification commands**
  Observed: `python3 -m py_compile train_gpt_simple.py simple_model.py simple_optim.py`, `python3 train_gpt_simple.py --help`, `bash -n run_simple.sh`, and `bash -n sync_to_server.sh` all passed.

Run:

```bash
python -m py_compile train_gpt_simple.py simple_model.py simple_optim.py
python train_gpt_simple.py --help
bash -n run_simple.sh
bash -n sync_to_server.sh
git status --short
```

Expected: Python files compile, help text prints, shell scripts are valid, and only intended changes remain.

- [x] **Step 2: Update this plan file with observed results**
  Observed: Recorded completed steps, blockers, and verification results directly in this plan file.

For each completed step above:

```markdown
- [x] Step name
  Observed: concise note about what actually changed or what command output verified.
```

Expected: the plan file becomes the durable execution log.

- [ ] **Step 3: Commit the final integrated workflow**
  Blocked: Pending final staging and commit after the execution log is written.

```bash
git add train_gpt_simple.py simple_model.py simple_optim.py run_simple.sh sync_to_server.sh README.md experiments/README.md docs/agent/plans/2026-04-22-simple-workflow-implementation.md
git commit -m "feat: simplify repo around train_gpt_simple"
```

- [ ] **Step 4: Prepare handoff notes**
  Blocked: Pending the final implementation commit and final response to the user.

Handoff summary should include:

```text
- what moved into archive
- which commands to use now
- what local verification passed
- any remaining server-side validation left for the user
```
