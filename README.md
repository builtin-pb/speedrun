# Modded NanoGPT

This repo is now organized around a simple training workflow for `train_gpt_simple.py`.

The intended setup is:
- edit code on your Mac
- sync to the server with `./sync_to_server.sh`
- run training on the server with `./run_simple.sh`

If you want the old speedrun code, records, and historical material, look under `archive/legacy_speedrun/`.

## What To Edit

The active surface of the repo is meant to stay small:
- `train_gpt_simple.py` is the main trainer
- `train_logging.py` contains W&B logging and training diagnostics helpers
- `run_simple.sh` is the server-side entrypoint
- `sync_to_server.sh` syncs the repo from your Mac to the server
- `experiments/` is for lightweight experiment notes
- `data/` stays intact and is shared with the server copy

Everything else from the old speedrun workflow is being moved under `archive/legacy_speedrun/`.

## Basic Workflow

### 1. Edit locally

Make code changes on your Mac. The server copy is for running, not for ad hoc edits.

### 2. Sync to server

From the repo root on your Mac:

```bash
./sync_to_server.sh
```

This should copy the active repo contents to the server while leaving large local history and archived material out of the way.

### 3. Run on server

From the repo root on the server:

```bash
./run_simple.sh
```

The goal is that plain defaults are already reasonable, so most runs should not need many flags.

## CLI Examples

Before running any of the commands below, export `WANDB_API_KEY`.

Use plain composable arguments. Override only what you are testing.

Run with defaults:

```bash
./run_simple.sh
```

Change optimizer and learning rate:

```bash
./run_simple.sh --adam-head-lr 0.002 --adam-embed-lr 0.2
```

Try a different model size:

```bash
./run_simple.sh --num-layers 8 --model-dim 512
```

Short smoke run:

```bash
./run_simple.sh --train-steps 200 --val-interval 50
```

Combine architecture and optimizer changes:

```bash
./run_simple.sh --num-layers 12 --model-dim 768 --muon-lr 0.02 --muon-weight-decay 0.02
```

Because the arguments are normal CLI flags, you can compose them freely:

```bash
./run_simple.sh --num-layers 6 --model-dim 384 --adam-head-lr 1e-3 --adam-weight-decay 0.1
```

## First-Time Setup

On any machine that will run training, install the Python dependencies first:

```bash
pip install -r requirements.txt
```

This trainer now requires W&B online logging. Set `WANDB_API_KEY` before launching runs:

```bash
export WANDB_API_KEY=...
```

Keep `data/` intact. If your server already has the dataset downloaded, leave it there.

## Experiment Tracking

Keep experiment notes lightweight. Record:
- the exact command you ran
- what you changed from the previous run
- the result you care about most
- any next guess

There is a starter notes stub in [experiments/README.md](experiments/README.md).

Training also logs to W&B by default. The dashboard is organized into:
- `main/*` for high-signal run health and throughput metrics
- `logits/*` for sampled logits and softcap diagnostics
- `attn_q/*`, `attn_k/*`, `attn_v/*`, `attn_proj/*`, `mlp_fc/*`, `mlp_proj/*`, `embed/*`, and `lm_head/*` for parameter-type diagnostics
- `block_00/*`, `block_01/*`, ... and `final/*` for sampled layer-level diagnostics that are not already captured by matrix norms

## Repo Direction

This repo is intentionally being boiled down to a simpler learning surface:
- one main trainer
- one sync command
- one run command
- a small number of files you actually need to touch

That makes it easier to iterate on architecture and optimizer ideas without dragging the full speedrun codebase into every experiment.
