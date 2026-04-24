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

Add linear LR warmup:

```bash
./run_simple.sh --warmup-frac 0.05
```

The default init policy is fixed in code:
- muP-style hidden-layer spectral init
- embedding init std `1.0`
- LM-head init std `1 / model_dim`
- attention and MLP residual projection init additionally scaled by `1 / sqrt(num_layers)`

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
./run_simple.sh --num-layers 6 --model-dim 384 --adam-head-lr 1e-3 --adam-weight-decay 0.1 --warmup-frac 0.05
```

Learning-rate schedule notes:
- `--warmup-frac` linearly ramps each optimizer group for `int(train_steps * warmup_frac)` updates; for example `0.05` means the first 5% of optimizer steps
- `--cooldown-frac` keeps the existing linear cooldown over the final fraction of training
- `--warmup-frac + --cooldown-frac` must not exceed `1`, so warmup, plateau, and cooldown stay non-overlapping
- defaults are `--warmup-frac 0.0` and `--cooldown-frac 0.5`

## First-Time Setup

On any machine that will run training, install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Run tests from the repo root with:

```bash
python3 -m unittest discover -s tests
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
- `logits/*` for sampled logits and softcap diagnostics; defaults sample 1 sequence per rank and can be changed with `--stability-sample-sequences`
- `matrix_attn_q/*`, `matrix_attn_k/*`, `matrix_attn_v/*`, `matrix_attn_proj/*`, `matrix_mlp_fc/*`, `matrix_mlp_proj/*`, and `matrix_embed/*` for matrix diagnostics, including parameter RMS, gradient RMS, and sampled activation-tail summaries
- `matrix_lm_head/*` for LM-head parameter and gradient diagnostics
- `layer_embed/*`, `layer_attn/*`, `layer_mlp/*`, and `layer_final/*` for sampled layer-level RMS diagnostics that are complementary to the matrix parameter/gradient metrics

### W&B Metric Reference

All metrics use `step` as their W&B x-axis unless noted otherwise. Metrics with `block_XX` repeat once for each transformer block.

Compatibility note: the layer, matrix, and global magnitude metrics were renamed from `*_l2` to `*_rms`. Existing W&B dashboards or alerts keyed to the old names need to be updated, and old/new runs are not numerically comparable without accounting for the metric definition change.

Run/index metrics:
- `step`
- `tokens_seen`

Main run metrics:
- `main/train_loss`
- `main/val_loss`
- `main/val_ppl`
- `main/train_time_s`
- `main/step_avg_ms`
- `main/step_time_ms`
- `main/tokens_per_sec`
- `main/instrumented_step_time_ms`
- `main/instrumented_tokens_per_sec`
- `main/lr_adam_head`
- `main/lr_adam_embed`
- `main/lr_muon`
- `main/model_num_params`
- `main/model_param_bytes`
- `main/global_grad_rms`
- `main/global_param_rms`
- `main/grad_nonfinite_count`
- `main/param_nonfinite_count`
- `main/grad_max_abs`
- `main/param_max_abs`
- `main/norm_instrumentation_active`
- `main/stability_replay_active`
- `main/norm_metrics_ms`
- `main/stability_metrics_ms`
- `main/stability_sample_sequences_per_rank`
- `main/stability_sample_tokens_per_rank`
- `main/stability_sample_fraction_per_rank`

Sampled logits metrics:
- `logits/mean`
- `logits/std`
- `logits/top1`
- `logits/top5_mean`
- `logits/bottom1`
- `logits/bottom5_mean`
- `logits/softcap_positive_saturation_frac`
- `logits/softcap_negative_saturation_frac`

Matrix metrics:
- `matrix_embed/param_rms`
- `matrix_embed/grad_rms`
- `matrix_embed/act_abs_p50`
- `matrix_embed/act_abs_p90`
- `matrix_embed/act_abs_p99`
- `matrix_lm_head/param_rms`
- `matrix_lm_head/grad_rms`
- `matrix_attn_q/block_XX_param_rms`
- `matrix_attn_q/block_XX_grad_rms`
- `matrix_attn_q/block_XX_act_abs_p50`
- `matrix_attn_q/block_XX_act_abs_p90`
- `matrix_attn_q/block_XX_act_abs_p99`
- `matrix_attn_k/block_XX_param_rms`
- `matrix_attn_k/block_XX_grad_rms`
- `matrix_attn_k/block_XX_act_abs_p50`
- `matrix_attn_k/block_XX_act_abs_p90`
- `matrix_attn_k/block_XX_act_abs_p99`
- `matrix_attn_v/block_XX_param_rms`
- `matrix_attn_v/block_XX_grad_rms`
- `matrix_attn_v/block_XX_act_abs_p50`
- `matrix_attn_v/block_XX_act_abs_p90`
- `matrix_attn_v/block_XX_act_abs_p99`
- `matrix_attn_proj/block_XX_param_rms`
- `matrix_attn_proj/block_XX_grad_rms`
- `matrix_attn_proj/block_XX_act_abs_p50`
- `matrix_attn_proj/block_XX_act_abs_p90`
- `matrix_attn_proj/block_XX_act_abs_p99`
- `matrix_mlp_fc/block_XX_param_rms`
- `matrix_mlp_fc/block_XX_grad_rms`
- `matrix_mlp_fc/block_XX_act_abs_p50`
- `matrix_mlp_fc/block_XX_act_abs_p90`
- `matrix_mlp_fc/block_XX_act_abs_p99`
- `matrix_mlp_proj/block_XX_param_rms`
- `matrix_mlp_proj/block_XX_grad_rms`
- `matrix_mlp_proj/block_XX_act_abs_p50`
- `matrix_mlp_proj/block_XX_act_abs_p90`
- `matrix_mlp_proj/block_XX_act_abs_p99`
- `matrix_other/*` for future parameters that do not match a known matrix group

Activation-tail quantiles are computed on a random capped sample of absolute values per observed tensor during stability replay, not on every element.

Sampled layer metrics:
- `layer_embed/activation_rms`
- `layer_attn/block_XX_input_rms`
- `layer_attn/block_XX_update_rms`
- `layer_attn/block_XX_output_rms`
- `layer_mlp/block_XX_input_rms`
- `layer_mlp/block_XX_update_rms`
- `layer_mlp/block_XX_output_rms`
- `layer_final/residual_rms`

## Repo Direction

This repo is intentionally being boiled down to a simpler learning surface:
- one main trainer
- one sync command
- one run command
- a small number of files you actually need to touch

That makes it easier to iterate on architecture and optimizer ideas without dragging the full speedrun codebase into every experiment.
