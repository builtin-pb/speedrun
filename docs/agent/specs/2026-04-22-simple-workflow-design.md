# Simple Workflow Design

**Date:** 2026-04-22

**Goal:** Restructure the repository around `train_gpt_simple.py` so it is easier to learn, easier to edit on a Mac, and easier to sync and run on a server with one command.

## Context

The current repository contains a full speedrun codebase, historical records, images, and multiple training entrypoints. That is useful for archival value, but it is too noisy for a beginner who wants to experiment primarily with architecture and optimizer changes in the simple trainer.

The desired workflow is:

- edit code locally on the Mac
- sync code to the server with `sync_to_server.sh`
- run training on the server with one main command
- tune experiments with normal CLI flags such as `--lr` and `--optimizer`

`data/` must remain intact because it is already used on the server.

## Design Summary

The repo will be split into an active learning-oriented surface and an archive surface.

- The active surface will keep only the files needed for the simple training workflow.
- Legacy speedrun material will move under `archive/legacy_speedrun/`.
- `sync_to_server.sh` will exclude the archive and local-only clutter so the server sync stays lean.
- `run_simple.sh` will be the thin server entrypoint that launches `torchrun` with good defaults and forwards any additional CLI flags to Python unchanged.
- `train_gpt_simple.py` will remain the main training entrypoint, but its responsibilities will be narrowed to orchestration and the training loop.
- Model code and optimizer code will move into helper files so experiments stay readable.

## Repo Layout

### Active Surface

- `train_gpt_simple.py`
  - main entrypoint
  - CLI parsing
  - distributed setup and teardown
  - training and validation loop
  - startup config summary
- `simple_model.py`
  - model building blocks
  - GPT architecture definition
- `simple_optim.py`
  - Muon implementation
  - optimizer parameter grouping
  - optimizer construction helpers
- `run_simple.sh`
  - one-command server launch wrapper around `torchrun`
- `sync_to_server.sh`
  - rsync helper that excludes archive and local-only files
- `README.md`
  - rewritten around the simple workflow only
- `data/`
  - unchanged
- `docs/agent/specs/`
  - design docs
- `docs/agent/plans/`
  - implementation plans
- `experiments/`
  - optional lightweight notes or saved commands for manual experiment tracking

### Archive Surface

Move these into `archive/legacy_speedrun/`:

- `records/`
- `train_gpt.py`
- `train_gpt_medium.py`
- `triton_kernels.py`
- `evals/`
- `img/`
- `Dockerfile`
- legacy speedrun-oriented documentation or scripts that are not required by the simple workflow

## Command-Line Interface

The first version will use plain arguments only. No smart default resolution will be added.

Examples:

- `./run_simple.sh`
- `./run_simple.sh --lr 1e-3 --optimizer adamw`
- `./run_simple.sh --num-layers 16 --model-dim 1024`
- `./run_simple.sh --train-steps 1000 --val-interval 100`

Rules:

- Flags are normal explicit CLI flags implemented with `argparse`.
- Defaults should be visible in code and printed at startup.
- Explicit user-supplied flags always override defaults directly.
- There will be no hidden coupling where changing one flag silently rewrites other flag values.

## Data Flow

1. Local edits are made in the active repo surface.
2. `sync_to_server.sh` copies the active repo to the server while excluding archive and local-only files.
3. The server runs `run_simple.sh`, optionally with CLI overrides.
4. `run_simple.sh` launches `torchrun` and forwards all extra arguments to `train_gpt_simple.py`.
5. `train_gpt_simple.py` parses flags, prints the resolved config, builds the model and optimizers, and executes training.

## Architecture Boundaries

### `train_gpt_simple.py`

Responsibilities:

- parse CLI arguments
- validate obvious invariants such as `model_dim % head_dim == 0`
- initialize distributed training
- construct the model via `simple_model.py`
- construct optimizers via `simple_optim.py`
- run training and validation
- log progress and shutdown cleanly

Non-responsibilities:

- raw model component definitions
- optimizer implementation details beyond orchestration

### `simple_model.py`

Responsibilities:

- RMSNorm helper
- linear layer wrapper
- rotary embedding
- attention, MLP, block, and GPT definitions

### `simple_optim.py`

Responsibilities:

- Newton-Schulz helper
- compiled Muon update
- Muon optimizer class
- parameter grouping rules for embeddings, head, and block matrices
- creation of optimizer objects from CLI config

## Error Handling

The simple workflow should fail fast with explicit messages for invalid inputs. Examples:

- `--head-dim` must divide `--model-dim`
- batch size assumptions must remain valid for the current world size
- data files must exist for the configured pattern

Failure mode priorities:

- clear startup validation over silent fallback
- short error messages tied to the exact flag or assumption that failed

## Testing and Verification

Local verification should avoid requiring a full GPU training run on the Mac.

Expected checks:

- Python import smoke checks for `train_gpt_simple.py`, `simple_model.py`, and `simple_optim.py`
- CLI help and argument parsing smoke check
- a small syntax-level or import-level verification command

Server-side validation remains the real execution test and is out of scope for local-only verification.

## README Direction

The README should stop presenting the repo primarily as the full speedrun codebase. It should instead describe:

- what the simplified workflow is for
- which files matter for learning and experimentation
- how to sync to the server
- how to run the simple trainer
- where archived legacy material lives

## Non-Goals

These are intentionally excluded from the first restructure:

- smart argument parsing or dependent defaults
- preset systems
- automatic experiment tracking infrastructure
- changes to `data/`
- changes to the underlying training objective or dataset format

## Implementation Risks

- moving files without updating sync exclusions could leave the server with stale or unnecessary content
- splitting the simple trainer incorrectly could make the code harder, not easier, to follow
- README drift could leave commands inconsistent with the new layout

Mitigations:

- keep the active surface very small
- keep `run_simple.sh` thin and transparent
- print the actual runtime configuration at startup
- verify imports and wrapper commands after the restructure

## Success Criteria

The redesign is successful if all of the following are true:

- the top-level repo feels centered on the simple trainer
- legacy material is still preserved under `archive/legacy_speedrun/`
- `data/` remains where it is
- `sync_to_server.sh` excludes archive content
- `run_simple.sh` is the main server command
- `train_gpt_simple.py` accepts plain explicit CLI flags
- model and optimizer logic are factored into helper files without changing the core training behavior unintentionally
