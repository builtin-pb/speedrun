# 2026-04-24 post-norm tuning

## Goal

Make post-norm consistently beat the default run `d2mokkr3` by at least `0.01` final val loss.

Target to beat:
- default `d2mokkr3`: final `main/val_loss = 3.27669`
- success threshold: `<= 3.26669`

## Starting point

Existing bad post-norm run:
- W&B run: `e1se8i9g`
- command delta: `--layer-norm-position post --residual-depth-scale none`
- final `main/val_loss = 3.51401`

Baseline comparison from logs:
- step 125: default `4.71022`, post-norm `5.14233`
- step 250: default `4.13694`, post-norm `4.59828`
- step 500: default `3.83819`, post-norm `4.23342`
- step 1000: default `3.63631`, post-norm `3.96221`
- step 3800: default `3.27669`, post-norm `3.51401`

## Observations

- The failing post-norm run used `residual_depth_scale=none`, while the default pre-norm baseline uses `init`.
- In the failing post-norm run, `layer_attn/block_*_output_rms`, `layer_mlp/block_*_input_rms`, `layer_mlp/block_*_output_rms`, and `layer_final/residual_rms` all collapse to about `1.0013` by the final checkpoint.
- At the same time, `layer_mlp/block_*_update_rms` becomes extremely large, often in the `200-1000` range.
- `matrix_mlp_proj/block_*_act_abs_*` is the clearest explosion signal in post-norm. Examples from the final summary:
  - block 00 `p50 = 660`, `p90 = 1584`, `p99 = 2544`
  - block 05 `p50 = 196`, `p90 = 580`, `p99 = 1256`
  - block 08 `p50 = 104.5`, `p90 = 298`, `p99 = 1808`
  - some intermediate blocks also degenerate to exact zero activations
- This looks less like a generic optimizer failure and more like an over-large residual branch whose effect is being hidden by the post-add norm.
- On the actual 3800-step schedule, LR warmup is actively harmful for post-norm in this repo. The same post-norm config that looks reasonable with no warmup falls far behind default by step 125 when warmup is enabled.
- `residual_depth_scale=forward` is the first setting that makes post-norm competitive with the default run at matched early validation checkpoints. `init` is much worse and `none` is unstable.
- Lowering `muon_lr` from `0.02` to `0.0125` helps the full-schedule post-norm run. At step 125 it nearly matches the default run, and it stays close through step 1000.
- Lower Muon momentum looks slightly helpful: `muon_momentum=0.90` beats the `0.95` version at step 125 in matched 4-GPU runs.
- Reducing Muon weight decay to `0.005` is neutral to slightly worse than the same config with weight decay `0.01` in the early screen.
- Slightly shrinking only the MLP residual projection (`mlp_proj_init_scale=0.85`) while keeping `muon_momentum=0.90` looks promising: it is the best configuration so far at step 125 in a matched 4-GPU screen.

## Current hypotheses

1. `residual_depth_scale=forward` is the correct residual scaling mode for post-norm in this codebase.
2. Post-norm wants gentler Muon dynamics, especially a smaller `muon_lr` and possibly lower `muon_momentum`.
3. The MLP residual projection is still the highest-leverage instability source, so shrinking `mlp_proj_init_scale` without shrinking the MLP expansion path as much is a better direction than reducing both MLP init scales together.
4. The remaining high-value flag-only knobs are `muon_momentum`, `mlp_proj_init_scale`, `cooldown_frac`, and possibly `adam_embed_lr`.

## First screen plan

- Test post-norm with `residual_depth_scale=init`.
- Test post-norm with `residual_depth_scale=forward`.
- Test reduced `muon_lr` plus warmup.
- Test smaller MLP init scales if residual scaling alone is not enough.

## Experiment log

### 2026-04-24 1. Context pass

Change:
- No code changes yet.
- Read `simple_model.py`, `train_gpt_simple.py`, `simple_optim.py`, repo notes, local logs, and W&B local summaries for `d2mokkr3` and `e1se8i9g`.

Result:
- Confirmed the current post-norm attempt is handicapped by `--residual-depth-scale none`.
- Confirmed the failure signature is giant MLP residual updates with post-add norms flattening hidden-state RMS to `~1`.

Next:
- Run short post-norm screening jobs centered on residual scaling, Muon LR, warmup, and MLP init scales.

### 2026-04-24 2. Residual scaling screen

Change:
- `post-rinit-s500`: `--layer-norm-position post --residual-depth-scale init --train-steps 500 --val-interval 125`
- `post-rforward-s500`: `--layer-norm-position post --residual-depth-scale forward --train-steps 500 --val-interval 125`

Result:
- `init` is clearly bad for post-norm here.
- `forward` is much better and immediately becomes the default branch for further tuning.
- Key checkpoints:
  - `post-rinit-s500`: step 125 val `5.07329` and killed early.
  - `post-rforward-s500`: step 125 val `4.64620`, step 250 val `4.17656`, then killed early around step 285.

Takeaway:
- The earlier bad `post-norm` result was mostly a residual-scaling problem, not proof that post-norm itself cannot work.

### 2026-04-24 3. Warmup and smaller Muon LR screen

Change:
- `post-rforward-warmup-muon0125-s500`: add `--warmup-frac 0.1 --muon-lr 0.0125`
- `post-rforward-warmup-muon0125-a025-s500`: same plus `--attention-scale 0.25`
- `post-rforward-warmup-muon0125-mlp05-050-s500`: same warmup config plus `--mlp-fc-init-scale 0.5 --mlp-proj-init-scale 0.5`
- `post-rforward-nowarmup-muon01-s500`: no warmup, `--muon-lr 0.01`

Result:
- Warmup hurts early learning for post-norm in this trainer.
- Attention scale `0.25` does not help.
- Halving both MLP init scales does not help.
- `muon_lr=0.01` with no warmup did not look clearly better than `0.0125` and was deprioritized.
- Key checkpoints:
  - warmup + `muon_lr=0.0125`: step 125 val `4.91870`, step 250 val `4.19869`, step 375 val `3.91907`, step 500 val `3.79668`
  - warmup + `muon_lr=0.0125` + `attention_scale=0.25`: step 125 val `4.91025`, step 250 val `4.22270`, then killed early
  - warmup + `muon_lr=0.0125` + half MLP init scales: step 125 val `4.91769`, then killed early

Takeaway:
- Keep `warmup_frac=0.0` for post-norm unless later evidence strongly contradicts this.

### 2026-04-24 4. Full-schedule probes around the promising post-norm config

Change:
- `post-rforward-warmup-muon0125-full`: `--layer-norm-position post --residual-depth-scale forward --warmup-frac 0.1 --muon-lr 0.0125`
- `post-rforward-a025-full`: same plus `--attention-scale 0.25`
- `post-rforward-muon0125-nowarmup-fullscan`: `--layer-norm-position post --residual-depth-scale forward --muon-lr 0.0125`
- `post-rforward-muon02-nowarmup-fullscan`: `--layer-norm-position post --residual-depth-scale forward`

Result:
- Warmup remains bad on the real 3800-step schedule.
- The strongest full-schedule candidate becomes `forward + no warmup + muon_lr=0.0125`.
- Compared to default, this candidate stays close through 1000 steps:
  - step 125: `4.72953` vs default `4.71022`
  - step 250: `4.13850` vs default `4.13694`
  - step 375: `3.95389` vs default `3.94632`
  - step 500: `3.85171` vs default `3.83819`
  - step 625: `3.78584` vs default `3.76442`
  - step 750: `3.73994` vs default `3.71581`
  - step 875: `3.70434` vs default `3.67346`
  - step 1000: `3.67652` vs default `3.63631`
- This run is clearly viable but not yet beating the default baseline.

Takeaway:
- Post-norm is now close enough that second-order optimizer/init details are worth tuning, rather than questioning the core setup.

### 2026-04-24 5. Muon momentum and weight decay screen

Change:
- Two matched 2-GPU screens from the best no-warmup config:
  - `post-rforward-muon0125-mom090-nowarmup-scan2g`: add `--muon-momentum 0.90`
  - `post-rforward-muon0125-wd0005-nowarmup-scan2g`: add `--muon-weight-decay 0.005`

Result:
- Lower Muon momentum is slightly better than the current best config at step 125.
- Lower Muon weight decay is slightly worse.
- Step-125 validation:
  - `mom090`: `4.70797`
  - `wd0005`: `4.72071`
  - current best reference (`muon_momentum=0.95`, 4-GPU): `4.72953`

Takeaway:
- Prioritize lower Muon momentum over lower Muon weight decay.

### 2026-04-24 6. 4-GPU confirmation screen for lower momentum and smaller MLP residual projection

Change:
- `post-rforward-muon0125-mom090-full4g`: `--muon-lr 0.0125 --muon-momentum 0.90`
- `post-rforward-muon0125-mom090-mlpproj085-full4g`: same plus `--mlp-proj-init-scale 0.85`

Interim result:
- At step 125, lower momentum alone is roughly tied with the default run and slightly better than the earlier `momentum=0.95` candidate.
- Shrinking only `mlp_proj_init_scale` to `0.85` with lower momentum is better still.
- Step-125 validation:
  - `mom090`: `4.71228`
  - `mom090 + mlp_proj_init_scale=0.85`: `4.70116`
  - default baseline: `4.71022`

Takeaway:
- Best current direction is:
  - `--layer-norm-position post`
  - `--residual-depth-scale forward`
  - `--muon-lr 0.0125`
  - `--muon-momentum 0.90`
  - `--mlp-proj-init-scale 0.85`
- Need to continue this line deeper into training to see whether the early win persists to final loss.

### 2026-04-25 7. Deeper continuation of the best 4-GPU post-norm line

Change:
- Continued the best 4-GPU run with `--muon-lr 0.0125 --muon-momentum 0.90 --mlp-proj-init-scale 0.85` to later checkpoints.

Result:
- The early win does not persist.
- Compared to default, this config is better at steps 125 and 250, about tied at 375, but then falls behind from 500 onward.
- Matched checkpoints versus default:
  - step 125: `4.70116` vs `4.71022` (better by `0.00906`)
  - step 250: `4.13216` vs `4.13694` (better by `0.00478`)
  - step 375: `3.94681` vs `3.94632` (worse by `0.00049`)
  - step 500: `3.84214` vs `3.83819` (worse by `0.00395`)
  - step 625: `3.77492` vs `3.76442` (worse by `0.01050`)
  - step 750: `3.72845` vs `3.71581` (worse by `0.01264`)
  - step 875: `3.68758` vs `3.67346` (worse by `0.01412`)

Takeaway:
- The problem is no longer early instability. The remaining gap is a late-schedule optimization/generalization issue.

### 2026-04-25 8. Local sweeps around the best post-norm recipe

Change:
- 2-GPU / 500-step screens around the `momentum=0.90, mlp_proj_init_scale=0.85` recipe:
  - `post-rforward-m0125-m090-mlpp085-lr014-s500`: raise `muon_lr` to `0.014`
  - `post-rforward-m0125-m090-mlpp085-embed02-s500`: lower `adam_embed_lr` to `0.2`
  - `post-rforward-m0125-m090-mlpp075-s500`: shrink `mlp_proj_init_scale` further to `0.75`
  - `post-rforward-m0125-m088-mlpp085-s500`: lower `muon_momentum` to `0.88`

Result:
- Raising Muon LR to `0.014` is the clear winner.
- Lowering `adam_embed_lr` hurts badly.
- Lowering momentum to `0.88` is not better than `0.90`.
- Shrinking `mlp_proj_init_scale` further to `0.75` is not better than `0.85`.
- Step-125 and step-250 validation:
  - `lr=0.014`: `4.66944`, `4.12008`
  - `adam_embed_lr=0.2`: `4.72236`, `4.14172`
  - `mlp_proj_init_scale=0.75`: `4.70675`, `4.13701`
  - `muon_momentum=0.88`: `4.70469`, `4.14457`

Takeaway:
- Promote `muon_lr=0.014` to full-schedule testing.

### 2026-04-25 9. Full 4-GPU runs with higher Muon LR and cooldown sweeps

Change:
- `post-rforward-m014-m090-mlpp085-full4g`: `--muon-lr 0.014 --muon-momentum 0.90 --mlp-proj-init-scale 0.85`
- `post-rforward-m014-m090-mlpp085-cd030-full4g`: same plus `--cooldown-frac 0.3`

Result:
- Increasing Muon LR to `0.014` is better than the earlier `0.0125` recipe throughout the observed window.
- Reducing `cooldown_frac` from `0.5` to `0.3` helps further and is the best full 4-GPU configuration so far.
- Comparison against default:
  - `lr=0.014`, cooldown `0.5`
    - step 125: `4.66740` vs `4.71022`
    - step 250: `4.12466` vs `4.13694`
    - step 375: `3.94557` vs `3.94632`
    - step 500: `3.84692` vs `3.83819`
    - step 625: `3.78457` vs `3.76442`
  - `lr=0.014`, cooldown `0.3`
    - step 125: `4.66965` vs `4.71022`
    - step 250: `4.12284` vs `4.13694`
    - step 375: `3.94148` vs `3.94632`
    - step 500: `3.84187` vs `3.83819`
    - step 625: `3.77772` vs `3.76442`

Takeaway:
- Best current observed configuration is:
  - `--layer-norm-position post`
  - `--residual-depth-scale forward`
  - `--muon-lr 0.014`
  - `--muon-momentum 0.90`
  - `--mlp-proj-init-scale 0.85`
  - `--cooldown-frac 0.3`
- It still trails the default by step 500+, but the late gap is smaller than before.

### 2026-04-25 10. GPU-count fidelity check and short 1000-step cooldown screens

Change:
- 8-GPU sanity check for the best 4-GPU candidate:
  - `post-rforward-m014-m090-mlpp085-cd030-full8g`
- 4-GPU / 1000-step shorter-schedule sweeps to probe later decay:
  - `post-rforward-m014-m090-mlpp085-cd020-s1000-4g`
  - `post-rforward-m0145-m090-mlpp085-cd020-s1000-4g`

Result:
- The user warning about GPU-count fidelity is valid. The 8-GPU run diverges from the 4-GPU curve enough that 4-GPU winners cannot be assumed to transfer exactly.
- On 8 GPUs with the same global batch size, the best 4-GPU config is still good early, but the step-125 val is worse than the 4-GPU result:
  - 8-GPU `lr=0.014, cooldown=0.3`: step 125 `4.68003`
  - 4-GPU `lr=0.014, cooldown=0.3`: step 125 `4.66965`
- The 1000-step cooldown sweep with `cooldown_frac=0.2` remains promising on 4 GPUs:
  - `lr=0.014`: step 125 `4.67008`, step 250 `4.12483`, step 375 `3.94235`
  - `lr=0.0145`: step 125 `4.66206`

Takeaway:
- Need to prioritize 8-GPU confirmation when narrowing finalists.
- The strongest remaining directions are still small schedule/LR adjustments around:
  - `muon_lr` in the `0.014-0.0145` range
  - `cooldown_frac` below `0.3`

### 2026-04-25 11. 8-GPU 1000-step completion and schedule diagnosis

Change:
- Let the active 8-GPU 1000-step run finish:
  - `post-rforward-m014-m090-mlpp085-cd030-s1000-8g`

Result:
- This run finishes very strongly and becomes the key diagnostic signal for the next round of work.
- Validation checkpoints:
  - step 125: `4.66573`
  - step 250: `4.11910`
  - step 375: `3.94396`
  - step 500: `3.84113`
  - step 625: `3.77714`
  - step 750: `3.70822`
  - step 875: `3.60883`
  - step 1000: `3.54993`
- Compared with the default baseline over the same steps:
  - better through step 375
  - slightly worse at step 500 and 625
  - clearly better by step 750 and beyond

Takeaway:
- Post-norm is not failing from early instability anymore.
- The main issue on the 3800-step horizon is schedule shape: the best post-norm recipe looks under-decayed in the late phase of the long run.

### 2026-04-25 12. Bounded trainer changes for Muon-only schedule control

Change:
- Added bounded, non-architectural Muon schedule controls in `train_gpt_simple.py`:
  - `--muon-cooldown-frac`
  - `--muon-lr-floor-scale`
  - `--muon-decay-start-step`
  - `--muon-decay-steps`
- Added tests covering schedule semantics and argument validation.

Verification:
- `python -m unittest tests.test_lr_schedule tests.test_train_args tests.test_simple_optim`

Result:
- The code change is working as intended, but the first Muon-only decay attempt was not good.
- Run: `post-rforward-m014-m090-mlpp085-cd030-mudec500x500-floor020-s1000-8g`
- Validation checkpoints:
  - step 125: `4.67709`
  - step 250: `4.12428`
  - step 375: `3.94656`
- This is worse than the flag-only reference at the same steps.

Takeaway:
- A late Muon-only taper starting at step 500 is too conservative.
- If Muon-specific decay helps, it likely needs to start earlier.

### 2026-04-25 13. Earlier Muon-only decay looks promising on 8 GPUs

Change:
- Started a more aggressive Muon-only decay screen:
  - `post-rforward-m014-m090-mlpp085-cd030-mudec200x800-floor020-s1000-8g`
- Also started and killed a residual-projection LR split screen after the early signal looked weaker:
  - `post-rforward-m014-m090-mlpp085-reslr070-s1000-8g`

Result:
- The earlier Muon-only decay is the best new result so far.
- `mudec200x800-floor020` checkpoints:
  - step 125: `4.67646`
  - step 250: `4.12188`
  - step 375: `3.92964`
  - step 500: `3.81902`
  - step 625: `3.74264`
- Comparison versus the default baseline:
  - step 125: worse by `0.03376`
  - step 250: better by `0.01506`
  - step 375: better by `0.01668`
  - step 500: better by `0.01917`
  - step 625: better by `0.02178`
- Comparison versus the best prior post-norm 8-GPU flag-only run (`d5ddf134...`):
  - step 375: `3.92964` vs `3.94007`
  - step 500: `3.81902` vs `3.83891`
- The residual-projection LR split looked worse early and was deprioritized:
  - `post-rforward-m014-m090-mlpp085-reslr070-s1000-8g`: step 125 `4.66076`

Takeaway:
- Earlier Muon-specific decay appears to fix the late underperformance problem without changing the model architecture.
- The next decisive test is a full 3800-step 8-GPU run with:
  - `--layer-norm-position post`
  - `--residual-depth-scale forward`
  - `--muon-lr 0.014`
  - `--muon-momentum 0.90`
  - `--mlp-proj-init-scale 0.85`
  - `--cooldown-frac 0.3`
  - `--muon-decay-start-step 200`
  - `--muon-decay-steps 800`
  - `--muon-lr-floor-scale 0.2`

### 2026-04-25 14. Full 8-GPU confirmation run started

Change:
- Started the long-horizon confirmation run:
  - `post-rforward-m014-m090-mlpp085-cd030-mudec200x800-floor020-full8g`

Interim result:
- Early curve is consistent with the successful short-horizon probe.
- First checkpoint:
  - step 125: `4.66360`
- This matches the earlier strong post-norm regime while preserving the schedule change that improved steps 500-625 on the 1000-step run.

Next:
- Let this 3800-step 8-GPU run continue and use it as the main candidate for final success against the `3.26669` target.

### 2026-04-25 15. Confirmation repeats and variance check

Change:
- Repeated the long-horizon Muon-decay candidate once more on 8 GPUs:
  - `post-rforward-m014-m090-mlpp085-cd030-mudec200x800-floor020-full8g-r2`
- Also kept comparing against the first long run:
  - `post-rforward-m014-m090-mlpp085-cd030-mudec200x800-floor020-full8g`

Result:
- The first long run remains strong through step 375:
  - step 125: `4.66360`
  - step 250: `4.11848`
  - step 375: `3.92948`
- The repeat landed noticeably worse by step 125 while still staying ahead of the default baseline:
  - step 125: `4.68738`
  - step 250: `4.12127`
- This confirms there is still meaningful run-to-run variance even on 8 GPUs.

Takeaway:
- The Muon-only decay change clearly moves the post-norm curve into a better late-training regime, but it is not yet stable enough to claim final success from a single short confirmation.
- The first long run is still the strongest candidate and should be treated as the main evidence-bearing line.

### 2026-04-25 16. Additional Muon schedule sweeps after the strong 200/800 result

Change:
- Tested nearby Muon floor/slope variants on 8 GPUs over 1000 steps:
  - `post-rforward-m014-m090-mlpp085-cd030-mudec200x800-floor010-s1000-8g`
  - `post-rforward-m014-m090-mlpp085-cd030-mudec200x1000-floor020-s1000-8g`

Result:
- `floor=0.1` looked weak enough in the very early curve that it was killed before the first validation checkpoint.
- Extending the decay horizon from `800` to `1000` while keeping floor `0.2` also underperformed the current best schedule early:
  - `mudec200x1000-floor020`: step 125 `4.66837`, step 250 `4.11638`
  - current best short-horizon Muon schedule (`mudec200x800-floor020`): step 125 `4.67646`, step 250 `4.12188`, step 375 `3.92964`, step 500 `3.81902`, step 625 `3.74264`

Takeaway:
- The best schedule seen so far is still:
  - `--muon-decay-start-step 200`
  - `--muon-decay-steps 800`
  - `--muon-lr-floor-scale 0.2`
- Lowering the floor or making the decay shallower did not reveal a clearly better nearby setting.
