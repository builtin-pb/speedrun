from __future__ import annotations

import math


LM_HEAD_INIT_MODES = ("mup", "embedding", "torch-default")


def spectral_init_std(in_features: int, out_features: int, *, scale: float = 1.0) -> float:
    return scale * math.sqrt(out_features / in_features) / (math.sqrt(in_features) + math.sqrt(out_features))


def residual_proj_init_scale(
    hidden_init_scale: float,
    *,
    num_layers: int,
    divide_by_sqrt_depth: bool,
) -> float:
    if not divide_by_sqrt_depth:
        return hidden_init_scale
    return hidden_init_scale / math.sqrt(num_layers)


def lm_head_init_std(
    mode: str,
    *,
    model_dim: int,
    embed_init_std: float,
    mup_lm_head_scale: float,
) -> float | None:
    if mode == "mup":
        return mup_lm_head_scale / model_dim
    if mode == "embedding":
        return embed_init_std
    if mode == "torch-default":
        return None
    raise ValueError(f"Unsupported lm_head init mode: {mode}")
